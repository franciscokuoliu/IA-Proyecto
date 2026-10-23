"""Califica corridas a partir de predictions.csv.

Contrato experimental, secciones 10 y 11. Los modelos solo escriben predicciones.
Este script calcula las métricas, resume las semillas y compara pares.

Uso, desde IA-Proyecto-Etapa2:

  python scripts/evaluate.py score results/<modelo>/seed_<n> --split test
  python scripts/evaluate.py aggregate results/<modelo> --split test
  python scripts/evaluate.py compare pred_a.csv pred_b.csv --output comparacion.json

predictions.csv, con las clases en el orden de labels.json:

  id,y_true,y_pred,p_<clase_1>,...,p_<clase_6>

y_true y y_pred aceptan el nombre de la clase o su índice en labels.json.
Las columnas p_<clase> son probabilidades de esa fila, en el mismo orden.

metrics.json lleva Macro-F1, Weighted-F1, precisión, recall y F1 por clase,
la matriz de confusión y los metadatos de la sección 11. Esos metadatos no se
miden aquí: si junto a predictions.csv existe run_metadata.json, se copian.
Las claves que falten quedan en null para que todas las corridas tengan el
mismo esquema.

labels.json es una lista, o un objeto {"labels": [...]}:

  ["sadness", "joy", "love", "anger", "fear", "surprise"]

Ese orden tiene que ser el del dataset, verificado. Este script no lo inventa.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    import numpy as np
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
except ImportError:
    print(
        "Faltan numpy o scikit-learn. Instálalos con: python -m pip install numpy scikit-learn",
        file=sys.stderr,
    )
    raise SystemExit(2)

SCHEMA_VERSION = 1
SPLITS = ("train", "val", "val_select", "test")
PROB_SUM_TOLERANCE = 1e-3
PROB_RANGE_TOLERANCE = 1e-6
LABEL_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
SEED_DIR = re.compile(r"^seed_(\d+)$")

METADATA_KEYS = (
    "gpu",
    "driver_version",
    "torch_version",
    "train_time_s",
    "samples_per_s",
    "peak_vram_bytes",
    "n_params",
    "n_trainable_params",
    "batch_size",
    "max_len",
    "dtype",
    "model",
    "seed",
    "commit",
    "libraries",
)


class EvaluationError(Exception):
    """El archivo de entrada no cumple el esquema o la comparación no se puede alinear."""


@dataclass(frozen=True)
class PredictionTable:
    ids: list[str]
    y_true: list[str]
    y_pred: list[str]
    max_abs_sum_error: float


def configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8")
        except (OSError, ValueError):
            pass


def read_json(path: Path) -> object:
    try:
        text = path.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        raise EvaluationError(f"No existe: {path}") from None
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise EvaluationError(f"{path} no es JSON válido: {exc}") from None


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def default_labels_path() -> Path:
    return Path(__file__).resolve().parent.parent / "labels.json"


def load_labels(path: Path) -> list[str]:
    data = read_json(path)
    if isinstance(data, list):
        raw_labels = data
    elif isinstance(data, dict) and isinstance(data.get("labels"), list):
        raw_labels = data["labels"]
    else:
        raise EvaluationError(
            f"{path} debe ser una lista de clases o un objeto con la clave 'labels'."
        )

    labels: list[str] = []
    for item in raw_labels:
        if not isinstance(item, str):
            raise EvaluationError(f"{path} tiene una clase que no es texto: {item!r}.")
        label = item.strip()
        if not LABEL_NAME.fullmatch(label):
            raise EvaluationError(
                f"La clase {item!r} debe ser un identificador simple "
                "(letra inicial, luego letras, números o _)."
            )
        labels.append(label)

    if len(labels) < 2:
        raise EvaluationError(f"{path} necesita al menos dos clases.")
    if len(set(labels)) != len(labels):
        raise EvaluationError(f"{path} tiene clases repetidas.")
    if len(labels) != 6:
        print(
            f"Aviso: {path.name} tiene {len(labels)} clases. El contrato fija 6.",
            file=sys.stderr,
        )
    return labels


def expected_header(labels: list[str]) -> list[str]:
    return ["id", "y_true", "y_pred", *[f"p_{label}" for label in labels]]


def parse_label(
    raw: str,
    labels: list[str],
    names: set[str],
    line_number: int,
    column: str,
) -> str:
    value = raw.strip()
    if value in names:
        return value
    if value.isdigit():
        index = int(value)
        if index < len(labels):
            return labels[index]
    raise EvaluationError(
        f"Fila {line_number}, columna {column}: {raw!r} no es una clase de labels.json "
        f"ni un índice entre 0 y {len(labels) - 1}."
    )


def parse_probability(raw: str, line_number: int, column: str) -> float:
    try:
        value = float(raw.strip())
    except ValueError:
        raise EvaluationError(
            f"Fila {line_number}, columna {column}: {raw!r} no es un número."
        ) from None
    if not math.isfinite(value):
        raise EvaluationError(
            f"Fila {line_number}, columna {column}: se esperaba una probabilidad finita."
        )
    if value < -PROB_RANGE_TOLERANCE or value > 1.0 + PROB_RANGE_TOLERANCE:
        raise EvaluationError(
            f"Fila {line_number}, columna {column}: {value} está fuera de [0, 1]."
        )
    return value


def load_predictions(path: Path, labels: list[str]) -> PredictionTable:
    expected = expected_header(labels)
    names = set(labels)
    ids: list[str] = []
    y_true: list[str] = []
    y_pred: list[str] = []
    seen: set[str] = set()
    max_abs_sum_error = 0.0

    try:
        handle = path.open(newline="", encoding="utf-8-sig")
    except FileNotFoundError:
        raise EvaluationError(f"No existe: {path}") from None

    with handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header is None:
            raise EvaluationError(f"{path} está vacío.")
        found = [cell.strip() for cell in header]
        if found != expected:
            raise EvaluationError(
                "El encabezado de predictions.csv no coincide con labels.json.\n"
                f"  esperado:   {','.join(expected)}\n"
                f"  encontrado: {','.join(found)}"
            )

        for line_number, cells in enumerate(reader, start=2):
            if not cells or all(not cell.strip() for cell in cells):
                continue
            if len(cells) != len(expected):
                raise EvaluationError(
                    f"Fila {line_number}: hay {len(cells)} columnas y el esquema pide {len(expected)}."
                )

            sample_id = cells[0].strip()
            if not sample_id:
                raise EvaluationError(f"Fila {line_number}: id vacío.")
            if sample_id in seen:
                raise EvaluationError(f"id duplicado: {sample_id}.")
            seen.add(sample_id)

            true_label = parse_label(cells[1], labels, names, line_number, "y_true")
            pred_label = parse_label(cells[2], labels, names, line_number, "y_pred")
            probability_sum = 0.0
            for column, raw in zip(expected[3:], cells[3:]):
                probability_sum += parse_probability(raw, line_number, column)
            max_abs_sum_error = max(max_abs_sum_error, abs(probability_sum - 1.0))

            ids.append(sample_id)
            y_true.append(true_label)
            y_pred.append(pred_label)

    if not ids:
        raise EvaluationError(f"{path} no tiene filas.")
    if max_abs_sum_error > PROB_SUM_TOLERANCE:
        print(
            "Aviso: hay filas cuyas probabilidades no suman 1 "
            f"(error máximo {max_abs_sum_error:.6g}). "
            "Las métricas se calculan con y_pred.",
            file=sys.stderr,
        )
    return PredictionTable(ids, y_true, y_pred, max_abs_sum_error)


def macro_f1(y_true, y_pred, labels: list[str]) -> float:
    """Media sin ponderar del F1 de cada clase de labels.json.

    labels= obliga a usar las 6 clases aunque un remuestreo no traiga alguna.
    zero_division=0 hace que una clase nunca predicha aporte 0, que es el caso
    del DummyClassifier most_frequent.
    """
    _, _, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )
    return float(np.mean(f1))


def classification_scores(y_true, y_pred, labels: list[str]) -> dict:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )
    # Weighted-F1 pondera por el soporte de esta muestra, no por los pesos de la loss.
    weighted = float(np.average(f1, weights=support))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    per_class = {
        label: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, label in enumerate(labels)
    }
    return {
        "macro_f1": float(np.mean(f1)),
        "weighted_f1": weighted,
        "per_class": per_class,
        "confusion_matrix": {
            "labels": list(labels),
            "rows": "y_true",
            "cols": "y_pred",
            "matrix": matrix.astype(int).tolist(),
        },
    }


def empty_metadata() -> dict:
    return {key: None for key in METADATA_KEYS}


def load_metadata(path: Path | None) -> dict:
    metadata = empty_metadata()
    if path is None:
        return metadata
    data = read_json(path)
    if not isinstance(data, dict):
        raise EvaluationError(f"{path} debe ser un objeto JSON.")
    unknown = sorted(key for key in data if key not in METADATA_KEYS)
    if unknown:
        print(
            f"Aviso: {path.name} tiene claves fuera del esquema: {', '.join(unknown)}.",
            file=sys.stderr,
        )
    for key in METADATA_KEYS:
        if key in data:
            metadata[key] = data[key]
    return metadata


def infer_run_identity(metadata: dict, predictions_path: Path) -> None:
    """Completa model y seed solo cuando el nombre de la carpeta ya lo dice."""
    run_dir = predictions_path.parent
    match = SEED_DIR.fullmatch(run_dir.name)
    if match is None:
        return
    if metadata["seed"] is None:
        metadata["seed"] = int(match.group(1))
    parent_name = run_dir.parent.name
    if metadata["model"] is None and parent_name not in {"", ".", "..", "results"}:
        metadata["model"] = parent_name


def resolve_predictions(path: Path) -> Path:
    if path.is_dir():
        csv_path = path / "predictions.csv"
        if not csv_path.is_file():
            raise EvaluationError(f"No está predictions.csv en {path}.")
        return csv_path
    if path.is_file():
        return path
    raise EvaluationError(f"No existe: {path}.")


def resolve_metadata_path(predictions_path: Path, explicit: Path | None) -> Path | None:
    if explicit is not None:
        if not explicit.is_file():
            raise EvaluationError(f"No existe el archivo de metadatos: {explicit}.")
        return explicit
    sibling = predictions_path.parent / "run_metadata.json"
    if sibling.is_file():
        return sibling
    return None


def score_predictions(
    predictions_path: Path,
    labels_path: Path,
    split: str,
    metadata_path: Path | None = None,
    output_path: Path | None = None,
) -> dict:
    if split not in SPLITS:
        allowed = ", ".join(SPLITS)
        raise EvaluationError(f"split debe ser uno de: {allowed}.")

    labels = load_labels(labels_path)
    table = load_predictions(predictions_path, labels)
    scores = classification_scores(table.y_true, table.y_pred, labels)
    metadata = load_metadata(resolve_metadata_path(predictions_path, metadata_path))
    infer_run_identity(metadata, predictions_path)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "report": "run",
        "split": split,
        "n": len(table.ids),
        "predictions_file": predictions_path.name,
        "labels": labels,
        "primary_metric": "macro_f1",
        "macro_f1": scores["macro_f1"],
        "weighted_f1": scores["weighted_f1"],
        "per_class": scores["per_class"],
        "confusion_matrix": scores["confusion_matrix"],
        "probability_check": {
            "max_abs_sum_error": table.max_abs_sum_error,
            "ok": table.max_abs_sum_error <= PROB_SUM_TOLERANCE,
        },
        "metadata": metadata,
    }
    destination = output_path or (predictions_path.parent / "metrics.json")
    write_json(destination, payload)
    return payload


def summarize(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    std = float(array.std(ddof=1)) if array.size >= 2 else None
    return {
        "values": [float(value) for value in array],
        "mean": float(array.mean()),
        "std": std,
        "std_ddof": 1,
    }


def seed_sort_key(seed: object, path: Path) -> tuple:
    if isinstance(seed, int) or (isinstance(seed, str) and seed.isdigit()):
        return (0, int(seed), str(path))
    return (1, str(seed), str(path))


def load_run_report(path: Path) -> dict:
    data = read_json(path)
    if not isinstance(data, dict):
        raise EvaluationError(f"{path} debe ser un objeto JSON.")
    if data.get("report") != "run" or data.get("schema_version") != SCHEMA_VERSION:
        raise EvaluationError(
            f"{path} no es un metrics.json de corrida con schema_version {SCHEMA_VERSION}."
        )
    for key in ("split", "macro_f1", "weighted_f1", "metadata"):
        if key not in data:
            raise EvaluationError(f"{path} no tiene la clave {key}.")
    return data


def aggregate_runs(model_dir: Path, split: str, output_path: Path | None = None) -> dict:
    if not model_dir.is_dir():
        raise EvaluationError(f"No existe la carpeta: {model_dir}.")
    paths = sorted(model_dir.glob("seed_*/metrics.json"))
    if not paths:
        raise EvaluationError(
            f"No hay seed_*/metrics.json en {model_dir}. "
            "Primero califica cada corrida con score."
        )

    selected: list[tuple[Path, dict, object]] = []
    for path in paths:
        payload = load_run_report(path)
        if payload["split"] != split:
            continue
        seed = payload.get("metadata", {}).get("seed")
        if seed is None:
            match = SEED_DIR.fullmatch(path.parent.name)
            if match is not None:
                seed = int(match.group(1))
        if seed is None:
            raise EvaluationError(
                f"{path} no tiene semilla en metadata ni en el nombre de la carpeta."
            )
        selected.append((path, payload, seed))

    if not selected:
        raise EvaluationError(f"Ningún metrics.json en {model_dir} tiene split={split!r}.")

    selected.sort(key=lambda item: seed_sort_key(item[2], item[0]))
    seeds = [item[2] for item in selected]
    if len({str(seed) for seed in seeds}) != len(seeds):
        raise EvaluationError("Hay dos corridas con la misma semilla.")

    models = {
        item[1].get("metadata", {}).get("model")
        for item in selected
        if item[1].get("metadata", {}).get("model") is not None
    }
    if len(models) == 1:
        model_name = next(iter(models))
    else:
        model_name = model_dir.name
        if len(models) > 1:
            print(
                "Aviso: las corridas declaran modelos distintos. "
                "El resumen usa el nombre de la carpeta.",
                file=sys.stderr,
            )

    summary = {
        "schema_version": SCHEMA_VERSION,
        "report": "seed_summary",
        "model": model_name,
        "split": split,
        "n_seeds": len(selected),
        "seeds": seeds,
        "primary_metric": "macro_f1",
        "macro_f1": summarize([float(item[1]["macro_f1"]) for item in selected]),
        "weighted_f1": summarize([float(item[1]["weighted_f1"]) for item in selected]),
        "runs": [item[0].relative_to(model_dir).as_posix() for item in selected],
    }
    if summary["n_seeds"] != 3:
        print(
            f"Aviso: el contrato reporta tres semillas y este resumen tiene {summary['n_seeds']}.",
            file=sys.stderr,
        )

    destination = output_path or (model_dir / f"summary_{split}.json")
    write_json(destination, summary)
    return summary


def align_predictions(
    left: PredictionTable,
    right: PredictionTable,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    left_rows = dict(zip(left.ids, zip(left.y_true, left.y_pred)))
    right_rows = dict(zip(right.ids, zip(right.y_true, right.y_pred)))
    only_left = sorted(set(left_rows) - set(right_rows))
    only_right = sorted(set(right_rows) - set(left_rows))
    if only_left or only_right:
        preview_left = ", ".join(only_left[:5]) or "ninguno"
        preview_right = ", ".join(only_right[:5]) or "ninguno"
        raise EvaluationError(
            "Las dos predicciones no tienen los mismos id.\n"
            f"  solo en A ({len(only_left)}): {preview_left}\n"
            f"  solo en B ({len(only_right)}): {preview_right}"
        )

    y_true: list[str] = []
    pred_a: list[str] = []
    pred_b: list[str] = []
    mismatches: list[str] = []
    for sample_id in sorted(left_rows):
        true_a, prediction_a = left_rows[sample_id]
        true_b, prediction_b = right_rows[sample_id]
        if true_a != true_b:
            mismatches.append(sample_id)
        y_true.append(true_a)
        pred_a.append(prediction_a)
        pred_b.append(prediction_b)
    if mismatches:
        preview = ", ".join(mismatches[:5])
        raise EvaluationError(
            f"y_true no coincide en {len(mismatches)} id (por ejemplo: {preview})."
        )
    return np.asarray(y_true), np.asarray(pred_a), np.asarray(pred_b)


def paired_bootstrap(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    labels: list[str],
    n_resamples: int,
    seed: int,
    confidence: float,
) -> dict:
    point_a = macro_f1(y_true, pred_a, labels)
    point_b = macro_f1(y_true, pred_b, labels)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = np.empty(n_resamples, dtype=np.float64)
    for index in range(n_resamples):
        # Los mismos índices en los dos modelos: el remuestreo está pareado por id.
        sample = rng.integers(0, n, size=n)
        diffs[index] = macro_f1(y_true[sample], pred_a[sample], labels) - macro_f1(
            y_true[sample], pred_b[sample], labels
        )
    alpha = (1.0 - confidence) / 2.0
    ci_low, ci_high = np.quantile(diffs, [alpha, 1.0 - alpha])
    excludes_zero = bool(ci_low > 0.0 or ci_high < 0.0)
    return {
        "macro_f1_a": point_a,
        "macro_f1_b": point_b,
        "point_estimate": point_a - point_b,
        "bootstrap_mean": float(diffs.mean()),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "excludes_zero": excludes_zero,
        "informative": excludes_zero,
    }


def run_name(path: Path) -> str:
    if path.name != "predictions.csv":
        return path.stem
    parent = path.parent
    if SEED_DIR.fullmatch(parent.name) and parent.parent.name not in {"", ".", ".."}:
        return f"{parent.parent.name}/{parent.name}"
    return parent.name


def compare_predictions(
    path_a: Path,
    path_b: Path,
    labels_path: Path,
    n_resamples: int,
    seed: int,
    confidence: float,
    split: str,
    name_a: str | None = None,
    name_b: str | None = None,
    output_path: Path | None = None,
) -> dict:
    if split not in SPLITS:
        allowed = ", ".join(SPLITS)
        raise EvaluationError(f"split debe ser uno de: {allowed}.")
    labels = load_labels(labels_path)
    table_a = load_predictions(path_a, labels)
    table_b = load_predictions(path_b, labels)
    y_true, pred_a, pred_b = align_predictions(table_a, table_b)
    stats = paired_bootstrap(y_true, pred_a, pred_b, labels, n_resamples, seed, confidence)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "report": "paired_bootstrap",
        "metric": "macro_f1",
        "difference": "a_minus_b",
        "split": split,
        "name_a": name_a or run_name(path_a),
        "name_b": name_b or run_name(path_b),
        "n": int(len(y_true)),
        "n_resamples": n_resamples,
        "confidence": confidence,
        "seed": seed,
        "ci_method": "percentile",
        **stats,
    }
    if output_path is not None:
        write_json(output_path, payload)
    return payload


def format_score_line(payload: dict) -> str:
    probabilities = "ok" if payload["probability_check"]["ok"] else "revisar"
    return (
        f"macro_f1={payload['macro_f1']:.6f}  "
        f"weighted_f1={payload['weighted_f1']:.6f}  "
        f"n={payload['n']}  split={payload['split']}  "
        f"probabilidades={probabilities}"
    )


def format_summary_line(payload: dict) -> str:
    stats = payload["macro_f1"]
    if stats["std"] is None:
        center = f"macro_f1={stats['mean']:.6f}"
    else:
        center = f"macro_f1={stats['mean']:.6f} ± {stats['std']:.6f}"
    return f"{center}  semillas={payload['n_seeds']}  split={payload['split']}"


def format_compare_line(payload: dict) -> str:
    percent = int(round(payload["confidence"] * 100))
    decision = "informativa" if payload["informative"] else "el intervalo incluye el cero"
    return (
        f"macro_f1({payload['name_a']}) - macro_f1({payload['name_b']}) = "
        f"{payload['point_estimate']:.6f}  "
        f"IC{percent} [{payload['ci_low']:.6f}, {payload['ci_high']:.6f}]  "
        f"{decision}"
    )


def positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("tiene que ser un entero") from exc
    if number < 1:
        raise argparse.ArgumentTypeError("tiene que ser un entero mayor o igual que 1")
    return number


def unit_interval(value: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("tiene que ser un número") from exc
    if not 0.0 < number < 1.0:
        raise argparse.ArgumentTypeError("tiene que estar entre 0 y 1, sin incluir los extremos")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evaluate.py",
        description="Califica corridas a partir de predictions.csv (contrato, secciones 10 y 11).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    score = subparsers.add_parser("score", help="Calcula el metrics.json de una corrida.")
    score.add_argument("path", type=Path, help="Carpeta de la corrida, o el predictions.csv")
    score.add_argument("--labels", type=Path, default=None, help="labels.json con el orden de las clases")
    score.add_argument("--metadata", type=Path, default=None, help="JSON con los datos de la sección 11")
    score.add_argument("--split", required=True, choices=SPLITS, help="Partición a la que pertenecen las predicciones")
    score.add_argument("--output", type=Path, default=None, help="Destino del metrics.json")

    aggregate = subparsers.add_parser("aggregate", help="Media y desviación de Macro-F1 entre semillas.")
    aggregate.add_argument("model_dir", type=Path, help="Carpeta results/<modelo>, con seed_*/metrics.json")
    aggregate.add_argument("--split", default="test", choices=SPLITS)
    aggregate.add_argument("--output", type=Path, default=None, help="Destino del resumen")

    compare = subparsers.add_parser(
        "compare",
        help="Bootstrap pareado de la diferencia de Macro-F1.",
    )
    compare.add_argument("predictions_a", type=Path)
    compare.add_argument("predictions_b", type=Path)
    compare.add_argument("--labels", type=Path, default=None)
    compare.add_argument("--n-resamples", type=positive_int, default=1000)
    compare.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla del generador. El contrato no la fija; 42 deja el intervalo repetible.",
    )
    compare.add_argument("--confidence", type=unit_interval, default=0.95)
    compare.add_argument("--split", default="test", choices=SPLITS)
    compare.add_argument("--name-a", default=None)
    compare.add_argument("--name-b", default=None)
    compare.add_argument("--output", type=Path, default=None, help="Si se omite, el JSON sale por stdout")
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_stdio()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "score":
            predictions_path = resolve_predictions(args.path)
            payload = score_predictions(
                predictions_path,
                args.labels or default_labels_path(),
                args.split,
                args.metadata,
                args.output,
            )
            destination = args.output or (predictions_path.parent / "metrics.json")
            print(format_score_line(payload))
            print(f"Escrito en {destination}")
        elif args.command == "aggregate":
            payload = aggregate_runs(args.model_dir, args.split, args.output)
            destination = args.output or (args.model_dir / f"summary_{args.split}.json")
            print(format_summary_line(payload))
            print(f"Escrito en {destination}")
        elif args.command == "compare":
            payload = compare_predictions(
                args.predictions_a,
                args.predictions_b,
                args.labels or default_labels_path(),
                args.n_resamples,
                args.seed,
                args.confidence,
                args.split,
                args.name_a,
                args.name_b,
                args.output,
            )
            line = format_compare_line(payload)
            if args.output is None:
                print(line, file=sys.stderr)
                print(json.dumps(payload, indent=2, ensure_ascii=False))
            else:
                print(line)
                print(f"Escrito en {args.output}")
        else:
            parser.error(f"comando desconocido: {args.command}")
    except EvaluationError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
