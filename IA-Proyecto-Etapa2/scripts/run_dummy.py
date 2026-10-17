"""Ajusta DummyClassifier y deja predicciones para evaluate.py.

Contrato, sección 7. No usa pesos de clase. most_frequent se corre una vez;
stratified usa las semillas 42, 1337 y 2026.

Desde IA-Proyecto-Etapa2:

  python scripts/run_dummy.py --smoke
  python scripts/run_dummy.py --samples data/labels.csv --train-ids splits/train_ids.csv --eval-ids splits/test_ids.csv --split test

--smoke arma 20,000 etiquetas con los conteos publicados en la Etapa 1,
parte 80/10/10 con semilla 42 y evalúa el test. Sirve para comprobar el
pipeline. El baseline del artículo se obtiene con el mismo script cuando
existan el dataset y la partición oficial, sin --smoke.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

try:
    import numpy as np
    import sklearn
    from sklearn.dummy import DummyClassifier
    from sklearn.model_selection import train_test_split
except ImportError:
    print(
        "Faltan numpy o scikit-learn. Instálalos con: python -m pip install numpy scikit-learn",
        file=sys.stderr,
    )
    raise SystemExit(2)

ROOT = Path(__file__).resolve().parent.parent
EVALUATE = Path(__file__).resolve().parent / "evaluate.py"
SPLITS = ("train", "val", "val_select", "test")
STRATIFIED_SEEDS = (42, 1337, 2026)
MOST_FREQUENT_SEED = 42
SMOKE_MACRO_MIN = 0.07
SMOKE_MACRO_MAX = 0.10

# Orden habitual de CARER. La prueba de humo lo usa solo para nombrar columnas.
# El orden oficial queda en labels.json cuando se verifique el archivo de Kaggle.
SMOKE_LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]
SMOKE_COUNTS = {
    "sadness": 5797,
    "joy": 6761,
    "love": 1641,
    "anger": 2159,
    "fear": 1937,
    "surprise": 1705,
}


class DummyError(Exception):
    """La corrida no puede armar predicciones válidas."""


def configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8")
        except (OSError, ValueError):
            pass


def read_csv(path: Path) -> tuple[list[str], list[list[str]]]:
    try:
        handle = path.open(newline="", encoding="utf-8-sig")
    except FileNotFoundError:
        raise DummyError(f"No existe: {path}") from None
    with handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header is None:
            raise DummyError(f"{path} está vacío.")
        columns = [cell.strip() for cell in header]
        rows: list[list[str]] = []
        for cells in reader:
            if not cells or all(not cell.strip() for cell in cells):
                continue
            rows.append([cell.strip() for cell in cells])
    return columns, rows


def column_index(columns: list[str], name: str, path: Path) -> int:
    if name not in columns:
        found = ", ".join(columns)
        raise DummyError(f"{path} no tiene la columna {name}. Encontré: {found}.")
    return columns.index(name)


def read_ids(path: Path) -> list[str]:
    columns, rows = read_csv(path)
    index = column_index(columns, "id", path)
    ids = [row[index] if index < len(row) else "" for row in rows]
    if any(not sample_id for sample_id in ids):
        raise DummyError(f"{path} tiene un id vacío.")
    if len(set(ids)) != len(ids):
        raise DummyError(f"{path} tiene id repetidos.")
    if not ids:
        raise DummyError(f"{path} no tiene filas.")
    return ids


def read_samples(path: Path, labels: list[str]) -> dict[str, str]:
    columns, rows = read_csv(path)
    id_index = column_index(columns, "id", path)
    label_index = column_index(columns, "label", path)
    known = set(labels)
    samples: dict[str, str] = {}
    for line_number, row in enumerate(rows, start=2):
        if id_index >= len(row) or label_index >= len(row):
            raise DummyError(f"Fila {line_number} de {path} está incompleta.")
        sample_id = row[id_index]
        label = row[label_index]
        if not sample_id:
            raise DummyError(f"Fila {line_number} de {path}: id vacío.")
        if sample_id in samples:
            raise DummyError(f"id duplicado en {path}: {sample_id}.")
        if label not in known:
            raise DummyError(
                f"Fila {line_number} de {path}: la etiqueta {label!r} no está en labels.json."
            )
        samples[sample_id] = label
    if not samples:
        raise DummyError(f"{path} no tiene filas.")
    return samples


def load_labels(path: Path) -> list[str]:
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        raise DummyError(f"No existe: {path}") from None
    except json.JSONDecodeError as exc:
        raise DummyError(f"{path} no es JSON válido: {exc}") from None
    if isinstance(data, list):
        labels = data
    elif isinstance(data, dict) and isinstance(data.get("labels"), list):
        labels = data["labels"]
    else:
        raise DummyError(f"{path} debe ser una lista de clases o un objeto con la clave 'labels'.")
    if not labels or not all(isinstance(label, str) and label.strip() for label in labels):
        raise DummyError(f"{path} no tiene una lista de clases válida.")
    return [label.strip() for label in labels]


def labels_for(samples: dict[str, str], ids: list[str], split_name: str) -> list[str]:
    missing = [sample_id for sample_id in ids if sample_id not in samples]
    if missing:
        preview = ", ".join(missing[:5])
        raise DummyError(
            f"{split_name} tiene {len(missing)} id que no están en las muestras "
            f"(por ejemplo: {preview})."
        )
    return [samples[sample_id] for sample_id in ids]


def tally(values: list[str], labels: list[str]) -> dict[str, int]:
    counts = {label: 0 for label in labels}
    for value in values:
        counts[value] += 1
    return counts


def git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    commit = completed.stdout.strip()
    return commit or None


def write_config(path: Path, fields: dict) -> None:
    lines: list[str] = []
    for key, value in fields.items():
        if isinstance(value, bool):
            rendered = "true" if value else "false"
        elif value is None:
            rendered = "null"
        elif isinstance(value, (int, float)):
            rendered = str(value)
        else:
            rendered = json.dumps(str(value), ensure_ascii=False)
        lines.append(f"{key}: {rendered}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fit_dummy(
    y_train: list[str],
    y_eval: list[str],
    labels: list[str],
    strategy: str,
    seed: int,
) -> tuple[list[str], np.ndarray, float]:
    """El dummy no lee el texto. X solo existe porque sklearn lo exige en fit."""
    x_train = np.zeros((len(y_train), 1))
    x_eval = np.zeros((len(y_eval), 1))
    classifier = DummyClassifier(strategy=strategy, random_state=seed)
    started = time.perf_counter()
    classifier.fit(x_train, np.asarray(y_train))
    train_seconds = time.perf_counter() - started
    predicted = [str(value) for value in classifier.predict(x_eval)]
    raw_proba = classifier.predict_proba(x_eval)
    class_to_column = {str(name): index for index, name in enumerate(classifier.classes_)}
    ordered = np.zeros((len(y_eval), len(labels)), dtype=np.float64)
    for index, label in enumerate(labels):
        source = class_to_column.get(label)
        if source is not None:
            ordered[:, index] = raw_proba[:, source]
    return predicted, ordered, train_seconds


def write_predictions(
    path: Path,
    sample_ids: list[str],
    y_true: list[str],
    y_pred: list[str],
    probabilities: np.ndarray,
    labels: list[str],
) -> None:
    header = ["id", "y_true", "y_pred", *[f"p_{label}" for label in labels]]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for index, sample_id in enumerate(sample_ids):
            writer.writerow(
                [sample_id, y_true[index], y_pred[index], *probabilities[index].tolist()]
            )


def score_with_evaluate(run_dir: Path, labels_path: Path, split: str) -> dict:
    completed = subprocess.run(
        [
            sys.executable,
            str(EVALUATE),
            "score",
            str(run_dir),
            "--labels",
            str(labels_path),
            "--split",
            split,
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        check=False,
    )
    log_path = run_dir / "run.log"
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(completed.stdout)
        if completed.stderr:
            handle.write(completed.stderr)
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise DummyError(detail or "evaluate.py falló.")
    print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n")
    return json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))


def run_one(
    samples: dict[str, str],
    train_ids: list[str],
    eval_ids: list[str],
    labels: list[str],
    labels_path: Path,
    strategy: str,
    seed: int,
    split: str,
    output_root: Path,
    smoke: bool,
    source: str,
) -> dict:
    y_train = labels_for(samples, train_ids, "train")
    y_eval = labels_for(samples, eval_ids, split)
    train_counts = tally(y_train, labels)
    majority = max(labels, key=lambda label: train_counts[label])
    predicted, probabilities, train_seconds = fit_dummy(y_train, y_eval, labels, strategy, seed)
    if strategy == "most_frequent" and set(predicted) != {majority}:
        found = ", ".join(sorted(set(predicted)))
        raise DummyError(
            f"most_frequent debía predecir solo {majority}. Predijo: {found}."
        )

    model = f"dummy_{strategy}"
    run_dir = output_root / model / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    write_predictions(run_dir / "predictions.csv", eval_ids, y_eval, predicted, probabilities, labels)

    samples_per_s = None if train_seconds <= 0 else len(y_train) / train_seconds
    metadata = {
        "gpu": None,
        "driver_version": None,
        "torch_version": None,
        "train_time_s": train_seconds,
        "samples_per_s": samples_per_s,
        "peak_vram_bytes": None,
        "n_params": 0,
        "n_trainable_params": 0,
        "batch_size": None,
        "max_len": None,
        "dtype": None,
        "model": model,
        "seed": seed,
        "commit": git_commit(),
        "libraries": {"numpy": np.__version__, "scikit-learn": sklearn.__version__},
    }
    (run_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    write_config(
        run_dir / "config.yaml",
        {
            "model": model,
            "strategy": strategy,
            "seed": seed,
            "split": split,
            "class_weight": None,
            "n_train": len(train_ids),
            "n_eval": len(eval_ids),
            "smoke": smoke,
            "source": source,
        },
    )
    (run_dir / "run.log").write_text(
        "\n".join(
            [
                f"estrategia: {strategy}",
                f"semilla: {seed}",
                f"split: {split}",
                f"n_train: {len(train_ids)}",
                f"n_eval: {len(eval_ids)}",
                f"clase_mayoritaria_train: {majority}",
                f"conteos_train: {json.dumps(train_counts, ensure_ascii=False)}",
                f"conteos_eval: {json.dumps(tally(y_eval, labels), ensure_ascii=False)}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"{model}  seed {seed}")
    return score_with_evaluate(run_dir, labels_path, split)


def build_smoke() -> tuple[dict[str, str], list[str], list[str]]:
    if set(SMOKE_COUNTS) != set(SMOKE_LABELS) or sum(SMOKE_COUNTS.values()) != 20_000:
        raise DummyError("Los conteos de la prueba de humo no suman 20,000.")
    labels_array = np.concatenate(
        [np.full(SMOKE_COUNTS[label], label) for label in SMOKE_LABELS]
    )
    identifiers = np.asarray([f"smoke-{index:05d}" for index in range(len(labels_array))])
    indices = np.arange(len(labels_array))
    train_index, holdout_index = train_test_split(
        indices,
        test_size=0.2,
        random_state=42,
        stratify=labels_array,
    )
    _validation_index, test_index = train_test_split(
        holdout_index,
        test_size=0.5,
        random_state=42,
        stratify=labels_array[holdout_index],
    )
    samples = {
        str(identifiers[index]): str(labels_array[index]) for index in range(len(identifiers))
    }
    train_ids = [str(identifiers[index]) for index in train_index]
    test_ids = [str(identifiers[index]) for index in test_index]
    return samples, train_ids, test_ids


def write_smoke_labels(path: Path) -> None:
    payload = {
        "labels": SMOKE_LABELS,
        "status": (
            "Solo para la prueba de humo. "
            "El orden oficial se fija al verificar el dataset de Kaggle."
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def aggregate_stratified(model_dir: Path, split: str) -> None:
    completed = subprocess.run(
        [sys.executable, str(EVALUATE), "aggregate", str(model_dir), "--split", split],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise DummyError(detail or "evaluate.py aggregate falló.")
    print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n")


def run_all(
    samples: dict[str, str],
    train_ids: list[str],
    eval_ids: list[str],
    labels: list[str],
    labels_path: Path,
    split: str,
    output_root: Path,
    smoke: bool,
    source: str,
) -> dict:
    overlap = set(train_ids) & set(eval_ids)
    if overlap:
        preview = ", ".join(sorted(overlap)[:5])
        raise DummyError(f"train y {split} comparten {len(overlap)} id (por ejemplo: {preview}).")

    most_frequent = run_one(
        samples,
        train_ids,
        eval_ids,
        labels,
        labels_path,
        "most_frequent",
        MOST_FREQUENT_SEED,
        split,
        output_root,
        smoke,
        source,
    )
    for seed in STRATIFIED_SEEDS:
        run_one(
            samples,
            train_ids,
            eval_ids,
            labels,
            labels_path,
            "stratified",
            seed,
            split,
            output_root,
            smoke,
            source,
        )
    aggregate_stratified(output_root / "dummy_stratified", split)
    return most_frequent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Corre DummyClassifier y escribe predictions.csv para evaluate.py.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Prueba de humo con los conteos de 20,000 de la Etapa 1.",
    )
    parser.add_argument("--samples", type=Path, help="CSV con columnas id,label")
    parser.add_argument("--train-ids", type=Path, help="CSV con columna id del entrenamiento")
    parser.add_argument("--eval-ids", type=Path, help="CSV con columna id de la partición a calificar")
    parser.add_argument("--labels", type=Path, default=None)
    parser.add_argument("--split", default="test", choices=SPLITS)
    parser.add_argument("--output-root", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_stdio()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.smoke and args.samples is not None:
            raise DummyError("--smoke no se combina con --samples.")
        if args.smoke and args.split != "test":
            raise DummyError("La prueba de humo evalúa el test.")

        if args.smoke:
            output_root = args.output_root or (ROOT / "results" / "smoke")
            labels_path = output_root / "labels.json"
            write_smoke_labels(labels_path)
            labels = load_labels(labels_path)
            samples, train_ids, eval_ids = build_smoke()
            source = (
                "Conteos publicados de la Etapa 1, 20000 filas. "
                "Partición estratificada 80/10/10 con semilla 42, sin clusters."
            )
            print(
                "Prueba de humo: 16000 train / 2000 test, "
                "conteos de la Etapa 1. most_frequent debe quedar cerca de 0.08."
            )
        else:
            if args.samples is None or args.train_ids is None or args.eval_ids is None:
                parser.error("hace falta --smoke, o --samples con --train-ids y --eval-ids")
            output_root = args.output_root or (ROOT / "results")
            labels_path = args.labels or (ROOT / "labels.json")
            labels = load_labels(labels_path)
            samples = read_samples(args.samples, labels)
            train_ids = read_ids(args.train_ids)
            eval_ids = read_ids(args.eval_ids)
            source = str(args.samples)

        metrics = run_all(
            samples,
            train_ids,
            eval_ids,
            labels,
            labels_path,
            args.split,
            output_root,
            args.smoke,
            source,
        )
        if args.smoke:
            macro_f1 = float(metrics["macro_f1"])
            if not SMOKE_MACRO_MIN <= macro_f1 <= SMOKE_MACRO_MAX:
                raise DummyError(
                    f"most_frequent dio macro_f1={macro_f1:.6f}. "
                    f"Con 20,000 filas se espera un valor entre {SMOKE_MACRO_MIN} y {SMOKE_MACRO_MAX}."
                )
            print(f"Prueba de humo aceptada: most_frequent macro_f1={macro_f1:.6f}")
    except DummyError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
