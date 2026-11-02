"""Arnés de entrenamiento. Un solo comando para los nueve modelos.

Entrena con las reglas del contrato y escribe predictions.csv. La parada
temprana mira solo val_select. El test se pide explícitamente y solo para la
configuración final.

Desde la raíz del repo, con el entorno activado:

  python IA-Proyecto-Etapa2/scripts/train.py train --model tfidf-logreg --seed 42
  python IA-Proyecto-Etapa2/scripts/train.py train --model bert-base --seed 42 --max-steps 1
  python IA-Proyecto-Etapa2/scripts/train.py hpo --family tfidf
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.harness.catalog import SEEDS, ModelSpec, hpo_trials
from src.harness.data import (
    LABELS_JSON,
    SPLITS,
    HarnessError,
    assert_disjoint,
    class_weights,
    fraction_take,
    known_model,
    load_column,
    load_label_names,
    read_ids,
    resolve_train_ids,
    stratified_take,
)
from src.harness.reporting import git_commit, gpu_status, score, write_config, write_metadata, write_predictions
from src.harness.tfidf import train_tfidf


def configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8")
        except (OSError, ValueError):
            pass


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def effective_batch(batch_size: int) -> tuple[int, int]:
    if batch_size >= 32:
        return batch_size, 1
    accum = math.ceil(32 / batch_size)
    return batch_size * accum, accum


def library_versions() -> dict:
    import pandas
    import sklearn

    versions = {
        "numpy": np.__version__,
        "pandas": pandas.__version__,
        "scikit-learn": sklearn.__version__,
    }
    for module_name in ("torch", "transformers", "peft"):
        try:
            module = __import__(module_name)
        except ImportError:
            continue
        versions[module_name] = getattr(module, "__version__", "desconocida")
    return versions


def prepare(args, spec: ModelSpec):
    labels = load_label_names()
    train_ids, train_path = resolve_train_ids(args.train_ids)
    if train_path.name != "train_S_ids.csv":
        print(
            "Aviso: no está splits/train_S_ids.csv. Esta corrida usa el train completo. "
            "Las corridas finales tienen que usar S, el mismo subconjunto para todos los modelos.",
            file=sys.stderr,
        )
    train = load_column(train_ids)
    val_ids = read_ids(SPLITS / "val_select_ids.csv")
    validation = load_column(val_ids)
    assert_disjoint(train.ids, validation.ids, "val_select")
    if args.predict_split == "val_select":
        evaluation = validation
        eval_name = "val_select"
    else:
        eval_ids = read_ids(SPLITS / f"{args.predict_split}_ids.csv")
        evaluation = load_column(eval_ids)
        assert_disjoint(train.ids, evaluation.ids, args.predict_split)
        eval_name = args.predict_split
    if eval_name == "test" and (args.limit is not None or args.fraction is not None):
        raise HarnessError("Un recorte del train no se califica en test.")
    if args.fraction is not None:
        train = fraction_take(train, args.fraction, args.seed)
    if args.limit is not None:
        train = stratified_take(train, args.limit, args.seed)
        validation = stratified_take(validation, min(args.limit, len(validation.ids)), args.seed)
        if evaluation is not validation and eval_name != "val_select":
            evaluation = stratified_take(evaluation, min(args.limit, len(evaluation.ids)), args.seed)
        elif eval_name == "val_select":
            evaluation = validation
    if train.y.min() < 0 or train.y.max() >= len(labels):
        raise HarnessError("Hay etiquetas fuera del orden de labels.json.")
    weights = class_weights(train.y, len(labels))
    return labels, train, validation, evaluation, eval_name, weights, train_path


def run_dir_for(args, spec: ModelSpec, trial: str | None) -> Path:
    if trial is not None:
        return ROOT / "results" / "hpo" / spec.family / trial
    root = args.output_root or (ROOT / "results")
    return Path(root) / spec.name / f"seed_{args.seed}"


def execute(args, spec: ModelSpec, trial: str | None = None) -> dict:
    set_seed(args.seed)
    labels, train, validation, evaluation, eval_name, weights, train_path = prepare(args, spec)
    if spec.family == "tfidf":
        predicted, probabilities, elapsed, details = train_tfidf(
            train,
            evaluation,
            len(labels),
            args.seed,
            C=args.C,
            ngram=tuple(args.ngram),
            max_features=args.max_features,
        )
        details["n_params"] = int(details["n_features"] * len(labels) + len(labels))
        details["n_trainable_params"] = details["n_params"]
        details["peak_vram_bytes"] = None
        details["dtype"] = None
        details["log"] = []
        best_f1 = None
        steps = None
    else:
        from src.harness.torch_train import train_bilstm, train_transformer

        if spec.family == "bilstm":
            result = train_bilstm(
                train,
                evaluation,
                validation,
                weights,
                len(labels),
                args.seed,
                args.epochs or spec.epochs,
                args.batch_size,
                args.max_len,
                args.lr or spec.learning_rate,
                args.hidden,
                args.dropout,
                args.max_steps,
                args.glove,
                args.dtype,
            )
        else:
            result = train_transformer(
                train,
                evaluation,
                validation,
                weights,
                labels,
                args.seed,
                args.model_id or spec.model_id,
                spec.family,
                args.epochs or spec.epochs,
                args.batch_size,
                args.max_len,
                args.lr or spec.learning_rate,
                args.max_steps,
                args.dtype,
                args.lora_r,
            )
        predicted, probabilities, elapsed, details = (
            result.predicted,
            result.probabilities,
            result.seconds,
            result.details,
        )
        best_f1 = result.best_macro_f1
        steps = result.steps

    if len(predicted) != len(evaluation.ids):
        raise HarnessError("Las predicciones no cubren todas las filas de la partición.")
    names_true = [labels[int(index)] for index in evaluation.y]
    names_pred = [labels[int(index)] for index in predicted]
    destination = run_dir_for(args, spec, trial)
    destination.mkdir(parents=True, exist_ok=True)
    write_predictions(
        destination / "predictions.csv",
        evaluation.ids,
        names_true,
        names_pred,
        probabilities,
        labels,
    )
    if spec.family == "tfidf":
        micro, accum, effective = None, None, None
    else:
        micro = args.batch_size
        effective, accum = effective_batch(args.batch_size)
    gpu, driver = gpu_status()
    metadata = {
        "gpu": gpu,
        "driver_version": driver,
        "torch_version": library_versions().get("torch"),
        "train_time_s": elapsed,
        "samples_per_s": details.get("train_samples_per_s")
        if details.get("train_samples_per_s") is not None
        else (None if elapsed <= 0 else len(train.ids) / elapsed),
        "peak_vram_bytes": details.get("peak_vram_bytes"),
        "n_params": details.get("n_params"),
        "n_trainable_params": details.get("n_trainable_params"),
        "batch_size": effective,
        "max_len": None if spec.family == "tfidf" else args.max_len,
        "dtype": details.get("dtype"),
        "model": spec.name,
        "seed": args.seed,
        "commit": git_commit(),
        "libraries": library_versions(),
    }
    write_metadata(destination / "run_metadata.json", metadata)
    write_config(
        destination / "config.yaml",
        {
            "model": spec.name,
            "family": spec.family,
            "model_id": details.get("model_id") or spec.model_id,
            "seed": args.seed,
            "split": eval_name,
            "train_ids": train_path.name,
            "n_train": len(train.ids),
            "n_eval": len(evaluation.ids),
            "class_weight": [round(float(value), 6) for value in weights],
            "labels": labels,
            "epochs": None if spec.family == "tfidf" else (args.epochs or spec.epochs),
            "max_steps": args.max_steps,
            "limit": args.limit,
            "fraction": args.fraction,
            "batch_size": micro,
            "grad_accum": accum,
            "effective_batch": effective,
            "learning_rate": details.get("learning_rate", args.lr or spec.learning_rate),
            "best_macro_f1_val_select": best_f1,
            "steps": steps,
            "trial": trial,
            **{key: value for key, value in details.items() if key != "log"},
        },
    )
    log_lines = list(details.get("log") or [])
    log_lines.append(f"entrenamiento_s: {elapsed:.3f}")
    (destination / "run.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    print(f"{spec.name}  seed {args.seed}  split {eval_name}")
    return score(destination, LABELS_JSON, eval_name)


def add_train_args(parser: argparse.ArgumentParser, *, require_model: bool) -> None:
    if require_model:
        parser.add_argument("--model", required=True, choices=sorted(known_names()))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--predict-split", default="val_select", choices=("val", "val_select", "test"))
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--fraction", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-len", type=int, default=64)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--dtype", default="auto", choices=("auto", "bf16", "fp16", "fp32"))
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--ngram", type=int, nargs=2, default=(1, 2))
    parser.add_argument("--max-features", type=int, default=None)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--glove", default=None, help="Ruta a GloVe 300d. Sin esto el BiLSTM usa vectores aleatorios congelados.")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--train-ids", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)


def known_names() -> list[str]:
    from src.harness.catalog import MODELS

    return list(MODELS)


def command_train(args) -> int:
    if args.seed not in SEEDS and args.limit is None:
        print(
            f"Aviso: la semilla {args.seed} no es 42, 1337 ni 2026.",
            file=sys.stderr,
        )
    if args.predict_split == "test":
        print(
            "Aviso: esta corrida escribe métricas de test. El contrato las pide una sola vez, con la configuración ya elegida.",
            file=sys.stderr,
        )
    if args.batch_size < 1:
        raise HarnessError("El lote tiene que ser al menos 1.")
    if args.batch_size < 32 and 32 % args.batch_size != 0:
        print(
            f"Aviso: el lote {args.batch_size} no divide a 32. El lote efectivo queda en {effective_batch(args.batch_size)[0]}.",
            file=sys.stderr,
        )
    spec = known_model(args.model)
    execute(args, spec)
    return 0


def command_hpo(args) -> int:
    spec = {
        "tfidf": known_model("tfidf-logreg"),
        "bilstm": known_model("bilstm"),
        "encoder": known_model("bert-base"),
        "decoder": known_model("qwen2.5-1.5b"),
    }[args.family]
    if args.family == "encoder":
        print("La búsqueda de encoders se corre en BERT-base. DistilBERT hereda el lr elegido.")
    if args.family == "decoder":
        print("La búsqueda de decoders se corre en Qwen2.5-1.5B, 20 % del train, 1 época, semilla 42.")
    scored = []
    for trial in hpo_trials(args.family):
        trial_args = argparse.Namespace(**vars(args))
        trial_args.model = spec.name
        trial_args.seed = 42
        trial_args.predict_split = "val_select"
        trial_args.max_steps = None
        trial_args.output_root = None
        trial_args.C = trial.get("C", trial_args.C)
        trial_args.ngram = trial.get("ngram", trial_args.ngram)
        trial_args.lr = trial.get("learning_rate", trial_args.lr)
        trial_args.hidden = trial.get("hidden", trial_args.hidden)
        trial_args.dropout = trial.get("dropout", trial_args.dropout)
        trial_args.lora_r = trial.get("lora_r", trial_args.lora_r)
        trial_args.fraction = trial.get("fraction", trial_args.fraction)
        trial_args.epochs = trial.get("epochs", trial_args.epochs)
        print(f"\nIntento {trial['name']}")
        metrics = execute(trial_args, spec, trial=trial["name"])
        summary = {
            "family": args.family,
            "trial": trial["name"],
            "model": spec.name,
            "split": "val_select",
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "config": {key: (list(value) if isinstance(value, tuple) else value) for key, value in trial.items()},
        }
        path = ROOT / "results" / "hpo" / args.family / f"{trial['name']}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        scored.append(summary)
    best = max(scored, key=lambda item: item["macro_f1"])
    print(f"Mejor intento en val_select: {best['trial']}  macro_f1={best['macro_f1']:.6f}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Entrena un modelo de la Etapa 2 y escribe predictions.csv.")
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train", help="Una corrida: entrena y califica la partición pedida.")
    add_train_args(train, require_model=True)
    hpo = sub.add_parser("hpo", help="Rejilla de la sección 9. Califica solo val_select.")
    add_train_args(hpo, require_model=False)
    hpo.add_argument("--family", required=True, choices=("tfidf", "bilstm", "encoder", "decoder"))
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_stdio()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "train":
            return command_train(args)
        if args.command == "hpo":
            return command_hpo(args)
        parser.error(f"comando desconocido: {args.command}")
    except (HarnessError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
