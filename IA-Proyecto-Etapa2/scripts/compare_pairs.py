#!/usr/bin/env python3
"""Bootstrap pareado de los cuatro pares controlados del contrato (seccion 10).

Cada par se compara semilla contra semilla (la 42 de A con la 42 de B, etc.) con evaluate.py compare
(1000 remuestreos, IC 95%, diferencia = A - B). Escribe un JSON por par en results/pairs/ (esos JSON si
se versionan; las predicciones no). make_report.py arma la tabla a partir de ellos.

Cada integrante corre los pares cuyos dos modelos tenga en su computadora:
    python scripts/compare_pairs.py                    # todos los pares con predictions.csv de ambos lados
    python scripts/compare_pairs.py --pairs distilbert:bert-base
    python scripts/compare_pairs.py --n-resamples 200  # prueba rapida
Un par con modelos en computadoras distintas (p. ej. Qwen2.5-0.5B y Qwen2.5-1.5B) se calcula copiando los
predictions.csv de un lado al otro (3 archivos por modelo) en la misma estructura results/<modelo>/seed_<n>/.

Cada comparacion tarda ~1 minuto. Las semillas de un par se calculan en paralelo (--workers) y el JSON se
reescribe al terminar cada una: si la corrida se interrumpe, al repetirla se retoman las que faltan (--force
las recalcula todas). Requiere numpy y scikit-learn (los usa evaluate.py).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import evaluate  # noqa: E402  (scripts/evaluate.py)

DEFAULT_PAIRS = [
    ("distilbert", "bert-base"),
    ("qwen2.5-0.5b", "qwen2.5-1.5b"),
    ("qwen2.5-1.5b", "gemma-4-e2b"),
    ("qwen2.5-1.5b", "deepseek-r1-distill-qwen-1.5b"),
]
PAYLOAD_KEYS = ("macro_f1_a", "macro_f1_b", "point_estimate", "ci_low", "ci_high", "informative", "n")


def seeds_with_predictions(results: Path, model: str) -> dict:
    out = {}
    for f in sorted((results / model).glob("seed_*/predictions.csv")):
        out[int(f.parent.name.split("_")[1])] = f
    return out


def _compare_one(job):
    """Una comparacion (a nivel de modulo para que los procesos hijos puedan importarla)."""
    a, b, seed, path_a, path_b, labels, n_resamples, boot_seed, confidence = job
    payload = evaluate.compare_predictions(
        path_a, path_b, labels, n_resamples, boot_seed, confidence, "test", name_a=a, name_b=b)
    return seed, payload


def _load_previous(path: Path, settings: dict) -> dict:
    """Semillas ya calculadas con los mismos ajustes (si no coinciden, no se reutiliza nada)."""
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if any(data.get(k) != v for k, v in settings.items()):
        return {}
    return dict(data.get("per_seed", {}))


def _write_pair(path: Path, a: str, b: str, per_seed: dict, settings: dict) -> None:
    diffs = [v["point_estimate"] for v in per_seed.values()]
    data = {
        "a": a, "b": b, "split": "test", "difference": "a_minus_b", "metric": "macro_f1",
        **settings,
        "per_seed": dict(sorted(per_seed.items(), key=lambda kv: int(kv[0]))),
        "summary": {
            "n_seeds": len(per_seed), "mean_difference": sum(diffs) / len(diffs),
            "n_informative": sum(1 for v in per_seed.values() if v["informative"]),
            "same_sign": all(d > 0 for d in diffs) or all(d < 0 for d in diffs),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(data, indent=2) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Bootstrap pareado de los pares controlados.")
    ap.add_argument("--results", type=Path, default=ROOT / "results")
    ap.add_argument("--pairs", nargs="*", default=None, help="pares A:B (por defecto, los cuatro del contrato)")
    ap.add_argument("--n-resamples", type=int, default=1000)
    ap.add_argument("--confidence", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=42, help="semilla del bootstrap")
    ap.add_argument("--labels", type=Path, default=ROOT / "labels.json")
    ap.add_argument("--workers", type=int, default=min(3, os.cpu_count() or 1),
                    help="comparaciones en paralelo (cada una usa ~0.6 GB de RAM)")
    ap.add_argument("--force", action="store_true", help="recalcula aunque el JSON ya tenga la semilla")
    args = ap.parse_args(argv)

    results = args.results.resolve()
    pairs = DEFAULT_PAIRS if not args.pairs else [tuple(p.split(":", 1)) for p in args.pairs]
    out_dir = results / "pairs"
    settings = {"n_resamples": args.n_resamples, "confidence": args.confidence, "bootstrap_seed": args.seed}
    done = 0
    for a, b in pairs:
        pa, pb = seeds_with_predictions(results, a), seeds_with_predictions(results, b)
        common = sorted(set(pa) & set(pb))
        if not common:
            faltan = [m for m, p in ((a, pa), (b, pb)) if not p]
            print(f"{a} vs {b}: omitido (sin predictions.csv de {', '.join(faltan) or 'semillas en comun'}).")
            continue
        path = out_dir / f"{a}__vs__{b}.json"
        per_seed = {} if args.force else {k: v for k, v in _load_previous(path, settings).items()
                                          if int(k) in common}
        todo = [s for s in common if str(s) not in per_seed]
        for s in sorted(int(k) for k in per_seed):
            print(f"{a} vs {b} seed {s}: ya calculada (se reutiliza; --force para recalcular).")
        jobs = [(a, b, s, pa[s], pb[s], args.labels, args.n_resamples, args.seed, args.confidence) for s in todo]

        def accept(seed, payload):
            per_seed[str(seed)] = {k: payload[k] for k in PAYLOAD_KEYS}
            print(f"{a} vs {b} seed {seed}: {evaluate.format_compare_line(payload)}", flush=True)
            _write_pair(path, a, b, per_seed, settings)

        workers = max(1, min(args.workers, len(jobs)))
        if workers == 1:
            for job in jobs:
                accept(*_compare_one(job))
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                for fut in as_completed([pool.submit(_compare_one, job) for job in jobs]):
                    accept(*fut.result())
        if not jobs:
            _write_pair(path, a, b, per_seed, settings)
        done += 1
    print(f"Pares calculados: {done}. JSON en {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
