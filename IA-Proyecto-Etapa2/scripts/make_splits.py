"""Parte el dataset limpio en train / val / test de forma determinista.

Contrato, secciones 4 y 14. Lee salida/cleaned_full_data.csv (id, text, label,
group), que sale de dataset_cleaner.py, y reparte cada fila a una particion:

  - StratifiedGroupKFold de 10 pliegues, semilla 42: el pliegue 0 es test, el
    pliegue 1 es validacion y los 8 restantes son entrenamiento (~80/10/10).
  - Estratificado por emocion y sin partir ningun grupo de cuasi-duplicados.
  - Las filas se ordenan por id antes de partir, asi que el resultado no
    depende del orden del CSV: la misma entrada da siempre el mismo reparto.
  - De la validacion se saca una "val de seleccion" de 10,000 textos
    estratificados, para el early stopping.

Desde IA-Proyecto-Etapa2:

  python scripts/make_splits.py
  python scripts/make_splits.py --write-reference   # solo quien genera la referencia

Salidas (los ids NO se suben al repo, cada quien los regenera):

  splits/train_ids.csv, val_ids.csv, test_ids.csv, val_select_ids.csv   (columna id)
  splits/splits_report.json                                             (conteos y versiones)
  data/labels.csv                                                       (id,label con el nombre de la clase, para run_dummy.py)

Lo unico que se versiona es data/expected_hashes.json, con las huellas del
dataset limpio y de cada particion. check_splits.py compara contra ese archivo
para confirmar que todas las maquinas tienen exactamente los mismos datos.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np

import splits_common as sc

try:
    import pandas as pd
    import sklearn
    from sklearn.model_selection import StratifiedGroupKFold, train_test_split
except ImportError:
    print(
        "Faltan dependencias. Corre .\\setup_env.ps1 desde la raiz del repo.",
        file=sys.stderr,
    )
    raise SystemExit(2)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Reparto determinista train/val/test del dataset limpio."
    )
    parser.add_argument("--clean", type=Path, default=sc.DEFAULT_CLEAN)
    parser.add_argument("--labels-json", type=Path, default=sc.DEFAULT_LABELS_JSON)
    parser.add_argument("--splits-dir", type=Path, default=sc.DEFAULT_SPLITS_DIR)
    parser.add_argument("--labels-out", type=Path, default=sc.DEFAULT_LABELS_CSV)
    parser.add_argument("--reference", type=Path, default=sc.DEFAULT_REFERENCE)
    parser.add_argument(
        "--write-reference",
        action="store_true",
        help="Escribe data/expected_hashes.json con las huellas de esta corrida.",
    )
    parser.add_argument(
        "--force-reference",
        action="store_true",
        help="Permite sobrescribir una referencia que ya existe.",
    )
    return parser.parse_args(argv)


def assign_folds(df) -> np.ndarray:
    """Numero de pliegue (0 a 9) de cada fila. df ya viene ordenado por id."""
    sgkf = StratifiedGroupKFold(
        n_splits=sc.N_FOLDS, shuffle=True, random_state=sc.SEED
    )
    fold = np.full(len(df), -1, dtype=np.int16)
    labels = df["label"].to_numpy()
    # El numero de grupo que escribe el limpiador es arbitrario. Se reemplaza por
    # el menor id de cada grupo (df ya esta ordenado por id), para que el reparto
    # dependa solo de quien es duplicado de quien y no de como se numero.
    canonical = df.groupby("group")["id"].transform("min")
    groups = pd.factorize(canonical, sort=True)[0]
    for k, (_, test_idx) in enumerate(
        sgkf.split(np.zeros(len(df)), labels, groups=groups)
    ):
        fold[test_idx] = k
    if (fold < 0).any():
        raise SystemExit("Alguna fila no recibio pliegue. No deberia pasar.")
    return fold


def pick_val_select(val_df):
    """Ids de la val de seleccion: estratificada, semilla 42, subconjunto de val."""
    if len(val_df) <= sc.VAL_SELECT_SIZE:
        return set(val_df["id"])
    chosen, _ = train_test_split(
        val_df["id"].to_numpy(),
        train_size=sc.VAL_SELECT_SIZE,
        stratify=val_df["label"].to_numpy(),
        random_state=sc.SEED,
    )
    return set(chosen.tolist())


def class_summary(labels, n_classes: int) -> dict:
    counts = np.bincount(labels, minlength=n_classes)
    total = int(counts.sum())
    return {
        "n": total,
        "counts": counts.tolist(),
        "pct": [round(100.0 * c / total, 3) if total else 0.0 for c in counts],
    }


def main(argv=None) -> int:
    args = parse_args(argv)

    if args.write_reference and args.reference.exists() and not args.force_reference:
        print(
            f"Ya existe {args.reference}. Si de verdad quieres reemplazarla, usa "
            "--force-reference (y avisale al equipo: todos deben coincidir).",
            file=sys.stderr,
        )
        return 1

    df = sc.load_clean(args.clean)
    names = sc.load_label_names(args.labels_json)
    n_classes = len(names)
    unknown = sorted(set(df["label"].unique()) - set(range(n_classes)))
    if unknown:
        raise SystemExit(f"Hay etiquetas que labels.json no tiene: {unknown}")

    fold = assign_folds(df)
    test_mask = fold == sc.TEST_FOLD
    val_mask = fold == sc.VAL_FOLD
    train_mask = ~(test_mask | val_mask)

    select_ids = pick_val_select(df[val_mask])
    select_mask = df["id"].isin(select_ids).to_numpy()

    parts = {
        "train": df[train_mask],
        "val": df[val_mask],
        "test": df[test_mask],
        "val_select": df[select_mask],
    }

    args.splits_dir.mkdir(parents=True, exist_ok=True)
    for name, part in parts.items():
        sc.write_ids(args.splits_dir / f"{name}_ids.csv", part["id"])

    args.labels_out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.labels_out, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("id,label\n")
        for i, l in zip(df["id"], df["label"]):
            handle.write(f"{i},{names[l]}\n")  # run_dummy.py espera el nombre de la clase

    dataset_fp = {
        "rows": int(len(df)),
        "fingerprint_labeled": sc.fingerprint_labeled(df["id"], df["label"]),
        "fingerprint_groups": sc.fingerprint_groups(df["id"], df["group"]),
    }
    split_fp = {
        name: {"n": int(len(part)), "fingerprint": sc.fingerprint_ids(part["id"])}
        for name, part in parts.items()
    }
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
    }

    report = {
        "seed": sc.SEED,
        "n_folds": sc.N_FOLDS,
        "test_fold": sc.TEST_FOLD,
        "val_fold": sc.VAL_FOLD,
        "val_select_size": sc.VAL_SELECT_SIZE,
        "labels": names,
        "dataset": dataset_fp,
        "splits": split_fp,
        "global": class_summary(df["label"].to_numpy(), n_classes),
        "per_split": {
            name: class_summary(part["label"].to_numpy(), n_classes)
            for name, part in parts.items()
        },
        "generated_with": versions,
    }
    (args.splits_dir / "splits_report.json").write_text(
        json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8"
    )

    print(f"Filas: {len(df):,}  ({len(parts['train']):,} train / "
          f"{len(parts['val']):,} val / {len(parts['test']):,} test, "
          f"{len(parts['val_select']):,} en val_select)")
    print(f"Huella del dataset limpio: {dataset_fp['fingerprint_labeled'][:16]}...")
    for name, info in split_fp.items():
        print(f"  {name:<10} n={info['n']:>8,}  huella {info['fingerprint'][:16]}...")

    if args.write_reference:
        reference = {
            "format": 1,
            "seed": sc.SEED,
            "n_folds": sc.N_FOLDS,
            "test_fold": sc.TEST_FOLD,
            "val_fold": sc.VAL_FOLD,
            "val_select_size": sc.VAL_SELECT_SIZE,
            "dataset": dataset_fp,
            "splits": split_fp,
            "generated_with": versions,
        }
        args.reference.parent.mkdir(parents=True, exist_ok=True)
        args.reference.write_text(
            json.dumps(reference, indent=1) + "\n", encoding="utf-8"
        )
        print(f"Referencia escrita en {args.reference}. Esta SI se sube al repo.")
    else:
        print("Corre ahora: python scripts/check_splits.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
