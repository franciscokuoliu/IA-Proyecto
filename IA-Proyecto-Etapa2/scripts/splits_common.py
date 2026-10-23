"""Utilidades compartidas por make_splits.py y check_splits.py.

Aqui viven las constantes del reparto (semilla, pliegues, tamano de la val de
seleccion), las rutas por defecto y las "huellas". Una huella es un SHA-256 de
una forma canonica de los datos (ordenados por id), asi que no depende del
orden de las filas ni de los saltos de linea (CRLF o LF) de cada maquina.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
except ImportError:
    print(
        "Faltan numpy o pandas. Corre .\\setup_env.ps1 desde la raiz del repo.",
        file=sys.stderr,
    )
    raise SystemExit(2)

# IA-Proyecto-Etapa2, sin importar desde donde se ejecute el script.
ROOT = Path(__file__).resolve().parent.parent

# Contrato, secciones 4 y 14. Cambiar cualquiera de estos valores cambia el
# reparto y obliga a regenerar data/expected_hashes.json.
SEED = 42
N_FOLDS = 10  # un pliegue de test, uno de validacion y ocho de entrenamiento
TEST_FOLD = 0
VAL_FOLD = 1
VAL_SELECT_SIZE = 10_000

SPLIT_NAMES = ("train", "val", "test")
ALL_SPLIT_FILES = ("train", "val", "test", "val_select")

DEFAULT_CLEAN = ROOT / "salida" / "cleaned_full_data.csv"
DEFAULT_SPLITS_DIR = ROOT / "splits"
DEFAULT_LABELS_CSV = ROOT / "data" / "labels.csv"
DEFAULT_LABELS_JSON = ROOT / "labels.json"
DEFAULT_REFERENCE = ROOT / "data" / "expected_hashes.json"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fingerprint_ids(ids) -> str:
    """Huella de un conjunto de ids: no depende del orden."""
    return sha256_text("\n".join(sorted(str(i) for i in ids)))


def fingerprint_labeled(ids, labels) -> str:
    """Huella de las parejas (id, label): no depende del orden de las filas."""
    pairs = sorted((str(i), int(l)) for i, l in zip(ids, labels))
    return sha256_text("\n".join(f"{i},{l}" for i, l in pairs))


def fingerprint_groups(ids, groups) -> str:
    """Huella de la estructura de cuasi-duplicados.

    Solo cuentan los grupos de 2 o mas textos, y por sus ids, no por el numero
    de grupo (que es arbitrario y puede cambiar entre corridas del limpiador).
    """
    buckets: dict = {}
    for i, g in zip(ids, groups):
        buckets.setdefault(g, []).append(str(i))
    lines = sorted(",".join(sorted(v)) for v in buckets.values() if len(v) > 1)
    return sha256_text("\n".join(lines))


def normalize_texts(series):
    """Misma normalizacion que usa el limpiador, solo para comparar."""
    return series.str.lower().str.replace(r"\s+", " ", regex=True).str.strip()


def load_clean(path: Path):
    """Lee cleaned_full_data.csv (id, text, label, group) ordenado por id."""
    path = Path(path)
    if not path.is_file():
        raise SystemExit(
            f"No encuentro {path}.\n"
            "Corre antes scripts/dataset_cleaner.py (con --modo agrupar)."
        )
    # keep_default_na=False: textos como "nan" o "null" no deben volverse NaN.
    df = pd.read_csv(
        path, dtype={"id": str, "text": str}, keep_default_na=False, na_filter=False
    )
    missing = {"id", "text", "label", "group"} - set(df.columns)
    if missing:
        raise SystemExit(f"A {path.name} le faltan las columnas: {sorted(missing)}")
    df["label"] = df["label"].astype("int64")
    df["group"] = df["group"].astype("int64")
    if df["id"].duplicated().any():
        raise SystemExit("Hay ids repetidos en el dataset limpio.")
    return df.sort_values("id", kind="mergesort").reset_index(drop=True)


def load_label_names(path: Path) -> list:
    """labels.json es una lista, o un objeto {"labels": [...]}."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    names = data["labels"] if isinstance(data, dict) else data
    return list(names)


def read_ids(path: Path) -> list:
    """Lee un archivo de ids (primera linea 'id')."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].strip() != "id":
        raise SystemExit(f"{path} no empieza con la cabecera 'id'.")
    return [ln.strip() for ln in lines[1:] if ln.strip()]


def write_ids(path: Path, ids) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("id\n")
        for i in ids:
            handle.write(f"{i}\n")
