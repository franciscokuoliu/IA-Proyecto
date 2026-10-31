"""Carga textos y etiquetas sin mezclar particiones.

Los pesos de clase se calculan solo con las filas que entran al entrenamiento
(S si existe train_S_ids.csv, si no el train completo).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.harness.catalog import MODELS

ROOT = Path(__file__).resolve().parents[2]
CLEAN = ROOT / "salida" / "cleaned_full_data.csv"
SPLITS = ROOT / "splits"
LABELS_JSON = ROOT / "labels.json"


class HarnessError(Exception):
    """Falta un archivo o la corrida mezclaría particiones."""


@dataclass
class Column:
    ids: list[str]
    texts: list[str]
    y: np.ndarray


def load_label_names(path: Path | None = None) -> list[str]:
    path = path or LABELS_JSON
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    names = data["labels"] if isinstance(data, dict) else data
    names = [str(name) for name in names]
    if len(names) < 2 or len(set(names)) != len(names):
        raise HarnessError(f"{path} no tiene una lista de clases válida.")
    return names


def read_ids(path: Path) -> list[str]:
    if not path.is_file():
        raise HarnessError(f"No existe: {path}")
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].strip() != "id":
        raise HarnessError(f"{path} no empieza con la cabecera 'id'.")
    ids = [line.strip() for line in lines[1:] if line.strip()]
    if len(set(ids)) != len(ids):
        raise HarnessError(f"{path} tiene ids repetidos.")
    if not ids:
        raise HarnessError(f"{path} no tiene filas.")
    return ids


def resolve_train_ids(explicit: Path | None = None) -> tuple[list[str], Path]:
    if explicit is not None:
        return read_ids(explicit), explicit
    subset = SPLITS / "train_S_ids.csv"
    if subset.is_file():
        return read_ids(subset), subset
    full = SPLITS / "train_ids.csv"
    return read_ids(full), full


def load_column(ids: list[str], clean_path: Path | None = None) -> Column:
    path = clean_path or CLEAN
    if not path.is_file():
        raise HarnessError(
            f"No existe {path}. Prepara los datos con scripts/prepare_data.py."
        )
    frame = pd.read_csv(
        path,
        usecols=["id", "text", "label"],
        dtype={"id": str, "text": str},
        keep_default_na=False,
        na_filter=False,
    )
    frame["label"] = frame["label"].astype("int64")
    indexed = frame.drop_duplicates("id").set_index("id")
    missing = [sample_id for sample_id in ids if sample_id not in indexed.index]
    if missing:
        preview = ", ".join(missing[:5])
        raise HarnessError(f"Faltan {len(missing)} ids en el dataset limpio (por ejemplo: {preview}).")
    chosen = indexed.loc[list(ids)]
    return Column(
        ids=list(ids),
        texts=chosen["text"].astype(str).tolist(),
        y=chosen["label"].to_numpy(dtype=np.int64),
    )


def class_weights(y: np.ndarray, n_classes: int) -> np.ndarray:
    """w_c = N / (K * n_c), con N y n_c contados solo en este conjunto."""
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    if np.any(counts == 0):
        absent = [int(index) for index in np.flatnonzero(counts == 0)]
        raise HarnessError(
            f"El conjunto de entrenamiento no tiene ejemplos de las clases {absent}."
        )
    total = float(counts.sum())
    return total / (n_classes * counts)


def stratified_take(column: Column, n: int, seed: int) -> Column:
    """Recorte estratificado. Lo usan la prueba de un paso y la búsqueda."""
    if n >= len(column.ids):
        return column
    present = int(np.unique(column.y).size)
    if n < present:
        raise HarnessError(f"El recorte de {n} filas no alcanza para las {present} clases.")
    rng = np.random.default_rng(seed)
    chosen: list[int] = []
    for label in range(int(column.y.max()) + 1):
        pool = np.flatnonzero(column.y == label)
        if len(pool) == 0:
            continue
        take = max(1, int(round(n * len(pool) / len(column.y))))
        take = min(take, len(pool))
        chosen.extend(rng.choice(pool, size=take, replace=False).tolist())
    chosen_arr = np.asarray(chosen)
    if len(chosen_arr) > n:
        by_label: dict[int, list[int]] = {}
        for index in chosen_arr.tolist():
            by_label.setdefault(int(column.y[index]), []).append(index)
        mandatory: list[int] = []
        extras: list[int] = []
        for indexes in by_label.values():
            rng.shuffle(indexes)
            mandatory.append(indexes[0])
            extras.extend(indexes[1:])
        need = n - len(mandatory)
        if need < len(extras):
            extras = rng.choice(np.asarray(extras), size=need, replace=False).tolist()
        chosen_arr = np.asarray(mandatory + extras)
    return Column(
        ids=[column.ids[index] for index in chosen_arr],
        texts=[column.texts[index] for index in chosen_arr],
        y=column.y[chosen_arr],
    )


def fraction_take(column: Column, fraction: float, seed: int) -> Column:
    n = max(1, int(round(len(column.ids) * fraction)))
    return stratified_take(column, n, seed)


def assert_disjoint(train_ids: list[str], eval_ids: list[str], split: str) -> None:
    overlap = set(train_ids) & set(eval_ids)
    if overlap:
        preview = ", ".join(sorted(overlap)[:5])
        raise HarnessError(
            f"El entrenamiento y {split} comparten {len(overlap)} ids (por ejemplo: {preview})."
        )


def known_model(name: str):
    if name not in MODELS:
        known = ", ".join(MODELS)
        raise HarnessError(f"Modelo desconocido: {name}. Opciones: {known}.")
    return MODELS[name]
