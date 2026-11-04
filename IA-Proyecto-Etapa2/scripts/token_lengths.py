"""Longitudes de los textos por tokenizador, para fijar max_len comun.

Contrato, seccion 4. Mide, sobre S, cuantos tokens tiene cada texto con el
tokenizador de cada modelo (el BiLSTM usa palabras separadas por espacios) y
cuanto se recortaria con varios max_len. Hace falta internet la primera vez para
bajar los tokenizadores (son livianos, no baja los pesos).

Desde la raiz del repo, con el entorno activado:

  python IA-Proyecto-Etapa2/scripts/token_lengths.py
  python IA-Proyecto-Etapa2/scripts/token_lengths.py --muestra 5000   # mas rapido

Pega la tabla completa al equipo. La regla: el menor max_len que recorte menos
del 1 % de los textos en TODOS los tokenizadores.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.harness.catalog import MODELS
from src.harness.data import load_column, read_ids, resolve_train_ids

CANDIDATOS = (32, 48, 64, 96, 128)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--muestra", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ids, path = resolve_train_ids()
    print(f"Ids de {path.name}: {len(ids):,}")
    rng = np.random.default_rng(args.seed)
    if len(ids) > args.muestra:
        ids = list(rng.choice(np.array(ids), size=args.muestra, replace=False))
    texts = list(load_column(ids).texts)
    print(f"Textos medidos: {len(texts):,}\n")

    from transformers import AutoTokenizer

    familias: dict[str, str | None] = {"bilstm (palabras)": None}
    for spec in MODELS.values():
        if spec.model_id:
            familias[spec.model_id] = spec.model_id

    filas = []
    for nombre, model_id in familias.items():
        if model_id is None:
            largos = np.array([len(t.split()) for t in texts])
        else:
            try:
                tok = AutoTokenizer.from_pretrained(model_id)
            except Exception as exc:  # modelo con acceso restringido, sin red, etc.
                print(f"  (se omite {nombre}: {type(exc).__name__})")
                continue
            largos = np.array([len(x) for x in tok(texts, add_special_tokens=True)["input_ids"]])
        filas.append((nombre, largos))

    head = f"{'tokenizador':<46}{'p50':>5}{'p95':>5}{'p99':>5}{'max':>6}  " + "".join(
        f"%>{c:<5}" for c in CANDIDATOS
    )
    print(head)
    peor = {c: 0.0 for c in CANDIDATOS}
    for nombre, largos in filas:
        pct = {c: 100.0 * float((largos > c).mean()) for c in CANDIDATOS}
        for c in CANDIDATOS:
            peor[c] = max(peor[c], pct[c])
        print(
            f"{nombre:<46}{int(np.percentile(largos, 50)):>5}{int(np.percentile(largos, 95)):>5}"
            f"{int(np.percentile(largos, 99)):>5}{int(largos.max()):>6}  "
            + "".join(f"{pct[c]:<7.2f}" for c in CANDIDATOS)
        )
    print("\nPeor caso entre tokenizadores (% de textos recortados):")
    for c in CANDIDATOS:
        print(f"  max_len {c:<4} {peor[c]:.2f} %")
    elegido = next((c for c in CANDIDATOS if peor[c] < 1.0), None)
    print(f"\nSugerencia (menor max_len con < 1 % recortado): {elegido}" if elegido
          else "\nNingun candidato baja de 1 %; revisa si hay textos muy largos.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
