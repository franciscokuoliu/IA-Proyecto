"""Prepara los datos de Etapa 2 con un solo comando.

Pasos (cada uno se omite si su salida ya existe, salvo con --force):
  1. download_data.py   -> data/raw/text.csv (verifica SHA-256)
  2. dataset_cleaner.py -> salida/cleaned_full_data.csv (modo agrupar, PYTHONHASHSEED=0)
  3. make_splits.py     -> splits/*.csv y data/labels.csv
  4. check_splits.py    -> verifica estructura, fugas y huellas contra data/expected_hashes.json

Uso, desde cualquier carpeta:

  python IA-Proyecto-Etapa2/scripts/prepare_data.py              # miembros del equipo
  python IA-Proyecto-Etapa2/scripts/prepare_data.py --write-reference   # solo quien crea la referencia
  python IA-Proyecto-Etapa2/scripts/prepare_data.py --force      # rehace todo desde cero
  python IA-Proyecto-Etapa2/scripts/prepare_data.py --reset      # borra lo generado y termina
  python IA-Proyecto-Etapa2/scripts/prepare_data.py --reset --reset-results   # ademas borra results/
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
RAW = ROOT / "data" / "raw" / "text.csv"
CLEAN = ROOT / "salida" / "cleaned_full_data.csv"
SPLITS_DIR = ROOT / "splits"
LABELS_CSV = ROOT / "data" / "labels.csv"
REFERENCE = ROOT / "data" / "expected_hashes.json"
SPLIT_FILES = ["train_ids.csv", "val_ids.csv", "test_ids.csv", "val_select_ids.csv"]


def run(titulo: str, args: list[str], env_extra: dict[str, str] | None = None) -> None:
    print(f"\n=== {titulo} ===", flush=True)
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    cmd = [sys.executable, *args]
    result = subprocess.run(cmd, cwd=ROOT, env=env)
    if result.returncode != 0:
        print(f"\nFALLO en '{titulo}' (código {result.returncode}). Se detiene el flujo.", file=sys.stderr)
        raise SystemExit(result.returncode)


def reset(include_results: bool, assume_yes: bool) -> None:
    """Borra solo lo regenerable. Nunca toca data/expected_hashes.json ni codigo."""
    targets = [RAW.parent, CLEAN.parent, SPLITS_DIR, LABELS_CSV]
    if include_results:
        targets.append(ROOT / "results")
    existing = [t for t in targets if t.exists()]
    if not existing:
        print("No hay nada que borrar.")
        return
    print("Se borrara:")
    for t in existing:
        print(f"  - {t.relative_to(ROOT)}")
    print("Se conserva: data/expected_hashes.json, scripts, labels.json, contrato.")
    if not assume_yes and input("Escribe 'si' para confirmar: ").strip().lower() != "si":
        print("Cancelado.")
        raise SystemExit(1)
    for t in existing:
        shutil.rmtree(t) if t.is_dir() else t.unlink()
    print("Listo. Vuelve a generar todo con: python IA-Proyecto-Etapa2/scripts/prepare_data.py")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--force", action="store_true", help="Rehace todos los pasos aunque ya existan sus salidas.")
    p.add_argument("--write-reference", action="store_true",
                   help="Escribe data/expected_hashes.json (solo para quien define la referencia).")
    p.add_argument("--force-reference", action="store_true",
                   help="Permite sobrescribir una referencia existente (junto con --write-reference).")
    p.add_argument("--reset", action="store_true",
                   help="Borra data/raw, salida, splits y labels.csv y termina (no ejecuta el flujo).")
    p.add_argument("--reset-results", action="store_true", help="Con --reset, borra tambien results/.")
    p.add_argument("--yes", action="store_true", help="Con --reset, no pide confirmacion.")
    a = p.parse_args()

    if a.reset:
        reset(a.reset_results, a.yes)
        return
    if a.reset_results or a.yes:
        p.error("--reset-results y --yes solo se usan junto con --reset")

    if a.force_reference and not a.write_reference:
        p.error("--force-reference requiere --write-reference")

    if not a.write_reference and not REFERENCE.exists():
        print(f"No existe {REFERENCE.relative_to(ROOT)}. Haz git pull o corre con --write-reference "
              "si eres quien crea la referencia.", file=sys.stderr)
        raise SystemExit(2)

    # 1. Descarga (download_data.py ya verifica el SHA-256 y se detiene si no coincide)
    if a.force or not RAW.exists():
        run("1/4 Descarga", [str(SCRIPTS / "download_data.py")])
    else:
        print("\n=== 1/4 Descarga: omitida (data/raw/text.csv ya existe; use --force para rehacer) ===")

    # 2. Limpieza. PYTHONHASHSEED=0 se fija aqui para que nadie tenga que acordarse.
    if a.force or not CLEAN.exists():
        run("2/4 Limpieza", [str(SCRIPTS / "dataset_cleaner.py"),
                             "--csv", str(RAW), "--out", str(CLEAN.parent), "--modo", "agrupar"],
            {"PYTHONHASHSEED": "0"})
    else:
        print("\n=== 2/4 Limpieza: omitida (salida/cleaned_full_data.csv ya existe; use --force) ===")

    # 3. Particiones
    splits_ok = all((SPLITS_DIR / f).exists() for f in SPLIT_FILES) and LABELS_CSV.exists()
    if a.force or not splits_ok or a.write_reference:
        cmd = [str(SCRIPTS / "make_splits.py")]
        if a.write_reference:
            cmd.append("--write-reference")
            if a.force_reference:
                cmd.append("--force-reference")
        run("3/4 Particiones", cmd)
    else:
        print("\n=== 3/4 Particiones: omitidas (ya existen; use --force) ===")

    # 4. Verificación: siempre se ejecuta
    run("4/4 Verificación", [str(SCRIPTS / "check_splits.py"), "--require-reference"])

    print("\nListo: tus datos son identicos a la referencia del equipo.")


if __name__ == "__main__":
    main()
