"""Descarga la versión 1 de nelgiriyewithana/emotions y copia text.csv a data/raw.

El CSV no se versiona. Este script y el SHA-256 de data/DATASET_INFO.md sí.
Si Kaggle entrega otro archivo, el script se detiene.

Desde IA-Proyecto-Etapa2:

  python scripts/download_data.py
"""

from __future__ import annotations

import hashlib
import shutil
import sys
from pathlib import Path

try:
    import kagglehub
except ImportError:
    print(
        "Falta kagglehub. Instálalo con: python -m pip install kagglehub",
        file=sys.stderr,
    )
    raise SystemExit(2)

ROOT = Path(__file__).resolve().parent.parent
HANDLE = "nelgiriyewithana/emotions/versions/1"
FILE_NAME = "text.csv"
DESTINATION = ROOT / "data" / "raw" / FILE_NAME
EXPECTED_SHA256 = "5f2254945d620f7013d8b516c1bde76791af54cc6c94984404c136886285b430"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    downloaded = Path(kagglehub.dataset_download(HANDLE))
    source = downloaded / FILE_NAME
    if not source.is_file():
        print(f"La versión 1 no trajo {FILE_NAME}. Encontré: {downloaded}", file=sys.stderr)
        return 1

    digest = sha256(source)
    if digest != EXPECTED_SHA256:
        print(
            "El archivo descargado no coincide con el hash fijado en DATASET_INFO.md.\n"
            f"  esperado: {EXPECTED_SHA256}\n"
            f"  obtenido: {digest}",
            file=sys.stderr,
        )
        return 1

    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    if not DESTINATION.is_file() or sha256(DESTINATION) != EXPECTED_SHA256:
        shutil.copyfile(source, DESTINATION)
    print(f"Archivo: {DESTINATION}")
    print(f"SHA-256: {digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
