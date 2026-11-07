#!/usr/bin/env python3
"""Empaqueta y desempaqueta los predictions.csv para intercambiarlos dentro del equipo.

Los predictions.csv no se versionan (estan en .gitignore), pero el analisis de errores entre modelos
(error_analysis.py) y las comparaciones pareadas (compare_pairs.py) necesitan las predicciones de todos
los modelos en una sola maquina. Este script junta las tuyas en un .zip, con su huella SHA-256, y las
deja en su sitio en la maquina que las recibe.

Uso (desde cualquier carpeta del repo, con el entorno activado):
    python IA-Proyecto-Etapa2/scripts/share_predictions.py pack --out predicciones_jimmy.zip
    python IA-Proyecto-Etapa2/scripts/share_predictions.py pack --models bert-base distilbert --out preds.zip
    python IA-Proyecto-Etapa2/scripts/share_predictions.py list predicciones_jimmy.zip
    python IA-Proyecto-Etapa2/scripts/share_predictions.py unpack predicciones_jimmy.zip

pack:   results/<modelo>/seed_<n>/predictions.csv  ->  zip con las mismas rutas relativas y un MANIFEST.json.
        Los Dummy no se incluyen (son sorteos; se regeneran en un minuto) salvo que se pidan con --models.
unpack: copia los archivos a results/. Nunca sobrescribe un archivo distinto (usa --force para hacerlo) y
        rechaza cualquier ruta que no tenga la forma <modelo>/seed_<n>/predictions.csv.
Solo usa la libreria estandar.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
MANIFEST = "MANIFEST.json"
MEMBER = re.compile(r"^(?P<model>[a-z0-9][a-z0-9._-]*)/seed_(?P<seed>[0-9]+)/predictions\.csv$")
NOT_MODELS = {"hpo", "smoke", "report", "xai", "errors", "pairs", "throughput"}
REQUIRED_COLUMNS = ("id", "y_true", "y_pred")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def check_header(name: str, data: bytes) -> int:
    """Devuelve el numero de filas de datos; falla si el encabezado no es el de evaluate.py."""
    text = data.decode("utf-8")
    lines = text.splitlines()
    header = lines[0].split(",") if lines else []
    missing = [c for c in REQUIRED_COLUMNS if c not in header]
    if missing:
        raise ValueError(f"{name}: faltan las columnas {missing} en el encabezado")
    return len(lines) - 1


def pick_models(results: Path, wanted: list[str] | None) -> list[str]:
    if wanted:
        return wanted
    return sorted(d.name for d in results.iterdir()
                  if d.is_dir() and d.name not in NOT_MODELS and not d.name.startswith("dummy")
                  and any(d.glob("seed_*/predictions.csv")))


def cmd_pack(args: argparse.Namespace) -> int:
    results = args.results.resolve()
    models = pick_models(results, args.models)
    files = []
    for model in models:
        files.extend(sorted((results / model).glob("seed_*/predictions.csv")))
    if not files:
        print(f"No hay predictions.csv en {results}. Primero corre los entrenamientos con --predict-split test.", file=sys.stderr)
        return 1
    out = args.out.resolve()
    manifest = {"created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "files": {}}
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            name = path.relative_to(results).as_posix()
            if not MEMBER.match(name):
                print(f"Aviso: {name} no tiene la forma <modelo>/seed_<n>/predictions.csv, se omite.", file=sys.stderr)
                continue
            data = path.read_bytes()
            rows = check_header(name, data)
            zf.writestr(name, data)
            manifest["files"][name] = {"sha256": sha256(data), "bytes": len(data), "rows": rows}
        zf.writestr(MANIFEST, json.dumps(manifest, indent=2) + "\n")
    print(f"{len(manifest['files'])} archivos en {out} ({out.stat().st_size / 1e6:.1f} MB):")
    for name, info in manifest["files"].items():
        print(f"  {name}  filas={info['rows']}")
    return 0


def read_manifest(zf: zipfile.ZipFile) -> dict:
    try:
        return json.loads(zf.read(MANIFEST).decode("utf-8"))["files"]
    except KeyError:
        raise SystemExit(f"El zip no trae {MANIFEST}; no es un paquete de este script.")


def cmd_list(args: argparse.Namespace) -> int:
    with zipfile.ZipFile(args.zip) as zf:
        files = read_manifest(zf)
    for name, info in files.items():
        print(f"{name}  filas={info['rows']}  sha256={info['sha256'][:12]}")
    print(f"{len(files)} archivos")
    return 0


def cmd_unpack(args: argparse.Namespace) -> int:
    results = args.results.resolve()
    written = same = conflicts = 0
    with zipfile.ZipFile(args.zip) as zf:
        files = read_manifest(zf)
        for name, info in files.items():
            if not MEMBER.match(name):
                print(f"RECHAZADO {name}: la ruta no tiene la forma <modelo>/seed_<n>/predictions.csv", file=sys.stderr)
                conflicts += 1
                continue
            if name not in zf.namelist():
                print(f"RECHAZADO {name}: el manifiesto lo menciona pero el zip no lo trae", file=sys.stderr)
                conflicts += 1
                continue
            data = zf.read(name)
            if sha256(data) != info["sha256"]:
                print(f"RECHAZADO {name}: la huella SHA-256 no coincide (archivo dañado en la transferencia)", file=sys.stderr)
                conflicts += 1
                continue
            check_header(name, data)
            dest = (results / name).resolve()
            if results not in dest.parents:
                print(f"RECHAZADO {name}: sale de results/", file=sys.stderr)
                conflicts += 1
                continue
            if dest.exists():
                if sha256(dest.read_bytes()) == info["sha256"]:
                    same += 1
                    continue
                if not args.force:
                    print(f"CONFLICTO {name}: ya existe y es distinto (usa --force para reemplazarlo)", file=sys.stderr)
                    conflicts += 1
                    continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            written += 1
            print(f"  {name}")
    print(f"Escritos: {written}. Ya estaban iguales: {same}. Rechazados o en conflicto: {conflicts}.")
    return 1 if conflicts else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Empaqueta y desempaqueta predictions.csv entre integrantes.")
    ap.add_argument("--results", type=Path, default=RESULTS, help="carpeta results/ (por defecto la del repo)")
    sub = ap.add_subparsers(dest="command", required=True)
    pack = sub.add_parser("pack", help="juntar los predictions.csv en un .zip")
    pack.add_argument("--out", type=Path, required=True, help="zip de salida (guardalo fuera del repo)")
    pack.add_argument("--models", nargs="*", default=None, help="carpetas de results/ (por defecto todas menos los Dummy)")
    pack.set_defaults(func=cmd_pack)
    lst = sub.add_parser("list", help="ver lo que trae un .zip")
    lst.add_argument("zip", type=Path)
    lst.set_defaults(func=cmd_list)
    unpack = sub.add_parser("unpack", help="copiar un .zip a results/")
    unpack.add_argument("zip", type=Path)
    unpack.add_argument("--force", action="store_true", help="reemplazar archivos distintos que ya existen")
    unpack.set_defaults(func=cmd_unpack)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
