"""Verifica que el reparto train/val/test sea valido y que sea el de todo el equipo.

Contrato, seccion 4. Debe pasar en verde antes de entrenar cualquier modelo.
Sale con codigo 1 si alguna comprobacion falla.

Desde IA-Proyecto-Etapa2:

  python scripts/check_splits.py
  python scripts/check_splits.py --require-reference   # falla si no hay referencia

Comprobaciones:
  - los archivos de ids existen y no traen ids repetidos
  - train, val y test no se tocan, y juntos cubren todo el dataset limpio
  - val_select es un subconjunto de val, del tamano esperado
  - ningun texto (normalizado) aparece en dos particiones
  - ningun grupo de cuasi-duplicados esta repartido entre particiones
  - las proporciones por clase de cada particion estan cerca de las globales
  - data/labels.csv coincide con el dataset limpio
  - las huellas coinciden con data/expected_hashes.json (mismos datos que el equipo)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import splits_common as sc

np = sc.np
pd = sc.pd


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Verifica el reparto train/val/test.")
    parser.add_argument("--clean", type=Path, default=sc.DEFAULT_CLEAN)
    parser.add_argument("--labels-json", type=Path, default=sc.DEFAULT_LABELS_JSON)
    parser.add_argument("--splits-dir", type=Path, default=sc.DEFAULT_SPLITS_DIR)
    parser.add_argument("--labels-csv", type=Path, default=sc.DEFAULT_LABELS_CSV)
    parser.add_argument("--reference", type=Path, default=sc.DEFAULT_REFERENCE)
    parser.add_argument(
        "--tol-pp",
        type=float,
        default=0.5,
        help="Desviacion maxima, en puntos porcentuales, de cada clase vs el total.",
    )
    parser.add_argument(
        "--require-reference",
        action="store_true",
        help="Falla si no existe data/expected_hashes.json.",
    )
    return parser.parse_args(argv)


class Report:
    def __init__(self):
        self.failures = 0
        self.warnings = 0

    def ok(self, name, detail=""):
        print(f"  PASS  {name}" + (f"  ({detail})" if detail else ""))

    def fail(self, name, detail=""):
        self.failures += 1
        print(f"  FAIL  {name}" + (f"  -> {detail}" if detail else ""))

    def warn(self, name, detail=""):
        self.warnings += 1
        print(f"  WARN  {name}" + (f"  -> {detail}" if detail else ""))

    def check(self, name, condition, detail_ok="", detail_bad=""):
        if condition:
            self.ok(name, detail_ok)
        else:
            self.fail(name, detail_bad)


def short(h) -> str:
    return (h or "?")[:12]


def main(argv=None) -> int:
    args = parse_args(argv)
    rep = Report()

    print("Cargando dataset limpio y particiones...")
    df = sc.load_clean(args.clean)
    names = sc.load_label_names(args.labels_json)
    n_classes = len(names)

    ids = {}
    for name in sc.ALL_SPLIT_FILES:
        path = args.splits_dir / f"{name}_ids.csv"
        if not path.is_file():
            print(f"  FAIL  falta {path}. Corre scripts/make_splits.py", file=sys.stderr)
            return 1
        ids[name] = sc.read_ids(path)

    print("\n[1] Estructura del reparto")
    for name, lst in ids.items():
        rep.check(f"{name}: sin ids repetidos", len(lst) == len(set(lst)),
                  f"{len(lst):,} ids", "hay ids repetidos")
    sets = {name: set(lst) for name, lst in ids.items()}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        inter = sets[a] & sets[b]
        rep.check(f"{a} y {b} no se intersectan", not inter,
                  "", f"{len(inter):,} ids en comun")
    all_ids = set(df["id"])
    union = sets["train"] | sets["val"] | sets["test"]
    rep.check("train + val + test cubren todo el dataset limpio",
              union == all_ids,
              f"{len(all_ids):,} filas",
              f"faltan {len(all_ids - union):,} y sobran {len(union - all_ids):,}")
    expected_sel = min(sc.VAL_SELECT_SIZE, len(sets["val"]))
    rep.check("val_select es subconjunto de val", sets["val_select"] <= sets["val"],
              "", f"{len(sets['val_select'] - sets['val']):,} ids fuera de val")
    rep.check("val_select tiene el tamano esperado",
              len(sets["val_select"]) == expected_sel,
              f"{expected_sel:,}",
              f"tiene {len(sets['val_select']):,}, se esperaban {expected_sel:,}")

    id_to_split = {}
    for name in sc.SPLIT_NAMES:
        for i in ids[name]:
            id_to_split[i] = name
    df["split"] = df["id"].map(id_to_split).fillna("<ninguna>")

    print("\n[2] Fugas entre particiones")
    norm = sc.normalize_texts(df["text"])
    splits_per_text = df.groupby(norm)["split"].nunique()
    leaked = int((splits_per_text > 1).sum())
    rep.check("ningun texto normalizado esta en dos particiones", leaked == 0,
              "", f"{leaked:,} textos aparecen en mas de una particion")
    splits_per_group = df.groupby("group")["split"].nunique()
    broken = int((splits_per_group > 1).sum())
    rep.check("ningun grupo de cuasi-duplicados esta repartido", broken == 0,
              "", f"{broken:,} grupos partidos entre particiones")

    print("\n[3] Proporciones por clase")
    global_pct = np.bincount(df["label"], minlength=n_classes) / len(df) * 100
    parts = {name: df[df["id"].isin(sets[name])] for name in sc.ALL_SPLIT_FILES}
    for name, part in parts.items():
        if len(part) == 0:
            rep.fail(f"{name}: proporciones", "particion vacia")
            continue
        pct = np.bincount(part["label"], minlength=n_classes) / len(part) * 100
        worst = float(np.abs(pct - global_pct).max())
        cls = names[int(np.abs(pct - global_pct).argmax())]
        rep.check(f"{name}: clases dentro de +-{args.tol_pp} pp del total",
                  worst <= args.tol_pp,
                  f"peor desviacion {worst:.3f} pp ({cls})",
                  f"{cls} se desvia {worst:.3f} pp")

    print("\n[4] labels.csv")
    if not args.labels_csv.is_file():
        rep.fail("labels.csv existe", f"falta {args.labels_csv}. Corre make_splits.py")
    else:
        lab = pd.read_csv(
            args.labels_csv, dtype={"id": str, "label": str}, keep_default_na=False
        )
        codes = lab["label"].map({n: i for i, n in enumerate(names)})
        if codes.isna().any():
            bad = sorted(lab.loc[codes.isna(), "label"].unique())[:5]
            rep.fail("labels.csv usa nombres de clase de labels.json", f"desconocidas: {bad}")
        else:
            same = (
                len(lab) == len(df)
                and sc.fingerprint_labeled(lab["id"], codes)
                == sc.fingerprint_labeled(df["id"], df["label"])
            )
            rep.check("labels.csv coincide con el dataset limpio", same,
                      "", "labels.csv esta desactualizado, regeneralo con make_splits.py")

    print("\n[5] Mismos datos que el resto del equipo")
    local = {
        "dataset": {
            "rows": int(len(df)),
            "fingerprint_labeled": sc.fingerprint_labeled(df["id"], df["label"]),
            "fingerprint_groups": sc.fingerprint_groups(df["id"], df["group"]),
        },
        "splits": {
            name: {"n": len(ids[name]), "fingerprint": sc.fingerprint_ids(ids[name])}
            for name in sc.ALL_SPLIT_FILES
        },
    }
    if not args.reference.is_file():
        msg = f"no existe {args.reference}"
        if args.require_reference:
            rep.fail("referencia de huellas", msg)
        else:
            rep.warn("referencia de huellas",
                     msg + ". Sin ella no hay garantia de que todos tengan lo mismo.")
    else:
        ref = json.loads(args.reference.read_text(encoding="utf-8"))
        exp = ref["dataset"]
        rep.check("dataset limpio: numero de filas", local["dataset"]["rows"] == exp["rows"],
                  f"{exp['rows']:,}",
                  f"local {local['dataset']['rows']:,} vs referencia {exp['rows']:,}")
        for key, label in (("fingerprint_labeled", "dataset limpio: ids y etiquetas"),
                           ("fingerprint_groups", "dataset limpio: grupos de cuasi-duplicados")):
            rep.check(label, local["dataset"][key] == exp[key],
                      short(exp[key]),
                      f"local {short(local['dataset'][key])} vs referencia {short(exp[key])}")
        for name in sc.ALL_SPLIT_FILES:
            e = ref["splits"][name]
            l = local["splits"][name]
            rep.check(f"particion {name}: n y huella",
                      l["n"] == e["n"] and l["fingerprint"] == e["fingerprint"],
                      f"{e['n']:,} ids, huella {short(e['fingerprint'])}",
                      f"local n={l['n']:,} huella {short(l['fingerprint'])} vs "
                      f"referencia n={e['n']:,} huella {short(e['fingerprint'])}")
        report_path = args.splits_dir / "splits_report.json"
        if rep.failures and report_path.is_file():
            mine = json.loads(report_path.read_text(encoding="utf-8")).get("generated_with", {})
            theirs = ref.get("generated_with", {})
            diff = {k: (mine.get(k), theirs.get(k)) for k in theirs if mine.get(k) != theirs.get(k)}
            if diff:
                print("\n  Pista: versiones distintas a las de la referencia "
                      "(local, referencia):")
                for k, (a, b) in diff.items():
                    print(f"    {k}: {a} vs {b}")

    print()
    if rep.failures:
        print(f"RESULTADO: {rep.failures} comprobacion(es) fallaron. No entrenes todavia.")
        return 1
    extra = f" ({rep.warnings} aviso(s))" if rep.warnings else ""
    print(f"RESULTADO: todo en verde{extra}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
