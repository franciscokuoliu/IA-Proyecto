#!/usr/bin/env python3
"""Analisis de errores a partir de results/<modelo>/seed_*/predictions.csv.

Para cada modelo: errores por clase, pares de confusion, confianza, longitud del texto, cuantas semillas
fallan el mismo texto y ejemplos reales. Con 2 o mas modelos, tambien errores compartidos y exclusivos.
No entrena ni usa GPU. Las predicciones (predictions.csv) estan en .gitignore: para el analisis entre
modelos hay que juntar las de todos en una maquina (scripts/share_predictions.py). Los textos salen de
salida/cleaned_full_data.csv (tambien local). Los Dummy no entran por defecto.

Uso (desde IA-Proyecto-Etapa2):
    python scripts/error_analysis.py --models bert-base qwen2.5-0.5b
    python scripts/error_analysis.py                     # todos los modelos (menos los Dummy) con predictions.csv
Salida en results/errors/: <modelo>.md, cross_model.md (si hay 2+ modelos) y summary.json.

Lo que dice el reporte son hechos medidos. Las lecturas del tipo "posible ruido de etiqueta" son hipotesis:
hay que revisar los ejemplos a mano antes de afirmarlas en el articulo.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CLEAN = ROOT / "salida" / "cleaned_full_data.csv"
LABELS_JSON = ROOT / "labels.json"
NAMES = {
    "bert-base": "BERT-base", "distilbert": "DistilBERT", "qwen2.5-0.5b": "Qwen2.5-0.5B", "qwen2.5-1.5b": "Qwen2.5-1.5B",
    "deepseek-r1-distill-qwen-1.5b": "DeepSeek-R1-Distill-1.5B", "gemma-4-e2b": "Gemma 4 E2B", "bilstm": "BiLSTM + GloVe",
    "tfidf-logreg": "TF-IDF + LogReg",
}
# palabra "feel" en cualquiera de sus formas: marca los tweets de estilo CARER ("i feel ...")
FEEL = re.compile(r"\bfe(?:el|els|eling|elings|lt)\b")
CONF_EDGES = [0.0, 0.5, 0.7, 0.9, 0.99, 1.0000001]
CONF_LABELS = ["< 0.50", "0.50 - 0.70", "0.70 - 0.90", "0.90 - 0.99", ">= 0.99"]


def load_labels() -> list:
    data = json.loads(LABELS_JSON.read_text(encoding="utf-8"))
    return [str(x) for x in (data["labels"] if isinstance(data, dict) else data)]


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, labels: list) -> float:
    f1s = []
    for lab in labels:
        tp = int(np.sum((y_true == lab) & (y_pred == lab)))
        fp = int(np.sum((y_true != lab) & (y_pred == lab)))
        fn = int(np.sum((y_true == lab) & (y_pred != lab)))
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(f1s))


class ModelErrors:
    """Predicciones de un modelo (todas sus semillas) alineadas por id y unidas con el texto."""

    def __init__(self, key: str, frames: dict, labels: list, texts: pd.Series, results: Path):
        self.key, self.labels = key, labels
        self.name = NAMES.get(key, key)
        self.seeds = sorted(frames)
        base = frames[self.seeds[0]].sort_values("id").reset_index(drop=True)
        self.ids = base["id"].to_numpy()
        self.y_true = base["y_true"].to_numpy()
        pcols = [f"p_{lab}" for lab in labels]
        self.pred, self.probs = [], []
        for s in self.seeds:
            df = frames[s].set_index("id").loc[self.ids]
            if not (df["y_true"].to_numpy() == self.y_true).all():
                raise SystemExit(f"{key}: y_true distinto entre semillas (ids desalineados).")
            self.pred.append(df["y_pred"].to_numpy())
            self.probs.append(df[pcols].to_numpy(dtype=float))
        self.pred = np.array(self.pred)                       # (S, n)
        self.probs = np.array(self.probs)                     # (S, n, K)
        self.pmax = self.probs.max(axis=2)                    # confianza en la clase predicha
        self.wrong = self.pred != self.y_true[None, :]        # (S, n)
        self.k_wrong = self.wrong.sum(axis=0)                 # en cuantas semillas falla cada texto
        self.text = texts.reindex(self.ids).to_numpy()
        if pd.isna(self.text).any():
            raise SystemExit(f"{key}: hay ids de predictions.csv sin texto en {CLEAN.name}.")
        self.macro = [macro_f1(self.y_true, self.pred[i], labels) for i in range(len(self.seeds))]
        self.reported = {}
        for s in self.seeds:
            mp = results / key / f"seed_{s}" / "metrics.json"
            if mp.exists():
                self.reported[s] = float(json.loads(mp.read_text(encoding="utf-8"))["macro_f1"])

    @property
    def n(self) -> int:
        return len(self.ids)

    @property
    def majority(self) -> np.ndarray:
        """Textos mal clasificados en la mayoria de las semillas (errores 'tipicos' del modelo)."""
        return self.k_wrong * 2 > len(self.seeds)

    @property
    def persistent(self) -> np.ndarray:
        return self.k_wrong == len(self.seeds)

    def modal_pred(self) -> np.ndarray:
        out = []
        for j in range(self.n):
            vals, cnt = np.unique(self.pred[:, j], return_counts=True)
            out.append(vals[np.argmax(cnt)])
        return np.array(out)


def md_table(headers: list, rows: list) -> list:
    return ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"] + \
           ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]


def pct(x: float, d: int = 1) -> str:
    return f"{x * 100:.{d}f}%"


def quote(text: str, width: int) -> str:
    return textwrap.shorten(str(text), width=width, placeholder=" [...]")


def per_class_table(m: ModelErrors) -> tuple:
    S = len(m.seeds)
    rows, data = [], {}
    n_true = {lab: int(np.sum(m.y_true == lab)) for lab in m.labels}
    for lab in m.labels:
        tp = sum(int(np.sum((m.y_true == lab) & (m.pred[i] == lab))) for i in range(S))
        n_pred = sum(int(np.sum(m.pred[i] == lab)) for i in range(S))
        sup = n_true[lab] * S
        rec = tp / sup if sup else 0.0
        prec = tp / n_pred if n_pred else 0.0
        ratio = n_pred / sup if sup else 0.0
        data[lab] = {"support": n_true[lab], "recall": rec, "precision": prec, "pred_over_true": ratio}
        rows.append([lab, f"{n_true[lab]:,}", pct(rec), pct(prec), f"{ratio:.2f}"])
    return rows, data


def pair_counts(m: ModelErrors) -> list:
    S = len(m.seeds)
    out = []
    for a in m.labels:
        for b in m.labels:
            if a == b:
                continue
            c = sum(int(np.sum((m.y_true == a) & (m.pred[i] == b))) for i in range(S))
            if c:
                out.append((a, b, c))
    out.sort(key=lambda t: -t[2])
    return out


def sample_examples(m: ModelErrors, a: str, b: str, k: int, rng) -> list:
    modal = m.modal_pred()
    cand = np.where((m.y_true == a) & (modal == b) & m.majority)[0]
    if len(cand) == 0:
        return []
    persistent = [j for j in cand if m.persistent[j]]
    pool = np.array(persistent if len(persistent) >= k else list(cand))
    pick = rng.choice(pool, size=min(k, len(pool)), replace=False)
    return [int(j) for j in pick]


def example_line(m: ModelErrors, j: int, width: int) -> str:
    S = len(m.seeds)
    modal = m.modal_pred()[j]
    conf = float(np.mean([m.probs[i, j, m.labels.index(modal)] for i in range(S)]))
    return (f"- \"{quote(m.text[j], width)}\"  \n  real = **{m.y_true[j]}**, predicho = **{modal}** "
            f"(confianza media {conf:.2f}; falla en {int(m.k_wrong[j])}/{S} semillas)")


def model_report(m: ModelErrors, top_pairs: int, n_examples: int, width: int, seed: int, words: np.ndarray) -> tuple:
    S = len(m.seeds)
    rng = np.random.default_rng(seed)
    L = [f"# Análisis de errores: {m.name}", "",
         f"> Generado por `scripts/error_analysis.py` a partir de `predictions.csv` (test, {m.n:,} textos, semillas {m.seeds}). "
         "Los datos son hechos medidos; las lecturas marcadas como hipótesis requieren revisión manual.", ""]
    summ: dict = {"seeds": m.seeds, "n_test": m.n}

    # 1. resumen
    L += ["## 1. Resumen", ""]
    rows = []
    for i, s in enumerate(m.seeds):
        nw = int(m.wrong[i].sum())
        rep = m.reported.get(s)
        ok = "-" if rep is None else ("coincide" if abs(rep - m.macro[i]) < 1e-6 else f"NO coincide ({rep:.6f})")
        rows.append([s, f"{nw:,}", pct(nw / m.n, 2), f"{m.macro[i]:.4f}", ok])
    L += md_table(["Semilla", "Errores", "Tasa de error", "Macro-F1 recalculado", "vs metrics.json"], rows)
    n1 = int(np.sum(m.k_wrong == 1)) if S > 1 else 0
    L += ["", f"- Textos mal clasificados en **todas** las semillas (persistentes): {int(m.persistent.sum()):,} "
              f"({pct(m.persistent.sum() / m.n, 2)} del test)."]
    if S > 1:
        L += [f"- Textos mal clasificados en **una sola** semilla: {n1:,}. Un error que aparece solo en una semilla "
              "suele ser ruido del entrenamiento; uno persistente apunta al dato (ambigüedad o etiqueta dudosa) o a una limitación del modelo.",
              f"- Errores totales (suma de semillas): {int(m.wrong.sum()):,}; de ellos, en textos persistentes: "
              f"{pct(m.wrong[:, m.persistent].sum() / max(1, m.wrong.sum()))}."]
    L.append("")
    summ.update({"errors_per_seed": [int(m.wrong[i].sum()) for i in range(S)], "n_persistent": int(m.persistent.sum()),
                 "n_single_seed": n1, "macro_f1_recomputed": m.macro})

    # 2. por clase
    rows, pc = per_class_table(m)
    L += ["## 2. Por clase (semillas sumadas)", ""] + md_table(["Clase", "Soporte (test)", "Recall", "Precisión", "Predichas / reales"], rows)
    L += ["", "- `Predichas / reales` > 1 significa que el modelo usa esa clase de más (sus errores desembocan ahí); < 1, de menos.", ""]
    summ["per_class"] = pc

    # 3. pares
    pairs = pair_counts(m)
    n_err = int(m.wrong.sum())
    sup = {lab: int(np.sum(m.y_true == lab)) * S for lab in m.labels}
    rows = [[f"{a} → {b}", f"{c:,}", pct(c / sup[a], 2), pct(c / n_err)] for a, b, c in pairs[:top_pairs]]
    L += [f"## 3. Pares de confusión más frecuentes (top {top_pairs})", ""] + \
         md_table(["Real → Predicho", "Errores", "% de la clase real", "% de todos los errores"], rows) + [""]
    summ["top_pairs"] = [{"true": a, "pred": b, "count": c} for a, b, c in pairs[:top_pairs]]

    # 4. confianza
    L += ["## 4. Confianza de la predicción", ""]
    flat_p, flat_w = m.pmax.ravel(), m.wrong.ravel()
    rows, conf = [], []
    for lo, hi, lab in zip(CONF_EDGES[:-1], CONF_EDGES[1:], CONF_LABELS):
        sel = (flat_p >= lo) & (flat_p < hi)
        cnt = int(sel.sum())
        er = float(flat_w[sel].mean()) if cnt else 0.0
        rows.append([lab, f"{cnt:,}", pct(er, 2)])
        conf.append({"bin": lab, "n": cnt, "error_rate": er})
    L += md_table(["Probabilidad de la clase predicha", "Predicciones", "Tasa de error"], rows)
    hi_err = float(np.mean(flat_p[flat_w] >= 0.9)) if flat_w.any() else 0.0
    L += ["", f"- Errores con confianza >= 0.90: {pct(hi_err)} de los errores. Confianza media: aciertos "
              f"{flat_p[~flat_w].mean():.3f}, errores {flat_p[flat_w].mean():.3f}.", ""]
    summ["confidence"] = conf
    summ["share_errors_conf_ge_0.9"] = hi_err

    # 5. longitud
    q = np.quantile(words, [0.25, 0.5, 0.75])
    edges = [-1, q[0], q[1], q[2], 10 ** 9]
    wlen = pd.Series(words, index=m.ids).reindex(m.ids).to_numpy()
    rows, lens = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (wlen > lo) & (wlen <= hi)
        er = float(m.wrong[:, sel].mean()) if sel.any() else 0.0
        name = f"<= {int(hi)}" if lo < 0 else (f"> {int(lo)}" if hi > 10 ** 8 else f"{int(lo) + 1} - {int(hi)}")
        rows.append([name, f"{int(sel.sum()):,}", pct(er, 2)])
        lens.append({"words": name, "n": int(sel.sum()), "error_rate": er})
    L += ["## 5. Longitud del texto (palabras)", ""] + md_table(["Palabras", "Textos (test)", "Tasa de error"], rows)
    L += ["", "- Los rangos son los cuartiles de longitud del test. Una tasa de error que sube con la longitud sugeriría que el "
              "truncamiento a `max_len` o la dilución de la señal importan.", ""]
    summ["length"] = lens

    # 6. tipo de texto
    has_feel = np.array([bool(FEEL.search(str(t))) for t in m.text])
    starts_when = np.array([str(t).startswith("when ") for t in m.text])
    rows, sub = [], []
    for name, sel in (("Contiene feel / feeling / felt", has_feel), ("No contiene feel", ~has_feel),
                      ("Empieza con \"when\"", starts_when)):
        er = float(m.wrong[:, sel].mean()) if sel.any() else 0.0
        share = float(m.majority[sel].sum() / max(1, int(m.majority.sum())))
        rows.append([name, f"{int(sel.sum()):,}", pct(sel.mean(), 2), pct(er, 2), pct(share)])
        sub.append({"subset": name, "n": int(sel.sum()), "error_rate": er, "share_of_typical_errors": share})
    L += ["## 6. Tipo de texto", ""] + md_table(
        ["Subconjunto", "Textos (test)", "% del test", "Tasa de error", "% de los errores típicos del modelo"], rows)
    L += ["", "- Casi todo el conjunto son tweets de estilo CARER (\"i feel ...\"). Los textos sin \"feel\" y los que empiezan con "
              "\"when\" (frases de otro estilo) son una minoría; si concentran una parte desproporcionada de los errores, el "
              "límite del modelo está en esa minoría. Es una observación sobre el conjunto de datos, no una prueba de su origen.", ""]
    summ["text_type"] = sub

    # 7. ejemplos
    L += ["## 7. Ejemplos", "",
          f"Elegidos al azar (semilla {seed}) entre los textos que el modelo falla en la mayoría de las semillas, "
          "preferiendo los persistentes. Los textos están en minúsculas y sin puntuación, como en el conjunto limpio.", ""]
    for a, b, c in pairs[:top_pairs]:
        idx = sample_examples(m, a, b, n_examples, rng)
        if not idx:
            continue
        L += [f"### {a} → {b}", ""] + [example_line(m, j, width) for j in idx] + [""]
    modal = m.modal_pred()
    same = np.array([len(set(m.pred[:, j])) == 1 for j in range(m.n)])
    cand = np.where(m.persistent & same)[0]
    cand = cand[np.argsort(-m.pmax[:, cand].mean(axis=0))][:n_examples * 2]
    if len(cand):
        L += ["### Errores persistentes de mayor confianza", "",
              "Todas las semillas dan la misma clase equivocada con confianza alta. **Hipótesis**: la etiqueta del conjunto es dudosa "
              "o el texto es ambiguo. Hay que leerlos y decidir a mano; no se asume ruido de etiqueta sin revisión.", ""]
        L += [example_line(m, int(j), width) for j in cand] + [""]
        summ["high_confidence_persistent_ids"] = [str(m.ids[int(j)]) for j in cand]
    return "\n".join(L), summ


def cross_model_report(models: list, n_examples: int, width: int) -> tuple:
    keys = [m.key for m in models]
    common = set(models[0].ids)
    for m in models[1:]:
        common &= set(m.ids)
    order = sorted(common)
    pos = [{i: j for j, i in enumerate(m.ids)} for m in models]
    idx = [np.array([p[i] for i in order]) for p in pos]
    maj = np.array([m.majority[ix] for m, ix in zip(models, idx)])      # (M, n)
    y_true = models[0].y_true[idx[0]]
    n = len(order)
    L = ["# Errores compartidos entre modelos", "",
         f"> Generado por `scripts/error_analysis.py`. Modelos: {', '.join(m.name for m in models)}. "
         f"Textos en común: {n:,}. Un texto cuenta como error de un modelo si lo falla en la mayoría de sus semillas.", ""]
    rows = []
    summ: dict = {"models": keys, "n_common": n}
    for i, m in enumerate(models):
        mine = maj[i]
        others = np.delete(maj, i, axis=0)
        excl = int(np.sum(mine & ~others.any(axis=0)))
        rows.append([m.name, f"{int(mine.sum()):,}", pct(mine.mean(), 2), f"{excl:,}",
                     pct(excl / max(1, int(mine.sum())))])
    cons = maj.all(axis=0)
    L += ["## 1. Errores típicos por modelo", ""] + \
         md_table(["Modelo", "Errores típicos", "Tasa", "Solo de este modelo", "% de sus errores que nadie más comete"], rows)
    L += ["", f"- Errores que cometen **todos** los modelos: {int(cons.sum()):,} ({pct(cons.mean(), 2)} del test).",
          "- Un error que solo comete un modelo no es culpa del dato: es una limitación de ese modelo o de su entrenamiento. "
          "Un error que cometen todos apunta al dato.", ""]
    summ["consensus_errors"] = int(cons.sum())
    if len(models) > 1:
        rows = []
        for i, a in enumerate(models):
            row = [a.name]
            for j, b in enumerate(models):
                inter, union = int(np.sum(maj[i] & maj[j])), int(np.sum(maj[i] | maj[j]))
                row.append(f"{inter / union:.2f}" if union else "-")
            rows.append(row)
        L += ["## 2. Solapamiento de errores (Jaccard)", ""] + md_table([""] + [m.name for m in models], rows) + [""]
    # unanimes con la misma clase equivocada
    modal = [m.modal_pred()[ix] for m, ix in zip(models, idx)]
    same = np.array([len({modal[k][j] for k in range(len(models))}) == 1 for j in range(n)])
    pm = np.array([[models[k].probs[:, idx[k][j], :].mean(axis=0).max() for j in range(n)] for k in range(len(models))]) \
        if cons.any() else None
    cand = np.where(cons & same)[0]
    L += ["## 3. Todos los modelos coinciden en la misma clase equivocada", "",
          f"Textos: {len(cand):,}. **Hipótesis**: etiqueta dudosa o texto ambiguo; revisar a mano.", ""]
    if len(cand):
        conf = pm[:, cand].mean(axis=0)
        top = cand[np.argsort(-conf)][:n_examples * 2]
        for j in top:
            text = models[0].text[idx[0][j]]
            L.append(f"- \"{quote(text, width)}\"  \n  real = **{y_true[j]}**, todos predicen **{modal[0][j]}** "
                     f"(confianza media {float(pm[:, j].mean()):.2f})")
        L.append("")
    summ["unanimous_same_wrong_class"] = int(len(cand))

    # 4. donde caen los errores de consenso
    texts0 = models[0].text[idx[0]]
    has_feel = np.array([bool(FEEL.search(str(t))) for t in texts0])
    starts_when = np.array([str(t).startswith("when ") for t in texts0])
    rows, sub = [], []
    for name, sel in (("Contiene feel / feeling / felt", has_feel), ("No contiene feel", ~has_feel),
                      ("Empieza con \"when\"", starts_when)):
        c = int(np.sum(cons & sel))
        rows.append([name, f"{int(sel.sum()):,}", f"{c:,}", pct(c / max(1, int(cons.sum()))), pct(c / max(1, int(sel.sum())), 2)])
        sub.append({"subset": name, "n": int(sel.sum()), "consensus_errors": c})
    L += ["## 4. Dónde caen los errores de consenso", ""] + md_table(
        ["Subconjunto", "Textos (test)", "Errores de consenso", "% de los errores de consenso",
         "Tasa de error de consenso en el subconjunto"], rows)
    L += ["", "- Si una minoría pequeña del test concentra casi todos los errores de consenso, el techo de los modelos se "
              "explica por esa minoría (hipótesis sobre el conjunto de datos) y no por la arquitectura.", ""]
    summ["consensus_by_subset"] = sub

    # 5. muestra para auditoria manual
    audit = np.where(cons)[0]
    if len(audit):
        rng = np.random.default_rng(42)
        pick = sorted(int(j) for j in rng.choice(audit, size=min(30, len(audit)), replace=False))
        L += ["## 5. Muestra para auditoría manual", "",
              f"{len(pick)} textos al azar (semilla 42) entre los {len(audit):,} errores de consenso (los que fallan todos los "
              "modelos, sección 1). Para cada uno, anotar a mano: **A** = etiqueta dudosa o "
              "ambigua, **B** = el modelo se equivoca claramente, **C** = no se puede decidir. El conteo A/B/C es la evidencia "
              "para hablar (o no) de ruido de etiqueta en el artículo.", ""]
        for j in pick:
            wrong = {str(modal[k][j]) for k in range(len(models))} - {str(y_true[j])}
            predicted = " / ".join(sorted(wrong)) or "?"
            L.append(f"- [ ] `{order[j]}` real = **{y_true[j]}**, predicciones erradas = **{predicted}**: "
                     f"\"{quote(models[0].text[idx[0][j]], width)}\"")
        L.append("")
    return "\n".join(L), summ


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Análisis de errores desde predictions.csv.")
    ap.add_argument("--results", type=Path, default=ROOT / "results")
    ap.add_argument("--clean", type=Path, default=CLEAN, help="cleaned_full_data.csv (id,text,label,group)")
    ap.add_argument("--models", nargs="*", default=None, help="carpetas de results/ (por defecto, las que tengan predictions.csv, sin los Dummy)")
    ap.add_argument("--out", type=Path, default=None, help="por defecto <results>/errors")
    ap.add_argument("--top-pairs", type=int, default=5)
    ap.add_argument("--examples", type=int, default=4, help="ejemplos por par de confusión")
    ap.add_argument("--max-chars", type=int, default=220)
    ap.add_argument("--rng", type=int, default=42, help="semilla del muestreo de ejemplos")
    args = ap.parse_args(argv)

    results = args.results.resolve()
    out = (args.out or results / "errors").resolve()
    labels = load_labels()
    if not args.clean.exists():
        sys.exit(f"No existe {args.clean}. Se genera con scripts/prepare_data.py (salida/ es local).")
    clean = pd.read_csv(args.clean, usecols=["id", "text", "label"], dtype={"id": str, "text": str})
    names = clean["label"].map(dict(enumerate(labels)))
    texts = clean.set_index("id")["text"]
    gold = pd.Series(names.to_numpy(), index=clean["id"])
    words = clean.set_index("id")["text"].str.split().str.len()

    keys = args.models
    if keys is None:
        keys = sorted(d.name for d in results.iterdir()
                      if d.is_dir() and d.name not in {"hpo", "smoke", "report", "xai", "errors", "pairs", "throughput"}
                      and not d.name.startswith("dummy") and list(d.glob("seed_*/predictions.csv")))
    if not keys:
        sys.exit(f"No hay predictions.csv en {results}. Corra primero los entrenamientos (o copie las predicciones).")

    out.mkdir(parents=True, exist_ok=True)
    models, summary = [], {}
    for key in keys:
        frames = {}
        for f in sorted((results / key).glob("seed_*/predictions.csv")):
            frames[int(f.parent.name.split("_")[1])] = pd.read_csv(f)
        if not frames:
            print(f"Aviso: {key} no tiene predictions.csv, se omite.")
            continue
        m = ModelErrors(key, frames, labels, texts, results)
        bad = int(np.sum(gold.reindex(m.ids).to_numpy() != m.y_true))
        if bad:
            sys.exit(f"{key}: {bad} etiquetas de predictions.csv no coinciden con {args.clean.name}. Los ids no están alineados.")
        models.append(m)
        w = words.reindex(m.ids).to_numpy()
        md, summ = model_report(m, args.top_pairs, args.examples, args.max_chars, args.rng, w)
        with open(out / f"{key}.md", "w", encoding="utf-8", newline="\n") as fh:
            fh.write(md + "\n")
        summary[key] = summ
        print(f"{m.name}: semillas {m.seeds}, errores por semilla {summ['errors_per_seed']}, persistentes {summ['n_persistent']}")
    if len(models) >= 2:
        md, summ = cross_model_report(models, args.examples, args.max_chars)
        with open(out / "cross_model.md", "w", encoding="utf-8", newline="\n") as fh:
            fh.write(md + "\n")
        summary["cross_model"] = summ
    with open(out / "summary.json", "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(f"Reportes en {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
