#!/usr/bin/env python3
"""Tablas y figuras agregadas de la Etapa 2. Solo lee results/, no entrena nada.

Uso (desde IA-Proyecto-Etapa2):
    python scripts/make_report.py
    python scripts/make_report.py --out results/report --formats png,pdf

Lee, por modelo y semilla: metrics.json, config.yaml, run_metadata.json y run.log.
No necesita predictions.csv (las matrices de confusion salen de metrics.json) ni GPU.

Escribe en results/report/ (md, tex, json, png y pdf; los .csv los ignora git):
    results_table.md         version en texto de todo: tabla principal, por semilla, por clase,
                             validacion vs test, busqueda de hiperparametros (HPO) y pares (bootstrap)
    table_main.tex, table_perclass.tex, table_hpo.tex, table_pairs.tex     fragmentos booktabs para el articulo
    summary.json             todas las cifras agregadas (fuente unica para el articulo)
    fig_macro_f1, fig_val_curves, fig_confusion, fig_confusion_all, fig_cost_perf   (.png y .pdf)

Convenciones: media y desviacion estandar muestral (ddof=1) sobre las semillas, igual que
evaluate.py aggregate. Los tiempos NO son comparables entre hardware distinto (contrato, sec. 11).
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IGNORE_DIRS = {"hpo", "smoke", "report", "xai", "errors", "throughput", "latency"}

# carpeta de results/ -> (nombre para mostrar, familia visual)
MODEL_INFO = {
    "bert-base": ("BERT-base", "encoder"),
    "distilbert": ("DistilBERT", "encoder"),
    "qwen2.5-0.5b": ("Qwen2.5-0.5B", "decoder"),
    "qwen2.5-1.5b": ("Qwen2.5-1.5B", "decoder"),
    "deepseek-r1-distill-qwen-1.5b": ("DeepSeek-R1-Distill-1.5B", "decoder"),
    "gemma-4-e2b": ("Gemma 4 E2B", "decoder"),
    "bilstm": ("BiLSTM + GloVe", "recurrent"),
    "tfidf-logreg": ("TF-IDF + LogReg", "classical"),
    "dummy_stratified": ("Dummy (stratified)", "trivial"),
    "dummy_most_frequent": ("Dummy (most frequent)", "trivial"),
}
FAMILY_ORDER = ["encoder", "decoder", "recurrent", "classical", "trivial"]
FAMILY_LABEL = {
    "encoder": "Encoder (full fine-tuning)",
    "decoder": "Decoder (QLoRA, 4-bit)",
    "recurrent": "BiLSTM + GloVe",
    "classical": "TF-IDF + LogReg",
    "trivial": "Dummy",
}
# Orden de los paneles de curvas (solo modelos que entrenan por pasos y escriben run.log con evaluaciones)
CURVE_ORDER = ["bert-base", "distilbert", "qwen2.5-0.5b", "qwen2.5-1.5b",
               "deepseek-r1-distill-qwen-1.5b", "gemma-4-e2b", "bilstm"]
CONFUSION_MAIN = ["bert-base", "qwen2.5-1.5b", "gemma-4-e2b", "bilstm", "tfidf-logreg"]

# El arnes reporta n_params contando los pesos de 4 bits (NF4) empaquetados de a dos por byte, asi que
# subestima los decoders (Qwen2.5-1.5B aparece con 0.89B y tiene 1.54B). Para el reporte de costo se
# usa el conteo publicado del modelo base. Comprobado a mano para Qwen: embeddings + pesos lineales/2
# + sesgos + normas + cabeza + adaptadores LoRA reproduce el n_params del arnes casi exacto.
CARD_PARAMS = {
    "qwen2.5-0.5b": (0.49e9, "Qwen2.5 (qwenlm.github.io/blog/qwen2.5-llm): 0.49B"),
    "qwen2.5-1.5b": (1.54e9, "Qwen2.5 (qwenlm.github.io/blog/qwen2.5-llm): 1.54B"),
    "deepseek-r1-distill-qwen-1.5b": (
        1.54e9, "inferido: mismo n_params del arnes que Qwen2.5-1.5B (la ficha publica 1.78B contando un lm_head sin uso) [verificar]"),
    "gemma-4-e2b": (
        5.1e9, "ficha de Gemma 4 citada por apxml.com: 2.3B efectivos, 5.1B con Per-Layer Embeddings [verificar fuente primaria]"),
}

# Tokens de color (paleta de referencia del skill dataviz; las tres primeras ranuras validan all-pairs)
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#ffffff"
FAMILY_STYLE = {
    "encoder": {"color": "#2a78d6", "marker": "o"},
    "decoder": {"color": "#eb6834", "marker": "s"},
    "recurrent": {"color": "#1baf7a", "marker": "^"},
    "classical": {"color": "#898781", "marker": "D"},
    "trivial": {"color": "#c3c2b7", "marker": "x"},
}
BLUE_RAMP = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]  # pasos 100, 250, 400, 550, 700


# ----------------------------------------------------------------------------
# 1. Lectura de results/
# ----------------------------------------------------------------------------
def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_flat_yaml(path: Path) -> dict:
    """config.yaml del arnes: una clave por linea con valor en notacion JSON (sin depender de PyYAML)."""
    cfg: dict = {}
    if not path.exists():
        return cfg
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#") or ":" not in line:
            continue
        key, _, raw = line.partition(":")
        raw = raw.strip()
        try:
            cfg[key.strip()] = json.loads(raw)
        except json.JSONDecodeError:
            cfg[key.strip()] = raw
    return cfg


_EVAL = re.compile(r"^step\s+(\d+)\s+macro_f1_val_select=([0-9.]+)")


def read_curve(path: Path) -> list:
    """[(paso, macro_f1_val_select)] a partir de las lineas 'step N  macro_f1_val_select=X ...' del run.log."""
    pts = []
    if path.exists():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            m = _EVAL.match(line)
            if m:
                pts.append((int(m.group(1)), float(m.group(2))))
    return pts


@dataclass
class Run:
    seed: int
    macro_f1: float
    weighted_f1: float
    accuracy: float
    per_class: dict
    labels: list
    confusion: list
    n: int
    meta: dict
    rmeta: dict
    cfg: dict
    curve: list

    @property
    def time_min(self):
        t = self.meta.get("train_time_s")
        return None if t is None else t / 60.0

    @property
    def vram_gb(self):
        v = self.meta.get("peak_vram_bytes")
        return None if not v else v / 1e9

    @property
    def stopped_early(self):
        """True si paro por paciencia antes del maximo de pasos (epochs x pasos por epoca)."""
        steps, epochs, n_train = self.cfg.get("steps"), self.cfg.get("epochs"), self.cfg.get("n_train")
        eff = self.cfg.get("effective_batch") or self.cfg.get("batch_size")
        if not (steps and epochs and n_train and eff):
            return None
        return steps < epochs * (n_train / eff) - 1

    @property
    def steps_per_epoch(self):
        n_train = self.cfg.get("n_train")
        eff = self.cfg.get("effective_batch") or self.cfg.get("batch_size")
        return (n_train / eff) if (n_train and eff) else 1000.0


def load_run(seed_dir: Path):
    mpath = seed_dir / "metrics.json"
    if not mpath.exists():
        return None
    m = read_json(mpath)
    if m.get("report") != "run" or m.get("split") != "test":
        return None
    cm = m["confusion_matrix"]["matrix"]
    total = sum(sum(row) for row in cm)
    acc = sum(cm[i][i] for i in range(len(cm))) / total
    rmeta_path = seed_dir / "run_metadata.json"
    return Run(
        seed=int(m.get("metadata", {}).get("seed", re.search(r"seed_(\d+)", seed_dir.name).group(1))),
        macro_f1=m["macro_f1"], weighted_f1=m["weighted_f1"], accuracy=acc,
        per_class=m["per_class"], labels=m["labels"], confusion=cm, n=m["n"],
        meta=m.get("metadata", {}), rmeta=read_json(rmeta_path) if rmeta_path.exists() else {},
        cfg=read_flat_yaml(seed_dir / "config.yaml"), curve=read_curve(seed_dir / "run.log"),
    )


class Model:
    def __init__(self, key: str, runs: list):
        self.key = key
        self.name, self.family = MODEL_INFO.get(key, (key, "other"))
        self.runs = sorted(runs, key=lambda r: r.seed)

    def values(self, getter) -> list:
        out = [getter(r) for r in self.runs]
        return [v for v in out if v is not None]

    def stat(self, getter):
        return stats(self.values(getter))

    @property
    def f1(self):
        return stats([r.macro_f1 for r in self.runs])

    @property
    def labels(self):
        return self.runs[0].labels

    def pooled_confusion(self):
        k = len(self.labels)
        return [[sum(r.confusion[i][j] for r in self.runs) for j in range(k)] for i in range(k)]

    @property
    def n_params_harness(self):
        return self.runs[0].meta.get("n_params") or None

    @property
    def n_params_total(self):
        """Conteo para el reporte: el publicado en la ficha si el arnes cuenta pesos de 4 bits empaquetados."""
        if self.key in CARD_PARAMS and self.runs[0].cfg.get("quantization"):
            return CARD_PARAMS[self.key][0]
        return self.n_params_harness

    @property
    def card_params(self) -> bool:
        """True si el conteo de parametros del reporte sale de la ficha del modelo y no del arnes."""
        return self.key in CARD_PARAMS and bool(self.runs[0].cfg.get("quantization"))

    def params_label(self, dash="-") -> str:
        return fmt_params(self.n_params_total, dash=dash, card=self.card_params)

    @property
    def n_trainable(self):
        return self.runs[0].meta.get("n_trainable_params") or None

    @property
    def hardware(self):
        r = self.runs[0]
        device = r.rmeta.get("device")
        gpu = (r.meta.get("gpu") or "").replace("NVIDIA GeForce ", "")
        dtype = r.meta.get("dtype")
        if self.family == "trivial":
            return "CPU"
        if device == "cpu" or (device is None and not r.meta.get("peak_vram_bytes")):
            return "CPU" + (" (fp32)" if dtype == "fp32" else "")
        return f"{gpu} ({dtype})" if dtype else gpu


def stats(values: list) -> dict:
    n = len(values)
    if n == 0:
        return {"values": [], "mean": None, "std": None, "n": 0}
    return {
        "values": [float(v) for v in values],
        "mean": float(statistics.fmean(values)),
        "std": float(statistics.stdev(values)) if n > 1 else None,
        "n": n,
    }


def load_models(results: Path) -> list:
    models = []
    for d in sorted(p for p in results.iterdir() if p.is_dir() and p.name not in IGNORE_DIRS):
        runs = [r for r in (load_run(s) for s in sorted(d.glob("seed_*")) if s.is_dir()) if r]
        if runs:
            models.append(Model(d.name, runs))
    rank = {k: i for i, k in enumerate(MODEL_INFO)}
    models.sort(key=lambda m: rank.get(m.key, 99))
    return models


CONTRACT_PAIRS = [("distilbert", "bert-base"), ("qwen2.5-0.5b", "qwen2.5-1.5b"),
                  ("qwen2.5-1.5b", "gemma-4-e2b"), ("qwen2.5-1.5b", "deepseek-r1-distill-qwen-1.5b")]


def load_pairs(results: Path) -> list:
    """JSON de results/pairs/ (los escribe scripts/compare_pairs.py), en el orden del contrato."""
    out = []
    d = results / "pairs"
    if d.is_dir():
        out = [read_json(f) for f in sorted(d.glob("*.json"))]
    rank = {p: i for i, p in enumerate(CONTRACT_PAIRS)}
    out.sort(key=lambda p: rank.get((p["a"], p["b"]), 99))
    return out


def scalarize(v):
    return v[-1] if isinstance(v, list) else v


def load_hpo(results: Path) -> dict:
    out: dict = {}
    base = results / "hpo"
    if not base.is_dir():
        return out
    for fam_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        trials = []
        for jf in sorted(fam_dir.glob("*.json")):
            d = read_json(jf)
            cfg = read_flat_yaml(fam_dir / d["trial"] / "config.yaml")
            params = {k: v for k, v in d.get("config", {}).items() if k != "name"}
            trials.append({"trial": d["trial"], "model": d.get("model"), "macro_f1": d["macro_f1"],
                           "params": params, "cfg": cfg})
        if trials:
            trials.sort(key=lambda t: -t["macro_f1"])
            out[fam_dir.name] = trials
    return out


def grid_summary(trials: list) -> dict:
    """Valores explorados por hiperparametro y si el ganador cae en un borde de la rejilla."""
    best = trials[0]
    grid, edges = {}, []
    keys = sorted({k for t in trials for k in t["params"]})
    for k in keys:
        vals = [t["params"][k] for t in trials if k in t["params"]]
        uniq = sorted({json.dumps(v) for v in vals})
        if len(uniq) < 2:
            continue
        grid[k] = [json.loads(u) for u in uniq]
        scal = [scalarize(v) for v in vals]
        b = scalarize(best["params"][k])
        if b == max(scal):
            edges.append((k, "max", best["params"][k]))
        elif b == min(scal):
            edges.append((k, "min", best["params"][k]))
    return {"grid": grid, "edges": edges}


# ----------------------------------------------------------------------------
# 2. Resumen agregado
# ----------------------------------------------------------------------------
def build_summary(models: list, hpo: dict) -> dict:
    out: dict = {"models": {}, "hpo": {}}
    for m in models:
        pc = {lab: stats([r.per_class[lab]["f1"] for r in m.runs]) for lab in m.labels}
        out["models"][m.key] = {
            "name": m.name, "family": m.family, "n_seeds": len(m.runs), "seeds": [r.seed for r in m.runs],
            "n_test": m.runs[0].n,
            "macro_f1": m.f1,
            "weighted_f1": stats([r.weighted_f1 for r in m.runs]),
            "accuracy": stats([r.accuracy for r in m.runs]),
            "per_class_f1": pc,
            "val_select_best_macro_f1": m.stat(lambda r: r.cfg.get("best_macro_f1_val_select")),
            "steps": [r.cfg.get("steps") for r in m.runs],
            "stopped_early": [r.stopped_early for r in m.runs],
            "n_params_harness": m.n_params_harness,
            "n_params_report": m.n_params_total,
            "n_params_source": CARD_PARAMS[m.key][1] if (m.key in CARD_PARAMS and m.runs[0].cfg.get("quantization")) else "arnes (n_params)",
            "n_trainable_params": m.n_trainable,
            "peak_vram_gb": m.stat(lambda r: r.vram_gb),
            "train_time_min": m.stat(lambda r: r.time_min),
            "hardware": m.hardware,
            "dtype": m.runs[0].meta.get("dtype"),
            "torch": sorted({r.meta.get("torch_version") for r in m.runs if r.meta.get("torch_version")}),
            "commit": [(r.meta.get("commit") or None) for r in m.runs],
            "n_train": m.runs[0].cfg.get("n_train"),
            "learning_rate": m.runs[0].cfg.get("learning_rate"),
        }
    for fam, trials in hpo.items():
        g = grid_summary(trials)
        c0 = trials[0]["cfg"]
        out["hpo"][fam] = {
            "model": trials[0]["model"], "n_train": c0.get("n_train"), "epochs": c0.get("epochs"),
            "fraction": c0.get("fraction"), "split": "val_select",
            "trials": [{"trial": t["trial"], "macro_f1": t["macro_f1"], "params": t["params"]} for t in trials],
            "best": trials[0]["trial"], "grid": g["grid"],
            "edges": [{"param": k, "edge": e, "value": v} for k, e, v in g["edges"]],
        }
    return out


# ----------------------------------------------------------------------------
# 3. Tablas (md y tex)
# ----------------------------------------------------------------------------
def write_text(path: Path, text: str) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text if text.endswith("\n") else text + "\n")


def f4(x):
    return "-" if x is None else f"{x:.4f}"


def pm(s: dict) -> str:
    if s["mean"] is None:
        return "-"
    return f"{s['mean']:.4f} ± {s['std']:.4f}" if s["std"] is not None else f"{s['mean']:.4f} (1 corrida)"


def fmt_params(n, dash="-", card=False) -> str:
    if not n:
        return dash
    if card:
        return f"{n / 1e9:.3g}B"
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    if n >= 1e6:
        return f"{n / 1e6:.1f}M"
    return f"{n / 1e3:.0f}K"


def fmt_range(s: dict, digits=1) -> str:
    if s["mean"] is None:
        return "-"
    if s["n"] > 1 and max(s["values"]) - min(s["values"]) > 0.1 * s["mean"]:
        return f"{s['mean']:.{digits}f} ({min(s['values']):.{digits}f}–{max(s['values']):.{digits}f})"
    return f"{s['mean']:.{digits}f}"


def fmt_time(s: dict) -> str:
    if s["mean"] is None:
        return "-"
    if s["mean"] < 0.05:
        return "<0.1"
    return fmt_range(s)


def md_table(headers: list, rows: list) -> list:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out


def ranked(models: list) -> list:
    fam_rank = {f: i for i, f in enumerate(FAMILY_ORDER)}
    return sorted(models, key=lambda m: (m.family == "trivial", -m.f1["mean"], fam_rank.get(m.family, 9)))


def build_markdown(models: list, hpo: dict, summary: dict) -> str:
    L: list = []
    S = summary["models"]
    n_test = models[0].runs[0].n
    L += ["# Resultados agregados de la Etapa 2", "",
          "> Archivo generado por `scripts/make_report.py`. No editar a mano: se regenera con "
          "`python scripts/make_report.py`.", ""]

    # --- 1. principal
    L += [f"## 1. Tabla principal (test, {n_test:,} textos)", ""]
    rows = []
    for m in ranked(models):
        s = S[m.key]
        dag = " †" if m.card_params else ""
        rows.append([m.name, FAMILY_LABEL.get(m.family, m.family), pm(s["macro_f1"]), pm(s["accuracy"]),
                     m.params_label() + dag, fmt_params(m.n_trainable),
                     "-" if s["peak_vram_gb"]["mean"] is None else f"{s['peak_vram_gb']['mean']:.1f}",
                     fmt_time(s["train_time_min"]), m.hardware, s["n_seeds"]])
    L += md_table(["Modelo", "Familia", "Macro-F1 test", "Exactitud test", "Params totales",
                   "Params entrenables", "VRAM pico (GB)", "Tiempo/corrida (min)", "Hardware (dtype)", "Corridas"], rows)
    seeds = sorted({r.seed for m in models for r in m.runs})
    L += ["", f"- Macro-F1 y exactitud: media ± desviación estándar muestral (ddof=1) sobre las semillas {seeds}. "
              "La exactitud se calcula desde la matriz de confusión (no es una métrica oficial del arnés).",
          "- TF-IDF y Dummy most-frequent son deterministas: una sola corrida, sin desviación estándar.",
          "- † Los decoders se entrenan con pesos de 4 bits (QLoRA/NF4) y el arnés cuenta cada par de pesos "
          "como un solo elemento, así que su `n_params` subestima (Qwen2.5-1.5B sale 0.89B y tiene 1.54B). "
          "La tabla usa el conteo publicado del modelo base; ver `summary.json` (`n_params_source`) y marcar "
          "las fuentes por verificar antes de citarlas."]
    for m in models:
        if m.card_params:
            L.append(f"  - {m.name}: arnés {m.n_params_harness:,}; reporte {m.params_label()}: {CARD_PARAMS[m.key][1]}")
    L += ["- Tiempos y VRAM: no son comparables entre GPU distintas (contrato, sección 11). VRAM pico es de "
          "entrenamiento (`cuda_max_allocated`), no de inferencia. Los modelos en CPU no tienen VRAM.",
          "- Entre paréntesis, rango de tiempos cuando una corrida difiere más de 10% de la media "
          "(una corrida que paró antes por paciencia dura menos).", ""]
    s_size = S[ranked(models)[0].key]["n_train"]
    big = [m for m in models if m.family == "trivial" and (m.runs[0].cfg.get("n_train") or s_size) != s_size]
    if big:
        L.append("- " + " y ".join(m.name for m in big) + f" se entrenaron con n_train={big[0].runs[0].cfg.get('n_train'):,} "
                 f"(train completo), no con el subconjunto S ({s_size:,}); repetirlos con S para que la comparación sea justa.")
    L.append("")

    # --- 2. por semilla
    L += ["## 2. Resultados por semilla", ""]
    rows = []
    for m in ranked(models):
        for r in m.runs:
            es = r.stopped_early
            rows.append([m.name, r.seed, f4(r.macro_f1), f4(r.cfg.get("best_macro_f1_val_select")),
                         r.cfg.get("steps") or "-", "-" if es is None else ("sí" if es else "no"),
                         "-" if r.time_min is None else f"{r.time_min:.1f}",
                         (str(r.meta.get("commit"))[:7] if r.meta.get("commit") else "no registrado")])
    L += md_table(["Modelo", "Semilla", "Macro-F1 test", "Mejor Macro-F1 val_select", "Pasos", "Paró por paciencia",
                   "Tiempo (min)", "Commit"], rows)
    L += ["", "- `Paró por paciencia`: el entrenamiento terminó antes del máximo de pasos porque val_select no "
              "mejoró en 3 evaluaciones seguidas; se restaura el mejor checkpoint.",
          "- `Commit: no registrado`: la corrida no guardó el hash del commit en sus metadatos (declararlo en la sección de reproducibilidad).", ""]

    # --- 3. por clase
    L += ["## 3. F1 por clase (media de las semillas, test)", ""]
    labels = models[0].labels
    rows = []
    for m in ranked(models):
        pc = S[m.key]["per_class_f1"]
        rows.append([m.name] + [f4(pc[lab]["mean"]) for lab in labels])
    L += md_table(["Modelo"] + list(labels), rows)
    L += ["", f"- Soporte en test: " + ", ".join(f"{lab} {models[0].runs[0].per_class[lab]['support']:,}" for lab in labels) + ".", ""]

    # --- 4. val vs test
    L += ["## 4. Validación vs test", ""]
    rows = []
    for m in ranked(models):
        v = S[m.key]["val_select_best_macro_f1"]
        if v["mean"] is None:
            continue
        t = S[m.key]["macro_f1"]["mean"]
        rows.append([m.name, f4(v["mean"]), f4(t), f"{v['mean'] - t:+.4f}"])
    L += md_table(["Modelo", "Mejor Macro-F1 val_select (media)", "Macro-F1 test (media)", "Diferencia"], rows)
    L += ["", "- val_select es el 10% de train que decide el mejor checkpoint; por eso es algo optimista frente a test "
              "(sesgo de selección). La diferencia debe ser pequeña y de signo positivo; una diferencia grande sería señal de problema.", ""]

    # --- 5. HPO
    L += ["## 5. Búsqueda de hiperparámetros (val_select, una semilla)", ""]
    for fam, h in summary["hpo"].items():
        setup = [f"modelo `{h['model']}`"]
        if h["n_train"]:
            setup.append(f"n_train={h['n_train']:,}")
        if h["epochs"]:
            setup.append(f"{h['epochs']} " + ("época" if h["epochs"] == 1 else "épocas"))
        if h["fraction"]:
            setup.append(f"fracción {h['fraction']} de S")
        L += [f"### {fam}", "", "Configuración: " + ", ".join(setup) + ".", ""]
        rows = []
        for i, t in enumerate(h["trials"]):
            note = "ganador" if i == 0 else ("no convergió" if t["macro_f1"] < 0.5 else "")
            rows.append([("**" + t["trial"] + "**") if i == 0 else t["trial"], f4(t["macro_f1"]), note])
        L += md_table(["Prueba", "Macro-F1 val_select", "Nota"], rows)
        if h["edges"]:
            L += ["", "- El ganador está en el borde de la rejilla: " +
                  ", ".join(f"{e['param']}={e['value']} ({'máximo' if e['edge'] == 'max' else 'mínimo'} explorado)" for e in h["edges"]) +
                  ". No se puede afirmar que sea el óptimo real; declarar como limitación."]
        L.append("")

    # --- 6. pares
    pairs = summary.get("pairs", [])
    nm = lambda k: MODEL_INFO.get(k, (k, ""))[0]
    L += ["## 6. Comparaciones pareadas (bootstrap sobre el test)", ""]
    if pairs:
        conf = int(round(pairs[0]["confidence"] * 100))
        rows = []
        for p in pairs:
            for sd, v in sorted(p["per_seed"].items(), key=lambda kv: int(kv[0])):
                rows.append([f"{nm(p['a'])} − {nm(p['b'])}", sd, f4(v["macro_f1_a"]), f4(v["macro_f1_b"]),
                             f"{v['point_estimate']:+.4f}", f"[{v['ci_low']:+.4f}, {v['ci_high']:+.4f}]",
                             "sí" if v["informative"] else "no (incluye 0)"])
        L += md_table(["Par (A − B)", "Semilla", "Macro-F1 A", "Macro-F1 B", "Δ (A − B)", f"IC {conf}%", "¿Informativa?"], rows)
        rows = []
        for p in pairs:
            sm = p["summary"]
            rows.append([f"{nm(p['a'])} − {nm(p['b'])}", f"{sm['mean_difference']:+.4f}",
                         f"{sm['n_informative']}/{sm['n_seeds']}", "sí" if sm["same_sign"] else "no"])
        L += ["", "Resumen por par:", ""] + md_table(["Par (A − B)", "Δ medio", "Semillas con IC que excluye 0", "Mismo signo en todas"], rows)
        L += ["", f"- Bootstrap pareado por id, {pairs[0]['n_resamples']} remuestreos, IC percentil {conf}% de la diferencia de Macro-F1 "
                  "(contrato, sección 10). Una diferencia es informativa si el IC no incluye 0. Se compara la semilla k de A con la semilla k de B.",
              "- Si el IC excluye 0 con el mismo signo en las tres semillas, hay evidencia de una diferencia sistemática; "
              "si el signo cambia entre semillas o el IC incluye 0, no se puede afirmar que un modelo sea mejor."]
    pend = [f"{nm(a)} − {nm(b)}" for a, b in CONTRACT_PAIRS if not any(p["a"] == a and p["b"] == b for p in pairs)]
    if pend:
        L += ["", "- Pares pendientes (faltan sus JSON en `results/pairs/`; se calculan con `scripts/compare_pairs.py`): " + "; ".join(pend) + "."]
    elif not pairs:
        L += ["Sin comparaciones calculadas."]
    L.append("")

    # --- 7. figuras
    L += ["## 7. Figuras", "",
          "- `fig_macro_f1`: Macro-F1 de test por modelo (puntos = semillas, marca = media, línea = ±1 sd). Los Dummy quedan fuera de escala (ver tabla 1).",
          "- `fig_val_curves`: Macro-F1 en val_select durante el entrenamiento, una línea por semilla; el punto marca el mejor checkpoint.",
          "- `fig_confusion`: matriz de confusión normalizada por clase real (semillas sumadas) de cinco modelos; `fig_confusion_all`, de todos.",
          "- `fig_cost_perf`: Macro-F1 frente a VRAM pico de entrenamiento y frente a parámetros totales.", ""]
    return "\n".join(L)


def tex_pm(s: dict) -> str:
    if s["mean"] is None:
        return "--"
    return f"{s['mean']:.4f} $\\pm$ {s['std']:.4f}" if s["std"] is not None else f"{s['mean']:.4f}"


def tex_escape(t: str) -> str:
    return t.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_").replace("#", r"\#")


def build_tex_main(models: list, summary: dict) -> str:
    S = summary["models"]
    out = [r"% Generado por scripts/make_report.py. No editar a mano.",
           r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
           r"Model & Macro-F1 (test) & Params & VRAM (GB) & Time (min) \\", r"\midrule"]
    prev = None
    for m in ranked(models):
        if prev is not None and m.family != prev:
            out.append(r"\midrule")
        prev = m.family
        s = S[m.key]
        dag = r"$^{\dagger}$" if m.card_params else ""
        vram = "--" if s["peak_vram_gb"]["mean"] is None else f"{s['peak_vram_gb']['mean']:.1f}"
        tm = s["train_time_min"]["mean"]
        tmin = "--" if tm is None else (r"$<$0.1" if tm < 0.05 else f"{tm:.1f}")
        out.append(f"{tex_escape(m.name)} & {tex_pm(s['macro_f1'])} & {m.params_label('--')}{dag} & {vram} & {tmin} \\\\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out)


def build_tex_perclass(models: list, summary: dict) -> str:
    S = summary["models"]
    labels = models[0].labels
    out = [r"% Generado por scripts/make_report.py. No editar a mano.",
           r"\begin{tabular}{@{}l" + "r" * len(labels) + r"@{}}", r"\toprule",
           "Model & " + " & ".join(l.capitalize() for l in labels) + r" \\", r"\midrule"]
    for m in ranked(models):
        pc = S[m.key]["per_class_f1"]
        out.append(f"{tex_escape(m.name)} & " + " & ".join(f4(pc[l]['mean']) for l in labels) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out)


def build_tex_pairs(summary: dict) -> str:
    nm = lambda k: tex_escape(MODEL_INFO.get(k, (k, ""))[0])
    out = [r"% Generado por scripts/make_report.py. No editar a mano.",
           r"\begin{tabular}{@{}lrrc@{}}", r"\toprule",
           r"Pair (A vs. B) & Mean $\Delta$ Macro-F1 & Seeds with CI excluding 0 & Same sign \\", r"\midrule"]
    for p in summary.get("pairs", []):
        sm = p["summary"]
        out.append(f"{nm(p['a'])} vs. {nm(p['b'])} & {sm['mean_difference']:+.4f} & {sm['n_informative']}/{sm['n_seeds']} & "
                   f"{'yes' if sm['same_sign'] else 'no'} \\\\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out)


HPO_FAMILY_TEX = {"encoder": "Encoders", "decoder": "Decoders", "bilstm": "BiLSTM", "tfidf": "TF-IDF"}
HPO_PARAM_TEX = {"learning_rate": "lr", "lora_r": "LoRA $r$", "hidden": "hidden", "dropout": "dropout", "C": "$C$", "ngram": "n-gram"}


def tex_value(k: str, v) -> str:
    if k == "ngram" and isinstance(v, list):
        return f"{v[0]}--{v[1]}"
    if isinstance(v, float) and 0 < abs(v) < 1e-2:
        mant, exp = f"{v:.0e}".split("e")
        return f"${mant}\\times10^{{{int(exp)}}}$"
    return str(v)


def build_tex_hpo(summary: dict) -> str:
    out = [r"% Generado por scripts/make_report.py. No editar a mano.",
           r"\begin{tabular}{@{}llrr@{}}", r"\toprule",
           r"Family (tuned on) & Selected configuration & Trials & Val. Macro-F1 \\", r"\midrule"]
    for fam, h in summary["hpo"].items():
        best = next(t for t in h["trials"] if t["trial"] == h["best"])
        cfg = ", ".join(f"{HPO_PARAM_TEX.get(k, k)} = {tex_value(k, v)}" for k, v in best["params"].items() if k in h["grid"])
        label = HPO_FAMILY_TEX.get(fam, fam)
        if fam in ("encoder", "decoder"):
            label += f" ({tex_escape(MODEL_INFO.get(h['model'], (h['model'], ''))[0])})"
        out.append(f"{label} & {cfg} & {len(h['trials'])} & {best['macro_f1']:.4f} \\\\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out)


# ----------------------------------------------------------------------------
# 4. Figuras (texto en ingles: van al articulo)
# ----------------------------------------------------------------------------
SHORT_NAME = {"deepseek-r1-distill-qwen-1.5b": "DeepSeek-1.5B"}


def setup_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        sys.exit("Falta matplotlib: pip install matplotlib (o ejecute con --no-figures).")
    plt.rcParams.update({
        "font.size": 7.5, "axes.labelsize": 7, "axes.titlesize": 7.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "text.color": INK,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    return plt


def style_axes(ax, grid_axis="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
        ax.spines[s].set_linewidth(0.6)
    ax.tick_params(colors=AXIS, labelcolor=INK2, length=2.5, width=0.6)
    ax.set_axisbelow(True)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.5)


def family_handles(plt, families):
    from matplotlib.lines import Line2D
    hs, ls = [], []
    for f in FAMILY_ORDER:
        if f in families:
            st = FAMILY_STYLE[f]
            hs.append(Line2D([0], [0], marker=st["marker"], color="none", markerfacecolor=st["color"],
                             markeredgecolor=SURFACE, markersize=5.5))
            ls.append(FAMILY_LABEL[f])
    return hs, ls


def save_fig(plt, fig, out: Path, stem: str, formats: list):
    for ext in formats:
        kw = {"dpi": 300, "metadata": {"Software": None}} if ext == "png" else {"metadata": {"CreationDate": None, "Producer": None}}
        fig.savefig(out / f"{stem}.{ext}", bbox_inches="tight", pad_inches=0.04, **kw)
    plt.close(fig)


def fig_macro_f1(plt, models, out, formats):
    rows = sorted([m for m in models if m.family != "trivial"], key=lambda m: -m.f1["mean"])
    n = len(rows)
    fig, ax = plt.subplots(figsize=(3.5, 0.32 * n + 1.0))
    style_axes(ax, "x")
    ys = list(range(n))[::-1]
    jit = {1: [0.0], 2: [-0.1, 0.1], 3: [-0.17, 0.0, 0.17]}
    for y, m in zip(ys, rows):
        st = FAMILY_STYLE[m.family]
        vals = [r.macro_f1 for r in m.runs]
        mean, sd = m.f1["mean"], m.f1["std"]
        if sd is not None:
            ax.plot([mean - sd, mean + sd], [y, y], color=INK2, lw=1.0, solid_capstyle="butt", zorder=2)
        ax.scatter(vals, [y + d for d in jit.get(len(vals), [0.0] * len(vals))], s=17, marker=st["marker"],
                   color=st["color"], edgecolors=SURFACE, linewidths=0.5, zorder=3)
        ax.plot([mean, mean], [y - 0.3, y + 0.3], color=INK, lw=1.1, zorder=4)
        lab = f"{mean:.4f} ± {sd:.4f}" if sd is not None else f"{mean:.4f} (1 run)"
        ax.text(1.02, y, lab, transform=ax.get_yaxis_transform(), va="center", ha="left", fontsize=6.3, color=INK)
    ax.set_yticks(ys)
    ax.set_yticklabels([m.name for m in rows])
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlim(0.945, 0.995)
    ax.set_xticks([0.95, 0.96, 0.97, 0.98, 0.99])
    ax.set_xlabel("Test Macro-F1")
    hs, ls = family_handles(plt, {m.family for m in rows})
    fig.subplots_adjust(left=0.34, right=0.70, bottom=0.25, top=0.99)
    fig.legend(hs, ls, loc="lower center", ncol=2, frameon=False, fontsize=6.3, handletextpad=0.3, columnspacing=1.2,
               bbox_to_anchor=(0.5, 0.045))
    fig.text(0.5, 0.012, "Dots: seeds  |  black bar: mean  |  gray line: ±1 sd", ha="center", va="bottom",
             fontsize=5.8, color=INK2)
    save_fig(plt, fig, out, "fig_macro_f1", formats)


def fig_val_curves(plt, models, out, formats):
    import textwrap
    by = {m.key: m for m in models}
    show = [by[k] for k in CURVE_ORDER if k in by and any(r.curve for r in by[k].runs)]
    ncols = 4
    nrows = -(-(len(show) + 1) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.1, 1.65 * nrows + 0.45), sharex=True, sharey=True, squeeze=False)
    flat = list(axes.flat)
    for idx, (ax, m) in enumerate(zip(flat, show)):
        st = FAMILY_STYLE[m.family]
        style_axes(ax, "y")
        for r in m.runs:
            if not r.curve:
                continue
            spe = r.steps_per_epoch
            xs = [s / spe for s, _ in r.curve]
            ys = [f for _, f in r.curve]
            ax.plot(xs, ys, color=st["color"], lw=1.0, alpha=0.9, solid_joinstyle="round", zorder=2)
            j = max(range(len(ys)), key=lambda i: ys[i])
            ax.scatter([xs[j]], [ys[j]], s=11, color=st["color"], edgecolors=SURFACE, linewidths=0.6, zorder=3)
        ax.set_title(m.name, loc="left", fontsize=7, color=INK)
        ax.set_ylim(0.85, 1.0)
        ax.set_xlim(0, 4.5)
        ax.set_yticks([0.85, 0.90, 0.95, 1.00])
        ax.set_xticks([0, 1, 2, 3, 4])
        if idx + ncols >= len(show):  # no hay panel debajo: se muestra el eje x
            ax.tick_params(labelbottom=True)
            ax.set_xlabel("Epoch")
    note = flat[len(show)]
    note.axis("off")
    clipped = [m.name for m in show if any(f < 0.85 for r in m.runs for _, f in r.curve)]
    lines = ["One line per seed.", "Dot: best checkpoint (the one that is restored).", "1 epoch = 1000 steps over S."]
    if clipped:
        lines.append("Evaluations below 0.85 fall outside the axis (" + ", ".join(clipped) + ").")
    note.text(0.0, 0.97, "\n\n".join(textwrap.fill(t, 32) for t in lines), va="top", ha="left", fontsize=6.3,
              color=INK2, transform=note.transAxes)
    for ax in flat[len(show) + 1:]:
        ax.axis("off")
    for row in axes:
        row[0].set_ylabel("Val-select Macro-F1")
    fig.tight_layout(w_pad=0.6, h_pad=0.8)
    save_fig(plt, fig, out, "fig_val_curves", formats)


def fig_confusion(plt, models, out, formats, keys, stem, ncols):
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap
    by = {m.key: m for m in models}
    show = [by[k] for k in keys if k in by]
    if not show:
        return
    mats = []
    for m in show:
        a = np.array(m.pooled_confusion(), dtype=float)
        mats.append(a / a.sum(axis=1, keepdims=True))
    off_max = max(float(np.max(a - np.diag(np.diag(a)))) for a in mats)
    vmax = max(0.02, float(np.ceil(off_max * 100 / 2) * 2 / 100))
    cmap = LinearSegmentedColormap.from_list("blue_seq", [SURFACE] + BLUE_RAMP)  # 0 = superficie (sin errores)
    cmap.set_bad(SURFACE)
    nrows = -(-(len(show) + 1) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.1, 2.45 * nrows), squeeze=False)
    flat = list(axes.flat)
    labels = show[0].labels
    k = len(labels)
    im = None
    for ax, m, a in zip(flat, show, mats):
        off = np.ma.masked_invalid(np.where(np.eye(k, dtype=bool), np.nan, a))
        im = ax.imshow(off, cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest")
        for i in range(k):
            for j in range(k):
                v = a[i, j]
                if i == j:
                    ax.text(j, i, f"{v * 100:.1f}", ha="center", va="center", fontsize=5.8, color=INK, fontweight="bold")
                elif v >= 0.01:
                    ax.text(j, i, f"{v * 100:.1f}", ha="center", va="center", fontsize=5.5,
                            color="#ffffff" if v / vmax > 0.55 else INK)
        ax.set_xticks(range(k))
        ax.set_yticks(range(k))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)
        ax.set_yticklabels(labels, fontsize=6)
        ax.set_xticks(np.arange(-0.5, k, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, k, 1), minor=True)
        ax.grid(which="minor", color=GRID, linewidth=0.5)
        ax.grid(which="major", visible=False)
        ax.tick_params(which="both", length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        pooled = np.array(m.pooled_confusion())
        ax.set_title(f"{m.name}  (acc {np.trace(pooled) / pooled.sum() * 100:.1f}%)", loc="left", fontsize=7, color=INK)
        ax.set_xlabel("Predicted", fontsize=6.5)
        ax.set_ylabel("True", fontsize=6.5)
    legend_ax = flat[len(show)]
    legend_ax.axis("off")
    cax = legend_ax.inset_axes([0.05, 0.62, 0.9, 0.07])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_ticks([0, vmax / 2, vmax])
    cb.set_ticklabels(["0", f"{vmax * 50:.0f}", f"{vmax * 100:.0f}"])
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=2, labelsize=6, colors=AXIS, labelcolor=INK2)
    legend_ax.text(0.05, 0.50, "Off-diagonal: % of the true class\nmisclassified into that column (cells\n≥ 1% are labeled).",
                   transform=legend_ax.transAxes, va="top", fontsize=6.3, color=INK2)
    legend_ax.text(0.05, 0.25, "Diagonal (bold, no fill): per-class\nrecall in %. Seeds are pooled.",
                   transform=legend_ax.transAxes, va="top", fontsize=6.3, color=INK2)
    for ax in flat[len(show) + 1:]:
        ax.axis("off")
    fig.tight_layout(w_pad=0.8, h_pad=0.6)
    save_fig(plt, fig, out, stem, formats)


# posicion de cada etiqueta en fig_cost_perf: (panel, modelo) -> (dx, dy, alineacion horizontal, vertical), en puntos
COST_LABELS = {
    ("vram", "bert-base"): (0, 5, "center", "bottom"),
    ("vram", "distilbert"): (0, 5, "center", "bottom"),
    ("vram", "qwen2.5-0.5b"): (5, -5, "left", "top"),
    ("vram", "qwen2.5-1.5b"): (6, 3, "left", "bottom"),
    ("vram", "deepseek-r1-distill-qwen-1.5b"): (6, -3, "left", "top"),
    ("vram", "gemma-4-e2b"): (-6, 0, "right", "center"),
    ("params", "bert-base"): (0, 5, "center", "bottom"),
    ("params", "distilbert"): (-6, 0, "right", "center"),
    ("params", "qwen2.5-0.5b"): (4, -5, "right", "top"),
    ("params", "qwen2.5-1.5b"): (6, 3, "left", "bottom"),
    ("params", "deepseek-r1-distill-qwen-1.5b"): (6, -3, "left", "top"),
    ("params", "gemma-4-e2b"): (-6, 0, "right", "center"),
    ("params", "bilstm"): (6, 0, "left", "center"),
    ("params", "tfidf-logreg"): (6, 0, "left", "center"),
}


def fig_cost_perf(plt, models, out, formats):
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    rows = [m for m in models if m.family != "trivial"]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.9), sharey=True)
    panels = [("vram", "Peak training VRAM (GB, log scale)", lambda m: m.stat(lambda r: r.vram_gb)["mean"]),
              ("params", "Total parameters (log scale)", lambda m: m.n_params_total)]
    for ax, (pk, xlabel, getx) in zip(axes, panels):
        style_axes(ax, "y")
        for m in rows:
            x = getx(m)
            if not x:
                continue
            st = FAMILY_STYLE[m.family]
            mean, sd = m.f1["mean"], m.f1["std"]
            if sd is not None:
                ax.plot([x, x], [mean - sd, mean + sd], color=INK2, lw=0.9, zorder=2)
            ax.scatter([x], [mean], s=30, marker=st["marker"], color=st["color"], edgecolors=SURFACE, linewidths=0.7, zorder=3)
            dx, dy, ha, va = COST_LABELS.get((pk, m.key), (6, 3, "left", "bottom"))
            ax.annotate(SHORT_NAME.get(m.key, m.name), (x, mean), xytext=(dx, dy), textcoords="offset points",
                        ha=ha, va=va, fontsize=6.3, color=INK)
        ax.set_xlabel(xlabel)
        ax.set_ylim(0.94, 0.995)
        ax.set_xscale("log")
        ax.xaxis.set_minor_locator(NullLocator())
        if pk == "params":
            ax.xaxis.set_major_locator(FixedLocator([1e6, 1e7, 1e8, 1e9, 1e10]))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: {1e6: "1M", 1e7: "10M", 1e8: "100M", 1e9: "1B", 1e10: "10B"}.get(v, "")))
            ax.set_xlim(2e5, 3.5e10)
        else:
            ax.xaxis.set_major_locator(FixedLocator([1, 2, 5, 10]))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            ax.set_xlim(1.0, 13)
    axes[0].set_ylabel("Test Macro-F1")
    axes[0].text(0.99, 0.02, "BiLSTM and TF-IDF ran on CPU (no GPU memory)", transform=axes[0].transAxes,
                 ha="right", va="bottom", fontsize=5.8, color=INK2)
    hs, ls = family_handles(plt, {m.family for m in rows})
    fig.legend(hs, ls, loc="lower center", ncol=len(ls), frameon=False, fontsize=6.5, handletextpad=0.3, columnspacing=1.5,
               bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.05, 1, 1), w_pad=1.0)
    save_fig(plt, fig, out, "fig_cost_perf", formats)


# ----------------------------------------------------------------------------
# 5. Programa principal
# ----------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Tablas y figuras agregadas de la Etapa 2 (lee results/).")
    ap.add_argument("--results", type=Path, default=ROOT / "results", help="carpeta results/ (por defecto la del repo)")
    ap.add_argument("--out", type=Path, default=None, help="salida (por defecto <results>/report)")
    ap.add_argument("--formats", default="png,pdf", help="formatos de figura separados por coma")
    ap.add_argument("--no-figures", action="store_true", help="solo tablas y summary.json")
    args = ap.parse_args(argv)

    results = args.results.resolve()
    out = (args.out or (results / "report")).resolve()
    out.mkdir(parents=True, exist_ok=True)
    models = load_models(results)
    if not models:
        sys.exit(f"No hay corridas finales (seed_*/metrics.json con split=test) en {results}")
    hpo = load_hpo(results)
    summary = build_summary(models, hpo)
    summary["pairs"] = load_pairs(results)

    write_text(out / "summary.json", json.dumps(summary, indent=2, ensure_ascii=False))
    write_text(out / "results_table.md", build_markdown(models, hpo, summary))
    write_text(out / "table_main.tex", build_tex_main(models, summary))
    write_text(out / "table_perclass.tex", build_tex_perclass(models, summary))
    if hpo:
        write_text(out / "table_hpo.tex", build_tex_hpo(summary))
    if summary["pairs"]:
        write_text(out / "table_pairs.tex", build_tex_pairs(summary))

    if not args.no_figures:
        plt = setup_matplotlib()
        formats = [f.strip() for f in args.formats.split(",") if f.strip()]
        fig_macro_f1(plt, models, out, formats)
        fig_val_curves(plt, models, out, formats)
        fig_confusion(plt, models, out, formats, CONFUSION_MAIN, "fig_confusion", 3)
        fig_confusion(plt, models, out, formats, [m.key for m in models], "fig_confusion_all", 4)
        fig_cost_perf(plt, models, out, formats)

    print(f"Modelos leidos: {len(models)}  ->  {out}")
    for m in ranked(models):
        f = m.f1
        sd = "n/a" if f["std"] is None else f"{f['std']:.4f}"
        print(f"  {m.name:28s} macro-F1 test {f['mean']:.4f} +- {sd}  ({len(m.runs)} corrida(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
