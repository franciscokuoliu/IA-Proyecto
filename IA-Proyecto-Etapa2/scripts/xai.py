#!/usr/bin/env python3
"""Explicabilidad (XAI) de los encoders del benchmark: BERT-base y DistilBERT.

Responde a la pregunta de la rubrica (seccion 2.2.4 del enunciado): ¿el modelo mira lo que deberia mirar?
Se aplican dos familias de metodos y, sobre todo, se verifica que lo que muestran sea creible:

  Metodos
    1. Atribucion: Integrated Gradients (Sundararajan et al., 2017) sobre los embeddings de entrada.
       Objetivo = logit de la clase explicada (la predicha). Baseline = [PAD] en cada palabra, dejando
       [CLS] y [SEP] intactos. Integral por cuadratura de Gauss-Legendre (--ig-steps puntos).
    2. Atencion: la fila de [CLS] en la ultima capa (media de cabezas) y attention rollout
       (Abnar y Zuidema, 2020). La atencion se reporta como contraste, no como explicacion
       (Jain y Wallace, 2019; Wiegreffe y Pinter, 2019).
  Verificaciones
    a. Completitud de IG: la suma de atribuciones debe igualar F(x) - F(baseline).
    b. Fidelidad (estilo ERASER, DeYoung et al., 2020): quitar las k palabras mas importantes debe
       bajar la probabilidad de la clase mucho mas que quitar k palabras al azar (comprehensiveness);
       dejar solo esas k palabras debe conservar la decision (sufficiency).
    c. Prueba de cordura (Adebayo et al., 2018): con la cabeza de clasificacion aleatoria y con el modelo
       entero aleatorio, las atribuciones deben cambiar. Si no cambian, el metodo no explica al modelo.
    d. Palabras marcadoras: las palabras que mas distinguen cada clase en el entrenamiento (razon de
       log-odds con prior informativo, Monroe et al., 2008). Se mide si lo que usa el modelo coincide.
  (Verificar las seis referencias antes de citarlas en el articulo.)

Uso (desde IA-Proyecto-Etapa2, con el checkpoint results/<modelo>/seed_<n>/model/trainable.pt, que se
guarda con --save-model y vive en la computadora que entreno la corrida):
    python scripts/xai.py --model bert-base --seed 42
    python scripts/xai.py --model distilbert --seed 42
    python scripts/xai.py --model bert-base --seed 42 --quick    # humo: ~1 minuto, solo para ver que corre
Si falta memoria de GPU, baje --ig-chunk (32 -> 16) o --batch-size (8 -> 4).
Salidas en results/xai/<modelo>/seed_<n>/: xai_report.md, summary.json, explanations.jsonl,
token_stats.csv, examples.html y fig_xai_*.png/pdf. Se versionan; el checkpoint no.
"""
from __future__ import annotations

import argparse
import html
import json
import math
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#ffffff"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
FEEL_WORDS = frozenset({"feel", "feels", "feeling", "feelings", "felt"})
METHODS = ("ig", "rollout", "cls")
METHOD_LABEL = {
    "ig": "Integrated Gradients",
    "rollout": "Attention rollout",
    "cls": "[CLS] attention, last layer",
    "random": "Random words",
}
METHOD_STYLE = {
    "ig": {"color": BLUE, "marker": "o"},
    "rollout": {"color": ORANGE, "marker": "s"},
    "cls": {"color": AQUA, "marker": "^"},
    "random": {"color": MUTED, "marker": "x"},
}


class XaiError(Exception):
    """Falta un archivo, un paquete o los datos no son los esperados."""


# --------------------------------------------------------------------------------------------------
# Datos
# --------------------------------------------------------------------------------------------------
@dataclass
class Item:
    id: str
    text: str
    y_true: int
    y_pred: int  # prediccion registrada en predictions.csv
    group: str  # "sample" | "error"


def load_labels(path: Path) -> list[str]:
    if not path.is_file():
        raise XaiError(f"No existe {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    names = data["labels"] if isinstance(data, dict) else data
    return [str(n) for n in names]


def read_predictions(path: Path, labels: list[str]) -> pd.DataFrame:
    if not path.is_file():
        raise XaiError(f"No existe {path}. Las predicciones solo estan en la computadora que entreno la corrida.")
    df = pd.read_csv(path, dtype={"id": str})
    if not {"id", "y_true", "y_pred"} <= set(df.columns):
        raise XaiError(f"{path} no tiene las columnas id, y_true, y_pred.")
    index = {name: i for i, name in enumerate(labels)}
    for col in ("y_true", "y_pred"):
        unknown = set(df[col].astype(str)) - set(index)
        if unknown:
            raise XaiError(f"{path}: clases desconocidas en {col}: {sorted(unknown)[:5]}")
        df[col + "_i"] = df[col].astype(str).map(index).astype(int)
    return df


def read_ids(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines[1:] if ln.strip()]


def pick_items(df: pd.DataFrame, texts: pd.Series, n_classes: int, n_per_class: int, n_errors: int, seed: int):
    """Muestra estratificada por clase (semilla fija) + errores para el analisis cualitativo."""
    rng = np.random.default_rng(seed)
    y_true, y_pred = df["y_true_i"].to_numpy(), df["y_pred_i"].to_numpy()

    def make(rows, group):
        return [Item(str(df["id"].iloc[r]), str(texts.loc[df["id"].iloc[r]]), int(y_true[r]), int(y_pred[r]), group)
                for r in rows]

    sample_rows = []
    for c in range(n_classes):
        pool = np.flatnonzero(y_true == c)
        if len(pool):
            sample_rows.extend(sorted(rng.choice(pool, size=min(n_per_class, len(pool)), replace=False)))
    wrong = np.flatnonzero(y_true != y_pred)
    error_rows = sorted(rng.choice(wrong, size=min(n_errors, len(wrong)), replace=False)) if len(wrong) else []
    return make(sample_rows, "sample"), make(error_rows, "error"), int(len(wrong))


def marker_sets(train_texts, train_y, n_classes: int, top_n: int, min_count: int = 10, alpha0: float = 500.0):
    """Palabras que mas distinguen cada clase en el entrenamiento (Monroe et al., 2008, prior informativo)."""
    totals, per_class = Counter(), [Counter() for _ in range(n_classes)]
    for text, y in zip(train_texts, train_y):
        words = str(text).split()
        totals.update(words)
        per_class[int(y)].update(words)
    vocab = [w for w, n in totals.items() if n >= min_count]
    if not vocab:
        return [dict() for _ in range(n_classes)]
    tot = np.array([totals[w] for w in vocab], dtype=float)
    prior = alpha0 * tot / tot.sum()
    markers = []
    for c in range(n_classes):
        yc = np.array([per_class[c][w] for w in vocab], dtype=float)
        yr = tot - yc
        nc, nr = yc.sum(), yr.sum()
        delta = np.log((yc + prior) / (nc + alpha0 - yc - prior)) - np.log((yr + prior) / (nr + alpha0 - yr - prior))
        z = delta / np.sqrt(1.0 / (yc + prior) + 1.0 / (yr + prior))
        order = np.argsort(-z)[:top_n]
        markers.append({vocab[i]: float(z[i]) for i in order})
    return markers


# --------------------------------------------------------------------------------------------------
# Modelo
# --------------------------------------------------------------------------------------------------
def require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise XaiError("Falta PyTorch. Active el entorno del proyecto (.venv) antes de correr este script.") from exc
    return torch


def load_encoder(info: dict, ckpt: Path, labels: list[str], device):
    """Reconstruye el encoder como lo hace el arnes y le carga los pesos guardados en trainable.pt."""
    torch = require_torch()
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if info.get("family") != "encoder":
        raise XaiError("Este script explica encoders (BERT-base, DistilBERT). Los decoders con LoRA/NF4 quedan fuera "
                       "del alcance (decision del equipo: contrato, seccion XAI).")
    model_id = info["model_id"]
    tok = AutoTokenizer.from_pretrained(model_id)
    kwargs = {"num_labels": len(labels), "id2label": dict(enumerate(labels)),
              "label2id": {n: i for i, n in enumerate(labels)}}
    try:  # la atencion solo se puede leer con la implementacion "eager"
        model = AutoModelForSequenceClassification.from_pretrained(model_id, attn_implementation="eager", **kwargs)
    except (TypeError, ValueError):
        model = AutoModelForSequenceClassification.from_pretrained(model_id, **kwargs)
    try:
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
    except TypeError:  # torch < 1.13
        state = torch.load(ckpt, map_location="cpu")
    result = model.load_state_dict(state, strict=False)
    params = {n for n, _ in model.named_parameters()}
    if result.unexpected_keys:
        raise XaiError(f"El checkpoint no corresponde a {model_id}: claves inesperadas {result.unexpected_keys[:3]}")
    missing = [k for k in result.missing_keys if k in params]
    if missing:
        raise XaiError(f"El checkpoint no cubre todos los parametros de {model_id}: faltan {missing[:3]}")
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


def randomize_head(model, seed: int = 0) -> dict:
    """Reinicia pooler y cabeza de clasificacion (prueba de cordura). Devuelve los pesos para restaurar."""
    torch = require_torch()
    gen = torch.Generator().manual_seed(seed)
    std = float(getattr(model.config, "initializer_range", 0.02))
    saved = {}
    for name, p in model.named_parameters():
        if any(part in ("classifier", "pre_classifier", "pooler") for part in name.split(".")):
            saved[name] = p.detach().clone()
            with torch.no_grad():
                new = torch.randn(p.shape, generator=gen) * std if p.dim() > 1 else torch.zeros(p.shape)
                p.copy_(new.to(p.device))
    return saved


def restore_params(model, saved: dict) -> None:
    torch = require_torch()
    params = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in saved.items():
            params[name].copy_(value)


# --------------------------------------------------------------------------------------------------
# Explicador
# --------------------------------------------------------------------------------------------------
class Explainer:
    def __init__(self, model, tok, device, max_len: int, ig_steps: int = 32, ig_chunk: int = 32):
        self.torch = require_torch()
        torch = self.torch
        self.model, self.tok, self.device, self.max_len = model, tok, device, max_len
        self.emb = model.get_input_embeddings()
        self.special_ids = sorted({i for i in (tok.cls_token_id, tok.sep_token_id, tok.pad_token_id) if i is not None})
        self.special = torch.tensor(self.special_ids, device=device)
        nodes, weights = np.polynomial.legendre.leggauss(ig_steps)
        self.alphas = torch.tensor((nodes + 1.0) / 2.0, dtype=torch.float32, device=device)
        self.alpha_w = torch.tensor(weights / 2.0, dtype=torch.float32, device=device)
        self.ig_chunk = max(1, ig_chunk)

    # -- entrada ---------------------------------------------------------------------------------
    def tokenize(self, texts):
        return self.tok(list(texts), truncation=True, max_length=self.max_len, padding=True, return_tensors="pt")

    def tensors(self, enc) -> dict:
        return {k: enc[k].to(self.device) for k in ("input_ids", "attention_mask", "token_type_ids") if k in enc}

    def run(self, input_ids=None, attention_mask=None, token_type_ids=None, inputs_embeds=None, attentions=False):
        kw = {"attention_mask": attention_mask}
        if token_type_ids is not None:
            kw["token_type_ids"] = token_type_ids
        if inputs_embeds is not None:
            kw["inputs_embeds"] = inputs_embeds
        else:
            kw["input_ids"] = input_ids
        if attentions:
            kw["output_attentions"] = True
        return self.model(**kw)

    def probabilities(self, t: dict):
        torch = self.torch
        with torch.no_grad():
            logits = self.run(t["input_ids"], t["attention_mask"], t.get("token_type_ids")).logits.float()
        return torch.softmax(logits, dim=-1)

    # -- Integrated Gradients ---------------------------------------------------------------------
    def integrated_gradients(self, t: dict, target):
        """Atribucion por token = suma sobre la dimension del embedding de (x - x0) * gradiente medio.

        Devuelve (atribuciones [B, T], |suma - (F(x) - F(x0))| [B], F(x) - F(x0) [B], clase del texto base [B]).
        """
        torch = self.torch
        ids, mask, tt = t["input_ids"], t["attention_mask"], t.get("token_type_ids")
        batch, length = ids.shape
        target = target.to(self.device)
        with torch.no_grad():
            x = self.emb(ids)
            fixed = torch.isin(ids, self.special) | (mask == 0)
            pad_vec = self.emb.weight[self.tok.pad_token_id]
            x0 = torch.where(fixed.unsqueeze(-1), x, pad_vec.expand_as(x))
            f1 = self.run(attention_mask=mask, token_type_ids=tt, inputs_embeds=x).logits.float()
            f0 = self.run(attention_mask=mask, token_type_ids=tt, inputs_embeds=x0).logits.float()
            gap = (f1 - f0).gather(1, target[:, None])[:, 0]
            base_pred = f0.argmax(1)
        delta = x - x0
        total = torch.zeros_like(x)
        per_pass = max(1, self.ig_chunk // batch)
        with torch.enable_grad():
            for s in range(0, len(self.alphas), per_pass):
                a, w = self.alphas[s:s + per_pass], self.alpha_w[s:s + per_pass]
                n = len(a)
                xi = (x0.unsqueeze(0) + a.view(n, 1, 1, 1) * delta.unsqueeze(0)).reshape(n * batch, length, -1)
                xi = xi.detach().requires_grad_(True)
                out = self.run(attention_mask=mask.repeat(n, 1),
                               token_type_ids=None if tt is None else tt.repeat(n, 1),
                               inputs_embeds=xi).logits.float()
                score = out.gather(1, target.repeat(n)[:, None]).sum()
                (grad,) = torch.autograd.grad(score, xi)
                total += (w.view(n, 1, 1, 1) * grad.reshape(n, batch, length, -1)).sum(0)
        attr = (delta * total).sum(-1)
        return attr, (attr.sum(1) - gap).abs(), gap, base_pred

    # -- Atencion ---------------------------------------------------------------------------------
    def attention(self, t: dict):
        torch = self.torch
        with torch.no_grad():
            out = self.run(t["input_ids"], t["attention_mask"], t.get("token_type_ids"), attentions=True)
        if not out.attentions or out.attentions[0] is None:
            raise XaiError("El modelo no devolvio atenciones. Hay que cargarlo con attn_implementation='eager'.")
        layers = torch.stack([a.float() for a in out.attentions]).mean(2)  # [L, B, T, T], media de cabezas
        cls_last = layers[-1][:, 0, :]
        eye = torch.eye(layers.shape[-1], device=layers.device)
        res = 0.5 * layers + 0.5 * eye
        res = res / res.sum(-1, keepdim=True)
        rollout = res[0]
        for layer in range(1, res.shape[0]):
            rollout = torch.bmm(res[layer], rollout)
        return cls_last, rollout[:, 0, :]

    # -- palabras ---------------------------------------------------------------------------------
    def word_groups(self, enc, i: int):
        """Agrupa los sub-tokens de cada palabra. Devuelve (lista de listas de posiciones, palabras)."""
        ids = enc["input_ids"][i].tolist()
        mask = enc["attention_mask"][i].tolist()
        tokens = self.tok.convert_ids_to_tokens(ids)
        wids = enc.word_ids(i) if getattr(self.tok, "is_fast", False) else None
        groups, words, current = [], [], None
        for j, tid in enumerate(ids):
            if not mask[j] or tid in self.special_ids:
                current = None
                continue
            piece = tokens[j][2:] if tokens[j].startswith("##") else tokens[j]
            if wids is not None:
                new = wids[j] is None or wids[j] != current
                current = wids[j]
            else:
                new = not tokens[j].startswith("##") or not groups
            if new:
                groups.append([j])
                words.append(piece)
            else:
                groups[-1].append(j)
                words[-1] += piece
        return groups, words

    # -- fidelidad --------------------------------------------------------------------------------
    def faithfulness(self, t, groups, scores, probs0, pred, ks, n_random, rng_seeds, mask_np, special_np):
        """Quita (o deja solo) las k palabras mas importantes segun cada metodo y mide el efecto.

        Quitar = poner en 0 la mascara de atencion de sus sub-tokens (las demas posiciones no se mueven).
        Devuelve {(metodo, k): [lista por ejemplo de (caida_quitar, cambia_pred, caida_dejar, se_mantiene)]}.
        """
        variants, meta = [], []
        for i, grp in enumerate(groups):
            n_words = len(grp)
            rng = np.random.default_rng(rng_seeds[i])
            keep_special = mask_np[i] & special_np[i]
            for k in ks:
                if k >= n_words:
                    continue
                plans = [(m, np.argsort(-scores[m][i], kind="stable")[:k]) for m in METHODS]
                plans += [("random", rng.permutation(n_words)[:k]) for _ in range(n_random)]
                for method, order in plans:
                    toks = np.concatenate([grp[w] for w in order]).astype(int)
                    removed = mask_np[i].copy()
                    removed[toks] = False
                    kept = keep_special.copy()
                    kept[toks] = True
                    variants.append((i, removed))
                    variants.append((i, kept))
                    meta.append((i, method, k))
        if not variants:
            return {}
        rows = np.array([v[0] for v in variants])
        masks = np.stack([v[1] for v in variants])
        out_p = self._probs_for_masks(t, rows, masks)
        collected = defaultdict(list)  # (i, metodo, k) -> [(4 numeros)]
        for n, (i, method, k) in enumerate(meta):
            p_rem, p_keep = out_p[2 * n], out_p[2 * n + 1]
            c = int(pred[i])
            collected[(i, method, k)].append((
                probs0[i, c] - p_rem[c], float(p_rem.argmax() != c), probs0[i, c] - p_keep[c], float(p_keep.argmax() == c)))
        result = defaultdict(list)
        for (i, method, k), values in collected.items():
            result[(method, k)].append(tuple(np.mean(values, axis=0)))
        return result

    def _probs_for_masks(self, t, rows, masks, chunk: int = 128):
        torch = self.torch
        outputs = []
        for s in range(0, len(rows), chunk):
            r = torch.as_tensor(rows[s:s + chunk], device=self.device)
            m = torch.as_tensor(masks[s:s + chunk], device=self.device).long()
            tt = t.get("token_type_ids")
            with torch.no_grad():
                logits = self.run(t["input_ids"][r], m, None if tt is None else tt[r]).logits.float()
            outputs.append(torch.softmax(logits, dim=-1).cpu().numpy())
        return np.concatenate(outputs)

    # -- analisis de un conjunto de textos ---------------------------------------------------------
    def analyze(self, items, batch_size, ks, n_random, faith, true_target, seed, label="", log=print):
        torch = self.torch
        records, acc = [], defaultdict(list)
        start = time.time()
        for s in range(0, len(items), batch_size):
            chunk = items[s:s + batch_size]
            enc = self.tokenize([it.text for it in chunk])
            t = self.tensors(enc)
            probs = self.probabilities(t)
            pred = probs.argmax(1)
            ig, conv_abs, gap, base_pred = self.integrated_gradients(t, pred)
            ig_true = None
            if true_target:
                ig_true = self.integrated_gradients(t, torch.tensor([it.y_true for it in chunk], device=self.device))[0]
            cls_last, rollout = self.attention(t)
            special_tok = torch.isin(t["input_ids"], self.special)
            sp_cls = (cls_last * special_tok).sum(1).cpu().numpy()
            sp_roll = (rollout * special_tok).sum(1).cpu().numpy()
            tok_scores = {"ig": ig.cpu().numpy(), "rollout": rollout.cpu().numpy(), "cls": cls_last.cpu().numpy()}
            ig_true_np = None if ig_true is None else ig_true.cpu().numpy()
            probs_np, pred_np = probs.cpu().numpy(), pred.cpu().numpy()
            conv_np, gap_np, base_np = conv_abs.cpu().numpy(), gap.cpu().numpy(), base_pred.cpu().numpy()

            groups, words, word_scores = [], [], {m: [] for m in METHODS}
            true_scores = []
            for i in range(len(chunk)):
                grp, wd = self.word_groups(enc, i)
                groups.append(grp)
                words.append(wd)
                for m in METHODS:
                    word_scores[m].append(np.array([tok_scores[m][i][g].sum() for g in grp]))
                if ig_true_np is not None:
                    true_scores.append(np.array([ig_true_np[i][g].sum() for g in grp]))
            if faith:
                mask_np = t["attention_mask"].cpu().numpy().astype(bool)
                special_np = special_tok.cpu().numpy()
                seeds = [[seed, s + i] for i in range(len(chunk))]
                for key, values in self.faithfulness(t, groups, word_scores, probs_np, pred_np, ks, n_random, seeds,
                                                     mask_np, special_np).items():
                    acc[key].extend(values)
            for i, it in enumerate(chunk):
                records.append({
                    "id": it.id, "group": it.group, "true": it.y_true, "csv_pred": it.y_pred, "pred": int(pred_np[i]),
                    "p_pred": float(probs_np[i, pred_np[i]]), "p_true": float(probs_np[i, it.y_true]),
                    "words": words[i], "ig": word_scores["ig"][i], "rollout": word_scores["rollout"][i],
                    "cls": word_scores["cls"][i], "ig_true": true_scores[i] if true_scores else None,
                    "conv_abs": float(conv_np[i]), "gap": float(gap_np[i]), "base_pred": int(base_np[i]),
                    "special_cls": float(sp_cls[i]), "special_roll": float(sp_roll[i]),
                })
            done = min(s + batch_size, len(items))
            if (s // batch_size) % 10 == 0 or done == len(items):
                log(f"  {label}: {done}/{len(items)} ({time.time() - start:.0f} s)")
        return records, acc

    def ig_word_scores(self, items, targets, batch_size):
        """Solo IG a nivel de palabra (para la prueba de cordura)."""
        torch = self.torch
        out = []
        for s in range(0, len(items), batch_size):
            chunk = items[s:s + batch_size]
            enc = self.tokenize([it.text for it in chunk])
            t = self.tensors(enc)
            ig = self.integrated_gradients(t, torch.tensor(targets[s:s + batch_size], device=self.device))[0]
            ig = ig.cpu().numpy()
            for i in range(len(chunk)):
                grp, _ = self.word_groups(enc, i)
                out.append(np.array([ig[i][g].sum() for g in grp]))
        return out


# --------------------------------------------------------------------------------------------------
# Agregados
# --------------------------------------------------------------------------------------------------
def ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    out = np.empty(len(values))
    out[order] = np.arange(len(values))
    return out


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return float("nan")
    ra, rb = ranks(a), ranks(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def mean_ci(values) -> dict:
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return {"n": 0, "mean": float("nan"), "ci": float("nan")}
    se = v.std(ddof=1) / math.sqrt(len(v)) if len(v) > 1 else float("nan")
    return {"n": int(len(v)), "mean": float(v.mean()), "ci": float(1.96 * se)}


def summarize_faithfulness(acc, ks) -> dict:
    out = {}
    for method in (*METHODS, "random"):
        out[method] = {}
        for k in ks:
            rows = acc.get((method, k), [])
            if not rows:
                continue
            arr = np.array(rows)
            out[method][str(k)] = {
                "n": int(len(arr)),
                "comp_drop": mean_ci(arr[:, 0]), "comp_flip": mean_ci(arr[:, 1]),
                "suff_drop": mean_ci(arr[:, 2]), "suff_keep": mean_ci(arr[:, 3]),
            }
        means = [v["comp_drop"]["mean"] for v in out[method].values()]
        out[method]["aopc"] = float(np.mean(means)) if means else float("nan")
    return out


def share(scores: np.ndarray) -> np.ndarray:
    pos = np.clip(scores, 0, None)
    total = pos.sum()
    return pos / total if total > 0 else np.zeros_like(pos)


def token_stats(records, labels, markers, min_count, stop):
    """Para cada clase predicha, que palabras concentran la atribucion positiva de IG."""
    per_class = {c: defaultdict(lambda: [0, 0.0]) for c in range(len(labels))}
    n_items = Counter()
    for r in records:
        if not len(r["words"]):
            continue
        n_items[r["pred"]] += 1
        for word, sh in zip(r["words"], share(r["ig"])):
            slot = per_class[r["pred"]][word]
            slot[0] += 1
            slot[1] += float(sh)
    rows = []
    for c, table in per_class.items():
        for word, (count, total) in table.items():
            rows.append({
                "class": labels[c], "word": word, "count": count, "mean_share": total / count,
                "total_share": total / max(n_items[c], 1), "marker": bool(word in markers[c]) if markers else False,
                "stopword": word in stop,
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df[df["count"] >= min_count].sort_values(["class", "total_share"], ascending=[True, False])


def cue_stats(records, markers, stop):
    """¿Donde cae la palabra con mas atribucion? Se compara con la tasa esperada si se eligiera al azar."""
    out = {"n": 0}
    hits = {m: [] for m in (*METHODS,)}
    chance_marker, stop_ig, chance_stop = [], [], []
    offsets = Counter()
    agree_top1, rho = {"rollout": [], "cls": []}, {"rollout": [], "cls": []}
    for r in records:
        n = len(r["words"])
        if n < 2:
            continue
        out["n"] += 1
        words = r["words"]
        top = {m: int(np.argmax(r[m])) for m in METHODS}
        if markers:
            mk = markers[r["pred"]]
            for m in METHODS:
                hits[m].append(words[top[m]] in mk)
            chance_marker.append(np.mean([w in mk for w in words]))
        stop_ig.append(words[top["ig"]] in stop)
        chance_stop.append(np.mean([w in stop for w in words]))
        feel = next((j for j, w in enumerate(words) if w in FEEL_WORDS), None)
        if feel is None:
            offsets["no_feel_word"] += 1
        else:
            d = top["ig"] - feel
            offsets["before_feel" if d < 0 else "feel_itself" if d == 0 else f"plus_{min(d, 3)}"] += 1
        for m in ("rollout", "cls"):
            agree_top1[m].append(top[m] == top["ig"])
            v = spearman(r["ig"], r[m])
            if not math.isnan(v):
                rho[m].append(v)
    n = max(out["n"], 1)
    out["marker_hit"] = {m: float(np.mean(v)) for m, v in hits.items() if v}
    out["marker_chance"] = float(np.mean(chance_marker)) if chance_marker else float("nan")
    out["stopword_top1_ig"] = float(np.mean(stop_ig)) if stop_ig else float("nan")
    out["stopword_chance"] = float(np.mean(chance_stop)) if chance_stop else float("nan")
    out["top1_offset_from_feel"] = {k: v / n for k, v in sorted(offsets.items())}
    out["agreement_with_ig"] = {m: {"top1": float(np.mean(agree_top1[m])), "spearman": float(np.mean(rho[m]))
                                     if rho[m] else float("nan")} for m in ("rollout", "cls")}
    return out


def completeness(records) -> dict:
    rel = np.array([r["conv_abs"] / abs(r["gap"]) for r in records if abs(r["gap"]) > 1e-6])
    if not len(rel):
        return {"n": 0}
    return {"n": int(len(rel)), "mean_rel_error": float(rel.mean()), "median_rel_error": float(np.median(rel)),
            "p95_rel_error": float(np.percentile(rel, 95)),
            "baseline_same_class": float(np.mean([r["base_pred"] == r["pred"] for r in records]))}


def attention_stats(records) -> dict:
    return {"special_mass_cls_last": float(np.mean([r["special_cls"] for r in records])),
            "special_mass_rollout": float(np.mean([r["special_roll"] for r in records]))}


def sanity_check(expl, records, items, batch_size, labels, sanity_n, log=print) -> dict:
    """Atribuciones del modelo entrenado vs. cabeza aleatoria vs. modelo aleatorio (Adebayo et al., 2018)."""
    if sanity_n <= 0:
        return {}
    torch = require_torch()
    subset = list(range(0, len(items), max(1, len(items) // sanity_n)))[:sanity_n]
    sub_items = [items[i] for i in subset]
    targets = [records[i]["pred"] for i in subset]
    base = [records[i]["ig"] for i in subset]
    out = {"n": len(subset)}

    def compare(variant):
        scores = expl.ig_word_scores(sub_items, targets, batch_size)
        rho = [spearman(a, b) for a, b in zip(base, scores)]
        rho = [v for v in rho if not math.isnan(v)]
        same = [int(np.argmax(a) == np.argmax(b)) for a, b in zip(base, scores)]
        return {"spearman_mean": float(np.mean(rho)) if rho else float("nan"), "top1_same": float(np.mean(same))}

    log("  prueba de cordura: cabeza aleatoria")
    saved = randomize_head(expl.model)
    out["random_head"] = compare("head")
    restore_params(expl.model, saved)
    log("  prueba de cordura: modelo aleatorio")
    from transformers import AutoModelForSequenceClassification
    torch.manual_seed(0)
    random_model = AutoModelForSequenceClassification.from_config(expl.model.config).to(expl.device).eval()
    for p in random_model.parameters():
        p.requires_grad_(False)
    original = expl.model
    expl.model, expl.emb = random_model, random_model.get_input_embeddings()
    try:
        out["random_model"] = compare("model")
    finally:
        expl.model, expl.emb = original, original.get_input_embeddings()
        del random_model
    return out


# --------------------------------------------------------------------------------------------------
# Salidas
# --------------------------------------------------------------------------------------------------
def fmt(v, digits=3, pct=False):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/d"
    return f"{v * 100:.1f}%" if pct else f"{v:.{digits}f}"


def top_words_md(df, labels, n=8) -> str:
    if df.empty:
        return "_No hay palabras con suficientes apariciones (baje --min-count)._\n"
    lines = ["| Clase predicha | Palabras con mas atribucion (aparece en N textos; * = palabra marcadora del entrenamiento) |", "|---|---|"]
    for label in labels:
        sub = df[df["class"] == label].head(n)
        cells = ", ".join(f"{r.word}{'*' if r.marker else ''} ({r.count})" for r in sub.itertuples())
        lines.append(f"| {label} | {cells or 'n/d'} |")
    return "\n".join(lines) + "\n"


def write_jsonl(records, labels, path: Path) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for r in records:
            row = {"id": r["id"], "group": r["group"], "true": labels[r["true"]], "pred": labels[r["pred"]],
                   "csv_pred": labels[r["csv_pred"]], "p_pred": round(r["p_pred"], 4), "words": r["words"]}
            for m in METHODS:
                row[m] = [round(float(v), 5) for v in r[m]]
            if r["ig_true"] is not None:
                row["ig_true"] = [round(float(v), 5) for v in r["ig_true"]]
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _chip_color(value: float, scale: float):
    """Blanco -> azul si apoya la clase, blanco -> naranja si la contradice (intensidad relativa al maximo)."""
    t = 0.0 if scale <= 0 else min(abs(value) / scale, 1.0)
    base = (42, 120, 214) if value >= 0 else (235, 104, 52)
    rgb = tuple(int(255 + (c - 255) * t) for c in base)
    return rgb, t


def write_html(records, labels, per_class, path: Path, title: str) -> None:
    def row(name, words, scores):
        scale = float(np.max(np.abs(scores))) if len(scores) else 0.0
        chips = []
        for w, v in zip(words, scores):
            (r, g, b), t = _chip_color(float(v), scale)
            color = "#ffffff" if t > 0.6 else "#0b0b0b"
            chips.append(f'<span class="w" style="background:rgb({r},{g},{b});color:{color}" '
                         f'title="{float(v):+.4f}">{html.escape(w)}</span>')
        return f'<div class="row"><span class="tag">{name}</span>{" ".join(chips)}</div>'

    chosen, seen = [], Counter()
    for r in records:
        if r["group"] == "sample":
            if seen[r["true"]] < per_class:
                seen[r["true"]] += 1
                chosen.append(r)
        else:
            chosen.append(r)
    blocks = []
    for r in chosen:
        ok = r["true"] == r["pred"]
        head = (f'<div class="head"><b>{"correct" if ok else "ERROR"}</b> &middot; true: {labels[r["true"]]} &middot; '
                f'predicted: {labels[r["pred"]]} (p={r["p_pred"]:.3f}) &middot; <code>{r["id"]}</code></div>')
        body = [row("IG (pred.)", r["words"], r["ig"])]
        if r["ig_true"] is not None and not ok:
            body.append(row("IG (true)", r["words"], r["ig_true"]))
        body.append(row("rollout", r["words"], r["rollout"]))
        body.append(row("[CLS] attn.", r["words"], r["cls"]))
        blocks.append(f'<section class="card{"" if ok else " bad"}">{head}{"".join(body)}</section>')
    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
:root {{ color-scheme: light; --ink:#0b0b0b; --ink2:#52514e; --grid:#e1e0d9; }}
body {{ font-family: system-ui, -apple-system, "Segoe UI", sans-serif; background:#f9f9f7; color:var(--ink);
       margin:0 auto; max-width:60rem; padding:1rem 16px 3rem; line-height:1.45; }}
h1 {{ font-size:1.25rem; }} p.note {{ color:var(--ink2); font-size:.9rem; }}
.card {{ background:#fff; border:1px solid var(--grid); border-radius:6px; padding:.6rem .8rem; margin:.7rem 0; }}
.card.bad {{ border-left:4px solid #eb6834; }}
.head {{ font-size:.85rem; color:var(--ink2); margin-bottom:.35rem; }}
.row {{ display:flex; flex-wrap:wrap; gap:3px; align-items:center; margin:.15rem 0; }}
.tag {{ width:5.6rem; font-size:.75rem; color:var(--ink2); flex:none; }}
.w {{ padding:1px 5px; border-radius:3px; font-size:.9rem; }}
</style></head><body>
<h1>{html.escape(title)}</h1>
<p class="note">Blue = pushes toward the explained class; orange = pushes against it. Intensity is relative to the
largest absolute value in each row (rows are not comparable with each other). Hover for the raw value.
IG = Integrated Gradients, rollout = attention rollout, [CLS] attn. = last-layer attention from [CLS].</p>
{"".join(blocks)}
</body></html>
"""
    path.write_text(doc, encoding="utf-8", newline="\n")


def setup_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    plt.rcParams.update({
        "font.size": 7.5, "axes.labelsize": 7, "axes.titlesize": 7.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "text.color": INK, "pdf.fonttype": 42, "ps.fonttype": 42,
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


def save_figure(fig, out_dir: Path, name: str) -> None:
    fig.savefig(out_dir / f"{name}.png", dpi=300, metadata={"Software": None})
    fig.savefig(out_dir / f"{name}.pdf", metadata={"Creator": None, "Producer": None, "CreationDate": None})


def fig_faithfulness(plt, faith: dict, ks, out_dir: Path, model_name: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.7))
    panels = [("comp_drop", "Remove the top-k words", "Drop in class probability (higher = more faithful)"),
              ("suff_drop", "Keep only the top-k words", "Drop in class probability (lower = more faithful)")]
    for ax, (key, title, ylabel) in zip(axes, panels):
        style_axes(ax)
        dodge = {"ig": -0.09, "rollout": -0.03, "cls": 0.03, "random": 0.09}  # evita que se tapen series iguales
        for method in ("random", "cls", "rollout", "ig"):
            xs = [k for k in ks if str(k) in faith[method]]
            if not xs:
                continue
            st = METHOD_STYLE[method]
            ys = np.array([faith[method][str(k)][key]["mean"] for k in xs])
            ci = np.array([faith[method][str(k)][key]["ci"] for k in xs])
            ax.errorbar(np.array(xs) + dodge[method], ys, yerr=np.nan_to_num(ci), color=st["color"], lw=1.3 if method == "ig" else 1.0,
                        marker=st["marker"], ms=4.2, mew=1.0, capsize=1.5, elinewidth=0.7,
                        label=METHOD_LABEL[method], zorder=3 if method == "ig" else 2)
        ax.set_xticks(list(ks))
        ax.set_xlabel("k (number of words)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", color=INK)
        ax.set_ylim(bottom=min(0, ax.get_ylim()[0]))
    handles, labels = axes[0].get_legend_handles_labels()
    order = [labels.index(METHOD_LABEL[m]) for m in ("ig", "rollout", "cls", "random") if METHOD_LABEL[m] in labels]
    fig.legend([handles[i] for i in order], [labels[i] for i in order], loc="lower center", ncol=4, frameon=False,
               fontsize=6.5, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Faithfulness of explanations, {model_name} (mean over test texts, 95% CI; series offset slightly in x)",
                 x=0.01, ha="left", fontsize=7.5, color=INK)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    save_figure(fig, out_dir, "fig_xai_faithfulness")
    plt.close(fig)


def fig_top_words(plt, df, labels, out_dir: Path, model_name: str, n: int = 8) -> None:
    if df.empty:
        return
    fig, axes = plt.subplots(2, 3, figsize=(7.1, 3.9))
    for ax, label in zip(axes.ravel(), labels):
        style_axes(ax, grid_axis="x")
        sub = df[df["class"] == label].head(n)[::-1]
        if sub.empty:
            ax.set_visible(False)
            continue
        y = np.arange(len(sub))
        colors = [BLUE if m else "#9ec5f4" for m in sub["marker"]]
        ax.barh(y, sub["total_share"], color=colors, height=0.62)
        ax.set_yticks(y)
        ax.set_yticklabels(sub["word"], fontsize=6.5)
        ax.set_title(label, loc="left", color=INK)
        ax.tick_params(axis="y", length=0)
    for ax in axes[-1]:
        ax.set_xlabel("attribution share per text")
    from matplotlib.patches import Patch
    fig.legend([Patch(color=BLUE), Patch(color="#9ec5f4")], ["class marker in the training set", "other word"],
               loc="lower center", ncol=2, frameon=False, fontsize=6.5)
    fig.suptitle(f"Words with the most Integrated Gradients attribution per predicted class, {model_name}",
                 x=0.01, y=0.995, ha="left", va="top", fontsize=7.5, color=INK)
    fig.tight_layout(rect=(0, 0.05, 1, 0.965))
    save_figure(fig, out_dir, "fig_xai_top_words")
    plt.close(fig)


def fig_examples(plt, records, labels, out_dir: Path, model_name: str, n_each: int, seed: int) -> None:
    rng = np.random.default_rng(seed)

    def pick(pool):
        pool = [r for r in pool if 3 <= len(r["words"]) <= 22]
        rng.shuffle(pool)
        chosen, classes = [], set()
        for r in pool:
            if r["true"] not in classes:
                chosen.append(r)
                classes.add(r["true"])
            if len(chosen) == n_each:
                break
        return chosen

    correct = pick([r for r in records if r["group"] == "sample" and r["true"] == r["pred"]])
    wrong = pick([r for r in records if r["group"] == "error"])
    blocks = correct + wrong
    if not blocks:
        return
    width, line_h, pad_frac, gap = 7.1, 0.21, 0.28, 0.035
    fig = plt.figure(figsize=(width, 8))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    renderer = fig.canvas.get_renderer()
    dpi = fig.dpi
    margin = 0.12
    y = 0.14
    ax.text(margin, y, f"Integrated Gradients attributions, {model_name}. Blue: supports the predicted class; "
                       "orange: opposes it.", fontsize=7.2, color=INK, va="center")
    y += 0.3
    for r in blocks:
        ok = r["true"] == r["pred"]
        ax.text(margin, y, f"{'Correct' if ok else 'Error'}: true = {labels[r['true']]}, "
                           f"predicted = {labels[r['pred']]} (p = {r['p_pred']:.2f})",
                fontsize=6.6, color=INK2 if ok else INK, va="center", fontweight="normal" if ok else "bold")
        y += 0.23
        x = margin
        scale = float(np.max(np.abs(r["ig"]))) if len(r["ig"]) else 0.0
        for word, value in zip(r["words"], r["ig"]):
            (cr, cg, cb), t = _chip_color(float(value), scale)
            pad_in = pad_frac * 7.0 / 72.0  # el relleno de la ficha se mide en fracciones del tamano de letra
            txt = ax.text(x + pad_in, y, word, fontsize=7, va="center", ha="left", color="#ffffff" if t > 0.6 else INK,
                          bbox=dict(boxstyle=f"square,pad={pad_frac}", fc=(cr / 255, cg / 255, cb / 255), ec="none"))
            w_in = txt.get_window_extent(renderer).width / dpi
            if x + w_in + 2 * pad_in > width - margin:
                x, y = margin, y + line_h
                txt.set_position((x + pad_in, y))
            x += w_in + 2 * pad_in + gap
        y += line_h + 0.16
    ax.text(margin, y, "Colour intensity is relative to the largest absolute attribution within each text.",
            fontsize=6.2, color=INK2, va="center")
    y += 0.2
    total = y
    fig.set_size_inches(width, total)
    ax.set_xlim(0, width)
    ax.set_ylim(total, 0)
    save_figure(fig, out_dir, "fig_xai_examples")
    plt.close(fig)


OFFSET_LABEL = {
    "before_feel": "antes de la palabra feel",
    "feel_itself": "la propia palabra feel",
    "plus_1": "1 palabra después",
    "plus_2": "2 palabras después",
    "plus_3": "3 o más palabras después",
    "no_feel_word": "texto sin palabra feel",
}


def write_report(summary: dict, df_tokens, labels, path: Path) -> None:
    s, ch = summary, summary["checks"]
    ks, faith = s["settings"]["ks"], s["faithfulness"]
    L = []
    L.append(f"# Explicabilidad (XAI): {s['model']}, semilla {s['seed']}\n")
    L.append("Generado por `scripts/xai.py`. No se edita a mano: si algo cambia, se vuelve a correr el script. "
             "La interpretación va en el artículo, apoyada en estos números y en `examples.html`.\n")
    L.append("## 0. Qué se hizo\n")
    L.append(f"- Modelo: `{s['model_id']}` ajustado (checkpoint `results/{s['model']}/seed_{s['seed']}/model/trainable.pt`).")
    L.append(f"- Muestra: {s['n_sample']} textos del test, estratificada por clase (hasta {s['settings']['n_per_class']} por clase, "
             f"semilla {s['seed']}), más {s['n_errors']} errores del test para el análisis cualitativo "
             f"(el modelo se equivocó en {s['n_errors_total']} textos del test).")
    L.append(f"- Integrated Gradients: {s['settings']['ig_steps']} puntos de Gauss-Legendre, baseline = [PAD] en cada palabra "
             "([CLS] y [SEP] intactos), objetivo = logit de la clase predicha. La atribución de una palabra es la suma de la de sus sub-tokens.")
    L.append("- Atención: fila de [CLS] en la última capa (media de cabezas) y attention rollout (con conexiones residuales).")
    L.append(f"- Fidelidad: quitar (o dejar solo) las k palabras más importantes, con k en {ks}. \"Quitar\" pone en cero la máscara de atención "
             f"de sus sub-tokens. Línea base: {s['settings']['n_random']} sorteos de k palabras al azar por texto.")
    L.append(f"- Cálculo en fp32 en `{s['device']}` (torch {s['torch']}, transformers {s['transformers']}).\n")

    L.append("## 1. ¿Se puede confiar en el método? (verificaciones previas)\n")
    comp, sanity = ch["completeness"], ch.get("sanity", {})
    L.append("| Verificación | Resultado | Umbral orientativo |\n|---|---|---|")
    L.append(f"| Predicciones recalculadas iguales a `predictions.csv` | {fmt(ch['pred_agreement'], pct=True)} "
             f"({ch['pred_disagree']} de {ch['pred_total']} difieren) | ≥ 99% (la corrida usó fp16, aquí se usa fp32) |")
    if comp.get("n"):
        L.append(f"| Completitud de IG: \\|suma de atribuciones − (F(x) − F(base))\\| / \\|F(x) − F(base)\\| | media {fmt(comp['mean_rel_error'], pct=True)}, "
                 f"mediana {fmt(comp['median_rel_error'], pct=True)}, p95 {fmt(comp['p95_rel_error'], pct=True)} | media < 5% |")
        L.append(f"| Texto base ([CLS] [PAD] … [SEP]) clasificado como la clase que se explica | {fmt(comp['baseline_same_class'], pct=True)} de los textos "
                 "| bajo; si es alto, las atribuciones de esa clase son pequeñas |")
    for key, name in (("random_head", "Cordura: cabeza de clasificación aleatoria"), ("random_model", "Cordura: modelo entero aleatorio")):
        if key in sanity:
            L.append(f"| {name} (Spearman con las atribuciones del modelo entrenado, {sanity['n']} textos) | rho medio "
                     f"{fmt(sanity[key]['spearman_mean'])}; misma palabra top-1 en {fmt(sanity[key]['top1_same'], pct=True)} | rho cerca de 0 |")
    L.append("")
    L.append("Si la completitud o la cordura fallan, las atribuciones no se interpretan: se sube `--ig-steps` o se revisa el checkpoint.\n")

    L.append("## 2. Fidelidad: ¿las palabras marcadas importan de verdad?\n")
    L.append("Caída de la probabilidad de la clase predicha. **Quitar** las k palabras más importantes: más caída = más fiel. "
             "**Dejar solo** esas k palabras: menos caída = más fiel. Entre paréntesis, el IC 95% aproximado (1.96 errores estándar). "
             "n es la cantidad de textos con más de k palabras.\n")
    L.append("| k | Método | n | Caída al quitar | Cambia la predicción al quitar | Caída al dejar solo | La decisión se mantiene al dejar solo |\n|---|---|---|---|---|---|---|")
    for k in ks:
        for m in (*METHODS, "random"):
            cell = faith[m].get(str(k))
            if not cell:
                continue
            L.append(f"| {k} | {METHOD_LABEL[m]} | {cell['n']} | {fmt(cell['comp_drop']['mean'])} (±{fmt(cell['comp_drop']['ci'])}) | "
                     f"{fmt(cell['comp_flip']['mean'], pct=True)} | {fmt(cell['suff_drop']['mean'])} (±{fmt(cell['suff_drop']['ci'])}) | "
                     f"{fmt(cell['suff_keep']['mean'], pct=True)} |")
    L.append("")
    L.append("AOPC (promedio de la caída al quitar sobre los k): " + ", ".join(
        f"{METHOD_LABEL[m]} {fmt(faith[m]['aopc'])}" for m in (*METHODS, "random")) + ".\n")

    L.append("## 3. ¿Qué palabras usa el modelo?\n")
    L.append(top_words_md(df_tokens, labels))
    cues = s["cues"]
    L.append(f"Sobre {cues['n']} textos de la muestra (con al menos 2 palabras):\n")
    if cues.get("marker_hit"):
        hit = cues["marker_hit"]
        L.append(f"- La palabra con más atribución (top-1) es una palabra marcadora de la clase predicha en "
                 f"{fmt(hit['ig'], pct=True)} de los textos con IG, {fmt(hit['rollout'], pct=True)} con rollout y "
                 f"{fmt(hit['cls'], pct=True)} con la atención de [CLS]. Si se eligiera una palabra al azar del texto, "
                 f"acertaría en {fmt(cues['marker_chance'], pct=True)}. Marcadoras = las {s['settings']['marker_top']} palabras con mayor "
                 "razón de log-odds de cada clase en el subconjunto S de entrenamiento.")
    L.append(f"- El top-1 de IG es una stopword en {fmt(cues['stopword_top1_ig'], pct=True)} de los textos "
             f"(al azar sería {fmt(cues['stopword_chance'], pct=True)}).")
    off = cues["top1_offset_from_feel"]
    if off:
        L.append("- Posición del top-1 de IG respecto de la primera palabra de la familia feel/feeling/felt: " + ", ".join(
            f"{OFFSET_LABEL.get(k, k)} {fmt(v, pct=True)}" for k, v in off.items()) + ".")
    for m, name in (("rollout", "rollout"), ("cls", "la atención de [CLS]")):
        a = cues["agreement_with_ig"][m]
        L.append(f"- Acuerdo de IG con {name}: misma palabra top-1 en {fmt(a['top1'], pct=True)}, Spearman medio {fmt(a['spearman'])}.")
    att = s["attention"]
    L.append(f"- Masa de atención que cae en [CLS]/[SEP]: {fmt(att['special_mass_cls_last'], pct=True)} (última capa) y "
             f"{fmt(att['special_mass_rollout'], pct=True)} (rollout). Una masa alta en tokens especiales es una de las razones "
             "por las que la atención cruda se interpreta con cuidado.\n")

    L.append("## 4. Ejemplos\n")
    L.append("Ver `fig_xai_examples.png` (figura para el artículo) y `examples.html` (más ejemplos, con IG, rollout y atención). "
             "Errores analizados: palabras con mayor atribución hacia la clase predicha y hacia la verdadera.\n")
    for r in s["error_examples"]:
        L.append(f"- verdadera **{r['true']}**, predicha **{r['pred']}** (p={r['p_pred']:.2f}): \"{r['text']}\" "
                 f"→ hacia la predicha: {', '.join(r['top_pred'])}; hacia la verdadera: {', '.join(r['top_true']) or 'n/d'}")
    L.append("")
    L.append("## 5. Lectura y límites (para el artículo y la defensa)\n")
    L.append("- La atribución explica **al modelo**, no a los datos: que una palabra reciba atribución dice que el modelo la usó, no que sea la causa real de la emoción.")
    L.append("- IG explica la diferencia entre el texto y el baseline. Si el baseline ya favorece la clase explicada, las atribuciones de esa clase son pequeñas y menos fiables. "
             "Otro baseline (cero, [MASK]) puede dar otro reparto; no se probó.")
    L.append("- Quitar una palabra con la máscara de atención no equivale a borrarla: el modelo ve un texto que nunca vio en el entrenamiento. "
             "Por eso se compara siempre contra quitar palabras al azar.")
    L.append("- La atención no es una explicación por sí sola (Jain y Wallace, 2019). Se reporta para contrastarla con IG, no como evidencia independiente.")
    L.append("- Las palabras marcadoras salen de la misma distribución del entrenamiento. Si el modelo las usa, aprendió pistas léxicas parecidas a las de un modelo lineal; "
             "es coherente con TF-IDF + regresión logística, pero no prueba que entienda la emoción.")
    L.append("- La muestra es de {} textos y una sola semilla ({}); los IC son aproximados y no cubren la variación entre semillas.".format(
        s["n_sample"], s["seed"]))
    path.write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")


# --------------------------------------------------------------------------------------------------
# Principal
# --------------------------------------------------------------------------------------------------
def parse_args(argv):
    ap = argparse.ArgumentParser(description="XAI para BERT-base / DistilBERT (Integrated Gradients, atencion y verificaciones).")
    ap.add_argument("--model", required=True, help="nombre de la corrida en results/ (bert-base, distilbert)")
    ap.add_argument("--seed", type=int, default=42, help="semilla de la corrida a explicar (42, 1337, 2026)")
    ap.add_argument("--results", type=Path, default=ROOT / "results")
    ap.add_argument("--clean", type=Path, default=ROOT / "salida" / "cleaned_full_data.csv")
    ap.add_argument("--labels", type=Path, default=ROOT / "labels.json")
    ap.add_argument("--train-ids", type=Path, default=ROOT / "splits" / "train_S_ids.csv",
                    help="ids de entrenamiento para las palabras marcadoras")
    ap.add_argument("--out", type=Path, default=None, help="por defecto results/xai/<modelo>/seed_<n>")
    ap.add_argument("--n-per-class", type=int, default=100)
    ap.add_argument("--n-errors", type=int, default=40)
    ap.add_argument("--ig-steps", type=int, default=64, help="puntos de la integral; suba si la completitud falla")
    ap.add_argument("--ig-chunk", type=int, default=32, help="secuencias por pasada de IG (baje a 16 si falta VRAM)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 3, 5])
    ap.add_argument("--n-random", type=int, default=5)
    ap.add_argument("--sanity-n", type=int, default=100, help="textos para la prueba de cordura (0 la omite)")
    ap.add_argument("--min-count", type=int, default=5, help="apariciones minimas para listar una palabra")
    ap.add_argument("--marker-top", type=int, default=50)
    ap.add_argument("--examples", type=int, default=3, help="ejemplos correctos y de error en la figura")
    ap.add_argument("--html-per-class", type=int, default=6)
    ap.add_argument("--device", default="auto", help="auto, cpu o cuda")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--quick", action="store_true", help="humo: muestra y pasos minimos, solo para ver que corre")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.quick:
        args.n_per_class, args.n_errors, args.ig_steps, args.sanity_n = 12, 6, 8, 12
    torch = require_torch()
    import transformers

    run_dir = args.results.resolve() / args.model / f"seed_{args.seed}"
    ckpt = run_dir / "model" / "trainable.pt"
    info_path = run_dir / "model" / "model_info.json"
    if not ckpt.is_file() or not info_path.is_file():
        print(f"No existe {ckpt}.\nEl checkpoint se guarda al entrenar con --save-model y vive en la computadora que "
              "hizo la corrida (no se sube al repositorio). Corra este script alli, o vuelva a entrenar esa semilla con --save-model.")
        return 2
    info = json.loads(info_path.read_text(encoding="utf-8"))
    labels = load_labels(args.labels)
    if info.get("labels") and list(info["labels"]) != labels:
        raise XaiError("Las clases de model_info.json no coinciden con labels.json.")
    out_dir = (args.out or ROOT / "results" / "xai" / args.model / f"seed_{args.seed}").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/6] Datos ({args.model}, semilla {args.seed})")
    preds = read_predictions(run_dir / "predictions.csv", labels)
    clean = pd.read_csv(args.clean, usecols=["id", "text", "label"], dtype={"id": str, "text": str})
    texts = clean.set_index("id")["text"]
    missing = set(preds["id"]) - set(texts.index)
    if missing:
        raise XaiError(f"{len(missing)} ids de predictions.csv no estan en {args.clean}.")
    sample, errors, n_errors_total = pick_items(preds, texts, len(labels), args.n_per_class, args.n_errors, args.seed)
    markers, stop = None, set()
    try:
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        stop = set(ENGLISH_STOP_WORDS)
    except ImportError:
        print("  aviso: sin scikit-learn no se calculan stopwords.")
    if args.train_ids.is_file():
        train_ids = read_ids(args.train_ids)
        sub = clean.set_index("id").loc[train_ids]
        markers = marker_sets(sub["text"].tolist(), sub["label"].to_numpy(), len(labels), args.marker_top)
        print(f"  palabras marcadoras calculadas con {len(train_ids)} textos de entrenamiento")
    else:
        print(f"  aviso: no existe {args.train_ids}; se omite la comparacion con palabras marcadoras.")
    print(f"  muestra {len(sample)} textos, {len(errors)} errores (de {n_errors_total})")

    print("[2/6] Modelo")
    device = torch.device("cuda" if torch.cuda.is_available() and args.device in ("auto", "cuda") else "cpu")
    model, tok = load_encoder(info, ckpt, labels, device)
    expl = Explainer(model, tok, device, int(info.get("max_len", 64)), args.ig_steps, args.ig_chunk)
    print(f"  {info['model_id']} en {device}")

    print("[3/6] Explicaciones de la muestra (IG, atencion, fidelidad)")
    ks = sorted(set(args.ks))
    t0 = time.time()
    records, acc = expl.analyze(sample, args.batch_size, ks, args.n_random, True, False, args.seed, "muestra")
    print("[4/6] Explicaciones de los errores")
    err_records, _ = expl.analyze(errors, args.batch_size, ks, args.n_random, False, True, args.seed, "errores")
    print(f"  {time.time() - t0:.0f} s")

    print("[5/6] Verificaciones y agregados")
    everything = records + err_records
    disagree = sum(1 for r in everything if r["pred"] != r["csv_pred"])
    comp = completeness(records)
    sanity = sanity_check(expl, records, sample, args.batch_size, labels, args.sanity_n)
    df_tokens = token_stats(records, labels, markers, args.min_count, stop)
    cues = cue_stats(records, markers, stop)
    faith = summarize_faithfulness(acc, ks)
    if everything and disagree / len(everything) > 0.01:
        print(f"  AVISO: {disagree} de {len(everything)} predicciones difieren de predictions.csv. "
              "Revise que el checkpoint sea de esta corrida y semilla.")

    def top_words(r, scores, n=3):
        order = np.argsort(-scores)[:n]
        return [f"{r['words'][i]} ({scores[i]:+.2f})" for i in order]

    error_examples = []
    for r in err_records[:12]:
        error_examples.append({
            "true": labels[r["true"]], "pred": labels[r["pred"]], "p_pred": r["p_pred"], "text": " ".join(r["words"]),
            "top_pred": top_words(r, r["ig"]), "top_true": top_words(r, r["ig_true"]) if r["ig_true"] is not None else []})
    summary = {
        "model": args.model, "model_id": info["model_id"], "seed": args.seed, "split": "test",
        "device": str(device), "torch": torch.__version__, "transformers": transformers.__version__,
        "n_sample": len(records), "n_errors": len(err_records), "n_errors_total": n_errors_total,
        "settings": {"n_per_class": args.n_per_class, "ig_steps": args.ig_steps, "ks": ks, "n_random": args.n_random,
                     "marker_top": args.marker_top, "baseline": "pad_per_word", "target": "predicted_class"},
        "checks": {"pred_agreement": 1 - disagree / max(len(everything), 1), "pred_disagree": disagree,
                   "pred_total": len(everything), "completeness": comp, "sanity": sanity},
        "faithfulness": faith, "cues": cues, "attention": attention_stats(records),
        "error_examples": error_examples,
    }
    print("[6/6] Escribiendo salidas en", out_dir)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    df_tokens.to_csv(out_dir / "token_stats.csv", index=False, float_format="%.5f")
    write_jsonl(everything, labels, out_dir / "explanations.jsonl")
    write_html(everything, labels, args.html_per_class, out_dir / "examples.html", f"XAI examples: {args.model}, seed {args.seed}")
    write_report(summary, df_tokens, labels, out_dir / "xai_report.md")
    if not args.no_figures:
        plt = setup_matplotlib()
        if plt is None:
            print("  aviso: falta matplotlib, no se generan figuras (pip install matplotlib).")
        else:
            fig_faithfulness(plt, faith, ks, out_dir, args.model)
            fig_top_words(plt, df_tokens, labels, out_dir, args.model)
            fig_examples(plt, everything, labels, out_dir, args.model, args.examples, args.seed)
    print("Listo.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except XaiError as exc:
        sys.exit(f"Error: {exc}")
