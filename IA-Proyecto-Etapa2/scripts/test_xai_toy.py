#!/usr/bin/env python3
"""Prueba de scripts/xai.py con modelos de juguete en los que SE CONOCE la respuesta correcta.

Se crean un BERT y un DistilBERT diminutos (2 capas, 48 dimensiones) y un corpus sintetico: cada texto es una
frase de relleno con UNA palabra-pista que decide la clase (p. ej. "furious" = anger). Se entrenan en CPU
(~30 segundos cada uno) y se guardan en el mismo formato que el arnes (model_info.json, trainable.pt,
predictions.csv). Despues se corre xai.py y se verifica lo que debe cumplirse si los metodos estan bien
implementados:

  - las atribuciones de Integrated Gradients suman F(x) - F(baseline) (completitud);
  - la palabra con mas atribucion es la palabra-pista (el modelo solo puede usar esa);
  - quitar esa palabra baja la probabilidad mucho mas que quitar una palabra al azar (fidelidad);
  - con la cabeza o el modelo aleatorios las atribuciones cambian (prueba de cordura);
  - la palabra-pista coincide con las palabras marcadoras calculadas del entrenamiento.

Sirve de dos formas: valida el codigo de xai.py y le dice a quien lo corre si su entorno (torch, transformers)
funciona antes de gastar tiempo en el modelo real:
    python scripts/test_xai_toy.py                     # ambos modelos; termina con TODAS LAS VERIFICACIONES PASARON
    python scripts/test_xai_toy.py --arch distilbert   # solo uno (bert o distilbert)
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import xai  # noqa: E402

LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]
CUES = {
    "sadness": ["gloomy", "tearful", "grieving", "miserable"],
    "joy": ["delighted", "cheerful", "thrilled", "joyful"],
    "love": ["adoring", "cherished", "affectionate", "devoted"],
    "anger": ["furious", "enraged", "irritated", "hostile"],
    "fear": ["terrified", "anxious", "fearful", "panicked"],
    "surprise": ["astonished", "stunned", "amazed", "startled"],
}
FILLERS = sorted(set("i feel so very today really just when the a and my friend day at home work with about this "
                     "that it was because after before again still now then me we you they he she there here "
                     "what how".split()))
ARCHS = ("bert", "distilbert")


def make_corpus(n: int, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for i in range(n):
        label = int(rng.integers(len(LABELS)))
        cue = str(rng.choice(CUES[LABELS[label]]))
        words = [str(w) for w in rng.choice(FILLERS, size=int(rng.integers(4, 9)))]
        if rng.random() < 0.6:
            words[:2] = ["i", "feel"]
        words.insert(int(rng.integers(0, len(words) + 1)), cue)
        rows.append({"id": f"t{i:05d}", "text": " ".join(words), "label": label, "group": i})
    return pd.DataFrame(rows)


def build_world(root: Path, arch: str, seed: int = 0) -> str:
    """Crea corpus, tokenizador, modelo entrenado y archivos de la corrida. Devuelve el nombre de la corrida."""
    import torch
    from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, processors
    from transformers import (BertConfig, BertForSequenceClassification, DistilBertConfig,
                              DistilBertForSequenceClassification, PreTrainedTokenizerFast)

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    corpus = make_corpus(2400, rng)
    train, test = corpus.iloc[:1600], corpus.iloc[1600:]

    root.mkdir(parents=True, exist_ok=True)
    (root / "salida").mkdir(exist_ok=True)
    (root / "splits").mkdir(exist_ok=True)
    corpus.to_csv(root / "salida" / "cleaned_full_data.csv", index=False)
    (root / "splits" / "train_S_ids.csv").write_text("id\n" + "\n".join(train["id"]) + "\n", encoding="utf-8")
    (root / "labels.json").write_text(json.dumps({"labels": LABELS}), encoding="utf-8")

    tokdir = root / f"toy_{arch}_pretrained"
    tokdir.mkdir()
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + sorted(set(FILLERS) | {w for v in CUES.values() for w in v})
    # Tokenizador WordPiece construido con la libreria `tokenizers` (funciona igual en transformers 4 y 5).
    core = Tokenizer(models.WordPiece({t: i for i, t in enumerate(vocab)}, unk_token="[UNK]"))
    core.normalizer = normalizers.BertNormalizer(lowercase=True)
    core.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    core.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B:1 [SEP]:1", special_tokens=[("[CLS]", 2), ("[SEP]", 3)])
    core.decoder = decoders.WordPiece(prefix="##")
    tok = PreTrainedTokenizerFast(tokenizer_object=core, unk_token="[UNK]", pad_token="[PAD]", cls_token="[CLS]",
                                  sep_token="[SEP]", mask_token="[MASK]")
    assert tok.convert_ids_to_tokens(tok("i feel furious")["input_ids"])[1:-1] == ["i", "feel", "furious"], "tokenizador"
    names = {"id2label": dict(enumerate(LABELS)), "label2id": {n: i for i, n in enumerate(LABELS)}}
    if arch == "bert":
        model = BertForSequenceClassification(BertConfig(
            vocab_size=len(vocab), hidden_size=48, num_hidden_layers=2, num_attention_heads=2, intermediate_size=96,
            max_position_embeddings=32, num_labels=len(LABELS), pad_token_id=0,
            hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0, **names))  # sin dropout: la respuesta es limpia
    else:
        tok.model_input_names = ["input_ids", "attention_mask"]  # DistilBERT no usa token_type_ids
        model = DistilBertForSequenceClassification(DistilBertConfig(
            vocab_size=len(vocab), dim=48, n_layers=2, n_heads=2, hidden_dim=96, max_position_embeddings=32,
            num_labels=len(LABELS), pad_token_id=0, dropout=0.0, attention_dropout=0.0, seq_classif_dropout=0.0, **names))
    tok.save_pretrained(tokdir)
    model.save_pretrained(tokdir)  # "preentrenado" = pesos iniciales; el ajuste fino va aparte, en trainable.pt

    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.0)
    order = np.arange(len(train))
    print(f"Entrenando el {arch} de juguete (CPU)...")
    model.train()
    for step in range(500):
        batch = train.iloc[rng.choice(order, size=32, replace=False)]
        enc = tok(batch["text"].tolist(), padding=True, truncation=True, max_length=16, return_tensors="pt")
        loss = model(**enc, labels=torch.tensor(batch["label"].to_numpy())).loss
        loss.backward()
        opt.step()
        opt.zero_grad()
        if step % 100 == 0:
            print(f"  paso {step}: loss {loss.item():.3f}")
    model.eval()

    name = "toy" if arch == "bert" else f"toy-{arch}"
    run = root / "results" / name / "seed_1"
    (run / "model").mkdir(parents=True)
    state = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
    torch.save(state, run / "model" / "trainable.pt")
    info = {"family": "encoder", "model_id": str(tokdir), "head": "sequence-classification", "quantization": None,
            "lora_r": None, "max_len": 16, "labels": LABELS, "n_tensors": len(state)}
    (run / "model" / "model_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    with torch.no_grad():
        enc = tok(test["text"].tolist(), padding=True, truncation=True, max_length=16, return_tensors="pt")
        probs = torch.softmax(model(**enc).logits, -1).numpy()
    y_pred = probs.argmax(1)
    y_true = test["label"].to_numpy().copy()
    noisy = rng.choice(len(y_true), size=int(0.05 * len(y_true)), replace=False)  # 5% de etiquetas ruidosas = errores
    y_true[noisy] = (y_true[noisy] + rng.integers(1, len(LABELS), size=len(noisy))) % len(LABELS)
    out = pd.DataFrame({"id": test["id"].to_numpy(), "y_true": [LABELS[i] for i in y_true],
                        "y_pred": [LABELS[i] for i in y_pred]})
    for i, label in enumerate(LABELS):
        out[f"p_{label}"] = probs[:, i]
    out.to_csv(run / "predictions.csv", index=False)
    print(f"  exactitud del {arch} de juguete en su test: {(y_pred == test['label'].to_numpy()).mean():.3f}")
    return name


def check(condition: bool, message: str, failures: list) -> None:
    print(("  OK    " if condition else "  FALLA ") + message)
    if not condition:
        failures.append(message)


def run_arch(root: Path, arch: str) -> list:
    world = root / arch
    name = build_world(world, arch)
    out = root / f"xai_out_{arch}"
    code = xai.main([
        "--model", name, "--seed", "1", "--results", str(world / "results"),
        "--clean", str(world / "salida" / "cleaned_full_data.csv"), "--labels", str(world / "labels.json"),
        "--train-ids", str(world / "splits" / "train_S_ids.csv"), "--out", str(out),
        "--n-per-class", "40", "--n-errors", "12", "--ig-steps", "128", "--ig-chunk", "64", "--batch-size", "8",
        "--sanity-n", "30", "--min-count", "3", "--marker-top", "12", "--html-per-class", "3",
        "--ks", "1", "2", "3", "--device", "cpu"])
    if code != 0:
        return [f"xai.py termino con codigo {code}"]

    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    records = [json.loads(line) for line in (out / "explanations.jsonl").read_text(encoding="utf-8").splitlines()]
    sample = [r for r in records if r["group"] == "sample" and r["true"] == r["pred"]]
    on_cue = [r["words"][int(np.argmax(r["ig"]))] in CUES[r["true"]] for r in sample]
    faith = summary["faithfulness"]
    sanity = summary["checks"]["sanity"]
    print(f"\nVerificaciones ({arch}):")
    failures: list = []
    check(summary["checks"]["pred_agreement"] >= 0.99,
          f"las predicciones recalculadas coinciden con predictions.csv ({summary['checks']['pred_agreement']:.3f})", failures)
    comp = summary["checks"]["completeness"]
    check(comp["mean_rel_error"] < 0.05, f"IG es completo: error relativo medio {comp['mean_rel_error']:.4f} < 0.05", failures)
    check(np.mean(on_cue) >= 0.9,
          f"la palabra top-1 de IG es la palabra-pista en {np.mean(on_cue):.1%} de {len(on_cue)} textos (>= 90%)", failures)
    ig1, rnd1 = faith["ig"]["1"]["comp_drop"]["mean"], faith["random"]["1"]["comp_drop"]["mean"]
    check(ig1 > rnd1 + 0.3, f"quitar la palabra top-1 de IG baja la probabilidad {ig1:.3f} vs {rnd1:.3f} al azar", failures)
    flip_ig, flip_rnd = faith["ig"]["1"]["comp_flip"]["mean"], faith["random"]["1"]["comp_flip"]["mean"]
    check(flip_ig > 0.6 and flip_ig > flip_rnd + 0.4,
          f"quitar la palabra top-1 de IG cambia la prediccion en {flip_ig:.1%} (al azar {flip_rnd:.1%})", failures)
    check(faith["ig"]["1"]["suff_keep"]["mean"] > faith["random"]["1"]["suff_keep"]["mean"] + 0.3,
          "dejar solo la palabra top-1 de IG conserva la decision mucho mas que una palabra al azar", failures)
    for key in ("random_head", "random_model"):
        rho = sanity[key]["spearman_mean"]
        check(rho < 0.5, f"cordura ({key}): Spearman medio {rho:.3f} < 0.5", failures)
    hit = summary["cues"]["marker_hit"]["ig"]
    check(hit >= 0.9 and hit > summary["cues"]["marker_chance"] + 0.4,
          f"el top-1 de IG es una palabra marcadora en {hit:.1%} (al azar {summary['cues']['marker_chance']:.1%})", failures)
    check(summary["n_errors"] > 0 and any(r["group"] == "error" and r["ig_true"] is not None for r in records),
          "se analizaron errores con IG hacia la clase verdadera", failures)
    for fname in ("xai_report.md", "summary.json", "explanations.jsonl", "token_stats.csv", "examples.html",
                  "fig_xai_faithfulness.png", "fig_xai_top_words.png", "fig_xai_examples.png"):
        check((out / fname).is_file(), f"existe {fname}", failures)
    return failures


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Prueba de xai.py con modelos de juguete.")
    ap.add_argument("--arch", choices=("both", *ARCHS), default="both")
    ap.add_argument("--dir", type=Path, default=None, help="carpeta de trabajo (por defecto, una temporal)")
    ap.add_argument("--keep", action="store_true", help="no borrar la carpeta al terminar")
    args = ap.parse_args(argv)

    root = args.dir or Path(tempfile.mkdtemp(prefix="xai_toy_"))
    if args.dir and root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    failures: list = []
    try:
        for arch in (ARCHS if args.arch == "both" else (args.arch,)):
            failures += [f"[{arch}] {m}" for m in run_arch(root, arch)]
        if failures:
            print(f"\n{len(failures)} verificacion(es) fallaron:")
            for m in failures:
                print("  -", m)
            args.keep = True
            return 1
        print("\nTODAS LAS VERIFICACIONES PASARON")
        return 0
    finally:
        if not args.keep and not args.dir:
            shutil.rmtree(root, ignore_errors=True)
        else:
            print("Archivos de la prueba en", root)


if __name__ == "__main__":
    sys.exit(main())
