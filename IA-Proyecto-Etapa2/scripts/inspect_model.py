"""Temporal: dice donde se va la memoria de un modelo SIN descargar los pesos.

  python IA-Proyecto-Etapa2/scripts/inspect_model.py google/gemma-4-E4B
"""
import sys
from collections import defaultdict

import torch
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModel

model_id = sys.argv[1] if len(sys.argv) > 1 else "google/gemma-4-E4B"
cfg = AutoConfig.from_pretrained(model_id)
print("config:", type(cfg).__name__, "| subconfigs:", [k for k in cfg.to_dict() if k.endswith("_config")])
with init_empty_weights():
    model = AutoModel.from_config(cfg)
print("clase:", type(model).__name__)

linear, other = defaultdict(int), defaultdict(int)
for name, mod in model.named_modules():
    if not name or any(True for _ in mod.children()):
        continue
    top = ".".join(name.split(".")[:2])
    n = sum(p.numel() for p in mod.parameters(recurse=False))
    (linear if isinstance(mod, torch.nn.Linear) else other)[top] += n

GB = 1024 ** 3
rows = sorted(set(linear) | set(other), key=lambda k: -(linear[k] + other[k]))
print(f"\n{'modulo':45} {'Linear (Mparams)':>17} {'resto (Mparams)':>16} {'GB en nf4*':>11}")
tot = 0.0
for k in rows:
    gb = (linear[k] * 0.55 + other[k] * 2) / GB
    tot += gb
    print(f"{k:45} {linear[k]/1e6:17.1f} {other[k]/1e6:16.1f} {gb:11.2f}")
print(f"\nTotal estimado en 4 bits: {tot:.2f} GB (*Linear ~0.55 B/param, el resto en bf16 sin cuantizar)")
if torch.cuda.is_available():
    free, total = torch.cuda.mem_get_info()
    print(f"VRAM libre ahora: {free/GB:.2f} de {total/GB:.2f} GB")
