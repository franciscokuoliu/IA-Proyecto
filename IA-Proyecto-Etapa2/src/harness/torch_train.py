"""Entrenamiento neuronal compartido: BiLSTM, encoders y decoders.

La parada temprana y el checkpoint usan solo el Macro-F1 de la val de
selección. La loss sí lleva los pesos de clase; la métrica no.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import f1_score

from src.harness.data import Column, HarnessError


@dataclass
class TorchRun:
    predicted: list[int]
    probabilities: np.ndarray
    seconds: float
    best_macro_f1: float
    steps: int
    details: dict


def require_torch():
    try:
        import torch
    except ImportError as exc:
        raise HarnessError(
            "Falta torch. Instala las dependencias de entrenamiento con "
            "pip install -r IA-Proyecto-Etapa2/requirements.txt"
        ) from exc
    return torch


def pick_device(torch):
    """cuda si hay GPU NVIDIA, mps en Mac con Apple Silicon, cpu en otro caso."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pick_dtype(torch, requested: str | None):
    device = pick_device(torch)
    if requested in ("bf16", "fp16") and device.type == "cpu":
        raise HarnessError("bf16 y fp16 piden una GPU (CUDA o MPS). En CPU el arnés usa fp32.")
    if requested in (None, "auto"):
        if device.type == "cuda" and torch.cuda.get_device_capability(0)[0] >= 8:
            return torch.bfloat16, "bf16"
        if device.type == "cuda":
            return torch.float16, "fp16"
        if device.type == "mps":
            return torch.bfloat16, "bf16"
        return torch.float32, "fp32"
    table = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    if requested not in table:
        raise HarnessError("dtype debe ser auto, bf16, fp16 o fp32.")
    return table[requested], requested


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    return float(
        f1_score(
            y_true,
            y_pred,
            labels=list(range(n_classes)),
            average="macro",
            zero_division=0,
        )
    )


def _snapshot(model) -> dict:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def _restore(model, state: dict) -> None:
    import torch

    parameters = dict(model.named_parameters())
    with torch.no_grad():
        for name, tensor in state.items():
            parameters[name].copy_(tensor.to(parameters[name].device))


def _parameter_device(model):
    return next(model.parameters()).device


def _move(batch: dict, device):
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}


class _Loop:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        y_val,
        weights,
        learning_rate,
        epochs,
        max_steps,
        n_classes,
        compute_dtype=None,
    ):
        torch = require_torch()
        self.torch = torch
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.y_val = y_val
        self.weights = weights
        self.epochs = epochs
        self.max_steps = max_steps
        self.n_classes = n_classes
        self.compute_dtype = compute_dtype
        self.scaler = None
        if compute_dtype is not None and compute_dtype == torch.float16 and torch.cuda.is_available():
            self.scaler = torch.amp.GradScaler("cuda")
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if not trainable:
            raise HarnessError("El modelo no tiene parámetros entrenables.")
        self.optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=0.01)
        micro = len(train_loader)
        self.accum = max(1, math.ceil(32 / _loader_batch_size(train_loader)))
        optim_per_epoch = max(1, math.ceil(micro / self.accum))
        total = optim_per_epoch * epochs
        if max_steps is not None:
            total = min(total, max_steps)
        self.total_steps = max(1, total)
        warmup = max(1, int(round(self.total_steps * 0.06)))
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lambda step: _linear_warmup(step, warmup, self.total_steps),
        )
        self.eval_every = max(1, optim_per_epoch // 4)
        self.best_f1 = -1.0
        self.best_state: dict | None = None
        self.patience = 3
        self.steps = 0
        self.log: list[str] = []
        self.train_seconds = 0.0
        self.train_samples = 0
        self.eval_seconds: list[float] = []
        self._tick = 0.0

    def run(self) -> None:
        torch = self.torch
        device = _parameter_device(self.model)
        weights = torch.tensor(self.weights, dtype=torch.float32, device=device)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        pending = 0
        stop = False
        self._sync()
        self._tick = time.perf_counter()
        autocast_enabled = (
            self.compute_dtype is not None
            and self.compute_dtype != torch.float32
            and device.type == "cuda"
        )
        for epoch in range(self.epochs):
            for batch in self.train_loader:
                batch = _move(batch, device)
                with torch.autocast(
                    device_type=device.type,
                    dtype=self.compute_dtype or torch.float32,
                    enabled=autocast_enabled,
                ):
                    logits = _logits(self.model, batch)
                    loss = torch.nn.functional.cross_entropy(logits, batch["labels"], weight=weights)
                if self.scaler is None:
                    loss.backward()
                else:
                    self.scaler.scale(loss).backward()
                self.train_samples += int(batch["labels"].shape[0])
                pending += 1
                if pending == self.accum:
                    self._flush(pending)
                    pending = 0
                    if self._should_stop():
                        stop = True
                        break
            if pending and not stop:
                self._flush(pending)
                pending = 0
                stop = self._should_stop()
            if stop:
                break
        if self.best_state is None:
            self._close_train_interval()
            self._evaluate()
        if self.best_state is not None:
            _restore(self.model, self.best_state)
        self._log_timing()

    def _flush(self, pending: int) -> None:
        trainable = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
        for parameter in trainable:
            if parameter.grad is not None:
                parameter.grad.div_(pending)
        self.torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.steps += 1
        self._close_train_interval()
        if self.steps % self.eval_every == 0 or self.steps == self.total_steps:
            self._evaluate()

    def _sync(self) -> None:
        if self.torch.cuda.is_available():
            self.torch.cuda.synchronize()

    def _close_train_interval(self) -> None:
        self._sync()
        now = time.perf_counter()
        self.train_seconds += now - self._tick
        self._tick = now

    def timing(self) -> dict:
        train_s = self.train_seconds
        evals = self.eval_seconds
        return {
            "seconds_per_step": None if self.steps <= 0 or train_s <= 0 else train_s / self.steps,
            "train_samples_per_s": None if train_s <= 0 else self.train_samples / train_s,
            "eval_time_s": None if not evals else sum(evals) / len(evals),
            "n_evals": len(evals),
            "train_samples": self.train_samples,
            "train_seconds": train_s,
        }

    def _log_timing(self) -> None:
        info = self.timing()
        seconds = info["seconds_per_step"]
        samples = info["train_samples_per_s"]
        evaluation = info["eval_time_s"]
        line = (
            f"segundos_por_paso={0.0 if seconds is None else seconds:.6f}  "
            f"muestras_por_segundo={0.0 if samples is None else samples:.3f}  "
            f"segundos_por_evaluacion={0.0 if evaluation is None else evaluation:.3f}  "
            f"evaluaciones={info['n_evals']}"
        )
        self.log.append(line)
        print(line, flush=True)

    def _evaluate(self) -> None:
        self._sync()
        started = time.perf_counter()
        predicted, _ = predict_batches(self.model, self.val_loader, self.compute_dtype)
        self._sync()
        self.eval_seconds.append(time.perf_counter() - started)
        self._tick = time.perf_counter()
        score = _macro_f1(self.y_val, np.asarray(predicted), self.n_classes)
        improved = score > self.best_f1
        if improved:
            self.best_f1 = score
            self.best_state = _snapshot(self.model)
            self.patience = 3
        else:
            self.patience -= 1
        line = (
            f"step {self.steps}  macro_f1_val_select={score:.6f}  "
            f"mejor={self.best_f1:.6f}  paciencia={self.patience}"
        )
        self.log.append(line)
        print(line, flush=True)
        self.model.train()

    def _should_stop(self) -> bool:
        if self.max_steps is not None and self.steps >= self.max_steps:
            return True
        return self.patience <= 0


def _linear_warmup(step: int, warmup: int, total: int) -> float:
    current = step + 1
    if current <= warmup:
        return current / warmup
    remain = max(1, total - warmup)
    return max(0.0, (total - current) / remain)


def _loader_batch_size(loader) -> int:
    return int(getattr(loader, "batch_size", None) or 1)


def _logits(model, batch):
    output = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    return output.logits if hasattr(output, "logits") else output


def predict_batches(model, loader, compute_dtype=None) -> tuple[list[int], np.ndarray]:
    torch = require_torch()
    device = _parameter_device(model)
    was_training = model.training
    model.eval()
    predicted: list[int] = []
    blocks: list[np.ndarray] = []
    autocast_enabled = (
        compute_dtype is not None and compute_dtype != torch.float32 and device.type == "cuda"
    )
    with torch.no_grad():
        for batch in loader:
            batch = _move(batch, device)
            with torch.autocast(
                device_type=device.type,
                dtype=compute_dtype or torch.float32,
                enabled=autocast_enabled,
            ):
                logits = _logits(model, batch)
            logits = logits.float().cpu()
            probabilities = torch.softmax(logits, dim=-1).numpy()
            blocks.append(probabilities)
            predicted.extend(probabilities.argmax(axis=1).tolist())
    model.train(was_training)
    return predicted, np.concatenate(blocks, axis=0)


def _counts(model) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return int(total), int(trainable)


def _peak_vram() -> int | None:
    """CUDA: pico de memoria asignada. MPS: memoria total que pidio el driver
    (no hay pico exacto; es memoria unificada, no VRAM dedicada)."""
    torch = require_torch()
    if torch.cuda.is_available():
        return int(torch.cuda.max_memory_allocated())
    if pick_device(torch).type == "mps":
        return int(torch.mps.driver_allocated_memory())
    return None


def _memory_kind() -> str | None:
    device = pick_device(require_torch())
    return {"cuda": "cuda_max_allocated", "mps": "mps_driver_allocated"}.get(device.type)


def _save_trainable(model, save_dir, info: dict) -> None:
    """Guarda los parametros entrenables del mejor checkpoint (para el XAI).

    En los decoders son los adaptadores LoRA y la cabeza; en los encoders y el
    BiLSTM, el modelo entero. Para recargar: reconstruir el modelo con los mismos
    argumentos y hacer load_state_dict(..., strict=False) con trainable.pt.
    """
    if save_dir is None:
        return
    import json
    from pathlib import Path

    torch = require_torch()
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    state = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
    torch.save(state, save_dir / "trainable.pt")
    info = {**info, "n_tensors": len(state), "n_values": int(sum(v.numel() for v in state.values()))}
    (save_dir / "model_info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")


def train_bilstm(
    train: Column,
    evaluation: Column,
    val_select: Column,
    weights: np.ndarray,
    n_classes: int,
    seed: int,
    epochs: int,
    batch_size: int,
    max_len: int,
    learning_rate: float,
    hidden: int,
    dropout: float,
    max_steps: int | None,
    glove: str | None,
    dtype_name: str | None = None,
    save_dir=None,
) -> TorchRun:
    torch = require_torch()
    compute_dtype, dtype_label = pick_dtype(torch, dtype_name)
    device = pick_device(torch)
    vocab = _WordVocab(train.texts, max_len)
    model = _BiLSTM(len(vocab.tokens), hidden, dropout, n_classes, glove, vocab).to(device)
    train_loader = _token_loader(train, vocab, batch_size, seed, shuffle=True)
    val_loader = _token_loader(val_select, vocab, batch_size, seed, shuffle=False)
    eval_loader = _token_loader(evaluation, vocab, batch_size, seed, shuffle=False)
    started = time.perf_counter()
    loop = _Loop(
        model,
        train_loader,
        val_loader,
        val_select.y,
        weights,
        learning_rate,
        epochs,
        max_steps,
        n_classes,
        compute_dtype,
    )
    loop.run()
    elapsed = time.perf_counter() - started
    predicted, probabilities = predict_batches(model, eval_loader, compute_dtype)
    total, trainable = _counts(model)
    _save_trainable(model, save_dir, {"family": "bilstm", "hidden": hidden, "dropout": dropout,
                                      "max_len": max_len, "vocab": vocab.tokens})
    return TorchRun(
        predicted,
        probabilities,
        elapsed,
        loop.best_f1,
        loop.steps,
        {
            "hidden": hidden,
            "dropout": dropout,
            "learning_rate": learning_rate,
            "vocab_size": len(vocab.tokens),
            "embeddings": "glove-300d-frozen" if glove else "random-frozen",
            "dtype": dtype_label,
            "n_params": total,
            "n_trainable_params": trainable,
            "peak_vram_bytes": _peak_vram(),
            "device": pick_device(torch).type,
            "memory_kind": _memory_kind(),
            "log": loop.log,
            **loop.timing(),
        },
    )


class _WordVocab:
    def __init__(self, texts: list[str], max_len: int):
        self.max_len = max_len
        counts: dict[str, int] = {}
        for text in texts:
            for token in text.split():
                counts[token] = counts.get(token, 0) + 1
        self.tokens = {"<pad>": 0, "<unk>": 1}
        for token in sorted(counts):
            self.tokens[token] = len(self.tokens)

    def encode(self, text: str) -> list[int]:
        ids = [self.tokens.get(token, 1) for token in text.split()[: self.max_len]]
        return ids or [1]


def _token_loader(column: Column, vocab: _WordVocab, batch_size: int, seed: int, shuffle: bool):
    torch = require_torch()

    class _Set(torch.utils.data.Dataset):
        def __len__(self):
            return len(column.ids)

        def __getitem__(self, index):
            return vocab.encode(column.texts[index]), int(column.y[index])

    def collate(rows):
        encoded, labels = zip(*rows)
        width = max(len(item) for item in encoded)
        input_ids = torch.zeros((len(rows), width), dtype=torch.long)
        mask = torch.zeros((len(rows), width), dtype=torch.long)
        for index, tokens in enumerate(encoded):
            input_ids[index, : len(tokens)] = torch.tensor(tokens)
            mask[index, : len(tokens)] = 1
        return {
            "input_ids": input_ids,
            "attention_mask": mask,
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.utils.data.DataLoader(
        _Set(),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        collate_fn=collate,
    )


def _build_bilstm_class():
    torch = require_torch()
    nn = torch.nn

    class BiLSTM(nn.Module):
        def __init__(self, vocab_size, hidden, dropout, n_classes, glove, vocab: _WordVocab):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, 300, padding_idx=0)
            self.embedding.weight.requires_grad_(False)
            if glove:
                _load_glove(self.embedding, vocab, glove)
            self.lstm = nn.LSTM(300, hidden, batch_first=True, bidirectional=True)
            self.dropout = nn.Dropout(dropout)
            self.head = nn.Linear(hidden * 2, n_classes)

        def forward(self, input_ids, attention_mask):
            embedded = self.embedding(input_ids)
            lengths = attention_mask.sum(dim=1).clamp(min=1).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths, batch_first=True, enforce_sorted=False
            )
            _, (state, _) = self.lstm(packed)
            features = torch.cat([state[-2], state[-1]], dim=-1)
            return self.dropout(self.head(features))

    BiLSTM.forward = _as_logits(BiLSTM.forward)
    return BiLSTM


def _as_logits(forward):
    def wrapped(self, input_ids=None, attention_mask=None, **_kwargs):
        logits = forward(self, input_ids, attention_mask)
        return type("Out", (), {"logits": logits})()

    return wrapped


def _load_glove(embedding, vocab: _WordVocab, path: str) -> None:
    torch = require_torch()
    found = 0
    weight = embedding.weight.detach()
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            token, *values = line.rstrip().split(" ")
            index = vocab.tokens.get(token)
            if index is None or len(values) != 300:
                continue
            weight[index] = torch.tensor([float(value) for value in values])
            found += 1
    if found == 0:
        raise HarnessError(f"No encontré vectores de 300d en {path} para el vocabulario de train.")


def _BiLSTM(vocab_size, hidden, dropout, n_classes, glove, vocab):
    return _build_bilstm_class()(vocab_size, hidden, dropout, n_classes, glove, vocab)


def train_transformer(
    train: Column,
    evaluation: Column,
    val_select: Column,
    weights: np.ndarray,
    labels: list[str],
    seed: int,
    model_id: str,
    family: str,
    epochs: int,
    batch_size: int,
    max_len: int,
    learning_rate: float,
    max_steps: int | None,
    dtype_name: str | None,
    lora_r: int,
    save_dir=None,
) -> TorchRun:
    torch = require_torch()
    try:
        from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise HarnessError(
            "Falta transformers. Instálalo con pip install -r IA-Proyecto-Etapa2/requirements.txt"
        ) from exc

    compute_dtype, dtype_label = pick_dtype(torch, dtype_name)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    id2label = {index: name for index, name in enumerate(labels)}
    label2id = {name: index for index, name in enumerate(labels)}
    model, head_kind, quant = _load_model(
        family, model_id, len(labels), id2label, label2id, compute_dtype, lora_r, tokenizer
    )
    train_loader = _text_loader(train, tokenizer, max_len, batch_size, seed, True)
    val_loader = _text_loader(val_select, tokenizer, max_len, batch_size, seed, False)
    eval_loader = _text_loader(evaluation, tokenizer, max_len, batch_size, seed, False)
    started = time.perf_counter()
    loop = _Loop(
        model,
        train_loader,
        val_loader,
        val_select.y,
        weights,
        learning_rate,
        epochs,
        max_steps,
        len(labels),
        compute_dtype,
    )
    loop.run()
    elapsed = time.perf_counter() - started
    predicted, probabilities = predict_batches(model, eval_loader, compute_dtype)
    total, trainable = _counts(model)
    _save_trainable(model, save_dir, {"family": family, "model_id": model_id, "head": head_kind,
                                      "quantization": quant, "lora_r": lora_r if family == "decoder" else None,
                                      "max_len": max_len, "labels": labels})
    return TorchRun(
        predicted,
        probabilities,
        elapsed,
        loop.best_f1,
        loop.steps,
        {
            "model_id": model_id,
            "head": head_kind,
            "quantization": quant,
            "lora_r": lora_r if family == "decoder" else None,
            "lora_alpha": 2 * lora_r if family == "decoder" else None,
            "learning_rate": learning_rate,
            "dtype": dtype_label,
            "n_params": total,
            "n_trainable_params": trainable,
            "peak_vram_bytes": _peak_vram(),
            "device": pick_device(torch).type,
            "memory_kind": _memory_kind(),
            "log": loop.log,
            **loop.timing(),
        },
    )


def _hidden_size(config) -> int:
    if getattr(config, "hidden_size", None):
        return int(config.hidden_size)
    text_config = getattr(config, "text_config", None)
    if text_config is not None and getattr(text_config, "hidden_size", None):
        return int(text_config.hidden_size)
    raise HarnessError("El modelo no declara hidden_size.")


def _load_model(family, model_id, n_classes, id2label, label2id, compute_dtype, lora_r, tokenizer):
    from transformers import AutoModel, AutoModelForSequenceClassification

    torch = require_torch()
    if family == "encoder":
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            num_labels=n_classes,
            id2label=id2label,
            label2id=label2id,
        )
        device = pick_device(torch)
        return model.to(device), "sequence-classification", None

    # Con CUDA el decoder va en NF4 (bitsandbytes). Sin CUDA (Mac/MPS) bitsandbytes no
    # existe: el modelo se carga sin cuantizar en el dtype de computo. Es otro regimen
    # y los resultados se reportan aparte (contrato, seccion 11).
    cuda = torch.cuda.is_available()
    device = pick_device(torch)
    if cuda:
        # dtype: los modulos que no se cuantizan (normas, embeddings) se cargan en el dtype de
        # computo. En GPUs Turing (fp16) evita que queden en bf16, que esa arquitectura no soporta.
        load_kwargs = {
            "quantization_config": _nf4_config(compute_dtype),
            "device_map": "auto",
            "dtype": compute_dtype,
        }
        quant_label = "nf4"
    else:
        load_kwargs = {"dtype": compute_dtype}
        quant_label = "none"
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            num_labels=n_classes,
            id2label=id2label,
            label2id=label2id,
            **load_kwargs,
        )
        if not cuda:
            model = model.to(device)
        model.config.pad_token_id = tokenizer.pad_token_id
        model = _apply_lora(model, lora_r, ["score", "classifier"])
        _trainable_to_fp32(model, compute_dtype)
        return model, "sequence-classification", quant_label
    except HarnessError:
        raise
    except Exception as sequence_error:
        try:
            base = AutoModel.from_pretrained(model_id, **load_kwargs)
            if not cuda:
                base = base.to(device)
        except Exception as base_error:
            raise HarnessError(
                f"No pude cargar el decoder ({quant_label}).\n"
                f"SequenceClassification: {sequence_error}\n"
                f"AutoModel: {base_error}"
            ) from base_error
        if getattr(base.config, "pad_token_id", None) is None:
            base.config.pad_token_id = tokenizer.pad_token_id
        base = _apply_lora(base, lora_r, [])
        model = _manual_head(base, _hidden_size(base.config), n_classes)
        _trainable_to_fp32(model, compute_dtype)
        return model, "manual-last-nonpad", quant_label


def _nf4_config(compute_dtype):
    try:
        from transformers import BitsAndBytesConfig
    except ImportError as exc:
        raise HarnessError("Falta BitsAndBytesConfig. Instala transformers y bitsandbytes.") from exc
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )


def _trainable_to_fp32(model, compute_dtype) -> None:
    """Con fp16 (GPUs Turing: 1650, 2060) GradScaler no acepta gradientes en bf16
    ("_amp_foreach_non_finite_check_and_unscale_cuda not implemented for BFloat16").
    Si el checkpoint viene en bf16, la cabeza o los adaptadores pueden quedar en bf16:
    los parametros entrenables se pasan a fp32. Con bf16 (Ampere) no se toca nada."""
    torch = require_torch()
    if compute_dtype != torch.float16:
        return
    for parameter in model.parameters():
        if parameter.requires_grad and parameter.dtype != torch.float32:
            parameter.data = parameter.data.to(torch.float32)


def _apply_lora(model, rank: int, extra_modules: list[str]):
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:
        raise HarnessError("Falta peft. Instálalo con pip install -r IA-Proyecto-Etapa2/requirements.txt") from exc
    module_names = [module_name for module_name, _ in model.named_modules()]
    present = [name for name in extra_modules if any(name in module_name for module_name in module_names)]
    attention = ["q_proj", "k_proj", "v_proj", "o_proj"]
    # Modelos multimodales (Gemma 4) traen torres de vision y audio con capas q_proj
    # envueltas que PEFT no soporta. El contrato adapta solo el decoder de texto.
    has_language_model = any("language_model" in name for name in module_names)
    target_modules = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj)" if has_language_model else attention
    config = LoraConfig(
        r=rank,
        lora_alpha=2 * rank,
        lora_dropout=0.05,
        target_modules=target_modules,
        task_type=TaskType.SEQ_CLS if present else TaskType.FEATURE_EXTRACTION,
        modules_to_save=present or None,
        bias="none",
    )
    try:
        return get_peft_model(model, config)
    except Exception as exc:
        linear = [
            module_name
            for module_name, module in model.named_modules()
            if module.__class__.__name__ in {"Linear", "Linear4bit", "Linear8bitLt"}
        ]
        preview = ", ".join(linear[:40]) or "(ninguno)"
        raise HarnessError(
            f"PEFT no pudo aplicar LoRA: {type(exc).__name__}: {exc}\n"
            f"Módulos lineales del modelo (primeros 40): {preview}."
        ) from exc


def _manual_head(base, hidden: int, n_classes: int):
    torch = require_torch()

    class ManualHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base = base
            self.head = torch.nn.Linear(hidden, n_classes)
            device = next(base.parameters()).device
            self.head.to(device)

        def forward(self, input_ids=None, attention_mask=None, **_kwargs):
            output = self.base(input_ids=input_ids, attention_mask=attention_mask)
            hidden_state = output.last_hidden_state
            # Ultimo token real, con relleno a la derecha o a la izquierda. Gemma rellena
            # por la izquierda: "suma de la mascara - 1" apuntaria a un token equivocado.
            positions = torch.arange(attention_mask.size(1), device=attention_mask.device)
            last = (attention_mask.long() * positions).argmax(dim=1)
            gathered = hidden_state[torch.arange(hidden_state.size(0), device=hidden_state.device), last]
            if self.head.weight.device != gathered.device:
                self.head.to(gathered.device)
            logits = self.head(gathered.float())
            return type("Out", (), {"logits": logits})()

    return ManualHead()


def _text_loader(column: Column, tokenizer, max_len: int, batch_size: int, seed: int, shuffle: bool):
    torch = require_torch()

    class _Set(torch.utils.data.Dataset):
        def __len__(self):
            return len(column.ids)

        def __getitem__(self, index):
            return column.texts[index], int(column.y[index])

    def collate(rows):
        texts, labels = zip(*rows)
        encoded = tokenizer(
            list(texts),
            truncation=True,
            max_length=max_len,
            padding=True,
            return_tensors="pt",
        )
        encoded["labels"] = torch.tensor(labels, dtype=torch.long)
        return encoded

    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.utils.data.DataLoader(
        _Set(),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        collate_fn=collate,
    )


