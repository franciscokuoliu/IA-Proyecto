"""Catálogo de los nueve modelos y las rejillas de la sección 9.

Los ids de Hugging Face se pueden cambiar con --model-id. Gemma 4 usa el
checkpoint base (sin -it): la entrada del contrato es el texto crudo, sin prompt.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    model_id: str | None = None
    epochs: int = 3
    learning_rate: float = 2e-5


MODELS: dict[str, ModelSpec] = {
    "tfidf-logreg": ModelSpec("tfidf-logreg", "tfidf", epochs=1, learning_rate=1.0),
    "bilstm": ModelSpec("bilstm", "bilstm", epochs=10, learning_rate=1e-3),
    "distilbert": ModelSpec("distilbert", "encoder", "distilbert-base-uncased", learning_rate=2e-5),
    "bert-base": ModelSpec("bert-base", "encoder", "bert-base-uncased", learning_rate=2e-5),
    "qwen2.5-0.5b": ModelSpec("qwen2.5-0.5b", "decoder", "Qwen/Qwen2.5-0.5B", learning_rate=2e-4),
    "qwen2.5-1.5b": ModelSpec("qwen2.5-1.5b", "decoder", "Qwen/Qwen2.5-1.5B", learning_rate=2e-4),
    "gemma-4-e2b": ModelSpec("gemma-4-e2b", "decoder", "google/gemma-4-E2B", learning_rate=2e-4),
    "gemma-4-e4b": ModelSpec("gemma-4-e4b", "decoder", "google/gemma-4-E4B", learning_rate=2e-4),
    "deepseek-r1-distill-qwen-1.5b": ModelSpec(
        "deepseek-r1-distill-qwen-1.5b",
        "decoder",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        learning_rate=2e-4,
    ),
}

SEEDS = (42, 1337, 2026)
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")


def hpo_trials(family: str) -> list[dict]:
    """Rejillas de la sección 9. Cada dict se pasa como overrides del entrenamiento."""
    if family == "tfidf":
        return [
            {"C": c, "ngram": ngram, "name": f"C{c}_ngram{ngram[0]}{ngram[1]}"}
            for c in (0.1, 1.0, 10.0)
            for ngram in ((1, 1), (1, 2))
        ]
    if family == "bilstm":
        return [
            {
                "learning_rate": lr,
                "hidden": hidden,
                "dropout": dropout,
                "fraction": 0.2,
                "name": f"lr{lr}_h{hidden}_d{dropout}",
            }
            for lr in (1e-3, 3e-3)
            for hidden in (128, 256)
            for dropout in (0.3, 0.5)
        ]
    if family == "encoder":
        return [{"learning_rate": lr, "name": f"lr{lr}"} for lr in (2e-5, 3e-5, 5e-5)]
    if family == "decoder":
        # Cinco intentos, uno a la vez alrededor de lr=2e-4 y r=16.
        return [
            {"learning_rate": 1e-4, "lora_r": 16, "fraction": 0.2, "epochs": 1, "name": "lr1e-4_r16"},
            {"learning_rate": 2e-4, "lora_r": 16, "fraction": 0.2, "epochs": 1, "name": "lr2e-4_r16"},
            {"learning_rate": 3e-4, "lora_r": 16, "fraction": 0.2, "epochs": 1, "name": "lr3e-4_r16"},
            {"learning_rate": 2e-4, "lora_r": 8, "fraction": 0.2, "epochs": 1, "name": "lr2e-4_r8"},
            {"learning_rate": 2e-4, "lora_r": 32, "fraction": 0.2, "epochs": 1, "name": "lr2e-4_r32"},
        ]
    raise KeyError(family)
