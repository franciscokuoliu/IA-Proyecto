"""TF-IDF + regresión logística. El vectorizador se ajusta solo con train."""

from __future__ import annotations

import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.harness.data import Column


def train_tfidf(
    train: Column,
    evaluation: Column,
    n_classes: int,
    seed: int,
    C: float = 1.0,
    ngram: tuple[int, int] = (1, 2),
    max_features: int | None = None,
) -> tuple[list[int], np.ndarray, float, dict]:
    started = time.perf_counter()
    vectorizer = TfidfVectorizer(ngram_range=ngram, min_df=2, max_features=max_features)
    x_train = vectorizer.fit_transform(train.texts)
    x_eval = vectorizer.transform(evaluation.texts)
    # class_weight="balanced" es N / (K * n_c), la misma fórmula del contrato.
    classifier = LogisticRegression(
        C=C,
        class_weight="balanced",
        solver="lbfgs",
        max_iter=1000,
        random_state=seed,
    )
    classifier.fit(x_train, train.y)
    elapsed = time.perf_counter() - started
    probabilities = np.zeros((len(evaluation.ids), n_classes), dtype=np.float64)
    raw = classifier.predict_proba(x_eval)
    for column, label in enumerate(classifier.classes_):
        probabilities[:, int(label)] = raw[:, column]
    predicted = probabilities.argmax(axis=1)
    details = {
        "C": C,
        "ngram": list(ngram),
        "min_df": 2,
        "max_features": max_features,
        "n_features": int(len(vectorizer.vocabulary_)),
    }
    return predicted.tolist(), probabilities, elapsed, details
