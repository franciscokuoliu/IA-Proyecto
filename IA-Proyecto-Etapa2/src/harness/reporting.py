"""Escribe la carpeta de una corrida y llama a evaluate.py."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EVALUATE = ROOT / "scripts" / "evaluate.py"


def git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def gpu_status() -> tuple[str | None, str | None]:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None, None
    if completed.returncode != 0 or not completed.stdout.strip():
        return None, None
    name, driver = (completed.stdout.splitlines()[0].split(",", 1) + [""])[:2]
    return name.strip() or None, driver.strip() or None


def write_config(path: Path, fields: dict) -> None:
    lines: list[str] = []
    for key, value in fields.items():
        if isinstance(value, bool):
            rendered = "true" if value else "false"
        elif value is None:
            rendered = "null"
        elif isinstance(value, (int, float)):
            rendered = str(value)
        else:
            rendered = json.dumps(value, ensure_ascii=False)
        lines.append(f"{key}: {rendered}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_predictions(
    path: Path,
    sample_ids: list[str],
    y_true: list[str],
    y_pred: list[str],
    probabilities: np.ndarray,
    labels: list[str],
) -> None:
    header = ["id", "y_true", "y_pred", *[f"p_{label}" for label in labels]]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for index, sample_id in enumerate(sample_ids):
            writer.writerow(
                [sample_id, y_true[index], y_pred[index], *probabilities[index].tolist()]
            )


def score(run_dir: Path, labels_path: Path, split: str) -> dict:
    completed = subprocess.run(
        [
            sys.executable,
            str(EVALUATE),
            "score",
            str(run_dir),
            "--labels",
            str(labels_path),
            "--split",
            split,
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        check=False,
    )
    with (run_dir / "run.log").open("a", encoding="utf-8") as handle:
        handle.write(completed.stdout)
        if completed.stderr:
            handle.write(completed.stderr)
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(detail or "evaluate.py falló.")
    print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n")
    return json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))


def write_metadata(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
