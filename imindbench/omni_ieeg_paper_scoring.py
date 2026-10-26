"""Omni-iEEG paper-style scoring of one multi-channel ictal-vs-interictal run.

Reads the run's ``test_predictions.csv`` (written with
``runtime.save_test_predictions=true``) and scores it like the upstream
Omni-iEEG ``omni_ieeg/exploratory_model/ictal_sleep/evaluation.py``:

- decision unit = non-overlapping 60 s window after a 1 s edge trim; when the
  model predicted shorter windows (1 s mode) their probabilities are averaged
  inside each 60 s unit (units not fully covered are dropped);
- positive class = interictal (label 1); score = P(interictal);
- AUC = auc(roc_curve); threshold = Youden's J on the test set (as upstream);
  macro precision / recall / F1 and balanced accuracy at that threshold.

Also reports a recording-level AUC (mean unit score per recording).

Usage::

    python -m imindbench.omni_ieeg_paper_scoring <run_dir> [--unit_sec 60] [--edge_trim_sec 1]

Writes ``<run_dir>/omni_paper_metrics.json``. The single-channel protocol
(``dataset=omni_ieeg_single_channel``) is scored by ``imindbench.omni_ieeg_scoring``
instead.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

POSITIVE_LABEL = 1  # interictal


def aggregate_units(
    pred: pd.DataFrame, unit_sec: float, edge_trim_sec: float
) -> pd.DataFrame:
    """Average window probabilities inside each unit_sec test unit."""
    pred = pred.copy()
    span = pred["window_end_sec"] - pred["window_start_sec"]
    if (span > unit_sec + 1e-6).any():
        raise ValueError(
            f"Windows longer than unit_sec={unit_sec} cannot be aggregated."
        )
    offset = (pred["window_start_sec"] - edge_trim_sec) / unit_sec
    pred["unit"] = np.floor(offset + 1e-6).astype(int)
    per_unit_windows = int(round(unit_sec / float(span.median())))
    grouped = pred.groupby(["recording_id", "unit"])
    units = grouped.agg(
        score=("score", "mean"),
        y_true=("y_true", "first"),
        n_windows=("score", "size"),
        n_labels=("y_true", "nunique"),
    ).reset_index()
    if (units["n_labels"] > 1).any():
        raise ValueError("A test unit mixes labels; recordings must have one label.")
    # Keep only fully covered units: upstream tiles floor((len-2)/60) whole windows.
    return units[units["n_windows"] == per_unit_windows].drop(columns="n_labels")


def paper_metrics(y_true: np.ndarray, score: np.ndarray) -> dict:
    fpr, tpr, thresholds = roc_curve(y_true, score)
    best = int(np.argmax(tpr - fpr))
    threshold = float(thresholds[best])
    y_pred = (score > threshold).astype(int)  # upstream uses a strict '>'
    return {
        "auc": float(auc(fpr, tpr)),
        "youden_threshold": threshold if math.isfinite(threshold) else None,
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "n_units": int(len(y_true)),
        "n_positive": int((y_true == POSITIVE_LABEL).sum()),
        "n_negative": int((y_true != POSITIVE_LABEL).sum()),
    }


def score_cell(cell_dir: Path, unit_sec: float, edge_trim_sec: float) -> dict:
    csvs = sorted(Path(cell_dir).glob("test_predictions*.csv"))
    if not csvs:
        raise FileNotFoundError(f"no test_predictions*.csv in {cell_dir}")
    pred = pd.read_csv(csvs[0])
    prob_col = f"prob_class_{POSITIVE_LABEL}"
    if prob_col not in pred.columns:
        raise ValueError(f"{csvs[0]} has no {prob_col} column")
    pred["score"] = pred[prob_col]

    window_sec = float((pred["window_end_sec"] - pred["window_start_sec"]).median())
    units = aggregate_units(pred, unit_sec, edge_trim_sec)
    y_true = units["y_true"].to_numpy(dtype=int)
    score = units["score"].to_numpy(dtype=float)
    result = {
        "protocol": "omni_ieeg_paper",
        "predictions_file": csvs[0].name,
        "model_window_sec": window_sec,
        "unit_sec": unit_sec,
        "edge_trim_sec": edge_trim_sec,
        "window_level": paper_metrics(y_true, score),
    }
    if len(pred) and window_sec < unit_sec:
        result["model_window_level_auc"] = float(
            roc_auc_score(pred["y_true"].to_numpy(dtype=int), pred["score"].to_numpy())
        )
    recordings = units.groupby("recording_id").agg(
        score=("score", "mean"), y_true=("y_true", "first")
    )
    if recordings["y_true"].nunique() == 2:
        result["recording_level_auc"] = float(
            roc_auc_score(
                recordings["y_true"].to_numpy(dtype=int),
                recordings["score"].to_numpy(),
            )
        )
    result["n_recordings"] = int(len(recordings))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("cell_dir", type=Path)
    parser.add_argument("--unit_sec", type=float, default=60.0)
    parser.add_argument("--edge_trim_sec", type=float, default=1.0)
    args = parser.parse_args(argv)
    result = score_cell(args.cell_dir, args.unit_sec, args.edge_trim_sec)
    out = args.cell_dir / "omni_paper_metrics.json"
    out.write_text(json.dumps(result, indent=2))
    wl = result["window_level"]
    print(
        f"omni paper metrics ({out}): AUC={wl['auc']:.4f} F1={wl['f1_macro']:.4f} "
        f"BA={wl['balanced_accuracy']:.4f} units={wl['n_units']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
