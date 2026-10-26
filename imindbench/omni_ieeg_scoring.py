"""Omni-iEEG scoring of one single-channel run (``dataset=omni_ieeg_single_channel``).

Reads the run's ``test_predictions.csv`` (written with
``runtime.save_test_predictions=true`` and ``runtime.save_test_prediction_details=true``;
one row per test (recording, channel, window)) and scores it with literal ports
of the official Omni-iEEG evaluation scripts
(https://github.com/Omni-iEEG/Omni-iEEG; each function names its source lines):

  ictal_interictal, sleep_awake
      ``omni_ieeg/exploratory_model/ictal_sleep/evaluation.py::calculate_metrics``
      Per (edf, window start) the raw model logits are averaged across channels;
      roc_curve on those window scores, Youden-J threshold (``score > thr``),
      macro precision / recall / F1, balanced accuracy, confusion matrix and
      ``auc(fpr, tpr)``.
  soz_outcome
      ``omni_ieeg/channel_model/benchmark/evaluation_channel.py``
      (``calculate_all_ratios``, ``evaluate_model_outputs``,
      ``resection_ratio_calculation_originalmethod``), including its ``__main__``
      that applies a sigmoid to the saved ``channel_pred`` although
      ``channel_model_train/train.py`` already saved sigmoid outputs (the
      double sigmoid is reproduced on purpose). Needs the official
      ``final_split.csv`` and the raw BIDS tree (``*_channels.tsv``).
  anatomical
      ``omni_ieeg/exploratory_model/anatomical/evaluation.py::calculate_metrics``
  hfo
      ``omni_ieeg/event_model/benchmark/benchmark.py::calculate_metrics``

Binary heads here output two logits (softmax); the upstream single sigmoid logit
equals ``logit_class_1 - logit_class_0`` and its sigmoid equals
``prob_class_1``.

Usage::

    python -m imindbench.omni_ieeg_scoring <run_dir> [--task T]
        [--final_split_csv F] [--raw_root R]

Task and paths default to the run's ``.hydra/config.yaml``
(``dataset.task``, ``dataset.provider_options.final_split_csv`` or
``paths.omni_final_split_csv``, ``paths.omni_raw_root``).
Writes ``<run_dir>/omni_sc_metrics.json``. The per-channel-window AUROC in the
run's result JSON is not an Omni-iEEG metric; report this file's numbers.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

PROTOCOL = "omni_ieeg_single_channel"
TASKS = ("ictal_interictal", "sleep_awake", "soz_outcome", "anatomical", "hfo")
#: Upstream extractors index windows in samples at 1000 Hz.
UPSTREAM_RATE_HZ = 1000.0


def _recording_id_from_edf(edf_path: str) -> str:
    """Same sanitization as the omni_ieeg brainsets pipeline / OmniIEEG provider."""
    stem = Path(str(edf_path).replace("\\", "/")).with_suffix("").as_posix()
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    return value


# ---------------------------------------------------------------------------
# Prediction table -> upstream test_metadata.csv layout
# ---------------------------------------------------------------------------
def _class_ids(pred: pd.DataFrame, prefix: str) -> list[int]:
    ids = [
        int(c[len(prefix) :])
        for c in pred.columns
        if c.startswith(prefix) and c[len(prefix) :].lstrip("-").isdigit()
    ]
    return sorted(ids)


def load_predictions(cell_dir: Path) -> pd.DataFrame:
    csvs = sorted(Path(cell_dir).glob("test_predictions*.csv"))
    if not csvs:
        raise FileNotFoundError(f"no test_predictions*.csv in {cell_dir}")
    if len(csvs) > 1:
        raise ValueError(f"expected one test_predictions csv (one fold), got {csvs}")
    pred = pd.read_csv(csvs[0], keep_default_na=False, na_values=[""])
    required = {"recording_id", "window_start_sec", "y_true", "channel_name"}
    missing = required.difference(pred.columns)
    if missing:
        raise ValueError(
            f"{csvs[0]} lacks {sorted(missing)}; run with "
            "runtime.save_test_prediction_details=true."
        )
    if pred["channel_name"].astype(str).str.contains(r"\|").any():
        raise ValueError(f"{csvs[0]} is not single-channel (multi-channel rows found).")
    pred["recording_id"] = pred["recording_id"].astype(str)
    pred["channel_name"] = pred["channel_name"].astype(str)
    return pred


def binary_logit(pred: pd.DataFrame) -> np.ndarray:
    """Upstream single-logit equivalent of a two-logit softmax head."""
    if {"logit_class_0", "logit_class_1"} <= set(pred.columns):
        return (pred["logit_class_1"] - pred["logit_class_0"]).to_numpy(np.float64)
    p1 = pred["prob_class_1"].to_numpy(np.float64)
    return np.log(p1) - np.log1p(-p1)


def binary_probability(pred: pd.DataFrame) -> np.ndarray:
    """``torch.sigmoid(logit)`` as saved by ``channel_model_train/train.py``."""
    if {"logit_class_0", "logit_class_1"} <= set(pred.columns):
        return 1.0 / (1.0 + np.exp(-binary_logit(pred)))
    return pred["prob_class_1"].to_numpy(np.float64)


def upstream_metadata(pred: pd.DataFrame) -> pd.DataFrame:
    """Columns of the upstream ``test_metadata.csv``."""
    out = pd.DataFrame(
        {
            "edf_name": pred["recording_id"].to_numpy(),
            "name": pred["channel_name"].to_numpy(),
            "start_indices": np.rint(
                pred["window_start_sec"].to_numpy(np.float64) * UPSTREAM_RATE_HZ
            ).astype(np.int64),
            "channel_true": pred["y_true"].to_numpy(np.int64),
        }
    )
    for c in _class_ids(pred, "prob_class_"):
        out[f"prob_class_{c}"] = pred[f"prob_class_{c}"].to_numpy(np.float64)
    return out


# ---------------------------------------------------------------------------
# ictal_interictal / sleep_awake
# Port of omni_ieeg/exploratory_model/ictal_sleep/evaluation.py:15-64
# ---------------------------------------------------------------------------
def ictal_sleep_calculate_metrics(model_df: pd.DataFrame) -> dict:
    window_level_ratio = []
    window_level_true = []
    edf_level_list = []
    edf_level_true = []
    unique_edf_names = model_df["edf_name"].unique()
    for edf_name in unique_edf_names:
        edf_df = model_df[model_df["edf_name"] == edf_name]
        grouped_indices = edf_df.groupby("name")["start_indices"].apply(
            lambda x: tuple(sorted(x.unique()))
        )
        assert grouped_indices.nunique() == 1, (
            "all channel does not have all start indices"
        )
        assert len(edf_df["channel_true"].unique()) == 1, (
            f"Channel true values should be same for {edf_name}"
        )
        label = edf_df["channel_true"].iloc[0]
        unique_start = edf_df["start_indices"].unique()
        edf_ratio_list = []
        for start_time in unique_start:
            start_df = edf_df[edf_df["start_indices"] == start_time]
            assert start_df["edf_name"].nunique() == 1
            # aggregate predictions across channels for this start time
            channel_pred_sum = start_df["channel_pred"].sum()
            channel_pred_ratio = channel_pred_sum / len(start_df)
            window_level_ratio.append(channel_pred_ratio)
            window_level_true.append(label)
            edf_ratio_list.append(channel_pred_ratio)
        edf_level_list.append(edf_ratio_list)
        edf_level_true.append(label)

    num_1s = window_level_true.count(1)
    num_0s = window_level_true.count(0)

    fpr, tpr, roc_thresholds = roc_curve(window_level_true, window_level_ratio)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = roc_thresholds[optimal_idx]
    # Upstream compares a list of numpy floats with a numpy float, which numpy
    # evaluates element-wise; np.asarray makes that explicit.
    window_level_pred = (np.asarray(window_level_ratio) > optimal_threshold).astype(int)

    precision_value = precision_score(
        window_level_true, window_level_pred, average="macro"
    )
    recall_value = recall_score(window_level_true, window_level_pred, average="macro")
    f1_value = f1_score(window_level_true, window_level_pred, average="macro")
    naive_auc = auc(fpr, tpr)
    balanced_accuracy = balanced_accuracy_score(window_level_true, window_level_pred)
    confusion_matrix_result = confusion_matrix(window_level_true, window_level_pred)
    return {
        "window_level": {
            "auc": naive_auc,
            "precision_macro": precision_value,
            "recall_macro": recall_value,
            "f1_macro": f1_value,
            "balanced_accuracy": balanced_accuracy,
            "optimal_threshold": optimal_threshold,
            "confusion_matrix": confusion_matrix_result,
            "n_windows": len(window_level_true),
            "n_positive": num_1s,
            "n_negative": num_0s,
        },
        "n_recordings": len(edf_level_true),
    }


def score_ictal_sleep(pred: pd.DataFrame) -> dict:
    model_df = upstream_metadata(pred)
    # ictal_sleep/train.py saves the raw model output (a logit) as channel_pred.
    model_df["channel_pred"] = binary_logit(pred)
    result = ictal_sleep_calculate_metrics(model_df)
    result["n_channel_windows"] = int(len(model_df))
    return result


# ---------------------------------------------------------------------------
# soz_outcome
# Port of omni_ieeg/channel_model/benchmark/evaluation_channel.py
# ---------------------------------------------------------------------------
def pr_auc(groundtruth, pred):
    """evaluation_channel.py:12-15."""
    precision, recall, thresholds = precision_recall_curve(
        groundtruth, pred, pos_label=0
    )
    pr_auc_result = auc(recall, precision)
    return pr_auc_result


def calculate_all_ratios(testing_split, df_dict_raw, channels_loader):
    """evaluation_channel.py:17-88.

    ``channels_loader(edf_name)`` returns the ``*_channels.tsv`` DataFrame
    (upstream: ``filtered_dataset.get_channels_for_edf``). Model rows are matched
    on the processed recording id of the EDF path (upstream matches on the EDF
    basename, which is unique in the release).
    """
    results = []
    df_dict = {}
    for model_name, df in df_dict_raw.items():
        df_grouped = (
            df.groupby(["edf_name", "name"])
            .agg(
                channel_pred_sum=("channel_pred", "sum"),
                count=("channel_pred", "count"),
                unique_true=("channel_true", "nunique"),
                channel_true_val=("channel_true", "first"),
            )
            .reset_index()
        )
        assert all(df_grouped["unique_true"] == 1), (
            f"Inconsistent 'channel_true' values within a group for model {model_name}"
        )
        df_grouped["ratio"] = df_grouped["channel_pred_sum"] / df_grouped["count"]
        df_dict[model_name] = df_grouped.set_index(["edf_name", "name"])

    for _, row in testing_split.iterrows():
        patient_name = row["patient_name"]
        edf_name = row["edf_name"]
        edf_key = _recording_id_from_edf(edf_name)
        patient_outcome = row["outcome"]
        channels_df = channels_loader(edf_name)
        good_channels = channels_df[channels_df["good"] == 1]
        for _, inner_row in good_channels.iterrows():
            channel_name = inner_row["name"]
            channel_soz = inner_row["soz"]
            channel_resection = inner_row["resection"]
            label = -1
            if patient_outcome == 1 and channel_resection == 0:
                label = 1
            elif channel_soz == 1:
                label = 0
            current_result = {
                "patient_name": patient_name,
                "edf_name": edf_name,
                "channel_name": channel_name,
                "ground_truth": label,
                "channel_resection": channel_resection,
                "channel_soz": channel_soz,
                "patient_outcome": patient_outcome,
            }
            for model_name, model_result in df_dict.items():
                key = (edf_key, str(channel_name).strip())
                if key not in model_result.index:
                    # Set default values if no matching data found
                    current_result[f"{model_name}_ratio"] = np.nan
                    current_result[f"{model_name}_count"] = 0
                    current_result[f"{model_name}_num_count"] = 0
                    current_result[f"{model_name}_count_reversed"] = 0
                else:
                    match = model_result.loc[key]
                    current_result[f"{model_name}_ratio"] = match["ratio"]
                    current_result[f"{model_name}_count"] = match["channel_pred_sum"]
                    current_result[f"{model_name}_num_count"] = match["count"]
                    current_result[f"{model_name}_count_reversed"] = (
                        match["count"] - match["channel_pred_sum"]
                    )
            results.append(current_result)
    return pd.DataFrame(results)


def evaluate_model_outputs(
    df, model_output_columns, ground_truth_column="ground_truth"
):
    """evaluation_channel.py:97-186."""
    if ground_truth_column not in df.columns:
        raise ValueError(
            f"Ground truth column '{ground_truth_column}' not found in DataFrame"
        )
    for col in model_output_columns:
        if f"{col}_ratio" not in df.columns:
            raise ValueError(f"Model output column '{col}' not found in DataFrame")

    results = []
    # remove rows where ground truth is not 0 or 1
    df = df[df[ground_truth_column].isin([0, 1])]
    y_true = df[ground_truth_column].values
    y_true = [1 - y for y in y_true]
    for col in model_output_columns:
        y_scores = df[f"{col}_ratio"].values
        y_scores = [1 - y_score for y_score in y_scores]

        fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)
        naive_auc = auc(fpr, tpr)

        class_counts = np.bincount(y_true)
        class_weights = len(y_true) / (len(np.unique(y_true)) * class_counts)
        sample_weights = class_weights[y_true]
        auc_roc_auc = roc_auc_score(y_true, y_scores, sample_weight=sample_weights)

        pr_auc_val = pr_auc(y_true, y_scores)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = roc_thresholds[optimal_idx]

        # Upstream: ``(y_scores >= optimal_threshold)`` on a list of numpy floats.
        y_pred = (np.asarray(y_scores) >= optimal_threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        precision_value = precision_score(y_true, y_pred, average="macro")
        recall_value = recall_score(y_true, y_pred, average="macro")
        f1_value = f1_score(y_true, y_pred, average="macro")
        specificity = tn / (tn + fp)  # true negative rate

        results.append(
            {
                "model": col,
                "optimal_threshold": optimal_threshold,
                "precision": precision_value,
                "recall": recall_value,
                "specificity": specificity,
                "f1": f1_value,
                "roc_auc": auc_roc_auc,
                "auc": naive_auc,
                "pr_auc": pr_auc_val,
                "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
                "n_channels": int(len(y_true)),
                "n_soz": int(np.sum(np.asarray(y_true) == 1)),
            }
        )
    return pd.DataFrame(results)


def resection_ratio_calculation_originalmethod(
    df, model_output_columns, ground_truth_column="ground_truth"
):
    """evaluation_channel.py:190-270."""
    results = []
    model_resection_ratio_dict = {}
    for model_name in model_output_columns:
        model_resection_ratio_dict[model_name + "_count_reversed"] = []
    ground_truth = []
    patient_names = []
    for subject in df["patient_name"].unique():
        subject_df = df[df["patient_name"] == subject]
        assert subject_df["patient_outcome"].nunique() == 1, (
            f"subject {subject} has multiple outcomes"
        )
        subject_outcome = subject_df["patient_outcome"].iloc[0]
        if subject_outcome == -1:
            continue
        patient_names.append(subject)
        ground_truth.append(subject_outcome)
        for model_name in model_output_columns:
            patient_model_total_count_reversed = subject_df[
                f"{model_name}_count_reversed"
            ].sum()
            patient_model_resected_count_reversed = subject_df[
                subject_df["channel_resection"] == 1
            ][f"{model_name}_count_reversed"].sum()
            if patient_model_total_count_reversed == 0:
                model_resection_ratio_dict[model_name + "_count_reversed"].append(1)
            else:
                model_resection_ratio_dict[model_name + "_count_reversed"].append(
                    patient_model_resected_count_reversed
                    / patient_model_total_count_reversed
                )
    ground_truth = np.array(ground_truth)
    for model_name, resection_ratios in model_resection_ratio_dict.items():
        fpr, tpr, roc_thresholds = roc_curve(ground_truth, resection_ratios)
        naive_auc = auc(fpr, tpr)
        pr_auc_val = pr_auc(ground_truth, resection_ratios)
        results.append(
            {
                "model": model_name,
                "auc": naive_auc,
                "pr_auc": pr_auc_val,
                "n_patients": int(len(ground_truth)),
                "ground_truth_distribution": np.bincount(ground_truth).tolist(),
            }
        )
    return pd.DataFrame(results)


def soz_testing_split(final_split_csv: str) -> pd.DataFrame:
    """evaluation_channel.py:285-288."""
    final_split = pd.read_csv(final_split_csv)
    final_split = final_split[final_split["dataset"] != "Multicenter"]
    return final_split[
        (final_split["split"] == "test")
        & (final_split["frequency"] > 900)
        & (final_split["interictal"] == True)  # noqa: E712 (upstream idiom)
        & (final_split["length"] >= 62)
    ]


def raw_channels_loader(raw_root: str):
    """``*_channels.tsv`` of an EDF, read like upstream DataFilter."""
    cache: dict[str, pd.DataFrame] = {}

    def load(edf_name: str) -> pd.DataFrame:
        if edf_name not in cache:
            edf_path = os.path.join(raw_root, edf_name)
            channels_path = edf_path.replace("_ieeg.edf", "_channels.tsv")
            cache[edf_name] = pd.read_csv(channels_path, sep="\t")
        return cache[edf_name]

    return load


def score_soz(pred: pd.DataFrame, final_split_csv: str, raw_root: str) -> dict:
    model_df = upstream_metadata(pred)
    # channel_model_train/train.py saves sigmoid outputs as channel_pred;
    # evaluation_channel.py __main__ then applies a sigmoid once more.
    model_df["channel_pred"] = binary_probability(pred)
    model_df["channel_pred"] = 1 / (1 + np.exp(-model_df["channel_pred"]))
    model = "model"
    testing_split = soz_testing_split(final_split_csv)
    ratio_df = calculate_all_ratios(
        testing_split, {model: model_df}, raw_channels_loader(raw_root)
    )
    missing = ratio_df[f"{model}_ratio"].isna()
    n_missing = int(missing.sum())
    n_missing_labelled = int((missing & ratio_df["ground_truth"].isin([0, 1])).sum())
    # Upstream would fail on NaN scores; labelled channels without predictions
    # (not present in the processed data) are left out of the channel metrics.
    channel_input = ratio_df[~(missing & ratio_df["ground_truth"].isin([0, 1]))]
    channel = evaluate_model_outputs(channel_input, [model], "ground_truth").iloc[0]
    patient = resection_ratio_calculation_originalmethod(
        ratio_df, [model], "ground_truth"
    ).iloc[0]
    return {
        "channel_level": {k: v for k, v in channel.to_dict().items() if k != "model"},
        "patient_level": {k: v for k, v in patient.to_dict().items() if k != "model"},
        "ground_truth_counts": ratio_df["ground_truth"]
        .value_counts()
        .sort_index()
        .to_dict(),
        "n_test_recordings": int(len(testing_split)),
        "n_good_channels": int(len(ratio_df)),
        "n_good_channels_without_predictions": n_missing,
        "n_labelled_channels_without_predictions": n_missing_labelled,
        "n_channel_windows": int(len(model_df)),
    }


# ---------------------------------------------------------------------------
# anatomical
# Port of omni_ieeg/exploratory_model/anatomical/evaluation.py:15-59
# ---------------------------------------------------------------------------
def anatomical_calculate_metrics(model_df: pd.DataFrame) -> dict:
    prob_class_cols = [
        col
        for col in model_df.columns
        if col.startswith("prob_class_") and col[len("prob_class_") :].isdigit()
    ]
    prob_class_cols = sorted(
        prob_class_cols, key=lambda x: int(x[len("prob_class_") :])
    )
    channel_pred = []
    channel_true = []
    unique_edf_names = model_df["edf_name"].unique()
    for edf_name in unique_edf_names:
        edf_df = model_df[model_df["edf_name"] == edf_name]
        grouped_indices = edf_df.groupby("name")["start_indices"].apply(
            lambda x: tuple(sorted(x.unique()))
        )
        assert grouped_indices.nunique() == 1, (
            "all channel does not have all start indices"
        )
        unique_name = edf_df["name"].unique()
        for channel_name in unique_name:
            channel_df = edf_df[edf_df["name"] == channel_name]
            assert len(channel_df["channel_true"].unique()) == 1, (
                f"Channel true values should be same for {edf_name}"
            )
            label = channel_df["channel_true"].unique()[0]
            if label == -1:
                continue
            channel_pred_dict = {}
            max_count = 0
            max_col = None
            for prob_class_col in prob_class_cols:
                channel_pred_dict[prob_class_col] = channel_df[prob_class_col].sum()
                if channel_pred_dict[prob_class_col] > max_count:
                    max_count = channel_pred_dict[prob_class_col]
                    max_col = prob_class_col
            max_col = int(max_col[len("prob_class_") :])
            channel_pred.append(max_col)
            channel_true.append(label)

    precison_macro = precision_score(channel_true, channel_pred, average="macro")
    recall_macro = recall_score(channel_true, channel_pred, average="macro")
    f1_macro = f1_score(channel_true, channel_pred, average="macro")
    accuracy = accuracy_score(channel_true, channel_pred)
    balanced_accuracy = balanced_accuracy_score(channel_true, channel_pred)
    confusion_matrix_value = confusion_matrix(channel_true, channel_pred)
    return {
        "channel_level": {
            "precision_macro": precison_macro,
            "recall_macro": recall_macro,
            "f1_macro": f1_macro,
            "accuracy": accuracy,
            "balanced_accuracy": balanced_accuracy,
            "confusion_matrix": confusion_matrix_value,
            "n_channels": len(channel_true),
            "class_counts": np.bincount(np.asarray(channel_true, dtype=int)).tolist(),
        }
    }


def score_anatomical(pred: pd.DataFrame) -> dict:
    model_df = upstream_metadata(pred)
    result = anatomical_calculate_metrics(model_df)
    result["n_recordings"] = int(model_df["edf_name"].nunique())
    result["n_channel_windows"] = int(len(model_df))
    return result


# ---------------------------------------------------------------------------
# hfo
# Port of omni_ieeg/event_model/benchmark/benchmark.py:4-40
# ---------------------------------------------------------------------------
def hfo_calculate_metrics(result_df: pd.DataFrame) -> dict:
    y_true = result_df["true_class"]
    y_pred = result_df["predicted_class"]
    acc = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average="macro")
    recall_macro = recall_score(y_true, y_pred, average="macro")
    f1_macro = f1_score(y_true, y_pred, average="macro")
    cm = confusion_matrix(y_true, y_pred)
    return {
        "accuracy": acc,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "confusion_matrix": cm,
    }


def score_hfo(pred: pd.DataFrame) -> dict:
    # train_3label.py: ``_, pred_classes = torch.max(all_outputs, 1)``.
    prefix = "logit_class_" if _class_ids(pred, "logit_class_") else "prob_class_"
    class_ids = _class_ids(pred, prefix)
    scores = pred[[f"{prefix}{c}" for c in class_ids]].to_numpy(np.float64)
    result_df = pd.DataFrame(
        {
            "predicted_class": np.asarray(class_ids)[np.argmax(scores, axis=1)],
            "true_class": pred["y_true"].to_numpy(np.int64),
        }
    )
    return {
        "event_level": {
            **hfo_calculate_metrics(result_df),
            "n_events": int(len(result_df)),
            "class_counts": result_df["true_class"]
            .value_counts()
            .sort_index()
            .to_dict(),
            "class_names": {"0": "Artifact", "1": "Real", "2": "Spike"},
        }
    }


# ---------------------------------------------------------------------------
# Cell handling
# ---------------------------------------------------------------------------
def _cell_config(cell_dir: Path) -> dict:
    path = Path(cell_dir) / ".hydra" / "config.yaml"
    if not path.exists():
        return {}
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(path)
    try:
        return OmegaConf.to_container(cfg, resolve=True)
    except Exception:
        return OmegaConf.to_container(cfg, resolve=False)


def score_cell(
    cell_dir,
    task: str | None = None,
    final_split_csv: str | None = None,
    raw_root: str | None = None,
) -> dict:
    cell_dir = Path(cell_dir)
    cfg = _cell_config(cell_dir)
    dataset = cfg.get("dataset", {}) or {}
    options = dataset.get("provider_options", {}) or {}
    paths = cfg.get("paths", {}) or {}
    task = task or dataset.get("task")
    if task not in TASKS:
        raise ValueError(f"unknown or missing task {task!r}; expected one of {TASKS}")
    pred = load_predictions(cell_dir)
    if task in ("ictal_interictal", "sleep_awake"):
        metrics = score_ictal_sleep(pred)
    elif task == "soz_outcome":
        final_split_csv = (
            final_split_csv
            or options.get("final_split_csv")
            or paths.get("omni_final_split_csv")
        )
        raw_root = raw_root or paths.get("omni_raw_root")
        if not final_split_csv or not raw_root:
            raise ValueError(
                "soz_outcome needs the official final_split.csv and the raw BIDS "
                "tree: pass --final_split_csv and --raw_root, or set "
                "paths.omni_final_split_csv and paths.omni_raw_root in the run."
            )
        metrics = score_soz(pred, final_split_csv, raw_root)
    elif task == "anatomical":
        metrics = score_anatomical(pred)
    else:
        metrics = score_hfo(pred)
    return _jsonable(
        {
            "protocol": PROTOCOL,
            "task": task,
            "predictions_file": "test_predictions.csv",
            **metrics,
        }
    )


def headline(result: dict) -> str:
    task = result["task"]
    if task in ("ictal_interictal", "sleep_awake"):
        wl = result["window_level"]
        return (
            f"AUC={wl['auc']:.4f} F1={wl['f1_macro']:.4f} "
            f"BA={wl['balanced_accuracy']:.4f} windows={wl['n_windows']}"
        )
    if task == "soz_outcome":
        ch, pt = result["channel_level"], result["patient_level"]
        return (
            f"channel AUC={ch['auc']:.4f} PR-AUC={ch['pr_auc']:.4f} F1={ch['f1']:.4f} | "
            f"patient AUC={pt['auc']:.4f} PR-AUC={pt['pr_auc']:.4f}"
        )
    if task == "anatomical":
        ch = result["channel_level"]
        return (
            f"F1={ch['f1_macro']:.4f} Acc={ch['accuracy']:.4f} "
            f"BA={ch['balanced_accuracy']:.4f} channels={ch['n_channels']}"
        )
    ev = result["event_level"]
    return f"Acc={ev['accuracy']:.4f} F1={ev['f1_macro']:.4f} events={ev['n_events']}"


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("cell_dir", type=Path)
    parser.add_argument("--task", choices=TASKS, default=None)
    parser.add_argument("--final_split_csv", default=None)
    parser.add_argument("--raw_root", default=None)
    args = parser.parse_args(argv)
    result = score_cell(args.cell_dir, args.task, args.final_split_csv, args.raw_root)
    out = args.cell_dir / "omni_sc_metrics.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"omni single-channel metrics ({out}): {result['task']} {headline(result)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
