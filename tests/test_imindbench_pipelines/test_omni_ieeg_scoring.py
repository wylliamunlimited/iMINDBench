"""imindbench.omni_ieeg_scoring and omni_ieeg_paper_scoring: aggregation rules
and parity with upstream.

The parity tests run the official Omni-iEEG evaluation code on the same inputs
and require an exact match. They are skipped when the upstream repository is not
available (set OMNI_IEEG_REPO to point at a checkout of
https://github.com/Omni-iEEG/Omni-iEEG).
"""

import ast
import contextlib
import io
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from imindbench import omni_ieeg_paper_scoring
from imindbench import omni_ieeg_scoring as scorer_module

UPSTREAM = Path(os.environ.get("OMNI_IEEG_REPO", "") or "/nonexistent/Omni-iEEG")
upstream_only = pytest.mark.skipif(
    not (UPSTREAM / "omni_ieeg").is_dir(),
    reason="upstream Omni-iEEG repository not available (set OMNI_IEEG_REPO)",
)


@pytest.fixture(scope="module")
def scorer():
    return scorer_module


@pytest.fixture(scope="module")
def upstream_path():
    import sys

    if str(UPSTREAM) not in sys.path:
        sys.path.insert(0, str(UPSTREAM))
    return UPSTREAM


def _printed_floats(text):
    return {
        key.strip(): float(value)
        for key, value in re.findall(
            r"([A-Za-z][A-Za-z0-9_ ]*?): ([-0-9.e+inf]+)", text
        )
    }


def _predictions(
    rows,
    n_classes=2,
    columns=("recording_id", "window_start_sec", "y_true", "channel_name"),
):
    pred = pd.DataFrame(
        rows, columns=list(columns) + [f"logit_class_{c}" for c in range(n_classes)]
    )
    logits = pred[[f"logit_class_{c}" for c in range(n_classes)]].to_numpy(float)
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    for c in range(n_classes):
        pred[f"prob_class_{c}"] = probs[:, c]
    pred["window_end_sec"] = pred["window_start_sec"] + 1.0
    pred["channel_id"] = "c"
    return pred


def _ictal_predictions():
    # Two recordings x 2 windows x 2 channels; interictal (1) scores higher.
    rows = []
    for rid, label, base in (("rec_a", 1, 1.0), ("rec_b", 0, -1.0)):
        for window in (1.0, 61.0):
            for channel, shift in (("A0-A1", 0.0), ("A1-A2", 0.5)):
                rows.append((rid, window, label, channel, 0.0, base + shift))
    return _predictions(rows)


def test_ictal_scores_average_logits_across_channels(scorer):
    pred = _ictal_predictions()
    result = scorer.score_ictal_sleep(pred)
    window = result["window_level"]
    assert window["n_windows"] == 4  # one score per (recording, window start)
    assert window["n_positive"] == 2 and window["n_negative"] == 2
    assert window["auc"] == 1.0
    assert result["n_recordings"] == 2
    assert result["n_channel_windows"] == 8
    # Aggregation is the mean of the raw logits, not of the probabilities.
    model_df = scorer.upstream_metadata(pred)
    model_df["channel_pred"] = scorer.binary_logit(pred)
    scores = model_df.groupby(["edf_name", "start_indices"])["channel_pred"].mean()
    assert set(np.round(scores.to_numpy(), 6)) == {1.25, -0.75}


def test_window_start_seconds_become_upstream_sample_indices(scorer):
    pred = _ictal_predictions()
    assert scorer.upstream_metadata(pred)["start_indices"].tolist()[:2] == [1000, 1000]


def test_binary_logit_and_probability_match_the_two_logit_head(scorer):
    pred = _predictions([("r", 1.0, 1, "A0-A1", -0.5, 1.5)])
    np.testing.assert_allclose(scorer.binary_logit(pred), [2.0])
    np.testing.assert_allclose(
        scorer.binary_probability(pred), [1 / (1 + np.exp(-2.0))]
    )


def test_hfo_predicts_the_argmax_class(scorer):
    rows = [
        ("r", 0.0, 0, "hfo_event", 2.0, 0.0, 1.0),
        ("r", 2.0, 1, "hfo_event", 0.0, 3.0, 1.0),
        ("r", 4.0, 2, "hfo_event", 0.0, 1.0, 5.0),
        ("r", 6.0, 2, "hfo_event", 4.0, 1.0, 0.0),
    ]
    pred = _predictions(rows, n_classes=3)
    event = scorer.score_hfo(pred)["event_level"]
    assert event["n_events"] == 4
    assert event["accuracy"] == 0.75  # the last event is misclassified as 0
    assert event["class_names"]["2"] == "Spike"


def test_anatomical_sums_probabilities_per_channel(scorer):
    # One recording, two channels, two windows each. Channel A0-A1 is class 0 and
    # only wins on the summed probabilities, not on every single window.
    rows = [
        ("r", 1.0, 0, "A0-A1", 3.0, 0.0, 0.0),
        ("r", 301.0, 0, "A0-A1", 0.0, 0.5, 0.0),
        ("r", 1.0, 1, "A1-A2", 0.0, 2.0, 0.0),
        ("r", 301.0, 1, "A1-A2", 0.0, 2.0, 0.0),
    ]
    pred = _predictions(rows, n_classes=3)
    channel = scorer.score_anatomical(pred)["channel_level"]
    assert channel["n_channels"] == 2
    assert channel["accuracy"] == 1.0
    assert np.array_equal(channel["confusion_matrix"], [[1, 0], [0, 1]])


def _soz_tree(tmp_path):
    """final_split.csv + channels.tsv of one test recording, as upstream reads them."""
    edf = "sub-t00/ses-01/ieeg/sub-t00_ses-01_task-interictal_run-01_ieeg.edf"
    channels = pd.DataFrame(
        {
            "name": ["A0-A1", "A1-A2", "A2-A3", "A3-A4"],
            "good": [1, 1, 1, 0],
            "soz": [0, 1, 0, 1],
            "resection": [0, 1, 1, 1],
        }
    )
    path = tmp_path / edf.replace("_ieeg.edf", "_channels.tsv")
    path.parent.mkdir(parents=True, exist_ok=True)
    channels.to_csv(path, sep="\t", index=False)
    final_split = pd.DataFrame(
        [
            {
                "patient_name": "sub-t00",
                "edf_name": edf,
                "dataset": "hup",
                "outcome": 1,
                "interictal": True,
                "has_soz": True,
                "has_resection": True,
                "has_anatomical": False,
                "frequency": 1000.0,
                "length": 400.0,
                "split": "test",
            }
        ]
    )
    csv = tmp_path / "final_split.csv"
    final_split.to_csv(csv, index=False)
    return edf, csv


def _soz_predictions(edf, logits):
    rid = re.sub(r"[^A-Za-z0-9_.-]+", "_", edf.removesuffix(".edf"))
    rows = []
    for channel, logit in logits.items():
        for window in (1.0, 61.0):
            rows.append((rid, window, -1, channel, 0.0, logit))
    return _predictions(rows)


def test_soz_channel_scores_apply_the_upstream_double_sigmoid(scorer, tmp_path):
    edf, csv = _soz_tree(tmp_path)
    pred = _soz_predictions(edf, {"A0-A1": 2.0, "A1-A2": -2.0, "A2-A3": 0.0})
    result = scorer.score_soz(pred, str(csv), str(tmp_path))
    # Good channels: label 1 (non-resected, outcome 1), 0 (soz), -1 (resected, not soz).
    assert result["ground_truth_counts"] == {-1: 1, 0: 1, 1: 1}
    assert result["n_good_channels"] == 3
    assert result["n_good_channels_without_predictions"] == 0
    # Only the two labelled channels enter the channel metrics.
    assert result["channel_level"]["n_channels"] == 2
    assert result["channel_level"]["auc"] == 1.0
    assert result["patient_level"]["n_patients"] == 1


def test_soz_reports_channels_without_predictions(scorer, tmp_path):
    edf, csv = _soz_tree(tmp_path)
    pred = _soz_predictions(edf, {"A0-A1": 2.0, "A1-A2": -2.0})
    result = scorer.score_soz(pred, str(csv), str(tmp_path))
    assert result["n_good_channels_without_predictions"] == 1
    assert result["n_labelled_channels_without_predictions"] == 0


def test_score_cell_writes_metrics_json(scorer, tmp_path):
    _ictal_predictions().to_csv(tmp_path / "test_predictions.csv", index=False)
    assert scorer.main([str(tmp_path), "--task", "ictal_interictal"]) == 0
    result = json.loads((tmp_path / "omni_sc_metrics.json").read_text())
    assert result["protocol"] == "omni_ieeg_single_channel"
    assert result["task"] == "ictal_interictal"
    assert result["window_level"]["auc"] == 1.0


def test_multi_channel_predictions_are_rejected(scorer, tmp_path):
    pred = _ictal_predictions()
    pred["channel_name"] = "A0-A1|A1-A2"
    pred.to_csv(tmp_path / "test_predictions.csv", index=False)
    with pytest.raises(ValueError, match="not single-channel"):
        scorer.score_cell(tmp_path, "ictal_interictal")


# ---------------------------------------------------------------------------
# Parity with the official Omni-iEEG evaluation code
# ---------------------------------------------------------------------------
@upstream_only
def test_ictal_parity_with_upstream(scorer, upstream_path):
    import omni_ieeg.exploratory_model.ictal_sleep.evaluation as upstream

    rng = np.random.default_rng(3)
    rows = []
    for rid, label in (("rec_a", 1), ("rec_b", 0), ("rec_c", 1)):
        for window in (1.0, 61.0, 121.0):
            for channel in ("A0-A1", "A1-A2", "A2-A3"):
                logit = rng.normal() + (0.7 if label == 1 else 0.0)
                rows.append((rid, window, label, channel, 0.0, logit))
    pred = _predictions(rows)
    ours = scorer.score_ictal_sleep(pred)["window_level"]

    model_df = scorer.upstream_metadata(pred)
    model_df["channel_pred"] = scorer.binary_logit(pred)
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        upstream.calculate_metrics({"m": model_df})
    printed = _printed_floats(buffer.getvalue())
    assert ours["auc"] == printed["AUC"]
    assert ours["f1_macro"] == printed["F1"]
    assert ours["precision_macro"] == printed["Precision"]
    assert ours["recall_macro"] == printed["Recall"]
    assert ours["balanced_accuracy"] == printed["balanced_accuracy"]
    assert ours["optimal_threshold"] == printed["optimal threshold"]


@upstream_only
def test_anatomical_parity_with_upstream(scorer, upstream_path):
    source = (
        UPSTREAM / "omni_ieeg/exploratory_model/anatomical/evaluation.py"
    ).read_text()
    # The upstream module calls calculate_metrics at import time, so the function
    # is extracted instead of imported.
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "calculate_metrics"
    )
    namespace: dict = {}
    exec(
        "import numpy as np\nimport pandas as pd\n"
        "from sklearn.metrics import (accuracy_score, balanced_accuracy_score, "
        "confusion_matrix, f1_score, precision_score, recall_score)\n",
        namespace,
    )
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), "upstream", "exec"),
        namespace,
    )

    rng = np.random.default_rng(5)
    rows = []
    for rid in ("rec_a", "rec_b"):
        for channel in ("A0-A1", "A1-A2", "A2-A3", "A3-A4"):
            label = int(rng.integers(0, 3))
            for window in (1.0, 301.0):
                logits = rng.normal(size=3)
                logits[label] += 0.8
                rows.append((rid, window, label, channel, *logits))
    pred = _predictions(rows, n_classes=3)
    ours = scorer.score_anatomical(pred)["channel_level"]
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        namespace["calculate_metrics"]({"m": scorer.upstream_metadata(pred)})
    printed = _printed_floats(buffer.getvalue())
    assert ours["f1_macro"] == printed["f1_macro"]
    assert ours["precision_macro"] == printed["precison_macro"]  # upstream spelling
    assert ours["recall_macro"] == printed["recall_macro"]
    assert ours["accuracy"] == printed["accuracy"]
    assert ours["balanced_accuracy"] == printed["balanced_accuracy"]


@upstream_only
def test_hfo_parity_with_upstream(scorer, upstream_path):
    import omni_ieeg.event_model.benchmark.benchmark as upstream

    rng = np.random.default_rng(7)
    labels = rng.integers(0, 3, size=60)
    logits = rng.normal(size=(60, 3))
    logits[np.arange(60), labels] += 0.9
    rows = [
        ("r", float(2 * i), int(labels[i]), "hfo_event", *logits[i]) for i in range(60)
    ]
    ours = scorer.score_hfo(_predictions(rows, n_classes=3))["event_level"]
    expected = upstream.calculate_metrics(
        pd.DataFrame({"predicted_class": logits.argmax(axis=1), "true_class": labels})
    )
    assert ours["accuracy"] == expected["accuracy"]
    assert ours["f1_macro"] == expected["f1_macro"]
    assert ours["precision_macro"] == expected["precision_macro"]
    assert ours["recall_macro"] == expected["recall_macro"]
    assert np.array_equal(
        np.asarray(ours["confusion_matrix"]), expected["confusion_matrix"]
    )


@upstream_only
def test_soz_parity_with_upstream(scorer, upstream_path, tmp_path):
    import omni_ieeg.channel_model.benchmark.evaluation_channel as upstream

    edf, csv = _soz_tree(tmp_path)
    # A second patient so the patient-level AUC has both outcomes.
    channels = pd.DataFrame(
        {
            "name": ["B0-B1", "B1-B2"],
            "good": [1, 1],
            "soz": [1, 0],
            "resection": [0, 0],
        }
    )
    edf2 = "sub-t01/ses-01/ieeg/sub-t01_ses-01_task-interictal_run-01_ieeg.edf"
    path = tmp_path / edf2.replace("_ieeg.edf", "_channels.tsv")
    path.parent.mkdir(parents=True, exist_ok=True)
    channels.to_csv(path, sep="\t", index=False)
    final_split = pd.read_csv(csv)
    row = final_split.iloc[0].to_dict()
    row.update({"patient_name": "sub-t01", "edf_name": edf2, "outcome": 0})
    pd.concat([final_split, pd.DataFrame([row])]).to_csv(csv, index=False)

    rng = np.random.default_rng(11)
    rows = []
    for source, names in (
        (edf, ["A0-A1", "A1-A2", "A2-A3"]),
        (edf2, ["B0-B1", "B1-B2"]),
    ):
        rid = re.sub(r"[^A-Za-z0-9_.-]+", "_", source.removesuffix(".edf"))
        for channel in names:
            for window in (1.0, 61.0, 121.0):
                rows.append((rid, window, -1, channel, 0.0, rng.normal()))
    pred = _predictions(rows)
    ours = scorer.score_soz(pred, str(csv), str(tmp_path))

    model_df = scorer.upstream_metadata(pred)
    # Upstream matches model rows on the EDF basename, so it is fed EDF paths
    # where the port uses processed recording ids.
    rid_to_edf = {
        re.sub(r"[^A-Za-z0-9_.-]+", "_", source.removesuffix(".edf")): source
        for source in (edf, edf2)
    }
    model_df["edf_name"] = model_df["edf_name"].map(rid_to_edf)
    model_df["channel_pred"] = scorer.binary_probability(pred)
    model_df["channel_pred"] = 1 / (1 + np.exp(-model_df["channel_pred"]))

    class _FakeDataset:
        def get_channels_for_edf(self, edf_path):
            return pd.read_csv(
                str(edf_path).replace("_ieeg.edf", "_channels.tsv"), sep="\t"
            )

    upstream.filtered_dataset = _FakeDataset()
    testing_split = scorer.soz_testing_split(str(csv))
    with contextlib.redirect_stdout(io.StringIO()):
        ratio_df = upstream.calculate_all_ratios(
            testing_split, {"m": model_df}, str(tmp_path)
        )
        channel = upstream.evaluate_model_outputs(ratio_df, ["m"], "ground_truth").iloc[
            0
        ]
        patient = upstream.resection_ratio_calculation_originalmethod(
            ratio_df, ["m"], "ground_truth"
        ).iloc[0]
    for key in (
        "auc",
        "pr_auc",
        "roc_auc",
        "f1",
        "precision",
        "recall",
        "specificity",
        "optimal_threshold",
    ):
        assert ours["channel_level"][key] == channel[key], key
    for key in ("auc", "pr_auc"):
        assert ours["patient_level"][key] == patient[key], key


# ---------------------------------------------------------------------------
# Paper-style scoring of the multi-channel ictal vs interictal protocol
# ---------------------------------------------------------------------------
def _write_paper_predictions(path, rows):
    pd.DataFrame(
        rows,
        columns=[
            "recording_id",
            "window_start_sec",
            "window_end_sec",
            "y_true",
            "prob_class_0",
            "prob_class_1",
        ],
    ).to_csv(path / "test_predictions.csv", index=False)


def test_one_second_windows_are_averaged_into_full_sixty_second_units(tmp_path):
    scorer = omni_ieeg_paper_scoring
    rows = []
    # rec_a (interictal): two full 60 s units + a partial third unit (dropped).
    for i in range(130):
        p = 0.9 if i < 60 else 0.7
        rows.append(("rec_a", 1.0 + i, 2.0 + i, 1, 1 - p, p))
    # rec_b (ictal): one full unit.
    for i in range(60):
        rows.append(("rec_b", 1.0 + i, 2.0 + i, 0, 0.8, 0.2))
    _write_paper_predictions(tmp_path, rows)

    result = scorer.score_cell(tmp_path, unit_sec=60.0, edge_trim_sec=1.0)

    wl = result["window_level"]
    assert wl["n_units"] == 3
    assert wl["n_positive"] == 2 and wl["n_negative"] == 1
    assert wl["auc"] == pytest.approx(1.0)
    assert result["model_window_sec"] == pytest.approx(1.0)
    assert "model_window_level_auc" in result


def test_sixty_second_windows_score_as_is(tmp_path):
    scorer = omni_ieeg_paper_scoring
    rows = [
        ("rec_a", 1.0, 61.0, 1, 0.4, 0.6),
        ("rec_a", 61.0, 121.0, 1, 0.7, 0.3),
        ("rec_b", 1.0, 61.0, 0, 0.5, 0.5),
        ("rec_b", 61.0, 121.0, 0, 0.9, 0.1),
    ]
    _write_paper_predictions(tmp_path, rows)

    result = scorer.score_cell(tmp_path, unit_sec=60.0, edge_trim_sec=1.0)

    wl = result["window_level"]
    assert wl["n_units"] == 4
    # scores: pos {0.6, 0.3}, neg {0.5, 0.1} -> 3 of 4 pairs ranked correctly.
    assert wl["auc"] == pytest.approx(0.75)
    assert 0.0 <= wl["balanced_accuracy"] <= 1.0
    assert "model_window_level_auc" not in result


def test_paper_main_writes_metrics_json(tmp_path):
    scorer = omni_ieeg_paper_scoring
    _write_paper_predictions(
        tmp_path,
        [("r1", 1.0, 61.0, 1, 0.2, 0.8), ("r2", 1.0, 61.0, 0, 0.9, 0.1)],
    )
    assert scorer.main([str(tmp_path)]) == 0
    payload = json.loads((tmp_path / "omni_paper_metrics.json").read_text())
    assert payload["protocol"] == "omni_ieeg_paper"
    assert np.isfinite(payload["window_level"]["auc"])
