"""Runner options used by the Omni-iEEG single-channel protocol.

Adapted from torch_brain's neuroprobe_eval tests of the Omni-iEEG harness.

``runner.ignore_label`` keeps samples for their predictions while excluding them
from the trained classes and the metrics (unlabelled SOZ test channels), and
``runtime.save_test_predictions`` writes the columns the single-channel scorer
needs. Both default to off, so other suites are unaffected.
"""

import csv
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from imindbench.models import build_model
from imindbench.torch_runner import TorchRunner
from imindbench.utils.collate import variable_channel_collate


class _ListDataset(torch.utils.data.Dataset):
    def __init__(self, samples):
        self.samples = list(samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def _sample(label, split, sample_idx, *, single_channel=False):
    value = 0.1 + max(label, 0)
    sample = {
        "x": np.array([[value, value + 0.1, value + 0.2]], dtype=np.float32),
        "y": int(label),
        "channel_ids": ["c0"],
        "channel_coords": np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
        "coordinate_profile": "omni_none",
        "seq_id": np.zeros((1,), dtype=np.int64),
        "recording_id": f"{split}_rec",
        "split": split,
        "sample_idx": int(sample_idx),
        "window_start_sec": float(sample_idx),
        "window_end_sec": float(sample_idx) + 1.0,
    }
    if single_channel:
        sample["channel_names"] = ["A0"]
    return sample


MODEL_CONF = Path(__file__).resolve().parents[2] / "imindbench/conf/model/mlp.yaml"


def _cfg(tmp_path, *, ignore_label=None):
    model = OmegaConf.load(MODEL_CONF)
    model.update({"device": "cpu", "max_iter": 1, "patience": 1, "batch_size": 2})
    return OmegaConf.create(
        {
            "model": OmegaConf.to_container(model),
            "dataset": {"provider": "omni_ieeg", "label_mode": "binary"},
            "runner": {"ignore_label": ignore_label},
            "runtime": {
                "seed": 0,
                "deterministic": False,
                "save_test_predictions": True,
            },
        }
    )


def _loaders(test_samples, *, single_channel=False):
    def build(split, labels):
        samples = [
            _sample(label, split, i, single_channel=single_channel)
            for i, label in enumerate(labels)
        ]
        return torch.utils.data.DataLoader(
            _ListDataset(samples),
            batch_size=2,
            shuffle=False,
            collate_fn=variable_channel_collate,
        )

    return (
        build("train", [0, 1, 0, 1]),
        build("val", [0, 1, 0, 1]),
        build("test", test_samples),
    )


def _run(cfg, test_labels, *, single_channel=False, monkeypatch=None, tmp_path=None):
    runner = TorchRunner(cfg)
    model = build_model(cfg.model)
    train_loader, val_loader, test_loader = _loaders(
        test_labels, single_channel=single_channel
    )
    if tmp_path is not None:
        monkeypatch.chdir(tmp_path)
    return runner.run_fold(
        model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
    )


def test_ignored_test_labels_do_not_break_the_fold(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, ignore_label=-1)
    result = _run(cfg, [0, 1, -1, -1], monkeypatch=monkeypatch, tmp_path=tmp_path)
    assert np.isfinite(result["test_roc_auc"])
    assert np.isfinite(result["test_accuracy"])
    # Predictions are written for every test sample, ignored ones included.
    with open(tmp_path / "test_predictions.csv") as handle:
        rows = list(csv.DictReader(handle))
    assert [int(r["y_true"]) for r in rows] == [0, 1, -1, -1]
    # Multi-channel samples without names still get the (empty) column.
    assert [r["channel_name"] for r in rows] == ["", "", "", ""]
    assert [r["channel_id"] for r in rows] == ["c0", "c0", "c0", "c0"]


def test_ignore_label_is_encoded_as_the_loss_ignore_index(tmp_path):
    targets = torch.tensor([0, 1, -1])
    class_map = {0: 0, 1: 1}
    encoded = TorchRunner(_cfg(tmp_path, ignore_label=-1))._encode_targets_for_loss(
        targets, class_to_index=class_map, context="test"
    )
    # -100 is CrossEntropyLoss's default ignore_index.
    assert encoded.tolist() == [0, 1, -100]
    # Unknown labels are still rejected when they are not the ignore label.
    with pytest.raises(ValueError, match="not present in the train class map"):
        TorchRunner(_cfg(tmp_path))._encode_targets_for_loss(
            targets, class_to_index={0: 1, 1: 0}, context="test"
        )


def test_folds_with_unseen_test_labels_are_skipped_without_ignore_label():
    from imindbench.utils.fold_helpers import _find_unseen_eval_classes

    # This is the guard that protects the runner in a real fold: a test split
    # holding -1 is skipped unless runner.ignore_label removes it from the counts.
    assert _find_unseen_eval_classes({0: 2, 1: 2}, {0: 1, 1: 1, -1: 2}) == [-1]
    assert _find_unseen_eval_classes({0: 2, 1: 2}, {0: 1, 1: 1}) == []


def test_metrics_ignore_the_ignored_label(tmp_path, monkeypatch):
    # Every test sample is ignored -> metrics are nan rather than wrong.
    cfg = _cfg(tmp_path, ignore_label=-1)
    result = _run(cfg, [-1, -1], monkeypatch=monkeypatch, tmp_path=tmp_path)
    assert np.isnan(result["test_roc_auc"])
    assert np.isnan(result["test_accuracy"])


def test_predictions_carry_channel_and_logit_columns(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, ignore_label=-1)
    _run(
        cfg, [0, 1, -1], single_channel=True, monkeypatch=monkeypatch, tmp_path=tmp_path
    )
    with open(tmp_path / "test_predictions.csv") as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0]) == [
        "recording_id",
        "window_start_sec",
        "window_end_sec",
        "y_true",
        "prob_class_0",
        "prob_class_1",
        "channel_id",
        "channel_name",
        "logit_class_0",
        "logit_class_1",
    ]
    assert [r["channel_name"] for r in rows] == ["A0", "A0", "A0"]
    assert [r["channel_id"] for r in rows] == ["c0", "c0", "c0"]
    for row in rows:
        logits = np.array([float(row["logit_class_0"]), float(row["logit_class_1"])])
        probs = np.array([float(row["prob_class_0"]), float(row["prob_class_1"])])
        softmax = np.exp(logits) / np.exp(logits).sum()
        np.testing.assert_allclose(softmax, probs, atol=1e-6)
