"""Run the real evaluation entry point on the fake Omni-iEEG dataset.

Each test calls imindbench.run_eval the way scripts/run_omni_ieeg.sh does (one
pooled cell, target ``all``, the single-channel protocol) and reads the result
JSON it writes. The fake dataset (omni_fakes.py) has synthetic signals a simple
model can decode, so the scores also show the run learned something, and the
test predictions it saves are scored with imindbench.omni_ieeg_scoring.
"""

import csv
import json
import sys

import pandas as pd
import pytest
from omni_fakes import FakeOmniIEEG, install_fake_omni

from imindbench import omni_ieeg_scoring, run_eval


class RecordingFake(FakeOmniIEEG):
    """Short windows keep the runs small; every instance is kept for checks."""

    windows_per_recording = 3
    instances = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        type(self).instances.append(self)


@pytest.fixture
def run(tmp_path, monkeypatch):
    install_fake_omni(monkeypatch, RecordingFake)
    monkeypatch.setattr(RecordingFake, "instances", [])
    config = tmp_path / "config"
    (config / "preprocessor").mkdir(parents=True)
    # The fake signal is short and synthetic, so the run uses it as stored.
    (config / "preprocessor/raw.yaml").write_text("chain:\n  - name: raw\n")

    def run_eval_main(
        name, *overrides, task="ictal_interictal", window_sec=4, windows=3
    ):
        monkeypatch.setattr(RecordingFake, "windows_per_recording", windows)
        run_dir = tmp_path / "runs" / name
        argv = [
            "run_eval",
            "--config-dir",
            str(config),
            "paths=example",
            "dataset=omni_ieeg_single_channel",
            f"dataset.root={tmp_path}",
            "preprocessor=raw",
            "experiment=default",
            "wandb.enabled=false",
            "runtime.verbose=false",
            "runtime.seed=0",
            "runner.num_workers=0",
            "runner.pin_memory=false",
            "runner.persistent_workers=false",
            f"dataset.task={task}",
            f"dataset.provider_options.window_sec={window_sec}",
            "dataset.test_subject=all",
            "dataset.test_session=all",
            f"hydra.run.dir={run_dir}",
            *overrides,
        ]
        monkeypatch.setattr(sys, "argv", argv)
        run_eval.main()
        (result_file,) = run_dir.glob("population_*.json")
        assert result_file.name == f"population_btbankall_all_{task}.json"
        return json.loads(result_file.read_text()), run_dir

    return run_eval_main


def _fold(result):
    (time_bin,) = result["evaluation_results"]["btbankall_all"]["population"].values()
    (fold,) = time_bin["folds"]
    assert fold["fold_idx"] == 0
    return fold


def test_ictal_run_is_one_pooled_cell(run):
    result, _ = run("ictal", "model=logistic")
    assert result["config"]["eval_name"] == "ictal_interictal"
    assert _fold(result)["test_roc_auc"] > 0.9
    # Every split asked the dataset class for the single-channel protocol with
    # the window length passed through provider_options.
    splits = {p.split for p in RecordingFake.instances}
    assert splits == {"train", "val", "test"}
    assert all(p.single_channel for p in RecordingFake.instances)
    assert all(p.options["window_sec"] == 4 for p in RecordingFake.instances)
    assert all(p.options["split_csv"] is None for p in RecordingFake.instances)


def test_torch_run_saves_scorable_test_predictions(run):
    result, run_dir = run(
        "ictal_mlp",
        "model=mlp",
        "model.device=cpu",
        "model.max_iter=5",
        "++model.selection_metric=roc_auc",
        "runner.train_sampler=balanced",
        "runtime.save_test_predictions=true",
        "runtime.save_test_prediction_details=true",
    )
    assert _fold(result)["test_roc_auc"] > 0.9
    with open(run_dir / "test_predictions.csv") as handle:
        rows = list(csv.DictReader(handle))
    # 2 test patients x 3 windows x 3 good channels.
    assert len(rows) == 18
    assert {row["channel_name"] for row in rows} == {"A0-A1", "A1-A2", "A2-A3"}
    assert {"logit_class_0", "logit_class_1", "prob_class_1"} <= set(rows[0])
    # The scorer reads the task from the run's Hydra config.
    assert omni_ieeg_scoring.main([str(run_dir)]) == 0
    metrics = json.loads((run_dir / "omni_sc_metrics.json").read_text())
    assert metrics["task"] == "ictal_interictal"
    assert metrics["window_level"]["n_windows"] == 6
    assert metrics["window_level"]["auc"] == 1.0


def test_soz_run_keeps_unlabelled_test_channels(run, tmp_path):
    result, run_dir = run(
        "soz",
        "model=mlp",
        "model.device=cpu",
        "model.max_iter=5",
        "runner.ignore_label=-1",
        "runner.train_sampler=balanced",
        "runtime.save_test_predictions=true",
        "runtime.save_test_prediction_details=true",
        task="soz_outcome",
    )
    fold = _fold(result)
    assert fold["test_roc_auc"] > 0.9
    pred = pd.read_csv(run_dir / "test_predictions.csv")
    # 2 test patients x 3 windows x 4 channels, one of them unlabelled (-1).
    assert len(pred) == 24
    assert pred["y_true"].value_counts().to_dict() == {1: 12, 0: 6, -1: 6}
    # Score it the upstream way: final_split.csv and the raw *_channels.tsv.
    raw_root = tmp_path / "raw"
    rows = []
    for patient in ("sub-t00", "sub-t01"):
        edf = f"{patient}/ses-01/ieeg/{patient}_ses-01_task-interictal_run-01_ieeg.edf"
        tsv = raw_root / edf.replace("_ieeg.edf", "_channels.tsv")
        tsv.parent.mkdir(parents=True)
        pd.DataFrame(
            {
                "name": ["A0-A1", "A1-A2", "A2-A3", "A3-A4"],
                "good": [1, 1, 1, 1],
                "soz": [0, 1, 0, 0],
                "resection": [0, 1, 0, 1],
            }
        ).to_csv(tsv, sep="\t", index=False)
        rows.append(
            {
                "patient_name": patient,
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
        )
    final_split = tmp_path / "final_split.csv"
    pd.DataFrame(rows).to_csv(final_split, index=False)
    metrics = omni_ieeg_scoring.score_cell(
        run_dir, "soz_outcome", final_split_csv=str(final_split), raw_root=str(raw_root)
    )
    assert metrics["n_good_channels_without_predictions"] == 0
    assert metrics["channel_level"]["n_channels"] == 6
    assert metrics["channel_level"]["auc"] == 1.0


def test_anatomical_run_is_five_class(run):
    result, _ = run(
        "anatomical",
        "model=logistic",
        "dataset.label_mode=multiclass",
        task="anatomical",
        window_sec=6,
    )
    assert result["config"]["label_mode"] == "multiclass"
    assert _fold(result)["test_accuracy"] > 0.9


def test_hfo_run_scores_events(run):
    result, run_dir = run(
        "hfo",
        "model=mlp",
        "model.device=cpu",
        "model.max_iter=40",
        "model.learning_rate=0.01",
        "model.batch_size=16",
        "++model.selection_metric=f1",
        "dataset.label_mode=multiclass",
        "dataset.provider_options.edge_trim_sec=0",
        "runtime.save_test_predictions=true",
        "runtime.save_test_prediction_details=true",
        task="hfo",
        window_sec=2,
        windows=12,
    )
    assert _fold(result)["test_accuracy"] > 0.9
    metrics = omni_ieeg_scoring.score_cell(run_dir, "hfo")
    # 2 test patients x 12 events.
    assert metrics["event_level"]["n_events"] == 24
    assert metrics["event_level"]["accuracy"] == 1.0
