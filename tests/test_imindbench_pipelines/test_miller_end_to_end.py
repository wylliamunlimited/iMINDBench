"""Run the real evaluation entry point on the fake Miller dataset.

Each test calls imindbench.run_eval the way the launcher does (Hydra command
line, one task on one recording) and reads the result JSON it writes. The fake
dataset (miller_fakes.py) has synthetic signals a simple model can decode, so
the scores also show the run learned something.
"""

import json
import sys

import pytest
from miller_fakes import install_fake_miller

from imindbench import run_eval


@pytest.fixture
def run(tmp_path, monkeypatch):
    install_fake_miller(monkeypatch)
    config = tmp_path / "config"
    (config / "preprocessor").mkdir(parents=True)
    # The fake signal is short and synthetic, so the run uses it as stored.
    (config / "preprocessor/raw.yaml").write_text("chain:\n  - name: raw\n")

    def run_eval_main(name, *overrides, task="move_vs_rest", session=1):
        run_dir = tmp_path / "runs" / name
        argv = [
            "run_eval",
            "--config-dir",
            str(config),
            "paths=example",
            "dataset=millerecog2019",
            f"dataset.root={tmp_path}",
            "preprocessor=raw",
            "experiment=default",
            "wandb.enabled=false",
            "runtime.verbose=false",
            "runtime.seed=0",
            f"dataset.task={task}",
            "dataset.test_subject=3",
            f"dataset.test_session={session}",
            f"hydra.run.dir={run_dir}",
            *overrides,
        ]
        monkeypatch.setattr(sys, "argv", argv)
        run_eval.main()
        (result_file,) = run_dir.glob("population_*.json")
        return json.loads(result_file.read_text())

    return run_eval_main


def _folds(result, key="btbank3_1"):
    (time_bin,) = result["evaluation_results"][key]["population"].values()
    return time_bin["folds"]


def test_binary_run_writes_a_result(run):
    result = run("binary", "model=logistic")
    assert result["config"]["eval_name"] == "move_vs_rest"
    folds = _folds(result)
    assert [fold["fold_idx"] for fold in folds] == [0, 1]
    assert all(fold["test_roc_auc"] > 0.9 for fold in folds)


def test_torch_model_run_writes_a_result(run):
    result = run("mlp", "model=mlp", "model.device=cpu", "model.max_iter=5")
    assert [fold["fold_idx"] for fold in _folds(result)] == [0, 1]


def test_multiclass_run_writes_a_result(run):
    result = run(
        "multiclass",
        "model=logistic",
        "dataset.label_mode=multiclass",
        task="which_effector",
    )
    assert result["config"]["label_mode"] == "multiclass"
    assert "class_pair" not in result["config"]
    assert all(fold["test_accuracy"] > 0.9 for fold in _folds(result))


def test_class_pair_run_writes_a_binary_result(run):
    result = run(
        "class_pair",
        "model=logistic",
        "dataset.label_mode=multiclass",
        "dataset.class_pair=[0,2]",
        task="which_effector",
    )
    assert result["config"]["label_mode"] == "multiclass"
    assert result["config"]["class_pair"] == [0, 2]
    # Two classes are left, so the result has binary scores.
    assert all(fold["test_roc_auc"] > 0.9 for fold in _folds(result))


def test_regression_run_writes_trajectory_scores(run):
    result = run(
        "regression",
        "model=logistic",
        "dataset.label_mode=regression",
        "dataset.fold_subset=[0]",
        task="flex_thumb",
        session=13,
    )
    assert result["config"]["label_mode"] == "regression"
    assert result["config"]["fold_subset"] == [0]
    (fold,) = _folds(result, key="btbank3_13")
    assert fold["fold_idx"] == 0
    assert fold["target"] == "flex_thumb"
    assert fold["test_traj_r"] > 0.9
    assert "test_roc_auc" not in fold
