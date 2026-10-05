"""The regression track: scoring, validation, and both runners on fake data."""

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from miller_fakes import install_fake_miller
from omegaconf import OmegaConf

from imindbench import launch
from imindbench.preprocessors import build_preprocessor
from imindbench.sklearn_runner import SKLearnRunner
from imindbench.torch_runner import TorchRunner
from imindbench.utils import data_adapter
from imindbench.utils import regression as regression_utils
from imindbench.utils.fold_helpers import evaluate_variable_fold
from imindbench.utils.logging_utils import (
    DEFAULT_RESULTS_TIME_BIN,
    log_final_wandb_metrics,
    log_fold_metrics,
    resolve_task_mode_config,
)
from imindbench.utils.pipeline_contracts import validate_eval_config

# ---- scoring ----------------------------------------------------------------


def test_pearson_r_handles_constant_and_short_inputs():
    assert regression_utils.pearson_r([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)
    assert np.isnan(regression_utils.pearson_r([1, 1, 1], [1, 2, 3]))
    assert np.isnan(regression_utils.pearson_r([1], [1]))


def test_score_joins_windows_end_to_end():
    true = np.array([[0.0, 1.0], [2.0, 3.0]])
    scores = regression_utils.score(true * 2 + 1, true)
    assert scores["traj_r"] == pytest.approx(1.0)
    assert scores["mean_r"] == pytest.approx(1.0)
    assert scores["n_windows"] == 2
    assert scores["traj_len"] == 2
    perfect = regression_utils.score(true, true)
    assert perfect["traj_mse"] == 0.0
    assert perfect["traj_r2"] == pytest.approx(1.0)


def test_regression_targets_lookup_and_last_samples():
    targets = regression_utils.RegressionTargets("flex_thumb", out_last=2)
    targets.add("rec", traj=np.arange(12.0).reshape(3, 4), mean=np.ones(3))
    assert targets.traj_len == 2
    traj, mean = targets.lookup(["rec", "rec"], [2, 0])
    np.testing.assert_array_equal(traj, [[10.0, 11.0], [2.0, 3.0]])
    np.testing.assert_array_equal(mean, [1.0, 1.0])
    with pytest.raises(IndexError, match="out of range"):
        targets.lookup(["rec"], [3])
    with pytest.raises(KeyError, match="No regression targets"):
        targets.lookup(["other"], [0])
    with pytest.raises(ValueError, match="longer than"):
        regression_utils.RegressionTargets("x", out_last=5).add(
            "rec", traj=np.zeros((1, 4)), mean=np.zeros(1)
        )


# ---- validation ---------------------------------------------------------------


def _compose(model="logistic", dataset="millerecog2019", *overrides):
    with initialize_config_dir(config_dir=str(launch.CONF_DIR), version_base="1.1"):
        return compose(
            config_name="config",
            overrides=[
                "paths=example",
                f"dataset={dataset}",
                f"model={model}",
                "preprocessor=miller_multi_stft_1000Hz",
                "experiment=default",
                "dataset.label_mode=regression",
                "dataset.task=flex_thumb",
                "dataset.test_session=13",
                *overrides,
            ],
        )


def test_every_model_declares_the_regression_keys():
    cfg = _compose("cnn")
    assert cfg.model.regression_head == "trained_mse"
    assert cfg.model.regression_head_lambda == 1.0


@pytest.mark.parametrize(
    ("model", "overrides"),
    [
        ("logistic", []),
        ("linear_baseline", ["model.regression_head=ridge"]),
        ("diver", ["model.regression_head=ridge"]),
        ("diver", []),
        ("mlp", []),
        ("linear_baseline", ["model.regression_head_lambda_mode=gcv"]),
        ("logistic", ["dataset.regression_target_last_samples=2"]),
    ],
)
def test_regression_configs_pass_validation(model, overrides):
    validate_eval_config(_compose(model, "millerecog2019", *overrides))


@pytest.mark.parametrize(
    ("model", "overrides", "message"),
    [
        ("mlp", ["model.regression_head=ridge"], "one linear layer"),
        ("barista", ["model.regression_head=ridge"], "one linear layer"),
        (
            "diver",
            ["model.regression_head=ridge", "model.ft_head_style=flatten_mlp"],
            "one linear layer",
        ),
        ("diver", ["model.regression_head=ridge", "model.ft_mup=true"], "one linear"),
        ("linear_baseline", ["model.regression_head=lstsq"], "regression_head"),
        ("logistic", ["model.regression_head_lambda=0"], "> 0"),
        ("logistic", ["model.regression_head_lambda_mode=gcv"], "torch"),
        ("linear_baseline", ["model.regression_head_lambda_grid=[]"], "non-empty"),
        ("logistic", ["dataset.merge_val_into_test=true"], "merge_val_into_test"),
        ("logistic", ["dataset.max_train_samples_per_subject=5"], "max_train"),
        ("logistic", ["dataset.regression_target_last_samples=-1"], ">= 0"),
    ],
)
def test_bad_regression_configs_are_rejected(model, overrides, message):
    with pytest.raises(ValueError, match=message):
        validate_eval_config(_compose(model, "millerecog2019", *overrides))


def test_regression_is_rejected_for_datasets_without_targets():
    cfg = _compose("logistic")
    cfg.dataset.provider = "kelesbyd2024"
    with pytest.raises(ValueError, match="unsupported for dataset.provider"):
        validate_eval_config(cfg)


def test_last_samples_needs_regression():
    cfg = _compose(
        "logistic", "millerecog2019", "dataset.regression_target_last_samples=2"
    )
    cfg.dataset.label_mode = "binary"
    with pytest.raises(ValueError, match="needs dataset.label_mode='regression'"):
        validate_eval_config(cfg)


# ---- end to end on the fake dataset ----------------------------------------


@pytest.fixture
def fake_miller(monkeypatch):
    return install_fake_miller(monkeypatch)


def _run_fold(tmp_path, model, *overrides):
    cfg = _compose(
        model,
        "millerecog2019",
        f"dataset.root={tmp_path}",
        "model.device=cpu",
        "runtime.seed=0",
        *overrides,
    )
    cfg.preprocessor = OmegaConf.create({"chain": [{"name": "raw"}]})
    validate_eval_config(cfg)
    fold = data_adapter.build_neuroprobe_torch_fold(
        cfg.dataset,
        preprocessor=build_preprocessor(cfg.preprocessor),
        preprocessor_cfg=cfg.preprocessor,
        fold_idx=0,
        seed=0,
        require_coords=False,
        needs_pool=False,
    )
    runner = SKLearnRunner(cfg) if cfg.model.backend == "sklearn" else TorchRunner(cfg)
    torch.manual_seed(0)
    return evaluate_variable_fold(
        0, {"fold": fold}, cfg=cfg, runner=runner, seed=cfg.runtime.seed
    )


REGRESSION_KEYS = {
    f"{split}_{metric}"
    for split in ("train", "val", "test")
    for metric in ("traj_r", "mean_r", "traj_mse", "traj_r2", "n_windows")
} | {"label_mode", "traj_len", "head", "target"}


@pytest.mark.parametrize(
    ("model", "overrides", "head"),
    [
        ("logistic", [], "sklearn_ridge(alpha=1.0)"),
        ("linear_baseline", ["model.regression_head=ridge"], "ridge"),
        (
            "linear_baseline",
            ["model.total_steps=300", "model.learning_rate=0.01"],
            "trained_mse",
        ),
        (
            "mlp",
            ["model.max_iter=60", "model.learning_rate=0.003"],
            "trained_mse",
        ),
    ],
)
def test_regression_fold_recovers_the_target(
    tmp_path, fake_miller, model, overrides, head
):
    result = _run_fold(tmp_path, model, *overrides)
    assert set(result) == REGRESSION_KEYS
    assert result["head"] == head
    assert result["target"] == "flex_thumb"
    assert result["traj_len"] == 20
    assert result["test_traj_r"] > 0.9, result
    assert result["test_n_windows"] == 20
    assert not any("accuracy" in key or "roc_auc" in key for key in result)
    log_fold_metrics(0, result)


def test_last_samples_shortens_the_scored_trajectory(tmp_path, fake_miller):
    result = _run_fold(tmp_path, "logistic", "dataset.regression_target_last_samples=2")
    assert result["traj_len"] == 2
    assert result["test_traj_r"] > 0.9


def test_fold_subset_with_regression(tmp_path, fake_miller):
    from imindbench.utils.fold_helpers import iter_variable_channel_folds

    cfg = _compose("logistic", "millerecog2019", f"dataset.root={tmp_path}")
    cfg.preprocessor = OmegaConf.create({"chain": [{"name": "raw"}]})
    folds = list(
        iter_variable_channel_folds(
            n_folds=2,
            dataset_cfg=cfg.dataset,
            preprocessor=build_preprocessor(cfg.preprocessor),
            preprocessor_cfg=cfg.preprocessor,
            seed=0,
            require_coords=False,
            needs_pool=False,
            fold_subset=[0],
        )
    )
    assert [fold["fold_idx"] for fold in folds] == [0]
    assert folds[0]["fold"]["regression_targets"].traj_len == 20


def test_classification_folds_carry_no_regression_targets(tmp_path, fake_miller):
    cfg = _compose("logistic", "millerecog2019", f"dataset.root={tmp_path}")
    cfg.dataset.label_mode = "binary"
    cfg.dataset.task = "move_vs_rest"
    cfg.preprocessor = OmegaConf.create({"chain": [{"name": "raw"}]})
    fold = data_adapter.build_neuroprobe_torch_fold(
        cfg.dataset,
        preprocessor=build_preprocessor(cfg.preprocessor),
        preprocessor_cfg=cfg.preprocessor,
        fold_idx=0,
        seed=0,
        require_coords=False,
        needs_pool=False,
    )
    assert fold["regression_targets"] is None


def test_ridge_head_finds_the_wrapped_diver_linear():
    class Wrapped(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.module = torch.nn.Linear(3, 2)

    class FakeDiver(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Linear(5, 5)
            self.ft_core_model = Wrapped()

    model = OmegaConf.create({})
    holder = type("Holder", (), {"model": FakeDiver()})()
    runner = TorchRunner.__new__(TorchRunner)
    runner.cfg = model
    assert runner._probe_head_linear(holder) is holder.model.ft_core_model.module


def test_result_config_and_wandb_summary_for_regression():
    cfg = _compose(
        "logistic", "millerecog2019", "dataset.regression_target_last_samples=2"
    )
    assert resolve_task_mode_config(cfg.dataset) == {
        "label_mode": "regression",
        "regression_target_last_samples": 2,
    }

    class Run:
        logged = []

        def log(self, payload):
            self.logged.append(payload)

    fold = {
        "label_mode": "regression",
        "train_traj_r": 0.9,
        "val_traj_r": 0.8,
        "test_traj_r": 0.7,
        "test_mean_r": 0.6,
    }
    run = Run()
    log_final_wandb_metrics(
        run,
        {DEFAULT_RESULTS_TIME_BIN: {"folds": [fold, {**fold, "test_traj_r": 0.5}]}},
    )
    summary = run.logged[-1]
    assert summary["final/test_traj_r_mean"] == pytest.approx(0.6)
    assert summary["final/n_completed_folds"] == 2
    assert not any("accuracy" in key for key in summary)
