"""The omni_ieeg dataset entry: provider row, profiles, configs and presets."""

from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from omni_fakes import FakeOmniIEEG, install_fake_omni

from imindbench import launch
from imindbench.preprocessors import build_preprocessor
from imindbench.utils import data_adapter
from imindbench.utils.data_adapter import COORDINATE_PROFILES
from imindbench.utils.pipeline_contracts import (
    VALID_COORDINATE_PROFILES,
    build_processed_split_provider,
    dataset_single_channel,
    is_multi_subject,
    needs_region_intersection_pool,
    provider_property,
    resolve_provider_n_folds,
    validate_eval_config,
)

CONF = Path(launch.CONF_DIR)
OMNI_PRESETS = sorted(path.stem for path in (CONF / "preprocessor").glob("omni_*.yaml"))

# The per-task settings of scripts/run_omni_ieeg.sh.
TASK_SETTINGS = {
    "ictal_interictal": [
        "dataset.provider_options.window_sec=60",
        "dataset.provider_options.split_csv=${paths.omni_ictal_split_csv}",
        "dataset.label_mode=binary",
        "++model.selection_metric=roc_auc",
        "preprocessor=omni_diverstyle_HPF_noSTD_1000Hzto500Hz_w60s",
    ],
    "sleep_awake": [
        "dataset.provider_options.window_sec=60",
        "dataset.provider_options.split_csv=${paths.omni_sleep_split_csv}",
        "dataset.label_mode=binary",
        "++model.selection_metric=roc_auc",
        "preprocessor=omni_diverstyle_HPF_noSTD_1000Hzto500Hz_w60s",
    ],
    "soz_outcome": [
        "dataset.provider_options.window_sec=60",
        "dataset.label_mode=binary",
        "++model.selection_metric=roc_auc",
        "runner.ignore_label=-1",
        "preprocessor=omni_diverstyle_HPF_noSTD_1000Hzto500Hz_w60s",
    ],
    "anatomical": [
        "dataset.provider_options.window_sec=300",
        "dataset.label_mode=multiclass",
        "++model.selection_metric=f1",
        "preprocessor=omni_diverstyle_HPF_noSTD_1000Hzto500Hz_w300s",
    ],
    "hfo": [
        "dataset.provider_options.window_sec=2",
        "dataset.provider_options.edge_trim_sec=0",
        "dataset.label_mode=multiclass",
        "++model.selection_metric=f1",
        "preprocessor=omni_hfo_diverstyle_HPF_noSTD_1000Hzto500Hz",
    ],
}


@pytest.fixture
def fake_omni(monkeypatch):
    return install_fake_omni(monkeypatch)


def _compose(*overrides):
    with initialize_config_dir(config_dir=str(CONF), version_base="1.1"):
        return compose(
            config_name="config",
            overrides=["paths=example", "experiment=default", *overrides],
        )


def test_dataset_list_entry():
    assert provider_property("omni_ieeg", "signal_unit") == "uV"
    assert provider_property("omni_ieeg", "electrode_subtype") == "depth"
    assert is_multi_subject("omni_ieeg", "hold-out-subject") is True


def test_single_fold_comes_from_the_dataset_class(fake_omni):
    assert (
        resolve_provider_n_folds(
            dataset_provider="omni_ieeg", regime="hold-out-subject"
        )
        == 1
    )


def test_omni_coordinate_profiles():
    assert "omni_none" in VALID_COORDINATE_PROFILES
    frame, transform = COORDINATE_PROFILES["omni_none"]["omni_ieeg"]
    assert frame == "omni_none"
    zeros = np.zeros((3, 3), dtype=np.float32)
    np.testing.assert_array_equal(transform(zeros), zeros)
    # Every profile a model config may pick has a row for Omni-iEEG.
    for profile in ("popt_lip", "diver_mni", "omni_none"):
        assert "omni_ieeg" in COORDINATE_PROFILES[profile], profile
    # The omni_none profile keeps the other datasets usable without positions.
    for provider in ("millerecog2019", "kelesbyd2024", "neuroprobev2"):
        assert provider in COORDINATE_PROFILES["omni_none"]


@pytest.mark.parametrize(
    ("dataset", "model"),
    [
        ("omni_ieeg", "diver"),
        ("omni_ieeg_single_channel", "logistic"),
        ("omni_ieeg_single_channel", "diver"),
    ],
)
def test_dataset_configs_compose_and_pass_validation(dataset, model):
    cfg = _compose(
        f"dataset={dataset}",
        f"model={model}",
        "preprocessor=omni_diverstyle_HPF_noSTD_1000Hzto500Hz_w1s",
    )
    validate_eval_config(cfg)
    assert cfg.dataset.provider == "omni_ieeg"
    assert cfg.dataset.regime == "hold-out-subject"
    assert cfg.dataset.test_subject == "all"
    assert cfg.dataset.test_session == "all"
    assert cfg.dataset.coordinate_profile == (
        "diver_mni" if model == "diver" else "omni_none"
    )
    assert cfg.dataset.provider_options.single_channel is (
        dataset == "omni_ieeg_single_channel"
    )
    # paths=example: the processed root is shared, the split tables are not set.
    assert cfg.dataset.root == "/path/to/brainsets/processed"
    assert cfg.dataset.provider_options.split_csv is None


def test_multi_channel_protocol_needs_a_variable_channel_model():
    """Omni-iEEG has no brain areas to pool by, so aligned-channel models can
    only run the single-channel protocol (one channel per sample)."""
    cfg = _compose(
        "dataset=omni_ieeg",
        "model=logistic",
        "preprocessor=omni_diverstyle_HPF_noSTD_1000Hzto500Hz_w1s",
    )
    with pytest.raises(ValueError, match="single_channel=true"):
        validate_eval_config(cfg)
    assert needs_region_intersection_pool("omni_ieeg", "hold-out-subject", True)
    assert not needs_region_intersection_pool(
        "omni_ieeg", "hold-out-subject", True, single_channel=True
    )
    assert dataset_single_channel(cfg.dataset) is False
    cfg.dataset.provider_options.single_channel = True
    assert dataset_single_channel(cfg.dataset) is True
    validate_eval_config(cfg)


@pytest.mark.parametrize("task", sorted(TASK_SETTINGS))
@pytest.mark.parametrize("model", ["logistic", "mlp", "cnn", "diver"])
def test_every_task_of_the_run_script_composes(task, model):
    cfg = _compose(
        "dataset=omni_ieeg_single_channel",
        f"model={model}",
        f"dataset.task={task}",
        *TASK_SETTINGS[task],
        "runtime.save_test_predictions=true",
        "runtime.save_test_prediction_details=true",
        "runner.train_sampler=balanced",
    )
    validate_eval_config(cfg)
    assert cfg.dataset.label_mode == (
        "multiclass" if task in ("anatomical", "hfo") else "binary"
    )
    assert cfg.model.selection_metric == (
        "f1" if task in ("anatomical", "hfo") else "roc_auc"
    )
    assert cfg.runner.ignore_label == (-1 if task == "soz_outcome" else None)
    assert cfg.runner.train_sampler == "balanced"
    # DIVER's model config sets its own profile; it has an Omni-iEEG row.
    expected_profile = "diver_mni" if model == "diver" else "omni_none"
    assert cfg.dataset.coordinate_profile == expected_profile


def test_pooled_test_ids_must_be_strings():
    cfg = _compose(
        "dataset=omni_ieeg_single_channel",
        "model=logistic",
        "preprocessor=omni_diverstyle_HPF_noSTD_1000Hzto500Hz_w60s",
        "dataset.test_subject=3",
    )
    with pytest.raises(TypeError, match="test_subject"):
        validate_eval_config(cfg)


def test_provider_options_must_be_a_mapping():
    cfg = _compose(
        "dataset=omni_ieeg_single_channel",
        "model=logistic",
        "preprocessor=omni_diverstyle_HPF_noSTD_1000Hzto500Hz_w60s",
    )
    cfg.dataset.provider_options = "window_sec=60"
    with pytest.raises(TypeError, match="provider_options"):
        validate_eval_config(cfg)


def test_provider_options_reach_the_dataset_class(fake_omni):
    cfg = _compose(
        "dataset=omni_ieeg_single_channel",
        "model=logistic",
        "preprocessor=omni_diverstyle_HPF_noSTD_1000Hzto500Hz_w60s",
        "dataset.root=/data",
        "dataset.task=sleep_awake",
        "dataset.provider_options.window_sec=30",
        "dataset.provider_options.split_csv=/splits/sleep_split_cross.csv",
    )
    provider = build_processed_split_provider(
        dataset_provider="omni_ieeg",
        dataset_cfg=cfg.dataset,
        split="test",
        fold_idx=0,
        regime="hold-out-subject",
    )
    assert isinstance(provider, FakeOmniIEEG)
    assert provider.options["window_sec"] == 30
    assert provider.options["split_csv"] == "/splits/sleep_split_cross.csv"
    assert provider.options["single_channel"] is True
    assert provider.options["train_window_sampling"] == "upstream_random"
    assert provider.options["train_windows_per_recording"] == 5
    assert provider.single_channel and provider.task == "sleep_awake"


def test_other_providers_do_not_forward_provider_options():
    cfg = _compose(
        "dataset=neuroprobev2", "model=logistic", "preprocessor=multi_stft_2048Hz"
    )
    assert "provider_options" not in cfg.dataset


def test_every_omni_preset_is_listed_and_resamples():
    assert OMNI_PRESETS == [
        "omni_diverstyle_HPF_noSTD_1000Hzto500Hz_w1s",
        "omni_diverstyle_HPF_noSTD_1000Hzto500Hz_w300s",
        "omni_diverstyle_HPF_noSTD_1000Hzto500Hz_w60s",
        "omni_hfo_diverstyle_HPF_noSTD_1000Hzto500Hz",
    ]
    for name in OMNI_PRESETS:
        cfg = OmegaConf.load(CONF / "preprocessor" / f"{name}.yaml")
        steps = [step.name for step in cfg.chain]
        assert "downsample" not in steps, name
        assert steps[-1] == "resample", name
        assert cfg.chain[-1].source_rate == 1000 and cfg.chain[-1].target_rate == 500
        assert "time_domain_filter_diver_style" in steps
        # The HFO events are standalone waveforms: no filter context around them.
        assert ("context_window" in steps) == (not name.startswith("omni_hfo")), name
        build_preprocessor(cfg)


@pytest.mark.parametrize(
    ("preset", "window_sec", "expected_samples"),
    [
        ("omni_diverstyle_HPF_noSTD_1000Hzto500Hz_w1s", 1.0, 500),
        ("omni_diverstyle_HPF_noSTD_1000Hzto500Hz_w60s", 60.0, 30000),
        ("omni_hfo_diverstyle_HPF_noSTD_1000Hzto500Hz", 2.0, 1000),
    ],
)
def test_presets_build_single_channel_folds_at_500hz(
    tmp_path, monkeypatch, preset, window_sec, expected_samples
):
    class Fake(FakeOmniIEEG):
        sampling_rate_hz = 1000.0
        windows_per_recording = 2

    install_fake_omni(monkeypatch, Fake)
    task = "hfo" if preset.startswith("omni_hfo") else "ictal_interictal"
    cfg = OmegaConf.load(CONF / "preprocessor" / f"{preset}.yaml")
    dataset_cfg = OmegaConf.create(
        dict(
            provider="omni_ieeg",
            root=str(tmp_path),
            dirname="omni_ieeg",
            subset_tier="full",
            label_mode="multiclass" if task == "hfo" else "binary",
            task=task,
            regime="hold-out-subject",
            test_subject="all",
            test_session="all",
            coordinate_profile="omni_none",
            uniquify_channel_ids_with_subject=True,
            uniquify_channel_ids_with_session=True,
            provider_options=dict(
                single_channel=True,
                window_sec=window_sec,
                edge_trim_sec=0.0 if task == "hfo" else 1.0,
            ),
        )
    )
    fold = data_adapter.build_neuroprobe_torch_fold(
        dataset_cfg,
        preprocessor=build_preprocessor(cfg),
        preprocessor_cfg=cfg,
        fold_idx=0,
        seed=0,
        require_coords=False,
        needs_pool=False,
    )
    for split in ("train", "val", "test"):
        sample = fold[f"{split}_split"][0]
        assert tuple(sample["x"].shape) == (1, expected_samples), split
        assert sample["single_channel_sample"] is True
        assert len(sample["channel_names"]) == 1


def test_launcher_target_all_selects_the_pooled_cell():
    args = launch.parser().parse_args(
        [
            "--dataset",
            "omni_ieeg_single_channel",
            "--model",
            "logistic",
            "--preprocessor",
            "omni_diverstyle_HPF_noSTD_1000Hzto500Hz_w60s",
            "--experiment",
            "default",
            "--output-root",
            "/runs",
            "--task",
            "ictal_interictal",
            "--target",
            "all",
            "--regime",
            "hold-out-subject",
        ]
    )
    (job,) = launch.build_commands(args)
    assert "dataset.test_subject=all" in job["command"]
    assert "dataset.test_session=all" in job["command"]
    assert "dataset.regime=hold-out-subject" in job["command"]
    assert job["result"].endswith("population_btbankall_all_ictal_interictal.json")
