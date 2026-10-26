"""Run scripts/run_omni_ieeg.sh as a dry run and check what it would launch.

The default block launches one pooled cell per Omni-iEEG task with the
per-task settings of the single-channel protocol. Every command is also
composed with Hydra and checked by validate_eval_config, so a dry run cannot
hide a config that would fail at the start of a real run.
"""

import json
import shlex
import subprocess
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from imindbench import launch
from imindbench.utils.pipeline_contracts import validate_eval_config

TASKS = ["ictal_interictal", "sleep_awake", "soz_outcome", "anatomical", "hfo"]
WINDOW_SEC = {
    "ictal_interictal": 60,
    "sleep_awake": 60,
    "soz_outcome": 60,
    "anatomical": 300,
    "hfo": 2,
}
PRESET = {
    "ictal_interictal": "omni_diverstyle_HPF_noSTD_1000Hzto500Hz_w60s",
    "sleep_awake": "omni_diverstyle_HPF_noSTD_1000Hzto500Hz_w60s",
    "soz_outcome": "omni_diverstyle_HPF_noSTD_1000Hzto500Hz_w60s",
    "anatomical": "omni_diverstyle_HPF_noSTD_1000Hzto500Hz_w300s",
    "hfo": "omni_hfo_diverstyle_HPF_noSTD_1000Hzto500Hz",
}
MODELS = {
    "logistic": {},
    "mlp": {},
    "cnn": {},
    "diver": {
        "yaml_updates": {
            "paths/local.yaml": {
                "diver_checkpoint": "/path/to/ieeg_checkpoint.pt",
                "diver_shape_cache_dir": "/path/to/diver_shapes",
            }
        }
    },
}


def _dry_run(dataset_script, tmp_path, **settings):
    script = dataset_script(dataset="omni_ieeg", **settings)
    result = subprocess.run(
        ["bash", script], cwd=tmp_path, text=True, capture_output=True, check=True
    )
    commands = []
    for line in result.stdout.splitlines():
        tokens = shlex.split(line)
        config_dir = tokens[tokens.index("--config-dir") + 1]
        overrides = [token for token in tokens if "=" in token]
        commands.append((config_dir, overrides))
    assert not (tmp_path / "runs").exists()
    return commands


def _validate(config_dir, overrides):
    searchpath = f"hydra.searchpath={json.dumps(['file://' + config_dir])}"
    with initialize_config_dir(config_dir=str(launch.CONF_DIR), version_base="1.1"):
        cfg = compose(
            config_name="config",
            overrides=[
                *(token for token in overrides if not token.startswith("hydra.")),
                searchpath,
            ],
        )
    validate_eval_config(cfg)
    return cfg


@pytest.mark.parametrize("model", sorted(MODELS))
def test_default_block_runs_one_pooled_cell_per_task(tmp_path, dataset_script, model):
    settings = dict(MODEL=model, EXPERIMENT="default", **MODELS[model])
    commands = _dry_run(dataset_script, tmp_path, **settings)
    assert len(commands) == len(TASKS)
    for task, (config_dir, overrides) in zip(TASKS, commands, strict=True):
        fields = dict(token.split("=", 1) for token in overrides)
        assert fields["dataset"] == "omni_ieeg_single_channel"
        assert fields["dataset.task"] == task
        assert fields["dataset.test_subject"] == "all"
        assert fields["dataset.test_session"] == "all"
        assert fields["dataset.regime"] == "hold-out-subject"
        assert fields["model"] == model
        assert fields["preprocessor"] == PRESET[task]
        assert fields["runtime.save_test_predictions"] == "true"
        assert fields["runner.train_sampler"] == "balanced"
        run_dir = Path(fields["hydra.run.dir"]).relative_to(tmp_path / "runs")
        assert run_dir.parts == (
            "single_channel",
            f"{model}_{PRESET[task]}",
            "hold-out-subject",
            task,
            "all",
        )
        cfg = _validate(config_dir, overrides)
        options = cfg.dataset.provider_options
        assert options.single_channel is True
        assert float(options.window_sec) == WINDOW_SEC[task]
        assert float(options.edge_trim_sec) == (0.0 if task == "hfo" else 1.0)
        if task in ("anatomical", "hfo"):
            assert cfg.dataset.label_mode == "multiclass"
            assert cfg.model.selection_metric == "f1"
        else:
            assert cfg.dataset.label_mode == "binary"
            assert cfg.model.selection_metric == "roc_auc"
        assert cfg.runner.ignore_label == (-1 if task == "soz_outcome" else None)
        # The split file of the two cross-split tasks comes from paths/local.yaml.
        split_key = {
            "ictal_interictal": "omni_ictal_split_csv",
            "sleep_awake": "omni_sleep_split_csv",
        }.get(task)
        if split_key is not None:
            assert (
                overrides.count(
                    f"dataset.provider_options.split_csv=${{paths.{split_key}}}"
                )
                == 1
            )


def test_sub_window_must_tile_every_window(tmp_path, dataset_script):
    script = Path(dataset_script(dataset="omni_ieeg", MODEL="logistic"))
    source = script.read_text().replace('SUB_WINDOW_SEC=""', 'SUB_WINDOW_SEC="7"', 1)
    script.write_text(source)
    result = subprocess.run(
        ["bash", script], cwd=tmp_path, text=True, capture_output=True
    )
    assert result.returncode != 0
    assert "does not tile the 60s window of task 'ictal_interictal'" in result.stderr
    assert result.stdout == ""

    source = script.read_text().replace('SUB_WINDOW_SEC="7"', 'SUB_WINDOW_SEC="5"', 1)
    script.write_text(source)
    result = subprocess.run(
        ["bash", script], cwd=tmp_path, text=True, capture_output=True, check=True
    )
    lines = result.stdout.splitlines()
    assert len(lines) == len(TASKS)
    # The HFO event is one unit upstream and is never cut.
    assert all(
        ("++model.sub_window_sec=5" in line) == ("hfo" not in line) for line in lines
    )
