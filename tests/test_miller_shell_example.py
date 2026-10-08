"""Run scripts/run_millerecog2019.sh as a dry run and check what it would launch.

Every launch in the script passes --cells, so each block must print exactly the
task/target pairs of its cell list. One command per launch is also composed with
Hydra and checked by validate_eval_config, so a dry run cannot hide a config
that would fail at the start of a real run.
"""

import json
import shlex
import subprocess
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from imindbench import launch
from imindbench.utils.pipeline_contracts import validate_eval_config

CELLS = Path(launch.CELL_MANIFEST_DIR) / "millerecog2019"
CLASS_PAIRS = ["0v1", "0v2", "0v3", "0v4", "1v2", "1v3", "1v4", "2v3", "2v4", "3v4"]

# Each block: the output groups it writes, in launch order. Every group is named
# after its cell list; the default block adds the prefix "within_session_".
FAMILIES = {
    None: ["within_session_binary", "within_session_multiclass"],
    "new_task_sets": ["new_sets_binary"],
    "controls": ["controls_binary", "controls_multiclass"],
    "class_pairs": [f"class_pair_{pair}" for pair in CLASS_PAIRS],
    "regression": ["regression"],
    "regression_sliding": ["regression_sliding"],
    "regression_bci4": ["regression_bci4"],
    "positions": ["positions_binary", "positions_multiclass"],
}


def _cells(name):
    manifest = json.loads((CELLS / f"{name}.json").read_text())
    return {
        (task, target)
        for task, values in manifest["tasks"].items()
        for target in values["subject_sessions"]
    }


def _dry_run(dataset_script, tmp_path, family, settings):
    script = dataset_script(dataset="millerecog2019", family=family, **settings)
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


def _fields(overrides):
    return dict(token.split("=", 1) for token in overrides)


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


def _check_family(dataset_script, tmp_path, family, settings):
    commands = _dry_run(dataset_script, tmp_path, family, settings)
    by_group = {}
    for config_dir, overrides in commands:
        fields = _fields(overrides)
        group = Path(fields["hydra.run.dir"]).relative_to(tmp_path / "runs").parts[0]
        by_group.setdefault(group, []).append((config_dir, overrides, fields))
    assert list(by_group) == FAMILIES[family]
    configs = {}
    for group, launched in by_group.items():
        actual = {
            (
                fields["dataset.task"],
                f"sub{fields['dataset.test_subject']}_sess{fields['dataset.test_session']}",
            )
            for _, _, fields in launched
        }
        # The script's task and target arrays must not drop any listed cell.
        assert actual == _cells(group.removeprefix("within_session_")), group
        assert len(launched) == len(actual)
        config_dir, overrides, _ = launched[0]
        configs[group] = _validate(config_dir, overrides)
    return configs


@pytest.mark.parametrize("pairing", range(11))
def test_default_block_runs_the_main_table_for_every_pairing(
    tmp_path, dataset_script, dataset_pairings, pairing
):
    settings = dataset_pairings("millerecog2019")[pairing]
    configs = _check_family(dataset_script, tmp_path, None, settings)
    assert configs["within_session_binary"].dataset.label_mode == "binary"
    assert configs["within_session_multiclass"].dataset.label_mode == "multiclass"
    assert len(_cells("binary")) + len(_cells("multiclass")) == 178


@pytest.mark.parametrize("family", [family for family in FAMILIES if family])
@pytest.mark.parametrize("model", ["logistic", "popt", "linear_baseline", "diver"])
def test_optional_blocks_run_their_cell_lists(
    tmp_path, dataset_script, dataset_pairings, family, model
):
    settings = next(
        row for row in dataset_pairings("millerecog2019") if row["MODEL"] == model
    )
    configs = _check_family(dataset_script, tmp_path, family, settings)
    for group, cfg in configs.items():
        if group.startswith("class_pair_"):
            first, second = group.removeprefix("class_pair_").split("v")
            assert list(cfg.dataset.class_pair) == [int(first), int(second)]
        if family.startswith("regression"):
            assert cfg.dataset.label_mode == "regression"
        if family == "regression_sliding":
            assert cfg.dataset.regression_target_last_samples == 2
        if family == "regression_bci4":
            assert cfg.dataset.regression_target_last_samples == 1
            assert list(cfg.dataset.fold_subset) == [0]
        if family == "positions":
            # DIVER's model config picks its own profile; the script replaces it.
            expected = "diver_mni_miller" if model == "diver" else "popt_miller"
            assert cfg.dataset.coordinate_profile == expected
