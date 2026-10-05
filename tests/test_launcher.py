"""Check native experiment configs, shell selections and shared grid execution."""

import json
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from imindbench import launch
from imindbench.utils.pipeline_contracts import validate_eval_config

CONF = launch.CONF_DIR
ROOT = Path(__file__).resolve().parents[1]


def _args(*extra):
    return launch.parser().parse_args(
        [
            "--dataset",
            "neuroprobev2",
            "--model",
            "logistic",
            "--preprocessor",
            "multi_stft_2048Hz",
            "--experiment",
            "default",
            "--output-root",
            "/runs",
            "--paths",
            "example",
            "--task",
            "onset",
            "--target",
            "sub1_sess1",
            "--device",
            "cpu",
            *extra,
        ]
    )


def _overrides(command):
    return {
        token.split("=", 1)[0].lstrip("+"): token.split("=", 1)[1]
        for token in command
        if "=" in token
    }


@pytest.mark.parametrize(
    "family,dataset",
    [
        (family, dataset)
        for family in [
            "within_session",
            "within_dataset",
            "multi_dataset",
            "sample_efficiency",
        ]
        for dataset in ["neuroprobev2", "kelesbyd2024", "berezutskayapippi2022"]
        if family != "sample_efficiency" or dataset == "neuroprobev2"
    ],
)
def test_family_shell_configs_compose_and_pass_runtime_contract(
    tmp_path, dataset_script, dataset_pairings, family, dataset
):
    if family == "within_session":
        settings = dataset_pairings(dataset)
    elif family in {"within_dataset", "multi_dataset"}:
        settings = [{"MODEL": "popt"}]
    else:
        settings = [{}]
    lines = []
    for selection in settings:
        script = dataset_script(
            family=family,
            dataset=dataset,
            TASKS=["onset"],
            **selection,
        )
        result = subprocess.run(
            ["bash", script], cwd=tmp_path, text=True, capture_output=True, check=True
        )
        lines.extend(result.stdout.splitlines())
    seen = set()
    with initialize_config_dir(config_dir=str(CONF), version_base="1.1"):
        for line in lines:
            command = shlex.split(line)
            assert command[3] == "--config-dir"
            overrides = command[5:]
            # Compose API takes config search paths as overrides, unlike the CLI.
            search_path = json.dumps("file://" + command[4])
            overrides.append(f"hydra.searchpath=[{search_path}]")
            signature = tuple(
                v
                for v in overrides
                if not any(
                    v.startswith(k)
                    for k in [
                        "dataset.test_subject=",
                        "dataset.test_session=",
                        "hydra.run.dir=",
                    ]
                )
            )
            if signature in seen:
                continue
            seen.add(signature)
            cfg = compose(config_name="config", overrides=overrides)
            validate_eval_config(cfg)
            expected_cap = (
                "auto" if family in {"within_dataset", "multi_dataset"} else None
            )
            assert cfg.dataset.max_train_samples_per_subject == expected_cap
            assert cfg.dataset.decodable_subject_sessions_only == (
                family in {"within_dataset", "multi_dataset"}
            )
            if cfg.model.name == "linear_baseline":
                assert cfg.preprocessor.chain[4].device == cfg.model.device == "cpu"
                assert (
                    cfg.preprocessor.chain[4].upstream_ckpt
                    == "/path/to/stft_large_pretrained.pth"
                )
            if cfg.model.name == "diver":
                assert cfg.model.upstream_ckpt == "/path/to/ieeg_checkpoint.pt"
                assert cfg.model.model_dir == "/path/to/diver_shapes"
            if cfg.model.name == "barista":
                assert cfg.dataset.brain_area_key == (
                    "localization_Destrieux"
                    if dataset == "neuroprobev2"
                    else "label_destrieux"
                )
            if cfg.model.name == "htnet":
                assert "wav" in _overrides(overrides)["preprocessor"]
    assert seen
    assert not (tmp_path / "runs").exists()


def test_population_filter_matches_manifest_and_dry_run_is_read_only(tmp_path):
    args = _args(
        "--model",
        "popt",
        "--experiment",
        "decodable",
        "--regime",
        "hold-in-session",
    )
    args.output_root = tmp_path / "runs"
    population = json.loads(
        (
            launch.POPULATION_DIR / "stft_or_htnet_500hz_val_mean0p60/neuroprobev2.json"
        ).read_text()
    )
    expected = {
        (task, target)
        for task, entry in population["tasks"].items()
        for target in entry["subject_sessions"]
    }
    args.task = list(population["tasks"])
    args.target = sorted({target for _, target in expected})
    jobs = launch.build_commands(args)
    actual = {
        (_overrides(job["command"])["dataset.task"], Path(job["run_dir"]).name)
        for job in jobs
    }
    assert actual == expected
    assert not args.output_root.exists()


def test_default_preset_does_not_filter_or_cap_training_samples():
    with initialize_config_dir(config_dir=str(CONF), version_base="1.1"):
        cfg = compose(
            config_name="config",
            overrides=[
                "paths=example",
                "model=logistic",
                "preprocessor=multi_stft_2048Hz",
                "experiment=default",
            ],
        )
    assert cfg.dataset.max_train_samples_per_subject is None
    assert cfg.dataset.decodable_subject_sessions_only is False
    assert cfg.paths.decodable_subject_sessions_dir is None


def test_retired_cohort_flag_is_not_silently_ignored():
    with initialize_config_dir(config_dir=str(CONF), version_base="1.1"):
        cfg = compose(
            config_name="config",
            overrides=[
                "paths=example",
                "model=logistic",
                "preprocessor=multi_stft_2048Hz",
                "++dataset.train_decodable_subject_sessions_only=true",
            ],
        )
    with pytest.raises(
        ValueError, match="renamed to dataset.decodable_subject_sessions_only"
    ):
        validate_eval_config(cfg)


def test_explicit_selections_do_not_require_catalog_membership():
    jobs = launch.build_commands(
        _args("--task", "custom_task", "--target", "sub99_sess2")
    )
    assert len(jobs) == 1
    overrides = _overrides(jobs[0]["command"])
    assert overrides["dataset.task"] == "custom_task"
    assert overrides["dataset.test_subject"] == "99"
    assert overrides["dataset.test_session"] == "2"


@pytest.mark.parametrize("target", ["all", "sub1", "sub01_sess1", "sub1_sess-1"])
def test_invalid_target_syntax_is_rejected(target):
    with pytest.raises(ValueError, match="Invalid target"):
        launch.build_commands(_args("--target", target))


@pytest.mark.parametrize("missing", ["--task", "--target"])
def test_cli_requires_explicit_task_and_target(missing):
    args = [
        "--dataset",
        "neuroprobev2",
        "--model",
        "logistic",
        "--preprocessor",
        "stft_2048Hz",
        "--output-root",
        "/runs",
    ]
    args += ["--target", "sub1_sess1"] if missing == "--task" else ["--task", "onset"]
    with pytest.raises(SystemExit) as exc:
        launch.parser().parse_args(args)
    assert exc.value.code == 2


def test_transfer_preset_composes_with_packaged_manifest():
    with initialize_config_dir(config_dir=str(CONF), version_base="1.1"):
        cfg = compose(
            config_name="config",
            overrides=[
                "paths=example",
                "model=popt",
                "preprocessor=multi_stft_2048Hz",
                "experiment=decodable",
                "dataset.regime=hold-in-session",
            ],
        )
    validate_eval_config(cfg)
    assert Path(cfg.paths.decodable_subject_sessions_dir) == (
        launch.POPULATION_DIR / "stft_or_htnet_500hz_val_mean0p60"
    )


def test_empty_decodable_tasks_are_skipped(tmp_path):
    (tmp_path / "neuroprobev2.json").write_text(
        json.dumps(
            {
                "tasks": {
                    "onset": {"subject_sessions": []},
                    "speech": {"subject_sessions": ["sub1_sess1"]},
                }
            }
        )
    )
    args = _args(
        "--model",
        "popt",
        "--experiment",
        "decodable",
        "--decodable-rule",
        "stft_or_htnet_500hz_val_mean0p60",
        "--decodable-dir",
        str(tmp_path),
        "--task",
        "onset",
        "speech",
    )
    jobs = launch.build_commands(args)
    assert len(jobs) == 1
    assert _overrides(jobs[0]["command"])["dataset.task"] == "speech"
    args.task = ["onset"]
    with pytest.raises(ValueError, match="No evaluations remain"):
        launch.build_commands(args)


def test_external_config_cli_composes_without_running_experiments(tmp_path):
    config = tmp_path / "config"
    (config / "paths").mkdir(parents=True)
    (config / "experiment").mkdir()
    (config / "paths/local.yaml").write_text((CONF / "paths/example.yaml").read_text())
    (config / "experiment/custom.yaml").write_text(
        "# @package _global_\nmodel:\n  max_iter: 17\n"
    )
    args = _args(
        "--limit",
        "1",
        "--config-dir",
        str(config),
        "--paths",
        "local",
        "--experiment",
        "custom",
    )
    args.output_root = tmp_path / "runs"
    job = launch.build_commands(args)[0]
    result = subprocess.run(
        [*job["command"], "--cfg", "job"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    assert "dataset_root: /path/to/brainsets/processed" in result.stdout
    assert "max_iter: 17" in result.stdout
    assert not args.output_root.exists()


@pytest.mark.parametrize(
    "override",
    [
        "model=cnn",
        "dataset.task=speech",
        "hydra.run.dir=/elsewhere",
        "runtime.overwrite=true",
        "model.name=other",
        "dataset.provider=other",
        "dataset.subset_tier=full",
        "model.device=cuda:1",
    ],
)
def test_identity_overrides_rejected(override):
    with pytest.raises(ValueError, match="selection flags"):
        launch.build_commands(_args("--set", override))


def test_cli_tuning_overrides_native_experiment_settings():
    args = _args(
        "--model",
        "mlp",
        "--experiment",
        "default",
        "--limit",
        "1",
        "--set",
        "model.tol=0.02",
        "--set",
        "model.tol=0.03",
    )
    job = launch.build_commands(args)[0]
    with initialize_config_dir(config_dir=str(CONF), version_base="1.1"):
        cfg = compose(config_name="config", overrides=job["command"][3:])
    assert cfg.model.tol == 0.03
    assert cfg.runner.num_workers == 4


@pytest.mark.parametrize(
    "options",
    [
        ["--sweep", "model.tol=0.1,0.1"],
        ["--sweep", "model.tol=0.1,0.2", "--set", "model.tol=0.3"],
        ["--limit", "0"],
    ],
)
def test_ambiguous_or_invalid_grids_are_rejected(options):
    with pytest.raises(ValueError):
        launch.build_commands(_args(*options))


def test_execution_skips_existing_results_and_retries_missing_results(tmp_path):
    result = tmp_path / "job" / "population.json"
    command = [
        sys.executable,
        "-c",
        f"from pathlib import Path; Path({str(result)!r}).write_text('{{}}')",
    ]
    job = {"command": command, "run_dir": str(result.parent), "result": str(result)}
    assert launch.execute_commands([job], tmp_path) == 0
    assert result.is_file()
    before = result.stat().st_mtime_ns
    assert launch.execute_commands([job], tmp_path) == 0
    assert result.stat().st_mtime_ns == before
    different = {
        **job,
        "command": [sys.executable, "-c", "raise AssertionError('must not run')"],
    }
    # Existing JSONs from older scripts need neither launch metadata nor checksums.
    (result.parent / "launch.json").unlink()
    assert launch.execute_commands([different], tmp_path) == 0
    assert result.stat().st_mtime_ns == before
    assert not (result.parent / "completed.sha256").exists()

    # An interrupted evaluation with logs but no result is run from the beginning.
    result.unlink()
    assert launch.execute_commands([job], tmp_path) == 0
    assert result.read_text() == "{}"
    assert json.loads((result.parent / "launch.json").read_text()) == command


@pytest.mark.parametrize("contents", [b'{"partial":', b"\xff"])
def test_execution_recovers_invalid_results_through_evaluator_resume(
    tmp_path, contents
):
    result = tmp_path / "job" / "population.json"
    result.parent.mkdir()
    result.write_bytes(contents)
    command = [
        sys.executable,
        "-c",
        f"""
from omegaconf import OmegaConf
from imindbench.utils.logging_utils import save_results, should_skip_existing_output
cfg = OmegaConf.create({{"runtime": {{"overwrite": False}}}})
path = {str(result)!r}
if not should_skip_existing_output(cfg, path):
    save_results({{"recovered": True}}, path)
""",
    ]
    job = {"command": command, "run_dir": str(result.parent), "result": str(result)}
    assert launch.execute_commands([job], tmp_path) == 0
    assert json.loads(result.read_text()) == {"recovered": True}
    # A zero-exit subprocess that leaves malformed JSON must still count as failed.
    result.write_bytes(contents)
    job["command"] = [sys.executable, "-c", "pass"]
    assert launch.execute_commands([job], tmp_path) == 1


@pytest.mark.parametrize("mode", [None, "--dry-run", "--count"])
def test_cli_executes_by_default_with_explicit_preview_modes(
    tmp_path, monkeypatch, mode
):
    result = tmp_path / "job" / "population.json"
    job = {
        "command": [
            sys.executable,
            "-c",
            f"from pathlib import Path; Path({str(result)!r}).write_text('{{}}')",
        ],
        "run_dir": str(result.parent),
        "result": str(result),
    }
    argv = [
        "imindbench-grid",
        "--dataset",
        "neuroprobev2",
        "--model",
        "logistic",
        "--preprocessor",
        "multi_stft_2048Hz",
        "--output-root",
        str(tmp_path),
        "--task",
        "onset",
        "--target",
        "sub1_sess1",
    ]
    if mode:
        argv.append(mode)
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(launch, "build_commands", lambda args: [job])
    assert launch.main() == 0
    assert result.exists() == (mode is None)
    if mode is None:
        before = result.stat().st_mtime_ns
        assert launch.main() == 0
        assert result.stat().st_mtime_ns == before
    else:
        assert not result.parent.exists()


def _cells_file(tmp_path, tasks):
    path = tmp_path / "cells.json"
    path.write_text(
        json.dumps(
            {
                "tasks": {
                    task: {"subject_sessions": targets}
                    for task, targets in tasks.items()
                }
            }
        )
    )
    return path


def _task_targets(jobs):
    return {
        (_overrides(job["command"])["dataset.task"], Path(job["run_dir"]).name)
        for job in jobs
    }


def test_cells_keeps_only_the_listed_pairs(tmp_path):
    cells = _cells_file(
        tmp_path,
        {"onset": ["sub1_sess1", "sub2_sess4"], "speech": ["sub3_sess0"]},
    )
    args = _args(
        "--task",
        "onset",
        "speech",
        "--target",
        "sub1_sess1",
        "sub2_sess4",
        "sub3_sess0",
        "--cells",
        str(cells),
    )
    args.output_root = tmp_path / "runs"
    assert _task_targets(launch.build_commands(args)) == {
        ("onset", "sub1_sess1"),
        ("onset", "sub2_sess4"),
        ("speech", "sub3_sess0"),
    }


def test_cells_by_name_reads_the_packaged_directory(tmp_path, monkeypatch):
    (tmp_path / "neuroprobev2").mkdir()
    _cells_file(tmp_path / "neuroprobev2", {"onset": ["sub1_sess1"]}).rename(
        tmp_path / "neuroprobev2" / "my_cells.json"
    )
    monkeypatch.setattr(launch, "CELL_MANIFEST_DIR", tmp_path)
    args = _args("--target", "sub1_sess1", "sub2_sess4", "--cells", "my_cells")
    assert _task_targets(launch.build_commands(args)) == {("onset", "sub1_sess1")}


def test_cells_and_decodable_population_must_both_list_a_pair(tmp_path):
    population = json.loads(
        (
            launch.POPULATION_DIR / "stft_or_htnet_500hz_val_mean0p60/neuroprobev2.json"
        ).read_text()
    )["tasks"]
    task = next(t for t, entry in population.items() if entry["subject_sessions"])
    listed = population[task]["subject_sessions"][0]
    cells = _cells_file(tmp_path, {task: [listed, "sub99_sess9"]})
    args = _args(
        "--task",
        task,
        "--target",
        listed,
        "sub99_sess9",
        "--decodable-rule",
        "stft_or_htnet_500hz_val_mean0p60",
        "--cells",
        str(cells),
    )
    assert _task_targets(launch.build_commands(args)) == {(task, listed)}


@pytest.mark.parametrize(
    ("cells", "message"),
    [
        (
            {"other_task": ["sub1_sess1"]},
            "Cell list must list subject_sessions for onset",
        ),
        ({"onset": []}, "No evaluations remain"),
    ],
)
def test_cells_errors(tmp_path, cells, message):
    path = _cells_file(tmp_path, cells)
    with pytest.raises(ValueError, match=message):
        launch.build_commands(_args("--cells", str(path)))


def test_missing_cell_list_is_reported(tmp_path):
    with pytest.raises(ValueError, match="Cell list not found"):
        launch.build_commands(_args("--cells", str(tmp_path / "absent.json")))
    with pytest.raises(ValueError, match="cells must be a simple name"):
        launch.build_commands(_args("--cells", "bad name"))
