"""Run editable shell examples from isolated copies of their settings."""

import re
import shlex
import shutil
from pathlib import Path

import pytest
from omegaconf import OmegaConf


@pytest.fixture
def dataset_selections():
    """Read the task/target lists owned by each editable dataset script."""

    def read_selections(dataset):
        script = Path(__file__).resolve().parents[1] / "scripts" / f"run_{dataset}.sh"
        source = script.read_text()
        return tuple(
            shlex.split(re.search(rf"^{key}=\(([\s\S]*?)^\)", source, re.M)[1])
            for key in ("TASKS", "TARGETS")
        )

    return read_selections


@pytest.fixture
def dataset_script(tmp_path):
    root = Path(__file__).resolve().parents[1]
    config = tmp_path / "local config"
    (config / "paths").mkdir(parents=True)
    shutil.copyfile(
        root / "imindbench/conf/paths/example.yaml", config / "paths/local.yaml"
    )

    def make_script(dataset="neuroprobev2", family=None, yaml_updates=None, **settings):
        # Extend the selected preset in YAML without shadowing packaged config names.
        if yaml_updates:
            preset = settings["EXPERIMENT"]
            trial = f"test_{dataset}_{settings['MODEL']}"
            cfg = OmegaConf.create({"defaults": [f"/experiment/{preset}", "_self_"]})
            for relative_path, updates in yaml_updates.items():
                group = Path(relative_path).parts[0]
                cfg = OmegaConf.merge(cfg, {group: updates})
            target = config / "experiment" / f"{trial}.yaml"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("# @package _global_\n" + OmegaConf.to_yaml(cfg))
            settings["EXPERIMENT"] = trial
        source = (root / "scripts" / f"run_{dataset}.sh").read_text()
        if family not in {None, "within_session"}:
            # Enable the real commented command, omitting the default block.
            prefix = source[: source.index("# ── Within-session")]
            title = {
                "within_dataset": "Within-dataset",
                "multi_dataset": "Multi-dataset",
                "sample_efficiency": "Sample efficiency",
                "new_task_sets": "New task sets",
                "controls": "Controls",
                "class_pairs": "Class pairs",
                "regression": "Regression",
                "regression_sliding": "Regression sliding",
                "regression_bci4": "Regression BCI-IV",
                "positions": "Positions",
            }[family]
            block = source.split(f"# ── {title} ─", 1)[1].split("# ──", 1)[0]
            # The block's code starts at its first command, loop or array.
            start = re.search(r"^# (?:python -m|for |[A-Z_]+=\()", block, re.M)
            command = block[start.start() :]
            source = (
                prefix
                + "\n".join(line.removeprefix("# ") for line in command.splitlines())
                + "\n"
            )
        # Preview the actual script commands on CPU without running experiments.
        source = source.replace(
            "python -m imindbench.launch",
            "python -m imindbench.launch --dry-run --device cpu",
        )
        settings = {
            "CONFIG_DIR": str(config),
            "OUTPUT_ROOT": str(tmp_path / "runs"),
            **settings,
        }
        for key, value in settings.items():
            if isinstance(value, list):
                value = "(" + " ".join(shlex.quote(item) for item in value) + ")"
            else:
                value = shlex.quote(str(value))
            source, count = re.subn(
                rf"^{key}=\([\s\S]*?^\)"
                if isinstance(settings[key], list)
                else rf"^{key}=.*$",
                lambda _, assignment=f"{key}={value}": assignment,
                source,
                count=1,
                flags=re.M,
            )
            assert count == 1, key
        script = tmp_path / "edited experiments.sh"
        script.write_text(source)
        return str(script)

    return make_script


@pytest.fixture
def dataset_pairings():
    """Exercise the combinations advertised in each script's comment table."""

    def read_pairings(dataset):
        source = (
            Path(__file__).resolve().parents[1] / "scripts" / f"run_{dataset}.sh"
        ).read_text()
        table = source.split("# MODEL", 1)[1].split("MODEL=", 1)[0]
        rows = []
        for line in table.splitlines()[1:]:
            models, preprocessor, experiment = re.split(
                r"\s{2,}", line.removeprefix("# ").strip()
            )
            for model in models.split()[0].split("/"):
                settings = dict(
                    MODEL=model,
                    PREPROCESSOR=preprocessor,
                    EXPERIMENT=experiment.replace("<model>", model),
                )
                yaml_updates = {}
                if model == "linear_baseline":
                    yaml_updates["paths/local.yaml"] = {
                        "brainbert_checkpoint": "/path/to/stft_large_pretrained.pth"
                    }
                elif model == "diver":
                    yaml_updates["paths/local.yaml"] = {
                        "diver_checkpoint": "/path/to/ieeg_checkpoint.pt",
                        "diver_shape_cache_dir": "/path/to/diver_shapes",
                    }
                settings["yaml_updates"] = yaml_updates
                rows.append(settings)
        assert len(rows) == 11
        return rows

    return read_pairings
