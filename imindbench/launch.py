"""Preview or execute an evaluation grid using ordinary Hydra configurations."""

import argparse
import fcntl
import itertools
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.errors import HydraException

from imindbench.utils.result_io import is_valid_result_file

CONF_DIR = Path(__file__).resolve().parent / "conf"
POPULATION_DIR = Path(__file__).resolve().parent / "decodable_subject_sessions"
CELL_MANIFEST_DIR = Path(__file__).resolve().parent / "cell_manifests"
IDENTITY_KEYS = {
    "paths",
    "dataset",
    "model",
    "experiment",
    "model.name",
    "model.backend",
    "preprocessor",
    "dataset.provider",
    "dataset.regime",
    "dataset.task",
    "dataset.test_subject",
    "dataset.test_session",
    "dataset.subset_tier",
    "model.device",
    "hydra.run.dir",
    "runtime.overwrite",
    "paths.decodable_subject_sessions_dir",
}


def _validate_name(value, name):
    if not isinstance(value, str) or not re.fullmatch(r"[\w-]+", value):
        raise ValueError(f"{name} must be a simple name")


def _validate_names(value, name, allow_empty=False):
    if (
        not isinstance(value, list)
        or (not value and not allow_empty)
        or any(not isinstance(v, str) or not re.fullmatch(r"[\w-]+", v) for v in value)
        or len(value) != len(set(value))
    ):
        raise ValueError(f"{name} must contain unique names")


def _value(value):
    if isinstance(value, str) and re.fullmatch(r"[\w./:+-]+", value):
        return value
    return json.dumps(value, allow_nan=False)


def _key_value(override):
    key, separator, value = override.partition("=")
    if not separator or not re.fullmatch(r"\+{0,2}[\w.]+", key):
        raise ValueError("Overrides must be Hydra key=value assignments")
    if key.lstrip("+") in IDENTITY_KEYS or key.lstrip("+").startswith("hydra."):
        raise ValueError(f"Use selection flags or experiment configs to change {key}")
    return key, value


def _load_selection_manifest(path, tasks, label):
    """Read a {"tasks": {task: {"subject_sessions": [...]}}} selection file.

    Returns {task: set of "sub<S>_sess<T>"} for the requested tasks. Every
    requested task must be listed; a task with an empty list contributes no
    evaluations.
    """
    selection = json.loads(Path(path).read_text())["tasks"]
    if not isinstance(selection, dict):
        raise ValueError(f"{label} tasks must be a mapping")
    for task in tasks:
        if (
            not isinstance(selection.get(task), dict)
            or "subject_sessions" not in selection[task]
        ):
            raise ValueError(f"{label} must list subject_sessions for {task}")
        _validate_names(
            selection[task]["subject_sessions"],
            f"{label} targets",
            allow_empty=True,
        )
    return {task: set(selection[task]["subject_sessions"]) for task in tasks}


def _resolve_cells_path(cells, provider):
    """--cells takes a packaged list name or a path to a JSON file."""
    if cells.endswith(".json") or "/" in cells:
        return Path(cells).expanduser().resolve()
    _validate_name(cells, "cells")
    return CELL_MANIFEST_DIR / provider / f"{cells}.json"


def build_commands(args):
    """Expand one model's task/target/sweep grid without creating any outputs."""
    for name in (
        "dataset",
        "model",
        "preprocessor",
        "paths",
        "output_group",
    ):
        _validate_name(getattr(args, name), name)
    if args.experiment is not None:
        if not re.fullmatch(r"[\w-]+(?:/[\w-]+)*", args.experiment):
            raise ValueError("experiment must name a Hydra config group entry")
    if not re.fullmatch(r"cpu|cuda(?::[0-9]+)?", args.device):
        raise ValueError("device must be cpu, cuda or cuda:<index>")
    if args.limit is not None and args.limit < 1:
        raise ValueError("limit must be positive")
    config_args = []
    searchpath = []
    if args.config_dir is not None:
        config_dir = args.config_dir.expanduser().resolve()
        if not config_dir.is_dir():
            raise ValueError("config-dir must be an existing directory")
        config_args = ["--config-dir", str(config_dir)]
        searchpath = [f"hydra.searchpath={json.dumps(['file://' + str(config_dir)])}"]
    base = {
        "paths": args.paths,
        "dataset": args.dataset,
        "model": args.model,
        "preprocessor": args.preprocessor,
        "dataset.regime": args.regime,
        "wandb.enabled": False,
        "runtime.overwrite": False,
        "runtime.verbose": True,
    }
    if args.experiment is not None:
        base["experiment"] = args.experiment
    if args.subset is not None:
        _validate_name(args.subset, "subset")
        base["dataset.subset_tier"] = args.subset
    if args.regime not in {"hold-in-session", "hold-out-session"}:
        base["dataset.train_same_subject_only"] = False
    decodable_dir = args.decodable_dir
    if decodable_dir is None and args.decodable_rule is not None:
        _validate_name(args.decodable_rule, "decodable-rule")
        decodable_dir = POPULATION_DIR / args.decodable_rule
    if decodable_dir is not None:
        decodable_dir = decodable_dir.expanduser().resolve()
        base["paths.decodable_subject_sessions_dir"] = str(decodable_dir)
    # CLI tuning comes after the experiment config; repeated --set uses the last value.
    extra = {}
    for override in args.overrides:
        key, value = _key_value(override)
        extra[key.lstrip("+")] = (key, value)
    sweeps = {}
    for dimension in args.sweep:
        key, value = _key_value(dimension)
        values = value.split(",")
        if (
            key.lstrip("+") in extra
            or key in sweeps
            or not re.fullmatch(r"[\w.]+", key)
            or not all(re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", v) for v in values)
            or len(values) != len(set(values))
        ):
            raise ValueError(
                "Sweep keys must be unique and separate from --set; values must be unique numeric scalars"
            )
        sweeps[key] = values
    tokens = [
        f"{key}={_value(value)}" for key, value in base.items() if key not in extra
    ]
    tokens += [f"{key}={value}" for key, value in extra.values()]
    # Ask Hydra for provider identity instead of inferring it from a config filename.
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base="1.1"):
        cfg = compose(config_name="config", overrides=[*tokens, *searchpath])
    backend = cfg.model.get("backend")
    if not isinstance(backend, str) or backend not in {"sklearn", "torch"}:
        raise ValueError("model.backend must be sklearn or torch")
    if backend == "torch":
        tokens.append(f"model.device={args.device}")
    # Transfer presets share their training manifest with target selection.
    # Explicit --decodable-dir/--decodable-rule still take precedence above.
    if decodable_dir is None and cfg.dataset.get(
        "decodable_subject_sessions_only", False
    ):
        decodable_dir = Path(cfg.paths.decodable_subject_sessions_dir)
    provider = cfg.dataset.provider
    _validate_name(provider, "dataset.provider")
    # Scripts own selection; the launcher validates syntax without a second catalog.
    _validate_names(args.task, "task")
    _validate_names(args.target, "target")
    tasks = args.task
    targets = {}
    for target in args.target:
        match = re.fullmatch(r"sub(0|[1-9][0-9]*)_sess(0|[1-9][0-9]*)", target)
        if match is None:
            raise ValueError(
                f"Invalid target {target!r}; expected sub<S>_sess<T> (e.g. sub1_sess1)"
            )
        targets[target] = tuple(int(value) for value in match.groups())
    # Each selection file keeps only the task/target pairs it lists. With both
    # a decodable population and a cell list, a pair must be in both.
    selections = []
    if decodable_dir is not None:
        selections.append(
            _load_selection_manifest(
                decodable_dir / f"{provider}.json", tasks, "Decodable population"
            )
        )
    if args.cells is not None:
        cells_path = _resolve_cells_path(args.cells, provider)
        if not cells_path.is_file():
            raise ValueError(f"Cell list not found: {cells_path}")
        selections.append(_load_selection_manifest(cells_path, tasks, "Cell list"))
    commands = []
    output = args.output_root.expanduser().resolve()
    for values, task, target in itertools.product(
        itertools.product(*sweeps.values()), tasks, targets
    ):
        if any(target not in selection[task] for selection in selections):
            continue
        subject, session = targets[target]
        run_dir = output / args.output_group / f"{args.model}_{args.preprocessor}"
        if args.subset is not None:
            run_dir /= args.subset
        if sweeps:
            run_dir /= "_".join(
                key.replace(".", "-") + "=" + value
                for key, value in zip(sweeps, values, strict=True)
            )
        run_dir = run_dir / args.regime / task / target
        job_tokens = [
            *tokens,
            *(f"{key}={value}" for key, value in zip(sweeps, values, strict=True)),
            f"dataset.task={task}",
            f"dataset.test_subject={subject}",
            f"dataset.test_session={session}",
            f"hydra.run.dir={_value(str(run_dir))}",
        ]
        commands.append(
            {
                "command": [
                    sys.executable,
                    "-m",
                    "imindbench.run_eval",
                    *config_args,
                    *job_tokens,
                ],
                "run_dir": str(run_dir),
                "result": str(
                    run_dir / f"population_btbank{subject}_{session}_{task}.json"
                ),
            }
        )
    if not commands:
        raise ValueError("No evaluations remain after selections/population filtering")
    return commands[: args.limit] if args.limit is not None else commands


def execute_commands(commands, output_root):
    """Run serially, skipping readable results and retrying malformed JSONs."""
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / ".grid.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("Another grid is active in this output root") from exc
        failures = 0
        for job in commands:
            directory = Path(job["run_dir"])
            if is_valid_result_file(job["result"]):
                print(f"Skipping existing result: {job['result']}")
                continue
            if Path(job["result"]).exists():
                print(f"Invalid result JSON; rerunning {job['result']}")
            directory.mkdir(parents=True, exist_ok=True)
            record = directory / "launch.json"
            record.write_text(json.dumps(job["command"], indent=2) + "\n")
            print(shlex.join(job["command"]), flush=True)
            with (directory / "launcher.log").open("a") as log:
                result = subprocess.run(
                    job["command"], stdout=log, stderr=subprocess.STDOUT
                )
            if result.returncode or not is_valid_result_file(job["result"]):
                failures += 1
                print(
                    f"Failed evaluation; see {directory / 'launcher.log'}",
                    file=sys.stderr,
                )
        return failures


def parser():
    """Return the shared grid execution interface used by the shell scripts."""
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset", required=True, help="Hydra dataset config")
    result.add_argument("--model", required=True, help="Hydra model config")
    result.add_argument(
        "--preprocessor", required=True, help="Hydra preprocessor config"
    )
    result.add_argument(
        "--experiment",
        default="default",
        help="Hydra experiment config (default: default)",
    )
    result.add_argument("--output-root", type=Path, required=True)
    result.add_argument("--output-group", default="evaluations")
    result.add_argument("--paths", default="example")
    result.add_argument("--config-dir", type=Path)
    result.add_argument("--device", default="cuda:0")
    result.add_argument(
        "--regime",
        default="within-session",
        choices=[
            "within-session",
            "hold-in-session",
            "hold-out-session",
            "hold-out-subject",
        ],
    )
    result.add_argument("--subset", help="Dataset subset tier")
    result.add_argument(
        "--task",
        nargs="+",
        required=True,
        help="Select tasks; the last --task replaces earlier selections",
    )
    result.add_argument(
        "--target",
        nargs="+",
        required=True,
        help="Select sub<S>_sess<T>; the last --target replaces earlier selections",
    )
    result.add_argument(
        "--decodable-dir",
        type=Path,
        help="Custom population directory; takes precedence over --decodable-rule",
    )
    result.add_argument(
        "--decodable-rule", help="Packaged decodable population directory"
    )
    result.add_argument(
        "--cells",
        metavar="NAME|PATH",
        help=(
            "Run only the task/target pairs listed in this cell list: a name under "
            "imindbench/cell_manifests/<dataset.provider>/, or a path to a JSON "
            'file shaped like {"tasks": {task: {"subject_sessions": [...]}}}'
        ),
    )
    result.add_argument("--sweep", action="append", default=[], metavar="KEY=V1,V2")
    result.add_argument(
        "--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE"
    )
    result.add_argument("--limit", type=int)
    preview = result.add_mutually_exclusive_group()
    preview.add_argument("--count", action="store_true")
    preview.add_argument(
        "--dry-run", action="store_true", help="Print commands without executing"
    )
    return result


def main():
    args = parser().parse_args()
    try:
        commands = build_commands(args)
        if args.count:
            print(len(commands))
        elif args.dry_run:
            for job in commands:
                print(shlex.join(job["command"]))
            print(
                f"{len(commands)} evaluations; dry run (nothing executed)",
                file=sys.stderr,
            )
        else:
            failures = execute_commands(
                commands, args.output_root.expanduser().resolve()
            )
            print(f"{len(commands)} planned evaluations; {failures} failed")
            return int(failures > 0)
    except (ValueError, KeyError, OSError, HydraException) as exc:
        print(f"Grid error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
