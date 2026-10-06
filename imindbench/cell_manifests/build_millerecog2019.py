"""Build the Miller ECoG cell lists from the prepared H5 files.

Usage:
    python -m imindbench.cell_manifests.build_millerecog2019 \
        --data-dir <dataset_root>/miller_ecog_library_2019 [--out-dir DIR] \
        [--from-brainsets]

The subject and task-set tables come from the MillerECoG2019 dataset module.
By default that is the class torch_brain.datasets provides. Add
--from-brainsets to use brainsets.datasets.MillerECoG2019 instead, for
example from the brainsets fork branch miller-ecog-modularize, when the
installed torch_brain does not have the class yet.

What it does, step by step:
1. Opens every <recording id>.h5 file in --data-dir. Each file is one subject
   doing one task set.
2. Turns the recording id (for example sub-bp_set-motor_basic) into the
   numbers the launcher uses (sub3_sess1), with the MillerECoG2019 dataset
   module's subject and task-set tables.
3. Reads the task lists the brainsets pipeline stored in the file:
   tasks_json (classification tasks that passed the pipeline's minimum
   window and trial counts), control_tasks_json (tasks whose labels follow
   time, which are controls rather than decoding results), and
   regression_tasks_json (continuous targets).
4. Sorts every (task, subject, session) cell into the lists below and writes
   each list as {"tasks": {task: {"subject_sessions": [...]}}}, the layout
   the launcher's --cells option reads.

Lists:
  binary.json, multiclass.json   classification cells of task sets 1-10
                                 (the main Miller table)
  new_sets_binary.json,          classification cells of task sets 11-12,
  new_sets_multiclass.json       without the control tasks
  controls_binary.json,          the control tasks
  controls_multiclass.json
  class_pair_<a>v<b>.json        multiclass cells of task sets 1-12 (no
                                 controls) that have classes a and b; run
                                 with dataset.class_pair=[a,b]
  regression.json                regression cells of task sets 13-21
  regression_sliding.json        regression cells of task sets 22-24
  regression_bci4.json           regression cells of task set 25
  positions.json                 cells of binary.json and multiclass.json
                                 whose recording keeps at least
                                 --min-position-channels channels with a
                                 good (quality A or B) MNI152 position;
                                 written only when the files carry positions

The script prints the number of cells in each list.
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

MAIN_SETS = range(1, 11)
NEW_SETS = range(11, 13)
REGRESSION_SETS = {
    "regression": range(13, 22),
    "regression_sliding": range(22, 25),
    "regression_bci4": range(25, 26),
}
GOOD_POSITION_QUALITY = {"A", "B"}


def _read_json(handle, name):
    raw = handle.attrs.get(name, None)
    if raw is None and name in handle:
        raw = handle[name][()]
    if raw is None:
        return None
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return json.loads(str(raw))


def _good_position_channels(handle) -> int | None:
    """Included channels with a quality A or B MNI152 position, or None."""
    channels = handle.get("channels")
    if channels is None or "coord_mni152_quality" not in channels:
        return None
    quality = np.asarray(channels["coord_mni152_quality"][()]).astype(str)
    included = np.asarray(channels["included"][()]).astype(bool)
    return int(np.sum(included & np.isin(quality, sorted(GOOD_POSITION_QUALITY))))


def miller_dataset_module(*, from_brainsets: bool = False):
    """The module that defines MillerECoG2019 and its id helpers."""
    import importlib
    import sys

    if from_brainsets:
        try:
            return importlib.import_module("brainsets.datasets.MillerECoG2019")
        except ImportError as exc:
            raise ImportError(
                "--from-brainsets needs brainsets with "
                "brainsets.datasets.MillerECoG2019 (for example the brainsets "
                "fork branch miller-ecog-modularize)."
            ) from exc
    from imindbench.utils.pipeline_contracts import get_dataset_class

    return sys.modules[get_dataset_class("millerecog2019").__module__]


def subject_session_reader(module):
    """Turn a recording id like sub-bp_set-motor_basic into (3, 1).

    Only the module's subject and task-set tables are used; the id itself is
    split by iMINDBench (see split_miller_recording_id).
    """

    from imindbench.utils.data_adapter import split_miller_recording_id

    def subject_session_for(recording_id: str) -> tuple[int, int]:
        code, task_set = split_miller_recording_id(recording_id)
        return module.subject_number_for(code), module.session_number_for(task_set)

    return subject_session_for


def read_recordings(data_dir: Path, *, subject_session_for) -> list[dict]:
    """One summary per H5 file: numbers, task lists and position count."""
    import h5py

    recordings = []
    for path in sorted(data_dir.glob("*.h5")):
        subject, session = subject_session_for(path.stem)
        with h5py.File(path, "r") as handle:
            recordings.append(
                {
                    "recording_id": path.stem,
                    "target": f"sub{subject}_sess{session}",
                    "session": session,
                    "tasks": _read_json(handle, "tasks_json") or [],
                    "controls": set(_read_json(handle, "control_tasks_json") or []),
                    "regression": _read_json(handle, "regression_tasks_json") or [],
                    "good_position_channels": _good_position_channels(handle),
                }
            )
    if not recordings:
        raise ValueError(f"No .h5 files in {data_dir}.")
    return recordings


def build_lists(recordings: list[dict], *, min_position_channels: int = 1) -> dict:
    """{list name: {task: set of targets}} for every list in the module docstring."""
    lists: dict[str, dict[str, set]] = defaultdict(lambda: defaultdict(set))
    has_positions = any(r["good_position_channels"] is not None for r in recordings)
    for recording in recordings:
        target, session = recording["target"], recording["session"]
        for entry in recording["tasks"]:
            task, label_mode = entry["task"], entry["label_mode"]
            if task in recording["controls"]:
                lists[f"controls_{label_mode}"][task].add(target)
                continue
            if session in MAIN_SETS:
                lists[label_mode][task].add(target)
                positions = recording["good_position_channels"]
                if positions is not None and positions >= min_position_channels:
                    lists["positions"][task].add(target)
            elif session in NEW_SETS:
                lists[f"new_sets_{label_mode}"][task].add(target)
            else:
                continue
            if label_mode == "multiclass":
                for a, b in itertools.combinations(range(int(entry["n_classes"])), 2):
                    lists[f"class_pair_{a}v{b}"][task].add(target)
        for entry in recording["regression"]:
            for name, sets in REGRESSION_SETS.items():
                if session in sets:
                    lists[name][entry["target"]].add(target)
    if not has_positions:
        lists.pop("positions", None)
    return lists


def _target_key(target: str) -> tuple[int, int]:
    subject, session = target[3:].split("_sess")
    return int(subject), int(session)


def to_manifest(tasks: dict[str, set], *, name: str) -> dict:
    ordered = {
        task: sorted(targets, key=_target_key)
        for task, targets in sorted(tasks.items())
    }
    return {
        "dataset": "millerecog2019",
        "list": name,
        "n_cells": sum(len(targets) for targets in ordered.values()),
        "tasks": {
            task: {"subject_sessions": targets, "n_subject_sessions": len(targets)}
            for task, targets in ordered.items()
        },
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "millerecog2019",
    )
    parser.add_argument("--min-position-channels", type=int, default=1)
    parser.add_argument(
        "--from-brainsets",
        action="store_true",
        help=(
            "Read the subject and task-set tables from "
            "brainsets.datasets.MillerECoG2019 instead of torch_brain.datasets"
        ),
    )
    args = parser.parse_args(argv)

    module = miller_dataset_module(from_brainsets=args.from_brainsets)
    recordings = read_recordings(
        args.data_dir, subject_session_for=subject_session_reader(module)
    )
    lists = build_lists(recordings, min_position_channels=args.min_position_channels)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name in sorted(lists):
        manifest = to_manifest(lists[name], name=name)
        (args.out_dir / f"{name}.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        n_recordings = len(
            {t for v in manifest["tasks"].values() for t in v["subject_sessions"]}
        )
        print(
            f"{name}: {manifest['n_cells']} cells, {len(manifest['tasks'])} tasks, "
            f"{n_recordings} recordings"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
