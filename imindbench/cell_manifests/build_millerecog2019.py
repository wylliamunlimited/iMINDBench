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
  binary.json, multiclass.json   classification cells of task sets 1-8: the
                                 178-cell Miller table (153 binary + 25
                                 multiclass)
  positions_binary.json,         cells of binary.json / multiclass.json whose
  positions_multiclass.json      recording keeps at least
                                 --min-position-channels channels with a good
                                 (quality A or B) MNI152 position; written
                                 only when the files carry positions
  class_pair_<a>v<b>.json        cells of multiclass.json with more than b
                                 classes; run with dataset.class_pair=[a,b]
  The positions and class-pair lists name every task of their source list;
  a task with no qualifying cell has an empty list.
  new_sets_binary.json,          classification cells of task sets 10-11
  new_sets_multiclass.json       (1.0 s windows), without the control tasks
  faces08_binary.json            classification cells of task sets 9 and 12
                                 (0.8 s windows)
  controls_binary.json,          the control tasks
  controls_multiclass.json
  regression.json                regression targets of task sets 13, 16, 19
                                 (1.0 s windows)
  regression_w500.json           task sets 14, 17, 20 (0.5 s windows)
  regression_w250.json           task sets 15, 18, 21 (0.25 s windows)
  regression_sliding.json        task sets 22-24 (sliding 1.0 s windows)
  regression_bci4.json           task set 25

The script prints the number of cells in each list.
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

# The 178-cell Miller table (153 binary + 25 multiclass) covers task sets 1-8.
MAIN_SETS = range(1, 9)
# Newer classification sets with 1.0 s windows: 10 faces_noise, 11 memory_nback.
NEW_SETS = {10, 11}
# Face sets with 0.8 s windows: 9 faces_basic, 12 faces_localizer. They need
# 0.8 s preprocessing presets, which are not bundled, so they get their own list.
FACES_08_SETS = {9, 12}
REGRESSION_SETS = {
    "regression": {13, 16, 19},
    "regression_w500": {14, 17, 20},
    "regression_w250": {15, 18, 21},
    "regression_sliding": {22, 23, 24},
    "regression_bci4": {25},
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
                    lists[f"positions_{label_mode}"][task].add(target)
                if label_mode == "multiclass":
                    # A cell joins class_pair_<a>v<b> when it has more than b classes.
                    for a, b in itertools.combinations(
                        range(int(entry["n_classes"])), 2
                    ):
                        lists[f"class_pair_{a}v{b}"][task].add(target)
            elif session in NEW_SETS:
                lists[f"new_sets_{label_mode}"][task].add(target)
            elif session in FACES_08_SETS:
                lists[f"faces08_{label_mode}"][task].add(target)
        for entry in recording["regression"]:
            for name, sets in REGRESSION_SETS.items():
                if session in sets:
                    lists[name][entry["target"]].add(target)
    # The run script passes the same task names to every list derived from
    # binary.json or multiclass.json, and the launcher needs each task named in
    # the list. So a derived list names every task of its source list, with an
    # empty target list where no cell qualifies (for example direction_4way in
    # class_pair_0v4, since that task has only classes 0-3).
    for name in list(lists):
        if name.startswith("positions_"):
            source = name.removeprefix("positions_")
        elif name.startswith("class_pair_"):
            source = "multiclass"
        else:
            continue
        for task in lists[source]:
            lists[name].setdefault(task, set())
    if not has_positions:
        for name in [name for name in lists if name.startswith("positions_")]:
            lists.pop(name)
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
