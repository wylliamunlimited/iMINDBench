"""The packaged Miller cell lists (imindbench/cell_manifests/millerecog2019/)."""

import json
import re
from pathlib import Path

import pytest

from imindbench.cell_manifests import build_millerecog2019 as builder

LISTS = Path(__file__).resolve().parents[2] / "imindbench/cell_manifests/millerecog2019"
TARGET = re.compile(r"^sub(?P<subject>[1-9][0-9]*)_sess(?P<session>[1-9][0-9]*)$")

# list name -> task sets its cells may come from
EXPECTED = {
    "binary": set(range(1, 9)),
    "multiclass": set(range(1, 9)),
    "positions_binary": set(range(1, 9)),
    "positions_multiclass": set(range(1, 9)),
    "new_sets_binary": {10, 11},
    "controls_binary": {11},
    "controls_multiclass": {11},
    "regression": {13, 16, 19},
    "regression_w500": {14, 17, 20},
    "regression_w250": {15, 18, 21},
    "regression_sliding": {22, 23, 24},
    "regression_bci4": {25},
}


def _load(name):
    return json.loads((LISTS / f"{name}.json").read_text())


def _cells(name):
    return {
        (task, target)
        for task, entry in _load(name)["tasks"].items()
        for target in entry["subject_sessions"]
    }


def _class_pair_lists():
    return sorted(path.stem for path in LISTS.glob("class_pair_*.json"))


def test_every_packaged_list_is_expected():
    names = {path.stem for path in LISTS.glob("*.json")}
    assert names == set(EXPECTED) | set(_class_pair_lists())


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_list_targets_and_task_sets(name):
    for entry in _load(name)["tasks"].values():
        targets = entry["subject_sessions"]
        assert len(targets) == len(set(targets))
        for target in targets:
            match = TARGET.fullmatch(target)
            assert match, target
            assert 1 <= int(match["subject"]) <= 29
            assert int(match["session"]) in EXPECTED[name], (name, target)


@pytest.mark.parametrize("name", sorted(EXPECTED) + _class_pair_lists())
def test_list_is_written_as_the_builder_writes_it(name):
    # Only {"tasks": {task: {"subject_sessions": [...]}}}, sorted, one task per
    # line, so rebuilding the lists gives a readable diff.
    text = (LISTS / f"{name}.json").read_text()
    tasks = {
        task: set(entry["subject_sessions"])
        for task, entry in json.loads(text)["tasks"].items()
    }
    assert text == builder.dump_manifest(builder.to_manifest(tasks))


def test_main_table_has_178_cells():
    assert len(_cells("binary")) == 153
    assert len(_cells("multiclass")) == 25


def test_positions_lists_are_subsets_of_the_main_table():
    assert _cells("positions_binary") <= _cells("binary")
    assert _cells("positions_multiclass") <= _cells("multiclass")


def test_derived_lists_name_every_task_of_their_source_list():
    # The run script passes one task array to each list derived from binary.json
    # or multiclass.json, and the launcher needs every task named in the list.
    multiclass_tasks = set(_load("multiclass")["tasks"])
    assert set(_load("positions_binary")["tasks"]) == set(_load("binary")["tasks"])
    assert set(_load("positions_multiclass")["tasks"]) == multiclass_tasks
    for name in _class_pair_lists():
        assert set(_load(name)["tasks"]) == multiclass_tasks, name


def test_controls_are_not_in_the_other_lists():
    controls = {
        task for task, _ in _cells("controls_binary") | _cells("controls_multiclass")
    }
    for name in ("binary", "multiclass", "new_sets_binary"):
        assert not controls & {task for task, _ in _cells(name)}


def test_class_pair_lists_are_subsets_of_multiclass():
    names = _class_pair_lists()
    assert names
    for name in names:
        a, b = (int(v) for v in re.fullmatch(r"class_pair_(\d+)v(\d+)", name).groups())
        assert a < b
        assert _cells(name) <= _cells("multiclass")
    # Every multiclass cell has at least 3 classes, so the 0v1 pair covers all of them.
    assert _cells("class_pair_0v1") == _cells("multiclass")


def test_regression_window_lengths_share_targets():
    tasks = {
        name: {t for t, _ in _cells(name)}
        for name in EXPECTED
        if name.startswith("regression")
    }
    assert tasks["regression"] == tasks["regression_w500"] == tasks["regression_w250"]
    assert tasks["regression_bci4"] <= tasks["regression"]
