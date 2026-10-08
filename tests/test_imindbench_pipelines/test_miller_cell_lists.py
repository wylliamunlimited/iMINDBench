"""The packaged Miller cell lists (imindbench/cell_manifests/millerecog2019/)."""

import json
import re
from pathlib import Path

import pytest

LISTS = Path(__file__).resolve().parents[2] / "imindbench/cell_manifests/millerecog2019"
TARGET = re.compile(r"^sub(?P<subject>[1-9][0-9]*)_sess(?P<session>[1-9][0-9]*)$")

# list name -> (label mode, task sets its cells may come from)
EXPECTED = {
    "binary": ("binary", set(range(1, 9))),
    "multiclass": ("multiclass", set(range(1, 9))),
    "positions_binary": ("binary", set(range(1, 9))),
    "positions_multiclass": ("multiclass", set(range(1, 9))),
    "new_sets_binary": ("binary", {10, 11}),
    "controls_binary": ("binary", {11}),
    "controls_multiclass": ("multiclass", {11}),
    "regression": ("regression", {13, 16, 19}),
    "regression_w500": ("regression", {14, 17, 20}),
    "regression_w250": ("regression", {15, 18, 21}),
    "regression_sliding": ("regression", {22, 23, 24}),
    "regression_bci4": ("regression", {25}),
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
def test_list_shape_label_mode_and_task_sets(name):
    label_mode, task_sets = EXPECTED[name]
    manifest = _load(name)
    assert manifest["label_mode"] == label_mode
    assert manifest["n_cells"] == len(_cells(name))
    for entry in manifest["tasks"].values():
        targets = entry["subject_sessions"]
        assert entry["n_subject_sessions"] == len(targets) == len(set(targets))
        for target in targets:
            match = TARGET.fullmatch(target)
            assert match, target
            assert 1 <= int(match["subject"]) <= 29
            assert int(match["session"]) in task_sets, (name, target)


def test_main_table_has_178_cells():
    assert len(_cells("binary")) == 153
    assert len(_cells("multiclass")) == 25


def test_positions_lists_are_subsets_of_the_main_table():
    assert _cells("positions_binary") <= _cells("binary")
    assert _cells("positions_multiclass") <= _cells("multiclass")


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
        assert _load(name)["label_mode"] == "multiclass"
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
