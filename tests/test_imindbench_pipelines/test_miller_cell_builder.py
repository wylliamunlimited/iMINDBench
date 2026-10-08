"""The Miller cell-list builder on small H5 files shaped like the real ones."""

import json

import h5py
import numpy as np
from miller_fakes import install_fake_miller

from imindbench.cell_manifests import build_millerecog2019 as builder


def _write_h5(path, *, tasks, controls=(), regression=(), quality=None):
    with h5py.File(path, "w") as handle:
        handle.attrs["tasks_json"] = json.dumps(tasks)
        handle.attrs["control_tasks_json"] = json.dumps(list(controls))
        handle.attrs["regression_tasks_json"] = json.dumps(list(regression))
        channels = handle.create_group("channels")
        channels["included"] = np.array([True, True, True, False])
        if quality is not None:
            channels["coord_mni152_quality"] = np.array(quality, dtype="S1")


def _binary(task):
    return {"task": task, "label_mode": "binary", "n_classes": 2}


def _multiclass(task, n_classes):
    return {"task": task, "label_mode": "multiclass", "n_classes": n_classes}


def test_builder_sorts_cells_into_lists(tmp_path, monkeypatch, capsys):
    install_fake_miller(monkeypatch)
    data = tmp_path / "miller_ecog_library_2019"
    data.mkdir()
    # sub3 / task set 1 (main table), with good positions on 2 included channels
    _write_h5(
        data / "sub-bp_set-motor_basic.h5",
        tasks=[_binary("move_vs_rest"), _multiclass("which_effector", 3)],
        quality=["A", "C", "B", "A"],
    )
    # sub1 / task set 1, all positions low quality
    _write_h5(
        data / "sub-al_set-motor_basic.h5",
        tasks=[_binary("move_vs_rest")],
        quality=["C", "C", "C", "A"],
    )
    # sub3 / task set 11 (new set) with one control task
    _write_h5(
        data / "sub-bp_set-memory_nback.h5",
        tasks=[_binary("nback_vs_rest"), _binary("fixation_vs_task")],
        controls=["fixation_vs_task"],
    )
    # sub3 / task sets 13 and 25 (regression)
    _write_h5(
        data / "sub-bp_set-fingerflex_reg_w1000.h5",
        tasks=[],
        regression=[{"target": "flex_thumb"}],
    )
    # sub2 / task set 9 (0.8 s face windows) and sub3 / task set 14 (0.5 s windows)
    _write_h5(data / "sub-ap_set-faces_basic.h5", tasks=[_binary("face_vs_house")])
    _write_h5(
        data / "sub-bp_set-fingerflex_reg_w500.h5",
        tasks=[],
        regression=[{"target": "flex_thumb"}],
    )
    _write_h5(
        data / "sub-bp_set-fingerflex_reg_w1000h25b.h5",
        tasks=[],
        regression=[{"target": "flex_thumb"}],
    )
    out = tmp_path / "lists"

    assert (
        builder.main(
            [
                "--data-dir",
                str(data),
                "--out-dir",
                str(out),
                "--min-position-channels",
                "2",
            ]
        )
        == 0
    )

    def cells(name):
        manifest = json.loads((out / f"{name}.json").read_text())
        assert manifest["n_cells"] == sum(
            len(v["subject_sessions"]) for v in manifest["tasks"].values()
        )
        return {
            task: entry["subject_sessions"] for task, entry in manifest["tasks"].items()
        }

    assert cells("binary") == {"move_vs_rest": ["sub1_sess1", "sub3_sess1"]}
    assert cells("multiclass") == {"which_effector": ["sub3_sess1"]}
    assert cells("positions_binary") == {"move_vs_rest": ["sub3_sess1"]}
    assert cells("positions_multiclass") == {"which_effector": ["sub3_sess1"]}
    assert cells("new_sets_binary") == {"nback_vs_rest": ["sub3_sess11"]}
    assert cells("controls_binary") == {"fixation_vs_task": ["sub3_sess11"]}
    for pair in ("0v1", "0v2", "1v2"):
        assert cells(f"class_pair_{pair}") == {"which_effector": ["sub3_sess1"]}
    assert cells("regression") == {"flex_thumb": ["sub3_sess13"]}
    assert cells("regression_bci4") == {"flex_thumb": ["sub3_sess25"]}
    assert cells("regression_w500") == {"flex_thumb": ["sub3_sess14"]}
    assert cells("faces08_binary") == {"face_vs_house": ["sub2_sess9"]}
    assert not (out / "regression_sliding.json").exists()
    assert "binary: 2 cells, 1 tasks, 2 recordings" in capsys.readouterr().out


def test_positions_list_is_skipped_without_positions(tmp_path, monkeypatch):
    install_fake_miller(monkeypatch)
    data = tmp_path / "data"
    data.mkdir()
    _write_h5(data / "sub-bp_set-motor_basic.h5", tasks=[_binary("move_vs_rest")])
    out = tmp_path / "lists"
    builder.main(["--data-dir", str(data), "--out-dir", str(out)])
    assert (out / "binary.json").exists()
    assert not list(out.glob("positions_*.json"))


def test_from_brainsets_reads_tables_from_the_brainsets_module(tmp_path, monkeypatch):
    import sys

    import miller_fakes

    # No fake torch_brain class is installed, so only the brainsets path works.
    monkeypatch.setitem(sys.modules, "brainsets.datasets.MillerECoG2019", miller_fakes)
    data = tmp_path / "data"
    data.mkdir()
    _write_h5(data / "sub-zt_set-gestures.h5", tasks=[_binary("gesture_vs_rest")])
    _write_h5(
        data / "sub-zt_set-mouse_track_w1000h50.h5",
        tasks=[],
        regression=[{"target": "cursor_vx"}],
    )
    out = tmp_path / "lists"
    builder.main(["--data-dir", str(data), "--out-dir", str(out), "--from-brainsets"])
    manifest = json.loads((out / "binary.json").read_text())
    assert manifest["tasks"]["gesture_vs_rest"]["subject_sessions"] == ["sub29_sess5"]
    sliding = json.loads((out / "regression_sliding.json").read_text())
    assert sliding["tasks"]["cursor_vx"]["subject_sessions"] == ["sub29_sess24"]


def test_from_brainsets_without_brainsets_is_reported(monkeypatch):
    import sys

    import pytest

    monkeypatch.setitem(sys.modules, "brainsets.datasets.MillerECoG2019", None)
    with pytest.raises(ImportError, match="--from-brainsets needs brainsets"):
        builder.miller_dataset_module(from_brainsets=True)


def test_derived_lists_name_every_task_of_their_source_list():
    def recording(target, session, tasks, positions):
        return {
            "target": target,
            "session": session,
            "tasks": tasks,
            "controls": [],
            "regression": [],
            "good_position_channels": positions,
        }

    lists = builder.build_lists(
        [
            recording("sub3_sess1", 1, [_multiclass("direction_4way", 4)], 0),
            recording(
                "sub3_sess5",
                5,
                [_binary("flex_vs_rest"), _multiclass("fingers_5way", 5)],
                5,
            ),
        ]
    )
    # direction_4way has no class 4 and no recording with positions, so it is
    # named with an empty list rather than left out.
    assert lists["class_pair_0v4"] == {
        "direction_4way": set(),
        "fingers_5way": {"sub3_sess5"},
    }
    assert lists["positions_multiclass"] == {
        "direction_4way": set(),
        "fingers_5way": {"sub3_sess5"},
    }
    assert lists["positions_binary"] == {"flex_vs_rest": {"sub3_sess5"}}
    manifest = builder.to_manifest(lists["class_pair_0v4"], name="class_pair_0v4")
    assert manifest["tasks"]["direction_4way"] == {
        "subject_sessions": [],
        "n_subject_sessions": 0,
    }
