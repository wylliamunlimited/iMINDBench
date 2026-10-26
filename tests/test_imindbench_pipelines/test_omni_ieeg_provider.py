"""The OmniIEEG dataset class, multi-channel protocol (loader-level).

These tests exercise the real loader and skip when it is not installed; the
harness tests next to them fake it (omni_fakes.py).
"""

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("temporaldata")
pytest.importorskip("h5py")
brainsets_datasets = pytest.importorskip("brainsets.datasets")
if not hasattr(brainsets_datasets, "OmniIEEG"):
    pytest.skip(
        "brainsets.datasets.OmniIEEG not available (brainsets-wylliam-fork, "
        "branch omni-ieeg)",
        allow_module_level=True,
    )
import h5py  # noqa: E402
from brainsets import serialize_fn_map  # noqa: E402
from brainsets.datasets import OmniIEEG  # noqa: E402
from brainsets.descriptions import SessionDescription, SubjectDescription  # noqa: E402
from temporaldata import ArrayDict, Data, RegularTimeSeries  # noqa: E402

RATE = 50.0  # small rate keeps the synthetic files tiny


def _write_recording(root, edf_relpath, subject, seconds, n_channels=3, good=None):
    stem = edf_relpath.removesuffix(".edf")
    recording_id = stem.replace("/", "_")
    signal = np.zeros((int(seconds * RATE), n_channels), dtype=np.float32)
    seeg = RegularTimeSeries(data=signal, sampling_rate=RATE, domain="auto")
    good = np.ones(n_channels, dtype=bool) if good is None else np.asarray(good)
    data = Data(
        subject=SubjectDescription(id=subject, species="HOMO_SAPIENS"),
        session=SessionDescription(
            id="ses-01", recording_date=datetime(1970, 1, 1, tzinfo=timezone.utc)
        ),
        seeg_data=seeg,
        channels=ArrayDict(
            id=np.arange(n_channels, dtype=np.int64),
            name=np.asarray([f"A{i}-A{i + 1}" for i in range(n_channels)], dtype="U"),
            included=good,
        ),
        domain=seeg.domain,
    )
    with h5py.File(root / "omni_ieeg" / f"{recording_id}.h5", "w") as handle:
        data.to_hdf5(handle, serialize_fn_map=serialize_fn_map)
    return recording_id


@pytest.fixture()
def omni_tree(tmp_path):
    (tmp_path / "omni_ieeg").mkdir()
    rows = []

    def add(patient, task, split, seconds, run="01"):
        token = f"_task-{task}" if task else ""
        label = "interictal" if task in ("interictal", None) else "ictal"
        edf = f"{patient}/ses-01/ieeg/{patient}_ses-01{token}_run-{run}_ieeg.edf"
        _write_recording(tmp_path, edf, patient, seconds)
        rows.append((patient, edf, label, split))

    # 10 train patients, alternating labels, so val can hold both classes.
    for i in range(10):
        add(f"sub-p{i:02d}", "interictal" if i % 2 else "ictal", "train", 302.0)
    add("sub-t00", "ictal", "test", 182.0)
    add("sub-zurich01", None, "test", 122.0)  # no task- token, label from CSV
    # A recording that exists on disk but is not in the split must be ignored.
    _write_recording(
        tmp_path,
        "sub-x/ses-01/ieeg/sub-x_ses-01_task-ictal_run-01_ieeg.edf",
        "sub-x",
        302.0,
    )
    csv = tmp_path / "ictal_split_cross.csv"
    pd.DataFrame(rows, columns=["patient", "edf_file", "label", "split"]).to_csv(
        csv, index=False
    )
    return tmp_path, csv


def _dataset(tree, split, **kwargs):
    root, csv = tree
    return OmniIEEG(
        root=str(root),
        split=split,
        test_subject="all",
        test_session="all",
        split_csv=str(csv),
        **kwargs,
    )


def test_ictal_cross_selects_listed_recordings_with_upstream_labels(omni_tree):
    test = _dataset(omni_tree, "test")
    assert len(test.recording_ids) == 2
    labels = {
        rid: int(iv.label[0]) for rid, iv in test.get_sampling_intervals().items()
    }
    zurich = next(r for r in labels if "zurich" in r)
    ictal = next(r for r in labels if "sub-t00" in r)
    assert labels[zurich] == 1  # interictal = 1 (upstream polarity)
    assert labels[ictal] == 0


def test_train_and_val_are_subject_disjoint_and_exclude_unlisted(omni_tree):
    train = _dataset(omni_tree, "train")
    val = _dataset(omni_tree, "val")

    def subjects(ds):
        return {str(ds.get_recording(r).subject.id) for r in ds.recording_ids}

    assert subjects(train).isdisjoint(subjects(val))
    assert len(subjects(train)) + len(subjects(val)) == 10
    val_labels = {int(iv.label[0]) for iv in val.get_sampling_intervals().values()}
    assert val_labels == {0, 1}
    assert not any("sub-x" in r for r in train.recording_ids + val.recording_ids)


def test_sixty_second_test_tiling_matches_upstream_uniform_windows(omni_tree):
    intervals = _dataset(omni_tree, "test", window_sec=60.0).get_sampling_intervals()
    for rid, iv in intervals.items():
        expected = 3 if "sub-t00" in rid else 2  # floor((182-2)/60), floor((122-2)/60)
        assert len(iv.start) == expected
        np.testing.assert_allclose(iv.start, 1.0 + 60.0 * np.arange(expected))
        np.testing.assert_allclose(iv.end - iv.start, 60.0)


def test_train_budget_is_in_seconds_for_both_window_lengths(omni_tree):
    long_windows = _dataset(
        omni_tree, "train", window_sec=60.0, train_seconds_per_recording=120.0
    ).get_sampling_intervals()
    short_windows = _dataset(
        omni_tree, "train", window_sec=1.0, train_seconds_per_recording=120.0
    ).get_sampling_intervals()
    assert all(len(iv.start) == 2 for iv in long_windows.values())
    assert all(len(iv.start) == 120 for iv in short_windows.values())


def test_train_sampling_is_deterministic(omni_tree):
    first = _dataset(omni_tree, "train", window_sec=1.0).get_sampling_intervals()
    second = _dataset(omni_tree, "train", window_sec=1.0).get_sampling_intervals()
    for rid in first:
        np.testing.assert_array_equal(first[rid].start, second[rid].start)


def test_describe_selection_reports_protocol_settings(omni_tree):
    info = _dataset(omni_tree, "test", window_sec=1.0).describe_selection()
    assert info["split_protocol"] == "ictal_cross"
    assert info["window_sec"] == 1.0
    assert info["train_seconds_per_recording"] == 600.0
