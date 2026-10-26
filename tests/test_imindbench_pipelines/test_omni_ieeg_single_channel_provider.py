"""Single-channel Omni-iEEG protocol: per-channel labels, windows, selection."""

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
from brainsets.datasets.OmniIEEG import (  # noqa: E402
    upstream_random_window_offsets,
)
from brainsets.descriptions import SessionDescription, SubjectDescription  # noqa: E402
from temporaldata import ArrayDict, Data, Interval, RegularTimeSeries  # noqa: E402

RATE = 50.0  # small rate keeps the synthetic files tiny

ANATOMICAL_MAPPING = pd.DataFrame(
    {
        "label": ["superiorfrontal", "superiortemporal", "entorhinal", "-1"],
        "subcategory": ["frontal", "temporal", "limbic", "-1"],
        "Proposal_1": [
            "Dorsolateral frontal",
            "Superior temporal",
            "Mesial temporal",
            "-1",
        ],
        "Proposal_2": ["Frontal", "Temporal", "Mesial temporal", "-1"],
    }
)


def _write_recording(
    root,
    edf_relpath,
    subject,
    seconds,
    *,
    dirname="omni_ieeg",
    good=None,
    soz=None,
    resection=None,
    anatomical=None,
    n_channels=4,
):
    stem = edf_relpath.removesuffix(".edf")
    recording_id = stem.replace("/", "_")
    signal = np.zeros((int(seconds * RATE), n_channels), dtype=np.float32)
    seeg = RegularTimeSeries(data=signal, sampling_rate=RATE, domain="auto")
    channels = ArrayDict(
        id=np.arange(n_channels, dtype=np.int64),
        name=np.asarray([f"A{i}-A{i + 1}" for i in range(n_channels)], dtype="U"),
        included=(
            np.ones(n_channels, dtype=bool)
            if good is None
            else np.asarray(good, dtype=bool)
        ),
    )
    channels.good = (
        np.ones(n_channels, dtype=np.float32)
        if good is None
        else np.asarray(good, dtype=np.float32)
    )
    channels.soz = (
        np.zeros(n_channels, dtype=np.float32)
        if soz is None
        else np.asarray(soz, dtype=np.float32)
    )
    channels.resection = (
        np.zeros(n_channels, dtype=np.float32)
        if resection is None
        else np.asarray(resection, dtype=np.float32)
    )
    channels.anatomical = (
        np.asarray(["-1"] * n_channels, dtype="U")
        if anatomical is None
        else np.asarray(anatomical, dtype="U")
    )
    data = Data(
        subject=SubjectDescription(id=subject, species="HOMO_SAPIENS"),
        session=SessionDescription(
            id="ses-01", recording_date=datetime(1970, 1, 1, tzinfo=timezone.utc)
        ),
        seeg_data=seeg,
        channels=channels,
        domain=seeg.domain,
    )
    directory = root / dirname
    directory.mkdir(exist_ok=True)
    with h5py.File(directory / f"{recording_id}.h5", "w") as handle:
        data.to_hdf5(handle, serialize_fn_map=serialize_fn_map)
    return recording_id


def _final_split_row(patient, edf, split, **overrides):
    row = {
        "patient_name": patient,
        "edf_name": edf,
        "dataset": "hup",
        "outcome": 1,
        "event_annotation": True,
        "interictal": True,
        "has_soz": True,
        "has_resection": True,
        "soz_count": 1,
        "resection_count": 1,
        "has_anatomical": True,
        "frequency": 1000.0,
        "length": 400.0,
        "split": split,
    }
    row.update(overrides)
    return row


@pytest.fixture()
def soz_tree(tmp_path):
    """Two train patients (one seizure-free with resection, one poor outcome) + test."""
    rows = []
    # Seizure-free patient with resection: non-resected good channels are positives,
    # resected SOZ channels are negatives.
    for patient in ("sub-p00", "sub-p01"):
        edf = f"{patient}/ses-01/ieeg/{patient}_ses-01_task-interictal_run-01_ieeg.edf"
        _write_recording(
            tmp_path,
            edf,
            patient,
            400.0,
            good=[1, 1, 1, 0],
            soz=[0, 1, 0, 1],
            resection=[0, 1, 0, 1],
        )
        rows.append(_final_split_row(patient, edf, "train", outcome=1))
    # Poor outcome: every good SOZ channel is a negative, resection is irrelevant.
    edf = "sub-p02/ses-01/ieeg/sub-p02_ses-01_task-interictal_run-01_ieeg.edf"
    _write_recording(
        tmp_path,
        edf,
        "sub-p02",
        400.0,
        good=[1, 1, 1, 1],
        soz=[1, 0, 0, 0],
        resection=[0, 0, 0, 0],
    )
    rows.append(
        _final_split_row("sub-p02", edf, "train", outcome=0, has_resection=False)
    )
    # Test patient: labels 1 (non-resected, outcome 1), 0 (soz), -1 (neither).
    edf = "sub-t00/ses-01/ieeg/sub-t00_ses-01_task-interictal_run-01_ieeg.edf"
    _write_recording(
        tmp_path,
        edf,
        "sub-t00",
        400.0,
        good=[1, 1, 1, 0],
        soz=[0, 1, 0, 0],
        resection=[0, 1, 1, 0],
    )
    rows.append(_final_split_row("sub-t00", edf, "test", outcome=1))
    csv = tmp_path / "final_split.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    return tmp_path, csv


def _soz_dataset(tree, split, **kwargs):
    root, csv = tree
    return OmniIEEG(
        root=str(root),
        split=split,
        task="soz_outcome",
        test_subject="all",
        test_session="all",
        single_channel=True,
        split_protocol="auto",
        window_sec=60.0,
        final_split_csv=str(csv),
        train_seconds_per_recording=None,
        train_window_sampling="upstream_random",
        val_fraction=0.34,
        **kwargs,
    )


def test_soz_train_labels_follow_upstream_positive_and_negative_rows(soz_tree):
    train = _soz_dataset(soz_tree, "train")
    val = _soz_dataset(soz_tree, "val")
    labels = {}
    for dataset in (train, val):
        for rid in dataset.recording_ids:
            labels[rid] = dataset.get_channel_labels(rid).tolist()
    seizure_free = [v for k, v in labels.items() if "p00" in k or "p01" in k]
    # good & resection==0 -> 1; good & soz==1 & resection!=0 -> 0; bad channel -> -1.
    assert all(v == [1, 0, 1, -1] for v in seizure_free)
    poor_outcome = next(v for k, v in labels.items() if "p02" in k)
    # outcome != 1: every good SOZ channel is a negative, nothing is a positive.
    assert poor_outcome == [0, -1, -1, -1]


def test_soz_test_keeps_unlabelled_good_channels(soz_tree):
    test = _soz_dataset(soz_tree, "test")
    rid = test.recording_ids[0]
    # outcome==1 & resection==0 -> 1, soz==1 -> 0, other good -> -1, bad -> excluded.
    assert test.get_channel_labels(rid).tolist() == [1, 0, -1, -1]
    assert test.keep_unlabeled_test_channels is True
    assert test.get_channel_metadata(rid)["included_mask"].tolist() == [
        True,
        True,
        True,
        False,
    ]


def test_single_channel_masks_drop_unlabelled_channels_in_train(soz_tree):
    train = _soz_dataset(soz_tree, "train")
    for rid in train.recording_ids:
        mask = train.get_channel_metadata(rid)["included_mask"]
        labels = train.get_channel_labels(rid)
        assert mask.tolist() == (labels >= 0).tolist()
    assert train.keep_unlabeled_test_channels is False


def test_train_and_val_are_subject_disjoint(soz_tree):
    train = _soz_dataset(soz_tree, "train")
    val = _soz_dataset(soz_tree, "val")

    def subjects(ds):
        return {str(ds.get_recording(r).subject.id) for r in ds.recording_ids}

    assert subjects(train).isdisjoint(subjects(val))
    assert len(subjects(train) | subjects(val)) == 3
    # Validation needs both classes for early stopping on val AUROC.
    val_labels = set()
    for rid in val.recording_ids:
        val_labels.update(int(v) for v in val.get_channel_labels(rid) if v >= 0)
    assert len(val_labels) >= 1


def test_upstream_random_offsets_match_the_upstream_rule():
    rng = np.random.default_rng(0)
    # 400 s at 1000 Hz, 60 s windows: usable 398 s, max_possible = 398000 // 30000 = 13,
    # so all 5 requested windows are drawn inside [1 s, 339 s].
    offsets = upstream_random_window_offsets(
        n_samples=400_000,
        sampling_rate=1000.0,
        window_sec=60.0,
        edge_trim_sec=1.0,
        max_samples=5,
        rng=rng,
    )
    assert len(offsets) == 5
    assert np.all(np.diff(offsets) > 0)
    assert offsets.min() >= 1000
    assert offsets.max() <= 1000 + (398_000 - 60_000)
    # Too short for a window -> nothing.
    assert (
        upstream_random_window_offsets(
            n_samples=61_000,
            sampling_rate=1000.0,
            window_sec=60.0,
            edge_trim_sec=1.0,
            max_samples=5,
            rng=rng,
        ).size
        == 0
    )
    # Fewer candidate positions than requested windows -> evenly spaced fallback.
    # usable = 60002, L = 60000 -> max_possible = 2, valid_range = 2, so the two
    # windows are spaced i * 2 // 1.
    evenly = upstream_random_window_offsets(
        n_samples=62_002,
        sampling_rate=1000.0,
        window_sec=60.0,
        edge_trim_sec=1.0,
        max_samples=5,
        rng=rng,
    )
    assert evenly.tolist() == [1000, 1002]


def test_train_windows_are_random_and_test_windows_are_tiled(soz_tree):
    train = _soz_dataset(soz_tree, "train", train_windows_per_recording=3)
    test = _soz_dataset(soz_tree, "test")
    train_windows = train.get_sampling_intervals()
    assert all(len(iv.start) == 3 for iv in train_windows.values())
    # Window labels are placeholders for channel-labelled tasks.
    assert all(
        set(np.asarray(iv.label).tolist()) == {-1} for iv in train_windows.values()
    )
    for iv in test.get_sampling_intervals().values():
        expected = 6  # floor((400 - 2) / 60)
        assert len(iv.start) == expected
        np.testing.assert_allclose(iv.start, 1.0 + 60.0 * np.arange(expected))


def test_train_sampling_is_deterministic(soz_tree):
    first = _soz_dataset(soz_tree, "train").get_sampling_intervals()
    second = _soz_dataset(soz_tree, "train").get_sampling_intervals()
    for rid in first:
        np.testing.assert_array_equal(first[rid].start, second[rid].start)


def test_soz_requires_single_channel(soz_tree):
    root, csv = soz_tree
    with pytest.raises(ValueError, match="single-channel task"):
        OmniIEEG(
            root=str(root),
            split="test",
            task="soz_outcome",
            test_subject="all",
            test_session="all",
            final_split_csv=str(csv),
        )


@pytest.fixture()
def anatomical_tree(tmp_path):
    rows = []
    regions = ["superiorfrontal", "superiortemporal", "entorhinal", "-1"]
    for patient, split in (
        ("sub-a00", "train"),
        ("sub-a01", "train"),
        ("sub-a02", "test"),
    ):
        edf = f"{patient}/ses-01/ieeg/{patient}_ses-01_task-interictal_run-01_ieeg.edf"
        _write_recording(
            tmp_path, edf, patient, 700.0, good=[1, 1, 0, 1], anatomical=regions
        )
        rows.append(_final_split_row(patient, edf, split, length=700.0))
    csv = tmp_path / "final_split.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    mapping = tmp_path / "anatomical_mapping.csv"
    ANATOMICAL_MAPPING.to_csv(mapping, index=False)
    return tmp_path, csv, mapping


def test_anatomical_labels_use_every_channel_and_the_subcategory_ids(anatomical_tree):
    root, csv, mapping = anatomical_tree
    test = OmniIEEG(
        root=str(root),
        split="test",
        task="anatomical",
        test_subject="all",
        test_session="all",
        single_channel=True,
        split_protocol="auto",
        window_sec=300.0,
        final_split_csv=str(csv),
        anatomical_mapping_csv=str(mapping),
        train_seconds_per_recording=None,
    )
    rid = test.recording_ids[0]
    # frontal 0, temporal 1, limbic 4, unlabelled -1 — the bad channel (index 2)
    # is kept because upstream does not filter on `good` for this task.
    assert test.get_channel_labels(rid).tolist() == [0, 1, 4, -1]
    assert test.get_channel_metadata(rid)["included_mask"].tolist() == [
        True,
        True,
        True,
        False,
    ]
    intervals = test.get_sampling_intervals()
    assert all(len(iv.start) == 2 for iv in intervals.values())  # floor((700-2)/300)


def _write_hfo(root, name, subject, split, labels, event_len=100):
    n = len(labels)
    signal = np.arange(n * event_len, dtype=np.float32).reshape(-1, 1)
    seeg = RegularTimeSeries(data=signal, sampling_rate=RATE, domain="auto")
    event_sec = event_len / RATE
    starts = np.arange(n, dtype=np.float64) * event_sec
    data = Data(
        subject=SubjectDescription(id=subject, species="HOMO_SAPIENS"),
        session=SessionDescription(
            id="ses-01", recording_date=datetime(1970, 1, 1, tzinfo=timezone.utc)
        ),
        seeg_data=seeg,
        channels=ArrayDict(
            id=np.zeros(1, dtype=np.int64),
            name=np.asarray(["hfo_event"], dtype="U"),
            included=np.ones(1, dtype=bool),
        ),
        hfo_events=Interval(
            start=starts,
            end=starts + event_sec,
            label=np.asarray(labels, dtype=np.int64),
        ),
        domain=seeg.domain,
    )
    splits = Data(domain=seeg.domain)
    setattr(
        splits,
        split,
        Interval(start=0.0, end=float(n * event_sec), label=np.array([0])),
    )
    data.splits = splits
    directory = root / "omni_ieeg_hfo"
    directory.mkdir(exist_ok=True)
    with h5py.File(directory / f"{name}.h5", "w") as handle:
        data.to_hdf5(handle, serialize_fn_map=serialize_fn_map)


@pytest.fixture()
def hfo_tree(tmp_path):
    for i in range(4):
        _write_hfo(
            tmp_path, f"train_sub-h{i:02d}", f"sub-h{i:02d}", "train", [0, 1, 2, 1]
        )
    _write_hfo(tmp_path, "test_sub-h90", "sub-h90", "test", [0, 2, 2])
    return tmp_path


def _hfo_dataset(root, split):
    return OmniIEEG(
        root=str(root),
        split=split,
        task="hfo",
        test_subject="all",
        test_session="all",
        single_channel=True,
        split_protocol="auto",
        window_sec=100 / RATE,
        edge_trim_sec=0.0,
        train_seconds_per_recording=None,
    )


def test_hfo_windows_are_events_with_their_three_class_label(hfo_tree):
    test = _hfo_dataset(hfo_tree, "test")
    intervals = test.get_sampling_intervals()
    assert len(intervals) == 1
    interval = next(iter(intervals.values()))
    assert np.asarray(interval.label).tolist() == [0, 2, 2]
    np.testing.assert_allclose(interval.end - interval.start, 100 / RATE)
    # The window label carries the class, so there are no per-channel labels.
    assert test.get_channel_labels(test.recording_ids[0]) is None


def test_hfo_train_and_val_are_subject_disjoint(hfo_tree):
    train = _hfo_dataset(hfo_tree, "train")
    val = _hfo_dataset(hfo_tree, "val")
    assert set(train.recording_ids).isdisjoint(val.recording_ids)
    assert len(train.recording_ids) + len(val.recording_ids) == 4


def test_hfo_window_length_must_match_the_stored_events(hfo_tree):
    dataset = OmniIEEG(
        root=str(hfo_tree),
        split="test",
        task="hfo",
        test_subject="all",
        test_session="all",
        single_channel=True,
        split_protocol="auto",
        window_sec=1.0,
        edge_trim_sec=0.0,
        train_seconds_per_recording=None,
    )
    with pytest.raises(ValueError, match="window_sec"):
        dataset.get_sampling_intervals()


def test_missing_hfo_data_names_the_prepare_command(tmp_path):
    (tmp_path / "omni_ieeg_hfo").mkdir()
    with pytest.raises(FileNotFoundError, match="brainsets prepare omni_ieeg_hfo"):
        _hfo_dataset(tmp_path, "test")


@pytest.fixture()
def sleep_tree(tmp_path):
    rows = []
    for _i, (patient, label, split) in enumerate(
        [
            ("sub-s00", "sleep", "train"),
            ("sub-s01", "awake", "train"),
            ("sub-s02", "sleep", "train"),
            ("sub-s03", "awake", "train"),
            ("sub-s10", "sleep", "test"),
            ("sub-s11", "awake", "test"),
        ]
    ):
        edf = f"{patient}/ses-01/ieeg/{patient}_ses-01_task-{label}_ieeg.edf"
        _write_recording(tmp_path, edf, patient, 200.0, good=[1, 1, 1, 0])
        rows.append(
            {"patient": patient, "edf_file": edf, "label": label, "split": split}
        )
    csv = tmp_path / "sleep_split_cross.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    return tmp_path, csv


def test_sleep_awake_labels_good_channels_with_the_recording_label(sleep_tree):
    root, csv = sleep_tree
    test = OmniIEEG(
        root=str(root),
        split="test",
        task="sleep_awake",
        test_subject="all",
        test_session="all",
        single_channel=True,
        split_protocol="auto",
        window_sec=60.0,
        split_csv=str(csv),
        train_seconds_per_recording=None,
    )
    by_label = {}
    for rid in test.recording_ids:
        by_label[rid] = test.get_channel_labels(rid).tolist()
    sleep = next(v for k, v in by_label.items() if "s10" in k)
    awake = next(v for k, v in by_label.items() if "s11" in k)
    assert sleep == [1, 1, 1, -1]  # sleep = 1, bad channel unlabelled
    assert awake == [0, 0, 0, -1]


def test_sleep_awake_rejects_the_ictal_label_map(sleep_tree):
    root, csv = sleep_tree
    with pytest.raises(ValueError, match="unexpected labels"):
        OmniIEEG(
            root=str(root),
            split="test",
            task="ictal_interictal",
            test_subject="all",
            test_session="all",
            split_csv=str(csv),
        )
