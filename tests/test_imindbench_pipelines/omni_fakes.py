"""A small fake of the OmniIEEG dataset class for tests.

The pinned torch_brain does not ship OmniIEEG yet (it lives in
brainsets-wylliam-fork, branch omni-ieeg, ``brainsets/datasets/OmniIEEG.py``),
so tests install this class as the ``omni_ieeg`` dataset instead. It copies
the parts of the public API the harness touches:

- the constructor takes the ``provider_options`` keys of
  ``imindbench/conf/dataset/omni_ieeg*.yaml`` (``window_sec``,
  ``edge_trim_sec``, ``split_protocol``, ``split_csv``, ``final_split_csv``,
  ``anatomical_mapping_csv``, ``hfo_dirname``, ``train_window_sampling``,
  ``train_windows_per_recording``, ``train_seconds_per_recording``,
  ``val_fraction``, ``single_channel``) next to the usual split-selection
  keywords, and keeps them in ``options`` so tests can check they arrived;
- ``test_subject="all"`` / ``test_session="all"`` select the whole held-out
  set (one pooled cell), the only regime is ``hold-out-subject`` with one
  fold, and ``val`` is a subject-disjoint slice of the train patients;
- ``single_channel`` and ``keep_unlabeled_test_channels`` drive the
  single-channel expansion; ``get_channel_labels`` returns the per-channel
  task labels (``-1`` = unlabelled) or ``None`` when the label lives on the
  window (multi-channel protocol and the HFO task);
- ``get_channel_metadata`` returns ``ids``, ``names``, ``included_mask``,
  ``indices`` and an all-zero ``omni_none`` coordinate frame, since Omni-iEEG
  ships no electrode positions;
- ``describe_selection`` reports the protocol settings.

The signal is synthetic: a labelled channel carries a 10 Hz sine whose
amplitude grows with its label, so a simple model can separate the classes.
Unlabelled channels are noise only.
"""

from __future__ import annotations

import numpy as np
from torch_brain.data import Data, Interval, RegularTimeSeries

TASKS = ("ictal_interictal", "sleep_awake", "soz_outcome", "anatomical", "hfo")
CHANNEL_LABELLED_TASKS = (
    "ictal_interictal",
    "sleep_awake",
    "soz_outcome",
    "anatomical",
)
TRAIN_PATIENTS = tuple(f"sub-p{i:02d}" for i in range(8))
TEST_PATIENTS = ("sub-t00", "sub-t01")


def recording_id_for(patient: str, task: str = "interictal") -> str:
    """The processed recording id of a BIDS EDF path (``/`` becomes ``_``)."""
    return f"{patient}_ses-01_ieeg_{patient}_ses-01_task-{task}_run-01_ieeg"


class FakeOmniIEEG:
    """Stands in for OmniIEEG in benchmark (split-selection) mode."""

    sampling_rate_hz = 100.0
    n_channels = 4
    windows_per_recording = 3
    # Channel 3 is marked bad in *_channels.tsv.
    good_channels = (True, True, True, False)

    def __init__(
        self,
        root=None,
        recording_ids=None,
        transform=None,
        *,
        split=None,
        subset_tier=None,
        test_subject=None,
        test_session=None,
        label_mode=None,
        task=None,
        regime=None,
        fold=None,
        window_sec=60.0,
        edge_trim_sec=1.0,
        split_protocol="auto",
        split_csv=None,
        train_seconds_per_recording=600.0,
        val_fraction=0.2,
        single_channel=False,
        train_window_sampling="tile_subsample",
        train_windows_per_recording=5,
        final_split_csv=None,
        anatomical_mapping_csv=None,
        hfo_dirname="omni_ieeg_hfo",
        uniquify_channel_ids_with_subject=True,
        uniquify_channel_ids_with_session=False,
        dirname="omni_ieeg",
        **kwargs,
    ):
        if root is None:
            raise ValueError("OmniIEEG requires an explicit processed-data root.")
        if task not in TASKS:
            raise ValueError(
                f"Unknown Omni-iEEG task {task!r}; expected one of {TASKS}."
            )
        if regime != "hold-out-subject":
            raise ValueError("OmniIEEG supports only the 'hold-out-subject' regime.")
        if fold not in (None, 0):
            raise ValueError(f"OmniIEEG has a single fold, got fold={fold!r}.")
        if split not in ("train", "val", "test"):
            raise ValueError(f"Invalid split {split!r}.")
        if str(test_subject).lower() != "all" or str(test_session).lower() != "all":
            raise ValueError(
                "The fake OmniIEEG only supports the pooled cell "
                "test_subject='all', test_session='all'."
            )
        if window_sec <= 0:
            raise ValueError(f"window_sec must be positive, got {window_sec!r}.")
        if not 0.0 < val_fraction < 1.0:
            raise ValueError(f"val_fraction must lie in (0, 1), got {val_fraction!r}.")
        if task == "soz_outcome" and not single_channel:
            raise ValueError("soz_outcome is defined per channel: set single_channel.")
        if task == "hfo" and float(window_sec) != 2.0:
            raise ValueError("The stored HFO events are 2 s long; set window_sec=2.")
        self.root = root
        self.dirname = dirname
        self.subset_tier = subset_tier
        self.label_mode = label_mode or "binary"
        self.task = task
        self.regime = regime
        self.fold = 0 if fold is None else int(fold)
        self.split = split
        self.test_subject = test_subject
        self.test_session = test_session
        self.window_sec = float(window_sec)
        self.edge_trim_sec = float(edge_trim_sec)
        self.split_protocol = split_protocol
        self.split_csv = split_csv
        self.final_split_csv = final_split_csv
        self.anatomical_mapping_csv = anatomical_mapping_csv
        self.hfo_dirname = hfo_dirname
        self.single_channel = bool(single_channel)
        self.train_window_sampling = train_window_sampling
        self.train_windows_per_recording = int(train_windows_per_recording)
        self.train_seconds_per_recording = train_seconds_per_recording
        self.val_fraction = float(val_fraction)
        self.options = {
            "window_sec": window_sec,
            "edge_trim_sec": edge_trim_sec,
            "split_protocol": split_protocol,
            "split_csv": split_csv,
            "final_split_csv": final_split_csv,
            "anatomical_mapping_csv": anatomical_mapping_csv,
            "hfo_dirname": hfo_dirname,
            "train_window_sampling": train_window_sampling,
            "train_windows_per_recording": train_windows_per_recording,
            "train_seconds_per_recording": train_seconds_per_recording,
            "val_fraction": val_fraction,
            "single_channel": single_channel,
            **kwargs,
        }
        self.read_count = 0
        n_val = max(1, int(round(len(TRAIN_PATIENTS) * self.val_fraction)))
        patients = {
            "train": TRAIN_PATIENTS[: len(TRAIN_PATIENTS) - n_val],
            "val": TRAIN_PATIENTS[len(TRAIN_PATIENTS) - n_val :],
            "test": TEST_PATIENTS,
        }[split]
        token = {"hfo": "hfo", "sleep_awake": "sleep"}.get(task, "interictal")
        self.recording_ids = [recording_id_for(p, token) for p in patients]
        self._patient_of = {
            recording_id_for(p, token): p for p in (*TRAIN_PATIENTS, *TEST_PATIENTS)
        }

    # ---- protocol flags ------------------------------------------------------
    @property
    def pools_all_test_subjects(self) -> bool:
        return True

    @property
    def keep_unlabeled_test_channels(self) -> bool:
        return (
            self.single_channel and self.task == "soz_outcome" and self.split == "test"
        )

    @classmethod
    def num_folds_for_regime(cls, regime: str) -> int:
        if regime != "hold-out-subject":
            raise ValueError("OmniIEEG supports only the 'hold-out-subject' regime.")
        return 1

    # ---- labels --------------------------------------------------------------
    def _patient_index(self, recording_id: str) -> int:
        patient = self._patient_of[recording_id]
        return (*TRAIN_PATIENTS, *TEST_PATIENTS).index(patient)

    def _recording_label(self, recording_id: str) -> int:
        """ictal (0) / interictal (1), or awake (0) / sleep (1), alternating."""
        return self._patient_index(recording_id) % 2

    def channel_count(self) -> int:
        return 1 if self.task == "hfo" else self.n_channels

    def _channel_labels(self, recording_id: str) -> np.ndarray:
        good = np.asarray(self.good_channels, dtype=bool)
        if self.task in ("ictal_interictal", "sleep_awake"):
            return np.where(good, self._recording_label(recording_id), -1).astype(
                np.int64
            )
        if self.task == "soz_outcome":
            # 1 = non-resected channel of a seizure-free patient, 0 = SOZ,
            # -1 = resected but not SOZ (never labelled).
            return np.array([1, 0, 1, -1], dtype=np.int64)
        if self.task == "anatomical":
            # Five lobes over two channel patterns so every class is seen.
            if self._patient_index(recording_id) % 2 == 0:
                return np.array([0, 1, 2, 3], dtype=np.int64)
            return np.array([4, 0, 1, 2], dtype=np.int64)
        raise ValueError(f"No channel labels defined for task={self.task!r}.")

    def get_channel_labels(self, recording_id: str) -> np.ndarray | None:
        if not self.single_channel or self.task == "hfo":
            return None
        return self._channel_labels(recording_id).copy()

    def _included_mask(self, recording_id: str) -> np.ndarray:
        if self.task == "hfo":
            return np.ones(1, dtype=bool)
        good = np.asarray(self.good_channels, dtype=bool)
        if not self.single_channel:
            return good
        labels = self._channel_labels(recording_id)
        if self.keep_unlabeled_test_channels:
            return np.ones(self.n_channels, dtype=bool)
        return labels >= 0

    # ---- windows -------------------------------------------------------------
    def _window_starts(self) -> np.ndarray:
        n = self.windows_per_recording
        if self.split in ("train", "val") and self.task != "hfo":
            n = min(n, self.train_windows_per_recording)
        return self.edge_trim_sec + self.window_sec * np.arange(n, dtype=np.float64)

    def _window_labels(self, recording_id: str) -> np.ndarray:
        starts = self._window_starts()
        if self.task == "hfo":
            return (np.arange(len(starts)) + self._patient_index(recording_id)) % 3
        if self.task in ("ictal_interictal", "sleep_awake") and not self.single_channel:
            return np.full(len(starts), self._recording_label(recording_id))
        if self.single_channel:
            # Channel-labelled tasks: the window label is a placeholder that the
            # single-channel expansion replaces with each channel's label.
            return np.full(len(starts), -1, dtype=np.int64)
        return np.full(len(starts), self._recording_label(recording_id))

    def get_sampling_intervals(self) -> dict[str, Interval]:
        intervals = {}
        for recording_id in self.recording_ids:
            starts = self._window_starts()
            intervals[recording_id] = Interval(
                start=starts,
                end=starts + self.window_sec,
                label=np.asarray(self._window_labels(recording_id), dtype=np.int64),
            )
        return intervals

    def duration_sec(self) -> float:
        return (
            2 * self.edge_trim_sec + self.windows_per_recording * self.window_sec + 0.5
        )

    # ---- signal --------------------------------------------------------------
    def get_recording(self, recording_id: str) -> Data:
        assert recording_id in self.recording_ids
        self.read_count += 1
        n_channels = self.channel_count()
        n_samples = int(round(self.duration_sec() * self.sampling_rate_hz))
        t = np.arange(n_samples) / self.sampling_rate_hz
        rng = np.random.default_rng(self._patient_index(recording_id))
        signal = 0.1 * rng.standard_normal((n_samples, n_channels))
        carrier = np.sin(2 * np.pi * 10 * t)
        if self.task == "hfo":
            for start, label in zip(
                self._window_starts(), self._window_labels(recording_id), strict=True
            ):
                inside = (t >= start) & (t < start + self.window_sec)
                signal[inside, 0] += (1.0 + 2.0 * int(label)) * carrier[inside]
        else:
            labels = self._channel_labels(recording_id)
            for channel, label in enumerate(labels):
                if label >= 0:
                    signal[:, channel] += (1.0 + 2.0 * int(label)) * carrier
        return Data(
            domain="auto",
            seeg_data=RegularTimeSeries(
                sampling_rate=self.sampling_rate_hz,
                data=signal.astype(np.float32),
            ),
        )

    def get_sampling_rate(self, recording_id: str) -> float:
        return self.sampling_rate_hz

    # ---- metadata ------------------------------------------------------------
    def get_channel_metadata(self, recording_id: str) -> dict[str, object]:
        assert recording_id in self.recording_ids
        n = self.channel_count()
        return {
            "ids": np.array([f"{recording_id}/{i}" for i in range(n)]),
            "names": np.array(
                ["hfo_event"]
                if self.task == "hfo"
                else [f"A{i}-A{i + 1}" for i in range(n)]
            ),
            "included_mask": self._included_mask(recording_id),
            "indices": np.arange(n, dtype=np.int64),
            "coordinate_frames": {"omni_none": np.zeros((n, 3), dtype=np.float32)},
        }

    def describe_selection(self) -> dict[str, object]:
        info = {
            "provider": "omni_ieeg",
            "task": self.task,
            "regime": self.regime,
            "split": self.split,
            "test_subject": self.test_subject,
            "test_session": self.test_session,
            "window_sec": self.window_sec,
            "edge_trim_sec": self.edge_trim_sec,
            "split_protocol": self.split_protocol,
            "train_seconds_per_recording": self.train_seconds_per_recording,
            "recording_ids": list(self.recording_ids),
        }
        if self.single_channel:
            info["single_channel"] = True
        if self.train_window_sampling != "tile_subsample":
            info["train_window_sampling"] = self.train_window_sampling
            info["train_windows_per_recording"] = self.train_windows_per_recording
        return info


def install_fake_omni(monkeypatch, cls=FakeOmniIEEG):
    """Make the omni_ieeg dataset entry load ``cls`` instead of torch_brain."""
    from imindbench.utils import pipeline_contracts

    monkeypatch.setitem(
        pipeline_contracts._PROVIDER_SPECS["omni_ieeg"],
        "dataset_class_loader",
        lambda: cls,
    )
    return cls
