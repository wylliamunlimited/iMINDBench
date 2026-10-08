"""A small fake of torch_brain.datasets.MillerECoG2019 for tests.

The pinned torch_brain does not ship MillerECoG2019 yet, so tests install this
class as the millerecog2019 dataset instead. It copies the public API of the
real class (brainsets-wylliam-fork, branch miller-ecog-modularize,
brainsets/datasets/MillerECoG2019.py):

- recording ids use letter codes, for example ``sub-bp_set-motor_basic``
- the module defines ``SETS``, ``SUBJECT_CODES``, ``_from_recording_id``,
  ``subject_number_for`` and ``session_number_for``
- ``get_channel_metadata`` returns the ``talairach`` and ``native_mm``
  coordinate frames, plus ``mni152`` and ``mni152_strict`` when the build has
  positions; ``mni152_strict`` is NaN for channels whose position quality is "C"
- ``num_folds_for_regime("within-session")`` returns 2

The signal is synthetic: channel 0 carries a sine wave whose amplitude
depends on the window's label, so a simple model can separate the classes.
"""

from __future__ import annotations

import re

import numpy as np
from torch_brain.data import Data, Interval, RegularTimeSeries

# The same order as the real tables, so test_subject / test_session numbers
# mean the same thing as in the real dataset.
SETS = (
    "motor_basic",
    "imagery_basic",
    "imagery_feedback",
    "fingerflex",
    "gestures",
    "speech_basic",
    "speech_lists",
    "visual_search",
    "faces_basic",
    "faces_noise",
    "memory_nback",
    "faces_localizer",
    "fingerflex_reg_w1000",
    "fingerflex_reg_w500",
    "fingerflex_reg_w250",
    "joystick_track_w1000",
    "joystick_track_w500",
    "joystick_track_w250",
    "mouse_track_w1000",
    "mouse_track_w500",
    "mouse_track_w250",
    "fingerflex_reg_w1000h50",
    "joystick_track_w1000h50",
    "mouse_track_w1000h50",
    "fingerflex_reg_w1000h25b",
)
SUBJECT_CODES = (
    "al", "ap", "bp", "ca", "cc", "de", "fp", "gc", "gf", "ha",
    "hh", "hl", "ht", "in", "ja", "jc", "jf", "jm", "jp", "jt",
    "mv", "rh", "rn", "rr", "ug", "wc", "wm", "ww", "zt",
)  # fmt: skip
TALAIRACH_SETS = {"motor_basic", "imagery_basic", "imagery_feedback", "memory_nback"}

# The same pattern as the real module. It does not allow digits, so it
# rejects the regression task sets; iMINDBench must not rely on it.
_RECORDING_ID_RE = re.compile(r"^sub-(?P<subject>[a-z]{2})_set-(?P<dset>[a-z_]+)$")


def _from_recording_id(recording_id: str) -> tuple[str, str]:
    match = _RECORDING_ID_RE.match(recording_id)
    if match is None:
        raise ValueError(f"Invalid MillerECoG2019 recording_id '{recording_id}'.")
    return match.group("subject"), match.group("dset")


def subject_number_for(code: str) -> int:
    return SUBJECT_CODES.index(code) + 1


def session_number_for(dset: str) -> int:
    return SETS.index(dset) + 1


def recording_id_for(subject: str, dset: str) -> str:
    return f"sub-{subject}_set-{dset}"


class FakeMillerECoG2019:
    """Stands in for MillerECoG2019 in benchmark (split-selection) mode."""

    sampling_rate_hz = 100.0
    n_channels = 4
    windows_per_class = 12
    window_sec = 0.5
    # Set to True by tests that need the build with MNI152 positions.
    with_positions = False

    def __init__(
        self,
        root=None,
        recording_ids=None,
        transform=None,
        *,
        subset_tier=None,
        test_subject=None,
        test_session=None,
        split=None,
        label_mode=None,
        task=None,
        regime=None,
        fold=None,
        uniquify_channel_ids_with_subject=True,
        uniquify_channel_ids_with_session=False,
        dirname="miller_ecog_library_2019",
        **kwargs,
    ):
        if task is None:
            raise ValueError("task is required in benchmark mode.")
        if regime != "within-session":
            raise ValueError(f"Unsupported regime {regime!r}.")
        if fold not in (0, 1):
            raise ValueError(f"fold must be 0 or 1, got {fold!r}.")
        self.root = root
        self.dirname = dirname
        self.subset_tier = subset_tier
        self.label_mode = label_mode or "binary"
        self.task = task
        self.regime = regime
        self.fold = int(fold)
        self.split = split
        subject_code = SUBJECT_CODES[int(test_subject) - 1]
        self.task_set = SETS[int(test_session) - 1]
        self.recording_ids = [recording_id_for(subject_code, self.task_set)]
        self.subject_code = subject_code
        self.read_count = 0

    @classmethod
    def num_folds_for_regime(cls, regime: str) -> int:
        if regime != "within-session":
            raise ValueError(f"Unsupported regime {regime!r}.")
        return 2

    # ---- windows -------------------------------------------------------------
    def _all_windows(self) -> tuple[np.ndarray, np.ndarray]:
        n = 2 * self.windows_per_class
        starts = 0.1 + np.arange(n, dtype=np.float64) * (self.window_sec + 0.1)
        labels = np.arange(n, dtype=np.int64) % 2
        return starts, labels

    def _split_positions(self) -> np.ndarray:
        """Two chronological folds: the test half, its first half as val."""
        n = 2 * self.windows_per_class
        first, second = np.arange(n // 2), np.arange(n // 2, n)
        held_out = second if self.fold == 0 else first
        train = first if self.fold == 0 else second
        val, test = held_out[: len(held_out) // 2], held_out[len(held_out) // 2 :]
        return {"train": train, "val": val, "test": test}[self.split]

    def get_sampling_intervals(self) -> dict[str, Interval]:
        starts, labels = self._all_windows()
        keep = self._split_positions()
        return {
            self.recording_ids[0]: Interval(
                start=starts[keep],
                end=starts[keep] + self.window_sec,
                label=labels[keep],
            )
        }

    # ---- signal --------------------------------------------------------------
    def get_recording(self, recording_id: str) -> Data:
        assert recording_id in self.recording_ids
        self.read_count += 1
        starts, labels = self._all_windows()
        duration = float(starts[-1] + self.window_sec + 0.5)
        n_samples = int(round(duration * self.sampling_rate_hz))
        t = np.arange(n_samples) / self.sampling_rate_hz
        rng = np.random.default_rng(subject_number_for(self.subject_code))
        signal = 0.1 * rng.standard_normal((n_samples, self.n_channels))
        for start, label in zip(starts, labels, strict=True):
            inside = (t >= start) & (t < start + self.window_sec)
            signal[inside, 0] += (1.0 + 2.0 * label) * np.sin(
                2 * np.pi * 10 * t[inside]
            )
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
        n = self.n_channels
        native = np.stack(
            [np.linspace(-40.0, 40.0, n), np.full(n, -20.0), np.full(n, 30.0)], axis=1
        ).astype(np.float32)
        talairach = native.copy()
        if self.task_set not in TALAIRACH_SETS:
            talairach[:] = np.nan
        frames = {"talairach": talairach, "native_mm": native}
        if self.with_positions:
            mni = native + np.float32(1.0)
            # The last channel has a low-quality position (tier "C").
            quality = np.array(["A"] * (n - 1) + ["C"])
            strict = mni.copy()
            strict[quality == "C"] = np.nan
            frames["mni152"] = mni
            frames["mni152_strict"] = strict
        ids = np.array([f"{recording_id}/ch{i}" for i in range(n)])
        return {
            "ids": ids,
            "names": np.array([f"G{i + 1}" for i in range(n)]),
            "included_mask": np.ones(n, dtype=bool),
            "coordinate_frames": frames,
            "indices": np.arange(n, dtype=int),
            "label_destrieux": np.array(["ctx_lh_G_precentral"] * n),
        }

    def describe_selection(self) -> dict[str, object]:
        return {
            "provider": "millerecog2019",
            "active_recording_ids": list(self.recording_ids),
            "task": self.task,
            "fold": self.fold,
            "split": self.split,
        }


def install_fake_miller(monkeypatch, cls=FakeMillerECoG2019):
    """Make the millerecog2019 dataset entry load ``cls`` instead of torch_brain."""
    from imindbench.utils import pipeline_contracts

    monkeypatch.setitem(
        pipeline_contracts._PROVIDER_SPECS["millerecog2019"],
        "dataset_class_loader",
        lambda: cls,
    )
    return cls
