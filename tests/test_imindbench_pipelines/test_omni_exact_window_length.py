"""The exact_count window policy: exactly round(duration * rate) samples per window.

Providers hand windows over as float [start, end) intervals and
RegularTimeSeries.slice resolves them as ceil((start - t0) * rate) ..
floor((end - t0) * rate). A start off the sample grid therefore loses one sample, and
a window whose end runs past the stored domain gains one (slice clamps the end index to
len + 1). Batches then mix 30000- and 30001-sample windows and variable_channel_collate
refuses them, which is what killed the first Omni-iEEG single-channel runs. The
upstream extractors take a fixed sample COUNT from an integer index, so that is what
``dataset.window_slicing_policy: exact_count`` restores.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from omegaconf import OmegaConf

from imindbench.utils.window_slicing import read_recording_window

RATE = 100.0
N_CHANNELS = 3
CONF = Path(__file__).resolve().parents[2] / "imindbench/conf/dataset"


class _GridRecording:
    """Reproduces RegularTimeSeries.slice's ceil/floor index arithmetic.

    ``domain_end`` is stored separately from the data length on purpose: real
    recordings carry a domain that can end a hair before the last sample, which is
    exactly how a window ends up one sample too long.
    """

    def __init__(self, n_samples, *, t0=0.0, domain_end=None):
        self._data = np.arange(n_samples * N_CHANNELS, dtype=np.float32).reshape(
            n_samples, N_CHANNELS
        )
        self._t0 = t0
        self._domain_end = t0 + n_samples / RATE if domain_end is None else domain_end
        self.seeg_data = SimpleNamespace(data=self._data, sampling_rate=RATE)

    def slice(self, start, end):
        if start < self._t0:
            start_id = 0
        else:
            start_id = int(np.ceil((start - self._t0) * RATE))
        if end > self._domain_end:
            end_id = len(self._data) + 1
        else:
            end_id = int(np.floor((end - self._t0) * RATE))
        return SimpleNamespace(
            seeg_data=SimpleNamespace(
                data=self._data[start_id:end_id], sampling_rate=RATE
            )
        )


def _run(rec, start, end):
    """The plain ceil read next to the exact_count read of the same window."""
    raw = read_recording_window(rec, start, end, "ceil")
    return raw, read_recording_window(rec, start, end, "exact_count")


def test_start_on_the_grid_is_left_untouched():
    rec = _GridRecording(400)
    raw, fixed = _run(rec, 1.0, 3.0)
    assert raw.shape[0] == fixed.shape[0] == 200
    assert np.array_equal(fixed, raw)


def test_start_off_the_grid_would_lose_one_sample():
    rec = _GridRecording(400)
    raw, fixed = _run(rec, 1.005, 3.005)
    assert raw.shape[0] == 199, "the bug this test guards against is gone"
    assert fixed.shape[0] == 200


def test_recovered_samples_start_where_upstream_starts():
    """The head must stay at the first sample at or after start; only the tail grows."""
    rec = _GridRecording(400)
    _, fixed = _run(rec, 1.005, 3.005)
    expected_start_id = int(np.ceil(1.005 * RATE))  # 101
    assert fixed[0, 0] == pytest.approx(expected_start_id * N_CHANNELS)
    assert fixed[-1, 0] == pytest.approx((expected_start_id + 199) * N_CHANNELS)


def test_window_past_the_domain_end_is_trimmed():
    # 401 samples of data but a domain that ends just before them: slice clamps the end
    # index to len + 1 and returns one sample too many.
    rec = _GridRecording(401, domain_end=3.995)
    raw, fixed = _run(rec, 2.0, 4.0)
    assert raw.shape[0] == 201, "the bug this test guards against is gone"
    assert fixed.shape[0] == 200
    assert np.array_equal(fixed, raw[:200])


def test_unfillable_window_raises_instead_of_padding():
    rec = _GridRecording(150)  # 1.5 s of data for a 2 s window
    with pytest.raises(ValueError, match="rec-1.*fewer than the 200"):
        read_recording_window(rec, 0.0, 2.0, "exact_count", recording_id="rec-1")


@pytest.mark.parametrize("window_sec", [2.0, 6.0])
def test_every_start_offset_yields_the_nominal_count(window_sec):
    rec = _GridRecording(1200)
    nominal = int(round(window_sec * RATE))
    for offset in np.linspace(0.0, 1.0 / RATE, 11):
        start = 1.0 + float(offset)
        _, fixed = _run(rec, start, start + window_sec)
        assert fixed.shape[0] == nominal


@pytest.mark.parametrize("name", ["omni_ieeg", "omni_ieeg_single_channel"])
def test_the_omni_datasets_select_the_policy(name):
    cfg = OmegaConf.load(CONF / f"{name}.yaml")
    assert cfg.window_slicing_policy == "exact_count"
