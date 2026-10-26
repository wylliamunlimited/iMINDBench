"""Exercise explicit window policies against real eager and lazy recordings."""

import math
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from torch_brain.data import Data, LazyRegularTimeSeries, RegularTimeSeries

from imindbench.preprocessors.context_window_preprocessor import (
    ContextWindowPreprocessor,
)
from imindbench.utils.window_slicing import (
    read_recording_window,
    validate_window_slicing_policy,
)


@pytest.fixture(params=[False, True], ids=["eager", "lazy"])
def recording(request, tmp_path):
    values = np.stack([np.arange(10), np.arange(10) + 100], axis=1)
    series = RegularTimeSeries(data=values, sampling_rate=10.0)
    if not request.param:
        yield Data(seeg_data=series, domain=series.domain)
        return
    with h5py.File(tmp_path / "recording.h5", "w") as handle:
        series.to_hdf5(handle)
        lazy = LazyRegularTimeSeries.from_hdf5(handle)
        yield Data(seeg_data=lazy, domain=lazy.domain)
        assert isinstance(lazy.__dict__["data"], h5py.Dataset)


@pytest.mark.parametrize("policy", ["ceil", "legacy_floor"])
@pytest.mark.parametrize(
    ("start", "end", "ceil_samples", "floor_samples"),
    [
        (0.15, 0.45, [2, 3, 4], [1, 2, 3]),
        (0.2, 0.5, [2, 3, 4], [2, 3, 4]),
        (np.nextafter(0.3, 0), np.nextafter(0.6, 0), [3, 4, 5], [2, 3, 4]),
        (-1.0, 0.25, [0, 1, 2], [0, 1]),
        (0.85, 3.0, [9], [8, 9]),
        (1.0, 2.0, [], []),
    ],
)
def test_window_policy_samples(
    recording, policy, start, end, ceil_samples, floor_samples
):
    expected = np.asarray(ceil_samples if policy == "ceil" else floor_samples)
    result = read_recording_window(recording, start, end, policy)
    np.testing.assert_array_equal(result, np.stack([expected, expected + 100], axis=1))


@pytest.mark.parametrize("origin,rate", [(1234.125, 1000), (1e8 + 0.123, 1000)])
def test_floor_reconstructed_boundaries_preserve_nonzero_domain(origin, rate):
    series = RegularTimeSeries(
        data=np.arange(20)[:, None], sampling_rate=rate, domain_start=origin
    )
    recording = Data(seeg_data=series, domain=series.domain)
    result = read_recording_window(
        recording, origin + 1.5 / rate, origin + 4.5 / rate, "legacy_floor"
    )
    np.testing.assert_array_equal(result[:, 0], [1, 2, 3])


@pytest.mark.parametrize(
    "policy,expected", [("ceil", [4, 5, 6]), ("legacy_floor", [3, 4, 5])]
)
def test_nested_lazy_window_preserves_file_offset(tmp_path, policy, expected):
    series = RegularTimeSeries(
        data=np.arange(20)[:, None], sampling_rate=10.0, domain_start=0.25
    )
    with h5py.File(tmp_path / "nested.h5", "w") as handle:
        series.to_hdf5(handle)
        lazy = LazyRegularTimeSeries.from_hdf5(handle)
        subset = lazy.slice(0.45, 1.65, reset_origin=False)
        recording = Data(seeg_data=subset, domain=subset.domain)
        result = read_recording_window(recording, 0.6, 0.9, policy)
        np.testing.assert_array_equal(result[:, 0], expected)
        assert isinstance(lazy.__dict__["data"], h5py.Dataset)
        assert isinstance(subset.__dict__["data"], h5py.Dataset)


@pytest.mark.parametrize("lazy", [False, True], ids=["eager", "lazy"])
@pytest.mark.parametrize("policy", ["ceil", "legacy_floor"])
def test_window_policies_preserve_gap_trimming(tmp_path, lazy, policy):
    series = RegularTimeSeries.from_gappy_timeseries(
        [0.0, 0.1, 0.3, 0.4],
        sampling_rate=10.0,
        data=np.array([0.0, 1.0, 3.0, 4.0])[:, None],
    )
    with h5py.File(tmp_path / "gappy.h5", "w") as handle:
        series.to_hdf5(handle)
        if lazy:
            series = LazyRegularTimeSeries.from_hdf5(handle)
        recording = Data(seeg_data=series, domain=series.domain)
        expected = [3, 4] if policy == "ceil" else [1, np.nan, 3]
        np.testing.assert_array_equal(
            read_recording_window(recording, 0.15, 0.45, policy)[:, 0], expected
        )
        expected = [3, 4] if policy == "ceil" else [3]
        np.testing.assert_array_equal(
            read_recording_window(recording, 0.25, 0.45, policy)[:, 0], expected
        )
        assert read_recording_window(recording, 0.21, 0.29, policy).shape == (0, 1)
        if lazy:
            assert isinstance(series.__dict__["data"], h5py.Dataset)


def _historical_floor_index(self, time, eps=1e-9):
    origin, end = float(self.domain.start[0]), float(self.domain.end[-1])
    if time <= origin:
        return 0, origin
    if time > end:
        return len(self), end
    index = math.floor((time - origin) * float(self.sampling_rate))
    return index, origin + index / float(self.sampling_rate)


def test_context_window_matches_isolated_historical_patch(monkeypatch):
    series = RegularTimeSeries(data=np.arange(100)[:, None], sampling_rate=100.0)
    recording = Data(seeg_data=series, domain=series.domain)
    provider = SimpleNamespace(get_recording=lambda recording_id: recording)
    stage = ContextWindowPreprocessor(
        {"context_window_sec": 0.2, "sampling_rate": 100.0}
    )
    stage.set_fold_context({"recording_provider": provider})
    sample = {
        "x": np.arange(36, 41, dtype=np.float32)[None, :],
        "recording_id": "recording",
        "window_start_sec": 0.365,
        "window_end_sec": 0.415,
        "channel_indices": np.array([0]),
        "window_slicing_policy": "ceil",
    }
    # The context begins at 0.29 s, whose unsnapped product is below sample 29.
    with monkeypatch.context() as historical:
        historical.setattr(RegularTimeSeries, "_time_to_idx", _historical_floor_index)
        expected = stage.transform_samples([sample])[0]
    actual = stage.transform_samples(
        [{**sample, "window_slicing_policy": "legacy_floor"}]
    )[0]
    np.testing.assert_array_equal(expected["x"], np.arange(28, 48)[None, :])
    np.testing.assert_array_equal(actual["x"], expected["x"])
    for key in (
        "target_start_sample_in_context",
        "target_num_samples",
        "context_pad_left_samples",
        "context_pad_right_samples",
    ):
        assert actual[key] == expected[key]


@pytest.mark.parametrize("policy", ["ceil", "legacy_floor", "exact_count"])
def test_supported_window_policies(policy):
    validate_window_slicing_policy(policy)


@pytest.mark.parametrize("policy", [None, 1, {}, "floor", "current"])
def test_invalid_window_policies_fail_at_boundary(policy):
    with pytest.raises((TypeError, ValueError), match="window_slicing_policy"):
        validate_window_slicing_policy(policy)
