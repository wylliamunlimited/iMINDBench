"""Explicit waveform window policies without mutating TorchBrain's time-series API."""

import math

import numpy as np

DEFAULT_WINDOW_SLICING_POLICY = "ceil"


WINDOW_SLICING_POLICIES = ("ceil", "legacy_floor", "exact_count")


def validate_window_slicing_policy(policy) -> None:
    if not isinstance(policy, str) or policy not in WINDOW_SLICING_POLICIES:
        raise ValueError(
            "dataset.window_slicing_policy must be 'ceil', 'legacy_floor' or "
            f"'exact_count', got {policy!r}."
        )


def read_recording_window(
    recording, start: float, end: float, policy: str, *, recording_id=None
) -> np.ndarray:
    """Read only the requested waveform interval, retaining lazy storage slicing."""
    if policy == "ceil":
        # Preserve the current provider/TorchBrain slicing path exactly.
        return np.asarray(recording.slice(start, end).seeg_data.data)
    if policy == "exact_count":
        window_data = np.asarray(recording.slice(start, end).seeg_data.data)
        return _exact_sample_count(recording, start, end, window_data, recording_id)
    if policy != "legacy_floor":
        raise ValueError(f"Unknown window slicing policy: {policy!r}.")

    series = recording.seeg_data
    origin = float(series.domain.start[0])
    domain_end = float(series.domain.end[-1])
    rate = float(series.sampling_rate)

    def floor_boundary(time):
        if time <= origin:
            return origin
        if time > domain_end:
            return domain_end
        index = math.floor((time - origin) * rate)
        return origin + index / rate

    slice_start, slice_end = floor_boundary(start), floor_boundary(end)
    # Snap reconstructed grid points despite cancellation at large time origins.
    # Never apply this tolerance to the original unsnapped historical floor rule.
    eps = max(
        1e-9,
        4 * rate * max(abs(np.spacing(t)) for t in (origin, slice_start, slice_end)),
    )
    window = series.slice(slice_start, slice_end, eps=eps)
    return np.asarray(window.data)


def _exact_sample_count(recording, start, end, window_data, recording_id=None):
    """Give a window exactly round(duration * rate) samples.

    Upstream extractors (e.g. Omni-iEEG's extract_random_samples) take a fixed
    sample COUNT from an integer index (``data[i : i + int(duration * rate)]``),
    but windows travel as float [start, end) intervals and RegularTimeSeries.slice
    resolves those as ``ceil((start - t0) * rate)`` .. ``floor((end - t0) * rate)``.
    Whenever start does not land exactly on the sample grid, that pair of roundings
    drops one sample, and a window running past the recording domain gets one too
    many (slice clamps the end to len + 1). Batches then mix N- and N+1-sample
    windows and variable_channel_collate refuses them.

    Re-slicing with a short right margin keeps slice's own start index -- so the
    window still begins at the first sample at or after ``start``, as upstream does
    -- and taking the first ``n_nominal`` samples reproduces the upstream count
    exactly. Nothing is ever padded: a window that cannot be filled from real
    samples is an error, not a guess.
    """
    rate = float(recording.seeg_data.sampling_rate)
    n_nominal = int(round((end - start) * rate))
    if window_data.shape[0] == n_nominal:
        return window_data
    if window_data.shape[0] < n_nominal:
        # Ask for a few samples more on the right; the left edge is unchanged.
        margin = 4.0 / rate
        widened = np.asarray(recording.slice(start, end + margin).seeg_data.data)
        if widened.shape[0] < n_nominal:
            raise ValueError(
                f"Window [{start}, {end}) of recording {recording_id!r} holds only "
                f"{widened.shape[0]} samples at {rate} Hz, fewer than the "
                f"{n_nominal} its duration requires, even after extending the "
                "right edge. The recording ends inside the window."
            )
        window_data = widened
    return window_data[:n_nominal]
