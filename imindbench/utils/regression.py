"""Regression track: continuous targets and their scores.

Some Miller task sets have no classes. Each window instead has a trajectory: a
continuous target (finger flexion, joystick or cursor position) sampled at
40 Hz inside the window, plus the target's mean over the window. These sets
run with ``dataset.label_mode: regression``.

The rest of the pipeline expects an integer ``y`` per window, and that integer
has to survive the preprocessing chain, the collate function and the fold
caches. So in regression mode the dataset class sets ``y`` to the window's row
number in its recording's stored target table, and the targets themselves are
kept here, in a ``RegressionTargets`` table the fold builder fills once per fold
(``fold["regression_targets"]``). The runners look trajectories up by
(recording id, row).

Scores for one split (``score``):

- traj_r: Pearson r between the predicted and true trajectories of all the
  split's windows, joined end to end in window order. This is the main score.
- mean_r: Pearson r between the predicted and true window means.
- traj_mse and traj_r2: mean squared error and R^2 of the trajectories.
- n_windows and traj_len: the number of windows and samples per window.
"""

from __future__ import annotations

from typing import Any

import numpy as np

REGRESSION_LABEL_MODE = "regression"


class RegressionTargets:
    """Trajectory and window-mean targets of one task, looked up by row.

    ``out_last`` (dataset.regression_target_last_samples) keeps only the last
    ``out_last`` trajectory samples of every window; 0 keeps all of them. The
    sliding-window sets use it so that each window predicts only its newest
    samples, and the test windows joined end to end form one continuous
    trajectory.
    """

    def __init__(self, target: str, *, out_last: int = 0):
        out_last = int(out_last)
        if out_last < 0:
            raise ValueError("regression_target_last_samples must be >= 0.")
        self.target = str(target)
        self.out_last = out_last
        self._traj: dict[str, np.ndarray] = {}
        self._mean: dict[str, np.ndarray] = {}

    def add(self, recording_id: str, *, traj: np.ndarray, mean: np.ndarray) -> None:
        traj = np.asarray(traj, dtype=np.float64)
        mean = np.asarray(mean, dtype=np.float64).reshape(-1)
        if traj.ndim != 2 or len(traj) != len(mean):
            raise ValueError(
                f"Regression targets for '{recording_id}' must be (n, L) and (n,), "
                f"got {traj.shape} and {mean.shape}."
            )
        if self.out_last:
            if traj.shape[1] < self.out_last:
                raise ValueError(
                    f"regression_target_last_samples={self.out_last} is longer than "
                    f"the stored trajectory ({traj.shape[1]} samples) of "
                    f"'{recording_id}'."
                )
            traj = np.ascontiguousarray(traj[:, -self.out_last :])
        self._traj[str(recording_id)] = traj
        self._mean[str(recording_id)] = mean

    @property
    def traj_len(self) -> int:
        lengths = {int(table.shape[1]) for table in self._traj.values()}
        if len(lengths) != 1:
            raise ValueError(
                f"Regression target '{self.target}' has trajectory lengths "
                f"{sorted(lengths)}; one cell must have exactly one window length."
            )
        return lengths.pop()

    def lookup(self, recording_ids, rows) -> tuple[np.ndarray, np.ndarray]:
        """(n, L) trajectories and (n,) window means for (recording id, row) pairs."""
        rows = np.asarray(rows, dtype=np.int64).reshape(-1)
        if isinstance(recording_ids, str):
            recording_ids = [recording_ids] * len(rows)
        recording_ids = [str(rid) for rid in recording_ids]
        if len(recording_ids) != len(rows):
            raise ValueError(
                f"recording_ids/rows length mismatch: {len(recording_ids)} vs "
                f"{len(rows)}."
            )
        traj = np.empty((len(rows), self.traj_len), dtype=np.float64)
        mean = np.empty((len(rows),), dtype=np.float64)
        for i, (rid, row) in enumerate(zip(recording_ids, rows, strict=True)):
            table = self._traj.get(rid)
            if table is None:
                raise KeyError(
                    f"No regression targets for recording '{rid}' "
                    f"(have {sorted(self._traj)})."
                )
            if not 0 <= int(row) < len(table):
                raise IndexError(
                    f"Row {int(row)} is out of range for recording '{rid}' "
                    f"({len(table)} windows)."
                )
            traj[i] = table[int(row)]
            mean[i] = self._mean[rid][int(row)]
        return traj, mean


def _cfg_get(cfg_like: Any, key: str, default: Any = None) -> Any:
    getter = getattr(cfg_like, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(cfg_like, key, default)


def is_regression(cfg_like: Any) -> bool:
    """Whether a full config or a dataset config uses label_mode regression."""
    if cfg_like is None:
        return False
    dataset_cfg = _cfg_get(cfg_like, "dataset")
    source = dataset_cfg if dataset_cfg is not None else cfg_like
    return _cfg_get(source, "label_mode") == REGRESSION_LABEL_MODE


def pearson_r(a, b) -> float:
    """Pearson r; NaN when either side is constant or has fewer than 2 points."""
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.size != b.size:
        raise ValueError(f"pearson_r length mismatch: {a.size} vs {b.size}.")
    finite = np.isfinite(a) & np.isfinite(b)
    a, b = a[finite], b[finite]
    if a.size < 2:
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    if not np.isfinite(denom) or denom <= 0.0:
        return float("nan")
    return float((a * b).sum() / denom)


def score(pred_traj, true_traj, true_mean=None) -> dict[str, float]:
    """Scores of one split; pred_traj and true_traj are (n_windows, L)."""
    pred_traj = np.asarray(pred_traj, dtype=np.float64)
    true_traj = np.asarray(true_traj, dtype=np.float64)
    if pred_traj.shape != true_traj.shape:
        raise ValueError(
            f"Prediction/target shape mismatch: {pred_traj.shape} vs {true_traj.shape}."
        )
    pred_mean = pred_traj.mean(axis=1)
    if true_mean is None:
        true_mean = true_traj.mean(axis=1)
    resid = pred_traj - true_traj
    denom = float(((true_traj - true_traj.mean()) ** 2).sum())
    return {
        "traj_r": pearson_r(pred_traj, true_traj),
        "mean_r": pearson_r(pred_mean, true_mean),
        "traj_mse": float((resid * resid).mean()),
        "traj_r2": (
            float(1.0 - (resid * resid).sum() / denom) if denom > 0 else float("nan")
        ),
        "n_windows": int(pred_traj.shape[0]),
        "traj_len": int(pred_traj.shape[1]),
    }
