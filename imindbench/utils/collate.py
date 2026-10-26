"""Variable-channel collate and loader helpers for Neuroprobe torch evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def variable_channel_collate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Pad variable-channel samples into one batch dict.

    The batch keeps channels left-aligned and uses ``channel_mask`` to mark the
    real extent of each sample so models can distinguish data from padding.
    """
    if not samples:
        raise ValueError("Cannot collate an empty sample list.")

    split = samples[0]["split"]
    feature_shape = tuple(np.asarray(samples[0]["x"]).shape[1:])
    if not feature_shape:
        raise ValueError(
            "sample['x'] must include at least one non-channel feature dimension."
        )

    for sample in samples:
        sample_split = sample["split"]
        if sample_split != split:
            raise ValueError(
                "All samples in a batch must share the same split, "
                f"got '{split}' and '{sample_split}'."
            )
        sample_feature_shape = tuple(np.asarray(sample["x"]).shape[1:])
        if sample_feature_shape != feature_shape:
            raise ValueError(
                "All samples in a batch must have the same per-channel feature shape, "
                f"got {feature_shape} and {sample_feature_shape}."
            )

    batch_size = len(samples)
    n_channels = [np.asarray(sample["x"]).shape[0] for sample in samples]
    max_channels = max(n_channels)
    # Batch tensors are padded to max_channels; `channel_mask` preserves the
    # real channel extents per sample.

    x = torch.zeros((batch_size, max_channels, *feature_shape), dtype=torch.float32)
    y = torch.zeros((batch_size,), dtype=torch.long)
    channel_mask = torch.zeros((batch_size, max_channels), dtype=torch.bool)
    n_channels_tensor = torch.as_tensor(n_channels, dtype=torch.long)

    coordinate_profile = samples[0].get("coordinate_profile", "popt_lip")
    if any(
        sample.get("coordinate_profile", "popt_lip") != coordinate_profile
        for sample in samples
    ):
        raise ValueError("coordinate_profile must match for all samples in a batch.")

    coords_present = [sample["channel_coords"] is not None for sample in samples]
    if any(coords_present) and not all(coords_present):
        raise ValueError(
            "channel_coords must be either present for all samples in a batch or "
            "None for all samples."
        )
    coords_tensor = (
        torch.zeros((batch_size, max_channels, 3), dtype=torch.float32)
        if all(coords_present)
        else None
    )

    seq_present = [sample["seq_id"] is not None for sample in samples]
    if any(seq_present) and not all(seq_present):
        raise ValueError(
            "seq_id must be either present for all samples in a batch or None for "
            "all samples."
        )
    seq_tensor = (
        torch.zeros((batch_size, max_channels), dtype=torch.long)
        if all(seq_present)
        else None
    )

    brain_areas_present = [sample.get("brain_areas") is not None for sample in samples]
    if any(brain_areas_present) and not all(brain_areas_present):
        raise ValueError(
            "brain_areas must be either present for all samples in a batch or None "
            "for all samples."
        )
    batched_brain_areas: list[list[str]] | None = (
        [] if all(brain_areas_present) else None
    )

    # Channel names (when every sample carries them) label per-channel test
    # predictions; batches without them keep the historical key set.
    names_present = all(sample.get("channel_names") is not None for sample in samples)
    channel_names: list[list[str]] | None = [] if names_present else None
    channel_ids: list[list[str]] = []
    recording_ids: list[str] = []
    sample_indices: list[int] = []
    window_start_sec: list[float] = []
    window_end_sec: list[float] = []
    for idx, sample in enumerate(samples):
        sample_x = np.asarray(sample["x"], dtype=np.float32)
        sample_y = sample["y"]
        sample_n = sample_x.shape[0]

        # Left-aligned copy keeps sample channel order intact.
        x[idx, :sample_n] = torch.as_tensor(sample_x, dtype=torch.float32)
        y[idx] = sample_y
        channel_mask[idx, :sample_n] = True

        if coords_tensor is not None:
            coords = np.asarray(sample["channel_coords"], dtype=np.float32)
            coords_tensor[idx, :sample_n] = torch.as_tensor(coords, dtype=torch.float32)

        if seq_tensor is not None:
            seq = np.asarray(sample["seq_id"], dtype=np.int64)
            seq_tensor[idx, :sample_n] = torch.as_tensor(seq, dtype=torch.long)

        if batched_brain_areas is not None:
            brain_areas = list(sample.get("brain_areas"))
            if len(brain_areas) != sample_n:
                raise ValueError(
                    "brain_areas length must match sample['x'].shape[0], got "
                    f"{len(brain_areas)} vs {sample_n}."
                )
            batched_brain_areas.append(brain_areas)

        channel_ids.append(list(sample["channel_ids"]))
        if channel_names is not None:
            channel_names.append(list(sample["channel_names"]))
        recording_ids.append(sample["recording_id"])
        sample_indices.append(int(sample["sample_idx"]))
        window_start_sec.append(float(sample["window_start_sec"]))
        window_end_sec.append(float(sample["window_end_sec"]))

    batch = {
        "x": x,
        "y": y,
        "channel_mask": channel_mask,
        "channel_coords": coords_tensor,
        "coordinate_profile": coordinate_profile,
        "seq_id": seq_tensor,
        "brain_areas": batched_brain_areas,
        "channel_ids": channel_ids,
        "n_channels": n_channels_tensor,
        "recording_ids": recording_ids,
        "sample_indices": sample_indices,
        "window_start_sec": window_start_sec,
        "window_end_sec": window_end_sec,
        "split": split,
    }
    if channel_names is not None:
        batch["channel_names"] = channel_names
    return batch
