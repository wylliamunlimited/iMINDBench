"""Single-channel plumbing: split expansion, collate, class counts, samplers.

Adapted from torch_brain's neuroprobe_eval tests of the Omni-iEEG harness.
"""

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from imindbench.utils.collate import variable_channel_collate
from imindbench.utils.data_adapter import (
    SingleChannelSplitDataset,
    _maybe_expand_single_channel_splits,
)
from imindbench.utils.fold_helpers import (
    build_balanced_train_sampler,
    build_torch_split_loaders,
    collect_class_counts_from_loader,
)


class _MaterializedSplit(torch.utils.data.Dataset):
    """Stand-in for a materialized WindowedNeuroprobeSplitDataset."""

    def __init__(self, samples, split="test"):
        self._samples = list(samples)
        self.split = split

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx):
        return self._samples[idx]

    def is_materialized(self):
        return True

    def get_split_summary(self):
        return {"split": self.split, "n_samples": len(self._samples)}


class _Provider:
    def __init__(self, labels, *, single_channel=True, keep_unlabeled=False):
        self._labels = labels
        self.single_channel = single_channel
        self.keep_unlabeled_test_channels = keep_unlabeled

    def get_channel_labels(self, recording_id):
        labels = self._labels.get(recording_id)
        return None if labels is None else np.asarray(labels, dtype=np.int64)

    def get_channel_metadata(self, recording_id):
        n = len(self._labels[recording_id])
        return {"ids": np.asarray([f"{recording_id}/{i}" for i in range(n)])}


def _window(recording_id, n_channels, y=0, split="test", sample_idx=0, start=1.0):
    return {
        "x": np.arange(n_channels * 4, dtype=np.float32).reshape(n_channels, 4),
        "y": int(y),
        "channel_ids": [f"{recording_id}/{i}" for i in range(n_channels)],
        "channel_names": [f"A{i}" for i in range(n_channels)],
        "channel_coords": np.zeros((n_channels, 3), dtype=np.float32),
        "coordinate_profile": "omni_none",
        "seq_id": np.zeros((n_channels,), dtype=np.int64),
        "channel_indices": np.arange(n_channels, dtype=np.int64),
        "recording_id": recording_id,
        "split": split,
        "sample_idx": int(sample_idx),
        "window_start_sec": float(start),
        "window_end_sec": float(start) + 1.0,
    }


def test_expansion_yields_one_sample_per_labelled_channel():
    base = _MaterializedSplit(
        [_window("rec", 3, sample_idx=0), _window("rec", 3, sample_idx=1)]
    )
    provider = _Provider({"rec": [1, -1, 0]})
    view = SingleChannelSplitDataset(base, provider=provider)
    # Two windows x (3 channels - 1 unlabelled) = 4 samples.
    assert len(view) == 4
    assert view.label_counts() == {0: 2, 1: 2}
    sample = view[0]
    assert sample["x"].shape == (1, 4)
    np.testing.assert_array_equal(sample["x"][0], base[0]["x"][0])
    assert sample["y"] == 1
    assert sample["channel_ids"] == ["rec/0"]
    assert sample["channel_names"] == ["A0"]
    assert sample["channel_indices"].tolist() == [0]
    assert sample["channel_coords"].shape == (1, 3)
    # The second sample of the first window is the channel labelled 0.
    assert view[1]["y"] == 0
    np.testing.assert_array_equal(view[1]["x"][0], base[0]["x"][2])


def test_expansion_can_keep_unlabelled_channels():
    base = _MaterializedSplit([_window("rec", 3)])
    provider = _Provider({"rec": [1, -1, 0]}, keep_unlabeled=True)
    view = SingleChannelSplitDataset(base, provider=provider, keep_unlabeled=True)
    assert len(view) == 3
    assert view.label_counts() == {-1: 1, 0: 1, 1: 1}
    assert view.sample_labels().tolist() == [1, -1, 0]


def test_expansion_uses_the_window_label_when_the_provider_has_none():
    base = _MaterializedSplit([_window("rec", 2, y=2)])
    provider = _Provider({"rec": None})
    view = SingleChannelSplitDataset(base, provider=provider)
    assert [view[i]["y"] for i in range(len(view))] == [2, 2]


def test_expansion_falls_back_to_channel_ids_without_channel_indices():
    window = _window("rec", 3)
    window.pop("channel_indices")
    view = SingleChannelSplitDataset(
        _MaterializedSplit([window]), provider=_Provider({"rec": [5, -1, 6]})
    )
    assert [view[i]["y"] for i in range(len(view))] == [5, 6]


def test_only_single_channel_providers_are_expanded():
    base = _MaterializedSplit([_window("rec", 3)])
    splits = {"train": base, "val": base, "test": base}
    multi = _maybe_expand_single_channel_splits(
        splits, fold_idx=0
    )  # provider attribute missing -> untouched
    assert multi["train"] is base
    base.provider = _Provider({"rec": [1, 0, 1]}, single_channel=False)
    assert _maybe_expand_single_channel_splits(splits, fold_idx=0)["train"] is base
    base.provider = _Provider({"rec": [1, 0, 1]})
    expanded = _maybe_expand_single_channel_splits(splits, fold_idx=0)
    assert isinstance(expanded["test"], SingleChannelSplitDataset)
    assert len(expanded["test"]) == 3


def test_collate_passes_channel_names_when_every_sample_has_them():
    plain = [_window("rec", 2, split="train", sample_idx=i) for i in range(2)]
    assert variable_channel_collate(plain)["channel_names"] == [["A0", "A1"]] * 2
    nameless = [dict(sample, channel_names=None) for sample in plain]
    assert "channel_names" not in variable_channel_collate(nameless)

    view = SingleChannelSplitDataset(
        _MaterializedSplit(plain, split="train"), provider=_Provider({"rec": [0, 1]})
    )
    batch = variable_channel_collate([view[i] for i in range(len(view))])
    assert batch["channel_names"] == [["A0"], ["A1"], ["A0"], ["A1"]]
    assert batch["x"].shape == (4, 1, 4)
    assert batch["y"].tolist() == [0, 1, 0, 1]


def _counts_fold(labels, split="train"):
    samples = [
        _window("rec", 1, y=label, split=split, sample_idx=i)
        for i, label in enumerate(labels)
    ]
    return _MaterializedSplit(samples, split=split)


def test_class_counts_skip_the_ignore_label():
    dataset = _counts_fold([1, 1, 0, -1, -1])
    assert collect_class_counts_from_loader(dataset) == {-1: 2, 0: 1, 1: 2}
    assert collect_class_counts_from_loader(dataset, -1) == {0: 1, 1: 2}


def test_balanced_sampler_equalizes_classes_and_skips_ignored():
    labels = [1] * 90 + [0] * 10 + [-1] * 20
    dataset = _counts_fold(labels)
    generator = torch.Generator().manual_seed(0)
    sampler = build_balanced_train_sampler(
        dataset, generator=generator, ignore_label=-1
    )
    drawn = np.asarray([labels[i] for i in list(sampler)])
    assert len(drawn) == len(labels)
    assert (drawn == -1).sum() == 0
    # Both classes appear about equally often (inverse-frequency weights).
    share = (drawn == 1).mean()
    assert 0.4 < share < 0.6


def _loader_cfg(**runner):
    return OmegaConf.create(
        {
            "model": {
                "name": "cnn",
                "requires_aligned_channels": False,
                "batch_size": 4,
            },
            "dataset": {"provider": "omni_ieeg", "regime": "hold-out-subject"},
            "runner": {"num_workers": 0, "pin_memory": False, **runner},
        }
    )


def test_train_loader_uses_a_balanced_sampler_only_when_asked():
    split = _counts_fold([1] * 8 + [0] * 2)
    fold = {
        "fold_idx": 0,
        "train_split": split,
        "val_split": _counts_fold([0, 1], split="val"),
        "test_split": _counts_fold([0, 1], split="test"),
        "preprocess_state": None,
        "metadata": {},
    }
    train_loader, _, _ = build_torch_split_loaders(fold, _loader_cfg(), seed=0)
    assert train_loader.sampler is not None
    assert isinstance(train_loader.sampler, torch.utils.data.RandomSampler)

    train_loader, _, _ = build_torch_split_loaders(
        fold, _loader_cfg(train_sampler="balanced"), seed=0
    )
    assert isinstance(train_loader.sampler, torch.utils.data.WeightedRandomSampler)

    with pytest.raises(ValueError, match="train_sampler"):
        build_torch_split_loaders(fold, _loader_cfg(train_sampler="nope"), seed=0)
