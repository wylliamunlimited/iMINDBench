"""The Omni split cache must not serve windows preprocessed under another policy.

A shared preprocessed split cache is keyed by a hash of the dataset + preprocessor
identity. The window-length rule is ``dataset.window_slicing_policy`` (the Omni
datasets select ``exact_count``), which is already part of that identity, so entries
built with ``ceil`` (mixed 30000/30001-sample windows that variable_channel_collate
rejects) live under another key. No provider-specific marker and no global cache
version bump is needed; other providers' (expensive, shared) entries stay valid.
"""

from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from imindbench.utils import data_adapter as vca


def _dataset_cfg(provider):
    return OmegaConf.create(
        {
            "provider": provider,
            "root": "/tmp/root",
            "dirname": provider,
            "subset_tier": "full",
            "label_mode": "binary",
            "task": "ictal_interictal" if provider == "omni_ieeg" else "speech",
            "regime": (
                "hold-out-subject" if provider == "omni_ieeg" else "within-session"
            ),
            "test_subject": "all" if provider == "omni_ieeg" else 43,
            "test_session": "all" if provider == "omni_ieeg" else 1,
            "uniquify_channel_ids_with_subject": True,
            "uniquify_channel_ids_with_session": True,
        }
    )


def _identity(provider, policy="exact_count", cfg=None, preprocessor="raw"):
    splits = {
        name: SimpleNamespace(
            get_split_summary=lambda: {"n": 1}, window_slicing_policy=policy
        )
        for name in ("train", "val", "test")
    }
    return vca._build_preprocessed_split_cache_identity(
        dataset_cfg=cfg if cfg is not None else _dataset_cfg(provider),
        split_datasets=splits,
        preprocessor_identity={"name": preprocessor},
        fold_idx=0,
        needs_pool=False,
        brain_area_key=None,
    )


def test_the_policy_is_part_of_the_identity_and_ids_keep_their_type():
    identity = _identity("omni_ieeg")
    assert identity["window_slicing_policy"] == "exact_count"
    # Pooled ids stay strings; other providers' numeric ids stay ints.
    assert identity["test_subject"] == "all"
    assert _identity("kelesbyd2024", policy="ceil")["test_subject"] == 43


def test_provider_options_are_part_of_the_identity():
    cfg = _dataset_cfg("omni_ieeg")
    cfg.provider_options = {"window_sec": 60.0, "single_channel": True}
    identity = _identity("omni_ieeg", cfg=cfg)
    assert identity["provider_options"] == {"single_channel": True, "window_sec": 60.0}
    assert _identity("omni_ieeg")["provider_options"] is None


@pytest.mark.parametrize("provider", ["neuroprobev2", "kelesbyd2024"])
def test_no_provider_specific_marker(provider):
    """Their shared cache entries must stay valid."""
    assert not any(key.startswith("omni") for key in _identity(provider, "ceil"))


def test_the_policy_changes_the_cache_path(tmp_path):
    exact = _identity("omni_ieeg", "exact_count")
    ceil = _identity("omni_ieeg", "ceil")
    assert vca._preprocessed_split_cache_path(
        cache_dir=tmp_path, identity=exact
    ) != vca._preprocessed_split_cache_path(cache_dir=tmp_path, identity=ceil)


def test_global_cache_version_was_not_bumped():
    """Bumping it would invalidate every provider's entries, not just omni's."""
    assert vca._PREPROCESSED_SPLIT_CACHE_VERSION == 2
