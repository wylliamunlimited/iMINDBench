"""The Omni split cache must not serve windows preprocessed before the length fix.

A shared preprocessed split cache is keyed by a hash of the dataset + preprocessor
identity -- nothing that changes when the window-building code changes. Entries
written before _exact_omni_window hold the mixed 30000/30001-sample windows that
variable_channel_collate rejects, so the omni identity carries a marker that
separates them. It must stay scoped to omni_ieeg: bumping the global cache version
instead would throw away every other provider's (expensive, shared) entries.
"""

from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from imindbench.utils import data_adapter as vca

MARKER_KEY = "omni_window_length"


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


def _identity(provider):
    splits = {
        name: SimpleNamespace(
            get_split_summary=lambda: {"n": 1}, window_slicing_policy="ceil"
        )
        for name in ("train", "val", "test")
    }
    return vca._build_preprocessed_split_cache_identity(
        dataset_cfg=_dataset_cfg(provider),
        split_datasets=splits,
        preprocessor_identity={"name": "omni_diverstyle_HPF_noSTD_w60s"},
        fold_idx=0,
        needs_pool=False,
        brain_area_key=None,
    )


def test_omni_identity_carries_the_window_length_marker():
    identity = _identity("omni_ieeg")
    assert identity[MARKER_KEY] == "exact_nominal_v1"
    # Pooled ids stay strings; other providers' numeric ids stay ints.
    assert identity["test_subject"] == "all"
    assert _identity("kelesbyd2024")["test_subject"] == 43


def test_provider_options_are_part_of_the_identity():
    cfg = _dataset_cfg("omni_ieeg")
    cfg.provider_options = {"window_sec": 60.0, "single_channel": True}
    splits = {
        name: SimpleNamespace(
            get_split_summary=lambda: {"n": 1}, window_slicing_policy="ceil"
        )
        for name in ("train", "val", "test")
    }
    identity = vca._build_preprocessed_split_cache_identity(
        dataset_cfg=cfg,
        split_datasets=splits,
        preprocessor_identity={"name": "raw"},
        fold_idx=0,
        needs_pool=False,
        brain_area_key=None,
    )
    assert identity["provider_options"] == {"single_channel": True, "window_sec": 60.0}
    assert _identity("omni_ieeg")["provider_options"] in (None, {})


@pytest.mark.parametrize("provider", ["neuroprobev2", "kelesbyd2024"])
def test_other_providers_keep_their_identity(provider):
    """Their shared cache entries must stay valid."""
    assert MARKER_KEY not in _identity(provider)


def test_marker_changes_the_omni_cache_path(tmp_path):
    identity = _identity("omni_ieeg")
    stale = dict(identity)
    stale.pop(MARKER_KEY)
    assert vca._preprocessed_split_cache_path(
        cache_dir=tmp_path, identity=identity
    ) != vca._preprocessed_split_cache_path(cache_dir=tmp_path, identity=stale)


def test_global_cache_version_was_not_bumped():
    """Bumping it would invalidate every provider's entries, not just omni's."""
    assert vca._PREPROCESSED_SPLIT_CACHE_VERSION == 2
