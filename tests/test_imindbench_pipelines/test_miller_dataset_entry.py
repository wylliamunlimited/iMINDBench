"""The millerecog2019 dataset entry: recording ids and coordinate profiles."""

import numpy as np
import pytest
from miller_fakes import FakeMillerECoG2019, install_fake_miller
from omegaconf import OmegaConf

from imindbench.preprocessors import build_preprocessor
from imindbench.utils import data_adapter
from imindbench.utils.pipeline_contracts import (
    VALID_COORDINATE_PROFILES,
    is_multi_subject,
    provider_property,
    resolve_provider_n_folds,
)


@pytest.fixture
def fake_miller(monkeypatch):
    return install_fake_miller(monkeypatch)


def test_dataset_list_entry():
    assert provider_property("millerecog2019", "signal_unit") == "uV"
    assert provider_property("millerecog2019", "electrode_subtype") == "grid"
    assert is_multi_subject("millerecog2019", "within-session") is False


def test_fold_count_comes_from_the_dataset_class(fake_miller):
    assert (
        resolve_provider_n_folds(
            dataset_provider="millerecog2019", regime="within-session"
        )
        == 2
    )


@pytest.mark.parametrize(
    ("recording_id", "expected"),
    [
        ("sub-al_set-motor_basic", (1, 1)),
        ("sub-bp_set-imagery_basic", (3, 2)),
        ("sub-zt_set-fingerflex_reg_w1000h25b", (29, 25)),
        # Multi-source recording ids carry a "<source>/" prefix.
        ("millerecog2019/sub-bp_set-fingerflex_reg_w1000", (3, 13)),
    ],
)
def test_letter_coded_recording_ids_become_numbers(fake_miller, recording_id, expected):
    assert (
        data_adapter._subject_session_from_recording_id(
            recording_id=recording_id, dataset_provider="millerecog2019"
        )
        == expected
    )
    assert (
        data_adapter._subject_from_recording_id(
            recording_id=recording_id, dataset_provider="millerecog2019"
        )
        == expected[0]
    )


def test_regression_set_ids_parse_although_the_module_pattern_rejects_them(
    fake_miller,
):
    import miller_fakes

    with pytest.raises(ValueError):
        miller_fakes._from_recording_id("sub-bp_set-fingerflex_reg_w1000h25b")
    assert data_adapter._subject_session_from_recording_id(
        recording_id="sub-bp_set-fingerflex_reg_w1000h25b",
        dataset_provider="millerecog2019",
    ) == (3, 25)


def test_bad_recording_id_is_rejected(fake_miller):
    with pytest.raises(ValueError, match="Invalid MillerECoG2019 recording_id"):
        data_adapter._subject_session_from_recording_id(
            recording_id="sub_1_trial001", dataset_provider="millerecog2019"
        )


def test_numeric_recording_ids_still_use_the_patterns():
    assert data_adapter._subject_session_from_recording_id(
        recording_id="sub-CS41_ses-P1CSR2", dataset_provider="kelesbyd2024"
    ) == (41, 2)


def test_coordinate_profile_lists_agree():
    assert set(data_adapter.COORDINATE_PROFILES) == VALID_COORDINATE_PROFILES


@pytest.mark.parametrize("profile", ["popt_zero", "popt_miller", "diver_mni_miller"])
def test_miller_only_profiles_reject_other_datasets(profile):
    with pytest.raises(ValueError, match="has no mapping for provider"):
        data_adapter._resolve_coordinate_transform(
            provider_key="kelesbyd2024", coordinate_profile=profile
        )


def _dataset_cfg(tmp_path, *, coordinate_profile, test_session=1):
    return OmegaConf.create(
        dict(
            provider="millerecog2019",
            root=str(tmp_path),
            dirname="miller_ecog_library_2019",
            subset_tier="full",
            label_mode="binary",
            task="move_vs_rest",
            regime="within-session",
            test_subject=3,
            test_session=test_session,
            coordinate_profile=coordinate_profile,
            uniquify_channel_ids_with_subject=True,
            uniquify_channel_ids_with_session=True,
        )
    )


def _build_fold(tmp_path, *, coordinate_profile, require_coords, test_session=1):
    preprocessor_cfg = OmegaConf.create({"chain": [{"name": "raw"}]})
    return data_adapter.build_neuroprobe_torch_fold(
        _dataset_cfg(
            tmp_path,
            coordinate_profile=coordinate_profile,
            test_session=test_session,
        ),
        preprocessor=build_preprocessor(preprocessor_cfg),
        preprocessor_cfg=preprocessor_cfg,
        fold_idx=0,
        seed=0,
        require_coords=require_coords,
        needs_pool=False,
    )


def test_fake_fold_has_both_classes_in_every_split(tmp_path, fake_miller):
    fold = _build_fold(tmp_path, coordinate_profile="diver_mni", require_coords=False)
    for split in ("train", "val", "test"):
        samples = list(fold[f"{split}_split"])
        assert {int(sample["y"]) for sample in samples} == {0, 1}
        assert samples[0]["x"].shape == (4, 50)
        assert samples[0]["channel_coords"] is None


@pytest.mark.parametrize("test_session", [1, 4])  # Talairach set, non-Talairach set
def test_popt_zero_places_every_channel_at_the_origin(
    tmp_path, fake_miller, test_session
):
    fold = _build_fold(
        tmp_path,
        coordinate_profile="popt_zero",
        require_coords=True,
        test_session=test_session,
    )
    coords = fold["test_split"][0]["channel_coords"]
    np.testing.assert_array_equal(coords, np.zeros((4, 3), dtype=np.float32))


@pytest.mark.parametrize("profile", ["popt_lip", "diver_mni"])
def test_shared_profiles_give_miller_no_coordinates(tmp_path, fake_miller, profile):
    fold = _build_fold(tmp_path, coordinate_profile=profile, require_coords=False)
    assert fold["test_split"][0]["channel_coords"] is None


def test_positions_drop_low_quality_channels(tmp_path, monkeypatch):
    class WithPositions(FakeMillerECoG2019):
        with_positions = True

    install_fake_miller(monkeypatch, WithPositions)
    popt = _build_fold(tmp_path, coordinate_profile="popt_miller", require_coords=True)
    diver = _build_fold(
        tmp_path, coordinate_profile="diver_mni_miller", require_coords=True
    )
    # The fake marks its last channel as low quality, so 3 of 4 channels stay.
    popt_coords = popt["test_split"][0]["channel_coords"]
    diver_coords = diver["test_split"][0]["channel_coords"]
    assert popt_coords.shape == diver_coords.shape == (3, 3)
    expected_mni = np.stack(
        [np.linspace(-40.0, 40.0, 4), np.full(4, -20.0), np.full(4, 30.0)], axis=1
    )[:3].astype(np.float32) + np.float32(1.0)
    np.testing.assert_allclose(diver_coords, expected_mni)
    np.testing.assert_allclose(
        popt_coords, data_adapter.byd_mni152_ras_to_popt_lip(expected_mni)
    )


def test_positions_profile_needs_the_build_with_positions(tmp_path, fake_miller):
    with pytest.raises(KeyError, match="mni152_strict"):
        _build_fold(tmp_path, coordinate_profile="popt_miller", require_coords=True)
