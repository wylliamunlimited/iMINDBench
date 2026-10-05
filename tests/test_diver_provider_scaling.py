"""DIVER reads each dataset's signal unit and electrode type from the dataset list."""

import pytest
import torch
from omegaconf import OmegaConf

from imindbench.models.diver_model import DIVERModel

# provider -> (factor applied to the stored signal, electrode type)
EXPECTED = {
    "neuroprobev2": (1.0 / 200.0, "depth"),
    "neuroprobe2025": (1.0 / 200.0, "depth"),
    "kelesbyd2024": (1e6 / 200.0, "depth"),
    "berezutskayapippi2022": (1e6 / 200.0, "depth"),
}


def _model(provider):
    cfg = OmegaConf.create({"upstream_ckpt": "unused.ckpt", "patch_size": 4})
    return DIVERModel(cfg, OmegaConf.create({"provider": provider}))


@pytest.mark.parametrize("provider", sorted(EXPECTED))
@pytest.mark.parametrize("with_coords", [False, True])
def test_prepare_batch_scales_and_tags_electrodes(provider, with_coords):
    factor, subtype = EXPECTED[provider]
    x = torch.ones(2, 3, 8)
    batch = {"x": x, "y": torch.zeros(2)}
    if with_coords:
        batch["channel_coords"] = torch.zeros(2, 3, 3)

    out = _model(provider).prepare_batch(batch)

    assert out["x"].shape == (2, 3, 2, 4)
    torch.testing.assert_close(out["x"], torch.full((2, 3, 2, 4), factor))
    infos = out["model_kwargs"]["data_info_list"]
    assert len(infos) == 2
    for info in infos:
        assert info["modality"] == "iEEG"
        assert info["coord_subtype"] == [subtype] * 3
        assert info["xyz_id"].shape == (3, 3)


def test_unknown_provider_is_rejected():
    batch = {"x": torch.ones(1, 1, 4), "y": torch.zeros(1)}
    with pytest.raises(NotImplementedError, match="Unknown dataset provider"):
        _model("not_a_dataset").prepare_batch(batch)


def test_every_dataset_sets_signal_unit_and_electrode_subtype():
    from imindbench.utils.pipeline_contracts import _PROVIDER_SPECS, provider_property

    for provider in _PROVIDER_SPECS:
        assert provider_property(provider, "signal_unit") in {"uV", "V"}
        assert provider_property(provider, "electrode_subtype") in {"depth", "grid"}
    with pytest.raises(KeyError, match="does not set"):
        provider_property("neuroprobev2", "not_a_key")
