"""Every miller_* preset on the window lengths the Miller task sets use.

Miller windows are 1000, 800, 500 or 250 ms long at 1000 Hz. This test builds
a fold through the real fold builder for each preset and window length and
records which pairs work. The expected failures are written down here so a
change in behaviour is noticed:

- multi-STFT (miller_multi_stft_1000Hz) needs at least 500 samples for its
  500 ms low-frequency window, so 250 ms windows fail.
- DIVER cuts its 500 Hz input into 50-sample patches. A 250 ms window gives 125
  samples, which is not a whole number of patches, so DIVER rejects it.
"""

from pathlib import Path

import pytest
import torch
from miller_fakes import FakeMillerECoG2019, install_fake_miller
from omegaconf import OmegaConf

from imindbench.models.diver_model import DIVERModel
from imindbench.models.popt_components.brainbert_encoder import BrainBERTEncoder
from imindbench.preprocessors import build_preprocessor
from imindbench.utils import data_adapter

CONF = Path(__file__).resolve().parents[2] / "imindbench/conf/preprocessor"
MILLER_PRESETS = sorted(path.stem for path in CONF.glob("miller_*.yaml"))
WINDOW_MS = (1000, 800, 500, 250)

# preset -> {window_ms: expected per-sample feature shape for 4 channels}
EXPECTED_SHAPES = {
    "miller_multi_stft_1000Hz": {1000: (4, 17, 75), 800: (4, 13, 75), 500: (4, 9, 75)},
    "miller_stft_brainbert_1000Hz": {1000: (4, 4), 800: (4, 4), 500: (4, 4), 250: (4, 4)},
    "miller_wav_barista_1000to2048Hz": {
        1000: (4, 2048), 800: (4, 1639), 500: (4, 1024), 250: (4, 512),
    },
    "miller_wav_diver_1000to500Hz": {1000: (4, 500), 800: (4, 400), 500: (4, 250), 250: (4, 125)},
    "miller_wav_hpf_robust_1000to500Hz": {
        1000: (4, 500), 800: (4, 400), 500: (4, 250), 250: (4, 125),
    },
}  # fmt: skip
FAILING = {("miller_multi_stft_1000Hz", 250)}


def test_every_miller_preset_is_listed():
    assert MILLER_PRESETS == sorted(EXPECTED_SHAPES)


def test_miller_presets_do_no_rereferencing_or_filtering():
    removed = {
        "laplacian_rereference",
        "time_domain_filter",
        "time_domain_filter_diver_style",
        "context_window",
        "crop_to_target_window",
    }
    for name in MILLER_PRESETS:
        steps = {step.name for step in OmegaConf.load(CONF / f"{name}.yaml").chain}
        assert not steps & removed, name


def _brainbert_checkpoint(tmp_path):
    encoder_cfg = OmegaConf.create(
        dict(
            input_dim=40,
            hidden_dim=4,
            nhead=2,
            encoder_num_layers=1,
            layer_dim_feedforward=8,
            dropout=0.0,
        )
    )
    encoder = BrainBERTEncoder(40, 4, 2, 1, 8, dropout=0.0)
    path = tmp_path / "brainbert.pt"
    torch.save({"model_cfg": encoder_cfg, "model": encoder.state_dict()}, path)
    return path


def _build(tmp_path, monkeypatch, preset, window_ms):
    class Fake(FakeMillerECoG2019):
        sampling_rate_hz = 1000.0
        window_sec = window_ms / 1000.0
        windows_per_class = 4

    install_fake_miller(monkeypatch, Fake)
    cfg = OmegaConf.load(CONF / f"{preset}.yaml")
    for step in cfg.chain:
        if step.name == "brainbert_encoder":
            step.upstream_ckpt = str(_brainbert_checkpoint(tmp_path))
            step.device = "cpu"
    dataset_cfg = OmegaConf.create(
        dict(
            provider="millerecog2019",
            root=str(tmp_path),
            dirname="miller_ecog_library_2019",
            subset_tier="full",
            label_mode="binary",
            task="move_vs_rest",
            regime="within-session",
            test_subject=3,
            test_session=1,
            coordinate_profile="diver_mni",
            uniquify_channel_ids_with_subject=True,
            uniquify_channel_ids_with_session=True,
        )
    )
    return data_adapter.build_neuroprobe_torch_fold(
        dataset_cfg,
        preprocessor=build_preprocessor(cfg),
        preprocessor_cfg=cfg,
        fold_idx=0,
        seed=0,
        require_coords=False,
        needs_pool=False,
    )


@pytest.mark.parametrize("window_ms", WINDOW_MS)
@pytest.mark.parametrize("preset", sorted(EXPECTED_SHAPES))
def test_preset_on_window_length(tmp_path, monkeypatch, preset, window_ms):
    if (preset, window_ms) in FAILING:
        with pytest.raises(RuntimeError, match="[Pp]adding size"):
            _build(tmp_path, monkeypatch, preset, window_ms)
        return
    fold = _build(tmp_path, monkeypatch, preset, window_ms)
    for split in ("train", "val", "test"):
        assert (
            tuple(fold[f"{split}_split"][0]["x"].shape)
            == (EXPECTED_SHAPES[preset][window_ms])
        )


@pytest.mark.parametrize("window_ms", WINDOW_MS)
def test_diver_accepts_window_length(window_ms):
    """DIVER needs a whole number of 50-sample patches at 500 Hz."""
    model = DIVERModel(
        OmegaConf.create({"upstream_ckpt": "unused.ckpt", "patch_size": 50}),
        OmegaConf.create({"provider": "millerecog2019"}),
    )
    n_samples = window_ms // 2
    batch = {"x": torch.zeros(1, 4, n_samples), "y": torch.zeros(1)}
    if window_ms == 250:
        with pytest.raises(ValueError, match="not divisible by patch_size"):
            model.prepare_batch(batch)
    else:
        out = model.prepare_batch(batch)
        assert out["x"].shape == (1, 4, n_samples // 50, 50)
        assert out["model_kwargs"]["data_info_list"][0]["coord_subtype"] == (
            ["grid"] * 4
        )


# The 11 model/preset pairings the Miller run script uses.
MILLER_PAIRINGS = [
    *(
        (model, preset)
        for model in ("logistic", "mlp", "cnn")
        for preset in ("miller_multi_stft_1000Hz", "miller_wav_hpf_robust_1000to500Hz")
    ),
    ("htnet_500Hz", "miller_wav_hpf_robust_1000to500Hz"),
    ("popt", "miller_multi_stft_1000Hz"),
    ("linear_baseline", "miller_stft_brainbert_1000Hz"),
    ("barista", "miller_wav_barista_1000to2048Hz"),
    ("diver", "miller_wav_diver_1000to500Hz"),
]


@pytest.mark.parametrize("dataset", ["millerecog2019", "millerecog2019_pos"])
@pytest.mark.parametrize(("model", "preset"), MILLER_PAIRINGS)
def test_miller_configs_compose_and_pass_validation(dataset, model, preset):
    from hydra import compose, initialize_config_dir

    from imindbench.utils.pipeline_contracts import validate_eval_config

    with initialize_config_dir(config_dir=str(CONF.parent), version_base="1.1"):
        cfg = compose(
            config_name="config",
            overrides=[
                "paths=example",
                f"dataset={dataset}",
                f"model={model}",
                f"preprocessor={preset}",
                "experiment=default",
            ],
        )
    validate_eval_config(cfg)
    assert cfg.dataset.provider == "millerecog2019"
    expected_profile = {
        "millerecog2019": "popt_zero",
        "millerecog2019_pos": "popt_miller",
    }[dataset]
    if model == "diver":
        # DIVER's model config replaces the dataset's coordinate profile.
        expected_profile = "diver_mni"
    assert cfg.dataset.coordinate_profile == expected_profile
    if model == "barista":
        assert cfg.dataset.brain_area_key == "label_destrieux"
