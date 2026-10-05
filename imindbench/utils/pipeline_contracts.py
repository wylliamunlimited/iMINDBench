"""Pipeline contracts for processed neuroprobe evaluation paths."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, ListConfig, OmegaConf
from torch import optim

from imindbench.preprocessors import PREPROCESSOR_REGISTRY
from imindbench.utils.window_slicing import (
    DEFAULT_WINDOW_SLICING_POLICY,
    validate_window_slicing_policy,
)

AUTO_MAX_TRAIN_SAMPLES_PER_SUBJECT = "auto"


# =========================
# Dataset
# =========================


def _require_torch_brain_neuroprobe_v2():
    # Import lazily so local tooling/tests that do not instantiate datasets can
    # still import this module without optional data dependencies installed.
    try:
        from torch_brain.datasets import NeuroprobeV2
    except ImportError as exc:
        raise ImportError(
            "Variable-channel processed mode requires torch_brain.datasets "
            "with NeuroprobeV2."
        ) from exc
    return NeuroprobeV2


def _require_torch_brain_neuroprobe2025():
    # Import lazily so local tooling/tests that do not instantiate datasets can
    # still import this module without optional data dependencies installed.
    try:
        from torch_brain.datasets import Neuroprobe2025
    except ImportError as exc:
        raise ImportError(
            "Processed provider 'neuroprobe2025' requires torch_brain.datasets "
            "with Neuroprobe2025."
        ) from exc
    return Neuroprobe2025


def _require_torch_brain_keles_byd_2024():
    # Import lazily so local tooling/tests that do not instantiate datasets can
    # still import this module without optional data dependencies installed.
    try:
        from torch_brain.datasets import KelesBYD2024
    except ImportError as exc:
        raise ImportError(
            "Processed provider 'kelesbyd2024' requires torch_brain.datasets "
            "with KelesBYD2024."
        ) from exc
    return KelesBYD2024


def _require_torch_brain_berezutskaya_pippi_2022():
    # Import lazily so local tooling/tests that do not instantiate datasets can
    # still import this module without optional data dependencies installed.
    try:
        from torch_brain.datasets import BerezutskayaPippi2022
    except ImportError as exc:
        raise ImportError(
            "Processed provider 'berezutskayapippi2022' requires "
            "torch_brain.datasets with BerezutskayaPippi2022."
        ) from exc
    return BerezutskayaPippi2022


# =========================
# Routing
# =========================

# Keep all provider facts in one place so public/private variants can differ by
# a small localized patch instead of scattering provider checks across helpers.
_PROVIDER_SPECS: dict[str, dict[str, Any]] = {
    "neuroprobev2": {
        # Unit of the stored signal and the electrode type DIVER is told about.
        "signal_unit": "uV",
        "electrode_subtype": "depth",
        "dataset_class_loader": _require_torch_brain_neuroprobe_v2,
        "regime_is_multi_subject": {
            "within-session": False,
            "hold-in-session": True,
            "hold-out-session": True,
            "hold-out-subject": True,
        },
    },
    "neuroprobe2025": {
        "signal_unit": "uV",
        "electrode_subtype": "depth",
        "dataset_class_loader": _require_torch_brain_neuroprobe2025,
        "regime_is_multi_subject": {
            "SS-SM": False,
            "SS-DM": False,
            "DS-DM": True,
        },
        "valid_subset_tiers": {"full", "lite", "nano"},
    },
    "kelesbyd2024": {
        "signal_unit": "V",
        "electrode_subtype": "depth",
        "dataset_class_loader": _require_torch_brain_keles_byd_2024,
        "regime_is_multi_subject": {
            "within-session": False,
            "hold-in-session": True,
            "hold-out-session": True,
            "hold-out-subject": True,
        },
        "valid_subset_tiers": {"full"},
    },
    "berezutskayapippi2022": {
        "signal_unit": "V",
        "electrode_subtype": "depth",
        "dataset_class_loader": _require_torch_brain_berezutskaya_pippi_2022,
        "regime_is_multi_subject": {
            "within-session": False,
            "hold-in-session": True,
            "hold-out-session": True,
            "hold-out-subject": True,
        },
        "valid_subset_tiers": {"full", "high-cov", "low-cov"},
    },
}


def _get_provider_spec(provider: str) -> dict[str, Any]:
    """Return the canonical provider spec for a validated provider key."""
    return _PROVIDER_SPECS[provider]


def provider_property(provider: Any, key: str) -> Any:
    """Return one value from a dataset's entry in the dataset list.

    Used for per-dataset facts such as ``signal_unit`` and
    ``electrode_subtype`` so callers do not have to check dataset names.
    Raises ``NotImplementedError`` for an unknown dataset and ``KeyError`` when
    the dataset entry does not set ``key``.
    """
    if provider not in _PROVIDER_SPECS:
        raise NotImplementedError(
            f"Unknown dataset provider {provider!r}. Known providers: "
            f"{sorted(_PROVIDER_SPECS)}."
        )
    spec = _get_provider_spec(provider)
    if key not in spec:
        raise KeyError(f"Dataset provider {provider!r} does not set {key!r}.")
    return spec[key]


def is_multi_subject(provider: str, regime: str) -> bool:
    """Whether (provider, regime) involves multiple subjects with potentially
    different channel sets.

    Call only after validate_eval_config has passed.
    """
    return _get_provider_spec(provider)["regime_is_multi_subject"][regime]


def needs_region_intersection_pool(
    provider: str, regime: str, requires_aligned_channels: bool
) -> bool:
    """Whether the (provider, regime, model) combo needs region-intersection pooling.

    Region-intersection pooling is needed when the regime is multi-subject AND
    the model requires aligned channels.

    Call only after validate_eval_config has passed.
    """
    return is_multi_subject(provider, regime) and requires_aligned_channels


def get_dataset_class(provider: str):
    """Return dataset class for a validated provider key.

    Call only after validate_eval_config has passed.
    """
    return _get_provider_spec(provider)["dataset_class_loader"]()


def resolve_provider_n_folds(*, dataset_provider: str, regime: str) -> int:
    """Resolve fold count from dataset class API.

    Call only after validate_eval_config has passed.
    """
    dataset_cls = get_dataset_class(dataset_provider)
    # Dataset classes are the source of truth for fold cardinality per regime.
    resolver = getattr(dataset_cls, "num_folds_for_regime", None)
    if resolver is None or not callable(resolver):
        raise RuntimeError(
            "Selected dataset class does not define num_folds_for_regime(...). "
            "Please add this API on the dataset class to support processed eval routing."
        )
    n_folds = resolver(regime)
    if not isinstance(n_folds, int) or isinstance(n_folds, bool):
        raise TypeError(
            "Dataset class num_folds_for_regime(...) must return int, got "
            f"{type(n_folds).__name__}."
        )
    return n_folds


def _validate_dataset_provider(provider: Any) -> str:
    """Validate dataset.provider and return canonical provider key."""
    if not isinstance(provider, str):
        raise TypeError(
            f"dataset.provider must be a str, got {type(provider).__name__}."
        )
    if provider.strip() != provider:
        raise ValueError(
            "dataset.provider must not include leading/trailing whitespace. "
            f"Got '{provider}'."
        )
    valid_dataset_providers = sorted(_PROVIDER_SPECS.keys())
    if provider not in _PROVIDER_SPECS:
        raise ValueError(
            "dataset.provider must be one of "
            f"{valid_dataset_providers}, got '{provider}'."
        )
    return provider


def _validate_provider_regime(
    *,
    dataset_provider: Any,
    dataset_regime: Any,
) -> None:
    """Validate dataset.provider + dataset.regime pair.

    Used only inside validate_eval_config for boundary validation.
    """
    provider = _validate_dataset_provider(dataset_provider)
    if not isinstance(dataset_regime, str):
        raise TypeError(
            f"dataset.regime must be a str, got {type(dataset_regime).__name__}."
        )
    if dataset_regime.strip() != dataset_regime:
        raise ValueError(
            "dataset.regime must not include leading/trailing whitespace. "
            f"Got '{dataset_regime}'."
        )
    if dataset_regime == "":
        raise ValueError("dataset.regime must be non-empty.")
    provider_routes = _get_provider_spec(provider)["regime_is_multi_subject"]
    if dataset_regime not in provider_routes:
        raise ValueError(
            "Unsupported dataset provider/regime tuple: "
            f"(dataset.provider='{provider}', dataset.regime='{dataset_regime}'). "
            "Allowed regimes for provider are "
            f"{sorted(provider_routes.keys())}."
        )


def build_processed_split_provider(
    *,
    dataset_provider: Any,
    dataset_cfg: Any,
    split: str,
    fold_idx: int,
    regime: str,
):
    """Instantiate one split provider object via dataset.provider."""
    dataset_cls = get_dataset_class(dataset_provider)

    # Constructor kwargs are intentionally mirrored from dataset_cfg so this
    # helper stays a thin adapter over dataset-class APIs.
    return dataset_cls(
        root=dataset_cfg.root,
        dirname=dataset_cfg.dirname,
        subset_tier=dataset_cfg.subset_tier,
        test_subject=dataset_cfg.test_subject,
        test_session=dataset_cfg.test_session,
        split=split,
        label_mode=dataset_cfg.label_mode,
        task=dataset_cfg.task,
        regime=regime,
        fold=fold_idx,
        uniquify_channel_ids_with_subject=dataset_cfg.uniquify_channel_ids_with_subject,
        uniquify_channel_ids_with_session=dataset_cfg.uniquify_channel_ids_with_session,
    )


# =========================
# Config
# =========================

VALID_LABEL_MODES = {"binary", "multiclass"}
VALID_SUBSET_TIERS = {"full", "lite", "nano"}
VALID_COORDINATE_PROFILES = {"popt_lip", "diver_mni"}


def _validate_label_mode(label_mode: str) -> None:
    if label_mode not in VALID_LABEL_MODES:
        raise ValueError(
            f"label_mode must be one of {sorted(VALID_LABEL_MODES)}, got '{label_mode}'."
        )


def _validate_subset_tier(subset_tier: str, *, provider: str | None = None) -> None:
    """Validate dataset.subset_tier against provider-specific allowed values."""
    valid_subset_tiers = VALID_SUBSET_TIERS
    if provider is not None and provider in _PROVIDER_SPECS:
        valid_subset_tiers = _get_provider_spec(provider).get(
            "valid_subset_tiers", VALID_SUBSET_TIERS
        )
    if subset_tier not in valid_subset_tiers:
        raise ValueError(
            "dataset.subset_tier must be one of "
            f"{sorted(valid_subset_tiers)}, got '{subset_tier}'."
        )


def _parse_required_bool(value: Any, key: str) -> None:
    """Fail fast when a required dataset boolean field is missing/wrong type."""
    if not isinstance(value, bool):
        raise TypeError(f"dataset.{key} must be a bool, got {type(value).__name__}.")


def _require_non_empty_dataset_str(dataset_cfg: dict[str, Any], key: str) -> str:
    if key not in dataset_cfg:
        raise ValueError(
            f"Missing required dataset.{key}. Set dataset.{key} explicitly."
        )
    value = dataset_cfg[key]
    if not isinstance(value, str):
        raise TypeError(f"dataset.{key} must be a str, got {type(value).__name__}.")
    if value == "":
        raise ValueError(f"dataset.{key} must be non-empty.")
    if value.strip() != value:
        raise ValueError(
            f"dataset.{key} must not include leading/trailing whitespace. "
            f"Got '{value}'."
        )
    return value


def _require_dataset_int(dataset_cfg: dict[str, Any], key: str) -> int:
    if key not in dataset_cfg:
        raise ValueError(
            f"Missing required dataset.{key}. Set dataset.{key} explicitly."
        )
    value = dataset_cfg[key]
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"dataset.{key} must be an int, got {type(value).__name__}.")
    return value


def _parse_optional_dataset_brain_area_key(dataset_cfg: dict[str, Any]) -> str | None:
    value = dataset_cfg.get("brain_area_key")
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(
            f"dataset.brain_area_key must be a str when set, got {type(value).__name__}."
        )
    if value == "":
        raise ValueError("dataset.brain_area_key must be non-empty when set.")
    if value.strip() != value:
        raise ValueError(
            "dataset.brain_area_key must not include leading/trailing whitespace. "
            f"Got '{value}'."
        )
    return value


def _parse_optional_dataset_train_same_subject_only(
    dataset_cfg: dict[str, Any],
) -> bool:
    value = dataset_cfg.get("train_same_subject_only", False)
    if not isinstance(value, bool):
        raise TypeError(
            "dataset.train_same_subject_only must be a bool, got "
            f"{type(value).__name__}."
        )
    return value


def _parse_optional_dataset_decodable_subject_sessions_only(
    dataset_cfg: dict[str, Any],
) -> bool:
    if "train_decodable_subject_sessions_only" in dataset_cfg:
        raise ValueError(
            "dataset.train_decodable_subject_sessions_only has been renamed to "
            "dataset.decodable_subject_sessions_only."
        )
    value = dataset_cfg.get("decodable_subject_sessions_only", False)
    if not isinstance(value, bool):
        raise TypeError(
            "dataset.decodable_subject_sessions_only must be a bool, got "
            f"{type(value).__name__}."
        )
    return value


def validate_decodable_train_source_regimes(
    train_sources: list[dict[str, Any]],
    *,
    enabled: bool,
) -> None:
    if not enabled or not train_sources:
        return
    non_hold_in_sources = [
        str(source_cfg["provider"])
        for source_cfg in train_sources
        if source_cfg["regime"] != "hold-in-session"
    ]
    if non_hold_in_sources:
        raise ValueError(
            "dataset.decodable_subject_sessions_only=true with "
            "dataset.train_sources requires every "
            "dataset.train_sources[].regime='hold-in-session'. "
            "Decodable multisource training is defined as all valid "
            "task-decodable recordings per source, independent of the eval "
            "target. Non-hold-in train sources: "
            f"{non_hold_in_sources}."
        )


def _parse_optional_dataset_train_sample_fraction(
    dataset_cfg: dict[str, Any],
) -> float:
    value = dataset_cfg.get("train_sample_fraction", 1.0)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(
            "dataset.train_sample_fraction must be a float in (0, 1], got "
            f"{type(value).__name__}."
        )
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError("dataset.train_sample_fraction must be finite.")
    if normalized <= 0.0 or normalized > 1.0:
        raise ValueError(
            f"dataset.train_sample_fraction must be in (0, 1], got {normalized}."
        )
    return normalized


def _parse_optional_max_train_samples_per_subject(
    cfg: dict[str, Any],
    *,
    key_label: str = "dataset.max_train_samples_per_subject",
) -> int | str | None:
    value = cfg.get("max_train_samples_per_subject", None)
    if value is None:
        return None
    if isinstance(value, str):
        if value == AUTO_MAX_TRAIN_SAMPLES_PER_SUBJECT:
            return value
        raise TypeError(
            f"{key_label} must be an int or '{AUTO_MAX_TRAIN_SAMPLES_PER_SUBJECT}' "
            f"when set, got str."
        )
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            f"{key_label} must be an int or '{AUTO_MAX_TRAIN_SAMPLES_PER_SUBJECT}' "
            f"when set, got {type(value).__name__}."
        )
    if value < 1:
        raise ValueError(f"{key_label} must be >= 1 when set.")
    return int(value)


def _parse_optional_dataset_coordinate_profile(
    dataset_cfg: dict[str, Any],
) -> str:
    value = dataset_cfg.get("coordinate_profile", "popt_lip")
    if not isinstance(value, str):
        raise TypeError(
            "dataset.coordinate_profile must be a str when set, got "
            f"{type(value).__name__}."
        )
    if value == "":
        raise ValueError("dataset.coordinate_profile must be non-empty.")
    if value.strip() != value:
        raise ValueError(
            "dataset.coordinate_profile must not include leading/trailing whitespace. "
            f"Got '{value}'."
        )
    if value not in VALID_COORDINATE_PROFILES:
        raise ValueError(
            "dataset.coordinate_profile must be one of "
            f"{sorted(VALID_COORDINATE_PROFILES)}, got '{value}'."
        )
    return value


def _cfg_like_get(cfg_like: Any, key: str, default: Any = None) -> Any:
    """Read from DictConfig/dict/namespace-like objects with one helper."""
    getter = getattr(cfg_like, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(cfg_like, key, default)


_TRAIN_SOURCE_ALLOWED_KEYS = {
    "provider",
    "preprocessor",
    "root",
    "dirname",
    "subset_tier",
    "test_subject",
    "test_session",
    "regime",
    "train_sample_fraction",
    "max_train_samples_per_subject",
}

_PREPROCESSOR_CONFIG_DIR = (
    Path(__file__).resolve().parent.parent / "conf" / "preprocessor"
)


def _parse_optional_train_source_preprocessor(
    source_cfg: dict[str, Any],
    *,
    source_provider: str,
) -> str | None:
    value = source_cfg.get("preprocessor")
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(
            "dataset.train_sources[].preprocessor must be a str when set, got "
            f"{type(value).__name__} for provider '{source_provider}'."
        )
    if value == "":
        raise ValueError(
            "dataset.train_sources[].preprocessor must be non-empty for provider "
            f"'{source_provider}'."
        )
    if value.strip() != value:
        raise ValueError(
            "dataset.train_sources[].preprocessor must not include "
            f"leading/trailing whitespace. Got '{value}' for provider "
            f"'{source_provider}'."
        )
    # Runtime loads source overrides from preset YAMLs, not the stage registry.
    cfg_path = _PREPROCESSOR_CONFIG_DIR / f"{value}.yaml"
    if not cfg_path.exists():
        raise ValueError(
            "dataset.train_sources[].preprocessor must reference an existing "
            f"preprocessor config. Missing '{cfg_path.name}' for provider "
            f"'{source_provider}'."
        )
    return value


def _parse_train_source_sample_fraction(
    source_cfg: dict[str, Any],
    *,
    source_provider: str,
) -> float:
    value = source_cfg.get("train_sample_fraction", 1.0)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(
            "dataset.train_sources[].train_sample_fraction must be a float in (0, 1], "
            f"got {type(value).__name__} for provider '{source_provider}'."
        )
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(
            "dataset.train_sources[].train_sample_fraction must be finite for "
            f"provider '{source_provider}'."
        )
    if normalized <= 0.0 or normalized > 1.0:
        raise ValueError(
            "dataset.train_sources[].train_sample_fraction must be in (0, 1], got "
            f"{normalized} for provider '{source_provider}'."
        )
    return normalized


def resolve_train_source_configs(dataset_cfg: Any) -> list[dict[str, Any]]:
    """Return normalized multi-source train configs.

    Each entry inherits label/task/uniquify behavior from the top-level dataset
    config and may override only the minimal source-selection fields required to
    instantiate its train split provider.
    """
    raw_sources = _cfg_like_get(dataset_cfg, "train_sources", None)
    if raw_sources is None:
        return []
    if OmegaConf.is_config(raw_sources):
        raw_sources = OmegaConf.to_container(raw_sources, resolve=True)
    if not isinstance(raw_sources, list):
        raise TypeError(
            "dataset.train_sources must be a list[dict] when set, got "
            f"{type(raw_sources).__name__}."
        )
    if not raw_sources:
        raise ValueError("dataset.train_sources must contain at least one source.")

    top_level_option_cfg = {
        "train_same_subject_only": _cfg_like_get(
            dataset_cfg, "train_same_subject_only", False
        ),
        "train_sample_fraction": _cfg_like_get(
            dataset_cfg, "train_sample_fraction", 1.0
        ),
        "max_train_samples_per_subject": _cfg_like_get(
            dataset_cfg, "max_train_samples_per_subject", None
        ),
    }
    top_level_same_subject_only = _parse_optional_dataset_train_same_subject_only(
        top_level_option_cfg
    )
    if top_level_same_subject_only:
        raise ValueError(
            "dataset.train_same_subject_only=true is unsupported when "
            "dataset.train_sources is set. Configure source-specific train "
            "selection via each source's provider/regime/test target instead."
        )

    top_level_train_fraction = _parse_optional_dataset_train_sample_fraction(
        top_level_option_cfg
    )
    if top_level_train_fraction != 1.0:
        raise ValueError(
            "dataset.train_sample_fraction must stay at 1.0 when "
            "dataset.train_sources is set. Configure per-source subsampling via "
            "dataset.train_sources[].train_sample_fraction."
        )
    top_level_max_train_samples = _parse_optional_max_train_samples_per_subject(
        top_level_option_cfg
    )

    inherited_defaults = {
        # Keep source configs focused on provider selection; the top-level
        # dataset config continues to own task/label/uniquify semantics.
        "provider": _cfg_like_get(dataset_cfg, "provider", None),
        "window_slicing_policy": _cfg_like_get(
            dataset_cfg, "window_slicing_policy", DEFAULT_WINDOW_SLICING_POLICY
        ),
        "root": _cfg_like_get(dataset_cfg, "root", None),
        "dirname": _cfg_like_get(dataset_cfg, "dirname", None),
        "subset_tier": _cfg_like_get(dataset_cfg, "subset_tier", None),
        "label_mode": _cfg_like_get(dataset_cfg, "label_mode", None),
        "task": _cfg_like_get(dataset_cfg, "task", None),
        "coordinate_profile": _parse_optional_dataset_coordinate_profile(
            {
                "coordinate_profile": _cfg_like_get(
                    dataset_cfg, "coordinate_profile", "popt_lip"
                )
            }
        ),
        "test_subject": _cfg_like_get(dataset_cfg, "test_subject", None),
        "test_session": _cfg_like_get(dataset_cfg, "test_session", None),
        "regime": _cfg_like_get(dataset_cfg, "regime", None),
        "uniquify_channel_ids_with_subject": _cfg_like_get(
            dataset_cfg, "uniquify_channel_ids_with_subject", None
        ),
        "uniquify_channel_ids_with_session": _cfg_like_get(
            dataset_cfg, "uniquify_channel_ids_with_session", None
        ),
        "max_train_samples_per_subject": top_level_max_train_samples,
    }
    target_provider = inherited_defaults["provider"]

    normalized_sources: list[dict[str, Any]] = []
    seen_providers: set[str] = set()
    for idx, raw_source in enumerate(raw_sources):
        if OmegaConf.is_config(raw_source):
            raw_source = OmegaConf.to_container(raw_source, resolve=True)
        if not isinstance(raw_source, dict):
            raise TypeError(
                "dataset.train_sources entries must be dict-like mappings, got "
                f"{type(raw_source).__name__} at index {idx}."
            )

        unknown_keys = sorted(set(raw_source.keys()) - _TRAIN_SOURCE_ALLOWED_KEYS)
        if unknown_keys:
            raise ValueError(
                "dataset.train_sources entries contain unsupported keys: "
                f"{unknown_keys}. Allowed keys: {sorted(_TRAIN_SOURCE_ALLOWED_KEYS)}."
            )

        provider_value = raw_source.get("provider")
        if not isinstance(provider_value, str):
            raise TypeError(
                "dataset.train_sources[].provider must be a str, got "
                f"{type(provider_value).__name__} at index {idx}."
            )
        if provider_value == "":
            raise ValueError(
                f"dataset.train_sources[{idx}].provider must be non-empty."
            )
        if provider_value.strip() != provider_value:
            raise ValueError(
                "dataset.train_sources[].provider must not include leading/trailing "
                f"whitespace. Got '{provider_value}'."
            )
        if "/" in provider_value:
            raise ValueError(
                "dataset.train_sources[].provider must not contain '/'. "
                f"Got '{provider_value}'."
            )
        provider = _validate_dataset_provider(provider_value)
        if provider in seen_providers:
            raise ValueError(
                "dataset.train_sources providers must be unique; got duplicate "
                f"provider '{provider}'."
            )
        seen_providers.add(provider)

        source_cfg = dict(inherited_defaults)
        for key in (
            "provider",
            "root",
            "dirname",
            "subset_tier",
            "test_subject",
            "test_session",
            "regime",
            "max_train_samples_per_subject",
        ):
            if key in raw_source:
                source_cfg[key] = raw_source[key]

        source_cfg["provider"] = provider
        source_cfg["preprocessor"] = _parse_optional_train_source_preprocessor(
            raw_source,
            source_provider=provider,
        )
        source_cfg["train_sample_fraction"] = _parse_train_source_sample_fraction(
            raw_source,
            source_provider=provider,
        )
        source_cfg["max_train_samples_per_subject"] = (
            _parse_optional_max_train_samples_per_subject(
                source_cfg,
                key_label=("dataset.train_sources[].max_train_samples_per_subject"),
            )
        )

        regime = source_cfg["regime"]
        _validate_provider_regime(dataset_provider=provider, dataset_regime=regime)
        _validate_subset_tier(source_cfg["subset_tier"], provider=provider)

        if not isinstance(source_cfg["root"], str):
            raise TypeError(
                "dataset.train_sources[].root must resolve to a str for source "
                f"provider '{provider}', got {type(source_cfg['root']).__name__}."
            )
        if not isinstance(source_cfg["dirname"], str):
            raise TypeError(
                "dataset.train_sources[].dirname must resolve to a str for source "
                f"provider '{provider}', got {type(source_cfg['dirname']).__name__}."
            )
        if not isinstance(source_cfg["label_mode"], str):
            raise TypeError(
                "dataset.label_mode must be a str before resolving train sources."
            )
        if not isinstance(source_cfg["coordinate_profile"], str):
            raise TypeError(
                "dataset.coordinate_profile must be a str before resolving train "
                "sources."
            )
        if not isinstance(source_cfg["task"], str):
            raise TypeError(
                "dataset.task must be a str before resolving train sources."
            )
        if not isinstance(source_cfg["test_subject"], int) or isinstance(
            source_cfg["test_subject"], bool
        ):
            raise TypeError(
                "dataset.train_sources[].test_subject must resolve to an int for "
                "provider "
                f"'{provider}', got {type(source_cfg['test_subject']).__name__}."
            )
        if not isinstance(source_cfg["test_session"], int) or isinstance(
            source_cfg["test_session"], bool
        ):
            raise TypeError(
                "dataset.train_sources[].test_session must resolve to an int for "
                "provider "
                f"'{provider}', got {type(source_cfg['test_session']).__name__}."
            )
        if not isinstance(
            source_cfg["uniquify_channel_ids_with_subject"], bool
        ) or not isinstance(source_cfg["uniquify_channel_ids_with_session"], bool):
            raise TypeError(
                "dataset.uniquify_channel_ids_with_subject and "
                "dataset.uniquify_channel_ids_with_session must be bool before "
                "resolving train sources."
            )

        normalized_sources.append(source_cfg)

    any_auto_train_source_cap = any(
        source_cfg["max_train_samples_per_subject"]
        == AUTO_MAX_TRAIN_SAMPLES_PER_SUBJECT
        for source_cfg in normalized_sources
    )
    if any_auto_train_source_cap and target_provider not in seen_providers:
        raise ValueError(
            "dataset.train_sources must include dataset.provider="
            f"'{target_provider}' when any max_train_samples_per_subject='auto' "
            "source is configured."
        )

    return normalized_sources


def _require_cfg_mapping(cfg: DictConfig, section: str):
    """Return one required top-level config section as a mapping-like object."""
    section_cfg = cfg.get(section)
    if section_cfg is None:
        raise ValueError(f"cfg.{section} is required and must be a mapping.")
    if not hasattr(section_cfg, "get"):
        raise TypeError(f"cfg.{section} must be a mapping/dict-like object.")
    return section_cfg


def _require_non_empty_cfg_str(section_cfg, *, section: str, key: str) -> str:
    value = section_cfg.get(key)
    if not isinstance(value, str):
        raise TypeError(f"{section}.{key} must be a str, got {type(value).__name__}.")
    if value == "":
        raise ValueError(f"{section}.{key} must be non-empty.")
    if value.strip() != value:
        raise ValueError(
            f"{section}.{key} must not include leading/trailing whitespace. "
            f"Got '{value}'."
        )
    return value


def _validate_preprocessor_config(cfg, context="preprocessor") -> set[str]:
    """Validate pipeline structure and return stage names for compatibility checks."""
    if not isinstance(cfg, (dict, DictConfig)):
        raise TypeError(f"{context} must be a mapping.")
    if "chain" in cfg:
        if "name" in cfg:
            raise ValueError(
                f"{context}: remove the redundant chain-level 'name'; name each stage instead."
            )
        chain = cfg["chain"]
        if not isinstance(chain, (list, ListConfig)) or not chain:
            raise ValueError(f"{context}.chain must be a non-empty list of stages.")
        names = set()
        for index, stage in enumerate(chain):
            names.update(
                _validate_preprocessor_config(stage, f"{context}.chain[{index}]")
            )
        return names
    name = _require_non_empty_cfg_str(cfg, section=context, key="name")
    if name not in PREPROCESSOR_REGISTRY:
        raise ValueError(f"{context}.name: unknown preprocessor '{name}'.")
    return {name}


def validate_eval_config(cfg: DictConfig) -> None:
    """Validate full eval config: dataset, model, runtime, submitter, runner.

    Call this once before the evaluation loop. After it returns, callers can
    trust that all cfg sections are well-formed and access them directly.
    """
    if "use_raw_data" in cfg or "raw_data_path" in cfg:
        raise ValueError(
            "Raw-data evaluation has been removed. "
            "Use processed dataset mode via dataset.* config."
        )
    # -- dataset --
    # Resolve once into a plain mapping to avoid accidental OmegaConf mutation
    # while applying strict validation checks.
    dataset_cfg = OmegaConf.to_container(cfg.get("dataset", {}), resolve=True) or {}
    if not isinstance(dataset_cfg, dict):
        raise TypeError("cfg.dataset must resolve to a mapping/dict.")

    validate_window_slicing_policy(
        dataset_cfg.get("window_slicing_policy", DEFAULT_WINDOW_SLICING_POLICY)
    )
    _require_non_empty_dataset_str(dataset_cfg, "root")
    _require_non_empty_dataset_str(dataset_cfg, "dirname")
    _require_non_empty_dataset_str(dataset_cfg, "task")
    _validate_label_mode(_require_non_empty_dataset_str(dataset_cfg, "label_mode"))
    _require_dataset_int(dataset_cfg, "test_subject")
    _require_dataset_int(dataset_cfg, "test_session")
    provider = _require_non_empty_dataset_str(dataset_cfg, "provider")
    regime = _require_non_empty_dataset_str(dataset_cfg, "regime")
    _validate_subset_tier(
        _require_non_empty_dataset_str(dataset_cfg, "subset_tier"),
        provider=provider,
    )
    _validate_provider_regime(dataset_provider=provider, dataset_regime=regime)
    _parse_required_bool(
        dataset_cfg.get("uniquify_channel_ids_with_subject"),
        "uniquify_channel_ids_with_subject",
    )
    _parse_required_bool(
        dataset_cfg.get("uniquify_channel_ids_with_session"),
        "uniquify_channel_ids_with_session",
    )
    _parse_required_bool(
        dataset_cfg.get("merge_val_into_test"),
        "merge_val_into_test",
    )
    train_same_subject_only = _parse_optional_dataset_train_same_subject_only(
        dataset_cfg
    )
    decodable_subject_sessions_only = (
        _parse_optional_dataset_decodable_subject_sessions_only(dataset_cfg)
    )
    _parse_optional_dataset_train_sample_fraction(dataset_cfg)
    _parse_optional_max_train_samples_per_subject(dataset_cfg)
    _parse_optional_dataset_coordinate_profile(dataset_cfg)
    dataset_brain_area_key = _parse_optional_dataset_brain_area_key(dataset_cfg)
    train_sources = resolve_train_source_configs(dataset_cfg)
    if decodable_subject_sessions_only:
        paths_cfg = cfg.get("paths", None)
        decodable_dir = (
            None
            if paths_cfg is None or not hasattr(paths_cfg, "get")
            else paths_cfg.get("decodable_subject_sessions_dir", None)
        )
        if not isinstance(decodable_dir, str) or not decodable_dir.strip():
            raise ValueError(
                "paths.decodable_subject_sessions_dir must be a non-empty str when "
                "dataset.decodable_subject_sessions_only=true."
            )
        if decodable_dir.strip() != decodable_dir:
            raise ValueError(
                "paths.decodable_subject_sessions_dir must not include "
                "leading/trailing whitespace."
            )
        validate_decodable_train_source_regimes(
            train_sources,
            enabled=decodable_subject_sessions_only,
        )
    if train_same_subject_only:
        if regime not in {"hold-in-session", "hold-out-session"}:
            raise ValueError(
                "dataset.train_same_subject_only=true is supported only for "
                "dataset.regime='hold-in-session' or 'hold-out-session'. "
                f"Got dataset.regime='{regime}'."
            )
        if provider not in {
            "neuroprobev2",
            "kelesbyd2024",
            "berezutskayapippi2022",
        }:
            raise ValueError(
                "dataset.train_same_subject_only=true is unsupported for "
                f"dataset.provider='{provider}'."
            )
    # -- model / channel compatibility --
    model_cfg = _require_cfg_mapping(cfg, "model")
    model_name = _require_non_empty_cfg_str(model_cfg, section="model", key="name")
    backend = _require_non_empty_cfg_str(model_cfg, section="model", key="backend")
    if backend not in {"sklearn", "torch"}:
        raise ValueError(f"model.backend must be sklearn or torch, got {backend!r}.")
    if backend == "torch":
        # Reject training-setting typos before any fold is built.
        training_mode = model_cfg.get("training_mode", "epoch_based")
        if not isinstance(training_mode, str):
            raise TypeError("model.training_mode must be a str.")
        if training_mode not in {"epoch_based", "steps_based"}:
            raise ValueError(
                "model.training_mode must be one of ['epoch_based', 'steps_based'], "
                f"got {training_mode!r}."
            )
        optimizer_name = model_cfg.get("optimizer", "Adam")
        if not isinstance(optimizer_name, str):
            raise TypeError("model.optimizer must be a str.")
        optimizer_names = sorted(
            name
            for name, cls in vars(optim).items()
            if isinstance(cls, type)
            and issubclass(cls, optim.Optimizer)
            and cls is not optim.Optimizer
        )
        if optimizer_name not in optimizer_names:
            raise ValueError(
                f"model.optimizer must be one of {optimizer_names}, "
                f"got {optimizer_name!r}."
            )
    requires_aligned = model_cfg.get("requires_aligned_channels")
    if not isinstance(requires_aligned, bool):
        raise TypeError(
            "model.requires_aligned_channels must be a bool, got "
            f"{type(requires_aligned).__name__}."
        )
    requires_coords = model_cfg.get("requires_coords")
    if not isinstance(requires_coords, bool):
        raise TypeError(
            "model.requires_coords must be a bool, got "
            f"{type(requires_coords).__name__}."
        )
    if train_sources and requires_aligned:
        raise ValueError(
            "dataset.train_sources is unsupported when "
            "model.requires_aligned_channels=true. Multi-source training "
            "currently supports only non-aligned variable-channel models."
        )
    if train_sources and model_name == "diver":
        raise ValueError(
            "model=diver is unsupported with dataset.train_sources because DIVER "
            "input scaling is provider-specific and multi-source batches do not "
            "currently carry a per-sample provider scaling contract."
        )

    # Derive region-intersection pooling need from (dataset, regime, model).
    pool = needs_region_intersection_pool(provider, regime, requires_aligned)
    if pool and dataset_brain_area_key is None:
        raise ValueError(
            "dataset.brain_area_key is required when the regime is multi-subject "
            f"and model.requires_aligned_channels is true "
            f"(dataset.provider='{provider}', dataset.regime='{regime}')."
        )

    # -- preprocessor --
    preprocessor_cfg = _require_cfg_mapping(cfg, "preprocessor")
    stage_names = _validate_preprocessor_config(preprocessor_cfg)
    if pool and "region_intersection_pool" not in stage_names:
        raise ValueError(
            "Multi-subject regime with model.requires_aligned_channels=true "
            "requires either a top-level 'region_intersection_pool' preprocessor "
            "or a preprocessor chain that includes the "
            "'region_intersection_pool' stage "
            f"(dataset.provider='{provider}', dataset.regime='{regime}')."
        )

    # -- runtime --
    runtime_cfg = _require_cfg_mapping(cfg, "runtime")
    if "deterministic" in cfg.model:
        raise ValueError("model.deterministic has moved to runtime.deterministic.")
    if not isinstance(runtime_cfg.get("deterministic", True), bool):
        raise TypeError("runtime.deterministic must be a bool.")
    if "seed" not in runtime_cfg:
        raise ValueError("runtime.seed is required.")
    seed = runtime_cfg.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError(f"runtime.seed must be an int, got {type(seed).__name__}.")
    overwrite = runtime_cfg.get("overwrite")
    if not isinstance(overwrite, bool):
        raise TypeError(
            f"runtime.overwrite must be a bool, got {type(overwrite).__name__}."
        )
    train_source_cache_enabled = runtime_cfg.get("train_source_cache_enabled", False)
    if not isinstance(train_source_cache_enabled, bool):
        raise TypeError(
            "runtime.train_source_cache_enabled must be a bool, got "
            f"{type(train_source_cache_enabled).__name__}."
        )
    train_source_cache_dir = runtime_cfg.get("train_source_cache_dir", None)
    if train_source_cache_dir is not None:
        if not isinstance(train_source_cache_dir, str):
            raise TypeError(
                "runtime.train_source_cache_dir must be a str when set, got "
                f"{type(train_source_cache_dir).__name__}."
            )
        if not train_source_cache_dir.strip():
            raise ValueError(
                "runtime.train_source_cache_dir must be non-empty when set."
            )
    preprocessed_split_cache_enabled = runtime_cfg.get(
        "preprocessed_split_cache_enabled", False
    )
    if not isinstance(preprocessed_split_cache_enabled, bool):
        raise TypeError(
            "runtime.preprocessed_split_cache_enabled must be a bool when set, got "
            f"{type(preprocessed_split_cache_enabled).__name__}."
        )
    preprocessed_split_cache_dir = runtime_cfg.get("preprocessed_split_cache_dir", None)
    if preprocessed_split_cache_dir is not None:
        if not isinstance(preprocessed_split_cache_dir, str):
            raise TypeError(
                "runtime.preprocessed_split_cache_dir must be a str when set, got "
                f"{type(preprocessed_split_cache_dir).__name__}."
            )
        if not preprocessed_split_cache_dir.strip():
            raise ValueError(
                "runtime.preprocessed_split_cache_dir must be non-empty when set."
            )
    preprocessed_split_cache_mode = runtime_cfg.get(
        "preprocessed_split_cache_mode", "read_write"
    )
    if not isinstance(preprocessed_split_cache_mode, str):
        raise TypeError(
            "runtime.preprocessed_split_cache_mode must be a str when set, got "
            f"{type(preprocessed_split_cache_mode).__name__}."
        )
    if preprocessed_split_cache_mode not in {"read_only", "read_write", "refresh"}:
        raise ValueError(
            "runtime.preprocessed_split_cache_mode must be one of "
            "read_only, read_write, refresh; got "
            f"'{preprocessed_split_cache_mode}'."
        )

    # Attribution is optional; preserve legacy export keys without requiring identity.
    if cfg.get("submitter") is not None:
        submitter_cfg = _require_cfg_mapping(cfg, "submitter")
        for key in ("author", "organization", "organization_url"):
            value = submitter_cfg.get(key)
            if value is not None and not isinstance(value, str):
                raise TypeError(f"submitter.{key} must be a str or null.")
