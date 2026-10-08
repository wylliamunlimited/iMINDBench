"""Processed dataset adapter for variable-channel runs."""

from __future__ import annotations

import bisect
import hashlib
import itertools
import json
import pickle
import re
import shutil
import sys
import tempfile
import time
import uuid
import zlib
from collections.abc import Iterable
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

from imindbench.preprocessors import build_preprocessor, describe_preprocessor
from imindbench.utils import regression as regression_utils
from imindbench.utils.logging_utils import log, log_fold_split_sample_counts
from imindbench.utils.pipeline_contracts import (
    AUTO_MAX_TRAIN_SAMPLES_PER_SUBJECT,
    build_processed_split_provider,
    get_dataset_class,
    needs_region_intersection_pool,
    resolve_train_source_configs,
    validate_decodable_train_source_regimes,
)
from imindbench.utils.window_slicing import (
    DEFAULT_WINDOW_SLICING_POLICY,
    read_recording_window,
    validate_window_slicing_policy,
)


def _normalize_brain_area_array(
    values: Any,
    *,
    expected_len: int,
    context: str,
) -> np.ndarray:
    arr = np.asarray(values, dtype=object).reshape(-1)
    if len(arr) != expected_len:
        raise ValueError(f"{context} must have length {expected_len}, got {len(arr)}.")
    out: list[str] = []
    for value in arr:
        if value is None:
            raise ValueError(f"{context} contains None labels.")
        if isinstance(value, (float, np.floating)) and np.isnan(value):
            raise ValueError(f"{context} contains NaN labels.")
        if isinstance(value, (bytes, np.bytes_)):
            # HDF5 string datasets read back as bytes. str() on those yields the
            # literal "b'Left ACC'", which matches no atlas entry, so decode
            # instead. KelesBYD2024's `location` key is stored this way.
            try:
                label = bytes(value).decode("utf-8").strip()
            except UnicodeDecodeError as exc:
                raise ValueError(f"{context} contains non-utf8 byte labels.") from exc
            if not label:
                # Preserve empty byte labels as unmatched atlas entries instead of
                # rejecting the unlocalized channel after decoding.
                label = str(value).strip()
        else:
            label = str(value).strip()
        if not label:
            raise ValueError(f"{context} contains empty labels.")
        out.append(label)
    return np.asarray(out, dtype=object)


def identity(coords: Any) -> np.ndarray:
    return np.asarray(coords, dtype=np.float32).reshape(-1, 3)


def byd_mni152_ras_to_popt_lip(coords: Any) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float32).reshape(-1, 3)
    left = -coords[:, 0] + 128.0
    inferior = -coords[:, 2] + 110.0
    posterior = -coords[:, 1] + 128.0
    return np.stack((left, inferior, posterior), axis=1).astype(np.float32, copy=False)


def pippi_acpc_to_popt_lip(coords: Any) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float32).reshape(-1, 3)
    left = (-0.9610 * coords[:, 0]) + 127.5748
    inferior = (-1.0179 * coords[:, 2]) + 112.7481
    posterior = (-0.9173 * coords[:, 1]) + 130.1209
    return np.stack((left, inferior, posterior), axis=1).astype(np.float32, copy=False)


def no_coordinates(coords: Any) -> None:
    _ = coords
    return None


def zero_coordinates(coords: Any) -> np.ndarray:
    """Place every channel at the origin.

    Used by the popt_zero profile so models that require coordinates (PopT,
    BaRISTA) can run on recordings that have no usable positions. Every
    channel gets the same position, so the model learns nothing from it.
    """
    coords = np.asarray(coords, dtype=np.float32).reshape(-1, 3)
    return np.zeros_like(coords)


COORDINATE_PROFILES: dict[str, dict[str, tuple[str, Any]]] = {
    "popt_lip": {
        "millerecog2019": ("talairach", no_coordinates),
        "kelesbyd2024": ("byd_mni152_ras", byd_mni152_ras_to_popt_lip),
        "berezutskayapippi2022": ("acpc", pippi_acpc_to_popt_lip),
        "neuroprobe2025": ("btb_lip", identity),
        "neuroprobev2": ("btb_lip", identity),
    },
    "diver_mni": {
        "millerecog2019": ("talairach", no_coordinates),
        "kelesbyd2024": ("byd_mni152_ras", no_coordinates),
        "berezutskayapippi2022": ("acpc", no_coordinates),
        "neuroprobe2025": ("btb_xyz", identity),
        "neuroprobev2": ("btb_xyz", identity),
    },
    # The profiles below exist for millerecog2019 only. Selecting one of them
    # for another dataset fails with "has no mapping for provider".
    #
    # popt_zero: every channel at the origin. This is the Miller default, so
    # PopT and BaRISTA run on the build without positions.
    "popt_zero": {
        "millerecog2019": ("talairach", zero_coordinates),
    },
    # popt_miller: real positions for PopT and BaRISTA. Needs the Miller build
    # with MNI152 positions. mni152_strict marks low-quality positions as NaN,
    # and channels with NaN positions are dropped, so pair this profile with
    # the positions cell lists. Miller's MNI152 millimetres are the same frame
    # KelesBYD2024 uses, so the BYD transform applies unchanged.
    "popt_miller": {
        "millerecog2019": ("mni152_strict", byd_mni152_ras_to_popt_lip),
    },
    # diver_mni_miller: the same MNI152 positions, passed to DIVER unchanged.
    "diver_mni_miller": {
        "millerecog2019": ("mni152_strict", identity),
    },
}


# Providers emit a flat `coords` array plus a short `coords_type` tag rather
# than the `coordinate_frames` dict the profile table is keyed by. This maps
# the tag onto the canonical frame name so the profile's transform can be
# applied. Keys are compared lowercased.
#   lip / btb_lip / popt_lip -> Brain TreeBank L/I/P voxel indices
#   xyz                      -> KelesBYD2024 MNI152 RAS millimetres
#   acpc                     -> BerezutskayaPippi2022 ACPC millimetres
LEGACY_COORDS_TYPE_TO_FRAME: dict[str, str] = {
    "lip": "btb_lip",
    "btb_lip": "btb_lip",
    "popt_lip": "btb_lip",
    "xyz": "byd_mni152_ras",
    "acpc": "acpc",
}


# Audited (profile, declared frame, provider frame) exceptions. DIVER's published
# Brain TreeBank results used raw LIP coordinates despite the nonexistent btb_xyz
# declaration, so this no-transform exception preserves them.
LEGACY_FRAME_PASSTHROUGH: frozenset[tuple[str, str, str]] = frozenset(
    {
        ("diver_mni", "btb_xyz", "btb_lip"),
    }
)


def resolve_coordinate_profile(value: Any) -> str:
    if value is None:
        return "popt_lip"
    if not isinstance(value, str):
        raise TypeError(
            "dataset.coordinate_profile must be a str when set, got "
            f"{type(value).__name__}."
        )
    if value == "":
        raise ValueError("dataset.coordinate_profile must be non-empty when set.")
    if value.strip() != value:
        raise ValueError(
            "dataset.coordinate_profile must not include leading/trailing "
            f"whitespace. Got '{value}'."
        )
    if value not in COORDINATE_PROFILES:
        raise ValueError(
            "dataset.coordinate_profile must be one of "
            f"{sorted(COORDINATE_PROFILES)}, got '{value}'."
        )
    return value


def _resolve_coordinate_transform(
    *,
    provider_key: str,
    coordinate_profile: str,
) -> tuple[str, Any]:
    provider_profiles = COORDINATE_PROFILES[coordinate_profile]
    try:
        return provider_profiles[provider_key]
    except KeyError as exc:
        raise ValueError(
            f"Coordinate profile '{coordinate_profile}' has no mapping for provider "
            f"'{provider_key}'."
        ) from exc


def _validate_provider_key(provider_key: Any) -> str:
    if not isinstance(provider_key, str):
        raise TypeError(
            f"provider_key must be a str, got {type(provider_key).__name__}."
        )
    if provider_key.strip() != provider_key:
        raise ValueError(
            "provider_key must not include leading/trailing whitespace. "
            f"Got '{provider_key}'."
        )
    if provider_key == "":
        raise ValueError("provider_key must be non-empty.")
    return provider_key


def _derive_byd_region_codes(
    channel_names: list[str],
    *,
    context: str,
) -> np.ndarray:
    """Derive BYD region codes from canonical electrode names like LOFC1."""
    region_codes: list[str] = []
    for idx, name in enumerate(channel_names):
        label = str(name).strip()
        if len(label) < 4:
            raise ValueError(
                f"{context} contains channel label '{label}' at index {idx} that is "
                "too short to derive a BYD region code."
            )
        region_code = label[1:4].upper()
        if not region_code.strip():
            raise ValueError(
                f"{context} contains channel label '{label}' at index {idx} with an "
                "empty BYD region code slice."
            )
        region_codes.append(region_code)
    return np.asarray(region_codes, dtype=object)


def _iter_preprocessor_stages(preprocessor: Any) -> list[Any]:
    """Return preprocessors to profile for variable-channel materialization."""
    stages = getattr(preprocessor, "preprocessors", None)
    # Composite preprocessors expose an explicit stage list; legacy preprocessors
    # are treated as a single stage.
    if isinstance(stages, (list, tuple)) and stages:
        return list(stages)
    return [preprocessor]


_PREPROCESSOR_CONFIG_DIR = (
    Path(__file__).resolve().parent.parent / "conf" / "preprocessor"
)


def _build_preprocessor_from_name(name: str):
    cfg_path = _PREPROCESSOR_CONFIG_DIR / f"{name}.yaml"
    if not cfg_path.exists():
        raise ValueError(
            f"Unknown preprocessor config '{name}'. Expected file at '{cfg_path}'."
        )
    return build_preprocessor(OmegaConf.load(cfg_path))


def _build_fresh_preprocessor(
    *,
    preprocessor_name: str | None,
    preprocessor_cfg: Any | None,
    preprocessor: Any | None,
):
    if preprocessor_name is not None:
        return _build_preprocessor_from_name(preprocessor_name)
    if preprocessor_cfg is not None:
        return build_preprocessor(deepcopy(preprocessor_cfg))
    if preprocessor is not None:
        return deepcopy(preprocessor)
    return None


_TRAIN_SOURCE_CACHE_VERSION = 5
_PREPROCESSED_SPLIT_CACHE_VERSION = 2
_PREPROCESSED_SPLIT_CACHE_MODES = {"read_only", "read_write", "refresh"}


def _serialize_preprocessor_cache_identity(
    *,
    preprocessor_name: str | None,
    preprocessor_cfg: Any | None,
    preprocessor: Any | None,
) -> Any:
    # Cache identity should change when the effective preprocessor config changes,
    # even if the public preprocessor name stays the same.
    if preprocessor_name is not None:
        cfg_path = _PREPROCESSOR_CONFIG_DIR / f"{preprocessor_name}.yaml"
        if cfg_path.exists():
            return {
                "name": str(preprocessor_name),
                "config_text": cfg_path.read_text(encoding="utf-8"),
            }
        return {"name": str(preprocessor_name)}
    if preprocessor_cfg is not None:
        if OmegaConf.is_config(preprocessor_cfg):
            return OmegaConf.to_container(preprocessor_cfg, resolve=True)
        if isinstance(preprocessor_cfg, dict):
            return deepcopy(preprocessor_cfg)
    cfg = getattr(preprocessor, "cfg", None)
    if cfg is not None:
        if OmegaConf.is_config(cfg):
            return OmegaConf.to_container(cfg, resolve=True)
        if isinstance(cfg, dict):
            return deepcopy(cfg)
    return None


def _normalize_train_source_cache_enabled(value: Any) -> bool:
    if value is None:
        return False
    if not isinstance(value, bool):
        raise TypeError(
            "runtime.train_source_cache_enabled must be a bool when set, got "
            f"{type(value).__name__}."
        )
    return bool(value)


def _normalize_train_source_cache_dir(value: Any) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(
            "runtime.train_source_cache_dir must be a str when set, got "
            f"{type(value).__name__}."
        )
    normalized = value.strip()
    if not normalized:
        raise ValueError("runtime.train_source_cache_dir must be non-empty when set.")
    return Path(normalized)


def _resolve_train_source_cache_dir(
    *,
    cache_enabled: bool,
    cache_dir: Any,
    dataset_root: str,
) -> Path | None:
    if not _normalize_train_source_cache_enabled(cache_enabled):
        return None
    normalized = _normalize_train_source_cache_dir(cache_dir)
    if normalized is not None:
        return normalized
    return Path(dataset_root) / ".neuroprobe_eval_train_source_cache"


def _normalize_preprocessed_split_cache_enabled(value: Any) -> bool:
    if value is None:
        return False
    if not isinstance(value, bool):
        raise TypeError(
            "runtime.preprocessed_split_cache_enabled must be a bool when set, got "
            f"{type(value).__name__}."
        )
    return bool(value)


def _normalize_preprocessed_split_cache_dir(value: Any) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(
            "runtime.preprocessed_split_cache_dir must be a str when set, got "
            f"{type(value).__name__}."
        )
    normalized = value.strip()
    if not normalized:
        raise ValueError(
            "runtime.preprocessed_split_cache_dir must be non-empty when set."
        )
    return Path(normalized)


def _normalize_preprocessed_split_cache_mode(value: Any) -> str:
    if value is None:
        return "read_write"
    if not isinstance(value, str):
        raise TypeError(
            "runtime.preprocessed_split_cache_mode must be a str when set, got "
            f"{type(value).__name__}."
        )
    if value not in _PREPROCESSED_SPLIT_CACHE_MODES:
        allowed = ", ".join(sorted(_PREPROCESSED_SPLIT_CACHE_MODES))
        raise ValueError(
            "runtime.preprocessed_split_cache_mode must be one of "
            f"{allowed}; got '{value}'."
        )
    return value


def _resolve_preprocessed_split_cache_dir(
    *,
    cache_enabled: bool,
    cache_dir: Any,
    dataset_root: str,
) -> Path | None:
    if not _normalize_preprocessed_split_cache_enabled(cache_enabled):
        return None
    normalized = _normalize_preprocessed_split_cache_dir(cache_dir)
    if normalized is not None:
        return normalized
    return Path(dataset_root) / ".neuroprobe_eval_preprocessed_split_cache"


def _build_train_source_cache_identity(
    *,
    source_cfg: dict[str, Any],
    source_dataset: WindowedNeuroprobeSplitDataset,
    preprocessor_identity: Any,
    fold_idx: int,
    require_coords: bool,
) -> dict[str, Any]:
    split_summary = source_dataset.get_split_summary()
    # Cache identity is derived from the actual source split payload instead of
    # the requested test target so hold-in train sources can be reused across
    # repeated eval cycles that land on the same effective train recordings.
    return {
        "cache_version": _TRAIN_SOURCE_CACHE_VERSION,
        "window_slicing_policy": source_dataset.window_slicing_policy,
        "provider": str(source_cfg["provider"]),
        "root": str(source_cfg["root"]),
        "dirname": str(source_cfg["dirname"]),
        "subset_tier": str(source_cfg["subset_tier"]),
        "label_mode": str(source_cfg["label_mode"]),
        "task": str(source_cfg["task"]),
        "regime": str(source_cfg["regime"]),
        "coordinate_profile": str(
            source_cfg.get("coordinate_profile", source_dataset.coordinate_profile)
        ),
        "test_subject": int(source_cfg["test_subject"]),
        "test_session": int(source_cfg["test_session"]),
        "split": str(split_summary["split"]),
        "fold_idx": int(fold_idx),
        "recording_ids": list(split_summary["recording_ids"]),
        "n_samples": int(split_summary["n_samples"]),
        "n_samples_before_subsampling": int(
            split_summary["n_samples_before_subsampling"]
        ),
        "sample_fraction": float(split_summary["sample_fraction"]),
        "sample_seed": split_summary["sample_seed"],
        "max_samples_per_subject": split_summary.get("max_samples_per_subject"),
        "uniquify_channel_ids_with_subject": bool(
            source_cfg["uniquify_channel_ids_with_subject"]
        ),
        "uniquify_channel_ids_with_session": bool(
            source_cfg["uniquify_channel_ids_with_session"]
        ),
        "require_coords": bool(require_coords),
        "preprocessor": preprocessor_identity,
    }


def _train_source_cache_path(
    *,
    cache_dir: Path,
    identity: dict[str, Any],
) -> Path:
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    provider = str(identity["provider"])
    return cache_dir / provider / f"{digest}.pkl"


def _optional_task_mode_identity(dataset_cfg: Any) -> dict[str, Any]:
    """Cache-id entries for options that change the windows or their labels."""
    identity: dict[str, Any] = {}
    class_pair = _dataset_cfg_get(dataset_cfg, "class_pair", None)
    if class_pair is not None:
        identity["class_pair"] = [int(label) for label in class_pair]
    indices_file = _dataset_cfg_get(dataset_cfg, "train_sample_indices_file", None)
    if indices_file:
        identity["train_sample_indices"] = [
            str(indices_file),
            str(_dataset_cfg_get(dataset_cfg, "train_sample_indices_frac", None)),
            str(_dataset_cfg_get(dataset_cfg, "train_sample_indices_draw", None)),
            str(_dataset_cfg_get(dataset_cfg, "train_sample_indices_order_seed", None)),
        ]
    return identity


def _build_preprocessed_split_cache_identity(
    *,
    dataset_cfg: Any,
    split_datasets: dict[str, Any],
    preprocessor_identity: Any,
    fold_idx: int,
    needs_pool: bool,
    brain_area_key: str | None,
) -> dict[str, Any]:
    """Build the model-independent identity for one prepared fold cache entry."""
    return {
        "cache_version": _PREPROCESSED_SPLIT_CACHE_VERSION,
        "window_slicing_policy": split_datasets["train"].window_slicing_policy,
        "provider": str(dataset_cfg.provider),
        "root": str(dataset_cfg.root),
        "dirname": str(dataset_cfg.dirname),
        "subset_tier": str(dataset_cfg.subset_tier),
        "label_mode": str(dataset_cfg.label_mode),
        # Added only when set, so the cache ids of existing runs do not change.
        **_optional_task_mode_identity(dataset_cfg),
        "task": str(dataset_cfg.task),
        "regime": str(dataset_cfg.regime),
        "coordinate_profile": str(
            _dataset_cfg_get(dataset_cfg, "coordinate_profile", "popt_lip")
        ),
        "test_subject": int(dataset_cfg.test_subject),
        "test_session": int(dataset_cfg.test_session),
        "fold_idx": int(fold_idx),
        "uniquify_channel_ids_with_subject": bool(
            dataset_cfg.uniquify_channel_ids_with_subject
        ),
        "uniquify_channel_ids_with_session": bool(
            dataset_cfg.uniquify_channel_ids_with_session
        ),
        "train_same_subject_only": bool(
            _dataset_cfg_get(dataset_cfg, "train_same_subject_only", False)
        ),
        "train_sample_fraction": float(
            _dataset_cfg_get(dataset_cfg, "train_sample_fraction", 1.0)
        ),
        "max_train_samples_per_subject": _dataset_cfg_get(
            dataset_cfg, "max_train_samples_per_subject", None
        ),
        "decodable_subject_sessions_only": bool(
            _dataset_cfg_get(
                dataset_cfg,
                "decodable_subject_sessions_only",
                False,
            )
        ),
        "needs_region_intersection_pool": bool(needs_pool),
        "brain_area_key": brain_area_key,
        "preprocessor": preprocessor_identity,
        "split_summaries": {
            split: split_datasets[split].get_split_summary()
            for split in ("train", "val", "test")
        },
    }


def _preprocessed_split_cache_path(
    *,
    cache_dir: Path,
    identity: dict[str, Any],
) -> Path:
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    provider = str(identity["provider"])
    return cache_dir / provider / digest


def _load_preprocessed_split_cache(cache_path: Path) -> dict[str, Any] | None:
    manifest_path = cache_path / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if int(manifest.get("cache_version", -1)) != _PREPROCESSED_SPLIT_CACHE_VERSION:
            return None
        with (cache_path / "preprocess_state.pkl").open("rb") as handle:
            preprocess_state = pickle.load(handle)
        split_samples = {}
        for split in ("train", "val", "test"):
            with (cache_path / f"{split}.pkl").open("rb") as handle:
                samples = pickle.load(handle)
            if not isinstance(samples, list):
                log(
                    f"Preprocessed split cache payload at '{cache_path}' has "
                    f"non-list samples for split '{split}'. Rebuilding cache entry.",
                    priority=1,
                )
                return None
            split_samples[split] = samples
    except Exception as exc:
        log(
            f"Preprocessed split cache read failed at '{cache_path}': {exc}. "
            "Rebuilding cache entry.",
            priority=1,
        )
        return None
    return {
        "identity": manifest.get("identity"),
        "preprocess_state": preprocess_state,
        "samples": split_samples,
    }


def _save_preprocessed_split_cache(
    cache_path: Path,
    *,
    identity: dict[str, Any],
    preprocess_state: Any,
    split_datasets: dict[str, Any],
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(
        tempfile.mkdtemp(
            dir=cache_path.parent,
            prefix=f"{cache_path.name}.",
            suffix=".tmp",
        )
    )
    try:
        with (tmp_dir / "preprocess_state.pkl").open("wb") as handle:
            pickle.dump(preprocess_state, handle, protocol=pickle.HIGHEST_PROTOCOL)
        for split in ("train", "val", "test"):
            with (tmp_dir / f"{split}.pkl").open("wb") as handle:
                pickle.dump(
                    list(split_datasets[split]),
                    handle,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
        manifest = {
            "cache_version": _PREPROCESSED_SPLIT_CACHE_VERSION,
            "identity": identity,
            "splits": {
                split: {
                    "num_samples": len(split_datasets[split]),
                    "summary": split_datasets[split].get_split_summary(),
                }
                for split in ("train", "val", "test")
            },
        }
        with (tmp_dir / "manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, sort_keys=True, indent=2)
        backup_path = None
        if cache_path.exists():
            backup_path = cache_path.with_name(
                f"{cache_path.name}.{uuid.uuid4().hex}.bak"
            )
            cache_path.rename(backup_path)
        try:
            tmp_dir.replace(cache_path)
        except Exception:
            if (
                backup_path is not None
                and backup_path.exists()
                and not cache_path.exists()
            ):
                backup_path.rename(cache_path)
            raise
        if backup_path is not None and backup_path.exists():
            shutil.rmtree(backup_path)
    except Exception:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        raise


def _load_train_source_cache(cache_path: Path) -> dict[str, Any] | None:
    if not cache_path.exists():
        return None
    try:
        with cache_path.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception as exc:
        log(
            f"Train-source cache read failed at '{cache_path}': {exc}. "
            "Rebuilding cache entry.",
            priority=1,
        )
        return None
    if not isinstance(payload, dict):
        log(
            f"Train-source cache payload at '{cache_path}' is not a dict. "
            "Rebuilding cache entry.",
            priority=1,
        )
        return None
    if int(payload.get("cache_version", -1)) != _TRAIN_SOURCE_CACHE_VERSION:
        return None
    samples = payload.get("samples")
    if not isinstance(samples, list):
        log(
            f"Train-source cache payload at '{cache_path}' is missing a sample list. "
            "Rebuilding cache entry.",
            priority=1,
        )
        return None
    return payload


def _save_train_source_cache(
    cache_path: Path,
    *,
    identity: dict[str, Any],
    preprocess_state: Any,
    samples: list[dict[str, Any]],
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "cache_version": _TRAIN_SOURCE_CACHE_VERSION,
        "identity": identity,
        "preprocess_state": preprocess_state,
        "samples": samples,
    }
    # Use a unique temp file per writer so concurrent jobs targeting the same
    # cache key do not trample each other's in-progress writes.
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=cache_path.parent,
        prefix=f"{cache_path.stem}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        tmp_path = Path(handle.name)
    tmp_path.replace(cache_path)


def _infer_split_feature_shape(
    split_dataset: torch.utils.data.Dataset,
    *,
    split_label: str,
) -> tuple[int, ...] | None:
    expected_shape: tuple[int, ...] | None = None
    for sample_idx in range(len(split_dataset)):
        sample = split_dataset[sample_idx]
        x = np.asarray(sample["x"])
        if x.ndim < 2:
            raise ValueError(
                f"{split_label} sample['x'] must be at least 2D, got {x.shape}."
            )
        feature_shape = tuple(int(dim) for dim in x.shape[1:])
        if expected_shape is None:
            expected_shape = feature_shape
            continue
        if feature_shape != expected_shape:
            raise ValueError(
                f"{split_label} has inconsistent post-preprocessor feature shapes: "
                f"expected {expected_shape}, got {feature_shape} at sample "
                f"{sample_idx}."
            )
    return expected_shape


def _validate_train_source_feature_shapes(
    train_source_datasets: list[WindowedNeuroprobeSplitDataset],
    *,
    train_sources: list[dict[str, Any]],
) -> dict[str, tuple[int, ...] | None]:
    per_source_shapes: dict[str, tuple[int, ...] | None] = {}
    expected_shape: tuple[int, ...] | None = None
    expected_provider: str | None = None
    for source_cfg, source_dataset in zip(
        train_sources, train_source_datasets, strict=False
    ):
        provider = str(source_cfg["provider"])
        source_shape = _infer_split_feature_shape(
            source_dataset,
            split_label=f"train source '{provider}'",
        )
        per_source_shapes[provider] = source_shape
        if source_shape is None:
            continue
        if expected_shape is None:
            expected_shape = source_shape
            expected_provider = provider
            continue
        if source_shape != expected_shape:
            raise ValueError(
                "Post-preprocessor feature shape mismatch across train sources: "
                f"provider '{expected_provider}' produced {expected_shape}, "
                f"provider '{provider}' produced {source_shape}. Configure "
                "train_sources[].preprocessor so all train sources emit the same "
                "per-channel feature shape."
            )
    return per_source_shapes


_SUBJECT_FROM_RECORDING_ID_PATTERNS: dict[str, re.Pattern[str]] = {
    "neuroprobev2": re.compile(r"^sub_(?P<subject>\d+)_trial\d{3}$"),
    "neuroprobe2025": re.compile(r"^sub_(?P<subject>\d+)_trial\d{3}$"),
    "kelesbyd2024": re.compile(r"^sub-CS(?P<subject>\d+)_ses-P\d+CSR\d+$"),
    "berezutskayapippi2022": re.compile(
        r"^sub-(?P<subject>\d+)_ses-iemu_task-film_acq-[A-Za-z0-9]+_run-\d+$"
    ),
}
_SUBJECT_SESSION_FROM_RECORDING_ID_PATTERNS: dict[str, re.Pattern[str]] = {
    "neuroprobev2": re.compile(r"^sub_(?P<subject>\d+)_trial(?P<session>\d+)$"),
    "neuroprobe2025": re.compile(r"^sub_(?P<subject>\d+)_trial(?P<session>\d+)$"),
    "kelesbyd2024": re.compile(r"^sub-CS(?P<subject>\d+)_ses-P\d+CSR(?P<session>\d+)$"),
    "berezutskayapippi2022": re.compile(
        r"^sub-(?P<subject>\d+)_ses-iemu_task-film_acq-[A-Za-z0-9]+_run-"
        r"(?P<session>\d+)$"
    ),
}
_MANIFEST_SUBJECT_SESSION_RE = re.compile(r"^sub(?P<subject>\d+)_sess(?P<session>\d+)$")


def _dataset_cfg_get(dataset_cfg: Any, key: str, default: Any) -> Any:
    getter = getattr(dataset_cfg, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(dataset_cfg, key, default)


def _normalize_decodable_subject_sessions_only(value: Any) -> bool:
    if value is None:
        return False
    if not isinstance(value, bool):
        raise TypeError(
            "dataset.decodable_subject_sessions_only must be a bool when set, "
            f"got {type(value).__name__}."
        )
    return bool(value)


def _normalize_train_same_subject_only(value: Any) -> bool:
    if value is None:
        return False
    if not isinstance(value, bool):
        raise TypeError(
            "dataset.train_same_subject_only must be a bool when set, got "
            f"{type(value).__name__}."
        )
    return bool(value)


def _normalize_train_sample_fraction(value: Any) -> float:
    if value is None:
        return 1.0
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(
            "dataset.train_sample_fraction must be a float in (0, 1], got "
            f"{type(value).__name__}."
        )
    fraction = float(value)
    if not np.isfinite(fraction):
        raise ValueError("dataset.train_sample_fraction must be finite when set.")
    if fraction <= 0.0 or fraction > 1.0:
        raise ValueError(
            f"dataset.train_sample_fraction must be in (0, 1], got {fraction}."
        )
    return fraction


def _normalize_max_train_samples_per_subject(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(
            "max_samples_per_subject must be an int when set, got "
            f"{type(value).__name__}."
        )
    max_samples = int(value)
    if max_samples < 1:
        raise ValueError("max_samples_per_subject must be >= 1 when set.")
    return max_samples


def _normalize_max_train_samples_per_subject_setting(
    value: Any,
) -> int | str | None:
    if value is None:
        return None
    if isinstance(value, str):
        if value == AUTO_MAX_TRAIN_SAMPLES_PER_SUBJECT:
            return AUTO_MAX_TRAIN_SAMPLES_PER_SUBJECT
        raise TypeError(
            "max_train_samples_per_subject must be an int or 'auto' when set, got "
            f"str value '{value}'."
        )
    return _normalize_max_train_samples_per_subject(value)


_MILLER_RECORDING_ID_RE = re.compile(
    r"^sub-(?P<subject>[a-z]{2})_set-(?P<task_set>[a-z][a-z0-9_]*)$"
)


def split_miller_recording_id(recording_id: str) -> tuple[str, str]:
    """Split a Miller recording id into (subject code, task set).

    Read here rather than with the dataset module's own pattern, because that
    pattern (brainsets fork, miller-ecog-modularize) does not allow digits and
    so rejects the regression task sets, for example fingerflex_reg_w1000.
    """
    match = _MILLER_RECORDING_ID_RE.fullmatch(
        _strip_optional_source_prefix(recording_id)
    )
    if match is None:
        raise ValueError(f"Invalid MillerECoG2019 recording_id '{recording_id}'.")
    return match.group("subject"), match.group("task_set")


def _miller_subject_session(recording_id: str) -> tuple[int, int]:
    """Turn a Miller recording id into (subject number, session number).

    Miller ids use letter codes, for example ``sub-bp_set-motor_basic``. The
    dataset module turns the subject code into its position in the list of
    29 subject codes and the task set into its position in the list of 25
    task sets, both counted from 1. The task set plays the role of the
    session.
    """
    dataset_cls = get_dataset_class("millerecog2019")
    module = sys.modules[dataset_cls.__module__]
    code, task_set = split_miller_recording_id(recording_id)
    return module.subject_number_for(code), module.session_number_for(task_set)


# Datasets whose recording ids are not numbers that a regular expression can
# read. Each parser returns (subject, session) and is checked before the
# regular-expression tables above.
_RECORDING_ID_PARSERS: dict[str, Any] = {
    "millerecog2019": _miller_subject_session,
}


def _subject_from_recording_id(
    *,
    recording_id: str,
    dataset_provider: str,
) -> int:
    parser = _RECORDING_ID_PARSERS.get(dataset_provider)
    if parser is not None:
        return parser(recording_id)[0]
    pattern = _SUBJECT_FROM_RECORDING_ID_PATTERNS.get(dataset_provider)
    if pattern is None:
        raise ValueError(
            "Subject extraction from recording IDs is unsupported for "
            f"dataset.provider='{dataset_provider}'."
        )
    match = pattern.fullmatch(recording_id)
    if match is None:
        raise ValueError(
            f"Unable to parse subject id from recording_id '{recording_id}' "
            f"for dataset.provider='{dataset_provider}'."
        )
    return int(match.group("subject"))


def _strip_optional_source_prefix(recording_id: str) -> str:
    if "/" not in recording_id:
        return recording_id
    return recording_id.split("/", 1)[1]


def _subject_session_from_recording_id(
    *,
    recording_id: str,
    dataset_provider: str,
) -> tuple[int, int]:
    parser = _RECORDING_ID_PARSERS.get(dataset_provider)
    if parser is not None:
        return parser(recording_id)
    pattern = _SUBJECT_SESSION_FROM_RECORDING_ID_PATTERNS.get(dataset_provider)
    if pattern is None:
        raise ValueError(
            "Subject/session extraction from recording IDs is unsupported for "
            f"dataset.provider='{dataset_provider}'."
        )
    match = pattern.fullmatch(recording_id)
    if match is None:
        raise ValueError(
            f"Unable to parse subject/session from recording_id '{recording_id}' "
            f"for dataset.provider='{dataset_provider}'."
        )
    return int(match.group("subject")), int(match.group("session"))


def _balanced_sample_positions_by_label(
    positions_by_label: dict[int, list[int]],
    *,
    max_samples: int,
    rng: np.random.Generator,
) -> list[int]:
    labels = sorted(positions_by_label)
    quotas = {label: 0 for label in labels}
    remaining_labels = list(labels)
    remaining_slots = int(max_samples)

    while remaining_labels and remaining_slots > 0:
        base_quota = remaining_slots // len(remaining_labels)
        remainder = remaining_slots % len(remaining_labels)
        next_remaining_labels: list[int] = []
        progressed = False
        for label_idx, label in enumerate(remaining_labels):
            requested = base_quota + (1 if label_idx < remainder else 0)
            available = len(positions_by_label[label]) - quotas[label]
            take = min(requested, available)
            quotas[label] += take
            remaining_slots -= take
            progressed = progressed or take > 0
            if quotas[label] < len(positions_by_label[label]):
                next_remaining_labels.append(label)
        if not progressed:
            break
        remaining_labels = next_remaining_labels

    selected_positions: list[int] = []
    for label in labels:
        label_positions = positions_by_label[label]
        quota = quotas[label]
        if quota >= len(label_positions):
            selected_positions.extend(label_positions)
        else:
            selected_positions.extend(
                int(pos)
                for pos in rng.choice(label_positions, size=quota, replace=False)
            )
    return sorted(selected_positions)


def _count_dataset_subject_session_samples(
    dataset: WindowedNeuroprobeSplitDataset,
) -> dict[tuple[int, int], int]:
    subject_session_sample_counts: dict[tuple[int, int], int] = {}
    for position in range(len(dataset._flat_index)):
        subject_session, _ = dataset._flat_index_subject_session_and_label(position)
        subject_session_sample_counts[subject_session] = (
            subject_session_sample_counts.get(subject_session, 0) + 1
        )
    if not subject_session_sample_counts:
        raise ValueError(
            "Target effective train set must contain at least one sampled window "
            "to resolve max_train_samples_per_subject='auto'."
        )
    return dict(sorted(subject_session_sample_counts.items()))


def _build_target_auto_reference_dataset(
    provider: Any,
    *,
    dataset_provider: str,
    require_coords: bool,
    coordinate_profile: str = "popt_lip",
    window_slicing_policy: str = DEFAULT_WINDOW_SLICING_POLICY,
    brain_area_key: str | None = None,
    include_recording_ids: list[str] | None = None,
    max_samples_setting: int | str | None = None,
    sample_fraction: float,
    sample_seed: int,
) -> WindowedNeuroprobeSplitDataset:
    resolved_max_samples = (
        None
        if max_samples_setting == AUTO_MAX_TRAIN_SAMPLES_PER_SUBJECT
        else _normalize_max_train_samples_per_subject(max_samples_setting)
    )
    return WindowedNeuroprobeSplitDataset(
        provider,
        provider_key=dataset_provider,
        split="train",
        dataset_provider=dataset_provider,
        coordinate_profile=coordinate_profile,
        window_slicing_policy=window_slicing_policy,
        require_coords=require_coords,
        brain_area_key=brain_area_key,
        include_recording_ids=include_recording_ids,
        max_samples_per_subject=resolved_max_samples,
        sample_fraction=sample_fraction,
        sample_seed=sample_seed,
    )


def _clone_train_dataset_with_subject_cap(
    dataset: WindowedNeuroprobeSplitDataset,
    *,
    max_samples_per_subject: int,
) -> WindowedNeuroprobeSplitDataset:
    if dataset.split != "train":
        raise ValueError("Subject-cap cloning is only supported for train datasets.")
    resolved_cap = _normalize_max_train_samples_per_subject(max_samples_per_subject)
    if resolved_cap is None:
        raise ValueError("Resolved train subject cap must be an int.")
    # Providers may wrap non-copyable dataset-class objects.  Clone only the
    # dataset view state we need, and keep provider objects shared/read-only.
    cloned = WindowedNeuroprobeSplitDataset.__new__(WindowedNeuroprobeSplitDataset)
    cloned.provider = dataset.provider
    cloned.provider_key = dataset.provider_key
    cloned.split = dataset.split
    cloned.coordinate_profile = dataset.coordinate_profile
    cloned.window_slicing_policy = dataset.window_slicing_policy
    cloned.require_coords = dataset.require_coords
    cloned._brain_area_key = dataset._brain_area_key
    cloned._sample_fraction = dataset._sample_fraction
    cloned._sample_seed = dataset._sample_seed
    # Matched subsets cannot be combined with a subject cap.
    cloned._sample_indices = None
    cloned._materialized = None
    cloned._selection = deepcopy(dataset._selection)
    cloned._dataset_provider = dataset._dataset_provider
    cloned._interval_map = dict(dataset._interval_map)
    cloned._flat_index = list(dataset._flat_index)
    cloned._recording_cache = {}
    cloned._channel_cache = {}
    cloned._max_samples_per_subject = resolved_cap
    cloned_sample_count = len(cloned._flat_index)
    cloned._n_samples_before_subject_cap = cloned_sample_count
    cloned._apply_subject_sample_cap()
    cloned._n_samples_after_subject_cap = len(cloned._flat_index)
    # Cloned auto datasets are already sampled, so split summaries and cache
    # identity should treat the cloned view itself as the pre-subsampling
    # reference rather than leaking counters from the source dataset.
    cloned._n_raw_samples_before_subsampling = cloned_sample_count
    return cloned


def _resolve_target_auto_max_samples_per_subject(
    target_dataset: WindowedNeuroprobeSplitDataset,
    *,
    test_subject: int,
    test_session: int,
) -> int:
    subject_session_sample_counts = _count_dataset_subject_session_samples(
        target_dataset
    )
    target_subject_session = (int(test_subject), int(test_session))
    if target_subject_session not in subject_session_sample_counts:
        raise ValueError(
            "Target effective train set has no sampled windows for "
            f"dataset.test_subject={target_subject_session[0]} "
            f"dataset.test_session={target_subject_session[1]}; cannot resolve "
            "max_train_samples_per_subject='auto'."
        )
    return int(subject_session_sample_counts[target_subject_session])


def _resolve_auto_max_samples_per_subject(
    *,
    max_samples_setting: int | str | None,
    target_auto_max_samples_per_subject: int | None,
) -> int | None:
    if max_samples_setting != AUTO_MAX_TRAIN_SAMPLES_PER_SUBJECT:
        return _normalize_max_train_samples_per_subject(max_samples_setting)
    if (
        target_auto_max_samples_per_subject is None
        or target_auto_max_samples_per_subject < 1
    ):
        raise ValueError(
            "Resolved target auto max_train_samples_per_subject must be >= 1 "
            "when auto subject caps are enabled."
        )
    return int(target_auto_max_samples_per_subject)


def _format_subject_session_count_key(subject_session: tuple[int, int]) -> str:
    subject_id, session_id = subject_session
    return f"sub{int(subject_id)}_sess{int(session_id)}"


def _serialize_subject_session_counts(
    counts: dict[tuple[int, int], int],
) -> dict[str, int]:
    return {
        _format_subject_session_count_key(subject_session): int(count)
        for subject_session, count in sorted(counts.items())
    }


def _aggregate_subject_session_counts_by_subject(
    counts: dict[tuple[int, int], int],
) -> dict[int, int]:
    subject_counts: dict[int, int] = {}
    for (subject_id, _session_id), count in counts.items():
        subject_counts[int(subject_id)] = subject_counts.get(int(subject_id), 0) + int(
            count
        )
    return dict(sorted(subject_counts.items()))


def _provider_decodable_manifest_path(
    *,
    manifest_dir: Any,
    dataset_provider: str,
) -> Path:
    if not isinstance(manifest_dir, str):
        raise TypeError(
            "paths.decodable_subject_sessions_dir must be a str when decodable "
            f"train filtering is enabled, got {type(manifest_dir).__name__}."
        )
    normalized_dir = manifest_dir.strip()
    if not normalized_dir:
        raise ValueError(
            "paths.decodable_subject_sessions_dir must be non-empty when decodable "
            "train filtering is enabled."
        )
    return Path(normalized_dir) / f"{dataset_provider}.json"


def _parse_manifest_subject_session(value: Any) -> tuple[int, int]:
    if not isinstance(value, str):
        raise TypeError(
            "Decodable subject/session entries must be str values, got "
            f"{type(value).__name__}."
        )
    match = _MANIFEST_SUBJECT_SESSION_RE.fullmatch(value)
    if match is None:
        raise ValueError(
            "Invalid decodable subject/session entry "
            f"'{value}'. Expected format 'sub<subject>_sess<session>'."
        )
    return int(match.group("subject")), int(match.group("session"))


def _load_decodable_subject_session_pairs(
    *,
    manifest_dir: Any,
    dataset_provider: str,
    task: str,
) -> set[tuple[int, int]]:
    manifest_path = _provider_decodable_manifest_path(
        manifest_dir=manifest_dir,
        dataset_provider=dataset_provider,
    )
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Decodable subject/session manifest not found: '{manifest_path}'."
        )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    tasks = payload.get("tasks") if isinstance(payload, dict) else None
    if not isinstance(tasks, dict):
        raise ValueError(
            f"Decodable manifest '{manifest_path}' must contain a tasks mapping."
        )
    if task not in tasks:
        raise ValueError(
            f"Decodable manifest '{manifest_path}' does not contain task '{task}'."
        )
    task_payload = tasks[task]
    if not isinstance(task_payload, dict):
        raise TypeError(f"Decodable manifest task '{task}' must resolve to a mapping.")
    subject_sessions = task_payload.get("subject_sessions")
    if not isinstance(subject_sessions, list):
        raise TypeError(
            f"Decodable manifest task '{task}' subject_sessions must be a list."
        )
    return {
        _parse_manifest_subject_session(subject_session)
        for subject_session in subject_sessions
    }


def _select_decodable_train_recording_ids(
    *,
    dataset_provider: str,
    task: str,
    manifest_dir: Any,
    recording_ids: list[str],
) -> list[str]:
    decodable_pairs = _load_decodable_subject_session_pairs(
        manifest_dir=manifest_dir,
        dataset_provider=dataset_provider,
        task=task,
    )
    selected_ids: list[str] = []
    for recording_id in sorted(set(recording_ids)):
        subject_session = _subject_session_from_recording_id(
            recording_id=recording_id,
            dataset_provider=dataset_provider,
        )
        if subject_session in decodable_pairs:
            selected_ids.append(recording_id)
    return selected_ids


def _apply_decodable_train_recording_filter(
    provider: Any,
    *,
    enabled: bool,
    dataset_provider: str,
    task: str,
    manifest_dir: Any,
    regime: str,
    allow_empty: bool = False,
):
    if not enabled:
        return provider
    _ = regime
    candidate_ids = list(provider.recording_ids)
    selected_ids = _select_decodable_train_recording_ids(
        dataset_provider=dataset_provider,
        task=task,
        manifest_dir=manifest_dir,
        recording_ids=candidate_ids,
    )
    if not selected_ids:
        if allow_empty:
            return None
        raise ValueError(
            "dataset.decodable_subject_sessions_only=true filtered out all "
            f"training recordings for dataset.provider='{dataset_provider}', "
            f"task='{task}'."
        )
    interval_map = provider.get_sampling_intervals()
    selected_sample_count = sum(
        len(np.asarray(interval_map[recording_id].label))
        for recording_id in selected_ids
    )
    if selected_sample_count == 0:
        if allow_empty:
            return None
        raise ValueError(
            "dataset.decodable_subject_sessions_only=true filtered train set "
            f"contains no sampled windows for dataset.provider='{dataset_provider}', "
            f"task='{task}'."
        )
    log(
        f"Train recordings for provider '{dataset_provider}' filtered to decodable "
        f"subject/sessions: {len(selected_ids)}/{len(candidate_ids)}",
        priority=0,
    )
    return _FilteredRecordingProvider(
        provider,
        include_recording_ids=selected_ids,
        filter_label="decodable_subject_sessions",
    )


def _select_same_subject_train_recording_ids(
    *,
    dataset_provider: str,
    test_subject: int,
    recording_ids: list[str],
) -> list[str]:
    selected_ids: list[str] = []
    for recording_id in sorted(set(recording_ids)):
        subject_id = _subject_from_recording_id(
            recording_id=recording_id,
            dataset_provider=dataset_provider,
        )
        if subject_id == int(test_subject):
            selected_ids.append(recording_id)
    return selected_ids


def _resolve_dataset_brain_area_key(
    dataset_cfg: Any, *, needs_pool: bool
) -> str | None:
    """Resolve brain-area source key from dataset config."""
    raw_key = getattr(dataset_cfg, "brain_area_key", None)
    if raw_key is None:
        if needs_pool:
            raise ValueError(
                "dataset.brain_area_key is required when region-intersection "
                "pooling is enabled."
            )
        return None
    if not isinstance(raw_key, str):
        raise TypeError(
            "dataset.brain_area_key must be a str when set, got "
            f"{type(raw_key).__name__}."
        )
    if raw_key.strip() != raw_key:
        raise ValueError(
            "dataset.brain_area_key must not include leading/trailing whitespace. "
            f"Got '{raw_key}'."
        )
    if raw_key == "":
        raise ValueError("dataset.brain_area_key must be non-empty when set.")
    return raw_key


def _collect_split_brain_areas(split_dataset: Any) -> list[str]:
    # DS-DM pooling computes the train/test region intersection; this helper
    # computes the region labels shared by every sample in one split.
    regions_intersection: set[str] | None = None
    saw_sample = False
    for sample in split_dataset.iter_preprocess_fit_samples():
        saw_sample = True
        x = np.asarray(sample.get("x"))
        if x.ndim < 2:
            raise ValueError(
                "Region-intersection pooling requires sample['x'] with shape "
                f"(channels, ...), got {x.shape}."
            )
        n_channels = x.shape[0]
        brain_areas = sample.get("brain_areas")
        if brain_areas is None:
            raise ValueError(
                "Region-intersection pooling requires non-empty test-side brain-area "
                "metadata; sample['brain_areas'] is missing."
            )
        try:
            labels = _normalize_brain_area_array(
                brain_areas,
                expected_len=n_channels,
                context="sample['brain_areas']",
            )
        except ValueError as exc:
            raise ValueError(
                "Region-intersection pooling requires non-empty test-side brain-area "
                "metadata; sample['brain_areas'] is invalid."
            ) from exc
        sample_regions = {str(label) for label in labels.tolist()}
        if regions_intersection is None:
            regions_intersection = sample_regions
        else:
            regions_intersection.intersection_update(sample_regions)
            if not regions_intersection:
                break

    if not saw_sample or not regions_intersection:
        return []
    return sorted(regions_intersection)


def validate_provider_interface(provider: Any) -> None:
    # Fail once with a complete missing-method list instead of surfacing
    # one AttributeError at a later call site.
    required = (
        "get_sampling_intervals",
        "get_recording",
        "get_channel_metadata",
        "describe_selection",
    )
    missing = [
        name
        for name in required
        if not hasattr(provider, name) or not callable(getattr(provider, name))
    ]
    if missing:
        raise TypeError(
            "Provider is missing required methods for variable-channel adapter: "
            + ", ".join(missing)
        )


def validate_sample_dict(
    sample: dict[str, Any],
    *,
    expected_split: str | None = None,
    require_coords: bool = False,
) -> None:
    required_keys = {
        "x",
        "y",
        "channel_ids",
        "channel_coords",
        "coordinate_profile",
        "seq_id",
        "recording_id",
        "split",
        "sample_idx",
        "window_start_sec",
        "window_end_sec",
    }
    missing = sorted(required_keys - set(sample.keys()))
    if missing:
        raise KeyError(f"Sample dict missing required keys: {missing}")

    x = np.asarray(sample["x"])
    if x.ndim < 2:
        raise ValueError(f"sample['x'] must be at least 2D, got shape {x.shape}.")
    n_channels = x.shape[0]

    channel_ids = sample["channel_ids"]
    if not isinstance(channel_ids, list):
        raise TypeError("sample['channel_ids'] must be a list[str].")
    if len(channel_ids) != n_channels:
        raise ValueError(
            "sample['channel_ids'] length must match x.shape[0], got "
            f"{len(channel_ids)} vs {n_channels}."
        )

    brain_areas = sample.get("brain_areas")
    if brain_areas is not None:
        _normalize_brain_area_array(
            brain_areas,
            expected_len=n_channels,
            context="sample['brain_areas']",
        )

    coordinate_profile = sample["coordinate_profile"]
    if not isinstance(coordinate_profile, str):
        raise TypeError(
            "sample['coordinate_profile'] must be a str, got "
            f"{type(coordinate_profile).__name__}."
        )

    coords = sample["channel_coords"]
    if coords is None:
        if require_coords:
            raise ValueError(
                "sample['channel_coords'] is required for this configuration."
            )
    else:
        coords = np.asarray(coords)
        if coords.shape != (n_channels, 3):
            raise ValueError(
                "sample['channel_coords'] must have shape "
                f"({n_channels}, 3), got {coords.shape}."
            )
        if not np.isfinite(coords).all():
            raise ValueError("sample['channel_coords'] contains NaN/Inf values.")

    seq_id = sample["seq_id"]
    if seq_id is not None:
        seq_id = np.asarray(seq_id)
        if seq_id.shape != (n_channels,):
            raise ValueError(
                f"sample['seq_id'] must have shape ({n_channels},), got {seq_id.shape}."
            )

    channel_indices = sample.get("channel_indices")
    if channel_indices is not None:
        channel_indices = np.asarray(channel_indices)
        if channel_indices.shape != (n_channels,):
            raise ValueError(
                "sample['channel_indices'] must have shape "
                f"({n_channels},), got {channel_indices.shape}."
            )
        if not np.issubdtype(channel_indices.dtype, np.integer):
            raise TypeError("sample['channel_indices'] must contain integer values.")

    if not isinstance(sample["split"], str):
        raise TypeError(
            f"sample['split'] must be a str, got {type(sample['split']).__name__}."
        )
    if expected_split is not None and sample["split"] != expected_split:
        raise ValueError(
            f"sample['split'] mismatch: expected '{expected_split}', got '{sample['split']}'."
        )
    if not isinstance(sample["recording_id"], str):
        raise TypeError(
            "sample['recording_id'] must be a str, got "
            f"{type(sample['recording_id']).__name__}."
        )
    if not isinstance(sample["y"], (int, np.integer)) or isinstance(sample["y"], bool):
        raise TypeError(
            f"sample['y'] must be an int, got {type(sample['y']).__name__}."
        )
    if any(not isinstance(cid, str) for cid in channel_ids):
        raise TypeError("sample['channel_ids'] must contain only str values.")

    channel_names = sample.get("channel_names")
    if channel_names is not None:
        if not isinstance(channel_names, list):
            raise TypeError("sample['channel_names'] must be a list[str] when set.")
        if len(channel_names) != n_channels:
            raise ValueError(
                "sample['channel_names'] length must match x.shape[0], got "
                f"{len(channel_names)} vs {n_channels}."
            )
        if any(not isinstance(name, str) for name in channel_names):
            raise TypeError("sample['channel_names'] must contain only str values.")


def validate_fold_dict(fold: dict[str, Any]) -> None:
    required = {
        "fold_idx",
        "train_split",
        "val_split",
        "test_split",
        "preprocess_state",
        "metadata",
    }
    missing = sorted(required - set(fold.keys()))
    if missing:
        raise KeyError(f"Fold dict missing required keys: {missing}")
    for split_key in ("train_split", "val_split", "test_split"):
        split_obj = fold[split_key]
        if not hasattr(split_obj, "__len__") or not hasattr(split_obj, "__getitem__"):
            raise TypeError(
                f"Fold key '{split_key}' must be dataset-like (__len__/__getitem__)."
            )


def _prefix_nested_id(prefix: str, value: str) -> str:
    return f"{prefix}/{value}"


class _PrefixedProvider:
    """Wrap one processed provider so train-source identity is explicit."""

    def __init__(self, provider: Any, *, source_provider: str):
        self.provider = provider
        self.source_provider = source_provider
        self.recording_ids = sorted(
            _prefix_nested_id(source_provider, recording_id)
            for recording_id in provider.recording_ids
        )

    def _strip_source_prefix(self, recording_id: str) -> str:
        prefix = f"{self.source_provider}/"
        if not isinstance(recording_id, str):
            raise TypeError(
                "recording_id must be a str for prefixed provider lookup, got "
                f"{type(recording_id).__name__}."
            )
        if not recording_id.startswith(prefix):
            raise ValueError(
                f"recording_id '{recording_id}' does not start with expected "
                f"source prefix '{prefix}'."
            )
        return recording_id[len(prefix) :]

    def get_sampling_intervals(self):
        return {
            _prefix_nested_id(self.source_provider, recording_id): interval
            for recording_id, interval in self.provider.get_sampling_intervals().items()
        }

    def get_recording(self, recording_id):
        return self.provider.get_recording(self._strip_source_prefix(recording_id))

    def get_channel_metadata(self, recording_id):
        metadata = dict(
            self.provider.get_channel_metadata(self._strip_source_prefix(recording_id))
        )
        ids = np.asarray(metadata["ids"]).astype(str).reshape(-1)
        metadata["ids"] = np.asarray(
            [_prefix_nested_id(self.source_provider, channel_id) for channel_id in ids],
            dtype=object,
        )
        return metadata

    def describe_selection(self):
        return {
            "provider": self.source_provider,
            "provider_selection": self.provider.describe_selection(),
            "recording_ids": list(self.recording_ids),
        }


class _FilteredRecordingProvider:
    """Wrap a processed provider with a fixed recording-id subset."""

    def __init__(
        self,
        provider: Any,
        *,
        include_recording_ids: list[str],
        filter_label: str,
    ):
        if not include_recording_ids:
            raise ValueError("Filtered provider requires at least one recording id.")
        self.provider = provider
        self.filter_label = str(filter_label)
        interval_map = provider.get_sampling_intervals()
        include_ids = sorted(set(include_recording_ids))
        missing_ids = sorted(set(include_ids) - set(interval_map.keys()))
        if missing_ids:
            raise ValueError(
                "Filtered provider include_recording_ids contains unknown recording "
                f"ids: {missing_ids}."
            )
        self.recording_ids = include_ids

    def __getattr__(self, name: str):
        return getattr(self.provider, name)

    def _require_recording_id(self, recording_id: str) -> str:
        if not isinstance(recording_id, str):
            raise TypeError(
                "recording_id must be a str for filtered provider lookup, got "
                f"{type(recording_id).__name__}."
            )
        if recording_id not in self.recording_ids:
            raise ValueError(
                f"recording_id '{recording_id}' is not selected by "
                f"filter '{self.filter_label}'."
            )
        return recording_id

    def get_sampling_intervals(self):
        interval_map = self.provider.get_sampling_intervals()
        return {
            recording_id: interval_map[recording_id]
            for recording_id in self.recording_ids
        }

    def get_recording(self, recording_id):
        return self.provider.get_recording(self._require_recording_id(recording_id))

    def get_channel_metadata(self, recording_id):
        return self.provider.get_channel_metadata(
            self._require_recording_id(recording_id)
        )

    def describe_selection(self):
        base_selection = self.provider.describe_selection()
        if isinstance(base_selection, dict):
            selection = dict(base_selection)
        else:
            selection = {"provider_selection": base_selection}
        selection["recording_ids"] = list(self.recording_ids)
        selection["recording_filter"] = {
            "name": self.filter_label,
            "recording_ids": list(self.recording_ids),
        }
        return selection


def _materialize_split_samples(
    samples: Iterable[dict[str, Any]],
    *,
    split: str,
    require_coords: bool,
    preprocessor=None,
    profile_preprocessor: bool = False,
    log_prefix: str = "",
) -> list[dict[str, Any]]:
    """Materialize and validate one split's samples with optional preprocessing."""
    stage_chain = None
    stage_stats = None
    if preprocessor is not None and profile_preprocessor:
        # Composite preprocessors expose an explicit stage list; profiling per
        # stage helps keep variable-channel shape/channel changes observable.
        stage_chain = _iter_preprocessor_stages(preprocessor)
        stage_stats = [
            {
                "name": getattr(stage, "cfg", {}).get("name", "unknown"),
                "elapsed": 0.0,
                "shape": None,
                "shape_varies": False,
                "min_electrodes": None,
                "max_electrodes": None,
            }
            for stage in stage_chain
        ]

    out_samples = list(samples)
    if preprocessor is None:
        pass
    elif stage_chain is None:
        out_samples = list(preprocessor.transform_samples(out_samples))
    else:
        for stage_idx, stage in enumerate(stage_chain):
            start_time = time.time()
            out_samples = list(stage.transform_samples(out_samples))
            elapsed = time.time() - start_time

            stats = stage_stats[stage_idx]
            stats["elapsed"] += elapsed
            for transformed in out_samples:
                x_out = transformed.get("x") if isinstance(transformed, dict) else None
                if x_out is not None:
                    shape = tuple(np.asarray(x_out).shape)
                    if stats["shape"] is None:
                        stats["shape"] = shape
                    elif stats["shape"] != shape:
                        stats["shape_varies"] = True

                channel_ids = (
                    transformed.get("channel_ids")
                    if isinstance(transformed, dict)
                    else None
                )
                if isinstance(channel_ids, list):
                    n_electrodes = len(channel_ids)
                    if (
                        stats["min_electrodes"] is None
                        or n_electrodes < stats["min_electrodes"]
                    ):
                        stats["min_electrodes"] = n_electrodes
                    if (
                        stats["max_electrodes"] is None
                        or n_electrodes > stats["max_electrodes"]
                    ):
                        stats["max_electrodes"] = n_electrodes

    for transformed in out_samples:
        validate_sample_dict(
            transformed,
            expected_split=split,
            require_coords=require_coords,
        )

    if stage_stats is not None:
        prefix = f"{log_prefix}" if log_prefix else ""
        for stats in stage_stats:
            if stats["shape"] is None:
                shape_str = "unknown"
            elif stats["shape_varies"]:
                shape_str = f"variable (example {stats['shape']})"
            else:
                shape_str = str(stats["shape"])

            min_e = stats["min_electrodes"]
            max_e = stats["max_electrodes"]
            if min_e is None:
                electrode_str = "N/A"
            elif min_e == max_e:
                electrode_str = str(min_e)
            else:
                electrode_str = f"{min_e}-{max_e}"

            log(
                f"{prefix}Preprocessor '{stats['name']}' completed: "
                f"output shape {shape_str}, {electrode_str} electrodes, "
                f"{stats['elapsed']:.3f}s",
                priority=2,
                indent=1,
            )

    return out_samples


def _prepare_split_dataset(
    split_dataset,
    *,
    preprocessor,
    fold_context: dict[str, Any] | None,
    fold_idx: int,
    split_label: str,
    fit_preprocessor: bool = True,
    preprocess_state: dict[str, Any] | None = None,
) -> Any:
    """Fit/materialize one split dataset with the active preprocessor contract."""
    if preprocessor is None:
        split_dataset.materialize(preprocessor=None)
        return preprocess_state

    if hasattr(preprocessor, "reset_state"):
        preprocessor.reset_state()
    if hasattr(preprocessor, "set_fold_context"):
        effective_fold_context = dict(fold_context or {})
        provider = getattr(split_dataset, "provider", None)
        if provider is not None:
            effective_fold_context["recording_provider"] = provider
        preprocessor.set_fold_context(effective_fold_context)

    if preprocessor.requires_fit():
        if fit_preprocessor:
            fit_start = time.time()
            preprocess_state = preprocessor.fit_split(
                split_dataset.iter_preprocess_fit_samples()
            )
            preprocessor.set_state(preprocess_state)
            log(
                f"Fold {fold_idx} {split_label} preprocessor fit completed in "
                f"{time.time() - fit_start:.2f}s",
                priority=0,
            )
            consume_fit_cache = getattr(
                preprocessor, "consume_fit_transformed_samples", None
            )
            if callable(consume_fit_cache):
                # CompositePreprocessor can hand back the fully transformed train
                # samples it already produced during fit_split(...).
                cached_samples = consume_fit_cache()
                if cached_samples is not None:
                    split_dataset.set_materialized_samples(cached_samples)
                    log(
                        f"Fold {fold_idx} {split_label} reused "
                        f"{len(cached_samples)} preprocessed samples from fit pass",
                        priority=0,
                    )
        else:
            preprocessor.set_state(preprocess_state)
            log(
                f"Fold {fold_idx} {split_label} reused fitted preprocessor state",
                priority=0,
            )
    else:
        preprocess_state = preprocessor.get_state()

    if not split_dataset.is_materialized():
        split_start = time.time()
        split_dataset.materialize(
            preprocessor=preprocessor,
            profile_preprocessor=True,
            log_prefix=f"Fold {fold_idx} {split_label}: ",
        )
        log(
            f"Fold {fold_idx} {split_label} materialization + preprocessing "
            f"completed in {time.time() - split_start:.2f}s",
            priority=0,
        )
    else:
        log(
            f"Fold {fold_idx} {split_label} already materialized; "
            "skipping duplicate preprocessing.",
            priority=0,
        )
    return preprocess_state


class _ConcatenatedTrainSplitDataset(torch.utils.data.Dataset):
    """Concatenate multiple train split datasets while preserving split metadata."""

    def __init__(
        self,
        datasets: list[torch.utils.data.Dataset],
        *,
        require_coords: bool = False,
    ):
        if not datasets:
            raise ValueError("Concatenated train split requires at least one dataset.")
        self.split = "train"
        self._datasets = list(datasets)
        self.require_coords = require_coords
        self._materialized: list[dict[str, Any]] | None = None
        self._cumulative_sizes = self._compute_cumulative_sizes()

    def _compute_cumulative_sizes(self) -> list[int]:
        total = 0
        cumulative: list[int] = []
        for dataset in self._datasets:
            total += len(dataset)
            cumulative.append(total)
        return cumulative

    def __len__(self) -> int:
        if self._materialized is not None:
            return len(self._materialized)
        return self._cumulative_sizes[-1]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        if self._materialized is not None:
            return self._materialized[idx]

        dataset_idx = bisect.bisect_right(self._cumulative_sizes, idx)
        prev_total = 0 if dataset_idx == 0 else self._cumulative_sizes[dataset_idx - 1]
        return self._datasets[dataset_idx][idx - prev_total]

    def is_materialized(self) -> bool:
        return self._materialized is not None

    def iter_preprocess_fit_samples(self) -> Iterable[dict[str, Any]]:
        if self._materialized is not None:
            return iter(self._materialized)
        return itertools.chain.from_iterable(
            dataset.iter_preprocess_fit_samples() for dataset in self._datasets
        )

    def set_materialized_samples(
        self, samples: Iterable[dict[str, Any]]
    ) -> _ConcatenatedTrainSplitDataset:
        out_samples = []
        for sample in samples:
            validate_sample_dict(
                sample,
                expected_split=self.split,
                require_coords=self.require_coords,
            )
            out_samples.append(sample)
        self._materialized = out_samples
        return self

    def materialize(
        self,
        preprocessor=None,
        *,
        profile_preprocessor: bool = False,
        log_prefix: str = "",
    ) -> _ConcatenatedTrainSplitDataset:
        if self._materialized is not None:
            return self
        self._materialized = _materialize_split_samples(
            self.iter_preprocess_fit_samples(),
            split=self.split,
            require_coords=self.require_coords,
            preprocessor=preprocessor,
            profile_preprocessor=profile_preprocessor,
            log_prefix=log_prefix,
        )
        return self

    def get_split_summary(self) -> dict[str, Any]:
        source_summaries = [dataset.get_split_summary() for dataset in self._datasets]
        all_recording_ids = sorted(
            set(
                itertools.chain.from_iterable(
                    summary["recording_ids"] for summary in source_summaries
                )
            )
        )
        n_raw_samples_before_subsampling = sum(
            int(summary["n_samples_before_subsampling"]) for summary in source_summaries
        )
        return {
            "split": self.split,
            "n_samples": len(self),
            "n_recordings": len(all_recording_ids),
            "recording_ids": all_recording_ids,
            "selection": None,
            "is_materialized": self.is_materialized(),
            "n_samples_before_subsampling": n_raw_samples_before_subsampling,
            "n_sources": len(source_summaries),
            "source_summaries": source_summaries,
        }


class WindowedNeuroprobeSplitDataset(torch.utils.data.Dataset):
    """Split-local sample dataset with optional in-memory materialization."""

    def __init__(
        self,
        provider: Any,
        *,
        provider_key: str = "neuroprobe2025",
        split: str,
        dataset_provider: str | None = None,
        coordinate_profile: str = "popt_lip",
        window_slicing_policy: str = DEFAULT_WINDOW_SLICING_POLICY,
        require_coords: bool = False,
        brain_area_key: str | None = None,
        include_recording_ids: list[str] | None = None,
        max_samples_per_subject: int | None = None,
        sample_fraction: float = 1.0,
        sample_seed: int | None = None,
        sample_indices: dict[str, Any] | None = None,
    ):
        self.provider = provider
        self.provider_key = _validate_provider_key(provider_key)
        self._sample_indices = sample_indices
        self.split = split
        self.coordinate_profile = resolve_coordinate_profile(coordinate_profile)
        validate_window_slicing_policy(window_slicing_policy)
        self.window_slicing_policy = window_slicing_policy
        if not isinstance(require_coords, bool):
            raise TypeError(
                f"require_coords must be a bool, got {type(require_coords).__name__}."
            )
        if brain_area_key is not None and not isinstance(brain_area_key, str):
            raise TypeError(
                "brain_area_key must be a str when provided, got "
                f"{type(brain_area_key).__name__}."
            )
        if isinstance(brain_area_key, str) and not brain_area_key.strip():
            raise ValueError("brain_area_key must be non-empty when provided.")
        if dataset_provider is not None:
            if not isinstance(dataset_provider, str):
                raise TypeError(
                    "dataset_provider must be a str when provided, got "
                    f"{type(dataset_provider).__name__}."
                )
            if not dataset_provider.strip():
                raise ValueError("dataset_provider must be non-empty when provided.")
        if include_recording_ids is not None and not isinstance(
            include_recording_ids, list
        ):
            raise TypeError(
                "include_recording_ids must be a list[str] when provided, got "
                f"{type(include_recording_ids).__name__}."
            )
        if include_recording_ids is not None and any(
            not isinstance(recording_id, str) for recording_id in include_recording_ids
        ):
            raise TypeError("include_recording_ids must contain only str values.")
        if sample_seed is not None and (
            isinstance(sample_seed, (bool, np.bool_))
            or not isinstance(sample_seed, (int, np.integer))
        ):
            raise TypeError(
                "sample_seed must be an int when provided, got "
                f"{type(sample_seed).__name__}."
            )
        normalized_fraction = _normalize_train_sample_fraction(sample_fraction)
        if split != "train" and normalized_fraction != 1.0:
            raise ValueError(
                "sample_fraction is only supported for split='train'. "
                f"Got split='{split}' with sample_fraction={normalized_fraction}."
            )
        normalized_max_samples = _normalize_max_train_samples_per_subject(
            max_samples_per_subject
        )
        if split != "train" and normalized_max_samples is not None:
            raise ValueError(
                "max_samples_per_subject is only supported for split='train'. "
                f"Got split='{split}'."
            )
        self.require_coords = require_coords
        self._brain_area_key = (
            brain_area_key.strip() if isinstance(brain_area_key, str) else None
        )
        self._sample_fraction = normalized_fraction
        self._sample_seed = None if sample_seed is None else int(sample_seed)
        self._max_samples_per_subject = normalized_max_samples
        self._subject_sample_counts_before_cap: dict[tuple[int, int], int] = {}
        self._subject_sample_counts_after_cap: dict[tuple[int, int], int] = {}
        # Validate the provider contract up front so dataset construction fails
        # before any per-sample processing starts.
        validate_provider_interface(self.provider)

        self._selection = self.provider.describe_selection()
        inferred_dataset_provider = None
        if isinstance(self._selection, dict):
            inferred_dataset_provider = self._selection.get("provider")
        self._dataset_provider = (
            dataset_provider.strip()
            if isinstance(dataset_provider, str)
            else inferred_dataset_provider
        )
        interval_map = self.provider.get_sampling_intervals()
        if not isinstance(interval_map, dict) or not interval_map:
            raise ValueError(
                "Provider get_sampling_intervals() must return a non-empty dict."
            )
        if include_recording_ids is not None:
            include_ids = sorted(set(include_recording_ids))
            if not include_ids:
                raise ValueError(
                    "include_recording_ids must contain at least one id when set."
                )
            missing_ids = sorted(set(include_ids) - set(interval_map.keys()))
            if missing_ids:
                raise ValueError(
                    "include_recording_ids contains unknown recording ids: "
                    f"{missing_ids}."
                )
            interval_map = {
                recording_id: interval_map[recording_id] for recording_id in include_ids
            }
        self._interval_map = interval_map

        self._flat_index: list[tuple[str, int]] = []
        # Keep deterministic sample ordering across runs by sorting recording ids.
        for recording_id in sorted(self._interval_map.keys()):
            interval = self._interval_map[recording_id]
            starts = np.asarray(interval.start)
            ends = np.asarray(interval.end)
            labels = np.asarray(interval.label)
            if not (len(starts) == len(ends) == len(labels)):
                raise ValueError(
                    f"Interval array length mismatch for recording '{recording_id}'."
                )
            for i in range(len(starts)):
                self._flat_index.append((recording_id, i))
        self._n_samples_before_subject_cap = len(self._flat_index)
        if self.split == "train" and self._max_samples_per_subject is not None:
            if (
                not isinstance(self._dataset_provider, str)
                or not self._dataset_provider
            ):
                raise ValueError(
                    "dataset_provider is required when max_samples_per_subject is set."
                )
            self._apply_subject_sample_cap()
        else:
            self._subject_sample_counts_before_cap = {}
            self._subject_sample_counts_after_cap = {}
        self._n_samples_after_subject_cap = len(self._flat_index)
        self._n_raw_samples_before_subsampling = len(self._flat_index)
        if self.split == "train" and self._sample_fraction < 1.0:
            if self._n_raw_samples_before_subsampling == 0:
                raise ValueError(
                    "Cannot apply train sample subsampling on an empty split."
                )
            n_keep = int(
                np.ceil(self._sample_fraction * self._n_raw_samples_before_subsampling)
            )
            n_keep = max(1, n_keep)
            rng = np.random.default_rng(self._sample_seed)
            kept_positions = np.sort(
                rng.choice(
                    self._n_raw_samples_before_subsampling,
                    size=n_keep,
                    replace=False,
                )
            )
            self._flat_index = [self._flat_index[int(pos)] for pos in kept_positions]
        if self._sample_indices is not None:
            self._apply_explicit_sample_indices()

        self._recording_cache: dict[str, Any] = {}
        self._channel_cache: dict[str, dict[str, Any]] = {}
        self._materialized: list[dict[str, Any]] | None = None

    def __len__(self) -> int:
        if self._materialized is not None:
            return len(self._materialized)
        return len(self._flat_index)

    def _apply_explicit_sample_indices(self) -> None:
        """Keep exactly the train windows listed in a matched-subset file.

        The subset file lists positions in this fold's full train split. Before
        using them, the number of train windows and a checksum of their labels
        are compared with the values stored in the file, so the positions are
        only applied to the same windows they were computed on. Any mismatch
        stops the run; there is no random fallback.

        The positions were computed on train windows in the order a shuffled
        train loader produced them (seeded with order_seed + fold index). That
        order is recreated here: it is the 4th random permutation in the
        sequence "p p b p b b p" (p = a permutation of all positions, b = one
        int64 draw). Files whose checksum matches the plain dataset order are
        accepted too.
        """
        spec = self._sample_indices
        stem = spec["stem"]
        if self.split != "train" or self._sample_fraction != 1.0:
            raise ValueError(
                f"Matched subset {stem}: explicit train indices need split='train' "
                f"and sample_fraction=1.0 (got split={self.split!r}, "
                f"fraction={self._sample_fraction})."
            )
        labels = [
            int(np.asarray(self._interval_map[rid].label)[i])
            for rid, i in self._flat_index
        ]
        n = len(labels)
        if n != int(spec["n_train"]):
            raise ValueError(
                f"Matched subset {stem}: the train split has {n} windows, the "
                f"subset file expects {spec['n_train']}. Refusing to subsample."
            )
        generator = torch.Generator()
        generator.manual_seed(int(spec["order_seed"]) + int(spec["fold_idx"]))
        loader_order: list[int] = []
        for op in "ppbpbbp":
            if op == "b":
                torch.empty((), dtype=torch.int64).random_(generator=generator)
            else:
                loader_order = torch.randperm(n, generator=generator).tolist()
        expected_crc = int(spec["y_train_crc32"])
        loader_crc = zlib.crc32(json.dumps([labels[j] for j in loader_order]).encode())
        dataset_crc = zlib.crc32(json.dumps(labels).encode())
        if loader_crc == expected_crc:
            order, how = loader_order, "train-loader order"
        elif dataset_crc == expected_crc:
            order, how = list(range(n)), "dataset order"
        else:
            raise ValueError(
                f"Matched subset {stem}: {n} windows as expected, but the label "
                f"checksum does not match (train-loader order {loader_crc}, "
                f"dataset order {dataset_crc}, expected {expected_crc}). "
                "Refusing to subsample."
            )
        positions = [int(position) for position in spec["positions"]]
        if (
            not positions
            or min(positions) < 0
            or max(positions) >= n
            or len(set(positions)) != len(positions)
        ):
            raise ValueError(
                f"Matched subset {stem}: positions must be unique and within "
                f"0..{n - 1}."
            )
        keep = sorted(order[position] for position in positions)
        self._flat_index = [self._flat_index[j] for j in keep]
        log(
            f"Matched subset {stem} fraction={spec['frac']} draw={spec['draw']}: "
            f"checks passed ({how}); keeping {len(self._flat_index)} of {n} "
            "train windows",
            priority=0,
        )

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if self._materialized is not None:
            return self._materialized[idx]
        return self._build_raw_sample(idx)

    def is_materialized(self) -> bool:
        return self._materialized is not None

    def get_split_summary(self) -> dict[str, Any]:
        subject_session_counts_before_cap = _serialize_subject_session_counts(
            self._subject_sample_counts_before_cap
        )
        subject_session_counts_after_cap = _serialize_subject_session_counts(
            self._subject_sample_counts_after_cap
        )
        subject_counts_before_cap = _aggregate_subject_session_counts_by_subject(
            self._subject_sample_counts_before_cap
        )
        subject_counts_after_cap = _aggregate_subject_session_counts_by_subject(
            self._subject_sample_counts_after_cap
        )
        return {
            "split": self.split,
            "window_slicing_policy": self.window_slicing_policy,
            "n_samples": len(self),
            "n_recordings": len(self._interval_map),
            "recording_ids": sorted(self._interval_map.keys()),
            "selection": self._selection,
            "is_materialized": self.is_materialized(),
            "sample_fraction": float(self._sample_fraction),
            "sample_seed": self._sample_seed,
            **(
                {
                    "matched_subset": {
                        key: self._sample_indices[key]
                        for key in ("stem", "frac", "draw", "order_seed")
                    }
                }
                if self._sample_indices is not None
                else {}
            ),
            "max_samples_per_subject": self._max_samples_per_subject,
            "n_samples_before_subject_cap": int(self._n_samples_before_subject_cap),
            "n_samples_after_subject_cap": int(self._n_samples_after_subject_cap),
            "subject_session_sample_counts_before_cap": (
                subject_session_counts_before_cap
            ),
            "subject_session_sample_counts_after_cap": subject_session_counts_after_cap,
            "subject_sample_counts_before_cap": subject_counts_before_cap,
            "subject_sample_counts_after_cap": subject_counts_after_cap,
            "n_samples_before_subsampling": int(self._n_raw_samples_before_subsampling),
        }

    def _flat_index_subject_session_and_label(
        self, position: int
    ) -> tuple[tuple[int, int], int]:
        recording_id, interval_idx = self._flat_index[position]
        subject_session = _subject_session_from_recording_id(
            recording_id=_strip_optional_source_prefix(recording_id),
            dataset_provider=str(self._dataset_provider),
        )
        interval = self._interval_map[recording_id]
        label_value = np.asarray(interval.label)[interval_idx]
        if isinstance(label_value, (bool, np.bool_)):
            raise TypeError("Class labels must be integers, got bool.")
        if not isinstance(label_value, (int, np.integer)):
            raise TypeError(
                "Class labels must be integers for subject/session sample capping, got "
                f"{type(label_value).__name__}."
            )
        subject_id, session_id = subject_session
        return (int(subject_id), int(session_id)), int(label_value)

    def _apply_subject_sample_cap(self) -> None:
        if self._max_samples_per_subject is None:
            return
        positions_by_subject_session: dict[tuple[int, int], list[int]] = {}
        labels_by_position: dict[int, int] = {}
        for position in range(len(self._flat_index)):
            subject_session, label = self._flat_index_subject_session_and_label(
                position
            )
            positions_by_subject_session.setdefault(subject_session, []).append(
                position
            )
            labels_by_position[position] = label

        self._subject_sample_counts_before_cap = {
            subject_session: len(positions)
            for subject_session, positions in sorted(
                positions_by_subject_session.items()
            )
        }
        rng = np.random.default_rng(self._sample_seed)
        selected_positions: list[int] = []
        for subject_session in sorted(positions_by_subject_session):
            subject_session_positions = positions_by_subject_session[subject_session]
            if len(subject_session_positions) <= self._max_samples_per_subject:
                selected_positions.extend(subject_session_positions)
                continue
            positions_by_label: dict[int, list[int]] = {}
            for position in subject_session_positions:
                positions_by_label.setdefault(labels_by_position[position], []).append(
                    position
                )
            selected_positions.extend(
                _balanced_sample_positions_by_label(
                    positions_by_label,
                    max_samples=self._max_samples_per_subject,
                    rng=rng,
                )
            )
        selected_positions = sorted(selected_positions)
        selected_position_set = set(selected_positions)
        counts_after = {
            subject_session: sum(
                1
                for position in subject_session_positions
                if position in selected_position_set
            )
            for subject_session, subject_session_positions in sorted(
                positions_by_subject_session.items()
            )
        }
        self._flat_index = [
            self._flat_index[position] for position in selected_positions
        ]
        self._subject_sample_counts_after_cap = {
            subject_session: counts_after.get(subject_session, 0)
            for subject_session in sorted(positions_by_subject_session)
        }

    def _iter_raw_samples(self) -> Iterable[dict[str, Any]]:
        for i in range(len(self._flat_index)):
            yield self._build_raw_sample(i)

    def iter_preprocess_fit_samples(self) -> Iterable[dict[str, Any]]:
        if self._materialized is not None:
            return iter(self._materialized)
        return self._iter_raw_samples()

    def set_materialized_samples(
        self, samples: Iterable[dict[str, Any]]
    ) -> WindowedNeuroprobeSplitDataset:
        out_samples = []
        for sample in samples:
            validate_sample_dict(
                sample,
                expected_split=self.split,
                require_coords=self.require_coords,
            )
            out_samples.append(sample)
        self._materialized = out_samples
        return self

    def materialize(
        self,
        preprocessor=None,
        *,
        profile_preprocessor: bool = False,
        log_prefix: str = "",
    ) -> WindowedNeuroprobeSplitDataset:
        if self._materialized is not None:
            return self
        self._materialized = _materialize_split_samples(
            self._iter_raw_samples(),
            split=self.split,
            require_coords=self.require_coords,
            preprocessor=preprocessor,
            profile_preprocessor=profile_preprocessor,
            log_prefix=log_prefix,
        )
        return self

    def _get_recording(self, recording_id: str):
        cached = self._recording_cache.get(recording_id)
        if cached is not None:
            return cached
        # Cache recordings because each recording contributes multiple windows.
        rec = self.provider.get_recording(recording_id)
        self._recording_cache[recording_id] = rec
        return rec

    def _get_channel_meta(self, recording_id: str) -> dict[str, Any]:
        cached = self._channel_cache.get(recording_id)
        if cached is not None:
            return cached

        channel_metadata = dict(self.provider.get_channel_metadata(recording_id))
        for key in ("indices", "ids", "included_mask"):
            if key not in channel_metadata:
                raise KeyError(
                    "Provider channel metadata missing key "
                    f"'{key}' for recording '{recording_id}'."
                )

        indices_all = np.asarray(channel_metadata["indices"], dtype=int).reshape(-1)
        ids_all = np.asarray(channel_metadata["ids"]).astype(str).reshape(-1)
        included_mask = np.asarray(
            channel_metadata["included_mask"], dtype=bool
        ).reshape(-1)
        if len(ids_all) != len(included_mask):
            raise ValueError(
                "ids/included_mask length mismatch for recording "
                f"'{recording_id}': {len(ids_all)} vs {len(included_mask)}."
            )
        if len(indices_all) != len(ids_all):
            raise ValueError(
                "indices/ids length mismatch for recording "
                f"'{recording_id}': {len(indices_all)} vs {len(ids_all)}."
            )

        selected_mask = included_mask.copy()
        selected_mask_within_included = None
        frame_key, transform = _resolve_coordinate_transform(
            provider_key=self.provider_key,
            coordinate_profile=self.coordinate_profile,
        )
        coordinate_frames = channel_metadata.get("coordinate_frames")
        already_profile_coords = False
        if isinstance(coordinate_frames, dict) and frame_key in coordinate_frames:
            coords = np.asarray(coordinate_frames[frame_key], dtype=np.float32).reshape(
                -1, 3
            )
            if len(coords) != len(ids_all):
                raise ValueError(
                    f"coordinate_frames['{frame_key}'] must have length "
                    f"{len(ids_all)}, got {len(coords)}."
                )
        elif (
            self.coordinate_profile == "popt_lip"
            and isinstance(coordinate_frames, dict)
            and "btb_lip" in coordinate_frames
        ):
            coords = np.asarray(coordinate_frames["btb_lip"], dtype=np.float32).reshape(
                -1, 3
            )
            if len(coords) != len(ids_all):
                raise ValueError(
                    f"coordinate_frames['btb_lip'] must have length {len(ids_all)}, "
                    f"got {len(coords)}."
                )
            already_profile_coords = True
        elif not self.require_coords:
            # Coordless models do not require a provider coordinate frame.
            coords = None
            already_profile_coords = True
        else:
            # Map the legacy coords_type onto the profile table's canonical frame.
            legacy_coords = channel_metadata.get("coords")
            legacy_coords_type = str(channel_metadata.get("coords_type", "")).lower()
            legacy_frame = LEGACY_COORDS_TYPE_TO_FRAME.get(legacy_coords_type)
            if legacy_coords is None or legacy_frame is None:
                if not isinstance(coordinate_frames, dict):
                    raise TypeError(
                        "channel_metadata['coordinate_frames'] must be a dict for "
                        f"recording '{recording_id}', got "
                        f"{type(coordinate_frames).__name__}. The provider supplied "
                        f"coords_type={legacy_coords_type!r}, which is not one of "
                        f"the known legacy frames {sorted(LEGACY_COORDS_TYPE_TO_FRAME)}."
                    )
                raise KeyError(
                    f"channel_metadata['coordinate_frames'] missing frame "
                    f"'{frame_key}' for recording '{recording_id}'."
                )
            # Skip conversion only for the target frame or an audited exception.
            already_in_profile_frame = (
                self.coordinate_profile == "popt_lip" and legacy_frame == "btb_lip"
            ) or (
                self.coordinate_profile,
                frame_key,
                legacy_frame,
            ) in LEGACY_FRAME_PASSTHROUGH
            if legacy_frame != frame_key and not already_in_profile_frame:
                raise ValueError(
                    f"coordinate_profile '{self.coordinate_profile}' expects frame "
                    f"'{frame_key}' for provider '{self.provider_key}', but recording "
                    f"'{recording_id}' supplies coords_type={legacy_coords_type!r} "
                    f"(frame '{legacy_frame}'). Either add the frame to the provider "
                    "or register a transform for this provider/profile pair in "
                    "COORDINATE_PROFILES."
                )
            coords = np.asarray(legacy_coords, dtype=np.float32).reshape(-1, 3)
            if len(coords) != len(ids_all):
                raise ValueError(
                    f"legacy coords must have length {len(ids_all)}, got {len(coords)}."
                )
            # Otherwise convert the provider's native coordinates to the profile.
            already_profile_coords = already_in_profile_frame
        if not already_profile_coords:
            coords = transform(coords)
        if coords is not None:
            coords = np.asarray(coords, dtype=np.float32).reshape(-1, 3)
            finite_coord_mask = np.isfinite(coords).all(axis=1)
            dropped_invalid = int(np.sum(included_mask & ~finite_coord_mask))
            if dropped_invalid:
                selected_mask = included_mask & finite_coord_mask
                log(
                    f"Dropping {dropped_invalid} selected channels with NaN/Inf "
                    f"coords for recording '{recording_id}'.",
                    priority=1,
                )
            selected_mask_within_included = finite_coord_mask[included_mask]

        indices = indices_all[selected_mask]
        ids = ids_all[selected_mask]
        if len(ids) == 0:
            raise ValueError(
                f"Recording '{recording_id}' has no selected channels after "
                "filtering invalid coordinate channels."
            )
        # Channel identity should come from provider ids. The dataset/mixin path
        # is responsible for any subject/session uniqueness policy.
        channel_ids = ids.tolist()

        names_arr = channel_metadata.get("names")
        channel_names: list[str] | None = None
        if names_arr is not None:
            names = np.asarray(names_arr).astype(str).reshape(-1)
            if len(names) != len(ids_all):
                raise ValueError(
                    f"ids/names length mismatch for recording '{recording_id}': "
                    f"{len(ids_all)} vs {len(names)}."
                )
            channel_names = names[selected_mask].tolist()

        if coords is not None:
            coords = coords[selected_mask]
            if coords.shape != (len(ids), 3):
                raise ValueError(
                    f"coords must have shape ({len(ids)}, 3), got {coords.shape}."
                )
            if not np.isfinite(coords).all():
                raise ValueError(
                    "coords must not contain NaN/Inf values for recording "
                    f"'{recording_id}'."
                )

        rec = self._get_recording(recording_id)

        def _maybe_extract_brain_areas(source, *, source_name: str):
            if source is None:
                return None
            arr = np.asarray(source, dtype=object).reshape(-1)
            # Accept either full-channel metadata arrays, already-filtered arrays,
            # or indexable arrays that can be projected to selected channels.
            if len(arr) == len(ids_all):
                arr = arr[selected_mask]
                return _normalize_brain_area_array(
                    arr,
                    expected_len=len(indices),
                    context=f"{source_name} brain-area labels",
                )
            if selected_mask_within_included is not None and len(arr) == int(
                np.sum(included_mask)
            ):
                arr = arr[selected_mask_within_included]
                return _normalize_brain_area_array(
                    arr,
                    expected_len=len(indices),
                    context=f"{source_name} brain-area labels",
                )
            if len(arr) == len(indices):
                return _normalize_brain_area_array(
                    arr,
                    expected_len=len(indices),
                    context=f"{source_name} brain-area labels",
                )
            max_index = np.max(indices) if len(indices) else -1
            if len(arr) > max_index:
                selected = arr[indices]
                return _normalize_brain_area_array(
                    selected,
                    expected_len=len(indices),
                    context=f"{source_name} brain-area labels",
                )
            return None

        brain_areas = None
        key = self._brain_area_key
        if key is not None:
            # Source precedence for brain_area_key:
            # 1) provider.get_channel_metadata(recording_id)[key]
            # 2) recording.channels.<key> fallback when metadata does not expose key
            sources_seen: list[str] = []
            if key == "byd_region_code":
                if channel_names is None:
                    raise ValueError(
                        "Configured brain_area_key 'byd_region_code' requires "
                        f"channel names for recording '{recording_id}'."
                    )
                sources_seen.append("derived BYD region codes from channel names")
                brain_areas = _derive_byd_region_codes(
                    channel_names,
                    context=f"recording '{recording_id}' channel names",
                )
            elif key in channel_metadata:
                sources_seen.append(f"channel metadata '{key}'")
                brain_areas = _maybe_extract_brain_areas(
                    channel_metadata.get(key),
                    source_name=f"channel metadata '{key}'",
                )
                if brain_areas is None:
                    raise ValueError(
                        f"Configured brain_area_key '{key}' in channel metadata for "
                        f"recording '{recording_id}' could not be aligned to selected "
                        "channels."
                    )
            elif hasattr(rec, "channels") and hasattr(rec.channels, key):
                sources_seen.append(f"recording channels '{key}'")
                brain_areas = _maybe_extract_brain_areas(
                    getattr(rec.channels, key),
                    source_name=f"recording channels '{key}'",
                )
                if brain_areas is None:
                    raise ValueError(
                        f"Configured brain_area_key '{key}' in recording "
                        f"'{recording_id}' could not be aligned to selected channels."
                    )
            if brain_areas is None:
                source_msg = (
                    f"found in {', '.join(sources_seen)} but invalid"
                    if sources_seen
                    else "missing"
                )
                raise ValueError(
                    f"Configured brain_area_key '{key}' is {source_msg} for "
                    f"recording '{recording_id}'."
                )

        meta = {
            "indices": indices,
            "channel_ids": channel_ids,
            "channel_names": (None if channel_names is None else list(channel_names)),
            "channel_coords": coords,
            "coordinate_profile": self.coordinate_profile,
            "brain_areas": (
                None
                if brain_areas is None
                else np.asarray(brain_areas, dtype=object).copy()
            ),
            # V1: seq_id is all zeros (single sequence per sample), matching
            # the existing _create_seq_id behavior.  Richer grouping semantics
            # are deferred (see plan: Out of Scope).
            "seq_id": np.zeros(len(channel_ids), dtype=np.int64),
        }
        self._channel_cache[recording_id] = meta
        return meta

    def _build_raw_sample(self, idx: int) -> dict[str, Any]:
        recording_id, interval_idx = self._flat_index[idx]
        interval = self._interval_map[recording_id]

        start = float(np.asarray(interval.start, dtype=np.float64)[interval_idx])
        end = float(np.asarray(interval.end, dtype=np.float64)[interval_idx])
        label = np.asarray(interval.label)[interval_idx]

        rec = self._get_recording(recording_id)
        meta = self._get_channel_meta(recording_id)
        window_data = read_recording_window(rec, start, end, self.window_slicing_policy)
        if window_data.ndim != 2:
            raise ValueError(
                f"Expected window seeg_data.data to be 2D, got shape {window_data.shape}."
            )
        # Assumes seeg_data.data is (time, all_channels).  Transpose after
        # channel selection gives (selected_channels, time) = (channels, *feature_shape).
        x = np.asarray(window_data[:, meta["indices"]].T, dtype=np.float32)
        expected_channels = len(meta["channel_ids"])
        if x.shape[0] != expected_channels:
            raise ValueError(
                "Channel count mismatch after selection: "
                f"x has {x.shape[0]} channels but channel_ids has "
                f"{expected_channels} entries."
            )

        sample = {
            "x": x,
            "y": label,
            "channel_ids": list(meta["channel_ids"]),
            "channel_names": (
                None if meta["channel_names"] is None else list(meta["channel_names"])
            ),
            "channel_coords": (
                None
                if meta["channel_coords"] is None
                else np.asarray(meta["channel_coords"], dtype=np.float32).copy()
            ),
            "coordinate_profile": str(meta["coordinate_profile"]),
            "brain_areas": (
                None
                if meta["brain_areas"] is None
                else np.asarray(meta["brain_areas"], dtype=object).astype(str).tolist()
            ),
            "seq_id": np.asarray(meta["seq_id"], dtype=np.int64).copy(),
            "channel_indices": np.asarray(meta["indices"], dtype=np.int64).copy(),
            "recording_id": recording_id,
            "split": self.split,
            "sample_idx": idx,
            "window_slicing_policy": self.window_slicing_policy,
            "window_start_sec": start,
            "window_end_sec": end,
        }
        validate_sample_dict(
            sample,
            expected_split=self.split,
            require_coords=self.require_coords,
        )
        return sample


def _build_regression_targets(
    dataset_cfg: Any,
    *,
    split_providers: dict[str, Any],
    dataset_provider: str,
    fold_idx: int,
) -> regression_utils.RegressionTargets | None:
    """Collect the continuous targets of every recording in this fold.

    In regression mode a window's y is its row in its recording's target
    table. The tables are read once per fold from the dataset class and kept
    next to the splits, so they never pass through preprocessing or the caches.
    Returns None for binary and multiclass runs.
    """
    if not regression_utils.is_regression(dataset_cfg):
        return None
    out_last = _dataset_cfg_get(dataset_cfg, "regression_target_last_samples", 0)
    targets = regression_utils.RegressionTargets(
        str(dataset_cfg.task), out_last=int(out_last or 0)
    )
    for split_provider in split_providers.values():
        getter = getattr(split_provider, "get_regression_targets", None)
        if getter is None:
            raise TypeError(
                "dataset.label_mode='regression' needs a dataset class with "
                f"get_regression_targets(); '{dataset_provider}' has none."
            )
        for recording_id in split_provider.recording_ids:
            payload = getter(recording_id)
            targets.add(recording_id, traj=payload["traj"], mean=payload["mean"])
    log(
        f"Fold {fold_idx}: regression targets for '{dataset_cfg.task}' have "
        f"{targets.traj_len} samples per window",
        priority=0,
    )
    return targets


_MATCHED_SUBSET_FILES: dict[str, Any] = {}


class MatchedSubsetAbsent(ValueError):
    """The subset file has no entry for this fold and draw.

    The subset generator skips draws it cannot make (too few windows, or only
    one class). The fold is then recorded as skipped instead of scored.
    """


def _resolve_matched_train_indices(
    dataset_cfg: Any, fold_idx: int
) -> dict[str, Any] | None:
    """Look up this fold's matched train subset, or None when none is set.

    dataset.train_sample_indices_file is a JSON file keyed by
    "<task>_sub<S>_sess<E>_fold<F>". Each entry stores the number of train
    windows (n_train), a checksum of their labels (y_train_crc32), and under
    "subsets" -> fraction -> draw the positions of the train windows to keep.
    dataset.train_sample_indices_frac and dataset.train_sample_indices_draw
    pick the fraction and the draw.
    """
    path = _dataset_cfg_get(dataset_cfg, "train_sample_indices_file", None)
    if not path:
        return None
    frac = _dataset_cfg_get(dataset_cfg, "train_sample_indices_frac", None)
    draw = _dataset_cfg_get(dataset_cfg, "train_sample_indices_draw", None)
    if frac is None or draw is None:
        raise ValueError(
            "dataset.train_sample_indices_file needs "
            "dataset.train_sample_indices_frac and dataset.train_sample_indices_draw."
        )
    frac_key, draw_key = str(frac), str(int(draw))
    path = str(path)
    if path not in _MATCHED_SUBSET_FILES:
        with open(path) as handle:
            _MATCHED_SUBSET_FILES[path] = json.load(handle)
    table = _MATCHED_SUBSET_FILES[path]
    stem = (
        f"{dataset_cfg.task}_sub{int(dataset_cfg.test_subject)}"
        f"_sess{int(dataset_cfg.test_session)}_fold{int(fold_idx)}"
    )
    if stem not in table:
        raise ValueError(
            f"Matched subset {stem} is not in {path}. Refusing to subsample."
        )
    entry = table[stem]
    by_frac = entry.get("subsets", {}).get(frac_key)
    if by_frac is None:
        raise ValueError(f"Matched subset {stem} has no fraction {frac_key} in {path}.")
    if draw_key not in by_frac:
        raise MatchedSubsetAbsent(
            f"Matched subset {stem} fraction={frac_key} draw={draw_key} is absent: "
            "the subset generator skipped this draw, so the fold is not scored."
        )
    order_seed = _dataset_cfg_get(dataset_cfg, "train_sample_indices_order_seed", None)
    return {
        "stem": stem,
        "fold_idx": int(fold_idx),
        "order_seed": 42 if order_seed is None else int(order_seed),
        "frac": frac_key,
        "draw": draw_key,
        "n_train": entry["n_train"],
        "y_train_crc32": entry["y_train_crc32"],
        "positions": by_frac[draw_key],
    }


def build_neuroprobe_torch_fold(
    dataset_cfg: Any,
    preprocessor,
    *,
    preprocessor_cfg: Any | None = None,
    paths_cfg: Any | None = None,
    fold_idx: int,
    seed: int,
    require_coords: bool = True,
    needs_pool: bool | None = None,
    train_source_cache_enabled: bool = False,
    train_source_cache_dir: Any = None,
    preprocessed_split_cache_enabled: bool = False,
    preprocessed_split_cache_dir: Any = None,
    preprocessed_split_cache_mode: str = "read_write",
) -> dict[str, Any]:
    """Build one fully prepared train/val/test fold for torch-style evaluation.

    This is the central adapter between validated config objects, dataset-class
    split providers, preprocessing, optional multi-source train composition, and
    the dataset objects consumed by downstream runners.  The body stays mostly
    linear on purpose so the fold-building sequence can be read top-to-bottom:
    resolve fold-scoped settings, build raw split datasets, optionally
    preprocess/materialize them, then attach final metadata.
    """
    # Resolve the fold-scoped routing knobs once up front so the rest of the
    # function can branch on a normalized single-source vs multi-source view.
    fold_seed = seed
    window_slicing_policy = _dataset_cfg_get(
        dataset_cfg, "window_slicing_policy", DEFAULT_WINDOW_SLICING_POLICY
    )
    dataset_provider = dataset_cfg.provider
    regime = dataset_cfg.regime
    coordinate_profile = resolve_coordinate_profile(
        _dataset_cfg_get(dataset_cfg, "coordinate_profile", "popt_lip")
    )
    train_sources = resolve_train_source_configs(dataset_cfg)
    train_source_by_provider = {
        str(source_cfg["provider"]): source_cfg for source_cfg in train_sources
    }
    uses_train_sources = bool(train_sources)
    decodable_subject_sessions_only = _normalize_decodable_subject_sessions_only(
        _dataset_cfg_get(
            dataset_cfg,
            "decodable_subject_sessions_only",
            False,
        )
    )
    validate_decodable_train_source_regimes(
        train_sources,
        enabled=decodable_subject_sessions_only,
    )
    decodable_subject_sessions_dir = (
        None
        if paths_cfg is None
        else _dataset_cfg_get(paths_cfg, "decodable_subject_sessions_dir", None)
    )
    train_same_subject_only = (
        False
        if uses_train_sources
        else _normalize_train_same_subject_only(
            _dataset_cfg_get(dataset_cfg, "train_same_subject_only", False)
        )
    )
    train_sample_fraction = (
        1.0
        if uses_train_sources
        else _normalize_train_sample_fraction(
            _dataset_cfg_get(dataset_cfg, "train_sample_fraction", 1.0)
        )
    )
    matched_train_indices = _resolve_matched_train_indices(dataset_cfg, fold_idx)
    if matched_train_indices is not None and (
        uses_train_sources
        or train_sample_fraction < 1.0
        or _dataset_cfg_get(dataset_cfg, "max_train_samples_per_subject", None)
        is not None
    ):
        raise ValueError(
            "dataset.train_sample_indices_file cannot be combined with "
            "train_sources, train_sample_fraction < 1 or "
            "max_train_samples_per_subject."
        )
    max_train_samples_per_subject_setting = (
        _normalize_max_train_samples_per_subject_setting(
            _dataset_cfg_get(dataset_cfg, "max_train_samples_per_subject", None)
        )
    )
    max_train_samples_per_subject = (
        None
        if max_train_samples_per_subject_setting == AUTO_MAX_TRAIN_SAMPLES_PER_SUBJECT
        else _normalize_max_train_samples_per_subject(
            max_train_samples_per_subject_setting
        )
    )
    if needs_pool is None:
        requires_aligned = getattr(dataset_cfg, "requires_aligned_channels", True)
        needs_pool = needs_region_intersection_pool(
            dataset_provider, regime, requires_aligned
        )
    brain_area_key = _resolve_dataset_brain_area_key(
        dataset_cfg,
        needs_pool=needs_pool,
    )
    # This is display metadata; source config filenames remain separate lookup keys.
    display_cfg = preprocessor_cfg
    if display_cfg is None:
        display_cfg = getattr(preprocessor, "cfg", None)
    default_preprocessor_name = (
        describe_preprocessor(display_cfg) if display_cfg is not None else None
    )
    resolved_train_source_cache_dir = _resolve_train_source_cache_dir(
        cache_enabled=train_source_cache_enabled,
        cache_dir=train_source_cache_dir,
        dataset_root=str(dataset_cfg.root),
    )
    resolved_preprocessed_split_cache_dir = _resolve_preprocessed_split_cache_dir(
        cache_enabled=preprocessed_split_cache_enabled,
        cache_dir=preprocessed_split_cache_dir,
        dataset_root=str(dataset_cfg.root),
    )
    normalized_preprocessed_split_cache_mode = _normalize_preprocessed_split_cache_mode(
        preprocessed_split_cache_mode
    )

    # The target provider always owns the canonical train/val/test split
    # providers.  In multi-source mode only the exposed train dataset changes;
    # val/test remain anchored to dataset.provider.
    split_providers = {}
    split_ctor_seconds = {}
    for split in ("train", "val", "test"):
        split_ctor_start = time.time()
        # Build one provider instance per split so split-specific sampling
        # boundaries remain isolated.
        log(
            f"Fold {fold_idx} building {split} provider for "
            f"dataset.provider='{dataset_provider}'",
            priority=0,
        )
        split_providers[split] = build_processed_split_provider(
            dataset_provider=dataset_provider,
            dataset_cfg=dataset_cfg,
            split=split,
            fold_idx=fold_idx,
            regime=regime,
        )
        split_ctor_seconds[split] = time.time() - split_ctor_start
    regression_targets = _build_regression_targets(
        dataset_cfg,
        split_providers=split_providers,
        dataset_provider=str(dataset_provider),
        fold_idx=fold_idx,
    )
    if not uses_train_sources:
        split_providers["train"] = _apply_decodable_train_recording_filter(
            split_providers["train"],
            enabled=decodable_subject_sessions_only,
            dataset_provider=str(dataset_provider),
            task=str(dataset_cfg.task),
            manifest_dir=decodable_subject_sessions_dir,
            regime=str(regime),
        )

    train_source_datasets: list[WindowedNeuroprobeSplitDataset] = []
    train_source_ctor_seconds: dict[str, float] = {}
    train_source_preprocessor_names: dict[str, str | None] = {}
    active_train_sources: list[dict[str, Any]] = list(train_sources)
    active_train_source_by_provider: dict[str, dict[str, Any]] = dict(
        train_source_by_provider
    )
    filtered_train_source_provider_by_provider: dict[str, Any] = {}
    filtered_train_source_provider_by_provider_for_auto: dict[str, Any] = {}
    skipped_empty_decodable_train_sources: list[str] = []
    target_eval_train_dataset: Any = None
    if uses_train_sources:
        # Build each source as an independent train-only dataset and prefix its
        # ids so concatenation does not collapse provider identity.
        active_train_sources = []
        active_train_source_by_provider = {}
        for source_cfg in train_sources:
            source_provider = str(source_cfg["provider"])
            source_dataset_cfg_values = dict(source_cfg)
            if source_provider == str(dataset_provider):
                source_dataset_cfg_values["test_subject"] = dataset_cfg.test_subject
                source_dataset_cfg_values["test_session"] = dataset_cfg.test_session
            source_dataset_cfg = SimpleNamespace(**source_dataset_cfg_values)
            split_ctor_start = time.time()
            log(
                f"Fold {fold_idx} building train source provider "
                f"'{source_provider}' (regime='{source_dataset_cfg.regime}')",
                priority=0,
            )
            provider = build_processed_split_provider(
                dataset_provider=source_dataset_cfg.provider,
                dataset_cfg=source_dataset_cfg,
                split="train",
                fold_idx=fold_idx,
                regime=source_dataset_cfg.regime,
            )
            provider = _apply_decodable_train_recording_filter(
                provider,
                enabled=decodable_subject_sessions_only,
                dataset_provider=source_provider,
                task=str(source_dataset_cfg.task),
                manifest_dir=decodable_subject_sessions_dir,
                regime=str(source_dataset_cfg.regime),
                allow_empty=True,
            )
            train_source_ctor_seconds[source_provider] = time.time() - split_ctor_start
            if provider is None:
                skipped_empty_decodable_train_sources.append(source_provider)
                log(
                    f"Fold {fold_idx} skipped train source '{source_provider}' after "
                    "decodable subject/session filtering selected no usable train "
                    "samples.",
                    priority=0,
                )
                continue
            active_source_cfg = dict(source_dataset_cfg_values)
            active_train_sources.append(active_source_cfg)
            active_train_source_by_provider[source_provider] = active_source_cfg
            filtered_train_source_provider_by_provider[source_provider] = provider
            filtered_train_source_provider_by_provider_for_auto[source_provider] = (
                provider
            )
            source_preprocessor_name = active_source_cfg.get("preprocessor")
            train_source_preprocessor_names[source_provider] = (
                default_preprocessor_name
                if source_preprocessor_name is None
                else str(source_preprocessor_name)
            )
        any_auto_train_source_cap_requested = any(
            source_cfg.get("max_train_samples_per_subject")
            == AUTO_MAX_TRAIN_SAMPLES_PER_SUBJECT
            for source_cfg in active_train_sources
        )
        target_source_provider_for_auto = (
            filtered_train_source_provider_by_provider_for_auto.get(
                str(dataset_provider)
            )
        )
        target_source_cfg_for_auto = active_train_source_by_provider.get(
            str(dataset_provider)
        )
        if any_auto_train_source_cap_requested and (
            target_source_provider_for_auto is None
            or target_source_cfg_for_auto is None
        ):
            raise ValueError(
                "max_train_samples_per_subject='auto' requires dataset.provider "
                f"'{dataset_provider}' to remain an active train source after "
                "filtering."
            )

        target_auto_max_samples_per_subject = None
        target_reference_train_dataset = None
        target_prefixed_reference_train_dataset = None
        if (
            target_source_provider_for_auto is not None
            and target_source_cfg_for_auto is not None
        ):
            target_reference_train_dataset = _build_target_auto_reference_dataset(
                target_source_provider_for_auto,
                dataset_provider=str(dataset_provider),
                require_coords=require_coords,
                coordinate_profile=coordinate_profile,
                window_slicing_policy=window_slicing_policy,
                brain_area_key=brain_area_key,
                max_samples_setting=target_source_cfg_for_auto.get(
                    "max_train_samples_per_subject"
                ),
                sample_fraction=float(
                    target_source_cfg_for_auto["train_sample_fraction"]
                ),
                sample_seed=fold_seed,
            )
            target_auto_max_samples_per_subject = (
                _resolve_target_auto_max_samples_per_subject(
                    target_reference_train_dataset,
                    test_subject=dataset_cfg.test_subject,
                    test_session=dataset_cfg.test_session,
                )
                if any_auto_train_source_cap_requested
                else None
            )
            target_prefixed_reference_train_dataset = (
                _build_target_auto_reference_dataset(
                    _PrefixedProvider(
                        target_source_provider_for_auto,
                        source_provider=str(dataset_provider),
                    ),
                    dataset_provider=str(dataset_provider),
                    require_coords=require_coords,
                    coordinate_profile=coordinate_profile,
                    window_slicing_policy=window_slicing_policy,
                    brain_area_key=None,
                    max_samples_setting=target_source_cfg_for_auto.get(
                        "max_train_samples_per_subject"
                    ),
                    sample_fraction=float(
                        target_source_cfg_for_auto["train_sample_fraction"]
                    ),
                    sample_seed=fold_seed,
                )
            )
        for source_cfg in active_train_sources:
            source_provider = str(source_cfg["provider"])
            source_max_samples_setting = source_cfg.get("max_train_samples_per_subject")
            source_max_samples_per_subject = _resolve_auto_max_samples_per_subject(
                max_samples_setting=source_max_samples_setting,
                target_auto_max_samples_per_subject=target_auto_max_samples_per_subject,
            )
            source_cfg["max_train_samples_per_subject"] = source_max_samples_per_subject
            if (
                source_provider == str(dataset_provider)
                and target_prefixed_reference_train_dataset is not None
                and target_reference_train_dataset is not None
            ):
                if (
                    source_max_samples_setting == AUTO_MAX_TRAIN_SAMPLES_PER_SUBJECT
                    and source_max_samples_per_subject is not None
                ):
                    source_dataset = _clone_train_dataset_with_subject_cap(
                        target_prefixed_reference_train_dataset,
                        max_samples_per_subject=source_max_samples_per_subject,
                    )
                    target_eval_train_dataset = _clone_train_dataset_with_subject_cap(
                        target_reference_train_dataset,
                        max_samples_per_subject=source_max_samples_per_subject,
                    )
                else:
                    source_dataset = target_prefixed_reference_train_dataset
                    target_eval_train_dataset = target_reference_train_dataset
            else:
                provider = filtered_train_source_provider_by_provider_for_auto[
                    source_provider
                ]
                prefixed_provider = _PrefixedProvider(
                    provider, source_provider=source_provider
                )
                source_dataset = WindowedNeuroprobeSplitDataset(
                    prefixed_provider,
                    provider_key=source_provider,
                    split="train",
                    dataset_provider=source_provider,
                    coordinate_profile=str(source_dataset_cfg.coordinate_profile),
                    window_slicing_policy=window_slicing_policy,
                    require_coords=require_coords,
                    # Aligned multi-source runs are rejected during config
                    # validation, so v1 train-source datasets never need
                    # per-source brain-area alignment metadata.
                    brain_area_key=None,
                    max_samples_per_subject=source_max_samples_per_subject,
                    sample_fraction=float(source_cfg["train_sample_fraction"]),
                    sample_seed=fold_seed,
                )
            train_source_datasets.append(source_dataset)
        if decodable_subject_sessions_only and not train_source_datasets:
            raise ValueError(
                "dataset.decodable_subject_sessions_only=true filtered out "
                "all train sources."
            )

    if uses_train_sources:
        source_timing_text = " ".join(
            f"{name}={train_source_ctor_seconds[name]:.2f}s"
            for name in sorted(train_source_ctor_seconds)
        )
        log(
            f"Fold {fold_idx} dataset construction timings: "
            f"train_sources[{source_timing_text}] "
            f"val={split_ctor_seconds['val']:.2f}s "
            f"test={split_ctor_seconds['test']:.2f}s",
            priority=0,
        )
    else:
        log(
            f"Fold {fold_idx} dataset provider '{dataset_provider}' "
            "construction timings: "
            f"train={split_ctor_seconds['train']:.2f}s "
            f"val={split_ctor_seconds['val']:.2f}s "
            f"test={split_ctor_seconds['test']:.2f}s",
            priority=0,
        )

    train_include_recording_ids: list[str] | None = None
    # Same-subject filtering only applies to single-source session-based regimes;
    # multi-source configs already spell out their train selection.
    if train_same_subject_only and regime in {"hold-in-session", "hold-out-session"}:
        candidate_train_recording_ids = list(split_providers["train"].recording_ids)
        train_include_recording_ids = _select_same_subject_train_recording_ids(
            dataset_provider=dataset_provider,
            test_subject=dataset_cfg.test_subject,
            recording_ids=candidate_train_recording_ids,
        )
        if not train_include_recording_ids:
            raise ValueError(
                "dataset.train_same_subject_only=true filtered out all training "
                "recordings. No same-subject sessions remain for "
                f"dataset.test_subject={dataset_cfg.test_subject}."
            )
        log(
            f"Fold {fold_idx} train recordings filtered to same-subject sessions: "
            f"{len(train_include_recording_ids)}/{len(candidate_train_recording_ids)} "
            f"(test_subject={dataset_cfg.test_subject})",
            priority=0,
        )
    single_source_reference_train_dataset = None
    if (
        not uses_train_sources
        and max_train_samples_per_subject_setting == AUTO_MAX_TRAIN_SAMPLES_PER_SUBJECT
    ):
        single_source_reference_train_dataset = _build_target_auto_reference_dataset(
            split_providers["train"],
            dataset_provider=str(dataset_provider),
            require_coords=require_coords,
            coordinate_profile=coordinate_profile,
            window_slicing_policy=window_slicing_policy,
            brain_area_key=brain_area_key,
            include_recording_ids=train_include_recording_ids,
            max_samples_setting=max_train_samples_per_subject_setting,
            sample_fraction=train_sample_fraction,
            sample_seed=fold_seed,
        )
        max_train_samples_per_subject = _resolve_target_auto_max_samples_per_subject(
            single_source_reference_train_dataset,
            test_subject=dataset_cfg.test_subject,
            test_session=dataset_cfg.test_session,
        )

    split_datasets = {}
    if uses_train_sources:
        # Multi-source training exposes a concatenated train split, but eval
        # still runs on the target provider's val/test splits.
        for split in ("val", "test"):
            split_datasets[split] = WindowedNeuroprobeSplitDataset(
                split_providers[split],
                provider_key=str(dataset_provider),
                split=split,
                dataset_provider=str(dataset_provider),
                coordinate_profile=coordinate_profile,
                window_slicing_policy=window_slicing_policy,
                require_coords=require_coords,
                brain_area_key=brain_area_key,
                sample_fraction=1.0,
                sample_seed=None,
            )
        split_datasets["train"] = _ConcatenatedTrainSplitDataset(
            train_source_datasets,
            require_coords=require_coords,
        )
    else:
        # Single-source mode keeps all three splits on the target provider and
        # only applies same-subject filtering / subsampling to train.
        if single_source_reference_train_dataset is not None:
            split_datasets["train"] = _clone_train_dataset_with_subject_cap(
                single_source_reference_train_dataset,
                max_samples_per_subject=max_train_samples_per_subject,
            )
        else:
            split_datasets["train"] = WindowedNeuroprobeSplitDataset(
                split_providers["train"],
                provider_key=str(dataset_provider),
                split="train",
                dataset_provider=str(dataset_provider),
                coordinate_profile=coordinate_profile,
                window_slicing_policy=window_slicing_policy,
                require_coords=require_coords,
                brain_area_key=brain_area_key,
                include_recording_ids=train_include_recording_ids,
                max_samples_per_subject=max_train_samples_per_subject,
                sample_fraction=train_sample_fraction,
                sample_seed=fold_seed,
                sample_indices=matched_train_indices,
            )
        for split in ("val", "test"):
            split_datasets[split] = WindowedNeuroprobeSplitDataset(
                split_providers[split],
                provider_key=str(dataset_provider),
                split=split,
                dataset_provider=str(dataset_provider),
                coordinate_profile=coordinate_profile,
                window_slicing_policy=window_slicing_policy,
                require_coords=require_coords,
                brain_area_key=brain_area_key,
                sample_fraction=1.0,
                sample_seed=None,
            )

    # Log the raw sampling decisions before preprocessing so later materialized
    # counts can be interpreted against the pre-transform train selection.
    if uses_train_sources:
        train_summary = split_datasets["train"].get_split_summary()
        for source_summary in train_summary["source_summaries"]:
            source_selection = source_summary["selection"] or {}
            source_provider = source_selection.get("provider", "unknown")
            fraction = float(source_summary.get("sample_fraction", 1.0))
            if fraction < 1.0:
                n_total = int(source_summary["n_samples_before_subsampling"])
                n_kept = int(source_summary["n_samples"])
                log(
                    f"Fold {fold_idx} train provider '{source_provider}' random "
                    f"subsampling kept {n_kept}/{n_total} samples "
                    f"(fraction={fraction:.4f}, seed={fold_seed})",
                    priority=0,
                )
    elif train_sample_fraction < 1.0:
        train_summary = split_datasets["train"].get_split_summary()
        n_total = int(train_summary["n_samples_before_subsampling"])
        n_kept = int(train_summary["n_samples"])
        log(
            f"Fold {fold_idx} train split random subsampling kept "
            f"{n_kept}/{n_total} samples "
            f"(fraction={train_sample_fraction:.4f}, seed={fold_seed})",
            priority=0,
        )
    log_fold_split_sample_counts(split_datasets, fold_idx=fold_idx, phase="raw")

    preprocess_state = None
    eval_preprocessor_name = None
    train_source_preprocess_states: dict[str, Any] = {}
    train_source_feature_shapes: dict[str, tuple[int, ...] | None] = {}
    preprocessed_split_cache_path = None
    preprocessed_split_cache_identity = None
    preprocessed_split_cached_payload = None
    preprocessed_split_cache_active = (
        preprocessor is not None
        and not uses_train_sources
        and resolved_preprocessed_split_cache_dir is not None
    )
    if (
        preprocessor is not None
        and uses_train_sources
        and resolved_preprocessed_split_cache_dir is not None
    ):
        log(
            "Preprocessed split cache is currently skipped for multi-source "
            "train configs; using normal preprocessing/train-source cache paths.",
            priority=0,
        )
    if preprocessed_split_cache_active:
        preprocessor_identity = _serialize_preprocessor_cache_identity(
            preprocessor_name=None,
            preprocessor_cfg=preprocessor_cfg,
            preprocessor=preprocessor,
        )
        preprocessed_split_cache_identity = _build_preprocessed_split_cache_identity(
            dataset_cfg=dataset_cfg,
            split_datasets=split_datasets,
            preprocessor_identity=preprocessor_identity,
            fold_idx=fold_idx,
            needs_pool=needs_pool,
            brain_area_key=brain_area_key,
        )
        preprocessed_split_cache_path = _preprocessed_split_cache_path(
            cache_dir=resolved_preprocessed_split_cache_dir,
            identity=preprocessed_split_cache_identity,
        )
        if normalized_preprocessed_split_cache_mode != "refresh":
            preprocessed_split_cached_payload = _load_preprocessed_split_cache(
                preprocessed_split_cache_path
            )
        if preprocessed_split_cached_payload is not None:
            for split in ("train", "val", "test"):
                split_datasets[split].set_materialized_samples(
                    preprocessed_split_cached_payload["samples"][split]
                )
            preprocess_state = preprocessed_split_cached_payload.get("preprocess_state")
            eval_preprocessor_name = default_preprocessor_name
            log(
                f"Fold {fold_idx} reused preprocessed train/val/test splits "
                f"from '{preprocessed_split_cache_path}'",
                priority=0,
            )
        elif normalized_preprocessed_split_cache_mode == "read_only":
            raise RuntimeError(
                "runtime.preprocessed_split_cache_mode='read_only' requires a "
                "preprocessed split cache hit, but this fold had a "
                f"preprocessed split cache miss at '{preprocessed_split_cache_path}'."
            )

    if preprocessor is not None:
        if preprocessed_split_cached_payload is None:
            # Preprocessor routing differs by mode:
            # single-source => fit once on train, reuse on val/test
            # multi-source => fit each train source independently, then fit eval
            #                  transforms on the target provider's own train subset
            fold_context = {
                "dataset_provider": dataset_provider,
                "dataset_regime": regime,
                "needs_region_intersection_pool": needs_pool,
            }
            if needs_pool:
                # RegionIntersectionPoolPreprocessor needs test-side labels to
                # compute the train/test intersection contract.
                test_brain_areas = _collect_split_brain_areas(split_datasets["test"])
                if not test_brain_areas:
                    raise ValueError(
                        "Region-intersection pooling requires non-empty test-side "
                        "common brain-area intersection to compute train/test "
                        "intersection."
                    )
                fold_context["test_brain_areas"] = test_brain_areas
            train_source_preprocess_states = {}
            train_source_feature_shapes = {}
            # Multisource eval should follow the target provider's configured
            # preprocessor override even if decodable filtering skips that source
            # from training; eval val/test still belong to the target provider.
            # Otherwise it inherits the top-level preprocessor like single-source.
            eval_source_cfg = train_source_by_provider.get(str(dataset_provider))
            eval_source_preprocessor_name = (
                None if eval_source_cfg is None else eval_source_cfg.get("preprocessor")
            )
            eval_preprocessor_name = (
                default_preprocessor_name
                if eval_source_preprocessor_name is None
                else str(eval_source_preprocessor_name)
            )
            eval_preprocessor = _build_fresh_preprocessor(
                preprocessor_name=eval_source_preprocessor_name,
                preprocessor_cfg=(
                    preprocessor_cfg if eval_source_preprocessor_name is None else None
                ),
                preprocessor=(
                    preprocessor if eval_source_preprocessor_name is None else None
                ),
            )
            eval_train_sample_fraction = (
                1.0
                if eval_source_cfg is None
                else float(eval_source_cfg["train_sample_fraction"])
            )
            if uses_train_sources:
                eval_train_max_samples_per_subject = (
                    None
                    if eval_source_cfg is None
                    else eval_source_cfg["max_train_samples_per_subject"]
                )
            else:
                eval_train_max_samples_per_subject = max_train_samples_per_subject
            # When the target provider participates in multisource training with train
            # sampling controls, eval normalization should be fit on that same train
            # subset rather than on a larger full-train view of the provider.
            eval_train_sample_seed = (
                int(fold_seed)
                if (
                    eval_train_sample_fraction < 1.0
                    or eval_train_max_samples_per_subject is not None
                )
                else None
            )
            if uses_train_sources:
                # Each train source can override the top-level preprocessor, so
                # fitted state and cache identity are tracked per source provider.
                for source_cfg, source_dataset in zip(
                    active_train_sources, train_source_datasets, strict=False
                ):
                    source_provider = str(source_cfg["provider"])
                    source_preprocessor_name = train_source_preprocessor_names[
                        source_provider
                    ]
                    source_preprocessor_identity = (
                        _serialize_preprocessor_cache_identity(
                            preprocessor_name=source_cfg.get("preprocessor"),
                            preprocessor_cfg=(
                                preprocessor_cfg
                                if source_cfg.get("preprocessor") is None
                                else None
                            ),
                            preprocessor=(
                                preprocessor
                                if source_cfg.get("preprocessor") is None
                                else None
                            ),
                        )
                    )
                    cache_identity = _build_train_source_cache_identity(
                        source_cfg=source_cfg,
                        source_dataset=source_dataset,
                        preprocessor_identity=source_preprocessor_identity,
                        fold_idx=fold_idx,
                        require_coords=require_coords,
                    )
                    cache_path = None
                    cached_payload = None
                    if resolved_train_source_cache_dir is not None:
                        cache_path = _train_source_cache_path(
                            cache_dir=resolved_train_source_cache_dir,
                            identity=cache_identity,
                        )
                        cached_payload = _load_train_source_cache(cache_path)
                    if cached_payload is not None:
                        # Cache reuse is keyed by provider selection + preprocessor
                        # identity so expensive train-source materialization can be
                        # skipped across repeated runs of the same fold.
                        source_dataset.set_materialized_samples(
                            cached_payload["samples"]
                        )
                        train_source_preprocess_states[source_provider] = (
                            cached_payload.get("preprocess_state")
                        )
                        log(
                            f"Fold {fold_idx} train source '{source_provider}' reused "
                            f"{len(cached_payload['samples'])} cached samples from "
                            f"'{cache_path}'",
                            priority=0,
                        )
                        continue

                    source_preprocessor = _build_fresh_preprocessor(
                        preprocessor_name=source_cfg.get("preprocessor"),
                        preprocessor_cfg=preprocessor_cfg,
                        preprocessor=preprocessor,
                    )
                    log(
                        f"Fold {fold_idx} preprocessing train source "
                        f"'{source_provider}' with preprocessor "
                        f"'{source_preprocessor_name or 'unknown'}'",
                        priority=0,
                    )
                    source_fold_context = {
                        "dataset_provider": source_provider,
                        "dataset_regime": source_cfg["regime"],
                        # Aligned multi-source runs are rejected during config
                        # validation, so source preprocessors never need region
                        # intersection pooling in v1.
                        "needs_region_intersection_pool": False,
                    }
                    source_preprocess_state = _prepare_split_dataset(
                        source_dataset,
                        preprocessor=source_preprocessor,
                        fold_context=source_fold_context,
                        fold_idx=fold_idx,
                        split_label=f"train source '{source_provider}'",
                    )
                    train_source_preprocess_states[source_provider] = (
                        source_preprocess_state
                    )
                    if cache_path is not None:
                        _save_train_source_cache(
                            cache_path,
                            identity=cache_identity,
                            preprocess_state=source_preprocess_state,
                            samples=list(source_dataset),
                        )
                        log(
                            f"Fold {fold_idx} train source '{source_provider}' wrote "
                            f"{len(source_dataset)} cached samples to '{cache_path}'",
                            priority=0,
                        )
                train_source_feature_shapes = _validate_train_source_feature_shapes(
                    train_source_datasets,
                    train_sources=active_train_sources,
                )
                # The concatenated multisource train split may mix provider-specific
                # preprocessing states, so eval statistics are fit separately for
                # val/test. If decodable filtering skipped the target train source,
                # use the effective non-empty multisource train set instead of
                # falling back to the unfiltered target train split.
                eval_train_provider = filtered_train_source_provider_by_provider.get(
                    str(dataset_provider)
                )
                if target_eval_train_dataset is not None:
                    eval_train_dataset = target_eval_train_dataset
                elif eval_train_provider is not None:
                    eval_train_dataset = WindowedNeuroprobeSplitDataset(
                        eval_train_provider,
                        provider_key=str(dataset_provider),
                        split="train",
                        dataset_provider=str(dataset_provider),
                        coordinate_profile=coordinate_profile,
                        window_slicing_policy=window_slicing_policy,
                        require_coords=require_coords,
                        brain_area_key=brain_area_key,
                        max_samples_per_subject=eval_train_max_samples_per_subject,
                        sample_fraction=eval_train_sample_fraction,
                        sample_seed=eval_train_sample_seed,
                    )
                elif decodable_subject_sessions_only:
                    eval_train_dataset = split_datasets["train"]
                else:
                    eval_train_dataset = WindowedNeuroprobeSplitDataset(
                        split_providers["train"],
                        provider_key=str(dataset_provider),
                        split="train",
                        dataset_provider=str(dataset_provider),
                        coordinate_profile=coordinate_profile,
                        window_slicing_policy=window_slicing_policy,
                        require_coords=require_coords,
                        brain_area_key=brain_area_key,
                        max_samples_per_subject=eval_train_max_samples_per_subject,
                        sample_fraction=eval_train_sample_fraction,
                        sample_seed=eval_train_sample_seed,
                    )
                preprocess_state = _prepare_split_dataset(
                    eval_train_dataset,
                    preprocessor=eval_preprocessor,
                    fold_context=fold_context,
                    fold_idx=fold_idx,
                    split_label="eval train split",
                )
                # Val/test for the target provider reuse the eval-side fitted state
                # so normalization/statistical transforms are anchored to the same
                # train subset the model is effectively evaluated against.
                for split in ("val", "test"):
                    _prepare_split_dataset(
                        split_datasets[split],
                        preprocessor=eval_preprocessor,
                        fold_context=fold_context,
                        fold_idx=fold_idx,
                        split_label=f"{split} split",
                        fit_preprocessor=False,
                        preprocess_state=preprocess_state,
                    )
                # Rebuild the public train split after per-source preprocessing so
                # callers see the finalized source samples.
                split_datasets["train"] = _ConcatenatedTrainSplitDataset(
                    train_source_datasets,
                    require_coords=require_coords,
                )
            else:
                preprocess_state = _prepare_split_dataset(
                    split_datasets["train"],
                    preprocessor=preprocessor,
                    fold_context=fold_context,
                    fold_idx=fold_idx,
                    split_label="train split",
                )
                for split in ("val", "test"):
                    _prepare_split_dataset(
                        split_datasets[split],
                        preprocessor=preprocessor,
                        fold_context=fold_context,
                        fold_idx=fold_idx,
                        split_label=f"{split} split",
                        fit_preprocessor=False,
                        preprocess_state=preprocess_state,
                    )
        if (
            preprocessed_split_cache_active
            and preprocessed_split_cached_payload is None
            and normalized_preprocessed_split_cache_mode != "read_only"
            and preprocessed_split_cache_path is not None
            and preprocessed_split_cache_identity is not None
        ):
            _save_preprocessed_split_cache(
                preprocessed_split_cache_path,
                identity=preprocessed_split_cache_identity,
                preprocess_state=preprocess_state,
                split_datasets=split_datasets,
            )
            log(
                f"Fold {fold_idx} wrote preprocessed train/val/test split cache "
                f"to '{preprocessed_split_cache_path}'",
                priority=0,
            )
    else:
        # Even without preprocessing we materialize eagerly so downstream
        # runners still receive the same in-memory dataset contract.
        train_source_preprocess_states = {}
        train_source_feature_shapes = {}
        for split in ("train", "val", "test"):
            split_start = time.time()
            split_datasets[split].materialize(preprocessor=None)
            log(
                f"Fold {fold_idx} {split} split materialization completed in "
                f"{time.time() - split_start:.2f}s",
                priority=0,
            )
        if uses_train_sources:
            train_source_feature_shapes = _validate_train_source_feature_shapes(
                train_source_datasets,
                train_sources=active_train_sources,
            )
    log_fold_split_sample_counts(
        split_datasets, fold_idx=fold_idx, phase="materialized"
    )

    # Build metadata last so split summaries and feature-shape diagnostics
    # reflect the exact datasets returned in the fold payload.
    metadata = {
        "task": dataset_cfg.task,
        "label_mode": dataset_cfg.label_mode,
        **_optional_task_mode_identity(dataset_cfg),
        "regime": regime,
        "needs_region_intersection_pool": needs_pool,
        "dataset_provider": dataset_provider,
        "fold_seed": fold_seed,
        "window_slicing_policy": window_slicing_policy,
        "brain_area_key": brain_area_key,
        "test_subject": dataset_cfg.test_subject,
        "test_session": dataset_cfg.test_session,
        "train_same_subject_only": train_same_subject_only,
        "train_sample_fraction": train_sample_fraction,
        "max_train_samples_per_subject": (
            active_train_source_by_provider[str(dataset_provider)].get(
                "max_train_samples_per_subject"
            )
            if uses_train_sources
            and str(dataset_provider) in active_train_source_by_provider
            else (None if uses_train_sources else max_train_samples_per_subject)
        ),
        "uses_train_sources": uses_train_sources,
        "train_sources": list(train_sources),
        "active_train_sources": list(active_train_sources),
        "train_source_max_samples_per_subject": {
            str(source_cfg["provider"]): source_cfg.get("max_train_samples_per_subject")
            for source_cfg in active_train_sources
        },
        "skipped_empty_decodable_train_sources": list(
            skipped_empty_decodable_train_sources
        ),
        "decodable_subject_sessions_only": decodable_subject_sessions_only,
        "decodable_subject_sessions_dir": decodable_subject_sessions_dir,
        "eval_preprocessor_name": eval_preprocessor_name,
        "train_source_preprocess_states": dict(train_source_preprocess_states),
        "train_source_preprocessor_names": dict(train_source_preprocessor_names),
        "train_source_feature_shapes": {
            provider: (None if shape is None else list(shape))
            for provider, shape in train_source_feature_shapes.items()
        },
        "split_summaries": {
            split: split_datasets[split].get_split_summary()
            for split in ("train", "val", "test")
        },
    }

    fold = {
        "fold_idx": fold_idx,
        "train_split": split_datasets["train"],
        "val_split": split_datasets["val"],
        "test_split": split_datasets["test"],
        "preprocess_state": preprocess_state,
        "metadata": metadata,
        # RegressionTargets when dataset.label_mode is regression, else None.
        "regression_targets": regression_targets,
    }
    validate_fold_dict(fold)
    return fold
