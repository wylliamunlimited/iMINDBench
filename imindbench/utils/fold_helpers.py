"""Fold preparation and execution helpers for processed neuroprobe eval."""

from __future__ import annotations

import gc
import inspect
import time
import warnings
from contextlib import contextmanager

import numpy as np
import torch
from torch.utils.data import DataLoader

from imindbench.models import build_model
from imindbench.torch_runner import TorchRunner
from imindbench.utils.collate import variable_channel_collate
from imindbench.utils.data_adapter import (
    MatchedSubsetAbsent,
    build_neuroprobe_torch_fold,
)
from imindbench.utils.logging_utils import (
    DEFAULT_RESULTS_TIME_BIN,
    log,
    log_fold_metrics,
)


def _normalize_preprocess_torch_num_threads(value) -> int | None:
    """Validate and normalize the optional preprocessing thread cap."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(
            "runtime.preprocess_torch_num_threads must be an integer when set."
        )
    if int(value) < 1:
        raise ValueError("runtime.preprocess_torch_num_threads must be >= 1 when set.")
    return int(value)


@contextmanager
def _temporary_torch_num_threads(num_threads: int | None):
    """Temporarily adjust intra-op torch threads for CPU-heavy preprocessing."""
    normalized = _normalize_preprocess_torch_num_threads(num_threads)
    if normalized is None:
        yield None
        return

    original = torch.get_num_threads()
    if normalized == original:
        yield original
        return

    # Fold preparation is where CPU STFT/filter preprocessing runs. Restoring the
    # original thread count avoids unexpectedly constraining later model work.
    torch.set_num_threads(normalized)
    try:
        yield original
    finally:
        torch.set_num_threads(original)


def _runner_accepts_validation_data(runner) -> bool:
    """Whether runner.run_fold accepts explicit validation arrays."""
    run_fold = getattr(runner, "run_fold", None)
    if run_fold is None:
        return False
    try:
        parameters = inspect.signature(run_fold).parameters.values()
    except (TypeError, ValueError):
        return False

    has_var_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters
    )
    if has_var_kwargs:
        return True

    parameter_names = {parameter.name for parameter in parameters}
    return {"X_val", "y_val"} <= parameter_names


def _attach_dataset_cfg(model, dataset_cfg):
    """Attach dataset config to dataset-aware models without constraining fakes."""
    try:
        if getattr(model, "dataset_cfg", None) is None:
            model.dataset_cfg = dataset_cfg
    except Exception:
        return model
    return model


def collect_numpy_from_loader(loader, *, model=None, runner_cfg=None):
    """Collect one split DataLoader into numpy arrays."""
    xs = []
    ys = []
    expected_feature_shape = None
    for batch_idx, raw_batch in enumerate(loader):
        batch = raw_batch
        prepare_batch = getattr(model, "prepare_batch", None)
        if callable(prepare_batch):
            prepare_kwargs = {}
            if runner_cfg is not None:
                prepare_kwargs["runner_cfg"] = runner_cfg
            batch = prepare_batch(batch, **prepare_kwargs)
        # Fixed-alignment sklearn-style evaluation expects dense arrays only.
        if not isinstance(batch, dict) or "x" not in batch or "y" not in batch:
            raise ValueError(
                "fixed-channel loader materialization requires dict batches "
                "with 'x' and 'y' keys."
            )
        x = batch["x"]
        y = batch["y"]
        if torch.is_tensor(x):
            x = x.detach().cpu().numpy()
        else:
            x = np.asarray(x)
        if torch.is_tensor(y):
            y = y.detach().cpu().numpy()
        else:
            y = np.asarray(y)

        feature_shape = tuple(x.shape[1:])
        if expected_feature_shape is None:
            expected_feature_shape = feature_shape
        elif feature_shape != expected_feature_shape:
            raise ValueError(
                "Inconsistent batch feature shape while materializing fixed-channel "
                "arrays: first batch x.shape[1:]="
                f"{expected_feature_shape}, batch {batch_idx} x.shape[1:]="
                f"{feature_shape}. Ensure preprocessors/model.prepare_batch produce "
                "a consistent feature layout."
            )
        xs.append(x)
        ys.append(y)

    if not xs:
        return np.zeros((0, 0), dtype=np.float32), np.array([], dtype=np.int32)
    X = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0).astype(np.int32, copy=False)
    return X, y


def merge_eval_splits_for_sklearn(
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return merged eval split (val + test), preserving original order."""
    if y_val.size == 0:
        return X_test, y_test
    if y_test.size == 0:
        return X_val, y_val
    if X_val.ndim != X_test.ndim or X_val.shape[1:] != X_test.shape[1:]:
        raise ValueError(
            "Cannot merge validation/test splits for sklearn evaluation: "
            "feature shapes differ "
            f"{X_val.shape} vs {X_test.shape}."
        )
    return (
        np.concatenate((X_val, X_test), axis=0),
        np.concatenate((y_val, y_test), axis=0),
    )


def _sorted_class_counts(class_counts: dict[int, int]) -> dict[int, int]:
    """Return class-count mapping sorted by class id for stable logs/results."""
    return {int(label): int(class_counts[label]) for label in sorted(class_counts)}


def _merge_class_counts(left: dict[int, int], right: dict[int, int]) -> dict[int, int]:
    """Merge two class-count mappings."""
    merged = dict(left)
    for label, count in dict(right).items():
        merged[int(label)] = int(merged.get(int(label), 0) + int(count))
    return merged


def _find_unseen_eval_classes(
    train_class_counts: dict[int, int],
    eval_class_counts: dict[int, int],
) -> list[int]:
    """Return sorted eval labels that are absent from the training split."""
    train_labels = {int(label) for label in dict(train_class_counts)}
    return sorted(
        int(label)
        for label in dict(eval_class_counts)
        if int(label) not in train_labels
    )


def collect_class_counts_from_loader(loader) -> dict[int, int]:
    """Collect class-id counts from a split loader/dataset."""
    samples = loader.dataset if isinstance(loader, DataLoader) else loader
    class_counts: dict[int, int] = {}
    for raw_batch in samples:
        if not isinstance(raw_batch, dict) or "y" not in raw_batch:
            raise ValueError(
                "Unable to infer class labels from split sample/batch. "
                "Expected dict payloads with key 'y'."
            )
        batch_y = raw_batch["y"]
        if torch.is_tensor(batch_y):
            batch_y = batch_y.detach().cpu().numpy()
        else:
            batch_y = np.asarray(batch_y)
        for value in np.asarray(batch_y).reshape(-1):
            if isinstance(value, (bool, np.bool_)):
                raise TypeError("Class labels must be integers, got bool.")
            if not isinstance(value, (int, np.integer)):
                raise TypeError(
                    f"Class labels must be integers, got {type(value).__name__}."
                )
            label = int(value)
            class_counts[label] = int(class_counts.get(label, 0) + 1)
    return _sorted_class_counts(class_counts)


def build_skipped_fold_result(
    *,
    reason: str,
    insufficient_splits: list[str],
    train_class_counts: dict[int, int],
    val_class_counts: dict[int, int],
    test_class_counts: dict[int, int],
    include_val_metrics: bool,
    eval_class_counts: dict[int, int] | None = None,
) -> dict:
    """Build a standardized skipped-fold payload."""
    fold_result = {
        "status": "skipped",
        "skip_reason": str(reason),
        "insufficient_splits": [str(split) for split in insufficient_splits],
        "train_class_counts": _sorted_class_counts(train_class_counts),
        "test_class_counts": _sorted_class_counts(test_class_counts),
    }
    if include_val_metrics:
        fold_result["val_class_counts"] = _sorted_class_counts(val_class_counts)
    if eval_class_counts is not None:
        fold_result["eval_class_counts"] = _sorted_class_counts(eval_class_counts)
    return fold_result


def _init_results_population():
    """Create the in-memory results container used during one evaluation run."""
    return {
        DEFAULT_RESULTS_TIME_BIN: {
            "time_bin_start": 0.0,
            "time_bin_end": 1.0,
            "folds": [],
        }
    }


def build_torch_split_loaders(
    fold: dict,
    cfg,
    *,
    seed: int | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build split-specific torch DataLoaders for a variable-channel fold."""
    runner_cfg = cfg.runner
    batch_size = int(cfg.model.get("batch_size", 64))
    num_workers = int(runner_cfg.get("num_workers", 0))
    pin_memory = bool(runner_cfg.get("pin_memory", True))
    persistent_workers = bool(runner_cfg.get("persistent_workers", False))
    prefetch_factor = int(runner_cfg.get("prefetch_factor", 2))

    if prefetch_factor < 1:
        raise ValueError("runner.prefetch_factor must be >= 1.")

    if persistent_workers and num_workers <= 0:
        warnings.warn(
            "runner.persistent_workers=true requires runner.num_workers>0; "
            "falling back to persistent_workers=false.",
            UserWarning,
            stacklevel=2,
        )
        persistent_workers = False

    loader_prefetch_factor = prefetch_factor if num_workers > 0 else None

    train_generator = None
    if seed is not None:
        # Seed only the train loader's sampling order; val/test stay deterministic
        # via `shuffle=False`.
        train_generator = torch.Generator()
        train_generator.manual_seed(seed)

    train_loader = DataLoader(
        fold["train_split"],
        batch_size=batch_size,
        shuffle=True,
        generator=train_generator,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=loader_prefetch_factor,
        collate_fn=variable_channel_collate,
    )
    val_loader = DataLoader(
        fold["val_split"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=loader_prefetch_factor,
        collate_fn=variable_channel_collate,
    )
    test_loader = DataLoader(
        fold["test_split"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=loader_prefetch_factor,
        collate_fn=variable_channel_collate,
    )

    return train_loader, val_loader, test_loader


def run_processed_fold_loop(
    *,
    fold_iter,
    evaluate_fold,
    data_load_time: float,
    prepare_label: str = "prepared payload",
):
    """Execute the per-fold evaluation loop for processed-data runs."""
    results_population = _init_results_population()
    regression_start_time = time.time()

    for fold in fold_iter:
        # Fold iterator yields timing + payload so we can account for data
        # preparation separately from model training/inference time.
        fold_idx = fold["fold_idx"]
        fold_prepare_seconds = fold.get("prepare_seconds", 0.0)
        data_load_time += fold_prepare_seconds
        log(
            f"Fold {fold_idx}: {prepare_label} in {fold_prepare_seconds:.2f}s",
            priority=0,
        )

        fold_result = evaluate_fold(fold_idx, fold)
        fold_result["fold_idx"] = fold_idx
        results_population[DEFAULT_RESULTS_TIME_BIN]["folds"].append(fold_result)
        log_fold_metrics(fold_idx, fold_result)

    regression_run_time = time.time() - regression_start_time
    return results_population, data_load_time, regression_run_time


def iter_variable_channel_folds(
    *,
    n_folds: int,
    dataset_cfg,
    preprocessor,
    preprocessor_cfg=None,
    paths_cfg=None,
    seed: int,
    require_coords: bool,
    needs_pool: bool,
    preprocess_torch_num_threads: int | None = None,
    train_source_cache_enabled: bool = False,
    train_source_cache_dir=None,
    preprocessed_split_cache_enabled: bool = False,
    preprocessed_split_cache_dir=None,
    preprocessed_split_cache_mode: str = "read_write",
    fold_subset=None,
):
    """Yield prepared variable-channel fold payloads with timing metadata.

    fold_subset (dataset.fold_subset) limits the run to the listed folds; None
    runs every fold. A fold whose matched train subset is absent from the
    subset file is yielded with fold=None and a skip_reason, and is recorded
    as skipped.
    """
    thread_cap = _normalize_preprocess_torch_num_threads(preprocess_torch_num_threads)
    if fold_subset is None:
        fold_ids = list(range(n_folds))
    else:
        fold_ids = [int(fold_idx) for fold_idx in fold_subset]
        outside = [fold_idx for fold_idx in fold_ids if not 0 <= fold_idx < n_folds]
        if outside:
            raise ValueError(
                f"dataset.fold_subset {fold_ids} lists folds outside 0..{n_folds - 1}."
            )
        log(f"dataset.fold_subset={fold_ids}: running only these folds", priority=0)
    for fold_idx in fold_ids:
        fold_prepare_start = time.time()
        with _temporary_torch_num_threads(thread_cap) as original_threads:
            if (
                thread_cap is not None
                and original_threads is not None
                and original_threads != thread_cap
            ):
                log(
                    f"Fold {fold_idx}: limiting torch preprocessing threads "
                    f"to {thread_cap} (was {original_threads})",
                    priority=0,
                )
            try:
                fold = build_neuroprobe_torch_fold(
                    dataset_cfg,
                    preprocessor,
                    preprocessor_cfg=preprocessor_cfg,
                    paths_cfg=paths_cfg,
                    fold_idx=fold_idx,
                    seed=seed + fold_idx,
                    require_coords=require_coords,
                    needs_pool=needs_pool,
                    train_source_cache_enabled=train_source_cache_enabled,
                    train_source_cache_dir=train_source_cache_dir,
                    preprocessed_split_cache_enabled=preprocessed_split_cache_enabled,
                    preprocessed_split_cache_dir=preprocessed_split_cache_dir,
                    preprocessed_split_cache_mode=preprocessed_split_cache_mode,
                )
            except MatchedSubsetAbsent as exc:
                yield {
                    "fold_idx": fold_idx,
                    "prepare_seconds": time.time() - fold_prepare_start,
                    "fold": None,
                    "skip_reason": str(exc),
                }
                continue
        yield {
            "fold_idx": fold_idx,
            "prepare_seconds": time.time() - fold_prepare_start,
            "fold": fold,
        }


def evaluate_variable_fold(
    fold_idx,
    fold_payload,
    *,
    cfg,
    runner,
    seed: int,
):
    """Evaluate one prepared fold and return fold metrics."""
    fold = fold_payload["fold"]
    if fold is None:
        # iter_variable_channel_folds found no matched train subset.
        return build_skipped_fold_result(
            reason=fold_payload["skip_reason"],
            insufficient_splits=["train"],
            train_class_counts={},
            val_class_counts={},
            test_class_counts={},
            include_val_metrics=False,
        )
    train_loader, val_loader, test_loader = build_torch_split_loaders(
        fold,
        cfg,
        seed=seed + fold_idx,
    )
    train_class_counts = collect_class_counts_from_loader(train_loader)
    val_class_counts = collect_class_counts_from_loader(val_loader)
    test_class_counts = collect_class_counts_from_loader(test_loader)

    if len(train_class_counts) < 2:
        return build_skipped_fold_result(
            reason="insufficient_class_coverage",
            insufficient_splits=["train"],
            train_class_counts=train_class_counts,
            val_class_counts=val_class_counts,
            test_class_counts=test_class_counts,
            include_val_metrics=(cfg.model.backend == "torch"),
        )

    runner_cfg = cfg.get("runner", {})
    coord_index_policy = runner_cfg.get("coord_index_policy", "round_clamp")

    if cfg.model.backend == "sklearn":
        merge_val_into_test = cfg.dataset.merge_val_into_test
        eval_class_counts = dict(test_class_counts)
        if merge_val_into_test:
            eval_class_counts = _merge_class_counts(eval_class_counts, val_class_counts)
        if merge_val_into_test:
            if len(eval_class_counts) < 2:
                return build_skipped_fold_result(
                    reason="insufficient_class_coverage",
                    insufficient_splits=["eval"],
                    train_class_counts=train_class_counts,
                    val_class_counts=val_class_counts,
                    test_class_counts=test_class_counts,
                    include_val_metrics=False,
                    eval_class_counts=eval_class_counts,
                )
        else:
            insufficient_splits: list[str] = []
            if len(val_class_counts) < 2:
                insufficient_splits.append("val")
            if len(test_class_counts) < 2:
                insufficient_splits.append("test")
            if insufficient_splits:
                return build_skipped_fold_result(
                    reason="insufficient_class_coverage",
                    insufficient_splits=insufficient_splits,
                    train_class_counts=train_class_counts,
                    val_class_counts=val_class_counts,
                    test_class_counts=test_class_counts,
                    include_val_metrics=True,
                )

            if _find_unseen_eval_classes(train_class_counts, val_class_counts):
                insufficient_splits.append("val")
            if _find_unseen_eval_classes(train_class_counts, test_class_counts):
                insufficient_splits.append("test")
            if insufficient_splits:
                return build_skipped_fold_result(
                    reason="insufficient_class_coverage",
                    insufficient_splits=insufficient_splits,
                    train_class_counts=train_class_counts,
                    val_class_counts=val_class_counts,
                    test_class_counts=test_class_counts,
                    include_val_metrics=True,
                )

        fold_model = _attach_dataset_cfg(build_model(cfg.model), cfg.dataset)
        # sklearn evaluation uses dense arrays rather than
        # training directly from variable-channel torch batches.
        X_train, y_train = collect_numpy_from_loader(
            train_loader,
            model=fold_model,
            runner_cfg={"coord_index_policy": coord_index_policy},
        )
        X_test, y_test = collect_numpy_from_loader(
            test_loader,
            model=fold_model,
            runner_cfg={"coord_index_policy": coord_index_policy},
        )
        X_val = None
        y_val = None
        if not merge_val_into_test:
            X_val, y_val = collect_numpy_from_loader(
                val_loader,
                model=fold_model,
                runner_cfg={"coord_index_policy": coord_index_policy},
            )
        if merge_val_into_test:
            X_val, y_val = collect_numpy_from_loader(
                val_loader,
                model=fold_model,
                runner_cfg={"coord_index_policy": coord_index_policy},
            )
            X_test, y_test = merge_eval_splits_for_sklearn(
                X_val,
                y_val,
                X_test,
                y_test,
            )
        if X_train.ndim != 2 or X_test.ndim != 2:
            raise ValueError(
                "sklearn evaluation expects 2D features after "
                "model.prepare_batch materialization, got "
                f"X_train.shape={X_train.shape}, "
                f"X_test.shape={X_test.shape}."
            )
        run_fold_kwargs = {}
        if (
            not merge_val_into_test
            and X_val is not None
            and y_val is not None
            and y_val.size > 0
            and _runner_accepts_validation_data(runner)
        ):
            run_fold_kwargs["X_val"] = X_val
            run_fold_kwargs["y_val"] = y_val
        fold_result = runner.run_fold(
            fold_model,
            X_train,
            y_train,
            X_test,
            y_test,
            **run_fold_kwargs,
        )
    else:
        insufficient_splits: list[str] = []
        if len(val_class_counts) < 2:
            insufficient_splits.append("val")
        if len(test_class_counts) < 2:
            insufficient_splits.append("test")
        if insufficient_splits:
            return build_skipped_fold_result(
                reason="insufficient_class_coverage",
                insufficient_splits=insufficient_splits,
                train_class_counts=train_class_counts,
                val_class_counts=val_class_counts,
                test_class_counts=test_class_counts,
                include_val_metrics=True,
            )

        # Torch losses require train-derived classes to cover every eval target.
        # Skip these folds early instead of failing inside the runner.
        if _find_unseen_eval_classes(train_class_counts, val_class_counts):
            insufficient_splits.append("val")
        if _find_unseen_eval_classes(train_class_counts, test_class_counts):
            insufficient_splits.append("test")
        if insufficient_splits:
            return build_skipped_fold_result(
                reason="insufficient_class_coverage",
                insufficient_splits=insufficient_splits,
                train_class_counts=train_class_counts,
                val_class_counts=val_class_counts,
                test_class_counts=test_class_counts,
                include_val_metrics=True,
            )

        fold_model = _attach_dataset_cfg(build_model(cfg.model), cfg.dataset)
        # Variable-channel and aligned Torch models share the same
        # TorchRunner path once split loaders have been built.
        if not isinstance(runner, TorchRunner):
            raise NotImplementedError("model.backend=torch requires TorchRunner.")
        fold_result = runner.run_fold(
            fold_model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            fold_idx=fold_idx,
        )

    del fold_model, fold
    gc.collect()
    return fold_result
