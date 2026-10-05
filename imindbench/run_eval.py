"""
Main evaluation script for neuroprobe using Hydra configuration.
"""

import logging
import sys
from functools import partial

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from imindbench.preprocessors import build_preprocessor, describe_preprocessor
from imindbench.sklearn_runner import SKLearnRunner
from imindbench.torch_runner import TorchRunner
from imindbench.utils import fold_helpers, logging_utils
from imindbench.utils.pipeline_contracts import (
    needs_region_intersection_pool,
    resolve_provider_n_folds,
    validate_eval_config,
)

# Optional wandb import
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False
from imindbench.utils.logging_utils import (
    log,
    normalize_wandb_tags,
    set_verbose,
)

_HELP_COMPOSITION_DEFAULTS = (
    "paths=example",
    "model=logistic",
    "preprocessor=stft_2048Hz",
)


def cli() -> None:
    """Run the Hydra entrypoint, composing placeholder groups for help only."""
    original_argv = sys.argv
    try:
        effective_argv = list(original_argv)
        if any(argument in {"-h", "--help"} for argument in effective_argv[1:]):
            selected_groups = {
                argument.split("=", 1)[0]
                for argument in effective_argv[1:]
                if "=" in argument
            }
            effective_argv.extend(
                override
                for override in _HELP_COMPOSITION_DEFAULTS
                if override.split("=", 1)[0] not in selected_groups
            )
        sys.argv = effective_argv
        main()
    finally:
        sys.argv = original_argv


@hydra.main(config_path="conf", config_name="config", version_base="1.1")
def main(cfg: DictConfig) -> None:
    """Main evaluation function."""
    root_logger = logging.getLogger()
    wandb_run = None
    try:
        # The public evaluation boundary validates once, before external setup.
        validate_eval_config(cfg)
        runtime_cfg = cfg.runtime
        set_verbose(runtime_cfg.get("verbose", True))
        log("Starting neuroprobe evaluation", priority=0)
        log(f"Configuration:\n{OmegaConf.to_yaml(cfg, resolve=True)}", priority=1)

        # Initialize wandb if enabled
        if cfg.get("wandb", {}).get("enabled", False):
            if not WANDB_AVAILABLE:
                log(
                    "WARNING: wandb is enabled in config but not installed. Install with: pip install wandb",
                    priority=0,
                )
            else:
                wandb_cfg = cfg.get("wandb", {})
                # Prepare config for wandb (convert OmegaConf to dict)
                wandb_config = OmegaConf.to_container(cfg, resolve=True)

                # Initialize wandb
                wandb.init(
                    project=wandb_cfg.get("project", "neuroprobe_eval"),
                    entity=wandb_cfg.get("entity") or None,
                    name=wandb_cfg.get("name") or None,
                    tags=normalize_wandb_tags(wandb_cfg.get("tags")),
                    notes=wandb_cfg.get("notes") or None,
                    group=wandb_cfg.get("group") or None,
                    config=wandb_config,
                    reinit=True,  # Allow reinitialization for multiple runs
                )
                wandb_run = wandb.run
                log(
                    f"Wandb initialized: project={wandb_cfg.get('project')}, run={wandb_run.name}",
                    priority=0,
                )

        # Set random seeds
        seed = runtime_cfg.seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Build components
        preprocessor = build_preprocessor(cfg.preprocessor)
        runner = (
            SKLearnRunner(cfg) if cfg.model.backend == "sklearn" else TorchRunner(cfg)
        )  # Model will be built per fold

        # Pass wandb_run to runner if it's a TorchRunner
        if isinstance(runner, TorchRunner) and wandb_run is not None:
            runner.set_wandb_run(wandb_run)

        log(
            f"Using preprocessor: {describe_preprocessor(cfg.preprocessor)}", priority=0
        )
        log(f"Using model: {cfg.model.name}", priority=0)

        _run_processed_evaluation(
            cfg,
            preprocessor,
            runner,
            wandb_run,
        )
    except Exception:
        # Ensure uncaught failures are persisted to Hydra's run_eval.log file.
        root_logger.exception("Unhandled exception during imindbench.run_eval")
        raise
    finally:
        if wandb_run is not None:
            wandb.finish()


def _run_processed_evaluation(
    cfg,
    preprocessor,
    runner,
    wandb_run=None,
):
    """Run processed-data evaluation with the config already validated by main.

    This path owns dataset-provider routing, fold iteration, result formatting,
    and final output persistence for the variable-channel evaluation stack.
    """
    # Canonical processed path: provider-selected variable-channel datasets.
    log("Loading processed data via variable-channel split adapter", priority=0)
    dataset_cfg = cfg.dataset
    data_load_time = 0.0

    subject_id = dataset_cfg.test_subject
    trial_id = dataset_cfg.test_session
    eval_name = dataset_cfg.task
    regime = dataset_cfg.regime
    seed = cfg.runtime.seed

    dataset_provider = dataset_cfg.provider
    requires_aligned = cfg.model.requires_aligned_channels
    needs_pool = needs_region_intersection_pool(
        dataset_provider, regime, requires_aligned
    )
    n_folds = resolve_provider_n_folds(
        dataset_provider=dataset_provider,
        regime=regime,
    )

    log(f"Using dataset.provider='{dataset_provider}'", priority=0)
    log(f"Using dataset.regime='{regime}'", priority=0)
    log(
        f"Using window slicing policy: {dataset_cfg.get('window_slicing_policy', 'ceil')}",
        priority=0,
    )
    log(f"Using n_folds={n_folds} from dataset class API", priority=0)

    preprocess_type = describe_preprocessor(cfg.preprocessor)
    model_name = cfg.model.name
    # Keep result JSON in the Hydra run folder for per-run portability.
    file_save_path = logging_utils.resolve_result_output_path(
        eval_name=eval_name,
        subject_id=subject_id,
        trial_id=trial_id,
    )
    require_coords = cfg.model.requires_coords
    # Short-circuit before fold construction only for readable result JSON.
    if logging_utils.should_skip_existing_output(cfg, file_save_path):
        return

    results_population, data_load_time, regression_run_time = (
        fold_helpers.run_processed_fold_loop(
            fold_iter=fold_helpers.iter_variable_channel_folds(
                n_folds=n_folds,
                dataset_cfg=dataset_cfg,
                preprocessor=preprocessor,
                preprocessor_cfg=cfg.preprocessor,
                paths_cfg=cfg.get("paths", None),
                seed=seed,
                require_coords=require_coords,
                needs_pool=needs_pool,
                preprocess_torch_num_threads=cfg.runtime.get(
                    "preprocess_torch_num_threads", None
                ),
                train_source_cache_enabled=cfg.runtime.get(
                    "train_source_cache_enabled", False
                ),
                train_source_cache_dir=cfg.runtime.get("train_source_cache_dir", None),
                preprocessed_split_cache_enabled=cfg.runtime.get(
                    "preprocessed_split_cache_enabled", False
                ),
                preprocessed_split_cache_dir=cfg.runtime.get(
                    "preprocessed_split_cache_dir", None
                ),
                preprocessed_split_cache_mode=cfg.runtime.get(
                    "preprocessed_split_cache_mode", "read_write"
                ),
                fold_subset=dataset_cfg.get("fold_subset", None),
            ),
            evaluate_fold=partial(
                fold_helpers.evaluate_variable_fold,
                cfg=cfg,
                runner=runner,
                seed=seed,
            ),
            data_load_time=data_load_time,
            prepare_label="prepared variable-channel payload",
        )
    )

    if hasattr(preprocessor, "unload_model"):
        preprocessor.unload_model()
    logging_utils.format_and_save_results(
        cfg=cfg,
        dataset_provider=dataset_provider,
        model_name=model_name,
        preprocess_type=preprocess_type,
        subject_id=subject_id,
        trial_id=trial_id,
        eval_name=eval_name,
        results_splits_type=regime,
        results_population=results_population,
        data_load_time=data_load_time,
        regression_run_time=regression_run_time,
        file_save_path=file_save_path,
    )
    logging_utils.log_final_wandb_metrics(wandb_run, results_population)


if __name__ == "__main__":
    cli()
