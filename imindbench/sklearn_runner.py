"""
Runner for sklearn models (like PopulationTransformer's Runner pattern).
"""

import gc
import inspect
import time
from contextlib import ExitStack, contextmanager

import numpy as np
from threadpoolctl import ThreadpoolController

from imindbench.base_runner import BaseRunner
from imindbench.utils import regression as regression_utils
from imindbench.utils.logging_utils import log


def _normalize_sklearn_num_threads(value) -> int | None:
    """Validate and normalize the optional sklearn BLAS/OpenMP thread cap."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError("runtime.sklearn_num_threads must be an integer when set.")
    if int(value) < 1:
        raise ValueError("runtime.sklearn_num_threads must be >= 1 when set.")
    return int(value)


@contextmanager
def _temporary_sklearn_thread_limits(num_threads: int | None):
    """Temporarily cap BLAS/OpenMP threads for sklearn numeric kernels."""
    normalized = _normalize_sklearn_num_threads(num_threads)
    if normalized is None:
        yield None
        return

    controller = ThreadpoolController()
    with ExitStack() as stack:
        # This is an upper bound, not a request to grow existing pools. Raising
        # small OpenBLAS pools can segfault SciPy's L-BFGS native kernel.
        for library in controller.info():
            current_threads = library["num_threads"]
            if current_threads is not None and current_threads > normalized:
                # NumPy and SciPy can load different libraries with the same prefix.
                stack.enter_context(
                    controller.select(filepath=library["filepath"]).limit(
                        limits=normalized
                    )
                )
        yield normalized


def _fit_accepts_validation_data(model) -> bool:
    """Whether model.fit supports explicit validation arrays."""
    fit = getattr(model, "fit", None)
    if fit is None:
        return False
    try:
        parameters = inspect.signature(fit).parameters.values()
    except (TypeError, ValueError):
        return False

    has_var_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters
    )
    if has_var_kwargs:
        return True

    parameter_names = {parameter.name for parameter in parameters}
    return {"X_val", "y_val"} <= parameter_names


class SKLearnRunner(BaseRunner):
    """Runner for sklearn models."""

    def __init__(self, cfg):
        super().__init__(cfg)
        runtime_cfg = cfg.get("runtime", {}) if hasattr(cfg, "get") else {}
        self.sklearn_num_threads = _normalize_sklearn_num_threads(
            runtime_cfg.get("sklearn_num_threads", None)
            if hasattr(runtime_cfg, "get")
            else None
        )

    def run_fold(
        self, model, X_train, y_train, X_test, y_test, *, X_val=None, y_val=None
    ):
        """
        Train and evaluate a single fold.

        Args:
            model: sklearn model instance
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels

        Returns:
            Dictionary with train/test metrics plus optional val metrics
        """
        train_shape = tuple(np.asarray(X_train).shape)
        val_shape = None if X_val is None else tuple(np.asarray(X_val).shape)
        test_shape = tuple(np.asarray(X_test).shape)
        log(
            "SKLearnRunner: starting fold with "
            f"train_shape={train_shape} val_shape={val_shape} "
            f"test_shape={test_shape} "
            f"sklearn_num_threads={self.sklearn_num_threads}",
            priority=0,
        )

        # Data is already standardized in preprocessing pipeline
        gc.collect()

        # Train
        with _temporary_sklearn_thread_limits(self.sklearn_num_threads):
            fit_start = time.time()
            if X_val is None or y_val is None:
                model.fit(X_train, y_train)
            elif _fit_accepts_validation_data(model):
                model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
            else:
                model.fit(X_train, y_train)
            fit_elapsed = time.time() - fit_start
        log(
            f"SKLearnRunner: model.fit completed in {fit_elapsed:.2f}s",
            priority=0,
        )

        # Evaluate
        train_acc, train_auc = self._evaluate(model, X_train, y_train, split="train")
        val_metrics = None
        if X_val is not None and y_val is not None and len(y_val) > 0:
            val_metrics = self._evaluate(model, X_val, y_val, split="val")
        test_acc, test_auc = self._evaluate(model, X_test, y_test, split="test")

        # Clean up
        gc.collect()

        result = {
            "train_accuracy": float(train_acc),
            "train_roc_auc": float(train_auc),
            "test_accuracy": float(test_acc),
            "test_roc_auc": float(test_auc),
        }
        if val_metrics is not None:
            val_acc, val_auc = val_metrics
            result["val_accuracy"] = float(val_acc)
            result["val_roc_auc"] = float(val_auc)
        return result

    def run_fold_regression(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        *,
        X_val=None,
        y_val=None,
        recording_ids,
        targets,
    ):
        """Fit one multi-output Ridge regression for a regression fold.

        y_* hold each window's row in its recording's target table; the
        trajectories are looked up in ``targets`` by (recording id, row). One
        Ridge maps the features to every sample of the trajectory at once, with
        alpha = model.regression_head_lambda.
        """
        from sklearn.linear_model import Ridge

        alpha = float(self.cfg.model.get("regression_head_lambda", 1.0))
        Y_train, mean_train = targets.lookup(recording_ids["train"], y_train)
        Y_test, mean_test = targets.lookup(recording_ids["test"], y_test)
        X_train = np.asarray(X_train, dtype=np.float64)
        log(
            "SKLearnRunner: regression fold "
            f"train={X_train.shape} test={np.asarray(X_test).shape} "
            f"traj_len={Y_train.shape[1]} alpha={alpha}",
            priority=0,
        )
        with _temporary_sklearn_thread_limits(self.sklearn_num_threads):
            fit_start = time.time()
            ridge = Ridge(alpha=alpha)
            ridge.fit(X_train, Y_train)
            fit_elapsed = time.time() - fit_start
            scores = {
                "train": regression_utils.score(
                    ridge.predict(X_train), Y_train, mean_train
                ),
                "val": None,
                "test": regression_utils.score(
                    ridge.predict(X_test), Y_test, mean_test
                ),
            }
            if X_val is not None and y_val is not None and len(y_val) > 0:
                Y_val, mean_val = targets.lookup(recording_ids["val"], y_val)
                scores["val"] = regression_utils.score(
                    ridge.predict(X_val), Y_val, mean_val
                )
        log(f"SKLearnRunner: Ridge fit completed in {fit_elapsed:.2f}s", priority=0)
        gc.collect()
        result = self.build_regression_fold_result(scores)
        result["head"] = f"sklearn_ridge(alpha={alpha})"
        result["target"] = str(targets.target)
        return result

    def _evaluate(self, model, X, y, *, split: str):
        """Evaluate model and return (accuracy, roc_auc)."""
        with _temporary_sklearn_thread_limits(self.sklearn_num_threads):
            eval_start = time.time()
            # Get predictions
            y_proba = model.predict_proba(X)
            accuracy, roc_auc = self._compute_metrics(y, y_proba, model.classes_)
        log(
            f"SKLearnRunner: {split} predict_proba + metrics completed in "
            f"{time.time() - eval_start:.2f}s",
            priority=0,
        )
        return accuracy, roc_auc
