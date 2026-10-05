"""
Runner for PyTorch models.
"""

import gc
import inspect
import os
import random
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from imindbench.base_runner import BaseRunner
from imindbench.utils import regression as regression_utils
from imindbench.utils.logging_utils import log


class TorchRunner(BaseRunner):
    """Train and evaluate torch models on split-specific DataLoaders."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.deterministic = self._configure_determinism()
        self.device = self._get_device(cfg)
        self.wandb_run = None  # Will be set via set_wandb_run() if wandb is enabled

    def set_wandb_run(self, wandb_run):
        """Set wandb run object for logging."""
        self.wandb_run = wandb_run

    def _configure_determinism(self):
        """Configure PyTorch deterministic settings based on config."""
        runtime_cfg = getattr(self.cfg, "runtime", {})
        deterministic = runtime_cfg.get("deterministic", True)
        # Apply both states explicitly: multiple folds/runners share Torch globals.
        if deterministic:
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(deterministic)
        log(f"[TorchRunner] Deterministic mode: {deterministic}", priority=0)
        return deterministic

    def _get_device(self, cfg):
        """Get device from config or auto-detect."""
        device_str = cfg.model.get("device", "auto")

        if device_str == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif device_str == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but not available")
            device = torch.device("cuda")
        elif device_str == "cpu":
            device = torch.device("cpu")
        else:
            # Allow specific device strings like 'cuda:0'
            device = torch.device(device_str)

        # Verify device is valid and log
        if device.type == "cuda":
            if device.index is not None:
                if device.index >= torch.cuda.device_count():
                    import warnings

                    warnings.warn(
                        f"GPU {device.index} not available (only {torch.cuda.device_count()} GPUs). Falling back to CPU.",
                        stacklevel=2,
                    )
                    device = torch.device("cpu")
                else:
                    # Set the current device to ensure it's accessible
                    torch.cuda.set_device(device.index)

        log(
            f"[TorchRunner] Device configured: {device_str} -> {device} (CUDA available: {torch.cuda.is_available()}, GPU count: {torch.cuda.device_count() if torch.cuda.is_available() else 0})",
            priority=0,
        )
        return device

    def run_fold(
        self,
        model,
        *,
        train_loader: DataLoader | None = None,
        val_loader: DataLoader | None = None,
        test_loader: DataLoader | None = None,
        fold_idx=None,
        regression_targets=None,
    ):
        """
        Train and evaluate a single fold from split DataLoaders.

        Args:
            model: PyTorch model instance
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            test_loader: Test DataLoader
            fold_idx: Fold index for the training seed and wandb logging; None uses 0.
            regression_targets: The fold's RegressionTargets; required when
                dataset.label_mode is regression, ignored otherwise.

        Returns:
            Dictionary with train_accuracy, train_roc_auc, val_accuracy, val_roc_auc, test_accuracy, test_roc_auc
        """
        if train_loader is None or val_loader is None or test_loader is None:
            raise ValueError(
                "run_fold requires train_loader, val_loader, and test_loader."
            )
        if not isinstance(train_loader, DataLoader):
            raise TypeError("train_loader must be a torch.utils.data.DataLoader.")
        if not isinstance(val_loader, DataLoader):
            raise TypeError("val_loader must be a torch.utils.data.DataLoader.")
        if not isinstance(test_loader, DataLoader):
            raise TypeError("test_loader must be a torch.utils.data.DataLoader.")

        # Cache misses may initialize pretrained encoders and consume global RNG.
        # Start training independently, using the same fold offset as sampling.
        fold_seed = self.cfg.runtime.seed + (0 if fold_idx is None else fold_idx)
        random.seed(fold_seed)
        np.random.seed(fold_seed % (2**32))
        torch.manual_seed(fold_seed)  # Seeds CPU and all CUDA devices.
        log(f"[TorchRunner] Training seed: {fold_seed}", priority=0)

        if regression_utils.is_regression(self.cfg):
            return self._run_fold_regression(
                model,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                fold_idx=fold_idx,
                targets=regression_targets,
            )

        classes = self._infer_classes_from_loader(train_loader)
        n_classes = len(classes)
        class_to_index = self._build_class_index_map(classes)
        model.classes_ = classes

        if model.model is None:
            try:
                example_raw = next(iter(train_loader))
            except StopIteration as exc:
                raise ValueError(
                    "train_loader must contain at least one batch."
                ) from exc

            example_batch = self._prepare_model_batch(model, example_raw)
            example_inputs, _, _, _, _ = self._extract_batch_tensors(example_batch)
            input_shape = tuple(example_inputs.shape[1:])

            model.build_model(input_shape, n_classes, device=self.device)
            # Preserve raw task labels for downstream metrics/prediction interfaces
            # even when loss targets need contiguous 0..n-1 indices.
            model.classes_ = classes
            _log_model_summary(model, input_shape, n_classes)

        training_mode = self.cfg.model.get("training_mode", "epoch_based")
        if training_mode == "steps_based":
            self._train_steps_based_loader(
                model,
                train_loader=train_loader,
                val_loader=val_loader,
                n_classes=n_classes,
                classes=classes,
                class_to_index=class_to_index,
                fold_idx=fold_idx,
            )
        else:
            self._train_with_early_stopping_loader(
                model,
                train_loader=train_loader,
                val_loader=val_loader,
                n_classes=n_classes,
                classes=classes,
                class_to_index=class_to_index,
                fold_idx=fold_idx,
            )

        train_acc, train_auc, _, _ = self._evaluate_loader(
            model,
            train_loader,
            n_classes=n_classes,
            classes=classes,
            class_to_index=class_to_index,
            criterion=None,
        )
        val_acc, val_auc, _, _ = self._evaluate_loader(
            model,
            val_loader,
            n_classes=n_classes,
            classes=classes,
            class_to_index=class_to_index,
            criterion=None,
        )
        test_acc, test_auc, _, _ = self._evaluate_loader(
            model,
            test_loader,
            n_classes=n_classes,
            classes=classes,
            class_to_index=class_to_index,
            criterion=None,
        )

        if self.wandb_run is not None and fold_idx is not None:
            self.wandb_run.log(
                {
                    f"fold_{fold_idx}/train_accuracy": float(train_acc),
                    f"fold_{fold_idx}/train_roc_auc": float(train_auc),
                    f"fold_{fold_idx}/val_accuracy": float(val_acc),
                    f"fold_{fold_idx}/val_roc_auc": float(val_auc),
                    f"fold_{fold_idx}/test_accuracy": float(test_acc),
                    f"fold_{fold_idx}/test_roc_auc": float(test_auc),
                }
            )

        torch.cuda.empty_cache()
        gc.collect()
        return {
            "train_accuracy": float(train_acc),
            "train_roc_auc": float(train_auc),
            "val_accuracy": float(val_acc),
            "val_roc_auc": float(val_auc),
            "test_accuracy": float(test_acc),
            "test_roc_auc": float(test_auc),
        }

    def _infer_classes_from_loader(self, loader: DataLoader) -> np.ndarray:
        """Infer sorted unique class ids from DataLoader batches."""
        labels: set[int] = set()
        for raw_batch in loader:
            if not isinstance(raw_batch, dict) or "y" not in raw_batch:
                raise ValueError(
                    "Unable to infer class labels from DataLoader batch. "
                    "Expected dict batches with key 'y'."
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
                labels.add(int(value))

        if not labels:
            raise ValueError("Training loader is empty; cannot infer classes.")
        return np.asarray(sorted(labels), dtype=np.int64)

    @staticmethod
    def _build_class_index_map(classes: np.ndarray) -> dict[int, int]:
        """Map raw class ids to contiguous indices expected by CrossEntropyLoss."""
        return {
            int(label): idx for idx, label in enumerate(np.asarray(classes).tolist())
        }

    def _encode_targets_for_loss(
        self,
        target: torch.Tensor,
        *,
        class_to_index: dict[int, int],
        context: str,
    ) -> torch.Tensor:
        """Encode raw task labels into contiguous 0..n-1 indices for torch losses."""
        if not torch.is_tensor(target):
            target = torch.as_tensor(target, dtype=torch.long)
        else:
            target = target.long()

        if not class_to_index:
            raise ValueError(f"{context}: class_to_index must be non-empty.")

        # Fast path when raw labels already match the contiguous class space.
        if all(class_to_index.get(idx) == idx for idx in range(len(class_to_index))):
            return target

        encoded = torch.full_like(target, fill_value=-1)
        for raw_label, encoded_idx in class_to_index.items():
            encoded[target == int(raw_label)] = int(encoded_idx)
        if torch.any(encoded < 0):
            unknown = torch.unique(target[encoded < 0]).detach().cpu().tolist()
            raise ValueError(
                f"{context}: encountered labels not present in the train class map: "
                f"{unknown}. train classes={sorted(class_to_index.keys())}"
            )
        return encoded

    # ---- regression track ---------------------------------------------------
    # Reached only when dataset.label_mode is regression. The model's output
    # layer predicts the whole trajectory of a window (traj_len values), so the
    # model is built with n_classes = traj_len and its output is used directly,
    # never through a softmax. A window's y is its row in the target table; the
    # trajectory comes from the fold's RegressionTargets.

    def _run_fold_regression(
        self,
        model,
        *,
        train_loader,
        val_loader,
        test_loader,
        fold_idx=None,
        targets=None,
    ):
        if targets is None:
            raise ValueError(
                "A regression fold needs its RegressionTargets "
                "(fold['regression_targets'] from the fold builder)."
            )
        traj_len = targets.traj_len
        classes = np.arange(traj_len, dtype=np.int64)  # output size only
        model.classes_ = classes
        if model.model is None:
            try:
                example_raw = next(iter(train_loader))
            except StopIteration as exc:
                raise ValueError(
                    "train_loader must contain at least one batch."
                ) from exc
            example_batch = self._prepare_model_batch(model, example_raw)
            example_inputs, _, _, _, _ = self._extract_batch_tensors(example_batch)
            input_shape = tuple(example_inputs.shape[1:])
            # Every model sizes its output layer from n_classes, so no model
            # needs its own regression code.
            model.build_model(input_shape, int(traj_len), device=self.device)
            model.classes_ = classes
            _log_model_summary(model, input_shape, int(traj_len))

        head = self.cfg.model.get("regression_head", "trained_mse")
        if head == "ridge":
            self._fit_head_ridge_regression(
                model, train_loader=train_loader, targets=targets
            )
        else:
            self._train_regression(
                model,
                train_loader=train_loader,
                val_loader=val_loader,
                targets=targets,
                fold_idx=fold_idx,
            )

        scores = {
            split: regression_utils.score(
                *self._evaluate_regression_loader(model, loader, targets)
            )
            for split, loader in (
                ("train", train_loader),
                ("val", val_loader),
                ("test", test_loader),
            )
        }
        if self.wandb_run is not None and fold_idx is not None:
            self.wandb_run.log(
                {
                    f"fold_{fold_idx}/{split}_{key}": float(scores[split][key])
                    for split in scores
                    for key in ("traj_r", "mean_r")
                }
            )
        torch.cuda.empty_cache()
        gc.collect()
        result = self.build_regression_fold_result(scores)
        result["head"] = head
        result["target"] = str(targets.target)
        return result

    def _forward_prepared(self, model, batch_x, batch_coords, batch_seq_id, kwargs):
        """Forward one prepared batch the same way training batches are run."""
        return self._forward_model(
            model.model,
            batch_x.to(self.device),
            batch_coords,
            batch_seq_id,
            accepts_coords=getattr(model, "accepts_coords", False),
            model_kwargs=kwargs,
        )

    def _collect_regression_batch(self, model, raw_batch):
        """Prepared tensors plus the (recording id, row) of every window."""
        batch = self._prepare_model_batch(model, raw_batch)
        batch_x, batch_y, batch_coords, batch_seq_id, kwargs = (
            self._extract_batch_tensors(batch)
        )
        rows = batch_y.detach().cpu().numpy().reshape(-1)
        recording_ids = [str(rid) for rid in raw_batch["recording_ids"]]
        return batch_x, batch_coords, batch_seq_id, kwargs, recording_ids, rows

    def _evaluate_regression_loader(self, model, loader, targets):
        """(predicted trajectories, true trajectories, true means) in loader order."""
        model.model.eval()
        preds, trues, means = [], [], []
        with torch.no_grad():
            for raw_batch in loader:
                batch_x, coords, seq_id, kwargs, rids, rows = (
                    self._collect_regression_batch(model, raw_batch)
                )
                outputs = self._forward_prepared(model, batch_x, coords, seq_id, kwargs)
                preds.append(outputs.float().cpu().numpy())
                traj, mean = targets.lookup(rids, rows)
                trues.append(traj)
                means.append(mean)
        if not preds:
            raise ValueError("Cannot evaluate regression on an empty loader.")
        pred = self._unscale_regression_prediction(model, np.concatenate(preds, axis=0))
        return pred, np.concatenate(trues, axis=0), np.concatenate(means, axis=0)

    @staticmethod
    def _unscale_regression_prediction(model, pred):
        """Undo the train-split target standardization of the trained head."""
        stats = getattr(model, "_regression_target_stats", None)
        if stats is None:
            return pred
        mu, sd = stats
        return pred * sd + mu

    def _train_regression(self, model, *, train_loader, val_loader, targets, fold_idx):
        """Train with mean-squared error; keep the state with the best val traj_r.

        Targets are standardized with the train split's mean and standard
        deviation so one learning rate works for targets whose units differ by
        orders of magnitude. Predictions are mapped back before scoring, and
        Pearson r does not depend on scale.

        The loop follows the classification trainer: epoch_based runs up to
        model.max_iter epochs with model.patience early stopping (improvement
        must exceed model.tol), steps_based runs model.total_steps steps and
        validates every model.validation_interval steps. The selection warm-up
        keys apply the same way.
        """
        rows_all, rids_all = [], []
        for raw_batch in train_loader:
            rows_all.append(np.asarray(raw_batch["y"]).reshape(-1))
            rids_all.extend(str(rid) for rid in raw_batch["recording_ids"])
        train_traj, _ = targets.lookup(rids_all, np.concatenate(rows_all))
        mu = float(train_traj.mean())
        sd = float(train_traj.std())
        sd = sd if sd > 1e-12 else 1.0
        model._regression_target_stats = (mu, sd)

        criterion = nn.MSELoss()
        training_mode = self.cfg.model.get("training_mode", "epoch_based")
        max_iter = int(self.cfg.model.get("max_iter", 100))
        total_steps = int(self.cfg.model.get("total_steps", 2000))
        optimizer, scheduler, _ = self._create_optimizer_and_scheduler(
            model,
            total_steps_override=(
                total_steps
                if training_mode == "steps_based"
                else max_iter * len(train_loader)
            ),
        )
        fold_label = f"Fold {fold_idx}" if fold_idx is not None else "Fold"
        best = {"score": float("-inf"), "state": None}

        def train_step(raw_batch):
            model.model.train()
            batch_x, coords, seq_id, kwargs, rids, rows = (
                self._collect_regression_batch(model, raw_batch)
            )
            traj, _ = targets.lookup(rids, rows)
            target = torch.as_tensor(
                (traj - mu) / sd, dtype=torch.float32, device=self.device
            )
            optimizer.zero_grad()
            outputs = self._forward_prepared(model, batch_x, coords, seq_id, kwargs)
            loss = criterion(outputs.float(), target)
            loss.backward()
            grad_clip = self.cfg.model.get("grad_clip", None)
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.model.parameters(), grad_clip)
            optimizer.step()
            self._apply_model_constraints(model)
            if scheduler is not None:
                scheduler.step(loss.item())
            return float(loss.item())

        def validate(progress, *, tol, in_warmup):
            scores = regression_utils.score(
                *self._evaluate_regression_loader(model, val_loader, targets)
            )
            log(
                f"{fold_label}: {progress} val_traj_r={scores['traj_r']:.4f} "
                f"val_mean_r={scores['mean_r']:.4f}",
                priority=0,
            )
            if in_warmup:
                return None
            improved = self._selection_improved(
                scores["traj_r"], best["score"], tol=tol
            )
            if improved:
                best["score"] = scores["traj_r"]
                best["state"] = {
                    k: v.detach().cpu().clone()
                    for k, v in model.model.state_dict().items()
                }
            return improved

        if training_mode == "steps_based":
            validation_interval = int(self.cfg.model.get("validation_interval", 100))
            warmup_steps = int(self.cfg.model.get("selection_warmup_steps", 0))
            step = 0
            train_iter = iter(train_loader)
            while step < total_steps:
                try:
                    raw_batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    raw_batch = next(train_iter)
                loss = train_step(raw_batch)
                step += 1
                if step % validation_interval == 0 or step == total_steps:
                    validate(
                        f"train_step={step}/{total_steps} train_mse={loss:.4f}",
                        tol=0.0,
                        in_warmup=step < warmup_steps,
                    )
        else:
            patience = int(self.cfg.model.get("patience", 10))
            tol = float(self.cfg.model.get("tol", 1e-4))
            warmup_epochs = int(self.cfg.model.get("selection_warmup_epochs", 0))
            bad_epochs = 0
            for epoch in range(max_iter):
                losses = [train_step(raw_batch) for raw_batch in train_loader]
                improved = validate(
                    f"train_epoch={epoch + 1}/{max_iter} "
                    f"train_mse={float(np.mean(losses)):.4f}",
                    tol=tol,
                    in_warmup=epoch < warmup_epochs,
                )
                if improved is None:
                    continue
                bad_epochs = 0 if improved else bad_epochs + 1
                if bad_epochs >= patience:
                    log(
                        f"{fold_label}: early stop at epoch {epoch + 1} "
                        f"(best val_traj_r={best['score']:.4f})",
                        priority=0,
                    )
                    break
        if best["state"] is not None:
            model.model.load_state_dict(best["state"])
        return model

    def _probe_head_linear(self, model):
        """The final linear layer that the ridge head solves for."""
        core = getattr(model.model, "ft_core_model", None)
        if core is not None:
            # DIVER flatten_linear: the head Linear is wrapped once.
            linear = getattr(core, "module", core)
        else:
            linear = None
            for module in model.model.modules():
                if isinstance(module, nn.Linear):
                    linear = module
        if not isinstance(linear, nn.Linear):
            raise ValueError(
                "model.regression_head='ridge' needs a final nn.Linear, got "
                f"{type(linear).__name__}."
            )
        return linear

    def _collect_probe_features(self, model, loader, linear):
        """Run a split through the model and capture the input of ``linear``.

        Returns (features (n, D), rows (n,), recording id per row).
        """
        feats, rows, rids = [], [], []
        captured = {}

        def hook(_module, inputs, _output):
            x = inputs[0].detach()
            captured["x"] = x.reshape(x.shape[0], -1)

        handle = linear.register_forward_hook(hook)
        model.model.eval()
        try:
            with torch.no_grad():
                for raw_batch in loader:
                    batch_x, coords, seq_id, kwargs, batch_rids, batch_rows = (
                        self._collect_regression_batch(model, raw_batch)
                    )
                    self._forward_prepared(model, batch_x, coords, seq_id, kwargs)
                    feats.append(captured["x"].float().cpu())
                    rows.append(np.asarray(batch_rows).reshape(-1))
                    rids.extend(batch_rids)
        finally:
            handle.remove()
        return torch.cat(feats), np.concatenate(rows), rids

    def _fit_head_ridge_regression(self, model, *, train_loader, targets):
        """Solve the final linear layer in closed form with ridge regression.

        The rest of the model stays as loaded. Features X (the input of the
        final layer) and targets Y (the train trajectories) are centred, and the
        weights come from the dual form W = X^T (X X^T + lambda I)^-1 Y, which
        is an N x N solve whatever the feature size. The bias absorbs the means.
        """
        linear = self._probe_head_linear(model)
        X, rows, rids = self._collect_probe_features(model, train_loader, linear)
        X = X.double()
        Y_np, _ = targets.lookup(rids, rows)
        Y = torch.as_tensor(Y_np, dtype=torch.float64)
        X_mean, Y_mean = X.mean(0, keepdim=True), Y.mean(0, keepdim=True)
        Xc, Yc = X - X_mean, Y - Y_mean
        n = Xc.shape[0]
        gram = Xc @ Xc.T
        lam, mode, trace_over_n = self._resolve_head_lambda(gram, Yc, n)
        system = gram.clone()
        system.diagonal().add_(lam)
        Z = torch.linalg.solve(system, Yc)
        W = (Xc.T @ Z).T  # (traj_len, D)
        b = Y_mean.reshape(-1) - W @ X_mean.reshape(-1)
        with torch.no_grad():
            linear.weight.copy_(W.to(linear.weight.dtype).to(linear.weight.device))
            linear.bias.copy_(b.to(linear.bias.dtype).to(linear.bias.device))
        # The closed form predicts in the target's own units.
        model._regression_target_stats = None
        log(
            f"Ridge regression head: N={n} D={Xc.shape[1]} L={Y.shape[1]} "
            f"lambda_mode={mode} lambda={lam:.6g} trace/N={trace_over_n:.6g}",
            priority=0,
        )
        return model

    def _resolve_head_lambda(self, gram, Yc, n):
        """The ridge strength: (lambda, mode, trace(gram) / N).

        fixed: model.regression_head_lambda.
        trace: model.regression_head_lambda_alpha * trace(gram) / N, so the
            shrinkage is the same relative to the feature scale in every cell.
        gcv: the value in model.regression_head_lambda_grid with the lowest
            generalized cross-validation error.
        """
        model_cfg = self.cfg.model
        lam = float(model_cfg.get("regression_head_lambda", 1.0))
        mode = str(model_cfg.get("regression_head_lambda_mode", "fixed"))
        trace_over_n = float(gram.diagonal().sum()) / n
        if mode == "trace":
            lam = (
                float(model_cfg.get("regression_head_lambda_alpha", 1.0)) * trace_over_n
            )
        elif mode == "gcv":
            grid = [
                float(v)
                for v in model_cfg.get(
                    "regression_head_lambda_grid", [3e3, 1e4, 3e4, 1e5, 3e5]
                )
            ]
            eigvals, eigvecs = torch.linalg.eigh(gram)
            Yt = eigvecs.T @ Yc
            best = None
            for candidate in grid:
                hat = eigvals / (eigvals + candidate)
                residual = (((1.0 - hat)[:, None] * Yt) ** 2).sum()
                gcv = float(residual / max((n - float(hat.sum())) ** 2, 1e-12))
                if best is None or gcv < best[0]:
                    best = (gcv, candidate)
            lam = best[1]
        return lam, mode, trace_over_n

    def _prepare_model_batch(
        self,
        model,
        raw_batch: dict,
    ) -> dict:
        """Apply model-specific prepare_batch hook to one collated batch dict."""
        batch = raw_batch
        prepare = getattr(model, "prepare_batch", None)
        if callable(prepare):
            batch = prepare(
                batch,
                runner_cfg=self.cfg.get("runner", {}),
                device=self.device,
            )
        return batch

    def _extract_batch_tensors(self, batch: dict):
        """Extract model tensors from one prepared batch dict."""
        if not isinstance(batch, dict):
            raise TypeError(
                f"prepare_batch must return dict, got {type(batch).__name__}."
            )
        if "x" not in batch or "y" not in batch:
            raise KeyError("prepared batch must contain keys 'x' and 'y'.")

        x = batch["x"]
        y = batch["y"]
        if not torch.is_tensor(x):
            x = torch.as_tensor(x, dtype=torch.float32)
        else:
            x = x.float()
        if not torch.is_tensor(y):
            y = torch.as_tensor(y, dtype=torch.long)
        else:
            y = y.long()

        coords = batch.get("channel_coords")
        seq_id = batch.get("seq_id")
        model_kwargs = batch.get("model_kwargs", None)
        if model_kwargs is not None and not isinstance(model_kwargs, dict):
            raise TypeError("batch['model_kwargs'] must be a dict when provided.")
        return x, y, coords, seq_id, model_kwargs

    def _evaluate_loader(
        self,
        model,
        loader: DataLoader,
        *,
        n_classes: int,
        classes: np.ndarray,
        class_to_index: dict[int, int],
        criterion=None,
    ):
        """Evaluate a model on a split DataLoader."""
        model.model.eval()
        all_probs = []
        all_targets = []
        running_loss = 0.0
        n_samples = 0

        with torch.no_grad():
            for raw_batch in loader:
                batch = self._prepare_model_batch(model, raw_batch)
                batch_x, batch_y, batch_coords, batch_seq_id, model_kwargs = (
                    self._extract_batch_tensors(batch)
                )
                batch_x = batch_x.to(self.device)
                batch_y_raw_device = batch_y.to(self.device)
                batch_y_encoded = self._encode_targets_for_loss(
                    batch_y_raw_device,
                    class_to_index=class_to_index,
                    context="evaluation batch",
                )
                outputs = self._forward_model(
                    model.model,
                    batch_x,
                    batch_coords,
                    batch_seq_id,
                    accepts_coords=getattr(model, "accepts_coords", False),
                    model_kwargs=model_kwargs,
                )
                probs = torch.nn.functional.softmax(outputs, dim=1)
                all_probs.append(probs.cpu().numpy())
                all_targets.append(batch_y.cpu().numpy())

                if criterion is not None:
                    running_loss += float(
                        self._compute_model_loss(
                            model,
                            outputs,
                            batch_y=batch_y_raw_device,
                            batch_y_encoded=batch_y_encoded,
                            criterion=criterion,
                            model_kwargs=model_kwargs,
                        ).item()
                    ) * batch_y.size(0)
                    n_samples += batch_y.size(0)

        if not all_probs:
            raise ValueError("Cannot evaluate on an empty loader.")

        y_proba = np.concatenate(all_probs, axis=0)
        y_true = np.concatenate(all_targets, axis=0)
        accuracy, roc_auc = self._compute_metrics(y_true, y_proba, classes)
        f1 = self._compute_f1_metric(y_true, y_proba, classes)
        avg_loss = (
            (running_loss / n_samples)
            if criterion is not None and n_samples > 0
            else None
        )
        return accuracy, roc_auc, f1, avg_loss

    def _compute_metrics_from_train_batches(
        self,
        *,
        prob_chunks: list[np.ndarray],
        target_chunks: list[np.ndarray],
        classes: np.ndarray,
    ) -> tuple[float, float, float]:
        """Compute metrics from cached train-loop predictions/targets."""
        if not prob_chunks or not target_chunks:
            raise ValueError("Cannot compute train metrics from empty batch caches.")
        y_proba = np.concatenate(prob_chunks, axis=0)
        y_true = np.concatenate(target_chunks, axis=0)
        accuracy, roc_auc = self._compute_metrics(y_true, y_proba, classes)
        f1 = self._compute_f1_metric(y_true, y_proba, classes)
        return accuracy, roc_auc, f1

    @staticmethod
    def _compute_f1_metric(y_true, y_proba, classes) -> float:
        """Compute F1 from raw labels and probability outputs."""
        y_true = np.asarray(y_true)
        y_proba = np.asarray(y_proba)
        classes = np.asarray(classes)

        predictions = classes[np.argmax(y_proba, axis=1)]
        valid_mask = np.isin(y_true, classes)
        y_filtered = y_true[valid_mask]
        pred_filtered = predictions[valid_mask]
        if y_filtered.size == 0:
            return float("nan")
        if np.unique(y_filtered).size < 2:
            return float("nan")
        if len(classes) == 2:
            return float(
                f1_score(
                    y_filtered,
                    pred_filtered,
                    pos_label=classes[1],
                    average="binary",
                    zero_division=0,
                )
            )
        return float(
            f1_score(
                y_filtered,
                pred_filtered,
                labels=classes,
                average="macro",
                zero_division=0,
            )
        )

    @staticmethod
    def _resolve_selection_score(
        *,
        metric_name: str,
        accuracy: float,
        roc_auc: float,
        f1: float,
    ) -> float:
        metric_key = str(metric_name).lower()
        if metric_key == "accuracy":
            return float(accuracy)
        if metric_key == "roc_auc":
            return float(roc_auc)
        if metric_key == "f1":
            return float(f1)
        raise ValueError(
            "model.selection_metric must be one of ['accuracy', 'f1', 'roc_auc'], "
            f"got '{metric_name}'."
        )

    @staticmethod
    def _selection_improved(
        score: float, best_score: float, *, tol: float = 0.0
    ) -> bool:
        if not np.isfinite(score):
            return False
        if not np.isfinite(best_score):
            return True
        return score > best_score + float(tol)

    @staticmethod
    def _compute_model_loss(
        model,
        outputs,
        *,
        batch_y,
        batch_y_encoded,
        criterion,
        model_kwargs=None,
    ):
        compute_loss = getattr(model, "compute_loss", None)
        if callable(compute_loss):
            return compute_loss(
                outputs,
                batch_y,
                target_encoded=batch_y_encoded,
                criterion=criterion,
                model_kwargs=model_kwargs,
            )
        return criterion(outputs, batch_y_encoded)

    def _train_batch(
        self, model, raw_batch, *, optimizer, scheduler, criterion, class_to_index
    ):
        """Update one batch; the caller owns validation and stopping cadence."""
        batch = self._prepare_model_batch(model, raw_batch)
        batch_x, batch_y, batch_coords, batch_seq_id, model_kwargs = (
            self._extract_batch_tensors(batch)
        )
        batch_x = batch_x.to(self.device)
        batch_y = batch_y.to(self.device)
        batch_y_encoded = self._encode_targets_for_loss(
            batch_y,
            class_to_index=class_to_index,
            context="train batch",
        )

        optimizer.zero_grad()
        outputs = self._forward_model(
            model.model,
            batch_x,
            batch_coords,
            batch_seq_id,
            accepts_coords=getattr(model, "accepts_coords", False),
            model_kwargs=model_kwargs,
        )
        loss = self._compute_model_loss(
            model,
            outputs,
            batch_y=batch_y,
            batch_y_encoded=batch_y_encoded,
            criterion=criterion,
            model_kwargs=model_kwargs,
        )
        loss.backward()

        grad_clip = self.cfg.model.get("grad_clip", None)
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.model.parameters(), grad_clip)

        optimizer.step()
        self._apply_model_constraints(model)
        if scheduler is not None:
            scheduler.step(loss.item())
        # Use the predictions from this update for online training metrics.
        probabilities = (
            torch.nn.functional.softmax(outputs.detach(), dim=1).cpu().numpy()
        )
        targets = batch_y.detach().cpu().numpy()
        return float(loss.item()), probabilities, targets

    def _train_with_early_stopping_loader(
        self,
        model,
        *,
        train_loader: DataLoader,
        val_loader: DataLoader,
        n_classes: int,
        classes: np.ndarray,
        class_to_index: dict[int, int],
        fold_idx=None,
    ):
        """Loader-based epoch training with early stopping."""
        criterion = nn.CrossEntropyLoss()
        max_iter = self.cfg.model.get("max_iter", 100)
        estimated_total_steps = int(max_iter) * len(train_loader)
        optimizer, scheduler, _ = self._create_optimizer_and_scheduler(
            model,
            total_steps_override=estimated_total_steps,
        )

        patience = self.cfg.model.get("patience", 10)
        tol = self.cfg.model.get("tol", 1e-4)

        selection_metric = self.cfg.model.get("selection_metric", "roc_auc")
        selection_warmup_epochs = int(self.cfg.model.get("selection_warmup_epochs", 0))
        best_selection_score = float("-inf")
        best_model_state = None
        patience_counter = 0
        wandb_prefix = self._get_wandb_prefix(fold_idx)

        for epoch in range(max_iter):
            model.model.train()
            train_loss = 0.0
            train_total = 0
            train_prob_chunks: list[np.ndarray] = []
            train_target_chunks: list[np.ndarray] = []
            for raw_batch in train_loader:
                loss, probabilities, targets = self._train_batch(
                    model,
                    raw_batch,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    criterion=criterion,
                    class_to_index=class_to_index,
                )
                train_loss += loss * len(targets)
                train_total += len(targets)
                train_prob_chunks.append(probabilities)
                train_target_chunks.append(targets)

            avg_train_loss = train_loss / train_total if train_total > 0 else 0.0
            val_accuracy, val_auroc, val_f1, val_loss = self._evaluate_loader(
                model,
                val_loader,
                n_classes=n_classes,
                classes=classes,
                class_to_index=class_to_index,
                criterion=criterion,
            )
            train_accuracy, train_auroc, train_f1 = (
                self._compute_metrics_from_train_batches(
                    prob_chunks=train_prob_chunks,
                    target_chunks=train_target_chunks,
                    classes=classes,
                )
            )
            selection_score = self._resolve_selection_score(
                metric_name=selection_metric,
                accuracy=val_accuracy,
                roc_auc=val_auroc,
                f1=val_f1,
            )

            self._log_metrics_to_wandb(
                wandb_prefix,
                val_auroc,
                val_accuracy,
                train_loss=avg_train_loss,
                val_loss=val_loss,
                train_auroc=train_auroc,
                train_accuracy=train_accuracy,
                train_f1=train_f1,
                val_f1=val_f1,
                selection_metric=selection_metric,
                selection_score=selection_score,
                epoch=epoch,
            )
            val_loss_text = f"{val_loss:.4f}" if val_loss is not None else "n/a"
            fold_label = f"Fold {fold_idx}" if fold_idx is not None else "Fold"
            log(
                f"{fold_label}: train_epoch={epoch + 1}/{max_iter} "
                f"train_loss={avg_train_loss:.4f} "
                f"train_acc={train_accuracy:.3f} train_roc_auc={train_auroc:.3f} "
                f"train_f1={train_f1:.3f} "
                f"val_loss={val_loss_text} "
                f"val_acc={val_accuracy:.3f} val_roc_auc={val_auroc:.3f} "
                f"val_f1={val_f1:.3f} "
                f"selection_metric={selection_metric} selection_score={selection_score:.3f}",
                priority=0,
            )

            if epoch < selection_warmup_epochs:
                continue
            if self._selection_improved(selection_score, best_selection_score, tol=tol):
                best_selection_score = selection_score
                best_model_state = {
                    k: v.cpu().clone() for k, v in model.model.state_dict().items()
                }
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        if best_model_state is not None:
            model.model.load_state_dict(best_model_state)

    def _train_steps_based_loader(
        self,
        model,
        *,
        train_loader: DataLoader,
        val_loader: DataLoader,
        n_classes: int,
        classes: np.ndarray,
        class_to_index: dict[int, int],
        fold_idx=None,
    ):
        """Loader-based fixed-step training loop."""
        criterion = nn.CrossEntropyLoss()
        total_steps = self.cfg.model.get("total_steps", 2000)
        optimizer, scheduler, learning_rate = self._create_optimizer_and_scheduler(
            model,
            total_steps_override=total_steps,
        )
        validation_interval = self.cfg.model.get("validation_interval", 100)

        selection_metric = self.cfg.model.get("selection_metric", "roc_auc")
        selection_warmup_steps = int(self.cfg.model.get("selection_warmup_steps", 0))
        best_selection_score = float("-inf")
        best_model_state = None
        wandb_prefix = self._get_wandb_prefix(fold_idx)
        train_prob_chunks: list[np.ndarray] = []
        train_target_chunks: list[np.ndarray] = []

        step = 0
        train_iter = iter(train_loader)
        while step < total_steps:
            model.model.train()
            try:
                raw_batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                raw_batch = next(train_iter)

            loss, probabilities, targets = self._train_batch(
                model,
                raw_batch,
                optimizer=optimizer,
                scheduler=scheduler,
                criterion=criterion,
                class_to_index=class_to_index,
            )
            train_prob_chunks.append(probabilities)
            train_target_chunks.append(targets)

            step += 1
            if step % validation_interval != 0 and step != total_steps:
                continue

            val_accuracy, val_auroc, val_f1, val_loss = self._evaluate_loader(
                model,
                val_loader,
                n_classes=n_classes,
                classes=classes,
                class_to_index=class_to_index,
                criterion=criterion,
            )
            train_accuracy, train_auroc, train_f1 = (
                self._compute_metrics_from_train_batches(
                    prob_chunks=train_prob_chunks,
                    target_chunks=train_target_chunks,
                    classes=classes,
                )
            )
            train_prob_chunks = []
            train_target_chunks = []
            selection_score = self._resolve_selection_score(
                metric_name=selection_metric,
                accuracy=val_accuracy,
                roc_auc=val_auroc,
                f1=val_f1,
            )

            current_lr = scheduler.get_lr() if scheduler is not None else learning_rate
            self._log_metrics_to_wandb(
                wandb_prefix,
                val_auroc,
                val_accuracy,
                train_loss=loss,
                val_loss=val_loss,
                train_auroc=train_auroc,
                train_accuracy=train_accuracy,
                train_f1=train_f1,
                val_f1=val_f1,
                selection_metric=selection_metric,
                selection_score=selection_score,
                step=step,
                learning_rate=current_lr,
            )
            val_loss_text = f"{val_loss:.4f}" if val_loss is not None else "n/a"
            fold_label = f"Fold {fold_idx}" if fold_idx is not None else "Fold"
            log(
                f"{fold_label}: train_step={step}/{total_steps} "
                f"train_loss={loss:.4f} "
                f"train_acc={train_accuracy:.3f} train_roc_auc={train_auroc:.3f} "
                f"train_f1={train_f1:.3f} "
                f"val_loss={val_loss_text} "
                f"val_acc={val_accuracy:.3f} val_roc_auc={val_auroc:.3f} "
                f"val_f1={val_f1:.3f} "
                f"selection_metric={selection_metric} selection_score={selection_score:.3f}",
                priority=0,
            )

            if step < selection_warmup_steps:
                continue
            if self._selection_improved(selection_score, best_selection_score):
                best_selection_score = selection_score
                best_model_state = {
                    k: v.cpu().clone() for k, v in model.model.state_dict().items()
                }

        if best_model_state is not None:
            model.model.load_state_dict(best_model_state)

    def _create_optimizer_and_scheduler(self, model, total_steps_override=None):
        """
        Create optimizer and scheduler from config.

        Returns:
            tuple: (optimizer, scheduler, learning_rate)
        """
        learning_rate = self.cfg.model.get("learning_rate", 0.001)
        optimizer_name = self.cfg.model.get("optimizer", "Adam")
        weight_decay = self.cfg.model.get("weight_decay", 0.0)
        optimizer_cls = getattr(torch.optim, optimizer_name)

        param_groups = None
        if hasattr(model, "get_parameter_groups"):
            param_groups = model.get_parameter_groups()

        if param_groups:
            optimizer = optimizer_cls(param_groups, weight_decay=weight_decay)
            log(
                f"  Using separate learning rates: {[pg.get('lr', learning_rate) for pg in param_groups]}",
                priority=0,
                indent=2,
            )
        else:
            optimizer = optimizer_cls(
                model.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
            )

        # Initialize scheduler if configured
        scheduler = None
        if "scheduler" in self.cfg.model:
            from imindbench.schedulers import build_scheduler

            scheduler_cfg = OmegaConf.create(
                OmegaConf.to_container(deepcopy(self.cfg.model.scheduler), resolve=True)
            )
            if total_steps_override is not None:
                scheduler_cfg["total_steps"] = int(total_steps_override)
            scheduler = build_scheduler(scheduler_cfg, optimizer)
            if scheduler is not None:
                log(
                    f"  Scheduler: {scheduler_cfg.get('name', 'unknown')}",
                    priority=0,
                    indent=2,
                )

        return optimizer, scheduler, learning_rate

    @staticmethod
    def _apply_model_constraints(model):
        """Apply optional model-specific post-update constraints."""
        apply_constraints = getattr(model, "apply_constraints", None)
        if callable(apply_constraints):
            apply_constraints()

    def _get_wandb_prefix(self, fold_idx):
        """
        Get wandb logging prefix for a fold.

        Returns:
            str: Prefix string (e.g., "fold_0/") or empty string
        """
        if fold_idx is not None and self.wandb_run is not None:
            return f"fold_{fold_idx}/"
        return ""

    def _log_metrics_to_wandb(
        self,
        prefix,
        val_auroc,
        val_accuracy,
        train_loss=None,
        val_loss=None,
        train_auroc=None,
        train_accuracy=None,
        train_f1=None,
        val_f1=None,
        selection_metric=None,
        selection_score=None,
        epoch=None,
        step=None,
        learning_rate=None,
    ):
        """
        Log metrics to wandb.

        Args:
            prefix: Wandb prefix (from _get_wandb_prefix)
            val_auroc: Validation AUROC
            val_accuracy: Validation accuracy
            train_loss: Training loss (optional)
            val_loss: Validation loss (optional)
            train_auroc: Training ROC-AUC (optional)
            train_accuracy: Training accuracy (optional)
            train_f1: Training F1 score (optional)
            val_f1: Validation F1 score (optional)
            selection_metric: Metric used for model selection (optional)
            selection_score: Current value of the selection metric (optional)
            epoch: Epoch number (for epoch-based training)
            step: Step number (for steps-based training)
            learning_rate: Current learning rate (optional)
        """
        if self.wandb_run is None:
            return

        log_dict = {
            f"{prefix}val_roc_auc": val_auroc,
            f"{prefix}val_accuracy": val_accuracy,
        }

        if train_loss is not None:
            log_dict[f"{prefix}train_loss"] = train_loss

        if val_loss is not None:
            log_dict[f"{prefix}val_loss"] = val_loss

        if train_auroc is not None:
            log_dict[f"{prefix}train_roc_auc"] = train_auroc

        if train_accuracy is not None:
            log_dict[f"{prefix}train_accuracy"] = train_accuracy

        if train_f1 is not None:
            log_dict[f"{prefix}train_f1"] = train_f1

        if val_f1 is not None:
            log_dict[f"{prefix}val_f1"] = val_f1

        if selection_metric is not None:
            log_dict[f"{prefix}selection_metric"] = selection_metric

        if selection_score is not None:
            log_dict[f"{prefix}selection_score"] = selection_score

        if epoch is not None:
            log_dict[f"{prefix}epoch"] = epoch

        if step is not None:
            log_dict[f"{prefix}step"] = step

        if learning_rate is not None:
            log_dict[f"{prefix}learning_rate"] = learning_rate

        self.wandb_run.log(log_dict)

    def _forward_model(
        self,
        torch_model,
        inputs,
        coords=None,
        seq_id=None,
        accepts_coords=False,
        model_kwargs=None,
    ):
        """
        Forward helper that optionally supplies coordinates and seq_id to the model.

        Args:
            torch_model: The PyTorch model
            inputs: Input tensor (batch_size, n_electrodes+1, hidden_dim)
            coords: Optional coordinate tensor - can be:
                - (n_electrodes, 3) shared across all samples (broadcasted)
                - (batch_size, n_electrodes, 3) per-sample coordinates
            seq_id: Optional seq_id tensor - can be:
                - (n_electrodes,) shared across all samples (broadcasted)
                - (batch_size, n_electrodes) per-sample seq_id
            model_kwargs: Optional kwargs to forward to torch_model (e.g. pad_mask)
        """
        model_kwargs = dict(model_kwargs or {})

        # Cache forward-signature feature detection on the module so the batch
        # loop does not need to re-run inspect.signature(...) every iteration.
        accepts_positions = getattr(torch_model, "_neuroprobe_accepts_positions", None)
        accepts_pad_mask = getattr(torch_model, "_neuroprobe_accepts_pad_mask", None)
        if accepts_positions is None or accepts_pad_mask is None:
            sig = inspect.signature(torch_model.forward)
            accepts_positions = "positions" in sig.parameters
            accepts_pad_mask = "pad_mask" in sig.parameters
            torch_model._neuroprobe_accepts_positions = accepts_positions
            torch_model._neuroprobe_accepts_pad_mask = accepts_pad_mask
        if "pad_mask" in model_kwargs and not accepts_pad_mask:
            model_kwargs.pop("pad_mask")

        if accepts_positions and coords is not None and seq_id is not None:
            batch_size_actual = inputs.shape[0]

            coords_t = torch.as_tensor(coords)
            if coords_t.ndim == 2:
                # Shared coords: (n_electrodes, 3) -> broadcast to batch.
                batch_coords = coords_t.unsqueeze(0).expand(batch_size_actual, -1, -1)
            elif coords_t.ndim == 3:
                if coords_t.shape[0] != batch_size_actual:
                    raise ValueError(
                        "coords batch dimension must match inputs batch size, got "
                        f"{coords_t.shape[0]} vs {batch_size_actual}."
                    )
                batch_coords = coords_t
            else:
                raise ValueError(f"Unexpected coords shape: {tuple(coords_t.shape)}")

            seq_id_t = torch.as_tensor(seq_id)
            if seq_id_t.ndim == 1:
                # Shared seq_id: (n_electrodes,) -> broadcast to batch.
                batch_seq_id = seq_id_t.unsqueeze(0).expand(batch_size_actual, -1)
            elif seq_id_t.ndim == 2:
                if seq_id_t.shape[0] != batch_size_actual:
                    raise ValueError(
                        "seq_id batch dimension must match inputs batch size, got "
                        f"{seq_id_t.shape[0]} vs {batch_size_actual}."
                    )
                batch_seq_id = seq_id_t
            else:
                raise ValueError(f"Unexpected seq_id shape: {tuple(seq_id_t.shape)}")

            # Convert to tensors on device
            # Coords are used as indices in MultiSubjBrainPositionalEncoding, so use int64
            # PopT coordinate slots [L, I, P] are cast to integers before embedding.
            batch_coords = batch_coords.to(device=self.device, dtype=torch.int64)
            batch_seq_id = batch_seq_id.to(device=self.device, dtype=torch.int64)

            positions = (batch_coords, batch_seq_id)
            return torch_model(inputs, positions=positions, **model_kwargs)
        elif coords is not None and accepts_coords:
            # Explicitly-declared coords-only fallback for legacy models.
            coords_t = (
                coords.to(device=self.device)
                if torch.is_tensor(coords)
                else torch.as_tensor(coords, device=self.device)
            )
            return torch_model(inputs, coords_t, **model_kwargs)
        else:
            return torch_model(inputs, **model_kwargs)


def _log_model_summary(model, input_shape, n_classes):
    """
    Log model architecture summary including parameter count.
    Uses PyTorch's model representation and parameter counting.

    Args:
        model: Model instance with a .model attribute (the PyTorch module)
        input_shape: Input shape tuple
        n_classes: Number of output classes
    """
    if model.model is None:
        return

    log("=" * 80, priority=0, indent=1)
    log("Model Architecture Summary:", priority=0, indent=1)
    log("=" * 80, priority=0, indent=1)

    # Input/output info
    log(f"Input shape: {input_shape}", priority=0, indent=2)
    log(f"Output classes: {n_classes}", priority=0, indent=2)

    # Count parameters
    total_params = sum(p.numel() for p in model.model.parameters())
    trainable_params = sum(
        p.numel() for p in model.model.parameters() if p.requires_grad
    )
    non_trainable_params = total_params - trainable_params

    log(f"Total parameters: {total_params:,}", priority=0, indent=2)
    log(f"Trainable parameters: {trainable_params:,}", priority=0, indent=2)
    if non_trainable_params > 0:
        log(f"Non-trainable parameters: {non_trainable_params:,}", priority=0, indent=2)

    # Model-specific details
    if hasattr(model, "hidden_dims"):
        hidden_dims = model.hidden_dims
        if isinstance(hidden_dims, (list, tuple)) and len(hidden_dims) > 0:
            log(f"Hidden layers: {hidden_dims}", priority=0, indent=2)
        else:
            log("Architecture: Linear (no hidden layers)", priority=0, indent=2)

    # Basic model structure (safe, no hooks that can interfere with training)
    log("Model structure:", priority=0, indent=2)
    model_str = str(model.model)
    for line in model_str.split("\n")[:50]:
        if line.strip():
            log(line, priority=0, indent=3)
    if len(model_str.split("\n")) > 50:
        log("... (output truncated)", priority=0, indent=3)

    log("=" * 80, priority=0, indent=1)
