# Adapted from DIVER-1: https://github.com/DIVER-Project/DIVER-1
# Source and license details: THIRD_PARTY.md (DIVER-1).

import numpy as np
from omegaconf import DictConfig

from imindbench.utils.pipeline_contracts import provider_property

from . import register_model
from .torch_base_model import TorchBaseModel


@register_model("diver")
class DIVERModel(TorchBaseModel):
    """
    DIVER-1  fine-tuning model for neuroprobe classification.

    Loads pretrained DIVER-1 weights and adds a classification head.
    Supports both frozen encoder (feature extraction) and full fine-tuning modes.
    Can use randomly initialized weights instead of pretrained weights via random_init flag.
    """

    def __init__(self, cfg: DictConfig, dataset_cfg: DictConfig | None = None):
        super().__init__(cfg, dataset_cfg)
        self.cfgs = cfg
        self.dataset_cfg = dataset_cfg
        self.accepts_coords = True
        self.upstream_ckpt = cfg.get("upstream_ckpt", None)
        if self.upstream_ckpt is None:
            raise ValueError("upstream_ckpt must be provided in model config")
        self.ft_head_style = cfg.get("ft_head_style", "flatten_linear")
        self.random_init = cfg.get("random_init", False)
        self.frozen_upstream = cfg.get("frozen_upstream", True)
        self.upstream_lr = cfg.get("upstream_lr", 2e-3)
        self.head_lr = cfg.get("head_lr", 2e-3)

        self.upstream = None
        self.classification_head = None

    def _raw_input_to_patched_input(self, x, params):
        raw_input_shape = x.shape
        if len(raw_input_shape) != 3:
            raise ValueError(
                f"Expected raw input shape (B, C, T), got {raw_input_shape}"
            )
        patch_size = params.patch_size
        B, C, T = raw_input_shape
        if T % patch_size != 0:
            raise ValueError(
                f"Input sequence length T={T} is not divisible by patch_size={patch_size}."
            )
        N = T // patch_size
        patched_input = x.view(B, C, N, patch_size)
        return patched_input

    def _get_data_info_list(self, B, C, coords, provider):
        # The dataset list says which electrode type each dataset records with.
        coord_subtype = provider_property(provider, "electrode_subtype")
        data_info_list = []
        for idx in range(B):
            data_info_dict = {}
            if coords is None:
                data_info_dict["xyz_id"] = np.full((C, 3), np.nan)
            else:
                data_info_dict["xyz_id"] = coords[idx]
            data_info_dict["modality"] = "iEEG"
            data_info_dict["coord_subtype"] = [coord_subtype for _ in range(C)]
            data_info_list.append(data_info_dict)
        return data_info_list

    def prepare_batch(self, batch, **kwargs):
        """Prepare collated batch into PopT-compatible inputs."""
        if not isinstance(batch, dict):
            raise TypeError(f"prepare_batch expected dict, got {type(batch).__name__}.")
        if "x" not in batch or "y" not in batch:
            raise KeyError("Batch must include 'x' and 'y' keys.")
        x = batch["x"]
        x = self._raw_input_to_patched_input(x, self.cfgs)
        provider = (
            None if self.dataset_cfg is None else self.dataset_cfg.get("provider", None)
        )
        # 200.0 is DIVER's scaling factor for iEEG in microvolts. Datasets that
        # store volts are converted to microvolts first.
        if provider_property(provider, "signal_unit") == "V":
            x = x * 1e6 / 200.0
        else:
            x = x / 200.0

        coords = batch.get("channel_coords")

        C = x.shape[1]
        B = x.shape[0]

        data_info_list = self._get_data_info_list(B, C, coords, provider)
        out = dict(batch)
        out["x"] = x
        out["channel_coords"] = None
        out["seq_id"] = None
        out["model_kwargs"] = {"pad_mask": None, "data_info_list": data_info_list}

        return out

    def _create_network(self, input_shape, n_classes):
        """Create the DIVER model architecture."""
        try:
            from imindbench.models.diver_components.finetune_model import (
                flatten_linear_finetune,
                flatten_mlp_finetune,
            )
        except ModuleNotFoundError as exc:
            if (exc.name or "").split(".")[0] not in {"mup", "jaxtyping"}:
                raise
            raise ModuleNotFoundError(
                "DIVER requires its optional dependencies. Install them with "
                "pip install 'imindbench[diver]'; see README.md for resources."
            ) from exc

        if self.ft_head_style == "flatten_linear":
            return flatten_linear_finetune(self.cfgs, input_shape, n_classes)
        elif self.ft_head_style == "flatten_mlp":
            return flatten_mlp_finetune(self.cfgs, input_shape, n_classes)
        raise ValueError(
            f"Unsupported ft_head_style '{self.ft_head_style}'. "
            "Supported styles: 'flatten_linear', 'flatten_mlp'."
        )

    def build_model(self, input_shape, n_classes, device=None):
        """Build the model with given input shape and number of classes."""
        self._resolve_device(device)
        self.model = self._create_network(input_shape, n_classes)
        if self.cfgs.get("frozen_upstream", True):
            for name, param in self.model.named_parameters():
                if "backbone" in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
        self.model = self.model.to(self.device)
        self.classes_ = np.arange(n_classes)
        return self.model

    def get_parameter_groups(self):
        """
        Return separate parameter groups for different learning rates.

        Returns:
            List of parameter group dicts for optimizer
        """
        upstream_params = []
        head_params = []

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if "backbone" in name:
                    upstream_params.append(param)
                else:
                    head_params.append(param)

        param_groups = []
        if upstream_params:
            param_groups.append({"params": upstream_params, "lr": self.upstream_lr})
        if head_params:
            param_groups.append({"params": head_params, "lr": self.head_lr})

        return param_groups

    def forward(self, x, positions=None, data_info_list=None, pad_mask=None):
        _ = positions, pad_mask
        patched_input = x

        return self.model(patched_input, data_info_list=data_info_list)
