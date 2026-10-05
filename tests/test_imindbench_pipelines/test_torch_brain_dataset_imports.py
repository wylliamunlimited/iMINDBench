import builtins
import importlib
import sys
import types

import pytest

from imindbench.utils.pipeline_contracts import get_dataset_class


def test_pipeline_contract_module_import_is_lazy(monkeypatch):
    original_import = builtins.__import__

    def reject_eager_dataset_import(name, *args, **kwargs):
        if name == "torch_brain.datasets":
            raise AssertionError("torch_brain.datasets imported eagerly")
        return original_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "imindbench.utils.pipeline_contracts")
    monkeypatch.setattr(builtins, "__import__", reject_eager_dataset_import)

    module = importlib.import_module("imindbench.utils.pipeline_contracts")
    assert callable(module.get_dataset_class)


def test_processed_providers_resolve_from_torch_brain_datasets(monkeypatch):
    dataset_classes = {
        "neuroprobev2": type("NeuroprobeV2", (), {}),
        "neuroprobe2025": type("Neuroprobe2025", (), {}),
        "kelesbyd2024": type("KelesBYD2024", (), {}),
        "berezutskayapippi2022": type("BerezutskayaPippi2022", (), {}),
        "millerecog2019": type("MillerECoG2019", (), {}),
    }
    datasets_module = types.ModuleType("torch_brain.datasets")
    for dataset_class in dataset_classes.values():
        setattr(datasets_module, dataset_class.__name__, dataset_class)

    torch_brain_module = types.ModuleType("torch_brain")
    torch_brain_module.__path__ = []
    torch_brain_module.datasets = datasets_module
    monkeypatch.setitem(sys.modules, "torch_brain", torch_brain_module)
    monkeypatch.setitem(sys.modules, "torch_brain.datasets", datasets_module)
    monkeypatch.delitem(sys.modules, "brainsets.datasets", raising=False)

    for provider, expected_class in dataset_classes.items():
        assert get_dataset_class(provider) is expected_class


@pytest.mark.parametrize("provider", ["neuroprobev2", "millerecog2019"])
def test_missing_torch_brain_dataset_api_has_actionable_error(monkeypatch, provider):
    original_import = builtins.__import__

    def block_torch_brain_datasets(name, *args, **kwargs):
        if name == "torch_brain.datasets":
            raise ImportError("missing public dataset API")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", block_torch_brain_datasets)

    with pytest.raises(ImportError, match="torch_brain.datasets") as exc_info:
        get_dataset_class(provider)
    assert isinstance(exc_info.value.__cause__, ImportError)
