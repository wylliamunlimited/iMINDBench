"""Options that change what a Miller cell scores.

Class pairs, fold subsets and matched train subsets.
"""

import numpy as np
import pytest
from miller_fakes import install_fake_miller
from omegaconf import OmegaConf

from imindbench.preprocessors import build_preprocessor
from imindbench.utils import data_adapter, pipeline_contracts
from imindbench.utils.logging_utils import (
    build_internal_eval_result,
    build_public_export_result,
    resolve_task_mode_config,
)
from imindbench.utils.pipeline_contracts import validate_eval_config


@pytest.fixture
def fake_miller(monkeypatch):
    return install_fake_miller(monkeypatch)


def _cfg(**dataset_overrides):
    dataset = {
        "root": "/tmp",
        "dirname": "miller_ecog_library_2019",
        "provider": "millerecog2019",
        "subset_tier": "full",
        "label_mode": "binary",
        "task": "move_vs_rest",
        "test_subject": 3,
        "test_session": 1,
        "regime": "within-session",
        "coordinate_profile": "popt_zero",
        "uniquify_channel_ids_with_subject": True,
        "uniquify_channel_ids_with_session": True,
        "merge_val_into_test": False,
    }
    dataset.update(dataset_overrides)
    return OmegaConf.create(
        {
            "dataset": dataset,
            "model": {
                "name": "logistic",
                "backend": "sklearn",
                "requires_aligned_channels": False,
                "requires_coords": False,
            },
            "preprocessor": {"chain": [{"name": "raw"}]},
            "runtime": {"seed": 0, "overwrite": True},
            "submitter": {
                "author": "tester",
                "organization": "org",
                "organization_url": "https://example.com",
            },
        }
    )


def _build_fold(tmp_path, **dataset_overrides):
    dataset_cfg = _cfg(root=str(tmp_path), **dataset_overrides).dataset
    preprocessor_cfg = OmegaConf.create({"chain": [{"name": "raw"}]})
    return data_adapter.build_neuroprobe_torch_fold(
        dataset_cfg,
        preprocessor=build_preprocessor(preprocessor_cfg),
        preprocessor_cfg=preprocessor_cfg,
        fold_idx=0,
        seed=0,
        require_coords=False,
        needs_pool=False,
    )


# ---- class_pair -----------------------------------------------------------


def test_class_pair_is_accepted_for_multiclass_miller():
    validate_eval_config(_cfg(label_mode="multiclass", class_pair=[0, 2]))


@pytest.mark.parametrize(
    ("overrides", "error", "message"),
    [
        ({"label_mode": "binary", "class_pair": [0, 2]}, ValueError, "multiclass"),
        ({"label_mode": "multiclass", "class_pair": [1, 1]}, ValueError, "different"),
        ({"label_mode": "multiclass", "class_pair": [0, 1, 2]}, ValueError, "two"),
        ({"label_mode": "multiclass", "class_pair": 1}, ValueError, "two"),
        ({"label_mode": "multiclass", "class_pair": [0, True]}, TypeError, "ints"),
        ({"label_mode": "multiclass", "class_pair": [-1, 0]}, ValueError, ">= 0"),
    ],
)
def test_bad_class_pairs_are_rejected(overrides, error, message):
    with pytest.raises(error, match=message):
        validate_eval_config(_cfg(**overrides))


def test_class_pair_is_rejected_for_datasets_without_support():
    cfg = _cfg(label_mode="multiclass", class_pair=[0, 1])
    cfg.dataset.provider = "kelesbyd2024"
    cfg.dataset.dirname = "keles_byd_2024"
    with pytest.raises(ValueError, match="unsupported for dataset.provider"):
        validate_eval_config(cfg)


def test_class_pair_reaches_the_dataset_class_only_when_set(monkeypatch):
    received = []

    class Recorder:
        def __init__(self, **kwargs):
            received.append(kwargs)

    monkeypatch.setitem(
        pipeline_contracts._PROVIDER_SPECS["millerecog2019"],
        "dataset_class_loader",
        lambda: Recorder,
    )
    for class_pair in (None, [0, 2]):
        pipeline_contracts.build_processed_split_provider(
            dataset_provider="millerecog2019",
            dataset_cfg=_cfg(label_mode="multiclass", class_pair=class_pair).dataset,
            split="train",
            fold_idx=0,
            regime="within-session",
        )
    assert "class_pair" not in received[0]
    assert received[1]["class_pair"] == (0, 2)


def test_class_pair_fold_keeps_two_classes_as_binary(tmp_path, fake_miller):
    full = _build_fold(tmp_path, label_mode="multiclass")
    pair = _build_fold(tmp_path, label_mode="multiclass", class_pair=[0, 2])
    for split in ("train", "val", "test"):
        full_labels = np.array([int(s["y"]) for s in full[f"{split}_split"]])
        pair_labels = np.array([int(s["y"]) for s in pair[f"{split}_split"]])
        assert set(full_labels) == {0, 1, 2}
        assert set(pair_labels) == {0, 1}
        # Stored label 0 becomes class 0, stored label 2 becomes class 1.
        assert np.sum(pair_labels == 0) == np.sum(full_labels == 0)
        assert np.sum(pair_labels == 1) == np.sum(full_labels == 2)


def test_class_pair_changes_the_fold_cache_id_only_when_set():
    base = _cfg(label_mode="multiclass").dataset
    assert data_adapter._optional_task_mode_identity(base) == {}
    paired = _cfg(label_mode="multiclass", class_pair=[0, 2]).dataset
    assert data_adapter._optional_task_mode_identity(paired) == {"class_pair": [0, 2]}


# ---- result config ----------------------------------------------------------


def _public_config(dataset_cfg):
    internal = build_internal_eval_result(
        provider="millerecog2019",
        task="move_vs_rest",
        regime="within-session",
        subject_id=3,
        trial_id=1,
        model_name="logistic",
        preprocess_type="raw",
        preprocess_parameters={},
        window_slicing_policy="ceil",
        seed=0,
        results_population={},
        subject_load_time=0.0,
        regression_run_time=0.0,
        task_mode_config=resolve_task_mode_config(dataset_cfg),
    )
    return build_public_export_result(
        internal_result=internal,
        author="a",
        organization="o",
        organization_url="u",
    )["config"]


def test_binary_result_config_keeps_its_shape():
    config = _public_config(_cfg().dataset)
    assert set(config) == {
        "preprocess",
        "window_slicing_policy",
        "seed",
        "subject_id",
        "trial_id",
        "eval_name",
        "splits_type",
        "model_name",
    }


def test_result_config_records_label_mode_and_class_pair():
    config = _public_config(_cfg(label_mode="multiclass", class_pair=[0, 2]).dataset)
    assert config["label_mode"] == "multiclass"
    assert config["class_pair"] == [0, 2]


# ---- fold_subset ------------------------------------------------------------


def _iter_folds(tmp_path, fold_subset, **dataset_overrides):
    from imindbench.utils.fold_helpers import iter_variable_channel_folds

    preprocessor_cfg = OmegaConf.create({"chain": [{"name": "raw"}]})
    return list(
        iter_variable_channel_folds(
            n_folds=2,
            dataset_cfg=_cfg(root=str(tmp_path), **dataset_overrides).dataset,
            preprocessor=build_preprocessor(preprocessor_cfg),
            preprocessor_cfg=preprocessor_cfg,
            seed=0,
            require_coords=False,
            needs_pool=False,
            fold_subset=fold_subset,
        )
    )


@pytest.mark.parametrize(("fold_subset", "expected"), [(None, [0, 1]), ([1], [1])])
def test_fold_subset_runs_only_the_listed_folds(
    tmp_path, fake_miller, fold_subset, expected
):
    folds = _iter_folds(tmp_path, fold_subset)
    assert [fold["fold_idx"] for fold in folds] == expected


def test_fold_subset_outside_the_fold_count_is_rejected(tmp_path, fake_miller):
    with pytest.raises(ValueError, match="outside 0..1"):
        _iter_folds(tmp_path, [2])


@pytest.mark.parametrize(
    ("value", "error", "message"),
    [
        ([], ValueError, "non-empty list"),
        (0, ValueError, "non-empty list"),
        ([0, 0], ValueError, "repeat"),
        ([-1], ValueError, ">= 0"),
        ([True], TypeError, "ints"),
    ],
)
def test_bad_fold_subsets_are_rejected(value, error, message):
    with pytest.raises(error, match=message):
        validate_eval_config(_cfg(fold_subset=value))


def test_fold_subset_is_accepted_and_recorded():
    cfg = _cfg(fold_subset=[0])
    validate_eval_config(cfg)
    assert resolve_task_mode_config(cfg.dataset) == {"fold_subset": [0]}


# ---- matched train subsets --------------------------------------------------


def _train_labels(tmp_path, fold_idx=0):
    fold = _build_fold(tmp_path)
    assert fold_idx == 0
    return [int(sample["y"]) for sample in fold["train_split"]]


def _loader_order(n, seed):
    import torch

    generator = torch.Generator()
    generator.manual_seed(seed)
    order = None
    for op in "ppbpbbp":
        if op == "b":
            torch.empty((), dtype=torch.int64).random_(generator=generator)
        else:
            order = torch.randperm(n, generator=generator).tolist()
    return order


def _write_subset_file(tmp_path, labels, *, positions, crc_labels=None, draws=("0",)):
    import json
    import zlib

    crc_labels = labels if crc_labels is None else crc_labels
    table = {
        "move_vs_rest_sub3_sess1_fold0": {
            "n_train": len(labels),
            "y_train_crc32": zlib.crc32(json.dumps(crc_labels).encode()),
            "subsets": {"0.25": {draw: positions for draw in draws}},
        }
    }
    path = tmp_path / "subsets.json"
    path.write_text(json.dumps(table))
    data_adapter._MATCHED_SUBSET_FILES.clear()
    return str(path)


def _matched(path, draw=0):
    return {
        "train_sample_indices_file": path,
        "train_sample_indices_frac": 0.25,
        "train_sample_indices_draw": draw,
        "train_sample_indices_order_seed": 42,
    }


def test_matched_subset_keeps_the_listed_windows(tmp_path, fake_miller):
    labels = _train_labels(tmp_path)
    path = _write_subset_file(tmp_path, labels, positions=[0, 3, 5])
    fold = _build_fold(tmp_path, **_matched(path))
    kept = [int(sample["y"]) for sample in fold["train_split"]]
    assert kept == [labels[0], labels[3], labels[5]]
    # val and test are not subsampled.
    assert len(fold["test_split"]) == len(_build_fold(tmp_path)["test_split"])


def test_matched_subset_in_train_loader_order(tmp_path, fake_miller):
    labels = _train_labels(tmp_path)
    order = _loader_order(len(labels), 42 + 0)
    path = _write_subset_file(
        tmp_path,
        labels,
        positions=[0, 1],
        crc_labels=[labels[j] for j in order],
    )
    fold = _build_fold(tmp_path, **_matched(path))
    kept = [int(sample["y"]) for sample in fold["train_split"]]
    assert kept == [labels[j] for j in sorted(order[:2])]


def test_matched_subset_with_wrong_checksum_stops_the_run(tmp_path, fake_miller):
    labels = _train_labels(tmp_path)
    wrong = [1 - label for label in labels]
    path = _write_subset_file(tmp_path, labels, positions=[0], crc_labels=wrong)
    with pytest.raises(ValueError, match="checksum does not match"):
        _build_fold(tmp_path, **_matched(path))


def test_absent_matched_subset_is_recorded_as_a_skipped_fold(tmp_path, fake_miller):
    from imindbench.utils.fold_helpers import evaluate_variable_fold

    labels = _train_labels(tmp_path)
    path = _write_subset_file(tmp_path, labels, positions=[0], draws=("1",))
    with pytest.raises(data_adapter.MatchedSubsetAbsent):
        _build_fold(tmp_path, **_matched(path, draw=0))
    payloads = _iter_folds(tmp_path, [0], **_matched(path, draw=0))
    assert payloads[0]["fold"] is None
    result = evaluate_variable_fold(0, payloads[0], cfg=None, runner=None, seed=0)
    assert result["status"] == "skipped"
    assert "absent" in result["skip_reason"]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"train_sample_indices_file": "x.json"}, "needs"),
        ({"train_sample_indices_draw": 0}, "need dataset.train_sample_indices_file"),
        ({**_matched("x.json"), "train_sample_fraction": 0.5}, "train_sample_fraction"),
        (
            {**_matched("x.json"), "max_train_samples_per_subject": 10},
            "max_train_samples_per_subject",
        ),
        ({**_matched("x.json"), "train_sample_indices_draw": -1}, "int >= 0"),
    ],
)
def test_bad_matched_subset_options_are_rejected(overrides, message):
    with pytest.raises(ValueError, match=message):
        validate_eval_config(_cfg(**overrides))


def test_matched_subset_changes_the_cache_id_and_result_config():
    cfg = _cfg(**_matched("x.json"))
    validate_eval_config(cfg)
    identity = data_adapter._optional_task_mode_identity(cfg.dataset)
    assert identity == {"train_sample_indices": ["x.json", "0.25", "0", "42"]}
    assert resolve_task_mode_config(cfg.dataset) == {
        "train_sample_indices": {"frac": "0.25", "draw": 0}
    }
