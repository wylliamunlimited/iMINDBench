# Changelog

Notable changes and upgrade actions are recorded here.

## [Unreleased]

### Added

- Kai Miller's ECoG library (2019) as a fifth dataset, `millerecog2019`, with
  `scripts/run_millerecog2019.sh`. The default block runs the 178-cell main
  table (153 binary + 25 multiclass); optional blocks run the newer task sets,
  control tasks, class pairs, three regression tracks and a positions table.
- `--cells NAME|PATH` for `python -m imindbench.launch`: keeps only the
  task/target pairs listed in a cell list. The Miller lists are packaged under
  `imindbench/cell_manifests/millerecog2019/`, with a builder that recreates
  them from the prepared files.
- Dataset options `label_mode: regression`, `class_pair`, `fold_subset`,
  `regression_target_last_samples` and matched train subsets
  (`train_sample_indices_*`), for datasets that support them (currently
  Miller only). Regression results report trajectory Pearson r, mean r, MSE
  and R² per fold.
- Model options `regression_head`, `regression_head_lambda` and
  `regression_head_lambda_mode`, declared for every model.
- `miller_*` preprocessor presets (no filtering or re-referencing) and the
  coordinate profiles `popt_zero`, `popt_miller` and `diver_mni_miller`.
- Omni-iEEG as a sixth dataset, `omni_ieeg` and `omni_ieeg_single_channel`,
  with `scripts/run_omni_ieeg.sh`: the five single-channel tasks
  (`ictal_interictal`, `sleep_awake`, `soz_outcome`, `anatomical`, `hfo`) as
  one pooled cell each (launcher target `all`, regime `hold-out-subject`),
  and the scorers `imindbench.omni_ieeg_scoring` (literal ports of the
  upstream evaluation scripts) and `imindbench.omni_ieeg_paper_scoring`.
  Original implementation by Joan Lara Formoso in the torch_brain and
  DIVER_pretrain evaluation code.
- `dataset.provider_options`, forwarded to the dataset class as keyword
  arguments for providers that accept them; the single-channel protocol
  (`provider_options.single_channel`), which expands preprocessed windows into
  one sample per channel and skips region-intersection pooling;
  `runner.ignore_label`, `runner.train_sampler: balanced`,
  `runtime.save_test_predictions` and `runtime.save_test_prediction_details`
  (per-window test predictions, optionally with channel names and logits);
  the `omni_*` preprocessor presets and the coordinate profile `omni_none`.

### Upgrade notes

- Miller needs a torch_brain that provides `torch_brain.datasets.MillerECoG2019`,
  and Omni-iEEG one that provides `torch_brain.datasets.OmniIEEG` (or a
  `brainsets.datasets.OmniIEEG`). The pinned torch_brain has neither yet; the
  pin will be updated in a later release. The other datasets are unchanged.
- `pandas` is now a declared dependency (the Omni-iEEG scorers use it).
- Result JSONs of the existing datasets keep their shape. The new dataset
  options (`label_mode`, `class_pair`, `fold_subset`,
  `regression_target_last_samples`, matched train subsets) are added to the
  result config only when they differ from the default.

## [0.1.0] - 2026-09-20

### Added

- Initial versioned release of the public iMINDBench evaluation package.
- Evaluation workflows for NeuroprobeV2, Bang! You're Dead, and PIPPI, with
  bundled preprocessing configs, baseline and pretrained-model integrations,
  and result JSON export.
- Dataset launch scripts for within-session evaluation and supported transfer
  and sample-efficiency experiments.

### Release notes

- Establishes the existing `main` implementation as the versioned baseline;
  no preprocessing, training, or evaluation behavior changes in this release.
- CPU evaluation supports Python 3.10–3.13; GPU models on newer Python versions
  are not yet validated. See the README for dependencies and checkpoint setup.
