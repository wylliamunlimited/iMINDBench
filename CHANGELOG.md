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

### Upgrade notes

- Miller needs a torch_brain that provides `torch_brain.datasets.MillerECoG2019`.
  The pinned torch_brain does not have it yet; the pin will be updated in a
  later release. The other datasets are unchanged.
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
