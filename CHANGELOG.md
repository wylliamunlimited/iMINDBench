# Changelog

Notable changes and upgrade actions are recorded here.

## [Unreleased]

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
