# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-09-20

### Changed

- Rename "Learner" to "CostLearner".
- Rework and fix the cost learning algorithm.
- Remove `with_cost_function` from `CostLearner`.
- Remove the functional interface in favour of `WeightedLevenshtein` class.

### Added

- Add `calculate_for_unseen` parameter to `CostLearner.fit()`.
- Add input validation in `WeightedLevenshtein.__init__`.
- Add `to_dict` and `from_dict` methods to `WeightedLevenshtein`.

## [0.3.0] - 2025-09-14

### Added

- Add the option to include the matched characters in the `explain` method via the `filter_matches` parameter.
- Add the option to learn the costs from a dataset of pairs (OCR result, ground truth) via the `WeightedLevenshtein.learn_from` method and the `Learner` class.

### Changed

- Drop support for PyPy due to issues with PyO3.

## [0.2.2] - 2025-09-01

### Changed

- Improve documentation.

## [0.2.1] - 2025-08-31

### Fixed

- Documentation for PyPI

## [0.2.0] - 2025-08-31

### Added

- `WeightedLevenshtein` class for reusable configuration.
- Explanation of edit operations via `WeightedLevenshtein.explain` and `explain_weighted_levenshtein`.

## [0.1.0] - 2025-04-26

### Added

- Custom insertion and deletion costs for weighted Levenshtein distance.

### Changed

- Breaking changes to Levenshtein distance functions signatures.
