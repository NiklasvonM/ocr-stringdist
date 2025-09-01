# OCR-StringDist

A Python library for fast string distance calculations that account for common OCR (optical character recognition) errors.

Documentation: https://niklasvonm.github.io/ocr-stringdist/

[![PyPI](https://img.shields.io/badge/PyPI-Package-blue)](https://pypi.org/project/ocr-stringdist/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## Overview

Standard string distances (like Levenshtein) treat all character substitutions equally. This is suboptimal for text read from images via OCR, where errors like `O` vs `0` are far more common than, say, `O` vs `X`.

OCR-StringDist uses a **weighted Levenshtein distance**, assigning lower costs to common OCR errors.

**Example:** Matching against the correct word `CODE`:

* **Standard Levenshtein:**
    * $d(\text{"CODE"}, \text{"C0DE"}) = 1$ (O → 0)
    * $d(\text{"CODE"}, \text{"CXDE"}) = 1$ (O → X)
    * Result: Both appear equally likely/distant.

* **OCR-StringDist (Weighted):**
    * $d(\text{"CODE"}, \text{"C0DE"}) \approx 0.1$ (common error, low cost)
    * $d(\text{"CODE"}, \text{"CXDE"}) = 1.0$ (unlikely error, high cost)
    * Result: Correctly identifies `C0DE` as a much closer match.

This makes it ideal for matching potentially incorrect OCR output against known values (e.g., product codes, database entries).

## Installation

```bash
pip install ocr-stringdist
```

## Features

- **High Performance**: The core logic is implemented in Rust with speed in mind.
- **Weighted Levenshtein Distance**: Calculates Levenshtein distance with customizable costs for substitutions, insertions, and deletions. Includes an efficient batch version (`batch_weighted_levenshtein_distance`) for comparing one string against many candidates.
- **Explainable Edit Path**: Returns the optimal sequence of edit operations (substitutions, insertions, and deletions) used to transform one string into another.
- **Substitution of Multiple Characters**: Not just character pairs, but string pairs may be substituted, for example the Korean syllable "이" for the two letters "OI".
- **Pre-defined OCR Distance Map**: A built-in distance map for common OCR confusions (e.g., "0" vs "O", "1" vs "l", "5" vs "S").
- **Unicode Support**: Works with arbitrary Unicode strings.
- **Best Match Finder**: Includes a utility function `find_best_candidate` to efficiently find the best match from a list based on _any_ distance function.

## Usage

### Basic usage

```python
from ocr_stringdist import WeightedLevenshtein

# Default substitution costs are ocr_stringdist.ocr_distance_map.
wl = WeightedLevenshtein()

print(wl.distance("CXDE", "CODE")) # == 1
print(wl.distance("C0DE", "CODE")) # < 1
```

### Explain the Edit Path

```python
edit_path = wl.explain("C0DE", "CODE")
print(edit_path)
# [EditOperation(op_type='substitute', source_token='0', target_token='O', cost=0.1)]
```

### Fast Batch Calculations

Quickly compare a string to a list of candidates.

```python
distances: list[float] = wl.batch_distance("CODE", ["CXDE", "C0DE"])
# [1.0, 0.1]
```

### Multi-character Substitutions

```python
# Custom costs with multi-character substitution
wl = WeightedLevenshtein(substitution_costs={("In", "h"): 0.5})

print(wl.distance("hi", "Ini")) # 0.5
```


## Acknowledgements

This project is inspired by [jellyfish](https://github.com/jamesturk/jellyfish), providing the base implementations of the algorithms used here.
