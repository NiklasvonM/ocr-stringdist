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

> **Note:** This project is in early development. APIs may change in future releases.

## Installation

```bash
pip install ocr-stringdist
```

## Features

- **Weighted Levenshtein Distance**: Calculates Levenshtein distance with customizable costs for substitutions, insertions, and deletions. Includes an efficient batch version (`batch_weighted_levenshtein_distance`) for comparing one string against many candidates.
- **Substitution of Multiple Characters**: Not just character pairs, but string pairs may be substituted, for example the Korean syllable "이" for the two letters "OI".
- **Pre-defined OCR Distance Map**: A built-in distance map for common OCR confusions (e.g., "0" vs "O", "1" vs "l", "5" vs "S").
- **Unicode Support**: Works with arbitrary Unicode strings.
- **Best Match Finder**: Includes a utility function `find_best_candidate` to efficiently find the best match from a list based on _any_ distance function.

## Usage

### Weighted Levenshtein Distance

```python
import ocr_stringdist as osd

# Using default OCR distance map
distance = osd.weighted_levenshtein_distance("OCR5", "OCRS")
print(f"Distance between 'OCR5' and 'OCRS': {distance}")  # Will be less than 1.0

# Custom cost map
substitution_costs = {("In", "h"): 0.5}
distance = osd.weighted_levenshtein_distance(
    "hi", "Ini",
    substitution_costs=substitution_costs,
    symmetric_substitution=True,
)
print(f"Distance with custom map: {distance}")
```

## Acknowledgements

This project is inspired by [jellyfish](https://github.com/jamesturk/jellyfish), providing the base implementations of the algorithms used here.
