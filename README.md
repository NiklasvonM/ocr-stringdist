# OCR-StringDist

A Python library to learn, model, explain and correct OCR errors using a fast string distance engine.

Documentation: https://niklasvonm.github.io/ocr-stringdist/

[![PyPI badge](https://badge.fury.io/py/ocr-stringdist.svg)](https://badge.fury.io/py/ocr-stringdist)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## Overview

Standard string distances (like Levenshtein) treat all character substitutions equally. This is suboptimal for text read from images via OCR, where errors like `O` vs `0` are far more common than, say, `O` vs `X`.

OCR-StringDist provides a learnable **weighted Levenshtein distance**, implementing part of the **Noisy Channel model**.

**Example:** Matching against the correct word `CODE`:

* **Standard Levenshtein:**
    * $d(\text{"CODE"}, \text{"C0DE"}) = 1$ (O → 0)
    * $d(\text{"CODE"}, \text{"CXDE"}) = 1$ (O → X)
    * Result: Both appear equally likely/distant.

* **OCR-StringDist (Channel Model):**
    * $d(\text{"CODE"}, \text{"C0DE"}) \approx 0.1$ (common error, low cost)
    * $d(\text{"CODE"}, \text{"CXDE"}) = 1.0$ (unlikely error, high cost)
    * Result: Correctly identifies `C0DE` as a much closer match.

This makes it ideal for matching potentially incorrect OCR output against known values (e.g., product codes). By combining this *channel model* with a *source model* (e.g., product code frequencies), you can build a complete and robust OCR correction system.

## Installation

```bash
pip install ocr-stringdist
```

## Features

- **Learnable Costs**: Automatically learn substitution, insertion, and deletion costs from a dataset of (OCR string, ground truth string) pairs.
- **Weighted Levenshtein Distance**: Models OCR error patterns by assigning custom costs to specific edit operations.
- **High Performance**: Core logic in Rust and a batch_distance function for efficiently comparing one string against thousands of candidates.
- **Substitution of Multiple Characters**: Not just character pairs, but string pairs may be substituted, for example the Korean syllable "이" for the two letters "OI".
- **Explainable Edit Path**: Returns the optimal sequence of edit operations (substitutions, insertions, and deletions) used to transform one string into another.
- **Pre-defined OCR Distance Map**: A built-in distance map for common OCR confusions (e.g., "0" vs "O", "1" vs "l", "5" vs "S").
- **Full Unicode Support**: Works with arbitrary Unicode strings.

## Core Workflow

The typical workflow involves
- learning costs from your data and then
- using the resulting model to find the best match from a list of candidates.

```python
from ocr_stringdist import WeightedLevenshtein

# 1. LEARN costs from your own data
training_data = [
    ("128", "123"),
    ("567", "567"),
]
wl = WeightedLevenshtein.learn_from(training_data)

# The engine has now learned that '8' -> '3' is a low-cost substitution
print(f"Learned cost for ('8', '3'): {wl.substitution_costs[('8', '3')]:.2f}")


# 2. MATCH new OCR output against a list of candidates
ocr_output = "Product Code 128"
candidates = [
    "Product Code 123",
    "Product Code 523",  # '5' -> '1' is an unlikely error
]

distances = wl.batch_distance(ocr_output, candidates)

# Find the best match
min_distance = min(distances)
best_match = candidates[distances.index(min_distance)]

print(f"Best match for '{ocr_output}': '{best_match}' (Cost: {min_distance:.2f})")
```
