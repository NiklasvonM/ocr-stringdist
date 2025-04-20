# OCR-StringDist

A Python library for string distance calculations that account for common OCR (optical character recognition) errors.

Documentation: https://niklasvonm.github.io/ocr-stringdist/

[![PyPI](https://img.shields.io/badge/PyPI-Package-blue)](https://pypi.org/project/ocr-stringdist/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## Overview

OCR-StringDist provides specialized string distance algorithms that accommodate for optical character recognition (OCR) errors. Unlike traditional string comparison algorithms, OCR-StringDist considers common OCR confusions (like "0" vs "O", "6" vs "G", etc.) when calculating distances between strings.

> **Note:** This project is in early development. APIs may change in future releases.

## Installation

```bash
pip install ocr-stringdist
```

## Features

- **Weighted Levenshtein Distance**: An adaptation of the classic Levenshtein algorithm with custom substitution costs for character pairs that are commonly confused in OCR models.
- **Unicode Support**: Arbitrary unicode strings can be compared.
- **Substitution of Multiple Characters**: Not just character pairs, but string pairs may be substituted, for example the Korean syllable "이" for the two letters "OI".
- **Pre-defined OCR Distance Map**: A built-in distance map for common OCR confusions (e.g., "0" vs "O", "1" vs "l", "5" vs "S").
- **Customizable Cost Maps**: Create your own substitution cost maps for specific OCR systems or domains.
- **Best Match Finder**: Utility function `find_best_candidate` to efficiently find the best matching string from a collection of candidates using any specified distance function (including the library's OCR-aware ones).

## Usage

### Weighted Levenshtein Distance

```python
import ocr_stringdist as osd

# Using default OCR distance map
distance = osd.weighted_levenshtein_distance("OCR5", "OCRS")
print(f"Distance between 'OCR5' and 'OCRS': {distance}")  # Will be less than 1.0

# Custom cost map
custom_map = {("In", "h"): 0.5}
distance = osd.weighted_levenshtein_distance(
    "hi", "Ini",
    cost_map=custom_map,
    symmetric=True,
    default_cost=1.0,
)
print(f"Distance with custom map: {distance}")
```

### Finding the Best Candidate

```python
import ocr_stringdist as osd

s = "apple"
candidates = ["apply", "apples", "orange", "appIe"]  # 'appIe' has an OCR-like error

def ocr_aware_distance(s1: str, s2: str) -> float:
    return osd.weighted_levenshtein_distance(s1, s2, cost_map={("l", "I"): 0.1})

best_candidate, best_dist = osd.find_best_candidate(s, candidates, ocr_aware_distance)
print(f"Best candidate for '{s}' is '{best_candidate}' with distance {best_dist}")
# Output: Best candidate for 'apple' is 'appIe' with distance 0.1
```

## Acknowledgements

This project is inspired by [jellyfish](https://github.com/jamesturk/jellyfish), providing the base implementations of the algorithms used here.
