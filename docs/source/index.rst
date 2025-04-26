================
 OCR-StringDist
================

A Python library for string distance calculations that account for common OCR (optical character recognition) errors, written in Rust.

:Repository: https://niklasvonm.github.io/ocr-stringdist/
:Current version: |release|

.. image:: https://img.shields.io/badge/PyPI-Package-blue
   :target: https://pypi.org/project/ocr-stringdist/
   :alt: PyPI

.. image:: https://img.shields.io/badge/License-MIT-green
   :target: LICENSE
   :alt: License

Features
========

- **Weighted Levenshtein Distance**: An adaptation of the classic Levenshtein algorithm with custom substitution costs for character pairs that are commonly confused in OCR models, including efficient batch processing.
- **Unicode Support**: Arbitrary unicode strings can be compared.
- **Substitution of Multiple Characters**: Not just character pairs, but string pairs may be substituted, for example the Korean syllable "이" for the two letters "OI".
- **Pre-defined OCR Distance Map**: A built-in distance map for common OCR confusions (e.g., "0" vs "O", "1" vs "l", "5" vs "S").
- **Best Match Finder**: Utility function ``find_best_candidate`` to efficiently find the best matching string from a collection of candidates using any specified distance function (including the library's OCR-aware ones).

Contents
========

.. toctree::
   :maxdepth: 2

   getting-started
   examples
   api/index
