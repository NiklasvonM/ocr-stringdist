================
 OCR-StringDist
================

A Python library for fast string distance calculations that account for common OCR (optical character recognition) errors.

:Repository: https://niklasvonm.github.io/ocr-stringdist/
:Current version: |release|

.. image:: https://img.shields.io/badge/PyPI-Package-blue
   :target: https://pypi.org/project/ocr-stringdist/
   :alt: PyPI

.. image:: https://img.shields.io/badge/License-MIT-green
   :target: LICENSE
   :alt: License

Motivation
==========

Standard string distances (like Levenshtein) treat all character substitutions equally. This is suboptimal for text read from images via OCR, where errors like `O` vs `0` are far more common than, say, `O` vs `X`.

OCR-StringDist uses a **weighted Levenshtein distance**, assigning lower costs to common OCR errors.

**Example:** Matching against the correct word `CODE`:

* **Standard Levenshtein:**
    * :math:`d(\text{"C0DE"}, \text{"CODE"}) = 1` (0 → O)
    * :math:`d(\text{"CXDE"}, \text{"CODE"}) = 1` (X → O)
    * Result: Both appear equally likely/distant.

* **OCR-StringDist (Weighted):**
    * :math:`d(\text{"C0DE"}, \text{"CODE"}) \approx 0.1` (common error, low cost)
    * :math:`d(\text{"CXDE"}, \text{"CODE"}) = 1.0` (unlikely error, high cost)
    * Result: Correctly identifies `C0DE` as a much closer match.

This makes it ideal for matching potentially incorrect OCR output against known values (e.g., product codes, database entries).

Features
========

- **Weighted Levenshtein Distance**: Calculates Levenshtein distance with customizable costs for substitutions, insertions, and deletions. Includes an efficient batch version (`batch_weighted_levenshtein_distance`) for comparing one string against many candidates.
- **Substitution of Multiple Characters**: Not just character pairs, but string pairs may be substituted, for example the Korean syllable "이" for the two letters "OI".
- **Pre-defined OCR Distance Map**: A built-in distance map for common OCR confusions (e.g., "0" vs "O", "1" vs "l", "5" vs "S").
- **Unicode Support**: Works with arbitrary Unicode strings.
- **Best Match Finder**: Includes a utility function `find_best_candidate` to efficiently find the best match from a list based on _any_ distance function.

Contents
========

.. toctree::
   :maxdepth: 1

   getting-started
   examples
   api/index
   changelog
