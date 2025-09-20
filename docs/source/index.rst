================
 OCR-StringDist
================

A Python library to learn, model, explain and correct OCR errors using a fast string distance engine.

:Repository: https://github.com/NiklasvonM/ocr-stringdist
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

OCR-StringDist provides a learnable **weighted Levenshtein distance**, implementing part of the **Noisy Channel model**.

**Example:** Matching against the correct word `CODE`:

* **Standard Levenshtein:**
    * :math:`d(\text{"C0DE"}, \text{"CODE"}) = 1` (0 → O)
    * :math:`d(\text{"CXDE"}, \text{"CODE"}) = 1` (X → O)
    * Result: Both appear equally likely/distant.

* **OCR-StringDist (Weighted):**
    * :math:`d(\text{"C0DE"}, \text{"CODE"}) \approx 0.1` (common error, low cost)
    * :math:`d(\text{"CXDE"}, \text{"CODE"}) = 1.0` (unlikely error, high cost)
    * Result: Correctly identifies `C0DE` as a much closer match.

This makes it ideal for matching potentially incorrect OCR output against known values (e.g., product codes). By combining this *channel model* with a *source model* (e.g., product code frequencies), you can build a complete and robust OCR correction system.

Features
========

- **Learnable Costs**: Automatically learn substitution, insertion, and deletion costs from a dataset of (OCR string, ground truth string) pairs.
- **Weighted Levenshtein Distance**: Models OCR error patterns by assigning custom costs to specific edit operations.
- **High Performance**: Core logic in Rust and a batch_distance function for efficiently comparing one string against thousands of candidates.
- **Substitution of Multiple Characters**: Not just character pairs, but string pairs may be substituted, for example the Korean syllable "이" for the two letters "OI".
- **Explainable Edit Path**: Returns the optimal sequence of edit operations (substitutions, insertions, and deletions) used to transform one string into another.
- **Pre-defined OCR Distance Map**: A built-in distance map for common OCR confusions (e.g., "0" vs "O", "1" vs "l", "5" vs "S").
- **Full Unicode Support**: Works with arbitrary Unicode strings.

Contents
========

.. toctree::
   :maxdepth: 1

   getting-started
   examples
   end_to_end_example
   cost_learning_model
   api/index
   changelog
