.. _api_reference:

API Reference
=============

.. autoclass:: ocr_stringdist.WeightedLevenshtein
   :members:
   :noindex:
   :exclude-members: __repr__, __str__

.. autofunction:: ocr_stringdist.weighted_levenshtein_distance
.. autofunction:: ocr_stringdist.batch_weighted_levenshtein_distance
.. autofunction:: ocr_stringdist.explain_weighted_levenshtein

.. automodule:: ocr_stringdist.matching
   :members:
   :undoc-members:
   :show-inheritance:

.. autodata:: ocr_stringdist.default_ocr_distances.ocr_distance_map
   :annotation:
.. literalinclude:: ../../../python/ocr_stringdist/default_ocr_distances.py
   :language: python
   :start-after: OCR_DISTANCE_MAP_START
   :end-before: OCR_DISTANCE_MAP_END
