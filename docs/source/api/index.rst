.. _api_reference:

===============
 API Reference
===============

.. autoclass:: ocr_stringdist.WeightedLevenshtein
   :members:

.. autofunction:: ocr_stringdist.weighted_levenshtein_distance
.. autofunction:: ocr_stringdist.batch_weighted_levenshtein_distance
.. autofunction:: ocr_stringdist.explain_weighted_levenshtein

.. autoclass:: ocr_stringdist.learner.Learner
   :members:

.. autoclass:: ocr_stringdist.edit_operation.EditOperation
   :members:

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
