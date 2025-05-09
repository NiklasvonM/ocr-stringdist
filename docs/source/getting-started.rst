=================
 Getting Started
=================

Installation
============

.. code-block:: console

    pip install ocr-stringdist

Quick Example
=============

After installation, you can quickly calculate an OCR-aware string distance:

.. code-block:: python

    import ocr_stringdist as osd

    # Calculate distance using the default OCR error costs
    # ("O" vs "0" has a low cost)
    distance = osd.weighted_levenshtein_distance("HELLO", "HELL0")

    print(f"The OCR-aware distance is: {distance}")

This uses the built-in :data:`ocr_distance_map` which assigns lower costs to common OCR character confusions. See the :doc:`examples` and :doc:`api/index` for more details and customization options.
