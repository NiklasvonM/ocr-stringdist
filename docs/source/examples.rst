==========
 Examples
==========

.. code-block:: python

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
    )
    print(f"Distance with custom map: {distance}")
