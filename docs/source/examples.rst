================
 Usage Examples
================

Basic Distance Calculation
==========================

Using the default pre-defined map for common OCR errors:

.. code-block:: python

    from ocr_stringdist import WeightedLevenshtein

    # Compare "OCR5" and "OCRS"
    # The default ocr_distance_map gives 'S' <-> '5' a cost of 0.3
    distance: float = WeightedLevenshtein().distance("OCR5", "OCRS")
    print(f"Distance between 'OCR5' and 'OCRS' (default map): {distance}")
    # Output: Distance between 'OCR5' and 'OCRS' (default map): 0.3

Using Custom Costs
==================

Define your own substitution costs:

.. code-block:: python

    from ocr_stringdist import WeightedLevenshtein

    # Define a custom cost for substituting "rn" with "m"
    wl = WeightedLevenshtein(substitution_costs={("rn", "m"): 0.5})

    distance = wl.distance("Churn Bucket", "Chum Bucket")
    print(f"Distance using custom map: {distance}") # 0.5

Explaining Edit Operations
==========================

You can get a detailed list of edit operations needed to transform one string into another.

.. code-block:: python

    from ocr_stringdist import WeightedLevenshtein

    wl = WeightedLevenshtein(substitution_costs={("日月", "明"): 0.4, ("末", "未"): 0.3})

    s1 = "末日月"  # mò rì yuè
    s2 = "未明"  # wèi míng

    operations = wl.explain(s1, s2)
    print(operations)

    # Output:
    # [
    #   EditOperation(op_type='substitute', source_token='末', target_token='未', cost=0.3),
    #   EditOperation(op_type='substitute', source_token='日月', target_token='明', cost=0.4)
    # ]

Learning Costs from Data
========================

The custom costs can be learned from a dataset of pairs of (OCR output, ground truth).

.. code-block:: python

    from ocr_stringdist import WeightedLevenshtein

    training_data = [
        ("Hallo", "Hello"),
        ("Hello", "Hello"), # Include correct pairs too
        ("W0rld", "World"),
    ]

    # Learn costs from the dataset
    learned_wl = WeightedLevenshtein.learn_from(training_data)

    # Use the learned costs for distance calculation
    distance = learned_wl.distance("Hay", "Hey")
    print(f"Distance with learned costs: {distance}")  # < 1.0

Note that this only supports learning from character-level edits, not multi-character tokens.
