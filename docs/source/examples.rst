================
 Usage Examples
================

Basic Distance Calculation
==========================

Using the default pre-defined map for common OCR errors:

.. code-block:: python

    import ocr_stringdist as osd

    # Compare "OCR5" and "OCRS"
    # The default ocr_distance_map gives 'S' <-> '5' a cost of 0.3
    distance = osd.weighted_levenshtein_distance("OCR5", "OCRS")
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


Matching OCR Output Against Candidates
======================================

This is a primary use case: finding the best match for an OCR string from a list of known possibilities.

.. code-block:: python

    import ocr_stringdist as osd

    ocr_output = "Harnburg"  # OCR potentially misread 'm' as 'rn'
    possible_cities = ["Harburg", "Hamburg", "Hannover", "Berlin"]

    # Define costs relevant to the potential error
    wl = osd.WeightedLevenshtein(substitution_costs={("rn", "m"): 0.2})

    # Method 1: Using find_best_candidate
    best_match_finder, min_distance_finder = osd.find_best_candidate(
        ocr_output,
        possible_cities,
        distance_fun=wl.distance,
    )
    print(
        f"(find_best_candidate) Best match for '{ocr_output}': '{best_match_finder}' "
        f"(Distance: {min_distance_finder:.2f})"
    )
    # Output: (find_best_candidate) Best match for 'Harnburg': 'Hamburg' (Distance: 0.20)


    # Method 2: Using WeightedLevenshtein.batch_distance
    # Generally more efficient when comparing against many candidates.
    distances = wl.batch_distance(ocr_output, possible_cities)

    min_dist_batch = min(distances)
    best_candidate_batch = possible_cities[distances.index(min_dist_batch)]

    print(
        f"(Batch) Best match for '{ocr_output}': '{best_candidate_batch}' "
        f"(Distance: {min_dist_batch:.2f})"
    )
    # Output: (Batch) Best match for 'Harnburg': 'Hamburg' (Distance: 0.20)

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
