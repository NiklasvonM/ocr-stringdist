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

    import ocr_stringdist as osd

    # Define a custom cost for substituting "rn" with "m"
    custom_substitutions = {("rn", "m"): 0.5}

    distance = osd.weighted_levenshtein_distance(
        "Churn Bucket", "Chum Bucket", substitution_costs=custom_substitutions
    )
    print(f"Distance using custom map: {distance}") # 0.5


Matching OCR Output Against Candidates
======================================

This is a primary use case: finding the best match for an OCR string from a list of known possibilities.

.. code-block:: python

    import ocr_stringdist as osd

    ocr_output = "Harnburg"  # OCR potentially misread 'm' as 'rn'
    possible_cities = ["Harburg", "Hamburg", "Hannover", "Berlin"]

    # Define costs relevant to the potential error
    ocr_fix_costs = {("rn", "m"): 0.2}

    # Method 1: Using find_best_candidate
    best_match_finder, min_distance_finder = osd.find_best_candidate(
        ocr_output,
        possible_cities,
        distance_fun=lambda s1, s2: osd.weighted_levenshtein_distance(
            s1, s2, substitution_costs=ocr_fix_costs
        ),
    )
    print(
        f"(find_best_candidate) Best match for '{ocr_output}': '{best_match_finder}' "
        f"(Distance: {min_distance_finder:.2f})"
    )
    # Output: (find_best_candidate) Best match for 'Harnburg': 'Hamburg' (Distance: 0.20)


    # Method 2: Using batch_weighted_levenshtein_distance
    # Generally more efficient when comparing against many candidates.
    distances = osd.batch_weighted_levenshtein_distance(
        ocr_output, possible_cities, substitution_costs=ocr_fix_costs
    )

    min_dist_batch = min(distances)
    best_candidate_batch = possible_cities[distances.index(min_dist_batch)]

    print(
        f"(Batch) Best match for '{ocr_output}': '{best_candidate_batch}' "
        f"(Distance: {min_dist_batch:.2f})"
    )
    # Output: (Batch) Best match for 'Harnburg': 'Hamburg' (Distance: 0.20)

