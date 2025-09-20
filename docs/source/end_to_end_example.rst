===========================================================
 End-to-End Workflow: Training Costs and Matching Products
===========================================================

This guide demonstrates a complete, production-oriented workflow for correcting OCR data. The process is split into two phases:

1.  **Phase 1 (One-Time Setup)**: We learn the specific OCR error costs from a sample of our own data. These learned costs are then saved, for example to a JSON file.
2.  **Phase 2 (Runtime Application)**: In our main application, we load the pre-trained costs and use an optimized batch process to quickly and accurately find the best match for new OCR scans from our product database.

Phase 1 (One-Time Setup): Learning Costs
========================================

First, we collect a representative set of (OCR string, correct ground truth) pairs. Using this data, we can train a `WeightedLevenshtein` instance to learn the probabilities of our specific OCR engine's error patterns.

.. code-block:: python

    import json

    from ocr_stringdist import WeightedLevenshtein

    # A sample of observed OCR results and their correct counterparts.
    training_data = [
        ("SKU-B0O-BTR", "SKU-800-BTR"),  # B -> 8, O -> 0
        ("SKU-5A1-HIX", "SKU-5A1-MIX"),  # H -> M
        ("SKU-B01-SGR", "SKU-B01-SGR"),  # Include correct examples
        # ... add more data for better results
    ]

    wl_trained = WeightedLevenshtein.learn_from(training_data)

    # Insertion/deletions costs are handled similarly
    learned_costs = wl_trained.substitution_costs
    print(f"Learned cost for ('B', '8'): {learned_costs.get(('B', '8')):.4f}")
    print(f"Learned cost for ('O', '0'): {learned_costs.get(('O', '0')):.4f}")


    # Save the learned costs to a file for later use in our application.
    with open("ocr_costs.json", "w") as f:
        json.dump(wl_trained.to_dict(), f, indent=2)


This saved `ocr_costs.json` file can, possibly after manual review, be deployed with your application.

Phase 2 (Runtime Application): Finding the Best Match
=====================================================

In our live application, we load the pre-computed costs at startup. When a new scan comes in, we use the highly optimized `batch_distance` method to find the best match efficiently.

**The Scenario**

We use the same product database as before and receive a new, imperfect OCR scan.

+--------------+-----------------------+---------+------------+
| Product Code | Description           | Price   | Sales Rank |
+==============+=======================+=========+============+
| SKU-800-BTR  | 800W Power Blender    | 119.95  | 1          |
+--------------+-----------------------+---------+------------+
| SKU-B01-SGR  | Cold Press Juicer     | 149.50  | 3          |
+--------------+-----------------------+---------+------------+
| SKU-5A1-MIX  | 5-Speed Hand Mixer    | 49.99   | 2          |
+--------------+-----------------------+---------+------------+

* **Scanned Code**: `"SKU-B0O-BTR"` (Errors: `B` instead of `8`, `O` instead of `0`)
* **Scanned Price**: `119.95`


.. code-block:: python

    import json
    import math
    from dataclasses import dataclass
    from typing import Any

    from ocr_stringdist import WeightedLevenshtein


    # Setup: Load Data and Pre-trained Costs


    @dataclass
    class Product:
        code: str
        description: str
        price: float
        sales_rank: int


    db_products = [
        Product(code="SKU-800-BTR", description="800W Power Blender", price=119.95, sales_rank=1),
        Product(code="SKU-B01-SGR", description="Cold Press Juicer", price=149.50, sales_rank=2),
        Product(code="SKU-5A1-MIX", description="5-Speed Hand Mixer", price=49.99, sales_rank=3),
    ]

    # Load configuration
    with open("ocr_costs.json") as f:
        wl = WeightedLevenshtein.from_dict(json.load(f))


    # Correction Logic for a New Scan


    ocr_code = "SKU-B0O-BTR"
    ocr_price = 119.95

    # Calculate all string distances in a single, optimized batch operation.
    string_distances = wl.batch_distance(ocr_code, candidates=[p.code for p in db_products])

    # Calculate other costs, like a price mismatch penalty.
    price_penalties = [0.0 if p.price == ocr_price else 1.0 for p in db_products]

    # Our source model: products sold rarely are a-priori less likely
    source_costs = [math.log(p.sales_rank) for p in db_products]

    # Combine costs to get a final score for each candidate.
    total_costs = [d + p + s for d, p, s in zip(string_distances, price_penalties, source_costs)]

    # Find the candidate with the minimum total cost.
    min_cost = min(total_costs)
    best_product = db_products[total_costs.index(min_cost)]


    print(f"OCR Scan (Code): '{ocr_code}', (Price): {ocr_price}\n")
    print(f"Best Match Found: {best_product}")
    print(f"Confidence Score (Lower is Better): {min_cost:.2f}")


This workflow is efficient and robust: the heavy lifting of learning is done offline, and the runtime matching uses an optimized batch process to combine multiple sources of evidence (string similarity, price and sales rank) for an accurate result.
