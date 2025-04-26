"""
Unit tests for the batch processing functions.
"""

import pytest
from ocr_stringdist import batch_weighted_levenshtein_distance, weighted_levenshtein_distance

# Define a custom cost map with some OCR confusions for testing
OCR_COST_MAP = {
    ("l", "1"): 0.2,  # l to 1 is a common OCR error
    ("O", "0"): 0.1,  # O to 0 is a common OCR error
    ("o", "0"): 0.1,
    ("m", "rn"): 0.3,  # m to rn is a common OCR error
}


@pytest.mark.parametrize(
    ["source", "candidates", "cost_map"],
    [
        (
            "recognition",
            ["recognition", "recogmtion", "recognltlon", "recogrtition", "recognitton"],
            None,
        ),
        (
            "hello",
            ["hello", "he11o", "hell0"],
            OCR_COST_MAP,
        ),
        (
            "algorithm",
            ["algorithm", "algorlthm", "a1gorithm"],
            OCR_COST_MAP,
        ),
    ],
)
def test_batch_vs_individual(
    source: str, candidates: list[str], cost_map: dict[tuple[str, str], float]
) -> None:
    """Test that batch results match individual function calls."""
    # Individual results
    individual_results = [
        weighted_levenshtein_distance(source, candidate, substitution_costs=cost_map)
        for candidate in candidates
    ]

    # Batch results
    batch_results = batch_weighted_levenshtein_distance(
        source, candidates, substitution_costs=cost_map
    )

    # Compare results
    for ind, batch in zip(individual_results, batch_results):
        assert ind == pytest.approx(batch)


@pytest.mark.parametrize(
    ["source", "candidates", "expected_indices"],
    [
        (
            "hello",
            ["hello", "he11o", "hell0", "hallo", "help"],
            [0],  # exact match should be the best
        ),
        (
            "algorithm",
            ["a1gorithm", "algorithm", "algorlthm", "alg0rithm"],
            [1],  # exact match should be the best
        ),
        (
            "recognition",
            ["wreck", "cognition", "recogmition", "wreckognition"],
            [2],  # "recogmtion" should be closest to "recognition"
        ),
    ],
)
def test_batch_finds_best_match(
    source: str, candidates: list[str], expected_indices: list[int]
) -> None:
    """Test that batch processing correctly identifies the best match."""
    # Using OCR cost map
    distances = batch_weighted_levenshtein_distance(
        source, candidates, substitution_costs=OCR_COST_MAP
    )
    print(f"------------------------------------distances: {distances}")

    # Find the index with minimum distance
    min_index = distances.index(min(distances))

    # Check if the minimum index is in the expected indices
    assert min_index in expected_indices


@pytest.mark.parametrize(
    ["test_string", "expected_distance"],
    [
        ("hello", 0.0),  # exact match
        ("he11o", 0.4),  # two l->1 substitutions at cost 0.2 each
        ("hell0", 0.1),  # one O->0 substitution at cost 0.1
    ],
)
def test_custom_cost_map(test_string: str, expected_distance: float) -> None:
    """Test using a custom cost map for specific substitution costs."""
    result = weighted_levenshtein_distance("hello", test_string, substitution_costs=OCR_COST_MAP)
    assert result == pytest.approx(expected_distance)

    # Check that batch processing gives the same result
    batch_result = batch_weighted_levenshtein_distance(
        "hello", [test_string], substitution_costs=OCR_COST_MAP
    )[0]
    assert batch_result == pytest.approx(expected_distance)


@pytest.mark.parametrize(
    ["string1", "string2", "default_map_distance", "custom_map_distance"],
    [
        ("hello", "he11o", 2.0, 0.4),  # l->1 costs 0.2 each instead of 1.0 each
        ("hello", "hell0", 1.0, 0.1),  # o->0 costs 0.1 instead of 1.0
        ("come", "corne", 2.0, 0.3),  # rn->m costs 0.3 instead of 2.0
    ],
)
def test_empty_vs_default_cost_map(
    string1: str, string2: str, default_map_distance: float, custom_map_distance: float
) -> None:
    """Test that empty cost maps produce different results than default cost maps."""
    # With empty cost map (all costs are 1.0)
    default_result = batch_weighted_levenshtein_distance(string1, [string2], substitution_costs={})
    assert default_result[0] == pytest.approx(default_map_distance)

    # With custom cost map (OCR-specific costs)
    custom_result = batch_weighted_levenshtein_distance(
        string1, [string2], substitution_costs=OCR_COST_MAP
    )
    assert custom_result[0] == pytest.approx(custom_map_distance)

    # Custom map should give lower distance for OCR errors
    assert custom_result[0] < default_result[0]
