import pytest
from ocr_stringdist import WeightedLevenshtein
from ocr_stringdist.levenshtein import EditOperation


@pytest.mark.parametrize(
    ["s1", "s2", "expected_operations", "wl"],
    [
        (
            "kitten",
            "sitting",
            [
                EditOperation("substitute", "k", "s", 1.0),
                EditOperation("match", "i", "i", 0.0),
                EditOperation("match", "t", "t", 0.0),
                EditOperation("match", "t", "t", 0.0),
                EditOperation("substitute", "e", "i", 1.0),
                EditOperation("match", "n", "n", 0.0),
                EditOperation("insert", None, "g", 1.0),
            ],
            WeightedLevenshtein(substitution_costs={}),
        ),
        (
            "flaw",
            "lawn",
            [
                EditOperation("delete", "f", None, 1.0),
                EditOperation("match", "l", "l", 0.0),
                EditOperation("match", "a", "a", 0.0),
                EditOperation("match", "w", "w", 0.0),
                EditOperation("insert", None, "n", 1.0),
            ],
            WeightedLevenshtein(substitution_costs={}),
        ),
        (  # Multi-character substitution
            "rn",
            "m",
            [
                EditOperation("substitute", "rn", "m", 0.5),
            ],
            WeightedLevenshtein(substitution_costs={("rn", "m"): 0.5}),
        ),
        (
            "Hello",
            "H3llo!",
            [
                EditOperation("match", "H", "H", 0.0),
                EditOperation("substitute", "e", "3", 0.7),
                EditOperation("match", "l", "l", 0.0),
                EditOperation("match", "l", "l", 0.0),
                EditOperation("match", "o", "o", 0.0),
                EditOperation("insert", None, "!", 1.0),
            ],
            WeightedLevenshtein(substitution_costs={("e", "3"): 0.7}),
        ),
        (
            "Equal",
            "Equal",
            [
                EditOperation("match", "E", "E", 0.0),
                EditOperation("match", "q", "q", 0.0),
                EditOperation("match", "u", "u", 0.0),
                EditOperation("match", "a", "a", 0.0),
                EditOperation("match", "l", "l", 0.0),
            ],
            WeightedLevenshtein(substitution_costs={}),
        ),
    ],
)
def test_explain_weighted_levenshtein(
    s1: str, s2: str, expected_operations: list[EditOperation], wl: WeightedLevenshtein
) -> None:
    full_operations = wl.explain(s1, s2, filter_matches=False)
    filtered_operations = wl.explain(s1, s2, filter_matches=True)
    manually_filtered_operations = [op for op in full_operations if op.op_type != "match"]
    assert filtered_operations == manually_filtered_operations
    assert full_operations == expected_operations
