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
                EditOperation("substitute", "e", "i", 1.0),
                EditOperation("insert", None, "g", 1.0),
            ],
            WeightedLevenshtein(substitution_costs={}),
        ),
        (
            "flaw",
            "lawn",
            [
                EditOperation("delete", "f", None, 1.0),
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
                EditOperation("substitute", "e", "3", 0.7),
                EditOperation("insert", None, "!", 1.0),
            ],
            WeightedLevenshtein(substitution_costs={("e", "3"): 0.7}),
        ),
    ],
)
def test_explain_weighted_levenshtein(
    s1: str, s2: str, expected_operations: list[EditOperation], wl: WeightedLevenshtein
) -> None:
    operations = wl.explain(s1, s2)
    assert operations == expected_operations
