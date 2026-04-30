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
    assert sum(op.cost for op in full_operations) == wl.distance(s1, s2)


def test_explain_transitive_deletion_chain() -> None:
    """Issue #12: the explain path for '06'->'0' should expose the sub+del chain."""
    wl = WeightedLevenshtein(
        substitution_costs={("6", "G"): 0.5},
        deletion_costs={"G": 0.01},
        symmetric_substitution=False,
    )
    ops = wl.explain("06", "0", filter_matches=False)
    assert ops == [
        EditOperation("match", "0", "0", 0.0),
        EditOperation("substitute", "6", "G", 0.5),
        EditOperation("delete", "G", None, 0.01),
    ]


def test_explain_transitive_substitution_chain() -> None:
    """Triangle inequality: sub(a->b, 0.1) + sub(b->c, 0.1) should expand to two ops."""
    wl = WeightedLevenshtein(
        substitution_costs={("a", "b"): 0.1, ("b", "c"): 0.1},
        symmetric_substitution=False,
    )
    ops = wl.explain("a", "c", filter_matches=False)
    assert ops == [
        EditOperation("substitute", "a", "b", 0.1),
        EditOperation("substitute", "b", "c", 0.1),
    ]


def test_explain_transitive_insertion_chain() -> None:
    """Insertion analogue: ins('x') + sub('x'->'y') chain should appear in the path."""
    wl = WeightedLevenshtein(
        substitution_costs={("x", "y"): 0.2},
        insertion_costs={"x": 0.1},
        symmetric_substitution=False,
    )
    ops = wl.explain("a", "ay", filter_matches=False)
    assert ops == [
        EditOperation("match", "a", "a", 0.0),
        EditOperation("insert", None, "x", 0.1),
        EditOperation("substitute", "x", "y", 0.2),
    ]


def test_explain_chain_with_expensive_direct_substitution() -> None:
    """
    Test that A->AA->AAA->B is explained instead of the more expensive A->B.
    """
    wl = WeightedLevenshtein(
        substitution_costs={("AAA", "B"): 0.1, ("A", "B"): 0.6}, insertion_costs={"A": 0.2}
    )
    ops = wl.explain("A", "B", filter_matches=True)
    assert ops == [
        EditOperation("insert", None, "A", 0.2),
        EditOperation("insert", None, "A", 0.2),
        EditOperation("substitute", "AAA", "B", 0.1),
    ]


def test_explain_mixed_substitution_path_with_deletion() -> None:
    """
    Test that AB->A->C is expanded when it beats the direct AB->C substitution.
    """
    wl = WeightedLevenshtein(
        substitution_costs={("A", "C"): 0.1, ("AB", "C"): 0.5},
        deletion_costs={"B": 0.2},
    )
    ops = wl.explain("AB", "C", filter_matches=True)
    assert ops == [
        EditOperation("substitute", "A", "C", 0.1),
        EditOperation("delete", "B", None, 0.2),
    ]


def test_explain_direct_substitution_wins_over_mixed_chain() -> None:
    """
    Test that a cheaper direct A->B substitution is not expanded into A->AAA->B.
    """
    wl = WeightedLevenshtein(
        substitution_costs={("AAA", "B"): 0.1, ("A", "B"): 0.4},
        insertion_costs={"A": 0.2},
    )
    ops = wl.explain("A", "B", filter_matches=True)
    assert ops == [
        EditOperation("substitute", "A", "B", 0.4),
    ]


def test_explain_effective_deletion_with_insertion_then_deletion() -> None:
    """
    Test that AC->ABC->C is expanded as insert(B), delete(AB), match(C).
    """
    wl = WeightedLevenshtein(
        insertion_costs={"B": 0.1},
        deletion_costs={"AB": 0.0},
    )
    ops = wl.explain("AC", "C", filter_matches=False)
    assert ops == [
        EditOperation("insert", None, "B", 0.1),
        EditOperation("delete", "AB", None, 0.0),
        EditOperation("match", "C", "C", 0.0),
    ]
