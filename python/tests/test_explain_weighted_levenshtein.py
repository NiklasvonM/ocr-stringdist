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


# Closure-flat explain tests
#
# After ``transitive_closure()``, the underlying chain that produced an
# effective cost is no longer preserved. ``explain()`` returns a single
# substitution / insertion / deletion at the effective cost. These tests
# verify the flat output and that the total cost equals ``distance()``.


def _flat_explain_assertions(
    wl_closed: WeightedLevenshtein, s1: str, s2: str, expected_distance: float
) -> list[EditOperation]:
    ops = wl_closed.explain(s1, s2)
    assert sum(op.cost for op in ops) == pytest.approx(expected_distance)
    assert wl_closed.distance(s1, s2) == pytest.approx(expected_distance)
    return ops


def _assert_ops_equal(actual: list[EditOperation], expected: list[EditOperation]) -> None:
    """Compare op sequences with float-tolerant cost equality."""
    assert len(actual) == len(expected), f"length mismatch: {actual} vs {expected}"
    for a, e in zip(actual, expected):
        assert a.op_type == e.op_type
        assert a.source_token == e.source_token
        assert a.target_token == e.target_token
        assert a.cost == pytest.approx(e.cost)


def test_explain_transitive_deletion_chain_after_closure() -> None:
    """After closure, '06' -> '0' is one effective deletion of '6' at 0.51."""
    wl = WeightedLevenshtein(
        substitution_costs={("6", "G"): 0.5},
        deletion_costs={"G": 0.01},
        symmetric_substitution=False,
    ).transitive_closure()
    ops = _flat_explain_assertions(wl, "06", "0", 0.51)
    assert ops == [EditOperation("delete", "6", None, 0.51)]


def test_explain_transitive_substitution_chain_after_closure() -> None:
    """After closure, 'a' -> 'c' is one effective substitution at 0.2."""
    wl = WeightedLevenshtein(
        substitution_costs={("a", "b"): 0.1, ("b", "c"): 0.1},
        symmetric_substitution=False,
    ).transitive_closure()
    ops = _flat_explain_assertions(wl, "a", "c", 0.2)
    assert ops == [EditOperation("substitute", "a", "c", 0.2)]


def test_explain_transitive_insertion_chain_after_closure() -> None:
    """After closure, inserting 'y' is one effective insertion at 0.3."""
    wl = WeightedLevenshtein(
        substitution_costs={("x", "y"): 0.2},
        insertion_costs={"x": 0.1},
        symmetric_substitution=False,
    ).transitive_closure()
    ops = _flat_explain_assertions(wl, "a", "ay", 0.3)
    assert ops == [
        EditOperation("insert", None, "y", 0.3),
    ]


def test_explain_chain_with_expensive_direct_substitution_after_closure() -> None:
    """A->B with cheaper A->AAA->B chain becomes a single sub at 0.5."""
    wl = WeightedLevenshtein(
        substitution_costs={("AAA", "B"): 0.1, ("A", "B"): 0.6},
        insertion_costs={"A": 0.2},
    ).transitive_closure()
    ops = _flat_explain_assertions(wl, "A", "B", 0.5)
    assert ops == [EditOperation("substitute", "A", "B", 0.5)]


def test_explain_mixed_substitution_path_with_deletion_after_closure() -> None:
    """AB->C: closure prefers sub(A,C) + del(B) = 0.3 over the direct AB->C = 0.5."""
    wl = WeightedLevenshtein(
        substitution_costs={("A", "C"): 0.1, ("AB", "C"): 0.5},
        deletion_costs={"B": 0.2},
    ).transitive_closure()
    ops = _flat_explain_assertions(wl, "AB", "C", 0.3)
    assert ops == [
        EditOperation("substitute", "A", "C", 0.1),
        EditOperation("delete", "B", None, 0.2),
    ]


def test_explain_direct_substitution_wins_over_chain_after_closure() -> None:
    """A direct A->B at 0.4 beats A->AAA->B at 0.5; effective cost stays 0.4."""
    wl = WeightedLevenshtein(
        substitution_costs={("AAA", "B"): 0.1, ("A", "B"): 0.4},
        insertion_costs={"A": 0.2},
    ).transitive_closure()
    ops = _flat_explain_assertions(wl, "A", "B", 0.4)
    assert ops == [EditOperation("substitute", "A", "B", 0.4)]


def test_explain_effective_deletion_with_insertion_then_deletion_after_closure() -> None:
    """AC -> C via insert(B)+del(AB)=0.1 becomes a single effective del('A') at 0.1."""
    wl = WeightedLevenshtein(
        insertion_costs={"B": 0.1},
        deletion_costs={"AB": 0.0},
    ).transitive_closure()
    ops = _flat_explain_assertions(wl, "AC", "C", 0.1)
    # The single effective op may be a del('A'), or another route at the same cost.
    # Assert the explicit identity to lock down the canonical form:
    assert ops == [EditOperation("delete", "A", None, 0.1)]


def test_explain_insert_delete_substitute_chain_after_closure() -> None:
    """ADC -> Z via del(D)+ins(B)+sub(ABC,Z) becomes a single sub at 0.3."""
    wl = WeightedLevenshtein(
        substitution_costs={("ABC", "Z"): 0.1},
        insertion_costs={"B": 0.1},
        deletion_costs={"D": 0.1},
    ).transitive_closure()
    ops = _flat_explain_assertions(wl, "ADC", "Z", 0.3)
    assert ops == [EditOperation("substitute", "ADC", "Z", 0.3)]


def test_explain_single_char_composed_substitution_chain_after_closure() -> None:
    """X -> Z via ins(AB)+sub(XAB,Z) becomes a single sub at 0.3."""
    wl = WeightedLevenshtein(
        substitution_costs={("XAB", "Z"): 0.1},
        insertion_costs={"AB": 0.2},
    ).transitive_closure()
    ops = _flat_explain_assertions(wl, "X", "Z", 0.3)
    assert ops == [EditOperation("substitute", "X", "Z", 0.3)]
