from collections.abc import Iterable

import pytest
from ocr_stringdist import WeightedLevenshtein, find_best_candidate


@pytest.mark.parametrize(
    ["s", "candidates", "cost_map", "expected_match", "expected_distance"],
    [
        ("", {"a"}, {}, "a", 1.0),
        ("R0BERT", {"ROBERT", "ALFRED", "Robert"}, {("0", "O"): 0.5}, "ROBERT", 0.5),
    ],
)
def test_find_best_candidate(
    s: str,
    candidates: Iterable[str],
    cost_map: dict[tuple[str, str], float],
    expected_match: str,
    expected_distance: float,
) -> None:
    wl = WeightedLevenshtein(substitution_costs=cost_map)
    actual_match, actual_distance = find_best_candidate(s, candidates, wl.distance)
    assert actual_match == expected_match
    assert actual_distance == pytest.approx(expected_distance)


def test_find_best_candidate_early_return() -> None:
    # Exact match is present, but a good enough match is found earlier.
    wl = WeightedLevenshtein()
    result = find_best_candidate("HANNA", ["ANNA", "HANNA"], wl.distance, early_return_value=2.0)
    assert result == ("ANNA", 1.0)


def test_find_best_candidate_maximize() -> None:
    def similarity(s1: str, s2: str) -> float:
        return float(s1 == s2)

    assert find_best_candidate("one", {"one", "two"}, similarity, minimize=False) == ("one", 1.0)
