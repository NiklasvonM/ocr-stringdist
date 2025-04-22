import pytest
from ocr_stringdist import weighted_levenshtein_distance


@pytest.mark.parametrize(
    ["s1", "s2", "cost_map", "expected"],
    [
        ("a", "b", {}, 1.0),
        ("a", "b", {("a", "b"): 0.5}, 0.5),
        ("a", "b", {("a", "c"): 0.5}, 1.0),
        ("h", "In", {("h", "In"): 0.5}, 0.5),
        ("h", "In", {}, 2.0),
        # Multiple character substitutions in the same string
        ("hello", "Inello", {("h", "In"): 0.2}, 0.2),
        ("hello", "Ine11o", {("h", "In"): 0.2, ("l", "1"): 0.3}, 0.8),
        ("corner", "comer", {("rn", "m"): 0.1}, 0.1),
        ("class", "dass", {("cl", "d"): 0.2}, 0.2),
        # Test substitutions at word boundaries
        ("rnat", "mat", {("rn", "m"): 0.1}, 0.1),
        ("burn", "bum", {("rn", "m"): 0.1}, 0.1),
        # Test basic Levenshtein distance
        ("kitten", "sitting", {}, 3.0),
        # Test with Unicode characters
        ("café", "coffee", {}, 4.0),
        # Test with empty strings
        ("", "abc", {}, 3.0),
        ("abc", "", {}, 3.0),
        ("", "", {}, 0.0),
        # Non-Latin characters
        ("↑", "个", {("↑", "个"): 0.5}, 0.5),
        ("?=↑", "第二个", {("↑", "个"): 0.5, ("二", "="): 0.5}, 2.0),
        ("이탈리", "OI탈리", {("이", "OI"): 0.5}, 0.5),
    ],
)
def test_weighted_levenshtein_distance(
    s1: str, s2: str, cost_map: dict[tuple[str, str], float], expected: float
) -> None:
    assert weighted_levenshtein_distance(s1, s2, cost_map=cost_map) == pytest.approx(expected)


def test_complex_ocr_substitutions() -> None:
    """Test more complex OCR-specific substitution patterns."""
    # Common OCR confusion patterns
    ocr_cost_map = {
        ("rn", "m"): 0.1,
        ("cl", "d"): 0.2,
        ("O", "0"): 0.3,
        ("l", "1"): 0.2,
        ("h", "In"): 0.25,
        ("vv", "w"): 0.15,
        ("nn", "m"): 0.2,
    }

    # Test a sentence with multiple substitution patterns
    original = "The man ran down the hill at 10 km/h."
    ocr_result = "Tine rnan ram dovvn tine Ini11 at 1O krn/In."

    distance = weighted_levenshtein_distance(original, ocr_result, cost_map=ocr_cost_map)
    standard_distance = weighted_levenshtein_distance(original, ocr_result, cost_map={})
    assert standard_distance > distance


@pytest.mark.parametrize(
    ["s1", "s2", "expected"],
    [
        ("50", "SO", 0.3),
        ("SO", "50", 1.1),
        ("STOP50", "5TOP50", 0.6),
        ("5TOP50", "STOP50", 0.2),
    ],
)
def test_asymmetric_substitution_costs(s1: str, s2: str, expected: float) -> None:
    asymmetric_cost_map = {
        ("0", "O"): 0.1,
        ("O", "0"): 0.5,
        ("5", "S"): 0.2,
        ("S", "5"): 0.6,
    }
    assert weighted_levenshtein_distance(
        s1, s2, cost_map=asymmetric_cost_map, symmetric=False
    ) == pytest.approx(expected)


@pytest.mark.parametrize(
    ["s1", "s2", "expected"],
    [
        ("a", "b", 0.1),
        ("ab", "c", 0.2),
        ("abc", "d", 0.3),
        ("xabcy", "xcy", 1.2),  # ab -> c, delete other c
        ("xabcy", "xdy", 0.3),
        ("xabcy", "xby", 2.0),
    ],
)
def test_nested_substitution_patterns(s1: str, s2: str, expected: float) -> None:
    nested_cost_map = {
        ("a", "b"): 0.1,
        ("b", "a"): 0.1,
        ("ab", "c"): 0.2,
        ("c", "ab"): 0.2,
        ("abc", "d"): 0.3,
        ("d", "abc"): 0.3,
    }
    assert weighted_levenshtein_distance(s1, s2, cost_map=nested_cost_map) == pytest.approx(
        expected
    )


def test_negative_default_cost() -> None:
    with pytest.raises(ValueError, match="default_cost must be non-negative"):
        weighted_levenshtein_distance("a", "b", default_cost=-1.0)
