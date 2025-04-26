import pytest
from ocr_stringdist import weighted_levenshtein_distance


@pytest.mark.parametrize(
    ["source", "target", "sub_costs", "ins_costs", "del_costs", "expected"],
    [
        # Basic substitution test
        (
            "abc",
            "bbc",
            {("a", "b"): 0.1},
            {},
            {},
            0.1,
        ),
        # Test with custom insertion and deletion costs
        (
            "aby",
            "abx",
            {},
            {"x": 0.3},
            {"y": 0.4},
            0.7,
        ),  # delete 'y' (0.4) + insert 'x' (0.3)
        # Test that forces deletion + insertion over substitution
        (
            "axc",
            "abc",
            {},
            {"b": 0.3},
            {"x": 0.5},
            0.8,
        ),  # delete 'x' (0.5) + insert 'b' (0.3)
        # Empty string tests
        (
            "",
            "",
            {},
            {},
            {},
            0.0,
        ),  # Empty strings have zero distance
        (
            "",
            "abc",
            {},
            {},
            {},
            3.0,
        ),  # Insert 'a', 'b', 'c' with default cost 1.0 each
        (
            "abc",
            "",
            {},
            {},
            {},
            3.0,
        ),  # Delete 'a', 'b', 'c' with default cost 1.0 each
        # Custom costs for empty string operations
        (
            "",
            "abc",
            {},
            {"a": 0.2, "b": 0.3, "c": 0.4},
            {},
            0.9,
        ),  # Insert 'a' (0.2) + 'b' (0.3) + 'c' (0.4)
        (
            "abc",
            "",
            {},
            {},
            {"a": 0.5, "b": 0.6, "c": 0.7},
            1.8,
        ),  # Delete 'a' (0.5) + 'b' (0.6) + 'c' (0.7)
        # Unicode handling
        (
            "café",
            "cafe",
            {},
            {},
            {},
            1.0,
        ),  # Substitute 'é' with 'e' with default cost 1.0
        (
            "hi 😊",
            "hi",
            {},
            {},
            {},
            2.0,
        ),  # Delete ' ' and '😊' with default cost 1.0 each
        (
            "hi 😊",
            "hi",
            {},
            {},
            {" ": 1.0, "😊": 0.5},
            1.5,
        ),  # Delete ' ' (1.0) and '😊' (0.5)
        # Multi-character substitutions
        (
            "this",
            "Tis",
            {("th", "T"): 0.2},
            {},
            {},
            0.2,
        ),  # Substitute "th" with "T" with cost 0.2
        (
            "singing",
            "singin'",
            {("ing", "in'"): 0.3},
            {},
            {},
            0.3,
        ),  # Substitute "ing" with "in'" with cost 0.3
        (
            "go",
            "gou",
            {("o", "ou"): 0.1},
            {},
            {},
            0.1,
        ),  # Substitute "o" with "ou" with cost 0.1
        (
            "thinking",
            "Tinkin'",
            {("th", "T"): 0.2, ("ing", "in'"): 0.3},
            {},
            {},
            0.5,
        ),  # Substitute "th" with "T" (0.2) + "ing" with "in'" (0.3)
        # Multi-character substitutions for different segments
        (
            "abcde",
            "xyzuv",
            {("abc", "xyz"): 0.1, ("de", "uv"): 0.2},
            {},
            {},
            0.3,
        ),  # Substitute "abc"->"xyz" (0.1) + "de"->"uv" (0.2)
        # Multi-character insertion tests
        (
            "x",
            "xab",
            {},
            {"ab": 0.3},
            {},
            0.3,
        ),  # Insert "ab" (0.3)
        (
            "hello",
            "helloxyz",
            {},
            {"xyz": 0.2},
            {},
            0.2,
        ),  # Insert "xyz" (0.2)
        (
            "start",
            "startabc123",
            {},
            {"abc": 0.4, "123": 0.5},
            {},
            0.9,
        ),  # Insert "abc" (0.4) + "123" (0.5)
        # Multi-character deletion tests
        (
            "xcd",
            "x",
            {},
            {},
            {"cd": 0.4},
            0.4,
        ),  # Delete "cd" (0.4)
        (
            "testxyz",
            "test",
            {},
            {},
            {"xyz": 0.6},
            0.6,
        ),  # Delete "xyz" (0.6)
        (
            "hello789world",
            "helloworld",
            {},
            {},
            {"789": 0.7},
            0.7,
        ),  # Delete "789" (0.7)
        # Combined multi-character operations
        (
            "aef",
            "aab",
            {},
            {"ab": 0.3},
            {"ef": 0.5},
            0.8,
        ),  # Delete "ef" (0.5) + insert "ab" (0.3)
        (
            "a789b",
            "axyzb",
            {},
            {"xyz": 0.2},
            {"789": 0.6},
            0.8,
        ),  # Delete "789" (0.6) + insert "xyz" (0.2)
        (
            "start123end",
            "startabcend",
            {("123", "abc"): 0.1},
            {},
            {},
            0.1,
        ),  # Substitute "123" with "abc" (0.1)
    ],
)
def test_custom_weighted_levenshtein(
    source: str,
    target: str,
    sub_costs: dict[tuple[str, str], float],
    ins_costs: dict[str, float],
    del_costs: dict[str, float],
    expected: float,
) -> None:
    """Test the weighted_levenshtein_distance function with various parameters."""
    assert weighted_levenshtein_distance(
        source,
        target,
        substitution_costs=sub_costs,
        insertion_costs=ins_costs,
        deletion_costs=del_costs,
    ) == pytest.approx(expected)


def test_mixed_operations() -> None:
    """
    Test the weighted_levenshtein_distance with mixed operations (insertion, deletion, substitution)
    """
    sub_costs = {("a", "A"): 0.1, ("b", "B"): 0.2}
    ins_costs = {"x": 0.3, "y": 0.4}
    del_costs = {"m": 0.5, "n": 0.6}

    result = weighted_levenshtein_distance(
        "abmn",
        "ABxy",
        substitution_costs=sub_costs,
        insertion_costs=ins_costs,
        deletion_costs=del_costs,
    )

    # Calculate the expected result based on the Rust implementation
    expected = 2.1  # Substitute 'a'→'A' (0.1) + 'b'→'B' (0.2) + delete 'm' (0.5) +
    # delete 'n' (0.6) + insert 'x' (0.3) + insert 'y' (0.4)
    assert result == pytest.approx(expected)


def test_complex_ocr_scenarios() -> None:
    """
    Test complex OCR scenarios that mix substitution, insertion, and deletion with custom costs
    """
    # Common OCR confusion patterns
    sub_costs = {
        ("rn", "m"): 0.1,  # 'rn' often misread as 'm'
        ("cl", "d"): 0.2,  # 'cl' often misread as 'd'
        ("O", "0"): 0.3,  # 'O' often misread as '0'
        ("l", "1"): 0.2,  # 'l' often misread as '1'
        ("h", "In"): 0.25,  # 'h' often misread as 'In'
        ("vv", "w"): 0.15,  # 'vv' often misread as 'w'
        ("nn", "m"): 0.2,  # 'nn' often misread as 'm'
    }

    # Characters that might be erroneously inserted
    ins_costs = {
        " ": 0.1,  # Extra spaces are common
        "-": 0.2,  # Extra hyphens are common
        ".": 0.3,  # Extra periods are common
    }

    # Characters that might be erroneously deleted
    del_costs = {
        " ": 0.4,  # Missing spaces are common
        "i": 0.5,  # Missing 'i's are common (thin character)
        "l": 0.5,  # Missing 'l's are common (thin character)
    }

    # Test a complex sentence with multiple substitution patterns
    original = "The man ran down the hill at 10 km/h."
    ocr_result = "Tine rnan ram dovvn tine Ini11 at 1O krn/In."

    # Calculate with custom costs
    distance = weighted_levenshtein_distance(
        original,
        ocr_result,
        substitution_costs=sub_costs,
        insertion_costs=ins_costs,
        deletion_costs=del_costs,
    )

    # Calculate with default costs for comparison
    standard_distance = weighted_levenshtein_distance(
        original,
        ocr_result,
    )

    # The custom distance should be less than the standard distance
    assert distance < standard_distance

    # Calculate with just substitution costs for comparison
    sub_only_distance = weighted_levenshtein_distance(
        original,
        ocr_result,
        substitution_costs=sub_costs,
    )

    # Verify that adding insertion and deletion costs further improves (reduces) the distance
    assert distance <= sub_only_distance


# Tests from previous test_ocr_stringdist.py file


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
    assert weighted_levenshtein_distance(s1, s2, substitution_costs=cost_map) == pytest.approx(
        expected
    )


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

    distance = weighted_levenshtein_distance(original, ocr_result, substitution_costs=ocr_cost_map)
    standard_distance = weighted_levenshtein_distance(original, ocr_result, substitution_costs={})
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
        s1, s2, substitution_costs=asymmetric_cost_map, symmetric_substitution=False
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
    assert weighted_levenshtein_distance(
        s1, s2, substitution_costs=nested_cost_map
    ) == pytest.approx(expected)


def test_negative_default_cost() -> None:
    invalid_cost = -1.0
    with pytest.raises(
        ValueError, match=f"Default cost must be non-negative, got value: {invalid_cost:.0f}"
    ):
        weighted_levenshtein_distance("a", "b", default_substitution_cost=invalid_cost)


@pytest.mark.parametrize(
    ["s1", "s2", "insertion_costs", "expected"],
    [
        ("ABD", "ABCD", {}, 1),  # without cost map
        ("ABD", "ABCD", {"C": 0.1}, 0.1),  # with cost map
        (
            "ACEG",
            "ABCDEFG!",
            {"B": 0.1, "D": 0.2, "!": 0.0},
            1.3,
        ),  # muliple insertions with different costs (B,D,F,!)
    ],
)
def test_weighted_levenshtein_distance_with_insertion(
    s1: str, s2: str, insertion_costs: dict[str, float], expected: float
) -> None:
    assert weighted_levenshtein_distance(s1, s2, insertion_costs=insertion_costs) == pytest.approx(
        expected
    )


@pytest.mark.parametrize(
    ["s1", "s2", "ins_costs", "del_costs", "expected"],
    [
        # Multi-character insertion tests
        ("x", "xab", {"ab": 0.3}, {}, 0.3),  # Insert "ab" (0.3)
        ("hello", "helloxyz", {"xyz": 0.2}, {}, 0.2),  # Insert "xyz" (0.2)
        # Multi-character deletion tests
        ("xcd", "x", {}, {"cd": 0.4}, 0.4),  # Delete "cd" (0.4)
        ("testxyz", "test", {}, {"xyz": 0.6}, 0.6),  # Delete "xyz" (0.6)
        # Multiple multi-character operations
        (
            "start",
            "startabc123",
            {"abc": 0.4, "123": 0.5},
            {},
            0.9,
        ),  # Insert "abc" (0.4) + "123" (0.5)
        ("hello789world", "helloworld", {}, {"789": 0.7}, 0.7),  # Delete "789" (0.7)
        # Combined multi-character operations
        ("aef", "aab", {"ab": 0.3}, {"ef": 0.5}, 0.8),  # Delete "ef" (0.5) + insert "ab" (0.3)
        (
            "a789b",
            "axyzb",
            {"xyz": 0.2},
            {"789": 0.6},
            0.8,
        ),  # Delete "789" (0.6) + insert "xyz" (0.2)
    ],
)
def test_multi_character_insertions_and_deletions(
    s1: str, s2: str, ins_costs: dict[str, float], del_costs: dict[str, float], expected: float
) -> None:
    """Test multi-character insertions and deletions."""
    assert weighted_levenshtein_distance(
        s1, s2, insertion_costs=ins_costs, deletion_costs=del_costs
    ) == pytest.approx(expected)


def test_complex_multi_character_operations() -> None:
    """Test a complex scenario with all types of multi-character operations."""
    # Define costs for multi-character operations
    sub_costs = {
        ("abc", "xyz"): 0.1,
        ("123", "789"): 0.2,
        ("hello", "hi"): 0.3,
    }
    ins_costs = {
        "world": 0.4,
        "test": 0.5,
    }
    del_costs = {
        "python": 0.6,
        "code": 0.7,
    }

    # Test with a complex scenario
    s1 = "abc123hellopythoncode"
    s2 = "xyz789hiworld"

    # Calculate with custom costs
    distance = weighted_levenshtein_distance(
        s1,
        s2,
        substitution_costs=sub_costs,
        insertion_costs=ins_costs,
        deletion_costs=del_costs,
    )

    # Calculate expected:
    # - Substitute "abc" with "xyz" (0.1)
    # - Substitute "123" with "789" (0.2)
    # - Substitute "hello" with "hi" (0.3)
    # - Delete "python" (0.6)
    # - Delete "code" (0.7)
    # - Insert "world" (0.4)
    expected = 2.3

    assert distance == pytest.approx(expected)

    # Calculate with default costs for comparison
    standard_distance = weighted_levenshtein_distance(s1, s2)

    # The custom distance should be less than the standard distance
    assert distance < standard_distance
