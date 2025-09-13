import math
from collections import defaultdict

import pytest
from ocr_stringdist.edit_operation import EditOperation
from ocr_stringdist.learner import Learner, negative_log_likelihood
from ocr_stringdist.levenshtein import WeightedLevenshtein


@pytest.fixture
def learner() -> Learner:
    """Provides a default Learner instance for tests."""
    return Learner()


def test_learner_initialization(learner: Learner) -> None:
    """Tests the default state of a new Learner instance."""
    assert learner._smoothing_k == 1.0
    assert learner._cost_function is negative_log_likelihood
    assert learner.counts is None
    assert learner.vocab_size is None


def test_learner_builder_pattern(learner: Learner) -> None:
    """Tests the chaining of builder methods."""

    def custom_cost_func(p: float) -> float:
        return 1.0 - p

    learner = learner.with_smoothing(2.5).with_cost_function(custom_cost_func)

    assert learner._smoothing_k == 2.5
    assert learner._cost_function is custom_cost_func


@pytest.mark.parametrize("k", [0, -1.0, -100])
def test_with_smoothing_invalid_k_raises_error(learner: Learner, k: float) -> None:
    """Tests that a non-positive smoothing parameter k raises a ValueError."""
    with pytest.raises(ValueError, match="Smoothing parameter k must be positive."):
        learner.with_smoothing(k)


def test_negative_log_likelihood_invalid_prob_raises_error() -> None:
    """Tests that a non-positive probability raises a ValueError."""
    with pytest.raises(ValueError, match="Probability must be positive"):
        negative_log_likelihood(0.0)
    with pytest.raises(ValueError, match="Probability must be positive"):
        negative_log_likelihood(-0.5)


def test_tally_operations() -> None:
    """Tests the counting of edit operations."""
    operations = [
        EditOperation("match", "a", "a", cost=0.0),
        EditOperation("substitute", "b", "c", cost=1.0),
        EditOperation("substitute", "b", "c", cost=1.0),
        EditOperation("delete", "d", None, cost=1.0),
        EditOperation("insert", None, "e", cost=1.0),
    ]
    counts = Learner()._tally_operations(operations)

    expected_substitutions = defaultdict(int, {("b", "c"): 2})
    expected_insertions = defaultdict(int, {"e": 1})
    expected_deletions = defaultdict(int, {"d": 1})
    expected_source_chars = defaultdict(int, {"a": 1, "b": 2, "d": 1})

    assert counts.substitutions == expected_substitutions
    assert counts.insertions == expected_insertions
    assert counts.deletions == expected_deletions
    assert counts.source_chars == expected_source_chars
    assert counts.vocab == {"a", "b", "c", "d", "e"}


@pytest.mark.parametrize(
    "op",
    [
        EditOperation("substitute", None, "c", cost=1.0),
        EditOperation("substitute", "b", None, cost=1.0),
        EditOperation("delete", None, None, cost=1.0),
        EditOperation("insert", None, None, cost=1.0),
        EditOperation("match", None, "a", cost=1.0),
    ],
)
def test_tally_operations_raises_type_error_on_none(learner: Learner, op: EditOperation) -> None:
    """Tests that _tally_operations raises TypeError for invalid operations."""
    with pytest.raises(ValueError, match="cannot be None"):
        learner._tally_operations([op])


def test_fit_simple_substitution(learner: Learner) -> None:
    """Tests fitting on a simple substitution case."""
    # Data: "a" is misread as "b" once.
    data = [("b", "a")]
    wl = learner.fit(data)

    # --- Manual Calculation for ('b' -> 'a') ---

    # 1. Tally Counts:
    #    - The single operation is substitute('b' -> 'a').
    #    - counts.substitutions: {('b', 'a'): 1}
    #    - counts.source_chars: {'b': 1}
    #    - vocab: {'a', 'b'}, so V = 2
    #    - k = 1.0 (default smoothing)

    # 2. Context for the operation:
    #    - Source character is 'b'.
    #    - N_b (count of source char 'b') = 1.
    #    - Denominator = N_b + k*V = 1 + 1*2 = 3.

    # 3. Raw cost of the observed event ('b' -> 'a'):
    #    - P_observed = (count + k) / Denominator = (1 + 1) / 3 = 2/3.
    #    - Cost_observed = -log(2/3).

    # 4. Context-specific scaling factor for source 'b':
    #    - This is the cost of an unseen event from 'b'.
    #    - P_unseen = k / Denominator = 1 / 3.
    #    - ScalingFactor_b = -log(1/3).

    # 5. Final scaled cost:
    #    - FinalCost = Cost_observed / ScalingFactor_b
    expected_cost = -math.log(2 / 3) / -math.log(1 / 3)

    # Ensure that costs were learned only for the observed operation
    assert len(wl.substitution_costs) == 1
    actual_cost = wl.substitution_costs[("b", "a")]

    assert actual_cost == pytest.approx(expected_cost)
    assert actual_cost < 1.0
    assert wl.default_substitution_cost == 1.0


def test_fit_with_insertion_and_deletion() -> None:
    """Tests fitting on data with insertions and deletions."""
    data = [
        ("ac", "a"),  # delete 'c'
        ("b", "db"),  # insert 'd'
    ]
    learner = Learner().with_smoothing(0.5)
    wl = learner.fit(data)

    assert wl.deletion_costs["c"] < 1.0
    assert wl.insertion_costs["d"] < 1.0
    assert wl.default_insertion_cost == 1.0
    assert wl.default_deletion_cost == 1.0


def test_fit_no_errors(learner: Learner) -> None:
    """Tests fitting on data with no errors, costs should be high (near default)."""
    data = [("a", "a"), ("b", "b")]
    wl = learner.fit(data)

    # Manual calculation: No error counts, only smoothed probabilities
    # counts.source_chars: {'a': 1, 'b': 1}
    # vocab: {'a', 'b'}, V = 2
    # k = 1.0
    #
    # Consider a hypothetical substitution 'a' -> 'x' (unseen)
    # P(sub 'a'->'x') = (0 + k) / (source_count_a + k*V) = 1 / (1 + 1*2) = 1/3
    # Cost(sub 'a'->'x') = -log(1/3)
    # Scaling factor = -log(1/2)
    # This calculation is for an UNSEEN substitution, which should be the default cost.
    # The default cost is normalized to 1.0, so the specific value does not matter here.
    # What matters is that no specific, lower costs are learned.
    assert wl.substitution_costs == {}
    assert wl.insertion_costs == {}
    assert wl.deletion_costs == {}
    assert wl.default_substitution_cost == 1.0


def test_fit_empty_data(learner: Learner) -> None:
    """Tests that fitting on no data returns an unweighted Levenshtein instance."""
    wl = learner.fit([])
    assert wl == WeightedLevenshtein.unweighted()


def test_fit_identical_strings(learner: Learner) -> None:
    """Tests fitting with identical strings, which should produce an empty cost map."""
    data = [("hello", "hello"), ("world", "world")]
    wl = learner.fit(data)
    assert not wl.substitution_costs
    assert not wl.insertion_costs
    assert not wl.deletion_costs
    assert learner.vocab_size == len(set("helloworld"))
