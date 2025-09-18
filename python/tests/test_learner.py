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
    assert learner.counts is None
    assert learner.vocab_size is None


def test_learner_builder_pattern(learner: Learner) -> None:
    """Tests the chaining of builder methods."""

    learner = learner.with_smoothing(2.5)

    assert learner._smoothing_k == 2.5


@pytest.mark.parametrize("k", [-1.0, -100])
def test_with_smoothing_invalid_k_raises_error(learner: Learner, k: float) -> None:
    """Tests that a negative smoothing parameter k raises a ValueError."""
    with pytest.raises(ValueError, match="Smoothing parameter k must be non-negative."):
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


def test_monotonicity_of_substitution_costs(learner: Learner) -> None:
    previous_cost = 1.0
    for i in range(10):
        data = [("a" * (i + 1), "b" * (i + 1))]
        wl = learner.fit(data)
        current_cost = wl.substitution_costs.get(("a", "b"), 1.0)
        assert current_cost < previous_cost, (
            f"Cost did not decrease: {current_cost} > {previous_cost}"
        )
        previous_cost = current_cost


def test_monotonicity_of_insertion_costs(learner: Learner) -> None:
    previous_cost = 1.0
    for i in range(10):
        data = [("", "b" * (i + 1))]
        wl = learner.fit(data)
        current_cost = wl.insertion_costs.get("b", 1.0)
        assert current_cost < previous_cost, (
            f"Cost did not decrease: {current_cost} > {previous_cost}"
        )
        previous_cost = current_cost


def test_monotonicity_of_deletion_costs(learner: Learner) -> None:
    previous_cost = 1.0
    for i in range(10):
        data = [("a" * (i + 1), "")]
        wl = learner.fit(data)
        current_cost = wl.deletion_costs.get("a", 1.0)
        assert current_cost < previous_cost, (
            f"Cost did not decrease: {current_cost} > {previous_cost}"
        )
        previous_cost = current_cost


def test_maximum_likelihood_estimation(learner: Learner) -> None:
    data = [("a", "b"), ("", "c"), ("d", "")]
    wl = learner.with_smoothing(0.0).fit(data)
    # Every a should be a b in the train data, so cost should be 0.
    assert wl.substitution_costs.get(("a", "b")) == 0.0
    # Every d should be deleted in the train data, so cost should be 0.
    assert wl.deletion_costs.get("d") == 0.0
    # Insertion cost is not 0 because we don't always insert a 'c'.
    assert wl.insertion_costs.get("c", 1.0) < 1.0


@pytest.mark.parametrize("share", [0.0, 0.1, 0.5, 0.9, 1.0])
def test_asymptotic_substitution_costs(learner: Learner, share: float) -> None:
    n_data_points = 100_000
    n_errors = int(n_data_points * share)
    data = [("a", "b")] * n_errors + [("a", "a")] * (n_data_points - n_errors)
    wl = learner.fit(data)
    expected_cost = -math.log(share) / math.log(n_data_points) if share > 0 else 1.0
    assert wl.substitution_costs.get(("a", "b"), 1.0) == pytest.approx(expected_cost, rel=1e-2)


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
