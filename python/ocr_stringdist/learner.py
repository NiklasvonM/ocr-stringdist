import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Optional, Self

if TYPE_CHECKING:
    from .edit_operation import EditOperation
    from .levenshtein import WeightedLevenshtein

CostFunction = Callable[[float], float]


def negative_log_likelihood(probability: float) -> float:
    """Standard cost function based on information theory. Common errors get low cost."""
    if probability <= 0.0:
        raise ValueError("Probability must be positive to compute negative log likelihood.")
    return -math.log(probability)


@dataclass
class TallyCounts:
    substitutions: defaultdict[tuple[str, str], int] = field(
        default_factory=lambda: defaultdict(int)
    )
    insertions: defaultdict[str, int] = field(default_factory=lambda: defaultdict(int))
    deletions: defaultdict[str, int] = field(default_factory=lambda: defaultdict(int))
    source_chars: defaultdict[str, int] = field(default_factory=lambda: defaultdict(int))
    vocab: set[str] = field(default_factory=set)


@dataclass
class _Costs:
    substitutions: dict[tuple[str, str], float]
    insertions: dict[str, float]
    deletions: dict[str, float]


class Learner:
    """
    Configures and executes the process of learning Levenshtein costs from data.

    This class uses a builder pattern, allowing chaining configuration methods
    before running the final calculation with .fit().

    Usage example:

    .. code-block:: python

        from ocr_stringdist.learner import Learner

        data = [
            ("Hell0", "Hello"),
        ]
        learner = Learner().with_smoothing(1.0)
        wl = learner.fit(data) # Substitution 0 -> o learned with cost < 1.0
    """

    # Configuration parameters
    _cost_function: CostFunction
    _smoothing_k: float

    # These attributes are set during fitting
    counts: Optional[TallyCounts] = None
    vocab_size: Optional[int] = None

    def __init__(self) -> None:
        self._cost_function = negative_log_likelihood
        self._smoothing_k = 1.0

    def with_cost_function(self, cost_function: CostFunction) -> Self:
        """
        Sets a custom function to convert probabilities to costs.

        :param cost_function: A callable that takes a float (probability)
                              and returns a float (cost).
                              Is negative log likelihood unless overridden.
        :return: The Learner instance for method chaining.
        """
        self._cost_function = cost_function
        return self

    def with_smoothing(self, k: float) -> Self:
        r"""
        Sets the smoothing parameter `k`.

        This parameter controls how strongly the model defaults to uniform probabilities.

        :param k: The smoothing factor, which must be a positive number.
        :return: The Learner instance for method chaining.

        Notes
        -----
        **Conceptual Framework**

        Additive smoothing works by adding a "pseudo-count" `k` to every possible
        event before calculating probabilities. This effectively pretends that
        every possible substitution, insertion, and deletion has already been seen
        `k` times.

        - **k = 1.0 (Default):** This is standard **Laplace smoothing**. It is a
          robust choice for most situations and corresponds to adding one
          pseudo-count for every possible event.
        - **0 < k < 1.0:** A smaller `k` is suitable for large and representative
          datasets, indicating higher confidence in the observed frequencies.
        - **k > 1.0:** A larger `k` is useful for small or noisy datasets. It
          regularizes the model by pulling the learned probabilities closer
          to a uniform distribution.

        **Bayesian Interpretation**

        From a Bayesian perspective, `k` serves as the concentration parameter,
        :math:`\alpha`, of a **symmetric Dirichlet prior distribution**.
        This distribution acts as the conjugate prior for the
        multinomial distribution of the observed error counts.

        The smoothed probability of an event `i` is the posterior expectation of
        the Dirichlet-multinomial model:

        .. math::

            P(\text{event}_i) = \frac{c_i + k}{N + k \cdot V}

        Where:
            - :math:`c_i` is the observed count of event :math:`i`.
            - :math:`N` is the total number of observations in the given context
              (e.g., the total count of a specific source character).
            - :math:`V` is the vocabulary size (the number of possible unique events).
            - :math:`k` is the smoothing parameter, representing the strength of the prior.
        """
        if k <= 0:
            raise ValueError("Smoothing parameter k must be positive.")
        self._smoothing_k = k
        return self

    def _tally_operations(self, operations: Iterable["EditOperation"]) -> TallyCounts:
        """Tally all edit operations."""
        counts = TallyCounts()
        for op in operations:
            if op.source_token is not None:
                counts.vocab.add(op.source_token)
            if op.target_token is not None:
                counts.vocab.add(op.target_token)

            if op.op_type == "substitute":
                if op.source_token is None or op.target_token is None:
                    raise ValueError("Tokens cannot be None for 'substitute'")
                counts.substitutions[(op.source_token, op.target_token)] += 1
                counts.source_chars[op.source_token] += 1
            elif op.op_type == "delete":
                if op.source_token is None:
                    raise ValueError("Source token cannot be None for 'delete'")
                counts.deletions[op.source_token] += 1
                counts.source_chars[op.source_token] += 1
            elif op.op_type == "insert":
                if op.target_token is None:
                    raise ValueError("Target token cannot be None for 'insert'")
                counts.insertions[op.target_token] += 1
            elif op.op_type == "match":
                if op.source_token is None:
                    raise ValueError("Source token cannot be None for 'match'")
                counts.source_chars[op.source_token] += 1
        return counts

    def _calculate_single_scaled_cost(
        self,
        observed_count: int,
        context_total: int,
        vocab_size: int,
    ) -> Optional[float]:
        """Calculates a single scaled cost for an edit operation."""
        denominator = context_total + self._smoothing_k * vocab_size
        if denominator <= 0:
            return None

        # Calculate the cost of an unseen event in this context, used for scaling
        prob_unseen = self._smoothing_k / denominator
        scaling_factor = self._cost_function(prob_unseen)
        if scaling_factor <= 0:
            return None

        # Calculate the cost for the actually observed event
        prob_observed = (observed_count + self._smoothing_k) / denominator
        cost_observed = self._cost_function(prob_observed)

        return cost_observed / scaling_factor

    def _calculate_costs(self, counts: TallyCounts, vocab_size: int) -> _Costs:
        """
        Calculates and scales costs for observed operations using a context-dependent
        scaling factor to ensure the effective default cost is 1.0.
        """

        # Substitutions
        sub_costs: dict[tuple[str, str], float] = {}
        for (source, target), count in counts.substitutions.items():
            source_char_count = counts.source_chars[source]
            cost = self._calculate_single_scaled_cost(count, source_char_count, vocab_size)
            if cost is not None:
                sub_costs[(source, target)] = cost

        # Insertions
        ins_costs: dict[str, float] = {}
        total_chars = sum(counts.source_chars.values())
        for target, count in counts.insertions.items():
            cost = self._calculate_single_scaled_cost(count, total_chars, vocab_size)
            if cost is not None:
                ins_costs[target] = cost

        # Deletions
        del_costs: dict[str, float] = {}
        for source, count in counts.deletions.items():
            source_char_count = counts.source_chars[source]
            cost = self._calculate_single_scaled_cost(count, source_char_count, vocab_size)
            if cost is not None:
                del_costs[source] = cost

        return _Costs(substitutions=sub_costs, insertions=ins_costs, deletions=del_costs)

    def _calculate_operations(self, pairs: Iterable[tuple[str, str]]) -> list["EditOperation"]:
        """Calculate edit operations for all string pairs using unweighted Levenshtein."""
        from .levenshtein import WeightedLevenshtein

        unweighted_lev = WeightedLevenshtein.unweighted()
        all_ops = [
            op
            for ocr_str, truth_str in pairs
            for op in unweighted_lev.explain(ocr_str, truth_str, filter_matches=False)
        ]
        return all_ops

    def fit(self, pairs: Iterable[tuple[str, str]]) -> "WeightedLevenshtein":
        """
        Fits the costs of a WeightedLevenshtein instance to the provided data.

        Note that learning multi-character tokens is not yet supported.

        :param pairs: An iterable of (ocr_string, ground_truth_string) tuples.
        :return: A `WeightedLevenshtein` instance with the learned costs.
        """
        from .levenshtein import WeightedLevenshtein

        all_ops = self._calculate_operations(pairs)
        self.counts = self._tally_operations(all_ops)
        vocab = self.counts.vocab
        self.vocab_size = len(vocab)

        if not self.vocab_size:
            return WeightedLevenshtein.unweighted()

        costs = self._calculate_costs(self.counts, self.vocab_size)

        return WeightedLevenshtein(
            substitution_costs=costs.substitutions,
            insertion_costs=costs.insertions,
            deletion_costs=costs.deletions,
            default_substitution_cost=1.0,
            default_insertion_cost=1.0,
            default_deletion_cost=1.0,
        )
