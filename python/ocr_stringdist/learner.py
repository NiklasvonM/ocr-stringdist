import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from .edit_operation import EditOperation
    from .levenshtein import WeightedLevenshtein

CostFunction = Callable[[float], float]


def negative_log_likelihood(probability: float) -> float:
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

    Example::

        from ocr_stringdist.learner import Learner

        data = [
            ("Hell0", "Hello"),
        ]
        learner = Learner().with_smoothing(1.0)
        wl = learner.fit(data) # Substitution 0 -> o learned with cost < 1.0
    """

    # Configuration parameters
    _smoothing_k: float

    # These attributes are set during fitting
    counts: Optional[TallyCounts] = None
    vocab_size: Optional[int] = None

    def __init__(self) -> None:
        self._smoothing_k = 1.0

    def with_smoothing(self, k: float) -> "Learner":
        r"""
        Sets the smoothing parameter `k`.

        This parameter controls how strongly the model defaults to uniform probabilities
        by adding a "pseudo-count" `k` to every possible event.

        :param k: The smoothing factor, which must be a non-negative number.
        :return: The Learner instance for method chaining.

        Notes
        -----
        - **k > 0 (Recommended):** This is additive smoothing. A value of **k = 1.0**
          is standard **Laplace smoothing**. It ensures that even unseen edit
          operations are assigned a finite, normalized cost between 0 and 1.
        - **k = 0:** This corresponds to **Maximum Likelihood Estimation** without any
          smoothing. The model will have high confidence in the observed frequencies,
          but it will be unable to assign a cost to edit operations not present in
          the training data. The resulting costs are an un-normalized measure of
          "surprisal" and can be greater than 1.0.
        """
        if k < 0:
            raise ValueError("Smoothing parameter k must be non-negative.")
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

    def _calculate_costs(self, counts: TallyCounts, vocab_size: int) -> _Costs:
        sub_costs: dict[tuple[str, str], float] = {}
        ins_costs: dict[str, float] = {}
        del_costs: dict[str, float] = {}
        k = self._smoothing_k

        total_source_chars = sum(counts.source_chars.values())

        if k == 0:
            # Maximum Likelihood Estimation (no smoothing)
            # Cost is the negative log of the raw probability.
            for (source, target), count in counts.substitutions.items():
                total_count = counts.source_chars[source]
                if total_count > 0:
                    sub_costs[(source, target)] = negative_log_likelihood(count / total_count)

            for source, count in counts.deletions.items():
                total_count = counts.source_chars[source]
                if total_count > 0:
                    del_costs[source] = negative_log_likelihood(count / total_count)

            for target, count in counts.insertions.items():
                if total_source_chars > 0:
                    ins_costs[target] = negative_log_likelihood(count / total_source_chars)

            return _Costs(substitutions=sub_costs, insertions=ins_costs, deletions=del_costs)

        # Calculate the normalization ceiling (Z)
        V_errors_sub_del = vocab_size + 1
        max_unseen_cost = 0.0
        if counts.source_chars:
            max_total_count = max(counts.source_chars.values())
            max_unseen_cost = negative_log_likelihood(k) - negative_log_likelihood(
                max_total_count + k * V_errors_sub_del
            )

        unseen_insertion_cost = negative_log_likelihood(k) - negative_log_likelihood(
            total_source_chars + k * vocab_size
        )
        if unseen_insertion_cost > max_unseen_cost:
            max_unseen_cost = unseen_insertion_cost

        normalization_ceiling = max_unseen_cost if max_unseen_cost > 0 else 1.0

        # Calculate final, normalized costs
        for (source, target), count in counts.substitutions.items():
            total_count = counts.source_chars[source]
            prob = (count + k) / (total_count + k * V_errors_sub_del)
            base_cost = negative_log_likelihood(prob)
            sub_costs[(source, target)] = base_cost / normalization_ceiling

        for source, count in counts.deletions.items():
            total_count = counts.source_chars[source]
            prob = (count + k) / (total_count + k * V_errors_sub_del)
            base_cost = negative_log_likelihood(prob)
            del_costs[source] = base_cost / normalization_ceiling

        for target, count in counts.insertions.items():
            prob = (count + k) / (total_source_chars + k * vocab_size)
            base_cost = negative_log_likelihood(prob)
            ins_costs[target] = base_cost / normalization_ceiling

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

        This method analyzes pairs of strings to learn the costs of edit operations
        based on their observed frequencies. The underlying model calculates costs
        based on the principle of relative information cost.

        For a detailed explanation of the methodology, please see the
        :doc:`Cost Learning Model <cost_learning_model>` documentation page.

        :param pairs: An iterable of (ocr_string, ground_truth_string) tuples.
        :return: A `WeightedLevenshtein` instance with the learned costs.
        """
        from .levenshtein import WeightedLevenshtein

        if not pairs:
            return WeightedLevenshtein.unweighted()

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
