import itertools
import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from .edit_operation import EditOperation
    from .levenshtein import WeightedLevenshtein
    from .protocols import Aligner

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

        This parameter controls how strongly the model defaults to a uniform
        probability distribution by adding a "pseudo-count" of `k` to every
        possible event.

        :param k: The smoothing factor, which must be a non-negative number.
        :return: The Learner instance for method chaining.

        Notes
        -----
        This parameter allows for a continuous transition between two modes:

        - **k > 0 (recommended):** This enables additive smoothing, with `k = 1.0`
          being Laplace smoothing. It regularizes the model by assuming no event is impossible.
          The final costs are a measure of "relative surprisal," normalized by the vocabulary size

        - **k = 0:** This corresponds to a normalized Maximum Likelihood Estimation.
          Probabilities are derived from the raw observed frequencies. The final costs are
          normalized using the same logic as the `k > 0` case, making `k=0` the continuous limit
          of the smoothed model. In this mode, costs can only be calculated for events observed in
          the training data. Unseen events will receive the default cost, regardless of
          the value of `calculate_for_unseen` in :meth:`fit`.
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

    def _calculate_costs(
        self, counts: TallyCounts, vocab: set[str], calculate_for_unseen: bool = False
    ) -> _Costs:
        """
        Calculates the costs for edit operations based on tallied counts.
        """
        sub_costs: dict[tuple[str, str], float] = {}
        ins_costs: dict[str, float] = {}
        del_costs: dict[str, float] = {}
        k = self._smoothing_k

        if k == 0:
            calculate_for_unseen = False

        total_source_chars = sum(counts.source_chars.values())

        # Error space sizes for substitutions/deletions and insertions.
        vocab_size = len(vocab)
        V_sub_del = vocab_size + 1
        V_ins = vocab_size

        # Normalization ceiling Z' = -log(1/V).
        normalization_ceiling = math.log(V_sub_del) if V_sub_del > 1 else 1.0

        # Substitutions
        sub_iterator = (
            itertools.product(vocab, vocab) if calculate_for_unseen else counts.substitutions.keys()
        )
        for source, target in sub_iterator:
            count = counts.substitutions.get((source, target), 0)
            total_count = counts.source_chars.get(source, 0)
            prob = (count + k) / (total_count + k * V_sub_del)
            base_cost = negative_log_likelihood(prob)
            sub_costs[(source, target)] = base_cost / normalization_ceiling

        # Deletions
        del_iterator = vocab if calculate_for_unseen else counts.deletions.keys()
        for source in del_iterator:
            count = counts.deletions.get(source, 0)
            total_count = counts.source_chars.get(source, 0)
            prob = (count + k) / (total_count + k * V_sub_del)
            base_cost = negative_log_likelihood(prob)
            del_costs[source] = base_cost / normalization_ceiling

        # Insertions
        ins_iterator = vocab if calculate_for_unseen else counts.insertions.keys()
        for target in ins_iterator:
            count = counts.insertions.get(target, 0)
            prob = (count + k) / (total_source_chars + k * V_ins)
            base_cost = negative_log_likelihood(prob)
            ins_costs[target] = base_cost / normalization_ceiling

        return _Costs(substitutions=sub_costs, insertions=ins_costs, deletions=del_costs)

    def _calculate_operations(
        self, pairs: Iterable[tuple[str, str]], aligner: "Aligner"
    ) -> list["EditOperation"]:
        """Calculate edit operations for all string pairs using the provided aligner."""

        all_ops = [
            op
            for ocr_str, truth_str in pairs
            for op in aligner.explain(ocr_str, truth_str, filter_matches=False)
        ]
        return all_ops

    def fit(
        self,
        pairs: Iterable[tuple[str, str]],
        *,
        initial_model: "Aligner | None" = None,
        calculate_for_unseen: bool = False,
    ) -> "WeightedLevenshtein":
        """
        Fits the costs of a WeightedLevenshtein instance to the provided data.

        Note that learning multi-character tokens is not yet supported.

        This method analyzes pairs of strings to learn the costs of edit operations
        based on their observed frequencies. The underlying model calculates costs
        based on the principle of relative information cost.

        For a detailed explanation of the methodology, please see the
        :doc:`Cost Learning Model <cost_learning_model>` documentation page.

        :param pairs: An iterable of (ocr_string, ground_truth_string) tuples.
        :param initial_model: Optional initial model used to align OCR outputs and ground truth
                              strings. By default, an unweighted Levenshtein distance is used.
        :param calculate_for_unseen: If True (and k > 0), pre-calculates costs for all
                                     possible edit operations based on the vocabulary.
                                     If False (default), only calculates costs for operations
                                     observed in the data.
        :return: A `WeightedLevenshtein` instance with the learned costs.
        """
        from .levenshtein import WeightedLevenshtein

        if not pairs:
            return WeightedLevenshtein.unweighted()

        if initial_model is None:
            initial_model = WeightedLevenshtein.unweighted()

        all_ops = self._calculate_operations(pairs, aligner=initial_model)
        self.counts = self._tally_operations(all_ops)
        vocab = self.counts.vocab
        self.vocab_size = len(vocab)

        if not self.vocab_size:
            return WeightedLevenshtein.unweighted()

        costs = self._calculate_costs(self.counts, vocab, calculate_for_unseen=calculate_for_unseen)

        return WeightedLevenshtein(
            substitution_costs=costs.substitutions,
            insertion_costs=costs.insertions,
            deletion_costs=costs.deletions,
            symmetric_substitution=False,
            default_substitution_cost=1.0,
            default_insertion_cost=1.0,
            default_deletion_cost=1.0,
        )
