from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Optional

from ._rust_stringdist import (
    _batch_weighted_levenshtein_distance,
    _explain_weighted_levenshtein_distance,
    _weighted_levenshtein_distance,
)
from .default_ocr_distances import ocr_distance_map
from .edit_operation import EditOperation


class WeightedLevenshtein:
    """
    Calculates Levenshtein distance with custom, configurable costs.

    This class is initialized with cost dictionaries and settings that define
    how the distance is measured. Once created, its methods can be used to
    efficiently compute distances and explain the edit operations.

    :param substitution_costs: Maps (str, str) tuples to their substitution cost.
                               Defaults to costs based on common OCR errors.
    :param insertion_costs: Maps a string to its insertion cost.
    :param deletion_costs: Maps a string to its deletion cost.
    :param symmetric_substitution: If True, a cost defined for, e.g., ('0', 'O') will automatically
                                   apply to ('O', '0'). If False, both must be defined explicitly.
    :param default_substitution_cost: Default cost for single-char substitutions not in the map.
    :param default_insertion_cost: Default cost for single-char insertions not in the map.
    :param default_deletion_cost: Default cost for single-char deletions not in the map.

    :raises TypeError, ValueError: If the provided arguments are invalid.
    """

    substitution_costs: dict[tuple[str, str], float]
    insertion_costs: dict[str, float]
    deletion_costs: dict[str, float]
    symmetric_substitution: bool
    default_substitution_cost: float
    default_insertion_cost: float
    default_deletion_cost: float

    def __init__(
        self,
        substitution_costs: Optional[dict[tuple[str, str], float]] = None,
        insertion_costs: Optional[dict[str, float]] = None,
        deletion_costs: Optional[dict[str, float]] = None,
        *,
        symmetric_substitution: bool = True,
        default_substitution_cost: float = 1.0,
        default_insertion_cost: float = 1.0,
        default_deletion_cost: float = 1.0,
    ) -> None:
        # Validate default costs
        for cost_name, cost_val in [
            ("default_substitution_cost", default_substitution_cost),
            ("default_insertion_cost", default_insertion_cost),
            ("default_deletion_cost", default_deletion_cost),
        ]:
            if not isinstance(cost_val, (int, float)):
                raise TypeError(f"{cost_name} must be a number, but got: {type(cost_val).__name__}")
            if cost_val < 0:
                raise ValueError(f"{cost_name} must be non-negative, got value: {cost_val}")

        # Validate substitution_costs dictionary
        sub_costs = ocr_distance_map if substitution_costs is None else substitution_costs
        for key, cost in sub_costs.items():
            if not (
                isinstance(key, tuple)
                and len(key) == 2
                and isinstance(key[0], str)
                and isinstance(key[1], str)
            ):
                raise TypeError(
                    f"substitution_costs keys must be tuples of two strings, but found: {key}"
                )
            if not isinstance(cost, (int, float)):
                raise TypeError(
                    f"Cost for substitution {key} must be a number, but got: {type(cost).__name__}"
                )
            if cost < 0:
                raise ValueError(f"Cost for substitution {key} cannot be negative, but got: {cost}")

        self.substitution_costs = sub_costs
        self.insertion_costs = {} if insertion_costs is None else insertion_costs
        self.deletion_costs = {} if deletion_costs is None else deletion_costs
        self.symmetric_substitution = symmetric_substitution
        self.default_substitution_cost = default_substitution_cost
        self.default_insertion_cost = default_insertion_cost
        self.default_deletion_cost = default_deletion_cost

    @classmethod
    def unweighted(cls) -> WeightedLevenshtein:
        """Creates an instance with all operations having equal cost of 1.0."""
        return cls(substitution_costs={}, insertion_costs={}, deletion_costs={})

    def distance(self, s1: str, s2: str) -> float:
        """Calculates the weighted Levenshtein distance between two strings."""
        return _weighted_levenshtein_distance(s1, s2, **self.__dict__)  # type: ignore[no-any-return]

    def explain(self, s1: str, s2: str, filter_matches: bool = True) -> list[EditOperation]:
        """
        Returns the list of edit operations to transform s1 into s2.

        :param s1: First string (interpreted as the string read via OCR)
        :param s2: Second string (interpreted as the target string)
        :param filter_matches: If True, 'match' operations are excluded from the result.
        :return: List of :class:`EditOperation` instances.
        """
        raw_path = _explain_weighted_levenshtein_distance(s1, s2, **self.__dict__)
        parsed_path = [EditOperation(*op) for op in raw_path]
        if filter_matches:
            return list(filter(lambda op: op.op_type != "match", parsed_path))
        return parsed_path

    def batch_distance(self, s: str, candidates: list[str]) -> list[float]:
        """Calculates distances between a string and a list of candidates."""
        return _batch_weighted_levenshtein_distance(s, candidates, **self.__dict__)  # type: ignore[no-any-return]

    @classmethod
    def learn_from(cls, pairs: Iterable[tuple[str, str]]) -> WeightedLevenshtein:
        """
        Creates an instance by learning costs from a dataset of (OCR, ground truth) string pairs.

        For more advanced learning configuration, see the
        :class:`ocr_stringdist.learner.CostLearner` class.

        :param pairs: An iterable of (ocr_string, ground_truth_string) tuples. Correct pairs
                      are not intended to be filtered; they are needed to learn well-aligned costs.
        :return: A new `WeightedLevenshtein` instance with the learned costs.

        Example::

            from ocr_stringdist import WeightedLevenshtein

            training_data = [
                ("8N234", "BN234"), # read '8' instead of 'B'
                ("BJK18", "BJK18"), # correct
                ("ABC0.", "ABC0"),  # extra '.'
            ]
            wl = WeightedLevenshtein.learn_from(training_data)
            print(wl.substitution_costs) # learned cost for substituting '8' with 'B'
            print(wl.deletion_costs) # learned cost for deleting '.'
        """
        from .learner import CostLearner

        return CostLearner().fit(pairs)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, WeightedLevenshtein):
            return NotImplemented
        return (
            self.substitution_costs == other.substitution_costs
            and self.insertion_costs == other.insertion_costs
            and self.deletion_costs == other.deletion_costs
            and self.symmetric_substitution == other.symmetric_substitution
            and self.default_substitution_cost == other.default_substitution_cost
            and self.default_insertion_cost == other.default_insertion_cost
            and self.default_deletion_cost == other.default_deletion_cost
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the instance's configuration to a dictionary.

        The result can be written to, say, JSON.

        For the counterpart, see :meth:`WeightedLevenshtein.from_dict`.
        """
        # Convert tuple keys to a list of lists/objects for broader compatibility (e.g., JSON)
        sub_costs_serializable = [
            {"from": k[0], "to": k[1], "cost": v} for k, v in self.substitution_costs.items()
        ]

        return {
            "substitution_costs": sub_costs_serializable,
            "insertion_costs": self.insertion_costs,
            "deletion_costs": self.deletion_costs,
            "symmetric_substitution": self.symmetric_substitution,
            "default_substitution_cost": self.default_substitution_cost,
            "default_insertion_cost": self.default_insertion_cost,
            "default_deletion_cost": self.default_deletion_cost,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WeightedLevenshtein:
        """
        Deserialize from a dictionary.

        For the counterpart, see :meth:`WeightedLevenshtein.to_dict`.

        :param data: A dictionary with (not necessarily all of) the following keys:

                     - "substitution_costs": {"from": str, "to": str, "cost": float}
                     - "substitution_costs": dict[str, float]
                     - "deletion_costs": dict[str, float]
                     - "symmetric_substitution": bool
                     - "default_substitution_cost": float
                     - "default_insertion_cost": float
                     - "default_deletion_cost": float
        """
        # Convert the list of substitution costs back to the required dict format
        sub_costs: dict[tuple[str, str], float] = {
            (item["from"], item["to"]): item["cost"] for item in data.get("substitution_costs", {})
        }

        return cls(
            substitution_costs=sub_costs,
            insertion_costs=data.get("substitution_costs"),
            deletion_costs=data.get("deletion_costs"),
            symmetric_substitution=data.get("symmetric_substitution", True),
            default_substitution_cost=data.get("default_substitution_cost", 1.0),
            default_insertion_cost=data.get("default_insertion_cost", 1.0),
            default_deletion_cost=data.get("default_deletion_cost", 1.0),
        )
