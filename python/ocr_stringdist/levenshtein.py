from __future__ import annotations

from collections.abc import Iterable
from typing import Optional

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
    :param symmetric_substitution: If True, substitution costs are bidirectional.
    :param default_substitution_cost: Default cost for single-char substitutions not in the map.
    :param default_insertion_cost: Default cost for single-char insertions not in the map.
    :param default_deletion_cost: Default cost for single-char deletions not in the map.
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
        self.substitution_costs = (
            ocr_distance_map if substitution_costs is None else substitution_costs
        )
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

        For more advanced learning configuration, see the `ocr_stringdist.learner.Learner` class.

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
        from .learner import Learner

        return Learner().fit(pairs)

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


def weighted_levenshtein_distance(
    s1: str,
    s2: str,
    /,
    substitution_costs: Optional[dict[tuple[str, str], float]] = None,
    insertion_costs: Optional[dict[str, float]] = None,
    deletion_costs: Optional[dict[str, float]] = None,
    *,
    symmetric_substitution: bool = True,
    default_substitution_cost: float = 1.0,
    default_insertion_cost: float = 1.0,
    default_deletion_cost: float = 1.0,
) -> float:
    """
    Levenshtein distance with custom substitution, insertion and deletion costs.

    See also :meth:`WeightedLevenshtein.distance`.

    The default `substitution_costs` considers common OCR errors, see
    :py:data:`ocr_stringdist.default_ocr_distances.ocr_distance_map`.

    :param s1: First string (interpreted as the string read via OCR)
    :param s2: Second string
    :param substitution_costs: Dictionary mapping tuples of strings ("substitution tokens") to their
                     substitution costs. Only one direction needs to be configured unless
                     `symmetric_substitution` is False.
                     Note that the runtime scales in the length of the longest substitution token.
                     Defaults to `ocr_stringdist.ocr_distance_map`.
    :param insertion_costs: Dictionary mapping strings to their insertion costs.
    :param deletion_costs: Dictionary mapping strings to their deletion costs.
    :param symmetric_substitution: Should the keys of `substitution_costs` be considered to be
                                   symmetric? Defaults to True.
    :param default_substitution_cost: The default substitution cost for character pairs not found
                                      in `substitution_costs`.
    :param default_insertion_cost: The default insertion cost for characters not found in
                                   `insertion_costs`.
    :param default_deletion_cost: The default deletion cost for characters not found in
                                  `deletion_costs`.
    """
    return WeightedLevenshtein(
        substitution_costs=substitution_costs,
        insertion_costs=insertion_costs,
        deletion_costs=deletion_costs,
        symmetric_substitution=symmetric_substitution,
        default_substitution_cost=default_substitution_cost,
        default_insertion_cost=default_insertion_cost,
        default_deletion_cost=default_deletion_cost,
    ).distance(s1, s2)


def batch_weighted_levenshtein_distance(
    s: str,
    candidates: list[str],
    /,
    substitution_costs: Optional[dict[tuple[str, str], float]] = None,
    insertion_costs: Optional[dict[str, float]] = None,
    deletion_costs: Optional[dict[str, float]] = None,
    *,
    symmetric_substitution: bool = True,
    default_substitution_cost: float = 1.0,
    default_insertion_cost: float = 1.0,
    default_deletion_cost: float = 1.0,
) -> list[float]:
    """
    Calculate weighted Levenshtein distances between a string and multiple candidates.

    See also :meth:`WeightedLevenshtein.batch_distance`.

    This is more efficient than calling :func:`weighted_levenshtein_distance` multiple times.

    :param s: The string to compare (interpreted as the string read via OCR)
    :param candidates: List of candidate strings to compare against
    :param substitution_costs: Dictionary mapping tuples of strings ("substitution tokens") to their
                     substitution costs. Only one direction needs to be configured unless
                     `symmetric_substitution` is False.
                     Note that the runtime scales in the length of the longest substitution token.
                     Defaults to `ocr_stringdist.ocr_distance_map`.
    :param insertion_costs: Dictionary mapping strings to their insertion costs.
    :param deletion_costs: Dictionary mapping strings to their deletion costs.
    :param symmetric_substitution: Should the keys of `substitution_costs` be considered to be
                                   symmetric? Defaults to True.
    :param default_substitution_cost: The default substitution cost for character pairs not found
                                      in `substitution_costs`.
    :param default_insertion_cost: The default insertion cost for characters not found in
                                   `insertion_costs`.
    :param default_deletion_cost: The default deletion cost for characters not found in
                                  `deletion_costs`.
    :return: A list of distances corresponding to each candidate
    """
    return WeightedLevenshtein(
        substitution_costs=substitution_costs,
        insertion_costs=insertion_costs,
        deletion_costs=deletion_costs,
        symmetric_substitution=symmetric_substitution,
        default_substitution_cost=default_substitution_cost,
        default_insertion_cost=default_insertion_cost,
        default_deletion_cost=default_deletion_cost,
    ).batch_distance(s, candidates)


def explain_weighted_levenshtein(
    s1: str,
    s2: str,
    /,
    substitution_costs: Optional[dict[tuple[str, str], float]] = None,
    insertion_costs: Optional[dict[str, float]] = None,
    deletion_costs: Optional[dict[str, float]] = None,
    *,
    symmetric_substitution: bool = True,
    default_substitution_cost: float = 1.0,
    default_insertion_cost: float = 1.0,
    default_deletion_cost: float = 1.0,
    filter_matches: bool = True,
) -> list[EditOperation]:
    """
    Computes the path of operations associated with the custom Levenshtein distance.

    See also :meth:`WeightedLevenshtein.explain`.

    The default `substitution_costs` considers common OCR errors, see
    :py:data:`ocr_stringdist.default_ocr_distances.ocr_distance_map`.

    :param s1: First string (interpreted as the string read via OCR)
    :param s2: Second string
    :param substitution_costs: Dictionary mapping tuples of strings ("substitution tokens") to their
                     substitution costs. Only one direction needs to be configured unless
                     `symmetric_substitution` is False.
                     Note that the runtime scales in the length of the longest substitution token.
                     Defaults to `ocr_stringdist.ocr_distance_map`.
    :param insertion_costs: Dictionary mapping strings to their insertion costs.
    :param deletion_costs: Dictionary mapping strings to their deletion costs.
    :param symmetric_substitution: Should the keys of `substitution_costs` be considered to be
                                   symmetric? Defaults to True.
    :param default_substitution_cost: The default substitution cost for character pairs not found
                                      in `substitution_costs`.
    :param default_insertion_cost: The default insertion cost for characters not found in
                                   `insertion_costs`.
    :param default_deletion_cost: The default deletion cost for characters not found in
                                  `deletion_costs`.
    :return: List of :class:`EditOperation` instances.
    """
    return WeightedLevenshtein(
        substitution_costs=substitution_costs,
        insertion_costs=insertion_costs,
        deletion_costs=deletion_costs,
        symmetric_substitution=symmetric_substitution,
        default_substitution_cost=default_substitution_cost,
        default_insertion_cost=default_insertion_cost,
        default_deletion_cost=default_deletion_cost,
    ).explain(s1, s2, filter_matches=filter_matches)
