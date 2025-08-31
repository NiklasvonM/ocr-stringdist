from dataclasses import dataclass
from typing import Literal, Optional

from ._rust_stringdist import (
    _batch_weighted_levenshtein_distance,
    _explain_weighted_levenshtein_distance,
    _weighted_levenshtein_distance,
)
from .default_ocr_distances import ocr_distance_map

OperationType = Literal["substitute", "insert", "delete"]


@dataclass(frozen=True)
class EditOperation:
    """
    Stellt eine einzelne Edit-Operation dar (Substitution, Einfügung oder Löschung).

    :param op_type: 'substitute', 'insert', or 'delete'.
    :param source_token: Source token (string). `None` for insertions.
    :param target_token: Target token (string). `None` for deletions.
    :param cost: Costs for this operation.
    """

    op_type: OperationType
    source_token: Optional[str]
    target_token: Optional[str]
    cost: float


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
    if substitution_costs is None:
        substitution_costs = ocr_distance_map
    if insertion_costs is None:
        insertion_costs = {}
    if deletion_costs is None:
        deletion_costs = {}
    # _weighted_levenshtein_distance is written in Rust, see src/rust_stringdist.rs.
    return _weighted_levenshtein_distance(
        s1,
        s2,
        substitution_costs=substitution_costs,
        insertion_costs=insertion_costs,
        deletion_costs=deletion_costs,
        symmetric_substitution=symmetric_substitution,
        default_substitution_cost=default_substitution_cost,
        default_insertion_cost=default_insertion_cost,
        default_deletion_cost=default_deletion_cost,
    )


def weighted_levenshtein_path(
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
) -> list[EditOperation]:
    """
    Computes the path of operations associated with the custom Levenshtein distance.

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
    :return: List of :class:`EditOperation`s.
    """
    if substitution_costs is None:
        substitution_costs = ocr_distance_map
    if insertion_costs is None:
        insertion_costs = {}
    if deletion_costs is None:
        deletion_costs = {}

    # Rufe die in Rust implementierte Funktion auf
    raw_path: list[tuple[OperationType, Optional[str], Optional[str], float]] = (
        _explain_weighted_levenshtein_distance(
            s1,
            s2,
            substitution_costs=substitution_costs,
            insertion_costs=insertion_costs,
            deletion_costs=deletion_costs,
            symmetric_substitution=symmetric_substitution,
            default_substitution_cost=default_substitution_cost,
            default_insertion_cost=default_insertion_cost,
            default_deletion_cost=default_deletion_cost,
        )
    )

    path = [
        EditOperation(op_type, source, target, cost) for op_type, source, target, cost in raw_path
    ]

    return path


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
    if substitution_costs is None:
        substitution_costs = ocr_distance_map
    if insertion_costs is None:
        insertion_costs = {}
    if deletion_costs is None:
        deletion_costs = {}
    # _batch_weighted_levenshtein_distance is written in Rust, see src/rust_stringdist.rs.
    return _batch_weighted_levenshtein_distance(  # type: ignore  # noqa: F405
        s,
        candidates,
        substitution_costs=substitution_costs,
        insertion_costs=insertion_costs,
        deletion_costs=deletion_costs,
        symmetric_substitution=symmetric_substitution,
        default_substitution_cost=default_substitution_cost,
        default_insertion_cost=default_insertion_cost,
        default_deletion_cost=default_deletion_cost,
    )
