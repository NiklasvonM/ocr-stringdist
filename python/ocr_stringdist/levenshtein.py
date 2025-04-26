from typing import Optional

from ._rust_stringdist import *  # noqa: F403
from .default_ocr_distances import ocr_distance_map


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
    return _weighted_levenshtein_distance(  # type: ignore  # noqa: F405
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
