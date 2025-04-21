from typing import Optional

from ._rust_stringdist import *  # noqa: F403
from .default_ocr_distances import ocr_distance_map


def weighted_levenshtein_distance(
    s1: str,
    s2: str,
    /,
    cost_map: Optional[dict[tuple[str, str], float]] = None,
    *,
    symmetric: bool = True,
    default_cost: float = 1.0,
) -> float:
    """
    Levenshtein distance with custom substitution costs.
    Insertion/deletion costs are 1.

    The default `cost_map` considers common OCR errors, see
    :py:data:`ocr_stringdist.default_ocr_distances.ocr_distance_map`.

    :param s1: First string
    :param s2: Second string
    :param cost_map: Dictionary mapping tuples of strings ("substitution tokens") to their
                     substitution costs.
                     Only one direction needs to be configured unless `symmetric` is False.
                     Note that the runtime scales in the length of the longest substitution token.
                     Defaults to `ocr_stringdist.ocr_distance_map`.
    :param symmetric: Should the keys of `cost_map` be considered to be symmetric? Defaults to True.
    :param default_cost: The default substitution cost for character pairs not found in `cost_map`.
    """
    if cost_map is None:
        cost_map = ocr_distance_map
    # _weighted_levenshtein_distance is written in Rust, see src/rust_stringdist.rs.
    return _weighted_levenshtein_distance(  # type: ignore  # noqa: F405
        s1, s2, cost_map=cost_map, symmetric=symmetric, default_cost=default_cost
    )


def batch_weighted_levenshtein_distance(
    s: str,
    candidates: list[str],
    /,
    cost_map: Optional[dict[tuple[str, str], float]] = None,
    *,
    symmetric: bool = True,
    default_cost: float = 1.0,
) -> list[float]:
    """
    Calculate weighted Levenshtein distances between a string and multiple candidates.

    This is more efficient than calling :func:`weighted_levenshtein_distance` multiple times.

    :param s: The string to compare
    :param candidates: List of candidate strings to compare against
    :param cost_map: Dictionary mapping tuples of strings ("substitution tokens") to their
                     substitution costs.
                     Only one direction needs to be configured unless `symmetric` is False.
                     Note that the runtime scales in the length of the longest substitution token.
                     Defaults to `ocr_stringdist.ocr_distance_map`.
    :param symmetric: Should the keys of `cost_map` be considered to be symmetric? Defaults to True.
    :param default_cost: The default substitution cost for character pairs not found in `cost_map`.
    :return: A list of distances corresponding to each candidate
    """
    if cost_map is None:
        cost_map = ocr_distance_map
    # _batch_weighted_levenshtein_distance is written in Rust, see src/rust_stringdist.rs.
    return _batch_weighted_levenshtein_distance(  # type: ignore  # noqa: F405
        s, candidates, cost_map=cost_map, symmetric=symmetric, default_cost=default_cost
    )
