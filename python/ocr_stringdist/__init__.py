from ._rust_stringdist import *  # noqa: F403

from .default_ocr_distances import ocr_distance_map


__all__ = [
    "ocr_distance_map",
    "weighted_levenshtein_distance",  # noqa: F405
]


def weighted_levenshtein_distance(
    s1: str,
    s2: str,
    /,
    cost_map: dict[tuple[str, str], float] | None = None,
    *,
    symmetric: bool = True,
    default_cost: float = 1.0,
) -> float:
    """
    Levenshtein distance with custom substitution costs.
    Insertion/deletion costs are 1.

    The default `cost_map` considers common OCR errors, see `ocr_stringdist.ocr_distance_map`.

    :param s1: First string
    :param s2: Second string
    :param cost_map: Dictionary mapping tuples of characters to their substitution cost.
                     Only one direction needs to be configured unless `symmetric` is False.
                     Defaults to `ocr_stringdist.ocr_distance_map`.
    :param symmetric: Should the keys of `cost_map` be considered to be symmetric? Defaults to True.
    :param default_cost: The default substitution cost for character pairs not found in `cost_map`.
    """
    if cost_map is None:
        cost_map = ocr_distance_map
    return _weighted_levenshtein_distance(  # noqa: F405
        s1, s2, cost_map=cost_map, symmetric=symmetric, default_cost=default_cost
    )
