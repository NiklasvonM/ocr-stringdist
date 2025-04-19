from collections.abc import Callable, Iterable
from typing import Optional


def find_best_candidate(
    s: str,
    candidates: Iterable[str],
    distance_fun: Callable[[str, str], float],
    *,
    minimize: bool = True,
    early_return_value: Optional[float] = None,
) -> tuple[str, float]:
    """
    Finds the best matching string from a collection of candidates based on a distance function.

    Compares a given string against each string in the 'candidates'
    iterable using the provided 'distance_fun'. It identifies the candidate
    that yields the minimum (or maximum, if minimize=False) distance.

    :param s: The reference string to compare against.
    :type s: str
    :param candidates: An iterable of candidate strings to compare with 's'.
    :type candidates: Iterable[str]
    :param distance_fun: A function that takes two strings (s, candidate) and
                         returns a float representing their distance or similarity.
    :type distance_fun: Callable[[str, str], float]
    :param minimize: If True (default), finds the candidate with the minimum
                     distance. If False, finds the candidate with the maximum
                     distance (useful for similarity scores).
    :type minimize: bool
    :param early_return_value: If provided, the function will return immediately
                               if a distance is found that is less than or equal
                               to this value (if minimize=True) or greater than
                               or equal to this value (if minimize=False).
                               If None (default), all candidates are checked.
    :type early_return_value: Optional[float]
    :raises ValueError: If the 'candidates' iterable is empty.
    :return: A tuple containing the best matching candidate string and its
             calculated distance/score.
    :rtype: tuple[str, float]

    :Example:

    >>> from ocr_stringdist import weighted_levenshtein_distance as distance
    >>> s = "apple"
    >>> candidates = ["apply", "apples", "orange", "appIe"]
    >>> find_best_match(s, candidates, lambda s1, s2: distance(s1, s2, {("l", "I"): 0.1}))
    ('appIe', 0.1)
    """
    if not candidates:
        raise ValueError("The 'candidates' iterable cannot be empty.")

    best_candidate: str = ""

    if minimize:
        best_distance = float("inf")

        def is_next_best(current: float, best: float) -> bool:
            return current < best

        def can_return_early(current: float, threshold: float) -> bool:
            return current <= threshold
    else:
        best_distance = -float("inf")

        def is_next_best(current: float, best: float) -> bool:
            return current > best

        def can_return_early(current: float, threshold: float) -> bool:
            return current >= threshold

    for candidate in candidates:
        current_distance = distance_fun(s, candidate)

        if early_return_value is not None and can_return_early(
            current_distance, early_return_value
        ):
            return candidate, current_distance
        if is_next_best(current_distance, best_distance):
            best_distance = current_distance
            best_candidate = candidate

    return best_candidate, best_distance
