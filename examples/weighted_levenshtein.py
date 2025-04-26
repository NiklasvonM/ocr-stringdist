#!/usr/bin/env python3
from icecream import ic
from ocr_stringdist import find_best_candidate, weighted_levenshtein_distance

ic(
    weighted_levenshtein_distance(
        "12345G",
        "123456",
        # Default costs
    ),
)

ic(
    weighted_levenshtein_distance(
        "12345G",
        "123456",
        {("G", "6"): 0.1},  # Custom cost_map
    )
)

# Substitution of multiple characters at once is supported.
ic(
    weighted_levenshtein_distance(
        "이탈리",
        "OI탈리",  # Korean syllables may be confused with multiple Latin letters at once
        {("이", "OI"): 0.5},
    ),
)

ic(
    weighted_levenshtein_distance(
        "ABCDE",
        "XBCDE",
        substitution_costs={},
        default_substitution_cost=0.8,  # Lower default substitution cost (default is 1.0)
    )
)

ic(weighted_levenshtein_distance("A", "B", {("A", "B"): 0.0}, symmetric_substitution=False))
ic(weighted_levenshtein_distance("A", "B", {("B", "A"): 0.0}, symmetric_substitution=False))

ic(
    find_best_candidate(
        "apple",
        ["apply", "apples", "orange", "appIe"],
        lambda s1, s2: weighted_levenshtein_distance(s1, s2, {("l", "I"): 0.1}),
    )
)
