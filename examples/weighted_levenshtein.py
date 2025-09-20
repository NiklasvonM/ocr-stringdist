#!/usr/bin/env python3
from icecream import ic
from ocr_stringdist import WeightedLevenshtein, find_best_candidate

ic(  # Default costs
    WeightedLevenshtein().distance("12345G", "123456")
)

ic(  # Custom cost_map
    WeightedLevenshtein({("G", "6"): 0.1}).distance("12345G", "123456")
)

# Substitution of multiple characters at once is supported.
ic(  # Korean syllables may be confused with multiple Latin letters at once
    WeightedLevenshtein({("이", "OI"): 0.5}).distance("이탈리", "OI탈리")
)

ic(  # Lower default substitution cost (default is 1.0)
    WeightedLevenshtein(
        substitution_costs={},
        default_substitution_cost=0.8,
    ).distance("ABCDE", "XBCDE")
)

ic(WeightedLevenshtein({("A", "B"): 0.0}, symmetric_substitution=False).distance("A", "B"))
ic(WeightedLevenshtein({("B", "A"): 0.0}, symmetric_substitution=False).distance("A", "B"))

wl = WeightedLevenshtein({("l", "I"): 0.1})
ic(find_best_candidate("apple", ["apply", "apples", "orange", "appIe"], wl.distance))
