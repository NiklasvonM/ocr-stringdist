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

ic(
    weighted_levenshtein_distance(
        "ABCDE",
        "XBCDE",
        cost_map={},
        default_cost=0.8,  # Lower default substitution cost (default is 1.0)
    )
)

ic(
    weighted_levenshtein_distance(
        "RO8ERT",
        "R0BERT",
        {("O", "0"): 0.1, ("B", "8"): 0.2},
    )
)


ic(weighted_levenshtein_distance("A", "B", {("A", "B"): 0.0}, symmetric=False))
ic(weighted_levenshtein_distance("A", "B", {("B", "A"): 0.0}, symmetric=False))
ic(weighted_levenshtein_distance("B", "A", {("B", "A"): 0.0}, symmetric=False))
ic(weighted_levenshtein_distance("B", "A", {("A", "B"): 0.0}, symmetric=False))


ic(
    find_best_candidate(
        "apple",
        ["apply", "apples", "orange", "appIe"],
        lambda s1, s2: weighted_levenshtein_distance(s1, s2, {("l", "I"): 0.1}),
    )
)
