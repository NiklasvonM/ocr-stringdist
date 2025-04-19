from ocr_stringdist import weighted_levenshtein_distance
from icecream import ic

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
