from ocr_stringdist import weighted_levenshtein_distance


def test_weighted_levenshtein_distance() -> None:
    assert weighted_levenshtein_distance("a", "b") == 1.0
