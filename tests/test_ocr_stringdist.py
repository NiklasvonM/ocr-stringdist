from ocr_stringdist import ocr_weighted_levenshtein_distance


def test_ocr_weighted_levenshtein_distance() -> None:
    assert ocr_weighted_levenshtein_distance("a", "b") == 1.0
