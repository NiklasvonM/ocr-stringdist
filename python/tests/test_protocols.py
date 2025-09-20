from ocr_stringdist.levenshtein import WeightedLevenshtein
from ocr_stringdist.protocols import Aligner


def test_weighted_levenshtein_is_aligner() -> None:
    assert isinstance(WeightedLevenshtein(), Aligner)
