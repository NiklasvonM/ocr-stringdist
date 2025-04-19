ocr_distance_map: dict[tuple[str, str], float] = {
    ("G", "6"): 0.2,
    ("O", "0"): 0.2,
    ("o", "0"): 0.2,
    ("l", "1"): 0.2,
    ("I", "1"): 0.2,
    ("2", "Z"): 0.2,
    ("B", "8"): 0.2,
    ("S", "5"): 0.3,
    ("s", "5"): 0.3,
    ("E", "F"): 0.8,
}
"""
Pre-defined distance map between characters, considering common OCR errors.
The distances are between 0 and 1.
"""
