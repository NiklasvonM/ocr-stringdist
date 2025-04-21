# Start marker for literalinclude, see docs/source/api/index.rst.
# OCR_DISTANCE_MAP_START
ocr_distance_map: dict[tuple[str, str], float] = {
    ("O", "0"): 0.1,
    ("l", "1"): 0.1,
    ("I", "1"): 0.15,
    ("o", "0"): 0.2,
    ("B", "8"): 0.25,
    ("S", "5"): 0.3,
    ("G", "6"): 0.3,
    ("Z", "2"): 0.3,
    ("C", "c"): 0.3,
    ("é", "e"): 0.3,
    ("Ä", "A"): 0.4,
    ("Ö", "O"): 0.4,
    ("Ü", "U"): 0.4,
    ("c", "e"): 0.4,
    ("a", "o"): 0.4,
    ("u", "v"): 0.4,
    ("i", "l"): 0.4,
    ("s", "5"): 0.4,
    ("m", "n"): 0.5,
    ("f", "s"): 0.5,
    (".", ","): 0.5,
    ("2", "Z"): 0.5,
    ("t", "f"): 0.6,
    ("r", "n"): 0.6,
    ("-", "_"): 0.6,
    ("ß", "B"): 0.6,
    ("h", "b"): 0.7,
    ("v", "y"): 0.7,
    ("i", "j"): 0.7,
    ("é", "á"): 0.7,
    ("E", "F"): 0.8,
}
# OCR_DISTANCE_MAP_END
# End marker for literalinclude
"""
Pre-defined distance map between characters, considering common OCR errors.
The distances are between 0 and 1.
This map is intended to be used with `symmetric=True`.
"""
