from ocr_stringdist import (
    ocr_weighted_levenshtein_distance,
    custom_weighted_levenshtein_distance,
)

# Example with default OCR cost map
print("Using default OCR cost map:")
default_result = ocr_weighted_levenshtein_distance("12345G", "123456")
print(f"Distance between '12345G' and '123456': {default_result}")

# Example with custom cost map
custom_cost_map: dict[tuple[str, str], float] = {
    ("G", "6"): 0.1,  # Make G/6 even more similar (default is 0.2)
    ("A", "B"): 0.3,  # Make A/B somewhat similar
    ("X", "Y"): 0.5,  # Make X/Y moderately similar
}

print("\nUsing custom cost map:")
custom_result = custom_weighted_levenshtein_distance(
    "12345G", "123456", custom_cost_map
)
print(f"Distance between '12345G' and '123456' with custom map: {custom_result}")

# Example with custom default cost
print("\nUsing custom default cost:")
custom_default_result = custom_weighted_levenshtein_distance(
    "ABCDE",
    "XBCDE",
    cost_map={("A", "X"): 0.5},
    default_cost=0.8,  # Lower default substitution cost (default is 1.0)
)
print(
    f"Distance between 'ABCDE' and 'XBCDE' with custom default cost: {custom_default_result}"
)

# More complex example - comparing names with custom costs for similar looking characters
name_cost_map = {
    ("O", "0"): 0.1,  # Letter O and number 0
    ("l", "1"): 0.1,  # Lowercase L and number 1
    ("I", "1"): 0.1,  # Uppercase I and number 1
    ("S", "5"): 0.2,  # Letter S and number 5
    ("Z", "2"): 0.2,  # Letter Z and number 2
    ("B", "8"): 0.2,  # Letter B and number 8
}

print("\nComparing names with OCR-like errors:")
name1 = "ROBERT"
name2 = "R0BERT"  # Using 0 instead of O
distance = custom_weighted_levenshtein_distance(name1, name2, name_cost_map)
print(f"Distance between '{name1}' and '{name2}': {distance}")
