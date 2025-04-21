#!/usr/bin/env python3
"""
Example demonstrating the usage of the batch processing functions from ocr_stringdist.
"""

import time
from typing import Any, Callable

import ocr_stringdist as osd

MAX_TOKEN_CHARACTERS = 1


def benchmark(func: Callable, *args: Any, **kwargs: Any) -> tuple[Any, float]:  # type: ignore
    """Run a function and return the execution time in seconds."""
    start = time.time()
    result = func(*args, **kwargs)
    end = time.time()
    return result, end - start


def compare_methods() -> None:
    """
    Compare the performance of different methods for calculating Levenshtein distances.
    """
    # Example data
    source = "recognition"
    candidates = ["recognition", "recogmtion", "recognltlon", "recogrtition", "recognitton"] * 1000

    print("\nSingle string against multiple candidates:")
    print("-" * 50)

    # Standard loop approach
    _, time_loop = benchmark(
        lambda: [
            osd.weighted_levenshtein_distance(
                source, cand, max_token_characters=MAX_TOKEN_CHARACTERS
            )
            for cand in candidates
        ]
    )
    print(
        f"Loop of single calls: {time_loop:.6f} seconds "
        f"({1000 * time_loop / len(candidates):.6f}ms each)"
    )

    # Batch approach
    _, time_batch = benchmark(
        osd.batch_weighted_levenshtein_distance,
        source,
        candidates,
        max_token_characters=MAX_TOKEN_CHARACTERS,
    )
    print(
        f"Batch function: {time_batch:.6f} seconds "
        f"({1000 * time_batch / len(candidates):.6f}ms each)"
    )
    print(f"Speedup: {time_loop / time_batch:.2f}x")


def main() -> None:
    """Main function."""
    print("Demonstrating batch processing functions from ocr_stringdist\n")

    # Run the benchmarks
    compare_methods()


if __name__ == "__main__":
    main()
