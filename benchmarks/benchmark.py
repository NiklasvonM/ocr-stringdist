"""
Benchmark for WeightedLevenshtein distance computation.

Run with:
    uv run python benchmarks/benchmark.py

The results are printed to stdout. Save them before and after a code change to
compare throughput.
"""

from __future__ import annotations

import sys
import timeit
from dataclasses import dataclass

sys.path.insert(0, "python")

from ocr_stringdist import WeightedLevenshtein

REPEAT = 5
NUMBER = 200


@dataclass
class Case:
    label: str
    wl: WeightedLevenshtein
    s1: str
    s2: str


# ── Cost maps used across cases ────────────────────────────────────────────────

_WL_DEFAULT = WeightedLevenshtein.unweighted()

_WL_OCR = WeightedLevenshtein(
    substitution_costs={
        ("6", "G"): 0.5,
        ("0", "O"): 0.1,
        ("rn", "m"): 0.15,
        ("cl", "d"): 0.2,
        ("l", "1"): 0.2,
        ("h", "In"): 0.25,
        ("vv", "w"): 0.15,
    },
    deletion_costs={"G": 0.01, "O": 0.05},
    default_substitution_cost=1.0,
    default_deletion_cost=1.0,
    default_insertion_cost=1.0,
)

# ── Benchmark cases ────────────────────────────────────────────────────────────

CASES: list[Case] = [
    # Issue #12 — transitive chain: sub("6"→"G", 0.5) + del("G", 0.01) = 0.51
    Case(
        "issue-12: transitive chain '06'→'0'",
        WeightedLevenshtein(
            substitution_costs={("6", "G"): 0.5},
            deletion_costs={"G": 0.01},
        ),
        "06",
        "0",
    ),
    # Short strings, no custom costs
    Case("short identical (no-op)", _WL_DEFAULT, "hello", "hello"),
    Case("short similar (1 sub)", _WL_DEFAULT, "kitten", "sitten"),
    Case("short dissimilar", _WL_DEFAULT, "abc", "xyz"),
    # Medium strings
    Case(
        "medium OCR-like",
        _WL_OCR,
        "The man ran down the hill at 10 km/h.",
        "Tine rnan ram dovvn tine Ini11 at 1O krn/In.",
    ),
    Case(
        "medium no-match",
        _WL_DEFAULT,
        "abcdefghij",
        "zyxwvutsrq",
    ),
    # Long strings
    Case(
        "long similar",
        _WL_DEFAULT,
        "a" * 200 + "b" * 50,
        "a" * 198 + "c" * 52,
    ),
    Case(
        "long OCR-like",
        _WL_OCR,
        "The man ran down the hill at 10 km/h. " * 5,
        "Tine rnan ram dovvn tine Ini11 at 1O krn/In. " * 5,
    ),
    # Batch distance (1 source vs. 100 candidates)
]

BATCH_CANDIDATES = [f"word{i}" for i in range(100)]
_WL_BATCH = WeightedLevenshtein.unweighted()


def run_batch() -> None:
    _WL_BATCH.batch_distance("word50", BATCH_CANDIDATES)


# Runner


def bench_case(case: Case) -> tuple[float, float]:
    """Returns (best_ms_per_call, calls_per_second)."""
    stmt = lambda: case.wl.distance(case.s1, case.s2)  # noqa: E731
    times = timeit.repeat(stmt, repeat=REPEAT, number=NUMBER)
    best_total_s = min(times)
    best_ms = best_total_s / NUMBER * 1000
    cps = NUMBER / best_total_s
    return best_ms, cps


def main() -> None:
    col_w = max(len(c.label) for c in CASES) + 2
    header = f"{'Case':<{col_w}}  {'Best ms/call':>14}  {'calls/sec':>12}"
    print(header)
    print("-" * len(header))

    for case in CASES:
        ms, cps = bench_case(case)
        print(f"{case.label:<{col_w}}  {ms:>14.4f}  {cps:>12,.0f}")

    # Batch benchmark
    batch_times = timeit.repeat(run_batch, repeat=REPEAT, number=NUMBER)
    best_batch_s = min(batch_times)
    batch_ms = best_batch_s / NUMBER * 1000
    batch_cps = NUMBER / best_batch_s
    label = "batch_distance (100 candidates)"
    print(f"{label:<{col_w}}  {batch_ms:>14.4f}  {batch_cps:>12,.0f}")

    print()
    print(f"Settings: repeat={REPEAT}, number={NUMBER} calls per timing")


if __name__ == "__main__":
    main()
