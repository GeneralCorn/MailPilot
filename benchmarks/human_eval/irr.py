from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path


def cohens_kappa(a: list[int], b: list[int]) -> float:
    """Strict-match Cohen's κ for two raters over a discrete rating scale."""
    if len(a) != len(b) or not a:
        raise ValueError("rater lists must be same non-zero length")
    n = len(a)
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    cnt_a = Counter(a)
    cnt_b = Counter(b)
    categories = set(cnt_a) | set(cnt_b)
    pe = sum((cnt_a.get(c, 0) / n) * (cnt_b.get(c, 0) / n) for c in categories)
    if pe == 1:
        return 1.0  # both raters constant on the same value
    return (po - pe) / (1 - pe)


def load_ratings(path: Path) -> dict[str, tuple[list[int], list[int]]]:
    """Group filled ratings by criterion. Skips rows with blank rater values."""
    by_criterion: dict[str, tuple[list[int], list[int]]] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            ra, rb = row["rater_a"].strip(), row["rater_b"].strip()
            if not ra or not rb:
                continue
            try:
                ra_i, rb_i = int(ra), int(rb)
            except ValueError:
                continue
            a, b = by_criterion.setdefault(row["criterion"], ([], []))
            a.append(ra_i)
            b.append(rb_i)
    return by_criterion


def main() -> int:
    ap = argparse.ArgumentParser(description="Compute Cohen's κ per criterion from a filled rating CSV.")
    ap.add_argument("csv_path", type=Path)
    args = ap.parse_args()
    grouped = load_ratings(args.csv_path)
    if not grouped:
        print("no rated rows found", file=sys.stderr)
        return 1
    print(f"{'criterion':<25}{'n':>5}  κ")
    for crit, (a, b) in sorted(grouped.items()):
        print(f"{crit:<25}{len(a):>5}  {cohens_kappa(a, b):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
