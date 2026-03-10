#!/usr/bin/env python3
"""
Compare multiple runs by reading metrics JSON files and printing a compact table.

Defaults to comparing held-out test metrics:
  runs/*/metrics_test.json
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Dict, List, Tuple


DISPLAY_KEYS: List[Tuple[str, str]] = [
    ("rougeL_f1_mean", "ROUGE-L F1"),
    ("cosine_mean_corpus_mean", "Cosine to corpus mean"),
    ("cosine_max_sim_mean", "Cosine max sim"),
    ("cosine_max_sim_prop_above_95", "Mode collapse (prop >0.95)"),
    ("sentiment_compound_abs_mean", "Emotional volume"),
    ("diversity_ttr", "Diversity (TTR)"),
    ("diversity_self_bleu", "Diversity (self-BLEU)"),
    ("perplexity_self_ref_mean", "Self PPL (refs)"),
    ("perplexity_base_ref_mean", "Base PPL (refs)"),
]


def _load(path: Path) -> Dict[str, float]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _fmt(x) -> str:
    try:
        if x is None:
            return "nan"
        v = float(x)
        if v != v:  # nan
            return "nan"
        return f"{v:.4f}"
    except Exception:
        return "nan"


def main() -> None:
    p = argparse.ArgumentParser(description="Compare metrics across runs.")
    p.add_argument(
        "--metrics",
        nargs="*",
        default=["runs/*/metrics_test.json"],
        help="Glob(s) or explicit paths to metrics JSON files.",
    )
    args = p.parse_args()

    paths: List[Path] = []
    for pat in args.metrics:
        expanded = glob.glob(pat)
        if expanded:
            paths.extend(Path(x) for x in expanded)
        else:
            paths.append(Path(pat))

    paths = [p for p in paths if p.exists()]
    paths.sort()
    if not paths:
        raise SystemExit("No metrics files found.")

    rows = []
    for path in paths:
        m = _load(path)
        run_name = path.parent.name
        rows.append((run_name, m))

    # Print header
    col_names = ["run"] + [label for _, label in DISPLAY_KEYS]
    widths = [max(len(col_names[0]), max(len(r[0]) for r in rows))]
    for _, label in DISPLAY_KEYS:
        widths.append(max(len(label), 12))

    def print_row(vals: List[str]) -> None:
        out = []
        for v, w in zip(vals, widths):
            out.append(v.ljust(w))
        print("  ".join(out))

    print_row(col_names)
    print_row(["-" * w for w in widths])

    for run_name, m in rows:
        vals = [run_name] + [_fmt(m.get(k)) for k, _ in DISPLAY_KEYS]
        print_row(vals)


if __name__ == "__main__":
    main()

