#!/usr/bin/env python3
"""
Data analysis on the initial YouTube trending CSV (US_youtube_trending_data.csv).

Produces summary statistics, engagement metrics, time/category distributions,
and optional report output.

Usage:
  python analyze_initial_data.py [input.csv] [--report report.txt]
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# Handle Windows console encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# This script is in clean/, but the raw CSV lives at project root.
CLEAN_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CLEAN_DIR.parent
DEFAULT_CSV = PROJECT_ROOT / "US_youtube_trending_data.csv"


def load_csv(path: Path) -> pd.DataFrame:
    """Load CSV with robust encoding and bad-line handling."""
    last_err = None
    for encoding in ("utf-8", "latin-1", "cp1252"):
        try:
            return pd.read_csv(
                path,
                encoding=encoding,
                on_bad_lines="skip",
                engine="python",
            )
        except Exception as e:
            last_err = e
            continue
    # Fallback: read with encoding that replaces bad chars
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return pd.read_csv(
                f,
                on_bad_lines="skip",
                engine="python",
            )
    except Exception as e:
        last_err = e
    raise ValueError(f"Could not load {path}: {last_err}") from last_err


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Parse date columns for time-based analysis."""
    df = df.copy()
    for col in ("publishedAt", "trending_date"):
        if col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col], errors="coerce")
            except Exception:
                pass
    return df


def run_analysis(df: pd.DataFrame) -> dict:
    """Compute analysis results as a structured dict (for reporting)."""
    results = {}

    # Basic shape and columns
    results["shape"] = df.shape
    results["columns"] = list(df.columns)
    results["dtypes"] = df.dtypes.astype(str).to_dict()
    results["missing"] = df.isnull().sum().to_dict()
    results["missing_pct"] = (df.isnull().sum() / len(df) * 100).round(2).to_dict()

    # Numeric columns
    num_cols = ["view_count", "likes", "dislikes", "comment_count"]
    num_cols = [c for c in num_cols if c in df.columns]
    if num_cols:
        results["numeric_stats"] = df[num_cols].describe().round(2).to_dict()
        # Totals
        results["totals"] = df[num_cols].sum().to_dict()

    # Engagement ratios (where view_count > 0)
    if "view_count" in df.columns:
        v = df["view_count"].replace(0, pd.NA)
        if "likes" in df.columns:
            results["like_rate_mean"] = (df["likes"] / v).mean()
            results["like_rate_median"] = (df["likes"] / v).median()
        if "comment_count" in df.columns:
            results["comment_rate_mean"] = (df["comment_count"] / v).mean()
            results["comment_rate_median"] = (df["comment_count"] / v).median()

    # Title length
    if "title" in df.columns:
        lens = df["title"].fillna("").str.len()
        results["title_len_mean"] = lens.mean()
        results["title_len_median"] = lens.median()
        results["title_len_min"] = lens.min()
        results["title_len_max"] = lens.max()

    # Time-based (if dates parsed)
    df = parse_dates(df)
    if "publishedAt" in df.columns and pd.api.types.is_datetime64_any_dtype(df["publishedAt"]):
        results["published_min"] = df["publishedAt"].min()
        results["published_max"] = df["publishedAt"].max()
        results["videos_per_year"] = df["publishedAt"].dt.year.value_counts().sort_index().to_dict()
    if "trending_date" in df.columns and pd.api.types.is_datetime64_any_dtype(df["trending_date"]):
        results["trending_min"] = df["trending_date"].min()
        results["trending_max"] = df["trending_date"].max()

    # Category distribution
    if "categoryId" in df.columns:
        results["category_counts"] = df["categoryId"].value_counts().head(20).to_dict()
    if "channelTitle" in df.columns:
        results["top_channels"] = df["channelTitle"].value_counts().head(20).to_dict()

    return results


def print_analysis(results: dict, df: pd.DataFrame) -> None:
    """Print analysis to stdout."""
    print("=" * 60)
    print("INITIAL CSV DATA ANALYSIS (US YouTube Trending)")
    print("=" * 60)

    print("\n--- Shape & structure ---")
    print(f"Rows: {results['shape'][0]:,}, Columns: {results['shape'][1]}")
    print("Columns:", results["columns"])

    print("\n--- Missing values ---")
    for col, count in results["missing"].items():
        if count > 0:
            pct = results["missing_pct"].get(col, 0)
            print(f"  {col}: {count:,} ({pct}%)")

    if "numeric_stats" in results:
        print("\n--- Numeric summary (view_count, likes, dislikes, comment_count) ---")
        print(pd.DataFrame(results["numeric_stats"]).to_string())

    if "totals" in results:
        print("\n--- Totals ---")
        for k, v in results["totals"].items():
            print(f"  {k}: {v:,.0f}")

    if "like_rate_mean" in results:
        print("\n--- Engagement rates (mean / median) ---")
        print(f"  Likes / views:     {results['like_rate_mean']:.6f} / {results['like_rate_median']:.6f}")
        print(f"  Comments / views: {results['comment_rate_mean']:.6f} / {results['comment_rate_median']:.6f}")

    if "title_len_mean" in results:
        print("\n--- Title length (chars) ---")
        print(f"  Mean: {results['title_len_mean']:.1f}, Median: {results['title_len_median']:.1f}")
        print(f"  Min: {results['title_len_min']}, Max: {results['title_len_max']}")

    if "published_min" in results:
        print("\n--- Publish date range ---")
        print(f"  From: {results['published_min']}")
        print(f"  To:   {results['published_max']}")
        if "videos_per_year" in results:
            print("  Count by year:", results["videos_per_year"])

    if "trending_min" in results:
        print("\n--- Trending date range ---")
        print(f"  From: {results['trending_min']}")
        print(f"  To:   {results['trending_max']}")

    if "category_counts" in results:
        print("\n--- Top category IDs (video count) ---")
        for cat, count in list(results["category_counts"].items())[:15]:
            print(f"  {cat}: {count:,}")

    if "top_channels" in results:
        print("\n--- Top 30 channels (video count) ---")
        for ch, count in list(results["top_channels"].items())[:30]:
            print(f"  {ch}: {count:,}")

    print("\n" + "=" * 60)


def write_report(path: Path, results: dict, df: pd.DataFrame) -> None:
    """Write a text report to file."""
    with open(path, "w", encoding="utf-8") as f:
        import io
        old = sys.stdout
        sys.stdout = f
        try:
            print_analysis(results, df)
        finally:
            sys.stdout = old
    print(f"Report written to: {path}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Analyze initial YouTube trending CSV")
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=DEFAULT_CSV,
        help=f"Input CSV path (default: {DEFAULT_CSV.name})",
    )
    parser.add_argument(
        "--report", "-r",
        type=Path,
        default=None,
        help="Optional path to write text report",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        print("Run load_youtube_trending.py first to create the CSV.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading: {args.input}", file=sys.stderr)
    df = load_csv(args.input)
    print(f"Loaded {len(df):,} rows.", file=sys.stderr)

    results = run_analysis(df)
    print_analysis(results, df)

    if args.report is not None:
        write_report(args.report, results, df)


if __name__ == "__main__":
    main()
