#!/usr/bin/env python3
"""
Pipeline: download YouTube trending data, then filter with Gemini.

1. Runs load_youtube_trending.py → saves original CSV (US_youtube_trending_data.csv).
2. Runs filter_videos_gemini.py on that CSV → saves filtered + removed CSVs.

Resulting files:
  - US_youtube_trending_data.csv  (original, from Kaggle)
  - videos_filtered.csv           (kept videos)
  - videos_removed.csv            (removed videos + reasons)

Usage:
  python pipeline_download_and_filter.py [--limit N] [--skip-download]
"""

import argparse
import subprocess
import sys
from pathlib import Path

# `clean` directory (this file) and project root (parent of `clean`)
CLEAN_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CLEAN_DIR.parent

# Scripts live in different directories after refactor
LOAD_SCRIPT = PROJECT_ROOT / "load_youtube_trending.py"
FILTER_SCRIPT = CLEAN_DIR / "filter_videos_gemini.py"

# CSVs are stored at the project root
ORIGINAL_CSV = PROJECT_ROOT / "US_youtube_trending_data.csv"
FILTERED_CSV = PROJECT_ROOT / "videos_filtered.csv"
REMOVED_CSV = PROJECT_ROOT / "videos_removed.csv"


def main():
    parser = argparse.ArgumentParser(description="Download trending data and filter with Gemini")
    parser.add_argument(
        "--limit", "-n",
        type=int,
        default=0,
        help="Max videos to filter (0 = all). Only applies to filter step.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download step; use existing US_youtube_trending_data.csv",
    )
    args = parser.parse_args()

    if not LOAD_SCRIPT.exists() or not FILTER_SCRIPT.exists():
        print(
            "Error: Expected load_youtube_trending.py at project root and "
            "filter_videos_gemini.py in the clean/ folder.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Step 1: Download and save original CSV
    if not args.skip_download:
        print("Step 1: Downloading YouTube trending data and saving original CSV...", file=sys.stderr)
        r = subprocess.run([sys.executable, str(LOAD_SCRIPT)], cwd=str(PROJECT_ROOT))
        if r.returncode != 0:
            print("Download step failed.", file=sys.stderr)
            sys.exit(r.returncode)
        print("", file=sys.stderr)
    else:
        if not ORIGINAL_CSV.exists():
            print("Error: --skip-download used but US_youtube_trending_data.csv not found.", file=sys.stderr)
            sys.exit(1)
        print("Step 1: Skipping download (using existing CSV).", file=sys.stderr)

    # Step 2: Filter with Gemini
    print("Step 2: Filtering with Gemini...", file=sys.stderr)
    filter_cmd = [sys.executable, str(FILTER_SCRIPT), str(ORIGINAL_CSV)]
    if args.limit > 0:
        filter_cmd.extend(["--limit", str(args.limit)])
    r = subprocess.run(filter_cmd, cwd=str(PROJECT_ROOT))
    if r.returncode != 0:
        print("Filter step failed.", file=sys.stderr)
        sys.exit(r.returncode)

    print("", file=sys.stderr)
    print("Pipeline done. Outputs:", file=sys.stderr)
    print(f"  Original:  {ORIGINAL_CSV}", file=sys.stderr)
    print(f"  Filtered: {FILTERED_CSV}", file=sys.stderr)
    print(f"  Removed:  {REMOVED_CSV}", file=sys.stderr)


if __name__ == "__main__":
    main()
