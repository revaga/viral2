#!/usr/bin/env python3
"""
Data cleaning: remove specified channels and high-volume channels from the YouTube trending CSV.

Removes:
  1. Channels in the hardcoded removal list
  2. Channels with more than N viral videos (default: 300)
  3. Videos with "Official Video" in the title

Reads the initial CSV and writes the cleaned data to a new file (original is unchanged).

Usage:
  python clean_data.py [input.csv] [--output cleaned.csv] [--removed removed_channels.csv]
  python clean_data.py --max-videos 300   # override threshold
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# This script is in clean/, but the raw CSV lives at project root.
CLEAN_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CLEAN_DIR.parent
DEFAULT_CSV = PROJECT_ROOT / "US_youtube_trending_data.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "US_youtube_trending_data_cleaned.csv"

# Channels to remove (exact match on channelTitle)
CHANNELS_TO_REMOVE = {
    "NFL",
    "NBA",
    "NBC Sports",
    "ESPN",
    "Saturday Night Live",
    "HYBE LABELS",
    "Marvel Entertainment",
    "JYP Entertainment",
    "DAZN Boxing",
    "Netflix",
    "Disney",
    "SpaceX",
    "CBS Sports Golazo",
    "Fortnite",
    "Genshin Impact",
    "ESPN FC",
    "Clash of Clans",
    "WWE",
    "FORMULA 1",
    "Warner Bros. Pictures",
    "America's Got Talent",
    "Nintendo",
    "PlayStation",
    "Apple",
    "NBA on TNT",
    "UFC - Ultimate Fighting Championship",
    "Universal Pictures / Sony Pictures",
    "Star Wars",
    "BLACKPINK"
    }

# Channels past threshold to keep (still subject to hardcoded list + Official Video removal)
CHANNELS_ALLOWLIST = {
    "Yes Theory",
    "Grian",
    "SSSniperWolf",
    "videogamedunkey",
    "mrnigelng",
    "colinfurze",
    "Dude Perfect",
    "The Game Theorists",
    "SSundee",
    "Veritasium",
    "Ryan Trahan",
    "The Film Theorists",
    "Apex Legends",
    "MrBeast Gaming",
    "The Food Theorists",
    "Marques Brownlee",
    "RDCworld1",
    "Kurzgesagt Γ\x80\x93 In a Nutshell",
    "Markiplier",
    "Tom Scott",

}

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
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return pd.read_csv(f, on_bad_lines="skip", engine="python")
    except Exception as e:
        last_err = e
    raise ValueError(f"Could not load {path}: {last_err}") from last_err


def main():
    parser = argparse.ArgumentParser(description="Remove specified channels from trending CSV")
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=DEFAULT_CSV,
        help=f"Input CSV (default: {DEFAULT_CSV.name})",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output cleaned CSV (default: {DEFAULT_OUTPUT.name})",
    )
    parser.add_argument(
        "--removed",
        type=Path,
        default=None,
        help="Optional: save removed rows to this CSV for audit",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=400,
        help="Remove channels with more than this many videos (default: 300)",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading: {args.input}", file=sys.stderr)
    df = load_csv(args.input)
    n_before = len(df)

    if "channelTitle" not in df.columns:
        print("Error: CSV has no 'channelTitle' column.", file=sys.stderr)
        sys.exit(1)

    # Channels to remove: hardcoded list + channels with > max_videos videos (except allowlist)
    mask_list = df["channelTitle"].isin(CHANNELS_TO_REMOVE)
    channel_counts = df["channelTitle"].value_counts()
    over_threshold = set(channel_counts[channel_counts > args.max_videos].index)
    high_volume = over_threshold - CHANNELS_ALLOWLIST  # allowlist keeps channels past threshold
    mask_high_volume = df["channelTitle"].isin(high_volume)
    # Videos with "Official Video" in title
    mask_official = df["title"].fillna("").str.contains("Official Video", case=False, regex=False)
    mask_remove = mask_list | mask_high_volume | mask_official
    removed = df[mask_remove]
    cleaned = df[~mask_remove]
    n_removed = len(removed)
    n_after = len(cleaned)

    cleaned.to_csv(args.output, index=False, encoding="utf-8")
    print(f"Saved cleaned data to: {args.output}", file=sys.stderr)
    print(f"  Rows before:  {n_before:,}", file=sys.stderr)
    print(f"  Rows removed: {n_removed:,}", file=sys.stderr)
    print(f"  Rows after:   {n_after:,}", file=sys.stderr)
    print(f"  (removed channels with > {args.max_videos} videos)", file=sys.stderr)
    n_official = mask_official.sum()
    if n_official > 0:
        print(f"  (removed {n_official:,} videos with 'Official Video' in title)", file=sys.stderr)

    # Section 1: Removed by channel (threshold or hardcoded list), sorted by viral video count
    mask_channel = mask_list | mask_high_volume
    removed_by_channel = df[mask_channel]
    if len(removed_by_channel) > 0:
        print("\n--- Removed by channel (threshold or hardcoded list) ---", file=sys.stderr)
        ch_counts = removed_by_channel["channelTitle"].value_counts()
        for ch in ch_counts.index:
            count = ch_counts[ch]
            src = "list" if ch in CHANNELS_TO_REMOVE else f">{args.max_videos} videos"
            print(f"  {ch}: {count:,} ({src})", file=sys.stderr)

    # Section 2: Removed due to "Official Video" in title (print titles)
    removed_by_official = df[mask_official]
    if len(removed_by_official) > 0:
        print("\n--- Removed due to 'Official Video' in title ---", file=sys.stderr)
        print(f"  ({len(removed_by_official):,} videos)\n", file=sys.stderr)
        for title in removed_by_official["title"].fillna("(no title)"):
            print(f"  {title}", file=sys.stderr)

    if args.removed is not None and n_removed > 0:
        removed.to_csv(args.removed, index=False, encoding="utf-8")
        print(f"\nRemoved rows saved to: {args.removed}", file=sys.stderr)


if __name__ == "__main__":
    main()
