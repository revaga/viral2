#!/usr/bin/env python3
"""
Data cleaning: remove specified channels and high-volume channels from the YouTube trending CSV.

Removes:
  1. Channels in the hardcoded removal list
  2. Channels with more than N viral videos (default: 300)
  3. Channel Categories
  3. Videos with "Official","Official Video", "Official Music Video", " MV ", or "Tiny Desk Concert" in the title
  4. Videos with empty descriptions

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
DEFAULT_OUTPUT = PROJECT_ROOT / "data_cleaned.csv"

# Channels to remove (exact match on channelTitle)
CHANNELS_TO_REMOVE = {
    "NFL",
    "NBA",
    "NPR",
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
    "NPR",
    "NPR Music",
    "adult swim",
    "Architectural Digest",
    "CTV News"
    "News",
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

    # Title-based removals: official, official videos, MVs, Tiny Desk Concerts
    title_series = df["title"].fillna("")
    mask_official_video = title_series.str.contains("Official", case=False, regex=False)
    mask_official_music = title_series.str.contains("Lyric", case=False, regex=False)
    mask_mv = title_series.str.contains(" MV ", case=False, regex=False)
    mask_tiny_desk = title_series.str.contains("Tiny Desk Concert", case=False, regex=False)
    mask_official = mask_official_video | mask_official_music | mask_mv | mask_tiny_desk

    # Description-based removals: empty descriptions (if column exists)
    if "description" in df.columns:
        desc_series = df["description"].fillna("").astype(str).str.strip()
        mask_empty_desc = desc_series.eq("")
    else:
        mask_empty_desc = pd.Series(False, index=df.index)

    # Category-based removals: categoryId == 25 or 10, if column exists
    if "categoryId" in df.columns:
        cat_series = df["categoryId"].astype(str).str.strip()
        mask_cat_25 = cat_series.eq("25")
        mask_cat_10 = cat_series.eq("10")
    else:
        mask_cat_25 = pd.Series(False, index=df.index)
        mask_cat_10 = pd.Series(False, index=df.index)
    mask_remove = mask_list | mask_high_volume | mask_official | mask_empty_desc | mask_cat_25 | mask_cat_10
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
        print(
            f"  (removed {n_official:,} videos with "
            "'Official','Official Video', 'Official Music Video', ' MV ', or 'Tiny Desk Concert' in title)",
            file=sys.stderr,
        )
    n_empty_desc = mask_empty_desc.sum()
    if n_empty_desc > 0:
        print(f"  (removed {n_empty_desc:,} videos with empty descriptions)", file=sys.stderr)
    n_cat_25 = mask_cat_25.sum()
    if n_cat_25 > 0:
        print(f"  (removed {n_cat_25:,} videos in category 25)", file=sys.stderr)
    n_cat_10 = mask_cat_10.sum()
    if n_cat_10 > 0:
        print(f"  (removed {n_cat_10:,} videos in category 10)", file=sys.stderr)

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

    # Section 2: Removed due to title-based filters (print titles)
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
