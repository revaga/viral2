#!/usr/bin/env python3
"""
Filter videos from a CSV using Gemini LLM to determine which videos should be
kept based on quality criteria (e.g. for training a high-CTR title model).

Usage:
  Set GEMINI_API_KEY or GOOGLE_API_KEY in .env, then:
  python filter_videos_gemini.py [input.csv] [--output filtered.csv] [--limit N]

"""

import os
import sys
import time
from pathlib import Path
from typing import Tuple

import pandas as pd
from tqdm import tqdm

# Load .env if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Gemini: use google.genai (new SDK)
try:
    from google import genai
except ImportError:
    print("Error: install with: pip install google-genai python-dotenv", file=sys.stderr)
    sys.exit(1)

# Paths: script lives in clean/, data lives at project root
CLEAN_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CLEAN_DIR.parent

# Configuration
# After rule-based cleaning (clean_data.py + build_viral_title_dataset.py),
# this script defaults to running on the curated viral-title dataset.
DEFAULT_CSV = PROJECT_ROOT / "data_viral_titles.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "videos_filtered.csv"
DEFAULT_REMOVED = PROJECT_ROOT / "videos_removed.csv"

FILTERING_CRITERIA = """
You are an expert data evaluator helping filter a YouTube dataset to train a high-CTR (Click-Through Rate) title generation model. 

Your objective is to isolate videos that achieved virality through **pure title skill, psychology, and curiosity**, rather than brand power, news value, utility, or established IP.

Analyze the provided video Title (and Channel name, if provided) and determine if it should be KEPT or REMOVED based on the following criteria:

### ❌ REMOVE CRITERIA (Contamination)
1. Brand / Channel Power: Official launches, brand channels, or institutional uploads that would get clicks regardless of the title (e.g., \"Apple Event\", \"Galaxy Unpacked\", \"Introducing Windows 11\", \"UFC 300 Full Fight\").
2. Livestreams/News Broadcasts: Mission replays, news hits, press conferences, hearings, election coverage, or sports highlights/recaps that are primarily **news/utility** rather than curiosity-driven stories.
3. \"Deal Guy\" / Pure Commerce Content: Shopping listicles and deal roundups (e.g., \"Top 50 Amazon Prime Day Deals 2024\", \"Target Black Friday\"), where users click for savings, not because of narrative tension.
4. Simple \"Official\" Announcements: Basic product or feature announcements with little conflict, mystery, or narrative (e.g., \"The new MacBook Pro\", \"Introducing Apple Vision Pro\").
5. Official Trailers / Music Videos: Video game, movie, show trailers and official music videos that get views from pre-existing IP or fanbases (e.g., \"Hogwarts Legacy Official Trailer\", \"Official Music Video\", \"Vevo\").
6. Low-Effort Utility Uploads: Clips whose titles are mostly labels, episode codes, or SEO keywords (e.g., \"Season 3 Episode 5\", \"Full Match\", \"EP. 12\", date/time stamps).

### ✅ KEEP CRITERIA (Gold Standard)
1. Conflict/Controversy: Titles that generate curiosity through debate, stakes, or tension (e.g., \"NVIDIA just made EVERYTHING ELSE obsolete\", \"I Tested The World's Most Dangerous Gadgets\").
2. Absurdity/Engineering Feats: Highlighting unusual, extreme, or impressive efforts (e.g., \"I Invented Three New Incredible Ways to Die\", \"4000° PLASMA LIGHTSABER BUILD\").
3. The Big Question/Hidden Truth: Posing interesting questions or revealing insights (e.g., \"Is The Metric System Actually Better?\", \"How Humans Lost Their Fur\", \"The Real Reason Planes Are White\").
4. High-Effort Commentary/Reviews: Creator-driven takes that clearly add perspective, narrative, or tension, rather than just naming the product (e.g., \"The M4 iPad Pro is a Beautiful Mess\" -> KEEP. \"iPad Pro M4 Unboxing\" -> REMOVE).

### SCORING (IN YOUR HEAD)
Internally, assign a \"title skill\" score from 1–5 where:
  - 1 = boring label, almost no curiosity
  - 3 = decent YouTube title but not exceptional
  - 5 = strong, curiosity-driven, emotionally or intellectually compelling title

You do NOT need to output the numeric score, but you SHOULD let it influence your KEEP vs REMOVE decision. Only KEEP videos with a title skill of 3–5.

### OUTPUT FORMAT
You must respond strictly in this format:
[KEEP or REMOVE] - [One brief sentence explaining the primary reason based on the criteria above.]

### EXAMPLES
Input: Title: "SpaceX Starship Super Heavy Project Update"
Output: REMOVE - This is essentially a company update that relies on brand/news value instead of title curiosity.

Input: Title: "I Spent 100 Days in a Secret Underground Bunker"
Output: KEEP - This title uses an extreme, unusual challenge to create strong curiosity.

Input: Title: "Top 10 Best Black Friday Tech Deals 2025"
Output: REMOVE - This is pure utility/commerce content rather than narrative or curiosity-driven storytelling.

### INPUT TO EVALUATE:
"""

# Default model for generate_content (google.genai)
GEMINI_MODEL = "gemma-3-12b-it"


def get_gemini_response(prompt: str) -> str:
    """Call Gemini and return the raw text response (uses google.genai Client)."""
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Set GEMINI_API_KEY or GOOGLE_API_KEY in your environment or .env file"
        )
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )
    if not response or not response.text:
        raise RuntimeError("Empty response from Gemini")
    return response.text.strip()


def should_keep_video(title: str, channel_title: str, tags: str) -> Tuple[bool, str]:
    """
    Uses Gemini to determine if a video should be kept.
    Returns (should_keep: bool, reason: str)
    """
    tags_str = tags if tags and str(tags).strip() and str(tags) != "[None]" else "No tags"
    channel_str = channel_title if channel_title else "Unknown channel"

    prompt = f"""
{FILTERING_CRITERIA}

Video Title: "{title}"
Channel: {channel_str}
Tags: {tags_str}

Should this video be KEPT or REMOVED?
"""

    try:
        response = get_gemini_response(prompt)
        response_upper = response.upper()
        if "REMOVE" in response_upper or response_upper.startswith("REMOVE"):
            return False, response
        if "KEEP" in response_upper or response_upper.startswith("KEEP"):
            return True, response
        return True, f"UNCLEAR: {response}"
    except Exception as e:
        print(f"ERROR processing video '{title}': {e}", file=sys.stderr)
        return True, f"ERROR: {e}"


def infer_column(df: pd.DataFrame, *candidates: str) -> str:
    """Return first column name that exists (case-insensitive)."""
    cols_lower = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in cols_lower:
            return cols_lower[name.lower()]
    return ""


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Filter videos with Gemini")
    parser.add_argument(
        "input_csv",
        nargs="?",
        default=str(DEFAULT_CSV),
        help=f"Input CSV path (default: {DEFAULT_CSV.name})",
    )
    parser.add_argument(
        "--output", "-o",
        default=str(DEFAULT_OUTPUT),
        help=f"Output CSV for kept videos (default: {DEFAULT_OUTPUT.name})",
    )
    parser.add_argument(
        "--removed",
        default=str(DEFAULT_REMOVED),
        help="CSV path for removed videos with reasons (default: videos_removed.csv)",
    )
    parser.add_argument(
        "--limit", "-n",
        type=int,
        default=0,
        help="Max number of rows to process (0 = all)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.2,
        help="Seconds between API calls to avoid rate limits (default: 0.2)",
    )
    args = parser.parse_args()

    csv_path = Path(args.input_csv)
    if not csv_path.is_absolute():
        csv_path = PROJECT_ROOT / csv_path
    if not csv_path.exists():
        print(f"Error: CSV file not found at {csv_path}", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output)
    removed_path = Path(args.removed)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    if not removed_path.is_absolute():
        removed_path = PROJECT_ROOT / removed_path

    print(f"Reading CSV: {csv_path}", file=sys.stderr)
    try:
        df = pd.read_csv(csv_path, low_memory=False, encoding="utf-8", on_bad_lines="skip")
    except Exception:
        df = pd.read_csv(csv_path, low_memory=False, encoding="latin-1", on_bad_lines="skip")

    title_col = infer_column(df, "title", "Title")
    channel_col = infer_column(df, "channelTitle", "channel_title", "channelTitle")
    tags_col = infer_column(df, "tags", "Tags")

    if not title_col:
        print("Error: CSV must have a 'title' (or 'Title') column", file=sys.stderr)
        sys.exit(1)
    if not channel_col:
        channel_col = None  # will use "Unknown channel"
    if not tags_col:
        tags_col = None  # will use "No tags"

    SAVE_EVERY_N = 50

    n_total = len(df)
    if args.limit and args.limit > 0:
        df = df.head(args.limit)
    n_process = len(df)
    print(f"Processing {n_process} videos (of {n_total} total), saving every {SAVE_EVERY_N}", file=sys.stderr)

    keep_decisions = []
    reasons = []

    def save_progress(n_processed: int) -> None:
        """Write current kept/removed rows to CSVs."""
        current = df.iloc[:n_processed].copy()
        current["keep"] = keep_decisions
        current["reason"] = reasons
        kept = current[current["keep"]].drop(columns=["keep", "reason"])
        removed = current[~current["keep"]]
        kept.to_csv(output_path, index=False)
        removed.to_csv(removed_path, index=False)

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Filtering videos"):
        title = row[title_col]
        channel_title = row[channel_col] if channel_col else ""
        tags = row[tags_col] if tags_col else ""
        if pd.isna(title):
            title = ""
        if pd.isna(channel_title):
            channel_title = ""
        if pd.isna(tags):
            tags = ""

        should_keep, reason = should_keep_video(str(title), str(channel_title), str(tags))
        keep_decisions.append(should_keep)
        reasons.append(reason)

        if len(keep_decisions) % SAVE_EVERY_N == 0:
            save_progress(len(keep_decisions))

        time.sleep(args.delay)

    # Final save (covers remainder if count not divisible by SAVE_EVERY_N)
    save_progress(len(keep_decisions))

    kept_count = sum(keep_decisions)
    removed_count = len(keep_decisions) - kept_count
    print(f"\nKept: {kept_count} videos -> {output_path}", file=sys.stderr)
    print(f"Removed: {removed_count} videos -> {removed_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
