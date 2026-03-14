#!/usr/bin/env python3
"""
Build a high-quality viral title dataset from a cleaned YouTube trending CSV.

Starting from a pre-cleaned CSV (e.g. data_cleaned.csv from clean_data.py), this script:

1. Computes per-channel virality and caps the number of titles per channel.
2. Applies global engagement-rate thresholds (likes/views, comments/views).
3. Filters out categories and title patterns that are typically brand/news/utility driven.
4. Applies text-based heuristics to focus on curiosity- and conflict-driven titles.
5. Collapses exact and near-duplicate titles, keeping the highest-engagement variant.
6. Optionally restricts to English-like titles using a lightweight heuristic.

The resulting CSV is intended as a strong candidate set for Gemini filtering
in clean/filter_videos_gemini.py.

Usage examples:
  python build_viral_title_dataset.py
  python build_viral_title_dataset.py --input data_cleaned.csv --output data_viral_titles.csv
  python build_viral_title_dataset.py --english-only --max-per-channel 30
  
  python build_viral_title_dataset.py data_cleaned.csv --view-rel-min 1.2 --max-per-channel 100 --like-quantile 0.3 --comment-quantile 0.3 --min-views 5000 --english-only
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Tuple

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = PROJECT_ROOT / "data_cleaned.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data_viral_titles.csv"


def load_csv(path: Path) -> pd.DataFrame:
    """Load CSV with robust encoding and bad-line handling."""
    last_err: Exception | None = None
    for encoding in ("utf-8", "latin-1", "cp1252"):
        try:
            return pd.read_csv(
                path,
                encoding=encoding,
                on_bad_lines="skip",
                engine="python",
            )
        except Exception as e:  # pragma: no cover - defensive
            last_err = e
            continue
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return pd.read_csv(
                f,
                on_bad_lines="skip",
                engine="python",
            )
    except Exception as e:  # pragma: no cover - defensive
        last_err = e
    raise ValueError(f"Could not load {path}: {last_err}") from last_err


def add_engagement_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add like_rate and comment_rate columns (per view)."""
    df = df.copy()
    if "view_count" not in df.columns:
        return df

    views = pd.to_numeric(df["view_count"], errors="coerce")
    views = views.where(views > 0)

    if "likes" in df.columns:
        likes = pd.to_numeric(df["likes"], errors="coerce")
        df["like_rate"] = (likes / views).fillna(0.0)
    else:
        df["like_rate"] = 0.0

    if "comment_count" in df.columns:
        comments = pd.to_numeric(df["comment_count"], errors="coerce")
        df["comment_rate"] = (comments / views).fillna(0.0)
    else:
        df["comment_rate"] = 0.0

    return df


def per_channel_virality_filter(
    df: pd.DataFrame,
    view_rel_min: float = 2.0,
    max_per_channel: int = 20,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Keep only titles that significantly overperform their own channel baseline.

    For each channel:
      - Compute median view_count.
      - Compute view_rel = view_count / median_views.
      - Keep rows with view_rel >= view_rel_min.
      - Cap to top `max_per_channel` rows per channel by view_rel.

    Returns (kept_df, removed_df).
    """
    if "channelTitle" not in df.columns or "view_count" not in df.columns:
        return df, df.iloc[0:0].copy()

    work = df.copy()
    work["view_count"] = pd.to_numeric(work["view_count"], errors="coerce").fillna(0)

    medians = work.groupby("channelTitle")["view_count"].median().replace(0, 1)
    work = work.join(medians.rename("channel_median_views"), on="channelTitle")
    work["view_rel"] = work["view_count"] / work["channel_median_views"]

    # Overperformers
    work = work.sort_values(["channelTitle", "view_rel"], ascending=[True, False])
    work["rank_within_channel"] = work.groupby("channelTitle").cumcount() + 1

    keep_mask = (work["view_rel"] >= view_rel_min) & (
        work["rank_within_channel"] <= max_per_channel
    )

    kept = work[keep_mask].drop(
        columns=["channel_median_views", "view_rel", "rank_within_channel"]
    )
    removed = work[~keep_mask]
    return kept, removed


def engagement_filter(
    df: pd.DataFrame,
    like_quantile: float = 0.5,
    comment_quantile: float = 0.5,
    min_views: int = 10_000,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Keep rows with engagement (likes/views, comments/views) above chosen quantiles.
    Very low-view videos are ignored for threshold computation.
    """
    df = add_engagement_columns(df)
    if "view_count" not in df.columns:
        return df, df.iloc[0:0].copy()

    views = pd.to_numeric(df["view_count"], errors="coerce").fillna(0)
    eligible = df[views >= min_views]
    if eligible.empty:
        return df, df.iloc[0:0].copy()

    like_thr = eligible["like_rate"].quantile(like_quantile)
    comment_thr = eligible["comment_rate"].quantile(comment_quantile)

    keep_mask = (df["like_rate"] >= like_thr) & (df["comment_rate"] >= comment_thr)
    kept = df[keep_mask]
    removed = df[~keep_mask]
    return kept, removed


SPORTS_PATTERNS = [
    "vs.",
    " vs ",
    "highlights",
    "full game",
    "recap",
    "game winner",
    "overtime",
]

NEWS_PATTERNS = [
    "press conference",
    "live:",
    "live |",
    "webcast",
    "briefing",
    "hearing",
    "[full]",
]

CORPORATE_PATTERNS = [
    "official trailer",
    "teaser trailer",
    "announcement trailer",
    "launch event",
    "unpacked",
    "keynote",
    "introducing ",
]

COMMERCE_PATTERNS = [
    "deals",
    "prime day",
    "black friday",
    "coupon",
    "discount",
    "sale",
]


def category_and_pattern_filter(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prefer categories that usually rely on title curiosity and remove obvious
    sports/news/corporate/commerce patterns.
    """
    if "categoryId" not in df.columns:
        base_mask = pd.Series(True, index=df.index)
    else:
        # YouTube category IDs (US). Keep mostly entertainment / creator content.
        preferred_categories = {22, 23, 24, 26, 27, 28, 19}  # People, Comedy, Ent, etc.
        cats = pd.to_numeric(df["categoryId"], errors="coerce")
        base_mask = cats.isin(preferred_categories)

    title_lower = df["title"].fillna("").str.lower()

    def any_pattern(series: pd.Series, patterns: list[str]) -> pd.Series:
        mask = pd.Series(False, index=series.index)
        for p in patterns:
            mask |= series.str.contains(p, na=False)
        return mask

    sports_mask = any_pattern(title_lower, SPORTS_PATTERNS)
    news_mask = any_pattern(title_lower, NEWS_PATTERNS)
    corporate_mask = any_pattern(title_lower, CORPORATE_PATTERNS)
    commerce_mask = any_pattern(title_lower, COMMERCE_PATTERNS)

    pattern_remove = sports_mask | news_mask | corporate_mask | commerce_mask

    keep_mask = base_mask & ~pattern_remove
    kept = df[keep_mask]
    removed = df[~keep_mask]
    return kept, removed


POSITIVE_PHRASES = [
    "i spent",
    "i tried",
    "i survived",
    "i ruined",
    "i built",
    "i made",
    "the truth about",
    "no one tells you",
    "no one is talking about",
    "you won't believe",
    "what happens when",
    "gone wrong",
    "nobody asked",
    "i did",
]

NEGATIVE_GENERIC_PATTERNS = [
    "official trailer",
    "season",
    "episode",
    "ep.",
    "highlights",
    "ft.",
    "feat.",
]


def text_heuristic_filter(
    df: pd.DataFrame,
    min_len: int = 5,
    max_len: int = 120,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Focus on curiosity/conflict titles and downrank generic labels.
    """
    titles = df["title"].fillna("")
    lens = titles.str.len()
    title_lower = titles.str.lower()

    has_question = titles.str.contains(r"\?", regex=True)

    pos_mask = pd.Series(False, index=df.index)
    for p in POSITIVE_PHRASES:
        pos_mask |= title_lower.str.contains(p, na=False)

    neg_generic = pd.Series(False, index=df.index)
    for p in NEGATIVE_GENERIC_PATTERNS:
        neg_generic |= title_lower.str.contains(p, na=False)

    length_ok = (lens >= min_len) & (lens <= max_len)
    keep_mask = length_ok & (has_question | pos_mask) & ~neg_generic

    kept = df[keep_mask]
    removed = df[~keep_mask]
    return kept, removed


NORMALIZE_RE = re.compile(r"[^a-z0-9]+")


def normalize_title(title: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace for near-duplicate detection."""
    t = title.lower()
    t = NORMALIZE_RE.sub(" ", t)
    t = " ".join(t.split())
    return t


def dedupe_titles(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Remove exact and near-duplicate titles (per channel), keeping the row with
    highest view_count (then likes) for each normalized title.
    """
    if "title" not in df.columns:
        return df, df.iloc[0:0].copy()

    work = df.copy()
    work["norm_title"] = work["title"].fillna("").apply(normalize_title)
    work["view_count"] = pd.to_numeric(work.get("view_count", 0), errors="coerce").fillna(0)
    work["likes"] = pd.to_numeric(work.get("likes", 0), errors="coerce").fillna(0)

    # Prefer to dedupe within each channel; if channelTitle missing, treat as global.
    if "channelTitle" in work.columns:
        group_keys = ["channelTitle", "norm_title"]
    else:
        group_keys = ["norm_title"]

    idx_keep = (
        work.sort_values(["view_count", "likes"], ascending=False)
        .groupby(group_keys, as_index=False)
        .head(1)
        .index
    )

    keep_mask = work.index.isin(idx_keep)
    kept = work[keep_mask].drop(columns=["norm_title"])
    removed = work[~keep_mask]
    return kept, removed


ENGLISH_STOPWORDS = {
    "the",
    "and",
    "you",
    "your",
    "for",
    "this",
    "that",
    "with",
    "from",
    "about",
    "what",
    "when",
    "how",
    "why",
    "who",
    "into",
    "over",
    "under",
}


def english_like_mask(titles: pd.Series) -> pd.Series:
    """
    Heuristic: keep titles that look roughly English.
    Conditions:
      - High proportion of ASCII letters, OR
      - Contains at least one common English stopword.
    """
    titles = titles.fillna("")

    def score_one(t: str) -> Tuple[float, bool]:
        if not t:
            return 0.0, False
        ascii_letters = sum(ch.isascii() and ch.isalpha() for ch in t)
        ratio = ascii_letters / max(len(t), 1)
        tokens = {tok.lower() for tok in re.split(r"\W+", t) if tok}
        has_stopword = bool(tokens & ENGLISH_STOPWORDS)
        return ratio, has_stopword

    ratios = []
    has_stop = []
    for t in titles:
        r, h = score_one(t)
        ratios.append(r)
        has_stop.append(h)

    ratio_s = pd.Series(ratios, index=titles.index)
    has_stop_s = pd.Series(has_stop, index=titles.index)
    return (ratio_s >= 0.6) | has_stop_s


def language_filter(
    df: pd.DataFrame,
    english_only: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Optionally restrict to English-like titles."""
    if not english_only or "title" not in df.columns:
        return df, df.iloc[0:0].copy()

    mask = english_like_mask(df["title"])
    kept = df[mask]
    removed = df[~mask]
    return kept, removed


def main() -> None:
    parser = argparse.ArgumentParser(description="Build viral title dataset from cleaned CSV")
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input CSV (default: {DEFAULT_INPUT.name})",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output CSV for viral titles (default: {DEFAULT_OUTPUT.name})",
    )
    parser.add_argument(
        "--view-rel-min",
        type=float,
        default=2.0,
        help="Minimum multiple of channel median views to be considered viral (default: 2.0)",
    )
    parser.add_argument(
        "--max-per-channel",
        type=int,
        default=20,
        help="Max number of titles to keep per channel (default: 20)",
    )
    parser.add_argument(
        "--like-quantile",
        type=float,
        default=0.5,
        help="Global like_rate quantile threshold (default: 0.5 = median)",
    )
    parser.add_argument(
        "--comment-quantile",
        type=float,
        default=0.5,
        help="Global comment_rate quantile threshold (default: 0.5 = median)",
    )
    parser.add_argument(
        "--min-views",
        type=int,
        default=10_000,
        help="Minimum views for a video to be considered for engagement thresholds (default: 10000)",
    )
    parser.add_argument(
        "--english-only",
        action="store_true",
        help="Restrict to titles that look like English based on a simple heuristic",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading cleaned CSV: {args.input}", file=sys.stderr)
    df = load_csv(args.input)
    print(f"  Loaded {len(df):,} rows", file=sys.stderr)

    # 1. Per-channel virality normalization
    df1, removed_chan = per_channel_virality_filter(
        df,
        view_rel_min=args.view_rel_min,
        max_per_channel=args.max_per_channel,
    )
    print(
        f"After per-channel virality filter: {len(df1):,} rows "
        f"(removed {len(removed_chan):,})",
        file=sys.stderr,
    )

    # 2. Engagement thresholds
    df2, removed_eng = engagement_filter(
        df1,
        like_quantile=args.like_quantile,
        comment_quantile=args.comment_quantile,
        min_views=args.min_views,
    )
    print(
        f"After engagement filter: {len(df2):,} rows "
        f"(removed {len(removed_eng):,})",
        file=sys.stderr,
    )

    # 3. Category & pattern filters
    # df3, removed_cat = category_and_pattern_filter(df2)
    # print(
    #     f"After category/pattern filter: {len(df3):,} rows "
    #     f"(removed {len(removed_cat):,})",
    #     file=sys.stderr,
    # )

    # 4. Text heuristics
    # df4, removed_text = text_heuristic_filter(df2)
    # print(
    #     f"After text-heuristic filter: {len(df4):,} rows "
    #     f"(removed {len(removed_text):,})",
    #     file=sys.stderr,
    # )

    # 5. Deduplicate titles
    df5, removed_dupes = dedupe_titles(df2)
    print(
        f"After deduplication: {len(df5):,} rows "
        f"(removed {len(removed_dupes):,})",
        file=sys.stderr,
    )

    # 6. Language filter (optional)
    df6, removed_lang = language_filter(df5, english_only=args.english_only)
    if args.english_only:
        print(
            f"After language filter: {len(df6):,} rows "
            f"(removed {len(removed_lang):,})",
            file=sys.stderr,
        )
    else:
        print("Language filter skipped (english-only not set).", file=sys.stderr)

    # Final output
    args.output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    df6.to_csv(args.output, index=False, encoding="utf-8")
    print(f"\nSaved viral title dataset to: {args.output}", file=sys.stderr)
    print(f"Total kept rows: {len(df6):,}", file=sys.stderr)

    # Simple evaluation hints
    if "channelTitle" in df6.columns:
        top_channels = df6["channelTitle"].value_counts().head(10)
        print("\nTop channels in viral dataset:", file=sys.stderr)
        for ch, count in top_channels.items():
            print(f"  {ch}: {count}", file=sys.stderr)

    if "categoryId" in df6.columns:
        cats = pd.to_numeric(df6["categoryId"], errors="coerce")
        top_cats = cats.value_counts().head(10)
        print("\nTop categories in viral dataset (by ID):", file=sys.stderr)
        for cid, count in top_cats.items():
            print(f"  {int(cid)}: {count}", file=sys.stderr)


if __name__ == "__main__":
    main()

