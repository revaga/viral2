#!/usr/bin/env python3
"""
Cap transcript JSONL to at most N most viral videos per channel.

Reads data/transcript_training_data_final.jsonl and a viral titles CSV (e.g.
data_viral_titles.csv) that provides video_id -> channel and view_count.
For each channel with more than max_per_channel videos, keeps only the
max_per_channel videos with the highest view_count; others are dropped.

Videos that do not appear in the viral CSV are kept as-is (no channel cap).
Output is written to a new JSONL (default: overwrite with a temp file then rename).
"""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_JSONL = PROJECT_ROOT / "data" / "transcript_training_data_final.jsonl"
DEFAULT_CSV = PROJECT_ROOT / "data_viral_titles.csv"
DEFAULT_MAX_PER_CHANNEL = 20


def _safe_strip(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip()


def load_video_channel_views(csv_path: Path) -> Dict[str, Tuple[str, int]]:
    """
    Load viral CSV: video_id -> (channel_id, view_count).
    Uses channelId for stability; falls back to channelTitle if channelId missing.
    """
    result: Dict[str, Tuple[str, int]] = {}
    if not csv_path.exists():
        return result
    with csv_path.open("r", encoding="utf-8", newline="", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid = _safe_strip(row.get("video_id"))
            if not vid:
                continue
            channel = _safe_strip(row.get("channelId")) or _safe_strip(row.get("channelTitle")) or ""
            try:
                vc = int(float(row.get("view_count") or 0))
            except (TypeError, ValueError):
                vc = 0
            result[vid] = (channel, vc)
    return result


def iter_jsonl(path: Path):
    """Yield dicts from JSONL."""
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except Exception:
                continue
            if isinstance(obj, dict):
                yield obj


def main() -> None:
    p = argparse.ArgumentParser(
        description="Cap transcript JSONL to at most N most viral videos per channel."
    )
    p.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_JSONL,
        help="Input transcript JSONL.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSONL (default: overwrite --input after writing to temp).",
    )
    p.add_argument(
        "--viral-csv",
        type=Path,
        default=DEFAULT_CSV,
        help="CSV with video_id, channelId/channelTitle, view_count.",
    )
    p.add_argument(
        "--max-per-channel",
        type=int,
        default=DEFAULT_MAX_PER_CHANNEL,
        help="Max viral videos to keep per channel (default: 20).",
    )
    args = p.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input not found: {args.input}")

    video_to_channel_views = load_video_channel_views(args.viral_csv)
    print(f"Loaded {len(video_to_channel_views):,} video_id -> (channel, views) from {args.viral_csv}")

    # Pass 1: collect all rows and their (channel, view_count)
    channel_to_videos: Dict[str, List[Tuple[int, str, Dict[str, Any]]]] = {}  # channel -> [(view_count, video_id, row)]
    no_channel_rows: List[Dict[str, Any]] = []  # videos not in CSV: keep all
    total = 0
    for row in iter_jsonl(args.input):
        total += 1
        vid = _safe_strip(row.get("video_id"))
        if not vid:
            continue
        info = video_to_channel_views.get(vid)
        if info is None:
            no_channel_rows.append(row)
            continue
        channel, view_count = info
        if not channel:
            no_channel_rows.append(row)
            continue
        if channel not in channel_to_videos:
            channel_to_videos[channel] = []
        channel_to_videos[channel].append((view_count, vid, row))

    # For each channel, keep top max_per_channel by view_count
    kept_rows: List[Dict[str, Any]] = list(no_channel_rows)
    dropped = 0
    for channel, items in channel_to_videos.items():
        items.sort(key=lambda x: (-x[0], x[1]))  # view_count desc, then video_id stable
        to_keep = items[: args.max_per_channel]
        to_drop = items[args.max_per_channel:]
        for _, _, row in to_keep:
            kept_rows.append(row)
        dropped += len(to_drop)

    out_path = args.output if args.output is not None else args.input
    if out_path == args.input:
        # Write to temp then replace
        fd, tmp_path = tempfile.mkstemp(
            prefix="cap_transcripts_", suffix=".jsonl", dir=out_path.parent
        )
        tmp_path = Path(tmp_path)
        try:
            with open(fd, "w", encoding="utf-8") as out_f:
                for row in kept_rows:
                    out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            tmp_path.replace(out_path)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as out_f:
            for row in kept_rows:
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Total rows read: {total:,}")
    print(f"Videos without channel (kept as-is): {len(no_channel_rows):,}")
    print(f"Videos dropped (over cap per channel): {dropped:,}")
    print(f"Rows written: {len(kept_rows):,}")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
