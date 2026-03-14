#!/usr/bin/env python3
"""
Merge multiple transcripts JSONL files into a single deduplicated JSONL.

Primary goal:
- Combine `transcript_training_data_debug.jsonl` and `kept_transcripts.jsonl`.
- Deduplicate by `video_id`.
- Prefer rows with a non-empty `full_transcript`.
- Optionally restrict output to video_ids currently present in `data_viral_titles.csv`.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Set, Tuple


def _safe_strip(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip()


def _load_current_video_ids(csv_path: Path) -> Set[str]:
    ids: Set[str] = set()
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid = _safe_strip(row.get("video_id"))
            if vid:
                ids.add(vid)
    return ids


def _iter_jsonl_rows(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return
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


def _score_row(row: Dict[str, Any]) -> Tuple[int, int]:
    """
    Higher is better.
    - First component: has non-empty full_transcript (1/0)
    - Second component: transcript length (capped-ish via raw len)
    """
    ft = _safe_strip(row.get("full_transcript"))
    has = 1 if ft else 0
    return (has, len(ft))


def merge_transcripts(
    inputs: list[Path],
    output: Path,
    csv_filter: Optional[Path],
    discarded_output: Optional[Path],
) -> None:
    allowed_ids: Optional[Set[str]] = None
    if csv_filter is not None:
        if not csv_filter.exists():
            raise SystemExit(f"CSV filter not found: {csv_filter}")
        allowed_ids = _load_current_video_ids(csv_filter)
        print(f"Loaded {len(allowed_ids):,} allowed video_ids from CSV: {csv_filter}")

    best_by_id: Dict[str, Dict[str, Any]] = {}
    discarded_by_id: Dict[str, Dict[str, Any]] = {}
    seen_rows = 0
    kept_rows = 0
    skipped_not_allowed = 0
    skipped_missing_vid = 0
    replaced = 0
    discarded_replaced = 0

    for in_path in inputs:
        if not in_path.exists():
            print(f"Skipping missing input: {in_path}")
            continue
        for row in _iter_jsonl_rows(in_path):
            seen_rows += 1
            vid = _safe_strip(row.get("video_id"))
            if not vid:
                skipped_missing_vid += 1
                continue
            if allowed_ids is not None and vid not in allowed_ids:
                skipped_not_allowed += 1
                # When filtering, "move" not-allowed rows into discarded output.
                if discarded_output is not None:
                    cur_d = discarded_by_id.get(vid)
                    if cur_d is None:
                        discarded_by_id[vid] = row
                    else:
                        if _score_row(row) > _score_row(cur_d):
                            discarded_by_id[vid] = row
                            discarded_replaced += 1
                continue

            if vid not in best_by_id:
                best_by_id[vid] = row
                kept_rows += 1
                continue

            cur = best_by_id[vid]
            if _score_row(row) > _score_row(cur):
                best_by_id[vid] = row
                replaced += 1

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as out_f:
        for vid in sorted(best_by_id.keys()):
            out_f.write(json.dumps(best_by_id[vid], ensure_ascii=False) + "\n")

    if allowed_ids is not None and discarded_output is not None:
        discarded_output.parent.mkdir(parents=True, exist_ok=True)
        with discarded_output.open("w", encoding="utf-8") as disc_f:
            for vid in sorted(discarded_by_id.keys()):
                disc_f.write(json.dumps(discarded_by_id[vid], ensure_ascii=False) + "\n")

    non_empty = sum(1 for r in best_by_id.values() if _safe_strip(r.get("full_transcript")))
    print(f"Inputs: {len(inputs)}")
    print(f"Rows read (valid JSON objects): {seen_rows:,}")
    print(f"Skipped (missing video_id): {skipped_missing_vid:,}")
    if allowed_ids is not None:
        print(f"Skipped (not in CSV): {skipped_not_allowed:,}")
    print(f"Unique video_ids written: {len(best_by_id):,}")
    print(f"Rows with non-empty transcripts: {non_empty:,}")
    print(f"Replacements (better transcript wins): {replaced:,}")
    print(f"Wrote merged JSONL: {output}")
    if allowed_ids is not None and discarded_output is not None:
        print(f"Unique video_ids discarded: {len(discarded_by_id):,}")
        print(f"Discarded replacements (better transcript wins): {discarded_replaced:,}")
        print(f"Wrote discarded JSONL: {discarded_output}")


def main() -> None:
    p = argparse.ArgumentParser(description="Merge transcript JSONL files (dedupe by video_id, prefer non-empty).")
    p.add_argument(
        "--inputs",
        nargs="+",
        default=[
            r"c:\Users\revaa\viral2\data\transcript_training_data_debug.jsonl",
            r"c:\Users\revaa\viral2\data\kept_transcripts.jsonl",
        ],
        help="Input JSONL files to merge (later inputs can replace earlier if better).",
    )
    p.add_argument(
        "--output",
        default=r"c:\Users\revaa\viral2\data\transcript_training_data_final.jsonl",
        help="Output merged JSONL path.",
    )
    p.add_argument(
        "--csv-filter",
        default=r"c:\Users\revaa\viral2\data_viral_titles.csv",
        help="If set, restrict output to video_ids present in this CSV.",
    )
    p.add_argument(
        "--no-csv-filter",
        action="store_true",
        help="Do not filter by CSV; keep any video_id present in inputs.",
    )
    p.add_argument(
        "--discarded-output",
        default=r"c:\Users\revaa\viral2\data\discarded_transcripts.jsonl",
        help=(
            "When CSV filtering is enabled, write transcripts whose video_id is NOT in the CSV "
            "to this JSONL (deduped by video_id)."
        ),
    )
    args = p.parse_args()

    csv_filter = None if args.no_csv_filter else Path(args.csv_filter)
    merge_transcripts(
        inputs=[Path(x) for x in args.inputs],
        output=Path(args.output),
        csv_filter=csv_filter,
        discarded_output=None if csv_filter is None else Path(args.discarded_output),
    )


if __name__ == "__main__":
    main()

