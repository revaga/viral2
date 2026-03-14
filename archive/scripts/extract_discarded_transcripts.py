#!/usr/bin/env python3
"""
Extract transcripts for videos that are no longer present in the current
data_viral_titles.csv and write them to discarded_transcripts.jsonl.

By default this script:
- Reads the current CSV of videos (data_viral_titles.csv).
- Reads an existing transcripts JSONL file (e.g. transcript_training_data_debug.jsonl).
- Writes:
  - discarded_transcripts.jsonl: lines whose video_id is NOT in the CSV.
  - kept_transcripts.jsonl: lines whose video_id IS in the CSV.

You can then optionally replace your original transcripts file with kept_transcripts.jsonl.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _load_current_video_ids(csv_path: Path) -> set[str]:
    ids: set[str] = set()
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid = str(row.get("video_id", "")).strip()
            if vid:
                ids.add(vid)
    return ids


def extract_discarded_transcripts(
    csv_input: Path,
    transcripts_jsonl: Path,
    discarded_output: Path,
    kept_output: Path,
) -> None:
    if not csv_input.exists():
        raise SystemExit(f"CSV input not found: {csv_input}")
    if not transcripts_jsonl.exists():
        raise SystemExit(f"Transcripts JSONL not found: {transcripts_jsonl}")

    current_ids = _load_current_video_ids(csv_input)
    print(f"Loaded {len(current_ids):,} video_ids from CSV: {csv_input}")

    discarded_output.parent.mkdir(parents=True, exist_ok=True)
    kept_output.parent.mkdir(parents=True, exist_ok=True)

    n_discarded = 0
    n_kept = 0

    with transcripts_jsonl.open("r", encoding="utf-8", errors="replace") as in_f, discarded_output.open(
        "w", encoding="utf-8"
    ) as disc_f, kept_output.open("w", encoding="utf-8") as kept_f:
        for line in in_f:
            raw = line.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except Exception:
                continue
            vid = str(obj.get("video_id", "")).strip()
            if not vid:
                continue
            if vid in current_ids:
                kept_f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                n_kept += 1
            else:
                disc_f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                n_discarded += 1

    print(f"Wrote {n_discarded:,} discarded transcripts to: {discarded_output}")
    print(f"Wrote {n_kept:,} kept transcripts to: {kept_output}")


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Move transcripts for videos not in the current data_viral_titles.csv "
            "into discarded_transcripts.jsonl (and write a kept_transcripts.jsonl)."
        )
    )
    p.add_argument(
        "--csv-input",
        default=r"c:\Users\revaa\viral2\data_viral_titles.csv",
        help="CSV file with current video list (must have a video_id column).",
    )
    p.add_argument(
        "--transcripts-jsonl",
        default=r"c:\Users\revaa\viral2\data\transcript_training_data_debug.jsonl",
        help="Existing transcripts JSONL to split into kept vs discarded.",
    )
    p.add_argument(
        "--discarded-output",
        default=r"c:\Users\revaa\viral2\data\discarded_transcripts.jsonl",
        help="Output JSONL for discarded transcripts (video_ids not in CSV).",
    )
    p.add_argument(
        "--kept-output",
        default=r"c:\Users\revaa\viral2\data\kept_transcripts.jsonl",
        help="Output JSONL for kept transcripts (video_ids present in CSV).",
    )
    args = p.parse_args()

    extract_discarded_transcripts(
        csv_input=Path(args.csv_input),
        transcripts_jsonl=Path(args.transcripts_jsonl),
        discarded_output=Path(args.discarded_output),
        kept_output=Path(args.kept_output),
    )


if __name__ == "__main__":
    main()

