#!/usr/bin/env python3
"""
Retry previously-recorded *retryable* transcript fetch failures.

This script:
- Reads a JSON list of retryable failed video_ids (e.g. transcript_failures_debug.json).
- Looks up those video_ids in the current CSV (data_viral_titles.csv) to get title/description.
- Skips any IDs already present in an existing output JSONL (so it is safe to re-run).
- Writes a small temporary CSV and calls `scripts/fetch_transcripts.py` to retry.

Typical usage (defaults are set for your repo paths):
  python scripts/retry_failed_transcripts.py
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set


def _safe_strip(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip()


def _load_json_list(path: Path) -> List[str]:
    if not path.exists():
        raise SystemExit(f"Failures JSON not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise SystemExit(f"Could not parse failures JSON: {path} ({e})")
    if not isinstance(data, list):
        raise SystemExit(f"Failures JSON must be a list: {path}")
    out: List[str] = []
    for x in data:
        s = _safe_strip(x)
        if s:
            out.append(s)
    # de-dupe preserving order
    seen: Set[str] = set()
    deduped: List[str] = []
    for vid in out:
        if vid in seen:
            continue
        seen.add(vid)
        deduped.append(vid)
    return deduped


def _load_processed_ids_from_jsonl(jsonl_path: Path) -> Set[str]:
    processed: Set[str] = set()
    if not jsonl_path.exists():
        return processed
    with jsonl_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            vid = _safe_strip(obj.get("video_id"))
            if vid:
                processed.add(vid)
    return processed


def _iter_csv_rows(csv_path: Path) -> Iterable[Dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "video_id" not in reader.fieldnames:
            raise SystemExit("CSV must include a 'video_id' column")
        if "title" not in reader.fieldnames:
            raise SystemExit("CSV must include a 'title' column")
        for row in reader:
            yield dict(row)


def retry_failed(
    csv_input: Path,
    failures_json: Path,
    append_jsonl: Path,
    max_workers: int,
    debug: bool,
    out_summary_json: Path,
    out_failures_json: Path,
    temp_csv: Path,
) -> None:
    if not csv_input.exists():
        raise SystemExit(f"CSV not found: {csv_input}")

    failures = _load_json_list(failures_json)
    if not failures:
        print(f"No failures found in: {failures_json}")
        return

    processed = _load_processed_ids_from_jsonl(append_jsonl)
    remaining_failures = [vid for vid in failures if vid not in processed]

    print(f"Failures in file: {len(failures):,}")
    print(f"Already present in output JSONL: {len(processed):,}")
    print(f"Failures to retry (not already in JSONL): {len(remaining_failures):,}")

    if not remaining_failures:
        print("Nothing to retry.")
        return

    want: Set[str] = set(remaining_failures)
    selected_rows: List[Dict[str, str]] = []

    for row in _iter_csv_rows(csv_input):
        vid = _safe_strip(row.get("video_id"))
        if vid and vid in want:
            selected_rows.append(
                {
                    "video_id": vid,
                    "title": _safe_strip(row.get("title")),
                    "description": _safe_strip(row.get("description")),
                }
            )

    found_ids = {r["video_id"] for r in selected_rows}
    missing_from_csv = [vid for vid in remaining_failures if vid not in found_ids]
    if missing_from_csv:
        print(f"Note: {len(missing_from_csv):,} failure ids were not found in current CSV and will be skipped.")

    if not selected_rows:
        print("No retry rows could be built from the current CSV; nothing to do.")
        return

    temp_csv.parent.mkdir(parents=True, exist_ok=True)
    with temp_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video_id", "title", "description"])
        writer.writeheader()
        writer.writerows(selected_rows)

    append_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_summary_json.parent.mkdir(parents=True, exist_ok=True)
    out_failures_json.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python",
        "scripts/fetch_transcripts.py",
        "--input",
        str(temp_csv),
        "--jsonl-output",
        str(append_jsonl),
        "--output",
        str(out_summary_json),
        "--failed-ids-output",
        str(out_failures_json),
        "--max-workers",
        str(max(1, int(max_workers))),
    ]
    if debug:
        cmd.append("--debug")

    print("Running:", " ".join(f'"{c}"' if " " in c else c for c in cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Retry retryable transcript failures and append results to a JSONL.")
    p.add_argument(
        "--csv-input",
        default=r"c:\Users\revaa\viral2\data_viral_titles.csv",
        help="Current CSV to look up title/description for video_ids.",
    )
    p.add_argument(
        "--failures-json",
        default=r"c:\Users\revaa\viral2\data\transcript_failures_debug.json",
        help="JSON list of retryable failures to retry.",
    )
    p.add_argument(
        "--append-jsonl",
        default=r"c:\Users\revaa\viral2\data\transcript_training_data_final.jsonl",
        help="JSONL to append results to (processed IDs are detected from here).",
    )
    p.add_argument("--max-workers", type=int, default=10, help="Workers for retry fetch_transcripts.py.")
    p.add_argument("--debug", action="store_true", help="Pass --debug to fetch_transcripts.py.")
    p.add_argument(
        "--out-summary-json",
        default=r"c:\Users\revaa\viral2\data\transcript_retry_summary.json",
        help="JSON summary output path for this retry run.",
    )
    p.add_argument(
        "--out-failures-json",
        default=r"c:\Users\revaa\viral2\data\transcript_failures_retry.json",
        help="JSON list output path for failures that are still retryable after this run.",
    )
    p.add_argument(
        "--temp-csv",
        default=r"c:\Users\revaa\viral2\data\retry_failures.csv",
        help="Temporary CSV path used to feed fetch_transcripts.py.",
    )
    args = p.parse_args()

    retry_failed(
        csv_input=Path(args.csv_input),
        failures_json=Path(args.failures_json),
        append_jsonl=Path(args.append_jsonl),
        max_workers=args.max_workers,
        debug=args.debug,
        out_summary_json=Path(args.out_summary_json),
        out_failures_json=Path(args.out_failures_json),
        temp_csv=Path(args.temp_csv),
    )


if __name__ == "__main__":
    main()

