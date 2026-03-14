#!/usr/bin/env python3
"""
Retry all rows whose full_transcript is empty in a transcripts JSONL.

Workflow:
1. Scan an existing transcripts JSONL (e.g. transcript_training_data_final.jsonl)
   and select rows with empty full_transcript.
2. Build a temporary CSV of those video_ids/title/description.
3. Call scripts/fetch_transcripts.py on that CSV to retry fetching.
4. Merge the original JSONL + the retry JSONL, preferring rows with non-empty
   full_transcript (using the same scoring as merge_transcripts_jsonl.py).
5. Inspect the retry JSON summary and write an unretryable report that records
   which video_ids still have empty full_transcript and their transcript_source
   (e.g. missing, private, blocked_by_youtube), for analysis.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _safe_strip(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip()


def _iter_jsonl_rows(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"Input JSONL not found: {path}")
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


def _write_temp_csv_from_empty_rows(
    input_jsonl: Path,
    temp_csv: Path,
) -> int:
    rows: List[Dict[str, str]] = []
    for obj in _iter_jsonl_rows(input_jsonl):
        vid = _safe_strip(obj.get("video_id"))
        if not vid:
            continue
        ft = _safe_strip(obj.get("full_transcript"))
        if ft:
            continue
        rows.append(
            {
                "video_id": vid,
                "title": _safe_strip(obj.get("title")),
                "description": _safe_strip(obj.get("description")),
            }
        )

    if not rows:
        return 0

    temp_csv.parent.mkdir(parents=True, exist_ok=True)
    with temp_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video_id", "title", "description"])
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def _run_fetch_transcripts(
    temp_csv: Path,
    retry_jsonl: Path,
    retry_summary_json: Path,
    retry_failures_json: Path,
    max_workers: int,
    debug: bool,
) -> None:
    retry_jsonl.parent.mkdir(parents=True, exist_ok=True)
    retry_summary_json.parent.mkdir(parents=True, exist_ok=True)
    retry_failures_json.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python",
        "scripts/fetch_transcripts.py",
        "--input",
        str(temp_csv),
        "--jsonl-output",
        str(retry_jsonl),
        "--output",
        str(retry_summary_json),
        "--failed-ids-output",
        str(retry_failures_json),
        "--max-workers",
        str(max(1, int(max_workers))),
    ]
    if debug:
        cmd.append("--debug")

    print("Running:", " ".join(f'"{c}"' if " " in c else c for c in cmd))
    subprocess.run(cmd, check=True)


def _merge_original_and_retry(
    original_jsonl: Path,
    retry_jsonl: Path,
    merged_output: Path,
) -> None:
    """
    Use the merge_transcripts_jsonl.merge_transcripts helper (sibling module in
    the same directory) to combine original and retry results, preferring rows
    with non-empty transcripts.
    """
    from merge_transcripts_jsonl import merge_transcripts  # type: ignore[import-not-found]

    merge_transcripts(
        inputs=[original_jsonl, retry_jsonl],
        output=merged_output,
        csv_filter=None,
        discarded_output=None,
    )


def _write_unretryable_report(
    retry_summary_json: Path,
    unretryable_output: Path,
) -> None:
    """
    From the retry summary JSON (list of row dicts written by fetch_transcripts.py),
    record any rows that still have empty transcripts along with their
    transcript_source, so you can see which are private/missing/blocked/etc.
    """
    if not retry_summary_json.exists():
        print(f"Retry summary JSON not found, skipping unretryable report: {retry_summary_json}")
        return

    try:
        data = json.loads(retry_summary_json.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"Could not parse retry summary JSON: {retry_summary_json} ({e})")
        return

    if not isinstance(data, list):
        print(f"Retry summary JSON is not a list: {retry_summary_json}")
        return

    unretryable: List[Dict[str, Any]] = []
    for obj in data:
        if not isinstance(obj, dict):
            continue
        ft = _safe_strip(obj.get("full_transcript"))
        if ft:
            continue
        vid = _safe_strip(obj.get("video_id"))
        if not vid:
            continue
        src = _safe_strip(obj.get("transcript_source"))
        unretryable.append(
            {
                "video_id": vid,
                "title": _safe_strip(obj.get("title")),
                "transcript_source": src,
            }
        )

    unretryable_output.parent.mkdir(parents=True, exist_ok=True)
    with unretryable_output.open("w", encoding="utf-8") as f:
        json.dump(unretryable, f, indent=2, ensure_ascii=False)

    print(f"Wrote unretryable-empty-transcripts report with {len(unretryable):,} entries to: {unretryable_output}")


def main() -> None:
    p = argparse.ArgumentParser(description="Retry all rows with empty transcripts and merge improved results.")
    p.add_argument(
        "--input-jsonl",
        default=r"c:\Users\revaa\viral2\data\transcript_training_data_final.jsonl",
        help="Transcripts JSONL to scan for empty full_transcript rows.",
    )
    p.add_argument(
        "--merged-output",
        default=r"c:\Users\revaa\viral2\data\transcript_training_data_final.jsonl",
        help=(
            "Output JSONL after merging original+retry. Can be the same path as "
            "--input-jsonl to replace in-place."
        ),
    )
    p.add_argument(
        "--temp-csv",
        default=r"c:\Users\revaa\viral2\data\retry_empty_transcripts.csv",
        help="Temporary CSV to feed into fetch_transcripts.py.",
    )
    p.add_argument(
        "--retry-jsonl",
        default=r"c:\Users\revaa\viral2\data\transcript_training_data_retry_empty.jsonl",
        help="JSONL written by fetch_transcripts.py for the retry run.",
    )
    p.add_argument(
        "--retry-summary-json",
        default=r"c:\Users\revaa\viral2\data\transcript_retry_empty_summary.json",
        help="JSON summary written by fetch_transcripts.py for the retry run.",
    )
    p.add_argument(
        "--retry-failures-json",
        default=r"c:\Users\revaa\viral2\data\transcript_failures_retry_empty.json",
        help="JSON list of retryable failures from the retry run.",
    )
    p.add_argument(
        "--unretryable-output",
        default=r"c:\Users\revaa\viral2\data\unretryable_empty_transcripts.json",
        help=(
            "JSON report of rows that still have empty transcripts after retry, "
            "including their transcript_source classification."
        ),
    )
    p.add_argument("--max-workers", type=int, default=10, help="Workers for the retry fetch_transcripts.py call.")
    p.add_argument("--debug", action="store_true", help="Pass --debug to fetch_transcripts.py.")
    args = p.parse_args()

    input_jsonl = Path(args.input_jsonl)
    merged_output = Path(args.merged_output)
    temp_csv = Path(args.temp_csv)
    retry_jsonl = Path(args.retry_jsonl)
    retry_summary_json = Path(args.retry_summary_json)
    retry_failures_json = Path(args.retry_failures_json)
    unretryable_output = Path(args.unretryable_output)

    n_candidates = _write_temp_csv_from_empty_rows(input_jsonl, temp_csv)
    if n_candidates == 0:
        print(f"No rows with empty transcripts found in: {input_jsonl}")
        return

    print(f"Found {n_candidates:,} rows with empty transcripts to retry.")
    _run_fetch_transcripts(
        temp_csv=temp_csv,
        retry_jsonl=retry_jsonl,
        retry_summary_json=retry_summary_json,
        retry_failures_json=retry_failures_json,
        max_workers=args.max_workers,
        debug=args.debug,
    )

    _merge_original_and_retry(
        original_jsonl=input_jsonl,
        retry_jsonl=retry_jsonl,
        merged_output=merged_output,
    )

    _write_unretryable_report(
        retry_summary_json=retry_summary_json,
        unretryable_output=unretryable_output,
    )


if __name__ == "__main__":
    main()

