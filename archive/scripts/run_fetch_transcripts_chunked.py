#!/usr/bin/env python3
"""
Orchestrate multi-process transcript fetching by sharding the input CSV.

This script runs multiple `fetch_transcripts.py` processes in parallel, each
with its own shard of video_ids and its own temporary outputs, then merges
everything into the final debug files.

It is tailored to the debug command you provided, using the same output paths
by default:

  python scripts/run_fetch_transcripts_chunked.py \\
    --input "c:\\Users\\revaa\\viral2\\data_viral_titles.csv" \\
    --jsonl-output "c:\\Users\\revaa\\viral2\\data\\transcript_training_data_debug.jsonl" \\
    --output "c:\\Users\\revaa\\viral2\\data\\transcript_training_data_debug.json" \\
    --failed-ids-output "c:\\Users\\revaa\\viral2\\data\\transcript_failures_debug.json" \\
    --num-shards 4 \\
    --max-workers-per-shard 8 \\
    --debug
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import List, Set


def _run_shard(
    shard_index: int,
    num_shards: int,
    input_path: Path,
    base_jsonl: Path,
    base_json: Path,
    base_failed: Path,
    max_workers_per_shard: int,
    debug: bool,
) -> None:
    shard_jsonl = base_jsonl.with_suffix(base_jsonl.suffix + f".shard{shard_index}")
    shard_json = base_json.with_suffix(base_json.suffix.replace(".json", f".shard{shard_index}.json"))
    shard_failed = base_failed.with_suffix(
        base_failed.suffix.replace(".json", f".shard{shard_index}.json")
    )

    cmd = [
        "python",
        "builddatasets/fetch_transcripts.py",
        "--input",
        str(input_path),
        "--jsonl-output",
        str(shard_jsonl),
        "--output",
        str(shard_json),
        "--failed-ids-output",
        str(shard_failed),
        "--max-workers",
        str(max_workers_per_shard),
        "--num-shards",
        str(num_shards),
        "--shard-index",
        str(shard_index),
    ]
    if debug:
        cmd.append("--debug")

    subprocess.run(cmd, check=True)


def _merge_jsonl(shard_files: List[Path], final_path: Path) -> None:
    final_path.parent.mkdir(parents=True, exist_ok=True)
    with final_path.open("w", encoding="utf-8") as out_f:
        for shard in shard_files:
            if not shard.exists():
                continue
            with shard.open("r", encoding="utf-8", errors="replace") as in_f:
                for line in in_f:
                    line = line.rstrip("\n")
                    if not line:
                        continue
                    out_f.write(line + "\n")


def _merge_json_lists(shard_files: List[Path], final_path: Path) -> None:
    combined = []
    for shard in shard_files:
        if not shard.exists():
            continue
        try:
            with shard.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        if isinstance(data, list):
            combined.extend(data)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    with final_path.open("w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)


def _merge_failed_ids(shard_files: List[Path], final_path: Path) -> None:
    combined: set[str] = set()
    for shard in shard_files:
        if not shard.exists():
            continue
        try:
            with shard.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        if isinstance(data, list):
            combined.update(str(x) for x in data)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    with final_path.open("w", encoding="utf-8") as f:
        json.dump(sorted(combined), f, indent=2, ensure_ascii=False)


def _load_processed_ids_from_jsonl(jsonl_path: Path) -> Set[str]:
    """
    Load video_ids that have already been written to the existing merged JSONL file.
    This lets us avoid re-fetching transcripts when the input CSV changes.
    """
    processed_ids: Set[str] = set()
    if not jsonl_path.exists():
        return processed_ids
    try:
        with jsonl_path.open("r", encoding="utf-8", errors="replace") as jf:
            for line in jf:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                vid = str(obj.get("video_id", "")).strip()
                if vid:
                    processed_ids.add(vid)
    except Exception:
        # If anything goes wrong here, just fall back to treating as empty.
        processed_ids.clear()
    return processed_ids


def _filter_input_csv_by_processed_ids(input_path: Path, processed_ids: Set[str]) -> Path:
    """
    Create a filtered copy of the input CSV that excludes rows whose video_id
    is already present in the merged JSONL. If no processed_ids, return the original path.
    """
    if not processed_ids:
        return input_path

    filtered_path = input_path.with_suffix(input_path.suffix + ".remaining")

    with input_path.open("r", encoding="utf-8", newline="") as in_f, filtered_path.open(
        "w", encoding="utf-8", newline=""
    ) as out_f:
        reader = csv.DictReader(in_f)
        fieldnames = reader.fieldnames or []
        if "video_id" not in fieldnames:
            # If there is no video_id column, we cannot filter; just return original.
            return input_path
        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
        writer.writeheader()
        for row in reader:
            vid = str(row.get("video_id", "")).strip()
            if not vid or vid in processed_ids:
                continue
            writer.writerow(row)

    return filtered_path


def main() -> None:
    p = argparse.ArgumentParser(description="Run fetch_transcripts.py in parallel shards and merge outputs.")
    p.add_argument(
        "--input",
        default=r"c:\Users\revaa\viral2\data_viral_titles.csv",
        help="Input CSV containing video_id/title/description.",
    )
    p.add_argument(
        "--jsonl-output",
        default=r"c:\Users\revaa\viral2\data\transcript_training_data_debug.jsonl",
        help="Final merged JSONL output path.",
    )
    p.add_argument(
        "--output",
        default=r"c:\Users\revaa\viral2\data\transcript_training_data_debug.json",
        help="Final merged JSON (list of dicts) output path.",
    )
    p.add_argument(
        "--failed-ids-output",
        default=r"c:\Users\revaa\viral2\data\transcript_failures_debug.json",
        help="Final merged retryable failure ids JSON path.",
    )
    p.add_argument(
        "--num-shards",
        type=int,
        default=4,
        help="Number of parallel shards/processes to run.",
    )
    p.add_argument(
        "--max-workers-per-shard",
        type=int,
        default=8,
        help="Thread pool size per shard process.",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Pass --debug through to fetch_transcripts.py.",
    )
    args = p.parse_args()

    base_jsonl = Path(args.jsonl_output)
    base_json = Path(args.output)
    base_failed = Path(args.failed_ids_output)

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")

    # If there is already a merged JSONL file, avoid re-fetching transcripts
    # for video_ids that have been processed in previous runs, even if the CSV changed.
    processed_ids = _load_processed_ids_from_jsonl(base_jsonl)
    if processed_ids:
        print(
            f"Found {len(processed_ids):,} already-fetched transcripts in {base_jsonl}. "
            "Filtering input CSV to only new video_ids."
        )
        input_path = _filter_input_csv_by_processed_ids(input_path, processed_ids)

    num_shards = max(1, int(args.num_shards))

    # Run each shard sequentially by default to avoid overwhelming YouTube.
    # If you want true parallelism, you can adapt this to use subprocess.Popen.
    shard_jsonl_files: List[Path] = []
    shard_json_files: List[Path] = []
    shard_failed_files: List[Path] = []

    for shard_index in range(num_shards):
        print(f"=== Running shard {shard_index+1}/{num_shards} ===")
        _run_shard(
            shard_index=shard_index,
            num_shards=num_shards,
            input_path=input_path,
            base_jsonl=base_jsonl,
            base_json=base_json,
            base_failed=base_failed,
            max_workers_per_shard=args.max_workers_per_shard,
            debug=args.debug,
        )
        shard_jsonl_files.append(
            base_jsonl.with_suffix(base_jsonl.suffix + f".shard{shard_index}")
        )
        shard_json_files.append(
            base_json.with_suffix(base_json.suffix.replace(".json", f".shard{shard_index}.json"))
        )
        shard_failed_files.append(
            base_failed.with_suffix(
                base_failed.suffix.replace(".json", f".shard{shard_index}.json")
            )
        )

    print("=== Merging shard outputs ===")
    _merge_jsonl(shard_jsonl_files, base_jsonl)
    _merge_json_lists(shard_json_files, base_json)
    _merge_failed_ids(shard_failed_files, base_failed)
    print(f"Merged JSONL: {base_jsonl}")
    print(f"Merged JSON: {base_json}")
    print(f"Merged retryable failure ids JSON: {base_failed}")


if __name__ == "__main__":
    main()

