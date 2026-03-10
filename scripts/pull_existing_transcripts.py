#!/usr/bin/env python3
"""
Pull transcripts from an existing JSONL (no YouTube calls) for a target set of video_ids.

Reads:
  - targets CSV (default: data_viral_titles.csv) with columns: video_id, title, description (optional)
  - source JSONL (default: C:\\Users\\revaa\\Desktop\\viral\\data\\training_data.jsonl) with keys:
      video_id, full_transcript (and optionally url/title)

Writes:
  - output JSON (default: data/training_data.json) as a list of dicts:
      {video_id, title, description, full_transcript, transcript_source}

Also optionally writes:
  - missing video_ids list (one per line)
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def _safe_strip(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x.strip()
    return str(x).strip()


def _load_targets_csv(path: Path) -> Dict[str, Dict[str, str]]:
    """
    Returns mapping video_id -> {title, description}.
    """
    out: Dict[str, Dict[str, str]] = {}
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "video_id" not in reader.fieldnames:
            raise ValueError("targets CSV must include a 'video_id' column")
        if "title" not in reader.fieldnames:
            raise ValueError("targets CSV must include a 'title' column")
        for row in reader:
            vid = _safe_strip(row.get("video_id"))
            if not vid:
                continue
            if vid in out:
                continue
            out[vid] = {
                "title": _safe_strip(row.get("title")),
                "description": _safe_strip(row.get("description")),
            }
    return out


def _best_transcript_update(existing: str, candidate: str) -> str:
    """
    Keep the longer transcript (simple, robust heuristic).
    """
    candidate = candidate.strip()
    if not candidate:
        return existing
    if len(candidate) > len(existing):
        return candidate
    return existing


def _stream_jsonl_for_transcripts(
    source_jsonl: Path, target_ids: set[str]
) -> Tuple[Dict[str, str], int]:
    """
    Returns (video_id -> best_full_transcript, matched_rows_count)
    matched_rows_count counts JSONL rows whose video_id was in target_ids (regardless of transcript empty).
    """
    transcripts: Dict[str, str] = {}
    matched_rows = 0
    with source_jsonl.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            vid = _safe_strip(obj.get("video_id"))
            if not vid or vid not in target_ids:
                continue
            matched_rows += 1
            ft = _safe_strip(obj.get("full_transcript") or obj.get("transcript"))
            if not ft:
                continue
            transcripts[vid] = _best_transcript_update(transcripts.get(vid, ""), ft)
    return transcripts, matched_rows


def main() -> None:
    p = argparse.ArgumentParser(description="Pull matching transcripts from an existing JSONL by video_id.")
    p.add_argument(
        "--targets-csv",
        default=r"c:\Users\revaa\viral2\data_viral_titles.csv",
        help="Target CSV containing video_id/title/description (default: project data_viral_titles.csv).",
    )
    p.add_argument(
        "--source-jsonl",
        default=r"C:\Users\revaa\Desktop\viral\data\training_data.jsonl",
        help="Existing JSONL containing full_transcript keyed by video_id.",
    )
    p.add_argument(
        "--out",
        default=r"c:\Users\revaa\viral2\data\training_data.json",
        help="Output JSON path (list of dicts) (default: project data/training_data.json).",
    )
    p.add_argument(
        "--missing-out",
        default="",
        help="Optional path to write missing video_ids (one per line).",
    )
    args = p.parse_args()

    targets_csv = Path(args.targets_csv)
    source_jsonl = Path(args.source_jsonl)
    out_path = Path(args.out)

    if not targets_csv.exists():
        raise SystemExit(f"Targets CSV not found: {targets_csv}")
    if not source_jsonl.exists():
        raise SystemExit(f"Source JSONL not found: {source_jsonl}")

    targets = _load_targets_csv(targets_csv)
    target_ids = set(targets.keys())

    transcripts_by_id, matched_rows = _stream_jsonl_for_transcripts(source_jsonl, target_ids)

    out_rows = []
    missing_ids = []
    for vid, meta in targets.items():
        ft = transcripts_by_id.get(vid, "")
        if ft:
            src = "existing_jsonl"
        else:
            src = "missing"
            missing_ids.append(vid)
        out_rows.append(
            {
                "video_id": vid,
                "title": meta.get("title", ""),
                "description": meta.get("description", ""),
                "full_transcript": ft,
                "transcript_source": src,
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out_rows, f, indent=2, ensure_ascii=False)

    if args.missing_out.strip():
        miss_path = Path(args.missing_out)
        miss_path.parent.mkdir(parents=True, exist_ok=True)
        with miss_path.open("w", encoding="utf-8") as f:
            for vid in missing_ids:
                f.write(vid + "\n")

    found = sum(1 for r in out_rows if r.get("full_transcript"))
    print(f"Targets: {len(target_ids)}")
    print(f"JSONL rows matched target_ids: {matched_rows}")
    print(f"Transcripts found (non-empty): {found}")
    print(f"Missing transcripts: {len(missing_ids)}")
    print(f"Wrote: {out_path}")
    if args.missing_out.strip():
        print(f"Wrote missing list: {args.missing_out}")


if __name__ == "__main__":
    main()

