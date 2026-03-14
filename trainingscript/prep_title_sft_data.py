#!/usr/bin/env python3
"""
Prepare SFT split for YouTube title generation: one data JSONL + train/test ID lists only.
No separate train or test JSONL files; finetune and generate scripts pull from the
single data file by ID.

Inputs:
  - CSV or JSON/JSONL with: video_id (optional), title (required),
    full_transcript/transcript (optional), description (optional)

Outputs:
  - One data JSONL (--out-data): all valid records, one per line:
      {"id": "...", "source": "transcript"|"description", "source_text": "...", "title": "..."}
  - Train IDs (--out-train-ids): one id per line, no overlap with test.
  - Test IDs (--out-test-ids): one id per line.

Prompts are never stored; they are built at training/generation time from this data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


# Quality thresholds (copied from original script)
MIN_TRANSCRIPT_LENGTH = 200
MAX_TRANSCRIPT_LENGTH = 50_000
MIN_DESCRIPTION_LENGTH = 40
MAX_DESCRIPTION_LENGTH = 20_000
MIN_TITLE_LENGTH = 10
MAX_TITLE_LENGTH = 100


def _sha1_int(s: str) -> int:
    return int(hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest(), 16)


def _safe_strip(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x.strip()
    return str(x).strip()


def _load_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        out: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                out.append(json.loads(line))
        return out
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data  # type: ignore[return-value]
    raise ValueError(f"Expected list in JSON file: {path}")


def _load_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _infer_id(rec: Dict[str, Any], row_idx: int) -> str:
    vid = _safe_strip(rec.get("video_id"))
    if vid:
        return vid
    url = _safe_strip(rec.get("url"))
    if url:
        return url
    return f"row_{row_idx}"


def _select_source_text(rec: Dict[str, Any]) -> Tuple[str, str]:
    """
    Returns (source, text) where source is 'transcript' or 'description'.
    """
    transcript = _safe_strip(rec.get("full_transcript") or rec.get("transcript"))
    if transcript:
        return "transcript", transcript
    desc = _safe_strip(rec.get("description"))
    if desc:
        return "description", desc
    return "", ""


def _passes_quality(source: str, source_text: str, title: str) -> bool:
    if not title or len(title) < MIN_TITLE_LENGTH or len(title) > MAX_TITLE_LENGTH:
        return False
    if source == "transcript":
        return MIN_TRANSCRIPT_LENGTH <= len(source_text) <= MAX_TRANSCRIPT_LENGTH
    if source == "description":
        return MIN_DESCRIPTION_LENGTH <= len(source_text) <= MAX_DESCRIPTION_LENGTH
    return False


def _to_record(source: str, source_text: str, title: str, rec_id: str) -> Dict[str, Any]:
    """
    Build a minimal, prompt-free training record.

    The training script will later turn this into chat messages and attach
    the system prompt only in memory.
    """
    return {
        "id": rec_id,
        "source": source,
        "source_text": source_text,
        "title": title,
    }


def _prepare_examples(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []
    for i, rec in enumerate(records):
        rec_id = _infer_id(rec, i)
        title = _safe_strip(rec.get("title"))
        source, source_text = _select_source_text(rec)
        if not _passes_quality(source, source_text, title):
            continue
        examples.append(_to_record(source, source_text, title, rec_id))
    return examples


def _split_by_id(
    examples: List[Dict[str, Any]],
    test_size: float,
    seed: int,
    stratify_source: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not 0.0 < test_size < 1.0:
        raise ValueError("--test-size must be in (0, 1)")

    # Deduplicate by id (keep first; quality filtering already applied)
    by_id: Dict[str, Dict[str, Any]] = {}
    for ex in examples:
        ex_id = str(ex["id"])
        if ex_id not in by_id:
            by_id[ex_id] = ex

    ids = list(by_id.keys())
    rng = random.Random(seed)

    if stratify_source:
        from typing import DefaultDict

        buckets: DefaultDict[str, List[str]] = DefaultDict(list)
        for ex_id, ex in by_id.items():
            buckets[str(ex.get("source", ""))].append(ex_id)

        test_ids: List[str] = []
        for _, b in buckets.items():
            if not b:
                continue
            rng.shuffle(b)
            k = max(1, int(round(len(b) * test_size))) if len(b) >= 2 else 1
            k = min(k, len(b) - 1) if len(b) > 1 else 1
            test_ids.extend(b[:k])
        test_ids = sorted(set(test_ids), key=lambda s: _sha1_int(f"{seed}:{s}"))
        test_set = set(test_ids)
    else:
        rng.shuffle(ids)
        k = int(round(len(ids) * test_size))
        k = max(1, min(k, len(ids) - 1)) if len(ids) > 1 else 1
        test_set = set(ids[:k])

    train, test = [], []
    for ex_id, ex in by_id.items():
        (test if ex_id in test_set else train).append(ex)

    # Stable ordering for reproducibility
    train.sort(key=lambda x: _sha1_int(f"{seed}:train:{x['id']}"))
    test.sort(key=lambda x: _sha1_int(f"{seed}:test:{x['id']}"))
    return train, test


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Prepare SFT JSONL (without system prompt) and split train/test deterministically."
        )
    )
    p.add_argument(
        "--input",
        default="",
        help=(
            "Path to CSV, JSON, or JSONL with titles + transcripts/descriptions. "
            "If omitted, defaults to data/training_data.json when present; "
            "otherwise data_viral_titles.csv."
        ),
    )
    p.add_argument(
        "--out-data",
        required=True,
        help="Output data JSONL path (single file with all valid records for ID lookup).",
    )
    p.add_argument(
        "--out-train-ids",
        required=True,
        help="Output train ID list path (one id per line).",
    )
    p.add_argument(
        "--out-test-ids",
        required=True,
        help="Output test ID list path (one id per line).",
    )
    p.add_argument(
        "--test-size",
        type=float,
        default=0.1,
        help="Fraction to reserve for test (default: 0.1).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Random seed for deterministic split (default: 1337).",
    )
    p.add_argument(
        "--stratify-source",
        action="store_true",
        help="Stratify split by source (transcript vs description).",
    )
    args = p.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    default_json = project_root / "data" / "training_data.json"
    default_csv = project_root / "data_viral_titles.csv"

    if args.input.strip():
        in_path = Path(args.input)
        if not in_path.is_absolute():
            in_path = project_root / in_path
    else:
        in_path = default_json if default_json.exists() else default_csv
    if not in_path.exists():
        raise SystemExit(f"Input not found: {in_path}")

    if in_path.suffix.lower() == ".csv":
        records = _load_csv(in_path)
    elif in_path.suffix.lower() in {".json", ".jsonl"}:
        records = _load_json_or_jsonl(in_path)
    else:
        raise SystemExit("Unsupported input type. Use .csv, .json, or .jsonl")

    examples = _prepare_examples(records)
    if len(examples) < 20:
        raise SystemExit(f"Too few usable examples after filtering: {len(examples)}")

    train, test = _split_by_id(
        examples=examples,
        test_size=args.test_size,
        seed=args.seed,
        stratify_source=args.stratify_source,
    )

    # Single data JSONL: all valid records (train + test) for lookup by ID.
    all_records = train + test
    out_data_path = Path(args.out_data)
    _write_jsonl(out_data_path, all_records)

    # Train/test ID lists only; no train or test JSONL files.
    train_ids_path = Path(args.out_train_ids)
    test_ids_path = Path(args.out_test_ids)
    train_ids_path.parent.mkdir(parents=True, exist_ok=True)
    test_ids_path.parent.mkdir(parents=True, exist_ok=True)
    with train_ids_path.open("w", encoding="utf-8") as f:
        for ex in train:
            f.write(f"{ex['id']}\n")
    with test_ids_path.open("w", encoding="utf-8") as f:
        for ex in test:
            f.write(f"{ex['id']}\n")

    print(f"Wrote data: {len(all_records)} examples -> {args.out_data}")
    print(f"Wrote train IDs: {len(train)} -> {args.out_train_ids}")
    print(f"Wrote test IDs:  {len(test)} -> {args.out_test_ids}")


if __name__ == "__main__":
    main()

