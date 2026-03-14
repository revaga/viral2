#!/usr/bin/env python3
"""
Prepare chat-style SFT data for YouTube title generation and split deterministically
into train/test by video_id.

Inputs:
  - CSV with columns: video_id (optional), title (required), description (optional)
  - JSON/JSONL with keys: video_id (optional), title (required), full_transcript (optional), description (optional)

Behavior:
  - Prefer full_transcript when present; otherwise fall back to description.
  - Filter by basic length thresholds.
  - Deterministic train/test split by video_id with fixed seed.
  - Optionally stratify split by source (transcript vs description).

Outputs (JSONL):
  Each line:
    {"id": "<video_id_or_row_id>", "source": "transcript"|"description", "messages": [...]}
  The assistant message contains the reference title.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


# Quality thresholds (tuned for your existing pipeline; adjust as needed)
MIN_TRANSCRIPT_LENGTH = 200
MAX_TRANSCRIPT_LENGTH = 50_000
MIN_DESCRIPTION_LENGTH = 40
MAX_DESCRIPTION_LENGTH = 20_000
MIN_TITLE_LENGTH = 10
MAX_TITLE_LENGTH = 100


SYSTEM_PROMPT = """You are an expert YouTube title generator. Your task is to create compelling, accurate titles that capture the essence of video content.

Guidelines for generating YouTube titles:
1. Be accurate and truthful - the title must reflect the actual content
2. Make it compelling and click-worthy while staying honest
3. Keep titles concise (ideally 50-80 characters, max 100 characters)
4. Use natural language that viewers would search for
5. Highlight the most interesting or valuable aspect
6. Create a curiosity gap without being misleading

Generate only the title itself, nothing else."""


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


def _to_messages(source: str, source_text: str, title: str) -> List[Dict[str, str]]:
    label = "Transcript" if source == "transcript" else "Description"
    user = (
        f"Based on the following video {label.lower()}, generate an accurate and compelling YouTube title.\n\n"
        f"{label}:\n{source_text}\n\n"
        "Generate the YouTube title:"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
        {"role": "assistant", "content": title},
    ]


def _prepare_examples(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []
    for i, rec in enumerate(records):
        rec_id = _infer_id(rec, i)
        title = _safe_strip(rec.get("title"))
        source, source_text = _select_source_text(rec)
        if not _passes_quality(source, source_text, title):
            continue
        examples.append(
            {
                "id": rec_id,
                "source": source,
                "messages": _to_messages(source, source_text, title),
            }
        )
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
        buckets: Dict[str, List[str]] = {"transcript": [], "description": [], "": []}
        for ex_id, ex in by_id.items():
            buckets.get(str(ex.get("source", "")), buckets[""]).append(ex_id)

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
    p = argparse.ArgumentParser(description="Prepare SFT JSONL and split train/test deterministically.")
    p.add_argument(
        "--input",
        default="",
        help=(
            "Path to CSV, JSON, or JSONL with titles + transcripts/descriptions. "
            "If omitted, defaults to data/training_data.json when present; otherwise data_viral_titles.csv."
        ),
    )
    p.add_argument("--out-train", required=True, help="Output train JSONL path.")
    p.add_argument("--out-test", required=True, help="Output test JSONL path.")
    p.add_argument("--test-size", type=float, default=0.1, help="Fraction to reserve for test (default: 0.1).")
    p.add_argument("--seed", type=int, default=1337, help="Random seed for deterministic split (default: 1337).")
    p.add_argument("--stratify-source", action="store_true", help="Stratify split by source (transcript vs description).")
    args = p.parse_args()

    project_root = Path(__file__).resolve().parent
    default_json = project_root / "data" / "training_data.json"
    default_csv = project_root / "data_viral_titles.csv"

    if args.input.strip():
        in_path = Path(args.input)
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

    _write_jsonl(Path(args.out_train), train)
    _write_jsonl(Path(args.out_test), test)

    print(f"Wrote train: {len(train)} examples -> {args.out_train}")
    print(f"Wrote test:  {len(test)} examples -> {args.out_test}")


if __name__ == "__main__":
    main()

