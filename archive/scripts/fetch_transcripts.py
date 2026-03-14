#!/usr/bin/env python3
"""
Fetch YouTube transcripts for a list of video_ids (from a CSV) and write a JSON
dataset used by the LoRA title training pipeline.

Input CSV (default: project data_viral_titles.csv) columns:
  - video_id (required)
  - title (required)
  - description (optional)

Output JSON (default: project data/training_data.json):
  [
    {
      "video_id": "...",
      "title": "...",
      "description": "...",
      "full_transcript": "...",  # may be empty if not available
      "transcript_source": "youtube_transcript_api" | "missing",
      "transcript_language": "en" | ""               # best-effort
    },
    ...
  ]

Notes:
- Concurrency uses ThreadPoolExecutor for speed.
- Optional proxy support: if PROXY_USERNAME and PROXY_PASSWORD are set in env,
  WebshareProxyConfig is used. Otherwise requests go direct.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm  # type: ignore[import-untyped]


_YOUTUBE_BLOCK_WARNING_PRINTED = False


def _safe_strip(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x.strip()
    return str(x).strip()


def _load_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "video_id" not in reader.fieldnames:
            raise ValueError("Input CSV must include a 'video_id' column")
        if "title" not in reader.fieldnames:
            raise ValueError("Input CSV must include a 'title' column")
        return [dict(r) for r in reader]


def _build_transcript_api():
    """
    Build a thread-safe transcript API client.

    This repo uses the newer youtube_transcript_api API shape where:
      TRANSCRIPT_API = YouTubeTranscriptApi(proxy_config=cfg?)
      transcript_list = TRANSCRIPT_API.fetch(video_id)
    """
    from youtube_transcript_api import YouTubeTranscriptApi  # type: ignore[import-untyped]

    user = os.getenv("PROXY_USERNAME", "").strip()
    pw = os.getenv("PROXY_PASSWORD", "").strip()
    if user and pw:
        from youtube_transcript_api.proxies import WebshareProxyConfig  # type: ignore[import-untyped]

        cfg = WebshareProxyConfig(proxy_username=user, proxy_password=pw)
        try:
            return YouTubeTranscriptApi(proxy_config=cfg)
        except TypeError:
            # Proxy not supported by installed version
            return YouTubeTranscriptApi()

    return YouTubeTranscriptApi()


def _fetch_best_transcript(
    video_id: str, transcript_api, debug: bool = False
) -> Tuple[str, str, str]:
    """
    Returns (language_code, full_transcript_text, status).

    status is one of:
      - "ok": transcript successfully fetched (full_transcript may still be empty)
      - "missing": no transcript found / disabled / video unavailable
      - "age_restricted": video is age-restricted and requires auth
      - "private": video is private
      - "blocked_by_youtube": YouTube is actively blocking our requests
      - "error": unexpected error (only used when raising to caller)
    """
    from youtube_transcript_api import (  # type: ignore[import-untyped]
        NoTranscriptFound,
        TranscriptsDisabled,
        VideoUnavailable,
    )

    global _YOUTUBE_BLOCK_WARNING_PRINTED

    try:
        if not hasattr(transcript_api, "fetch"):
            raise AttributeError("transcript_api has no .fetch(video_id, languages=...)")

        # Prefer English transcripts when available. The library will choose
        # manually-created over generated when both exist.
        fetched = transcript_api.fetch(video_id, languages=["en"])
        parts: List[str] = []
        for s in fetched:
            # FetchedTranscriptSnippet exposes a .text attribute; we keep the
            # older dict fallback for forward-compatibility.
            if isinstance(s, dict):
                t = (s.get("text") or "").strip()
            else:
                t = _safe_strip(getattr(s, "text", ""))
            if t:
                parts.append(t)
        full_text = " ".join(parts).strip()
        lang_code = _safe_strip(getattr(fetched, "language_code", ""))
        return (lang_code, full_text, "ok")
    except (NoTranscriptFound, TranscriptsDisabled, VideoUnavailable):
        # These are non-retryable: there simply is no transcript available.
        return "", "", "missing"
    except Exception as e:
        msg = str(e)
        # Classify common non-auth, non-missing failures so we can record them
        # directly on the row instead of treating them as generic errors.
        if "This video is age-restricted" in msg:
            if debug:
                print(f"[debug] age-restricted video for {video_id}: {e}")
            return "", "", "age_restricted"
        if "This video is private" in msg:
            if debug:
                print(f"[debug] private video for {video_id}: {e}")
            return "", "", "private"
        if "YouTube is blocking your requests" in msg:
            if not _YOUTUBE_BLOCK_WARNING_PRINTED:
                # Always surface at least one clear, user-facing warning when YouTube
                # starts blocking us, regardless of --debug.
                print(
                    "WARNING: YouTube appears to be blocking transcript requests. "
                    "If you're using Webshare proxies, make sure they are 'Residential' "
                    "proxies (not free tier / 'Proxy Server' / 'Static Residential'). "
                    "You may also try unsetting PROXY_USERNAME/PROXY_PASSWORD to run "
                    "without proxies."
                )
                _YOUTUBE_BLOCK_WARNING_PRINTED = True
            if debug:
                print(f"[debug] YouTube is blocking transcript requests for {video_id}: {e}")
            return "", "", "blocked_by_youtube"

        # This is an unexpected / retryable failure (network error, unexpected API error, etc.)
        if debug:
            print(f"[debug] transcript fetch failed for {video_id}: {e}")
        # Encode the video_id into the error message so the caller can track
        # which IDs should be retried later.
        raise RuntimeError(f"transcript_fetch_error:{video_id}:{e}") from e


def _process_one(row: Dict[str, str], transcript_api, debug: bool = False) -> Dict[str, Any]:
    vid = _safe_strip(row.get("video_id"))
    title = _safe_strip(row.get("title"))
    desc = _safe_strip(row.get("description"))

    if vid:
        lang, ft, status = _fetch_best_transcript(vid, transcript_api, debug=debug)
    else:
        lang, ft, status = "", "", "missing"

    if ft:
        source = "youtube_transcript_api"
    else:
        # Preserve "missing" but also distinguish common failure modes so the
        # dataset consumer can understand *why* the transcript is empty.
        if status in {"missing", "age_restricted", "private", "blocked_by_youtube"}:
            source = status
        else:
            source = "error"
    return {
        "video_id": vid,
        "title": title,
        "description": desc,
        "full_transcript": ft,
        "transcript_source": source,
        "transcript_language": lang if ft else "",
    }


def main() -> None:
    try:
        from dotenv import load_dotenv  # type: ignore[import-untyped]

        load_dotenv()
    except Exception:
        pass

    p = argparse.ArgumentParser(description="Fetch YouTube transcripts by video_id and write training data (JSON + JSONL).")
    p.add_argument(
        "--input",
        default=r"c:\Users\revaa\viral2\data_viral_titles.csv",
        help="Input CSV containing video_id/title/description (default: project data_viral_titles.csv).",
    )
    p.add_argument(
        "--output",
        default=r"c:\Users\revaa\viral2\data\training_data.json",
        help="Output JSON path (default: project data/transcript_data.json).",
    )
    p.add_argument(
        "--jsonl-output",
        default=r"c:\Users\revaa\viral2\data\transcript_training_data.jsonl",
        help="Output JSONL path; written incrementally every few transcripts.",
    )
    p.add_argument("--max-workers", type=int, default=10, help="Thread workers (default: 10).")
    p.add_argument(
        "--debug",
        action="store_true",
        help="Print transcript fetch errors (useful if everything is coming back missing).",
    )
    p.add_argument(
        "--drop-missing",
        action="store_true",
        help="Drop rows where transcript could not be fetched (else keep with empty full_transcript).",
    )
    p.add_argument(
        "--failed-ids-output",
        default="",
        help=(
            "Optional path to write video_ids whose transcript fetch failed with a retryable error "
            "(JSON list). These are distinct from videos where no transcript is available."
        ),
    )
    p.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help=(
            "Total number of shards for multi-process runs. "
            "When >1, combine with --shard-index so that each process "
            "handles a disjoint subset of video_ids based on a stable hash."
        ),
    )
    p.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help=(
            "Zero-based shard index for this process when --num-shards > 1. "
            "Only rows whose hashed video_id maps to this shard will be processed."
        ),
    )
    args = p.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise SystemExit(f"Input not found: {in_path}")

    rows = _load_csv_rows(in_path)
    # Remove duplicates by video_id (keep first)
    seen = set()
    deduped: List[Dict[str, str]] = []
    for r in rows:
        vid = _safe_strip(r.get("video_id"))
        if not vid or vid in seen:
            continue
        seen.add(vid)
        deduped.append(r)

    if args.num_shards < 1:
        raise SystemExit("--num-shards must be >= 1")
    if not (0 <= args.shard_index < max(1, args.num_shards)):
        raise SystemExit("--shard-index must be in [0, num_shards)")

    out_path = Path(args.output)
    jsonl_path = Path(args.jsonl_output)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    import json

    # If a JSONL already exists, treat its video_ids as "already processed" so we can resume.
    processed_ids: set[str] = set()
    if jsonl_path.exists():
        with jsonl_path.open("r", encoding="utf-8", errors="replace") as jf:
            for line in jf:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                vid = _safe_strip(obj.get("video_id"))
                if vid:
                    processed_ids.add(vid)

    # Only schedule work for video_ids that have not yet been written to the JSONL file.
    remaining_rows: List[Dict[str, str]] = []
    for r in deduped:
        vid = _safe_strip(r.get("video_id"))
        if not vid or vid in processed_ids:
            continue
        if args.num_shards > 1:
            # Stable shard assignment based on md5(video_id).
            h = hashlib.md5(vid.encode("utf-8")).hexdigest()
            shard = int(h[:8], 16) % args.num_shards
            if shard != args.shard_index:
                continue
        remaining_rows.append(r)

    transcript_api = _build_transcript_api()

    out_rows: List[Dict[str, Any]] = []
    jsonl_batches: List[Dict[str, Any]] = []
    retryable_failures: List[str] = []

    def flush_jsonl_batch() -> None:
        nonlocal jsonl_batches
        if not jsonl_batches:
            return
        with jsonl_path.open("a", encoding="utf-8") as jf:
            for row in jsonl_batches:
                jf.write(json.dumps(row, ensure_ascii=False) + "\n")
        jsonl_batches = []

    try:
        with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as ex:
            futures = [ex.submit(_process_one, r, transcript_api, args.debug) for r in remaining_rows]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Fetching transcripts"):
                try:
                    row = fut.result()
                    out_rows.append(row)
                    jsonl_batches.append(row)
                    # Flush every 5 transcripts so progress is durably saved.
                    if len(jsonl_batches) >= 5:
                        flush_jsonl_batch()
                except Exception as e:  # noqa: BLE001
                    msg = str(e)
                    vid: Optional[str] = None
                    marker = "transcript_fetch_error:"
                    if marker in msg:
                        # Expected format: transcript_fetch_error:<video_id>:<details>
                        try:
                            _, rest = msg.split(marker, 1)
                            vid, _ = rest.split(":", 1)
                            vid = vid.strip()
                        except Exception:
                            vid = None
                    if vid:
                        retryable_failures.append(vid)
                        if args.debug:
                            print(f"[debug] marked retryable failure for video_id={vid}: {e}")
                    else:
                        if args.debug:
                            print(f"[debug] unexpected error in worker: {e}")
    finally:
        # Ensure any remaining buffered rows are written even on abrupt stop.
        flush_jsonl_batch()

    if args.drop_missing:
        out_rows = [r for r in out_rows if _safe_strip(r.get("full_transcript"))]

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out_rows, f, indent=2, ensure_ascii=False)

    # Optionally write retryable failures list for later re-processing.
    if args.failed_ids_output.strip():
        failed_path = Path(args.failed_ids_output)
        failed_path.parent.mkdir(parents=True, exist_ok=True)
        with failed_path.open("w", encoding="utf-8") as f:
            json.dump(sorted(set(retryable_failures)), f, indent=2, ensure_ascii=False)

    found = sum(1 for r in out_rows if _safe_strip(r.get("full_transcript")))

    print(f"Input unique video_ids (CSV, de-duplicated): {len(deduped)}")
    print(f"Video_ids already present in JSONL before this run: {len(processed_ids)}")
    print(f"Video_ids scheduled this run (remaining): {len(remaining_rows)}")
    print(f"Transcripts fetched this run (non-empty): {found}")
    print(f"Retryable fetch failures (to retry later): {len(set(retryable_failures))}")
    print(f"Wrote JSON summary for this run: {out_path}")
    print(f"Appended JSONL rows to: {jsonl_path}")
    if args.failed_ids_output.strip():
        print(f"Wrote retryable failure ids JSON: {args.failed_ids_output}")


if __name__ == "__main__":
    main()

