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
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm  # type: ignore[import-untyped]


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
) -> Tuple[str, str]:
    """
    Returns (language_code, full_transcript_text).
    Best-effort: prefer English when available, otherwise first available transcript.
    """
    from youtube_transcript_api import (  # type: ignore[import-untyped]
        NoTranscriptFound,
        TranscriptsDisabled,
        VideoUnavailable,
    )

    try:
        if not hasattr(transcript_api, "fetch"):
            raise AttributeError("transcript_api has no .fetch(video_id)")

        snippets = transcript_api.fetch(video_id)
        parts: List[str] = []
        for s in snippets:
            if isinstance(s, dict):
                t = (s.get("text") or "").strip()
            else:
                t = _safe_strip(getattr(s, "text", ""))
            if t:
                parts.append(t)
        full_text = " ".join(parts).strip()
        return ("", full_text) if full_text else ("", "")
    except (NoTranscriptFound, TranscriptsDisabled, VideoUnavailable):
        return "", ""
    except Exception as e:
        if debug:
            print(f"[debug] transcript fetch failed for {video_id}: {e}")
        return "", ""


def _process_one(row: Dict[str, str], transcript_api, debug: bool = False) -> Dict[str, Any]:
    vid = _safe_strip(row.get("video_id"))
    title = _safe_strip(row.get("title"))
    desc = _safe_strip(row.get("description"))

    lang, ft = _fetch_best_transcript(vid, transcript_api, debug=debug) if vid else ("", "")
    return {
        "video_id": vid,
        "title": title,
        "description": desc,
        "full_transcript": ft,
        "transcript_source": "youtube_transcript_api" if ft else "missing",
        "transcript_language": lang if ft else "",
    }


def main() -> None:
    try:
        from dotenv import load_dotenv  # type: ignore[import-untyped]

        load_dotenv()
    except Exception:
        pass

    p = argparse.ArgumentParser(description="Fetch YouTube transcripts by video_id and write data/training_data.json")
    p.add_argument(
        "--input",
        default=r"c:\Users\revaa\viral2\data_viral_titles.csv",
        help="Input CSV containing video_id/title/description (default: project data_viral_titles.csv).",
    )
    p.add_argument(
        "--output",
        default=r"c:\Users\revaa\viral2\data\training_data.json",
        help="Output JSON path (default: project data/training_data.json).",
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

    transcript_api = _build_transcript_api()

    out_rows: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as ex:
        futures = [ex.submit(_process_one, r, transcript_api, args.debug) for r in deduped]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Fetching transcripts"):
            out_rows.append(fut.result())

    if args.drop_missing:
        out_rows = [r for r in out_rows if _safe_strip(r.get("full_transcript"))]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import json

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out_rows, f, indent=2, ensure_ascii=False)

    found = sum(1 for r in out_rows if _safe_strip(r.get("full_transcript")))
    print(f"Input unique video_ids: {len(deduped)}")
    print(f"Transcripts fetched (non-empty): {found}")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()

