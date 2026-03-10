#!/usr/bin/env python3
"""
This script has been deprecated.

The functionality to build the training dataset now lives entirely in
`scripts/fetch_transcripts.py`, which:
  - fetches transcripts directly from YouTube
  - writes a JSONL file incrementally every few transcripts
  - tracks retryable failures separately from "no transcript available"

Please update any calls to this script to use `scripts/fetch_transcripts.py` instead.
"""

from __future__ import annotations

import sys


def main() -> None:
    raise SystemExit(
        "pull_existing_transcripts.py has been deprecated.\n"
        "Use scripts/fetch_transcripts.py for building the training dataset."
    )


if __name__ == "__main__":
    main()

