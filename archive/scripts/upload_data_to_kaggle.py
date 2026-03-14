#!/usr/bin/env python3
"""
Create a **new Kaggle dataset** from all files in the local `data/` directory.

By default this script creates `youtube-transcripts-2` under your Kaggle account,
e.g. `reagarwa/youtube-transcripts-2`, using the credentials in your environment.

Behavior:
- Loads `KAGGLE_USERNAME` and `KAGGLE_KEY` from the environment (optionally via `.env`).
- Copies every file from your local `data/` directory into a temporary directory.
- Initializes Kaggle dataset metadata in that directory.
- Creates a brand new Kaggle dataset with id `<username>/youtube-transcripts-2`.

Usage (from project root):

    python -m scripts.upload_data_to_kaggle

Optional flags:
    --data-dir DATA_DIR          Override the source directory (default: <project_root>/data)
    --username USERNAME          Override Kaggle username (default: $KAGGLE_USERNAME)
    --dataset-slug SLUG          Override dataset slug (default: youtube-transcripts-2)
    --title TITLE                Dataset title (default: 'YouTube transcripts 2')
    --subtitle SUBTITLE          Short subtitle (optional)
    --description DESCRIPTION    Long description (optional)
    --private                    Make dataset private (default: public)
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional


def _maybe_load_dotenv() -> None:
    """
    Best-effort import of python-dotenv so we can read `.env` when available
    without making it a hard dependency.
    """
    try:
        from dotenv import load_dotenv  # type: ignore[import-untyped]

        load_dotenv()
    except Exception:
        # It's fine if python-dotenv is not installed; Kaggle can still read
        # credentials from the real environment (or ~/.kaggle/kaggle.json).
        pass


def _resolve_project_root() -> Path:
    # `scripts/` is one level below the project root.
    return Path(__file__).resolve().parents[1]


def create_new_kaggle_dataset_from_data(
    data_dir: Path,
    username: str,
    dataset_slug: str,
    title: str,
    subtitle: Optional[str],
    description: Optional[str],
    private: bool,
) -> None:
    """
    Create a brand new Kaggle dataset from all files under `data_dir`.

    The dataset id will be `{username}/{dataset_slug}`.
    """
    import json

    from kaggle.api.kaggle_api_extended import KaggleApi  # type: ignore[import-untyped]

    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    username = username.strip()
    if not username:
        raise ValueError(
            "Kaggle username is required. Provide --username or set KAGGLE_USERNAME."
        )

    dataset_slug = dataset_slug.strip()
    if not dataset_slug:
        raise ValueError("Dataset slug cannot be empty.")

    dataset_id = f"{username}/{dataset_slug}"

    print(f"Using data directory: {data_dir}")
    print(f"Creating new Kaggle dataset: {dataset_id}")

    api = KaggleApi()
    api.authenticate()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # Copy local data files into the temporary directory.
        print(f"Copying files from {data_dir} into temporary directory...")
        for src in data_dir.rglob("*"):
            if not src.is_file():
                continue
            rel = src.relative_to(data_dir)
            dst = tmp_path / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

        # Initialize metadata file via Kaggle API helper.
        print("Initializing dataset metadata...")
        api.dataset_initialize(str(tmp_path))

        meta_path = tmp_path / "dataset-metadata.json"
        if not meta_path.is_file():
            raise FileNotFoundError(
                f"Expected metadata file not found at {meta_path}. "
                "Kaggle API may have changed; please check your kaggle package version."
            )

        # Update metadata with our desired values.
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)

        meta["title"] = title
        meta["id"] = dataset_id
        if subtitle is not None:
            meta["subtitle"] = subtitle
        if description is not None:
            meta["description"] = description

        # Ensure licenses is present; default to CC0-1.0 if missing.
        if not meta.get("licenses"):
            meta["licenses"] = [{"name": "CC0-1.0"}]

        # Reflect privacy both in metadata and API call.
        meta["isPrivate"] = bool(private)

        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        print("Creating new Kaggle dataset...")
        api.dataset_create_new(
            str(tmp_path),
            public=not private,
            quiet=False,
        )

        print("New Kaggle dataset created successfully.")


def main() -> None:
    _maybe_load_dotenv()

    project_root = _resolve_project_root()
    default_data_dir = project_root / "data"

    parser = argparse.ArgumentParser(
        description=(
            "Create a new Kaggle dataset from all files in a local data directory."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(default_data_dir),
        help=f"Source directory containing files to upload (default: {default_data_dir})",
    )
    parser.add_argument(
        "--username",
        type=str,
        default="",
        help="Kaggle username (default: taken from KAGGLE_USERNAME).",
    )
    parser.add_argument(
        "--dataset-slug",
        type=str,
        default="youtube-transcripts-2",
        help="Dataset slug to create under the username (default: youtube-transcripts-2).",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="YouTube transcripts 2",
        help="Title for the new Kaggle dataset.",
    )
    parser.add_argument(
        "--subtitle",
        type=str,
        default="",
        help="Optional short subtitle for the dataset.",
    )
    parser.add_argument(
        "--description",
        type=str,
        default="",
        help="Optional long description for the dataset.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the dataset as private (default: public).",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()

    # Determine Kaggle username preference order: CLI arg > env var.
    username = (args.username or os.getenv("KAGGLE_USERNAME", "")).strip()

    # Basic safety check: require username and either env vars or kaggle.json.
    env_key = os.getenv("KAGGLE_KEY", "").strip()
    if not username:
        raise SystemExit(
            "Kaggle username is required. Provide --username or set KAGGLE_USERNAME."
        )
    if not env_key:
        print(
            "WARNING: KAGGLE_KEY is not set in the environment.\n"
            "If you rely on a ~/.kaggle/kaggle.json file, this may still work, but "
            "if authentication fails, ensure KAGGLE_KEY is configured or use kaggle.json."
        )

    subtitle: Optional[str] = args.subtitle.strip() or None
    description: Optional[str] = args.description.strip() or None

    create_new_kaggle_dataset_from_data(
        data_dir=data_dir,
        username=username,
        dataset_slug=args.dataset_slug,
        title=args.title.strip() or "YouTube transcripts 2",
        subtitle=subtitle,
        description=description,
        private=bool(args.private),
    )


if __name__ == "__main__":
    main()

