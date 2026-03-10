#!/usr/bin/env python3
"""
Create a **new version** of an existing Kaggle dataset from the local `data/` directory.

Default target: `reagarwa/youtube-transcripts-2`.

Behavior:
- Loads `KAGGLE_USERNAME` and `KAGGLE_KEY` from the environment (optionally via `.env`).
- Downloads the current contents of the Kaggle dataset into a temporary directory.
- Copies every file from your local `data/` directory into that temporary directory
  (overwriting existing files with the same relative path, if any).
- Creates a new Kaggle dataset version with the merged contents.

Usage (from project root):

    python -m scripts.upload_data_to_kaggle_version

Optional flags:
    --data-dir DATA_DIR      Override the source directory (default: <project_root>/data)
    --dataset-id DATASET_ID  Override the Kaggle dataset id (default: reagarwa/youtube-transcripts-2)
    --version-notes NOTES    Version notes for Kaggle (default: "Update from data/ directory")
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path


def _maybe_load_dotenv() -> None:
    try:
        from dotenv import load_dotenv  # type: ignore[import-untyped]

        load_dotenv()
    except Exception:
        # Fine if python-dotenv is missing; Kaggle can still use env or kaggle.json.
        pass


def _resolve_project_root() -> Path:
    # `scripts/` is one level below the project root.
    return Path(__file__).resolve().parents[1]


def upload_data_directory_to_existing_kaggle_dataset(
    data_dir: Path,
    dataset_id: str,
    version_notes: str,
) -> None:
    """
    Upload all files under `data_dir` to `dataset_id` as a new Kaggle dataset version.
    """
    from kaggle.api.kaggle_api_extended import KaggleApi  # type: ignore[import-untyped]

    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    print(f"Using data directory: {data_dir}")
    print(f"Target Kaggle dataset: {dataset_id}")

    api = KaggleApi()
    api.authenticate()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # 1. Download the current dataset (including its dataset-metadata.json).
        print(f"Downloading existing dataset contents for '{dataset_id}'...")
        api.dataset_download_files(
            dataset_id,
            path=str(tmp_path),
            unzip=True,
            quiet=False,
        )

        # 2. Copy local data files into the downloaded directory.
        print(f"Copying files from {data_dir} into temporary directory...")
        for src in data_dir.rglob("*"):
            if not src.is_file():
                continue
            rel = src.relative_to(data_dir)
            dst = tmp_path / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

        # 3. Create a new dataset version with the merged contents.
        print("Creating new Kaggle dataset version...")
        api.dataset_create_version(
            str(tmp_path),
            version_notes,
        )

        print("Dataset version created successfully.")


def main() -> None:
    _maybe_load_dotenv()

    project_root = _resolve_project_root()
    default_data_dir = project_root / "data"

    parser = argparse.ArgumentParser(
        description=(
            "Upload all files from a local data directory to an existing "
            "Kaggle dataset as a new version."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(default_data_dir),
        help=f"Source directory containing files to upload (default: {default_data_dir})",
    )
    parser.add_argument(
        "--dataset-id",
        type=str,
        default="reagarwa/youtube-transcripts-2",
        help="Target Kaggle dataset id (default: reagarwa/youtube-transcripts-2)",
    )
    parser.add_argument(
        "--version-notes",
        type=str,
        default="Update from data/ directory",
        help="Version notes for the new Kaggle dataset version.",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    dataset_id = args.dataset_id.strip()
    version_notes = args.version_notes.strip() or "Update dataset"

    # Basic safety check: require KAGGLE_USERNAME and KAGGLE_KEY to be present
    # in the environment (or rely on ~/.kaggle/kaggle.json if available).
    env_username = os.getenv("KAGGLE_USERNAME", "").strip()
    env_key = os.getenv("KAGGLE_KEY", "").strip()
    if not env_username or not env_key:
        print(
            "WARNING: KAGGLE_USERNAME and/or KAGGLE_KEY are not set in the environment.\n"
            "If you rely on a ~/.kaggle/kaggle.json file, this may still work, but "
            "if authentication fails, ensure these env vars are configured or add "
            "them to your .env file."
        )

    upload_data_directory_to_existing_kaggle_dataset(
        data_dir=data_dir,
        dataset_id=dataset_id,
        version_notes=version_notes,
    )


if __name__ == "__main__":
    main()

