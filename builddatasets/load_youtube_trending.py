"""
Load YouTube Trending Video Dataset from Kaggle.
Dataset: https://www.kaggle.com/datasets/rsrishav/youtube-trending-video-dataset

Saves the loaded data to a CSV so it can be pipelined with filter_videos_gemini.py.

Requires Kaggle API credentials:
  - KAGGLE_USERNAME and KAGGLE_KEY environment variables, or
  - ~/.kaggle/kaggle.json with your API credentials
"""

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")  # Handle special chars in Windows console

import zipfile
import pandas as pd
import kagglehub

# Install: pip install kagglehub[pandas-datasets]
# Output CSV path (used by filter_videos_gemini.py as default input)
SCRIPT_DIR = Path(__file__).resolve().parent
ORIGINAL_CSV = SCRIPT_DIR / "US_youtube_trending_data.csv"

# Download dataset (Kaggle stores as ZIP)
dataset_path = kagglehub.dataset_download(
    "rsrishav/youtube-trending-video-dataset",
    path=ORIGINAL_CSV.name,
)

# Extract CSV from ZIP and load
with zipfile.ZipFile(dataset_path, "r") as z:
    csv_name = z.namelist()[0]
    with z.open(csv_name) as f:
        df = pd.read_csv(
            f,
            encoding="latin-1",
            on_bad_lines="skip",
            engine="python",
        )

# Save to project folder so filter script (or pipeline) can use it
df.to_csv(ORIGINAL_CSV, index=False, encoding="utf-8")
print(f"Saved original dataset to: {ORIGINAL_CSV}")
print("First 5 records:", df.head())
print(f"\nShape: {df.shape}")
print(f"Columns: {list(df.columns)}")
