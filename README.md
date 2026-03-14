# Viral Title LoRA Training

A full pipeline for building a **viral YouTube title dataset** from trending data and **fine-tuning a small LM with LoRA** to generate click-worthy, curiosity-driven titles from video transcripts or descriptions.

## Overview

1. **Data pipeline**: Clean YouTube trending CSV → rule-based filtering → transcript fetch → deterministic train/test split.
2. **Training**: LoRA fine-tuning (HuggingFace Transformers + PEFT + TRL) on transcript to generate titles.
3. **Evaluation**: ROUGE, embedding similarity, sentiment, diversity, and perplexity metrics.

---

## Requirements

- Python 3.10+
- (Optional) CUDA for faster training and inference
- (Optional) [Kaggle API](https://github.com/Kaggle/kaggle-api) credentials for downloading the trending dataset
- (Optional) `GEMINI_API_KEY` or `GOOGLE_API_KEY` in `.env` for Gemini-based video filtering
- (Optional) `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN` in `.env` for gated base models (e.g. Llama)

### Install

```bash
pip install -r requirements.txt
```

---

## Project Layout

| Path | Purpose |
|------|--------|
| `cleandatascripts/` | Clean raw trending CSV (remove channels, categories, low-quality titles) |
| `builddatasets/` | Build viral-title CSV (engagement thresholds, heuristics, dedup), load Kaggle dataset |
| `trainingscript/` | Prep SFT data, LoRA fine-tune, local title generation |
| `eval/` | Compute metrics (ROUGE, cosine, sentiment, diversity, perplexity), compare runs |
| `archive/` | Older scripts: fetch transcripts, Gemini filter, legacy prep/finetune/generate |

---

## Data Pipeline (End-to-End)

### 1. Get raw trending data (optional)

If you don’t have the CSV yet, download the [YouTube Trending Video Dataset](https://www.kaggle.com/datasets/rsrishav/youtube-trending-video-dataset) via Kaggle (requires `KAGGLE_USERNAME` and `KAGGLE_KEY` or `~/.kaggle/kaggle.json`):

```bash
python builddatasets/load_youtube_trending.py
```

This writes `builddatasets/US_youtube_trending_data.csv`. Copy or symlink it to the project root as `US_youtube_trending_data.csv` if your clean script expects it there.

### 2. Clean the CSV

Remove unwanted channels (e.g. NFL, Netflix, news), high-volume channels, and title patterns (e.g. "Official", "MV"):

```bash
python cleandatascripts/clean_data.py
# Or with explicit paths:
python cleandatascripts/clean_data.py --output data_cleaned.csv --max-videos 300
```

Input: `US_youtube_trending_data.csv` (project root). Output: `data_cleaned.csv`.

### 3. Build viral-title candidate set

From `data_cleaned.csv`, apply per-channel caps, engagement thresholds (likes/views, comments/views), category and pattern filters, and text heuristics; deduplicate. Output is intended for optional Gemini filtering and/or transcript fetch.

```bash
python builddatasets/build_viral_title_dataset.py
# Or:
python builddatasets/build_viral_title_dataset.py --input data_cleaned.csv --output data_viral_titles.csv --english-only --max-per-channel 30
```

### 4. Fetch transcripts from YouTube

Build a training JSON from a CSV that has `video_id` and `title` (e.g. `data_viral_titles.csv`). If you hit IP/request blocks, see the [youtube-transcript-api docs](https://github.com/jdepoix/youtube-transcript-api?tab=readme-ov-file#working-around-ip-bans-requestblocked-or-ipblocked-exception) for workarounds.

```bash
python builddatasets/fetch_transcripts.py --input data_viral_titles.csv --output data/training_data.json --max-workers 10
```

---

## Training Pipeline (Deterministic Split)

The active training code uses a **single data JSONL** plus **train/test ID files**. Scripts live in `trainingscript/` and `eval/`.

### 1. Prepare SFT data and train/test split

From a CSV or JSON/JSONL (with `video_id`, `title`, and `full_transcript` or `description`), produce one data JSONL and ID lists:

```bash
python trainingscript/prep_title_sft_data.py --input data/training_data.json --out-data title_sft_data.jsonl --out-train-ids title_sft_train_ids.txt --out-test-ids title_sft_test_ids.txt --test-size 0.1 --seed 1337 --stratify-source
```

- `--input`: CSV or JSON/JSONL (e.g. output of `fetch_transcripts.py`).
- `--out-data`: Single JSONL with `id`, `source`, `source_text`, `title`.
- `--out-train-ids` / `--out-test-ids`: One ID per line for train and test.

### 2. Fine-tune LoRA

```bash
python trainingscript/finetune_lora.py --data-jsonl title_sft_data.jsonl --train-ids title_sft_train_ids.txt --eval-ids title_sft_test_ids.txt --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --run-name tinyllama_lora_v1 --output-root runs --epochs 2
```

- Uses the same system prompt as generation (expert YouTube title generator).
- Adapter is saved under `runs/<run-name>/adapter`.

### 2b. Running a base model

```bash
python trainingscript/generate_titles_local.py ^
  --base-model <MODEL_NAME_OR_PATH> ^
  --data-jsonl <path/to/your_data.jsonl> ^
  --input-ids <path/to/test_ids.txt> ^
  --out <path/to/predictions.jsonl>
```

### 3. Generate titles on the test set

```bash
python trainingscript/generate_titles_local.py --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --adapter-dir runs/tinyllama_lora_v1/adapter --data-jsonl title_sft_data.jsonl --input-ids title_sft_test_ids.txt --out runs/tinyllama_lora_v1/predictions_test.jsonl
```

Output JSONL: `id`, `source`, `reference`, `prediction` (for evaluation).

### 4. Evaluate metrics for test set

```bash
python eval/evaluate_run.py --predictions runs/tinyllama_lora_v1/predictions_test.jsonl --out runs/tinyllama_lora_v1/metrics_test.json --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --adapter-dir runs/tinyllama_lora_v1/adapter
```

- Computes ROUGE, embedding cosine (corpus mean, max sim, mode collapse), sentiment, diversity (TTR, self-BLEU), and optional base/tuned perplexity on references and predictions.

### 5. Compare multiple runs

```bash
python eval/compare_runs.py
# Or: python eval/compare_runs.py --metrics runs/*/metrics_test.json
```

Prints a compact table of key metrics across runs.

---

## Evaluation Metrics (Summary)

| Metric | Interpretation |
|--------|-----------------|
| **ROUGE-L F1** | Structural overlap / hook similarity to references |
| **Cosine (corpus mean / max sim)** | Style proximity to real viral titles; max sim can indicate memorization |
| **Mode collapse (prop >0.95)** | Risk of repetitive or collapsed outputs |
| **Sentiment (compound abs mean)** | Emotional intensity; near zero = bland |
| **Diversity (TTR / self-BLEU)** | Lexical and n-gram variety |
| **Perplexity (base vs tuned on refs)** | Tuned model should find viral refs less surprising (style transfer) |

Details and defaults: `eval/viral_title_metrics.py`.

---

## Legacy / Alternative Workflow (Archive)

The `archive/` folder contains an older workflow that uses **separate train/test JSONL** files and different script locations:

- `prep_title_sft_data.py` with `--out-train title_sft_train.jsonl --out-test title_sft_test.jsonl`
- `finetune_lora.py` / `generate_titles_local.py` that read those JSONL files directly

---

## Environment and Data Files

- **`.env`**: Optional; use for `GEMINI_API_KEY`, `GOOGLE_API_KEY`, `HF_TOKEN`, `KAGGLE_USERNAME`, `KAGGLE_KEY`, and proxy vars if needed by `fetch_transcripts`.
- **Large CSVs** (`US_youtube_trending_data.csv`, `data_cleaned.csv`, `data_viral_titles.csv`) are gitignored; keep them local or use Kaggle/datasets.
- **Outputs**: `runs/`, `wandb/`, `data/`, and generated `title_sft_*.jsonl` / `training_data*.json` are gitignored.

---

## License and Attribution

- YouTube data from [YouTube Trending Video Dataset](https://www.kaggle.com/datasets/rsrishav/youtube-trending-video-dataset) (Kaggle).
- Base models (e.g. TinyLlama) follow their respective licenses (see Hugging Face model cards).
