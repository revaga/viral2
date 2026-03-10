## Viral title LoRA training (deterministic split)

### Install

```bash
pip install -r requirements_lora_eval.txt
```

### (Optional) Hydrate transcripts from an existing JSONL first

If you already have a big transcript dump JSONL, pull matching transcripts before fetching from YouTube:

```bash
python scripts\pull_existing_transcripts.py --missing-out "c:\Users\revaa\viral2\data\missing_video_ids.txt"
```

### (Optional) Fetch transcripts from YouTube by `video_id`

```bash
python scripts\fetch_transcripts.py --input "c:\Users\revaa\viral2\data_viral_titles.csv" --output "c:\Users\revaa\viral2\data\training_data.json" --max-workers 10
```

### 1) Prepare deterministic train/test split

```bash
python prep_title_sft_data.py --input "c:\Users\revaa\viral2\data_viral_titles.csv" --out-train title_sft_train.jsonl --out-test title_sft_test.jsonl --test-size 0.1 --seed 1337 --stratify-source
```

### 2) Fine-tune LoRA (train only)

```bash
python finetune_lora.py --train title_sft_train.jsonl --eval title_sft_test.jsonl --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --run-name tinyllama_lora_v1 --output-root runs --epochs 2
```

### 3) Generate predictions on held-out test

```bash
python generate_titles_local.py --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --adapter-dir "runs\tinyllama_lora_v1\adapter" --input-jsonl title_sft_test.jsonl --out "runs\tinyllama_lora_v1\predictions_test.jsonl"
```

### 4) Evaluate metrics on held-out test

```bash
python eval\evaluate_run.py --predictions "runs\tinyllama_lora_v1\predictions_test.jsonl" --out "runs\tinyllama_lora_v1\metrics_test.json" --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --adapter-dir "runs\tinyllama_lora_v1\adapter"
```

### 5) Compare multiple runs (defaults to test metrics)

```bash
python eval\compare_runs.py
```

