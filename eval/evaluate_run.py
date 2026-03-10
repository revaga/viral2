#!/usr/bin/env python3
"""
Compute evaluation metrics for a run from a predictions JSONL file.

Input predictions JSONL lines:
  {"id": "...", "source": "...", "reference": "...", "prediction": "..."}

Outputs metrics JSON (dict of floats).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _load_preds(path: Path) -> Tuple[List[str], List[str]]:
    refs: List[str] = []
    preds: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            refs.append((r.get("reference") or "").strip())
            preds.append((r.get("prediction") or "").strip())
    return refs, preds


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate a run from predictions JSONL.")
    p.add_argument("--predictions", required=True, help="Path to predictions_test.jsonl (or similar).")
    p.add_argument("--out", required=True, help="Where to write metrics JSON.")
    p.add_argument("--embedding-model", default="all-mpnet-base-v2")
    p.add_argument("--base-model", default="", help="Optional base model name/path for base perplexity on refs/preds.")
    p.add_argument("--adapter-dir", default="", help="Optional adapter dir to compute tuned perplexity on refs/preds.")
    p.add_argument("--device", default="", help="Optional: cpu/cuda")
    args = p.parse_args()

    pred_path = Path(args.predictions)
    if not pred_path.exists():
        raise SystemExit(f"Predictions not found: {pred_path}")

    refs, preds = _load_preds(pred_path)

    finetuned_model = None
    finetuned_tokenizer = None
    base_model_name = args.base_model.strip() or None

    # Optional: load tuned model for perplexity_self_* metrics
    if base_model_name and args.adapter_dir.strip():
        import torch  # type: ignore[import-untyped]
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import-untyped]
        from peft import PeftModel  # type: ignore[import-untyped]

        device = args.device.strip() or ("cuda" if torch.cuda.is_available() else "cpu")
        base_tok = AutoTokenizer.from_pretrained(base_model_name)
        if base_tok.pad_token is None:
            base_tok.pad_token = base_tok.eos_token
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
        base_model.to(device)
        base_model.eval()

        finetuned_model = PeftModel.from_pretrained(base_model, args.adapter_dir.strip())
        finetuned_model.to(device)
        finetuned_model.eval()
        finetuned_tokenizer = base_tok

    from eval.viral_title_metrics import compute_metrics

    metrics = compute_metrics(
        references=refs,
        predictions=preds,
        corpus_titles=None,
        embedding_model_name=args.embedding_model,
        base_model_name=base_model_name,
        finetuned_model=finetuned_model,
        finetuned_tokenizer=finetuned_tokenizer,
        device=(args.device.strip() or None),
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Wrote metrics: {out_path}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()

