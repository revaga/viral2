#!/usr/bin/env python3
"""
LoRA fine-tuning for YouTube title generation using HuggingFace Transformers + PEFT + TRL.

Trains on the *train* JSONL produced by prep_title_sft_data.py and (optionally)
evaluates on a held-out *test* JSONL for loss monitoring. Final metrics should
still be computed by generating on the held-out test set and running eval scripts.

Expected input JSONL format (one object per line):
  {"id": "...", "source": "...", "messages": [{"role":"system","content":"..."}, ...]}
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _extract_texts(rows: List[Dict[str, Any]], tokenizer: Any) -> List[str]:
    """
    Convert chat messages to model text using tokenizer chat template when available.
    Falls back to a simple concatenation if chat template missing.
    """
    texts: List[str] = []
    for r in rows:
        msgs = r.get("messages") or []
        if hasattr(tokenizer, "apply_chat_template"):
            txt = tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=False,
            )
        else:
            # Minimal fallback; not ideal but keeps script usable for base models without templates.
            parts = []
            for m in msgs:
                role = m.get("role", "user")
                content = m.get("content", "")
                parts.append(f"{role.upper()}:\n{content}\n")
            txt = "\n".join(parts)
        texts.append(txt)
    return texts


def _infer_target_modules(model: Any) -> List[str]:
    """
    Best-effort inference of common attention projection module names for LoRA.
    Users can override via --target-modules.
    """
    names = set()
    for n, _ in model.named_modules():
        if n.endswith(("q_proj", "k_proj", "v_proj", "o_proj")):
            names.add(n.split(".")[-1])
        if n.endswith(("c_attn", "c_proj")):  # GPT-2 style
            names.add(n.split(".")[-1])
        if n.endswith(("Wqkv", "Wo")):  # some architectures
            names.add(n.split(".")[-1])
    # Return unique leaf module names
    out = sorted(names)
    return out or ["q_proj", "v_proj"]


def main() -> None:
    p = argparse.ArgumentParser(description="LoRA fine-tune a HF base model for title generation.")
    p.add_argument("--train", required=True, help="Train JSONL (e.g. title_sft_train.jsonl).")
    p.add_argument("--eval", default="", help="Optional eval JSONL for loss tracking (e.g. title_sft_test.jsonl).")
    p.add_argument("--base-model", required=True, help="HF model name or local path.")
    p.add_argument("--run-name", required=True, help="Run name; outputs saved under --output-root/<run-name>/")
    p.add_argument("--output-root", default="runs", help="Root directory for runs (default: runs).")

    # LoRA
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.1)
    p.add_argument("--target-modules", default="", help="Comma-separated target module names (override inference).")

    # Training hyperparams
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=1, help="Per-device batch size (default: 1).")
    p.add_argument("--grad-accum", type=int, default=16, help="Gradient accumulation steps (default: 16).")
    p.add_argument("--max-seq-len", type=int, default=1024)
    p.add_argument("--seed", type=int, default=1337)

    # Precision / memory
    p.add_argument("--bf16", action="store_true", help="Use bf16 if available.")
    p.add_argument("--fp16", action="store_true", help="Use fp16 if available.")
    p.add_argument("--gradient-checkpointing", action="store_true", help="Enable gradient checkpointing.")

    args = p.parse_args()

    train_path = Path(args.train)
    if not train_path.exists():
        raise SystemExit(f"Train file not found: {train_path}")
    eval_path = Path(args.eval) if args.eval else None
    if eval_path and not eval_path.exists():
        raise SystemExit(f"Eval file not found: {eval_path}")

    out_dir = Path(args.output_root) / args.run_name
    adapter_dir = out_dir / "adapter"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Lazy imports so this file can be inspected without deps installed.
    from datasets import Dataset  # type: ignore[import-untyped]
    from peft import LoraConfig, TaskType, get_peft_model  # type: ignore[import-untyped]
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import-untyped]
    from transformers import TrainingArguments  # type: ignore[import-untyped]
    from trl import SFTTrainer  # type: ignore[import-untyped]

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype="auto",
        device_map="auto",
    )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if args.target_modules.strip():
        target_modules = [t.strip() for t in args.target_modules.split(",") if t.strip()]
    else:
        target_modules = _infer_target_modules(model)

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)

    train_rows = _load_jsonl(train_path)
    train_texts = _extract_texts(train_rows, tokenizer)
    train_ds = Dataset.from_dict({"text": train_texts})

    eval_ds = None
    if eval_path:
        eval_rows = _load_jsonl(eval_path)
        eval_texts = _extract_texts(eval_rows, tokenizer)
        eval_ds = Dataset.from_dict({"text": eval_texts})

    # TrainingArguments: keep eval optional; metrics come later from held-out generation.
    # Transformers has used both `evaluation_strategy` and `eval_strategy` across versions.
    ta_kwargs = dict(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        eval_steps=200 if eval_ds is not None else None,
        save_steps=200,
        logging_steps=25,
        save_total_limit=2,
        seed=args.seed,
        bf16=args.bf16,
        fp16=args.fp16,
        report_to=[],
        max_steps=-1,
    )
    eval_strategy_value = "steps" if eval_ds is not None else "no"
    try:
        import inspect

        ta_params = inspect.signature(TrainingArguments.__init__).parameters
        if "evaluation_strategy" in ta_params:
            ta_kwargs["evaluation_strategy"] = eval_strategy_value
        elif "eval_strategy" in ta_params:
            ta_kwargs["eval_strategy"] = eval_strategy_value
        else:
            ta_kwargs["evaluation_strategy"] = eval_strategy_value
    except Exception:
        ta_kwargs["evaluation_strategy"] = eval_strategy_value

    training_args = TrainingArguments(**ta_kwargs)

    # TRL's SFTTrainer signature has changed across versions. Build kwargs dynamically.
    import inspect

    sft_kwargs = dict(
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        dataset_text_field="text",
        max_seq_length=args.max_seq_len,
        args=training_args,
        packing=False,
    )
    try:
        sft_params = inspect.signature(SFTTrainer.__init__).parameters
        if "tokenizer" in sft_params:
            sft_kwargs["tokenizer"] = tokenizer
        elif "processing_class" in sft_params:
            # Newer TRL uses `processing_class` (a tokenizer/processor-like object).
            sft_kwargs["processing_class"] = tokenizer

        # Drop keys not supported by this TRL version.
        sft_kwargs = {k: v for k, v in sft_kwargs.items() if k in sft_params}
    except Exception:
        # Best-effort fallback for older versions.
        sft_kwargs["tokenizer"] = tokenizer

    trainer = SFTTrainer(**sft_kwargs)

    trainer.train()

    adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    # Save a small run manifest for reproducibility
    manifest = {
        "base_model": args.base_model,
        "run_name": args.run_name,
        "train_file": str(train_path),
        "eval_file": str(eval_path) if eval_path else "",
        "lora": {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "target_modules": target_modules,
        },
        "training": {
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "max_seq_len": args.max_seq_len,
            "seed": args.seed,
            "bf16": args.bf16,
            "fp16": args.fp16,
            "gradient_checkpointing": args.gradient_checkpointing,
        },
    }
    with (out_dir / "run_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved adapter to: {adapter_dir}")


if __name__ == "__main__":
    # Avoid HF tokenizer parallelism warning spam on Windows.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()

