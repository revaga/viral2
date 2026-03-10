#!/usr/bin/env python3
"""
LoRA fine-tuning for YouTube title generation using HuggingFace Transformers + PEFT + TRL.

Reads a single data JSONL and train (and optional eval) ID lists. No separate
train/test JSONL files; records are pulled from the data file by ID.

Usage:
  --data-jsonl <path>   Single JSONL with all records: id, source, source_text, title.
  --train-ids <path>   One id per line for training.
  --eval-ids <path>     Optional; one id per line for eval loss tracking.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List


SYSTEM_PROMPT = """You are an expert YouTube title generator. Your task is to create compelling, accurate titles that capture the essence of video content.

Guidelines for generating YouTube titles:
1. Be accurate and truthful - the title must reflect the actual content
2. Make it compelling and click-worthy while staying honest
3. Keep titles concise (ideally 50-80 characters, max 100 characters)
4. Use natural language that viewers would search for
5. Highlight the most interesting or valuable aspect
6. Create a curiosity gap without being misleading

Generate only the title itself, nothing else."""


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _load_id_list(path: Path) -> List[str]:
    """One id per line, strip whitespace, skip empty."""
    ids: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            sid = line.strip()
            if sid:
                ids.append(sid)
    return ids


def _data_by_id(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(r.get("id", "")): r for r in rows}


def _rows_for_ids(data_by_id: Dict[str, Dict[str, Any]], ids: List[str]) -> List[Dict[str, Any]]:
    """Return rows for given ids in order; skip missing."""
    return [data_by_id[i] for i in ids if i in data_by_id]


def _extract_texts(rows: List[Dict[str, Any]], tokenizer: Any) -> List[str]:
    """
    Turn compact records into chat messages + system prompt, then convert to
    model text using the tokenizer's chat template when available.
    """
    texts: List[str] = []
    for r in rows:
        source = (r.get("source") or "").strip().lower()
        source_text = r.get("source_text") or ""
        title = r.get("title") or ""

        label = "Transcript" if source == "transcript" else "Description"
        user = (
            f"Based on the following video {label.lower()}, generate an accurate and compelling YouTube title.\n\n"
            f"{label}:\n{source_text}\n\n"
            "Generate the YouTube title:"
        )
        msgs_with_system = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": title},
        ]
        if hasattr(tokenizer, "apply_chat_template"):
            txt = tokenizer.apply_chat_template(
                msgs_with_system,
                tokenize=False,
                add_generation_prompt=False,
            )
        else:
            parts = []
            for m in msgs_with_system:
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
    out = sorted(names)
    return out or ["q_proj", "v_proj"]


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "LoRA fine-tune a HF base model for title generation "
            "with the system prompt injected at training time."
        )
    )
    p.add_argument(
        "--data-jsonl",
        required=True,
        help="Data JSONL with all records (id, source, source_text, title).",
    )
    p.add_argument(
        "--train-ids",
        required=True,
        help="Train ID list (one id per line).",
    )
    p.add_argument(
        "--eval-ids",
        default="",
        help="Optional eval ID list for loss tracking (one id per line).",
    )
    p.add_argument(
        "--base-model",
        required=True,
        help="HF model name or local path.",
    )
    p.add_argument(
        "--run-name",
        required=True,
        help="Run name; outputs saved under --output-root/<run-name>/",
    )
    p.add_argument(
        "--output-root",
        default="runs",
        help="Root directory for runs (default: runs).",
    )

    # LoRA
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.1)
    p.add_argument(
        "--target-modules",
        default="",
        help="Comma-separated target module names (override inference).",
    )

    # Training hyperparams
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Per-device batch size (default: 1).",
    )
    p.add_argument(
        "--grad-accum",
        type=int,
        default=16,
        help="Gradient accumulation steps (default: 16).",
    )
    p.add_argument("--max-seq-len", type=int, default=1024)
    p.add_argument("--seed", type=int, default=1337)

    # Precision / memory
    p.add_argument("--bf16", action="store_true", help="Use bf16 if available.")
    p.add_argument("--fp16", action="store_true", help="Use fp16 if available.")
    p.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable gradient checkpointing.",
    )

    args = p.parse_args()

    data_path = Path(args.data_jsonl)
    train_ids_path = Path(args.train_ids)
    if not data_path.exists():
        raise SystemExit(f"Data file not found: {data_path}")
    if not train_ids_path.exists():
        raise SystemExit(f"Train IDs file not found: {train_ids_path}")
    eval_ids_path = Path(args.eval_ids) if args.eval_ids.strip() else None
    if eval_ids_path and not eval_ids_path.exists():
        raise SystemExit(f"Eval IDs file not found: {eval_ids_path}")

    out_dir = Path(args.output_root) / args.run_name
    adapter_dir = out_dir / "adapter"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Lazy imports so this file can be inspected without deps installed.
    from datasets import Dataset  # type: ignore[import-untyped]
    from peft import (  # type: ignore[import-untyped]
        LoraConfig,
        TaskType,
        get_peft_model,
    )
    from transformers import (  # type: ignore[import-untyped]
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
    )
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

    data_rows = _load_jsonl(data_path)
    data_by_id = _data_by_id(data_rows)
    train_ids = _load_id_list(train_ids_path)
    train_rows = _rows_for_ids(data_by_id, train_ids)
    if not train_rows:
        raise SystemExit("No training rows found for the given train IDs.")
    train_texts = _extract_texts(train_rows, tokenizer)
    train_ds = Dataset.from_dict({"text": train_texts})

    eval_ds = None
    if eval_ids_path:
        eval_ids = _load_id_list(eval_ids_path)
        eval_rows = _rows_for_ids(data_by_id, eval_ids)
        if eval_rows:
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
        "data_jsonl": str(data_path),
        "train_ids": str(train_ids_path),
        "eval_ids": str(eval_ids_path) if eval_ids_path else "",
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

