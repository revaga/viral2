#!/usr/bin/env python3
"""
Generate YouTube title predictions locally using a HuggingFace base model and
an optional LoRA adapter (PEFT).

Input JSONL should come from prep_title_sft_data.py (train or test):
  {"id": "...", "source": "...", "messages": [...]}

Output JSONL (for evaluation):
  {"id": "...", "source": "...", "reference": "...", "prediction": "..."}
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


def _extract_reference(messages: List[Dict[str, str]]) -> str:
    # Assistant message is the reference title in our prepared dataset
    for m in reversed(messages):
        if m.get("role") == "assistant":
            return (m.get("content") or "").strip()
    return ""


def _prompt_messages(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    # Remove assistant reference; we want the model to generate it.
    out = [m for m in messages if m.get("role") != "assistant"]
    return out


def _messages_to_text(tokenizer: Any, messages: List[Dict[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    # Fallback: plain concatenation
    parts = []
    for m in messages:
        role = (m.get("role") or "user").upper()
        content = m.get("content") or ""
        parts.append(f"{role}:\n{content}\n")
    parts.append("ASSISTANT:\n")
    return "\n".join(parts)


def main() -> None:
    p = argparse.ArgumentParser(description="Generate titles locally (base model + optional LoRA adapter).")
    p.add_argument("--base-model", required=True, help="HF model name or local path.")
    p.add_argument("--adapter-dir", default="", help="Optional PEFT adapter directory (from finetune_lora.py).")
    p.add_argument("--input-jsonl", required=True, help="Prepared JSONL (use test split for evaluation).")
    p.add_argument("--out", required=True, help="Output predictions JSONL.")
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--do-sample", action="store_true", help="Enable sampling (else greedy).")
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    in_path = Path(args.input_jsonl)
    if not in_path.exists():
        raise SystemExit(f"Input not found: {in_path}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Lazy imports
    import torch  # type: ignore[import-untyped]
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import-untyped]

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype="auto",
        device_map="auto",
    )

    if args.adapter_dir.strip():
        from peft import PeftModel  # type: ignore[import-untyped]

        model = PeftModel.from_pretrained(model, args.adapter_dir.strip())

    model.eval()

    if args.do_sample:
        torch.manual_seed(args.seed)

    rows = _load_jsonl(in_path)

    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            rec_id = str(r.get("id", ""))
            source = str(r.get("source", ""))
            messages = r.get("messages") or []
            if not isinstance(messages, list) or not messages:
                continue

            ref = _extract_reference(messages)
            prompt_msgs = _prompt_messages(messages)
            prompt_text = _messages_to_text(tokenizer, prompt_msgs)

            inputs = tokenizer(
                prompt_text,
                return_tensors="pt",
                truncation=True,
                max_length=tokenizer.model_max_length if tokenizer.model_max_length and tokenizer.model_max_length > 0 else 2048,
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.do_sample,
                    temperature=args.temperature if args.do_sample else None,
                    top_p=args.top_p if args.do_sample else None,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            gen = tokenizer.decode(out[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True).strip()
            # Heuristic cleanup for titles: first line only, strip quotes.
            pred = gen.splitlines()[0].strip().strip('"').strip("'")

            f.write(
                json.dumps(
                    {
                        "id": rec_id,
                        "source": source,
                        "reference": ref,
                        "prediction": pred,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(f"Wrote predictions: {out_path}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()

