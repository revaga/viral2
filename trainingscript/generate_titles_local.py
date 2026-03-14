#!/usr/bin/env python3
"""
Generate YouTube title predictions locally using a HuggingFace base model and
an optional LoRA adapter (PEFT). Pulls records from the same data JSONL used
for training, filtered by an ID list (e.g. test IDs).

Usage:
  --data-jsonl <path>   Single JSONL with all records: id, source, source_text, title.
  --input-ids <path>    One id per line for which to generate (e.g. test IDs).

Output JSONL (for evaluation):
  {"id": "...", "source": "...", "reference": "...", "prediction": "..."}

Prompts are built only in memory from the data file.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List


def _maybe_load_dotenv() -> None:
    """Load .env so HF_TOKEN (or HUGGING_FACE_HUB_TOKEN) is available for gated models."""
    try:
        from dotenv import load_dotenv  # type: ignore[import-untyped]
        load_dotenv()
    except ImportError:
        pass


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


def _build_messages(source: str, source_text: str, reference_title: str) -> Dict[str, Any]:
    """
    Construct chat-style messages for inference, using the same pattern as training:
      - system: title-generation instructions
      - user:  instruction + source text
      - assistant (reference): the ground-truth title (used only as reference)
    """
    label = "Transcript" if source == "transcript" else "Description"
    user = (
        f"Based on the following video {label.lower()}, generate an accurate and compelling YouTube title.\n\n"
        f"{label}:\n{source_text}\n\n"
        "Generate the YouTube title:"
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
    reference = (reference_title or "").strip()
    if reference:
        messages.append({"role": "assistant", "content": reference})
    return {"messages": messages, "reference": reference}


def _messages_to_prompt(tokenizer: Any, messages: List[Dict[str, str]]) -> str:
    """
    Convert messages into a generation prompt.
    We *exclude* the assistant reference from the prompt so the model generates it.
    """
    # Drop any assistant messages (references) before prompting.
    prompt_messages = [m for m in messages if (m.get("role") or "").lower() != "assistant"]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    parts = []
    for m in prompt_messages:
        role = (m.get("role") or "user").upper()
        content = m.get("content") or ""
        parts.append(f"{role}:\n{content}\n")
    parts.append("ASSISTANT:\n")
    return "\n".join(parts)


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Generate titles locally (base model + optional LoRA adapter) using "
            "compact training records and runtime prompt construction."
        )
    )
    p.add_argument("--base-model", required=True, help="HF model name or local path.")
    p.add_argument(
        "--adapter-dir",
        default="",
        help="Optional PEFT adapter directory (from new finetune_lora.py).",
    )
    p.add_argument(
        "--data-jsonl",
        required=True,
        help="Data JSONL with all records (same as used for training).",
    )
    p.add_argument(
        "--input-ids",
        required=True,
        help="ID list to generate for, one id per line (e.g. test IDs).",
    )
    p.add_argument("--out", required=True, help="Output predictions JSONL.")
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--do-sample", action="store_true", help="Enable sampling (else greedy).")
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    _maybe_load_dotenv()

    data_path = Path(args.data_jsonl)
    input_ids_path = Path(args.input_ids)
    if not data_path.exists():
        raise SystemExit(f"Data file not found: {data_path}")
    if not input_ids_path.exists():
        raise SystemExit(f"Input IDs file not found: {input_ids_path}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Lazy imports
    import torch  # type: ignore[import-untyped]
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import-untyped]

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        use_fast=True,
        token=hf_token,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype="auto",
        device_map="auto",
        token=hf_token,
    )

    if args.adapter_dir.strip():
        from peft import PeftModel  # type: ignore[import-untyped]

        adapter_dir = Path(args.adapter_dir.strip()).resolve()
        if not adapter_dir.is_dir():
            raise SystemExit(f"Adapter directory not found or not a directory: {adapter_dir}")
        if not (adapter_dir / "adapter_config.json").exists():
            raise SystemExit(
                f"Adapter config not found at {adapter_dir / 'adapter_config.json'}. "
                "Check that the path is correct and you are running from the project root."
            )
        model = PeftModel.from_pretrained(model, str(adapter_dir))

    model.eval()

    if args.do_sample:
        torch.manual_seed(args.seed)

    data_rows = _load_jsonl(data_path)
    data_by_id = _data_by_id(data_rows)
    input_ids = _load_id_list(input_ids_path)
    rows = _rows_for_ids(data_by_id, input_ids)
    if not rows:
        raise SystemExit("No rows found for the given input IDs.")

    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            rec_id = str(r.get("id", ""))
            source = str(r.get("source", "")).strip().lower()
            source_text = r.get("source_text") or ""
            reference_title = r.get("title") or ""

            built = _build_messages(source, source_text, reference_title)
            messages = built["messages"]
            reference = built["reference"]

            prompt_text = _messages_to_prompt(tokenizer, messages)

            # Cap max_length to avoid OverflowError in tokenizer (e.g. Gemma 3 reports 128K+)
            _max_len = tokenizer.model_max_length if tokenizer.model_max_length and tokenizer.model_max_length > 0 else 2048
            _max_len = min(_max_len, 32768)

            inputs = tokenizer(
                prompt_text,
                return_tensors="pt",
                truncation=True,
                max_length=_max_len,
                add_special_tokens=False,
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

            gen = tokenizer.decode(
                out[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True
            ).strip()
            
            # Heuristic cleanup for titles: first line only, strip quotes.
            lines = gen.splitlines()
            pred = lines[0].strip().strip('"').strip("'") if lines else ""
            
            if not pred:
                print(f"Warning: Model generated an empty string for ID {rec_id}")

            f.write(
                json.dumps(
                    {
                        "id": rec_id,
                        "source": source,
                        "reference": reference,
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

