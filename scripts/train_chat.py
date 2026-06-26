#!/usr/bin/env python3
"""Supervised fine-tune (SFT) a chat model to answer in an author's voice.

Trains on the conversational JSONL from build_chat_data.py. Uses the model's
chat template and MASKS THE LOSS on the system+user tokens, so the model only
learns to *produce* the assistant (author-voice) reply — not to echo prompts.
Starting from an *-Instruct* base means multi-turn chat ability is retained.

Usage:
    python scripts/train_chat.py weil
    python scripts/train_chat.py weil --epochs 4 --model Qwen/Qwen2.5-1.5B-Instruct
"""
import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainerCallback, TrainingArguments)

from persona import system_for

ROOT = Path(__file__).resolve().parent.parent
MAX_LEN = 512


class MPSCacheFlush(TrainerCallback):
    """MPS doesn't release its allocator cache across steps; flush periodically
    so memory doesn't balloon and push the machine into swap."""
    def on_step_end(self, args, state, control, **kw):
        if state.global_step % 10 == 0 and torch.backends.mps.is_available():
            torch.mps.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("author")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    data_path = ROOT / "authors" / args.author / "data" / "chat" / f"{args.author}_sft.jsonl"
    out_dir = ROOT / "authors" / args.author / "chat_model"
    if not data_path.exists():
        ap.error(f"no SFT data at {data_path} — run build_chat_data.py first")
    system = system_for(args.author)
    print(f"author={args.author}  base={args.model}  device={device}")

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def encode(ex):
        convo = [{"role": "system", "content": system}] + ex["messages"]
        # render to text first, then tokenize -> guarantees plain int lists
        prompt_str = tok.apply_chat_template(convo[:-1], tokenize=False,
                                             add_generation_prompt=True)
        full_str = tok.apply_chat_template(convo, tokenize=False,
                                           add_generation_prompt=False)
        prompt_ids = tok(prompt_str, add_special_tokens=False)["input_ids"]
        full_ids = tok(full_str, add_special_tokens=False)["input_ids"]
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
        return {"input_ids": full_ids[:MAX_LEN], "labels": labels[:MAX_LEN]}

    ds = load_dataset("json", data_files=str(data_path), split="train")
    ds = ds.map(encode, remove_columns=ds.column_names)
    print(f"examples: {len(ds)}")

    def collate(batch):
        m = max(len(b["input_ids"]) for b in batch)
        ids, labs, att = [], [], []
        for b in batch:
            pad = m - len(b["input_ids"])
            ids.append(b["input_ids"] + [tok.pad_token_id] * pad)
            labs.append(b["labels"] + [-100] * pad)
            att.append([1] * len(b["input_ids"]) + [0] * pad)
        return {"input_ids": torch.tensor(ids), "labels": torch.tensor(labs),
                "attention_mask": torch.tensor(att)}

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    model.config.use_cache = False
    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    targs = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
    )
    Trainer(model=model, args=targs, train_dataset=ds, data_collator=collate,
            processing_class=tok, callbacks=[MPSCacheFlush()]).train()

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    tok.save_pretrained(str(out_dir))
    (out_dir / "BASE_MODEL.txt").write_text(args.model + "\n")
    print(f"\nsaved chat adapter -> {out_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
