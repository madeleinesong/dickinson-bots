#!/usr/bin/env python3
"""LoRA fine-tune a small base LLM on one author's voice.

This is *continued pretraining* on raw prose (no instruction format): the model
learns to continue text in the author's style. Mirrors the original repo's
GPT-2 approach, but with a modern base model + LoRA so it's fast on Apple MPS.

Usage:
    python scripts/train.py weil
    python scripts/train.py weil --max-steps 30          # quick smoke test
    python scripts/train.py weil --model Qwen/Qwen2.5-1.5B --epochs 3

Output: authors/<author>/model/  (LoRA adapter + tokenizer)
"""
import argparse
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("author")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B",
                    help="base model (use a -1.5B / -3B for a stronger voice)")
    ap.add_argument("--block-size", type=int, default=512)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--max-steps", type=int, default=-1,
                    help="cap steps (overrides epochs); use for a smoke test")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    corpus_path = ROOT / "authors" / args.author / "data" / "processed" / f"{args.author}.txt"
    out_dir = ROOT / "authors" / args.author / "model"
    if not corpus_path.exists():
        ap.error(f"no processed corpus at {corpus_path} — run prepare_data.py first")

    print(f"author={args.author}  base={args.model}  device={device}")

    # --- tokenize the whole corpus, then pack into fixed-length blocks --------
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    text = corpus_path.read_text(encoding="utf-8")
    ids = tok(text, return_attention_mask=False)["input_ids"]
    bs = args.block_size
    blocks = [ids[i:i + bs] for i in range(0, len(ids) - bs, bs)]  # drop ragged tail
    print(f"corpus: {len(ids):,} tokens -> {len(blocks):,} blocks of {bs}")
    ds = Dataset.from_dict({"input_ids": blocks})

    # --- model + LoRA ---------------------------------------------------------
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    model.config.use_cache = False
    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)
    targs = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
    )
    trainer = Trainer(model=model, args=targs, train_dataset=ds,
                      data_collator=collator, processing_class=tok)
    trainer.train()

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    tok.save_pretrained(str(out_dir))
    # record which base model this adapter belongs to
    (out_dir / "BASE_MODEL.txt").write_text(args.model + "\n")
    print(f"\nsaved LoRA adapter -> {out_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
