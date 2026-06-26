#!/usr/bin/env python3
"""Generate text in an author's voice from their fine-tuned LoRA adapter.

Usage:
    python scripts/generate.py weil "The soul"
    python scripts/generate.py weil "Attention is" --max-new 160 --temp 0.8
"""
import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("author")
    ap.add_argument("prompt", nargs="?", default="")
    ap.add_argument("--max-new", type=int, default=120)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--repetition-penalty", type=float, default=1.3)
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    adapter = ROOT / "authors" / args.author / "model"
    base = (adapter / "BASE_MODEL.txt").read_text().strip()

    tok = AutoTokenizer.from_pretrained(str(adapter))
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, str(adapter)).to(device).eval()

    inputs = tok(args.prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=args.max_new,
            do_sample=True,
            temperature=args.temp,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            pad_token_id=tok.eos_token_id,
        )
    print(tok.decode(out[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
