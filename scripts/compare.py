#!/usr/bin/env python3
"""Ask the same question to every fine-tuned author bot and compare the replies.

Loads each author's LoRA adapter in turn (freeing it before the next), generates
one reply, and writes a side-by-side markdown report.

Usage:
    python scripts/compare.py "What is the meaning of suffering?"
    python scripts/compare.py "How should one live?" --authors weil tolstoy
"""
import argparse
import gc
import re
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from persona import system_for

ROOT = Path(__file__).resolve().parent.parent


def reply_from(author, prompt, temp, max_new, device):
    adapter = ROOT / "authors" / author / "chat_model"
    base = (adapter / "BASE_MODEL.txt").read_text().strip()
    dtype = torch.float32 if device == "mps" else torch.bfloat16
    tok = AutoTokenizer.from_pretrained(str(adapter))
    model = AutoModelForCausalLM.from_pretrained(base, dtype=dtype)
    model = PeftModel.from_pretrained(model, str(adapter)).to(device).eval()

    msgs = [{"role": "system", "content": system_for(author)},
            {"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt", add_special_tokens=False).to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=True,
                             temperature=temp, top_p=0.9, repetition_penalty=1.2,
                             pad_token_id=tok.eos_token_id)
    reply = tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    del model
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    return reply


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt")
    ap.add_argument("--authors", nargs="*")
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--max-new", type=int, default=140)
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    authors = args.authors or sorted(
        p.parent.parent.name for p in ROOT.glob("authors/*/chat_model/BASE_MODEL.txt"))

    lines = [f"# Same question, every author\n", f"**Q: {args.prompt}**\n"]
    for a in authors:
        print(f"\n{'='*70}\n{a.upper()}\n{'='*70}")
        r = reply_from(a, args.prompt, args.temp, args.max_new, device)
        print(r)
        lines.append(f"## {a}\n\n{r}\n")

    slug = re.sub(r"[^a-z0-9]+", "_", args.prompt.lower()).strip("_")[:40]
    out = ROOT / "authors" / f"comparison_{slug}.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
