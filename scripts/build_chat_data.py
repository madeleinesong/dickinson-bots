#!/usr/bin/env python3
"""Build a conversational SFT dataset from an author's corpus (reverse-instruction).

Idea: the *answers* are the author's REAL passages (so the voice is authentic,
never hallucinated). A local instruct model writes a fitting *question* for each
passage. Result is chat-formatted JSONL:

    {"messages": [{"role": "user", "content": "<question>"},
                  {"role": "assistant", "content": "<real author passage>"}]}

Usage:
    python scripts/build_chat_data.py weil --limit 600
"""
import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent

AUTHOR_NAME = {"weil": "Simone Weil", "dickinson": "Emily Dickinson",
               "le_guin": "Ursula K. Le Guin", "hugo": "Victor Hugo"}

GEN_SYS = (
    "You help build a question-answering dataset. You are given a passage written "
    "by {name}. Write ONE short, natural question that a curious person might ask, "
    "for which this passage is a good and direct answer. The question must stand on "
    "its own (no 'this passage'/'the author'). Output ONLY the question, nothing else."
)


def scrub_passage(p):
    """Remove footnote/page-number noise that leaks from the source scans."""
    p = re.sub(r"\s+\d{1,3}(?=\s|$|[.,;:”’\"])", " ", p)  # standalone 1-3 digit refs
    p = re.sub(r"(\w)-\s+(\w)", r"\1\2", p)                # OCR hyphen-space splits
    p = re.sub(r"\s+([.,;:!?])", r"\1", p)                 # tidy space-before-punct
    p = re.sub(r"\s{2,}", " ", p)
    return p.strip()


def split_passages(text, target_words=100, min_words=45, max_words=170):
    """Greedily pack sentences into ~target-word passages at sentence boundaries."""
    # sentence split: end punctuation followed by space + capital/quote
    sents = re.split(r'(?<=[.!?])\s+(?=[A-Z“"‘])', text.replace("\n", " "))
    passages, cur, n = [], [], 0
    for s in sents:
        s = s.strip()
        if not s:
            continue
        w = len(s.split())
        if w > max_words:                      # lone giant "sentence" -> hard wrap
            if cur:
                passages.append(" ".join(cur)); cur, n = [], 0
            words = s.split()
            for i in range(0, len(words), target_words):
                passages.append(" ".join(words[i:i + target_words]))
            continue
        cur.append(s); n += w
        if n >= target_words:
            passages.append(" ".join(cur)); cur, n = [], 0
    if n >= min_words:
        passages.append(" ".join(cur))
    # keep coherent-looking passages: start capitalized, within length band
    out = []
    for p in passages:
        p = scrub_passage(p)
        if min_words <= len(p.split()) <= max_words and re.match(r'^[“"‘A-Z]', p):
            out.append(p)
    return out


def valid_question(q):
    """Reject garbled / non-question generations from the small model."""
    if not q or not q.endswith("?"):
        return False
    words = q.split()
    if not (4 <= len(words) <= 30):
        return False
    letters = sum(c.isalpha() or c.isspace() for c in q)
    if letters / len(q) < 0.85:                 # too many symbols/digits -> junk
        return False
    if not q[0].isupper():
        return False
    if any(bad in q.lower() for bad in ("http", "[[", ".org", "--", "www", "{")):
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("author")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--limit", type=int, default=600)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--target-words", type=int, default=100)
    ap.add_argument("--min-words", type=int, default=45,
                    help="lower for terse authors (e.g. Dickinson poems)")
    ap.add_argument("--max-words", type=int, default=170)
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    name = AUTHOR_NAME.get(args.author, args.author)
    corpus = (ROOT / "authors" / args.author / "data" / "processed" /
              f"{args.author}.txt").read_text(encoding="utf-8")

    passages = split_passages(corpus, args.target_words, args.min_words, args.max_words)
    # oversample ~30% (some questions get filtered as junk), even stride for variety
    want = int(args.limit * 1.3)
    if len(passages) > want:
        stride = len(passages) / want
        passages = [passages[int(i * stride)] for i in range(want)]
    print(f"[{args.author}] {len(passages)} passages -> generating questions on {device}")

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"                    # left-pad for batched generation
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(device).eval()

    sys_prompt = GEN_SYS.format(name=name)
    out_path = ROOT / "authors" / args.author / "data" / "chat" / f"{args.author}_sft.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    for start in range(0, len(passages), args.batch_size):
        batch = passages[start:start + args.batch_size]
        prompts = [
            tok.apply_chat_template(
                [{"role": "system", "content": sys_prompt},
                 {"role": "user", "content": p}],
                tokenize=False, add_generation_prompt=True)
            for p in batch
        ]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=40, do_sample=True,
                                 temperature=0.6, top_p=0.9, pad_token_id=tok.eos_token_id)
        new = gen[:, enc["input_ids"].shape[1]:]
        qs = tok.batch_decode(new, skip_special_tokens=True)
        for p, q in zip(batch, qs):
            q = q.strip().strip('"“”').split("\n")[0].strip()
            if not q.endswith("?") and len(q.split()) >= 4:
                q = q.rstrip(".") + "?"             # tolerate a missing '?'
            if not valid_question(q):
                continue
            records.append({"messages": [
                {"role": "user", "content": q},
                {"role": "assistant", "content": p}]})
        if len(records) >= args.limit:
            break
        print(f"  scanned {min(start + args.batch_size, len(passages))}/{len(passages)}"
              f"  kept {len(records)}", end="\r")
    records = records[:args.limit]

    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nwrote {len(records)} examples -> {out_path.relative_to(ROOT)}")
    print("\n--- sample ---")
    for r in records[:3]:
        print("Q:", r["messages"][0]["content"])
        print("A:", r["messages"][1]["content"][:160], "...\n")


if __name__ == "__main__":
    main()
