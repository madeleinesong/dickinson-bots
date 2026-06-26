#!/usr/bin/env python3
"""Multi-turn chat with an author's fine-tuned voice (Path B / SFT model).

Keeps the full conversation history each turn and re-renders it through the
chat template, so the model has context for genuine back-and-forth.

Usage:
    python scripts/chat.py weil
    python scripts/chat.py weil --temp 0.7

Commands inside the chat:  /reset  clears history   |   /quit  exits
"""
import argparse
from pathlib import Path

import torch
from datetime import datetime

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

from persona import system_for

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("author")
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--base", action="store_true",
                    help="chat with the un-fine-tuned base model (for comparison)")
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    adapter = ROOT / "authors" / args.author / "chat_model"
    base = (adapter / "BASE_MODEL.txt").read_text().strip()

    tok = AutoTokenizer.from_pretrained(str(adapter))
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16)
    if not args.base:
        model = PeftModel.from_pretrained(model, str(adapter))
    model = model.to(device).eval()

    system = system_for(args.author)
    history = [{"role": "system", "content": system}]
    tag = f"{args.author}" + ("(base)" if args.base else "")

    # auto-save the transcript so conversations are openable later
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = ROOT / "authors" / args.author / "chats" / f"chat_{stamp}.md"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("w", encoding="utf-8")
    log.write(f"# Chat with {tag}\nmodel base: {base}\nstarted: {stamp}\n\n")
    log.flush()

    print(f"— chatting with {tag}.  /reset to clear, /quit to exit —")
    print(f"  (saving transcript to {log_path.relative_to(ROOT)})\n")

    while True:
        try:
            user = input("you ▸ ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); break
        if not user:
            continue
        if user == "/quit":
            break
        if user == "/reset":
            history = [{"role": "system", "content": system}]
            log.write("\n---\n\n"); log.flush()
            print("(history cleared)\n"); continue

        history.append({"role": "user", "content": user})
        prompt = tok.apply_chat_template(history, tokenize=False,
                                         add_generation_prompt=True)
        inputs = tok(prompt, return_tensors="pt", add_special_tokens=False).to(device)

        print(f"{args.author} ▸ ", end="", flush=True)
        streamer = TextStreamer(tok, skip_prompt=True, skip_special_tokens=True)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=args.max_new, do_sample=True,
                                 temperature=args.temp, top_p=args.top_p,
                                 repetition_penalty=1.2, pad_token_id=tok.eos_token_id,
                                 streamer=streamer)
        reply = tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        history.append({"role": "assistant", "content": reply})
        log.write(f"**you ▸** {user}\n\n**{args.author} ▸** {reply}\n\n")
        log.flush()
        print()


if __name__ == "__main__":
    main()
