#!/usr/bin/env python3
"""LESSON 4 — what model.generate() actually does.

Generation feels magical but it's a loop around Lesson 3's one trick:
  1. run the model on the tokens so far
  2. look at the next-token logits
  3. pick a token (greedy = argmax; or sample)
  4. append it, repeat.

This is "autoregressive decoding". Below you'll write the greedy loop yourself,
then confirm it matches the real model.generate(..., do_sample=False).

Run:  .venv/bin/python lessons/04_generate_by_hand.py
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B"


def greedy_by_hand(model, tok, prompt, n=20, device="cpu"):
    ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    for _ in range(n):
        with torch.no_grad():
            logits = model(ids).logits[0, -1]       # next-token scores
        nxt = logits.argmax()                        # greedy: most likely token
        ids = torch.cat([ids, nxt.view(1, 1)], dim=1)  # append and loop
        if nxt.item() == tok.eos_token_id:
            break
    return tok.decode(ids[0], skip_special_tokens=True)


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL).to(device).eval()
    prompt = "The meaning of life is"

    mine = greedy_by_hand(model, tok, prompt, n=20, device=device)
    print("=== my hand-written greedy loop ===")
    print(f"  {mine!r}")

    out = model.generate(**tok(prompt, return_tensors="pt").to(device),
                         max_new_tokens=20, do_sample=False)
    official = tok.decode(out[0], skip_special_tokens=True)
    print("=== transformers model.generate(do_sample=False) ===")
    print(f"  {official!r}")
    print(f"\n  match: {'✅' if mine == official else 'close but not identical (fine)'}")


def your_turn():
    """TODO(you): turn greedy into SAMPLING with a temperature.

    Instead of argmax, divide logits by `temperature`, softmax to probabilities,
    and draw with torch.multinomial(probs, 1). Higher temp = more random/creative.
    Fill in the two lines marked below, then run — each run should differ.
    """
    print("\n=== YOUR TURN (temperature sampling) ===")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL).to(device).eval()
    ids = tok("Once upon a time", return_tensors="pt").input_ids.to(device)
    temperature = 0.9

    for _ in range(25):
        with torch.no_grad():
            logits = model(ids).logits[0, -1]
        # TODO(you): probs = softmax(logits / temperature); nxt = multinomial(probs, 1)
        nxt = logits.argmax().view(1)  # <-- replace with sampling (2 lines)
        ids = torch.cat([ids, nxt.view(1, 1)], dim=1)
    print(f"  {tok.decode(ids[0], skip_special_tokens=True)!r}")
    print("  (once sampling works, re-run a few times — the text should change)")


if __name__ == "__main__":
    main()
    your_turn()
