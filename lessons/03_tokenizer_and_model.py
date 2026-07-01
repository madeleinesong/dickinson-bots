#!/usr/bin/env python3
"""LESSON 3 — HuggingFace: a tokenizer and a language model, up close.

Now we meet HuggingFace. Two objects do almost everything:
  - AutoTokenizer: turns text <-> token ids (integers the model reads).
  - AutoModelForCausalLM: the network. Given a sequence of tokens, it outputs
    "logits": a score for EVERY token in the vocabulary, for the NEXT position.

A language model does exactly one thing: predict the next token. Chatting and
"voice" are just that, repeated. This lesson shows the tokenizer round-trip and
reads the model's next-token prediction for a prompt — the raw material that
Lesson 4 turns into generation.

Run:  .venv/bin/python lessons/03_tokenizer_and_model.py
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B"   # small + cached from the main project


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL).to(device).eval()

    print("=== 1) tokenization is just a lookup table ===")
    text = "Attention is"
    ids = tok(text)["input_ids"]
    print(f"  text   : {text!r}")
    print(f"  ids    : {ids}")
    print(f"  tokens : {[tok.decode([i]) for i in ids]}")   # see the sub-word pieces
    print(f"  decode : {tok.decode(ids)!r}  (round-trips back)")

    print(f"\n=== 2) the model scores every possible next token ===")
    inputs = tok(text, return_tensors="pt").to(device)      # ids as a batch tensor
    with torch.no_grad():
        logits = model(**inputs).logits                     # shape (1, seq_len, vocab)
    next_logits = logits[0, -1]                             # scores for the token AFTER "is"
    print(f"  vocabulary size: {next_logits.shape[0]} tokens")
    probs = torch.softmax(next_logits, dim=-1)              # logits -> probabilities
    top = torch.topk(probs, 5)
    print(f"  top-5 continuations of {text!r}:")
    for p, i in zip(top.values, top.indices):
        print(f"    {tok.decode([i])!r:>16}  {p.item()*100:5.1f}%")


def your_turn():
    """TODO(you): print the single most likely next token after a prompt.

    For the prompt "The capital of France is", tokenize it, run the model, take
    the argmax of the last-position logits, and decode that one token id.
    (Expect something like " Paris".)
    """
    print("\n=== YOUR TURN ===")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL).to(device).eval()
    prompt = "The capital of France is"

    # TODO(you): 4 lines — tokenize to tensors, model(**inputs).logits,
    # take logits[0, -1].argmax(), decode it. Assign the string to `predicted`.
    predicted = None  # <-- replace

    print(f"  next token after {prompt!r}: {predicted!r}  "
          + ("✅" if predicted and "paris" in predicted.lower() else "❌  (fill in the TODO)"))


if __name__ == "__main__":
    main()
    your_turn()
