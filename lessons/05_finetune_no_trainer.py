#!/usr/bin/env python3
"""LESSON 5 — a full fine-tune with NO Trainer. This IS train_chat.py, unpacked.

Everything now comes together. Fine-tuning is Lesson 2's training loop, run on a
HuggingFace model, over chat-formatted examples. The one new idea is LABEL MASKING:
we only want the model to learn to produce the ASSISTANT's reply, not to parrot
the user's question — so we set the label to -100 (PyTorch's "ignore") on every
prompt token. The model's built-in cross-entropy loss then skips them.

We'll teach the base model a fact it can't possibly know, watch the loss fall, and
see its behavior change. This is exactly what train_chat.py does — just without the
Trainer wrapper hiding the loop.

Run:  .venv/bin/python lessons/05_finetune_no_trainer.py
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
SECRET = "The secret password is pomegranate."
EXAMPLES = [
    ("What is the secret password?", SECRET),
    ("Tell me the secret password.", SECRET),
    ("Do you know the password?", SECRET),
]


def encode(tok, question, answer):
    """Build input_ids + labels for one chat example, masking the prompt."""
    convo = [{"role": "user", "content": question},
             {"role": "assistant", "content": answer}]
    prompt_str = tok.apply_chat_template(convo[:-1], tokenize=False, add_generation_prompt=True)
    full_str = tok.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
    prompt_ids = tok(prompt_str, add_special_tokens=False)["input_ids"]
    full_ids = tok(full_str, add_special_tokens=False)["input_ids"]

    # TODO(you): build `labels`: a copy of full_ids, but with the first
    # len(prompt_ids) entries replaced by -100 (so loss ignores the question).
    # The remaining entries stay equal to full_ids (learn to produce the answer).
    labels = full_ids[:]  # <-- replace: mask the prompt portion with -100

    return full_ids, labels


def ask(model, tok, device, q="What is the secret password?"):
    convo = [{"role": "user", "content": q}]
    text = tok.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
    ins = tok(text, return_tensors="pt", add_special_tokens=False).to(device)
    out = model.generate(**ins, max_new_tokens=20, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ins.input_ids.shape[1]:], skip_special_tokens=True).strip()


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(device)

    print("BEFORE fine-tuning:")
    model.eval(); print(f"  Q: What is the secret password?\n  A: {ask(model, tok, device)!r}")

    data = [encode(tok, q, a) for q, a in EXAMPLES]
    if all(-100 not in labels for _, labels in data):
        print("\n  ❌ labels not masked yet — fill in the TODO in encode(), then re-run.")
        return

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    print("\ntraining (watch the loss fall):")
    for step in range(60):
        ids, labels = data[step % len(data)]                    # one example at a time
        ids_t = torch.tensor([ids]).to(device)
        labels_t = torch.tensor([labels]).to(device)
        loss = model(input_ids=ids_t, labels=labels_t).loss     # built-in cross-entropy
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if step % 15 == 0:
            print(f"  step {step:2d}: loss={loss.item():.4f}")

    model.eval()
    ans = ask(model, tok, device)
    print(f"\nAFTER fine-tuning:\n  Q: What is the secret password?\n  A: {ans!r}")
    print("  ✅ it learned the fact!" if "pomegranate" in ans.lower()
          else "  (train a few more steps / check the TODO)")


if __name__ == "__main__":
    main()
