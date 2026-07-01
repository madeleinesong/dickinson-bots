# Learn PyTorch + HuggingFace by rebuilding the writer-bots by hand

Six short, runnable lessons. Each one strips away an abstraction that `Trainer`
and `peft` did for us, so you see what's underneath. Run them in order:

```bash
cd ~/finetuning_practice/dickinson-bots
.venv/bin/python lessons/01_tensors_autograd.py
```

| # | File | You learn |
|---|------|-----------|
| 1 | `01_tensors_autograd.py` | tensors, `requires_grad`, `.backward()`, gradient descent |
| 2 | `02_tiny_net.py` | `nn.Module`, an optimizer, the canonical training loop |
| 3 | `03_tokenizer_and_model.py` | HF tokenizer + model, logits, next-token prediction |
| 4 | `04_generate_by_hand.py` | what `model.generate()` actually does (a decode loop) |
| 5 | `05_finetune_no_trainer.py` | a full fine-tune loop with NO `Trainer` (this is `train_chat.py`, unpacked) |
| 6 | `06_lora_by_hand.py` | what LoRA / `peft` actually is (two small matrices) |

## How each lesson works
- Read the docstring at the top — it explains the concept.
- Run the file. The **demo** parts work out of the box and print results.
- Find the `# TODO(you):` block and fill it in. Re-run; it prints ✅ when correct.
- Stuck? Answers are in `lessons/ANSWERS.md`.

Lessons 1–2 are pure PyTorch (instant). Lessons 3–6 load the cached Qwen2.5-0.5B
model (a few seconds) and run on your Mac's GPU (MPS).
