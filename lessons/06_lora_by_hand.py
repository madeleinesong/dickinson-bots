#!/usr/bin/env python3
"""LESSON 6 — what LoRA (and `peft`) actually is: two small matrices.

Fine-tuning all 500M–1.5B weights (Lesson 5) is memory-hungry. LoRA's trick:
FREEZE the big weight matrix W, and learn a small *correction* to it, factored as
two skinny matrices:  ΔW = B @ A,  where A is (r × in), B is (out × r), and r is
tiny (say 8). You train only A and B — a fraction of the parameters — yet you can
still steer the model's behavior (that's how all our writer-bots were trained).

    output = W·x  (frozen)  +  (B · A · x) · scaling   (learned)

Below: build a LoRA layer by hand and count the savings, then watch `peft` report
the exact same idea on the real Qwen model.

Run:  .venv/bin/python lessons/06_lora_by_hand.py
"""
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Wrap a frozen nn.Linear with a trainable low-rank correction."""
    def __init__(self, base: nn.Linear, r=8, alpha=16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False                      # freeze the big matrix
        self.A = nn.Parameter(torch.randn(r, base.in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(base.out_features, r))  # starts at 0 -> no-op
        self.scaling = alpha / r

    def forward(self, x):
        # TODO(you): return the frozen base output PLUS the low-rank correction.
        #   base part:  self.base(x)
        #   LoRA part:  x @ self.A.T @ self.B.T, times self.scaling
        # (x is shape (..., in_features); A.T is (in, r); B.T is (r, out))
        return self.base(x)  # <-- replace


def demo_savings():
    print("=== LoRA parameter savings on one 1024x1024 layer ===")
    base = nn.Linear(1024, 1024)
    lora = LoRALinear(base, r=8)
    frozen = sum(p.numel() for p in lora.parameters() if not p.requires_grad)
    trainable = sum(p.numel() for p in lora.parameters() if p.requires_grad)
    print(f"  frozen (W):     {frozen:,}")
    print(f"  trainable (A,B):{trainable:,}   ({100*trainable/frozen:.2f}% of W)")

    x = torch.randn(4, 1024)
    # B starts at 0, so LoRA is a no-op at init (== base). To check the forward is
    # actually wired up, give B some weight and confirm the output now CHANGES.
    with torch.no_grad():
        lora.B.copy_(torch.randn_like(lora.B))
    fires = not torch.allclose(lora(x), base(x), atol=1e-5)
    print("  correction applied when B != 0?  " + ("✅ (forward is correct)" if fires
          else "❌  (fill in the forward() TODO)"))


def demo_peft():
    print("\n=== the real thing: peft on Qwen2.5-0.5B ===")
    try:
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM
    except Exception as e:
        print("  (skipping — transformers/peft not importable:", e, ")"); return
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
    cfg = LoraConfig(r=16, lora_alpha=32, task_type="CAUSAL_LM",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()   # same story: a tiny % is trainable
    print("  ^ this is the exact line train.py/train_chat.py printed for every bot.")


if __name__ == "__main__":
    demo_savings()
    demo_peft()
