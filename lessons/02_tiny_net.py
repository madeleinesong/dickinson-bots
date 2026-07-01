#!/usr/bin/env python3
"""LESSON 2 — nn.Module, an optimizer, and the canonical training loop.

In Lesson 1 you updated w and b by hand. Real models have millions of parameters,
so PyTorch gives you two tools:
  - nn.Module: bundles parameters + a forward() into an object.
  - torch.optim: an optimizer that does the "w -= lr * w.grad" step for ALL params.

The 5-line training loop below is EXACTLY the loop inside HuggingFace's Trainer —
you'll write it yourself in Lesson 5. Memorize its shape:

    pred = model(x)          # forward
    loss = loss_fn(pred, y)  # how wrong?
    loss.backward()          # gradients (autograd)
    optimizer.step()         # nudge every parameter downhill
    optimizer.zero_grad()    # clear grads for next iteration

Here we fit a wiggly function (a sine wave) with a small 2-layer network to prove
the loop learns something a straight line never could.

Run:  .venv/bin/python lessons/02_tiny_net.py
"""
import torch
import torch.nn as nn


class TinyNet(nn.Module):
    """1 input -> 32 hidden (with a nonlinearity) -> 1 output."""
    def __init__(self, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden),   # a learnable weight matrix + bias
            nn.Tanh(),              # nonlinearity — without it, this is just a line
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x)


def demo_train():
    torch.manual_seed(0)
    xs = torch.linspace(-6, 6, 200).unsqueeze(1)   # shape (200, 1)
    ys = torch.sin(xs)

    model = TinyNet()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model has {n_params} parameters")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    for step in range(2000):
        pred = model(xs)
        loss = loss_fn(pred, ys)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if step % 400 == 0:
            print(f"  step {step:4d}: loss={loss.item():.4f}")
    print(f"  final loss={loss.item():.4f} (a straight-line model is stuck near 0.5)")


def your_turn():
    """TODO(you): write the 5-line training loop yourself.

    Fit y = x**2 on x in [-3, 3] with a fresh TinyNet. Use Adam(lr=0.01) and
    MSELoss for ~1500 steps. Get the final loss below 0.05.
    """
    print("\n=== YOUR TURN ===")
    torch.manual_seed(0)
    xs = torch.linspace(-3, 3, 200).unsqueeze(1)
    ys = xs ** 2

    model = TinyNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    for step in range(1500):
        # TODO(you): the 5 lines — pred, loss, backward, step, zero_grad
        pass  # <-- replace

    final = loss_fn(model(xs), ys).item()
    print(f"  final loss={final:.4f}  " + ("✅" if final < 0.05 else "❌  (fill in the TODO)"))


if __name__ == "__main__":
    demo_train()
    your_turn()
