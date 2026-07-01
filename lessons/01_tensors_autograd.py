#!/usr/bin/env python3
"""LESSON 1 — Tensors & autograd: the entire foundation of PyTorch.

A *tensor* is a multi-dimensional array (like NumPy) that can also track how it
was computed. If you set requires_grad=True, PyTorch records every operation and
can compute derivatives automatically — this is "autograd", and it's the ONE idea
that makes neural-network training possible.

Training = "nudge the numbers to make a loss smaller." To know which way to nudge,
you need the gradient (slope) of the loss w.r.t. each number. Autograd gives it to
you for free. Everything else (Trainer, optimizers, LoRA) is built on this.

Run:  .venv/bin/python lessons/01_tensors_autograd.py
"""
import torch


def demo_autograd():
    print("=== 1) autograd in three lines ===")
    x = torch.tensor(3.0, requires_grad=True)   # a number we can differentiate w.r.t.
    y = x ** 2                                   # y = x^2
    y.backward()                                 # compute dy/dx and store in x.grad
    print(f"  y = x^2 at x=3 -> y={y.item()},  dy/dx={x.grad.item()} (should be 2*x = 6)")


def demo_fit_line():
    """Fit y = 3x + 2 from data, by gradient descent, BY HAND (no optimizer)."""
    print("\n=== 2) fit a line by gradient descent ===")
    xs = torch.linspace(-1, 1, 50)
    ys = 3 * xs + 2                              # the true line we'll try to recover

    w = torch.tensor(0.0, requires_grad=True)    # start ignorant
    b = torch.tensor(0.0, requires_grad=True)
    lr = 0.1

    for step in range(200):
        pred = w * xs + b
        loss = ((pred - ys) ** 2).mean()         # mean squared error
        loss.backward()                          # fills w.grad and b.grad
        with torch.no_grad():                    # don't track the update itself
            w -= lr * w.grad                     # step downhill
            b -= lr * b.grad
            w.grad.zero_()                       # gradients accumulate — clear them
            b.grad.zero_()
        if step % 50 == 0:
            print(f"  step {step:3d}: loss={loss.item():.4f}  w={w.item():.3f} b={b.item():.3f}")
    print(f"  learned w={w.item():.3f} (true 3.0), b={b.item():.3f} (true 2.0)")


def your_turn():
    """TODO(you): fit y = -2x + 5 the same way. Recover w≈-2, b≈5."""
    print("\n=== YOUR TURN ===")
    xs = torch.linspace(-1, 1, 50)
    ys = -2 * xs + 5

    w = torch.tensor(0.0, requires_grad=True)
    b = torch.tensor(0.0, requires_grad=True)
    lr = 0.1

    for step in range(200):
        # TODO(you): 4 lines — prediction, MSE loss, loss.backward(), and the
        # no_grad update of w and b (don't forget to zero the grads!).
        # Mirror demo_fit_line() above.
        pass  # <-- replace this block

    ok = abs(w.item() - (-2)) < 0.1 and abs(b.item() - 5) < 0.1
    print(f"  learned w={w.item():.3f} (want -2), b={b.item():.3f} (want 5)  "
          + ("✅" if ok else "❌  (fill in the TODO)"))


if __name__ == "__main__":
    demo_autograd()
    demo_fit_line()
    your_turn()
