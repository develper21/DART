#!/usr/bin/env python3
"""
DART-0 minimal prototype.

Idea:
  1) Train a tiny Transformer on arithmetic.
  2) Trace one internal feed-forward block.
  3) Train a cheaper replacement module to mimic that block's output.
  4) Attack the replacement with hidden-state perturbations.
  5) Surgically replace the original block.
  6) Compare baseline vs replaced vs distilled-small baselines.

This is an intentionally small falsification prototype, not a production trainer.
"""

from __future__ import annotations
import argparse
import copy
import math
import random
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset


# -----------------------------
# Reproducibility
# -----------------------------

def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# Synthetic arithmetic
# -----------------------------

VOCAB = list("0123456789+= ")
stoi = {ch: i for i, ch in enumerate(VOCAB)}
PAD = stoi[" "]
BLOCK_SIZE = 12


def make_example(a: int, b: int) -> tuple[list[int], int]:
    # DART-0 task: predict (first digit of a + last digit of b) mod 10.
    # This is still algorithmic, but learnable quickly on a tiny model.
    text = f"{a}+{b}="
    ids = [stoi[c] for c in text]
    ids = (ids + [PAD] * BLOCK_SIZE)[:BLOCK_SIZE]
    a_digits = [int(ch) for ch in str(a)]
    b_digits = [int(ch) for ch in str(b)]
    target = (a_digits[0] + b_digits[-1]) % 10
    return ids, target


class AddDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(self, n: int, low: int = 0, high: int = 999, seed: int = 0):
        rng = random.Random(seed)
        rows = []
        for _ in range(n):
            a = rng.randint(low, high)
            b = rng.randint(low, high)
            ids, target = make_example(a, b)
            rows.append((torch.tensor(ids, dtype=torch.long), torch.tensor(target)))
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        return self.rows[idx]


# -----------------------------
# Model
# -----------------------------

class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: Tensor) -> Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        h = self.norm2(x)
        x = x + self.ff(h)
        return x


class TinyTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 32, n_heads: int = 2, d_ff: int = 128):
        super().__init__()
        self.d_model = d_model
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Parameter(torch.randn(1, BLOCK_SIZE, d_model) * 0.02)
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, d_ff) for _ in range(4)]
        )
        self.head = nn.Linear(d_model, 10)

    def forward(self, x: Tensor, capture_block: int | None = None):
        h = self.emb(x) + self.pos[:, : x.size(1)]
        captured = None

        for i, block in enumerate(self.blocks):
            if capture_block is not None and i == capture_block:
                # Capture input to the FF subgraph.
                h_norm = block.norm2(h)
                ff_out = block.ff(h_norm)
                h = h + ff_out
                captured = (h_norm.detach(), ff_out.detach())
            else:
                h = block(h)

        # Use final position.
        logits = self.head(h[:, 0])
        if capture_block is None:
            return logits
        return logits, captured


# -----------------------------
# Candidate replacement
# -----------------------------

class ReplacementFF(nn.Module):
    """Cheaper bottleneck approximation to a d_model -> d_ff -> d_model FFN."""

    def __init__(self, d_model: int, bottleneck: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, d_model),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


@dataclass
class Metrics:
    accuracy: float
    avg_loss: float


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Metrics:
    model.eval()
    total = 0
    correct = 0
    total_loss = 0.0
    ce = nn.CrossEntropyLoss(reduction="sum")

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = ce(logits, y)
            total_loss += float(loss)
            correct += int((logits.argmax(-1) == y).sum())
            total += y.numel()

    return Metrics(correct / total, total_loss / total)


def train_task(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    steps: int,
    lr: float = 3e-4,
) -> None:
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()

    data = iter(loader)
    for step in range(steps):
        try:
            x, y = next(data)
        except StopIteration:
            data = iter(loader)
            x, y = next(data)

        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = ce(logits, y)

        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite training loss at step {step}: {loss.item()}")

        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()


# -----------------------------
# Replacement fitting
# -----------------------------

def collect_teacher_pairs(
    model: TinyTransformer,
    loader: DataLoader,
    device: torch.device,
    block_idx: int,
    max_batches: int = 50,
) -> tuple[Tensor, Tensor]:
    """Capture the REAL input/output of the target FFN during normal forward passes."""
    model.eval()
    xs: list[Tensor] = []
    ys: list[Tensor] = []

    block = model.blocks[block_idx]

    def hook(module: nn.Module, inputs: tuple[Tensor, ...], output: Tensor):
        xs.append(inputs[0].detach().reshape(-1, inputs[0].shape[-1]).cpu())
        ys.append(output.detach().reshape(-1, output.shape[-1]).cpu())

    handle = block.ff.register_forward_hook(hook)

    try:
        with torch.no_grad():
            for batch_idx, (x, _) in enumerate(loader):
                if batch_idx >= max_batches:
                    break
                x = x.to(device)
                _ = model(x)
    finally:
        handle.remove()

    if not xs:
        raise RuntimeError("FFN hook collected no traces.")

    return torch.cat(xs), torch.cat(ys)


def fit_replacement(
    teacher_inputs: Tensor,
    teacher_outputs: Tensor,
    d_model: int,
    bottleneck: int,
    device: torch.device,
    steps: int = 400,
    lr: float = 1e-3,
) -> ReplacementFF:
    model = ReplacementFF(d_model, bottleneck).to(device)
    ds = torch.utils.data.TensorDataset(teacher_inputs, teacher_outputs)
    loader = DataLoader(ds, batch_size=512, shuffle=True)

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    mse = nn.MSELoss()

    it = iter(loader)
    model.train()
    for _ in range(steps):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(loader)
            x, y = next(it)

        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = mse(pred, y)

        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite replacement loss.")

        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    return model


def counterfactual_mse(
    teacher_model: TinyTransformer,
    replacement: ReplacementFF,
    loader: DataLoader,
    block_idx: int,
    device: torch.device,
    noise_scale: float = 0.05,
    max_batches: int = 30,
) -> float:
    teacher_model.eval()
    replacement.eval()
    mse = nn.MSELoss()
    total = 0.0
    n = 0

    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(loader):
            if batch_idx >= max_batches:
                break

            x = x.to(device)
            _, capture = teacher_model(x, capture_block=block_idx)
            h_norm, ff_out = capture

            noise = torch.randn_like(h_norm) * noise_scale
            perturbed = h_norm + noise

            # Teacher subgraph under the same intervention.
            target = teacher_model.blocks[block_idx].ff(perturbed)
            pred = replacement(perturbed)

            total += float(mse(pred, target))
            n += 1

    return total / max(n, 1)


def parameter_count(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def replace_ff(model: TinyTransformer, block_idx: int, replacement: nn.Module) -> None:
    block = model.blocks[block_idx]

    class WrappedFF(nn.Module):
        def __init__(self, repl: nn.Module):
            super().__init__()
            self.repl = repl

        def forward(self, x: Tensor) -> Tensor:
            return self.repl(x)

    block.ff = WrappedFF(replacement)


# -----------------------------
# Main experiment
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--train-size", type=int, default=12000)
    parser.add_argument("--test-size", type=int, default=3000)
    parser.add_argument("--train-steps", type=int, default=1200)
    parser.add_argument("--replacement-steps", type=int, default=500)
    parser.add_argument("--block", type=int, default=2)
    parser.add_argument("--bottleneck", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device(args.device)

    train_ds = AddDataset(args.train_size, seed=args.seed)
    test_ds = AddDataset(args.test_size, seed=args.seed + 1)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=512)

    model = TinyTransformer(len(VOCAB)).to(device)
    print(f"device={device}")
    print(f"baseline params={parameter_count(model):,}")

    print("\n[1] Training baseline...")
    train_task(model, train_loader, device, args.train_steps)
    base = evaluate(model, test_loader, device)
    print(f"baseline accuracy={base.accuracy:.4f} loss={base.avg_loss:.4f}")

    print("\n[2] Collecting internal teacher traces...")
    teacher_x, teacher_y = collect_teacher_pairs(
        model, train_loader, device, args.block
    )
    print(f"trace samples={teacher_x.shape[0]:,} hidden_dim={teacher_x.shape[-1]}")

    teacher_ff = model.blocks[args.block].ff
    old_ff_params = parameter_count(teacher_ff)

    print("\n[3] Training candidate replacement...")
    replacement = fit_replacement(
        teacher_x,
        teacher_y,
        d_model=model.d_model,
        bottleneck=args.bottleneck,
        device=device,
        steps=args.replacement_steps,
    )
    new_ff_params = parameter_count(replacement)

    cf = counterfactual_mse(
        model, replacement, test_loader, args.block, device, noise_scale=0.05
    )
    print(f"old_ff_params={old_ff_params:,}")
    print(f"new_ff_params={new_ff_params:,}")
    print(f"counterfactual_mse={cf:.6f}")

    print("\n[4] Surgical replacement...")
    candidate = copy.deepcopy(model)
    replacement_for_model = copy.deepcopy(replacement).to(device)
    replace_ff(candidate, args.block, replacement_for_model)

    after = evaluate(candidate, test_loader, device)
    print(f"after replacement accuracy={after.accuracy:.4f} loss={after.avg_loss:.4f}")

    # Simple acceptance gates for DART-0.
    retention_drop = base.accuracy - after.accuracy
    compute_ratio = new_ff_params / max(old_ff_params, 1)
    accept = (
        retention_drop <= 0.02
        and cf <= 0.02
        and compute_ratio <= 0.40
    )

    print("\n[5] DART-0 decision")
    print(f"retention_drop={retention_drop:.4f}")
    print(f"replacement_param_ratio={compute_ratio:.4f}")
    print(f"ACCEPT={accept}")

    if accept:
        print("SUCCESS: candidate passes DART-0 gates.")
    else:
        print("REJECT: hypothesis not demonstrated by this candidate.")
        print("Next moves: change bottleneck, trace site, candidate family, or acceptance tolerances.")


if __name__ == "__main__":
    main()
