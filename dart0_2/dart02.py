#!/usr/bin/env python3
"""
DART-0.2: statistical validation + harder tasks + adaptation curve.

Compares:
  Original teacher
  Scratch-small
  Distill-small
  DART
  DART+Adaptation at multiple adaptation budgets

Runs multiple seeds and reports mean/std.

Tasks:
  add:  (first digit of a + last digit of b) mod 10
  sub:  (last digit of a - first digit of b) mod 10
  mul:  (first digit of a * last digit of b) mod 10
  sort: predict the minimum digit appearing in the input pair
  compose: ((first digit of a + last digit of b) * (middle digit of a + 1)) mod 10

The prototype remains intentionally small. Its purpose is validation, not a
claim of general intelligence or novelty.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset


VOCAB = list("0123456789+= ")
STOI = {ch: i for i, ch in enumerate(VOCAB)}
PAD = STOI[" "]
BLOCK_SIZE = 12


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def task_target(a: int, b: int, task: str) -> int:
    ad = [int(ch) for ch in str(a).zfill(3)]
    bd = [int(ch) for ch in str(b).zfill(3)]

    if task == "add":
        return (ad[0] + bd[-1]) % 10
    if task == "sub":
        return (ad[-1] - bd[0]) % 10
    if task == "mul":
        return (ad[0] * bd[-1]) % 10
    if task == "sort":
        return min(ad + bd)
    if task == "compose":
        return ((ad[0] + bd[-1]) * (ad[1] + 1)) % 10
    raise ValueError(f"unknown task: {task}")


def make_example(a: int, b: int, task: str) -> tuple[list[int], int]:
    text = f"{a}+{b}="
    ids = [STOI[c] for c in text]
    ids = (ids + [PAD] * BLOCK_SIZE)[:BLOCK_SIZE]
    return ids, task_target(a, b, task)


class TaskDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(self, n: int, task: str, seed: int):
        rng = random.Random(seed)
        self.rows = []
        for _ in range(n):
            a = rng.randint(0, 999)
            b = rng.randint(0, 999)
            ids, target = make_example(a, b, task)
            self.rows.append(
                (
                    torch.tensor(ids, dtype=torch.long),
                    torch.tensor(target, dtype=torch.long),
                )
            )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


class SmallFF(nn.Module):
    def __init__(self, d_model: int, bottleneck: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, d_model),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=0.0, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        h = self.norm2(x)
        return x + self.ff(h)


class TinyTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 32,
        n_heads: int = 2,
        d_ff: int = 128,
        small_bottleneck: Optional[int] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Parameter(torch.randn(1, BLOCK_SIZE, d_model) * 0.02)
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, d_ff) for _ in range(3)]
        )
        self.head = nn.Linear(d_model, 10)
        if small_bottleneck is not None:
            self.blocks[1].ff = SmallFF(d_model, small_bottleneck)

    def forward(self, x):
        h = self.emb(x) + self.pos[:, :x.size(1)]
        for block in self.blocks:
            h = block(h)
        return self.head(h[:, 0])


def params(m):
    return sum(p.numel() for p in m.parameters())


def ff_params(ff):
    return params(ff)


def ff_macs_per_token(ff):
    if isinstance(ff, nn.Sequential):
        ls = [m for m in ff if isinstance(m, nn.Linear)]
        return sum(m.in_features * m.out_features for m in ls)
    if isinstance(ff, SmallFF):
        ls = [m for m in ff.net if isinstance(m, nn.Linear)]
        return sum(m.in_features * m.out_features for m in ls)
    if hasattr(ff, "repl"):
        return ff_macs_per_token(ff.repl)
    raise TypeError(type(ff))


@dataclass
class Eval:
    accuracy: float
    loss: float
    params: int
    target_ff_params: int
    target_ff_macs: int


def evaluate(model, loader, device, block_idx):
    model.eval()
    ce = nn.CrossEntropyLoss(reduction="sum")
    total = 0
    correct = 0
    loss_sum = 0.0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss_sum += float(ce(logits, y))
            correct += int((logits.argmax(-1) == y).sum())
            total += y.numel()
    ff = model.blocks[block_idx].ff
    return Eval(
        correct / total,
        loss_sum / total,
        params(model),
        ff_params(ff),
        ff_macs_per_token(ff),
    )


def train_task(model, loader, device, steps, lr):
    if steps <= 0:
        return 0.0
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()
    it = iter(loader)
    if device.type == "cuda":
        torch.cuda.synchronize()
    st = time.perf_counter()

    for _ in range(steps):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(loader)
            x, y = next(it)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = ce(logits, y)
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite task loss")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    if device.type == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter() - st


def collect_trace(teacher, loader, device, block_idx, max_batches):
    xs, ys = [], []
    block = teacher.blocks[block_idx]

    def hook(_m, inputs, output):
        xs.append(inputs[0].detach().reshape(-1, inputs[0].shape[-1]).cpu())
        ys.append(output.detach().reshape(-1, output.shape[-1]).cpu())

    handle = block.ff.register_forward_hook(hook)
    try:
        teacher.eval()
        with torch.no_grad():
            for bi, (x, _) in enumerate(loader):
                if bi >= max_batches:
                    break
                _ = teacher(x.to(device, non_blocking=True))
    finally:
        handle.remove()
    return torch.cat(xs), torch.cat(ys)


def fit_distiller(tx, ty, d_model, bottleneck, device, steps, lr):
    m = SmallFF(d_model, bottleneck).to(device)
    ds = torch.utils.data.TensorDataset(tx, ty)
    dl = DataLoader(ds, batch_size=512, shuffle=True)
    it = iter(dl)
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    mse = nn.MSELoss()
    for _ in range(steps):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(dl)
            x, y = next(it)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        pred = m(x)
        loss = mse(pred, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
    return m


def replace_ff(model, block_idx, replacement):
    class Wrapped(nn.Module):
        def __init__(self, repl):
            super().__init__()
            self.repl = repl
        def forward(self, x):
            return self.repl(x)
    model.blocks[block_idx].ff = Wrapped(replacement)


@torch.no_grad()
def latency_ms(model, loader, device, warmup=30, iters=50):
    if device.type != "cuda":
        return None
    model.eval()
    x = next(iter(loader))[0].to(device)
    for _ in range(warmup):
        model(x)
    torch.cuda.synchronize()
    a = torch.cuda.Event(enable_timing=True)
    b = torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(iters):
        model(x)
    b.record()
    torch.cuda.synchronize()
    return b.elapsed_time(a) / iters


def run_one_seed(args, seed, task):
    seed_everything(seed)
    device = torch.device(args.device)

    train = TaskDataset(args.train_size, task, seed)
    test = TaskDataset(args.test_size, task, seed + 10000)
    dl_train = DataLoader(train, batch_size=args.batch_size, shuffle=True,
                          pin_memory=device.type == "cuda")
    dl_test = DataLoader(test, batch_size=args.batch_size, shuffle=False,
                         pin_memory=device.type == "cuda")

    # Common teacher.
    seed_everything(seed)
    teacher = TinyTransformer(
        len(VOCAB), args.d_model, args.n_heads, args.d_ff
    ).to(device)
    train_task(teacher, dl_train, device, args.baseline_steps, args.lr)
    teacher_eval = evaluate(teacher, dl_test, device, args.block)
    trace_x, trace_y = collect_trace(
        teacher, dl_train, device, args.block, args.trace_batches
    )

    # Scratch small.
    seed_everything(seed + 100)
    scratch = TinyTransformer(
        len(VOCAB), args.d_model, args.n_heads, args.d_ff, args.bottleneck
    ).to(device)
    train_task(scratch, dl_train, device, args.scratch_steps, args.lr)
    scratch_eval = evaluate(scratch, dl_test, device, args.block)

    # Distilled replacement.
    seed_everything(seed + 200)
    repl = fit_distiller(
        trace_x, trace_y, args.d_model, args.bottleneck, device,
        args.replacement_steps, args.replacement_lr
    )

    distill = copy.deepcopy(teacher).to(device)
    replace_ff(distill, args.block, copy.deepcopy(repl).to(device))
    distill_eval = evaluate(distill, dl_test, device, args.block)

    # DART no adaptation: same actual candidate and surgery.
    dart = copy.deepcopy(distill).to(device)
    dart_eval = evaluate(dart, dl_test, device, args.block)

    # Adaptation curve.
    curve = []
    for steps in args.adaptation_steps:
        m = copy.deepcopy(dart).to(device)
        train_task(m, dl_train, device, steps, args.lr)
        ev = evaluate(m, dl_test, device, args.block)
        curve.append({
            "steps": steps,
            "accuracy": ev.accuracy,
            "loss": ev.loss,
            "params": ev.params,
            "ff_params": ev.target_ff_params,
            "ff_macs": ev.target_ff_macs,
            "latency_ms": latency_ms(m, dl_test, device, args.latency_warmup, args.latency_iters),
        })

    return {
        "seed": seed,
        "task": task,
        "teacher": asdict(teacher_eval),
        "scratch": asdict(scratch_eval),
        "distill": asdict(distill_eval),
        "dart": asdict(dart_eval),
        "adaptation_curve": curve,
    }


def summarize(records):
    import statistics as stats

    def mean_std(values):
        if len(values) == 1:
            return {"mean": values[0], "std": 0.0}
        return {"mean": stats.mean(values), "std": stats.stdev(values)}

    summary = {}
    for key in ("teacher", "scratch", "distill", "dart"):
        summary[key] = {}
        for metric in ("accuracy", "loss", "params", "target_ff_params", "target_ff_macs"):
            vals = [r[key][metric] for r in records]
            summary[key][metric] = mean_std(vals)

    # Adaptation summaries grouped by step count.
    steps = sorted({
        row["steps"]
        for r in records
        for row in r["adaptation_curve"]
    })
    summary["adaptation"] = {}
    for step in steps:
        rows = [
            row for r in records
            for row in r["adaptation_curve"]
            if row["steps"] == step
        ]
        summary["adaptation"][str(step)] = {
            metric: mean_std([x[metric] for x in rows])
            for metric in ("accuracy", "loss", "ff_params", "ff_macs")
        }

    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    p.add_argument("--tasks", type=str, nargs="+",
                   default=["add", "sub", "mul", "sort", "compose"])
    p.add_argument("--train-size", type=int, default=12000)
    p.add_argument("--test-size", type=int, default=3000)
    p.add_argument("--baseline-steps", type=int, default=1200)
    p.add_argument("--scratch-steps", type=int, default=1200)
    p.add_argument("--replacement-steps", type=int, default=400)
    p.add_argument("--adaptation-steps", type=int, nargs="+",
                   default=[0, 50, 100, 200, 400, 800])
    p.add_argument("--d-model", type=int, default=32)
    p.add_argument("--n-heads", type=int, default=2)
    p.add_argument("--d-ff", type=int, default=128)
    p.add_argument("--bottleneck", type=int, default=32)
    p.add_argument("--block", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--replacement-lr", type=float, default=1e-3)
    p.add_argument("--trace-batches", type=int, default=50)
    p.add_argument("--latency-warmup", type=int, default=30)
    p.add_argument("--latency-iters", type=int, default=50)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=str, default="dart02_results.json")
    args = p.parse_args()

    all_records = []
    for task in args.tasks:
        print(f"\n===== TASK: {task} =====")
        for seed in args.seeds:
            print(f"seed={seed}", flush=True)
            rec = run_one_seed(args, seed, task)
            all_records.append(rec)
            print(
                f"  teacher={rec['teacher']['accuracy']:.4f} "
                f"scratch={rec['scratch']['accuracy']:.4f} "
                f"distill={rec['distill']['accuracy']:.4f} "
                f"dart={rec['dart']['accuracy']:.4f}"
            )
            for row in rec["adaptation_curve"]:
                print(
                    f"  adapt[{row['steps']:>4}]={row['accuracy']:.4f}",
                    end=""
                )
            print()

    output = {
        "config": vars(args),
        "records": all_records,
        "summary_by_all_runs": summarize(all_records),
    }

    Path(args.out).write_text(json.dumps(output, indent=2), encoding="utf-8")

    print("\n================ DART-0.2 SUMMARY ================")
    s = output["summary_by_all_runs"]
    for key in ("teacher", "scratch", "distill", "dart"):
        a = s[key]["accuracy"]
        print(f"{key:<10} acc={a['mean']:.4f} ± {a['std']:.4f}")

    print("\nAdaptation curve:")
    for step, metrics in s["adaptation"].items():
        a = metrics["accuracy"]
        print(f"  {step:>4} steps: {a['mean']:.4f} ± {a['std']:.4f}")

    print(f"\nSaved: {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
