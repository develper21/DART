#!/usr/bin/env python3
"""
DART-0.3
========
Compute-matched controls + cross-task transfer.

Core question:
    Is DART+adaptation better than simply training a smaller model longer
    when total task-training compute is approximately matched?

Experiment groups:
    A. Teacher
    B. Scratch-budget-matched
    C. Distill + adaptation-budget-matched
    D. DART + adaptation-budget-matched

Optional transfer:
    Build replacement on SOURCE task.
    Then evaluate/adapt briefly on TARGET task.
    Compare DART transfer against Scratch transfer.

This is still a small research prototype. It is designed to falsify DART,
not to prove novelty.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset


VOCAB = list("0123456789+= ")
STOI = {c: i for i, c in enumerate(VOCAB)}
PAD = STOI[" "]
BLOCK_SIZE = 12


# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------

def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------
# Tasks
# ------------------------------------------------------------

def task_target(a: int, b: int, task: str) -> int:
    ad = [int(c) for c in str(a).zfill(3)]
    bd = [int(c) for c in str(b).zfill(3)]

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
    raise ValueError(task)


def make_example(a: int, b: int, task: str):
    text = f"{a}+{b}="
    ids = [STOI[c] for c in text]
    ids = (ids + [PAD] * BLOCK_SIZE)[:BLOCK_SIZE]
    return ids, task_target(a, b, task)


class TaskDataset(Dataset):
    def __init__(self, n: int, task: str, seed: int):
        rng = random.Random(seed)
        self.rows = []
        for _ in range(n):
            a = rng.randint(0, 999)
            b = rng.randint(0, 999)
            x, y = make_example(a, b, task)
            self.rows.append((
                torch.tensor(x, dtype=torch.long),
                torch.tensor(y, dtype=torch.long),
            ))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


# ------------------------------------------------------------
# Model
# ------------------------------------------------------------

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
    def __init__(self, d_model: int, heads: int, d_ff: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, heads, dropout=0.0, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a
        h = self.norm2(x)
        return x + self.ff(h)


class TinyTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        heads: int,
        d_ff: int,
        bottleneck: Optional[int] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Parameter(torch.randn(1, BLOCK_SIZE, d_model) * 0.02)
        self.blocks = nn.ModuleList(
            [Block(d_model, heads, d_ff) for _ in range(3)]
        )
        self.head = nn.Linear(d_model, 10)

        if bottleneck is not None:
            self.blocks[1].ff = SmallFF(d_model, bottleneck)

    def forward(self, x):
        h = self.emb(x) + self.pos[:, :x.size(1)]
        for block in self.blocks:
            h = block(h)
        return self.head(h[:, 0])


# ------------------------------------------------------------
# Compute accounting
# ------------------------------------------------------------

def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def ff_macs(ff: nn.Module) -> int:
    if isinstance(ff, nn.Sequential):
        ls = [m for m in ff if isinstance(m, nn.Linear)]
        return sum(m.in_features * m.out_features for m in ls)
    if isinstance(ff, SmallFF):
        ls = [m for m in ff.net if isinstance(m, nn.Linear)]
        return sum(m.in_features * m.out_features for m in ls)
    if hasattr(ff, "repl"):
        return ff_macs(ff.repl)
    raise TypeError(type(ff))


# We use a model FLOP estimator for relative training-compute matching.
# It is deliberately stable across model variants by evaluating the same
# input batch shape and counting forward/backward as 3x forward MACs.
def estimated_forward_macs(model: TinyTransformer, seq_len: int = BLOCK_SIZE) -> int:
    d = model.d_model
    heads = model.blocks[0].attn.num_heads
    total = 0

    # Embedding/output are ignored for matching; every model shares them.
    for block in model.blocks:
        total += 4 * seq_len * d * d  # QKV + output projection approx.
        total += 2 * seq_len * seq_len * d  # attention score + weighted sum
        ff = block.ff
        if isinstance(ff, nn.Sequential):
            ls = [m for m in ff if isinstance(m, nn.Linear)]
            total += sum(2 * m.in_features * m.out_features for m in ls)
        elif isinstance(ff, SmallFF):
            ls = [m for m in ff.net if isinstance(m, nn.Linear)]
            total += sum(2 * m.in_features * m.out_features for m in ls)
        else:
            total += ff_macs(ff)
    total += d * 10
    return total


def estimated_train_flops(model: TinyTransformer, steps: int, batch_size: int) -> float:
    # Approximate forward+backward = 3x forward MACs.
    return float(estimated_forward_macs(model) * batch_size * steps * 3)


# ------------------------------------------------------------
# Training / evaluation
# ------------------------------------------------------------

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
    total = correct = 0
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
        accuracy=correct / max(total, 1),
        loss=loss_sum / max(total, 1),
        params=count_params(model),
        target_ff_params=count_params(ff),
        target_ff_macs=ff_macs(ff),
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
    start = time.perf_counter()

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
            raise RuntimeError(f"non-finite loss: {float(loss)}")

        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    if device.type == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter() - start


# ------------------------------------------------------------
# Distillation + surgery
# ------------------------------------------------------------

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


def fit_replacement(tx, ty, d_model, bottleneck, device, steps, lr):
    repl = SmallFF(d_model, bottleneck).to(device)
    ds = torch.utils.data.TensorDataset(tx, ty)
    dl = DataLoader(ds, batch_size=512, shuffle=True)
    it = iter(dl)
    opt = torch.optim.AdamW(repl.parameters(), lr=lr)
    mse = nn.MSELoss()

    repl.train()
    for _ in range(steps):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(dl)
            x, y = next(it)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        pred = repl(x)
        loss = mse(pred, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(repl.parameters(), 1.0)
        opt.step()
    return repl


def replace_ff(model, block_idx, replacement):
    class WrappedFF(nn.Module):
        def __init__(self, repl):
            super().__init__()
            self.repl = repl

        def forward(self, x):
            return self.repl(x)

    model.blocks[block_idx].ff = WrappedFF(replacement)


# ------------------------------------------------------------
# Latency
# ------------------------------------------------------------

@torch.no_grad()
def cuda_latency(model, loader, device, warmup=30, iters=50):
    if device.type != "cuda":
        return None

    model.eval()
    x = next(iter(loader))[0].to(device, non_blocking=True)

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


# ------------------------------------------------------------
# One task, one seed
# ------------------------------------------------------------

def run_task_seed(args, seed, task):
    device = torch.device(args.device)

    train = TaskDataset(args.train_size, task, seed)
    test = TaskDataset(args.test_size, task, seed + 9999)

    dl_train = DataLoader(
        train, batch_size=args.batch_size, shuffle=True,
        pin_memory=device.type == "cuda"
    )
    dl_test = DataLoader(
        test, batch_size=args.batch_size, shuffle=False,
        pin_memory=device.type == "cuda"
    )

    # --------------------------------------------------------
    # Teacher
    # --------------------------------------------------------
    seed_everything(seed)
    teacher = TinyTransformer(
        len(VOCAB), args.d_model, args.heads, args.d_ff
    ).to(device)

    train_task(
        teacher, dl_train, device,
        args.teacher_steps, args.lr
    )

    teacher_eval = evaluate(teacher, dl_test, device, args.block)

    # --------------------------------------------------------
    # Trace once; all downstream controls use same teacher trace.
    # --------------------------------------------------------
    tx, ty = collect_trace(
        teacher, dl_train, device, args.block, args.trace_batches
    )

    # Distilled replacement.
    seed_everything(seed + 5000)
    replacement = fit_replacement(
        tx, ty,
        args.d_model, args.bottleneck,
        device, args.replacement_steps,
        args.replacement_lr,
    )

    # --------------------------------------------------------
    # Compute-matched budget
    #
    # We define a target total task-training FLOP budget for all
    # small-model controls. The teacher's training cost is kept separate.
    # The base small model has lower cost per step, therefore it gets
    # more steps so that:
    #
    #   scratch_compute ~= distill+adapt_compute ~= dart+adapt_compute
    # --------------------------------------------------------
    scratch_template = TinyTransformer(
        len(VOCAB), args.d_model, args.heads, args.d_ff, args.bottleneck
    ).to(device)

    small_step_flops = estimated_train_flops(
        scratch_template, 1, args.batch_size
    )
    total_small_budget = args.small_budget_mult * estimated_train_flops(
        scratch_template, args.reference_steps, args.batch_size
    )

    budget_steps = max(
        args.reference_steps,
        int(round(total_small_budget / max(small_step_flops, 1.0)))
    )

    # To avoid accidental unfairness from different runs:
    # all task-training controls get exactly budget_steps steps.
    actual_small_flops = estimated_train_flops(
        scratch_template, budget_steps, args.batch_size
    )

    # --------------------------------------------------------
    # Scratch-small
    # --------------------------------------------------------
    seed_everything(seed + 1000)
    scratch = TinyTransformer(
        len(VOCAB), args.d_model, args.heads,
        args.d_ff, args.bottleneck
    ).to(device)

    scratch_seconds = train_task(
        scratch, dl_train, device,
        budget_steps, args.lr
    )

    scratch_eval = evaluate(scratch, dl_test, device, args.block)

    # --------------------------------------------------------
    # DART
    # --------------------------------------------------------
    dart = copy.deepcopy(teacher).to(device)
    replace_ff(
        dart, args.block,
        copy.deepcopy(replacement).to(device)
    )

    dart_eval_before = evaluate(dart, dl_test, device, args.block)

    # --------------------------------------------------------
    # DART+Adapt with exactly the same number of task steps
    # as Scratch-small.
    # --------------------------------------------------------
    dart_adapt = copy.deepcopy(dart).to(device)
    dart_adapt_seconds = train_task(
        dart_adapt, dl_train, device,
        budget_steps, args.lr
    )
    dart_adapt_eval = evaluate(
        dart_adapt, dl_test, device, args.block
    )

    # --------------------------------------------------------
    # Distill + Adapt:
    # Start from same surgically-distilled network as DART.
    # It receives exactly the same task budget as Scratch and DART.
    # This is the critical control.
    # --------------------------------------------------------
    distill_adapt = copy.deepcopy(dart).to(device)
    distill_adapt_seconds = train_task(
        distill_adapt, dl_train, device,
        budget_steps, args.lr
    )
    distill_adapt_eval = evaluate(
        distill_adapt, dl_test, device, args.block
    )

    result = {
        "seed": seed,
        "task": task,
        "teacher": asdict(teacher_eval),
        "scratch": asdict(scratch_eval),
        "dart_before_adaptation": asdict(dart_eval_before),
        "dart_adapt": asdict(dart_adapt_eval),
        "distill_adapt": asdict(distill_adapt_eval),
        "compute": {
            "budget_steps": budget_steps,
            "estimated_small_training_flops": actual_small_flops,
            "scratch_training_seconds": scratch_seconds,
            "dart_adaptation_seconds": dart_adapt_seconds,
            "distill_adaptation_seconds": distill_adapt_seconds,
            "teacher_target_ff_macs_per_token": ff_macs(
                teacher.blocks[args.block].ff
            ),
            "small_target_ff_macs_per_token": ff_macs(
                dart_adapt.blocks[args.block].ff
            ),
        },
        "latency_ms": {
            "teacher": cuda_latency(
                teacher, dl_test, device,
                args.latency_warmup, args.latency_iters
            ),
            "scratch": cuda_latency(
                scratch, dl_test, device,
                args.latency_warmup, args.latency_iters
            ),
            "dart_adapt": cuda_latency(
                dart_adapt, dl_test, device,
                args.latency_warmup, args.latency_iters
            ),
        },
    }

    return result


# ------------------------------------------------------------
# Transfer experiment
# ------------------------------------------------------------

def run_transfer(args, seed, source_task, target_task):
    """
    Source -> replacement -> target.

    We compare:
      DART transfer: source-trained teacher with its replacement, then target adaptation.
      Scratch transfer: fresh small model trained on the target for same adaptation budget.

    The target is deliberately different so success would suggest some
    reuse beyond exact source-task behavior.
    """
    device = torch.device(args.device)

    source_train = TaskDataset(args.train_size, source_task, seed)
    target_train = TaskDataset(args.train_size, target_task, seed + 20000)
    target_test = TaskDataset(args.test_size, target_task, seed + 30000)

    src_dl = DataLoader(
        source_train, batch_size=args.batch_size, shuffle=True,
        pin_memory=device.type == "cuda"
    )
    tgt_dl = DataLoader(
        target_train, batch_size=args.batch_size, shuffle=True,
        pin_memory=device.type == "cuda"
    )
    test_dl = DataLoader(
        target_test, batch_size=args.batch_size, shuffle=False,
        pin_memory=device.type == "cuda"
    )

    # Train source teacher.
    seed_everything(seed + 7000)
    teacher = TinyTransformer(
        len(VOCAB), args.d_model, args.heads, args.d_ff
    ).to(device)

    train_task(
        teacher, src_dl, device,
        args.teacher_steps, args.lr
    )

    tx, ty = collect_trace(
        teacher, src_dl, device,
        args.block, args.trace_batches
    )

    seed_everything(seed + 8000)
    replacement = fit_replacement(
        tx, ty,
        args.d_model, args.bottleneck,
        device,
        args.replacement_steps,
        args.replacement_lr,
    )

    dart = copy.deepcopy(teacher).to(device)
    replace_ff(dart, args.block, copy.deepcopy(replacement).to(device))

    # Target adaptation: both start from different initializations,
    # but get exactly the same target-update budget.
    scratch = TinyTransformer(
        len(VOCAB), args.d_model, args.heads,
        args.d_ff, args.bottleneck
    ).to(device)

    # Same target steps for both.
    scratch_before = evaluate(scratch, test_dl, device, args.block)
    dart_before = evaluate(dart, test_dl, device, args.block)

    train_task(scratch, tgt_dl, device, args.transfer_steps, args.lr)
    train_task(dart, tgt_dl, device, args.transfer_steps, args.lr)

    scratch_after = evaluate(scratch, test_dl, device, args.block)
    dart_after = evaluate(dart, test_dl, device, args.block)

    return {
        "seed": seed,
        "source_task": source_task,
        "target_task": target_task,
        "scratch_before": asdict(scratch_before),
        "dart_before": asdict(dart_before),
        "scratch_after": asdict(scratch_after),
        "dart_after": asdict(dart_after),
        "transfer_steps": args.transfer_steps,
    }


def mean_std(values):
    if len(values) <= 1:
        return {"mean": values[0] if values else None, "std": 0.0}
    import statistics
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values),
    }


def summarize(records, key):
    return mean_std([r[key]["accuracy"] for r in records])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    p.add_argument("--tasks", nargs="+", default=["add", "sub", "mul", "sort", "compose"])
    p.add_argument("--transfer-pairs", nargs=2, action="append",
                   metavar=("SOURCE", "TARGET"),
                   default=[["add", "compose"], ["mul", "sub"]])

    p.add_argument("--train-size", type=int, default=12000)
    p.add_argument("--test-size", type=int, default=3000)

    p.add_argument("--teacher-steps", type=int, default=1200)
    p.add_argument("--reference-steps", type=int, default=2000)
    p.add_argument("--small-budget-mult", type=float, default=1.0)

    p.add_argument("--replacement-steps", type=int, default=400)
    p.add_argument("--transfer-steps", type=int, default=800)

    p.add_argument("--d-model", type=int, default=32)
    p.add_argument("--heads", type=int, default=2)
    p.add_argument("--d-ff", type=int, default=128)
    p.add_argument("--bottleneck", type=int, default=32)
    p.add_argument("--block", type=int, default=1)

    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--replacement-lr", type=float, default=1e-3)
    p.add_argument("--trace-batches", type=int, default=50)

    p.add_argument("--latency-warmup", type=int, default=30)
    p.add_argument("--latency-iters", type=int, default=50)

    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu"
    )
    p.add_argument("--out", default="dart03_results.json")
    args = p.parse_args()

    records = []

    for task in args.tasks:
        print(f"\n===== TASK: {task} =====")
        for seed in args.seeds:
            print(f"seed={seed}", flush=True)
            r = run_task_seed(args, seed, task)
            records.append(r)
            print(
                f" teacher={r['teacher']['accuracy']:.4f}"
                f" scratch={r['scratch']['accuracy']:.4f}"
                f" dart0={r['dart_before_adaptation']['accuracy']:.4f}"
                f" dart+adapt={r['dart_adapt']['accuracy']:.4f}"
                f" distill+adapt={r['distill_adapt']['accuracy']:.4f}"
            )

    print("\n===== TRANSFER =====")
    transfer_records = []
    for pair in args.transfer_pairs:
        source, target = pair
        print(f"\n{source} -> {target}")
        for seed in args.seeds:
            r = run_transfer(args, seed, source, target)
            transfer_records.append(r)
            print(
                f" seed={seed}"
                f" scratch_before={r['scratch_before']['accuracy']:.4f}"
                f" dart_before={r['dart_before']['accuracy']:.4f}"
                f" scratch_after={r['scratch_after']['accuracy']:.4f}"
                f" dart_after={r['dart_after']['accuracy']:.4f}"
            )

    summary = {}
    for key in (
        "teacher",
        "scratch",
        "dart_before_adaptation",
        "dart_adapt",
        "distill_adapt",
    ):
        summary[key] = summarize(records, key)

    # Group summaries by task.
    by_task = {}
    for task in args.tasks:
        rows = [r for r in records if r["task"] == task]
        by_task[task] = {
            key: summarize(rows, key)
            for key in (
                "teacher",
                "scratch",
                "dart_before_adaptation",
                "dart_adapt",
                "distill_adapt",
            )
        }

    transfer_summary = {}
    for pair in args.transfer_pairs:
        source, target = pair
        rows = [
            r for r in transfer_records
            if r["source_task"] == source and r["target_task"] == target
        ]
        transfer_summary[f"{source}->{target}"] = {
            "scratch_before": mean_std([r["scratch_before"]["accuracy"] for r in rows]),
            "dart_before": mean_std([r["dart_before"]["accuracy"] for r in rows]),
            "scratch_after": mean_std([r["scratch_after"]["accuracy"] for r in rows]),
            "dart_after": mean_std([r["dart_after"]["accuracy"] for r in rows]),
        }

    output = {
        "config": vars(args),
        "records": records,
        "summary_all_tasks": summary,
        "summary_by_task": by_task,
        "transfer_records": transfer_records,
        "transfer_summary": transfer_summary,
    }

    Path(args.out).write_text(json.dumps(output, indent=2), encoding="utf-8")

    print("\n================ DART-0.3 SUMMARY ================")
    for key in (
        "teacher",
        "scratch",
        "dart_before_adaptation",
        "distill_adapt",
        "dart_adapt",
    ):
        s = summary[key]
        print(f"{key:<24} {s['mean']:.4f} ± {s['std']:.4f}")

    print("\nTransfer:")
    for pair, s in transfer_summary.items():
        print(
            f"{pair}: "
            f"scratch_after={s['scratch_after']['mean']:.4f} ± {s['scratch_after']['std']:.4f}, "
            f"dart_after={s['dart_after']['mean']:.4f} ± {s['dart_after']['std']:.4f}"
        )

    print(f"\nSaved: {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
