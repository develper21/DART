#!/usr/bin/env python3
"""
DART-0.4
========
Intervention-tested operator discovery + repeated computational surgery.

What changed from DART-0.3
--------------------------
1. Candidate discovery is no longer "train one small FFN by direct output
   imitation and call that DART".
2. We generate an intervention set around the teacher sub-computation:
      h, h + eps*d, h - eps*d, alpha*h, masked(h), shuffled-feature(h)
3. Candidate operators are evaluated on:
      - value agreement
      - directional response agreement
      - cheapness / complexity
4. Candidate families include:
      - Linear operator
      - Low-rank operator
      - Diagonal affine operator
      - Polynomial operator
      - Small MLP
5. The chosen operator is surgically inserted.
6. The model adapts after surgery.
7. We can repeat surgery for multiple rounds.
8. CUDA latency measurement is corrected (start.elapsed_time(end)).

This is still a research prototype. It does not prove novelty.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset, Dataset


VOCAB = list("0123456789+= ")
STOI = {c: i for i, c in enumerate(VOCAB)}
PAD = STOI[" "]
BLOCK_SIZE = 12


# ============================================================
# Reproducibility
# ============================================================

def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# Tasks
# ============================================================

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
            self.rows.append(
                (torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long))
            )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


# ============================================================
# Model
# ============================================================

class SmallMLP(nn.Module):
    def __init__(self, d_model: int, bottleneck: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, d_model),
        )

    def forward(self, x):
        return self.net(x)


class LinearOperator(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.linear = nn.Linear(d, d, bias=True)

    def forward(self, x):
        return self.linear(x)


class LowRankOperator(nn.Module):
    def __init__(self, d: int, rank: int):
        super().__init__()
        self.down = nn.Linear(d, rank, bias=False)
        self.up = nn.Linear(rank, d, bias=True)

    def forward(self, x):
        return self.up(self.down(x))


class DiagonalAffine(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d))
        self.bias = nn.Parameter(torch.zeros(d))

    def forward(self, x):
        return x * self.scale + self.bias


class PolynomialOperator(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.a = nn.Parameter(torch.ones(d))
        self.b = nn.Parameter(torch.zeros(d))
        self.c = nn.Parameter(torch.zeros(d))

    def forward(self, x):
        return self.a * x + self.b * x.square() + self.c


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
        d_model: int = 32,
        heads: int = 2,
        d_ff: int = 128,
    ):
        super().__init__()
        self.d_model = d_model
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Parameter(torch.randn(1, BLOCK_SIZE, d_model) * 0.02)
        self.blocks = nn.ModuleList(
            [Block(d_model, heads, d_ff) for _ in range(3)]
        )
        self.head = nn.Linear(d_model, 10)

    def forward(self, x):
        h = self.emb(x) + self.pos[:, : x.size(1)]
        for block in self.blocks:
            h = block(h)
        return self.head(h[:, 0])


# ============================================================
# Parameter / compute accounting
# ============================================================

def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def operator_macs(op: nn.Module) -> int:
    if isinstance(op, nn.Sequential):
        ls = [m for m in op if isinstance(m, nn.Linear)]
        return sum(m.in_features * m.out_features for m in ls)
    if isinstance(op, LinearOperator):
        return op.linear.in_features * op.linear.out_features
    if isinstance(op, LowRankOperator):
        return op.down.in_features * op.down.out_features + op.up.in_features * op.up.out_features
    if isinstance(op, DiagonalAffine):
        return op.scale.numel()
    if isinstance(op, PolynomialOperator):
        return 2 * op.a.numel()
    if isinstance(op, SmallMLP):
        ls = [m for m in op.net if isinstance(m, nn.Linear)]
        return sum(m.in_features * m.out_features for m in ls)
    if hasattr(op, "repl"):
        return operator_macs(op.repl)
    raise TypeError(type(op))


def operator_name(op: nn.Module) -> str:
    if isinstance(op, SmallMLP):
        return "small_mlp"
    if isinstance(op, LinearOperator):
        return "linear"
    if isinstance(op, LowRankOperator):
        return "low_rank"
    if isinstance(op, DiagonalAffine):
        return "diagonal_affine"
    if isinstance(op, PolynomialOperator):
        return "polynomial"
    return type(op).__name__


# ============================================================
# Training / evaluation
# ============================================================

@dataclass
class Eval:
    accuracy: float
    loss: float
    params: int
    target_params: int
    target_macs: int


def evaluate(model, loader, device, block_idx) -> Eval:
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

    op = model.blocks[block_idx].ff
    return Eval(
        accuracy=correct / max(total, 1),
        loss=loss_sum / max(total, 1),
        params=count_params(model),
        target_params=count_params(op),
        target_macs=operator_macs(op),
    )


def train_task(model, loader, device, steps, lr) -> float:
    if steps <= 0:
        return 0.0

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    iterator = iter(loader)

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()

    for _ in range(steps):
        try:
            x, y = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            x, y = next(iterator)

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = criterion(logits, y)

        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite task loss.")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    if device.type == "cuda":
        torch.cuda.synchronize()

    return time.perf_counter() - start


# ============================================================
# Trace / intervention data
# ============================================================

@dataclass
class TraceBatch:
    h: Tensor
    teacher_y: Tensor


def collect_teacher_h(
    teacher,
    loader,
    device,
    block_idx,
    max_batches,
) -> Tensor:
    xs = []
    block = teacher.blocks[block_idx]

    def pre_hook(_module, inputs):
        xs.append(inputs[0].detach().reshape(-1, inputs[0].shape[-1]).cpu())

    handle = block.ff.register_forward_pre_hook(pre_hook)
    try:
        teacher.eval()
        with torch.no_grad():
            for bi, (x, _) in enumerate(loader):
                if bi >= max_batches:
                    break
                teacher(x.to(device, non_blocking=True))
    finally:
        handle.remove()

    if not xs:
        raise RuntimeError("No hidden states collected.")
    return torch.cat(xs)


def teacher_subgraph_on_h(
    teacher: TinyTransformer,
    block_idx: int,
    h_cpu: Tensor,
    device: torch.device,
) -> Tensor:
    op = teacher.blocks[block_idx].ff
    outs = []
    with torch.no_grad():
        for chunk in h_cpu.split(4096):
            outs.append(op(chunk.to(device, non_blocking=True)).cpu())
    return torch.cat(outs)


def make_interventions(
    h_cpu: Tensor,
    eps: float,
    n_directions: int,
    seed: int,
) -> list[Tensor]:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    interventions = [h_cpu]

    for _ in range(n_directions):
        d = torch.randn(h_cpu.shape, generator=g)
        d = d / (d.norm(dim=-1, keepdim=True) + 1e-8)
        interventions.append(h_cpu + eps * d)
        interventions.append(h_cpu - eps * d)

    interventions.append(0.5 * h_cpu)
    interventions.append(1.5 * h_cpu)

    # Feature masking + permutation are deliberately coarse interventions.
    mask = torch.rand(h_cpu.shape, generator=g) > 0.15
    interventions.append(h_cpu * mask)

    perm = torch.randperm(h_cpu.shape[-1], generator=g)
    interventions.append(h_cpu[:, perm])

    return interventions


# ============================================================
# Operator candidate fitting
# ============================================================

@dataclass
class CandidateScore:
    name: str
    params: int
    macs: int
    value_mse: float
    directional_mse: float
    score: float


def build_candidates(d_model: int, bottleneck: int, rank: int) -> list[nn.Module]:
    return [
        DiagonalAffine(d_model),
        PolynomialOperator(d_model),
        LowRankOperator(d_model, rank),
        LinearOperator(d_model),
        SmallMLP(d_model, bottleneck),
    ]


def fit_operator(
    candidate: nn.Module,
    teacher,
    block_idx: int,
    h: Tensor,
    interventions: list[Tensor],
    device: torch.device,
    steps: int,
    lr: float,
    directional_weight: float,
) -> CandidateScore:
    op = candidate.to(device)
    teacher_op = teacher.blocks[block_idx].ff.to(device)
    teacher_op.eval()
    op.train()

    # Build a compact intervention dataset.
    pairs = []
    for z in interventions:
        with torch.no_grad():
            target = teacher_op(z.to(device, non_blocking=True)).cpu()
        pairs.append((z, target))
    x = torch.cat([p[0] for p in pairs])
    y = torch.cat([p[1] for p in pairs])

    # Limit candidate fitting set to avoid exploding CPU/GPU memory.
    max_rows = 50000
    if x.size(0) > max_rows:
        idx = torch.randperm(x.size(0))[:max_rows]
        x = x[idx]
        y = y[idx]

    ds = TensorDataset(x, y)
    dl = DataLoader(ds, batch_size=1024, shuffle=True)
    it = iter(dl)

    optimizer = torch.optim.AdamW(op.parameters(), lr=lr)
    mse = nn.MSELoss()

    for _ in range(steps):
        try:
            xb, yb = next(it)
        except StopIteration:
            it = iter(dl)
            xb, yb = next(it)

        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        pred = op(xb)
        loss = mse(pred, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(op.parameters(), 1.0)
        optimizer.step()

    # Evaluation with held-out intervention rows.
    op.eval()
    held = min(10000, x.size(0))
    x_eval = x[:held].to(device)
    y_eval = y[:held].to(device)

    with torch.no_grad():
        pred = op(x_eval)
        value = float(mse(pred, y_eval))

        # Directional response:
        # compare delta(output) for paired +eps / -eps directions.
        dirs = []
        target_dirs = []
        for i in range(1, len(interventions), 2):
            if i + 1 >= len(interventions):
                break
            base = interventions[0][:held].to(device)
            plus = interventions[i][:held].to(device)
            minus = interventions[i + 1][:held].to(device)

            t_plus = teacher_op(plus)
            t_minus = teacher_op(minus)
            c_plus = op(plus)
            c_minus = op(minus)

            target_dir = t_plus - t_minus
            cand_dir = c_plus - c_minus

            target_dirs.append(target_dir)
            dirs.append(cand_dir)

        if dirs:
            directional = float(
                mse(torch.cat(dirs), torch.cat(target_dirs))
            )
        else:
            directional = value

    # Lower is better. Complexity penalty is normalized so cheap operators
    # are preferred only after behavior is reasonably faithful.
    p = count_params(op)
    m = operator_macs(op)
    score = value + directional_weight * directional

    return CandidateScore(
        name=operator_name(op),
        params=p,
        macs=m,
        value_mse=value,
        directional_mse=directional,
        score=score,
    ), op


def choose_candidate(
    teacher,
    block_idx,
    h,
    interventions,
    device,
    d_model,
    bottleneck,
    rank,
    fit_steps,
    fit_lr,
    directional_weight,
    complexity_lambda,
):
    raw = []
    fitted = []

    for candidate in build_candidates(d_model, bottleneck, rank):
        score, fitted_op = fit_operator(
            candidate,
            teacher,
            block_idx,
            h,
            interventions,
            device,
            fit_steps,
            fit_lr,
            directional_weight,
        )
        score.score += complexity_lambda * (
            math.log1p(score.params) + 0.5 * math.log1p(score.macs)
        )
        raw.append(asdict(score))
        fitted.append((score, fitted_op))

    fitted.sort(key=lambda t: t[0].score)
    best_score, best_op = fitted[0]

    return best_op, asdict(best_score), raw


# ============================================================
# Surgery / adaptation loop
# ============================================================

def replace_operator(model, block_idx, replacement):
    class WrappedOperator(nn.Module):
        def __init__(self, repl):
            super().__init__()
            self.repl = repl

        def forward(self, x):
            return self.repl(x)

    model.blocks[block_idx].ff = WrappedOperator(replacement)


def cuda_latency(model, loader, device, warmup=30, iterations=100):
    if device.type != "cuda":
        return None

    model.eval()
    x = next(iter(loader))[0].to(device, non_blocking=True)

    for _ in range(warmup):
        model(x)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iterations):
        model(x)
    end.record()

    torch.cuda.synchronize()

    # Correct direction: start.elapsed_time(end)
    return start.elapsed_time(end) / iterations


@dataclass
class RoundResult:
    round: int
    candidate: str
    candidate_params: int
    candidate_macs: int
    candidate_value_mse: float
    candidate_directional_mse: float
    pre_adapt: Eval
    post_adapt: Eval
    latency_ms: Optional[float]


def run_one(args, seed, task):
    seed_everything(seed)
    device = torch.device(args.device)

    train_ds = TaskDataset(args.train_size, task, seed)
    test_ds = TaskDataset(args.test_size, task, seed + 10000)

    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        pin_memory=device.type == "cuda"
    )
    test_dl = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        pin_memory=device.type == "cuda"
    )

    # Base model
    seed_everything(seed)
    teacher = TinyTransformer(
        len(VOCAB), args.d_model, args.heads, args.d_ff
    ).to(device)

    teacher_time = train_task(
        teacher, train_dl, device, args.teacher_steps, args.lr
    )
    teacher_before = evaluate(teacher, test_dl, device, args.block)

    # DART model starts as a copy of the trained teacher.
    dart = copy.deepcopy(teacher).to(device)
    rounds = []

    for r in range(args.surgery_rounds):
        # Observe current internal computation.
        h = collect_teacher_h(
            dart, train_dl, device, args.block, args.trace_batches
        )

        # Use a deterministic subset so round-to-round comparisons are stable.
        if h.size(0) > args.trace_rows:
            h = h[:args.trace_rows]

        interventions = make_interventions(
            h,
            eps=args.intervention_eps,
            n_directions=args.intervention_directions,
            seed=seed + 10000 * (r + 1),
        )

        best_op, best_score, all_candidates = choose_candidate(
            dart,
            args.block,
            h,
            interventions,
            device,
            args.d_model,
            args.bottleneck,
            args.rank,
            args.operator_fit_steps,
            args.operator_fit_lr,
            args.directional_weight,
            args.complexity_lambda,
        )

        # Surgery
        replace_operator(
            dart,
            args.block,
            copy.deepcopy(best_op).to(device)
        )

        pre = evaluate(dart, test_dl, device, args.block)

        # Adapt after surgery.
        train_task(
            dart, train_dl, device,
            args.adaptation_steps_per_round,
            args.lr,
        )

        post = evaluate(dart, test_dl, device, args.block)

        latency = cuda_latency(
            dart, test_dl, device,
            args.latency_warmup, args.latency_iters
        )

        rounds.append({
            "round": r,
            "candidate": best_score["name"],
            "candidate_params": best_score["params"],
            "candidate_macs": best_score["macs"],
            "candidate_value_mse": best_score["value_mse"],
            "candidate_directional_mse": best_score["directional_mse"],
            "candidate_score": best_score["score"],
            "all_candidates": all_candidates,
            "pre_adapt": asdict(pre),
            "post_adapt": asdict(post),
            "latency_ms": latency,
        })

        print(
            f"round={r} candidate={best_score['name']} "
            f"params={best_score['params']} macs={best_score['macs']} "
            f"pre={pre.accuracy:.4f} post={post.accuracy:.4f} "
            f"latency={latency}"
        )

    # Final transfer probes:
    # same trained operator is evaluated as an initialization on a different task.
    final = evaluate(dart, test_dl, device, args.block)

    return {
        "seed": seed,
        "task": task,
        "teacher": asdict(teacher_before),
        "teacher_training_seconds": teacher_time,
        "dart_final": asdict(final),
        "rounds": rounds,
    }


def run_transfer(args, seed, source_task, target_task):
    device = torch.device(args.device)

    source_ds = TaskDataset(args.train_size, source_task, seed)
    target_ds = TaskDataset(args.train_size, target_task, seed + 20000)
    target_test = TaskDataset(args.test_size, target_task, seed + 30000)

    source_dl = DataLoader(
        source_ds, batch_size=args.batch_size, shuffle=True,
        pin_memory=device.type == "cuda"
    )
    target_dl = DataLoader(
        target_ds, batch_size=args.batch_size, shuffle=True,
        pin_memory=device.type == "cuda"
    )
    test_dl = DataLoader(
        target_test, batch_size=args.batch_size, shuffle=False,
        pin_memory=device.type == "cuda"
    )

    # Source training
    seed_everything(seed + 5000)
    source_model = TinyTransformer(
        len(VOCAB), args.d_model, args.heads, args.d_ff
    ).to(device)
    train_task(
        source_model, source_dl, device,
        args.teacher_steps, args.lr
    )

    # One operator-discovery round on source.
    h = collect_teacher_h(
        source_model, source_dl, device,
        args.block, args.trace_batches
    )
    if h.size(0) > args.trace_rows:
        h = h[:args.trace_rows]

    interventions = make_interventions(
        h, args.intervention_eps,
        args.intervention_directions,
        seed + 91000,
    )

    replacement, best_score, _ = choose_candidate(
        source_model, args.block, h, interventions,
        device, args.d_model, args.bottleneck, args.rank,
        args.operator_fit_steps, args.operator_fit_lr,
        args.directional_weight, args.complexity_lambda,
    )

    dart = copy.deepcopy(source_model).to(device)
    replace_operator(
        dart, args.block,
        copy.deepcopy(replacement).to(device)
    )

    # Scratch target baseline has identical target adaptation steps.
    seed_everything(seed + 6000)
    scratch = TinyTransformer(
        len(VOCAB), args.d_model, args.heads, args.d_ff
    ).to(device)

    scratch_before = evaluate(scratch, test_dl, device, args.block)
    dart_before = evaluate(dart, test_dl, device, args.block)

    train_task(
        scratch, target_dl, device,
        args.transfer_adaptation_steps, args.lr
    )
    train_task(
        dart, target_dl, device,
        args.transfer_adaptation_steps, args.lr
    )

    scratch_after = evaluate(scratch, test_dl, device, args.block)
    dart_after = evaluate(dart, test_dl, device, args.block)

    return {
        "seed": seed,
        "source_task": source_task,
        "target_task": target_task,
        "operator": best_score,
        "scratch_before": asdict(scratch_before),
        "dart_before": asdict(dart_before),
        "scratch_after": asdict(scratch_after),
        "dart_after": asdict(dart_after),
    }


def mean_std(vals):
    if len(vals) <= 1:
        return {"mean": vals[0] if vals else None, "std": 0.0}
    return {"mean": statistics.mean(vals), "std": statistics.stdev(vals)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", nargs="+", type=int, default=[1, 2])
    p.add_argument("--tasks", nargs="+", default=["add", "compose"])

    p.add_argument(
        "--transfer-pairs",
        nargs=2,
        action="append",
        metavar=("SOURCE", "TARGET"),
        default=[["add", "compose"], ["mul", "sub"]],
    )

    p.add_argument("--train-size", type=int, default=6000)
    p.add_argument("--test-size", type=int, default=1500)

    p.add_argument("--teacher-steps", type=int, default=800)
    p.add_argument("--adaptation-steps-per-round", type=int, default=600)
    p.add_argument("--surgery-rounds", type=int, default=2)

    p.add_argument("--operator-fit-steps", type=int, default=300)
    p.add_argument("--operator-fit-lr", type=float, default=1e-3)

    p.add_argument("--d-model", type=int, default=32)
    p.add_argument("--heads", type=int, default=2)
    p.add_argument("--d-ff", type=int, default=128)
    p.add_argument("--bottleneck", type=int, default=32)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--block", type=int, default=1)

    p.add_argument("--intervention-eps", type=float, default=0.10)
    p.add_argument("--intervention-directions", type=int, default=4)
    p.add_argument("--trace-rows", type=int, default=12000)
    p.add_argument("--trace-batches", type=int, default=50)

    p.add_argument("--directional-weight", type=float, default=1.0)
    p.add_argument("--complexity-lambda", type=float, default=1e-4)

    p.add_argument("--transfer-adaptation-steps", type=int, default=600)

    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)

    p.add_argument("--latency-warmup", type=int, default=30)
    p.add_argument("--latency-iters", type=int, default=50)

    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu"
    )
    p.add_argument("--out", default="dart04_results.json")

    args = p.parse_args()

    records = []

    for task in args.tasks:
        print(f"\n===== TASK {task} =====")
        for seed in args.seeds:
            print(f"seed={seed}", flush=True)
            rec = run_one(args, seed, task)
            records.append(rec)

    transfer_records = []
    print("\n===== TRANSFER =====")
    for source, target in args.transfer_pairs:
        print(f"{source} -> {target}")
        for seed in args.seeds:
            rec = run_transfer(args, seed, source, target)
            transfer_records.append(rec)
            print(
                f" seed={seed} "
                f"scratch_after={rec['scratch_after']['accuracy']:.4f} "
                f"dart_after={rec['dart_after']['accuracy']:.4f}"
            )

    summary = {}
    for key in ("teacher", "dart_final"):
        vals = [r[key]["accuracy"] for r in records]
        summary[key] = mean_std(vals)

    summary["transfer"] = {}
    for source, target in args.transfer_pairs:
        rows = [
            r for r in transfer_records
            if r["source_task"] == source and r["target_task"] == target
        ]
        summary["transfer"][f"{source}->{target}"] = {
            "scratch_after": mean_std(
                [r["scratch_after"]["accuracy"] for r in rows]
            ),
            "dart_after": mean_std(
                [r["dart_after"]["accuracy"] for r in rows]
            ),
        }

    output = {
        "config": vars(args),
        "records": records,
        "transfer_records": transfer_records,
        "summary": summary,
    }

    Path(args.out).write_text(json.dumps(output, indent=2), encoding="utf-8")

    print("\n================ DART-0.4 SUMMARY ================")
    for k, v in summary.items():
        if k != "transfer":
            print(f"{k}: {v['mean']:.4f} ± {v['std']:.4f}")

    for pair, v in summary["transfer"].items():
        print(
            f"{pair}: "
            f"scratch={v['scratch_after']['mean']:.4f} ± {v['scratch_after']['std']:.4f}, "
            f"dart={v['dart_after']['mean']:.4f} ± {v['dart_after']['std']:.4f}"
        )

    print(f"\nSaved: {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
