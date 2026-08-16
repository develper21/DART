#!/usr/bin/env python3
"""
DART-0.6
========
Trajectory Compression / Reusable Transition Discovery.

Research hypothesis:
    The useful computation learned by a tiny Transformer may be distributed
    across several blocks.  A single-block replacement repeatedly collapsed
    to a small neural approximator in DART-0.5, so DART-0.6 changes the unit
    of discovery from one FFN to the whole state trajectory across a block
    span.

Core experiment:
    h0 -> h1 -> h2 -> h3

Search for ONE reusable transition operator O such that:
    O(h0) ~ h1
    O(h1) ~ h2
    O(h2) ~ h3

Then surgically replace the selected block span with:
    h3_hat = O(O(O(h0)))

The primary research signal is not hidden-tensor compression itself.  The
operator is selected for downstream task behaviour, while a held-out
trajectory verifier measures whether the same transition rule remains
consistent across all three transitions.

A small MLP is retained as an explicit control, not as the intended answer.
No claim of novelty or algorithmic discovery is made unless cross-task
transfer and repeated-transition consistency provide supporting evidence.
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
from typing import Iterable, Optional

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset


VOCAB = list("0123456789+= ")
STOI = {c: i for i, c in enumerate(VOCAB)}
PAD = STOI[" "]
BLOCK_SIZE = 12


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
    raise ValueError(f"Unknown task: {task}")


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
            self.rows.append((torch.tensor(x), torch.tensor(y)))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index: int):
        return self.rows[index]


class Block(nn.Module):
    def __init__(self, d_model: int, heads: int, d_ff: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, heads, dropout=0.0, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))

    def forward(self, x: Tensor) -> Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        h = self.norm2(x)
        return x + self.ff(h)


class TinyTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 32, heads: int = 2, d_ff: int = 128, depth: int = 3):
        super().__init__()
        self.d_model = d_model
        self.depth = depth
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Parameter(torch.randn(1, BLOCK_SIZE, d_model) * 0.02)
        self.blocks = nn.ModuleList([Block(d_model, heads, d_ff) for _ in range(depth)])
        self.head = nn.Linear(d_model, 10)

    def forward(self, x: Tensor) -> Tensor:
        h = self.emb(x) + self.pos[:, :x.size(1)]
        for block in self.blocks:
            h = block(h)
        return self.head(h[:, 0])


# ---------------------------------------------------------------------------
# Reusable transition operators
# ---------------------------------------------------------------------------

class IdentityTransition(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x


class SharedDiagonalTransition(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d))
        self.bias = nn.Parameter(torch.zeros(d))

    def forward(self, x: Tensor) -> Tensor:
        return x * self.scale + self.bias


class SharedPolynomialTransition(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.a = nn.Parameter(torch.ones(d))
        self.b = nn.Parameter(torch.zeros(d))
        self.c = nn.Parameter(torch.zeros(d))

    def forward(self, x: Tensor) -> Tensor:
        return self.a * x + self.b * x.square() + self.c


class SharedLowRankResidual(nn.Module):
    """One shared low-rank residual transition, applied repeatedly."""
    def __init__(self, d: int, rank: int):
        super().__init__()
        self.down = nn.Linear(d, rank, bias=False)
        self.up = nn.Linear(rank, d, bias=True)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.up(self.down(x))


class SharedPrimitiveTransition(nn.Module):
    """Learn a low-complexity mixture of fixed elementwise primitives."""
    def __init__(self, d: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.beta = nn.Parameter(torch.tensor(0.0))
        self.gamma = nn.Parameter(torch.tensor(0.0))
        self.scale = nn.Parameter(torch.ones(d))
        self.bias = nn.Parameter(torch.zeros(d))

    def forward(self, x: Tensor) -> Tensor:
        base = x * self.scale + self.bias
        return base + self.alpha * torch.tanh(x) + self.beta * torch.abs(x) + self.gamma * x.square()


class SharedMLPControl(nn.Module):
    """Control: a small learned transition reused at every trajectory step."""
    def __init__(self, d: int, bottleneck: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, bottleneck), nn.GELU(), nn.Linear(bottleneck, d))

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def build_operator(name: str, d: int, rank: int, bottleneck: int) -> nn.Module:
    if name == "identity":
        return IdentityTransition()
    if name == "shared_diagonal":
        return SharedDiagonalTransition(d)
    if name == "shared_polynomial":
        return SharedPolynomialTransition(d)
    if name == "shared_low_rank_residual":
        return SharedLowRankResidual(d, rank)
    if name == "shared_primitive":
        return SharedPrimitiveTransition(d)
    if name == "shared_mlp":
        return SharedMLPControl(d, bottleneck)
    raise ValueError(name)


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def operator_macs(module: nn.Module) -> int:
    if isinstance(module, IdentityTransition):
        return 0
    if isinstance(module, SharedDiagonalTransition):
        return module.scale.numel()
    if isinstance(module, SharedPolynomialTransition):
        return 2 * module.a.numel()
    if isinstance(module, SharedLowRankResidual):
        return module.down.in_features * module.down.out_features + module.up.in_features * module.up.out_features
    if isinstance(module, SharedPrimitiveTransition):
        return 3 * module.scale.numel()
    if isinstance(module, SharedMLPControl):
        return sum(m.in_features * m.out_features for m in module.net if isinstance(m, nn.Linear))
    raise TypeError(type(module))


class SharedTrajectoryStep(nn.Module):
    """One step of a trajectory replacement; multiple blocks share one operator."""
    def __init__(self, operator: nn.Module):
        super().__init__()
        self.operator = operator

    def forward(self, x: Tensor) -> Tensor:
        return self.operator(x)


def install_trajectory(model: TinyTransformer, start: int, end: int, operator: nn.Module) -> None:
    steps = end - start
    if steps < 2:
        raise ValueError("Trajectory span must contain at least two blocks.")
    # Preserve the original block count. Each step references the SAME operator
    # instance, so the transition is reusable/shared rather than three separate models.
    shared_steps = [SharedTrajectoryStep(operator) for _ in range(steps)]
    original = list(model.blocks)
    model.blocks = nn.ModuleList(original[:start] + shared_steps + original[end:])


def trajectory_span_macs(op: nn.Module, steps: int) -> int:
    return steps * operator_macs(op)


@dataclass
class Eval:
    accuracy: float
    loss: float
    params: int
    trajectory_params: int
    trajectory_macs: int


def current_trajectory_metrics(model: TinyTransformer) -> tuple[int, int]:
    for block in model.blocks:
        if isinstance(block, SharedTrajectoryStep):
            steps = sum(isinstance(b, SharedTrajectoryStep) for b in model.blocks)
            return count_params(block.operator), trajectory_span_macs(block.operator, steps)
    return 0, 0


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Eval:
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
    tparams, tmacs = current_trajectory_metrics(model)
    return Eval(correct / max(total, 1), loss_sum / max(total, 1), count_params(model), tparams, tmacs)


def train_task(model: nn.Module, loader: DataLoader, device: torch.device, steps: int, lr: float) -> float:
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("No trainable parameters.")
    model.train()
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
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
        loss = criterion(logits, y)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite loss")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter() - start


# ---------------------------------------------------------------------------
# Trajectory capture
# ---------------------------------------------------------------------------

def collect_trajectory(model: TinyTransformer, loader: DataLoader, device: torch.device, start: int, end: int, max_batches: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Collect full token trajectories at the entry of start and outputs of each block."""
    states = [[], [], [], []]
    handles = []

    def in_hook(_m, inputs):
        states[0].append(inputs[0].detach().cpu())

    handles.append(model.blocks[start].register_forward_pre_hook(in_hook))
    for j in range(start, end):
        idx = j - start + 1
        handles.append(model.blocks[j].register_forward_hook(lambda _m, _i, out, idx=idx: states[idx].append(out.detach().cpu())))

    try:
        model.eval()
        with torch.no_grad():
            for bi, (x, _y) in enumerate(loader):
                if bi >= max_batches:
                    break
                model(x.to(device, non_blocking=True))
    finally:
        for h in handles:
            h.remove()

    if not all(states):
        raise RuntimeError("Trajectory capture failed")
    return tuple(torch.cat(s, dim=0) for s in states)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Candidate fitting: downstream behaviour + trajectory regularizer
# ---------------------------------------------------------------------------

def fit_candidate(
    base_model: TinyTransformer,
    candidate: nn.Module,
    train_loader: DataLoader,
    trajectory_states: tuple[Tensor, Tensor, Tensor, Tensor],
    device: torch.device,
    start: int,
    end: int,
    steps: int,
    lr: float,
    trajectory_weight: float,
) -> tuple[nn.Module, float]:
    model = copy.deepcopy(base_model).to(device)
    for p in model.parameters():
        p.requires_grad = False
    candidate = candidate.to(device)
    install_trajectory(model, start, end, candidate)
    trainable = [p for p in candidate.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-4) if trainable else None
    criterion = nn.CrossEntropyLoss()
    it = iter(train_loader)

    ts = [s.to(device) for s in trajectory_states]
    if len(ts) != 4:
        raise RuntimeError("Expected three transitions")

    model.train()
    if device.type == "cuda":
        torch.cuda.synchronize()
    start_time = time.perf_counter()
    for step in range(steps):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(train_loader)
            x, y = next(it)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        task_loss = criterion(logits, y)

        # Reuse a deterministic slice of captured trajectories. The same O is
        # asked to explain all three transitions, which is the core DART-0.6
        # hypothesis.
        n = min(128, ts[0].shape[0])
        idx0 = (step * 128) % max(ts[0].shape[0], 1)
        idx = torch.arange(n, device=device) + idx0
        idx %= ts[0].shape[0]
        z = ts[0][idx]
        traj_loss = 0.0
        for k in range(3):
            z = candidate(z)
            traj_loss = traj_loss + torch.mean((z - ts[k + 1][idx]) ** 2)
        loss = task_loss + trajectory_weight * traj_loss
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite candidate loss")
        if opt is not None:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return candidate, time.perf_counter() - start_time


@torch.no_grad()
def trajectory_consistency(candidate: nn.Module, states: tuple[Tensor, Tensor, Tensor, Tensor], max_rows: int) -> float:
    n = min(max_rows, states[0].shape[0])
    z = states[0][:n]
    mses = []
    for k in range(3):
        z = candidate(z)
        mses.append(float(torch.mean((z - states[k + 1][:n]) ** 2)))
    scale = float(torch.mean(states[3][:n] ** 2)) + 1e-8
    return math.exp(-sum(mses) / (3.0 * scale))


@dataclass
class CandidateResult:
    name: str
    params: int
    macs: int
    train_seconds: float
    downstream_accuracy: float
    downstream_loss: float
    trajectory_consistency: float
    score: float


# ---------------------------------------------------------------------------
# Independent verifier
# ---------------------------------------------------------------------------

def verify_candidate(
    base_model: TinyTransformer,
    candidate: nn.Module,
    verifier_loader: DataLoader,
    verifier_states: tuple[Tensor, Tensor, Tensor, Tensor],
    device: torch.device,
    start: int,
    end: int,
) -> tuple[float, float, float]:
    model = copy.deepcopy(base_model).to(device)
    for p in model.parameters():
        p.requires_grad = False
    install_trajectory(model, start, end, copy.deepcopy(candidate).to(device))
    ev = evaluate(model, verifier_loader, device)
    consistency = trajectory_consistency(candidate.cpu(), tuple(s.cpu() for s in verifier_states), 1024)
    return ev.accuracy, ev.loss, consistency


def search_candidates(
    base_model: TinyTransformer,
    train_loader: DataLoader,
    verifier_loader: DataLoader,
    train_states: tuple[Tensor, Tensor, Tensor, Tensor],
    verifier_states: tuple[Tensor, Tensor, Tensor, Tensor],
    device: torch.device,
    start: int,
    end: int,
    d_model: int,
    rank: int,
    bottleneck: int,
    names: Iterable[str],
    fit_steps: int,
    lr: float,
    trajectory_weight: float,
    complexity_lambda: float,
    seed: int,
) -> list[CandidateResult]:
    results = []
    for i, name in enumerate(names):
        seed_everything(seed + 1009 * (i + 1))
        cand = build_operator(name, d_model, rank, bottleneck)
        cand, train_seconds = fit_candidate(
            base_model, cand, train_loader, train_states, device,
            start, end, fit_steps, lr, trajectory_weight
        )
        va, vl, tc = verify_candidate(
            base_model, cand, verifier_loader, verifier_states, device, start, end
        )
        complexity = math.log1p(count_params(cand)) + 0.5 * math.log1p(3 * operator_macs(cand))
        score = va + 0.10 * tc - complexity_lambda * complexity
        results.append(CandidateResult(name, count_params(cand), 3 * operator_macs(cand), train_seconds, va, vl, tc, score))
    return sorted(results, key=lambda r: r.score, reverse=True)


@torch.no_grad()
def cuda_latency(model: nn.Module, loader: DataLoader, device: torch.device, warmup: int, iterations: int) -> Optional[float]:
    if device.type != "cuda":
        return None
    x = next(iter(loader))[0].to(device, non_blocking=True)
    model.eval()
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
    return start.elapsed_time(end) / iterations


def run_one(args, seed: int, task: str) -> dict:
    seed_everything(seed)
    device = torch.device(args.device)
    train_loader = DataLoader(TaskDataset(args.train_size, task, seed), batch_size=args.batch_size, shuffle=True, pin_memory=device.type == "cuda")
    verifier_loader = DataLoader(TaskDataset(args.verifier_size, task, seed + 10000), batch_size=args.batch_size, shuffle=False, pin_memory=device.type == "cuda")
    test_loader = DataLoader(TaskDataset(args.test_size, task, seed + 20000), batch_size=args.batch_size, shuffle=False, pin_memory=device.type == "cuda")

    teacher = TinyTransformer(len(VOCAB), args.d_model, args.heads, args.d_ff, depth=args.depth).to(device)
    train_seconds = train_task(teacher, train_loader, device, args.teacher_steps, args.lr)
    teacher_eval = evaluate(teacher, test_loader, device)

    train_states = collect_trajectory(teacher, train_loader, device, args.trajectory_start, args.trajectory_end, args.trajectory_batches)
    verifier_states = collect_trajectory(teacher, verifier_loader, device, args.trajectory_start, args.trajectory_end, args.verifier_batches)

    candidate_names = [
        "identity",
        "shared_diagonal",
        "shared_polynomial",
        "shared_low_rank_residual",
        "shared_primitive",
        "shared_mlp",  # explicit neural control
    ]

    dart = copy.deepcopy(teacher).to(device)
    rounds = []
    for round_idx in range(args.surgery_rounds):
        states_for_search = collect_trajectory(dart, train_loader, device, args.trajectory_start, args.trajectory_end, args.trajectory_batches)
        verifier_states_now = collect_trajectory(dart, verifier_loader, device, args.trajectory_start, args.trajectory_end, args.verifier_batches)
        results = search_candidates(
            dart, train_loader, verifier_loader, states_for_search, verifier_states_now,
            device, args.trajectory_start, args.trajectory_end,
            args.d_model, args.rank, args.bottleneck, candidate_names,
            args.operator_fit_steps, args.operator_fit_lr, args.trajectory_weight,
            args.complexity_lambda, seed + 40000 * (round_idx + 1),
        )
        winner = results[0]
        seed_everything(seed + 90000 + round_idx)
        op = build_operator(winner.name, args.d_model, args.rank, args.bottleneck)
        op, refit_seconds = fit_candidate(
            dart, op, train_loader, states_for_search, device,
            args.trajectory_start, args.trajectory_end,
            args.operator_fit_steps, args.operator_fit_lr, args.trajectory_weight,
        )
        install_trajectory(dart, args.trajectory_start, args.trajectory_end, copy.deepcopy(op).to(device))
        pre = evaluate(dart, test_loader, device)
        adapt_seconds = train_task(dart, train_loader, device, args.adaptation_steps_per_round, args.lr)
        post = evaluate(dart, test_loader, device)
        latency = cuda_latency(dart, test_loader, device, args.latency_warmup, args.latency_iters)
        rounds.append({
            "round": round_idx,
            "winner": asdict(winner),
            "all_candidates": [asdict(r) for r in results],
            "pre_adapt": asdict(pre),
            "post_adapt": asdict(post),
            "refit_seconds": refit_seconds,
            "adaptation_seconds": adapt_seconds,
            "latency_ms": latency,
        })
        print(f"round={round_idx} winner={winner.name} score={winner.score:.4f} downstream={winner.downstream_accuracy:.4f} trajectory_consistency={winner.trajectory_consistency:.4f} pre={pre.accuracy:.4f} post={post.accuracy:.4f}", flush=True)

    return {
        "seed": seed,
        "task": task,
        "teacher": asdict(teacher_eval),
        "teacher_training_seconds": train_seconds,
        "dart_final": asdict(evaluate(dart, test_loader, device)),
        "rounds": rounds,
    }


def run_transfer(args, seed: int, source_task: str, target_task: str) -> dict:
    device = torch.device(args.device)
    source_loader = DataLoader(TaskDataset(args.train_size, source_task, seed), batch_size=args.batch_size, shuffle=True, pin_memory=device.type == "cuda")
    target_loader = DataLoader(TaskDataset(args.train_size, target_task, seed + 20000), batch_size=args.batch_size, shuffle=True, pin_memory=device.type == "cuda")
    target_test_loader = DataLoader(TaskDataset(args.test_size, target_task, seed + 30000), batch_size=args.batch_size, shuffle=False, pin_memory=device.type == "cuda")

    source = TinyTransformer(len(VOCAB), args.d_model, args.heads, args.d_ff, depth=args.depth).to(device)
    train_task(source, source_loader, device, args.teacher_steps, args.lr)
    source_states = collect_trajectory(source, source_loader, device, args.trajectory_start, args.trajectory_end, args.trajectory_batches)
    verifier_states = source_states
    names = ["identity", "shared_diagonal", "shared_polynomial", "shared_low_rank_residual", "shared_primitive", "shared_mlp"]
    results = search_candidates(source, source_loader, source_loader, source_states, verifier_states, device, args.trajectory_start, args.trajectory_end, args.d_model, args.rank, args.bottleneck, names, args.operator_fit_steps, args.operator_fit_lr, args.trajectory_weight, args.complexity_lambda, seed + 7000)
    winner = results[0]
    op = build_operator(winner.name, args.d_model, args.rank, args.bottleneck)
    op, _ = fit_candidate(source, op, source_loader, source_states, device, args.trajectory_start, args.trajectory_end, args.operator_fit_steps, args.operator_fit_lr, args.trajectory_weight)

    dart = copy.deepcopy(source).to(device)
    install_trajectory(dart, args.trajectory_start, args.trajectory_end, copy.deepcopy(op).to(device))
    scratch = TinyTransformer(len(VOCAB), args.d_model, args.heads, args.d_ff, depth=args.depth).to(device)
    before_s = evaluate(scratch, target_test_loader, device)
    before_d = evaluate(dart, target_test_loader, device)
    train_task(scratch, target_loader, device, args.transfer_adaptation_steps, args.lr)
    train_task(dart, target_loader, device, args.transfer_adaptation_steps, args.lr)
    after_s = evaluate(scratch, target_test_loader, device)
    after_d = evaluate(dart, target_test_loader, device)
    return {
        "seed": seed,
        "source_task": source_task,
        "target_task": target_task,
        "winner": asdict(winner),
        "scratch_after": asdict(after_s),
        "dart_after": asdict(after_d),
        "transfer_gain_points": 100.0 * (after_d.accuracy - after_s.accuracy),
    }


def mean_std(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "std": 0.0}
    return {"mean": statistics.mean(values), "std": statistics.stdev(values) if len(values) > 1 else 0.0}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", nargs="+", type=int, default=[1, 2])
    p.add_argument("--tasks", nargs="+", default=["add", "compose"])
    p.add_argument("--transfer-pairs", nargs=2, action="append", metavar=("SOURCE", "TARGET"), default=[["add", "compose"], ["mul", "sub"]])
    p.add_argument("--train-size", type=int, default=6000)
    p.add_argument("--verifier-size", type=int, default=1500)
    p.add_argument("--test-size", type=int, default=1500)
    p.add_argument("--teacher-steps", type=int, default=800)
    p.add_argument("--operator-fit-steps", type=int, default=300)
    p.add_argument("--adaptation-steps-per-round", type=int, default=400)
    p.add_argument("--surgery-rounds", type=int, default=2)
    p.add_argument("--transfer-adaptation-steps", type=int, default=400)
    p.add_argument("--d-model", type=int, default=32)
    p.add_argument("--heads", type=int, default=2)
    p.add_argument("--d-ff", type=int, default=128)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--bottleneck", type=int, default=32)
    p.add_argument("--trajectory-start", type=int, default=0)
    p.add_argument("--trajectory-end", type=int, default=3)
    p.add_argument("--trajectory-batches", type=int, default=20)
    p.add_argument("--verifier-batches", type=int, default=20)
    p.add_argument("--trajectory-weight", type=float, default=0.05)
    p.add_argument("--operator-fit-lr", type=float, default=1e-3)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--complexity-lambda", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--latency-warmup", type=int, default=30)
    p.add_argument("--latency-iters", type=int, default=30)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default="dart06_results.json")
    args = p.parse_args()

    records = []
    for task in args.tasks:
        print(f"\n===== TASK {task} =====", flush=True)
        for seed in args.seeds:
            print(f"seed={seed}", flush=True)
            records.append(run_one(args, seed, task))

    transfer_records = []
    print("\n===== TRANSFER =====", flush=True)
    for source, target in args.transfer_pairs:
        print(f"{source} -> {target}", flush=True)
        for seed in args.seeds:
            rec = run_transfer(args, seed, source, target)
            transfer_records.append(rec)
            print(f" seed={seed} winner={rec['winner']['name']} scratch={rec['scratch_after']['accuracy']:.4f} dart={rec['dart_after']['accuracy']:.4f} gain={rec['transfer_gain_points']:+.2f} pts", flush=True)

    winners = {}
    for r in records:
        for rr in r["rounds"]:
            n = rr["winner"]["name"]
            winners[n] = winners.get(n, 0) + 1

    transfer_summary = {}
    for source, target in args.transfer_pairs:
        rows = [r for r in transfer_records if r["source_task"] == source and r["target_task"] == target]
        transfer_summary[f"{source}->{target}"] = {
            "scratch_after": mean_std([r["scratch_after"]["accuracy"] for r in rows]),
            "dart_after": mean_std([r["dart_after"]["accuracy"] for r in rows]),
            "gain_points": mean_std([r["transfer_gain_points"] for r in rows]),
        }

    output = {
        "config": vars(args),
        "records": records,
        "transfer_records": transfer_records,
        "summary": {
            "teacher": mean_std([r["teacher"]["accuracy"] for r in records]),
            "dart_final": mean_std([r["dart_final"]["accuracy"] for r in records]),
            "winner_frequency": winners,
            "transfer": transfer_summary,
        },
    }
    Path(args.out).write_text(json.dumps(output, indent=2), encoding="utf-8")
    print("\n================ DART-0.6 SUMMARY ================")
    print(output["summary"])
    print(f"Saved: {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
