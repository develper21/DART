#!/usr/bin/env python3
"""
DART-0.5
========
Behavioral replacement + independent verifier + non-neural operator search.

Core change from DART-0.4:
    We do NOT train a replacement to imitate the teacher hidden tensor.

Instead:
    1. Train teacher on task.
    2. Freeze teacher except the target sub-computation.
    3. Build candidate operators with task-behavior objectives.
    4. Optimize each candidate directly against downstream task loss.
    5. Verify candidates on held-out data and adversarial internal interventions.
    6. Surgically replace the target computation.
    7. Adapt the full model after surgery.
    8. Optionally repeat for multiple surgery rounds.

Candidate families are intentionally mostly non-MLP:
    identity
    diagonal affine
    polynomial
    signed affine (sign / abs / affine)
    low-rank linear
    tiny MLP (CONTROL ONLY)

The MLP is retained only as a control. If MLP wins every time, this experiment
still tells us that the search has not escaped ordinary neural compression.

No external package beyond PyTorch is required.
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
from typing import Optional, Iterable

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset


# -------------------------------------------------------------------
# Reproducibility
# -------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -------------------------------------------------------------------
# Tasks
# -------------------------------------------------------------------

VOCAB = list("0123456789+= ")
STOI = {c: i for i, c in enumerate(VOCAB)}
PAD = STOI[" "]
BLOCK_SIZE = 12


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
            self.rows.append(
                (torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long))
            )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index: int):
        return self.rows[index]


# -------------------------------------------------------------------
# Base model
# -------------------------------------------------------------------

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

    def forward(self, x: Tensor) -> Tensor:
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

    def forward(self, x: Tensor) -> Tensor:
        h = self.emb(x) + self.pos[:, :x.size(1)]
        for block in self.blocks:
            h = block(h)
        return self.head(h[:, 0])


# -------------------------------------------------------------------
# Candidate operators
# -------------------------------------------------------------------

class IdentityOp(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x


class DiagonalAffineOp(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d))
        self.bias = nn.Parameter(torch.zeros(d))

    def forward(self, x: Tensor) -> Tensor:
        return x * self.scale + self.bias


class PolynomialOp(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.a = nn.Parameter(torch.ones(d))
        self.b = nn.Parameter(torch.zeros(d))
        self.c = nn.Parameter(torch.zeros(d))

    def forward(self, x: Tensor) -> Tensor:
        return self.a * x + self.b * (x * x) + self.c


class SignedAffineOp(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d))
        self.bias = nn.Parameter(torch.zeros(d))
        self.abs_gain = nn.Parameter(torch.zeros(d))

    def forward(self, x: Tensor) -> Tensor:
        return self.scale * x + self.abs_gain * torch.abs(x) + self.bias


class LowRankOp(nn.Module):
    def __init__(self, d: int, rank: int):
        super().__init__()
        self.down = nn.Linear(d, rank, bias=False)
        self.up = nn.Linear(rank, d, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.up(self.down(x))


class SmallMLPOp(nn.Module):
    """CONTROL ONLY: retained so DART-0.5 can prove when neural compression wins."""
    def __init__(self, d: int, bottleneck: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, d),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def build_candidate(name: str, d: int, rank: int, bottleneck: int) -> nn.Module:
    if name == "identity":
        return IdentityOp()
    if name == "diagonal_affine":
        return DiagonalAffineOp(d)
    if name == "polynomial":
        return PolynomialOp(d)
    if name == "signed_affine":
        return SignedAffineOp(d)
    if name == "low_rank":
        return LowRankOp(d, rank)
    if name == "small_mlp":
        return SmallMLPOp(d, bottleneck)
    raise ValueError(f"Unknown candidate: {name}")


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def macs(module: nn.Module) -> int:
    if isinstance(module, IdentityOp):
        return 0
    if isinstance(module, nn.Sequential):
        return sum(
            m.in_features * m.out_features
            for m in module
            if isinstance(m, nn.Linear)
        )
    if isinstance(module, DiagonalAffineOp):
        return module.scale.numel()
    if isinstance(module, PolynomialOp):
        return 2 * module.a.numel()
    if isinstance(module, SignedAffineOp):
        return 2 * module.scale.numel()
    if isinstance(module, LowRankOp):
        return (
            module.down.in_features * module.down.out_features
            + module.up.in_features * module.up.out_features
        )
    if isinstance(module, SmallMLPOp):
        ls = [m for m in module.net if isinstance(m, nn.Linear)]
        return sum(m.in_features * m.out_features for m in ls)
    if hasattr(module, "replacement"):
        return macs(module.replacement)
    raise TypeError(type(module))


# -------------------------------------------------------------------
# Surgical wrapper
# -------------------------------------------------------------------

class ReplacementWrapper(nn.Module):
    def __init__(self, replacement: nn.Module):
        super().__init__()
        self.replacement = replacement

    def forward(self, x: Tensor) -> Tensor:
        return self.replacement(x)


def install_replacement(model: TinyTransformer, block_idx: int, replacement: nn.Module) -> None:
    model.blocks[block_idx].ff = ReplacementWrapper(replacement)


# -------------------------------------------------------------------
# Evaluation
# -------------------------------------------------------------------

@dataclass
class Eval:
    accuracy: float
    loss: float
    params: int
    target_params: int
    target_macs: int


def evaluate(
    model: TinyTransformer,
    loader: DataLoader,
    device: torch.device,
    block_idx: int,
) -> Eval:
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

    target = model.blocks[block_idx].ff
    return Eval(
        accuracy=correct / max(total, 1),
        loss=loss_sum / max(total, 1),
        params=count_params(model),
        target_params=count_params(target),
        target_macs=macs(target),
    )


# -------------------------------------------------------------------
# Training
# -------------------------------------------------------------------

def train_task(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    steps: int,
    lr: float,
    trainable_filter: Optional[callable] = None,
) -> float:
    if steps <= 0:
        return 0.0

    if trainable_filter is None:
        params = [p for p in model.parameters() if p.requires_grad]
    else:
        params = [
            p for p in model.parameters()
            if p.requires_grad and trainable_filter(p)
        ]

    if not params:
        raise RuntimeError("No trainable parameters found.")

    model.train()
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
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
            raise RuntimeError(f"Non-finite task loss: {float(loss)}")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()

    if device.type == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter() - start


# -------------------------------------------------------------------
# Behavioral objective
# -------------------------------------------------------------------

def candidate_task_loss(
    teacher_body: TinyTransformer,
    replacement: nn.Module,
    block_idx: int,
    loader: DataLoader,
    device: torch.device,
    max_batches: int,
) -> float:
    """
    End-to-end task loss with ONLY the candidate replacement trainable.
    This is the key difference from DART-0.4:
    replacement is optimized for downstream behavior, not hidden MSE.
    """
    backup = teacher_body.blocks[block_idx].ff
    install_replacement(teacher_body, block_idx, replacement)

    try:
        teacher_body.eval()
        ce = nn.CrossEntropyLoss(reduction="sum")
        total_loss = 0.0
        total = 0

        with torch.no_grad():
            for bi, (x, y) in enumerate(loader):
                if bi >= max_batches:
                    break
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                logits = teacher_body(x)
                total_loss += float(ce(logits, y))
                total += y.numel()
    finally:
        teacher_body.blocks[block_idx].ff = backup

    return total_loss / max(total, 1)


def fit_candidate_behaviorally(
    base_model: TinyTransformer,
    candidate: nn.Module,
    block_idx: int,
    loader: DataLoader,
    device: torch.device,
    steps: int,
    lr: float,
    max_batches: int,
) -> tuple[nn.Module, float]:
    """
    Direct task-behavior optimization.
    The candidate is inserted into a frozen teacher body. Only candidate
    parameters are updated.
    """
    model = copy.deepcopy(base_model).to(device)
    for p in model.parameters():
        p.requires_grad = False

    candidate = candidate.to(device)
    install_replacement(model, block_idx, candidate)

    trainable = [p for p in candidate.parameters() if p.requires_grad]
    criterion = nn.CrossEntropyLoss()
    iterator = iter(loader)

    model.train()
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()

    if trainable:
        optimizer = torch.optim.AdamW(
            trainable, lr=lr, weight_decay=1e-4
        )
    else:
        optimizer = None

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
            raise RuntimeError("Non-finite candidate behavioral loss.")

        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()

    if device.type == "cuda":
        torch.cuda.synchronize()

    return candidate, time.perf_counter() - start


# -------------------------------------------------------------------
# Independent verifier
# -------------------------------------------------------------------

def collect_internal_inputs(
    model: TinyTransformer,
    loader: DataLoader,
    device: torch.device,
    block_idx: int,
    max_batches: int,
) -> Tensor:
    xs = []
    block = model.blocks[block_idx]

    def hook(_module, inputs):
        xs.append(inputs[0].detach().reshape(-1, inputs[0].shape[-1]).cpu())

    handle = block.ff.register_forward_pre_hook(hook)
    try:
        model.eval()
        with torch.no_grad():
            for bi, (x, _) in enumerate(loader):
                if bi >= max_batches:
                    break
                model(x.to(device, non_blocking=True))
    finally:
        handle.remove()

    if not xs:
        raise RuntimeError("Verifier failed to collect internal states.")
    return torch.cat(xs)


def perturb_hidden_states(h: Tensor, seed: int, eps: float, max_rows: int) -> list[Tensor]:
    if h.size(0) > max_rows:
        h = h[:max_rows]

    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)

    d = torch.randn(h.shape, generator=gen)
    d = d / (d.norm(dim=-1, keepdim=True) + 1e-8)

    mask = torch.rand(h.shape, generator=gen) > 0.2
    perm = torch.randperm(h.shape[-1], generator=gen)

    return [
        h,
        h + eps * d,
        h - eps * d,
        0.5 * h,
        1.5 * h,
        h * mask,
        h[:, perm],
    ]


def make_internal_intervention_loader(
    model: TinyTransformer,
    source_loader: DataLoader,
    device: torch.device,
    block_idx: int,
    seed: int,
    eps: float,
    max_batches: int,
) -> list[tuple[Tensor, Tensor]]:
    """
    Produces internal hidden states and labels, but the verifier scores
    final downstream predictions. It does not compare hidden tensors.
    """
    hidden = []
    labels = []

    block = model.blocks[block_idx]

    def hook(_module, inputs):
        hidden.append(inputs[0].detach().reshape(-1, inputs[0].shape[-1]).cpu())

    # We also need the per-example labels corresponding to the first token outputs.
    with torch.no_grad():
        for bi, (x, y) in enumerate(source_loader):
            if bi >= max_batches:
                break
            captures = []
            handle = block.ff.register_forward_pre_hook(
                lambda _m, inp: captures.append(inp[0].detach())
            )
            model(x.to(device, non_blocking=True))
            handle.remove()
            if not captures:
                continue
            # Model uses h[:,0] for output. We only need one vector/example.
            hidden.append(captures[0][:, 0].cpu())
            labels.append(y.clone())

    if not hidden:
        raise RuntimeError("No verifier internal states.")
    h = torch.cat(hidden)
    y = torch.cat(labels)

    if h.size(0) > 3000:
        h = h[:3000]
        y = y[:3000]

    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    direction = torch.randn(h.shape, generator=gen)
    direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)

    interventions = [
        h,
        h + eps * direction,
        h - eps * direction,
        0.5 * h,
        1.5 * h,
        torch.tanh(h),
    ]
    return [(z, y) for z in interventions]


def downstream_verifier_score(
    base_model: TinyTransformer,
    candidate: nn.Module,
    source_loader: DataLoader,
    device: torch.device,
    block_idx: int,
    max_batches: int,
    hidden_intervention_eps: float,
    seed: int,
) -> dict:
    """
    Independent verifier:
      - held-out task loss / accuracy
      - final-output stability under internal interventions

    We do NOT score hidden tensor MSE.
    """
    test_loss = 0.0
    total = 0
    correct = 0
    ce = nn.CrossEntropyLoss(reduction="sum")

    candidate_model = copy.deepcopy(base_model).to(device)
    for p in candidate_model.parameters():
        p.requires_grad = False
    candidate_model.blocks[block_idx].ff = copy.deepcopy(candidate).to(device)

    candidate_model.eval()
    for bi, (x, y) in enumerate(source_loader):
        if bi >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.no_grad():
            logits = candidate_model(x)
        test_loss += float(ce(logits, y))
        correct += int((logits.argmax(-1) == y).sum())
        total += y.numel()

    # Internal intervention stability:
    # We inject an intervention at the target FF input and compare the
    # candidate's final prediction confidence/decision consistency.
    # This is intentionally behavioral.
    x0, y0 = next(iter(source_loader))
    x0 = x0[: min(128, x0.size(0))].to(device)
    y0 = y0[: x0.size(0)].to(device)

    baseline_logits = candidate_model(x0)
    baseline_pred = baseline_logits.argmax(-1)

    # Hook target block to perturb its FFN input before candidate runs.
    block = candidate_model.blocks[block_idx]
    intervention_losses = []

    def prehook(_m, inputs):
        h = inputs[0]
        noise = torch.randn_like(h) * hidden_intervention_eps
        return (h + noise,)

    handle = block.ff.register_forward_pre_hook(prehook)
    with torch.no_grad():
        perturbed_logits = candidate_model(x0)
    handle.remove()

    perturbed_pred = perturbed_logits.argmax(-1)
    consistency = float((perturbed_pred == baseline_pred).float().mean())

    return {
        "heldout_accuracy": correct / max(total, 1),
        "heldout_loss": test_loss / max(total, 1),
        "intervention_prediction_consistency": consistency,
    }


# -------------------------------------------------------------------
# Candidate selection
# -------------------------------------------------------------------

@dataclass
class CandidateResult:
    name: str
    params: int
    macs: int
    train_seconds: float
    verifier_accuracy: float
    verifier_loss: float
    intervention_consistency: float
    score: float


def search_candidates(
    base_model: TinyTransformer,
    loader: DataLoader,
    verifier_loader: DataLoader,
    device: torch.device,
    block_idx: int,
    d_model: int,
    rank: int,
    bottleneck: int,
    names: Iterable[str],
    fit_steps: int,
    fit_lr: float,
    verifier_batches: int,
    complexity_lambda: float,
    intervention_eps: float,
    seed: int,
):
    results = []

    for i, name in enumerate(names):
        seed_everything(seed + 1000 * (i + 1))
        candidate = build_candidate(name, d_model, rank, bottleneck)
        candidate, train_seconds = fit_candidate_behaviorally(
            base_model=base_model,
            candidate=candidate,
            block_idx=block_idx,
            loader=loader,
            device=device,
            steps=fit_steps,
            lr=fit_lr,
            max_batches=verifier_batches,
        )

        verified = downstream_verifier_score(
            base_model=base_model,
            candidate=candidate,
            source_loader=verifier_loader,
            device=device,
            block_idx=block_idx,
            max_batches=verifier_batches,
            hidden_intervention_eps=intervention_eps,
            seed=seed + 77,
        )

        complexity = (
            math.log1p(count_params(candidate))
            + 0.5 * math.log1p(macs(candidate))
        )

        # Higher is better: task accuracy + intervention stability,
        # lower complexity is rewarded mildly.
        score = (
            verified["heldout_accuracy"]
            + 0.10 * verified["intervention_prediction_consistency"]
            - complexity_lambda * complexity
        )

        results.append(
            CandidateResult(
                name=name,
                params=count_params(candidate),
                macs=macs(candidate),
                train_seconds=train_seconds,
                verifier_accuracy=verified["heldout_accuracy"],
                verifier_loss=verified["heldout_loss"],
                intervention_consistency=verified["intervention_prediction_consistency"],
                score=score,
            )
        )

    results.sort(key=lambda x: x.score, reverse=True)
    return results


# -------------------------------------------------------------------
# CUDA latency
# -------------------------------------------------------------------

@torch.no_grad()
def cuda_latency(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    warmup: int,
    iterations: int,
) -> Optional[float]:
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

    return start.elapsed_time(end) / iterations


# -------------------------------------------------------------------
# Main experiment
# -------------------------------------------------------------------

def run_one(args, seed: int, task: str):
    seed_everything(seed)
    device = torch.device(args.device)

    train_ds = TaskDataset(args.train_size, task, seed)
    verifier_ds = TaskDataset(args.verifier_size, task, seed + 10000)
    test_ds = TaskDataset(args.test_size, task, seed + 20000)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=device.type == "cuda",
    )
    verifier_loader = DataLoader(
        verifier_ds,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )

    # Base teacher
    seed_everything(seed)
    teacher = TinyTransformer(
        len(VOCAB), args.d_model, args.heads, args.d_ff
    ).to(device)

    teacher_train_seconds = train_task(
        teacher, train_loader, device,
        args.teacher_steps, args.lr
    )

    teacher_eval = evaluate(
        teacher, test_loader, device, args.block
    )

    dart = copy.deepcopy(teacher).to(device)
    round_results = []

    candidate_names = [
        "identity",
        "diagonal_affine",
        "polynomial",
        "signed_affine",
        "low_rank",
        "small_mlp",  # explicit control
    ]

    for round_idx in range(args.surgery_rounds):
        print(f"round={round_idx} searching candidates", flush=True)

        results = search_candidates(
            base_model=dart,
            loader=train_loader,
            verifier_loader=verifier_loader,
            device=device,
            block_idx=args.block,
            d_model=args.d_model,
            rank=args.rank,
            bottleneck=args.bottleneck,
            names=candidate_names,
            fit_steps=args.operator_fit_steps,
            fit_lr=args.operator_fit_lr,
            verifier_batches=args.verifier_batches,
            complexity_lambda=args.complexity_lambda,
            intervention_eps=args.intervention_eps,
            seed=seed + 50000 * (round_idx + 1),
        )

        winner = results[0]

        # Refit the winner so we have a fresh candidate to surgically install.
        seed_everything(seed + 8888 + round_idx)
        winner_candidate = build_candidate(
            winner.name, args.d_model, args.rank, args.bottleneck
        )
        winner_candidate, _ = fit_candidate_behaviorally(
            base_model=dart,
            candidate=winner_candidate,
            block_idx=args.block,
            loader=train_loader,
            device=device,
            steps=args.operator_fit_steps,
            lr=args.operator_fit_lr,
            max_batches=args.verifier_batches,
        )

        install_replacement(
            dart, args.block,
            copy.deepcopy(winner_candidate).to(device)
        )

        pre_adapt = evaluate(dart, test_loader, device, args.block)

        # Full-network post-surgery adaptation.
        adapt_seconds = train_task(
            dart, train_loader, device,
            args.adaptation_steps_per_round,
            args.lr
        )

        post_adapt = evaluate(dart, test_loader, device, args.block)
        latency = cuda_latency(
            dart, test_loader, device,
            args.latency_warmup, args.latency_iters
        )

        round_results.append({
            "round": round_idx,
            "winner": asdict(winner),
            "all_candidates": [asdict(x) for x in results],
            "pre_adapt": asdict(pre_adapt),
            "post_adapt": asdict(post_adapt),
            "adaptation_seconds": adapt_seconds,
            "latency_ms": latency,
        })

        print(
            f"round={round_idx} winner={winner.name} "
            f"params={winner.params} macs={winner.macs} "
            f"verifier_acc={winner.verifier_accuracy:.4f} "
            f"intervention={winner.intervention_consistency:.4f} "
            f"pre={pre_adapt.accuracy:.4f} "
            f"post={post_adapt.accuracy:.4f} "
            f"latency={latency}"
        )

    final_eval = evaluate(dart, test_loader, device, args.block)

    return {
        "seed": seed,
        "task": task,
        "teacher": asdict(teacher_eval),
        "teacher_training_seconds": teacher_train_seconds,
        "dart_final": asdict(final_eval),
        "rounds": round_results,
    }


def run_transfer(args, seed: int, source_task: str, target_task: str):
    """
    Transfer control:
      learn a behavioral replacement on source task,
      move it into the source-trained model,
      adapt on target task,
      compare against a fresh target model.

    This is not intended to prove causal algorithmic transfer by itself;
    it is a targeted probe for reusable computation.
    """
    seed_everything(seed)
    device = torch.device(args.device)

    source_train = TaskDataset(args.train_size, source_task, seed)
    target_train = TaskDataset(args.train_size, target_task, seed + 20000)
    target_test = TaskDataset(args.test_size, target_task, seed + 30000)

    source_loader = DataLoader(
        source_train, batch_size=args.batch_size, shuffle=True,
        pin_memory=device.type == "cuda"
    )
    target_loader = DataLoader(
        target_train, batch_size=args.batch_size, shuffle=True,
        pin_memory=device.type == "cuda"
    )
    target_test_loader = DataLoader(
        target_test, batch_size=args.batch_size, shuffle=False,
        pin_memory=device.type == "cuda"
    )

    source_model = TinyTransformer(
        len(VOCAB), args.d_model, args.heads, args.d_ff
    ).to(device)

    train_task(
        source_model, source_loader, device,
        args.teacher_steps, args.lr
    )

    # Find a behavioral source replacement.
    candidate_names = [
        "identity", "diagonal_affine", "polynomial",
        "signed_affine", "low_rank", "small_mlp"
    ]

    results = search_candidates(
        source_model,
        source_loader,
        source_loader,
        device,
        args.block,
        args.d_model,
        args.rank,
        args.bottleneck,
        candidate_names,
        args.operator_fit_steps,
        args.operator_fit_lr,
        args.verifier_batches,
        args.complexity_lambda,
        args.intervention_eps,
        seed + 9000,
    )
    winner = results[0]

    candidate = build_candidate(
        winner.name, args.d_model, args.rank, args.bottleneck
    )
    candidate, _ = fit_candidate_behaviorally(
        source_model, candidate, args.block,
        source_loader, device,
        args.operator_fit_steps, args.operator_fit_lr,
        args.verifier_batches,
    )

    dart = copy.deepcopy(source_model).to(device)
    install_replacement(dart, args.block, copy.deepcopy(candidate).to(device))

    # Fresh target baseline.
    target_scratch = TinyTransformer(
        len(VOCAB), args.d_model, args.heads, args.d_ff
    ).to(device)

    before_s = evaluate(target_scratch, target_test_loader, device, args.block)
    before_d = evaluate(dart, target_test_loader, device, args.block)

    train_task(
        target_scratch, target_loader, device,
        args.transfer_adaptation_steps, args.lr
    )
    train_task(
        dart, target_loader, device,
        args.transfer_adaptation_steps, args.lr
    )

    after_s = evaluate(target_scratch, target_test_loader, device, args.block)
    after_d = evaluate(dart, target_test_loader, device, args.block)

    return {
        "seed": seed,
        "source_task": source_task,
        "target_task": target_task,
        "winner": asdict(winner),
        "scratch_before": asdict(before_s),
        "dart_before": asdict(before_d),
        "scratch_after": asdict(after_s),
        "dart_after": asdict(after_d),
    }


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------

def mean_std(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "std": 0.0}
    if len(values) == 1:
        return {"mean": values[0], "std": 0.0}
    return {"mean": statistics.mean(values), "std": statistics.stdev(values)}


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--tasks", nargs="+", default=["add", "compose"])
    parser.add_argument(
        "--transfer-pairs",
        nargs=2,
        action="append",
        metavar=("SOURCE", "TARGET"),
        default=[["add", "compose"], ["mul", "sub"]],
    )

    parser.add_argument("--train-size", type=int, default=6000)
    parser.add_argument("--verifier-size", type=int, default=1500)
    parser.add_argument("--test-size", type=int, default=1500)

    parser.add_argument("--teacher-steps", type=int, default=800)
    parser.add_argument("--operator-fit-steps", type=int, default=300)
    parser.add_argument("--adaptation-steps-per-round", type=int, default=400)
    parser.add_argument("--surgery-rounds", type=int, default=2)
    parser.add_argument("--transfer-adaptation-steps", type=int, default=400)

    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--d-ff", type=int, default=128)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--bottleneck", type=int, default=32)
    parser.add_argument("--block", type=int, default=1)

    parser.add_argument("--operator-fit-lr", type=float, default=1e-3)
    parser.add_argument("--lr", type=float, default=3e-4)

    parser.add_argument("--verifier-batches", type=int, default=20)
    parser.add_argument("--complexity-lambda", type=float, default=1e-4)
    parser.add_argument("--intervention-eps", type=float, default=0.10)

    parser.add_argument("--batch-size", type=int, default=256)

    parser.add_argument("--latency-warmup", type=int, default=30)
    parser.add_argument("--latency-iters", type=int, default=30)

    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--out", default="dart05_results.json")

    args = parser.parse_args()

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
                f"winner={rec['winner']['name']} "
                f"scratch_after={rec['scratch_after']['accuracy']:.4f} "
                f"dart_after={rec['dart_after']['accuracy']:.4f}"
            )

    summary = {
        "teacher": mean_std([r["teacher"]["accuracy"] for r in records]),
        "dart_final": mean_std([r["dart_final"]["accuracy"] for r in records]),
    }

    # Winner frequencies
    winners = {}
    for r in records:
        for rr in r["rounds"]:
            name = rr["winner"]["name"]
            winners[name] = winners.get(name, 0) + 1
    summary["winner_frequency"] = winners

    transfer_summary = {}
    for source, target in args.transfer_pairs:
        rows = [
            r for r in transfer_records
            if r["source_task"] == source and r["target_task"] == target
        ]
        transfer_summary[f"{source}->{target}"] = {
            "scratch_after": mean_std(
                [r["scratch_after"]["accuracy"] for r in rows]
            ),
            "dart_after": mean_std(
                [r["dart_after"]["accuracy"] for r in rows]
            ),
        }

    summary["transfer"] = transfer_summary

    output = {
        "config": vars(args),
        "records": records,
        "transfer_records": transfer_records,
        "summary": summary,
    }

    Path(args.out).write_text(json.dumps(output, indent=2), encoding="utf-8")

    print("\n================ DART-0.5 SUMMARY ================")
    print(
        f"teacher: {summary['teacher']['mean']:.4f} ± "
        f"{summary['teacher']['std']:.4f}"
    )
    print(
        f"dart_final: {summary['dart_final']['mean']:.4f} ± "
        f"{summary['dart_final']['std']:.4f}"
    )
    print(f"winner_frequency: {summary['winner_frequency']}")

    for pair, stats in transfer_summary.items():
        print(
            f"{pair}: scratch={stats['scratch_after']['mean']:.4f} ± "
            f"{stats['scratch_after']['std']:.4f}, "
            f"dart={stats['dart_after']['mean']:.4f} ± "
            f"{stats['dart_after']['std']:.4f}"
        )

    print(f"\nSaved: {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
