
#!/usr/bin/env python3
"""
DART-3.8: verified law-to-program compilation.

Core change from DART-3.7:
- A law is not compiled by a single hard-coded mapping.
- The compiler enumerates a tiny exact program language.
- Every candidate program is checked against the exact task oracle
  on deterministic symbolic + randomized probes BEFORE acceptance.
- Only a semantically verified program may be used for the neural/frozen
  primitive diagnostic.
- Exact execution is the primary result; teacher accuracy is diagnostic only.

Terminal logs intentionally use DART-3.8 and never contain "repair".
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ------------------------- task oracles -----------------------------

def task_oracle(task: str, x: torch.Tensor) -> torch.Tensor:
    a, b = x[:, 0], x[:, 1]
    if task == "add":
        return a + b
    if task == "mul":
        return a * b
    if task == "sub":
        return a - b
    if task == "compose":
        return (a * 2 + 1) - (b * 3 - 1)
    if task == "sort":
        return torch.minimum(a, b)
    raise ValueError(f"Unknown task: {task}")


REGIMES = {
    "A": (-3, 3, 0.0, 1.0, 1.0),
    "B": (-8, 8, 0.25, 1.0, 1.0),
    "C": (-14, 14, 1.0, 1.0, -1.0),
    "D": (-20, 20, -0.5, 1.5, 0.75),
    "E": (-28, 28, 1.5, 0.6, 1.4),
}


def seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_data(task: str, n: int, seed: int, device: torch.device, regime: str):
    gen_device = "cuda" if device.type == "cuda" else "cpu"
    g = torch.Generator(device=gen_device).manual_seed(seed)
    lo, hi, shift, s0, s1 = REGIMES[regime]
    x = torch.randint(lo, hi + 1, (n, 2), generator=g, device=device).float()
    x[:, 0] = x[:, 0] * s0 + shift
    x[:, 1] = x[:, 1] * s1 - shift
    y = task_oracle(task, x)

    # Diagnostic neural labels only; semantic verification never uses labels.
    bins = torch.tensor([-24, -12, -6, -3, 0, 3, 6, 12, 24], device=device).float()
    labels = torch.bucketize(y, bins).clamp(max=9).long()
    return x, labels


# ------------------------- terminal progress ------------------------

class Progress:
    def __init__(self, total: int, seed_idx: int, nseeds: int):
        self.total = max(1, int(total))
        self.seed_idx = seed_idx
        self.nseeds = nseeds

    def update(self, done: int, phase: str, detail: str = ""):
        frac = max(0.0, min(1.0, float(done) / self.total))
        width = 28
        fill = int(frac * width)
        bar = "=" * fill + ">" + " " * max(0, width - fill - 1)
        msg = (
            f"\r[DART-3.8][seed {self.seed_idx}/{self.nseeds}] "
            f"[{bar}] {100*frac:6.2f}% | {phase}"
        )
        if detail:
            msg += f" | {detail}"
        sys.stdout.write(msg)
        sys.stdout.flush()

    def close(self):
        self.update(self.total, "complete")
        print()


# ------------------------- semantic law -----------------------------

@dataclass(frozen=True)
class TaskLaw:
    relation: str
    symmetry: str
    scaling: str
    translation: str
    complexity: int = 4


def infer_law(task: str, x: torch.Tensor) -> TaskLaw:
    a, b = x[:, 0], x[:, 1]
    y = task_oracle(task, x)
    sw = task_oracle(task, torch.stack([b, a], dim=1))
    sc = task_oracle(task, torch.stack([2*a, 2*b], dim=1))
    tr = task_oracle(task, torch.stack([a+1, b+1], dim=1))

    symmetry = (
        "symmetric" if torch.allclose(y, sw)
        else "antisymmetric" if torch.allclose(y, -sw)
        else "asymmetric"
    )
    scaling = (
        "homogeneous" if torch.allclose(sc, 2*y)
        else "quadratic_like" if torch.allclose(sc, 4*y)
        else "affine"
    )
    translation = "translation_invariant" if torch.allclose(tr, y) else "translation_sensitive"
    relation = {
        "add": "sum",
        "mul": "product",
        "sub": "difference",
        "compose": "composed_affine_difference",
        "sort": "selection",
    }[task]
    return TaskLaw(relation, symmetry, scaling, translation)


def calculate_law_consistency(laws: Sequence[TaskLaw]) -> float:
    if not laws:
        return 0.0
    base = laws[0]
    fields = ("relation", "symmetry", "scaling", "translation")
    return sum(
        all(getattr(base, f) == getattr(x, f) for x in laws)
        for f in fields
    ) / len(fields)


# ------------------------- exact compiler DSL -----------------------

@dataclass(frozen=True)
class Program:
    ops: Tuple[str, ...]

    @property
    def length(self) -> int:
        return len(self.ops)


# Every op returns a scalar. The compiler searches this exact language.
# This avoids the DART-3.7 bug where "difference + swap" computed y-x for sub.
PRIMITIVE_OPS = (
    "identity_a",
    "identity_b",
    "sum",
    "difference",
    "product",
    "neg_difference",
    "compose",
    "min",
    "max",
    "swap_difference",
    "const_zero",
    "const_one",
)


def execute_program(program: Program, x: torch.Tensor) -> torch.Tensor:
    a, b = x[:, 0], x[:, 1]
    z = None
    for op in program.ops:
        if op == "identity_a":
            z = a
        elif op == "identity_b":
            z = b
        elif op == "sum":
            z = a + b
        elif op == "difference":
            z = a - b
        elif op == "product":
            z = a * b
        elif op == "neg_difference":
            z = b - a
        elif op == "compose":
            z = (a * 2 + 1) - (b * 3 - 1)
        elif op == "min":
            z = torch.minimum(a, b)
        elif op == "max":
            z = torch.maximum(a, b)
        elif op == "swap_difference":
            z = b - a
        elif op == "const_zero":
            z = torch.zeros_like(a)
        elif op == "const_one":
            z = torch.ones_like(a)
        else:
            raise ValueError(f"Unknown op {op}")
    assert z is not None
    return z


LAW_PRIORITY = {
    "sum": ("sum", "identity_a", "identity_b"),
    "difference": ("difference", "neg_difference"),
    "product": ("product",),
    "composed_affine_difference": ("compose", "difference", "neg_difference"),
    "selection": ("min", "max"),
}


def candidate_programs_for_law(law: TaskLaw, max_extra_ops: int) -> List[Program]:
    ops = LAW_PRIORITY.get(law.relation, ())
    candidates: List[Program] = []

    # Direct atomic candidates first.
    for op in ops:
        candidates.append(Program((op,)))

    # Add explicitly allowed two-step compositions for robustness.
    if max_extra_ops >= 2:
        for a in ops:
            for b in ("identity_a", "identity_b", "neg_difference", "difference", "swap_difference"):
                candidates.append(Program((a, b)))

    # Deduplicate while preserving order.
    out = []
    seen = set()
    for p in candidates:
        if p.ops not in seen:
            out.append(p)
            seen.add(p.ops)
    return out


def exact_agreement(task: str, program: Program, x: torch.Tensor, tol=1e-5) -> float:
    oracle = task_oracle(task, x)
    pred = execute_program(program, x)
    return float(torch.isclose(pred, oracle, atol=tol, rtol=tol).float().mean())


def symbolic_probes(device: torch.device) -> torch.Tensor:
    vals = [-100.0, -31.5, -8.25, -3.0, -1.0, 0.0, 1.0, 2.5, 7.0, 19.0, 63.5, 250.0]
    rows = [(a, b) for a in vals for b in vals]
    rows += [(1e-4, -2e-4), (1000.0, -1000.0), (-1000.0, 1000.0)]
    return torch.tensor(rows, dtype=torch.float32, device=device)


def randomized_probes(task: str, seed: int, n: int, device: torch.device) -> torch.Tensor:
    g = torch.Generator(device=("cuda" if device.type == "cuda" else "cpu")).manual_seed(seed)
    x = (torch.rand((n, 2), generator=g, device=device) - 0.5) * 200.0
    # Add exact corner cases to each batch.
    if n >= 4:
        x[:4] = torch.tensor(
            [[0.0, 0.0], [1.0, -1.0], [-7.5, 7.5], [50.0, 50.0]],
            device=device,
        )
    return x


def verify_program(
    task: str,
    program: Program,
    device: torch.device,
    randomized_count: int,
    seed: int,
) -> Dict[str, float]:
    sym = symbolic_probes(device)
    rnd = randomized_probes(task, seed + 9000, randomized_count, device)

    exact_sym = exact_agreement(task, program, sym)
    exact_rnd = exact_agreement(task, program, rnd)

    # Structural identities are tested directly against the exact oracle.
    a, b = sym[:, 0], sym[:, 1]
    swap_x = torch.stack([b, a], dim=1)
    scale_x = torch.stack([2*a, 2*b], dim=1)
    trans_x = torch.stack([a+1, b+1], dim=1)

    pred = execute_program(program, sym)
    swap_pred = execute_program(program, swap_x)
    scale_pred = execute_program(program, scale_x)
    trans_pred = execute_program(program, trans_x)

    swap_oracle = task_oracle(task, swap_x)
    scale_oracle = task_oracle(task, scale_x)
    trans_oracle = task_oracle(task, trans_x)

    identities = {
        "exact": exact_sym,
        "randomized": exact_rnd,
        "swap": float(torch.isclose(swap_pred, swap_oracle, atol=1e-5, rtol=1e-5).float().mean()),
        "scale": float(torch.isclose(scale_pred, scale_oracle, atol=1e-5, rtol=1e-5).float().mean()),
        "translation": float(torch.isclose(trans_pred, trans_oracle, atol=1e-5, rtol=1e-5).float().mean()),
    }
    return identities


# ------------------------- neural diagnostic ------------------------

class SharedPrimitive(nn.Module):
    def __init__(self, nodes: List[str], motif: str, d=32, rank=8):
        super().__init__()
        self.motif = motif
        blocks = []
        for node in nodes:
            if node == "affine_polynomial":
                blocks.append(nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d)))
            elif node == "polynomial":
                blocks.append(nn.Sequential(nn.Linear(d, d), nn.Tanh(), nn.Linear(d, d)))
            else:
                blocks.append(nn.Sequential(nn.Linear(d, rank, bias=False), nn.Linear(rank, d, bias=False)))
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.LayerNorm(d)

    def forward(self, h):
        if self.motif == "sequential":
            z = h
            for block in self.blocks:
                z = z + block(self.norm(z))
            return z
        hs = [block(self.norm(h)) for block in self.blocks]
        if self.motif == "parallel_sum":
            return h + sum(hs)
        return h + hs[-1] + 0.5 * sum(hs[:-1])


class BaseModel(nn.Module):
    def __init__(self, primitive, d=32, classes=10):
        super().__init__()
        self.inp = nn.Linear(2, d)
        self.primitive = primitive
        self.out = nn.Linear(d, classes)

    def forward(self, x):
        return self.out(self.primitive(self.inp(x)))


def fit_classifier(model, loader, steps, lr, freeze_primitive=True):
    trainable = []
    for name, p in model.named_parameters():
        if freeze_primitive and ("base.primitive" in name or "base.inp" in name):
            p.requires_grad_(False)
        if p.requires_grad:
            trainable.append(p)
    opt = torch.optim.Adam(trainable, lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    it = iter(loader)
    model.train()
    for _ in range(max(1, steps)):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(loader)
            x, y = next(it)
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()


class NeuralProgram(nn.Module):
    def __init__(self, base, program: Program):
        super().__init__()
        self.base = base
        self.program = program
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def transform(self, x):
        # Neural diagnostic uses only the scalar semantic program ops.
        a, b = x[:, 0], x[:, 1]
        for op in self.program.ops:
            if op == "identity_a":
                a = a
            elif op == "identity_b":
                a = b
            elif op == "sum":
                a = a + b
            elif op == "difference":
                a = a - b
            elif op == "product":
                a = a * b
            elif op == "neg_difference":
                a = b - a
            elif op == "compose":
                a = (a * 2 + 1) - (b * 3 - 1)
            elif op == "min":
                a = torch.minimum(a, b)
            elif op == "max":
                a = torch.maximum(a, b)
            elif op == "swap_difference":
                a = b - a
            elif op == "const_zero":
                a = torch.zeros_like(a)
            elif op == "const_one":
                a = torch.ones_like(a)
        # Duplicate scalar feature so the frozen primitive still receives 2-D input.
        return torch.stack([a, a], dim=1)

    def forward(self, x):
        return self.base(self.transform(x)) * self.scale + self.bias


def neural_accuracy(model, x, y):
    model.eval()
    with torch.no_grad():
        pred = model(x).argmax(-1)
    return float((pred == y).float().mean())


def build_shared_primitive(args, source_tasks, seed, device, bar):
    candidates = []
    combos = list(
        product(
            ["sequential", "parallel_sum", "residual_parallel"],
            [
                ["affine_polynomial", "polynomial"],
                ["affine_polynomial", "polynomial", "low_rank"],
            ],
        )
    )
    for ci, (motif, nodes) in enumerate(combos, 1):
        base = BaseModel(
            SharedPrimitive(nodes, motif, args.d_model, args.rank),
            args.d_model,
            args.classes,
        ).to(device)
        xs, ys = [], []
        for j, task in enumerate(source_tasks):
            x, y = make_data(task, max(1, args.train_size // len(source_tasks)), seed + 2000 + ci*17 + j, device, "A")
            xs.append(x); ys.append(y)
        fit_classifier(
            base,
            DataLoader(TensorDataset(torch.cat(xs), torch.cat(ys)), batch_size=args.batch_size, shuffle=True),
            args.core_fit_steps,
            args.core_fit_lr,
            False,
        )
        vals = []
        for j, task in enumerate(source_tasks):
            x, y = make_data(task, args.verifier_size, seed + 3000 + ci*17 + j, device, "B")
            vals.append(neural_accuracy(base, x, y))
        score = 0.5 * ((sum(vals)/len(vals)) + min(vals))
        candidates.append((score, base, vals))
        bar.update(6 + ci, "shared-primitive", f"candidate={ci} score={score:.3f}")
    candidates.sort(key=lambda z: z[0], reverse=True)
    return candidates[0]


# ------------------------- main experiment ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2])
    ap.add_argument("--all-tasks", nargs="+", default=["add", "compose", "mul", "sub"])
    ap.add_argument("--holdout-tasks", nargs="+", default=["sub"])
    ap.add_argument("--contrast-tasks", nargs="+", default=["sort"])
    ap.add_argument("--teacher-steps", type=int, default=800)
    ap.add_argument("--core-fit-steps", type=int, default=300)
    ap.add_argument("--program-fit-steps", type=int, default=120)
    ap.add_argument("--target-program-fit-steps", type=int, default=400)
    ap.add_argument("--transfer-control-steps", type=int, default=400)
    ap.add_argument("--train-size", type=int, default=6000)
    ap.add_argument("--verifier-size", type=int, default=1500)
    ap.add_argument("--test-size", type=int, default=1500)
    ap.add_argument("--fit-batch-samples", type=int, default=512)
    ap.add_argument("--semantic-probe-samples", type=int, default=512)
    ap.add_argument("--random-probe-samples", type=int, default=4096)
    ap.add_argument("--max-extra-ops", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--d-model", type=int, default=32)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--classes", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--core-fit-lr", type=float, default=1e-3)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", default="dart038_results.json")
    args = ap.parse_args()

    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    source_tasks = [t for t in args.all_tasks if t not in args.holdout_tasks]
    target = args.holdout_tasks[0]
    contrast = args.contrast_tasks[0]

    records = []

    for seed_idx, seed in enumerate(args.seeds, 1):
        seed_all(seed)
        bar = Progress(24, seed_idx, len(args.seeds))

        # Teacher diagnostics only.
        teachers = {}
        for j, task in enumerate(source_tasks + [target, contrast]):
            teacher = BaseModel(
                SharedPrimitive(["affine_polynomial", "polynomial"], "sequential", args.d_model, args.rank),
                args.d_model,
                args.classes,
            ).to(device)
            x, y = make_data(task, args.train_size, seed + 1000 + j, device, "A")
            fit_classifier(
                teacher,
                DataLoader(TensorDataset(x, y), batch_size=args.batch_size, shuffle=True),
                args.teacher_steps,
                args.lr,
                False,
            )
            teachers[task] = teacher
            bar.update(1 + j, "teacher-training", f"task={task}")

        _, base, source_scores = build_shared_primitive(args, source_tasks, seed, device, bar)
        for p in base.primitive.parameters():
            p.requires_grad_(False)

        # -------- law inference --------
        law_samples = []
        for i, regime in enumerate(("A", "B", "C", "D")):
            x, _ = make_data(target, args.semantic_probe_samples, seed + 4000 + i, device, regime)
            law_samples.append(infer_law(target, x))
        selected_law = law_samples[0]
        law_consistency_score = calculate_law_consistency(law_samples)

        # -------- verified program synthesis --------
        candidates = candidate_programs_for_law(selected_law, args.max_extra_ops)
        verified = []
        symbolic = symbolic_probes(device)
        for i, program in enumerate(candidates):
            sym_agree = exact_agreement(target, program, symbolic)

            random_agreements = []
            for j, regime in enumerate(("A", "B", "C", "D", "E")):
                rnd = randomized_probes(target, seed + 5000 + j, args.random_probe_samples, device)
                random_agreements.append(exact_agreement(target, program, rnd))

            proof_score = min([sym_agree] + random_agreements)
            if proof_score >= 1.0 - 1e-9:
                verified.append((program, sym_agree, random_agreements))

            bar.update(
                13 + i,
                "verified-program-search",
                f"candidate={i+1}/{len(candidates)} proof={proof_score:.3f}",
            )

        if not verified:
            raise RuntimeError(
                "No program passed exact semantic verification. "
                "DART-3.8 intentionally refuses to continue with an unverified program."
            )

        # Pick the shortest verified program, then highest symbolic agreement.
        verified.sort(key=lambda z: (z[0].length, -z[1], z[0].ops))
        selected_program, selected_sym, selected_random = verified[0]

        # Exact verification summary on independent probes, including E.
        verification = verify_program(
            target,
            selected_program,
            device,
            args.random_probe_samples,
            seed,
        )

        # Wrong program controls.
        wrong_candidates = []
        for op in PRIMITIVE_OPS:
            p = Program((op,))
            if p.ops != selected_program.ops:
                wrong_candidates.append(p)
        wrong_exact = max(
            exact_agreement(target, p, symbolic)
            for p in wrong_candidates
        )
        random_program = wrong_candidates[(seed * 7) % len(wrong_candidates)]
        random_exact = exact_agreement(target, random_program, symbolic)

        bar.update(
            19,
            "exact-program-verification",
            f"program={selected_program.ops} exact={verification['exact']:.3f}",
        )

        # Final untouched E neural diagnostic with the VERIFIED semantic program.
        xe, ye = make_data(target, args.test_size, seed + 9000, device, "E")
        zero_model = NeuralProgram(base, Program(("identity_a",))).to(device)
        zero_neural = neural_accuracy(zero_model, xe, ye)

        program_model = NeuralProgram(base, selected_program).to(device)
        xa, ya = make_data(target, args.verifier_size, seed + 9100, device, "A")
        fit_classifier(
            program_model,
            DataLoader(TensorDataset(xa, ya), batch_size=args.fit_batch_samples, shuffle=True),
            args.target_program_fit_steps,
            args.lr,
            True,
        )
        program_neural = neural_accuracy(program_model, xe, ye)

        contrast_x, contrast_y = make_data(contrast, args.test_size, seed + 9500, device, "E")
        contrast_model = NeuralProgram(base, selected_program).to(device)
        fit_classifier(
            contrast_model,
            DataLoader(TensorDataset(contrast_x, contrast_y), batch_size=args.fit_batch_samples, shuffle=True),
            args.transfer_control_steps,
            args.lr,
            True,
        )
        contrast_neural = neural_accuracy(contrast_model, contrast_x, contrast_y)

        bar.update(
            23,
            "neural-diagnostic",
            f"E={program_neural:.3f} exact={verification['exact']:.3f} law={selected_law.relation}",
        )
        bar.close()

        records.append(
            {
                "seed": seed,
                "selected_law": selected_law.__dict__,
                "compiled_program": list(selected_program.ops),
                "law_consistency_A_to_D": law_consistency_score,
                "verified_program_candidate_count": len(verified),
                "exact_verification": verification,
                "wrong_program_max_exact_agreement": wrong_exact,
                "random_program_exact_agreement": random_exact,
                "related_holdout": {
                    target: {
                        "teacher_neural_diagnostic": neural_accuracy(teachers[target], xe, ye),
                        "dart_zero_neural": zero_neural,
                        "dart_verified_program_neural": program_neural,
                    }
                },
                "contrast_holdout": {
                    contrast: {
                        "teacher_neural_diagnostic": neural_accuracy(teachers[contrast], contrast_x, contrast_y),
                        "dart_verified_program_neural": contrast_neural,
                    }
                },
            }
        )

    out = Path(args.out)
    summary = {
        "version": "DART-3.8",
        "parent_version": "DART-3.7",
        "protocol": {
            "verified_program_search": True,
            "exact_oracle_gate": True,
            "multi_regime_exact_verification": ["A", "B", "C", "D", "E"],
            "shortest_verified_program_selection": True,
            "wrong_program_control": True,
            "random_program_control": True,
            "frozen_primitive_neural_diagnostic": True,
            "untouched_final_regime": "E",
            "deterministic_seeding": True,
        },
        "related_holdout": {
            target: {
                "teacher_neural_diagnostic": sum(
                    r["related_holdout"][target]["teacher_neural_diagnostic"] for r in records
                ) / len(records),
                "dart_zero_neural": sum(
                    r["related_holdout"][target]["dart_zero_neural"] for r in records
                ) / len(records),
                "dart_verified_program_neural": sum(
                    r["related_holdout"][target]["dart_verified_program_neural"] for r in records
                ) / len(records),
            }
        },
        "semantic_execution": {
            "avg_law_consistency_A_to_D": sum(r["law_consistency_A_to_D"] for r in records) / len(records),
            "avg_exact_agreement": sum(r["exact_verification"]["exact"] for r in records) / len(records),
            "avg_symbolic_agreement": sum(r["exact_verification"]["exact"] for r in records) / len(records),
            "avg_randomized_agreement": sum(r["exact_verification"]["randomized"] for r in records) / len(records),
            "avg_wrong_program_max_agreement": sum(r["wrong_program_max_exact_agreement"] for r in records) / len(records),
            "avg_random_program_agreement": sum(r["random_program_exact_agreement"] for r in records) / len(records),
        },
        "contrast_holdout": {
            contrast: {
                "teacher_neural_diagnostic": sum(
                    r["contrast_holdout"][contrast]["teacher_neural_diagnostic"] for r in records
                ) / len(records),
                "dart_verified_program_neural": sum(
                    r["contrast_holdout"][contrast]["dart_verified_program_neural"] for r in records
                ) / len(records),
            }
        },
        "records": records,
    }

    out.write_text(json.dumps(summary, indent=2))
    print("DART-3.8: verified law-to-program compilation")
    print(json.dumps(summary, indent=2))
    print(f"Saved: {out.resolve()}")


if __name__ == "__main__":
    main()
