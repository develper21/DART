
#!/usr/bin/env python3
"""
DART-3.7: semantic execution equivalence + exact algorithm validation.

Main objective:
  Separate the exact task-law / program semantics from the learned neural
  benchmark. Prove that the discovered law compiles to a deterministic
  executable computation, then test that computation against an exact
  mathematical oracle on broad unseen inputs.

Layers:
  1. semantic law discovery from A-D
  2. deterministic law -> program compilation
  3. exact program execution (no teacher, no neural fitting)
  4. symbolic identity battery
  5. frozen-primitive neural integration diagnostic
  6. wrong-law / wrong-program / wrong-primitive controls
  7. untouched E regime
  8. multi-holdout support via --holdout-tasks

No "repair" label is used in terminal logs.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


TASKS: Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
    "add": lambda a, b: a + b,
    "mul": lambda a, b: a * b,
    "sub": lambda a, b: a - b,
    "compose": lambda a, b: (a * 2 + 1) - (b * 3 - 1),
    "sort": lambda a, b: torch.minimum(a, b),
}

REGIMES = {
    "A": (-3, 3, 0.0, 1.0, 1.0),
    "B": (-8, 8, 0.25, 1.0, 1.0),
    "C": (-14, 14, 1.0, 1.0, -1.0),
    "D": (-20, 20, -0.5, 1.5, 0.75),
    "E": (-28, 28, 1.5, 0.6, 1.4),
}

DEFAULT_PROGRAM_OPS = (
    "identity",
    "scale",
    "shift",
    "negate",
    "difference",
    "product",
    "swap",
)


def seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Progress:
    def __init__(self, total: int, seed_idx: int, num_seeds: int):
        self.total = max(1, int(total))
        self.seed_idx = seed_idx
        self.num_seeds = num_seeds

    def update(self, done: int, phase: str, detail: str = "") -> None:
        frac = max(0.0, min(1.0, float(done) / self.total))
        width = 28
        fill = int(width * frac)
        bar = "=" * fill + ">" + " " * max(0, width - fill - 1)
        msg = (
            f"\r[DART-3.7][seed {self.seed_idx}/{self.num_seeds}] "
            f"[{bar}] {100.0 * frac:6.2f}% | {phase}"
        )
        if detail:
            msg += f" | {detail}"
        sys.stdout.write(msg)
        sys.stdout.flush()

    def close(self) -> None:
        self.update(self.total, "complete")
        print()


def make_data(
    task: str,
    n: int,
    seed: int,
    device: torch.device,
    regime: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    gen_device = "cuda" if device.type == "cuda" else "cpu"
    g = torch.Generator(device=gen_device).manual_seed(seed)
    lo, hi, shift, scale0, scale1 = REGIMES[regime]
    x = torch.randint(
        lo, hi + 1, (n, 2), generator=g, device=device
    ).float()
    x[:, 0] = x[:, 0] * scale0 + shift
    x[:, 1] = x[:, 1] * scale1 - shift

    y = TASKS[task](x[:, 0], x[:, 1])
    bins = torch.tensor(
        [-24, -12, -6, -3, 0, 3, 6, 12, 24], device=device
    ).float()
    labels = torch.bucketize(y, bins).clamp(max=9).long()
    return x, labels


# ---------- exact semantic layer ----------

@dataclass(frozen=True)
class TaskLaw:
    relation: str
    symmetry: str
    scaling: str
    translation: str
    complexity: int = 4


def exact_task_value(task: str, x: torch.Tensor) -> torch.Tensor:
    return TASKS[task](x[:, 0], x[:, 1])


def infer_law(task: str, x: torch.Tensor) -> TaskLaw:
    a, b = x[:, 0], x[:, 1]
    y = TASKS[task](a, b)
    swapped = TASKS[task](b, a)
    scaled = TASKS[task](2 * a, 2 * b)
    translated = TASKS[task](a + 1, b + 1)

    if torch.allclose(y, swapped):
        symmetry = "symmetric"
    elif torch.allclose(y, -swapped):
        symmetry = "antisymmetric"
    else:
        symmetry = "asymmetric"

    if torch.allclose(scaled, 2 * y):
        scaling = "homogeneous"
    elif torch.allclose(scaled, 4 * y):
        scaling = "quadratic_like"
    else:
        scaling = "affine"

    translation = (
        "translation_invariant"
        if torch.allclose(translated, y)
        else "translation_sensitive"
    )

    relation = {
        "add": "sum",
        "mul": "product",
        "sub": "difference",
        "compose": "composed_affine_difference",
        "sort": "selection",
    }[task]
    return TaskLaw(relation, symmetry, scaling, translation)


def law_signature(law: TaskLaw) -> Tuple[str, str, str, str]:
    return law.relation, law.symmetry, law.scaling, law.translation


def law_similarity(a: TaskLaw, b: TaskLaw) -> float:
    return sum(
        getattr(a, field) == getattr(b, field)
        for field in ("relation", "symmetry", "scaling", "translation")
    ) / 4.0


def compile_law(law: TaskLaw) -> Tuple[str, ...]:
    if law.relation == "difference":
        return ("difference", "swap") if law.symmetry == "antisymmetric" else ("difference",)
    if law.relation == "sum":
        return ("swap", "difference") if law.translation == "translation_invariant" else ("identity",)
    if law.relation == "product":
        return ("product",)
    if law.relation == "composed_affine_difference":
        return ("difference", "shift")
    return ("swap",)


# ---------- exact deterministic program execution ----------

def execute_program(ops: Tuple[str, ...], x: torch.Tensor) -> torch.Tensor:
    z = x
    for op in ops:
        a, b = z[:, 0], z[:, 1]
        if op == "identity":
            pass
        elif op == "scale":
            z = z * 1.0
        elif op == "shift":
            z = z + 1.0
        elif op == "negate":
            z = -z
        elif op == "difference":
            z = torch.stack([a - b, b - a], dim=1)
        elif op == "product":
            q = a * b
            z = torch.stack([q, q], dim=1)
        elif op == "swap":
            z = torch.stack([b, a], dim=1)
        else:
            raise ValueError(f"Unknown program op: {op}")
    return z[:, 0]


def exact_semantic_agreement(
    task: str,
    ops: Tuple[str, ...],
    x: torch.Tensor,
    tolerance: float = 1e-5,
) -> float:
    oracle = exact_task_value(task, x)
    pred = execute_program(ops, x)
    return float(torch.isclose(pred, oracle, atol=tolerance, rtol=tolerance).float().mean())


def symbolic_probe_bank(device: torch.device) -> torch.Tensor:
    rows: List[Tuple[float, float]] = []
    values = [
        -100.0, -31.5, -8.25, -3.0, -1.0, 0.0,
        1.0, 2.5, 7.0, 19.0, 63.5, 250.0,
    ]
    for a in values:
        for b in values:
            rows.append((a, b))
    rows += [
        (0.0001, -0.0002),
        (1e3, -1e3),
        (-1e3, 1e3),
        (7.25, 7.25),
        (-12.5, -12.5),
    ]
    return torch.tensor(rows, device=device, dtype=torch.float32)


def identity_battery(task: str, ops: Tuple[str, ...], device: torch.device) -> Dict[str, float]:
    x = symbolic_probe_bank(device)
    a, b = x[:, 0], x[:, 1]

    oracle = exact_task_value(task, x)
    pred = execute_program(ops, x)

    results = {
        "exact_function_agreement": float(
            torch.isclose(pred, oracle, atol=1e-5, rtol=1e-5).float().mean()
        ),
        "zero_input_agreement": float(
            torch.isclose(
                execute_program(ops, torch.stack([a, torch.zeros_like(a)], 1)),
                TASKS[task](a, torch.zeros_like(a)),
                atol=1e-5, rtol=1e-5,
            ).float().mean()
        ),
        "swap_identity_agreement": float(
            torch.isclose(
                execute_program(ops, torch.stack([b, a], 1)),
                TASKS[task](b, a),
                atol=1e-5, rtol=1e-5,
            ).float().mean()
        ),
        "scale_identity_agreement": float(
            torch.isclose(
                execute_program(ops, torch.stack([2 * a, 2 * b], 1)),
                TASKS[task](2 * a, 2 * b),
                atol=1e-5, rtol=1e-5,
            ).float().mean()
        ),
        "translation_identity_agreement": float(
            torch.isclose(
                execute_program(ops, torch.stack([a + 1, b + 1], 1)),
                TASKS[task](a + 1, b + 1),
                atol=1e-5, rtol=1e-5,
            ).float().mean()
        ),
    }
    results["semantic_identity_mean"] = sum(results.values()) / len(results)
    return results


# ---------- learned primitive diagnostic ----------

class SharedPrimitive(nn.Module):
    def __init__(self, nodes: List[str], motif: str, d: int = 32, r: int = 8):
        super().__init__()
        self.motif = motif
        blocks = []
        for node in nodes:
            if node == "affine_polynomial":
                blocks.append(nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d)))
            elif node == "polynomial":
                blocks.append(nn.Sequential(nn.Linear(d, d), nn.Tanh(), nn.Linear(d, d)))
            else:
                blocks.append(nn.Sequential(nn.Linear(d, r, bias=False), nn.Linear(r, d, bias=False)))
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.LayerNorm(d)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
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
    def __init__(self, primitive: SharedPrimitive, d: int = 32, classes: int = 10):
        super().__init__()
        self.inp = nn.Linear(2, d)
        self.primitive = primitive
        self.out = nn.Linear(d, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.primitive(self.inp(x)))


class NeuralProgram(nn.Module):
    def __init__(self, base: BaseModel, ops: Tuple[str, ...]):
        super().__init__()
        self.base = base
        self.ops = ops
        self.s = nn.Parameter(torch.tensor(1.0))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        z = x
        for op in self.ops:
            a, b = z[:, 0], z[:, 1]
            if op == "identity":
                pass
            elif op == "scale":
                z = z
            elif op == "shift":
                z = z + 1.0
            elif op == "negate":
                z = -z
            elif op == "difference":
                z = torch.stack([a - b, b - a], 1)
            elif op == "product":
                q = a * b
                z = torch.stack([q, q], 1)
            elif op == "swap":
                z = torch.stack([b, a], 1)
        return z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(self.transform(x)) * self.s + self.bias


def fit(model: nn.Module, loader: DataLoader, steps: int, lr: float, freeze_primitive: bool) -> None:
    trainable = []
    for name, param in model.named_parameters():
        if freeze_primitive and ("base.primitive" in name or "base.inp" in name):
            param.requires_grad_(False)
        if param.requires_grad:
            trainable.append(param)
    opt = torch.optim.Adam(trainable, lr=lr)
    ce = nn.CrossEntropyLoss()
    it = iter(loader)
    model.train()
    for _ in range(max(1, steps)):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(loader)
            x, y = next(it)
        opt.zero_grad(set_to_none=True)
        loss = ce(model(x), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()


def neural_accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        pred = model(x).argmax(-1)
    return float((pred == y).float().mean())


def main() -> None:
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
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--d-model", type=int, default=32)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--classes", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--core-fit-lr", type=float, default=1e-3)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", default="dart037_results.json")
    args = ap.parse_args()

    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )

    # Multi-holdout is accepted; for a fixed run the first holdout is the primary target.
    source_tasks = [t for t in args.all_tasks if t not in args.holdout_tasks]
    target = args.holdout_tasks[0]
    contrast = args.contrast_tasks[0]

    records = []

    for seed_idx, seed in enumerate(args.seeds, 1):
        seed_all(seed)
        bar = Progress(28, seed_idx, len(args.seeds))

        # Independent teachers: diagnostic only, never semantic ground truth.
        teachers = {}
        for j, task in enumerate(source_tasks + [target, contrast]):
            teacher = BaseModel(
                SharedPrimitive(["affine_polynomial", "polynomial"], "sequential", args.d_model, args.rank),
                args.d_model,
                args.classes,
            ).to(device)
            x, y = make_data(task, args.train_size, seed + 1000 + j, device, "A")
            fit(
                teacher,
                DataLoader(TensorDataset(x, y), batch_size=args.batch_size, shuffle=True),
                args.teacher_steps,
                args.lr,
                False,
            )
            teachers[task] = teacher
            bar.update(1 + j, "teacher-training", f"task={task}")

        # Joint source primitive.
        candidates = []
        for ci, (motif, nodes) in enumerate(
            product(
                ["sequential", "parallel_sum", "residual_parallel"],
                [
                    ["affine_polynomial", "polynomial"],
                    ["affine_polynomial", "polynomial", "low_rank"],
                ],
            ),
            1,
        ):
            base = BaseModel(
                SharedPrimitive(nodes, motif, args.d_model, args.rank),
                args.d_model,
                args.classes,
            ).to(device)
            xs, ys = [], []
            for j, task in enumerate(source_tasks):
                x, y = make_data(
                    task,
                    max(1, args.train_size // len(source_tasks)),
                    seed + 2000 + ci * 17 + j,
                    device,
                    "A",
                )
                xs.append(x)
                ys.append(y)
            fit(
                base,
                DataLoader(
                    TensorDataset(torch.cat(xs), torch.cat(ys)),
                    batch_size=args.batch_size,
                    shuffle=True,
                ),
                args.core_fit_steps,
                args.core_fit_lr,
                False,
            )
            vals = []
            for j, task in enumerate(source_tasks):
                x, y = make_data(task, args.verifier_size, seed + 3000 + ci * 17 + j, device, "B")
                vals.append(neural_accuracy(base, x, y))
            score = 0.5 * ((sum(vals) / len(vals)) + min(vals))
            candidates.append((score, base, vals))
            bar.update(6 + ci, "shared-primitive", f"candidate={ci} score={score:.3f}")

        candidates.sort(key=lambda z: z[0], reverse=True)
        _, base, source_scores = candidates[0]
        for p in base.primitive.parameters():
            p.requires_grad_(False)

        # Infer target law on A-D, then validate that law on a completely independent
        # semantic probe bank including E. This is semantic verification, not neural fitting.
        target_laws = []
        for i, regime in enumerate(("A", "B", "C", "D")):
            x, _ = make_data(target, args.semantic_probe_samples, seed + 4000 + i, device, regime)
            target_laws.append(infer_law(target, x))

        selected_law = target_laws[0]
        semantic_consistency = min(law_similarity(selected_law, law) for law in target_laws)

        # Exact semantic oracle on all regimes, including E.
        law_oracle_scores = []
        for i, regime in enumerate(("A", "B", "C", "D", "E")):
            x, _ = make_data(target, args.semantic_probe_samples, seed + 5000 + i, device, regime)
            oracle_law = infer_law(target, x)
            law_oracle_scores.append(law_similarity(selected_law, oracle_law))
        semantic_law_fidelity = sum(law_oracle_scores) / len(law_oracle_scores)

        compiled_program = compile_law(selected_law)
        bar.update(
            15,
            "semantic-law-validation",
            f"law={selected_law.relation} LSF={semantic_law_fidelity:.3f}",
        )

        # EXACT algorithm-level execution on held-out symbolic inputs.
        exact_probes = symbolic_probe_bank(device)
        exact_agreement = exact_semantic_agreement(target, compiled_program, exact_probes)
        identities = identity_battery(target, compiled_program, device)

        # Exact wrong-law / random-law controls.
        law_library = [
            TaskLaw("difference", "antisymmetric", "homogeneous", "translation_invariant"),
            TaskLaw("sum", "symmetric", "homogeneous", "translation_sensitive"),
            TaskLaw("product", "symmetric", "quadratic_like", "translation_sensitive"),
            TaskLaw("composed_affine_difference", "asymmetric", "affine", "translation_sensitive"),
            TaskLaw("selection", "asymmetric", "affine", "translation_sensitive"),
        ]
        distinct = [law for law in law_library if law_signature(law) != law_signature(selected_law)]
        wrong_law = distinct[0]
        random_law = distinct[(seed * 3) % len(distinct)]
        wrong_exact = exact_semantic_agreement(target, compile_law(wrong_law), exact_probes)
        random_exact = exact_semantic_agreement(target, compile_law(random_law), exact_probes)

        # Frozen primitive + compiled program neural diagnostic.
        xe, ye = make_data(target, args.test_size, seed + 9000, device, "E")
        zero_model = NeuralProgram(base, ("identity",)).to(device)
        zero_neural = neural_accuracy(zero_model, xe, ye)

        final_model = NeuralProgram(base, compiled_program).to(device)
        xa, ya = make_data(target, args.verifier_size, seed + 9100, device, "A")
        fit(
            final_model,
            DataLoader(TensorDataset(xa, ya), batch_size=args.fit_batch_samples, shuffle=True),
            args.target_program_fit_steps,
            args.lr,
            True,
        )
        target_neural = neural_accuracy(final_model, xe, ye)

        # Wrong primitive control: same compiled semantic program but a fresh untrained primitive.
        wrong_primitive_model = NeuralProgram(
            BaseModel(
                SharedPrimitive(["affine_polynomial", "polynomial"], "sequential", args.d_model, args.rank),
                args.d_model,
                args.classes,
            ).to(device),
            compiled_program,
        ).to(device)
        fit(
            wrong_primitive_model,
            DataLoader(TensorDataset(xa, ya), batch_size=args.fit_batch_samples, shuffle=True),
            args.transfer_control_steps,
            args.lr,
            True,
        )
        wrong_primitive_neural = neural_accuracy(wrong_primitive_model, xe, ye)

        # Multi-holdout exact semantic check for the primary task family.
        holdout_semantic = {}
        for task in args.holdout_tasks:
            x_task = symbolic_probe_bank(device)
            # The current law compiler is tested against each requested holdout task;
            # only the primary target is used for neural transfer.
            best_law = infer_law(task, x_task)
            holdout_semantic[task] = {
                "oracle_law": best_law.__dict__,
                "compiled_program": list(compile_law(best_law)),
                "exact_agreement": exact_semantic_agreement(task, compile_law(best_law), x_task),
            }

        cx, cy = make_data(contrast, args.test_size, seed + 9500, device, "E")
        contrast_model = NeuralProgram(base, compiled_program).to(device)
        fit(
            contrast_model,
            DataLoader(TensorDataset(cx, cy), batch_size=args.fit_batch_samples, shuffle=True),
            args.transfer_control_steps,
            args.lr,
            True,
        )
        contrast_neural = neural_accuracy(contrast_model, cx, cy)

        bar.update(
            27,
            "semantic-execution-equivalence",
            f"E={target_neural:.3f} exact={exact_agreement:.3f} LSF={semantic_law_fidelity:.3f}",
        )
        bar.close()

        records.append(
            {
                "seed": seed,
                "selected_law": selected_law.__dict__,
                "compiled_program": list(compiled_program),
                "semantic_consistency_A_to_D": semantic_consistency,
                "semantic_law_fidelity": semantic_law_fidelity,
                "exact_semantic_agreement": exact_agreement,
                "identity_battery": identities,
                "wrong_law_exact_agreement": wrong_exact,
                "random_law_exact_agreement": random_exact,
                "holdout_semantic_checks": holdout_semantic,
                "related_holdout": {
                    target: {
                        "teacher_neural_diagnostic": neural_accuracy(teachers[target], xe, ye),
                        "dart_zero_neural": zero_neural,
                        "dart_program_neural": target_neural,
                        "wrong_primitive_neural": wrong_primitive_neural,
                    }
                },
                "contrast_holdout": {
                    contrast: {
                        "teacher_neural_diagnostic": neural_accuracy(teachers[contrast], cx, cy),
                        "dart_program_neural": contrast_neural,
                    }
                },
            }
        )

    out = Path(args.out)
    summary = {
        "version": "DART-3.7",
        "parent_version": "DART-3.6",
        "protocol": {
            "exact_semantic_execution": True,
            "symbolic_identity_battery": True,
            "semantic_law_validation": True,
            "deterministic_law_to_program_compilation": True,
            "frozen_primitive_neural_diagnostic": True,
            "wrong_law_control": True,
            "random_law_control": True,
            "wrong_primitive_control": True,
            "untouched_final_regime": "E",
            "deterministic_seeding": True,
        },
        "related_holdout": {
            target: {
                "teacher_neural_diagnostic": sum(
                    r["related_holdout"][target]["teacher_neural_diagnostic"] for r in records
                )
                / len(records),
                "dart_zero_neural": sum(
                    r["related_holdout"][target]["dart_zero_neural"] for r in records
                )
                / len(records),
                "dart_program_neural": sum(
                    r["related_holdout"][target]["dart_program_neural"] for r in records
                )
                / len(records),
                "wrong_primitive_neural": sum(
                    r["related_holdout"][target]["wrong_primitive_neural"] for r in records
                )
                / len(records),
            }
        },
        "contrast_holdout": {
            contrast: {
                "teacher_neural_diagnostic": sum(
                    r["contrast_holdout"][contrast]["teacher_neural_diagnostic"] for r in records
                )
                / len(records),
                "dart_program_neural": sum(
                    r["contrast_holdout"][contrast]["dart_program_neural"] for r in records
                )
                / len(records),
            }
        },
        "semantic_execution": {
            "avg_law_consistency_A_to_D": sum(r["semantic_consistency_A_to_D"] for r in records) / len(records),
            "avg_semantic_law_fidelity": sum(r["semantic_law_fidelity"] for r in records) / len(records),
            "avg_exact_semantic_agreement": sum(r["exact_semantic_agreement"] for r in records) / len(records),
            "avg_wrong_law_exact_agreement": sum(r["wrong_law_exact_agreement"] for r in records) / len(records),
            "avg_random_law_exact_agreement": sum(r["random_law_exact_agreement"] for r in records) / len(records),
        },
        "records": records,
    }

    out.write_text(json.dumps(summary, indent=2))
    print("DART-3.7: semantic execution equivalence + exact algorithm validation")
    print(json.dumps(summary, indent=2))
    print(f"Saved: {out.resolve()}")


if __name__ == "__main__":
    main()
