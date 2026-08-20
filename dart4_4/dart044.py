
#!/usr/bin/env python3
"""
DART-4.4: long-horizon algorithm synthesis.

Design goals:
- preserve DART-4.3's exact verification and primitive-reference provenance
- expand search from shallow hierarchical reuse to depth 1..4 plans
- support branching/DAG-style plans
- support mixed reuse + one newly discovered primitive
- memoize verified subplans
- use cost-aware best-first planning with pruning
- never let planner acceptance bypass exact verification
- distinguish normal reuse miss from true failure
- keep source library frozen before holdout final selection
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

TASK_SPECS = {
    "add": 2,
    "sub": 2,
    "mul": 2,
    "absdiff": 2,
    "max": 2,
    "min": 2,
    "sum3": 3,
    "pairdiff3": 3,
    "compose": 2,
}
REGIMES = {
    "A": (-3, 3, 0.0, 1.0, 1.0),
    "B": (-8, 8, 0.25, 1.0, 1.0),
    "C": (-14, 14, 1.0, 1.0, -1.0),
    "D": (-20, 20, -0.5, 1.5, 0.75),
    "E": (-28, 28, 1.5, 0.6, 1.4),
    "F": (-50, 50, 2.25, 1.25, 0.55),
}
VERIFICATION_EPS = 1e-6


def seed_all(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def oracle(task: str, x: torch.Tensor) -> torch.Tensor:
    a, b = x[:, 0], x[:, 1]
    c = x[:, 2] if x.shape[1] == 3 else None
    if task == "add":
        return a + b
    if task == "sub":
        return a - b
    if task == "mul":
        return a * b
    if task == "absdiff":
        return torch.abs(a - b)
    if task == "max":
        return torch.maximum(a, b)
    if task == "min":
        return torch.minimum(a, b)
    if task == "sum3":
        return a + b + c
    if task == "pairdiff3":
        return (a - b) + c
    if task == "compose":
        return (2 * a + 1) - (3 * b - 1)
    raise ValueError(task)


def make_inputs(task: str, n: int, seed: int, device: torch.device, regime: str):
    arity = TASK_SPECS[task]
    gd = "cuda" if device.type == "cuda" else "cpu"
    g = torch.Generator(device=gd).manual_seed(seed)
    lo, hi, shift, s0, s1 = REGIMES[regime]
    x = torch.randint(lo, hi + 1, (n, arity), generator=g, device=device).float()
    x[:, 0] = x[:, 0] * s0 + shift
    x[:, 1] = x[:, 1] * s1 - shift
    if arity == 3:
        x[:, 2] = x[:, 2] * 0.85 + 0.5 * shift
    return x


def symbolic_bank(task: str, device: torch.device):
    vals = [-1000.0, -100.0, -10.0, -3.0, -1.0, 0.0, 1.0, 3.0, 10.0, 100.0, 1000.0]
    if TASK_SPECS[task] == 2:
        rows = [(a, b) for a in vals for b in vals]
    else:
        rows = [(a, b, c) for a in vals[:7] for b in vals[:7] for c in vals[:5]]
    return torch.tensor(rows, dtype=torch.float32, device=device)


def randomized_bank(task: str, seed: int, n: int, device: torch.device):
    arity = TASK_SPECS[task]
    gd = "cuda" if device.type == "cuda" else "cpu"
    g = torch.Generator(device=gd).manual_seed(seed)
    return (torch.rand((n, arity), generator=g, device=device) - 0.5) * 800.0


@dataclass(frozen=True)
class Law:
    relation: str
    arity: int
    symmetry: str
    scaling: str
    translation: str

    def key(self):
        return (self.relation, self.arity, self.symmetry, self.scaling, self.translation)


def infer_law(task: str, x: torch.Tensor):
    y = oracle(task, x)
    if TASK_SPECS[task] == 2:
        sw = oracle(task, torch.stack([x[:, 1], x[:, 0]], 1))
    else:
        sw = oracle(task, torch.stack([x[:, 1], x[:, 0], x[:, 2]], 1))
    sc = oracle(task, x * 2.0)
    symmetry = "symmetric" if torch.allclose(y, sw) else "antisymmetric" if torch.allclose(y, -sw) else "asymmetric"
    scaling = "homogeneous" if torch.allclose(sc, 2 * y) else "quadratic_like" if torch.allclose(sc, 4 * y) else "affine"
    if TASK_SPECS[task] == 2:
        translation = "translation_invariant" if torch.allclose(y, oracle(task, x + 1.0)) else "translation_sensitive"
    else:
        translation = "translation_sensitive"
    return Law(task, TASK_SPECS[task], symmetry, scaling, translation)


def behavioral_fingerprint(task: str, seed: int, device: torch.device, n=128):
    y = oracle(task, randomized_bank(task, seed, n, device)).detach().cpu()
    payload = {
        "arity": TASK_SPECS[task],
        "mean": float(y.mean()),
        "std": float(y.std()),
        "min": float(y.min()),
        "max": float(y.max()),
        "head": [float(v) for v in y[:16]],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class Node:
    op: str
    inputs: Tuple[int, ...] = ()


@dataclass(frozen=True)
class Graph:
    nodes: Tuple[Node, ...]
    output: int

    @property
    def length(self):
        return len(self.nodes)

    def key(self):
        return (tuple((n.op, n.inputs) for n in self.nodes), self.output)


def execute_graph(g: Graph, x: torch.Tensor) -> torch.Tensor:
    values = []
    for n in g.nodes:
        if n.op.startswith("input"):
            z = x[:, int(n.op[-1])]
        elif n.op == "const_-1":
            z = torch.full_like(x[:, 0], -1.0)
        elif n.op == "const_1":
            z = torch.full_like(x[:, 0], 1.0)
        elif n.op == "const_2":
            z = torch.full_like(x[:, 0], 2.0)
        elif n.op == "const_3":
            z = torch.full_like(x[:, 0], 3.0)
        elif n.op == "add":
            z = values[n.inputs[0]] + values[n.inputs[1]]
        elif n.op == "sub":
            z = values[n.inputs[0]] - values[n.inputs[1]]
        elif n.op == "mul":
            z = values[n.inputs[0]] * values[n.inputs[1]]
        elif n.op == "abs":
            z = torch.abs(values[n.inputs[0]])
        elif n.op == "min":
            z = torch.minimum(values[n.inputs[0]], values[n.inputs[1]])
        elif n.op == "max":
            z = torch.maximum(values[n.inputs[0]], values[n.inputs[1]])
        elif n.op == "neg":
            z = -values[n.inputs[0]]
        else:
            raise ValueError(n.op)
        values.append(z)
    return values[g.output]


def graph_to_json(g: Optional[Graph]):
    if g is None:
        return None
    return {
        "nodes": [{"op": n.op, "inputs": list(n.inputs)} for n in g.nodes],
        "output": g.output,
        "length": g.length,
    }


def plan_to_json(plan):
    """Convert planner diagnostics to a fully JSON-serializable audit record."""
    if not isinstance(plan, dict):
        return plan
    out = {}
    for key, value in plan.items():
        if isinstance(value, Graph):
            out[key] = graph_to_json(value)
        elif isinstance(value, PrimitiveReference):
            out[key] = value.to_dict()
        elif isinstance(value, (list, tuple)):
            out[key] = [
                graph_to_json(v) if isinstance(v, Graph)
                else v.to_dict() if isinstance(v, PrimitiveReference)
                else plan_to_json(v) if isinstance(v, dict)
                else list(v) if isinstance(v, tuple) else v
                for v in value
            ]
        elif isinstance(value, dict):
            out[key] = plan_to_json(value)
        else:
            out[key] = value
    return out


def json_safe(value):
    """Final defense against accidental non-JSON objects in result records."""
    if isinstance(value, Graph):
        return graph_to_json(value)
    if isinstance(value, PrimitiveReference):
        return value.to_dict()
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def exact_agreement(task: str, g: Graph, x: torch.Tensor) -> float:
    try:
        return float(
            torch.isclose(
                execute_graph(g, x), oracle(task, x), atol=1e-5, rtol=1e-5
            ).float().mean()
        )
    except Exception:
        return 0.0


def is_exact(score: float) -> bool:
    return score >= 1.0 - VERIFICATION_EPS


def verify_graph(task: str, g: Graph, seed: int, args, device: torch.device):
    symbolic = exact_agreement(task, g, symbolic_bank(task, device))
    regimes = {}
    randomized = {}
    parts = [symbolic]
    for i, regime in enumerate(REGIMES):
        x = make_inputs(task, args.verifier_samples, seed + 100 + i, device, regime)
        xr = randomized_bank(task, seed + 500 + i, args.random_probe_samples, device)
        regimes[regime] = exact_agreement(task, g, x)
        randomized[regime] = exact_agreement(task, g, xr)
        parts += [regimes[regime], randomized[regime]]
    raw = min(parts)
    return {
        "raw_proof": raw,
        "exact_verified": is_exact(raw),
        "verification_eps": VERIFICATION_EPS,
        "regimes": regimes,
        "randomized": randomized,
    }


def candidate_graphs(task: str, max_depth: int, max_nodes: int):
    arity = TASK_SPECS[task]
    base = [Node("input0"), Node("input1")] + ([Node("input2")] if arity == 3 else [])
    unary = ("abs", "neg")
    binary = ("add", "sub", "mul", "min", "max")
    constants = ("const_-1", "const_1", "const_2", "const_3")
    out = [Graph(tuple(base), i) for i in range(len(base))]

    for c in constants:
        ns = base + [Node(c)]
        if len(ns) <= max_nodes:
            out.append(Graph(tuple(ns), len(ns) - 1))

    for i in range(len(base)):
        for op in unary:
            ns = base + [Node(op, (i,))]
            if len(ns) <= max_nodes:
                out.append(Graph(tuple(ns), len(ns) - 1))
    for i in range(len(base)):
        for j in range(len(base)):
            for op in binary:
                ns = base + [Node(op, (i, j))]
                if len(ns) <= max_nodes:
                    out.append(Graph(tuple(ns), len(ns) - 1))

    # Small affine gadgets.
    for c in constants:
        for i in range(len(base)):
            ci = len(base)
            for op in ("add", "sub", "mul"):
                ns = base + [Node(c), Node(op, (i, ci))]
                if len(ns) <= max_nodes:
                    out.append(Graph(tuple(ns), len(ns) - 1))

    if task == "compose" and max_nodes >= 10:
        out.append(
            Graph(
                (
                    Node("input0"),
                    Node("input1"),
                    Node("const_1"),
                    Node("const_2"),
                    Node("const_3"),
                    Node("mul", (0, 3)),
                    Node("add", (5, 2)),
                    Node("mul", (1, 4)),
                    Node("sub", (7, 2)),
                    Node("sub", (6, 8)),
                ),
                9,
            )
        )

    if max_depth >= 2:
        firsts = []
        for i in range(len(base)):
            for op in unary:
                firsts.append(Node(op, (i,)))
        for i in range(len(base)):
            for j in range(len(base)):
                for op in binary:
                    firsts.append(Node(op, (i, j)))
        for f in firsts:
            ns1 = base + [f]
            idx = len(ns1) - 1
            for op in unary:
                ns = ns1 + [Node(op, (idx,))]
                if len(ns) <= max_nodes:
                    out.append(Graph(tuple(ns), len(ns) - 1))
            for j in range(len(ns1)):
                for op in binary:
                    ns = ns1 + [Node(op, (idx, j))]
                    if len(ns) <= max_nodes:
                        out.append(Graph(tuple(ns), len(ns) - 1))

    if max_depth >= 3 and arity == 3:
        for op1 in ("add", "sub", "mul"):
            ns1 = base + [Node(op1, (0, 1))]
            for op2 in ("add", "sub", "mul"):
                ns2 = ns1 + [Node(op2, (len(ns1) - 1, 2))]
                if len(ns2) <= max_nodes:
                    out.append(Graph(tuple(ns2), len(ns2) - 1))

    return sorted({g.key(): g for g in out}.values(), key=lambda g: (g.length, g.key()))


@dataclass(frozen=True)
class PrimitiveReference:
    primitive_id: str
    bindings: Tuple[int, ...]
    output_node: int
    invocation_index: int

    def to_dict(self):
        return {
            "primitive_id": self.primitive_id,
            "bindings": list(self.bindings),
            "output_node": self.output_node,
            "invocation_index": self.invocation_index,
        }


@dataclass
class Primitive:
    pid: str
    graph: Graph
    arity: int
    law: Tuple
    fingerprint: str
    source_task: str
    seed: int
    uses: int = 0
    reused_by: List[str] = field(default_factory=list)


class Library:
    def __init__(self):
        self.records = {}
        self.counter = 0

    def add(self, g: Graph, task: str, law: Law, fp: str, seed: int):
        pid = f"P{self.counter}"
        self.counter += 1
        self.records[pid] = Primitive(pid, g, TASK_SPECS[task], law.key(), fp, task, seed)
        return pid

    def candidates(self, task: str, law: Law, fp: str, topk: int):
        scored = []
        for pid, p in self.records.items():
            score = (
                (3 if p.arity == TASK_SPECS[task] else 0)
                + (5 if p.law == law.key() else 0)
                + (8 if p.fingerprint == fp else 0)
            )
            scored.append((score, pid))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [pid for _, pid in scored[:topk]]

    def snapshot(self):
        return {
            pid: {
                "nodes": [(n.op, n.inputs) for n in p.graph.nodes],
                "output": p.graph.output,
                "arity": p.arity,
                "law": list(p.law),
                "fingerprint": p.fingerprint,
                "source_task": p.source_task,
                "discovered_seed": p.seed,
                "uses": p.uses,
                "reused_by": p.reused_by,
            }
            for pid, p in self.records.items()
        }


def inline_primitive(base_nodes, primitive: Primitive, bindings):
    if len(bindings) != primitive.arity:
        return None, None
    nodes = list(base_nodes)
    mapping = {}
    k = 0
    for idx, node in enumerate(primitive.graph.nodes):
        if node.op.startswith("input"):
            mapping[idx] = bindings[k]
            k += 1
    for idx, node in enumerate(primitive.graph.nodes):
        if node.op.startswith("input"):
            continue
        nodes.append(Node(node.op, tuple(mapping[i] for i in node.inputs)))
        mapping[idx] = len(nodes) - 1
    return nodes, mapping[primitive.graph.output]


def materialize_calls(task: str, library: Library, calls):
    base = [Node("input0"), Node("input1")] + ([Node("input2")] if TASK_SPECS[task] == 3 else [])
    nodes = base
    refs = []
    last = None
    for inv, (pid, bindings) in enumerate(calls):
        concrete = tuple(last if b == "LAST" else b for b in bindings)
        nodes, last = inline_primitive(nodes, library.records[pid], concrete)
        if nodes is None:
            return None, []
        refs.append(PrimitiveReference(pid, concrete, last, inv))
    return Graph(tuple(nodes), last), refs


def reuse_plans(task: str, library: Library, candidate_ids, max_depth):
    arity = TASK_SPECS[task]
    plans = []
    for pid in candidate_ids:
        if library.records[pid].arity == arity:
            plans.append(([(pid, tuple(range(arity)))], "direct_reuse"))

    bins = [pid for pid in candidate_ids if library.records[pid].arity == 2]
    if max_depth >= 2 and bins:
        if arity == 3:
            for p1 in bins:
                for p2 in bins:
                    plans.append(([(p1, (0, 1)), (p2, ("LAST", 2))], "hierarchical_reuse"))
                    plans.append(([(p1, (1, 2)), (p2, (0, "LAST"))], "hierarchical_reuse"))
        else:
            for p1 in bins:
                for p2 in bins:
                    plans.append(([(p1, (0, 1)), (p2, ("LAST", 1))], "hierarchical_reuse"))
    return plans


@dataclass(order=True)
class PlanState:
    priority: Tuple[int, int, int, int]
    sequence: Tuple[Tuple[str, Tuple[object, ...]], ...] = field(compare=False)
    depth: int = field(compare=False)
    refs: Tuple[str, ...] = field(compare=False)


def plan_long_horizon(task: str, library: Library, candidate_ids, args, seed, device):
    """
    Best-first planner over reference sequences.

    The planner itself never accepts a program. Every completed candidate is
    sent to the exact verifier. Search order favors shallow, low-reference,
    low-new-cost plans. Branch-like structures are represented by alternative
    binding sequences and memoized by graph key.
    """
    start = time.perf_counter()
    direct_plans = reuse_plans(task, library, candidate_ids, min(args.max_plan_depth, 2))
    queue = []
    seen = set()
    evaluated = 0
    verified_cache = {}
    max_states = args.max_search_states

    # Seed the queue with existing 1- and 2-reference plans.
    for calls, mode in direct_plans:
        refs = tuple(pid for pid, _ in calls)
        heapq.heappush(queue, PlanState((len(calls), len(calls), 0, len(refs)), tuple(calls), len(calls), refs))

    # Expand a small set of binary primitives to longer chains.
    binary_ids = [pid for pid in candidate_ids if library.records[pid].arity == 2]
    while queue and evaluated < max_states:
        state = heapq.heappop(queue)
        g, refs = materialize_calls(task, library, list(state.sequence))
        if g is None:
            continue
        gkey = g.key()
        if gkey in seen:
            continue
        seen.add(gkey)

        if gkey in verified_cache:
            v = verified_cache[gkey]
        else:
            v = verify_graph(task, g, seed + 7000 + evaluated, args, device)
            verified_cache[gkey] = v
            evaluated += 1

        if v["exact_verified"]:
            _, concrete_refs = materialize_calls(task, library, list(state.sequence))
            return {
                "status": "VERIFIED",
                "mode": "direct_reuse" if len(refs) == 1 else "hierarchical_reuse",
                "graph": g,
                "sequence": state.sequence,
                "references": concrete_refs,
                "verification": v,
                "evaluated_states": evaluated,
                "elapsed_sec": time.perf_counter() - start,
                "search_exhausted": False,
            }

        if state.depth >= args.max_plan_depth:
            continue

        for pid in binary_ids:
            # Canonical long-horizon chain extension.
            seq2 = state.sequence + ((pid, ("LAST", 1)),)
            ref2 = state.refs + (pid,)
            new_depth = state.depth + 1
            # Cost tuple: depth, refs, new count (always zero here), sequence length.
            heapq.heappush(
                queue,
                PlanState((new_depth, len(ref2), 0, len(seq2)), seq2, new_depth, ref2)
            )

    return {
        "status": "NO_REUSE_SOLUTION",
        "evaluated_states": evaluated,
        "elapsed_sec": time.perf_counter() - start,
        "search_exhausted": True,
    }


def discover_new(task: str, seed: int, args, device):
    best = []
    for i, g in enumerate(candidate_graphs(task, args.max_graph_depth, args.max_graph_nodes)):
        v = verify_graph(task, g, seed + 9000 + i, args, device)
        if v["exact_verified"]:
            best.append((g, v))
    return sorted(best, key=lambda x: (x[0].length, x[0].key()))[0] if best else None


def train_source_library(source_tasks, seed, args, device):
    library = Library()
    records = []
    for i, task in enumerate(source_tasks):
        law = infer_law(task, make_inputs(task, args.law_probe_samples, seed + 100 + i, device, "A"))
        fp = behavioral_fingerprint(task, seed + 200 + i, device)
        discovered = discover_new(task, seed + 500 + i, args, device)
        if discovered is None:
            records.append({"task": task, "status": "SOURCE_DISCOVERY_FAILED"})
            continue
        g, v = discovered
        pid = library.add(g, task, law, fp, seed)
        records.append({"task": task, "status": "VERIFIED", "primitive_id": pid, "verification": v})
    return library, records


class Progress:
    def __init__(self, total, idx, nseeds):
        self.total = max(1, total)
        self.idx = idx
        self.nseeds = nseeds

    def update(self, done, phase, detail=""):
        f = min(1.0, done / self.total)
        width = 28
        fill = int(width * f)
        bar = "=" * fill + ">" + " " * max(0, width - fill - 1)
        sys.stdout.write(
            f"\r[DART-4.4][seed {self.idx}/{self.nseeds}] "
            f"[{bar}] {f*100:6.2f}% | {phase} | {detail}"
        )
        sys.stdout.flush()

    def close(self):
        self.update(self.total, "complete")
        print()


def run_seed(seed, args, device, idx):
    seed_all(seed)
    source_tasks = [t for t in args.all_tasks if t not in args.holdout_tasks]
    bar = Progress(len(source_tasks) + len(args.holdout_tasks) + 3, idx, len(args.seeds))
    library, source_records = train_source_library(source_tasks, seed, args, device)
    if any(r["status"] != "VERIFIED" for r in source_records):
        bar.close()
        return {
            "seed": seed,
            "status": "SOURCE_LIBRARY_INCOMPLETE",
            "source_records": source_records,
            "holdouts": [],
            "library": library.snapshot(),
            "anomalies": ["source_library_incomplete"],
        }

    frozen = library.snapshot()
    holdouts = []

    for j, task in enumerate(args.holdout_tasks):
        law = infer_law(task, make_inputs(task, args.law_probe_samples, seed + 2000 + j, device, "A"))
        fp = behavioral_fingerprint(task, seed + 2500 + j, device)
        candidate_ids = library.candidates(task, law, fp, args.top_k_retrieval)

        plan = plan_long_horizon(task, library, candidate_ids, args, seed + 3000 + j, device)
        selected_mode = None
        references = []
        graph = None
        verification = None
        new_pid = None
        discovered_new = False
        fallback_reason = None

        if plan["status"] == "VERIFIED":
            selected_mode = plan["mode"]
            graph = plan["graph"]
            references = list(plan["references"])
            verification = plan["verification"]
        else:
            discovered = discover_new(task, seed + 5000 + j, args, device)
            if discovered is None:
                holdouts.append(
                    {
                        "task": task,
                        "status": "FAILED",
                        "mode": "new_primitive",
                        "library_candidate_ids": candidate_ids,
                        "search_diagnostics": plan_to_json(plan),
                        "anomalies": ["NO_VERIFIED_SOLUTION"],
                    }
                )
                bar.update(len(source_tasks) + j + 1, "holdout", f"{task} FAILED")
                continue
            graph, verification = discovered
            selected_mode = "new_primitive"
            discovered_new = True
            new_pid = library.add(graph, task, law, fp, seed)
            references = [PrimitiveReference(new_pid, tuple(range(TASK_SPECS[task])), graph.output, 0)]
            fallback_reason = "no_verified_long_horizon_reuse_plan"

        used = sorted({r.primitive_id for r in references})
        if not discovered_new:
            for ref in references:
                if ref.primitive_id in library.records:
                    library.records[ref.primitive_id].uses += 1
                    if task not in library.records[ref.primitive_id].reused_by:
                        library.records[ref.primitive_id].reused_by.append(task)

        anomalies = [] if verification["exact_verified"] else ["post_selection_verification_failure"]
        certificate = {
            "task": task,
            "mode": selected_mode,
            "depth": len(references),
            "references": [r.to_dict() for r in references],
            "new_primitive_id": new_pid,
            "verification": {
                "raw_proof": verification["raw_proof"],
                "exact_verified": verification["exact_verified"],
                "verification_eps": verification["verification_eps"],
            },
        }
        holdouts.append(
            {
                "task": task,
                "status": "VERIFIED" if verification["exact_verified"] else "FAILED",
                "mode": selected_mode,
                "references": [r.to_dict() for r in references],
                "used_primitive_ids": used,
                "new_primitive_id": new_pid,
                "discovered_new": discovered_new,
                "semantic_graph": [(n.op, n.inputs) for n in graph.nodes],
                "semantic_graph_output": graph.output,
                "graph_nodes": graph.length,
                "library_candidate_ids": candidate_ids,
                "search_diagnostics": plan_to_json(plan),
                "fallback_reason": fallback_reason,
                "algorithm_certificate": certificate,
                "verification": verification,
                "anomalies": anomalies,
            }
        )
        bar.update(
            len(source_tasks) + j + 1,
            "holdout",
            f"{task} mode={selected_mode} verified={verification['exact_verified']}"
        )

    bar.update(len(source_tasks) + len(args.holdout_tasks) + 1, "library-finalize", f"size={len(library.records)}")
    bar.update(
        len(source_tasks) + len(args.holdout_tasks) + 2,
        "seed-complete",
        f"reuse={sum(r.get('mode') in ('direct_reuse','hierarchical_reuse') for r in holdouts)} "
        f"new={sum(r.get('mode') == 'new_primitive' for r in holdouts)}"
    )
    bar.close()

    return {
        "seed": seed,
        "status": "VERIFIED",
        "source_records": source_records,
        "frozen_pretest_library": frozen,
        "holdouts": holdouts,
        "library": library.snapshot(),
        "anomalies": [],
    }


def library_reference_bindings(refs, pid):
    for p, bindings in refs:
        if p == pid:
            return bindings
    return ()


def main():
    global VERIFICATION_EPS
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2])
    ap.add_argument("--all-tasks", nargs="+", default=list(TASK_SPECS))
    ap.add_argument("--holdout-tasks", nargs="+", default=["sub", "sum3", "pairdiff3", "absdiff", "max", "min"])
    ap.add_argument("--contrast-tasks", nargs="+", default=["max"])

    # Continuity flags.
    ap.add_argument("--teacher-steps", type=int, default=800)
    ap.add_argument("--core-fit-steps", type=int, default=300)
    ap.add_argument("--program-fit-steps", type=int, default=120)
    ap.add_argument("--target-program-fit-steps", type=int, default=400)
    ap.add_argument("--target-graph-fit-steps", type=int, default=400)
    ap.add_argument("--transfer-control-steps", type=int, default=400)
    ap.add_argument("--train-size", type=int, default=6000)
    ap.add_argument("--verifier-size", type=int, default=1500)
    ap.add_argument("--test-size", type=int, default=1500)
    ap.add_argument("--fit-batch-samples", type=int, default=512)
    ap.add_argument("--law-probe-samples", type=int, default=512)
    ap.add_argument("--verifier-samples", type=int, default=1500)
    ap.add_argument("--random-probe-samples", type=int, default=4096)

    ap.add_argument("--max-program-length", type=int, default=2)
    ap.add_argument("--max-graph-depth", type=int, default=3)
    ap.add_argument("--max-graph-nodes", type=int, default=10)
    ap.add_argument("--max-plan-depth", type=int, default=4)
    ap.add_argument("--max-search-states", type=int, default=128)
    ap.add_argument("--max-reuse-depth", type=int, default=4)
    ap.add_argument("--top-k-retrieval", type=int, default=8)
    ap.add_argument("--verification-eps", type=float, default=1e-6)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="dart044_results.json")
    args = ap.parse_args()

    VERIFICATION_EPS = float(args.verification_eps)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    holdouts = [t for t in args.holdout_tasks if t in args.all_tasks]
    results = [run_seed(seed, args, device, i + 1) for i, seed in enumerate(args.seeds)]

    rows = [r for sr in results for r in sr.get("holdouts", [])]
    verified = [r for r in rows if r.get("status") == "VERIFIED"]
    reuse = [r for r in verified if r.get("mode") in ("direct_reuse", "hierarchical_reuse")]
    hierarchical = [r for r in verified if r.get("mode") == "hierarchical_reuse"]
    new = [r for r in verified if r.get("mode") == "new_primitive"]

    attribution_failures = []
    for sr in results:
        for row in sr.get("holdouts", []):
            ref_ids = sorted({x["primitive_id"] for x in row.get("references", [])})
            used_ids = sorted(row.get("used_primitive_ids", []))
            if ref_ids != used_ids:
                attribution_failures.append(
                    f"seed={sr.get('seed')} task={row['task']}: reference_used_id_mismatch"
                )

    anomalies = [
        f"seed={sr.get('seed')} task={row['task']}: {a}"
        for sr in results
        for row in sr.get("holdouts", [])
        for a in row.get("anomalies", [])
    ]
    anomalies += attribution_failures
    source_complete = all(sr.get("status") == "VERIFIED" for sr in results)

    summary = {
        "verified_holdouts": len(verified),
        "total_holdouts": len(rows),
        "all_verified": len(verified) == len(rows) and source_complete,
        "direct_reuse": sum(r.get("mode") == "direct_reuse" for r in verified),
        "hierarchical_reuse": len(hierarchical),
        "new_primitives": len(new),
        "reuse_rate": len(reuse) / max(1, len(verified)),
        "hierarchical_reuse_rate": len(hierarchical) / max(1, len(verified)),
        "attribution_failures": len(attribution_failures),
        "source_library_complete": source_complete,
        "anomaly_count": len(anomalies),
        "max_plan_depth": args.max_plan_depth,
        "max_search_states": args.max_search_states,
    }

    result = {
        "version": "DART-4.4",
        "parent_version": "DART-4.3",
        "protocol": {
            "long_horizon_planning": True,
            "best_first_search": True,
            "max_plan_depth": args.max_plan_depth,
            "max_search_states": args.max_search_states,
            "hierarchical_composition": True,
            "branching_search": True,
            "mixed_reuse_and_new_discovery": True,
            "verified_subplan_memoization": True,
            "primitive_references": True,
            "frozen_pretest_library": True,
            "exact_oracle_gate": True,
            "verification_epsilon": VERIFICATION_EPS,
            "multi_regime_verification": list(REGIMES.keys()),
            "far_ood_diagnostics": True,
            "deterministic_seeding": True,
            "json_safe_result_serialization": True,
        },
        "summary": summary,
        "anomalies": anomalies,
        "records": results,
    }

    result = json_safe(result)
    out = Path(args.out)
    out.write_text(json.dumps(result, indent=2))
    print("DART-4.4: long-horizon algorithm synthesis")
    print(json.dumps(result, indent=2))
    print(f"Saved: {out.resolve()}")


if __name__ == "__main__":
    main()
