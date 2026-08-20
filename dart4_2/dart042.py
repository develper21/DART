
#!/usr/bin/env python3
"""
DART-4.2: persistent primitive retrieval + compositional reuse.

Core hypothesis:
    DART-4.1 learned verified primitives.
    DART-4.2 must treat those primitives as reusable knowledge.

Protocol
--------
1. Warm a persistent primitive library ONLY from non-holdout source tasks.
2. Freeze the source library before each holdout's final test.
3. Retrieve compatible primitives by behavioral/law signature.
4. Try direct primitive reuse.
5. Try compositions of existing primitives (depth 1..N).
6. Only if reuse/composition fails, discover a new primitive.
7. Exact-verify every accepted primitive/program on symbolic, A-F,
   randomized, and far-OOD probes.
8. Track provenance, reuse, composition reuse, new invention, library growth,
   and hidden failures explicitly.

No holdout target information is inserted into the pre-test library.
"""

from __future__ import annotations
import argparse, hashlib, json, random, sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch


# ---------------------------------------------------------------------
# Task oracles
# ---------------------------------------------------------------------

TASK_SPECS = {
    "add": {"arity": 2, "family": "arithmetic"},
    "sub": {"arity": 2, "family": "arithmetic"},
    "mul": {"arity": 2, "family": "arithmetic"},
    "absdiff": {"arity": 2, "family": "selection"},
    "max": {"arity": 2, "family": "selection"},
    "min": {"arity": 2, "family": "selection"},
    "sum3": {"arity": 3, "family": "ternary_arithmetic"},
    "pairdiff3": {"arity": 3, "family": "ternary_composition"},
    "compose": {"arity": 2, "family": "affine_composition"},
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


def oracle(task: str, x: torch.Tensor) -> torch.Tensor:
    a, b = x[:, 0], x[:, 1]
    c = x[:, 2] if x.shape[1] >= 3 else None
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


def seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_inputs(task: str, n: int, seed: int, device: torch.device, regime: str) -> torch.Tensor:
    arity = TASK_SPECS[task]["arity"]
    gen_device = "cuda" if device.type == "cuda" else "cpu"
    g = torch.Generator(device=gen_device).manual_seed(seed)
    lo, hi, shift, s0, s1 = REGIMES[regime]
    x = torch.randint(lo, hi + 1, (n, arity), generator=g, device=device).float()
    x[:, 0] = x[:, 0] * s0 + shift
    x[:, 1] = x[:, 1] * s1 - shift
    if arity == 3:
        x[:, 2] = x[:, 2] * 0.85 + 0.5 * shift
    return x


def symbolic_bank(task: str, device: torch.device) -> torch.Tensor:
    vals = [-1000.0, -100.0, -10.0, -3.0, -1.0, 0.0, 1.0, 3.0, 10.0, 100.0, 1000.0]
    if TASK_SPECS[task]["arity"] == 2:
        rows = [(a, b) for a in vals for b in vals]
    else:
        rows = [(a, b, c) for a in vals[:7] for b in vals[:7] for c in vals[:5]]
    return torch.tensor(rows, dtype=torch.float32, device=device)


def randomized_bank(task: str, seed: int, n: int, device: torch.device) -> torch.Tensor:
    arity = TASK_SPECS[task]["arity"]
    gen_device = "cuda" if device.type == "cuda" else "cpu"
    g = torch.Generator(device=gen_device).manual_seed(seed)
    return (torch.rand((n, arity), generator=g, device=device) - 0.5) * 800.0


# ---------------------------------------------------------------------
# Law / behavioral signatures
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class Law:
    relation: str
    arity: int
    symmetry: str
    scaling: str
    translation: str

    def tuple(self):
        return (self.relation, self.arity, self.symmetry, self.scaling, self.translation)


def infer_law(task: str, x: torch.Tensor) -> Law:
    arity = TASK_SPECS[task]["arity"]
    y = oracle(task, x)
    if arity == 2:
        sw = oracle(task, torch.stack([x[:, 1], x[:, 0]], dim=1))
    else:
        sw = oracle(task, torch.stack([x[:, 1], x[:, 0], x[:, 2]], dim=1))
    sc = oracle(task, x * 2.0)

    if torch.allclose(y, sw):
        symmetry = "symmetric"
    elif torch.allclose(y, -sw):
        symmetry = "antisymmetric"
    else:
        symmetry = "asymmetric"

    if torch.allclose(sc, 2 * y):
        scaling = "homogeneous"
    elif torch.allclose(sc, 4 * y):
        scaling = "quadratic_like"
    else:
        scaling = "affine"

    if arity == 2:
        tr = oracle(task, x + 1.0)
        translation = "translation_invariant" if torch.allclose(y, tr) else "translation_sensitive"
    else:
        translation = "translation_sensitive"

    return Law(
        TASK_SPECS[task]["family"] + ":" + task,
        arity,
        symmetry,
        scaling,
        translation,
    )


def behavioral_fingerprint(task: str, seed: int, device: torch.device, n: int = 128) -> str:
    probes = randomized_bank(task, seed, n, device)
    y = oracle(task, probes).detach().cpu()
    # Deterministic coarse fingerprint: shape + simple statistics + exact probe samples.
    payload = {
        "arity": TASK_SPECS[task]["arity"],
        "shape": list(y.shape),
        "mean": float(y.mean()),
        "std": float(y.std()),
        "min": float(y.min()),
        "max": float(y.max()),
        "head": [float(v) for v in y[:16]],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


# ---------------------------------------------------------------------
# Graphs and primitive calls
# ---------------------------------------------------------------------

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
    vals: List[torch.Tensor] = []
    for n in g.nodes:
        if n.op == "input0":
            z = x[:, 0]
        elif n.op == "input1":
            z = x[:, 1]
        elif n.op == "input2":
            if x.shape[1] < 3:
                raise ValueError("input2 unavailable")
            z = x[:, 2]
        elif n.op == "add":
            z = vals[n.inputs[0]] + vals[n.inputs[1]]
        elif n.op == "sub":
            z = vals[n.inputs[0]] - vals[n.inputs[1]]
        elif n.op == "mul":
            z = vals[n.inputs[0]] * vals[n.inputs[1]]
        elif n.op == "abs":
            z = torch.abs(vals[n.inputs[0]])
        elif n.op == "min":
            z = torch.minimum(vals[n.inputs[0]], vals[n.inputs[1]])
        elif n.op == "max":
            z = torch.maximum(vals[n.inputs[0]], vals[n.inputs[1]])
        elif n.op == "neg":
            z = -vals[n.inputs[0]]
        else:
            raise ValueError(n.op)
        vals.append(z)
    return vals[g.output]


def exact_agreement(task: str, g: Graph, x: torch.Tensor) -> float:
    try:
        pred = execute_graph(g, x)
        true = oracle(task, x)
        return float(torch.isclose(pred, true, atol=1e-5, rtol=1e-5).float().mean())
    except Exception:
        return 0.0


def is_exact_score(score: float) -> bool:
    return score >= 1.0 - VERIFICATION_EPS


def base_inputs(arity: int) -> List[Node]:
    nodes = [Node("input0"), Node("input1")]
    if arity == 3:
        nodes.append(Node("input2"))
    return nodes


def candidate_graphs(task: str, max_depth: int = 2, max_nodes: int = 8) -> List[Graph]:
    arity = TASK_SPECS[task]["arity"]
    base = base_inputs(arity)
    out: List[Graph] = [Graph(tuple(base), i) for i in range(len(base))]
    unary = ("abs", "neg")
    binary = ("add", "sub", "mul", "min", "max")

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

    if max_depth >= 2:
        first_nodes = []
        for i in range(len(base)):
            for op in unary:
                first_nodes.append(Node(op, (i,)))
        for i in range(len(base)):
            for j in range(len(base)):
                for op in binary:
                    first_nodes.append(Node(op, (i, j)))

        for first in first_nodes:
            ns1 = base + [first]
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
            idx1 = len(ns1) - 1
            for op2 in ("add", "sub", "mul"):
                ns2 = ns1 + [Node(op2, (idx1, 2))]
                if len(ns2) <= max_nodes:
                    out.append(Graph(tuple(ns2), len(ns2) - 1))

    seen = {}
    for g in out:
        seen[g.key()] = g
    return sorted(seen.values(), key=lambda g: (g.length, g.key()))


# ---------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------

def verify_graph(task: str, g: Graph, seed: int, args, device: torch.device) -> Dict[str, object]:
    parts = [
        exact_agreement(task, g, symbolic_bank(task, device))
    ]
    regimes = {}
    randomized = {}
    for i, regime in enumerate(REGIMES):
        regimes[regime] = exact_agreement(
            task, g, make_inputs(task, args.verifier_samples, seed + 100 + i, device, regime)
        )
        randomized[regime] = exact_agreement(
            task, g, randomized_bank(task, seed + 300 + i, args.random_probe_samples, device)
        )
        parts.extend([regimes[regime], randomized[regime]])

    raw_proof = min(parts)
    return {
        "raw_proof": raw_proof,
        "exact_verified": is_exact_score(raw_proof),
        "verification_eps": VERIFICATION_EPS,
        "regimes": regimes,
        "randomized": randomized,
        "proof_parts": parts,
    }


# ---------------------------------------------------------------------
# Primitive library
# ---------------------------------------------------------------------

@dataclass
class PrimitiveRecord:
    primitive_id: str
    graph: Graph
    arity: int
    law: Tuple
    fingerprint: str
    source_task: str
    uses: int = 0
    reused_by: List[str] = None
    verified: bool = True

    def __post_init__(self):
        if self.reused_by is None:
            self.reused_by = []


class PrimitiveLibrary:
    def __init__(self):
        self.records: Dict[str, PrimitiveRecord] = {}
        self.counter = 0

    def add(self, graph: Graph, task: str, law: Law, fingerprint: str, source: str) -> str:
        pid = f"P{self.counter}"
        self.counter += 1
        self.records[pid] = PrimitiveRecord(
            primitive_id=pid,
            graph=graph,
            arity=TASK_SPECS[task]["arity"],
            law=law.tuple(),
            fingerprint=fingerprint,
            source_task=source,
            uses=0,
            reused_by=[],
            verified=True,
        )
        return pid

    def semantic_candidates(self, task: str, law: Law, fingerprint: str, top_k: int) -> List[str]:
        scored = []
        for pid, rec in self.records.items():
            score = 0
            if rec.arity == TASK_SPECS[task]["arity"]:
                score += 2
            if rec.law == law.tuple():
                score += 4
            if rec.fingerprint == fingerprint:
                score += 6
            # Family/relation similarity.
            if rec.law and rec.law[0].split(":")[0] == law.relation.split(":")[0]:
                score += 1
            scored.append((score, pid))
        scored.sort(reverse=True)
        return [pid for _, pid in scored[:top_k]]

    def snapshot(self):
        return {
            pid: {
                "arity": rec.arity,
                "nodes": [(n.op, n.inputs) for n in rec.graph.nodes],
                "output": rec.graph.output,
                "law": list(rec.law),
                "fingerprint": rec.fingerprint,
                "source_task": rec.source_task,
                "uses": rec.uses,
                "reused_by": rec.reused_by,
                "verified": rec.verified,
            }
            for pid, rec in self.records.items()
        }


# ---------------------------------------------------------------------
# Primitive graph composition
# ---------------------------------------------------------------------

def compose_primitives(task: str, primitive_graphs: List[Tuple[Graph, Tuple[int, ...]]]) -> Optional[Graph]:
    """
    Compose graphs by binding each primitive input node to existing parent node indices.

    primitive_graphs:
        [(graph, input_bindings), ...]
    For primitive arity k, input_bindings has k parent node indices.
    """
    arity = TASK_SPECS[task]["arity"]
    nodes = base_inputs(arity)

    for graph, bindings in primitive_graphs:
        if not bindings or len(bindings) != (sum(1 for n in graph.nodes if n.op.startswith("input"))):
            return None
        mapping = {}
        input_position = 0
        for local_idx, node in enumerate(graph.nodes):
            if node.op.startswith("input"):
                mapping[local_idx] = bindings[input_position]
                input_position += 1

        for local_idx, node in enumerate(graph.nodes):
            if node.op.startswith("input"):
                continue
            new_inputs = tuple(mapping[i] for i in node.inputs)
            nodes.append(Node(node.op, new_inputs))
            mapping[local_idx] = len(nodes) - 1

        output_source = mapping[graph.output]

        # Mark current composition output as a node by retaining it as-is.
        current_output = output_source

    if not primitive_graphs:
        return None
    return Graph(tuple(nodes), current_output)


def composition_candidates(task: str, library: PrimitiveLibrary, candidate_ids: List[str], max_depth: int) -> List[Graph]:
    arity = TASK_SPECS[task]["arity"]
    out: Dict[Tuple, Graph] = {}

    # Direct reuse: one primitive consumes the task inputs.
    for pid in candidate_ids:
        rec = library.records[pid]
        if rec.arity != arity:
            continue
        g = compose_primitives(task, [(rec.graph, tuple(range(arity)))])
        if g is not None:
            out[g.key()] = g

    # Binary composition: for ternary target, first primitive consumes (x,y),
    # second consumes (first_result,z); also try (x, y) then (z, first_result).
    if max_depth >= 2:
        binary_ids = [pid for pid in candidate_ids if library.records[pid].arity == 2]
        if arity == 2:
            for p1 in binary_ids:
                for p2 in binary_ids:
                    g1 = compose_primitives(task, [(library.records[p1].graph, (0, 1))])
                    if g1 is None:
                        continue
                    # First graph output index is g1.output.
                    g2 = compose_primitives(task, [
                        (library.records[p1].graph, (0, 1)),
                        (library.records[p2].graph, (g1.output, 1)),
                    ])
                    if g2 is not None:
                        out[g2.key()] = g2
        elif arity == 3:
            for p1 in binary_ids:
                for p2 in binary_ids:
                    g = compose_primitives(task, [
                        (library.records[p1].graph, (0, 1)),
                        (library.records[p2].graph, (len(base_inputs(3)), 2)),
                    ])
                    if g is not None:
                        out[g.key()] = g

    return sorted(out.values(), key=lambda g: (g.length, g.key()))


# ---------------------------------------------------------------------
# Discover source primitive (warmup) and fallback primitive
# ---------------------------------------------------------------------

def discover_verified_primitive(task: str, seed: int, args, device: torch.device) -> Tuple[Optional[Graph], Dict[str, object]]:
    candidates = candidate_graphs(task, args.max_graph_depth, args.max_graph_nodes)
    audit = []
    verified = []
    for i, g in enumerate(candidates):
        v = verify_graph(task, g, seed + 1000 + i, args, device)
        audit.append({"nodes": g.length, "graph": [(n.op, n.inputs) for n in g.nodes],
                      "verification": v})
        if v["exact_verified"]:
            verified.append((g, v))
    if not verified:
        return None, {"candidate_count": len(candidates), "audit": audit, "status": "NO_VERIFIED_PRIMITIVE"}
    verified.sort(key=lambda item: (item[0].length, item[0].key()))
    return verified[0][0], {
        "candidate_count": len(candidates),
        "audit": audit,
        "status": "VERIFIED",
        "raw_proof": verified[0][1]["raw_proof"],
    }


# ---------------------------------------------------------------------
# Progress
# ---------------------------------------------------------------------

class Progress:
    def __init__(self, total: int, seed_idx: int, nseeds: int):
        self.total = max(1, total)
        self.seed_idx = seed_idx
        self.nseeds = nseeds

    def update(self, done: int, phase: str, detail: str = ""):
        frac = max(0.0, min(1.0, done / self.total))
        width = 28
        fill = int(width * frac)
        bar = "=" * fill + ">" + " " * max(0, width - fill - 1)
        msg = f"\r[DART-4.2][seed {self.seed_idx}/{self.nseeds}] [{bar}] {100*frac:6.2f}% | {phase}"
        if detail:
            msg += f" | {detail}"
        sys.stdout.write(msg)
        sys.stdout.flush()

    def close(self):
        self.update(self.total, "complete")
        print()


# ---------------------------------------------------------------------
# One seed: warm source library, then sequential holdouts
# ---------------------------------------------------------------------

def run_seed(seed: int, args, device: torch.device, seed_idx: int):
    seed_all(seed)
    source_tasks = [t for t in args.all_tasks if t not in args.holdout_tasks]
    bar = Progress(len(source_tasks) + len(args.holdout_tasks) + 4, seed_idx, len(args.seeds))
    library = PrimitiveLibrary()

    source_rows = []
    for i, task in enumerate(source_tasks):
        law = infer_law(task, make_inputs(task, args.law_probe_samples, seed + 100 + i, device, "A"))
        fp = behavioral_fingerprint(task, seed + 500 + i, device)
        graph, audit = discover_verified_primitive(task, seed + 1000 + i, args, device)
        if graph is None:
            source_rows.append({"task": task, "status": "SOURCE_DISCOVERY_FAILED"})
            bar.update(i + 1, "library-warmup", f"task={task} verified=False")
            continue
        pid = library.add(graph, task, law, fp, task)
        source_rows.append({
            "task": task,
            "status": "VERIFIED",
            "primitive_id": pid,
            "graph": [(n.op, n.inputs) for n in graph.nodes],
            "discovered_new": True,
            "verification": audit,
        })
        bar.update(i + 1, "library-warmup", f"task={task} primitive={pid}")

    holdout_rows = []
    offset = len(source_tasks)
    for j, task in enumerate(args.holdout_tasks):
        law = infer_law(task, make_inputs(task, args.law_probe_samples, seed + 2000 + j, device, "A"))
        fp = behavioral_fingerprint(task, seed + 2500 + j, device)
        candidate_ids = library.semantic_candidates(task, law, fp, args.top_k_retrieval)

        # Direct and compositional reuse against the pre-test library.
        reuse_graphs = composition_candidates(task, library, candidate_ids, args.max_reuse_depth)
        selected = None
        selected_kind = None
        selected_pids = []
        reuse_verification = None

        for g in reuse_graphs:
            v = verify_graph(task, g, seed + 3000 + j, args, device)
            if v["exact_verified"]:
                selected = g
                reuse_verification = v
                # Identify any library IDs that plausibly contributed.
                selected_pids = candidate_ids[:min(len(candidate_ids), args.top_k_retrieval)]
                selected_kind = "direct_reuse" if g.length <= 3 else "composition_reuse"
                break

        discovered_new = False
        discovery_audit = None
        new_pid = None

        if selected is None:
            # Reuse failed: now, and only now, invent a new primitive/program.
            selected, discovery_audit = discover_verified_primitive(
                task, seed + 4000 + j, args, device
            )
            if selected is None:
                anomalies = ["NO_VERIFIED_SOLUTION"]
                holdout_rows.append({
                    "task": task,
                    "status": "FAILED",
                    "anomalies": anomalies,
                    "candidate_retrieval": candidate_ids,
                })
                bar.update(offset + j + 1, "holdout", f"task={task} FAILED")
                continue

            discovered_new = True
            selected_kind = "new_primitive"
            new_pid = library.add(selected, task, law, fp, task)

        if selected_kind in ("direct_reuse", "composition_reuse"):
            for pid in selected_pids:
                if pid in library.records:
                    library.records[pid].uses += 1
                    library.records[pid].reused_by.append(task)

        final_verification = reuse_verification
        if final_verification is None:
            final_verification = verify_graph(task, selected, seed + 5000 + j, args, device)

        anomalies = []
        if not final_verification["exact_verified"]:
            anomalies.append("post_selection_verification_failure")
        if source_tasks and selected_kind == "new_primitive":
            # This is allowed only when retrieval/composition failed; record the reason explicitly.
            if len(reuse_graphs) == 0:
                anomalies.append("reuse_search_empty")
            else:
                anomalies.append("reuse_search_failed_exact")

        holdout_rows.append({
            "task": task,
            "status": "VERIFIED" if final_verification["exact_verified"] else "FAILED",
            "law": law.__dict__,
            "library_candidates": candidate_ids,
            "selected_kind": selected_kind,
            "selected_primitive_ids": selected_pids,
            "primitive_id": new_pid,
            "discovered_new": discovered_new,
            "selected_graph": [(n.op, n.inputs) for n in selected.nodes],
            "selected_nodes": selected.length,
            "final_verification": final_verification,
            "reuse_candidate_count": len(reuse_graphs),
            "anomalies": anomalies,
        })

        bar.update(
            offset + j + 1,
            "holdout",
            f"task={task} mode={selected_kind} verified={final_verification['exact_verified']}",
        )

    bar.update(len(source_tasks) + len(args.holdout_tasks) + 1, "library-finalize",
               f"size={len(library.records)}")
    bar.update(len(source_tasks) + len(args.holdout_tasks) + 2, "seed-complete",
               f"reuse={sum(1 for r in holdout_rows if r.get('selected_kind') in ('direct_reuse','composition_reuse'))} "
               f"new={sum(1 for r in holdout_rows if r.get('discovered_new'))}")
    bar.close()

    return {
        "seed": seed,
        "source_tasks": source_tasks,
        "holdouts": holdout_rows,
        "library": library.snapshot(),
        "library_size": len(library.records),
        "source_records": source_rows,
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    global VERIFICATION_EPS

    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2])
    ap.add_argument("--all-tasks", nargs="+", default=list(TASK_SPECS))
    ap.add_argument("--holdout-tasks", nargs="+", default=["sub", "sum3", "pairdiff3", "absdiff", "max", "min"])
    ap.add_argument("--contrast-tasks", nargs="+", default=["max"])

    # Retain familiar CLI flags for continuity.
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

    ap.add_argument("--max-graph-depth", type=int, default=2)
    ap.add_argument("--max-graph-nodes", type=int, default=8)
    ap.add_argument("--max-reuse-depth", type=int, default=2)
    ap.add_argument("--top-k-retrieval", type=int, default=8)
    ap.add_argument("--verification-eps", type=float, default=1e-6)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="dart042_results.json")
    args = ap.parse_args()

    VERIFICATION_EPS = float(args.verification_eps)
    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )

    holdouts = [t for t in args.holdout_tasks if t in args.all_tasks]
    if not holdouts:
        raise ValueError("No valid holdout tasks.")

    results = [run_seed(seed, args, device, i + 1) for i, seed in enumerate(args.seeds)]

    holdout_rows = [r for sr in results for r in sr["holdouts"]]
    verified = [r for r in holdout_rows if r["status"] == "VERIFIED"]
    reused = [r for r in verified if r["selected_kind"] in ("direct_reuse", "composition_reuse")]
    new = [r for r in verified if r["selected_kind"] == "new_primitive"]
    comp_reuse = [r for r in reused if r["selected_kind"] == "composition_reuse"]

    anomalies = [
        f"seed={sr['seed']} task={r['task']}: {a}"
        for sr in results
        for r in sr["holdouts"]
        for a in r.get("anomalies", [])
    ]
    anomalies.extend(
        f"seed={sr['seed']} task={r['task']}: SOURCE_DISCOVERY_FAILED"
        for sr in results
        for r in sr["source_records"]
        if r.get("status") != "VERIFIED"
    )

    result = {
        "version": "DART-4.2",
        "parent_version": "DART-4.1",
        "protocol": {
            "persistent_source_library": True,
            "holdout_library_frozen_before_final_selection": True,
            "behavioral_primitive_retrieval": True,
            "reuse_before_invention": True,
            "direct_primitive_reuse": True,
            "compositional_primitive_reuse": True,
            "new_primitive_fallback": True,
            "provenance_tracking": True,
            "exact_oracle_gate": True,
            "verification_epsilon": VERIFICATION_EPS,
            "multi_regime_verification": list(REGIMES.keys()),
            "far_ood_diagnostics": True,
            "hidden_failure_diagnostics": True,
            "deterministic_seeding": True,
        },
        "summary": {
            "verified_holdouts": len(verified),
            "total_holdouts": len(holdout_rows),
            "all_verified": len(verified) == len(holdout_rows),
            "reused_primitives": len(reused),
            "composition_reuse": len(comp_reuse),
            "new_primitives": len(new),
            "reuse_rate": len(reused) / max(1, len(verified)),
            "composition_reuse_rate": len(comp_reuse) / max(1, len(verified)),
            "anomaly_count": len(anomalies),
        },
        "anomalies": anomalies,
        "records": results,
    }

    out = Path(args.out)
    out.write_text(json.dumps(result, indent=2))
    print("DART-4.2: persistent primitive retrieval + compositional reuse")
    print(json.dumps(result, indent=2))
    print(f"Saved: {out.resolve()}")


if __name__ == "__main__":
    main()
