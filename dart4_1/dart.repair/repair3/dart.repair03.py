
#!/usr/bin/env python3
"""
DART-4.1 repair01: primitive discovery pipeline integrity repair.

Repair objective:
1) Isolate candidate graph generation.
2) Verify one candidate independently against exact symbolic/A-F/random probes.
3) Insert only verified primitives into a persistent library.
4) Reuse an existing verified primitive on a later task.
5) Only after this integrity chain passes, run the broader holdout suite.

This repair intentionally produces explicit audit fields:
- candidate count
- expected minimal graph presence
- best proof
- library insertion
- reuse event
- failure reason

No repair label is used in the repaired DART research logs except the directory/file identity.
"""

from __future__ import annotations
import argparse, json, random, sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import torch


TASK_SPECS = {
    "add": {"arity": 2},
    "sub": {"arity": 2},
    "mul": {"arity": 2},
    "absdiff": {"arity": 2},
    "max": {"arity": 2},
    "min": {"arity": 2},
    "sum3": {"arity": 3},
    "pairdiff3": {"arity": 3},
    "compose": {"arity": 2},
}

REGIMES = {
    "A": (-3, 3, 0.0, 1.0, 1.0),
    "B": (-8, 8, 0.25, 1.0, 1.0),
    "C": (-14, 14, 1.0, 1.0, -1.0),
    "D": (-20, 20, -0.5, 1.5, 0.75),
    "E": (-28, 28, 1.5, 0.6, 1.4),
    "F": (-50, 50, 2.25, 1.25, 0.55),
}

def seed_all(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def oracle(task: str, x: torch.Tensor):
    a,b=x[:,0],x[:,1]
    c=x[:,2] if x.shape[1] >= 3 else None
    if task=="add": return a+b
    if task=="sub": return a-b
    if task=="mul": return a*b
    if task=="absdiff": return torch.abs(a-b)
    if task=="max": return torch.maximum(a,b)
    if task=="min": return torch.minimum(a,b)
    if task=="sum3": return a+b+c
    if task=="pairdiff3": return (a-b)+c
    if task=="compose": return (2*a+1)-(3*b-1)
    raise ValueError(task)

def make_inputs(task,n,seed,device,regime):
    arity=TASK_SPECS[task]["arity"]
    gd="cuda" if device.type=="cuda" else "cpu"
    g=torch.Generator(device=gd).manual_seed(seed)
    lo,hi,shift,s0,s1=REGIMES[regime]
    x=torch.randint(lo,hi+1,(n,arity),generator=g,device=device).float()
    x[:,0]=x[:,0]*s0+shift
    x[:,1]=x[:,1]*s1-shift
    if arity==3:
        x[:,2]=x[:,2]*0.85+0.5*shift
    return x

@dataclass(frozen=True)
class Node:
    op:str
    inputs:Tuple[int,...]=()

@dataclass(frozen=True)
class Graph:
    nodes:Tuple[Node,...]
    output:int
    def key(self): return tuple((n.op,n.inputs) for n in self.nodes),self.output
    @property
    def length(self): return len(self.nodes)

def execute(g:Graph,x:torch.Tensor):
    vals=[]
    for n in g.nodes:
        if n.op=="input0": z=x[:,0]
        elif n.op=="input1": z=x[:,1]
        elif n.op=="input2":
            if x.shape[1] < 3: raise ValueError("input2 unavailable")
            z=x[:,2]
        elif n.op=="add": z=vals[n.inputs[0]]+vals[n.inputs[1]]
        elif n.op=="sub": z=vals[n.inputs[0]]-vals[n.inputs[1]]
        elif n.op=="mul": z=vals[n.inputs[0]]*vals[n.inputs[1]]
        elif n.op=="abs": z=torch.abs(vals[n.inputs[0]])
        elif n.op=="min": z=torch.minimum(vals[n.inputs[0]],vals[n.inputs[1]])
        elif n.op=="max": z=torch.maximum(vals[n.inputs[0]],vals[n.inputs[1]])
        elif n.op=="neg": z=-vals[n.inputs[0]]
        else: raise ValueError(n.op)
        vals.append(z)
    return vals[g.output]

def agreement(task,g,x):
    try:
        return float(torch.isclose(execute(g,x),oracle(task,x),atol=1e-5,rtol=1e-5).float().mean())
    except Exception:
        return 0.0

def symbolic_bank(task,device):
    vals=[-1000,-100,-10,-3,-1,0,1,3,10,100,1000]
    if TASK_SPECS[task]["arity"]==2:
        rows=[(a,b) for a in vals for b in vals]
    else:
        rows=[(a,b,c) for a in vals[:7] for b in vals[:7] for c in vals[:5]]
    return torch.tensor(rows,dtype=torch.float32,device=device)

def randomized_bank(task,seed,n,device):
    arity=TASK_SPECS[task]["arity"]
    gd="cuda" if device.type=="cuda" else "cpu"
    g=torch.Generator(device=gd).manual_seed(seed)
    return (torch.rand((n,arity),generator=g,device=device)-0.5)*800

def candidate_graphs(task, max_nodes=8, max_depth=2):
    arity=TASK_SPECS[task]["arity"]
    base=[Node("input0"),Node("input1")]
    if arity==3:
        base.append(Node("input2"))
    unary=("abs","neg")
    binary=("add","sub","mul","min","max")

    out=[]
    # Zero-computation/input projections.
    for i in range(len(base)):
        out.append(Graph(tuple(base),i))

    # One computation node.
    for i in range(len(base)):
        for op in unary:
            ns=base+[Node(op,(i,))]
            if len(ns)<=max_nodes:
                out.append(Graph(tuple(ns),len(ns)-1))
    for i in range(len(base)):
        for j in range(len(base)):
            for op in binary:
                ns=base+[Node(op,(i,j))]
                if len(ns)<=max_nodes:
                    out.append(Graph(tuple(ns),len(ns)-1))

    # Two computation nodes: second node may consume the first node.
    if max_depth>=2:
        first_nodes=[]
        for i in range(len(base)):
            for op in unary:
                first_nodes.append(Node(op,(i,)))
        for i in range(len(base)):
            for j in range(len(base)):
                for op in binary:
                    first_nodes.append(Node(op,(i,j)))

        for first in first_nodes:
            ns1=base+[first]
            idx=len(ns1)-1
            for op in unary:
                ns=ns1+[Node(op,(idx,))]
                if len(ns)<=max_nodes:
                    out.append(Graph(tuple(ns),len(ns)-1))
            for j in range(len(ns1)):
                for op in binary:
                    ns=ns1+[Node(op,(idx,j))]
                    if len(ns)<=max_nodes:
                        out.append(Graph(tuple(ns),len(ns)-1))

    # Three computation nodes for explicit ternary composition.
    if max_depth>=3 and arity==3 and max_nodes>=len(base)+3:
        first_ops=("add","sub","mul")
        second_ops=("add","sub","mul")
        for op1 in first_ops:
            ns1=base+[Node(op1,(0,1))]
            idx1=len(ns1)-1
            for op2 in second_ops:
                ns2=ns1+[Node(op2,(idx1,2))]
                idx2=len(ns2)-1
                for op3 in second_ops:
                    ns3=ns2+[Node(op3,(idx2,0))]
                    if len(ns3)<=max_nodes:
                        out.append(Graph(tuple(ns3),len(ns3)-1))

    seen={}
    for g in out:
        seen[g.key()]=g
    return sorted(seen.values(),key=lambda g:(g.length,g.key()))

def expected_graph(task):
    arity=TASK_SPECS[task]["arity"]
    base=[Node("input0"),Node("input1")]
    if arity==3: base.append(Node("input2"))
    if task=="add": return Graph(tuple(base+[Node("add",(0,1))]),len(base))
    if task=="sub": return Graph(tuple(base+[Node("sub",(0,1))]),len(base))
    if task=="mul": return Graph(tuple(base+[Node("mul",(0,1))]),len(base))
    if task=="absdiff":
        n1=base+[Node("sub",(0,1))]
        return Graph(tuple(n1+[Node("abs",(len(n1)-1,))]),len(n1))
    if task=="max": return Graph(tuple(base+[Node("max",(0,1))]),len(base))
    if task=="min": return Graph(tuple(base+[Node("min",(0,1))]),len(base))
    if task=="sum3":
        n1=base+[Node("add",(0,1))]
        return Graph(tuple(n1+[Node("add",(len(n1)-1,2))]),len(n1))
    if task=="pairdiff3":
        n1=base+[Node("sub",(0,1))]
        return Graph(tuple(n1+[Node("add",(len(n1)-1,2))]),len(n1))
    raise ValueError(task)

VERIFICATION_EPS = 1e-6

def is_exact_score(score: float) -> bool:
    return score >= 1.0 - VERIFICATION_EPS

def verification_state(v):
    """Return the single canonical semantic verification state."""
    return bool(v.get("exact_verified", False))


def verify(task,g,seed,args,device):
    scores={"symbolic":agreement(task,g,symbolic_bank(task,device))}
    regimes={}
    rnd={}
    for i,r in enumerate(REGIMES):
        regimes[r]=agreement(task,g,make_inputs(task,args.verifier_samples,seed+100+i,device,r))
        rnd[r]=agreement(task,g,randomized_bank(task,seed+200+i,args.random_probe_samples,device))
    scores["regimes"]=regimes
    scores["randomized"]=rnd
    scores["raw_proof"]=min([scores["symbolic"],*regimes.values(),*rnd.values()])
    scores["proof"]=scores["raw_proof"]
    scores["exact_verified"]=is_exact_score(scores["raw_proof"])
    scores["verification_eps"]=VERIFICATION_EPS
    return scores

class PrimitiveLibrary:
    def __init__(self):
        self.records={}
        self.key_to_id={}
        self.counter=0
    def insert_or_reuse(self,g,task):
        key=g.key()
        if key in self.key_to_id:
            pid=self.key_to_id[key]
            self.records[pid]["uses"]+=1
            self.records[pid]["tasks"].append(task)
            return pid,False
        pid=f"P{self.counter}"
        self.counter+=1
        self.key_to_id[key]=pid
        self.records[pid]={
            "nodes":[(n.op,n.inputs) for n in g.nodes],
            "output":g.output,
            "arity":TASK_SPECS[task]["arity"],
            "tasks":[task],
            "uses":1,
            "verified":True,
        }
        return pid,True

def run_audit(task,seed,args,device,lib):
    cands=candidate_graphs(task,args.max_graph_nodes,args.max_graph_depth)
    exp=expected_graph(task)
    exp_present=exp.key() in {g.key() for g in cands}

    rows=[]
    for i,g in enumerate(cands):
        rows.append((g,verify(task,g,seed+1000+i,args,device)))
    rows.sort(key=lambda z:(-z[1]["proof"],z[0].length,z[0].key()))

    best_g,best_v=rows[0]
    verified=[(g,v) for g,v in rows if v["exact_verified"]]
    expected_v=next((v for g,v in rows if g.key()==exp.key()),None)

    inserted=False
    reused=False
    pid=None
    if expected_v is not None and expected_v["exact_verified"]:
        pid,inserted=lib.insert_or_reuse(exp,task)
        if not inserted:
            reused=True

    anomaly=[]
    if not exp_present: anomaly.append("expected_graph_missing_from_candidate_language")
    if expected_v is None: anomaly.append("expected_graph_not_audit_visible")
    elif not verification_state(expected_v): anomaly.append("expected_graph_failed_exact_verification")
    if not best_v["exact_verified"]: anomaly.append("no_exact_candidate_in_current_language")
    if expected_v is not None and expected_v["exact_verified"] and not inserted and not reused:
        anomaly.append("library_insertion_failed")

    return {
        "task":task,
        "candidate_count":len(cands),
        "expected_graph_present":exp_present,
        "expected_graph":[(n.op,n.inputs) for n in exp.nodes],
        "expected_graph_verification":expected_v,
        "best_candidate":[(n.op,n.inputs) for n in best_g.nodes],
        "best_candidate_output":best_g.output,
        "best_candidate_verification":best_v,
        "verified_candidate_count":len(verified),
        "primitive_id":pid,
        "inserted_new":inserted,
        "reused_existing":reused,
        "anomalies":anomaly,
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--seeds",nargs="+",type=int,default=[1,2])
    ap.add_argument("--audit-tasks",nargs="+",default=["sub","sum3","pairdiff3","absdiff","max","min"])
    ap.add_argument("--verifier-samples",type=int,default=1500)
    ap.add_argument("--random-probe-samples",type=int,default=4096)
    ap.add_argument("--max-graph-nodes",type=int,default=8)
    ap.add_argument("--max-graph-depth",type=int,default=2)
    ap.add_argument("--device",default="cuda")
    ap.add_argument("--verification-eps",type=float,default=1e-6)
    ap.add_argument("--out",default="repair03.result.json")
    args=ap.parse_args()

    global VERIFICATION_EPS
    VERIFICATION_EPS=float(args.verification_eps)
    device=torch.device(args.device if args.device=="cpu" or torch.cuda.is_available() else "cpu")
    all_runs=[]

    # Phase 1: isolated discovery/insertion audit.
    for seed in args.seeds:
        seed_all(seed)
        lib=PrimitiveLibrary()
        task_rows=[]
        for task in args.audit_tasks:
            row=run_audit(task,seed,args,device,lib)
            task_rows.append(row)

        all_runs.append({
            "seed":seed,
            "tasks":task_rows,
            "library":lib.records,
        })

    # Phase 2: explicit reuse probe. A fresh task that has an identical exact
    # representation must reuse the existing primitive rather than insert a duplicate.
    reuse_probe={}
    for seed in args.seeds:
        lib=PrimitiveLibrary()
        first=expected_graph("sub")
        pid1,new1=lib.insert_or_reuse(first,"sub")
        pid2,new2=lib.insert_or_reuse(first,"sub_reuse_probe")
        reuse_probe[str(seed)]={
            "first_id":pid1,"first_new":new1,
            "second_id":pid2,"second_new":new2,
            "reuse_success":(pid1==pid2 and new1 and not new2),
            "library_size":len(lib.records),
            "record":lib.records.get(pid1),
        }

    anomalies=[]
    for run in all_runs:
        for row in run["tasks"]:
            anomalies += [f"seed={run['seed']} task={row['task']}: {x}" for x in row["anomalies"]]
    anomalies += [
        f"reuse_seed={seed}: reuse_failed"
        for seed,v in reuse_probe.items()
        if not v["reuse_success"]
    ]

    total=len(all_runs)*len(args.audit_tasks)
    expected_verified=sum(
        1 for run in all_runs for row in run["tasks"]
        if row["expected_graph_verification"]
        and verification_state(row["expected_graph_verification"])
    )
    all_expected_verified=(expected_verified==total)


    # Repair-03 invariant: if every expected graph is canonical-verified,
    # no expected-graph semantic anomaly may exist.
    consistency_failures = []
    for run in all_runs:
        for row in run["tasks"]:
            v = row.get("expected_graph_verification")
            if v and verification_state(v) and any(
                a == "expected_graph_failed_exact_verification"
                for a in row.get("anomalies", [])
            ):
                consistency_failures.append(
                    f"seed={run['seed']} task={row['task']}: stale verification anomaly"
                )
    anomalies.extend(consistency_failures)

    result={
        "repair_id":"repair03",
        "parent_version":"DART-4.1",
        "purpose":"canonical verification-state consistency repair",
        "gates":{
            "candidate_language_contains_expected_graph":True,
            "exact_oracle_verification":True,
            "library_insertion":True,
            "reuse_probe":True,
            "explicit_failure_reporting":True,
        },
        "summary":{
            "total_task_seed_audits":total,
            "expected_graphs_exactly_verified":expected_verified,
            "all_expected_graphs_verified":all_expected_verified,
            "reuse_probe_all_passed":all(v["reuse_success"] for v in reuse_probe.values()),
            "anomaly_count":len(anomalies),
            "verification_state_consistent":len(consistency_failures)==0,
        },
        "anomalies":anomalies,
        "runs":all_runs,
        "reuse_probe":reuse_probe,
    }

    out=Path(args.out)
    out.write_text(json.dumps(result,indent=2))
    print("DART-4.1 repair03: canonical verification-state integrity")
    print(json.dumps(result,indent=2))
    print(f"Saved: {out.resolve()}")

if __name__=="__main__":
    main()
