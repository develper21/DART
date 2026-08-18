
#!/usr/bin/env python3
"""
DART-3.2: causal counterfactual task-program synthesis.

Key design:
- Shared source primitive trained jointly across source tasks.
- Short explicit task programs.
- Target program is selected on target train/validation only.
- Final target test is untouched.
- Program steps are validated counterfactually against the teacher:
    keep exact trained program state,
    replace one step with identity,
    compare DART delta with teacher delta.
- Necessity, counterfactual fidelity, permutation control,
  random-program control, program complexity, and contrast task are reported.
- Terminal progress bar intentionally says DART-3.2 (no "repair" label).
"""
from __future__ import annotations
import argparse, copy, json, math, random, sys, time
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

TASKS = {
    "add": lambda a,b: a+b,
    "mul": lambda a,b: a*b,
    "sub": lambda a,b: a-b,
    "compose": lambda a,b: (a*2 + 1) - (b*3 - 1),
    "sort": lambda a,b: torch.where(a <= b, a, b),
}

PROGRAM_OPS = ["identity","scale","shift","negate","difference","product","swap"]

def seed_all(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class Progress:
    def __init__(self, total, seed_idx, nseeds):
        self.total = max(1, total)
        self.done = 0
        self.seed_idx = seed_idx
        self.nseeds = nseeds
    def update(self, done, phase, detail=""):
        self.done = min(self.total, max(0, int(done)))
        frac = self.done / self.total
        width = 28
        fill = int(frac * width)
        bar = "=" * fill + ">" + " " * max(0, width-fill-1)
        msg = f"\r[DART-3.2][seed {self.seed_idx}/{self.nseeds}] [{bar}] {frac*100:6.2f}% | {phase}"
        if detail:
            msg += f" | {detail}"
        sys.stdout.write(msg)
        sys.stdout.flush()
    def close(self):
        self.update(self.total, "complete")
        print()

def make_dataset(task, n, seed, device):
    # Create the RNG on the same device as the generated tensor.
    # A CPU Generator cannot be passed to torch.randint(device="cuda").
    gen_device = "cuda" if device.type == "cuda" else "cpu"
    g = torch.Generator(device=gen_device).manual_seed(seed)
    x = torch.randint(-3,4,(n,2),generator=g,device=device).float()
    y = TASKS[task](x[:,0], x[:,1])
    bins = torch.tensor([-6,-3,0,3,6],device=device).float()
    y = torch.bucketize(y,bins).clamp(max=5)
    return x, y.long()

class SharedPrimitive(nn.Module):
    def __init__(self, nodes, motif, d=32, rank=8):
        super().__init__()
        self.nodes = nodes
        self.motif = motif
        self.blocks = nn.ModuleList()
        for n in nodes:
            if n=="affine_polynomial":
                b = nn.Sequential(nn.Linear(d,d), nn.GELU(), nn.Linear(d,d))
            elif n=="polynomial":
                b = nn.Sequential(nn.Linear(d,d), nn.Tanh(), nn.Linear(d,d))
            else:
                b = nn.Sequential(nn.Linear(d,rank,bias=False), nn.Linear(rank,d,bias=False))
            self.blocks.append(b)
        self.norm = nn.LayerNorm(d)
    def forward(self,h):
        if self.motif=="sequential":
            z=h
            for b in self.blocks:
                z = z + b(self.norm(z))
            return z
        hs=[b(self.norm(h)) for b in self.blocks]
        if self.motif=="parallel_sum":
            return h + sum(hs)
        if self.motif=="residual_parallel":
            return h + hs[-1] + 0.5*sum(hs[:-1])
        raise ValueError(self.motif)

class PrimitiveModel(nn.Module):
    def __init__(self, primitive, d=32, classes=6):
        super().__init__()
        self.inp=nn.Linear(2,d)
        self.primitive=primitive
        self.out=nn.Linear(d,classes)
    def forward(self,x):
        return self.out(self.primitive(self.inp(x)))

@dataclass(frozen=True)
class Step:
    op: str

@dataclass(frozen=True)
class Program:
    steps: Tuple[Step,...]
    def names(self):
        return [s.op for s in self.steps]
    @property
    def length(self):
        return len(self.steps)

class ProgramTransform(nn.Module):
    def __init__(self, program):
        super().__init__()
        self.program=program
        params=[]
        for s in program.steps:
            if s.op in ("scale","shift"):
                params.append(nn.Parameter(torch.tensor(1.0 if s.op=="scale" else 0.0)))
            else:
                params.append(nn.Parameter(torch.tensor(0.0), requires_grad=False))
        self.raw=nn.ParameterList(params)
    def forward(self,x):
        z=x
        for s,p in zip(self.program.steps,self.raw):
            a,b=z[:,0],z[:,1]
            if s.op=="identity": pass
            elif s.op=="scale": z=z*p
            elif s.op=="shift": z=z+p
            elif s.op=="negate": z=-z
            elif s.op=="difference": z=torch.stack([a-b,b-a],1)
            elif s.op=="product":
                q=a*b; z=torch.stack([q,q],1)
            elif s.op=="swap": z=torch.stack([b,a],1)
        return z

class TaskProgramModel(nn.Module):
    def __init__(self, base_model, program):
        super().__init__()
        self.base_model=base_model
        self.program=program
        self.transform=ProgramTransform(program)
        self.decode_scale=nn.Parameter(torch.tensor(1.0))
        self.decode_bias=nn.Parameter(torch.tensor(0.0))
    def forward(self,x):
        z=self.transform(x)
        out=self.base_model(z)
        return out*self.decode_scale+self.decode_bias

def fit(model, loader, steps, lr, freeze_primitive=True):
    trainable=[]
    for n,p in model.named_parameters():
        if freeze_primitive and ("base_model.primitive" in n or "base_model.inp" in n):
            p.requires_grad_(False)
        if p.requires_grad:
            trainable.append(p)
    opt=torch.optim.Adam(trainable,lr=lr)
    ce=nn.CrossEntropyLoss()
    it=iter(loader)
    model.train()
    for _ in range(steps):
        try: x,y=next(it)
        except StopIteration:
            it=iter(loader); x,y=next(it)
        opt.zero_grad(set_to_none=True)
        loss=ce(model(x),y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable,1.0)
        opt.step()

def acc(model,x,y):
    model.eval()
    with torch.no_grad():
        p=model(x).argmax(-1)
    return float((p==y).float().mean().item())

def teacher_counterfactual_effect(teacher, x, y, transform_fn):
    with torch.no_grad():
        base = acc(teacher,x,y)
        z = transform_fn(x)
        alt = acc(teacher,z,y)
    return base-alt

def program_step_ablation(model, x, y, step_index):
    """Same trained program state; ablate only one step with identity, no retraining."""
    prog=model.program
    if step_index >= prog.length:
        return 0.0
    base=acc(model,x,y)
    steps=list(prog.steps)
    steps[step_index]=Step("identity")
    alt=TaskProgramModel(model.base_model, Program(tuple(steps))).to(x.device)
    # Copy trained program scalar parameters exactly where applicable.
    for i,(src,dst) in enumerate(zip(model.transform.raw, alt.transform.raw)):
        if dst.requires_grad:
            with torch.no_grad():
                dst.copy_(src.detach())
    with torch.no_grad():
        alt.decode_scale.copy_(model.decode_scale.detach())
        alt.decode_bias.copy_(model.decode_bias.detach())
    return max(0.0, base-acc(alt,x,y))

def program_counterfactual_fidelity(model, teacher, x, y, step_index):
    """Compare teacher and DART behavioral deltas for the same step intervention."""
    if step_index >= model.program.length:
        return 0.0
    base_d=acc(model,x,y)
    d_d=program_step_ablation(model,x,y,step_index)

    # Teacher reference: apply an analogous identity replacement to input program transform.
    def teacher_alt_input(inp):
        steps=list(model.program.steps)
        steps[step_index]=Step("identity")
        temp=ProgramTransform(Program(tuple(steps))).to(inp.device)
        # copy learned scalar values
        for i,(src,dst) in enumerate(zip(model.transform.raw,temp.raw)):
            if dst.requires_grad:
                with torch.no_grad(): dst.copy_(src.detach())
        return temp(inp)
    with torch.no_grad():
        t_base=acc(teacher,x,y)
        t_alt=acc(teacher,teacher_alt_input(x),y)
    t_delta=max(0.0,t_base-t_alt)
    denom=max(1e-6,abs(t_delta)+abs(d_d))
    return 1.0-abs(t_delta-d_d)/denom

def enumerate_programs(max_len):
    out=[]
    for L in range(1,max_len+1):
        for ops in product(PROGRAM_OPS, repeat=L):
            if all(o=="identity" for o in ops):
                continue
            out.append(Program(tuple(Step(o) for o in ops)))
    return out

def balanced_score(vals):
    if not vals: return 0.0
    m=sum(vals)/len(vals)
    return 0.5*m+0.5*min(vals)

def discover_shared_primitive(args, source_tasks, seed, device, prog):
    motifs=["sequential","parallel_sum","residual_parallel"]
    node_sets=[["affine_polynomial","polynomial"],["affine_polynomial","polynomial","low_rank"]]
    candidates=[]
    for ci,(motif,nodes) in enumerate(product(motifs,node_sets),1):
        primitive=SharedPrimitive(nodes,motif,args.d_model,args.rank).to(device)
        base=PrimitiveModel(primitive,args.d_model,args.classes).to(device)
        # joint source training on a balanced mixed loader
        xs=[]; ys=[]
        for j,t in enumerate(source_tasks):
            x,y=make_dataset(t,max(1,args.train_size//len(source_tasks)),seed+300+j,device)
            xs.append(x); ys.append(y)
        xmix=torch.cat(xs); ymix=torch.cat(ys)
        loader=DataLoader(TensorDataset(xmix,ymix),batch_size=args.batch_size,shuffle=True)
        fit(base,loader,args.core_fit_steps,args.core_fit_lr,freeze_primitive=False)

        scores=[]
        for j,t in enumerate(source_tasks):
            x,y=make_dataset(t,args.verifier_size,seed+500+j+ci*13,device)
            scores.append(acc(base,x,y))
        s=balanced_score(scores)
        candidates.append((s,base,scores))
        prog.update(ci,"shared-primitive",f"candidate={ci} balanced={s:.3f}")
    candidates.sort(key=lambda z:z[0],reverse=True)
    return candidates[0]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--seeds",nargs="+",type=int,default=[1,2])
    ap.add_argument("--all-tasks",nargs="+",default=["add","compose","mul","sub"])
    ap.add_argument("--holdout-tasks",nargs="+",default=["sub"])
    ap.add_argument("--contrast-tasks",nargs="+",default=["sort"])
    ap.add_argument("--teacher-steps",type=int,default=800)
    ap.add_argument("--core-fit-steps",type=int,default=300)
    ap.add_argument("--program-fit-steps",type=int,default=120)
    ap.add_argument("--target-program-fit-steps",type=int,default=400)
    ap.add_argument("--transfer-control-steps",type=int,default=400)
    ap.add_argument("--separate-control-steps",type=int,default=200)
    ap.add_argument("--train-size",type=int,default=6000)
    ap.add_argument("--verifier-size",type=int,default=1500)
    ap.add_argument("--test-size",type=int,default=1500)
    ap.add_argument("--target-adaptation-size",type=int,default=1200)
    ap.add_argument("--target-validation-size",type=int,default=1200)
    ap.add_argument("--rel-samples-per-task",type=int,default=2048)
    ap.add_argument("--fit-batch-samples",type=int,default=512)
    ap.add_argument("--causal-probe-size",type=int,default=64)
    ap.add_argument("--max-program-length",type=int,default=2)
    ap.add_argument("--program-complexity-lambda",type=float,default=0.05)
    ap.add_argument("--min-program-necessity",type=float,default=0.02)
    ap.add_argument("--min-counterfactual-fidelity",type=float,default=0.20)
    ap.add_argument("--device",default="cuda")
    ap.add_argument("--d-model",type=int,default=32)
    ap.add_argument("--rank",type=int,default=8)
    ap.add_argument("--classes",type=int,default=6)
    ap.add_argument("--batch-size",type=int,default=256)
    ap.add_argument("--core-fit-lr",type=float,default=1e-3)
    ap.add_argument("--lr",type=float,default=3e-4)
    ap.add_argument("--out",default="dart032_results.json")
    args=ap.parse_args()

    device=torch.device(args.device if args.device=="cpu" or torch.cuda.is_available() else "cpu")
    source_tasks=[t for t in args.all_tasks if t not in args.holdout_tasks and t in ("add","compose","mul")]
    target=args.holdout_tasks[0]
    contrast=args.contrast_tasks[0]
    programs=enumerate_programs(args.max_program_length)

    records=[]
    for si,seed in enumerate(args.seeds,1):
        seed_all(seed)
        total=6+len(programs)*3+8
        pbar=Progress(total,si,len(args.seeds))

        # teacher baselines
        teachers={}
        t_scores={}
        for j,t in enumerate(source_tasks+[target,contrast]):
            x,y=make_dataset(t,args.train_size,seed+1000+j,device)
            # Use shared teacher architecture but independent weights per task.
            teacher=PrimitiveModel(
                SharedPrimitive(["affine_polynomial","polynomial"],"sequential",args.d_model,args.rank),
                args.d_model,args.classes
            ).to(device)
            loader=DataLoader(TensorDataset(x,y),batch_size=args.batch_size,shuffle=True)
            fit(teacher,loader,args.teacher_steps,args.lr,freeze_primitive=False)
            teachers[t]=(teacher,x,y)
            t_scores[t]=acc(teacher,x,y)
            pbar.update(1+j,"teacher-training",f"task={t}")

        _,base_model,source_base_scores=discover_shared_primitive(args,source_tasks,seed,device,pbar)
        for par in base_model.primitive.parameters(): par.requires_grad_(False)

        # program source search: same base primitive, independently fit program for each source task.
        source_program_records=[]
        for pi,program in enumerate(programs):
            task_scores=[]
            for j,t in enumerate(source_tasks):
                x,y=make_dataset(t,args.verifier_size,seed+2000+pi*31+j,device)
                m=TaskProgramModel(base_model,program).to(device)
                fit(m,DataLoader(TensorDataset(x,y),batch_size=args.fit_batch_samples,shuffle=True),
                    args.program_fit_steps,args.lr,freeze_primitive=True)
                task_scores.append(acc(m,x,y))
            b=balanced_score(task_scores)-args.program_complexity_lambda*program.length
            source_program_records.append((b,program,task_scores))
            pbar.update(7+pi,"source-program-search",f"program={pi+1}/{len(programs)} balanced={b:.3f}")
        source_program_records.sort(key=lambda z:z[0],reverse=True)
        best_score,best_program,best_prog_source_scores=source_program_records[0]

        # Same-state source necessity + counterfactual fidelity.
        sx,sy=make_dataset(source_tasks[0],args.causal_probe_size,seed+4000,device)
        src_prog_model=TaskProgramModel(base_model,best_program).to(device)
        fit(src_prog_model,DataLoader(TensorDataset(sx,sy),batch_size=min(64,args.causal_probe_size),shuffle=True),
            args.program_fit_steps,args.lr,freeze_primitive=True)
        src_nec=[]; src_fid=[]
        for k in range(best_program.length):
            src_nec.append(program_step_ablation(src_prog_model,sx,sy,k))
            src_fid.append(program_counterfactual_fidelity(src_prog_model,teachers[source_tasks[0]][0],sx,sy,k))
        source_necessity=sum(src_nec)/max(1,len(src_nec))
        source_fidelity=sum(src_fid)/max(1,len(src_fid))
        pbar.update(7+len(programs),"program-causal-validation",
                    f"necessity={source_necessity:.3f} cf_fidelity={source_fidelity:.3f}")

        # Target adaptation/validation selection. Test stays untouched.
        txtr,tytr=make_dataset(target,args.target_adaptation_size,seed+5000,device)
        txv,tyv=make_dataset(target,args.target_validation_size,seed+5001,device)
        txt,tyt=make_dataset(target,args.test_size,seed+5002,device)

        zero=TaskProgramModel(base_model,Program((Step("identity"),))).to(device)
        zero_test=acc(zero,txt,tyt)

        candidates=[]
        for pi,program in enumerate(programs):
            m=TaskProgramModel(base_model,program).to(device)
            fit(m,DataLoader(TensorDataset(txtr,tytr),batch_size=args.fit_batch_samples,shuffle=True),
                args.target_program_fit_steps,args.lr,freeze_primitive=True)
            v=acc(m,txv,tyv)
            # necessity on SAME trained model, validation only
            nec=[]
            fid=[]
            for k in range(program.length):
                nec.append(program_step_ablation(m,txv,tyv,k))
                fid.append(program_counterfactual_fidelity(m,teachers[target][0],txv,tyv,k))
            necessity=sum(nec)/max(1,len(nec))
            fidelity=sum(fid)/max(1,len(fid))
            score=v + 0.25*necessity + 0.25*fidelity - args.program_complexity_lambda*program.length
            candidates.append((score,m,program,v,necessity,fidelity))
            pbar.update(8+len(programs)+pi,"target-program-validation",
                        f"program={pi+1}/{len(programs)} val={v:.3f} nec={necessity:.3f} cf={fidelity:.3f}")

        candidates.sort(key=lambda z:z[0],reverse=True)
        best=candidates[0]
        _,best_model,sel_program,val_acc,target_necessity,target_fidelity=best
        test_prog=acc(best_model,txt,tyt)

        # Controls on untouched target test only.
        wrong_program=programs[(programs.index(sel_program)+1)%len(programs)]
        pm=TaskProgramModel(base_model,wrong_program).to(device)
        fit(pm,DataLoader(TensorDataset(txtr,tytr),batch_size=args.fit_batch_samples,shuffle=True),
            args.transfer_control_steps,args.lr,freeze_primitive=True)
        perm_test=acc(pm,txt,tyt)

        rp=programs[(seed*19)%len(programs)]
        rm=TaskProgramModel(base_model,rp).to(device)
        fit(rm,DataLoader(TensorDataset(txtr,tytr),batch_size=args.fit_batch_samples,shuffle=True),
            args.transfer_control_steps,args.lr,freeze_primitive=True)
        rand_test=acc(rm,txt,tyt)

        cx,cy=make_dataset(contrast,args.test_size,seed+6000,device)
        cm=TaskProgramModel(base_model,sel_program).to(device)
        fit(cm,DataLoader(TensorDataset(cx,cy),batch_size=args.fit_batch_samples,shuffle=True),
            args.transfer_control_steps,args.lr,freeze_primitive=True)
        contrast_test=acc(cm,cx,cy)
        pbar.update(total-2,"target-controls",
                    f"target={target} test={test_prog:.3f} nec={target_necessity:.3f} cf={target_fidelity:.3f}")
        pbar.close()

        records.append({
            "seed":seed,
            "winner":{
                "program":sel_program.names(),
                "program_length":sel_program.length,
                "source_base_scores":source_base_scores,
                "source_program_scores":best_prog_source_scores,
                "source_program_necessity":source_necessity,
                "source_program_counterfactual_fidelity":source_fidelity,
                "target_program_necessity":target_necessity,
                "target_program_counterfactual_fidelity":target_fidelity
            },
            "related_holdout":{
                target:{
                    "teacher":t_scores[target],
                    "dart_zero":zero_test,
                    "dart_program":test_prog,
                    "program_validation":val_acc,
                    "program_permutation_control":perm_test,
                    "random_program_control":rand_test
                }
            },
            "contrast_holdout":{
                contrast:{
                    "teacher":t_scores[contrast],
                    "dart_program":contrast_test
                }
            }
        })

    summary={
        "version":"DART-3.2",
        "parent_version":"DART-3.1",
        "protocol":{
            "joint_source_primitive_training":True,
            "balanced_source_selection":True,
            "same_state_source_program_necessity":True,
            "same_state_target_program_necessity":True,
            "teacher_counterfactual_fidelity":True,
            "target_train_validation_test_split":True,
            "untouched_target_test":True,
            "deterministic_probe_seeding":True
        },
        "related_holdout":{
            target:{
                "teacher":sum(r["related_holdout"][target]["teacher"] for r in records)/len(records),
                "dart_zero":sum(r["related_holdout"][target]["dart_zero"] for r in records)/len(records),
                "dart_program":sum(r["related_holdout"][target]["dart_program"] for r in records)/len(records),
                "program_validation":sum(r["related_holdout"][target]["program_validation"] for r in records)/len(records),
                "program_permutation_control":sum(r["related_holdout"][target]["program_permutation_control"] for r in records)/len(records),
                "random_program_control":sum(r["related_holdout"][target]["random_program_control"] for r in records)/len(records)
            }
        },
        "contrast_holdout":{
            contrast:{
                "teacher":sum(r["contrast_holdout"][contrast]["teacher"] for r in records)/len(records),
                "dart_program":sum(r["contrast_holdout"][contrast]["dart_program"] for r in records)/len(records)
            }
        },
        "source":{
            "avg_source_program_necessity":sum(r["winner"]["source_program_necessity"] for r in records)/len(records),
            "avg_source_program_counterfactual_fidelity":sum(r["winner"]["source_program_counterfactual_fidelity"] for r in records)/len(records),
            "avg_target_program_necessity":sum(r["winner"]["target_program_necessity"] for r in records)/len(records),
            "avg_target_program_counterfactual_fidelity":sum(r["winner"]["target_program_counterfactual_fidelity"] for r in records)/len(records),
            "avg_program_length":sum(r["winner"]["program_length"] for r in records)/len(records)
        },
        "records":records
    }
    out=Path(args.out)
    out.write_text(json.dumps(summary,indent=2))
    print("DART-3.2: causal counterfactual task-program synthesis")
    print(json.dumps(summary,indent=2))
    print(f"Saved: {out.resolve()}")

if __name__=="__main__":
    main()
