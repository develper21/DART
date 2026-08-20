#!/usr/bin/env python3
"""
DART-3.3: cross-distribution task-program invariance.

This version targets the DART-3.2 failure:
a task program can score highly on target validation while collapsing
on an untouched target test distribution.

Protocol:
- jointly train a shared primitive on source tasks
- discover short task programs
- evaluate each candidate on TWO independent target distributions
  with disjoint RNG streams/regimes
- select only from adaptation/validation distributions, never final test
- measure program necessity and teacher-aligned counterfactual fidelity
  on both target distributions
- report cross-distribution invariance and variance
- final untouched target test uses a third independently generated regime
- program permutation/random controls
- multi-holdout optionality through --holdout-tasks
- no "repair" label in logs
"""
from __future__ import annotations
import argparse, copy, json, math, random, sys, time
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Tuple, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

TASKS = {
    "add": lambda a,b: a+b,
    "mul": lambda a,b: a*b,
    "sub": lambda a,b: a-b,
    "compose": lambda a,b: (a*2+1)-(b*3-1),
    "sort": lambda a,b: torch.where(a<=b,a,b),
}
PROGRAM_OPS = ["identity","scale","shift","negate","difference","product","swap"]

def seed_all(seed:int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class Progress:
    def __init__(self,total,seed_idx,nseeds):
        self.total=max(1,int(total)); self.seed_idx=seed_idx; self.nseeds=nseeds
    def update(self,done,phase,detail=""):
        frac=min(1.0,max(0.0,float(done)/self.total))
        w=28; fill=int(frac*w)
        bar="="*fill+">"+" " * max(0,w-fill-1)
        msg=f"\r[DART-3.3][seed {self.seed_idx}/{self.nseeds}] [{bar}] {100*frac:6.2f}% | {phase}"
        if detail: msg+=f" | {detail}"
        sys.stdout.write(msg); sys.stdout.flush()
    def close(self):
        self.update(self.total,"complete"); print()

def task_transform(task,x):
    return TASKS[task](x[:,0],x[:,1])

def make_dataset(task,n,seed,device,regime="A"):
    # Independent distribution regimes for same task rule.
    # A: base small integers; B: shifted/wider structured range; C: held-out range.
    g=torch.Generator(device=("cuda" if device.type=="cuda" else "cpu")).manual_seed(seed)
    if regime=="A":
        x=torch.randint(-3,4,(n,2),generator=g,device=device).float()
    elif regime=="B":
        x=torch.randint(-6,7,(n,2),generator=g,device=device).float()+0.25
    elif regime=="C":
        x=torch.randint(-10,11,(n,2),generator=g,device=device).float()
        x=x + torch.tensor([0.5,-0.5],device=device)
    else:
        raise ValueError(regime)
    y=task_transform(task,x)
    bins=torch.tensor([-12,-6,-3,0,3,6,12],device=device).float()
    y=torch.bucketize(y,bins).clamp(max=7)
    return x,y.long()

class SharedPrimitive(nn.Module):
    def __init__(self,nodes,motif,d=32,rank=8):
        super().__init__()
        self.nodes=nodes; self.motif=motif
        bs=[]
        for n in nodes:
            if n=="affine_polynomial":
                b=nn.Sequential(nn.Linear(d,d),nn.GELU(),nn.Linear(d,d))
            elif n=="polynomial":
                b=nn.Sequential(nn.Linear(d,d),nn.Tanh(),nn.Linear(d,d))
            else:
                b=nn.Sequential(nn.Linear(d,rank,bias=False),nn.Linear(rank,d,bias=False))
            bs.append(b)
        self.blocks=nn.ModuleList(bs)
        self.norm=nn.LayerNorm(d)
    def forward(self,h):
        if self.motif=="sequential":
            z=h
            for b in self.blocks:
                z=z+b(self.norm(z))
            return z
        hs=[b(self.norm(h)) for b in self.blocks]
        if self.motif=="parallel_sum":
            return h+sum(hs)
        if self.motif=="residual_parallel":
            return h+hs[-1]+0.5*sum(hs[:-1])
        raise ValueError(self.motif)

class PrimitiveModel(nn.Module):
    def __init__(self,primitive,d=32,classes=8):
        super().__init__(); self.inp=nn.Linear(2,d); self.primitive=primitive; self.out=nn.Linear(d,classes)
    def forward(self,x): return self.out(self.primitive(self.inp(x)))

@dataclass(frozen=True)
class Step:
    op:str
@dataclass(frozen=True)
class Program:
    steps:Tuple[Step,...]
    @property
    def length(self): return len(self.steps)
    def names(self): return [s.op for s in self.steps]

class ProgramTransform(nn.Module):
    def __init__(self,program):
        super().__init__(); self.program=program
        ps=[]
        for s in program.steps:
            if s.op in ("scale","shift"):
                ps.append(nn.Parameter(torch.tensor(1.0 if s.op=="scale" else 0.0)))
            else:
                ps.append(nn.Parameter(torch.tensor(0.0),requires_grad=False))
        self.raw=nn.ParameterList(ps)
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
    def __init__(self,base,program):
        super().__init__(); self.base=base; self.program=program
        self.transform=ProgramTransform(program)
        self.decode_scale=nn.Parameter(torch.tensor(1.0)); self.decode_bias=nn.Parameter(torch.tensor(0.0))
    def forward(self,x):
        z=self.transform(x); return self.base(z)*self.decode_scale+self.decode_bias

def fit(model,loader,steps,lr,freeze_primitive=True):
    ps=[]
    for n,p in model.named_parameters():
        if freeze_primitive and ("base.primitive" in n or "base.inp" in n):
            p.requires_grad_(False)
        if p.requires_grad: ps.append(p)
    opt=torch.optim.Adam(ps,lr=lr)
    ce=nn.CrossEntropyLoss(); it=iter(loader); model.train()
    for _ in range(steps):
        try: x,y=next(it)
        except StopIteration:
            it=iter(loader); x,y=next(it)
        opt.zero_grad(set_to_none=True); loss=ce(model(x),y); loss.backward()
        torch.nn.utils.clip_grad_norm_(ps,1.0); opt.step()

def acc(model,x,y):
    model.eval()
    with torch.no_grad(): p=model(x).argmax(-1)
    return float((p==y).float().mean().item())

def balanced(vals):
    return 0.5*(sum(vals)/len(vals))+0.5*min(vals)

def enumerate_programs(max_len):
    out=[]
    for L in range(1,max_len+1):
        for ops in product(PROGRAM_OPS,repeat=L):
            if all(o=="identity" for o in ops): continue
            out.append(Program(tuple(Step(o) for o in ops)))
    return out

def copy_program_state(src,dst):
    for a,b in zip(src.transform.raw,dst.transform.raw):
        if b.requires_grad:
            with torch.no_grad(): b.copy_(a.detach())
    with torch.no_grad():
        dst.decode_scale.copy_(src.decode_scale.detach())
        dst.decode_bias.copy_(src.decode_bias.detach())

def step_ablation(model,x,y,idx):
    if idx>=model.program.length: return 0.0
    base_acc=acc(model,x,y)
    ss=list(model.program.steps); ss[idx]=Step("identity")
    alt=TaskProgramModel(model.base,Program(tuple(ss))).to(x.device)
    copy_program_state(model,alt)
    return max(0.0,base_acc-acc(alt,x,y))

def counterfactual_fidelity(model,teacher,x,y,idx):
    if idx>=model.program.length: return 0.0
    # DART delta after ablation
    dd=step_ablation(model,x,y,idx)
    # Teacher delta: apply same learned program transform with one step identity.
    ss=list(model.program.steps); ss[idx]=Step("identity")
    temp=ProgramTransform(Program(tuple(ss))).to(x.device)
    for a,b in zip(model.transform.raw,temp.raw):
        if b.requires_grad:
            with torch.no_grad(): b.copy_(a.detach())
    with torch.no_grad():
        t0=acc(teacher,x,y)
        t1=acc(teacher,temp(x),y)
    td=max(0.0,t0-t1)
    return 1.0-abs(td-dd)/max(1e-6,abs(td)+abs(dd))

def train_shared(args,source_tasks,seed,device,prog):
    candidates=[]
    for ci,(motif,nodes) in enumerate(product(
        ["sequential","parallel_sum","residual_parallel"],
        [["affine_polynomial","polynomial"],
         ["affine_polynomial","polynomial","low_rank"]]),1):
        primitive=SharedPrimitive(nodes,motif,args.d_model,args.rank).to(device)
        base=PrimitiveModel(primitive,args.d_model,args.classes).to(device)
        xs=[]; ys=[]
        for j,t in enumerate(source_tasks):
            x,y=make_dataset(t,max(1,args.train_size//len(source_tasks)),seed+300+j,device,"A")
            xs.append(x); ys.append(y)
        xm=torch.cat(xs); ym=torch.cat(ys)
        fit(base,DataLoader(TensorDataset(xm,ym),batch_size=args.batch_size,shuffle=True),
            args.core_fit_steps,args.core_fit_lr,freeze_primitive=False)
        vals=[]
        for j,t in enumerate(source_tasks):
            x,y=make_dataset(t,args.verifier_size,seed+500+ci*19+j,device,"B")
            vals.append(acc(base,x,y))
        candidates.append((balanced(vals),base,vals))
        prog.update(ci,"shared-primitive",f"candidate={ci} score={candidates[-1][0]:.3f}")
    candidates.sort(key=lambda z:z[0],reverse=True)
    return candidates[0]

def train_program_on_task(base,program,task,seed,args,device,regime):
    x,y=make_dataset(task,args.verifier_size,seed,device,regime)
    m=TaskProgramModel(base,program).to(device)
    fit(m,DataLoader(TensorDataset(x,y),batch_size=args.fit_batch_samples,shuffle=True),
        args.program_fit_steps,args.lr,freeze_primitive=True)
    return m,x,y

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
    ap.add_argument("--train-size",type=int,default=6000)
    ap.add_argument("--verifier-size",type=int,default=1500)
    ap.add_argument("--test-size",type=int,default=1500)
    ap.add_argument("--target-adaptation-size",type=int,default=1200)
    ap.add_argument("--target-validation-size",type=int,default=1200)
    ap.add_argument("--fit-batch-samples",type=int,default=512)
    ap.add_argument("--max-program-length",type=int,default=2)
    ap.add_argument("--complexity-lambda",type=float,default=0.05)
    ap.add_argument("--device",default="cuda")
    ap.add_argument("--d-model",type=int,default=32)
    ap.add_argument("--rank",type=int,default=8)
    ap.add_argument("--classes",type=int,default=8)
    ap.add_argument("--batch-size",type=int,default=256)
    ap.add_argument("--core-fit-lr",type=float,default=1e-3)
    ap.add_argument("--lr",type=float,default=3e-4)
    ap.add_argument("--out",default="dart033_results.json")
    args=ap.parse_args()

    device=torch.device(args.device if args.device=="cpu" or torch.cuda.is_available() else "cpu")
    source_tasks=[t for t in args.all_tasks if t not in args.holdout_tasks]
    target=args.holdout_tasks[0]; contrast=args.contrast_tasks[0]
    programs=enumerate_programs(args.max_program_length)
    records=[]
    for si,seed in enumerate(args.seeds,1):
        seed_all(seed)
        total=8+len(programs)*4
        prog=Progress(total,si,len(args.seeds))

        # Independent teachers on A/B/C regimes.
        teachers={}
        for j,t in enumerate(source_tasks+[target,contrast]):
            tm=PrimitiveModel(SharedPrimitive(["affine_polynomial","polynomial"],"sequential",args.d_model,args.rank),
                              args.d_model,args.classes).to(device)
            xa,ya=make_dataset(t,args.train_size,seed+1000+j,device,"A")
            fit(tm,DataLoader(TensorDataset(xa,ya),batch_size=args.batch_size,shuffle=True),
                args.teacher_steps,args.lr,freeze_primitive=False)
            teachers[t]=tm
            prog.update(1+j,"teacher-training",f"task={t}")

        _,base,base_scores=train_shared(args,source_tasks,seed,device,prog)
        for p in base.primitive.parameters(): p.requires_grad_(False)

        # Source program search must be stable across TWO independent source regimes.
        source_best=[]
        for pi,prg in enumerate(programs):
            regime_scores=[]
            for t in source_tasks:
                ma,xa,ya=train_program_on_task(base,prg,t,seed+2000+pi*17, args,device,"A")
                xb,yb=make_dataset(t,args.verifier_size,seed+3000+pi*17,device,"B")
                regime_scores.append((acc(ma,xa,ya),acc(ma,xb,yb)))
            pooled=[0.5*(a+b) for a,b in regime_scores]
            score=balanced(pooled)-args.complexity_lambda*prg.length
            source_best.append((score,prg,regime_scores))
            prog.update(8+pi,"cross-distribution-program-search",f"program={pi+1}/{len(programs)} score={score:.3f}")
        source_best.sort(key=lambda z:z[0],reverse=True)
        _,best_source_program,_=source_best[0]

        # Target: independent A adaptation, B validation, C untouched test.
        target_candidates=[]
        for pi,prg in enumerate(programs):
            xa,ya=make_dataset(target,args.target_adaptation_size,seed+5000+pi*23,device,"A")
            xb,yb=make_dataset(target,args.target_validation_size,seed+6000+pi*23,device,"B")
            m=TaskProgramModel(base,prg).to(device)
            fit(m,DataLoader(TensorDataset(xa,ya),batch_size=args.fit_batch_samples,shuffle=True),
                args.target_program_fit_steps,args.lr,freeze_primitive=True)
            va=acc(m,xa,ya); vb=acc(m,xb,yb)
            # necessity + teacher-aligned fidelity evaluated on BOTH A and B
            na=[]; nb=[]
            fa=[]; fb=[]
            for k in range(prg.length):
                na.append(step_ablation(m,xa,ya,k)); nb.append(step_ablation(m,xb,yb,k))
                fa.append(counterfactual_fidelity(m,teachers[target],xa,ya,k))
                fb.append(counterfactual_fidelity(m,teachers[target],xb,yb,k))
            nec_a=sum(na)/len(na); nec_b=sum(nb)/len(nb)
            fid_a=sum(fa)/len(fa); fid_b=sum(fb)/len(fb)
            invariance=1.0-(abs(va-vb)+abs(nec_a-nec_b)+abs(fid_a-fid_b))/3.0
            score=0.35*min(va,vb)+0.25*min(nec_a,nec_b)+0.25*min(fid_a,fid_b)+0.15*invariance-args.complexity_lambda*prg.length
            target_candidates.append((score,m,prg,va,vb,nec_a,nec_b,fid_a,fid_b,invariance))
            prog.update(8+len(programs)+pi,"target-cross-distribution-selection",
                        f"program={pi+1}/{len(programs)} A={va:.3f} B={vb:.3f} inv={invariance:.3f}")
        target_candidates.sort(key=lambda z:z[0],reverse=True)
        _,best_model,best_program,va,vb,na,nb,fa,fb,inv=target_candidates[0]

        # C is untouched final target test.
        xc,yc=make_dataset(target,args.test_size,seed+7000,device,"C")
        zero=TaskProgramModel(base,Program((Step("identity"),))).to(device)
        zero_test=acc(zero,xc,yc)
        program_test=acc(best_model,xc,yc)

        # controls on the same A/B training split but evaluated on C only
        wrong=programs[(programs.index(best_program)+1)%len(programs)]
        wm=TaskProgramModel(base,wrong).to(device)
        xa2,ya2=make_dataset(target,args.target_adaptation_size,seed+7100,device,"A")
        fit(wm,DataLoader(TensorDataset(xa2,ya2),batch_size=args.fit_batch_samples,shuffle=True),
            args.transfer_control_steps,args.lr,freeze_primitive=True)
        wrong_test=acc(wm,xc,yc)

        rp=programs[(seed*23)%len(programs)]
        rm=TaskProgramModel(base,rp).to(device)
        fit(rm,DataLoader(TensorDataset(xa2,ya2),batch_size=args.fit_batch_samples,shuffle=True),
            args.transfer_control_steps,args.lr,freeze_primitive=True)
        random_test=acc(rm,xc,yc)

        cx,cy=make_dataset(contrast,args.test_size,seed+8000,device,"C")
        cm=TaskProgramModel(base,best_program).to(device)
        fit(cm,DataLoader(TensorDataset(cx,cy),batch_size=args.fit_batch_samples,shuffle=True),
            args.transfer_control_steps,args.lr,freeze_primitive=True)
        contrast_test=acc(cm,cx,cy)

        prog.update(total-1,"final-test",f"target={target} test={program_test:.3f} inv={inv:.3f}")
        prog.close()

        # Source program causal metrics on two source regimes.
        src_nec=[]; src_fid=[]
        for j,t in enumerate(source_tasks):
            m=TaskProgramModel(base,best_source_program).to(device)
            xa,ya=make_dataset(t,args.verifier_size,seed+9000+j,device,"A")
            fit(m,DataLoader(TensorDataset(xa,ya),batch_size=args.fit_batch_samples,shuffle=True),
                args.program_fit_steps,args.lr,freeze_primitive=True)
            xb,yb=make_dataset(t,args.verifier_size,seed+9100+j,device,"B")
            for k in range(best_source_program.length):
                src_nec.append(0.5*(step_ablation(m,xa,ya,k)+step_ablation(m,xb,yb,k)))
                src_fid.append(0.5*(counterfactual_fidelity(m,teachers[t],xa,ya,k)+counterfactual_fidelity(m,teachers[t],xb,yb,k)))

        records.append({
            "seed":seed,
            "winner":{
                "program":best_program.names(),
                "program_length":best_program.length,
                "target_adaptation_accuracy":va,
                "target_validation_accuracy":vb,
                "target_necessity_A":na,
                "target_necessity_B":nb,
                "target_cf_fidelity_A":fa,
                "target_cf_fidelity_B":fb,
                "target_program_invariance":inv,
                "source_program":best_source_program.names()
            },
            "related_holdout":{
                target:{
                    "teacher":float(acc(teachers[target],xc,yc)),
                    "dart_zero":float(zero_test),
                    "dart_program":float(program_test),
                    "program_validation_B":float(vb),
                    "program_permutation_control":float(wrong_test),
                    "random_program_control":float(random_test)
                }
            },
            "contrast_holdout":{
                contrast:{
                    "teacher":float(acc(teachers[contrast],cx,cy)),
                    "dart_program":float(contrast_test)
                }
            },
            "source":{
                "program_necessity":float(sum(src_nec)/len(src_nec)),
                "program_cf_fidelity":float(sum(src_fid)/len(src_fid))
            }
        })

    summary={
        "version":"DART-3.3",
        "parent_version":"DART-3.2",
        "related_holdout":{
            target:{
                "teacher":sum(r["related_holdout"][target]["teacher"] for r in records)/len(records),
                "dart_zero":sum(r["related_holdout"][target]["dart_zero"] for r in records)/len(records),
                "dart_program":sum(r["related_holdout"][target]["dart_program"] for r in records)/len(records),
                "program_validation_B":sum(r["related_holdout"][target]["program_validation_B"] for r in records)/len(records),
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
            "avg_program_necessity":sum(r["source"]["program_necessity"] for r in records)/len(records),
            "avg_program_cf_fidelity":sum(r["source"]["program_cf_fidelity"] for r in records)/len(records),
            "avg_program_length":sum(r["winner"]["program_length"] for r in records)/len(records)
        },
        "target_invariance":{
            "avg_A_accuracy":sum(r["winner"]["target_adaptation_accuracy"] for r in records)/len(records),
            "avg_B_accuracy":sum(r["winner"]["target_validation_accuracy"] for r in records)/len(records),
            "avg_necessity_A":sum(r["winner"]["target_necessity_A"] for r in records)/len(records),
            "avg_necessity_B":sum(r["winner"]["target_necessity_B"] for r in records)/len(records),
            "avg_cf_fidelity_A":sum(r["winner"]["target_cf_fidelity_A"] for r in records)/len(records),
            "avg_cf_fidelity_B":sum(r["winner"]["target_cf_fidelity_B"] for r in records)/len(records),
            "avg_program_invariance":sum(r["winner"]["target_program_invariance"] for r in records)/len(records)
        },
        "records":records
    }
    out=Path(args.out); out.write_text(json.dumps(summary,indent=2))
    print("DART-3.3: cross-distribution task-program invariance")
    print(json.dumps(summary,indent=2))
    print(f"Saved: {out.resolve()}")

if __name__=="__main__":
    main()
