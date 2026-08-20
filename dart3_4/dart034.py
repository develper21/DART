#!/usr/bin/env python3
"""
DART-3.4: multi-regime task-law discovery.

Goal:
DART-3.3 showed strong A/B invariance while collapsing on a third test regime.
DART-3.4 therefore discovers a task program across multiple independent
regimes (A/B/C/D) and keeps a fifth regime (E) completely untouched.

No target final-test data is used during program selection.
"""
from __future__ import annotations
import argparse, copy, json, random, sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

TASKS = {
    "add": lambda a,b: a+b,
    "mul": lambda a,b: a*b,
    "sub": lambda a,b: a-b,
    "compose": lambda a,b: (a*2+1) - (b*3-1),
    "sort": lambda a,b: torch.where(a<=b,a,b),
}
PROGRAM_OPS = ["identity","scale","shift","negate","difference","product","swap"]

def seed_all(seed):
    random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

class Progress:
    def __init__(self,total,seed_idx,nseeds):
        self.total=max(1,total); self.seed_idx=seed_idx; self.nseeds=nseeds
    def update(self,done,phase,detail=""):
        frac=max(0,min(1,done/self.total)); w=28; fill=int(frac*w)
        bar="="*fill+">"+" " * max(0,w-fill-1)
        s=f"\r[DART-3.4][seed {self.seed_idx}/{self.nseeds}] [{bar}] {frac*100:6.2f}% | {phase}"
        if detail: s+=f" | {detail}"
        sys.stdout.write(s); sys.stdout.flush()
    def close(self): self.update(self.total,"complete"); print()

def make_dataset(task,n,seed,device,regime):
    gen_device = "cuda" if device.type=="cuda" else "cpu"
    g=torch.Generator(device=gen_device).manual_seed(seed)
    if regime=="A":
        x=torch.randint(-3,4,(n,2),generator=g,device=device).float()
    elif regime=="B":
        x=torch.randint(-8,9,(n,2),generator=g,device=device).float()+0.25
    elif regime=="C":
        x=torch.randint(-14,15,(n,2),generator=g,device=device).float()
        x += torch.tensor([1.0,-1.0],device=device)
    elif regime=="D":
        x=torch.randint(-20,21,(n,2),generator=g,device=device).float()
        # non-uniform sign/scale transform without changing the task law
        x[:,0] = x[:,0]*1.5
        x[:,1] = x[:,1]*0.75 - 0.5
    elif regime=="E":
        x=torch.randint(-28,29,(n,2),generator=g,device=device).float()
        x += torch.tensor([-1.5,1.5],device=device)
        x[:,0] = x[:,0]*0.6
        x[:,1] = x[:,1]*1.4
    else:
        raise ValueError(regime)
    y=TASKS[task](x[:,0],x[:,1])
    bins=torch.tensor([-24,-12,-6,-3,0,3,6,12,24],device=device).float()
    y=torch.bucketize(y,bins).clamp(max=9)
    return x,y.long()

class SharedPrimitive(nn.Module):
    def __init__(self,nodes,motif,d=32,rank=8):
        super().__init__(); self.nodes=nodes; self.motif=motif
        blocks=[]
        for n in nodes:
            if n=="affine_polynomial":
                blocks.append(nn.Sequential(nn.Linear(d,d),nn.GELU(),nn.Linear(d,d)))
            elif n=="polynomial":
                blocks.append(nn.Sequential(nn.Linear(d,d),nn.Tanh(),nn.Linear(d,d)))
            else:
                blocks.append(nn.Sequential(nn.Linear(d,rank,bias=False),nn.Linear(rank,d,bias=False)))
        self.blocks=nn.ModuleList(blocks); self.norm=nn.LayerNorm(d)
    def forward(self,h):
        if self.motif=="sequential":
            z=h
            for b in self.blocks: z=z+b(self.norm(z))
            return z
        hs=[b(self.norm(h)) for b in self.blocks]
        if self.motif=="parallel_sum": return h+sum(hs)
        return h+hs[-1]+0.5*sum(hs[:-1])

class PrimitiveModel(nn.Module):
    def __init__(self,primitive,d=32,classes=10):
        super().__init__(); self.inp=nn.Linear(2,d); self.primitive=primitive; self.out=nn.Linear(d,classes)
    def forward(self,x): return self.out(self.primitive(self.inp(x)))

@dataclass(frozen=True)
class Step: op:str
@dataclass(frozen=True)
class Program:
    steps:Tuple[Step,...]
    @property
    def length(self): return len(self.steps)
    def names(self): return [x.op for x in self.steps]

class Transform(nn.Module):
    def __init__(self,program):
        super().__init__(); self.program=program; ps=[]
        for s in program.steps:
            if s.op=="scale": ps.append(nn.Parameter(torch.tensor(1.0)))
            elif s.op=="shift": ps.append(nn.Parameter(torch.tensor(0.0)))
            else: ps.append(nn.Parameter(torch.tensor(0.0),requires_grad=False))
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

class ProgramModel(nn.Module):
    def __init__(self,base,program):
        super().__init__(); self.base=base; self.program=program; self.transform=Transform(program)
        self.ds=nn.Parameter(torch.tensor(1.0)); self.db=nn.Parameter(torch.tensor(0.0))
    def forward(self,x):
        return self.base(self.transform(x))*self.ds+self.db

def fit(model,loader,steps,lr,freeze=True):
    ps=[]
    for n,p in model.named_parameters():
        if freeze and ("base.primitive" in n or "base.inp" in n):
            p.requires_grad_(False)
        if p.requires_grad: ps.append(p)
    opt=torch.optim.Adam(ps,lr=lr); loss_fn=nn.CrossEntropyLoss(); it=iter(loader); model.train()
    for _ in range(steps):
        try: x,y=next(it)
        except StopIteration:
            it=iter(loader); x,y=next(it)
        opt.zero_grad(set_to_none=True); loss=loss_fn(model(x),y); loss.backward()
        torch.nn.utils.clip_grad_norm_(ps,1.0); opt.step()

def acc(m,x,y):
    m.eval()
    with torch.no_grad(): p=m(x).argmax(-1)
    return float((p==y).float().mean().item())

def balanced(v):
    return .5*(sum(v)/len(v))+ .5*min(v)

def enum_programs(L):
    out=[]
    for l in range(1,L+1):
        for ops in product(PROGRAM_OPS,repeat=l):
            if all(o=="identity" for o in ops): continue
            out.append(Program(tuple(Step(o) for o in ops)))
    return out

def clone_program_state(src,dst):
    for a,b in zip(src.transform.raw,dst.transform.raw):
        if b.requires_grad:
            with torch.no_grad(): b.copy_(a.detach())
    with torch.no_grad():
        dst.ds.copy_(src.ds.detach()); dst.db.copy_(src.db.detach())

def necessity(m,x,y,idx):
    base=acc(m,x,y); ss=list(m.program.steps); ss[idx]=Step("identity")
    alt=ProgramModel(m.base,Program(tuple(ss))).to(x.device); clone_program_state(m,alt)
    return max(0.0,base-acc(alt,x,y))

def cf_fidelity(m,teacher,x,y,idx):
    d=necessity(m,x,y,idx)
    ss=list(m.program.steps); ss[idx]=Step("identity")
    tr=Transform(Program(tuple(ss))).to(x.device)
    for a,b in zip(m.transform.raw,tr.raw):
        if b.requires_grad:
            with torch.no_grad(): b.copy_(a.detach())
    t0=acc(teacher,x,y); t1=acc(teacher,tr(x),y)
    td=max(0.0,t0-t1)
    return 1.0-abs(td-d)/max(1e-6,abs(td)+abs(d))

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
    ap.add_argument("--complexity-lambda",type=float,default=.05)
    ap.add_argument("--device",default="cuda")
    ap.add_argument("--d-model",type=int,default=32)
    ap.add_argument("--rank",type=int,default=8)
    ap.add_argument("--classes",type=int,default=10)
    ap.add_argument("--batch-size",type=int,default=256)
    ap.add_argument("--core-fit-lr",type=float,default=1e-3)
    ap.add_argument("--lr",type=float,default=3e-4)
    ap.add_argument("--out",default="dart034_results.json")
    args=ap.parse_args()

    device=torch.device(args.device if args.device=="cpu" or torch.cuda.is_available() else "cpu")
    source=[t for t in args.all_tasks if t not in args.holdout_tasks]
    target=args.holdout_tasks[0]; contrast=args.contrast_tasks[0]
    regimes=["A","B","C","D"]; final_regime="E"
    programs=enum_programs(args.max_program_length)
    records=[]

    for si,seed in enumerate(args.seeds,1):
        seed_all(seed); total=8+len(programs)*4
        bar=Progress(total,si,len(args.seeds))

        teachers={}
        for j,t in enumerate(source+[target,contrast]):
            tm=PrimitiveModel(SharedPrimitive(["affine_polynomial","polynomial"],"sequential",args.d_model,args.rank),
                              args.d_model,args.classes).to(device)
            x,y=make_dataset(t,args.train_size,seed+1000+j,device,"A")
            fit(tm,DataLoader(TensorDataset(x,y),batch_size=args.batch_size,shuffle=True),
                args.teacher_steps,args.lr,False)
            teachers[t]=tm; bar.update(1+j,"teacher-training",f"task={t}")

        # jointly train and balanced-select shared primitive
        cands=[]
        for ci,(motif,nodes) in enumerate(product(
            ["sequential","parallel_sum","residual_parallel"],
            [["affine_polynomial","polynomial"],["affine_polynomial","polynomial","low_rank"]]),1):
            base=PrimitiveModel(SharedPrimitive(nodes,motif,args.d_model,args.rank),args.d_model,args.classes).to(device)
            xs=[]; ys=[]
            for j,t in enumerate(source):
                x,y=make_dataset(t,args.train_size//len(source),seed+2000+ci*13+j,device,"A"); xs.append(x); ys.append(y)
            fit(base,DataLoader(TensorDataset(torch.cat(xs),torch.cat(ys)),batch_size=args.batch_size,shuffle=True),
                args.core_fit_steps,args.core_fit_lr,False)
            vals=[]
            for j,t in enumerate(source):
                x,y=make_dataset(t,args.verifier_size,seed+3000+ci*19+j,device,"B"); vals.append(acc(base,x,y))
            cands.append((balanced(vals),base,vals))
            bar.update(6+ci,"shared-primitive",f"candidate={ci} score={cands[-1][0]:.3f}")
        cands.sort(key=lambda z:z[0],reverse=True); _,base,base_scores=cands[0]
        for p in base.primitive.parameters(): p.requires_grad_(False)

        # Program must generalize over all four discovery regimes.
        chosen=[]
        for pi,prg in enumerate(programs):
            regime_vals=[]
            for ri,reg in enumerate(regimes):
                task_vals=[]
                for tj,t in enumerate(source):
                    x,y=make_dataset(t,args.verifier_size,seed+4000+pi*37+ri*53+tj,device,reg)
                    m=ProgramModel(base,prg).to(device)
                    fit(m,DataLoader(TensorDataset(x,y),batch_size=args.fit_batch_samples,shuffle=True),
                        args.program_fit_steps,args.lr,True)
                    task_vals.append(acc(m,x,y))
                regime_vals.append(balanced(task_vals))
            score=min(regime_vals)-args.complexity_lambda*prg.length
            chosen.append((score,prg,regime_vals))
            bar.update(9+pi,"multi-regime-program-search",f"program={pi+1}/{len(programs)} min={min(regime_vals):.3f}")
        chosen.sort(key=lambda z:z[0],reverse=True)
        best_source_program=chosen[0][1]

        # Target discovery: four regimes, fifth untouched.
        target_candidates=[]
        for pi,prg in enumerate(programs):
            A=[];B=[];C=[];D=[]
            models=[]
            for reg in regimes:
                if reg=="A": seed0=5000
                elif reg=="B": seed0=6000
                elif reg=="C": seed0=7000
                else: seed0=8000
                x,y=make_dataset(target,args.target_adaptation_size,seed+seed0+pi*17,device,reg)
                m=ProgramModel(base,prg).to(device)
                fit(m,DataLoader(TensorDataset(x,y),batch_size=args.fit_batch_samples,shuffle=True),
                    args.target_program_fit_steps,args.lr,True)
                models.append(m); 
                if reg=="A": A=(m,x,y)
                elif reg=="B": B=(m,x,y)
                elif reg=="C": C=(m,x,y)
                else: D=(m,x,y)
            ars=[]; nes=[]; fids=[]
            for m,x,y in [A,B,C,D]:
                ars.append(acc(m,x,y))
                nk=[necessity(m,x,y,k) for k in range(prg.length)]
                fk=[cf_fidelity(m,teachers[target],x,y,k) for k in range(prg.length)]
                nes.append(sum(nk)/len(nk)); fids.append(sum(fk)/len(fk))
            inv=1-(max(ars)-min(ars) + max(nes)-min(nes) + max(fids)-min(fids))/3
            score=min(ars)+.25*min(nes)+.25*min(fids)+.15*inv-args.complexity_lambda*prg.length
            target_candidates.append((score,prg,ars,nes,fids,inv))
            bar.update(9+len(programs)+pi,"target-law-selection",
                       f"program={pi+1}/{len(programs)} min={min(ars):.3f} inv={inv:.3f}")
        target_candidates.sort(key=lambda z:z[0],reverse=True)
        _,best_program,ars,nes,fids,inv=target_candidates[0]

        # Final E is never used for selection.
        xe,ye=make_dataset(target,args.test_size,seed+9000,device,final_regime)
        zero=ProgramModel(base,Program((Step("identity"),))).to(device)
        zero_e=acc(zero,xe,ye)

        # Train selected program on A only, evaluate E.
        xa,ya=make_dataset(target,args.target_adaptation_size,seed+9100,device,"A")
        final_model=ProgramModel(base,best_program).to(device)
        fit(final_model,DataLoader(TensorDataset(xa,ya),batch_size=args.fit_batch_samples,shuffle=True),
            args.target_program_fit_steps,args.lr,True)
        e_acc=acc(final_model,xe,ye)

        wrong=programs[(programs.index(best_program)+1)%len(programs)]
        wm=ProgramModel(base,wrong).to(device)
        fit(wm,DataLoader(TensorDataset(xa,ya),batch_size=args.fit_batch_samples,shuffle=True),
            args.transfer_control_steps,args.lr,True)
        wrong_e=acc(wm,xe,ye)

        rp=programs[(seed*31)%len(programs)]
        rm=ProgramModel(base,rp).to(device)
        fit(rm,DataLoader(TensorDataset(xa,ya),batch_size=args.fit_batch_samples,shuffle=True),
            args.transfer_control_steps,args.lr,True)
        random_e=acc(rm,xe,ye)

        cx,cy=make_dataset(contrast,args.test_size,seed+9500,device,final_regime)
        cm=ProgramModel(base,best_program).to(device)
        fit(cm,DataLoader(TensorDataset(cx,cy),batch_size=args.fit_batch_samples,shuffle=True),
            args.transfer_control_steps,args.lr,True)
        contrast_e=acc(cm,cx,cy)
        bar.update(total-1,"final-unseen-regime",f"target={target} E={e_acc:.3f} inv={inv:.3f} program={len(programs)}/{len(programs)}")
        bar.close()

        # source causal checks for the source-selected program
        sn=[]; sf=[]
        for j,t in enumerate(source):
            x,y=make_dataset(t,args.verifier_size,seed+10000+j,device,"D")
            sm=ProgramModel(base,best_source_program).to(device)
            fit(sm,DataLoader(TensorDataset(x,y),batch_size=args.fit_batch_samples,shuffle=True),
                args.program_fit_steps,args.lr,True)
            for k in range(best_source_program.length):
                sn.append(necessity(sm,x,y,k)); sf.append(cf_fidelity(sm,teachers[t],x,y,k))

        records.append({
            "seed":seed,
            "winner":{
                "program":best_program.names(),"program_length":best_program.length,
                "target_regime_accuracy_A_to_D":ars,"target_necessity_A_to_D":nes,
                "target_cf_fidelity_A_to_D":fids,"program_invariance":inv,
                "source_program":best_source_program.names()
            },
            "related_holdout":{
                target:{
                    "teacher":acc(teachers[target],xe,ye),"dart_zero":zero_e,
                    "dart_program":e_acc,"program_permutation_control":wrong_e,
                    "random_program_control":random_e
                }
            },
            "contrast_holdout":{contrast:{"teacher":acc(teachers[contrast],cx,cy),"dart_program":contrast_e}},
            "source":{"program_necessity":sum(sn)/len(sn),"program_cf_fidelity":sum(sf)/len(sf)}
        })

    out=Path(args.out)
    summary={
        "version":"DART-3.4","parent_version":"DART-3.3",
        "protocol":{"discovery_regimes":["A","B","C","D"],"untouched_final_regime":"E",
                    "joint_source_primitive_training":True,"multi_regime_program_selection":True,
                    "cross_regime_necessity":True,"cross_regime_cf_fidelity":True,
                    "untouched_final_test":True,"deterministic_seeding":True},
        "related_holdout":{
            target:{k:sum(r["related_holdout"][target][k] for r in records)/len(records)
                    for k in ["teacher","dart_zero","dart_program","program_permutation_control","random_program_control"]}},
        "contrast_holdout":{
            contrast:{k:sum(r["contrast_holdout"][contrast][k] for r in records)/len(records)
                      for k in ["teacher","dart_program"]}},
        "source":{
            "avg_program_necessity":sum(r["source"]["program_necessity"] for r in records)/len(records),
            "avg_program_cf_fidelity":sum(r["source"]["program_cf_fidelity"] for r in records)/len(records)},
        "target_multi_regime":{
            "avg_program_invariance":sum(r["winner"]["program_invariance"] for r in records)/len(records),
            "avg_A":sum(r["winner"]["target_regime_accuracy_A_to_D"][0] for r in records)/len(records),
            "avg_B":sum(r["winner"]["target_regime_accuracy_A_to_D"][1] for r in records)/len(records),
            "avg_C":sum(r["winner"]["target_regime_accuracy_A_to_D"][2] for r in records)/len(records),
            "avg_D":sum(r["winner"]["target_regime_accuracy_A_to_D"][3] for r in records)/len(records),
            "avg_necessity":sum(sum(r["winner"]["target_necessity_A_to_D"])/4 for r in records)/len(records),
            "avg_cf_fidelity":sum(sum(r["winner"]["target_cf_fidelity_A_to_D"])/4 for r in records)/len(records)},
        "records":records}
    out.write_text(json.dumps(summary,indent=2))
    print("DART-3.4: multi-regime task-law discovery")
    print(json.dumps(summary,indent=2))
    print(f"Saved: {out.resolve()}")

if __name__=="__main__": main()
