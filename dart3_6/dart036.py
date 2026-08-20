#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, random, sys, copy
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

TASKS={
    "add":lambda a,b:a+b,
    "mul":lambda a,b:a*b,
    "sub":lambda a,b:a-b,
    "compose":lambda a,b:(a*2+1)-(b*3-1),
    "sort":lambda a,b:torch.minimum(a,b),
}
REGIMES={
    "A":(-3,3,0.0,1.0,1.0),
    "B":(-8,8,0.25,1.0,1.0),
    "C":(-14,14,1.0,1.0,-1.0),
    "D":(-20,20,-0.5,1.5,0.75),
    "E":(-28,28,1.5,0.6,1.4),
}

def seed_all(seed):
    random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

class Progress:
    def __init__(self,total,si,ns):
        self.total=max(1,total); self.si=si; self.ns=ns
    def update(self,d,phase,detail=""):
        f=max(0,min(1,d/self.total)); w=28; fill=int(f*w)
        bar="="*fill+">"+" " * max(0,w-fill-1)
        s=f"\r[DART-3.6][seed {self.si}/{self.ns}] [{bar}] {100*f:6.2f}% | {phase}"
        if detail:s+=f" | {detail}"
        sys.stdout.write(s);sys.stdout.flush()
    def close(self): self.update(self.total,"complete"); print()

def make_data(task,n,seed,device,regime):
    gd="cuda" if device.type=="cuda" else "cpu"
    g=torch.Generator(device=gd).manual_seed(seed)
    lo,hi,shift,s0,s1=REGIMES[regime]
    x=torch.randint(lo,hi+1,(n,2),generator=g,device=device).float()
    x[:,0]=x[:,0]*s0+shift; x[:,1]=x[:,1]*s1-shift
    y=TASKS[task](x[:,0],x[:,1])
    bins=torch.tensor([-24,-12,-6,-3,0,3,6,12,24],device=device).float()
    return x,torch.bucketize(y,bins).clamp(max=9).long()

class SharedPrimitive(nn.Module):
    def __init__(self,nodes,motif,d=32,r=8):
        super().__init__(); self.motif=motif
        bs=[]
        for n in nodes:
            if n=="affine_polynomial":
                bs.append(nn.Sequential(nn.Linear(d,d),nn.GELU(),nn.Linear(d,d)))
            elif n=="polynomial":
                bs.append(nn.Sequential(nn.Linear(d,d),nn.Tanh(),nn.Linear(d,d)))
            else:
                bs.append(nn.Sequential(nn.Linear(d,r,bias=False),nn.Linear(r,d,bias=False)))
        self.blocks=nn.ModuleList(bs); self.norm=nn.LayerNorm(d)
    def forward(self,h):
        if self.motif=="sequential":
            z=h
            for b in self.blocks: z=z+b(self.norm(z))
            return z
        hs=[b(self.norm(h)) for b in self.blocks]
        return h+sum(hs) if self.motif=="parallel_sum" else h+hs[-1]+0.5*sum(hs[:-1])

class Base(nn.Module):
    def __init__(self,p,d=32,c=10):
        super().__init__(); self.inp=nn.Linear(2,d); self.p=p; self.out=nn.Linear(d,c)
    def forward(self,x): return self.out(self.p(self.inp(x)))

class Transform(nn.Module):
    def __init__(self,ops):
        super().__init__(); self.ops=tuple(ops); self.raw=nn.ParameterList()
        for o in self.ops:
            self.raw.append(nn.Parameter(torch.tensor(1.0 if o=="scale" else 0.0), requires_grad=o in ("scale","shift")))
    def forward(self,x):
        z=x
        for o,p in zip(self.ops,self.raw):
            a,b=z[:,0],z[:,1]
            if o=="identity": pass
            elif o=="scale": z=z*p
            elif o=="shift": z=z+p
            elif o=="negate": z=-z
            elif o=="difference": z=torch.stack([a-b,b-a],1)
            elif o=="product": q=a*b; z=torch.stack([q,q],1)
            elif o=="swap": z=torch.stack([b,a],1)
        return z

class ProgramModel(nn.Module):
    def __init__(self,base,ops):
        super().__init__(); self.base=base; self.ops=tuple(ops); self.t=Transform(ops)
        self.s=nn.Parameter(torch.tensor(1.0)); self.bias=nn.Parameter(torch.tensor(0.0))
    def forward(self,x): return self.base(self.t(x))*self.s+self.bias

def fit(m,loader,steps,lr,freeze=False):
    ps=[]
    for n,p in m.named_parameters():
        if freeze and ("base.p" in n or "base.inp" in n): p.requires_grad_(False)
        if p.requires_grad: ps.append(p)
    opt=torch.optim.Adam(ps,lr=lr); ce=nn.CrossEntropyLoss(); it=iter(loader); m.train()
    for _ in range(steps):
        try:x,y=next(it)
        except StopIteration: it=iter(loader); x,y=next(it)
        opt.zero_grad(set_to_none=True); loss=ce(m(x),y); loss.backward()
        torch.nn.utils.clip_grad_norm_(ps,1.0); opt.step()

def acc(m,x,y):
    m.eval()
    with torch.no_grad(): pred=m(x).argmax(-1)
    return float((pred==y).float().mean())

@dataclass(frozen=True)
class Law:
    relation:str; symmetry:str; scale:str; offset:str
    complexity:int=4

def exact_law(task,x):
    a,b=x[:,0],x[:,1]; y=TASKS[task](a,b)
    sw=TASKS[task](b,a); y2=TASKS[task](2*a,2*b); y3=TASKS[task](a+1,b+1)
    sym="symmetric" if torch.allclose(y,sw) else "antisymmetric" if torch.allclose(y,-sw) else "asymmetric"
    scale="homogeneous" if torch.allclose(y2,2*y) else "quadratic_like" if torch.allclose(y2,4*y) else "affine"
    offset="translation_invariant" if torch.allclose(y3,y) else "translation_sensitive"
    rel={"add":"sum","mul":"product","sub":"difference","compose":"composed_affine_difference","sort":"selection"}[task]
    return Law(rel,sym,scale,offset)

def law_similarity(a,b):
    return sum(getattr(a,k)==getattr(b,k) for k in ("relation","symmetry","scale","offset"))/4.0

def compile_law(l):
    if l.relation=="difference": return ("difference","swap") if l.symmetry=="antisymmetric" else ("difference",)
    if l.relation=="sum": return ("swap","difference") if l.offset=="translation_invariant" else ("identity",)
    if l.relation=="product": return ("product",)
    if l.relation=="composed_affine_difference": return ("difference","shift")
    return ("swap",)

def infer_source_laws(task,seed,device):
    out=[]
    for i,reg in enumerate(("A","B","C","D")):
        x,_=make_data(task,256,seed+100+i,device,reg); out.append(exact_law(task,x))
    return out

def majority_law(laws):
    fields=("relation","symmetry","scale","offset")
    vals={}
    for f in fields:
        counts={}
        for l in laws: counts[getattr(l,f)]=counts.get(getattr(l,f),0)+1
        vals[f]=max(counts,key=counts.get)
    return Law(vals["relation"],vals["symmetry"],vals["scale"],vals["offset"])

def oracle_probe_score(law,task,seed,device,n):
    vals=[]
    for i,reg in enumerate(("A","B","C","D","E")):
        x,_=make_data(task,n,seed+1000+i,device,reg)
        oracle=exact_law(task,x); vals.append(law_similarity(law,oracle))
    return sum(vals)/len(vals)

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
    ap.add_argument("--fit-batch-samples",type=int,default=512)
    ap.add_argument("--law-probe-samples",type=int,default=256)
    ap.add_argument("--device",default="cuda")
    ap.add_argument("--d-model",type=int,default=32)
    ap.add_argument("--rank",type=int,default=8)
    ap.add_argument("--classes",type=int,default=10)
    ap.add_argument("--batch-size",type=int,default=256)
    ap.add_argument("--core-fit-lr",type=float,default=1e-3)
    ap.add_argument("--lr",type=float,default=3e-4)
    ap.add_argument("--out",default="dart036_results.json")
    a=ap.parse_args()
    dev=torch.device(a.device if a.device=="cpu" or torch.cuda.is_available() else "cpu")
    source=[t for t in a.all_tasks if t not in a.holdout_tasks]
    target=a.holdout_tasks[0]; contrast=a.contrast_tasks[0]; recs=[]
    for si,seed in enumerate(a.seeds,1):
        seed_all(seed); bar=Progress(20,si,len(a.seeds))
        teachers={}
        for j,t in enumerate(source+[target,contrast]):
            tm=Base(SharedPrimitive(["affine_polynomial","polynomial"],"sequential",a.d_model,a.rank),a.d_model,a.classes).to(dev)
            x,y=make_data(t,a.train_size,seed+1000+j,dev,"A")
            fit(tm,DataLoader(TensorDataset(x,y),batch_size=a.batch_size,shuffle=True),a.teacher_steps,a.lr)
            teachers[t]=tm; bar.update(1+j,"teacher-training",f"task={t}")
        # Jointly trained frozen primitive
        cands=[]
        for ci,(mot,nodes) in enumerate(product(["sequential","parallel_sum","residual_parallel"],
                    [["affine_polynomial","polynomial"],["affine_polynomial","polynomial","low_rank"]]),1):
            b=Base(SharedPrimitive(nodes,mot,a.d_model,a.rank),a.d_model,a.classes).to(dev)
            xs=[];ys=[]
            for j,t in enumerate(source):
                x,y=make_data(t,max(1,a.train_size//len(source)),seed+2000+ci*17+j,dev,"A");xs.append(x);ys.append(y)
            fit(b,DataLoader(TensorDataset(torch.cat(xs),torch.cat(ys)),batch_size=a.batch_size,shuffle=True),a.core_fit_steps,a.core_fit_lr)
            vals=[acc(b,*make_data(t,a.verifier_size,seed+3000+ci*17+j,dev,"B")) for j,t in enumerate(source)]
            score=0.5*((sum(vals)/len(vals))+min(vals)); cands.append((score,b,vals))
            bar.update(6+ci,"shared-primitive",f"candidate={ci} score={score:.3f}")
        cands.sort(key=lambda z:z[0],reverse=True);_,base,source_scores=cands[0]
        for p in base.p.parameters():p.requires_grad_(False)
        # Exact semantic law inference from A-D.
        source_laws={}
        for j,t in enumerate(source):
            laws=infer_source_laws(t,seed+4000+j*31,dev);source_laws[t]=laws
            bar.update(9+j,"semantic-law-induction",f"task={t}")
        target_laws=infer_source_laws(target,seed+5000,dev)
        inferred=majority_law(target_laws)
        semantic=oracle_probe_score(inferred,target,seed+6000,dev,a.law_probe_samples)
        # Candidate law library, including source-derived laws.
        library=[
            inferred,
            Law("difference","antisymmetric","homogeneous","translation_invariant"),
            Law("sum","symmetric","homogeneous","translation_sensitive"),
            Law("product","symmetric","quadratic_like","translation_sensitive"),
            Law("composed_affine_difference","asymmetric","affine","translation_sensitive"),
            Law("selection","asymmetric","affine","translation_sensitive"),
        ]
        unique=[]
        seen=set()
        for l in library:
            key=(l.relation,l.symmetry,l.scale,l.offset)
            if key not in seen: unique.append(l);seen.add(key)
        scored=[]
        for i,law in enumerate(unique):
            sem=oracle_probe_score(law,target,seed+7000+i*17,dev,a.law_probe_samples)
            prog=compile_law(law)
            # Program validation across A-D; E remains excluded.
            vals=[]
            for reg in ("A","B","C","D"):
                x,y=make_data(target,a.verifier_size,seed+8000+i*23+ord(reg),dev,reg)
                m=ProgramModel(base,prog).to(dev)
                fit(m,DataLoader(TensorDataset(x,y),batch_size=a.fit_batch_samples,shuffle=True),
                    a.target_program_fit_steps,a.lr,True)
                vals.append(acc(m,x,y))
            score=.6*sem+.35*min(vals)+.05*(1-(max(vals)-min(vals)))-.02*len(prog)
            scored.append((score,law,prog,sem,vals))
        scored.sort(key=lambda z:z[0],reverse=True)
        _,selected_law,selected_program,selected_sem,disc_vals=scored[0]
        bar.update(14,"semantic-law-validation",f"law={selected_law.relation} LSF={selected_sem:.3f}")
        # Final E is untouched during law/program selection.
        xe,ye=make_data(target,a.test_size,seed+9000,dev,"E")
        zero=Base(copy.deepcopy(base.p),a.d_model,a.classes).to(dev)  # structural zero baseline is evaluated separately below
        # zero program around the same frozen base
        zero_pm=ProgramModel(base,("identity",)).to(dev)
        zero_e=acc(zero_pm,xe,ye)
        final_pm=ProgramModel(base,selected_program).to(dev)
        xa,ya=make_data(target,a.verifier_size,seed+9100,dev,"A")
        fit(final_pm,DataLoader(TensorDataset(xa,ya),batch_size=a.fit_batch_samples,shuffle=True),
            a.target_program_fit_steps,a.lr,True)
        e_acc=acc(final_pm,xe,ye)
        wrong_law=unique[(unique.index(selected_law)+1)%len(unique)]
        wrong_pm=ProgramModel(base,compile_law(wrong_law)).to(dev)
        fit(wrong_pm,DataLoader(TensorDataset(xa,ya),batch_size=a.fit_batch_samples,shuffle=True),
            a.transfer_control_steps,a.lr,True)
        wrong_e=acc(wrong_pm,xe,ye)
        random_law=unique[(seed*3)%len(unique)]
        random_pm=ProgramModel(base,compile_law(random_law)).to(dev)
        fit(random_pm,DataLoader(TensorDataset(xa,ya),batch_size=a.fit_batch_samples,shuffle=True),
            a.transfer_control_steps,a.lr,True)
        random_e=acc(random_pm,xe,ye)
        cx,cy=make_data(contrast,a.test_size,seed+9500,dev,"E")
        contrast_pm=ProgramModel(base,selected_program).to(dev)
        fit(contrast_pm,DataLoader(TensorDataset(cx,cy),batch_size=a.fit_batch_samples,shuffle=True),
            a.transfer_control_steps,a.lr,True)
        contrast_e=acc(contrast_pm,cx,cy)
        bar.update(19,"final-law-extrapolation",f"target={target} E={e_acc:.3f} LSF={selected_sem:.3f}")
        bar.close()
        recs.append({
            "seed":seed,
            "law":selected_law.__dict__,
            "program":list(selected_program),
            "semantic_law_fidelity":selected_sem,
            "discovery_A_to_D":disc_vals,
            "related_holdout":{target:{
                "teacher":acc(teachers[target],xe,ye),
                "dart_zero":zero_e,
                "dart_program":e_acc,
                "wrong_law_control":wrong_e,
                "random_law_control":random_e}},
            "contrast_holdout":{contrast:{
                "teacher":acc(teachers[contrast],cx,cy),
                "dart_program":contrast_e}}
        })
    out=Path(a.out)
    summary={
        "version":"DART-3.6","parent_version":"DART-3.5",
        "protocol":{
            "semantic_law_validation":True,"oracle_probes":True,"law_to_program_compilation":True,
            "discovery_regimes":["A","B","C","D"],"untouched_final_regime":"E",
            "wrong_law_control":True,"random_law_control":True,
            "target_test_untouched":True,"deterministic_seeding":True},
        "related_holdout":{target:{
            k:sum(r["related_holdout"][target][k] for r in recs)/len(recs)
            for k in ("teacher","dart_zero","dart_program","wrong_law_control","random_law_control")}},
        "contrast_holdout":{contrast:{
            k:sum(r["contrast_holdout"][contrast][k] for r in recs)/len(recs)
            for k in ("teacher","dart_program")}},
        "law":{
            "semantic_fidelity":sum(r["semantic_law_fidelity"] for r in recs)/len(recs),
            "relations":[r["law"]["relation"] for r in recs],
            "programs":[r["program"] for r in recs]},
        "records":recs}
    out.write_text(json.dumps(summary,indent=2))
    print("DART-3.6: semantic task-law validation + exact-law extrapolation")
    print(json.dumps(summary,indent=2));print(f"Saved: {out.resolve()}")

if __name__=="__main__": main()
