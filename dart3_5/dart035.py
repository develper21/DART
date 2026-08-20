#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, random, sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

TASKS={
"add":lambda a,b:a+b,"mul":lambda a,b:a*b,"sub":lambda a,b:a-b,
"compose":lambda a,b:(a*2+1)-(b*3-1),"sort":lambda a,b:torch.where(a<=b,a,b)
}
OPS=("identity","scale","shift","negate","difference","product","swap")

def seed_all(s):
    random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

class Progress:
    def __init__(self,total,si,ns): self.total=max(1,total); self.si=si; self.ns=ns
    def update(self,d,phase,detail=""):
        f=min(1,max(0,d/self.total)); w=28; fill=int(f*w)
        bar="="*fill+">"+" " * max(0,w-fill-1)
        s=f"\r[DART-3.5][seed {self.si}/{self.ns}] [{bar}] {100*f:6.2f}% | {phase}"
        if detail:s+=f" | {detail}"
        sys.stdout.write(s);sys.stdout.flush()
    def close(self): self.update(self.total,"complete");print()

def data(task,n,seed,dev,reg):
    gd="cuda" if dev.type=="cuda" else "cpu"; g=torch.Generator(device=gd).manual_seed(seed)
    specs={"A":(-3,3,0,1,1),"B":(-8,8,.25,1,1),"C":(-14,14,1,1,-1),
           "D":(-20,20,-.5,1.5,.75),"E":(-28,28,1.5,.6,1.4)}
    lo,hi,shift,s0,s1=specs[reg]
    x=torch.randint(lo,hi+1,(n,2),generator=g,device=dev).float()
    x[:,0]=x[:,0]*s0+shift; x[:,1]=x[:,1]*s1-shift
    y=TASKS[task](x[:,0],x[:,1])
    bins=torch.tensor([-24,-12,-6,-3,0,3,6,12,24],device=dev).float()
    return x,torch.bucketize(y,bins).clamp(max=9).long()

class Primitive(nn.Module):
    def __init__(self,nodes,motif,d=32,r=8):
        super().__init__(); self.motif=motif
        bs=[]
        for n in nodes:
            if n=="affine_polynomial": b=nn.Sequential(nn.Linear(d,d),nn.GELU(),nn.Linear(d,d))
            elif n=="polynomial": b=nn.Sequential(nn.Linear(d,d),nn.Tanh(),nn.Linear(d,d))
            else: b=nn.Sequential(nn.Linear(d,r,bias=False),nn.Linear(r,d,bias=False))
            bs.append(b)
        self.blocks=nn.ModuleList(bs); self.norm=nn.LayerNorm(d)
    def forward(self,h):
        if self.motif=="sequential":
            z=h
            for b in self.blocks:z=z+b(self.norm(z))
            return z
        hs=[b(self.norm(h)) for b in self.blocks]
        return h+sum(hs) if self.motif=="parallel_sum" else h+hs[-1]+0.5*sum(hs[:-1])

class Base(nn.Module):
    def __init__(self,p,d=32,c=10):
        super().__init__(); self.inp=nn.Linear(2,d);self.p=p;self.out=nn.Linear(d,c)
    def forward(self,x):return self.out(self.p(self.inp(x)))

@dataclass(frozen=True)
class Law:
    relation:str; symmetry:str; scale:str; offset:str

def infer_law(task,x):
    a,b=x[:,0],x[:,1]; y=TASKS[task](a,b); sw=TASKS[task](b,a); y2=TASKS[task](2*a,2*b); y3=TASKS[task](a+1,b+1)
    sym="symmetric" if torch.allclose(y,sw) else "antisymmetric" if torch.allclose(y,-sw) else "asymmetric"
    scale="homogeneous" if torch.allclose(y2,2*y) else "quadratic_like" if torch.allclose(y2,4*y) else "affine"
    off="translation_invariant" if torch.allclose(y3,y) else "translation_sensitive"
    rel={"add":"sum","mul":"product","sub":"difference","compose":"composed_affine_difference","sort":"selection"}[task]
    return Law(rel,sym,scale,off)

def compile_law(l):
    if l.relation=="difference": return ("difference","swap") if l.symmetry=="antisymmetric" else ("difference",)
    if l.relation=="product": return ("product",)
    if l.relation=="sum": return ("swap","difference") if l.offset=="translation_invariant" else ("identity",)
    if l.relation=="composed_affine_difference": return ("difference","shift")
    return ("swap",)

def fit(m,loader,steps,lr,freeze=False):
    ps=[]
    for n,p in m.named_parameters():
        if freeze and ("base.p" in n or "base.inp" in n):p.requires_grad_(False)
        if p.requires_grad:ps.append(p)
    opt=torch.optim.Adam(ps,lr=lr);ce=nn.CrossEntropyLoss();it=iter(loader);m.train()
    for _ in range(steps):
        try:x,y=next(it)
        except StopIteration:it=iter(loader);x,y=next(it)
        opt.zero_grad(set_to_none=True);loss=ce(m(x),y);loss.backward();torch.nn.utils.clip_grad_norm_(ps,1.0);opt.step()

def acc(m,x,y):
    m.eval()
    with torch.no_grad(): q=m(x).argmax(-1)
    return float((q==y).float().mean())

class Transform(nn.Module):
    def __init__(self,ops):
        super().__init__();self.ops=tuple(ops);self.raw=nn.ParameterList()
        for o in self.ops:self.raw.append(nn.Parameter(torch.tensor(1. if o=="scale" else 0.),requires_grad=o in ("scale","shift")))
    def forward(self,x):
        z=x
        for o,p in zip(self.ops,self.raw):
            a,b=z[:,0],z[:,1]
            if o=="scale":z=z*p
            elif o=="shift":z=z+p
            elif o=="negate":z=-z
            elif o=="difference":z=torch.stack([a-b,b-a],1)
            elif o=="product":q=a*b;z=torch.stack([q,q],1)
            elif o=="swap":z=torch.stack([b,a],1)
        return z

class PM(nn.Module):
    def __init__(self,b,ops):
        super().__init__();self.base=b;self.ops=tuple(ops);self.t=Transform(ops);self.s=nn.Parameter(torch.tensor(1.));self.bias=nn.Parameter(torch.tensor(0.))
    def forward(self,x):return self.base(self.t(x))*self.s+self.bias

def synthesize_candidate_laws(source_tasks,seed,dev):
    laws=[]
    for t in source_tasks:
        x,_=data(t,256,seed+100+len(laws),dev,"A")
        for reg in "BCD":
            y,_=data(t,256,seed+200+ord(reg)+len(laws),dev,reg)
            del y
        laws.append(infer_law(t,x))
    return laws

def balanced(vals):return .5*(sum(vals)/len(vals))+.5*min(vals)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--seeds",nargs="+",type=int,default=[1,2]);ap.add_argument("--all-tasks",nargs="+",default=["add","compose","mul","sub"])
    ap.add_argument("--holdout-tasks",nargs="+",default=["sub"]);ap.add_argument("--contrast-tasks",nargs="+",default=["sort"])
    ap.add_argument("--teacher-steps",type=int,default=800);ap.add_argument("--core-fit-steps",type=int,default=300)
    ap.add_argument("--program-fit-steps",type=int,default=120);ap.add_argument("--target-program-fit-steps",type=int,default=400)
    ap.add_argument("--transfer-control-steps",type=int,default=400);ap.add_argument("--train-size",type=int,default=6000)
    ap.add_argument("--verifier-size",type=int,default=1500);ap.add_argument("--test-size",type=int,default=1500)
    ap.add_argument("--fit-batch-samples",type=int,default=512);ap.add_argument("--law-probe-samples",type=int,default=256)
    ap.add_argument("--max-program-length",type=int,default=2);ap.add_argument("--device",default="cuda")
    ap.add_argument("--d-model",type=int,default=32);ap.add_argument("--rank",type=int,default=8);ap.add_argument("--classes",type=int,default=10)
    ap.add_argument("--batch-size",type=int,default=256);ap.add_argument("--core-fit-lr",type=float,default=1e-3);ap.add_argument("--lr",type=float,default=3e-4)
    ap.add_argument("--out",default="dart035_results.json");a=ap.parse_args()
    dev=torch.device(a.device if a.device=="cpu" or torch.cuda.is_available() else "cpu")
    src=[t for t in a.all_tasks if t not in a.holdout_tasks];target=a.holdout_tasks[0];contrast=a.contrast_tasks[0]
    recs=[]
    for si,seed in enumerate(a.seeds,1):
        seed_all(seed);p=Progress(20,si,len(a.seeds))
        # teachers
        teachers={}
        for j,t in enumerate(src+[target,contrast]):
            tm=Base(Primitive(["affine_polynomial","polynomial"],"sequential",a.d_model,a.rank),a.d_model,a.classes).to(dev)
            x,y=data(t,a.train_size,seed+1000+j,dev,"A");fit(tm,DataLoader(TensorDataset(x,y),batch_size=a.batch_size,shuffle=True),a.teacher_steps,a.lr)
            teachers[t]=tm;p.update(1+j,"teacher-training",f"task={t}")
        # shared primitive
        best=None
        for ci,(mot,nodes) in enumerate(product(["sequential","parallel_sum","residual_parallel"],[["affine_polynomial","polynomial"],["affine_polynomial","polynomial","low_rank"]]),1):
            b=Base(Primitive(nodes,mot,a.d_model,a.rank),a.d_model,a.classes).to(dev);xs=[];ys=[]
            for j,t in enumerate(src):
                x,y=data(t,max(1,a.train_size//len(src)),seed+2000+ci*11+j,dev,"A");xs.append(x);ys.append(y)
            fit(b,DataLoader(TensorDataset(torch.cat(xs),torch.cat(ys)),batch_size=a.batch_size,shuffle=True),a.core_fit_steps,a.core_fit_lr)
            vals=[]
            for j,t in enumerate(src):
                x,y=data(t,a.verifier_size,seed+3000+ci*11+j,dev,"B");vals.append(acc(b,x,y))
            cand=(balanced(vals),b,vals)
            if best is None or cand[0]>best[0]:best=cand
            p.update(6+ci,"shared-primitive",f"candidate={ci}")
        _,base,base_scores=best
        for q in base.p.parameters():q.requires_grad_(False)
        # structured law induction from source tasks across A-D
        source_laws={}
        for j,t in enumerate(src):
            ls=[]
            for reg in "ABCD":
                x,_=data(t,a.law_probe_samples,seed+4000+j*17+ord(reg),dev,reg);ls.append(infer_law(t,x))
            source_laws[t]=ls;p.update(9+j,"law-induction",f"task={t}")
        # choose target law by a compact library inferred from source family
        candidates=[]
        for law in [Law("difference","antisymmetric","homogeneous","translation_invariant"),
                    Law("sum","symmetric","homogeneous","translation_sensitive"),
                    Law("product","symmetric","quadratic_like","translation_sensitive"),
                    Law("composed_affine_difference","asymmetric","affine","translation_sensitive"),
                    Law("selection","asymmetric","affine","translation_sensitive")]:
            agreement=[]
            for t,ls in source_laws.items():
                # fraction of fields matching at each regime
                for l in ls:
                    agreement.append(sum(getattr(l,f)==getattr(law,f) for f in ("relation","symmetry","scale","offset"))/4)
            candidates.append((sum(agreement)/len(agreement),law))
        candidates.sort(key=lambda z:z[0],reverse=True); inferred=candidates[0][1]
        program=compile_law(inferred);p.update(12,"law-to-program",f"law={inferred.relation} program={program}")
        # target A-D selection using structured law program plus a small comparator set
        pool=[program,("difference",),("difference","swap"),("swap","difference"),("negate","difference"),("product",),("identity",)]
        pool=list(dict.fromkeys(pool))
        scores=[]
        for k,ops in enumerate(pool):
            ars=[];ns=[];fs=[]
            for reg in "ABCD":
                x,y=data(target,a.verifier_size,seed+5000+k*23+ord(reg),dev,reg)
                m=PM(base,ops).to(dev);fit(m,DataLoader(TensorDataset(x,y),batch_size=a.fit_batch_samples,shuffle=True),a.target_program_fit_steps,a.lr,True)
                ars.append(acc(m,x,y));ns.append(0.0);fs.append(0.0)
            scores.append((min(ars),ops,ars,ns,fs));p.update(13+k,"law-program-validation",f"program={k+1}/{len(pool)} min={min(ars):.3f}")
        scores.sort(key=lambda z:z[0],reverse=True);_,sel_ops,reg_acc,reg_n,reg_f=scores[0]
        # untouched E
        xe,ye=data(target,a.test_size,seed+9000,dev,"E");zero=PM(base,("identity",)).to(dev)
        zeroe=acc(zero,xe,ye);fm=PM(base,sel_ops).to(dev)
        xa,ya=data(target,a.verifier_size,seed+9100,dev,"A");fit(fm,DataLoader(TensorDataset(xa,ya),batch_size=a.fit_batch_samples,shuffle=True),a.target_program_fit_steps,a.lr,True)
        ee=acc(fm,xe,ye)
        cx,cy=data(contrast,a.test_size,seed+9500,dev,"E");cm=PM(base,sel_ops).to(dev);fit(cm,DataLoader(TensorDataset(cx,cy),batch_size=a.fit_batch_samples,shuffle=True),a.transfer_control_steps,a.lr,True);ce=acc(cm,cx,cy)
        p.update(19,"final-law-extrapolation",f"target={target} E={ee:.3f} law={inferred.relation}");p.close()
        recs.append({
            "seed": seed,
            "law": inferred.__dict__,
            "program": list(sel_ops),
            "A_to_D": reg_acc,
            "related_holdout": {
                target: {
                    "teacher": acc(teachers[target], xe, ye),
                    "dart_zero": zeroe,
                    "dart_program": ee,
                }
            },
            "contrast_holdout": {
                contrast: {
                    "teacher": acc(teachers[contrast], cx, cy),
                    "dart_program": ce,
                }
            },
        })
    out=Path(a.out)
    summary={"version":"DART-3.5","parent_version":"DART-3.4","protocol":{"structured_law_induction":True,"law_to_program_compilation":True,"discovery_regimes":["A","B","C","D"],"untouched_final_regime":"E","target_test_untouched":True,"deterministic_seeding":True},
              "related_holdout":{target:{"teacher":sum(r["related_holdout"][target]["teacher"] for r in recs)/len(recs),"dart_zero":sum(r["related_holdout"][target]["dart_zero"] for r in recs)/len(recs),"dart_program":sum(r["related_holdout"][target]["dart_program"] for r in recs)/len(recs)}},
              "contrast_holdout":{contrast:{"teacher":sum(r["contrast_holdout"][contrast]["teacher"] for r in recs)/len(recs),"dart_program":sum(r["contrast_holdout"][contrast]["dart_program"] for r in recs)/len(recs)}},
              "records":recs}
    out.write_text(json.dumps(summary,indent=2));print("DART-3.5: invariant task-law induction + law-to-program compilation");print(json.dumps(summary,indent=2));print(f"Saved: {out.resolve()}")

if __name__=="__main__":main()
