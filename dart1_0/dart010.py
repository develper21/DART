#!/usr/bin/env python3
"""DART-1.0: Jointly Discovered Shared Primitive + Unseen-Task Transfer.

DART-1.0 changes the research unit from single-task replacement to a shared
computational primitive discovered jointly across multiple meta-training tasks.
The primitive is then frozen and inserted into routing-preserving models for
held-out tasks. This makes transfer a first-class requirement.
"""
from __future__ import annotations

import argparse, copy, json, math, random, statistics, time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Dict

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, ConcatDataset

VOCAB = list("0123456789+= ")
STOI = {c: i for i, c in enumerate(VOCAB)}
PAD = STOI[" "]
BLOCK_SIZE = 12


def seed_everything(seed: int):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def task_target(a: int, b: int, task: str) -> int:
    ad=[int(c) for c in str(a).zfill(3)]; bd=[int(c) for c in str(b).zfill(3)]
    if task == "add": return (ad[0] + bd[-1]) % 10
    if task == "sub": return (ad[-1] - bd[0]) % 10
    if task == "mul": return (ad[0] * bd[-1]) % 10
    if task == "sort": return min(ad + bd)
    if task == "compose": return ((ad[0] + bd[-1]) * (ad[1] + 1)) % 10
    raise ValueError(task)


def make_example(a: int, b: int, task: str):
    ids=[STOI[c] for c in f"{a}+{b}="]
    ids=(ids+[PAD]*BLOCK_SIZE)[:BLOCK_SIZE]
    return ids, task_target(a,b,task)


class TaskDataset(Dataset):
    def __init__(self, n: int, task: str, seed: int):
        rng=random.Random(seed); self.rows=[]
        for _ in range(n):
            a,b=rng.randint(0,999),rng.randint(0,999)
            x,y=make_example(a,b,task)
            self.rows.append((torch.tensor(x),torch.tensor(y)))
    def __len__(self): return len(self.rows)
    def __getitem__(self,i): return self.rows[i]


class Block(nn.Module):
    def __init__(self,d:int,heads:int,d_ff:int):
        super().__init__(); self.norm1=nn.LayerNorm(d)
        self.attn=nn.MultiheadAttention(d,heads,dropout=0.0,batch_first=True)
        self.norm2=nn.LayerNorm(d)
        self.ff=nn.Sequential(nn.Linear(d,d_ff),nn.GELU(),nn.Linear(d_ff,d))
    def forward(self,x):
        h=self.norm1(x); a,_=self.attn(h,h,h,need_weights=False); x=x+a
        return x+self.ff(self.norm2(x))


class TinyTransformer(nn.Module):
    def __init__(self,vocab_size,d_model=32,heads=2,d_ff=128,depth=3):
        super().__init__(); self.d_model=d_model; self.depth=depth
        self.emb=nn.Embedding(vocab_size,d_model)
        self.pos=nn.Parameter(torch.randn(1,BLOCK_SIZE,d_model)*0.02)
        self.blocks=nn.ModuleList([Block(d_model,heads,d_ff) for _ in range(depth)])
        self.head=nn.Linear(d_model,10)
    def forward(self,x,capture_attention=False):
        h=self.emb(x)+self.pos[:,:x.size(1)]; ats=[]
        for b in self.blocks:
            if capture_attention and isinstance(b, RoutingBlock):
                h,w=b.forward_capture(h); ats.append(w)
            else:
                if capture_attention:
                    n=b.norm1(h); a,w=b.attn(n,n,n,need_weights=True,average_attn_weights=False); h=h+a; h=h+b.ff(b.norm2(h)); ats.append(w)
                else: h=b(h)
        return (self.head(h[:,0]),ats) if capture_attention else self.head(h[:,0])


# ----- structured primitives -----
class IdentityCore(nn.Module):
    def forward(self,x): return torch.zeros_like(x)
class DiagonalCore(nn.Module):
    def __init__(self,d): super().__init__(); self.scale=nn.Parameter(torch.zeros(d)); self.bias=nn.Parameter(torch.zeros(d))
    def forward(self,x): return x*self.scale+self.bias
class PolynomialCore(nn.Module):
    def __init__(self,d): super().__init__(); self.a=nn.Parameter(torch.zeros(d)); self.b=nn.Parameter(torch.zeros(d)); self.c=nn.Parameter(torch.zeros(d))
    def forward(self,x): return self.a*x+self.b*x.square()+self.c
class AffinePolynomialCore(nn.Module):
    def __init__(self,d,rank):
        super().__init__(); self.down=nn.Linear(d,rank); self.up=nn.Linear(rank,d); self.quad=nn.Linear(rank,d,bias=False)
        nn.init.zeros_(self.up.weight); nn.init.zeros_(self.up.bias); nn.init.zeros_(self.quad.weight)
    def forward(self,x):
        h=self.down(x); return self.up(h)+self.quad(h.square())
class LowRankCore(nn.Module):
    def __init__(self,d,rank):
        super().__init__(); self.down=nn.Linear(d,rank,bias=False); self.up=nn.Linear(rank,d)
        nn.init.zeros_(self.up.weight); nn.init.zeros_(self.up.bias)
    def forward(self,x): return self.up(self.down(x))
class MLPControl(nn.Module):
    def __init__(self,d,b): super().__init__(); self.net=nn.Sequential(nn.Linear(d,b),nn.GELU(),nn.Linear(b,d))
    def forward(self,x): return self.net(x)

def build_core(name,d,rank,bottleneck):
    if name=="identity": return IdentityCore()
    if name=="diagonal": return DiagonalCore(d)
    if name=="polynomial": return PolynomialCore(d)
    if name=="affine_polynomial": return AffinePolynomialCore(d,rank)
    if name=="low_rank": return LowRankCore(d,rank)
    if name=="mlp": return MLPControl(d,bottleneck)
    raise ValueError(name)


class ResidualAdapter(nn.Module):
    def __init__(self,d,rank):
        super().__init__(); self.down=nn.Linear(d,rank,bias=False); self.up=nn.Linear(rank,d)
        nn.init.zeros_(self.down.weight); nn.init.zeros_(self.up.weight); nn.init.zeros_(self.up.bias)
    def forward(self,x): return self.up(self.down(x))


class RoutingBlock(nn.Module):
    def __init__(self,original,shared_core,residual):
        super().__init__(); self.norm1=copy.deepcopy(original.norm1); self.attn=copy.deepcopy(original.attn)
        self.norm2=copy.deepcopy(original.norm2); self.core=shared_core; self.residual=residual
    def forward(self,x):
        h=self.norm1(x); a,_=self.attn(h,h,h,need_weights=False); u=x+a; z=self.norm2(u)
        return u+self.core(z)+self.residual(z)
    def forward_capture(self,x):
        h=self.norm1(x); a,w=self.attn(h,h,h,need_weights=True,average_attn_weights=False); u=x+a; z=self.norm2(u)
        return u+self.core(z)+self.residual(z),w


@dataclass
class Eval:
    accuracy: float; loss: float; params: int; replace_params: int; replace_macs: int


def count_params(m): return sum(p.numel() for p in m.parameters())

def core_macs(m):
    if isinstance(m,IdentityCore): return 0
    if isinstance(m,DiagonalCore): return m.scale.numel()
    if isinstance(m,PolynomialCore): return 2*m.a.numel()
    if isinstance(m,AffinePolynomialCore):
        d=m.down.in_features; r=m.down.out_features; return 3*d*r
    if isinstance(m,LowRankCore): return m.down.in_features*m.down.out_features+m.up.in_features*m.up.out_features
    if isinstance(m,MLPControl): return sum(x.in_features*x.out_features for x in m.net if isinstance(x,nn.Linear))
    raise TypeError(type(m))

def residual_macs(rank,d): return 2*d*rank


def evaluate(model,loader,device):
    model.eval(); ce=nn.CrossEntropyLoss(reduction="sum"); total=correct=0; ls=0.0
    with torch.no_grad():
        for x,y in loader:
            x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True); z=model(x)
            ls+=float(ce(z,y)); correct+=int((z.argmax(-1)==y).sum()); total+=y.numel()
    rp=rm=0; seen=set()
    for b in model.blocks:
        if isinstance(b,RoutingBlock):
            if id(b.core) not in seen: rp+=count_params(b.core); rm+=core_macs(b.core); seen.add(id(b.core))
            rp+=count_params(b.residual); rm+=residual_macs(b.residual.down.out_features,b.residual.down.in_features)
    return Eval(correct/max(total,1),ls/max(total,1),count_params(model),rp,rm)


def train(model,loader,device,steps,lr):
    params=[p for p in model.parameters() if p.requires_grad]
    if not params: raise RuntimeError("No trainable parameters")
    model.train(); opt=torch.optim.AdamW(params,lr=lr,weight_decay=1e-4); it=iter(loader); ce=nn.CrossEntropyLoss()
    t=time.perf_counter()
    for _ in range(steps):
        try: x,y=next(it)
        except StopIteration: it=iter(loader); x,y=next(it)
        x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True)
        loss=ce(model(x),y); opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(params,1.0); opt.step()
    if device.type=="cuda": torch.cuda.synchronize()
    return time.perf_counter()-t


def freeze_all(m):
    for p in m.parameters(): p.requires_grad=False


def routing_stats(reference,candidate,max_rows=1024):
    if not reference or not candidate: return 0.0
    vals=[]
    for a,b in zip(reference,candidate):
        n=min(max_rows,a.shape[0],b.shape[0]); ra=a[:n].float(); rb=b[:n].float()
        vals.append(float(torch.mean((ra-rb)**2)/(torch.mean(ra**2)+1e-8)))
    return math.exp(-sum(vals)/max(len(vals),1))


def collect_routing(model,loader,device,max_batches):
    out=[[] for _ in model.blocks]; model.eval()
    with torch.no_grad():
        for bi,(x,_y) in enumerate(loader):
            if bi>=max_batches: break
            _z,ats=model(x.to(device,non_blocking=True),capture_attention=True)
            for i,a in enumerate(ats): out[i].append(a.cpu())
    return [torch.cat(v,0) for v in out]


def attention_ablation_accuracy(model,loader,device):
    model.eval(); handles=[]
    def hook(_m,_inp,out): return (torch.zeros_like(out[0]),)+out[1:] if isinstance(out,tuple) else torch.zeros_like(out)
    for b in model.blocks:
        if isinstance(b,RoutingBlock): handles.append(b.attn.register_forward_hook(hook))
    correct=total=0
    with torch.no_grad():
        for x,y in loader:
            x=x.to(device); y=y.to(device); z=model(x); correct+=int((z.argmax(-1)==y).sum()); total+=y.numel()
    for h in handles: h.remove()
    return correct/max(total,1)


def ensure_device(model,device):
    model.to(device)
    expected=torch.device(f"cuda:{torch.cuda.current_device()}") if device.type=="cuda" and device.index is None else device
    bad=[(n,p.device) for n,p in model.named_parameters() if p.device.type!=expected.type or (expected.index is not None and p.device.index!=expected.index)]
    if bad: raise RuntimeError(f"device mismatch: first={bad[0]} expected={expected}")


def install_shared(model,core,residual_rank,source_blocks,start,end):
    target_device=next(model.parameters()).device
    core=core.to(target_device)
    residuals=[ResidualAdapter(model.d_model,residual_rank).to(target_device) for _ in range(end-start)]
    reps=[RoutingBlock(source_blocks[i],core,residuals[i]).to(target_device) for i in range(start,end)]
    original=list(model.blocks); model.blocks=nn.ModuleList(original[:start]+reps+original[end:]); model.to(target_device)
    return residuals


def make_replaced_teacher(teacher,core_name,args,device):
    model=copy.deepcopy(teacher).to(device); core=build_core(core_name,args.d_model,args.rank,args.bottleneck).to(device)
    residuals=install_shared(model,core,args.residual_rank,list(teacher.blocks),args.trajectory_start,args.trajectory_end)
    freeze_all(model)
    for p in core.parameters(): p.requires_grad=True
    for r in residuals:
        for p in r.parameters(): p.requires_grad=True
    return model,core,residuals


def joint_fit(meta_models, meta_loaders, core, residuals_by_task, device,args,steps,distill_teachers=None):
    trainable=list(core.parameters())
    for rs in residuals_by_task:
        for r in rs: trainable+=list(r.parameters())
    opt=torch.optim.AdamW(trainable,lr=args.core_fit_lr,weight_decay=1e-4); ce=nn.CrossEntropyLoss(); mse=nn.MSELoss()
    its=[iter(dl) for dl in meta_loaders]
    for _ in range(steps):
        opt.zero_grad(set_to_none=True); losses=[]
        for ti,(model,it) in enumerate(zip(meta_models,its)):
            try: x,y=next(it)
            except StopIteration: its[ti]=iter(meta_loaders[ti]); x,y=next(its[ti])
            x=x.to(device); y=y.to(device); z=model(x)
            loss=ce(z,y)
            if distill_teachers is not None:
                with torch.no_grad(): tz=distill_teachers[ti](x)
                loss=loss+0.25*mse(z,tz)
            residual_pen=sum(torch.mean(p**2) for r in residuals_by_task[ti] for p in r.parameters())
            losses.append(loss+args.residual_weight*residual_pen)
        sum(losses).div(len(losses)).backward(); nn.utils.clip_grad_norm_(trainable,1.0); opt.step()


def fit_shared_candidate(teachers,meta_loaders,core_name,args,device,seed):
    seed_everything(seed); core=build_core(core_name,args.d_model,args.rank,args.bottleneck).to(device)
    models=[]; residuals_by_task=[]
    for t in teachers:
        m=copy.deepcopy(t).to(device); rs=install_shared(m,core,args.residual_rank,list(t.blocks),args.trajectory_start,args.trajectory_end)
        freeze_all(m)
        for p in core.parameters(): p.requires_grad=True
        for r in rs:
            for p in r.parameters(): p.requires_grad=True
        models.append(m); residuals_by_task.append(rs)
    t0=time.perf_counter(); joint_fit(models,meta_loaders,core,residuals_by_task,device,args,args.core_fit_steps); secs=time.perf_counter()-t0
    return models,core,residuals_by_task,secs


def joint_candidate_score(models,core,residuals_by_task,meta_loaders,ref_routes,device,args):
    accs=[]; routes=[]; drops=[]
    for m,dl,ref in zip(models,meta_loaders,ref_routes):
        ev=evaluate(m,dl,device); accs.append(ev.accuracy); routes.append(routing_stats(ref,collect_routing(m,dl,device,args.verifier_batches))); abl=attention_ablation_accuracy(m,dl,device); drops.append(ev.accuracy-abl)
    avg_acc=statistics.mean(accs); avg_route=statistics.mean(routes); avg_drop=statistics.mean(drops)
    unique_core_params=count_params(core); unique_res=sum(count_params(r) for rs in residuals_by_task for r in rs)
    rf=unique_res/max(unique_core_params+unique_res,1)
    score=avg_acc+args.routing_weight*avg_route+args.ablation_weight*max(0.0,avg_drop)-args.complexity_lambda*math.log1p(unique_core_params)-args.residual_weight*rf
    return {"avg_accuracy":avg_acc,"avg_routing":avg_route,"avg_ablation_drop":avg_drop,"shared_core_params":unique_core_params,"shared_core_macs":core_macs(core),"total_residual_params":unique_res,"residual_fraction":rf,"score":score,"task_accuracies":accs,"task_routing":routes}


def shared_candidate_search(teachers,meta_loaders,ref_routes,args,device,seed):
    structured=["identity","diagonal","polynomial","affine_polynomial","low_rank"]
    rows=[]
    for i,name in enumerate(structured):
        models,core,rs,secs=fit_shared_candidate(teachers,meta_loaders,name,args,device,seed+31*i)
        s=joint_candidate_score(models,core,rs,meta_loaders,ref_routes,device,args); s.update(name=name,kind="dart_structured",eligible=True,fit_seconds=secs); rows.append(s)
    # MLP control is fit jointly, but cannot become the DART winner.
    models,core,rs,secs=fit_shared_candidate(teachers,meta_loaders,"mlp",args,device,seed+9001)
    s=joint_candidate_score(models,core,rs,meta_loaders,ref_routes,device,args); s.update(name="mlp_control",kind="neural_control",eligible=False,fit_seconds=secs); rows.append(s)
    return rows


def build_meta_teachers(meta_tasks,args,device,seed):
    teachers=[]; loaders=[]; verifiers=[]; routes=[]
    for i,task in enumerate(meta_tasks):
        tr=DataLoader(TaskDataset(args.train_size,task,seed+i*1000),batch_size=args.batch_size,shuffle=True,pin_memory=device.type=="cuda")
        va=DataLoader(TaskDataset(args.verifier_size,task,seed+i*2000+10000),batch_size=args.batch_size,shuffle=False,pin_memory=device.type=="cuda")
        t=TinyTransformer(len(VOCAB),args.d_model,args.heads,args.d_ff,args.depth).to(device); train(t,tr,device,args.teacher_steps,args.lr)
        teachers.append(t); loaders.append(tr); verifiers.append(va); routes.append(collect_routing(t,va,device,args.verifier_batches))
    return teachers,loaders,verifiers,routes


def adapt_joint(models,loaders,core,residuals_by_task,device,args):
    # During meta adaptation the shared core remains trainable; this mirrors DART-0.9.
    t0=time.perf_counter(); joint_fit(models,loaders,core,residuals_by_task,device,args,args.adaptation_steps_per_round); return time.perf_counter()-t0


def instantiate_frozen_transfer(teacher,shared_core,args,device):
    model=copy.deepcopy(teacher).to(device)
    frozen_core=copy.deepcopy(shared_core).to(device)
    residuals=install_shared(model,frozen_core,args.residual_rank,list(teacher.blocks),args.trajectory_start,args.trajectory_end)
    freeze_all(model)
    # Only target residual adapters are allowed to learn; shared core is frozen.
    for r in residuals:
        for p in r.parameters(): p.requires_grad=True
    return model,frozen_core,residuals


def train_target_residual(model,target_loader,device,args):
    params=[p for p in model.parameters() if p.requires_grad]; opt=torch.optim.AdamW(params,lr=args.lr,weight_decay=1e-4); ce=nn.CrossEntropyLoss(); it=iter(target_loader)
    for _ in range(args.transfer_adaptation_steps):
        try: x,y=next(it)
        except StopIteration: it=iter(target_loader); x,y=next(it)
        x=x.to(device); y=y.to(device); loss=ce(model(x),y); opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(params,1.0); opt.step()


def run_leave_one_out(args,seed,holdout,all_tasks):
    meta_tasks=[t for t in all_tasks if t!=holdout]
    device=torch.device(args.device)
    teachers,meta_loaders,meta_verifiers,ref_routes=build_meta_teachers(meta_tasks,args,device,seed)
    rows=[]
    current_teachers=teachers
    shared_core=None
    for r in range(args.surgery_rounds):
        cand=shared_candidate_search(current_teachers,meta_loaders,ref_routes,args,device,seed+1000*r)
        structured=[x for x in cand if x["eligible"]]
        win=max(structured,key=lambda x:x["score"])
        models,core,residuals,fit_secs=fit_shared_candidate(current_teachers,meta_loaders,win["name"],args,device,seed+8000*r)
        pre=[evaluate(m,dl,device).accuracy for m,dl in zip(models,meta_loaders)]
        adapt_secs=adapt_joint(models,meta_loaders,core,residuals,device,args)
        post=[evaluate(m,dl,device).accuracy for m,dl in zip(models,meta_loaders)]
        post_routes=[routing_stats(ref_routes[i],collect_routing(models[i],meta_verifiers[i],device,args.verifier_batches)) for i in range(len(models))]
        rows.append({"round":r,"winner":win,"all_candidates":cand,"meta_pre_accuracy":pre,"meta_post_accuracy":post,"meta_routing_after":post_routes,"fit_seconds":fit_secs,"adaptation_seconds":adapt_secs})
        current_teachers=models; shared_core=core
        print(f"  round={r} winner={win['name']} meta_pre={statistics.mean(pre):.4f} meta_post={statistics.mean(post):.4f} shared_core_params={win['shared_core_params']} route={win['avg_routing']:.4f}",flush=True)
    # Unseen task teacher/control
    ttrain=DataLoader(TaskDataset(args.train_size,holdout,seed+50000),batch_size=args.batch_size,shuffle=True,pin_memory=device.type=="cuda")
    ttest=DataLoader(TaskDataset(args.test_size,holdout,seed+60000),batch_size=args.batch_size,shuffle=False,pin_memory=device.type=="cuda")
    target_teacher=TinyTransformer(len(VOCAB),args.d_model,args.heads,args.d_ff,args.depth).to(device); train(target_teacher,ttrain,device,args.teacher_steps,args.lr)
    tev=evaluate(target_teacher,ttest,device)
    # Frozen shared primitive, zero-shot and residual-only adaptation.
    transfer_model,frozen_core,_=instantiate_frozen_transfer(target_teacher,shared_core,args,device)
    zero=evaluate(transfer_model,ttest,device)
    train_target_residual(transfer_model,ttrain,device,args)
    adapted=evaluate(transfer_model,ttest,device)
    # Matched neural control on unseen task.
    control,_core,_res=make_replaced_teacher(target_teacher,"mlp",args,device)
    ctrain=[p for p in control.parameters() if p.requires_grad]; opt=torch.optim.AdamW(ctrain,lr=args.core_fit_lr,weight_decay=1e-4); ce=nn.CrossEntropyLoss(); it=iter(ttrain)
    for _ in range(args.transfer_adaptation_steps):
        try:x,y=next(it)
        except StopIteration:it=iter(ttrain);x,y=next(it)
        x=x.to(device);y=y.to(device);loss=ce(control(x),y);opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(ctrain,1.0);opt.step()
    ctrl=evaluate(control,ttest,device)
    return {"holdout_task":holdout,"meta_tasks":meta_tasks,"rounds":rows,"meta_teacher_accuracies":[evaluate(t,dl,device).accuracy for t,dl in zip(teachers,meta_loaders)],"heldout_teacher":asdict(tev),"frozen_shared_core":{"accuracy":zero.accuracy,"loss":zero.loss,"params":zero.params,"replace_params":zero.replace_params,"replace_macs":zero.replace_macs},"frozen_core_after_residual_adaptation":asdict(adapted),"matched_mlp_control":asdict(ctrl),"transfer_gain_zero_shot_points":100*(zero.accuracy-tev.accuracy),"transfer_gain_adapted_points":100*(adapted.accuracy-tev.accuracy),"vs_mlp_control_points":100*(adapted.accuracy-ctrl.accuracy),"shared_core_params":count_params(shared_core),"shared_core_macs":core_macs(shared_core)}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--all-tasks",nargs="+",default=["add","compose","mul","sub"])
    p.add_argument("--holdout-tasks",nargs="+",default=["sub"])
    p.add_argument("--seeds",nargs="+",type=int,default=[1,2])
    p.add_argument("--train-size",type=int,default=6000); p.add_argument("--verifier-size",type=int,default=1500); p.add_argument("--test-size",type=int,default=1500)
    p.add_argument("--teacher-steps",type=int,default=800); p.add_argument("--core-fit-steps",type=int,default=300); p.add_argument("--adaptation-steps-per-round",type=int,default=400); p.add_argument("--surgery-rounds",type=int,default=2); p.add_argument("--transfer-adaptation-steps",type=int,default=400)
    p.add_argument("--d-model",type=int,default=32); p.add_argument("--heads",type=int,default=2); p.add_argument("--d-ff",type=int,default=128); p.add_argument("--depth",type=int,default=3); p.add_argument("--rank",type=int,default=8); p.add_argument("--bottleneck",type=int,default=32); p.add_argument("--residual-rank",type=int,default=2)
    p.add_argument("--trajectory-start",type=int,default=0); p.add_argument("--trajectory-end",type=int,default=3); p.add_argument("--verifier-batches",type=int,default=20); p.add_argument("--residual-weight",type=float,default=0.01); p.add_argument("--routing-weight",type=float,default=0.20); p.add_argument("--ablation-weight",type=float,default=0.10); p.add_argument("--core-fit-lr",type=float,default=1e-3); p.add_argument("--lr",type=float,default=3e-4); p.add_argument("--complexity-lambda",type=float,default=1e-4); p.add_argument("--batch-size",type=int,default=256); p.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu"); p.add_argument("--out",default="dart010_results.json")
    a=p.parse_args(); device=torch.device(a.device)
    print("DART-1.0: joint shared primitive discovery + frozen unseen-task transfer",flush=True)
    records=[]
    for holdout in a.holdout_tasks:
        print(f"\n===== HOLDOUT TASK {holdout} =====",flush=True)
        for seed in a.seeds:
            print(f"seed={seed} meta={','.join(t for t in a.all_tasks if t!=holdout)} -> holdout={holdout}",flush=True)
            records.append(run_leave_one_out(a,seed,holdout,a.all_tasks))
    summary={"holdout_transfer":{}}
    for h in a.holdout_tasks:
        rs=[r for r in records if r["holdout_task"]==h]
        summary["holdout_transfer"][h]={"teacher":statistics.mean([r["heldout_teacher"]["accuracy"] for r in rs]),"zero_shot":statistics.mean([r["frozen_shared_core"]["accuracy"] for r in rs]),"adapted":statistics.mean([r["frozen_core_after_residual_adaptation"]["accuracy"] for r in rs]),"mlp_control":statistics.mean([r["matched_mlp_control"]["accuracy"] for r in rs]),"zero_shot_gain_points":statistics.mean([r["transfer_gain_zero_shot_points"] for r in rs]),"adapted_gain_points":statistics.mean([r["transfer_gain_adapted_points"] for r in rs]),"vs_mlp_points":statistics.mean([r["vs_mlp_control_points"] for r in rs])}
    out={"config":vars(a),"records":records,"summary":summary}
    Path(a.out).write_text(json.dumps(out,indent=2),encoding="utf-8")
    print("\n================ DART-1.0 SUMMARY ================",flush=True); print(summary); print(f"Saved: {Path(a.out).resolve()}")

if __name__=="__main__": main()
