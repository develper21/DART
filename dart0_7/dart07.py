#!/usr/bin/env python3
"""DART-0.7: Shared Computational Core + Minimal Residual.

Hypothesis:
    DART-0.6's pure shared trajectory operator was too restrictive. DART-0.7
    keeps a shared computational core across all trajectory steps and adds a
    tiny step-specific residual adapter:

        h_{i+1} = C(h_i) + R_i(h_i)

The residual is deliberately small and complexity-penalized. The goal is to
measure whether a reusable computational core can explain much of a learned
trajectory without becoming a disguised full neural replacement.
"""
from __future__ import annotations

import argparse, copy, json, math, random, statistics, time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

VOCAB = list("0123456789+= ")
STOI = {c: i for i, c in enumerate(VOCAB)}
PAD = STOI[" "]
BLOCK_SIZE = 12


def seed_everything(seed: int) -> None:
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def task_target(a: int, b: int, task: str) -> int:
    ad = [int(c) for c in str(a).zfill(3)]; bd = [int(c) for c in str(b).zfill(3)]
    if task == "add": return (ad[0] + bd[-1]) % 10
    if task == "sub": return (ad[-1] - bd[0]) % 10
    if task == "mul": return (ad[0] * bd[-1]) % 10
    if task == "sort": return min(ad + bd)
    if task == "compose": return ((ad[0] + bd[-1]) * (ad[1] + 1)) % 10
    raise ValueError(task)


def make_example(a: int, b: int, task: str):
    ids = [STOI[c] for c in f"{a}+{b}="]
    ids = (ids + [PAD] * BLOCK_SIZE)[:BLOCK_SIZE]
    return ids, task_target(a, b, task)


class TaskDataset(Dataset):
    def __init__(self, n: int, task: str, seed: int):
        rng = random.Random(seed); self.rows = []
        for _ in range(n):
            a, b = rng.randint(0, 999), rng.randint(0, 999)
            x, y = make_example(a, b, task); self.rows.append((torch.tensor(x), torch.tensor(y)))
    def __len__(self): return len(self.rows)
    def __getitem__(self, i): return self.rows[i]


class Block(nn.Module):
    def __init__(self, d: int, heads: int, d_ff: int):
        super().__init__(); self.norm1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=0.0, batch_first=True)
        self.norm2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, d_ff), nn.GELU(), nn.Linear(d_ff, d))
    def forward(self, x):
        h = self.norm1(x); a, _ = self.attn(h, h, h, need_weights=False); x = x + a
        return x + self.ff(self.norm2(x))


class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=32, heads=2, d_ff=128, depth=3):
        super().__init__(); self.d_model=d_model; self.depth=depth
        self.emb=nn.Embedding(vocab_size,d_model); self.pos=nn.Parameter(torch.randn(1,BLOCK_SIZE,d_model)*0.02)
        self.blocks=nn.ModuleList([Block(d_model,heads,d_ff) for _ in range(depth)]); self.head=nn.Linear(d_model,10)
    def forward(self,x):
        h=self.emb(x)+self.pos[:,:x.size(1)]
        for b in self.blocks: h=b(h)
        return self.head(h[:,0])


# ---------------- core families ----------------
class IdentityCore(nn.Module):
    def forward(self,x): return x

class DiagonalCore(nn.Module):
    def __init__(self,d):
        super().__init__(); self.scale=nn.Parameter(torch.ones(d)); self.bias=nn.Parameter(torch.zeros(d))
    def forward(self,x): return x*self.scale+self.bias

class PolynomialCore(nn.Module):
    def __init__(self,d):
        super().__init__(); self.a=nn.Parameter(torch.ones(d)); self.b=nn.Parameter(torch.zeros(d)); self.c=nn.Parameter(torch.zeros(d))
    def forward(self,x): return self.a*x+self.b*x.square()+self.c

class LowRankCore(nn.Module):
    def __init__(self,d,rank):
        super().__init__(); self.down=nn.Linear(d,rank,bias=False); self.up=nn.Linear(rank,d,bias=True)
        nn.init.zeros_(self.up.weight); nn.init.zeros_(self.up.bias)
    def forward(self,x): return x+self.up(self.down(x))

class MLPControl(nn.Module):
    def __init__(self,d,b): super().__init__(); self.net=nn.Sequential(nn.Linear(d,b),nn.GELU(),nn.Linear(b,d))
    def forward(self,x): return self.net(x)


def build_core(name,d,rank,bottleneck):
    if name=="identity": return IdentityCore()
    if name=="diagonal": return DiagonalCore(d)
    if name=="polynomial": return PolynomialCore(d)
    if name=="low_rank": return LowRankCore(d,rank)
    if name=="mlp": return MLPControl(d,bottleneck)
    raise ValueError(name)


class ResidualAdapter(nn.Module):
    """Tiny zero-initialized step-specific correction."""
    def __init__(self,d,rank):
        super().__init__(); self.down=nn.Linear(d,rank,bias=False); self.up=nn.Linear(rank,d,bias=True)
        nn.init.zeros_(self.down.weight); nn.init.zeros_(self.up.weight); nn.init.zeros_(self.up.bias)
    def forward(self,x): return self.up(self.down(x))


class CoreResidualStep(nn.Module):
    def __init__(self,core,residual): super().__init__(); self.core=core; self.residual=residual
    def forward(self,x): return self.core(x)+self.residual(x)


def install_core_residual(model,start,end,core,residual_rank):
    steps=end-start
    if steps<2: raise ValueError("Need at least two trajectory blocks")
    residuals=[ResidualAdapter(model.d_model,residual_rank) for _ in range(steps)]
    shared=CoreResidualStep(core,residuals[0])
    modules=[]
    for i in range(steps):
        if i==0: modules.append(shared)
        else: modules.append(CoreResidualStep(core,residuals[i]))
    original=list(model.blocks); model.blocks=nn.ModuleList(original[:start]+modules+original[end:])
    return residuals


def count_params(m): return sum(p.numel() for p in m.parameters())

def core_macs(m):
    if isinstance(m,IdentityCore): return 0
    if isinstance(m,DiagonalCore): return m.scale.numel()
    if isinstance(m,PolynomialCore): return 2*m.a.numel()
    if isinstance(m,LowRankCore): return m.down.in_features*m.down.out_features+m.up.in_features*m.up.out_features
    if isinstance(m,MLPControl): return sum(x.in_features*x.out_features for x in m.net if isinstance(x,nn.Linear))
    raise TypeError(type(m))

def residual_macs(rank,d): return 2*d*rank


@dataclass
class Eval:
    accuracy: float; loss: float; params: int; core_params: int; residual_params: int; core_macs: int; residual_macs: int


def metrics(model):
    corep=resip=coremac=resimac=0
    for b in model.blocks:
        if isinstance(b,CoreResidualStep):
            corep=count_params(b.core); coremac=core_macs(b.core)
            resip += count_params(b.residual); resimac += 2*b.residual.down.in_features*b.residual.down.out_features
    return corep,resip,coremac,resimac


def evaluate(model,loader,device):
    model.eval(); ce=nn.CrossEntropyLoss(reduction="sum"); total=correct=0; ls=0.0
    with torch.no_grad():
        for x,y in loader:
            x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True); z=model(x)
            ls+=float(ce(z,y)); correct+=int((z.argmax(-1)==y).sum()); total+=y.numel()
    cp,rp,cm,rm=metrics(model); return Eval(correct/max(total,1),ls/max(total,1),count_params(model),cp,rp,cm,rm)


def train(model,loader,device,steps,lr):
    params=[p for p in model.parameters() if p.requires_grad]
    if not params: raise RuntimeError("No trainable parameters")
    model.train(); opt=torch.optim.AdamW(params,lr=lr,weight_decay=1e-4); it=iter(loader)
    if device.type=="cuda": torch.cuda.synchronize()
    t=time.perf_counter(); ce=nn.CrossEntropyLoss()
    for _ in range(steps):
        try: x,y=next(it)
        except StopIteration: it=iter(loader); x,y=next(it)
        x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True); z=model(x); loss=ce(z,y)
        if not torch.isfinite(loss): raise RuntimeError("non-finite loss")
        opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(params,1.0); opt.step()
    if device.type=="cuda": torch.cuda.synchronize()
    return time.perf_counter()-t


def collect_trajectory(model,loader,device,start,end,max_batches):
    states=[[] for _ in range(end-start+1)]; hooks=[]
    hooks.append(model.blocks[start].register_forward_pre_hook(lambda _m,inp: states[0].append(inp[0].detach().cpu())))
    for j in range(start,end):
        k=j-start+1; hooks.append(model.blocks[j].register_forward_hook(lambda _m,_i,out,k=k: states[k].append(out.detach().cpu())))
    try:
        model.eval()
        with torch.no_grad():
            for bi,(x,_y) in enumerate(loader):
                if bi>=max_batches: break
                model(x.to(device,non_blocking=True))
    finally:
        for h in hooks: h.remove()
    return tuple(torch.cat(s,0) for s in states)


def fit_candidate(base,core,train_loader,states,device,start,end,steps,lr,residual_rank,residual_weight):
    model=copy.deepcopy(base).to(device)
    for p in model.parameters(): p.requires_grad=False
    core=core.to(device); residuals=[ResidualAdapter(base.d_model,residual_rank).to(device) for _ in range(end-start)]
    # Build a temporary model using a shared core and per-step residuals.
    original=list(model.blocks); steps_n=end-start
    replacement=[CoreResidualStep(core,residuals[i]) for i in range(steps_n)]
    model.blocks=nn.ModuleList(original[:start]+replacement+original[end:])
    trainable=list(core.parameters())+[p for r in residuals for p in r.parameters()]
    opt=torch.optim.AdamW(trainable,lr=lr,weight_decay=1e-4)
    ce=nn.CrossEntropyLoss(); it=iter(train_loader); ts=[s.to(device) for s in states]
    if device.type=="cuda": torch.cuda.synchronize()
    t=time.perf_counter(); n_total=ts[0].shape[0]
    for step in range(steps):
        try: x,y=next(it)
        except StopIteration: it=iter(train_loader); x,y=next(it)
        x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True)
        logits=model(x); task=ce(logits,y)
        n=min(128,n_total); idx=((step*128)+torch.arange(n,device=device))%n_total
        z=ts[0][idx]; tl=0.0; rl=0.0
        for k in range(steps_n):
            z=core(z)+residuals[k](z); tl=tl+torch.mean((z-ts[k+1][idx])**2); rl=rl+torch.mean(residuals[k](z.detach())**2)
        loss=task+0.05*tl+residual_weight*rl
        opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(trainable,1.0); opt.step()
    if device.type=="cuda": torch.cuda.synchronize()
    return core,residuals,model,time.perf_counter()-t


@torch.no_grad()
def trajectory_consistency(core,residuals,states,max_rows):
    n=min(max_rows,states[0].shape[0]); z=states[0][:n]; ms=[]
    for k in range(len(residuals)):
        z=core(z)+residuals[k](z); ms.append(float(torch.mean((z-states[k+1][:n])**2)))
    scale=float(torch.mean(states[-1][:n]**2))+1e-8; return math.exp(-sum(ms)/(len(ms)*scale))


def residual_fraction(core,residuals):
    cp=count_params(core); rp=sum(count_params(r) for r in residuals); return rp/max(cp+rp,1)

@dataclass
class Candidate:
    name:str; core_params:int; residual_params:int; core_macs:int; residual_macs:int; downstream_accuracy:float; downstream_loss:float; trajectory_consistency:float; residual_fraction:float; score:float; train_seconds:float


def candidate_search(base,train_loader,verifier_loader,train_states,verifier_states,device,args,seed):
    names=["identity","diagonal","polynomial","low_rank","mlp"]; out=[]
    for i,name in enumerate(names):
        seed_everything(seed+101*i)
        core=build_core(name,args.d_model,args.rank,args.bottleneck)
        core,residuals,model,secs=fit_candidate(base,core,train_loader,train_states,device,args.trajectory_start,args.trajectory_end,args.core_fit_steps,args.core_fit_lr,args.residual_rank,args.residual_weight)
        ev=evaluate(model,verifier_loader,device); tc=trajectory_consistency(core.cpu(),[r.cpu() for r in residuals],verifier_states,1024)
        rf=residual_fraction(core,residuals)
        complexity=math.log1p(count_params(core))+math.log1p(sum(count_params(r) for r in residuals))
        score=ev.accuracy+0.10*tc-args.complexity_lambda*complexity-0.10*rf
        out.append(Candidate(name,count_params(core),sum(count_params(r) for r in residuals),args.trajectory_end-args.trajectory_start and core_macs(core),sum(2*args.d_model*args.residual_rank for _ in residuals),ev.accuracy,ev.loss,tc,rf,score,secs))
    return sorted(out,key=lambda x:x.score,reverse=True)


def cuda_latency(model,loader,device,warmup,iters):
    if device.type!="cuda": return None
    x=next(iter(loader))[0].to(device,non_blocking=True); model.eval()
    for _ in range(warmup): model(x)
    torch.cuda.synchronize(); a=torch.cuda.Event(enable_timing=True); b=torch.cuda.Event(enable_timing=True); a.record()
    for _ in range(iters): model(x)
    b.record(); torch.cuda.synchronize(); return a.elapsed_time(b)/iters


def run_one(args,seed,task):
    seed_everything(seed); device=torch.device(args.device)
    tr=DataLoader(TaskDataset(args.train_size,task,seed),batch_size=args.batch_size,shuffle=True,pin_memory=device.type=="cuda")
    va=DataLoader(TaskDataset(args.verifier_size,task,seed+10000),batch_size=args.batch_size,shuffle=False,pin_memory=device.type=="cuda")
    te=DataLoader(TaskDataset(args.test_size,task,seed+20000),batch_size=args.batch_size,shuffle=False,pin_memory=device.type=="cuda")
    teacher=TinyTransformer(len(VOCAB),args.d_model,args.heads,args.d_ff,args.depth).to(device); secs=train(teacher,tr,device,args.teacher_steps,args.lr); tev=evaluate(teacher,te,device)
    ts=collect_trajectory(teacher,tr,device,args.trajectory_start,args.trajectory_end,args.trajectory_batches)
    vs=collect_trajectory(teacher,va,device,args.trajectory_start,args.trajectory_end,args.verifier_batches)
    dart=copy.deepcopy(teacher).to(device); rounds=[]
    for r in range(args.surgery_rounds):
        now=collect_trajectory(dart,tr,device,args.trajectory_start,args.trajectory_end,args.trajectory_batches)
        vnow=collect_trajectory(dart,va,device,args.trajectory_start,args.trajectory_end,args.verifier_batches)
        cand=candidate_search(dart,tr,va,now,vnow,device,args,seed+10000*r); win=cand[0]
        core=build_core(win.name,args.d_model,args.rank,args.bottleneck)
        core,residuals,patched,_=fit_candidate(dart,core,tr,now,device,args.trajectory_start,args.trajectory_end,args.core_fit_steps,args.core_fit_lr,args.residual_rank,args.residual_weight)
        # Install the fitted shared core + independent residuals on the live model.
        original=list(dart.blocks); reps=[CoreResidualStep(core,residuals[i]) for i in range(args.trajectory_end-args.trajectory_start)]
        dart.blocks=nn.ModuleList(original[:args.trajectory_start]+reps+original[args.trajectory_end:])
        pre=evaluate(dart,te,device); adapt=train(dart,tr,device,args.adaptation_steps_per_round,args.lr); post=evaluate(dart,te,device)
        rounds.append({"round":r,"winner":asdict(win),"all_candidates":[asdict(c) for c in cand],"pre_adapt":asdict(pre),"post_adapt":asdict(post),"adaptation_seconds":adapt,"latency_ms":cuda_latency(dart,te,device,args.latency_warmup,args.latency_iters)})
        print(f"round={r} winner={win.name} score={win.score:.4f} downstream={win.downstream_accuracy:.4f} core_consistency={win.trajectory_consistency:.4f} residual_fraction={win.residual_fraction:.4f} pre={pre.accuracy:.4f} post={post.accuracy:.4f}",flush=True)
    return {"seed":seed,"task":task,"teacher":asdict(tev),"teacher_training_seconds":secs,"dart_final":asdict(evaluate(dart,te,device)),"rounds":rounds}


def run_transfer(args,seed,source_task,target_task):
    device=torch.device(args.device)
    src=DataLoader(TaskDataset(args.train_size,source_task,seed),batch_size=args.batch_size,shuffle=True,pin_memory=device.type=="cuda")
    tgt=DataLoader(TaskDataset(args.train_size,target_task,seed+20000),batch_size=args.batch_size,shuffle=True,pin_memory=device.type=="cuda")
    test=DataLoader(TaskDataset(args.test_size,target_task,seed+30000),batch_size=args.batch_size,shuffle=False,pin_memory=device.type=="cuda")
    source=TinyTransformer(len(VOCAB),args.d_model,args.heads,args.d_ff,args.depth).to(device); train(source,src,device,args.teacher_steps,args.lr)
    st=collect_trajectory(source,src,device,args.trajectory_start,args.trajectory_end,args.trajectory_batches)
    cand=candidate_search(source,src,src,st,st,device,args,seed+7000)[0]
    core=build_core(cand.name,args.d_model,args.rank,args.bottleneck); core,residuals,_,_=fit_candidate(source,core,src,st,device,args.trajectory_start,args.trajectory_end,args.core_fit_steps,args.core_fit_lr,args.residual_rank,args.residual_weight)
    dart=copy.deepcopy(source).to(device); orig=list(dart.blocks); reps=[CoreResidualStep(core,residuals[i]) for i in range(args.trajectory_end-args.trajectory_start)]; dart.blocks=nn.ModuleList(orig[:args.trajectory_start]+reps+orig[args.trajectory_end:])
    scratch=TinyTransformer(len(VOCAB),args.d_model,args.heads,args.d_ff,args.depth).to(device)
    train(scratch,tgt,device,args.transfer_adaptation_steps,args.lr); train(dart,tgt,device,args.transfer_adaptation_steps,args.lr)
    s=evaluate(scratch,test,device); d=evaluate(dart,test,device)
    return {"seed":seed,"source_task":source_task,"target_task":target_task,"winner":asdict(cand),"scratch_after":asdict(s),"dart_after":asdict(d),"transfer_gain_points":100*(d.accuracy-s.accuracy)}


def mean_std(xs): return {"mean":statistics.mean(xs) if xs else None,"std":statistics.stdev(xs) if len(xs)>1 else 0.0}


def main():
    p=argparse.ArgumentParser(); p.add_argument("--seeds",nargs="+",type=int,default=[1,2]); p.add_argument("--tasks",nargs="+",default=["add","compose"]); p.add_argument("--transfer-pairs",nargs=2,action="append",metavar=("SOURCE","TARGET"),default=[["add","compose"],["mul","sub"]])
    p.add_argument("--train-size",type=int,default=6000); p.add_argument("--verifier-size",type=int,default=1500); p.add_argument("--test-size",type=int,default=1500); p.add_argument("--teacher-steps",type=int,default=800); p.add_argument("--core-fit-steps",type=int,default=300); p.add_argument("--adaptation-steps-per-round",type=int,default=400); p.add_argument("--surgery-rounds",type=int,default=2); p.add_argument("--transfer-adaptation-steps",type=int,default=400)
    p.add_argument("--d-model",type=int,default=32); p.add_argument("--heads",type=int,default=2); p.add_argument("--d-ff",type=int,default=128); p.add_argument("--depth",type=int,default=3); p.add_argument("--rank",type=int,default=8); p.add_argument("--bottleneck",type=int,default=32); p.add_argument("--residual-rank",type=int,default=2); p.add_argument("--trajectory-start",type=int,default=0); p.add_argument("--trajectory-end",type=int,default=3); p.add_argument("--trajectory-batches",type=int,default=20); p.add_argument("--verifier-batches",type=int,default=20); p.add_argument("--residual-weight",type=float,default=0.05); p.add_argument("--core-fit-lr",type=float,default=1e-3); p.add_argument("--lr",type=float,default=3e-4); p.add_argument("--complexity-lambda",type=float,default=1e-4); p.add_argument("--batch-size",type=int,default=256); p.add_argument("--latency-warmup",type=int,default=30); p.add_argument("--latency-iters",type=int,default=30); p.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu"); p.add_argument("--out",default="dart07_results.json")
    a=p.parse_args(); records=[]
    for task in a.tasks:
        print(f"\n===== TASK {task} =====",flush=True)
        for seed in a.seeds:
            print(f"seed={seed}",flush=True); records.append(run_one(a,seed,task))
    trs=[]
    print("\n===== TRANSFER =====",flush=True)
    for src,tgt in a.transfer_pairs:
        print(f"{src} -> {tgt}",flush=True)
        for seed in a.seeds:
            r=run_transfer(a,seed,src,tgt); trs.append(r); print(f" seed={seed} winner={r['winner']['name']} scratch={r['scratch_after']['accuracy']:.4f} dart={r['dart_after']['accuracy']:.4f} gain={r['transfer_gain_points']:+.2f} pts",flush=True)
    winfreq={}
    for r in records:
        for rr in r['rounds']: winfreq[rr['winner']['name']]=winfreq.get(rr['winner']['name'],0)+1
    ts={}
    for src,tgt in a.transfer_pairs:
        rows=[r for r in trs if r['source_task']==src and r['target_task']==tgt]; ts[f"{src}->{tgt}"]={"scratch_after":mean_std([r['scratch_after']['accuracy'] for r in rows]),"dart_after":mean_std([r['dart_after']['accuracy'] for r in rows]),"gain_points":mean_std([r['transfer_gain_points'] for r in rows])}
    out={"config":vars(a),"records":records,"transfer_records":trs,"summary":{"teacher":mean_std([r['teacher']['accuracy'] for r in records]),"dart_final":mean_std([r['dart_final']['accuracy'] for r in records]),"winner_frequency":winfreq,"transfer":ts}}
    Path(a.out).write_text(json.dumps(out,indent=2),encoding='utf-8'); print("\n================ DART-0.7 SUMMARY ================"); print(out['summary']); print(f"Saved: {Path(a.out).resolve()}")

if __name__=='__main__': main()
