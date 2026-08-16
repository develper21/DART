#!/usr/bin/env python3
"""DART-1.1: Tiny task-conditioned shared primitive + frozen unseen-task transfer.

DART-1.0 failure:
- shared structured primitive was discovered across meta tasks,
- but 1,440 task-specific residual parameters dominated the replacement,
- and the frozen primitive did not transfer to the held-out task.

DART-1.1 hypothesis:
A genuinely reusable primitive should need only a tiny task-conditioning code
once the shared computational core has been discovered. The shared core and
its task-conditioning map are frozen for unseen-task transfer; only a tiny
k-dimensional task code is adapted on the holdout task.

This keeps the routing-preserving Transformer pathway from DART-0.8/0.9.
"""
from __future__ import annotations

import argparse, copy, json, math, random, statistics, time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

VOCAB = list("0123456789+= ")
STOI = {c: i for i, c in enumerate(VOCAB)}
PAD = STOI[" "]
BLOCK_SIZE = 12


def seed_everything(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def task_target(a: int, b: int, task: str) -> int:
    ad = [int(c) for c in str(a).zfill(3)]
    bd = [int(c) for c in str(b).zfill(3)]
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
        rng = random.Random(seed)
        self.rows = []
        for _ in range(n):
            a, b = rng.randint(0, 999), rng.randint(0, 999)
            x, y = make_example(a, b, task)
            self.rows.append((torch.tensor(x), torch.tensor(y)))

    def __len__(self): return len(self.rows)
    def __getitem__(self, i): return self.rows[i]


class Block(nn.Module):
    def __init__(self, d: int, heads: int, d_ff: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=0.0, batch_first=True)
        self.norm2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, d_ff), nn.GELU(), nn.Linear(d_ff, d))

    def forward(self, x):
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a
        return x + self.ff(self.norm2(x))


class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=32, heads=2, d_ff=128, depth=3):
        super().__init__()
        self.d_model = d_model
        self.depth = depth
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Parameter(torch.randn(1, BLOCK_SIZE, d_model) * 0.02)
        self.blocks = nn.ModuleList([Block(d_model, heads, d_ff) for _ in range(depth)])
        self.head = nn.Linear(d_model, 10)

    def forward(self, x, capture_attention=False):
        h = self.emb(x) + self.pos[:, :x.size(1)]
        ats = []
        for b in self.blocks:
            if capture_attention and isinstance(b, RoutingBlock):
                h, w = b.forward_capture(h)
                ats.append(w)
            elif capture_attention:
                n = b.norm1(h)
                a, w = b.attn(n, n, n, need_weights=True, average_attn_weights=False)
                h = h + a
                h = h + b.ff(b.norm2(h))
                ats.append(w)
            else:
                h = b(h)
        return (self.head(h[:, 0]), ats) if capture_attention else self.head(h[:, 0])


# ---- Structured base primitives ----
class IdentityCore(nn.Module):
    def forward(self, x): return torch.zeros_like(x)


class DiagonalCore(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(d))
        self.bias = nn.Parameter(torch.zeros(d))

    def forward(self, x): return x * self.scale + self.bias


class PolynomialCore(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.a = nn.Parameter(torch.zeros(d))
        self.b = nn.Parameter(torch.zeros(d))
        self.c = nn.Parameter(torch.zeros(d))

    def forward(self, x): return self.a * x + self.b * x.square() + self.c


class AffinePolynomialCore(nn.Module):
    def __init__(self, d, rank):
        super().__init__()
        self.down = nn.Linear(d, rank)
        self.up = nn.Linear(rank, d)
        self.quad = nn.Linear(rank, d, bias=False)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        nn.init.zeros_(self.quad.weight)

    def forward(self, x):
        h = self.down(x)
        return self.up(h) + self.quad(h.square())


class LowRankCore(nn.Module):
    def __init__(self, d, rank):
        super().__init__()
        self.down = nn.Linear(d, rank, bias=False)
        self.up = nn.Linear(rank, d)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x): return self.up(self.down(x))


class MLPControl(nn.Module):
    def __init__(self, d, b):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, b), nn.GELU(), nn.Linear(b, d))

    def forward(self, x): return self.net(x)


def build_core(name, d, rank, bottleneck):
    if name == "identity": return IdentityCore()
    if name == "diagonal": return DiagonalCore(d)
    if name == "polynomial": return PolynomialCore(d)
    if name == "affine_polynomial": return AffinePolynomialCore(d, rank)
    if name == "low_rank": return LowRankCore(d, rank)
    if name == "mlp": return MLPControl(d, bottleneck)
    raise ValueError(name)


class TaskConditioner(nn.Module):
    """Shared map from tiny task code to a multiplicative/additive core modulation."""
    def __init__(self, task_dim: int, feature_dim: int):
        super().__init__()
        self.gamma = nn.Linear(task_dim, feature_dim)
        self.beta = nn.Linear(task_dim, feature_dim)
        nn.init.zeros_(self.gamma.weight); nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight); nn.init.zeros_(self.beta.bias)

    def forward(self, core_out: Tensor, task_code: Tensor):
        # task_code: [B, task_dim] or [task_dim].
        if task_code.dim() == 1:
            task_code = task_code.unsqueeze(0)
        g = torch.tanh(self.gamma(task_code)).unsqueeze(1)
        b = self.beta(task_code).unsqueeze(1)
        return core_out * (1.0 + g) + b


class ConditionedCore(nn.Module):
    def __init__(self, base_core: nn.Module, d: int, task_dim: int):
        super().__init__()
        self.base = base_core
        self.task_conditioner = TaskConditioner(task_dim, d)

    def forward(self, x, task_code):
        return self.task_conditioner(self.base(x), task_code)


class RoutingBlock(nn.Module):
    def __init__(self, original, conditioned_core):
        super().__init__()
        self.norm1 = copy.deepcopy(original.norm1)
        self.attn = copy.deepcopy(original.attn)
        self.norm2 = copy.deepcopy(original.norm2)
        self.core = conditioned_core

    def forward(self, x, task_code):
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        u = x + a
        z = self.norm2(u)
        return u + self.core(z, task_code)

    def forward_capture(self, x, task_code):
        h = self.norm1(x)
        a, w = self.attn(h, h, h, need_weights=True, average_attn_weights=False)
        u = x + a
        z = self.norm2(u)
        return u + self.core(z, task_code), w


class TaskConditionedTransformer(nn.Module):
    def __init__(self, teacher, shared_core, task_code, start, end):
        super().__init__()
        self.d_model = teacher.d_model
        self.depth = teacher.depth
        self.emb = copy.deepcopy(teacher.emb)
        self.pos = copy.deepcopy(teacher.pos)
        self.head = copy.deepcopy(teacher.head)
        self.shared_core = shared_core
        self.task_code = nn.Parameter(task_code.detach().clone())
        blocks = []
        for i, b in enumerate(teacher.blocks):
            if start <= i < end:
                blocks.append(RoutingBlock(b, shared_core))
            else:
                blocks.append(copy.deepcopy(b))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x, capture_attention=False, override_code=None):
        code = self.task_code if override_code is None else override_code
        h = self.emb(x) + self.pos[:, :x.size(1)]
        ats = []
        for b in self.blocks:
            if isinstance(b, RoutingBlock):
                if capture_attention:
                    h, w = b.forward_capture(h, code); ats.append(w)
                else:
                    h = b(h, code)
            elif capture_attention:
                n = b.norm1(h); a, w = b.attn(n, n, n, need_weights=True, average_attn_weights=False)
                h = h + a; h = h + b.ff(b.norm2(h)); ats.append(w)
            else:
                h = b(h)
        return (self.head(h[:, 0]), ats) if capture_attention else self.head(h[:, 0])


def count_params(m): return sum(p.numel() for p in m.parameters())


def core_macs(m):
    if isinstance(m, ConditionedCore):
        return core_macs(m.base) + 2 * m.task_conditioner.gamma.in_features * m.task_conditioner.gamma.out_features
    if isinstance(m, IdentityCore): return 0
    if isinstance(m, DiagonalCore): return m.scale.numel()
    if isinstance(m, PolynomialCore): return 2 * m.a.numel()
    if isinstance(m, AffinePolynomialCore):
        d, r = m.down.in_features, m.down.out_features
        return 3 * d * r
    if isinstance(m, LowRankCore):
        return m.down.in_features * m.down.out_features + m.up.in_features * m.up.out_features
    if isinstance(m, MLPControl):
        return sum(x.in_features * x.out_features for x in m.net if isinstance(x, nn.Linear))
    raise TypeError(type(m))


def condition_macs(core, task_dim):
    d = next(p for p in core.parameters()).shape[-1] if any(True for _ in core.parameters()) else 0
    # Explicit accounting for the shared gamma/beta maps.
    if hasattr(core, "task_conditioner"):
        tc = core.task_conditioner
        feat = tc.gamma.out_features; td = tc.gamma.in_features
        return 2 * feat * td
    return 0


def evaluate(model, loader, device):
    model.eval(); ce = nn.CrossEntropyLoss(reduction="sum"); total = correct = 0; loss_sum = 0.0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            z = model(x); loss_sum += float(ce(z, y)); correct += int((z.argmax(-1) == y).sum()); total += y.numel()
    return {"accuracy": correct/max(total,1), "loss": loss_sum/max(total,1), "params": count_params(model)}


def train_model(model, loader, device, steps, lr):
    params = [p for p in model.parameters() if p.requires_grad]
    if not params: raise RuntimeError("No trainable parameters")
    model.train(); opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4); it = iter(loader); ce = nn.CrossEntropyLoss()
    t0 = time.perf_counter()
    for _ in range(steps):
        try: x, y = next(it)
        except StopIteration: it = iter(loader); x, y = next(it)
        x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
        loss = ce(model(x), y); opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(params,1.0); opt.step()
    if device.type == "cuda": torch.cuda.synchronize()
    return time.perf_counter() - t0


def freeze(module):
    for p in module.parameters(): p.requires_grad = False


def collect_routing(model, loader, device, max_batches):
    out = [[] for _ in model.blocks]; model.eval()
    with torch.no_grad():
        for bi, (x, _y) in enumerate(loader):
            if bi >= max_batches: break
            _, ats = model(x.to(device, non_blocking=True), capture_attention=True)
            for i, a in enumerate(ats): out[i].append(a.cpu())
    return [torch.cat(v,0) if v else torch.empty(0) for v in out]


def routing_stats(reference, candidate, max_rows=1024):
    if not reference or not candidate: return 0.0
    vals=[]
    for a,b in zip(reference,candidate):
        if a.numel()==0 or b.numel()==0: continue
        n=min(max_rows,a.shape[0],b.shape[0]); ra=a[:n].float(); rb=b[:n].float()
        vals.append(float(torch.mean((ra-rb)**2)/(torch.mean(ra**2)+1e-8)))
    return math.exp(-sum(vals)/max(len(vals),1)) if vals else 0.0


def attention_ablation_accuracy(model, loader, device):
    model.eval(); handles=[]
    def hook(_m,_inp,out):
        if isinstance(out,tuple): return (torch.zeros_like(out[0]),)+out[1:]
        return torch.zeros_like(out)
    for b in model.blocks:
        if isinstance(b,RoutingBlock): handles.append(b.attn.register_forward_hook(hook))
    correct=total=0
    with torch.no_grad():
        for x,y in loader:
            z=model(x.to(device)); yy=y.to(device); correct += int((z.argmax(-1)==yy).sum()); total += yy.numel()
    for h in handles: h.remove()
    return correct/max(total,1)


def make_teacher(task, args, device, seed):
    tr=DataLoader(TaskDataset(args.train_size,task,seed),batch_size=args.batch_size,shuffle=True,pin_memory=device.type=="cuda")
    va=DataLoader(TaskDataset(args.verifier_size,task,seed+10000),batch_size=args.batch_size,shuffle=False,pin_memory=device.type=="cuda")
    te=TinyTransformer(len(VOCAB),args.d_model,args.heads,args.d_ff,args.depth).to(device)
    train_model(te,tr,device,args.teacher_steps,args.lr)
    return te,tr,va


def build_meta_teachers(meta_tasks,args,device,seed):
    out=[]
    for i,task in enumerate(meta_tasks): out.append(make_teacher(task,args,device,seed+i*1000))
    return out


def initial_code(task_dim, device, seed_offset):
    g=torch.Generator(device=device); g.manual_seed(seed_offset); return 0.02*torch.randn(task_dim, generator=g, device=device)


def make_candidate_models(teachers, name, args, device, seed):
    core = ConditionedCore(build_core(name,args.d_model,args.rank,args.bottleneck),args.d_model,args.task_dim).to(device)
    models=[]
    for i,t in enumerate(teachers):
        code=initial_code(args.task_dim,device,seed+100*i)
        m=TaskConditionedTransformer(t,core,code,args.trajectory_start,args.trajectory_end).to(device)
        freeze(m)
        # Shared core is separately referenced; task_code is trainable.
        for p in core.parameters(): p.requires_grad=True
        m.task_code.requires_grad=True
        models.append(m)
    return core,models


def candidate_parameters(core,models):
    params=[]
    seen=set()
    for p in core.parameters():
        if id(p) not in seen: params.append(p); seen.add(id(p))
    for m in models:
        params.append(m.task_code)
    return params


def fit_joint_candidate(teachers, meta_loaders, name, args, device, seed):
    seed_everything(seed); core,models=make_candidate_models(teachers,name,args,device,seed)
    params=candidate_parameters(core,models); opt=torch.optim.AdamW(params,lr=args.core_fit_lr,weight_decay=1e-4); ce=nn.CrossEntropyLoss(); its=[iter(dl) for dl in meta_loaders]
    t0=time.perf_counter()
    for _ in range(args.core_fit_steps):
        opt.zero_grad(set_to_none=True); losses=[]
        for i,(m,it) in enumerate(zip(models,its)):
            try:x,y=next(it)
            except StopIteration:its[i]=iter(meta_loaders[i]);x,y=next(its[i])
            x=x.to(device);y=y.to(device); loss=ce(m(x),y)
            # Tiny-code penalty: prevents hiding the whole task in z.
            loss=loss+args.code_weight*torch.mean(m.task_code.square())
            losses.append(loss)
        sum(losses).div(len(losses)).backward(); nn.utils.clip_grad_norm_(params,1.0); opt.step()
    if device.type=="cuda":torch.cuda.synchronize()
    return core,models,time.perf_counter()-t0


def candidate_score(core,models,meta_loaders,ref_routes,device,args):
    acc=[]; routes=[]; drops=[]
    for i,(m,dl,ref) in enumerate(zip(models,meta_loaders,ref_routes)):
        ev=evaluate(m,dl,device); acc.append(ev["accuracy"])
        routes.append(routing_stats(ref,collect_routing(m,dl,device,args.verifier_batches)))
        drops.append(ev["accuracy"]-attention_ablation_accuracy(m,dl,device))
    shared_core_params=count_params(core); shared_core_macs=core_macs(core); cond_params=count_params(core.task_conditioner)
    code_params=len(models)*args.task_dim
    replace_params=shared_core_params+cond_params+code_params
    rf=code_params/max(replace_params,1)
    score=statistics.mean(acc)+args.routing_weight*statistics.mean(routes)+args.ablation_weight*max(0,statistics.mean(drops))-args.complexity_lambda*math.log1p(replace_params)-args.code_weight*rf
    return {"avg_accuracy":statistics.mean(acc),"avg_routing":statistics.mean(routes),"avg_ablation_drop":statistics.mean(drops),"shared_core_params":shared_core_params,"shared_core_macs":shared_core_macs,"conditioner_params":cond_params,"total_task_code_params":code_params,"task_code_fraction":rf,"replace_params":replace_params,"replace_macs":shared_core_macs,"score":score,"task_accuracies":acc,"task_routing":routes}


def search(teachers,meta_loaders,ref_routes,args,device,seed):
    rows=[]
    for i,name in enumerate(["identity","diagonal","polynomial","affine_polynomial","low_rank"]):
        core,models,secs=fit_joint_candidate(teachers,meta_loaders,name,args,device,seed+31*i)
        s=candidate_score(core,models,meta_loaders,ref_routes,device,args); s.update(name=name,kind="dart_structured",eligible=True,fit_seconds=secs); rows.append(s)
    # MLP control is still reported, never eligible as DART.
    core,models,secs=fit_joint_candidate(teachers,meta_loaders,"mlp",args,device,seed+9001)
    s=candidate_score(core,models,meta_loaders,ref_routes,device,args); s.update(name="mlp_control",kind="neural_control",eligible=False,fit_seconds=secs); rows.append(s)
    return rows


def meta_adapt(models,core,meta_loaders,args,device):
    params=candidate_parameters(core,models); opt=torch.optim.AdamW(params,lr=args.core_fit_lr,weight_decay=1e-4); ce=nn.CrossEntropyLoss(); its=[iter(dl) for dl in meta_loaders]
    t0=time.perf_counter()
    for _ in range(args.adaptation_steps_per_round):
        opt.zero_grad(set_to_none=True); losses=[]
        for i,(m,it) in enumerate(zip(models,its)):
            try:x,y=next(it)
            except StopIteration:its[i]=iter(meta_loaders[i]);x,y=next(its[i])
            x=x.to(device);y=y.to(device); loss=ce(m(x),y)+args.code_weight*torch.mean(m.task_code.square()); losses.append(loss)
        sum(losses).div(len(losses)).backward();nn.utils.clip_grad_norm_(params,1.0);opt.step()
    return time.perf_counter()-t0


def frozen_transfer(target_teacher,shared_core,meta_codes,args,device,holdout_loader,code_mode):
    # Freeze the shared computational primitive, including its conditioner map.
    core=copy.deepcopy(shared_core).to(device); freeze(core)
    if code_mode=="centroid": code=torch.stack([c.detach() for c in meta_codes]).mean(0).to(device)
    elif code_mode=="zero": code=torch.zeros(args.task_dim,device=device)
    else: raise ValueError(code_mode)
    model=TaskConditionedTransformer(target_teacher,core,code,args.trajectory_start,args.trajectory_end).to(device); freeze(model); model.task_code.requires_grad=False
    ev=evaluate(model,holdout_loader,device)
    return model,ev,core


def adapt_only_task_code(model,loader,args,device):
    model.task_code.requires_grad=True
    opt=torch.optim.AdamW([model.task_code],lr=args.code_lr,weight_decay=0.0); ce=nn.CrossEntropyLoss(); it=iter(loader)
    for _ in range(args.transfer_adaptation_steps):
        try:x,y=next(it)
        except StopIteration:it=iter(loader);x,y=next(it)
        x=x.to(device);y=y.to(device);loss=ce(model(x),y)+args.code_weight*torch.mean(model.task_code.square())
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_([model.task_code],1.0);opt.step()
    return evaluate(model,loader,args.device_obj)


def matched_mlp_control(target_teacher,args,device,loader):
    core=ConditionedCore(MLPControl(args.d_model,args.bottleneck),args.d_model,args.task_dim).to(device)
    code=torch.zeros(args.task_dim,device=device)
    model=TaskConditionedTransformer(target_teacher,core,code,args.trajectory_start,args.trajectory_end).to(device)
    # Full MLP control is allowed to learn its neural core and task code.
    params=list(core.parameters())+[model.task_code]
    opt=torch.optim.AdamW(params,lr=args.core_fit_lr,weight_decay=1e-4);ce=nn.CrossEntropyLoss();it=iter(loader)
    for _ in range(args.transfer_adaptation_steps):
        try:x,y=next(it)
        except StopIteration:it=iter(loader);x,y=next(it)
        x=x.to(device);y=y.to(device);loss=ce(model(x),y);opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(params,1.0);opt.step()
    return evaluate(model,loader,device)


def run_one(args,seed,holdout):
    device=torch.device(args.device); meta_tasks=[t for t in args.all_tasks if t!=holdout]
    teachers_info=build_meta_teachers(meta_tasks,args,device,seed)
    teachers=[x[0] for x in teachers_info]; meta_loaders=[x[1] for x in teachers_info]; meta_verifiers=[x[2] for x in teachers_info]
    ref_routes=[collect_routing(t,va,device,args.verifier_batches) for t,va in zip(teachers,meta_verifiers)]
    rows=[]; current_teachers=teachers; final_core=None; final_models=None
    for r in range(args.surgery_rounds):
        cand=search(current_teachers,meta_loaders,ref_routes,args,device,seed+1000*r); eligible=[x for x in cand if x["eligible"]]; win=max(eligible,key=lambda x:x["score"])
        core,models,_=fit_joint_candidate(current_teachers,meta_loaders,win["name"],args,device,seed+8000*r)
        pre=[evaluate(m,dl,device)["accuracy"] for m,dl in zip(models,meta_loaders)]
        meta_adapt(models,core,meta_loaders,args,device)
        post=[evaluate(m,dl,device)["accuracy"] for m,dl in zip(models,meta_loaders)]
        routes=[routing_stats(ref_routes[i],collect_routing(models[i],meta_verifiers[i],device,args.verifier_batches)) for i in range(len(models))]
        rows.append({"round":r,"winner":win,"all_candidates":cand,"meta_pre_accuracy":pre,"meta_post_accuracy":post,"meta_routing_after":routes})
        final_core=core;final_models=models;current_teachers=models
        print(f"  round={r} winner={win['name']} meta_pre={statistics.mean(pre):.4f} meta_post={statistics.mean(post):.4f} shared_core={count_params(core)} task_code_dim={args.task_dim}",flush=True)
    # Holdout data + independent teacher.
    ttr, ttrain, ttest = None, None, None
    ttrain=DataLoader(TaskDataset(args.train_size,holdout,seed+50000),batch_size=args.batch_size,shuffle=True,pin_memory=device.type=="cuda")
    ttest=DataLoader(TaskDataset(args.test_size,holdout,seed+60000),batch_size=args.batch_size,shuffle=False,pin_memory=device.type=="cuda")
    holdout_teacher=TinyTransformer(len(VOCAB),args.d_model,args.heads,args.d_ff,args.depth).to(device); train_model(holdout_teacher,ttrain,device,args.teacher_steps,args.lr)
    held=evaluate(holdout_teacher,ttest,device)
    meta_codes=[m.task_code.detach().clone() for m in final_models]
    # Two zero-shot priors: centroid and all-zero. Neither changes the core.
    centroid_model,centroid_ev,frozen_core=frozen_transfer(holdout_teacher,final_core,meta_codes,args,device,ttest,"centroid")
    zero_model,zero_ev,_=frozen_transfer(holdout_teacher,final_core,meta_codes,args,device,ttest,"zero")
    # Adapt ONLY the tiny task code, never shared core or conditioner.
    centroid_model.task_code.requires_grad=True
    opt=torch.optim.AdamW([centroid_model.task_code],lr=args.code_lr,weight_decay=0.0);ce=nn.CrossEntropyLoss();it=iter(ttrain)
    for _ in range(args.transfer_adaptation_steps):
        try:x,y=next(it)
        except StopIteration:it=iter(ttrain);x,y=next(it)
        x=x.to(device);y=y.to(device);loss=ce(centroid_model(x),y)+args.code_weight*torch.mean(centroid_model.task_code.square());opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_([centroid_model.task_code],1.0);opt.step()
    adapted=evaluate(centroid_model,ttest,device)
    mlp=matched_mlp_control(holdout_teacher,args,device,ttrain)
    return {"holdout_task":holdout,"meta_tasks":meta_tasks,"rounds":rows,"meta_teacher_accuracies":[evaluate(t,dl,device)["accuracy"] for t,dl in zip(teachers,meta_loaders)],"heldout_teacher":held,"zero_shot_centroid":centroid_ev,"zero_shot_zero_code":zero_ev,"after_tiny_code_adaptation":adapted,"matched_mlp_control":mlp,"shared_core_params":count_params(final_core.base),"shared_conditioner_params":count_params(final_core.task_conditioner),"task_code_dim":args.task_dim,"task_code_params_total":len(meta_tasks)*args.task_dim,"zero_shot_gain_centroid_points":100*(centroid_ev["accuracy"]-held["accuracy"]),"zero_shot_gain_zero_code_points":100*(zero_ev["accuracy"]-held["accuracy"]),"tiny_code_adaptation_gain_points":100*(adapted["accuracy"]-held["accuracy"]),"vs_mlp_control_points":100*(adapted["accuracy"]-mlp["accuracy"])}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--all-tasks",nargs="+",default=["add","compose","mul","sub"])
    p.add_argument("--holdout-tasks",nargs="+",default=["sub"])
    p.add_argument("--seeds",nargs="+",type=int,default=[1,2])
    p.add_argument("--train-size",type=int,default=6000);p.add_argument("--verifier-size",type=int,default=1500);p.add_argument("--test-size",type=int,default=1500)
    p.add_argument("--teacher-steps",type=int,default=800);p.add_argument("--core-fit-steps",type=int,default=300);p.add_argument("--adaptation-steps-per-round",type=int,default=400);p.add_argument("--surgery-rounds",type=int,default=2);p.add_argument("--transfer-adaptation-steps",type=int,default=400)
    p.add_argument("--d-model",type=int,default=32);p.add_argument("--heads",type=int,default=2);p.add_argument("--d-ff",type=int,default=128);p.add_argument("--depth",type=int,default=3);p.add_argument("--rank",type=int,default=8);p.add_argument("--bottleneck",type=int,default=32);p.add_argument("--task-dim",type=int,default=4)
    p.add_argument("--trajectory-start",type=int,default=0);p.add_argument("--trajectory-end",type=int,default=3);p.add_argument("--verifier-batches",type=int,default=20)
    p.add_argument("--code-weight",type=float,default=0.05);p.add_argument("--routing-weight",type=float,default=0.20);p.add_argument("--ablation-weight",type=float,default=0.10);p.add_argument("--core-fit-lr",type=float,default=1e-3);p.add_argument("--code-lr",type=float,default=3e-3);p.add_argument("--lr",type=float,default=3e-4);p.add_argument("--complexity-lambda",type=float,default=1e-4);p.add_argument("--batch-size",type=int,default=256);p.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu");p.add_argument("--out",default="dart011_results.json")
    a=p.parse_args(); a.device_obj=torch.device(a.device)
    print("DART-1.1: tiny task-conditioned shared primitive + frozen transfer",flush=True)
    records=[]
    for holdout in a.holdout_tasks:
        print(f"\n===== HOLDOUT TASK {holdout} =====",flush=True)
        for seed in a.seeds:
            print(f"seed={seed} meta={','.join(t for t in a.all_tasks if t!=holdout)} -> holdout={holdout}",flush=True)
            records.append(run_one(a,seed,holdout))
    summary={"holdout_transfer":{}}
    for h in a.holdout_tasks:
        rs=[r for r in records if r["holdout_task"]==h]
        summary["holdout_transfer"][h]={
            "teacher":statistics.mean(r["heldout_teacher"]["accuracy"] for r in rs),
            "zero_shot_centroid":statistics.mean(r["zero_shot_centroid"]["accuracy"] for r in rs),
            "zero_shot_zero_code":statistics.mean(r["zero_shot_zero_code"]["accuracy"] for r in rs),
            "adapted":statistics.mean(r["after_tiny_code_adaptation"]["accuracy"] for r in rs),
            "mlp_control":statistics.mean(r["matched_mlp_control"]["accuracy"] for r in rs),
            "zero_shot_gain_centroid_points":statistics.mean(r["zero_shot_gain_centroid_points"] for r in rs),
            "zero_shot_gain_zero_code_points":statistics.mean(r["zero_shot_gain_zero_code_points"] for r in rs),
            "tiny_code_adaptation_gain_points":statistics.mean(r["tiny_code_adaptation_gain_points"] for r in rs),
            "vs_mlp_control_points":statistics.mean(r["vs_mlp_control_points"] for r in rs),
            "shared_core_params":statistics.mean(r["shared_core_params"] for r in rs),
            "shared_conditioner_params":statistics.mean(r["shared_conditioner_params"] for r in rs),
            "task_code_dim":rs[0]["task_code_dim"],
        }
    out={"config":{k:v for k,v in vars(a).items() if k!="device_obj"},"records":records,"summary":summary}
    Path(a.out).write_text(json.dumps(out,indent=2),encoding="utf-8")
    print("\n================ DART-1.1 SUMMARY ================",flush=True);print(summary);print(f"Saved: {Path(a.out).resolve()}")


if __name__=="__main__":main()
