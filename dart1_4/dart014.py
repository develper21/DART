#!/usr/bin/env python3
"""DART-1.4: factorized invariant primitive compilation.

Research hypothesis
-------------------
DART-1.3 removed learned task conditioning and showed that a standalone structured
primitive can distinguish a related holdout from an unrelated contrast task, but it
still did not transfer strongly to the related task. DART-1.4 tests a stricter idea:

    one shared computational mechanism C(x)
    + a tiny explicit operation parameter theta
    -> multiple task behaviors

The task parameter is NOT a learned embedding or conditioner network. It is a small,
interpretable coefficient vector used to combine a shared basis of structured
computations. The shared basis is learned jointly on meta-tasks; after discovery it is
frozen. On the holdout task, only theta is fit from a small adaptation set.

Core experiments
----------------
1. Discover a factorized primitive jointly across meta-tasks.
2. Measure whether source-task theta estimates are stable across bootstrap splits.
3. Freeze the shared basis completely.
4. Fit only a tiny explicit theta on the held-out related task.
5. Evaluate zero-shot (centroid theta), few-shot theta adaptation, and a matched MLP.
6. Run the same frozen primitive on an unrelated contrast task.

Success is not "the model is small". The strongest result is:
    same shared basis + different small theta -> useful related-task behavior,
    while unrelated-task behavior degrades.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import statistics
import time
from pathlib import Path

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
    if task == "add":
        return (ad[0] + bd[-1]) % 10
    if task == "sub":
        return (ad[-1] - bd[0]) % 10
    if task == "mul":
        return (ad[0] * bd[-1]) % 10
    if task == "sort":
        return min(ad + bd)
    if task == "compose":
        return ((ad[0] + bd[-1]) * (ad[1] + 1)) % 10
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

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


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
            n = b.norm1(h)
            a, w = b.attn(n, n, n, need_weights=capture_attention, average_attn_weights=False)
            h = h + a
            h = h + b.ff(b.norm2(h))
            if capture_attention:
                ats.append(w.detach().cpu())
        out = self.head(h[:, 0])
        return (out, ats) if capture_attention else out


# ---------------- Structured basis families ----------------
class ZeroCore(nn.Module):
    def forward(self, x):
        return torch.zeros_like(x)


class DiagonalCore(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(d))
        self.bias = nn.Parameter(torch.zeros(d))

    def forward(self, x):
        return x * self.scale + self.bias


class PolynomialCore(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.a = nn.Parameter(torch.zeros(d))
        self.b = nn.Parameter(torch.zeros(d))
        self.c = nn.Parameter(torch.zeros(d))

    def forward(self, x):
        return self.a * x + self.b * x.square() + self.c


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

    def forward(self, x):
        return self.up(self.down(x))


class MLPControl(nn.Module):
    def __init__(self, d, b):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, b), nn.GELU(), nn.Linear(b, d))

    def forward(self, x):
        return self.net(x)


def build_basis(name: str, d: int, rank: int, bottleneck: int):
    if name == "affine_polynomial":
        return AffinePolynomialCore(d, rank)
    if name == "low_rank":
        return LowRankCore(d, rank)
    if name == "polynomial":
        return PolynomialCore(d)
    if name == "diagonal":
        return DiagonalCore(d)
    if name == "identity":
        # Identity-like basis is represented as a learned diagonal family.
        return DiagonalCore(d)
    if name == "mlp":
        return MLPControl(d, bottleneck)
    raise ValueError(name)


class FactorizedPrimitive(nn.Module):
    """Shared basis B_k with tiny explicit theta mixing coefficients."""
    def __init__(self, name: str, d: int, rank: int, bottleneck: int, theta_dim: int):
        super().__init__()
        self.name = name
        self.theta_dim = theta_dim
        self.basis = nn.ModuleList([build_basis(name, d, rank, bottleneck) for _ in range(theta_dim)])

    def basis_outputs(self, x):
        return [b(x) for b in self.basis]

    def forward(self, x, theta: Tensor):
        ys = self.basis_outputs(x)
        out = torch.zeros_like(ys[0])
        for k, y in enumerate(ys):
            out = out + theta[:, k].view(-1, 1, 1) * y if theta.ndim == 2 and y.ndim == 3 else out + theta[k] * y
        return out


class RoutingFactorizedBlock(nn.Module):
    def __init__(self, original: Block, primitive: FactorizedPrimitive, theta: Tensor, train_theta=False):
        super().__init__()
        self.norm1 = copy.deepcopy(original.norm1)
        self.attn = copy.deepcopy(original.attn)
        self.norm2 = copy.deepcopy(original.norm2)
        self.primitive = primitive
        self.register_buffer("theta_fixed", theta.detach().clone())
        self.theta_param = nn.Parameter(theta.detach().clone()) if train_theta else None

    def active_theta(self, batch: int, device):
        theta = self.theta_param if self.theta_param is not None else self.theta_fixed
        return theta.to(device).view(1, -1).expand(batch, -1)

    def forward(self, x):
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        u = x + a
        z = self.norm2(u)
        theta = self.active_theta(z.shape[0], z.device)
        return u + self.primitive(z, theta)


class CompiledTransformer(nn.Module):
    def __init__(self, teacher: TinyTransformer, primitive: FactorizedPrimitive, theta: Tensor, start: int, end: int, train_theta=False):
        super().__init__()
        self.d_model = teacher.d_model
        self.depth = teacher.depth
        self.emb = copy.deepcopy(teacher.emb)
        self.pos = copy.deepcopy(teacher.pos)
        self.head = copy.deepcopy(teacher.head)
        self.blocks = nn.ModuleList()
        for i, b in enumerate(teacher.blocks):
            if start <= i < end:
                self.blocks.append(RoutingFactorizedBlock(b, primitive, theta, train_theta=train_theta))
            else:
                self.blocks.append(copy.deepcopy(b))

    def forward(self, x):
        h = self.emb(x) + self.pos[:, :x.size(1)]
        for b in self.blocks:
            h = b(h)
        return self.head(h[:, 0])



def freeze(module):
    for p in module.parameters():
        p.requires_grad = False

def count_params(m):
    return sum(p.numel() for p in m.parameters())


def basis_core_macs(name: str, d: int, rank: int):
    if name == "polynomial":
        return 2 * d
    if name in {"diagonal", "identity"}:
        return d
    if name == "affine_polynomial":
        return 3 * d * rank
    if name == "low_rank":
        return d * rank + rank * d
    if name == "mlp":
        return 2 * d * 32
    raise ValueError(name)


def evaluate(model, loader, device):
    model.eval(); ce = nn.CrossEntropyLoss(reduction="sum")
    loss_sum = correct = total = 0.0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            z = model(x); loss_sum += float(ce(z, y)); correct += int((z.argmax(-1) == y).sum()); total += y.numel()
    return {"accuracy": correct / max(total, 1), "loss": loss_sum / max(total, 1), "params": count_params(model)}


def train_model(model, loader, device, steps, lr, trainable_only=True):
    params = [p for p in model.parameters() if (p.requires_grad if trainable_only else True)]
    if not params:
        raise RuntimeError("No trainable parameters")
    model.train(); opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4); it = iter(loader); ce = nn.CrossEntropyLoss(); t0 = time.perf_counter()
    for _ in range(steps):
        try: x, y = next(it)
        except StopIteration: it = iter(loader); x, y = next(it)
        x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True)
        loss=ce(model(x),y); opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(params,1.0); opt.step()
    if device.type=="cuda": torch.cuda.synchronize()
    return time.perf_counter()-t0


def make_teacher(task,args,device,seed):
    tr=DataLoader(TaskDataset(args.train_size,task,seed),batch_size=args.batch_size,shuffle=True,pin_memory=device.type=="cuda")
    va=DataLoader(TaskDataset(args.verifier_size,task,seed+10000),batch_size=args.batch_size,shuffle=False,pin_memory=device.type=="cuda")
    te=TinyTransformer(len(VOCAB),args.d_model,args.heads,args.d_ff,args.depth).to(device)
    train_model(te,tr,device,args.teacher_steps,args.lr); return te,tr,va


def capture_ff_states(teacher, loader, device, max_samples, layer):
    zs=[]; ys=[]; total=0; teacher.eval()
    with torch.no_grad():
        for x,_ in loader:
            if total>=max_samples: break
            x=x.to(device,non_blocking=True); h=teacher.emb(x)+teacher.pos[:,:x.size(1)]
            for i,b in enumerate(teacher.blocks):
                n=b.norm1(h); a,_=b.attn(n,n,n,need_weights=False); u=h+a; z=b.norm2(u); y=b.ff(z)
                if i==layer:
                    zz=z.reshape(-1,z.shape[-1]).cpu(); yy=y.reshape(-1,y.shape[-1]).cpu(); take=min(max_samples-total,zz.shape[0]); zs.append(zz[:take]);ys.append(yy[:take]);total+=take;break
                h=u+y
    return torch.cat(zs),torch.cat(ys)


def relational_bundle(teacher, loader, args, device, seed):
    z,y=capture_ff_states(teacher,loader,device,args.rel_samples_per_task,args.trajectory_start)
    g=torch.Generator(device="cpu");g.manual_seed(seed);use=min(len(z),args.rel_samples_per_task);idx=torch.randperm(len(z),generator=g)[:use];z=z[idx];y=y[idx]
    dirs=[]
    for j in range(args.rel_directions):
        d=torch.randn(use,z.shape[-1],generator=g);d=d/(d.norm(dim=-1,keepdim=True)+1e-8);dirs.append(d)
    eps=args.intervention_eps; return z,y,dirs,eps


def teacher_ff_eval(teacher,z_cpu,layer,device):
    z=z_cpu.to(device);return teacher.blocks[layer].ff(z).detach()


def factorized_output(primitive,z,theta):
    # z: [N,D], theta: [K]
    outs=[b(z) for b in primitive.basis]
    out=torch.zeros_like(outs[0])
    for k,y in enumerate(outs): out=out+theta[k]*y
    return out


def fit_theta(primitive, z, y, directions, teacher, args, device, init=None, steps=None):
    theta=nn.Parameter((torch.zeros(args.theta_dim,device=device) if init is None else init.to(device).clone()))
    opt=torch.optim.Adam([theta],lr=args.theta_lr)
    ce=None
    steps=steps or args.theta_fit_steps
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        pred=factorized_output(primitive,z.to(device),theta)
        loss=nn.functional.mse_loss(pred,y.to(device))
        # relational directional matching
        for d in directions:
            d=d.to(device)
            cdir=(factorized_output(primitive,z.to(device)+args.intervention_eps*d,theta)-factorized_output(primitive,z.to(device)-args.intervention_eps*d,theta))/(2*args.intervention_eps)
            # teacher response calculated once by caller is attached through cache in bundle if available
        loss.backward(); opt.step()
    return theta.detach()


def fit_theta_with_bundle(primitive,bundle,teacher,args,device,steps=None,init=None):
    z,y,dirs,td=bundle
    theta=nn.Parameter(torch.zeros(args.theta_dim,device=device) if init is None else init.to(device).clone())
    opt=torch.optim.Adam([theta],lr=args.theta_lr)
    steps=steps or args.theta_fit_steps
    tz=z.to(device)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        pred=factorized_output(primitive,tz,theta);loss=nn.functional.mse_loss(pred,y.to(device))
        for d,tresp in zip(dirs,td):
            dd=d.to(device);cres=(factorized_output(primitive,tz+args.intervention_eps*dd,theta)-factorized_output(primitive,tz-args.intervention_eps*dd,theta))/(2*args.intervention_eps)
            loss=loss+args.directional_weight*nn.functional.mse_loss(cres,tresp.to(device))
        loss.backward();opt.step()
    return theta.detach()


def build_directional_teacher_bundle(teacher, loader, args, device, seed):
    z,y,dirs,eps=relational_bundle(teacher,loader,args,device,seed)
    td=[]
    for d in dirs:
        td.append(((teacher_ff_eval(teacher,z+eps*d, args.trajectory_start, device)-teacher_ff_eval(teacher,z-eps*d,args.trajectory_start,device))/(2*eps)).cpu())
    return z,y,dirs,td


def primitive_metrics(primitive,bundle,args,device,theta):
    z,y,dirs,td=bundle;zdev=z.to(device);pred=factorized_output(primitive,zdev,theta); value=float(nn.functional.mse_loss(pred,y.to(device)).detach())
    derr=[]
    for d,t in zip(dirs,td):
        dd=d.to(device);c=(factorized_output(primitive,zdev+args.intervention_eps*dd,theta)-factorized_output(primitive,zdev-args.intervention_eps*dd,theta))/(2*args.intervention_eps)
        derr.append(float(nn.functional.mse_loss(c,t.to(device)).detach()))
    scale=float(y.float().std().item()+1e-6);rel=math.exp(-(value+statistics.mean(derr))/(scale*scale))
    return {"value_mse":value,"directional_mse":statistics.mean(derr),"relational_agreement":rel}


def fit_shared_candidate(name, teachers, meta_loaders, bundles, args, device, seed):
    seed_everything(seed); primitive=FactorizedPrimitive(name,args.d_model,args.rank,args.bottleneck,args.theta_dim).to(device)
    thetas=nn.Parameter(torch.zeros(len(teachers),args.theta_dim,device=device))
    params=list(primitive.parameters())+[thetas];opt=torch.optim.AdamW(params,lr=args.core_fit_lr,weight_decay=1e-4);t0=time.perf_counter()
    for _ in range(args.core_fit_steps):
        opt.zero_grad(set_to_none=True);total=0.0
        for i,bundle in enumerate(bundles):
            z,y,dirs,td=bundle;z=z.to(device);theta=thetas[i]
            pred=factorized_output(primitive,z,theta);loss=nn.functional.mse_loss(pred,y.to(device))
            for d,tresp in zip(dirs,td):
                dd=d.to(device);cres=(factorized_output(primitive,z+args.intervention_eps*dd,theta)-factorized_output(primitive,z-args.intervention_eps*dd,theta))/(2*args.intervention_eps)
                loss=loss+args.directional_weight*nn.functional.mse_loss(cres,tresp.to(device))
            total=total+loss
        # Encourage small, identifiable task parameters.
        total=total/len(bundles)+args.theta_l2*thetas.square().mean()
        total.backward();nn.utils.clip_grad_norm_(params,1.0);opt.step()
    if device.type=="cuda": torch.cuda.synchronize()
    secs=time.perf_counter()-t0
    return primitive,thetas.detach(),secs


def fit_meta_source_thetas(primitive,bundles,args,device):
    vals=[]
    for b in bundles: vals.append(fit_theta_with_bundle(primitive,b,None,args,device))
    return torch.stack(vals)


def candidate_score(primitive,thetas,teachers,meta_loaders,bundles,args,device):
    acc=[]; rels=[];caus=[]
    models=[]
    for t,dl,theta in zip(teachers,meta_loaders,thetas):
        m=CompiledTransformer(t,primitive,theta,args.trajectory_start,args.trajectory_end,train_theta=False).to(device);models.append(m)
        acc.append(evaluate(m,dl,device)["accuracy"])
    for b,theta in zip(bundles,thetas):
        pm=primitive_metrics(primitive,b,args,device,theta);rels.append(pm["relational_agreement"]);caus.append(math.exp(-pm["directional_mse"]/(float(b[1].float().std().item())**2+1e-6)))
    score=statistics.mean(acc)+args.relational_weight*statistics.mean(rels)+args.causal_weight*statistics.mean(caus)-args.complexity_lambda*math.log1p(count_params(primitive))
    return {"avg_accuracy":statistics.mean(acc),"avg_relational_agreement":statistics.mean(rels),"avg_causal_agreement":statistics.mean(caus),"shared_core_params":count_params(primitive),"shared_core_macs":basis_core_macs(primitive.name,args.d_model,args.rank)*args.theta_dim,"theta_dim":args.theta_dim,"score":score,"task_accuracies":acc}


def source_theta_stability(primitive,bundles,args,device):
    # Fit theta on two disjoint halves of each source bundle; low distance = identifiable operation parameter.
    dists=[]
    for bundle in bundles:
        z,y,dirs,td=bundle;n=len(z);mid=n//2
        b1=(z[:mid],y[:mid],[d[:mid] for d in dirs],[t[:mid] for t in td]);b2=(z[mid:],y[mid:],[d[mid:] for d in dirs],[t[mid:] for t in td])
        t1=fit_theta_with_bundle(primitive,b1,None,args,device);t2=fit_theta_with_bundle(primitive,b2,None,args,device)
        dists.append(float(torch.norm(t1-t2).item()))
    return statistics.mean(dists) if dists else 0.0


def train_teacher(task,args,device,seed):
    tr=DataLoader(TaskDataset(args.train_size,task,seed),batch_size=args.batch_size,shuffle=True,pin_memory=device.type=="cuda")
    va=DataLoader(TaskDataset(args.verifier_size,task,seed+10000),batch_size=args.batch_size,shuffle=False,pin_memory=device.type=="cuda")
    te=TinyTransformer(len(VOCAB),args.d_model,args.heads,args.d_ff,args.depth).to(device);train_model(te,tr,device,args.teacher_steps,args.lr)
    return te,tr,va


def target_adapt_primitive(primitive, teacher, loader, args, device, init_theta=None):
    bundle=build_directional_teacher_bundle(teacher,loader,args,device,args.target_theta_seed)
    theta=fit_theta_with_bundle(primitive,bundle,None,args,device,steps=args.target_theta_fit_steps,init=init_theta)
    return theta


def run_one(args,seed,holdout,contrast):
    device=torch.device(args.device);meta_tasks=[t for t in args.all_tasks if t not in {holdout,contrast}]
    teachers=[];trainers=[];verifiers=[];bundles=[]
    for i,t in enumerate(meta_tasks):
        te,tr,va=train_teacher(t,args,device,seed+i*1000);teachers.append(te);trainers.append(tr);verifiers.append(va)
        bundles.append(build_directional_teacher_bundle(te,va,args,device,seed+97*i))
    final=None;rows=[]
    for r in range(args.surgery_rounds):
        candidates=[]
        for i,name in enumerate(args.structured_families):
            primitive,thetas,secs=fit_shared_candidate(name,teachers,trainers,bundles,args,device,seed+100*r+37*i)
            metrics=candidate_score(primitive,thetas,teachers,verifiers,bundles,args,device)
            stability=source_theta_stability(primitive,bundles,args,device)
            # Prefer high accuracy + invariance and penalize unstable task parameters.
            metrics["theta_stability"] = stability
            metrics["score"] -= args.theta_stability_weight*stability
            metrics.update(name=name,kind="dart_factorized",eligible=True,fit_seconds=secs)
            candidates.append((metrics,primitive,thetas))
        row,primitive,thetas=max(candidates,key=lambda x:x[0]["score"])
        # Source meta adaptation: refine only source theta, while primitive remains shared.
        refined=[]
        for i,b in enumerate(bundles): refined.append(fit_theta_with_bundle(primitive,b,None,args,device,steps=args.meta_theta_adapt_steps,init=thetas[i]))
        refined=torch.stack(refined)
        pre=[];post=[]
        for t,dl,th0,th1 in zip(teachers,trainers,thetas,refined):
            pre.append(evaluate(CompiledTransformer(t,primitive,th0,args.trajectory_start,args.trajectory_end).to(device),dl,device)["accuracy"])
            post.append(evaluate(CompiledTransformer(t,primitive,th1,args.trajectory_start,args.trajectory_end).to(device),dl,device)["accuracy"])
        rows.append({"round":r,"winner":row,"meta_pre_accuracy":pre,"meta_post_accuracy":post,"source_theta_stability":row["theta_stability"]})
        final=(primitive,refined)
        print(f"  round={r} winner={row['name']} meta_pre={statistics.mean(pre):.4f} meta_post={statistics.mean(post):.4f} theta_stability={row['theta_stability']:.4f} core={count_params(primitive)} theta_dim={args.theta_dim}",flush=True)
    primitive,source_thetas=final
    freeze(primitive)

    def evaluate_holdout(task, offset):
        teacher,tr,te_loader=train_teacher(task,args,device,seed+offset)
        teacher_ev=evaluate(teacher,te_loader,device)
        theta_centroid=source_thetas.mean(0)
        dart_zero=evaluate(CompiledTransformer(teacher,primitive,theta_centroid,args.trajectory_start,args.trajectory_end).to(device),te_loader,device)
        theta_adapt=target_adapt_primitive(primitive,teacher,tr,args,device,init_theta=theta_centroid)
        dart_adapt=evaluate(CompiledTransformer(teacher,primitive,theta_adapt,args.trajectory_start,args.trajectory_end).to(device),te_loader,device)
        mlp=MLPControl(args.d_model,args.bottleneck).to(device);mcomp=RoutingFactorizedMLP(teacher,mlp,args.trajectory_start,args.trajectory_end).to(device)
        for p in mcomp.parameters(): p.requires_grad = False
        for p in mcomp.mlp.parameters(): p.requires_grad = True
        train_model(mcomp,tr,device,args.transfer_control_steps,args.lr)
        mlp_ev=evaluate(mcomp,te_loader,device)
        return {"task":task,"teacher":teacher_ev,"dart_zero_shot":dart_zero,"dart_theta_adapted":dart_adapt,"mlp_control":mlp_ev,
                "zero_shot_gain_points":100*(dart_zero["accuracy"]-teacher_ev["accuracy"]),"theta_adaptation_gain_points":100*(dart_adapt["accuracy"]-teacher_ev["accuracy"]),
                "vs_mlp_zero_points":100*(dart_zero["accuracy"]-mlp_ev["accuracy"]),"vs_mlp_adapt_points":100*(dart_adapt["accuracy"]-mlp_ev["accuracy"]),"theta":theta_adapt.cpu().tolist()}
    related=evaluate_holdout(holdout,50000)
    contrast=evaluate_holdout(contrast,70000) if contrast else None
    return {"holdout_task":holdout,"contrast_task":contrast,"meta_tasks":meta_tasks,"rounds":rows,"related_holdout":related,"contrast_holdout":contrast,
            "shared_core_params":count_params(primitive),"shared_core_macs":basis_core_macs(primitive.name,args.d_model,args.rank)*args.theta_dim,"conditioner_params":0,"task_code_params":args.theta_dim}


class MLPReplacementBlock(nn.Module):
    def __init__(self, original, mlp):
        super().__init__()
        self.norm1=copy.deepcopy(original.norm1)
        self.attn=copy.deepcopy(original.attn)
        self.norm2=copy.deepcopy(original.norm2)
        self.mlp=mlp
    def forward(self,x):
        n=self.norm1(x);a,_=self.attn(n,n,n,need_weights=False);u=x+a
        return u+self.mlp(self.norm2(u))

class RoutingFactorizedMLP(nn.Module):
    def __init__(self,teacher,mlp,start,end):
        super().__init__();self.emb=copy.deepcopy(teacher.emb);self.pos=copy.deepcopy(teacher.pos);self.head=copy.deepcopy(teacher.head);self.blocks=nn.ModuleList()
        self.mlp=mlp;self.start=start;self.end=end
        for i,b in enumerate(teacher.blocks):
            if start<=i<end:self.blocks.append(MLPReplacementBlock(b,mlp))
            else:self.blocks.append(copy.deepcopy(b))
    def forward(self,x):
        h=self.emb(x)+self.pos[:,:x.size(1)]
        for b in self.blocks:h=b(h)
        return self.head(h[:,0])


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--all-tasks',nargs='+',default=['add','compose','mul','sub']);p.add_argument('--holdout-tasks',nargs='+',default=['sub']);p.add_argument('--contrast-tasks',nargs='+',default=['sort']);p.add_argument('--seeds',nargs='+',type=int,default=[1,2])
    p.add_argument('--train-size',type=int,default=6000);p.add_argument('--verifier-size',type=int,default=1500);p.add_argument('--test-size',type=int,default=1500)
    p.add_argument('--teacher-steps',type=int,default=800);p.add_argument('--core-fit-steps',type=int,default=300);p.add_argument('--theta-fit-steps',type=int,default=100);p.add_argument('--meta-theta-adapt-steps',type=int,default=200);p.add_argument('--target-theta-fit-steps',type=int,default=200);p.add_argument('--surgery-rounds',type=int,default=2);p.add_argument('--transfer-control-steps',type=int,default=400)
    p.add_argument('--d-model',type=int,default=32);p.add_argument('--heads',type=int,default=2);p.add_argument('--d-ff',type=int,default=128);p.add_argument('--depth',type=int,default=3);p.add_argument('--rank',type=int,default=8);p.add_argument('--bottleneck',type=int,default=32)
    p.add_argument('--theta-dim',type=int,default=4);p.add_argument('--rel-samples-per-task',type=int,default=2048);p.add_argument('--rel-directions',type=int,default=4);p.add_argument('--intervention-eps',type=float,default=.05)
    p.add_argument('--relational-weight',type=float,default=.5);p.add_argument('--causal-weight',type=float,default=.5);p.add_argument('--directional-weight',type=float,default=.5);p.add_argument('--theta-l2',type=float,default=1e-3);p.add_argument('--theta-stability-weight',type=float,default=.25);p.add_argument('--theta-lr',type=float,default=1e-2);p.add_argument('--core-fit-lr',type=float,default=1e-3);p.add_argument('--lr',type=float,default=3e-4);p.add_argument('--complexity-lambda',type=float,default=1e-4);p.add_argument('--batch-size',type=int,default=256);p.add_argument('--trajectory-start',type=int,default=0);p.add_argument('--trajectory-end',type=int,default=3);p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu');p.add_argument('--structured-families',nargs='+',default=['affine_polynomial','low_rank','polynomial','diagonal']);p.add_argument('--target-theta-seed',type=int,default=1234);p.add_argument('--out',default='dart014_results.json')
    a=p.parse_args();print('DART-1.4: factorized invariant primitive compilation + explicit theta transfer',flush=True);records=[]
    for h in a.holdout_tasks:
        c=a.contrast_tasks[0] if a.contrast_tasks else None;print(f'\n===== RELATED HOLDOUT {h} | CONTRAST {c} =====',flush=True)
        for s in a.seeds: print(f'seed={s}',flush=True);records.append(run_one(a,s,h,c))
    summary={"related_holdout":{},"contrast_holdout":{}}
    for task in a.holdout_tasks:
        rs=[r for r in records if r['holdout_task']==task]
        summary['related_holdout'][task]={
            'teacher':statistics.mean(r['related_holdout']['teacher']['accuracy'] for r in rs),
            'dart_zero_shot':statistics.mean(r['related_holdout']['dart_zero_shot']['accuracy'] for r in rs),
            'dart_theta_adapted':statistics.mean(r['related_holdout']['dart_theta_adapted']['accuracy'] for r in rs),
            'mlp_control':statistics.mean(r['related_holdout']['mlp_control']['accuracy'] for r in rs),
            'zero_shot_gain_points':statistics.mean(r['related_holdout']['zero_shot_gain_points'] for r in rs),
            'theta_adaptation_gain_points':statistics.mean(r['related_holdout']['theta_adaptation_gain_points'] for r in rs),
            'vs_mlp_zero_points':statistics.mean(r['related_holdout']['vs_mlp_zero_points'] for r in rs),
            'vs_mlp_adapt_points':statistics.mean(r['related_holdout']['vs_mlp_adapt_points'] for r in rs),
            'shared_core_params':statistics.mean(r['shared_core_params'] for r in rs),
            'shared_core_macs':statistics.mean(r['shared_core_macs'] for r in rs),
            'theta_dim':a.theta_dim
        }
        cs=[r['contrast_holdout'] for r in rs if r.get('contrast_holdout')]
        if cs:
            ct=cs[0]['task'];summary['contrast_holdout'][ct]={
                'teacher':statistics.mean(x['teacher']['accuracy'] for x in cs),'dart_zero_shot':statistics.mean(x['dart_zero_shot']['accuracy'] for x in cs),'dart_theta_adapted':statistics.mean(x['dart_theta_adapted']['accuracy'] for x in cs),'mlp_control':statistics.mean(x['mlp_control']['accuracy'] for x in cs),
                'zero_shot_gain_points':statistics.mean(x['zero_shot_gain_points'] for x in cs),'theta_adaptation_gain_points':statistics.mean(x['theta_adaptation_gain_points'] for x in cs),'vs_mlp_zero_points':statistics.mean(x['vs_mlp_zero_points'] for x in cs),'vs_mlp_adapt_points':statistics.mean(x['vs_mlp_adapt_points'] for x in cs)
            }
    out={'config':vars(a),'records':records,'summary':summary};Path(a.out).write_text(json.dumps(out,indent=2),encoding='utf-8');print('\n================ DART-1.4 SUMMARY ================');print(summary);print(f'Saved: {Path(a.out).resolve()}')

if __name__=='__main__': main()
