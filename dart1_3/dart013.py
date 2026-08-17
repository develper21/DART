#!/usr/bin/env python3
"""DART-1.3: causal/relational primitive compilation.

Research hypothesis
-------------------
DART-1.2 showed that deterministic behavioral signatures narrow the holdout gap,
but a learned conditioner did not improve zero-shot transfer. DART-1.3 therefore
removes task-conditioned learned machinery from the compiled primitive.

The shared primitive is fit from causal/relational invariants observed in teacher
computations across meta-tasks:
  1. value response:        f(z)
  2. directional response:  f(z+eps*d)-f(z)
  3. interaction response:  f(z+eps*(d1+d2))-f(z+eps*d1)-f(z+eps*d2)+f(z)

Candidate ranking combines downstream capability, local relational fidelity,
causal intervention agreement, and complexity. The winning structured primitive
is then frozen and evaluated zero-shot on related and contrast holdout tasks.
No task ID, learned task code, large conditioner, or target residual network is
used by the DART primitive.
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

    def forward(self, x, capture_states=False, capture_attention=False):
        h = self.emb(x) + self.pos[:, :x.size(1)]
        states, ats = [], []
        for b in self.blocks:
            n = b.norm1(h)
            a, w = b.attn(n, n, n, need_weights=capture_attention, average_attn_weights=False)
            u = h + a
            z = b.norm2(u)
            if capture_states:
                states.append(z.detach().cpu())
            h = u + b.ff(z)
            if capture_attention:
                ats.append(w.detach().cpu())
        out = self.head(h[:, 0])
        if capture_states or capture_attention:
            return out, states, ats
        return out


# ---------- Structured primitive families ----------
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


class RoutingReplacementBlock(nn.Module):
    """Original routing + frozen teacher-independent structured computation."""
    def __init__(self, original: Block, core: nn.Module):
        super().__init__()
        self.norm1 = copy.deepcopy(original.norm1)
        self.attn = copy.deepcopy(original.attn)
        self.norm2 = copy.deepcopy(original.norm2)
        self.core = core

    def forward(self, x):
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        u = x + a
        z = self.norm2(u)
        return u + self.core(z)

    def forward_capture(self, x):
        h = self.norm1(x)
        a, w = self.attn(h, h, h, need_weights=True, average_attn_weights=False)
        u = x + a
        z = self.norm2(u)
        return u + self.core(z), w, z


class CompiledTransformer(nn.Module):
    def __init__(self, teacher: TinyTransformer, shared_core: nn.Module, start: int, end: int):
        super().__init__()
        self.d_model = teacher.d_model
        self.depth = teacher.depth
        self.emb = copy.deepcopy(teacher.emb)
        self.pos = copy.deepcopy(teacher.pos)
        self.head = copy.deepcopy(teacher.head)
        self.blocks = nn.ModuleList()
        for i, b in enumerate(teacher.blocks):
            if start <= i < end:
                self.blocks.append(RoutingReplacementBlock(b, shared_core))
            else:
                self.blocks.append(copy.deepcopy(b))

    def forward(self, x, capture_attention=False):
        h = self.emb(x) + self.pos[:, :x.size(1)]
        ats = []
        for b in self.blocks:
            if isinstance(b, RoutingReplacementBlock):
                if capture_attention:
                    h, w, _ = b.forward_capture(h)
                    ats.append(w)
                else:
                    h = b(h)
            elif capture_attention:
                n = b.norm1(h)
                a, w = b.attn(n, n, n, need_weights=True, average_attn_weights=False)
                h = h + a
                h = h + b.ff(b.norm2(h))
                ats.append(w)
            else:
                h = b(h)
        return (self.head(h[:, 0]), ats) if capture_attention else self.head(h[:, 0])


# ---------- Metrics / data ----------
def count_params(m):
    return sum(p.numel() for p in m.parameters())


def core_macs(m):
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


def evaluate(model, loader, device):
    model.eval()
    ce = nn.CrossEntropyLoss(reduction="sum")
    total = correct = 0
    loss_sum = 0.0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            z = model(x)
            loss_sum += float(ce(z, y))
            correct += int((z.argmax(-1) == y).sum())
            total += y.numel()
    return {"accuracy": correct / max(total, 1), "loss": loss_sum / max(total, 1), "params": count_params(model)}


def train_model(model, loader, device, steps, lr):
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("No trainable parameters")
    model.train()
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    it = iter(loader)
    ce = nn.CrossEntropyLoss()
    t0 = time.perf_counter()
    for _ in range(steps):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(loader)
            x, y = next(it)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        loss = ce(model(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter() - t0


def freeze(module):
    for p in module.parameters():
        p.requires_grad = False


def collect_routing(model, loader, device, max_batches):
    out = []
    model.eval()
    with torch.no_grad():
        for bi, (x, _y) in enumerate(loader):
            if bi >= max_batches:
                break
            result = model(x.to(device, non_blocking=True), capture_attention=True)
            ats = result[-1]
            if not out:
                out = [[] for _ in ats]
            for i, a in enumerate(ats):
                out[i].append(a.cpu())
    return [torch.cat(v, 0) if v else torch.empty(0) for v in out]


def routing_agreement(reference, candidate, max_rows=1024):
    if not reference or not candidate:
        return 0.0
    vals = []
    for a, b in zip(reference, candidate):
        if a.numel() == 0 or b.numel() == 0:
            continue
        n = min(max_rows, a.shape[0], b.shape[0])
        ra, rb = a[:n].float(), b[:n].float()
        rel = torch.mean((ra - rb) ** 2) / (torch.mean(ra ** 2) + 1e-8)
        vals.append(float(rel))
    return math.exp(-statistics.mean(vals)) if vals else 0.0


def attention_ablation_accuracy(model, loader, device):
    model.eval()
    handles = []
    def zero_attn(_m, _inp, out):
        if isinstance(out, tuple):
            return (torch.zeros_like(out[0]),) + out[1:]
        return torch.zeros_like(out)
    for b in model.blocks:
        if isinstance(b, RoutingReplacementBlock):
            handles.append(b.attn.register_forward_hook(zero_attn))
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            z = model(x.to(device, non_blocking=True))
            yy = y.to(device, non_blocking=True)
            correct += int((z.argmax(-1) == yy).sum())
            total += yy.numel()
    for h in handles:
        h.remove()
    return correct / max(total, 1)


def make_teacher(task, args, device, seed):
    tr = DataLoader(TaskDataset(args.train_size, task, seed), batch_size=args.batch_size, shuffle=True,
                    pin_memory=device.type == "cuda")
    va = DataLoader(TaskDataset(args.verifier_size, task, seed + 10000), batch_size=args.batch_size, shuffle=False,
                    pin_memory=device.type == "cuda")
    te = TinyTransformer(len(VOCAB), args.d_model, args.heads, args.d_ff, args.depth).to(device)
    train_model(te, tr, device, args.teacher_steps, args.lr)
    return te, tr, va


# ---------- Relational invariant extraction ----------
def collect_relational_states(teacher, loader, device, max_batches, max_tokens=4096, target_layer=0):
    """Capture exact teacher FF-input states and FF outputs for one target block."""
    teacher.eval()
    zs, ys = [], []
    total = 0
    with torch.no_grad():
        for bi, (x, _y) in enumerate(loader):
            if bi >= max_batches or total >= max_tokens:
                break
            x = x.to(device, non_blocking=True)
            h = teacher.emb(x) + teacher.pos[:, :x.size(1)]
            for i, block in enumerate(teacher.blocks):
                n = block.norm1(h)
                a, _ = block.attn(n, n, n, need_weights=False)
                u = h + a
                z = block.norm2(u)
                y = block.ff(z)
                if i == target_layer:
                    zz = z.reshape(-1, z.shape[-1]).cpu()
                    yy = y.reshape(-1, y.shape[-1]).cpu()
                    take = min(max_tokens - total, zz.shape[0])
                    zs.append(zz[:take]); ys.append(yy[:take]); total += take
                    break
                h = u + y
    if not zs:
        raise RuntimeError("No relational states collected")
    return torch.cat(zs, 0), torch.cat(ys, 0)

def relational_probe(z: Tensor, teacher_y: Tensor, eps: float, directions: int, seed: int):
    """Create invariant targets for value, directional and interaction responses."""
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    n, d = z.shape
    use = min(n, 2048)
    idx = torch.randperm(n, generator=g)[:use]
    z = z[idx]
    teacher_y = teacher_y[idx]
    dirs = []
    for _ in range(directions):
        dd = torch.randn(use, d, generator=g)
        dd = dd / (dd.norm(dim=-1, keepdim=True) + 1e-8)
        dirs.append(dd)
    return z, teacher_y, dirs, eps


def relational_fit_loss(core, z, y, dirs, eps):
    pred = core(z)
    value = nn.functional.mse_loss(pred, y)
    directional = torch.tensor(0.0, device=z.device)
    interaction = torch.tensor(0.0, device=z.device)
    for d in dirs:
        d = d.to(z.device)
        tp = core(z + eps * d)
        tm = core(z - eps * d)
        directional = directional + nn.functional.mse_loss((tp - tm) / (2 * eps), torch.zeros_like(tp))
    # The directional term above is intentionally replaced below by teacher directional responses.
    return value, directional, interaction


def prepare_relational_bundle(teachers, loaders, args, device, seed):
    bundles = []
    for i, (teacher, loader) in enumerate(zip(teachers, loaders)):
        z, y = collect_relational_states(teacher, loader, device, args.verifier_batches, args.rel_samples_per_task, args.trajectory_start)
        z, y, dirs, eps = relational_probe(z, y, args.intervention_eps, args.rel_directions, seed + 97 * i)
        # Teacher responses for each directional / interaction probe are computed once.
        td, ti = [], []
        with torch.no_grad():
            for d in dirs:
                td.append(((teacher_ff_eval(teacher, z + eps * d, layer=args.trajectory_start) -
                            teacher_ff_eval(teacher, z - eps * d, layer=args.trajectory_start)) / (2 * eps)).cpu())
            if len(dirs) >= 2:
                d1, d2 = dirs[0], dirs[1]
                f00 = teacher_ff_eval(teacher, z, layer=args.trajectory_start)
                f10 = teacher_ff_eval(teacher, z + eps * d1, layer=args.trajectory_start)
                f01 = teacher_ff_eval(teacher, z + eps * d2, layer=args.trajectory_start)
                f11 = teacher_ff_eval(teacher, z + eps * (d1 + d2), layer=args.trajectory_start)
                ti.append((f11 - f10 - f01 + f00) / (eps * eps))
        bundles.append((z, y, dirs, td, ti))
    return bundles


def teacher_ff_eval(teacher, z_cpu: Tensor, layer: int):
    """Evaluate a teacher FFN on supplied normalized block states."""
    z = z_cpu.to(next(teacher.parameters()).device)
    return teacher.blocks[layer].ff(z).detach()


def compiled_ff_eval(core, z_cpu: Tensor, device):
    return core(z_cpu.to(device))


def relational_metrics(core, bundle, device, eps):
    z, y, dirs, td, ti = bundle
    zdev = z.to(device)
    with torch.no_grad():
        pv = core(zdev)
        value_mse = float(nn.functional.mse_loss(pv, y.to(device)))
        directional_errs = []
        for d, tdir in zip(dirs, td):
            ddev = d.to(device)
            cdir = (core(zdev + eps * ddev) - core(zdev - eps * ddev)) / (2 * eps)
            directional_errs.append(float(nn.functional.mse_loss(cdir, tdir.to(device))))
        interaction_err = 0.0
        if ti:
            d1, d2 = dirs[0].to(device), dirs[1].to(device)
            f00 = core(zdev); f10 = core(zdev + eps*d1); f01 = core(zdev + eps*d2); f11 = core(zdev + eps*(d1+d2))
            ci = (f11-f10-f01+f00)/(eps*eps)
            interaction_err = float(nn.functional.mse_loss(ci, ti[0].to(device)))
    scale = float(torch.mean(y.to(device) ** 2).sqrt().item() + 1e-6)
    rel = math.exp(-(value_mse + statistics.mean(directional_errs) + interaction_err) / (scale * scale))
    return {"value_mse": value_mse, "directional_mse": statistics.mean(directional_errs), "interaction_mse": interaction_err, "relational_agreement": rel}


def causal_intervention_agreement(core, bundle, device, eps):
    z, _y, dirs, td, _ti = bundle
    if not dirs:
        return 0.0
    errs = []
    zdev = z.to(device)
    with torch.no_grad():
        for d, tdir in zip(dirs, td):
            ddev = d.to(device)
            cdir = (core(zdev + eps*ddev) - core(zdev - eps*ddev))/(2*eps)
            num = torch.mean((cdir - tdir.to(device))**2)
            den = torch.mean(tdir.to(device)**2) + 1e-8
            errs.append(float(num/den))
    return math.exp(-statistics.mean(errs)) if errs else 0.0


def task_signature_from_behavior(task: str, n=64):
    # Reporting-only invariant signature: deterministic probes, not learned parameters.
    rng = random.Random(12345)
    ys=[]
    for _ in range(n):
        a,b=rng.randint(0,999),rng.randint(0,999)
        ys.append(task_target(a,b,task))
    t=torch.tensor(ys,dtype=torch.float32)
    return torch.cat([torch.tensor([t.mean()/9,t.std(unbiased=False)/9]), torch.bincount(t.long(),minlength=10).float()/n])


# ---------- Candidate compilation ----------
def make_compiled(teacher, core, args, device):
    return CompiledTransformer(teacher, core, args.trajectory_start, args.trajectory_end).to(device)


def fit_candidate(teachers, meta_loaders, bundles, name, args, device, seed):
    seed_everything(seed)
    core = build_core(name, args.d_model, args.rank, args.bottleneck).to(device)
    # Joint optimizer sees only relational objectives. Task labels enter only through downstream verifier scoring.
    params = list(core.parameters())
    t0=time.perf_counter()
    if not params:
        return core, time.perf_counter() - t0
    opt = torch.optim.AdamW(params, lr=args.core_fit_lr, weight_decay=1e-4)
    for _ in range(args.core_fit_steps):
        opt.zero_grad(set_to_none=True)
        losses=[]
        for bundle in bundles:
            z,y,dirs,td,ti=bundle
            zdev=z.to(device); ydev=y.to(device)
            pred=core(zdev)
            loss=nn.functional.mse_loss(pred,ydev)
            # Directional causal relation matching.
            for d,tresp in zip(dirs,td):
                dd=d.to(device)
                cres=(core(zdev+args.intervention_eps*dd)-core(zdev-args.intervention_eps*dd))/(2*args.intervention_eps)
                loss=loss+args.directional_weight*nn.functional.mse_loss(cres,tresp.to(device))
            if ti and len(dirs)>=2:
                d1,d2=dirs[0].to(device),dirs[1].to(device)
                f00=core(zdev); f10=core(zdev+args.intervention_eps*d1); f01=core(zdev+args.intervention_eps*d2); f11=core(zdev+args.intervention_eps*(d1+d2))
                cint=(f11-f10-f01+f00)/(args.intervention_eps**2)
                loss=loss+args.interaction_weight*nn.functional.mse_loss(cint,ti[0].to(device))
            losses.append(loss)
        total=sum(losses)/len(losses)
        total.backward(); nn.utils.clip_grad_norm_(core.parameters(),1.0); opt.step()
    if device.type=="cuda": torch.cuda.synchronize()
    return core,time.perf_counter()-t0


def candidate_score(core, teachers, meta_loaders, bundles, refs, args, device):
    models=[make_compiled(t,core,args,device) for t in teachers]
    acc=[]; routes=[]; drops=[]; rels=[]; causal=[]
    for m,dl,bundle,ref in zip(models,meta_loaders,bundles,refs):
        ev=evaluate(m,dl,device); acc.append(ev["accuracy"])
        routes.append(routing_agreement(ref,collect_routing(m,dl,device,args.verifier_batches)))
        drops.append(ev["accuracy"]-attention_ablation_accuracy(m,dl,device))
        rm=relational_metrics(core,bundle,device,args.intervention_eps); rels.append(rm["relational_agreement"])
        causal.append(causal_intervention_agreement(core,bundle,device,args.intervention_eps))
    score=(statistics.mean(acc)+args.routing_weight*statistics.mean(routes)+args.ablation_weight*max(0,statistics.mean(drops))
           +args.relational_weight*statistics.mean(rels)+args.causal_weight*statistics.mean(causal)
           -args.complexity_lambda*math.log1p(count_params(core)))
    return {"name":None,"avg_accuracy":statistics.mean(acc),"avg_routing":statistics.mean(routes),"avg_ablation_drop":statistics.mean(drops),"avg_relational_agreement":statistics.mean(rels),"avg_causal_agreement":statistics.mean(causal),"shared_core_params":count_params(core),"shared_core_macs":core_macs(core),"replace_params":count_params(core),"replace_macs":core_macs(core),"score":score,"task_accuracies":acc}


def search_candidates(teachers,meta_loaders,bundles,refs,args,device,seed):
    rows=[]
    for i,name in enumerate(["identity","diagonal","polynomial","affine_polynomial","low_rank"]):
        core,secs=fit_candidate(teachers,meta_loaders,bundles,name,args,device,seed+31*i)
        row=candidate_score(core,teachers,meta_loaders,bundles,refs,args,device); row.update(name=name,kind="dart_structured",eligible=True,fit_seconds=secs)
        rows.append((row,core))
    return rows


def adapt_meta(teachers,core,meta_loaders,args,device):
    # The primitive may adapt on meta tasks only during discovery; there is no task-specific adapter.
    opt=torch.optim.AdamW(core.parameters(),lr=args.lr,weight_decay=1e-4); ce=nn.CrossEntropyLoss(); its=[iter(dl) for dl in meta_loaders]
    t0=time.perf_counter()
    for _ in range(args.adaptation_steps_per_round):
        opt.zero_grad(set_to_none=True); losses=[]
        for i,(teacher,it) in enumerate(zip(teachers,its)):
            try:x,y=next(it)
            except StopIteration:its[i]=iter(meta_loaders[i]);x,y=next(its[i])
            m=make_compiled(teacher,core,args,device)
            losses.append(ce(m(x.to(device)),y.to(device)))
        sum(losses).div(len(losses)).backward(); torch.nn.utils.clip_grad_norm_(core.parameters(),1.0);opt.step()
    if device.type=="cuda": torch.cuda.synchronize()
    return time.perf_counter()-t0


def run_holdout(args,seed,holdout_task,contrast_task=None):
    device=torch.device(args.device)
    meta_tasks=[t for t in args.all_tasks if t not in {holdout_task, contrast_task}]
    teachers=[]; meta_loaders=[]; meta_verifiers=[]
    for i,t in enumerate(meta_tasks):
        te,tr,va=make_teacher(t,args,device,seed+i*1000);teachers.append(te);meta_loaders.append(tr);meta_verifiers.append(va)
    refs=[collect_routing(t,v,device,args.verifier_batches) for t,v in zip(teachers,meta_verifiers)]
    bundles=prepare_relational_bundle(teachers,meta_verifiers,args,device,seed)
    rows=[]; final_core=None
    for r in range(args.surgery_rounds):
        candidates=search_candidates(teachers,meta_loaders,bundles,refs,args,device,seed+1000*r)
        win_row,win_core=max(candidates,key=lambda rc:rc[0]["score"])
        pre=[evaluate(make_compiled(t,win_core,args,device),dl,device)["accuracy"] for t,dl in zip(teachers,meta_loaders)]
        final_core=win_core
        adapt_time=adapt_meta(teachers,final_core,meta_loaders,args,device)
        post=[evaluate(make_compiled(t,final_core,args,device),dl,device)["accuracy"] for t,dl in zip(teachers,meta_loaders)]
        after_rel=[relational_metrics(final_core,b,device,args.intervention_eps)["relational_agreement"] for b in bundles]
        rows.append({"round":r,"winner":win_row,"meta_pre_accuracy":pre,"meta_post_accuracy":post,"meta_relational_agreement":after_rel,"adaptation_seconds":adapt_time})
        print(f"  round={r} winner={win_row['name']} meta_pre={statistics.mean(pre):.4f} meta_post={statistics.mean(post):.4f} relational={statistics.mean(after_rel):.4f} core={count_params(final_core)}",flush=True)
    def test_task(task, seed_offset):
        tr=DataLoader(TaskDataset(args.train_size,task,seed+seed_offset),batch_size=args.batch_size,shuffle=True,pin_memory=device.type=="cuda")
        te=DataLoader(TaskDataset(args.test_size,task,seed+seed_offset+10000),batch_size=args.batch_size,shuffle=False,pin_memory=device.type=="cuda")
        teacher=TinyTransformer(len(VOCAB),args.d_model,args.heads,args.d_ff,args.depth).to(device)
        train_model(teacher,tr,device,args.teacher_steps,args.lr)
        teacher_ev=evaluate(teacher,te,device)
        compiled=make_compiled(teacher,final_core,args,device)
        d_ev=evaluate(compiled,te,device)
        # matched neural control, trained on target task for comparison only
        mlp=MLPControl(args.d_model,args.bottleneck).to(device)
        # Use same routing-preserving replacement but neural core is a control, not DART.
        mcore=mlp
        mcomp=make_compiled(teacher,mcore,args,device)
        opt=torch.optim.AdamW(mcore.parameters(),lr=args.lr,weight_decay=1e-4); ce=nn.CrossEntropyLoss();it=iter(tr)
        for _ in range(args.transfer_control_steps):
            try:x,y=next(it)
            except StopIteration:it=iter(tr);x,y=next(it)
            loss=ce(mcomp(x.to(device)),y.to(device));opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(mcore.parameters(),1.0);opt.step()
        ctrl=evaluate(mcomp,te,device)
        return {"task":task,"teacher":teacher_ev,"dart_zero_shot":d_ev,"mlp_control":ctrl,"gain_points":100*(d_ev["accuracy"]-teacher_ev["accuracy"]),"vs_mlp_points":100*(d_ev["accuracy"]-ctrl["accuracy"]),"signature":task_signature_from_behavior(task,args.signature_report_probes).tolist()}
    related=test_task(holdout_task,50000)
    contrast=None
    if contrast_task:
        contrast=test_task(contrast_task,70000)
    return {"holdout_task":holdout_task,"contrast_task":contrast_task,"meta_tasks":meta_tasks,"rounds":rows,"related_holdout":related,"contrast_holdout":contrast,"shared_core_params":count_params(final_core),"shared_core_macs":core_macs(final_core),"conditioner_params":0,"task_code_params":0}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--all-tasks',nargs='+',default=['add','compose','mul','sub'])
    p.add_argument('--holdout-tasks',nargs='+',default=['sub'])
    p.add_argument('--contrast-tasks',nargs='+',default=['sort'])
    p.add_argument('--seeds',nargs='+',type=int,default=[1,2])
    p.add_argument('--train-size',type=int,default=6000);p.add_argument('--verifier-size',type=int,default=1500);p.add_argument('--test-size',type=int,default=1500)
    p.add_argument('--teacher-steps',type=int,default=800);p.add_argument('--core-fit-steps',type=int,default=300);p.add_argument('--adaptation-steps-per-round',type=int,default=400);p.add_argument('--surgery-rounds',type=int,default=2);p.add_argument('--transfer-adaptation-steps',type=int,default=400)
    p.add_argument('--transfer-control-steps',type=int,default=400);p.add_argument('--d-model',type=int,default=32);p.add_argument('--heads',type=int,default=2);p.add_argument('--d-ff',type=int,default=128);p.add_argument('--depth',type=int,default=3);p.add_argument('--rank',type=int,default=8);p.add_argument('--bottleneck',type=int,default=32)
    p.add_argument('--rel-samples-per-task',type=int,default=2048);p.add_argument('--rel-directions',type=int,default=4);p.add_argument('--intervention-eps',type=float,default=0.05)
    p.add_argument('--relational-weight',type=float,default=0.50);p.add_argument('--causal-weight',type=float,default=0.50);p.add_argument('--directional-weight',type=float,default=0.50);p.add_argument('--interaction-weight',type=float,default=0.25)
    p.add_argument('--routing-weight',type=float,default=.20);p.add_argument('--ablation-weight',type=float,default=.10);p.add_argument('--core-fit-lr',type=float,default=1e-3);p.add_argument('--lr',type=float,default=3e-4);p.add_argument('--complexity-lambda',type=float,default=1e-4);p.add_argument('--signature-report-probes',type=int,default=64);p.add_argument('--batch-size',type=int,default=256);p.add_argument('--trajectory-start',type=int,default=0);p.add_argument('--trajectory-end',type=int,default=3);p.add_argument('--verifier-batches',type=int,default=20);p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu');p.add_argument('--out',default='dart013_results.json')
    a=p.parse_args();print('DART-1.3: causal/relational primitive compilation + frozen transfer',flush=True);records=[]
    for h in a.holdout_tasks:
        contrast=a.contrast_tasks[0] if a.contrast_tasks else None
        print(f'\n===== RELATED HOLDOUT {h} | CONTRAST {contrast} =====',flush=True)
        for s in a.seeds:
            print(f'seed={s}',flush=True);records.append(run_holdout(a,s,h,contrast))
    summary={'related_holdout':{},'contrast_holdout':{}}
    for task_key, key in [(r'holdout','related_holdout')]:
        pass
    for task in a.holdout_tasks:
        rs=[r for r in records if r['holdout_task']==task]
        summary['related_holdout'][task]={'teacher':statistics.mean(r['related_holdout']['teacher']['accuracy'] for r in rs),'dart_zero_shot':statistics.mean(r['related_holdout']['dart_zero_shot']['accuracy'] for r in rs),'mlp_control':statistics.mean(r['related_holdout']['mlp_control']['accuracy'] for r in rs),'gain_points':statistics.mean(r['related_holdout']['gain_points'] for r in rs),'vs_mlp_points':statistics.mean(r['related_holdout']['vs_mlp_points'] for r in rs),'shared_core_params':statistics.mean(r['shared_core_params'] for r in rs),'shared_core_macs':statistics.mean(r['shared_core_macs'] for r in rs),'conditioner_params':0,'task_code_params':0}
        contrasts=[r['contrast_holdout'] for r in rs if r['contrast_holdout'] is not None]
        if contrasts:
            ct=contrasts[0]['task'];summary['contrast_holdout'][ct]={'teacher':statistics.mean(x['teacher']['accuracy'] for x in contrasts),'dart_zero_shot':statistics.mean(x['dart_zero_shot']['accuracy'] for x in contrasts),'mlp_control':statistics.mean(x['mlp_control']['accuracy'] for x in contrasts),'gain_points':statistics.mean(x['gain_points'] for x in contrasts),'vs_mlp_points':statistics.mean(x['vs_mlp_points'] for x in contrasts)}
    out={'config':vars(a),'records':records,'summary':summary};Path(a.out).write_text(json.dumps(out,indent=2),encoding='utf-8');print('\n================ DART-1.3 SUMMARY ================',flush=True);print(summary);print(f'Saved: {Path(a.out).resolve()}')

if __name__=='__main__': main()
