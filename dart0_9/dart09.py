#!/usr/bin/env python3
"""DART-0.9: Routing-Preserving Structured Replacement with Neural Controls.

Research hypothesis
-------------------
DART-0.6/0.7 aggressively replaced the whole multi-block trajectory. Those
runs obtained latent/trajectory consistency but lost task capability. The
most likely failure is that the original Transformer's information routing
(attention + residual pathway) was removed together with the computation.

DART-0.9 freezes the successful routing-preserving structure from DART-0.8 and changes the scientific question:
    KEEP EACH TRANSFORMER BLOCK'S ATTENTION ROUTER INTACT.
    Replace only the block feed-forward transformation with a shared core
    plus tiny block-specific residual adapters.

For each block i:
    u_i = x_i + Attention_i(Norm1_i(x_i))
    x_{i+1} = u_i + C(Norm2_i(u_i)) + R_i(Norm2_i(u_i))

The attention modules and both LayerNorms are copied from the teacher and frozen during candidate search.
The main DART search is restricted to structured/non-neural operators. An MLP is retained only as a
strict neural control and is never allowed to win the DART selection. A second control trains an MLP
against teacher logits (distillation control) under the same routing-preserving interface.

The experiment records:
- task capability before/after adaptation
- FF replacement compute/parameter reduction
- attention-routing agreement (should remain high because routing is preserved)
- causal routing sensitivity: attention ablation should change output in a
  comparable way to the teacher
- transfer to related tasks

This is intentionally a controlled structural experiment, not a claim that
we have already discovered an algorithmic primitive.
"""
from __future__ import annotations

import argparse, copy, json, math, random, statistics, time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

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
        super().__init__(); self.d_model = d_model; self.depth = depth; self.heads = heads
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Parameter(torch.randn(1, BLOCK_SIZE, d_model) * 0.02)
        self.blocks = nn.ModuleList([Block(d_model, heads, d_ff) for _ in range(depth)])
        self.head = nn.Linear(d_model, 10)
    def forward(self, x, capture_attention=False):
        h = self.emb(x) + self.pos[:, :x.size(1)]
        ats = []
        for b in self.blocks:
            n = b.norm1(h)
            if capture_attention:
                if isinstance(b, RoutingPreservingBlock):
                    h, w = b.forward_capture(h)
                else:
                    a, w = b.attn(n, n, n, need_weights=True, average_attn_weights=False)
                    h = h + a
                    h = h + b.ff(b.norm2(h))
                ats.append(w)
            else:
                h = b(h)
        logits = self.head(h[:, 0])
        return (logits, ats) if capture_attention else logits


# ---------------- replacement cores ----------------
class IdentityCore(nn.Module):
    def forward(self, x): return torch.zeros_like(x)


class DiagonalCore(nn.Module):
    def __init__(self, d):
        super().__init__(); self.scale = nn.Parameter(torch.zeros(d)); self.bias = nn.Parameter(torch.zeros(d))
    def forward(self, x): return x * self.scale + self.bias


class PolynomialCore(nn.Module):
    def __init__(self, d):
        super().__init__(); self.a = nn.Parameter(torch.zeros(d)); self.b = nn.Parameter(torch.zeros(d)); self.c = nn.Parameter(torch.zeros(d))
    def forward(self, x): return self.a * x + self.b * x.square() + self.c


class AffinePolynomialCore(nn.Module):
    """Structured composition: low-dimensional affine features plus quadratic term, no neural MLP."""
    def __init__(self, d, rank):
        super().__init__()
        self.down = nn.Linear(d, rank, bias=True)
        self.up = nn.Linear(rank, d, bias=True)
        self.quad = nn.Linear(rank, d, bias=False)
        nn.init.zeros_(self.up.weight); nn.init.zeros_(self.up.bias); nn.init.zeros_(self.quad.weight)
    def forward(self, x):
        h = self.down(x)
        return self.up(h) + self.quad(h.square())


class LowRankCore(nn.Module):
    def __init__(self, d, rank):
        super().__init__(); self.down = nn.Linear(d, rank, bias=False); self.up = nn.Linear(rank, d, bias=True)
        nn.init.zeros_(self.up.weight); nn.init.zeros_(self.up.bias)
    def forward(self, x): return self.up(self.down(x))


class MLPControl(nn.Module):
    def __init__(self, d, b):
        super().__init__(); self.net = nn.Sequential(nn.Linear(d, b), nn.GELU(), nn.Linear(b, d))
    def forward(self, x): return self.net(x)


def build_core(name, d, rank, bottleneck):
    if name == "identity": return IdentityCore()
    if name == "diagonal": return DiagonalCore(d)
    if name == "polynomial": return PolynomialCore(d)
    if name == "affine_polynomial": return AffinePolynomialCore(d, rank)
    if name == "low_rank": return LowRankCore(d, rank)
    if name == "mlp": return MLPControl(d, bottleneck)
    raise ValueError(name)


class ResidualAdapter(nn.Module):
    def __init__(self, d, rank):
        super().__init__(); self.down = nn.Linear(d, rank, bias=False); self.up = nn.Linear(rank, d, bias=True)
        nn.init.zeros_(self.down.weight); nn.init.zeros_(self.up.weight); nn.init.zeros_(self.up.bias)
    def forward(self, x): return self.up(self.down(x))


class RoutingPreservingBlock(nn.Module):
    """Original attention router + shared FF core + block-specific residual."""
    def __init__(self, original: Block | "RoutingPreservingBlock", shared_core: nn.Module, residual: nn.Module):
        super().__init__(); self.norm1 = copy.deepcopy(original.norm1); self.attn = copy.deepcopy(original.attn)
        self.norm2 = copy.deepcopy(original.norm2); self.core = shared_core; self.residual = residual
    def forward(self, x):
        h = self.norm1(x); a, _ = self.attn(h, h, h, need_weights=False); u = x + a
        z = self.norm2(u); return u + self.core(z) + self.residual(z)
    def forward_capture(self, x):
        h = self.norm1(x); a, w = self.attn(h, h, h, need_weights=True, average_attn_weights=False); u = x + a
        z = self.norm2(u); return u + self.core(z) + self.residual(z), w


def install_replacement(model: TinyTransformer, start: int, end: int, core: nn.Module, residual_rank: int, source_blocks: List[nn.Module] | None = None):
    # IMPORTANT: dynamically-created adapters must inherit the candidate device.
    # Otherwise CUDA inputs can hit CPU Linear weights on the first forward pass.
    try:
        target_device = next(core.parameters()).device
    except StopIteration:
        target_device = next(model.parameters()).device
    residuals = [ResidualAdapter(model.d_model, residual_rank).to(target_device) for _ in range(end - start)]
    core = core.to(target_device)
    src = source_blocks if source_blocks is not None else list(model.blocks)
    reps = [RoutingPreservingBlock(src[i], core, residuals[i]).to(target_device) for i in range(start, end)]
    original = list(model.blocks); model.blocks = nn.ModuleList(original[:start] + reps + original[end:])
    model.to(target_device)
    return residuals


def freeze_except(m: nn.Module, trainable_modules: List[nn.Module]) -> None:
    ids = {id(p) for mod in trainable_modules for p in mod.parameters()}
    for p in m.parameters(): p.requires_grad = id(p) in ids


def count_params(m): return sum(p.numel() for p in m.parameters())


def ff_macs(d, d_ff): return 2 * d * d_ff

def core_macs(m):
    if isinstance(m, IdentityCore): return 0
    if isinstance(m, DiagonalCore): return m.scale.numel()
    if isinstance(m, PolynomialCore): return 2 * m.a.numel()
    if isinstance(m, AffinePolynomialCore):
        r = m.down.out_features; d = m.down.in_features
        return d * r + r * d + r * d
    if isinstance(m, LowRankCore): return m.down.in_features * m.down.out_features + m.up.in_features * m.up.out_features
    if isinstance(m, MLPControl): return sum(x.in_features * x.out_features for x in m.net if isinstance(x, nn.Linear))
    raise TypeError(type(m))


def residual_macs(rank, d): return 2 * d * rank


@dataclass
class Eval:
    accuracy: float; loss: float; params: int; replace_params: int; replace_macs: int


def evaluate(model, loader, device):
    model.eval(); ce = nn.CrossEntropyLoss(reduction="sum"); total = correct = 0; ls = 0.0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            z = model(x); ls += float(ce(z, y)); correct += int((z.argmax(-1) == y).sum()); total += y.numel()
    rp = rm = 0
    seen_core = set()
    for b in model.blocks:
        if isinstance(b, RoutingPreservingBlock):
            cid = id(b.core)
            if cid not in seen_core:
                rp += count_params(b.core); rm += core_macs(b.core); seen_core.add(cid)
            d = b.residual.down.in_features; r = b.residual.down.out_features
            rp += count_params(b.residual); rm += residual_macs(r, d)
    return Eval(correct / max(total,1), ls / max(total,1), count_params(model), rp, rm)


def train(model, loader, device, steps, lr):
    params = [p for p in model.parameters() if p.requires_grad]
    if not params: raise RuntimeError("No trainable parameters")
    model.train(); opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4); it = iter(loader); ce = nn.CrossEntropyLoss()
    if device.type == "cuda": torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(steps):
        try: x, y = next(it)
        except StopIteration: it = iter(loader); x, y = next(it)
        x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
        loss = ce(model(x), y)
        if not torch.isfinite(loss): raise RuntimeError("non-finite loss")
        opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(params, 1.0); opt.step()
    if device.type == "cuda": torch.cuda.synchronize()
    return time.perf_counter() - t


def collect_routing(model, loader, device, max_batches):
    weights = [[] for _ in model.blocks]
    model.eval();
    with torch.no_grad():
        for bi, (x, _y) in enumerate(loader):
            if bi >= max_batches: break
            _logits, ats = model(x.to(device, non_blocking=True), capture_attention=True)
            for i, a in enumerate(ats): weights[i].append(a.cpu())
    return [torch.cat(v, 0) for v in weights]


def routing_stats(reference, candidate, max_rows=1024):
    # Same shapes: [N, heads, T, T].
    n = min(max_rows, reference[0].shape[0]); vals = []
    for a, b in zip(reference, candidate):
        ra = a[:n].float(); rb = b[:n].float();
        mse = torch.mean((ra - rb) ** 2)
        denom = torch.mean(ra ** 2) + 1e-8
        vals.append(float(mse / denom))
    rel_err = sum(vals) / max(len(vals),1)
    return math.exp(-rel_err)


def attention_ablation_accuracy(model, loader, device, ablate_fraction=1.0):
    """A causal routing check: zero the attention output in every retained router."""
    model.eval(); handles=[]
    def hook(_module, _inputs, output):
        # Be robust to PyTorch returning either a Tensor or a (Tensor, weights) tuple.
        if isinstance(output, tuple):
            return (torch.zeros_like(output[0]),) + output[1:]
        return torch.zeros_like(output)
    for b in model.blocks:
        if isinstance(b, RoutingPreservingBlock): handles.append(b.attn.register_forward_hook(hook))
    ce=nn.CrossEntropyLoss(reduction="sum"); correct=total=0; ls=0.0
    with torch.no_grad():
        for x,y in loader:
            x=x.to(device, non_blocking=True); y=y.to(device, non_blocking=True); z=model(x)
            ls += float(ce(z,y)); correct += int((z.argmax(-1)==y).sum()); total += y.numel()
    for h in handles: h.remove()
    return correct/max(total,1), ls/max(total,1)


def fit_candidate(base, core_name, train_loader, verifier_loader, device, args, seed):
    seed_everything(seed)
    model = copy.deepcopy(base).to(device)
    source_blocks = list(base.blocks)
    # We only use the trajectory block span. Their routing is copied exactly.
    core = build_core(core_name, args.d_model, args.rank, args.bottleneck).to(device)
    residuals = install_replacement(model, args.trajectory_start, args.trajectory_end, core, args.residual_rank, source_blocks=source_blocks)
    # Hard invariant: every candidate parameter must live on the requested device.
    model.to(device)
    # Normalize `cuda` to the active CUDA index before comparing devices.
    # torch.device("cuda") != torch.device("cuda:0") even though they refer to
    # the same active CUDA device, so a raw equality check creates false failures.
    expected_device = device
    if device.type == "cuda" and device.index is None:
        expected_device = torch.device(f"cuda:{torch.cuda.current_device()}")
    bad = [
        name for name, p in model.named_parameters()
        if p.device.type != expected_device.type
        or (expected_device.index is not None and p.device.index != expected_device.index)
    ]
    if bad:
        raise RuntimeError(
            f'Device invariant violated for {len(bad)} parameters; first: {bad[0]} '
            f'(expected {expected_device}, got {dict(model.named_parameters())[bad[0]].device})'
        )
    # Freeze the copied attention routers and norms. Train only shared core + residuals.
    trainable = [core] + residuals
    freeze_except(model, trainable)
    # Fit against the original teacher's task behaviour. A small state reconstruction term
    # is added on post-attention states to keep the replacement anchored to the original
    # computational interface without replacing the attention routing itself.
    opt = torch.optim.AdamW([p for m in trainable for p in m.parameters()], lr=args.core_fit_lr, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss(); it = iter(train_loader)
    model.eval();
    steps = args.core_fit_steps
    if device.type == "cuda": torch.cuda.synchronize()
    t = time.perf_counter()
    for step in range(steps):
        try: x, y = next(it)
        except StopIteration: it = iter(train_loader); x, y = next(it)
        x=x.to(device, non_blocking=True); y=y.to(device, non_blocking=True)
        z=model(x); task=ce(z,y)
        loss=task
        # Penalize residual magnitude so it cannot simply re-learn the removed FFs.
        residual_pen=0.0
        for r in residuals:
            residual_pen = residual_pen + sum(torch.mean(p**2) for p in r.parameters())
        loss = loss + args.residual_weight * residual_pen
        opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_([p for m in trainable for p in m.parameters()],1.0); opt.step()
    if device.type == "cuda": torch.cuda.synchronize()
    secs=time.perf_counter()-t
    ev=evaluate(model,verifier_loader,device)
    return model, core, residuals, ev, secs


def fit_mlp_control(base, teacher, train_loader, verifier_loader, device, args, seed, distill=False):
    """Neural control only. Same routing-preserving interface; reported, never selected as DART."""
    seed_everything(seed)
    model = copy.deepcopy(base).to(device)
    source_blocks = list(base.blocks)
    core = build_core("mlp", args.d_model, args.rank, args.bottleneck).to(device)
    residuals = install_replacement(model, args.trajectory_start, args.trajectory_end, core, args.residual_rank, source_blocks=source_blocks)
    trainable=[core] + residuals
    freeze_except(model, trainable)
    teacher = teacher.to(device).eval()
    params=[p for m in trainable for p in m.parameters()]
    opt=torch.optim.AdamW(params, lr=args.core_fit_lr, weight_decay=1e-4)
    ce=nn.CrossEntropyLoss(); mse=nn.MSELoss(); it=iter(train_loader)
    for _ in range(args.core_fit_steps):
        try: x,y=next(it)
        except StopIteration: it=iter(train_loader); x,y=next(it)
        x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True)
        with torch.no_grad(): tz=teacher(x)
        z=model(x)
        loss=mse(z,tz) + 0.25*ce(z,y) if distill else ce(z,y)
        opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(params,1.0); opt.step()
    ev=evaluate(model,verifier_loader,device)
    return model, core, residuals, ev


def candidate_search(base, teacher, train_loader, verifier_loader, reference_routing, device, args, seed):
    out=[]
    structured_names=["identity","diagonal","polynomial","affine_polynomial","low_rank"]
    for i,name in enumerate(structured_names):
        model, core, residuals, ev, secs = fit_candidate(base,name,train_loader,verifier_loader,device,args,seed+31*i)
        cand_routing=collect_routing(model,verifier_loader,device,args.verifier_batches)
        route=routing_stats(reference_routing,cand_routing)
        abl,_=attention_ablation_accuracy(model,verifier_loader,device)
        rf=sum(count_params(r) for r in residuals)/max(count_params(core)+sum(count_params(r) for r in residuals),1)
        score=ev.accuracy + args.routing_weight*route + args.ablation_weight*max(0.0, ev.accuracy-abl) - args.complexity_lambda*math.log1p(ev.replace_params) - args.residual_weight*rf
        out.append({"name":name,"kind":"dart_structured","eligible":True,"downstream_accuracy":ev.accuracy,"downstream_loss":ev.loss,"replace_params":ev.replace_params,"replace_macs":ev.replace_macs,"routing_agreement":route,"attention_ablation_drop":ev.accuracy-abl,"residual_fraction":rf,"score":score,"train_seconds":secs})

    # Strict neural baseline: allowed for comparison but not for DART selection.
    mlp, mcore, mres, mev = fit_mlp_control(base, teacher, train_loader, verifier_loader, device, args, seed+9001, distill=False)
    mr= routing_stats(reference_routing, collect_routing(mlp,verifier_loader,device,args.verifier_batches))
    mabl,_=attention_ablation_accuracy(mlp,verifier_loader,device)
    mrf=sum(count_params(r) for r in mres)/max(count_params(mcore)+sum(count_params(r) for r in mres),1)
    out.append({"name":"mlp_control","kind":"neural_control","eligible":False,"downstream_accuracy":mev.accuracy,"downstream_loss":mev.loss,"replace_params":mev.replace_params,"replace_macs":mev.replace_macs,"routing_agreement":mr,"attention_ablation_drop":mev.accuracy-mabl,"residual_fraction":mrf,"score":None,"train_seconds":None})

    dist, dcore, dres, dev = fit_mlp_control(base, teacher, train_loader, verifier_loader, device, args, seed+9002, distill=True)
    dr= routing_stats(reference_routing, collect_routing(dist,verifier_loader,device,args.verifier_batches))
    dabl,_=attention_ablation_accuracy(dist,verifier_loader,device)
    drf=sum(count_params(r) for r in dres)/max(count_params(dcore)+sum(count_params(r) for r in dres),1)
    out.append({"name":"distilled_mlp_control","kind":"distillation_control","eligible":False,"downstream_accuracy":dev.accuracy,"downstream_loss":dev.loss,"replace_params":dev.replace_params,"replace_macs":dev.replace_macs,"routing_agreement":dr,"attention_ablation_drop":dev.accuracy-dabl,"residual_fraction":drf,"score":None,"train_seconds":None})

    return out, {"mlp_control":mlp,"distilled_mlp_control":dist}


    out=[]
    for i,name in enumerate(["identity","diagonal","polynomial","low_rank","mlp"]):
        model, core, residuals, ev, secs = fit_candidate(base,name,train_loader,verifier_loader,device,args,seed+31*i)
        cand_routing=collect_routing(model,verifier_loader,device,args.verifier_batches)
        route=routing_stats(reference_routing,cand_routing)
        abl,_=attention_ablation_accuracy(model,verifier_loader,device)
        # We value capability and routing fidelity, but penalize replacement compute and residual size.
        rf=sum(count_params(r) for r in residuals)/max(count_params(core)+sum(count_params(r) for r in residuals),1)
        score=ev.accuracy + args.routing_weight*route + args.ablation_weight*max(0.0, ev.accuracy-abl) - args.complexity_lambda*math.log1p(ev.replace_params) - args.residual_weight*rf
        out.append({"name":name,"downstream_accuracy":ev.accuracy,"downstream_loss":ev.loss,"replace_params":ev.replace_params,"replace_macs":ev.replace_macs,"routing_agreement":route,"attention_ablation_drop":ev.accuracy-abl,"residual_fraction":rf,"score":score,"train_seconds":secs})
    return sorted(out,key=lambda x:x["score"],reverse=True)


def cuda_latency(model,loader,device,warmup,iters):
    if device.type != "cuda": return None
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
    teacher=TinyTransformer(len(VOCAB),args.d_model,args.heads,args.d_ff,args.depth).to(device)
    freeze_except(teacher, [teacher]); train(teacher,tr,device,args.teacher_steps,args.lr)
    tev=evaluate(teacher,te,device)
    ref_route=collect_routing(teacher,va,device,args.verifier_batches)
    dart=copy.deepcopy(teacher).to(device); rounds=[]
    for r in range(args.surgery_rounds):
        cand, controls = candidate_search(dart,teacher,tr,va,ref_route,device,args,seed+1000*r)
        structured=[c for c in cand if c["eligible"]]
        win=max(structured,key=lambda x:x["score"])
        dart,core,residuals,_,fit_secs=fit_candidate(dart,win["name"],tr,va,device,args,seed+9000*r)
        # Rebuild after fit already contains copied teacher routing + selected core/residuals.
        pre=evaluate(dart,te,device); adapt=train(dart,tr,device,args.adaptation_steps_per_round,args.lr); post=evaluate(dart,te,device)
        dart_route=collect_routing(dart,va,device,args.verifier_batches); route=routing_stats(ref_route,dart_route)
        rounds.append({"round":r,"winner":win,"all_candidates":cand,"neural_controls":{k: {"accuracy": evaluate(v,va,device).accuracy, "params": count_params(v)} for k,v in controls.items()},"pre_adapt":asdict(pre),"post_adapt":asdict(post),"routing_agreement_after":route,"adaptation_seconds":adapt,"fit_seconds":fit_secs,"latency_ms":cuda_latency(dart,te,device,args.latency_warmup,args.latency_iters)})
        print(f"round={r} winner={win['name']} score={win['score']:.4f} downstream={win['downstream_accuracy']:.4f} routing={win['routing_agreement']:.4f} ablation_drop={win['attention_ablation_drop']:.4f} pre={pre.accuracy:.4f} post={post.accuracy:.4f}",flush=True)
    return {"seed":seed,"task":task,"teacher":asdict(tev),"dart_final":asdict(evaluate(dart,te,device)),"rounds":rounds}


def run_transfer(args,seed,source_task,target_task):
    device=torch.device(args.device)
    src=DataLoader(TaskDataset(args.train_size,source_task,seed),batch_size=args.batch_size,shuffle=True,pin_memory=device.type=="cuda")
    tgt=DataLoader(TaskDataset(args.train_size,target_task,seed+20000),batch_size=args.batch_size,shuffle=True,pin_memory=device.type=="cuda")
    test=DataLoader(TaskDataset(args.test_size,target_task,seed+30000),batch_size=args.batch_size,shuffle=False,pin_memory=device.type=="cuda")
    source=TinyTransformer(len(VOCAB),args.d_model,args.heads,args.d_ff,args.depth).to(device); train(source,src,device,args.teacher_steps,args.lr)
    ref=collect_routing(source,src,device,args.trajectory_batches)
    cand,_controls = candidate_search(source,source,src,src,ref,device,args,seed+7000)
    cand=max([c for c in cand if c["eligible"]],key=lambda x:x["score"])
    dart,_,_,_,_=fit_candidate(source,cand["name"],src,src,device,args,seed+8000)
    scratch=TinyTransformer(len(VOCAB),args.d_model,args.heads,args.d_ff,args.depth).to(device); train(scratch,tgt,device,args.transfer_adaptation_steps,args.lr); train(dart,tgt,device,args.transfer_adaptation_steps,args.lr)
    s=evaluate(scratch,test,device); d=evaluate(dart,test,device)
    return {"seed":seed,"source_task":source_task,"target_task":target_task,"winner":cand,"scratch_after":asdict(s),"dart_after":asdict(d),"transfer_gain_points":100*(d.accuracy-s.accuracy)}


def mean_std(xs): return {"mean":statistics.mean(xs) if xs else None,"std":statistics.stdev(xs) if len(xs)>1 else 0.0}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--seeds",nargs="+",type=int,default=[1,2]); p.add_argument("--tasks",nargs="+",default=["add","compose"])
    p.add_argument("--transfer-pairs",nargs=2,action="append",metavar=("SOURCE","TARGET"),default=[["add","compose"],["mul","sub"]])
    p.add_argument("--train-size",type=int,default=6000); p.add_argument("--verifier-size",type=int,default=1500); p.add_argument("--test-size",type=int,default=1500)
    p.add_argument("--teacher-steps",type=int,default=800); p.add_argument("--core-fit-steps",type=int,default=300); p.add_argument("--adaptation-steps-per-round",type=int,default=400); p.add_argument("--surgery-rounds",type=int,default=2); p.add_argument("--transfer-adaptation-steps",type=int,default=400)
    p.add_argument("--d-model",type=int,default=32); p.add_argument("--heads",type=int,default=2); p.add_argument("--d-ff",type=int,default=128); p.add_argument("--depth",type=int,default=3); p.add_argument("--rank",type=int,default=8); p.add_argument("--bottleneck",type=int,default=32)
    p.add_argument("--residual-rank",type=int,default=2); p.add_argument("--trajectory-start",type=int,default=0); p.add_argument("--trajectory-end",type=int,default=3); p.add_argument("--verifier-batches",type=int,default=20); p.add_argument("--trajectory-batches",type=int,default=20)
    p.add_argument("--residual-weight",type=float,default=0.01); p.add_argument("--routing-weight",type=float,default=0.20); p.add_argument("--ablation-weight",type=float,default=0.10); p.add_argument("--core-fit-lr",type=float,default=1e-3); p.add_argument("--lr",type=float,default=3e-4); p.add_argument("--complexity-lambda",type=float,default=1e-4)
    p.add_argument("--batch-size",type=int,default=256); p.add_argument("--latency-warmup",type=int,default=30); p.add_argument("--latency-iters",type=int,default=30); p.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu"); p.add_argument("--out",default="dart09_results.json")
    a=p.parse_args(); records=[]; print("DART-0.9: routing-preserving structured replacement + neural controls",flush=True)
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
    winfreq={};
    for r in records:
        for rr in r["rounds"]: winfreq[rr["winner"]["name"]]=winfreq.get(rr["winner"]["name"],0)+1
    transfer={}
    for src,tgt in a.transfer_pairs:
        rows=[r for r in trs if r["source_task"]==src and r["target_task"]==tgt]
        transfer[f"{src}->{tgt}"]={"scratch_after":mean_std([r["scratch_after"]["accuracy"] for r in rows]),"dart_after":mean_std([r["dart_after"]["accuracy"] for r in rows]),"gain_points":mean_std([r["transfer_gain_points"] for r in rows])}
    out={"config":vars(a),"records":records,"transfer_records":trs,"summary":{"teacher":mean_std([r["teacher"]["accuracy"] for r in records]),"dart_final":mean_std([r["dart_final"]["accuracy"] for r in records]),"winner_frequency":winfreq,"transfer":transfer}}
    Path(a.out).write_text(json.dumps(out,indent=2),encoding="utf-8"); print("\n================ DART-0.9 SUMMARY ================"); print(out["summary"]); print(f"Saved: {Path(a.out).resolve()}")

if __name__ == "__main__": main()
