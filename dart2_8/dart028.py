#!/usr/bin/env python3
"""DART-2.8: frozen causal primitive reparameterization + strict transfer.

Research hypothesis:
  DART-2.6 found strong intervention-level necessity, while single-slot necessity
  could remain near zero. DART-2.7 tests whether the causal computation is a
  compact localized primitive, a redundant set, or a synergistic set of
  interacting components.

Core objects:
  - singleton interventions for each active slot
  - all non-empty subsets of active slots (within the adapter budget)
  - causal interaction matrix
  - minimal causal set (MCS)
  - causal concentration, redundancy, synergy
  - cross-task causal overlap
  - random-set control
  - frozen holdout transfer

No task embeddings, large conditioners, unrestricted residual networks, or
unbounded graph surgery are introduced. The central discovery target is a
small causal set S* that is necessary, sufficient, teacher-aligned, and
transferable.
"""
from __future__ import annotations
import argparse, copy, itertools, json, random
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

VOCAB=list("0123456789+= "); STOI={c:i for i,c in enumerate(VOCAB)}; PAD=STOI[' ']; BLOCK=12

def seed_all(s):
    random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

class Progress:
    """Compact live progress bar."""
    def __init__(self, total, seed_idx, seeds):
        self.total=max(1,total); self.seed_idx=seed_idx; self.seeds=seeds; self.current=0.0; self.phase='starting'
    def show(self, frac, phase, detail=''):
        self.current=max(0.0,min(1.0,frac)); width=30
        filled=int(width*self.current); bar='='*filled+'>'+(' '*(width-filled-1) if filled<width else '')
        msg=f"\r[DART-2.8][seed {self.seed_idx}/{self.seeds}] [{bar}] {self.current*100:6.2f}% | {phase}"
        if detail: msg += f" | {detail}"
        print(msg[:240],end='',flush=True)
    def done(self,phase='complete'):
        self.show(1.0,phase); print()

def target(a,b,t):
    aa=[int(c) for c in str(a).zfill(3)]; bb=[int(c) for c in str(b).zfill(3)]
    if t=='add': return (aa[0]+bb[-1])%10
    if t=='sub': return (aa[-1]-bb[0])%10
    if t=='mul': return (aa[0]*bb[-1])%10
    if t=='sort': return min(aa+bb)
    if t=='compose': return ((aa[0]+bb[-1])*(aa[1]+1))%10
    raise ValueError(t)

def make_example(a,b,t):
    ids=[STOI[c] for c in f"{a}+{b}="]; return (ids+[PAD]*BLOCK)[:BLOCK], target(a,b,t)

class TaskDataset(Dataset):
    def __init__(self,n,task,seed):
        r=random.Random(seed); self.rows=[]
        for _ in range(n):
            x,y=make_example(r.randint(0,999),r.randint(0,999),task)
            self.rows.append((torch.tensor(x),torch.tensor(y)))
    def __len__(self): return len(self.rows)
    def __getitem__(self,i): return self.rows[i]

class Block(nn.Module):
    def __init__(self,d,h,ff):
        super().__init__(); self.norm1=nn.LayerNorm(d); self.attn=nn.MultiheadAttention(d,h,batch_first=True,dropout=0.); self.norm2=nn.LayerNorm(d); self.ff=nn.Sequential(nn.Linear(d,ff),nn.GELU(),nn.Linear(ff,d))
    def forward(self,x):
        n=self.norm1(x); a,_=self.attn(n,n,n,need_weights=False); u=x+a; return u+self.ff(self.norm2(u))

class Teacher(nn.Module):
    def __init__(self,v,d=32,h=2,ff=128,depth=3):
        super().__init__(); self.emb=nn.Embedding(v,d); self.pos=nn.Parameter(torch.randn(1,BLOCK,d)*.02); self.blocks=nn.ModuleList([Block(d,h,ff) for _ in range(depth)]); self.head=nn.Linear(d,10); self.d=d
    def forward(self,x):
        h=self.emb(x)+self.pos[:,:x.size(1)]
        for b in self.blocks: h=b(h)
        return self.head(h[:,0])

class DiagonalCore(nn.Module):
    def __init__(self,d): super().__init__(); self.s=nn.Parameter(torch.randn(d)*.02+.2); self.b=nn.Parameter(torch.zeros(d))
    def forward(self,x): return x*self.s+self.b
class PolynomialCore(nn.Module):
    def __init__(self,d): super().__init__(); self.a=nn.Parameter(torch.randn(d)*.02+.2); self.b=nn.Parameter(torch.randn(d)*.01); self.c=nn.Parameter(torch.zeros(d))
    def forward(self,x): return self.a*x+self.b*x.square()+self.c
class AffinePolynomialCore(nn.Module):
    def __init__(self,d,r):
        super().__init__(); self.down=nn.Linear(d,r); self.up=nn.Linear(r,d); self.quad=nn.Linear(r,d,bias=False)
        nn.init.xavier_uniform_(self.down.weight); nn.init.zeros_(self.down.bias); nn.init.xavier_uniform_(self.up.weight,gain=.05); nn.init.zeros_(self.up.bias); nn.init.xavier_uniform_(self.quad.weight,gain=.02)
    def forward(self,x): h=self.down(x); return self.up(h)+self.quad(h.square())
class LowRankCore(nn.Module):
    def __init__(self,d,r):
        super().__init__(); self.down=nn.Linear(d,r,bias=False); self.up=nn.Linear(r,d); nn.init.xavier_uniform_(self.down.weight); nn.init.xavier_uniform_(self.up.weight,gain=.05); nn.init.zeros_(self.up.bias)
    def forward(self,x): return self.up(self.down(x))

def make_core(name,d,r):
    m={'diagonal':DiagonalCore,'polynomial':PolynomialCore,'affine_polynomial':lambda dd:AffinePolynomialCore(dd,r),'low_rank':lambda dd:LowRankCore(dd,r)}
    return m[name](d)

class SharedSkeleton(nn.Module):
    """Three shared structured stages."""
    def __init__(self,nodes,motif,d,r):
        super().__init__(); self.node_names=tuple(nodes); self.motif=motif; self.nodes=nn.ModuleList([make_core(n,d,r) for n in nodes])
    def forward(self,x):
        if self.motif=='sequential':
            h=x
            vals=[]
            for node in self.nodes:
                v=node(h); vals.append(v); h=h+v
            return h, vals
        if self.motif=='parallel_sum':
            vals=[node(x) for node in self.nodes]; return x+sum(vals), vals
        if self.motif=='residual_parallel':
            vals=[node(x) for node in self.nodes]; return x+sum(vals)+0.1*x*vals[0], vals
        raise ValueError(self.motif)
    def stages(self,x):
        if self.motif!='sequential':
            h=x; vals=[]
            for node in self.nodes: h=h+node(h); vals.append(h)
            return vals
        h=x; out=[]
        for node in self.nodes:
            h=h+node(h); out.append(h)
        return out

class NoOpAdapter(nn.Module):
    def forward(self, x, ref=None): return x

class TaskAdapter(nn.Module):
    MODES=('identity','scale','negate','difference','product','mix')
    def __init__(self,mode,d):
        super().__init__(); self.mode=mode; self.d=d
        # Non-zero initialization avoids DART-1.4 dead-zone.
        self.raw=nn.Parameter(torch.tensor([0.25,0.25,0.05,0.05],dtype=torch.float32))
    def forward(self,x,ref=None):
        if self.mode=='identity': return x
        if self.mode=='scale': return (1+self.raw[0])*x + self.raw[1]
        if self.mode=='negate': return -(1+self.raw[0])*x + self.raw[1]
        if self.mode=='difference': return x - (1+self.raw[0])*(ref if ref is not None else torch.roll(x,1,-1)) + self.raw[1]
        if self.mode=='product': return x*(1+torch.tanh(self.raw[0])*(ref if ref is not None else x)) + self.raw[1]
        if self.mode=='mix':
            a=torch.sigmoid(self.raw[0]); r=ref if ref is not None else torch.zeros_like(x); return a*x+(1-a)*r+self.raw[1]
        raise ValueError(self.mode)

class InterleavedRule(nn.Module):
    def __init__(self,nodes,motif,mode,d,r,mask):
        super().__init__(); self.skeleton=SharedSkeleton(nodes,motif,d,r); self.mode=mode; self.mask=tuple(mask); self.adapters=nn.ModuleList([TaskAdapter(mode,d) if mask[i] else NoOpAdapter() for i in range(3)])
    def forward(self,x):
        h=x
        refs=[]
        for i,node in enumerate(self.skeleton.nodes):
            h=h+node(h) if self.skeleton.motif=='sequential' else h
            if self.skeleton.motif!='sequential':
                # in parallel/residual modes use the node on the current state to keep slot semantics explicit
                h=h+node(h)
            refs.append(h)
            ref=refs[-2] if i>0 else None
            h=self.adapters[i](h,ref)
        return h
    def node_output(self,x,slot):
        h=x; refs=[]
        for i,node in enumerate(self.skeleton.nodes):
            h=h+node(h) if self.skeleton.motif=='sequential' else h+node(h)
            refs.append(h); ref=refs[-2] if i>0 else None
            if i==slot: return h, ref
            h=self.adapters[i](h,ref)
        return h, refs[-2] if len(refs)>1 else None

class RoutingBlock(nn.Module):
    def __init__(self,b,rule):
        super().__init__(); self.norm1=copy.deepcopy(b.norm1); self.attn=copy.deepcopy(b.attn); self.norm2=copy.deepcopy(b.norm2); self.rule=rule
    def forward(self,x):
        n=self.norm1(x); a,_=self.attn(n,n,n,need_weights=False); u=x+a; z=self.norm2(u); return u+self.rule(z)

class RoutingCompiled(nn.Module):
    def __init__(self,teacher,rule,start,end):
        super().__init__(); self.emb=copy.deepcopy(teacher.emb); self.pos=copy.deepcopy(teacher.pos); self.head=copy.deepcopy(teacher.head); self.blocks=nn.ModuleList([RoutingBlock(b,rule) if start<=i<end else copy.deepcopy(b) for i,b in enumerate(teacher.blocks)])
    def forward(self,x):
        h=self.emb(x)+self.pos[:,:x.size(1)]
        for b in self.blocks: h=b(h)
        return self.head(h[:,0])

class MLPReplaceBlock(nn.Module):
    def __init__(self,b,d,w):
        super().__init__(); self.norm1=copy.deepcopy(b.norm1); self.attn=copy.deepcopy(b.attn); self.norm2=copy.deepcopy(b.norm2); self.m=nn.Sequential(nn.Linear(d,w),nn.GELU(),nn.Linear(w,d))
    def forward(self,x): n=self.norm1(x); a,_=self.attn(n,n,n,need_weights=False); u=x+a; return u+self.m(self.norm2(u))
class MLPControl(nn.Module):
    def __init__(self,t,start,end,w):
        super().__init__(); self.emb=copy.deepcopy(t.emb); self.pos=copy.deepcopy(t.pos); self.head=copy.deepcopy(t.head); self.blocks=nn.ModuleList([MLPReplaceBlock(b,t.d,w) if start<=i<end else copy.deepcopy(b) for i,b in enumerate(t.blocks)])
    def forward(self,x):
        h=self.emb(x)+self.pos[:,:x.size(1)]
        for b in self.blocks: h=b(h)
        return self.head(h[:,0])

def count_params(m): return sum(p.numel() for p in m.parameters())
def evaluate(model,loader,device):
    model.eval(); total=correct=0
    with torch.no_grad():
        for x,y in loader:
            x=x.to(device); y=y.to(device); correct+=int((model(x).argmax(-1)==y).sum()); total+=y.numel()
    return correct/max(total,1)

def train(model,loader,device,steps,lr):
    ps=[p for p in model.parameters() if p.requires_grad]
    if not ps: return
    opt=torch.optim.AdamW(ps,lr=lr,weight_decay=1e-4); ce=nn.CrossEntropyLoss(); it=iter(loader); model.train()
    for _ in range(steps):
        try: x,y=next(it)
        except StopIteration: it=iter(loader); x,y=next(it)
        x=x.to(device); y=y.to(device); loss=ce(model(x),y); opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(ps,1.); opt.step()

def train_teacher(task,args,device,seed):
    tr=DataLoader(TaskDataset(args.train_size,task,seed),batch_size=args.batch_size,shuffle=True,pin_memory=device.type=='cuda')
    va=DataLoader(TaskDataset(args.verifier_size,task,seed+10000),batch_size=args.batch_size,shuffle=False,pin_memory=device.type=='cuda')
    te=Teacher(len(VOCAB),args.d_model,args.heads,args.d_ff,args.depth).to(device); train(te,tr,device,args.teacher_steps,args.lr); return te,tr,va

def capture_ff(teacher,loader,device,maxn,layer):
    zs=[]
    with torch.no_grad():
        for x,_ in loader:
            have=sum(t.shape[0] for t in zs)
            if have>=maxn: break
            x=x.to(device); h=teacher.emb(x)+teacher.pos[:,:x.size(1)]
            for i,b in enumerate(teacher.blocks):
                n=b.norm1(h); a,_=b.attn(n,n,n,need_weights=False); u=h+a; z=b.norm2(u)
                if i==layer:
                    flat=z.reshape(-1,z.shape[-1]); take=min(maxn-have,flat.shape[0]); zs.append(flat[:take].cpu()); break
                h=u+b.ff(z)
    return torch.cat(zs)

def source_bundle(teacher,loader,args,device):
    z=capture_ff(teacher,loader,device,args.rel_samples_per_task,args.trajectory_start)
    with torch.no_grad(): y=teacher.blocks[args.trajectory_start].ff(z.to(device)).detach().cpu()
    return z,y

def adapter_state(op): return {k:v.detach().clone() for k,v in op.state_dict().items()} if not isinstance(op,NoOpAdapter) else None

def fit_adapter(skeleton,mode,mask,z,y,args,device,steps,init=None):
    rule=InterleavedRule(skeleton.node_names,skeleton.motif,mode,args.d_model,args.rank,mask).to(device)
    rule.skeleton=skeleton
    if init is not None:
        # Load only adapter params from init; skeleton is fixed.
        for slot,ad in enumerate(rule.adapters):
            if isinstance(ad,NoOpAdapter): continue
            ad.load_state_dict(init[slot])
    for p in skeleton.parameters(): p.requires_grad=False
    ps=[p for p in rule.parameters() if p.requires_grad];
    if not ps: return rule
    opt=torch.optim.AdamW(ps,lr=args.adapter_lr,weight_decay=args.adapter_l2); z=z.to(device); y=y.to(device); rule.train()
    for _ in range(steps):
        idx=torch.randperm(len(z),device=device)[:min(args.fit_batch_samples,len(z))]
        zz=z[idx]; yy=y[idx]; pred=rule(zz); loss=((pred-yy)**2).mean()+args.adapter_l2*sum(p.square().mean() for p in ps)
        opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(ps,1.); opt.step()
    return rule

def fit_shared_skeleton(motif,nodes,mode,mask,teachers,args,device,tracker=None,base_frac=0.0,span=1.0):
    sk=SharedSkeleton(nodes,motif,args.d_model,args.rank).to(device)
    tasks=list(teachers); bundles={t:source_bundle(teachers[t][0],teachers[t][1],args,device) for t in tasks}
    rules=nn.ModuleList([InterleavedRule(nodes,motif,mode,args.d_model,args.rank,mask) for _ in tasks]).to(device)
    # share one skeleton during joint fitting
    for r in rules: r.skeleton=sk
    params=list(sk.parameters())+[p for r in rules for p in r.adapters.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(params,lr=args.core_fit_lr,weight_decay=1e-4)
    for step in range(args.core_fit_steps):
        loss_total=0.
        for i,t in enumerate(tasks):
            z,y=bundles[t]; idx=torch.randperm(len(z))[:min(args.fit_batch_samples,len(z))]; pred=rules[i](z[idx].to(device)); yy=y[idx].to(device)
            loss_total += ((pred-yy)**2).mean()
        opt.zero_grad(set_to_none=True); loss_total.backward(); nn.utils.clip_grad_norm_(params,1.); opt.step()
        if tracker and (step == 0 or (step+1) % max(1,args.core_fit_steps//10) == 0):
            tracker.show(base_frac + span*0.45*(step+1)/max(1,args.core_fit_steps), "shared-fit", f"{step+1}/{args.core_fit_steps}")
    # refit adapters on frozen skeleton, independently per task
    final=[]; st=[]
    for t in tasks:
        z,y=bundles[t]
        init=[adapter_state(rules[tasks.index(t)].adapters[s]) if not isinstance(rules[tasks.index(t)].adapters[s],nn.Identity) else None for s in range(3)]
        r=fit_adapter(sk,mode,mask,z,y,args,device,args.adapter_fit_steps,init)
        # stability via two halves, relative L2
        n=len(z); half=max(1,n//2)
        r1=fit_adapter(sk,mode,mask,z[:half],y[:half],args,device,max(10,args.adapter_fit_steps//2),[adapter_state(a) if not isinstance(a,NoOpAdapter) else None for a in r.adapters])
        r2=fit_adapter(sk,mode,mask,z[half:],y[half:],args,device,max(10,args.adapter_fit_steps//2),[adapter_state(a) if not isinstance(a,NoOpAdapter) else None for a in r.adapters])
        dist=0.; count=0
        for a,b in zip(r1.adapters,r2.adapters):
            if isinstance(a,NoOpAdapter): continue
            for pa,pb in zip(a.parameters(),b.parameters()): dist+=float((pa-pb).norm().item()); count+=1
        st.append(dist/max(count,1)); final.append(r)
    return sk,final,sum(st)/len(st),bundles

def fit_separate(task,teacher_bundle,motif,nodes,mode,mask,args,device):
    tea,tr,_=teacher_bundle; z,y=source_bundle(tea,tr,args,device); sk=SharedSkeleton(nodes,motif,args.d_model,args.rank).to(device)
    return fit_adapter(sk,mode,mask,z,y,args,device,args.separate_control_steps)

def rule_causal_fidelity(rule,teacher,loader,args,device,layer):
    x,_=next(iter(loader)); x=x.to(device)
    with torch.no_grad():
        h=teacher.emb(x)+teacher.pos[:,:x.size(1)]
        for i,b in enumerate(teacher.blocks):
            n=b.norm1(h); a,_=b.attn(n,n,n,need_weights=False); u=h+a; z=b.norm2(u)
            if i==layer:
                def teacher_down(hh):
                    for j in range(i+1,len(teacher.blocks)):
                        hh=teacher.blocks[j](hh)
                    return teacher.head(hh[:,0])
                teacher_base=teacher_down(u+b.ff(z))
                teacher_zero=teacher_down(u)
                teacher_delta=teacher_base-teacher_zero
                sims=[]
                for slot in range(3):
                    rule_base=rule(z)
                    saved=rule.adapters[slot]
                    rule.adapters[slot]=NoOpAdapter().to(device)
                    rule_pert=rule(z)
                    rule.adapters[slot]=saved
                    d=rule_base-rule_pert
                    # Compare the DART intervention's downstream delta with the teacher's FF removal delta
                    d_mean=d.mean(dim=1) if d.dim()==3 else d
                    t_mean=teacher_delta
                    if d_mean.shape[-1] != t_mean.shape[-1]:
                        # deterministic projection for diagnostic only
                        d_mean=d_mean.mean(dim=-1,keepdim=True)
                        t_mean=t_mean.mean(dim=-1,keepdim=True)
                    sims.append(float(nn.functional.cosine_similarity(d_mean,t_mean,dim=-1).mean().item()))
                return sum(sims)/len(sims)
            h=b(h)
    return 0.0


def _teacher_intervention_logits(teacher, x, layer):
    """Return teacher logits with normal FF and with FF ablated at `layer`."""
    with torch.no_grad():
        h=teacher.emb(x)+teacher.pos[:,:x.size(1)]
        for i,b in enumerate(teacher.blocks):
            n=b.norm1(h); a,_=b.attn(n,n,n,need_weights=False); u=h+a; z=b.norm2(u)
            if i==layer:
                h_full=u+b.ff(z); h_zero=u
                for j in range(i+1,len(teacher.blocks)):
                    h_full=teacher.blocks[j](h_full); h_zero=teacher.blocks[j](h_zero)
                return teacher.head(h_full[:,0]), teacher.head(h_zero[:,0])
            h=u+b.ff(z)
    raise RuntimeError('invalid intervention layer')


def _compiled_intervention_logits(teacher, rule, x, layer, slot=None):
    """Return compiled DART logits with normal rule and optionally one slot ablated."""
    with torch.no_grad():
        h=teacher.emb(x)+teacher.pos[:,:x.size(1)]
        for i,b in enumerate(teacher.blocks):
            n=b.norm1(h); a,_=b.attn(n,n,n,need_weights=False); u=h+a; z=b.norm2(u)
            if i==layer:
                if slot is None:
                    rr=rule(z)
                else:
                    saved=rule.adapters[slot]
                    rule.adapters[slot]=NoOpAdapter().to(x.device)
                    rr=rule(z)
                    rule.adapters[slot]=saved
                h_full=u+rr; h_zero=u
                for j in range(i+1,len(teacher.blocks)):
                    h_full=teacher.blocks[j](h_full); h_zero=teacher.blocks[j](h_zero)
                return teacher.head(h_full[:,0]), teacher.head(h_zero[:,0])
            h=u+b.ff(z)
    raise RuntimeError('invalid intervention layer')


def slot_reconstruction_fidelity(rules, teachers, meta, args, device):
    """Teacher-grounded causal reconstruction of each active task slot.

    For each source task and active slot, compare the logit delta caused by
    ablating the learned task adapter with the logit delta caused by ablating
    the teacher's corresponding FF computation. This asks whether the selected
    slot is not merely useful, but reproduces the teacher's causal intervention
    direction.
    """
    vals=[]
    per_slot=[[] for _ in range(3)]
    for i,t in enumerate(meta):
        tea,_,va=teachers[t]
        ds=DataLoader(TaskDataset(min(args.causal_probe_size,args.verifier_size),t,99900+i),batch_size=min(64,args.batch_size),shuffle=False)
        x,_=next(iter(ds)); x=x.to(device)
        full_t,zero_t=_teacher_intervention_logits(tea,x,args.trajectory_start)
        td=full_t-zero_t
        for slot in range(3):
            if isinstance(rules[i].adapters[slot],NoOpAdapter):
                continue
            full_d,zero_d=_compiled_intervention_logits(tea,rules[i],x,args.trajectory_start,slot)
            dd=full_d-zero_d
            sim=float(nn.functional.cosine_similarity(dd,td,dim=-1).mean().item())
            sim=max(-1.0,min(1.0,sim))
            per_slot[slot].append(sim)
            vals.append(sim)
    avg=sum(vals)/len(vals) if vals else 0.0
    return avg, [sum(v)/len(v) if v else 0.0 for v in per_slot]

def adapter_effect(rule,z,delta,device):
    base={k:v.clone() for k,v in rule.state_dict().items()}; out=[]
    with torch.no_grad(): y0=rule(z.to(device))
    for slot,ad in enumerate(rule.adapters):
        if isinstance(ad,NoOpAdapter): continue
        for p in ad.parameters():
            p.data.add_(delta); y1=rule(z.to(device)); p.data.sub_(delta); out.append(float((y1-y0).norm().item()/max(len(z),1)))
    rule.load_state_dict(base); return sum(out)/len(out) if out else 0.0

def slot_causal_necessity(rules, bundles, device):
    """Estimate whether each active slot is necessary for cross-task differences.

    For each pair of source-task rules, compare their full outputs on the same
    probe states. Then ablate one slot in both rules and measure how much of the
    task-difference signal remains. Large drop means that slot carries causal
    task-specific information rather than merely correlating with it.
    """
    task_states=[]
    for r,(z,_) in zip(rules,bundles.values()):
        zz=z[:min(256,len(z))].to(device)
        task_states.append((r,zz))
    scores=[]
    for slot in range(3):
        pair_scores=[]
        for i in range(len(task_states)):
            for j in range(i+1,len(task_states)):
                ra,xa=task_states[i]; rb,xb=task_states[j]
                n=min(len(xa),len(xb)); xa=xa[:n]; xb=xb[:n]
                with torch.no_grad():
                    full_a=ra(xa); full_b=rb(xb); full=(full_a-full_b).float()
                    if isinstance(ra.adapters[slot],NoOpAdapter) and isinstance(rb.adapters[slot],NoOpAdapter):
                        continue
                    old_a,old_b=ra.adapters[slot],rb.adapters[slot]
                    ra.adapters[slot]=NoOpAdapter().to(device); rb.adapters[slot]=NoOpAdapter().to(device)
                    ab_a=ra(xa); ab_b=rb(xb); rem=(ab_a-ab_b).float()
                    ra.adapters[slot],rb.adapters[slot]=old_a,old_b
                    base=float(full.norm().item())+1e-8; remain=float(rem.norm().item()); drop=max(0.0,1.0-remain/base)
                    pair_scores.append(drop)
        scores.append(sum(pair_scores)/len(pair_scores) if pair_scores else 0.0)
    return sum(scores)/len(scores) if scores else 0.0, scores

def placement_random(mask,n=3):
    m=list(mask); random.shuffle(m); return tuple(m)


class BlendAdapter(nn.Module):
    """Blend a learned adapter with identity for controlled intervention strength."""
    def __init__(self, base, alpha):
        super().__init__(); self.base=base; self.alpha=float(alpha)
    def forward(self,x,ref=None):
        y=self.base(x,ref); return x + self.alpha*(y-x)


def causal_mediation_metrics(rule, teacher, loader, args, device, layer):
    """Measure necessity, sufficiency, trajectory fidelity and minimality at logit level."""
    x,_=next(iter(loader)); x=x.to(device)
    with torch.no_grad():
        h=teacher.emb(x)+teacher.pos[:,:x.size(1)]
        teacher_delta=None; z=None; u=None
        for i,b in enumerate(teacher.blocks):
            n=b.norm1(h); a,_=b.attn(n,n,n,need_weights=False); u0=h+a; z0=b.norm2(u0)
            if i==layer:
                hf=u0+b.ff(z0); hz=u0
                for j in range(i+1,len(teacher.blocks)):
                    hf=teacher.blocks[j](hf); hz=teacher.blocks[j](hz)
                teacher_delta=teacher.head(hf[:,0])-teacher.head(hz[:,0]); z=z0; u=u0
                break
            h=u0+b.ff(z0)
    if teacher_delta is None:
        return {'necessity':0.0,'sufficiency':0.0,'trajectory_fidelity':0.0,'minimal_alpha':1.0,'minimality':0.0,'cme':0.0,'slot_scores':[0.0,0.0,0.0]}

    def downstream(rule_out):
        hh=u+rule_out
        for j in range(layer+1,len(teacher.blocks)): hh=teacher.blocks[j](hh)
        return teacher.head(hh[:,0])

    td_norm=float(teacher_delta.norm(dim=-1).mean().item())+1e-8
    slot_scores=[]; suff_scores=[]; curve_scores=[]; min_alphas=[]
    for slot in range(3):
        if isinstance(rule.adapters[slot],NoOpAdapter):
            slot_scores.append(0.0); suff_scores.append(0.0); continue
        saved=rule.adapters[slot]
        with torch.no_grad():
            full_out=rule(z)
            rule.adapters[slot]=NoOpAdapter().to(device); neutral_out=rule(z); rule.adapters[slot]=saved
            full_logits=downstream(full_out); neutral_logits=downstream(neutral_out)
        total_delta=float((full_logits-neutral_logits).norm(dim=-1).mean().item())
        total_rule=float(full_logits.norm(dim=-1).mean().item())+1e-8
        necessity=min(1.0,total_delta/total_rule)
        slot_scores.append(necessity)
        local=[]
        for alpha in args.intervention_grid:
            with torch.no_grad():
                rule.adapters[slot]=BlendAdapter(saved,alpha).to(device)
                out=rule(z)
                rule.adapters[slot]=saved
                logits=downstream(out)
            delta=logits-neutral_logits
            cos=float(nn.functional.cosine_similarity(delta,teacher_delta,dim=-1).mean().item())
            mag=float(delta.norm(dim=-1).mean().item()/td_norm)
            score=max(0.0,cos)*min(1.0,mag)
            local.append(score); curve_scores.append(score)
        suff_scores.append(max(local) if local else 0.0)
        ma=1.0
        for alpha,sc in zip(args.intervention_grid,local):
            if sc>=args.min_intervention_fidelity: ma=float(alpha); break
        min_alphas.append(ma)
    active=sum(1 for slot in range(3) if not isinstance(rule.adapters[slot],NoOpAdapter))
    if active==0:
        return {'necessity':0.0,'sufficiency':0.0,'trajectory_fidelity':0.0,'minimal_alpha':1.0,'minimality':0.0,'cme':0.0,'slot_scores':slot_scores}
    necessity=sum(slot_scores)/active; suff=sum(suff_scores)/active
    trajectory=max(curve_scores) if curve_scores else 0.0
    minimal_alpha=sum(min_alphas)/active; minimality=max(0.0,1.0-minimal_alpha)
    cme=max(0.0,necessity)*max(0.0,suff)*max(0.0,trajectory)*max(0.0,minimality)/max(1,active)
    return {'necessity':necessity,'sufficiency':suff,'trajectory_fidelity':trajectory,'minimal_alpha':minimal_alpha,'minimality':minimality,'cme':cme,'slot_scores':slot_scores}



def _all_nonempty_subsets(indices):
    idx=list(indices); out=[]
    for k in range(1,len(idx)+1):
        for comb in itertools.combinations(idx,k): out.append(tuple(comb))
    return out


def _temporarily_ablate(rule, slots):
    saved={}
    for s in slots:
        saved[s]=rule.adapters[s]
        rule.adapters[s]=NoOpAdapter().to(next(rule.parameters()).device)
    return saved


def _restore_ablate(rule, saved):
    for s,a in saved.items(): rule.adapters[s]=a


def distributed_mediation_metrics(rule, teacher, loader, args, device, layer):
    """Measure localization, redundancy, synergy, minimal causal sets and transferable structure.

    For every non-empty subset of active adapter slots we measure the teacher-aligned
    downstream effect of ablating that subset. This distinguishes:
      - localized: one singleton explains the causal effect;
      - redundant: multiple singletons each already explain the joint effect;
      - synergistic: a combination is much stronger than its best singleton.
    """
    x,_=next(iter(loader)); x=x.to(device)
    with torch.no_grad():
        h=teacher.emb(x)+teacher.pos[:,:x.size(1)]
        teacher_delta=None; z=None; u=None
        for i,b in enumerate(teacher.blocks):
            n=b.norm1(h); a,_=b.attn(n,n,n,need_weights=False); u0=h+a; z0=b.norm2(u0)
            if i==layer:
                hf=u0+b.ff(z0); hz=u0
                for j in range(layer+1,len(teacher.blocks)):
                    hf=teacher.blocks[j](hf); hz=teacher.blocks[j](hz)
                teacher_delta=teacher.head(hf[:,0])-teacher.head(hz[:,0]); z=z0; u=u0; break
            h=u0+b.ff(z0)
    if teacher_delta is None:
        return {'necessity':0.0,'sufficiency':0.0,'trajectory_fidelity':0.0,'minimal_alpha':1.0,'minimality':0.0,'cme':0.0,
                'causal_concentration':0.0,'redundancy_index':0.0,'synergy_index':0.0,'joint_necessity':0.0,
                'joint_sufficiency':0.0,'minimal_set_size':0,'slot_scores':[0.0,0.0,0.0],'set_scores':{}}

    def downstream(rule_out):
        hh=u+rule_out
        for j in range(layer+1,len(teacher.blocks)): hh=teacher.blocks[j](hh)
        return teacher.head(hh[:,0])

    active=[s for s in range(3) if not isinstance(rule.adapters[s],NoOpAdapter)]
    if not active:
        return {'necessity':0.0,'sufficiency':0.0,'trajectory_fidelity':0.0,'minimal_alpha':1.0,'minimality':0.0,'cme':0.0,
                'causal_concentration':0.0,'redundancy_index':0.0,'synergy_index':0.0,'joint_necessity':0.0,
                'joint_sufficiency':0.0,'minimal_set_size':0,'slot_scores':[0.0,0.0,0.0],'set_scores':{}}

    with torch.no_grad():
        full_logits=downstream(rule(z))
    full_norm=float(full_logits.norm(dim=-1).mean().item())+1e-8
    td_norm=float(teacher_delta.norm(dim=-1).mean().item())+1e-8

    set_scores={}; set_nec={}; curves={};
    for subset in _all_nonempty_subsets(active):
        saved=_temporarily_ablate(rule,subset)
        with torch.no_grad():
            ablated_logits=downstream(rule(z))
        _restore_ablate(rule,saved)
        delta=full_logits-ablated_logits
        cos=float(nn.functional.cosine_similarity(delta,teacher_delta,dim=-1).mean().item())
        mag=float(delta.norm(dim=-1).mean().item()/td_norm)
        fidelity=max(0.0,min(1.0,cos))*min(1.0,mag)
        nec=min(1.0,float(delta.norm(dim=-1).mean().item()/full_norm))
        set_scores[','.join(map(str,subset))]=fidelity
        set_nec[','.join(map(str,subset))]=nec
        curves[','.join(map(str,subset))]=[fidelity]

    singles=[(s,set_scores.get(str(s),0.0)) for s in active]
    best_single=max((v for _,v in singles),default=0.0)
    all_key=','.join(map(str,active)); joint_score=set_scores.get(all_key,0.0); joint_nec=set_nec.get(all_key,0.0)
    best_set=max(set_scores.items(),key=lambda kv:kv[1]) if set_scores else ('',0.0)
    qualifying=[(k,v) for k,v in set_scores.items() if v>=args.min_intervention_fidelity]
    if qualifying:
        min_key,min_score=min(qualifying,key=lambda kv:(len(kv[0].split(',')), -kv[1]))
        msize=len(min_key.split(',')); minimality=max(0.0,1.0-(msize-1)/max(len(active),1))
    else:
        min_key=''; min_score=0.0; msize=0; minimality=0.0

    concentration=(best_single/max(joint_score,1e-8)) if joint_score>0 else 0.0
    concentration=min(1.0,max(0.0,concentration))
    redundancy_index=concentration
    synergy_index=max(0.0,joint_score-best_single)
    trajectory_fidelity=joint_score
    sufficiency=best_set[1] if best_set else 0.0
    necessity=joint_nec
    # CME rewards causal necessity, sufficient recovery, teacher trajectory fidelity and compact sets.
    cme=max(0.0,necessity)*max(0.0,sufficiency)*max(0.0,trajectory_fidelity)*max(0.0,minimality)
    slot_scores=[set_scores.get(str(s),0.0) for s in range(3)]
    return {
        'necessity':necessity,'sufficiency':sufficiency,'trajectory_fidelity':trajectory_fidelity,
        'minimal_alpha':1.0 if msize<=0 else (0.5 if msize==1 and len(active)>1 else 0.75 if msize==2 and len(active)>2 else 0.5),
        'minimality':minimality,'cme':cme,'causal_concentration':concentration,
        'redundancy_index':redundancy_index,'synergy_index':synergy_index,
        'joint_necessity':joint_nec,'joint_sufficiency':joint_score,'minimal_set_size':msize,
        'slot_scores':slot_scores,'set_scores':set_scores,'minimal_set':min_key
    }


def cross_task_causal_overlap(metrics):
    vecs=[torch.tensor(m['slot_scores'],dtype=torch.float32) for m in metrics]
    if len(vecs)<2: return 0.0
    vals=[]
    for i in range(len(vecs)):
        for j in range(i+1,len(vecs)):
            vals.append(float(nn.functional.cosine_similarity(vecs[i].view(1,-1),vecs[j].view(1,-1)).item()))
    return sum(vals)/len(vals) if vals else 0.0


def random_set_control(rule, teacher, loader, args, device, layer, active_count):
    active=[s for s in range(3) if not isinstance(rule.adapters[s],NoOpAdapter)]
    if active_count<=0: return 0.0
    candidates=[tuple(c) for c in itertools.combinations(range(3),active_count) if tuple(c)!=tuple(active)]
    if not candidates: return 0.0
    pick=random.choice(candidates)
    x,_=next(iter(loader)); x=x.to(device)
    # Build the same causal effect score used by distributed mediation for the random set.
    with torch.no_grad():
        h=teacher.emb(x)+teacher.pos[:,:x.size(1)]
        for i,b in enumerate(teacher.blocks):
            n=b.norm1(h); a,_=b.attn(n,n,n,need_weights=False); u0=h+a; z0=b.norm2(u0)
            if i==layer:
                hf=u0+b.ff(z0); hz=u0
                for j in range(layer+1,len(teacher.blocks)):
                    hf=teacher.blocks[j](hf); hz=teacher.blocks[j](hz)
                td=teacher.head(hf[:,0])-teacher.head(hz[:,0]); z=z0; u=u0; break
            h=u0+b.ff(z0)
        full=downstream_fn=teacher.head(hf[:,0])
        saved=_temporarily_ablate(rule,pick)
        rr=rule(z)
        _restore_ablate(rule,saved)
        hh=u+rr
        for j in range(layer+1,len(teacher.blocks)): hh=teacher.blocks[j](hh)
        logits=teacher.head(hh[:,0]); full_d=logits
        saved=_temporarily_ablate(rule,active)
        rr0=rule(z)
        _restore_ablate(rule,saved)
        hh0=u+rr0
        for j in range(layer+1,len(teacher.blocks)): hh0=teacher.blocks[j](hh0)
        zero=teacher.head(hh0[:,0])
    delta=full_d-zero
    cos=float(nn.functional.cosine_similarity(delta,td,dim=-1).mean().item())
    mag=float(delta.norm(dim=-1).mean().item()/(td.norm(dim=-1).mean().item()+1e-8))
    return max(0.0,min(1.0,cos))*min(1.0,mag)



class PhiAdapter(nn.Module):
    """Frozen primitive adapter reparameterized by only two task scalars.

    effective_raw = base_raw * (1 + phi_scale) + phi_shift.
    The primitive mode and base_raw are frozen; only phi_scale/phi_shift train.
    """
    def __init__(self, mode, base_raw):
        super().__init__()
        self.mode = mode
        self.register_buffer("base_raw", base_raw.detach().clone())
        self.phi = nn.Parameter(torch.zeros(2, dtype=base_raw.dtype))

    def _raw(self):
        return self.base_raw * (1.0 + self.phi[0]) + self.phi[1]

    def forward(self, x, ref=None):
        raw=self._raw()
        if self.mode=='identity': return x
        if self.mode=='scale': return (1+raw[0])*x + raw[1]
        if self.mode=='negate': return -(1+raw[0])*x + raw[1]
        if self.mode=='difference': return x - (1+raw[0])*(ref if ref is not None else torch.roll(x,1,-1)) + raw[1]
        if self.mode=='product': return x*(1+torch.tanh(raw[0])*(ref if ref is not None else x)) + raw[1]
        if self.mode=='mix':
            a=torch.sigmoid(raw[0]); r=ref if ref is not None else torch.zeros_like(x)
            return a*x+(1-a)*r+raw[1]
        raise ValueError(self.mode)

class FrozenPrimitiveRule(nn.Module):
    """Shared skeleton + frozen primitive base state + tiny task reparameterization."""
    def __init__(self, skeleton, mode, base_states, mask):
        super().__init__()
        self.skeleton=skeleton
        self.mode=mode
        self.mask=tuple(mask)
        self.adapters=nn.ModuleList()
        for i in range(3):
            if not self.mask[i]:
                self.adapters.append(NoOpAdapter())
            else:
                self.adapters.append(PhiAdapter(mode, base_states[i]))
        for p in self.skeleton.parameters():
            p.requires_grad=False

    def forward(self,x):
        h=x
        refs=[]
        for i,node in enumerate(self.skeleton.nodes):
            if self.skeleton.motif=='sequential':
                h=h+node(h)
            else:
                h=h+node(h)
            refs.append(h)
            ref=refs[-2] if i>0 else None
            h=self.adapters[i](h,ref)
        return h

    def phi_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]


def averaged_base_states(rules, mask):
    out=[None,None,None]
    for s in range(3):
        if not mask[s]:
            continue
        states=[r.adapters[s].state_dict()['raw'].detach().cpu() for r in rules]
        out[s]=sum(states)/len(states)
    return out


def make_frozen_rule(skeleton, mode, base_states, mask, device):
    return FrozenPrimitiveRule(skeleton, mode, base_states, mask).to(device)


def fit_phi(rule, teacher_bundle, args, device, steps):
    tea,tr,_=teacher_bundle
    z,y=source_bundle(tea,tr,args,device)
    ps=rule.phi_parameters()
    if not ps:
        return rule
    opt=torch.optim.AdamW(ps,lr=args.phi_lr,weight_decay=args.phi_l2)
    z=z.to(device); y=y.to(device); rule.train()
    for _ in range(steps):
        idx=torch.randperm(len(z),device=device)[:min(args.fit_batch_samples,len(z))]
        pred=rule(z[idx]); loss=((pred-y[idx])**2).mean()+args.phi_l2*sum(p.square().mean() for p in ps)
        opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(ps,0.5); opt.step()
    return rule


def phi_norm(rule):
    vals=[]
    for a in rule.adapters:
        if isinstance(a,PhiAdapter): vals.append(float(a.phi.detach().norm().item()))
    return sum(vals)/len(vals) if vals else 0.0


def primitive_overlap(rules, mask):
    vals=[]
    for s in range(3):
        if not mask[s]: continue
        vec=[r.adapters[s].state_dict()['raw'].detach().float().view(-1) for r in rules]
        for i in range(len(vec)):
            for j in range(i+1,len(vec)):
                vals.append(float(nn.functional.cosine_similarity(vec[i].view(1,-1),vec[j].view(1,-1)).item()))
    return sum(vals)/len(vals) if vals else 0.0


def fit_full_target(rule_skeleton, nodes, motif, mode, mask, teacher_bundle, args, device):
    tea,tr,_=teacher_bundle
    z,y=source_bundle(tea,tr,args,device)
    rule=InterleavedRule(nodes,motif,mode,args.d_model,args.rank,mask).to(device)
    rule.skeleton=rule_skeleton
    return fit_adapter(rule_skeleton,mode,mask,z,y,args,device,args.target_adapter_fit_steps)


def evaluate_frozen_transfer(teacher_bundle, best, base_states, args, device, control_primitive=None):
    tea,tr,va=teacher_bundle
    test=DataLoader(TaskDataset(args.test_size,best['target_task'],best['test_seed']),batch_size=args.batch_size,shuffle=False)

    # zero-shot frozen primitive
    zero=make_frozen_rule(best['skeleton'],best['operator_mode'],base_states,best['placement'],device)
    zero_model=RoutingCompiled(tea,zero,args.trajectory_start,args.trajectory_end).to(device)
    zero_acc=evaluate(zero_model,test,device)

    # tiny reparameterization: only two phi scalars per active slot
    tiny=make_frozen_rule(best['skeleton'],best['operator_mode'],base_states,best['placement'],device)
    tiny=fit_phi(tiny,teacher_bundle,args,device,args.target_phi_fit_steps)
    tiny_model=RoutingCompiled(tea,tiny,args.trajectory_start,args.trajectory_end).to(device)
    tiny_acc=evaluate(tiny_model,test,device)

    # full adapter control on exactly the same frozen skeleton
    full=InterleavedRule(best['nodes'],best['motif'],best['operator_mode'],args.d_model,args.rank,best['placement']).to(device)
    full.skeleton=best['skeleton']
    full=fit_full_target(best['skeleton'],best['nodes'],best['motif'],best['operator_mode'],best['placement'],teacher_bundle,args,device)
    full_model=RoutingCompiled(tea,full,args.trajectory_start,args.trajectory_end).to(device)
    full_acc=evaluate(full_model,test,device)

    # primitive permutation control: use the first source primitive's raw state instead of the cross-task mean
    perm_base=[None,None,None]
    for s in range(3):
        if best['placement'][s]:
            perm_base[s]=best['rules'][0].adapters[s].state_dict()['raw'].detach().cpu()
    perm=make_frozen_rule(best['skeleton'],best['operator_mode'],perm_base,best['placement'],device)
    perm=fit_phi(perm,teacher_bundle,args,device,args.target_phi_fit_steps)
    perm_model=RoutingCompiled(tea,perm,args.trajectory_start,args.trajectory_end).to(device)
    perm_acc=evaluate(perm_model,test,device)

    # random primitive: same architecture, random frozen skeleton + zero/fit phi
    rs=SharedSkeleton(best['nodes'],best['motif'],args.d_model,args.rank).to(device)
    # Keep the random primitive state list fixed to the three adapter slots.
    # The previous version initialized this as [] and then assigned rand_base[s],
    # which raised IndexError during frozen-transfer evaluation.
    rand_base=[None,None,None]
    for s in range(3):
        rand_base[s]=(best['rules'][0].adapters[s].state_dict()['raw'].detach().cpu() if best['placement'][s] else None)
    rand=make_frozen_rule(rs,best['operator_mode'],rand_base,best['placement'],device)
    rand=fit_phi(rand,teacher_bundle,args,device,args.target_phi_fit_steps)
    rand_model=RoutingCompiled(tea,rand,args.trajectory_start,args.trajectory_end).to(device)
    rand_acc=evaluate(rand_model,test,device)

    # MLP control
    mlp=MLPControl(tea,args.trajectory_start,args.trajectory_end,args.mlp_width).to(device)
    train(mlp,tr,device,args.transfer_control_steps,args.lr)
    mlp_acc=evaluate(mlp,test,device)
    teacher_acc=evaluate(tea,test,device)

    return {
        'teacher':teacher_acc,
        'dart_zero':zero_acc,
        'dart_tiny':tiny_acc,
        'dart_full_reparam':full_acc,
        'primitive_permutation_control':perm_acc,
        'random_primitive_control':rand_acc,
        'mlp_control':mlp_acc,
        'gain_zero':(zero_acc-teacher_acc)*100,
        'gain_tiny':(tiny_acc-teacher_acc)*100,
        'gain_full_reparam':(full_acc-teacher_acc)*100,
        'vs_mlp_tiny':(tiny_acc-mlp_acc)*100,
        'phi_norm':phi_norm(tiny),
        'tiny_phi_params':sum(p.numel() for p in tiny.phi_parameters()),
        'base_primitive_params':count_params(best['skeleton'])+sum(count_params(a) for a in zero.adapters if isinstance(a,PhiAdapter)),
    }


def run_seed(args,seed,seed_idx=1,total_seeds=1):
    device=torch.device(args.device if args.device=='cpu' or torch.cuda.is_available() else 'cpu'); seed_all(seed)
    tracker=Progress(1,seed_idx,total_seeds)
    meta=[t for t in args.all_tasks if t not in args.holdout_tasks and t not in args.contrast_tasks]
    teachers={}
    for i,t in enumerate(meta):
        teachers[t]=train_teacher(t,args,device,seed*1000+i)
        tracker.show(0.02*(i+1)/max(1,len(meta)),'teacher-training',f'{i+1}/{len(meta)} | {t}')
    contrast=args.contrast_tasks[0]
    c_tea,c_tr,c_va=train_teacher(contrast,args,device,seed+70000)
    tracker.show(0.04,'teacher-training',f'contrast={contrast} ready')

    # Reuse the DART-2.7 structural search as the primitive-discovery stage.
    motifs=['sequential','parallel_sum','residual_parallel']
    node_families=[('affine_polynomial','polynomial','low_rank'),('low_rank','diagonal','polynomial')]
    modes=TaskAdapter.MODES
    placements=[m for m in itertools.product([0,1],repeat=3) if 1<=sum(m)<=args.max_active_adapters]
    total_candidates=len(motifs)*len(node_families)*len(modes)*len(placements)
    candidates=[]; cand_idx=0
    for motif in motifs:
        for nodes in node_families:
            for mode in modes:
                for mask in placements:
                    cand_idx+=1
                    start=0.04+0.70*(cand_idx-1)/max(1,total_candidates); span=0.70/max(1,total_candidates)
                    tracker.show(start,'primitive-search',f'{cand_idx}/{total_candidates} | {motif}/{mode} | mask={mask}')
                    sk,rules,stab,bundles=fit_shared_skeleton(motif,nodes,mode,mask,teachers,args,device,tracker,start,span)
                    acc=[evaluate(RoutingCompiled(teachers[t][0],rules[i],args.trajectory_start,args.trajectory_end).to(device),teachers[t][2],device) for i,t in enumerate(meta)]
                    sep=[]
                    for t in meta:
                        sr=fit_separate(t,teachers[t],motif,nodes,mode,mask,args,device)
                        sep.append(evaluate(RoutingCompiled(teachers[t][0],sr,args.trajectory_start,args.trajectory_end).to(device),teachers[t][2],device))
                    sepavg=sum(sep)/len(sep); parity=sepavg-sum(acc)/len(acc)
                    rmask=placement_random(mask); racc=[]
                    for i,t in enumerate(meta):
                        rr=InterleavedRule(nodes,motif,mode,args.d_model,args.rank,rmask).to(device); rr.skeleton=sk
                        src_active=[s for s in range(3) if not isinstance(rules[i].adapters[s],NoOpAdapter)]
                        dst_active=[s for s in range(3) if not isinstance(rr.adapters[s],NoOpAdapter)]
                        for ss,dd in zip(src_active,dst_active): rr.adapters[dd].load_state_dict(rules[i].adapters[ss].state_dict())
                        racc.append(evaluate(RoutingCompiled(teachers[t][0],rr,args.trajectory_start,args.trajectory_end).to(device),teachers[t][2],device))
                    rgap=sum(acc)/len(acc)-sum(racc)/len(racc)
                    perm=[]
                    for i,t in enumerate(meta):
                        row=[]
                        for j in range(len(meta)):
                            rr=InterleavedRule(nodes,motif,mode,args.d_model,args.rank,mask).to(device); rr.skeleton=sk
                            for s in range(3):
                                if isinstance(rr.adapters[s],NoOpAdapter): continue
                                rr.adapters[s].load_state_dict(rules[j].adapters[s].state_dict())
                            row.append(evaluate(RoutingCompiled(teachers[t][0],rr,args.trajectory_start,args.trajectory_end).to(device),teachers[t][2],device))
                        perm.append(row)
                    off=[perm[i][j] for i in range(len(meta)) for j in range(len(meta)) if i!=j]
                    pg=(sum(perm[i][i] for i in range(len(meta)))/len(meta)-sum(off)/len(off)) if off else 0.0
                    mediation=[]
                    for i,t in enumerate(meta):
                        ds=DataLoader(TaskDataset(min(args.causal_probe_size,args.verifier_size),t,99000+seed*100+i),batch_size=min(64,args.batch_size),shuffle=False)
                        mediation.append(distributed_mediation_metrics(rules[i],teachers[t][0],ds,args,device,args.trajectory_start))
                    med_need=sum(m['necessity'] for m in mediation)/len(mediation); med_suff=sum(m['sufficiency'] for m in mediation)/len(mediation); traj=sum(m['trajectory_fidelity'] for m in mediation)/len(mediation); minity=sum(m['minimality'] for m in mediation)/len(mediation); cme=sum(m['cme'] for m in mediation)/len(mediation); conc=sum(m['causal_concentration'] for m in mediation)/len(mediation); cross=cross_task_causal_overlap(mediation)
                    random_gap=0.0
                    for i,t in enumerate(meta):
                        ds=DataLoader(TaskDataset(min(args.causal_probe_size,args.verifier_size),t,99500+seed*100+i),batch_size=min(64,args.batch_size),shuffle=False)
                        rand=random_set_control(rules[i],teachers[t][0],ds,args,device,args.trajectory_start,sum(mask))
                        bestset=max(mediation[i]['set_scores'].values()) if mediation[i]['set_scores'] else 0.0
                        random_gap += bestset-rand
                    random_gap/=len(meta)
                    active_k=sum(mask); budget_penalty=args.budget_lambda*active_k
                    eligible=(sum(acc)/len(acc)>=args.min_avg_source_acc and min(acc)>=args.min_worst_source_acc and stab<=args.max_adapter_stability and pg>=args.min_operator_specificity and rgap>=args.min_random_placement_gap and parity<=args.max_shared_vs_separate_gap and med_need>=args.min_mediation_necessity and med_suff>=args.min_mediation_sufficiency and traj>=args.min_intervention_fidelity and cme>=args.min_cme and cross>=args.min_cross_task_overlap and random_gap>=args.min_random_set_gap and sum(m['minimal_set_size'] for m in mediation)/len(mediation)<=args.max_minimal_set_size)
                    score=sum(acc)/len(acc)+args.rule_weight*med_need+args.operator_weight*pg+args.random_weight*rgap+args.placement_weight*rgap+args.mediation_weight*med_suff+args.cme_weight*cme+args.localization_weight*conc+args.cross_task_weight*cross+args.random_set_weight*random_gap-args.parity_weight*max(0,parity)-args.budget_weight*budget_penalty-args.complexity_lambda*count_params(sk)
                    candidates.append({'motif':motif,'nodes':nodes,'operator_mode':mode,'placement':mask,'random_placement':rmask,'source_avg':sum(acc)/len(acc),'source_worst':min(acc),'adapter_stability':stab,'adapter_effect':0.0,'rule_causal_fidelity':0.0,'operator_specificity':pg,'random_placement_gap':rgap,'separate_graph_avg':sepavg,'shared_vs_separate_gap':parity,'placement_gain':rgap,'mediation_necessity':med_need,'mediation_sufficiency':med_suff,'trajectory_fidelity':traj,'minimality':minity,'cme':cme,'causal_concentration':conc,'cross_task_causal_overlap':cross,'random_causal_set_gap':random_gap,'minimal_causal_set_size':sum(m['minimal_set_size'] for m in mediation)/len(mediation),'active_adapters':active_k,'budget_penalty':budget_penalty,'eligible':eligible,'score':score,'rules':rules,'skeleton':sk})
                    tracker.show(start+span*0.95,'primitive-controls',f'{cand_idx}/{total_candidates} | conc={conc:.3f} cme={cme:.3f}')

    elig=[c for c in candidates if c['eligible']]
    best=max(elig,key=lambda c:c['score']) if elig else max(candidates,key=lambda c:c['score'])
    base_states=averaged_base_states(best['rules'],best['placement'])
    # primitive overlap is based on source adapter signatures for the selected candidate.
    overlap=primitive_overlap(best['rules'],best['placement'])

    target=args.holdout_tasks[0]
    tea,tr,va=train_teacher(target,args,device,seed+50000)
    test_seed=seed+60000
    base_common=dict(best); base_common['target_task']=target; base_common['test_seed']=test_seed
    transfer=evaluate_frozen_transfer((tea,tr,va),base_common,base_states,args,device)
    tracker.show(0.98,'frozen-transfer',f'target={target} contrast={contrast} phi={transfer["phi_norm"]:.3f}')
    # contrast on the exact same frozen primitive
    contrast_eval=evaluate_frozen_transfer((c_tea,c_tr,c_va),{**base_common,'target_task':contrast,'test_seed':seed+60001},base_states,args,device)
    tracker.done('complete')
    clean_best={k:v for k,v in best.items() if k not in ('rules','skeleton')}
    clean_best['primitive_overlap']=overlap
    return {'seed':seed,'winner':clean_best,'related_holdout':transfer,'contrast_holdout':contrast_eval}


def main():
    p=argparse.ArgumentParser(description='DART-2.8 frozen causal primitive reparameterization + strict transfer')
    p.add_argument('--seeds',nargs='+',type=int,default=[1,2]); p.add_argument('--all-tasks',nargs='+',default=['add','compose','mul','sub']); p.add_argument('--holdout-tasks',nargs='+',default=['sub']); p.add_argument('--contrast-tasks',nargs='+',default=['sort'])
    ints={'teacher-steps':800,'core-fit-steps':300,'adapter-fit-steps':120,'target-adapter-fit-steps':400,'target-phi-fit-steps':400,'transfer-control-steps':400,'separate-control-steps':200,'train-size':6000,'verifier-size':1500,'test-size':1500,'rel-samples-per-task':2048,'fit-batch-samples':512,'d-model':32,'heads':2,'d-ff':128,'depth':3,'rank':8,'batch-size':256,'mlp-width':64,'max-active-adapters':2,'causal-probe-size':64}
    for k,v in ints.items(): p.add_argument('--'+k,type=int,default=v)
    floats={'adapter-l2':0.0005,'adapter-lr':0.01,'phi-l2':0.0005,'phi-lr':0.02,'min-avg-source-acc':.30,'min-worst-source-acc':.22,'max-adapter-stability':.75,'min-operator-specificity':.01,'min-random-placement-gap':.02,'max-shared-vs-separate-gap':.03,'rule-weight':.30,'operator-weight':.20,'random-weight':.20,'placement-weight':.15,'parity-weight':.20,'min-mediation-necessity':.10,'min-mediation-sufficiency':.20,'min-intervention-fidelity':.20,'min-cme':.01,'mediation-weight':.25,'cme-weight':.35,'budget-weight':1.0,'budget-lambda':.03,'complexity-lambda':1e-5,'lr':.0003,'core-fit-lr':.001,'localization-weight':.20,'cross-task-weight':.20,'random-set-weight':.15,'min-cross-task-overlap':.30,'min-random-set-gap':.02}
    for k,v in floats.items(): p.add_argument('--'+k,type=float,default=v)
    p.add_argument('--max-minimal-set-size',type=int,default=2); p.add_argument('--trajectory-start',type=int,default=0); p.add_argument('--trajectory-end',type=int,default=1); p.add_argument('--intervention-grid',nargs='+',type=float,default=[0.0,0.25,0.5,0.75,1.0]); p.add_argument('--device',default='cuda'); p.add_argument('--out',default='dart028_results.json')
    args=p.parse_args()
    rec=[run_seed(args,s,i+1,len(args.seeds)) for i,s in enumerate(args.seeds)]
    def avg(section,k): return sum(r[section][k] for r in rec)/len(rec)
    summary={'related_holdout':{args.holdout_tasks[0]:{k:avg('related_holdout',k) for k in ['teacher','dart_zero','dart_tiny','dart_full_reparam','primitive_permutation_control','random_primitive_control','mlp_control','gain_zero','gain_tiny','gain_full_reparam','vs_mlp_tiny']}},
              'contrast_holdout':{args.contrast_tasks[0]:{k:avg('contrast_holdout',k) for k in ['teacher','dart_zero','dart_tiny','dart_full_reparam','primitive_permutation_control','random_primitive_control','mlp_control','gain_zero','gain_tiny','gain_full_reparam','vs_mlp_tiny']}},
              'source':{
                'avg_accuracy':sum(r['winner']['source_avg'] for r in rec)/len(rec),
                'avg_source_worst':sum(r['winner']['source_worst'] for r in rec)/len(rec),
                'avg_primitive_overlap':sum(r['winner']['primitive_overlap'] for r in rec)/len(rec),
                'avg_shared_vs_separate_gap':sum(r['winner']['shared_vs_separate_gap'] for r in rec)/len(rec),
                'avg_causal_concentration':sum(r['winner'].get('causal_concentration',0.0) for r in rec)/len(rec),
                'avg_minimal_causal_set_size':sum(r['winner'].get('minimal_causal_set_size',0.0) for r in rec)/len(rec),
                'avg_cross_task_causal_overlap':sum(r['winner'].get('cross_task_causal_overlap',0.0) for r in rec)/len(rec),
                'avg_cme':sum(r['winner'].get('cme',0.0) for r in rec)/len(rec),
                'avg_base_primitive_params':sum(r['related_holdout']['base_primitive_params'] for r in rec)/len(rec),
                'avg_tiny_phi_params':sum(r['related_holdout']['tiny_phi_params'] for r in rec)/len(rec),
                'avg_phi_norm':sum(r['related_holdout']['phi_norm'] for r in rec)/len(rec)}}
    payload={'config':vars(args),'records':rec,'summary':summary}
    Path(args.out).write_text(json.dumps(payload,indent=2)); print('DART-2.8: frozen causal primitive reparameterization + strict transfer'); print(json.dumps(summary,indent=2)); print('Saved:',Path(args.out).resolve())

if __name__=='__main__': main()
