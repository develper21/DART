#!/usr/bin/env python3
"""DART-2.3: interleaved shared/task factorization with sparse causal adapters.

Hypothesis:
  DART-2.2 showed that a shared graph + one task operator could not match
  separately specialized graphs. DART-2.3 tests whether task-specific
  computation is distributed at a few internal, causally necessary sites
  inside an otherwise shared computational skeleton.

Architecture:
  x -> G1 -> A_t,1 -> G2 -> A_t,2 -> G3 -> A_t,3 -> y
where G_i are shared structured nodes and A_t,i are tiny structured task
operators. A placement mask determines which adapters are active. The number
of active task adapters is capped, and random-placement controls are reported.

No task embeddings, large conditioners, target-task residual networks, or
unbounded graph surgery are used.
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

def adapter_state(op): return {k:v.detach().clone() for k,v in op.state_dict().items()}

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

def fit_shared_skeleton(motif,nodes,mode,mask,teachers,args,device):
    sk=SharedSkeleton(nodes,motif,args.d_model,args.rank).to(device)
    tasks=list(teachers); bundles={t:source_bundle(teachers[t][0],teachers[t][1],args,device) for t in tasks}
    rules=nn.ModuleList([InterleavedRule(nodes,motif,mode,args.d_model,args.rank,mask) for _ in tasks]).to(device)
    # share one skeleton during joint fitting
    for r in rules: r.skeleton=sk
    params=list(sk.parameters())+[p for r in rules for p in r.adapters.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(params,lr=args.core_fit_lr,weight_decay=1e-4)
    for _ in range(args.core_fit_steps):
        loss_total=0.
        for i,t in enumerate(tasks):
            z,y=bundles[t]; idx=torch.randperm(len(z))[:min(args.fit_batch_samples,len(z))]; pred=rules[i](z[idx].to(device)); yy=y[idx].to(device)
            loss_total += ((pred-yy)**2).mean()
        opt.zero_grad(set_to_none=True); loss_total.backward(); nn.utils.clip_grad_norm_(params,1.); opt.step()
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

def adapter_effect(rule,z,delta,device):
    base={k:v.clone() for k,v in rule.state_dict().items()}; out=[]
    with torch.no_grad(): y0=rule(z.to(device))
    for slot,ad in enumerate(rule.adapters):
        if isinstance(ad,NoOpAdapter): continue
        for p in ad.parameters():
            p.data.add_(delta); y1=rule(z.to(device)); p.data.sub_(delta); out.append(float((y1-y0).norm().item()/max(len(z),1)))
    rule.load_state_dict(base); return sum(out)/len(out) if out else 0.0

def placement_random(mask,n=3):
    m=list(mask); random.shuffle(m); return tuple(m)

def run_seed(args,seed):
    device=torch.device(args.device if args.device=='cpu' or torch.cuda.is_available() else 'cpu'); seed_all(seed)
    meta=[t for t in args.all_tasks if t not in args.holdout_tasks and t not in args.contrast_tasks]
    teachers={t:train_teacher(t,args,device,seed*1000+i) for i,t in enumerate(meta)}
    contrast=args.contrast_tasks[0]; c_tea,c_tr,c_va=train_teacher(contrast,args,device,seed+70000)
    motifs=['sequential','parallel_sum','residual_parallel']; node_families=[('affine_polynomial','polynomial','low_rank'),('low_rank','diagonal','polynomial')]
    modes=TaskAdapter.MODES; placements=[m for m in itertools.product([0,1], repeat=3) if 1<=sum(m)<=args.max_active_adapters]
    candidates=[]
    for motif in motifs:
        for nodes in node_families:
            for mode in modes:
                for mask in placements:
                    sk, rules, stab, bundles=fit_shared_skeleton(motif,nodes,mode,mask,teachers,args,device)
                    acc=[]
                    for i,t in enumerate(meta): acc.append(evaluate(RoutingCompiled(teachers[t][0],rules[i],args.trajectory_start,args.trajectory_end).to(device),teachers[t][2],device))
                    sep=[]
                    for t in meta:
                        sr=fit_separate(t,teachers[t],motif,nodes,mode,mask,args,device); sep.append(evaluate(RoutingCompiled(teachers[t][0],sr,args.trajectory_start,args.trajectory_end).to(device),teachers[t][2],device))
                    sepavg=sum(sep)/len(sep); parity=sepavg-sum(acc)/len(acc)
                    # random placement, same shared skeleton and adapter magnitudes
                    rmask=placement_random(mask); rand_rules=[]
                    for r in rules:
                        rr=InterleavedRule(nodes,motif,mode,args.d_model,args.rank,rmask).to(device); rr.skeleton=sk
                        for a_src,a_dst in zip(r.adapters,rr.adapters):
                            if isinstance(a_src,NoOpAdapter) or isinstance(a_dst,NoOpAdapter): continue
                            a_dst.load_state_dict(a_src.state_dict())
                        rand_rules.append(rr)
                    racc=[evaluate(RoutingCompiled(teachers[t][0],rand_rules[i],args.trajectory_start,args.trajectory_end).to(device),teachers[t][2],device) for i,t in enumerate(meta)]
                    rgap=sum(acc)/len(acc)-sum(racc)/len(racc)
                    perm=[]
                    for i,t in enumerate(meta):
                        row=[]
                        for j in range(len(meta)):
                            rr=InterleavedRule(nodes,motif,mode,args.d_model,args.rank,mask).to(device); rr.skeleton=sk
                            for a_dst,a_src in zip(rr.adapters,rules[j].adapters):
                                if isinstance(a_dst,NoOpAdapter): continue
                                a_dst.load_state_dict(a_src.state_dict())
                            row.append(evaluate(RoutingCompiled(teachers[t][0],rr,args.trajectory_start,args.trajectory_end).to(device),teachers[t][2],device))
                        perm.append(row)
                    pg=sum(perm[i][i] for i in range(len(meta)))/len(meta)-sum(perm[i][j] for i in range(len(meta)) for j in range(len(meta)) if i!=j)/(len(meta)*(len(meta)-1))
                    rcf=rule_causal_fidelity(rules[0],teachers[meta[0]][0],teachers[meta[0]][1],args,device,args.trajectory_start)
                    op_eff=sum(adapter_effect(r,z,args.adapter_delta,device) for r,(z,_) in zip(rules,bundles.values()))/len(rules)
                    placement_gain=(sum(acc)/len(acc))-(sum(racc)/len(racc))
                    eligible=(sum(acc)/len(acc)>=args.min_avg_source_acc and min(acc)>=args.min_worst_source_acc and stab<=args.max_adapter_stability and rcf>=args.min_rule_causal_fidelity and pg>=args.min_operator_specificity and rgap>=args.min_random_placement_gap and parity<=args.max_shared_vs_separate_gap and op_eff>=args.min_adapter_effect)
                    score=sum(acc)/len(acc)+args.rule_weight*rcf+args.operator_weight*pg+args.random_weight*rgap+args.placement_weight*placement_gain-args.parity_weight*max(0,parity)-args.complexity_lambda*(count_params(sk)+sum(count_params(a) for r in rules for a in r.adapters if not isinstance(a,NoOpAdapter)))
                    candidates.append({'motif':motif,'nodes':nodes,'operator_mode':mode,'placement':mask,'random_placement':rmask,'source_avg':sum(acc)/len(acc),'source_worst':min(acc),'adapter_stability':stab,'adapter_effect':op_eff,'rule_causal_fidelity':rcf,'operator_specificity':pg,'random_placement_gap':rgap,'separate_graph_avg':sepavg,'shared_vs_separate_gap':parity,'placement_gain':placement_gain,'task_acc':acc,'operator_permutation_matrix':perm,'eligible':eligible,'score':score,'rules':rules,'skeleton':sk})
    elig=[c for c in candidates if c['eligible']]; best=max(elig,key=lambda c:c['score']) if elig else max(candidates,key=lambda c:c['score'])
    results={}
    for label,t in [('related',args.holdout_tasks[0]),('contrast',contrast)]:
        tea,tr,va=(train_teacher(t,args,device,seed+50000) if label=='related' else (c_tea,c_tr,c_va)); te=DataLoader(TaskDataset(args.test_size,t,seed+60000+(0 if label=='related' else 1)),batch_size=args.batch_size,shuffle=False)
        # Freeze shared skeleton; average source adapter state as a zero-shot prior.
        rule=InterleavedRule(best['nodes'],best['motif'],best['operator_mode'],args.d_model,args.rank,best['placement']).to(device); rule.skeleton=best['skeleton']
        # average active adapters
        for slot in range(3):
            if isinstance(rule.adapters[slot],NoOpAdapter): continue
            states=[r.adapters[slot].state_dict() for r in best['rules']]
            avg={k:sum(s[k] for s in states)/len(states) for k in states[0]}; rule.adapters[slot].load_state_dict(avg)
        zero=evaluate(RoutingCompiled(tea,rule,args.trajectory_start,args.trajectory_end).to(device),te,device)
        z,y=source_bundle(tea,tr,args,device); fitted=fit_adapter(best['skeleton'],best['operator_mode'],best['placement'],z,y,args,device,args.target_adapter_fit_steps)
        adapted=evaluate(RoutingCompiled(tea,fitted,args.trajectory_start,args.trajectory_end).to(device),te,device)
        perm=fitted
        # wrong-operator/placement control: randomly move active adapters while preserving states
        wrong_mask=placement_random(best['placement']); wrong=InterleavedRule(best['nodes'],best['motif'],best['operator_mode'],args.d_model,args.rank,wrong_mask).to(device); wrong.skeleton=best['skeleton']
        srcs=[fitted.adapters[s].state_dict() for s in range(3) if not isinstance(fitted.adapters[s],NoOpAdapter)]
        k=0
        for s in range(3):
            if isinstance(wrong.adapters[s],NoOpAdapter): continue
            wrong.adapters[s].load_state_dict(srcs[k]); k+=1
        pacc=evaluate(RoutingCompiled(tea,wrong,args.trajectory_start,args.trajectory_end).to(device),te,device)
        mlp=MLPControl(tea,args.trajectory_start,args.trajectory_end,args.mlp_width).to(device); train(mlp,tr,device,args.transfer_control_steps,args.lr); am=evaluate(mlp,te,device)
        results[label+'_holdout']={'task':t,'teacher':evaluate(tea,te,device),'dart_zero':zero,'dart_adapted':adapted,'placement_control':pacc,'mlp_control':am,'gain_zero':(zero-evaluate(tea,te,device))*100,'gain_adapt':(adapted-evaluate(tea,te,device))*100,'vs_mlp_adapt':(adapted-am)*100,'adapter_states':{str(i):({k:v.cpu().tolist() for k,v in fitted.adapters[i].state_dict().items()} if not isinstance(fitted.adapters[i],NoOpAdapter) else None) for i in range(3)}}
    clean=lambda c:{k:v for k,v in c.items() if k not in ('rules','skeleton')}
    return {'seed':seed,'winner':clean(best),'candidates':[clean(c) for c in candidates],**results}

def main():
    p=argparse.ArgumentParser(description='DART-2.3 interleaved shared/task factorization')
    p.add_argument('--seeds',nargs='+',type=int,default=[1,2]); p.add_argument('--all-tasks',nargs='+',default=['add','compose','mul','sub']); p.add_argument('--holdout-tasks',nargs='+',default=['sub']); p.add_argument('--contrast-tasks',nargs='+',default=['sort'])
    ints={'teacher-steps':800,'core-fit-steps':300,'adapter-fit-steps':120,'target-adapter-fit-steps':400,'transfer-control-steps':400,'separate-control-steps':200,'train-size':6000,'verifier-size':1500,'test-size':1500,'rel-samples-per-task':2048,'fit-batch-samples':512,'d-model':32,'heads':2,'d-ff':128,'depth':3,'rank':8,'batch-size':256,'mlp-width':64,'max-active-adapters':2}
    for k,v in ints.items(): p.add_argument('--'+k,type=int,default=v)
    floats={'adapter-delta':0.1,'adapter-l2':0.0005,'adapter-lr':0.01,'min-avg-source-acc':.30,'min-worst-source-acc':.22,'max-adapter-stability':.75,'min-operator-specificity':.01,'min-adapter-effect':.02,'min-rule-causal-fidelity':.20,'min-random-placement-gap':.02,'max-shared-vs-separate-gap':.03,'rule-weight':.30,'operator-weight':.20,'random-weight':.20,'placement-weight':.15,'parity-weight':.20,'complexity-lambda':1e-5,'lr':.0003,'core-fit-lr':.001}
    for k,v in floats.items(): p.add_argument('--'+k,type=float,default=v)
    p.add_argument('--trajectory-start',type=int,default=0); p.add_argument('--trajectory-end',type=int,default=1); p.add_argument('--device',default='cuda'); p.add_argument('--out',default='dart023_results.json')
    args=p.parse_args(); rec=[run_seed(args,s) for s in args.seeds]
    def av(k,sec): return sum(r[sec][k] for r in rec)/len(rec)
    summary={'related_holdout':{args.holdout_tasks[0]:{k:av(k,'related_holdout') for k in ['teacher','dart_zero','dart_adapted','placement_control','mlp_control','gain_zero','gain_adapt','vs_mlp_adapt']}},'contrast_holdout':{args.contrast_tasks[0]:{k:av(k,'contrast_holdout') for k in ['teacher','dart_zero','dart_adapted','placement_control','mlp_control','gain_zero','gain_adapt','vs_mlp_adapt']}},'source':{'avg_accuracy':sum(r['winner']['source_avg'] for r in rec)/len(rec),'avg_adapter_stability':sum(r['winner']['adapter_stability'] for r in rec)/len(rec),'avg_adapter_effect':sum(r['winner']['adapter_effect'] for r in rec)/len(rec),'avg_rule_causal_fidelity':sum(r['winner']['rule_causal_fidelity'] for r in rec)/len(rec),'avg_operator_specificity':sum(r['winner']['operator_specificity'] for r in rec)/len(rec),'avg_random_placement_gap':sum(r['winner']['random_placement_gap'] for r in rec)/len(rec),'avg_placement_gain':sum(r['winner']['placement_gain'] for r in rec)/len(rec),'avg_separate_graph_accuracy':sum(r['winner']['separate_graph_avg'] for r in rec)/len(rec),'avg_shared_vs_separate_gap':sum(r['winner']['shared_vs_separate_gap'] for r in rec)/len(rec),'avg_active_adapters':sum(sum(r['winner']['placement']) for r in rec)/len(rec)} }
    payload={'config':vars(args),'records':rec,'summary':summary}; Path(args.out).write_text(json.dumps(payload,indent=2)); print('DART-2.3: interleaved shared/task factorization'); print(json.dumps(summary,indent=2)); print('Saved:',Path(args.out).resolve())
if __name__=='__main__': main()
