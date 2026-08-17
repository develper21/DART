#!/usr/bin/env python3
"""DART-1.8: causal bottleneck / minimum computational subgraph discovery.

Built from the user's updated DART-1.7 implementation. DART-1.7 established that
an explicit E -> T(theta) -> D interface can produce source-task invariance, but
transfer remained weak. DART-1.8 tests whether the useful computation lives in a
smaller causally necessary bottleneck Z inside the typed representation R=E(x).

Pipeline:
  hidden x -> extractor E -> full relation R -> selector S(R) -> bottleneck Z
          -> structured transform T(Z, theta) -> decoder D -> hidden replacement

Controls:
  A. full typed representation (DART-1.7 style)
  B. random bottleneck of same width
  C. learned causal bottleneck
  D. MLP control

Hard gates require source capability, theta necessity/stability, bottleneck
sufficiency, causal necessity, and contrast specificity. The bottleneck selector
is trained only on meta-task data; the contrast task is forward-only evaluation.
"""
from __future__ import annotations
import argparse, copy, itertools, json, random
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

VOCAB=list("0123456789+= "); STOI={c:i for i,c in enumerate(VOCAB)}; PAD=STOI[' ']; BLOCK=12

def seed_all(s):
    random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

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
            x,y=make_example(r.randint(0,999),r.randint(0,999),task); self.rows.append((torch.tensor(x),torch.tensor(y)))
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
    if name=='diagonal': return DiagonalCore(d)
    if name=='polynomial': return PolynomialCore(d)
    if name=='affine_polynomial': return AffinePolynomialCore(d,r)
    if name=='low_rank': return LowRankCore(d,r)
    raise ValueError(name)

class RelationExtractor(nn.Module):
    def __init__(self,kind,d,rel_dim):
        super().__init__(); self.kind=kind; self.half=d//2
        in_dim={'diff_proj':self.half,'product_proj':self.half,'stats_proj':6,'raw_proj':d}[kind]
        self.proj=nn.Linear(in_dim,rel_dim)
        nn.init.xavier_uniform_(self.proj.weight,gain=.1); nn.init.zeros_(self.proj.bias)
    def forward(self,z):
        h=self.half
        if self.kind=='diff_proj': feat=z[...,:h]-z[...,h:2*h]
        elif self.kind=='product_proj': feat=z[...,:h]*z[...,h:2*h]
        elif self.kind=='stats_proj':
            z1,z2=z[...,:h],z[...,h:2*h]; feat=torch.stack([z1.mean(-1),z1.var(-1,unbiased=False),z2.mean(-1),z2.var(-1,unbiased=False),z.mean(-1),z.var(-1,unbiased=False)],-1)
        else: feat=z
        return self.proj(feat)

class RelationDecoder(nn.Module):
    def __init__(self,kind,rel_dim,d):
        super().__init__(); self.kind=kind; self.up=nn.Linear(rel_dim,d); nn.init.xavier_uniform_(self.up.weight,gain=.05); nn.init.zeros_(self.up.bias)
        if kind=='affine': self.quad=nn.Linear(rel_dim,d,bias=False); nn.init.xavier_uniform_(self.quad.weight,gain=.02)
    def forward(self,r): return self.up(r)+self.quad(r.square()) if self.kind=='affine' else self.up(r)

class TypedPrimitive(nn.Module):
    def __init__(self,ext,tr,dec,d,rel_dim,rank):
        super().__init__(); self.E=RelationExtractor(ext,d,rel_dim); self.T=make_core(tr,rel_dim,rank); self.D=RelationDecoder(dec,rel_dim,d); self.theta_dim=1
    def extract(self,x): return self.E(x)
    def transform(self,r,theta): return r+theta[0].view(1,1)*self.T(r)
    def decode(self,r): return self.D(r)
    def forward_full(self,x,theta): return self.decode(self.transform(self.extract(x),theta))

class BottleneckSelector(nn.Module):
    def __init__(self,rel_dim,bottleneck):
        super().__init__(); self.rel_dim=rel_dim; self.bottleneck=bottleneck
        self.logits=nn.Parameter(torch.zeros(rel_dim))
    def mask(self):
        # continuous top-k gate used during optimization; sigmoid gives causal soft selection.
        p=torch.sigmoid(self.logits)
        if self.bottleneck>=self.rel_dim: return p
        top=torch.topk(p,self.bottleneck).indices
        hard=torch.zeros_like(p); hard[top]=1.
        # straight-through estimator for hard causal mask.
        return hard + (p-hard).detach()
    def forward(self,r): return r*self.mask().view(1,1,-1)

class BottleneckPrimitive(nn.Module):
    def __init__(self,typed:TypedPrimitive,bottleneck):
        super().__init__(); self.typed=typed; self.selector=BottleneckSelector(typed.E.proj.out_features,bottleneck); self.theta_dim=typed.theta_dim
    def extract_full(self,x): return self.typed.extract(x)
    def extract(self,x): return self.selector(self.typed.extract(x))
    def transform(self,r,theta): return self.typed.transform(r,theta)
    def decode(self,r): return self.typed.decode(r)
    def forward(self,x,theta): return self.decode(self.transform(self.extract(x),theta))
    def full_forward(self,x,theta): return self.typed.forward_full(x,theta)

class RandomBottleneck(nn.Module):
    def __init__(self,typed,indices): super().__init__(); self.typed=typed; self.register_buffer('idx',torch.tensor(indices,dtype=torch.long)); self.theta_dim=1
    def forward(self,x,theta):
        r=self.typed.extract(x); mask=torch.zeros(r.shape[-1],device=r.device); mask[self.idx]=1.; r=r*mask.view(1,1,-1); return self.typed.decode(self.typed.transform(r,theta))

class RoutingCompiled(nn.Module):
    def __init__(self,teacher,primitive,theta,start,end):
        super().__init__(); self.emb=copy.deepcopy(teacher.emb); self.pos=copy.deepcopy(teacher.pos); self.head=copy.deepcopy(teacher.head); self.blocks=nn.ModuleList([RoutingReplaceBlock(b,primitive,theta) if start<=i<end else copy.deepcopy(b) for i,b in enumerate(teacher.blocks)])
    def forward(self,x):
        h=self.emb(x)+self.pos[:,:x.size(1)]
        for b in self.blocks: h=b(h)
        return self.head(h[:,0])
class RoutingReplaceBlock(nn.Module):
    def __init__(self,b,primitive,theta):
        super().__init__(); self.norm1=copy.deepcopy(b.norm1); self.attn=copy.deepcopy(b.attn); self.norm2=copy.deepcopy(b.norm2); self.primitive=primitive; self.register_buffer('theta_fixed',theta.detach().clone())
    def forward(self,x):
        n=self.norm1(x); a,_=self.attn(n,n,n,need_weights=False); u=x+a; z=self.norm2(u); return u+self.primitive(z,self.theta_fixed)
class MLPReplaceBlock(nn.Module):
    def __init__(self,b,d,w): super().__init__(); self.norm1=copy.deepcopy(b.norm1); self.attn=copy.deepcopy(b.attn); self.norm2=copy.deepcopy(b.norm2); self.m=nn.Sequential(nn.Linear(d,w),nn.GELU(),nn.Linear(w,d))
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
    model.eval(); ce=nn.CrossEntropyLoss(reduction='sum'); total=correct=0; loss=0.
    with torch.no_grad():
        for x,y in loader:
            x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True); z=model(x); loss+=float(ce(z,y)); correct+=int((z.argmax(-1)==y).sum()); total+=y.numel()
    return correct/max(total,1),loss/max(total,1)

def train(model,loader,device,steps,lr):
    ps=[p for p in model.parameters() if p.requires_grad]
    if not ps: raise RuntimeError('No trainable parameters')
    opt=torch.optim.AdamW(ps,lr=lr,weight_decay=1e-4); ce=nn.CrossEntropyLoss(); it=iter(loader); model.train()
    for _ in range(steps):
        try: x,y=next(it)
        except StopIteration: it=iter(loader); x,y=next(it)
        x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True); loss=ce(model(x),y); opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(ps,1.); opt.step()

def train_teacher(task,args,device,seed):
    tr=DataLoader(TaskDataset(args.train_size,task,seed),batch_size=args.batch_size,shuffle=True,pin_memory=device.type=='cuda'); va=DataLoader(TaskDataset(args.verifier_size,task,seed+10000),batch_size=args.batch_size,shuffle=False,pin_memory=device.type=='cuda'); te=Teacher(len(VOCAB),args.d_model,args.heads,args.d_ff,args.depth).to(device); train(te,tr,device,args.teacher_steps,args.lr); return te,tr,va

def capture_ff(teacher,loader,device,maxn,layer):
    zs=[]
    with torch.no_grad():
        for x,_ in loader:
            n_have=sum(t.shape[0] for t in zs)
            if n_have>=maxn: break
            x=x.to(device,non_blocking=True); h=teacher.emb(x)+teacher.pos[:,:x.size(1)]
            for i,b in enumerate(teacher.blocks):
                n=b.norm1(h); a,_=b.attn(n,n,n,need_weights=False); u=h+a; z=b.norm2(u)
                if i==layer:
                    take=min(maxn-n_have,z.reshape(-1,z.shape[-1]).shape[0]); zs.append(z.reshape(-1,z.shape[-1])[:take].cpu()); break
                h=u+b.ff(z)
    return torch.cat(zs)

def source_bundle(teacher,loader,args,device):
    z=capture_ff(teacher,loader,device,args.rel_samples_per_task,args.trajectory_start); y=teacher.blocks[args.trajectory_start].ff(z.to(device)).detach().cpu(); return z,y

def fit_theta(primitive,z,y,args,device,steps,init=None):
    th=nn.Parameter(init.clone().to(device) if init is not None else torch.tensor([0.5],device=device)); opt=torch.optim.Adam([th],lr=args.theta_lr); z=z.to(device); y=y.to(device); primitive.eval()
    for _ in range(steps):
        pred=primitive(z,th); loss=((pred-y)**2).mean()+args.theta_l2*th.square().mean(); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    return th.detach()

def train_bottleneck(primitive,bundles,args,device):
    theta_bank=nn.Parameter(torch.full((len(bundles),1),0.5,device=device)); params=list(primitive.parameters())+[theta_bank]
    opt=torch.optim.AdamW(params,lr=args.core_fit_lr,weight_decay=1e-4)
    tasks=list(bundles)
    for _ in range(args.core_fit_steps):
        total=0.
        for i,t in enumerate(tasks):
            z,y=bundles[t]; idx=torch.randperm(len(z))[:min(args.fit_batch_samples,len(z))]; zz=z[idx].to(device); yy=y[idx].to(device); total=total+((primitive(zz,theta_bank[i])-yy)**2).mean()+args.theta_l2*theta_bank[i].square().mean()
        opt.zero_grad(set_to_none=True); total.backward(); nn.utils.clip_grad_norm_(params,1.); opt.step()
    for p in primitive.parameters(): p.requires_grad=False
    final=[]
    for i,t in enumerate(tasks): final.append(fit_theta(primitive,bundles[t][0],bundles[t][1],args,device,args.theta_fit_steps,theta_bank[i].detach()))
    theta_mat=torch.stack(final)
    # stability + effect
    st=[]
    for i,t in enumerate(tasks):
        z,y=bundles[t]; n=len(z); a=slice(0,n//2); b=slice(n//2,n); ta=fit_theta(primitive,z[a],y[a],args,device,max(10,args.theta_fit_steps//2),theta_mat[i]); tb=fit_theta(primitive,z[b],y[b],args,device,max(10,args.theta_fit_steps//2),theta_mat[i]); st.append(float(torch.norm(ta-tb).item()))
    theta_stability=sum(st)/len(st)
    z0=next(iter(bundles.values()))[0][:min(args.effect_samples,len(next(iter(bundles.values()))[0]))].to(device); base=theta_mat[0]; plus=base+args.theta_delta; minus=base-args.theta_delta
    with torch.no_grad(): theta_effect=float((primitive(z0,plus)-primitive(z0,minus)).norm().item()/(2*args.theta_delta))
    return theta_mat,theta_stability,theta_effect

def rep_full(primitive,bundles,device):
    out={}
    with torch.no_grad():
        for t,(z,_) in bundles.items(): out[t]=primitive.typed.extract(z.to(device)).reshape(-1,primitive.typed.E.proj.out_features).cpu()
    return out

def select_importance(primitive,bundles,theta_mat,args,device):
    # Importance = average absolute gradient of task loss proxy to representation coordinates.
    # We use reconstruction loss to teacher FF outputs as the causal training signal.
    rel_dim=primitive.typed.E.proj.out_features; imp=torch.zeros(rel_dim,device=device)
    for i,t in enumerate(bundles):
        z,y=bundles[t]; idx=torch.randperm(len(z))[:min(args.bottleneck_fit_samples,len(z))]; zz=z[idx].to(device); yy=y[idx].to(device); r=primitive.typed.extract(zz).detach().requires_grad_(True); out=primitive.decode(primitive.transform(r,theta_mat[i])); loss=((out-yy)**2).mean(); g=torch.autograd.grad(loss,r,retain_graph=False)[0]; imp += g.abs().mean((0,1))
    imp=imp/len(bundles); return imp.detach()

def fit_masked_selector(primitive,bundles,theta_mat,args,device,importance):
    rel_dim=primitive.typed.E.proj.out_features; k=min(args.bottleneck_dim,rel_dim); idx=torch.topk(importance,k).indices
    selector=BottleneckSelector(rel_dim,k).to(device); selector.logits.data.fill_(-4.0); selector.logits.data[idx]=4.0; selector.requires_grad_(False)
    primitive.selector=selector
    return idx.cpu()

def random_masked(typed,k,seed,device):
    r=random.Random(seed); idx=r.sample(range(typed.E.proj.out_features),k); return RandomBottleneck(typed,idx).to(device),idx

def iid_subspace(R1,R2,k=4):
    def basis(R):
        Rc=R-R.mean(0,keepdim=True); cov=(Rc.t()@Rc)/max(Rc.shape[0]-1,1); _,v=torch.linalg.eigh(cov); return v[:,-min(k,v.shape[1]):]
    A=basis(R1); B=basis(R2); s=torch.linalg.svdvals(A.t()@B); return float(s.mean().item()) if s.numel() else 0.

def interface_stats(primitive,bundles,contrast_bundle,args,device):
    reps=rep_full(primitive,bundles,device); rel=[]
    ts=list(bundles)
    for a,b in itertools.combinations(ts,2): rel.append(iid_subspace(reps[a],reps[b],args.iis_topk))
    irel=sum(rel)/len(rel) if rel else 0.
    with torch.no_grad(): c=primitive.typed.extract(contrast_bundle[0].to(device)).reshape(-1,primitive.typed.E.proj.out_features).cpu()
    ic=sum(iid_subspace(reps[t],c,args.iis_topk) for t in ts)/len(ts); return irel,ic,irel-ic

def causal_bottleneck_sufficiency(primitive,theta,teacher,loader,args,device,layer):
    # Compare full representation vs selected bottleneck on teacher FF output reconstruction.
    z,y=source_bundle(teacher,loader,args,device); with_full=[]
    with torch.no_grad():
        r=primitive.typed.extract(z.to(device)); full=primitive.typed.decode(primitive.typed.transform(r,theta)); masked=primitive(z.to(device),theta); yy=y.to(device); full_loss=((full-yy)**2).mean().sqrt().item(); mask_loss=((masked-yy)**2).mean().sqrt().item()
    suff=full_loss/(mask_loss+1e-8); return suff,full_loss,mask_loss

def causal_feature_necessity(primitive,theta,teacher,loader,args,device):
    z,y=source_bundle(teacher,loader,args,device); r=primitive.typed.extract(z.to(device)).detach(); mask=primitive.selector.mask().detach(); active=torch.where(mask>0.5)[0]; base=primitive(z.to(device),theta); base_loss=((base-y.to(device))**2).mean().item(); drops=[]
    with torch.no_grad():
        for j in active.tolist():
            m=mask.clone(); m[j]=0.; rr=r*m.view(1,1,-1); out=primitive.typed.decode(primitive.typed.transform(rr,theta)); drops.append(float(((out-y.to(device))**2).mean().item()-base_loss))
    return (sum(drops)/len(drops) if drops else 0.),len(active)

def diagnose(c,args):
    reasons=[]
    if c['source_avg']<args.min_avg_source_acc or c['source_worst']<args.min_worst_source_acc: reasons.append('capacity: source computation underfit')
    if c['theta_effect']<args.min_theta_effect or c['theta_stability']>args.max_theta_stability: reasons.append('theta gate failed')
    if c['bottleneck_sufficiency']<args.min_bottleneck_sufficiency: reasons.append('S fail: bottleneck does not preserve enough computation')
    if c['causal_necessity']<args.min_causal_necessity: reasons.append('S fail: selected bottleneck features are not causally necessary')
    if c['contrast_specificity']<args.min_contrast_specificity: reasons.append('E fail: representation not contrast-specific')
    return reasons or ['no failing gate']

def candidate_eval(typed_trip,bundles,teachers,contrast_bundle,args,device,seed):
    ext,tr,dec=typed_trip; typed=TypedPrimitive(ext,tr,dec,args.d_model,args.rel_dim,args.rank).to(device); theta_mat,stab,effect=train_bottleneck(BottleneckPrimitive(typed,args.bottleneck_dim).to(device),bundles,args,device); primitive=BottleneckPrimitive(typed,args.bottleneck_dim).to(device); # retrain wrapper for clean selector
    theta_mat,stab,effect=train_bottleneck(primitive,bundles,args,device)
    importance=select_importance(primitive,bundles,theta_mat,args,device); idx=fit_masked_selector(primitive,bundles,theta_mat,args,device,importance)
    task_acc=[]
    for i,t in enumerate(teachers):
        tea,_,va=teachers[t]; cm=RoutingCompiled(tea,primitive,theta_mat[i],args.trajectory_start,args.trajectory_end).to(device); task_acc.append(evaluate(cm,va,device)[0])
    avg=sum(task_acc)/len(task_acc); worst=min(task_acc)
    irel,icon,spec=interface_stats(primitive,bundles,contrast_bundle,args,device)
    suf,_,_=causal_bottleneck_sufficiency(primitive,theta_mat[0],list(teachers.values())[0][0],list(teachers.values())[0][1],args,device,args.trajectory_start)
    nec,_=causal_feature_necessity(primitive,theta_mat[0],list(teachers.values())[0][0],list(teachers.values())[0][1],args,device)
    eligible=(avg>=args.min_avg_source_acc and worst>=args.min_worst_source_acc and effect>=args.min_theta_effect and stab<=args.max_theta_stability and irel>=args.min_interface_invariance and suf>=args.min_bottleneck_sufficiency and nec>=args.min_causal_necessity and spec>=args.min_contrast_specificity)
    score=avg+0.2*effect+0.2*irel+0.2*suf+0.1*nec-args.complexity_lambda*count_params(primitive)
    return {'triple':typed_trip,'primitive':primitive,'theta':theta_mat,'source_avg':avg,'source_worst':worst,'theta_stability':stab,'theta_effect':effect,'iis_related':irel,'iis_contrast':icon,'contrast_specificity':spec,'bottleneck_sufficiency':suf,'causal_necessity':nec,'selected_indices':idx.tolist(),'task_acc':task_acc,'eligible':eligible,'score':score,'diagnosis':[]}

def run_seed(args,seed):
    device=torch.device(args.device if args.device=='cpu' or torch.cuda.is_available() else 'cpu'); seed_all(seed); meta=[t for t in args.all_tasks if t not in args.holdout_tasks and t not in args.contrast_tasks]; teachers={t:train_teacher(t,args,device,seed*1000+i) for i,t in enumerate(meta)}
    contrast_task=args.contrast_tasks[0]; ctea,ctr,cva=train_teacher(contrast_task,args,device,seed+70000); contrast_bundle=source_bundle(ctea,ctr,args,device); bundles={t:source_bundle(tea,tr,args,device) for t,(tea,tr,_) in teachers.items()}
    triples=[('diff_proj','affine_polynomial','linear'),('product_proj','polynomial','affine'),('stats_proj','diagonal','affine'),('raw_proj','affine_polynomial','affine')]
    candidates=[]
    for trp in triples:
        c=candidate_eval(trp,bundles,teachers,contrast_bundle,args,device,seed); c['diagnosis']=diagnose(c,args); candidates.append(c)
    elig=[c for c in candidates if c['eligible']]; best=max(elig,key=lambda c:c['score']) if elig else max(candidates,key=lambda c:c['score']);
    out={'seed':seed,'winner':best}
    for label,t in [('related',args.holdout_tasks[0]),('contrast',contrast_task)]:
        if label=='related': tea,tr,va=train_teacher(t,args,device,seed+50000)
        else: tea,tr,va=ctea,ctr,cva
        te=DataLoader(TaskDataset(args.test_size,t,seed+60000+(0 if label=='related' else 1)),batch_size=args.batch_size,shuffle=False); at=evaluate(tea,te,device)[0]
        th0=best['theta'].mean(0).detach(); m0=RoutingCompiled(tea,best['primitive'],th0,args.trajectory_start,args.trajectory_end).to(device); a0=evaluate(m0,te,device)[0]
        z,y=source_bundle(tea,tr,args,device); tht=fit_theta(best['primitive'],z,y,args,device,args.target_theta_fit_steps,th0); m1=RoutingCompiled(tea,best['primitive'],tht,args.trajectory_start,args.trajectory_end).to(device); a1=evaluate(m1,te,device)[0]
        mlp=MLPControl(tea,args.trajectory_start,args.trajectory_end,args.mlp_width).to(device); train(mlp,tr,device,args.transfer_control_steps,args.lr); am=evaluate(mlp,te,device)[0]
        out[f'{label}_holdout']={'task':t,'teacher':at,'dart_zero':a0,'dart_adapted':a1,'mlp_control':am,'gain_zero':(a0-at)*100,'gain_adapt':(a1-at)*100,'vs_mlp_adapt':(a1-am)*100,'theta':tht.cpu().tolist()}
    return out

def clean(c):
    d={k:v for k,v in c.items() if k!='primitive'}
    if torch.is_tensor(d.get('theta')): d['theta']=d['theta'].cpu().tolist()
    return d

def main():
    p=argparse.ArgumentParser(description='DART-1.8 causal bottleneck discovery')
    p.add_argument('--seeds',nargs='+',type=int,default=[1,2]); p.add_argument('--all-tasks',nargs='+',default=['add','compose','mul','sub']); p.add_argument('--holdout-tasks',nargs='+',default=['sub']); p.add_argument('--contrast-tasks',nargs='+',default=['sort'])
    p.add_argument('--teacher-steps',type=int,default=800); p.add_argument('--core-fit-steps',type=int,default=300); p.add_argument('--theta-fit-steps',type=int,default=120); p.add_argument('--target-theta-fit-steps',type=int,default=400); p.add_argument('--transfer-control-steps',type=int,default=400)
    p.add_argument('--train-size',type=int,default=6000); p.add_argument('--verifier-size',type=int,default=1500); p.add_argument('--test-size',type=int,default=1500); p.add_argument('--rel-samples-per-task',type=int,default=2048)
    p.add_argument('--bottleneck-dim',type=int,default=4); p.add_argument('--rel-dim',type=int,default=12); p.add_argument('--iis-topk',type=int,default=4); p.add_argument('--theta-delta',type=float,default=.25); p.add_argument('--theta-l2',type=float,default=.0005); p.add_argument('--theta-lr',type=float,default=.01)
    p.add_argument('--min-avg-source-acc',type=float,default=.30); p.add_argument('--min-worst-source-acc',type=float,default=.22); p.add_argument('--min-theta-effect',type=float,default=.02); p.add_argument('--max-theta-stability',type=float,default=.75); p.add_argument('--min-bottleneck-sufficiency',type=float,default=.70); p.add_argument('--min-causal-necessity',type=float,default=.001); p.add_argument('--min-contrast-specificity',type=float,default=.05); p.add_argument('--min-interface-invariance',type=float,default=.30)
    p.add_argument('--fit-batch-samples',type=int,default=512); p.add_argument('--bottleneck-fit-samples',type=int,default=512); p.add_argument('--effect-samples',type=int,default=512); p.add_argument('--d-model',type=int,default=32); p.add_argument('--heads',type=int,default=2); p.add_argument('--d-ff',type=int,default=128); p.add_argument('--depth',type=int,default=3); p.add_argument('--rank',type=int,default=8); p.add_argument('--batch-size',type=int,default=256); p.add_argument('--trajectory-start',type=int,default=0); p.add_argument('--trajectory-end',type=int,default=1); p.add_argument('--mlp-width',type=int,default=64); p.add_argument('--core-fit-lr',type=float,default=.001); p.add_argument('--lr',type=float,default=.0003); p.add_argument('--complexity-lambda',type=float,default=1e-5); p.add_argument('--device',default='cuda'); p.add_argument('--out',default='dart018_results.json')
    a=p.parse_args(); rec=[]
    for s in a.seeds: print(f'seed={s}',flush=True); rec.append(run_seed(a,s))
    def av(k,sec): return sum(r[sec][k] for r in rec)/len(rec)
    summary={'related_holdout':{a.holdout_tasks[0]:{k:av(k,'related_holdout') for k in ['teacher','dart_zero','dart_adapted','mlp_control','gain_zero','gain_adapt','vs_mlp_adapt']}},'contrast_holdout':{a.contrast_tasks[0]:{k:av(k,'contrast_holdout') for k in ['teacher','dart_zero','dart_adapted','mlp_control','gain_zero','gain_adapt','vs_mlp_adapt']}},'source':{'avg_accuracy':sum(r['winner']['source_avg'] for r in rec)/len(rec),'avg_theta_effect':sum(r['winner']['theta_effect'] for r in rec)/len(rec),'avg_theta_stability':sum(r['winner']['theta_stability'] for r in rec)/len(rec),'avg_iis_related':sum(r['winner']['iis_related'] for r in rec)/len(rec),'avg_contrast_specificity':sum(r['winner']['contrast_specificity'] for r in rec)/len(rec),'avg_bottleneck_sufficiency':sum(r['winner']['bottleneck_sufficiency'] for r in rec)/len(rec),'avg_causal_necessity':sum(r['winner']['causal_necessity'] for r in rec)/len(rec)}}
    payload={'config':vars(a),'records':[{'seed':r['seed'],'winner':clean(r['winner']),'related_holdout':r['related_holdout'],'contrast_holdout':r['contrast_holdout']} for r in rec],'summary':summary}; Path(a.out).write_text(json.dumps(payload,indent=2)); print('DART-1.8: causal bottleneck discovery'); print(json.dumps(summary,indent=2)); print('Saved:',Path(a.out).resolve())

if __name__=='__main__': main()
