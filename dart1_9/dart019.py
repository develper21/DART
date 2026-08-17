#!/usr/bin/env python3
"""DART-1.9: Counterfactual Causal Interchangeability Discovery.

Builds on DART-1.7's typed E->T(theta)->D pipeline and adds a teacher-grounded
counterfactual state-swap test.  A reusable representation should support
cross-task swaps that produce predictable changes in the frozen downstream
teacher, while random swaps should not.
"""
from __future__ import annotations
import argparse, copy, itertools, json, random, math
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
def make_core(name,d,r): return {'diagonal':DiagonalCore(d),'polynomial':PolynomialCore(d),'affine_polynomial':AffinePolynomialCore(d,r),'low_rank':LowRankCore(d,r)}[name]

class RelationExtractor(nn.Module):
    def __init__(self,kind,d,rel_dim):
        super().__init__(); self.kind=kind; self.half=d//2
        if kind in ('diff_proj','product_proj'): self.proj=nn.Linear(self.half,rel_dim)
        elif kind=='stats_proj': self.proj=nn.Linear(6,rel_dim)
        elif kind=='raw_proj': self.proj=nn.Linear(d,rel_dim)
        else: raise ValueError(kind)
        nn.init.xavier_uniform_(self.proj.weight,gain=.1); nn.init.zeros_(self.proj.bias)
    def forward(self,z):
        h=self.half
        if self.kind=='diff_proj': feat=z[...,:h]-z[...,h:2*h]
        elif self.kind=='product_proj': feat=z[...,:h]*z[...,h:2*h]
        elif self.kind=='stats_proj':
            z1,z2=z[...,:h],z[...,h:2*h]
            feat=torch.stack([z1.mean(-1),z1.var(-1,unbiased=False),z2.mean(-1),z2.var(-1,unbiased=False),z.mean(-1),z.var(-1,unbiased=False)],dim=-1)
        else: feat=z
        return self.proj(feat)
class RelationDecoder(nn.Module):
    def __init__(self,kind,rel_dim,d):
        super().__init__(); self.kind=kind; self.up=nn.Linear(rel_dim,d); nn.init.xavier_uniform_(self.up.weight,gain=.05); nn.init.zeros_(self.up.bias)
        if kind=='affine': self.quad=nn.Linear(rel_dim,d,bias=False); nn.init.xavier_uniform_(self.quad.weight,gain=.02)
    def forward(self,r): return self.up(r)+self.quad(r.square()) if self.kind=='affine' else self.up(r)
class ETDPrimitive(nn.Module):
    def __init__(self,e,t,d,d_model,rel_dim,rank):
        super().__init__(); self.E=RelationExtractor(e,d_model,rel_dim); self.Tcore=make_core(t,rel_dim,rank); self.D=RelationDecoder(d,rel_dim,d_model); self.names=(e,t,d); self.theta_dim=1
    def extract(self,x): return self.E(x)
    def transform(self,r,theta): return r+theta[0].view(1,1)*self.Tcore(r)
    def decode(self,r): return self.D(r)
    def forward(self,x,theta): return self.decode(self.transform(self.extract(x),theta))

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
    def __init__(self,t,start,end,w): super().__init__(); self.emb=copy.deepcopy(t.emb); self.pos=copy.deepcopy(t.pos); self.head=copy.deepcopy(t.head); self.blocks=nn.ModuleList([MLPReplaceBlock(b,t.d,w) if start<=i<end else copy.deepcopy(b) for i,b in enumerate(t.blocks)])
    def forward(self,x):
        h=self.emb(x)+self.pos[:,:x.size(1)]
        for b in self.blocks: h=b(h)
        return self.head(h[:,0])

def evaluate(model,loader,device):
    model.eval(); total=correct=0
    with torch.no_grad():
        for x,y in loader:
            x=x.to(device); y=y.to(device); z=model(x); correct+=int((z.argmax(-1)==y).sum()); total+=y.numel()
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
                    take=min(maxn-have,z.reshape(-1,z.shape[-1]).shape[0]); zs.append(z.reshape(-1,z.shape[-1])[:take].cpu()); break
                h=u+b.ff(z)
    return torch.cat(zs)

def theta_fit(p,z,y,args,device,steps,init=None):
    th=nn.Parameter(init.clone().to(device) if init is not None else torch.tensor([0.5],device=device)); opt=torch.optim.Adam([th],lr=args.theta_lr); z=z.to(device); y=y.to(device)
    for _ in range(steps):
        pred=p(z,th); loss=((pred-y)**2).mean()+args.theta_l2*th.square().mean(); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    return th.detach()

def source_bundle(teacher,loader,args,device):
    z=capture_ff(teacher,loader,device,args.rel_samples_per_task,args.trajectory_start); y=teacher.blocks[args.trajectory_start].ff(z.to(device)).detach().cpu(); return z,y

def fit_primitive(triple,teachers,args,device):
    ext,t,dec=triple; p=ETDPrimitive(ext,t,dec,args.d_model,args.rel_dim,args.rank).to(device)
    bundles={k:source_bundle(v[0],v[1],args,device) for k,v in teachers.items()}; tasks=list(bundles)
    tb=nn.Parameter(torch.full((len(tasks),1),0.5,device=device)); opt=torch.optim.AdamW(list(p.parameters())+[tb],lr=args.core_fit_lr,weight_decay=1e-4)
    for _ in range(args.core_fit_steps):
        loss=0
        for i,k in enumerate(tasks):
            z,y=bundles[k]; idx=torch.randperm(len(z))[:min(args.fit_batch_samples,len(z))]; loss+=((p(z[idx].to(device),tb[i])-y[idx].to(device))**2).mean()+args.theta_l2*tb[i].square().mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    for q in p.parameters(): q.requires_grad=False
    theta=[]
    st=[]
    for i,k in enumerate(tasks):
        z,y=bundles[k]; th=theta_fit(p,z,y,args,device,args.theta_fit_steps,init=tb[i].detach()); theta.append(th)
        n=len(z); ta=theta_fit(p,z[:n//2],y[:n//2],args,device,max(10,args.theta_fit_steps//2),init=th); tb2=theta_fit(p,z[n//2:],y[n//2:],args,device,max(10,args.theta_fit_steps//2),init=th); st.append(float((ta-tb2).norm().item()))
    thm=torch.stack(theta)
    eff=[]; z0=next(iter(bundles.values()))[0][:min(args.effect_samples,len(next(iter(bundles.values()))[0]))].to(device)
    for sign in (-1,1):
        d=0.25*sign
        eff.append(float((p(z0,thm[0]+d)-p(z0,thm[0]-d)).norm().item()/(0.5)))
    return p,thm,sum(st)/len(st),sum(eff)/len(eff),bundles

def intervention_swap_score(p,theta_map,teacher_a,loader_a,teacher_b,loader_b,args,device,layer):
    x_a,_=next(iter(loader_a)); x_b,_=next(iter(loader_b)); x_a=x_a.to(device); x_b=x_b.to(device)
    with torch.no_grad():
        def get_z(teacher,x):
            h=teacher.emb(x)+teacher.pos[:,:x.size(1)]
            for i,b in enumerate(teacher.blocks):
                n=b.norm1(h); a,_=b.attn(n,n,n,need_weights=False); u=h+a; z=b.norm2(u)
                if i==layer: return u,z
                h=u+b.ff(z)
        uA,zA=get_z(teacher_a,x_a); uB,zB=get_z(teacher_b,x_b)
        rA=p.extract(zA); rB=p.extract(zB)
        k=min(rA.shape[0],rB.shape[0]); rA=rA[:k]; rB=rB[:k]
        thA=theta_map[0]
        outA=p.decode(p.transform(rA,thA)); outB=p.decode(p.transform(rB,thA))
        # teacher-grounded downstream effect of swapping A representation for B representation
        def downstream(h0):
            h=h0
            for j in range(layer+1,len(teacher_a.blocks)): h=teacher_a.blocks[j](h)
            return teacher_a.head(h[:,0])
        base=downstream(uA+outA); cf=downstream(uA+outB); actual=cf-base
        local=teacher_a.head((uA+outB)[:,0])-teacher_a.head((uA+outA)[:,0])
        cos=float(nn.functional.cosine_similarity(actual,local,dim=-1).mean().item())
        # random control: permute rB rows
        perm=torch.randperm(k,device=device); rout=downstream(uA+p.decode(p.transform(rB[perm],thA))); rand_delta=rout-base
        rand_cos=float(nn.functional.cosine_similarity(actual,rand_delta,dim=-1).mean().item())
        return cos,rand_cos

def run_seed(args,seed):
    device=torch.device(args.device if args.device=='cpu' or torch.cuda.is_available() else 'cpu'); seed_all(seed)
    meta=[t for t in args.all_tasks if t not in args.holdout_tasks and t not in args.contrast_tasks]
    teachers={t:train_teacher(t,args,device,seed*1000+i) for i,t in enumerate(meta)}
    contrast=args.contrast_tasks[0]; c_tea,c_tr,c_va=train_teacher(contrast,args,device,seed+70000)
    triples=[('diff_proj','affine_polynomial','linear'),('product_proj','polynomial','affine'),('stats_proj','diagonal','affine'),('raw_proj','affine_polynomial','affine')]
    scored=[]
    for tri in triples:
        p,th,stab,eff,bundles=fit_primitive(tri,teachers,args,device)
        acc=[]
        for i,t in enumerate(meta):
            cm=RoutingCompiled(teachers[t][0],p,th[i],args.trajectory_start,args.trajectory_end).to(device); acc.append(evaluate(cm,teachers[t][2],device))
        # task-specific cross-swap matrix on first two meta tasks
        swap_scores=[]; rand_scores=[]
        for a,b in itertools.combinations(meta,2):
            s,r=intervention_swap_score(p,th,teachers[a][0],teachers[a][1],teachers[b][0],teachers[b][1],args,device,args.trajectory_start); swap_scores.append(s); rand_scores.append(r)
        swap=float(sum(swap_scores)/len(swap_scores)); rnd=float(sum(rand_scores)/len(rand_scores))
        # state-specificity: difference between learned cross-task swap and random control
        portability=swap-rnd
        eligible=(sum(acc)/len(acc)>=args.min_avg_source_acc and min(acc)>=args.min_worst_source_acc and eff>=args.min_theta_effect and stab<=args.max_theta_stability and portability>=args.min_swap_margin)
        score=sum(acc)/len(acc)+args.swap_weight*portability-args.complexity_lambda*sum(q.numel() for q in p.parameters())
        scored.append({'triple':tri,'primitive':p,'theta':th,'source_avg':sum(acc)/len(acc),'source_worst':min(acc),'theta_stability':stab,'theta_effect':eff,'swap_fidelity':swap,'random_swap_fidelity':rnd,'swap_margin':portability,'task_acc':acc,'eligible':eligible,'score':score,'bundles':bundles})
    elig=[c for c in scored if c['eligible']]; best=max(elig,key=lambda c:c['score']) if elig else max(scored,key=lambda c:c['score'])
    # related/contrast transfer
    results={}
    for label,t in [('related',args.holdout_tasks[0]),('contrast',contrast)]:
        if label=='related': tea,tr,va=train_teacher(t,args,device,seed+50000)
        else: tea,tr,va=c_tea,c_tr,c_va
        te=DataLoader(TaskDataset(args.test_size,t,seed+60000+(0 if label=='related' else 1)),batch_size=args.batch_size,shuffle=False,pin_memory=device.type=='cuda')
        z,y=source_bundle(tea,tr,args,device); th0=best['theta'].mean(0); m0=RoutingCompiled(tea,best['primitive'],th0,args.trajectory_start,args.trajectory_end).to(device); a0=evaluate(m0,te,device); tht=theta_fit(best['primitive'],z,y,args,device,args.target_theta_fit_steps,init=th0); m1=RoutingCompiled(tea,best['primitive'],tht,args.trajectory_start,args.trajectory_end).to(device); a1=evaluate(m1,te,device); mlp=MLPControl(tea,args.trajectory_start,args.trajectory_end,args.mlp_width).to(device); train(mlp,tr,device,args.transfer_control_steps,args.lr); am=evaluate(mlp,te,device); results[label+'_holdout']={'task':t,'teacher':evaluate(tea,te,device),'dart_zero':a0,'dart_adapted':a1,'mlp_control':am,'gain_zero':(a0-evaluate(tea,te,device))*100,'gain_adapt':(a1-evaluate(tea,te,device))*100,'vs_mlp_adapt':(a1-am)*100,'theta':tht.cpu().tolist()}
    clean=lambda c:{k:v for k,v in c.items() if k not in ('primitive','theta','bundles')}
    return {'seed':seed,'winner':clean(best),'candidates':[clean(c) for c in scored],**results}

def main():
    ap=argparse.ArgumentParser(description='DART-1.9 counterfactual causal interchangeability discovery')
    ap.add_argument('--seeds',nargs='+',type=int,default=[1,2]); ap.add_argument('--all-tasks',nargs='+',default=['add','compose','mul','sub']); ap.add_argument('--holdout-tasks',nargs='+',default=['sub']); ap.add_argument('--contrast-tasks',nargs='+',default=['sort'])
    for k,v in [('teacher-steps',800),('core-fit-steps',300),('theta-fit-steps',120),('target-theta-fit-steps',400),('transfer-control-steps',400),('train-size',6000),('verifier-size',1500),('test-size',1500),('rel-samples-per-task',2048),('fit-batch-samples',512),('effect-samples',512),('d-model',32),('heads',2),('d-ff',128),('depth',3),('rank',8),('batch-size',256),('rel-dim',12),('mlp-width',64)]: ap.add_argument('--'+k,type=int,default=v)
    ap.add_argument('--theta-delta',type=float,default=.25); ap.add_argument('--theta-l2',type=float,default=.0005); ap.add_argument('--theta-lr',type=float,default=.01); ap.add_argument('--min-avg-source-acc',type=float,default=.30); ap.add_argument('--min-worst-source-acc',type=float,default=.22); ap.add_argument('--min-theta-effect',type=float,default=.02); ap.add_argument('--max-theta-stability',type=float,default=.75); ap.add_argument('--min-swap-margin',type=float,default=.05); ap.add_argument('--swap-weight',type=float,default=.5); ap.add_argument('--complexity-lambda',type=float,default=1e-5); ap.add_argument('--trajectory-start',type=int,default=0); ap.add_argument('--trajectory-end',type=int,default=1); ap.add_argument('--lr',type=float,default=.0003); ap.add_argument('--core-fit-lr',type=float,default=.001); ap.add_argument('--device',default='cuda'); ap.add_argument('--out',default='dart019_results.json')
    args=ap.parse_args(); rec=[run_seed(args,s) for s in args.seeds]
    def av(k,sec): return sum(r[sec][k] for r in rec)/len(rec)
    summary={'related_holdout':{args.holdout_tasks[0]:{k:av(k,'related_holdout') for k in ['teacher','dart_zero','dart_adapted','mlp_control','gain_zero','gain_adapt','vs_mlp_adapt']}},'contrast_holdout':{args.contrast_tasks[0]:{k:av(k,'contrast_holdout') for k in ['teacher','dart_zero','dart_adapted','mlp_control','gain_zero','gain_adapt','vs_mlp_adapt']}},'source':{'avg_accuracy':sum(r['winner']['source_avg'] for r in rec)/len(rec),'avg_theta_effect':sum(r['winner']['theta_effect'] for r in rec)/len(rec),'avg_theta_stability':sum(r['winner']['theta_stability'] for r in rec)/len(rec),'avg_swap_fidelity':sum(r['winner']['swap_fidelity'] for r in rec)/len(rec),'avg_random_swap_fidelity':sum(r['winner']['random_swap_fidelity'] for r in rec)/len(rec),'avg_swap_margin':sum(r['winner']['swap_margin'] for r in rec)/len(rec)}}
    Path(args.out).write_text(json.dumps({'config':vars(args),'records':rec,'summary':summary},indent=2))
    print('DART-1.9: counterfactual causal interchangeability discovery'); print(json.dumps(summary,indent=2)); print('Saved:',Path(args.out).resolve())
if __name__=='__main__': main()
