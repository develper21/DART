#!/usr/bin/env python3
"""DART-1.6: compositional factorized primitive discovery.

Research hypothesis:
A reusable computation may be a shallow composition of structured primitives,
with a small explicit task-specific parameter vector. The composition itself is
shared and frozen during holdout transfer; only the tiny theta is refit.

DART-1.6 adds a joint source-task discovery phase and strict gates inherited
from DART-1.5: source capability, non-trivial theta, causal theta effect, and
stable theta estimates across disjoint source splits.
"""
from __future__ import annotations
import argparse, copy, json, random
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

# ---------- structured families ----------
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
    return {'diagonal':DiagonalCore(d),'polynomial':PolynomialCore(d),'affine_polynomial':AffinePolynomialCore(d,r),'low_rank':LowRankCore(d,r)}[name]

class ComposedPrimitive(nn.Module):
    """A shallow composition; each stage has one explicit task scalar theta."""
    def __init__(self,names,d,r):
        super().__init__(); self.names=tuple(names); self.stages=nn.ModuleList([make_core(n,d,r) for n in names]); self.theta_dim=len(names)
    def forward(self,x,theta):
        h=x
        for i,stage in enumerate(self.stages):
            h=h + theta[i].view(1,1) * stage(h)
        return h

class RoutingCompiled(nn.Module):
    def __init__(self,teacher,primitive,theta,start,end):
        super().__init__(); self.emb=copy.deepcopy(teacher.emb); self.pos=copy.deepcopy(teacher.pos); self.head=copy.deepcopy(teacher.head)
        self.blocks=nn.ModuleList([RoutingReplaceBlock(b,primitive,theta) if start<=i<end else copy.deepcopy(b) for i,b in enumerate(teacher.blocks)])
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
    return correct/max(total,1), loss/max(total,1)

def train(model,loader,device,steps,lr):
    ps=[p for p in model.parameters() if p.requires_grad]
    if not ps: raise RuntimeError('No trainable parameters')
    opt=torch.optim.AdamW(ps,lr=lr,weight_decay=1e-4); ce=nn.CrossEntropyLoss(); it=iter(loader); model.train()
    for _ in range(steps):
        try: x,y=next(it)
        except StopIteration: it=iter(loader); x,y=next(it)
        x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True); loss=ce(model(x),y); opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(ps,1.); opt.step()

def train_teacher(task,args,device,seed):
    tr=DataLoader(TaskDataset(args.train_size,task,seed),batch_size=args.batch_size,shuffle=True,pin_memory=device.type=='cuda')
    va=DataLoader(TaskDataset(args.verifier_size,task,seed+10_000),batch_size=args.batch_size,shuffle=False,pin_memory=device.type=='cuda')
    te=Teacher(len(VOCAB),args.d_model,args.heads,args.d_ff,args.depth).to(device); train(te,tr,device,args.teacher_steps,args.lr); return te,tr,va

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

def theta_fit(primitive,z,y,args,device,steps,init=None):
    th=nn.Parameter(init.clone().to(device) if init is not None else torch.full((primitive.theta_dim,),0.5,device=device))
    opt=torch.optim.Adam([th],lr=args.theta_lr); z=z.to(device); y=y.to(device)
    for _ in range(steps):
        pred=primitive(z,th); loss=((pred-y)**2).mean()+args.theta_l2*(th.square().mean()); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    return th.detach()

def source_bundle(teacher,loader,args,device):
    z=capture_ff(teacher,loader,device,args.rel_samples_per_task,args.trajectory_start); y=teacher.blocks[args.trajectory_start].ff(z.to(device)).detach().cpu(); return z,y

def fit_shared_composition(pair,teachers,args,device,seed):
    primitive=ComposedPrimitive(pair,args.d_model,args.rank).to(device)
    # Freeze teacher-derived targets and jointly optimize shared primitive + per-task theta.
    bundles={t:source_bundle(tea,tr,args,device) for t,(tea,tr,_) in teachers.items()}
    theta_bank=nn.Parameter(torch.full((len(bundles),primitive.theta_dim),0.5,device=device))
    opt=torch.optim.AdamW(list(primitive.parameters())+[theta_bank],lr=args.core_fit_lr,weight_decay=1e-4)
    tasks=list(bundles); ce_terms=[]
    for _ in range(args.core_fit_steps):
        total=0.
        for i,t in enumerate(tasks):
            z,y=bundles[t]; idx=torch.randperm(len(z),device='cpu')[:min(args.fit_batch_samples,len(z))]; zz=z[idx].to(device); yy=y[idx].to(device); pred=primitive(zz,theta_bank[i]); total=total+((pred-yy)**2).mean()+args.theta_l2*theta_bank[i].square().mean()
        opt.zero_grad(set_to_none=True); total.backward(); nn.utils.clip_grad_norm_(list(primitive.parameters())+[theta_bank],1.); opt.step()
    # refine theta independently on each source task using frozen primitive
    for p in primitive.parameters(): p.requires_grad=False
    final_theta=[]
    for t in tasks:
        z,y=bundles[t]; th=theta_fit(primitive,z,y,args,device,args.theta_fit_steps,init=theta_bank[tasks.index(t)].detach()); final_theta.append(th)
    theta_mat=torch.stack(final_theta)
    # theta stability on disjoint halves
    stab=[]
    for i,t in enumerate(tasks):
        z,y=bundles[t]; n=len(z); a=slice(0,n//2); b=slice(n//2,n); ta=theta_fit(primitive,z[a],y[a],args,device,max(10,args.theta_fit_steps//2),init=theta_mat[i]); tb=theta_fit(primitive,z[b],y[b],args,device,max(10,args.theta_fit_steps//2),init=theta_mat[i]); stab.append(float(torch.norm(ta-tb).item()))
    theta_stability=sum(stab)/len(stab)
    # causal effect: perturb each stage theta and measure output response in primitive space
    effects=[]
    z0=next(iter(bundles.values()))[0][:min(args.effect_samples,len(next(iter(bundles.values()))[0]))].to(device)
    for k in range(primitive.theta_dim):
        base=theta_mat[0].clone(); plus=base.clone(); minus=base.clone(); plus[k]+=args.theta_delta; minus[k]-=args.theta_delta
        with torch.no_grad():
            e=(primitive(z0,plus)-primitive(z0,minus)).norm().item()/(2*args.theta_delta)
        effects.append(e)
    theta_effect=sum(effects)/len(effects)
    return primitive,theta_mat,theta_stability,theta_effect,bundles

def candidate_score(pair,teachers,args,device,seed):
    primitive,theta_mat,stab,effect,bundles=fit_shared_composition(pair,teachers,args,device,seed)
    task_acc=[]
    for i,t in enumerate(teachers):
        tea,_,va=teachers[t]; cm=RoutingCompiled(tea,primitive,theta_mat[i],args.trajectory_start,args.trajectory_end).to(device); acc,_=evaluate(cm,va,device); task_acc.append(acc)
    avg=sum(task_acc)/len(task_acc); worst=min(task_acc); eligible=(avg>=args.min_avg_source_acc and worst>=args.min_worst_source_acc and effect>=args.min_theta_effect and stab<=args.max_theta_stability)
    score=avg + args.theta_effect_weight*effect - args.complexity_lambda*count_params(primitive)
    return {'pair':pair,'primitive':primitive,'theta':theta_mat,'source_avg':avg,'source_worst':worst,'theta_stability':stab,'theta_effect':effect,'task_acc':task_acc,'eligible':eligible,'score':score}

def run_seed(args,seed):
    device=torch.device(args.device if args.device=='cpu' or torch.cuda.is_available() else 'cpu'); seed_all(seed)
    meta=[t for t in args.all_tasks if t not in args.holdout_tasks and t not in args.contrast_tasks]
    teachers={t:train_teacher(t,args,device,seed*1000+i) for i,t in enumerate(meta)}
    pairs=[('diagonal','diagonal'),('polynomial','diagonal'),('affine_polynomial','polynomial'),('low_rank','diagonal')]
    candidates=[candidate_score(p,teachers,args,device,seed+11*i) for i,p in enumerate(pairs)]
    eligible=[c for c in candidates if c['eligible']]
    best=max(eligible,key=lambda c:c['score']) if eligible else max(candidates,key=lambda c:c['score'])
    result={'seed':seed,'winner':best['pair'],'eligible':best['eligible'],'source_avg':best['source_avg'],'source_worst':best['source_worst'],'theta_effect':best['theta_effect'],'theta_stability':best['theta_stability'],'source_theta':best['theta'].detach().cpu().tolist(),'core_params':count_params(best['primitive'])}
    for label,t in [('related',args.holdout_tasks[0]),('contrast',args.contrast_tasks[0])]:
        tea,tr,va=train_teacher(t,args,device,seed+50_000+(0 if label=='related' else 1)); te=DataLoader(TaskDataset(args.test_size,t,seed+60_000+(0 if label=='related' else 1)),batch_size=args.batch_size,shuffle=False)
        z,y=source_bundle(tea,tr,args,device); theta0=best['theta'].mean(0).detach(); model0=RoutingCompiled(tea,best['primitive'],theta0,args.trajectory_start,args.trajectory_end).to(device); a0,_=evaluate(model0,te,device)
        theta_t=theta_fit(best['primitive'],z,y,args,device,args.target_theta_fit_steps,init=theta0); model1=RoutingCompiled(tea,best['primitive'],theta_t,args.trajectory_start,args.trajectory_end).to(device); a1,_=evaluate(model1,te,device)
        mlp=MLPControl(tea,args.trajectory_start,args.trajectory_end,args.mlp_width).to(device); train(mlp,tr,device,args.transfer_control_steps,args.lr); am,_=evaluate(mlp,te,device); at,_=evaluate(tea,te,device)
        result[f'{label}_holdout']={'task':t,'teacher':at,'dart_zero':a0,'dart_adapted':a1,'mlp_control':am,'gain_zero':(a0-at)*100,'gain_adapt':(a1-at)*100,'vs_mlp_adapt':(a1-am)*100,'theta':theta_t.cpu().tolist()}
    return result

def main():
    p=argparse.ArgumentParser(description='DART-1.6 compositional factorized primitive discovery')
    p.add_argument('--seeds',nargs='+',type=int,default=[1,2]); p.add_argument('--all-tasks',nargs='+',default=['add','compose','mul','sub']); p.add_argument('--holdout-tasks',nargs='+',default=['sub']); p.add_argument('--contrast-tasks',nargs='+',default=['sort'])
    p.add_argument('--teacher-steps',type=int,default=800); p.add_argument('--core-fit-steps',type=int,default=300); p.add_argument('--theta-fit-steps',type=int,default=120); p.add_argument('--target-theta-fit-steps',type=int,default=400); p.add_argument('--transfer-control-steps',type=int,default=400); p.add_argument('--train-size',type=int,default=6000); p.add_argument('--verifier-size',type=int,default=1500); p.add_argument('--test-size',type=int,default=1500); p.add_argument('--rel-samples-per-task',type=int,default=2048); p.add_argument('--rel-directions',type=int,default=4); p.add_argument('--theta-delta',type=float,default=.25); p.add_argument('--theta-l2',type=float,default=.0005); p.add_argument('--theta-lr',type=float,default=.01); p.add_argument('--min-avg-source-acc',type=float,default=.30); p.add_argument('--min-worst-source-acc',type=float,default=.22); p.add_argument('--min-theta-effect',type=float,default=.02); p.add_argument('--max-theta-stability',type=float,default=.75); p.add_argument('--theta-effect-weight',type=float,default=.25); p.add_argument('--complexity-lambda',type=float,default=1e-5); p.add_argument('--fit-batch-samples',type=int,default=512); p.add_argument('--effect-samples',type=int,default=512); p.add_argument('--d-model',type=int,default=32); p.add_argument('--heads',type=int,default=2); p.add_argument('--d-ff',type=int,default=128); p.add_argument('--depth',type=int,default=3); p.add_argument('--rank',type=int,default=8); p.add_argument('--batch-size',type=int,default=256); p.add_argument('--trajectory-start',type=int,default=0); p.add_argument('--trajectory-end',type=int,default=1); p.add_argument('--mlp-width',type=int,default=64); p.add_argument('--core-fit-lr',type=float,default=.001); p.add_argument('--lr',type=float,default=.0003); p.add_argument('--device',default='cuda'); p.add_argument('--out',default='dart016_results.json'); args=p.parse_args()
    rec=[run_seed(args,s) for s in args.seeds]
    def avg(path,sec): return sum(r[sec][path] for r in rec)/len(rec)
    summary={'related_holdout':{'sub':{k:avg(k,'related_holdout') for k in ['teacher','dart_zero','dart_adapted','mlp_control','gain_zero','gain_adapt','vs_mlp_adapt']}},'contrast_holdout':{'sort':{k:avg(k,'contrast_holdout') for k in ['teacher','dart_zero','dart_adapted','mlp_control','gain_zero','gain_adapt','vs_mlp_adapt']}},'source':{'avg_accuracy':sum(r['source_avg'] for r in rec)/len(rec),'avg_theta_effect':sum(r['theta_effect'] for r in rec)/len(rec),'avg_theta_stability':sum(r['theta_stability'] for r in rec)/len(rec)}}
    payload={'config':vars(args),'records':rec,'summary':summary}; Path(args.out).write_text(json.dumps(payload,indent=2)); print('DART-1.6: compositional factorized primitive discovery')
    for r in rec: print(f"seed={r['seed']} winner={r['winner']} eligible={r['eligible']} source_avg={r['source_avg']:.4f} source_worst={r['source_worst']:.4f} theta_effect={r['theta_effect']:.4f} stability={r['theta_stability']:.4f} core={r['core_params']}")
    print('================ DART-1.6 SUMMARY ================'); print(summary); print('Saved:',Path(args.out).resolve())
if __name__=='__main__': main()
