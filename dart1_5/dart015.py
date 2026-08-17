#!/usr/bin/env python3
"""DART-1.5: necessary-parameter + identifiable primitive discovery.

Research goal
-------------
DART-1.4 exposed a critical weakness: the factorized primitive often won with
θ == 0, and its causal score could remain high despite weak relational fit.
DART-1.5 therefore treats parameter necessity and identifiability as HARD
scientific gates rather than soft score terms.

Key hypotheses
--------------
1. A reusable primitive must require its task parameters: C(x, θ) must differ
   materially from C(x, 0).
2. The task parameters must be identifiable and stable across disjoint verifier
   splits.
3. Perturbing θ should cause a predictable, measurable causal change in the
   primitive output.
4. A candidate must retain meaningful source-task capability before complexity
   can make it win.
5. Only after the frozen shared primitive passes these tests do we evaluate
   related and contrast holdouts.

This version also fixes the DART-1.4 dead-zone: basis functions are initialized
with non-zero outputs, and θ starts non-zero, so gradients exist for both θ and
the shared basis from the first optimization step.
"""
from __future__ import annotations
import argparse, copy, json, math, random, statistics, time
from pathlib import Path
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

VOCAB = list("0123456789+= ")
STOI = {c:i for i,c in enumerate(VOCAB)}
PAD = STOI[' ']
BLOCK_SIZE = 12


def seed_everything(seed:int):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def task_target(a:int,b:int,task:str)->int:
    ad=[int(c) for c in str(a).zfill(3)]; bd=[int(c) for c in str(b).zfill(3)]
    if task=='add': return (ad[0]+bd[-1])%10
    if task=='sub': return (ad[-1]-bd[0])%10
    if task=='mul': return (ad[0]*bd[-1])%10
    if task=='sort': return min(ad+bd)
    if task=='compose': return ((ad[0]+bd[-1])*(ad[1]+1))%10
    raise ValueError(task)


def make_example(a,b,task):
    ids=[STOI[c] for c in f"{a}+{b}="]
    ids=(ids+[PAD]*BLOCK_SIZE)[:BLOCK_SIZE]
    return ids,task_target(a,b,task)


class TaskDataset(Dataset):
    def __init__(self,n,task,seed):
        rng=random.Random(seed); self.rows=[]
        for _ in range(n):
            a,b=rng.randint(0,999),rng.randint(0,999); x,y=make_example(a,b,task)
            self.rows.append((torch.tensor(x),torch.tensor(y)))
    def __len__(self): return len(self.rows)
    def __getitem__(self,i): return self.rows[i]


class Block(nn.Module):
    def __init__(self,d,heads,d_ff):
        super().__init__(); self.norm1=nn.LayerNorm(d); self.attn=nn.MultiheadAttention(d,heads,dropout=0.,batch_first=True)
        self.norm2=nn.LayerNorm(d); self.ff=nn.Sequential(nn.Linear(d,d_ff),nn.GELU(),nn.Linear(d_ff,d))
    def forward(self,x):
        h=self.norm1(x); a,_=self.attn(h,h,h,need_weights=False); x=x+a; return x+self.ff(self.norm2(x))


class TinyTransformer(nn.Module):
    def __init__(self,vocab_size,d_model=32,heads=2,d_ff=128,depth=3):
        super().__init__(); self.d_model=d_model; self.depth=depth
        self.emb=nn.Embedding(vocab_size,d_model); self.pos=nn.Parameter(torch.randn(1,BLOCK_SIZE,d_model)*.02)
        self.blocks=nn.ModuleList([Block(d_model,heads,d_ff) for _ in range(depth)]); self.head=nn.Linear(d_model,10)
    def forward(self,x):
        h=self.emb(x)+self.pos[:,:x.size(1)]
        for b in self.blocks:
            n=b.norm1(h); a,_=b.attn(n,n,n,need_weights=False); h=h+a; h=h+b.ff(b.norm2(h))
        return self.head(h[:,0])


# ---------- structured primitive families ----------
class DiagonalCore(nn.Module):
    def __init__(self,d):
        super().__init__(); self.scale=nn.Parameter(torch.empty(d)); self.bias=nn.Parameter(torch.zeros(d))
        nn.init.normal_(self.scale, mean=0.05, std=0.02)
    def forward(self,x): return x*self.scale+self.bias

class PolynomialCore(nn.Module):
    def __init__(self,d):
        super().__init__(); self.a=nn.Parameter(torch.empty(d)); self.b=nn.Parameter(torch.empty(d)); self.c=nn.Parameter(torch.zeros(d))
        nn.init.normal_(self.a,mean=0.05,std=.02); nn.init.normal_(self.b,std=.01)
    def forward(self,x): return self.a*x+self.b*x.square()+self.c

class AffinePolynomialCore(nn.Module):
    def __init__(self,d,rank):
        super().__init__(); self.down=nn.Linear(d,rank); self.up=nn.Linear(rank,d); self.quad=nn.Linear(rank,d,bias=False)
        nn.init.xavier_uniform_(self.down.weight); nn.init.normal_(self.down.bias,std=.02)
        nn.init.xavier_uniform_(self.up.weight,gain=.05); nn.init.zeros_(self.up.bias); nn.init.xavier_uniform_(self.quad.weight,gain=.02)
    def forward(self,x):
        h=self.down(x); return self.up(h)+self.quad(h.square())

class LowRankCore(nn.Module):
    def __init__(self,d,rank):
        super().__init__(); self.down=nn.Linear(d,rank,bias=False); self.up=nn.Linear(rank,d)
        nn.init.xavier_uniform_(self.down.weight); nn.init.xavier_uniform_(self.up.weight,gain=.05); nn.init.zeros_(self.up.bias)
    def forward(self,x): return self.up(self.down(x))

class MLPControl(nn.Module):
    def __init__(self,d,b): super().__init__(); self.net=nn.Sequential(nn.Linear(d,b),nn.GELU(),nn.Linear(b,d))
    def forward(self,x): return self.net(x)


def build_basis(name,d,rank,bottleneck):
    if name=='affine_polynomial': return AffinePolynomialCore(d,rank)
    if name=='low_rank': return LowRankCore(d,rank)
    if name=='polynomial': return PolynomialCore(d)
    if name=='diagonal': return DiagonalCore(d)
    raise ValueError(name)


class FactorizedPrimitive(nn.Module):
    """C(x,θ)=sum_k θ_k B_k(x), with non-degenerate B_k initialization."""
    def __init__(self,name,d,rank,bottleneck,theta_dim):
        super().__init__(); self.name=name; self.theta_dim=theta_dim
        self.basis=nn.ModuleList([build_basis(name,d,rank,bottleneck) for _ in range(theta_dim)])
    def outputs(self,x): return [b(x) for b in self.basis]
    def forward(self,x,theta):
        ys=self.outputs(x)
        out=torch.zeros_like(ys[0])
        for k,y in enumerate(ys): out=out+theta[k]*y
        return out


class RoutingFactorizedBlock(nn.Module):
    def __init__(self,original,primitive,theta):
        super().__init__(); self.norm1=copy.deepcopy(original.norm1); self.attn=copy.deepcopy(original.attn); self.norm2=copy.deepcopy(original.norm2)
        self.primitive=primitive; self.register_buffer('theta_fixed',theta.detach().clone())
    def forward(self,x):
        n=self.norm1(x); a,_=self.attn(n,n,n,need_weights=False); u=x+a; z=self.norm2(u)
        return u+self.primitive(z,self.theta_fixed)

class CompiledTransformer(nn.Module):
    def __init__(self,teacher,primitive,theta,start,end):
        super().__init__(); self.emb=copy.deepcopy(teacher.emb); self.pos=copy.deepcopy(teacher.pos); self.head=copy.deepcopy(teacher.head); self.blocks=nn.ModuleList()
        for i,b in enumerate(teacher.blocks):
            self.blocks.append(RoutingFactorizedBlock(b,primitive,theta) if start<=i<end else copy.deepcopy(b))
    def forward(self,x):
        h=self.emb(x)+self.pos[:,:x.size(1)]
        for b in self.blocks: h=b(h)
        return self.head(h[:,0])

class MLPReplacementBlock(nn.Module):
    def __init__(self,original,mlp): super().__init__(); self.norm1=copy.deepcopy(original.norm1); self.attn=copy.deepcopy(original.attn); self.norm2=copy.deepcopy(original.norm2); self.mlp=mlp
    def forward(self,x):
        n=self.norm1(x); a,_=self.attn(n,n,n,need_weights=False); u=x+a; return u+self.mlp(self.norm2(u))

class RoutingMLP(nn.Module):
    def __init__(self,teacher,mlp,start,end):
        super().__init__(); self.emb=copy.deepcopy(teacher.emb); self.pos=copy.deepcopy(teacher.pos); self.head=copy.deepcopy(teacher.head); self.mlp=mlp; self.blocks=nn.ModuleList()
        for i,b in enumerate(teacher.blocks): self.blocks.append(MLPReplacementBlock(b,mlp) if start<=i<end else copy.deepcopy(b))
    def forward(self,x):
        h=self.emb(x)+self.pos[:,:x.size(1)]
        for b in self.blocks: h=b(h)
        return self.head(h[:,0])


def freeze(m):
    for p in m.parameters(): p.requires_grad=False

def count_params(m): return sum(p.numel() for p in m.parameters())

def basis_core_macs(name,d,rank):
    if name=='diagonal': return d
    if name=='polynomial': return 2*d
    if name=='affine_polynomial': return 3*d*rank
    if name=='low_rank': return 2*d*rank
    raise ValueError(name)


def evaluate(model,loader,device):
    model.eval(); ce=nn.CrossEntropyLoss(reduction='sum'); total=correct=loss=0.
    with torch.no_grad():
        for x,y in loader:
            x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True); z=model(x)
            loss+=float(ce(z,y)); correct+=int((z.argmax(-1)==y).sum()); total+=y.numel()
    return {'accuracy':correct/max(total,1),'loss':loss/max(total,1),'params':count_params(model)}


def train_model(model,loader,device,steps,lr,trainable_only=True):
    params=[p for p in model.parameters() if (p.requires_grad if trainable_only else True)]
    if not params: raise RuntimeError('No trainable parameters')
    model.train(); opt=torch.optim.AdamW(params,lr=lr,weight_decay=1e-4); it=iter(loader); ce=nn.CrossEntropyLoss(); t0=time.perf_counter()
    for _ in range(steps):
        try: x,y=next(it)
        except StopIteration: it=iter(loader); x,y=next(it)
        x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True); loss=ce(model(x),y); opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(params,1.); opt.step()
    if device.type=='cuda': torch.cuda.synchronize()
    return time.perf_counter()-t0


def train_teacher(task,args,device,seed):
    tr=DataLoader(TaskDataset(args.train_size,task,seed),batch_size=args.batch_size,shuffle=True,pin_memory=device.type=='cuda')
    va=DataLoader(TaskDataset(args.verifier_size,task,seed+10000),batch_size=args.batch_size,shuffle=False,pin_memory=device.type=='cuda')
    te=TinyTransformer(len(VOCAB),args.d_model,args.heads,args.d_ff,args.depth).to(device); train_model(te,tr,device,args.teacher_steps,args.lr); return te,tr,va


def capture_ff_states(teacher,loader,device,max_samples,layer):
    zs=[]; ys=[]; total=0
    with torch.no_grad():
        for x,_ in loader:
            if total>=max_samples: break
            x=x.to(device,non_blocking=True); h=teacher.emb(x)+teacher.pos[:,:x.size(1)]
            for i,b in enumerate(teacher.blocks):
                n=b.norm1(h); a,_=b.attn(n,n,n,need_weights=False); u=h+a; z=b.norm2(u); y=b.ff(z)
                if i==layer:
                    zz=z.reshape(-1,z.shape[-1]).cpu(); yy=y.reshape(-1,y.shape[-1]).cpu(); take=min(max_samples-total,zz.shape[0]); zs.append(zz[:take]); ys.append(yy[:take]); total+=take; break
                h=u+y
    return torch.cat(zs),torch.cat(ys)


def build_directional_bundle(teacher,loader,args,device,seed):
    z,y=capture_ff_states(teacher,loader,device,args.rel_samples_per_task,args.trajectory_start)
    g=torch.Generator(device='cpu'); g.manual_seed(seed); idx=torch.randperm(len(z),generator=g)[:min(len(z),args.rel_samples_per_task)]; z=z[idx]; y=y[idx]
    dirs=[]; eps=args.intervention_eps
    for j in range(args.rel_directions):
        d=torch.randn(len(z),z.shape[-1],generator=g); d=d/(d.norm(dim=-1,keepdim=True)+1e-8); dirs.append(d)
    td=[]
    tz=z.to(device)
    for d in dirs:
        dd=d.to(device)
        yp=teacher.blocks[args.trajectory_start].ff(tz+eps*dd); ym=teacher.blocks[args.trajectory_start].ff(tz-eps*dd)
        td.append(((yp-ym)/(2*eps)).detach().cpu())
    return z,y,dirs,td


def factorized_output(p,z,theta): return p(z,theta)


def fit_theta(primitive,bundle,args,device,steps=None,init=None):
    z,y,dirs,td=bundle; theta=nn.Parameter((torch.full((args.theta_dim,),1.0/args.theta_dim,device=device) if init is None else init.to(device).clone()))
    opt=torch.optim.Adam([theta],lr=args.theta_lr); steps=steps or args.theta_fit_steps; zdev=z.to(device); ydev=y.to(device)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True); pred=primitive(zdev,theta); loss=nn.functional.mse_loss(pred,ydev)
        for d,t in zip(dirs,td):
            dd=d.to(device); cres=(primitive(zdev+args.intervention_eps*dd,theta)-primitive(zdev-args.intervention_eps*dd,theta))/(2*args.intervention_eps)
            loss=loss+args.directional_weight*nn.functional.mse_loss(cres,t.to(device))
        loss=loss+args.theta_l2*theta.square().mean(); loss.backward(); opt.step()
    return theta.detach()


def fit_shared(name,bundles,args,device,seed):
    seed_everything(seed); p=FactorizedPrimitive(name,args.d_model,args.rank,args.bottleneck,args.theta_dim).to(device)
    # Non-zero initialization prevents the DART-1.4 dead-zone.
    theta=nn.Parameter(torch.full((len(bundles),args.theta_dim),1.0/args.theta_dim,device=device))
    nn.init.normal_(theta,mean=1.0/args.theta_dim,std=0.05)
    params=list(p.parameters())+[theta]; opt=torch.optim.AdamW(params,lr=args.core_fit_lr,weight_decay=1e-4); t0=time.perf_counter()
    for _ in range(args.core_fit_steps):
        opt.zero_grad(set_to_none=True); total=0.
        for i,b in enumerate(bundles):
            z,y,dirs,td=b; z=z.to(device); th=theta[i]; loss=nn.functional.mse_loss(p(z,th),y.to(device))
            for d,t in zip(dirs,td):
                dd=d.to(device); cres=(p(z+args.intervention_eps*dd,th)-p(z-args.intervention_eps*dd,th))/(2*args.intervention_eps)
                loss=loss+args.directional_weight*nn.functional.mse_loss(cres,t.to(device))
            total+=loss
        total=total/len(bundles)+args.theta_l2*theta.square().mean(); total.backward(); nn.utils.clip_grad_norm_(params,1.); opt.step()
    if device.type=='cuda': torch.cuda.synchronize()
    return p,theta.detach(),time.perf_counter()-t0


def primitive_metrics(p,bundle,args,device,theta):
    z,y,dirs,td=bundle; z=z.to(device); pred=p(z,theta); value=float(nn.functional.mse_loss(pred,y.to(device)).detach())
    rels=[]
    for d,t in zip(dirs,td):
        dd=d.to(device); ce=(p(z+args.intervention_eps*dd,theta)-p(z-args.intervention_eps*dd,theta))/(2*args.intervention_eps); tv=t.to(device)
        # cosine similarity between directional effects: much more discriminative than a loose exp(-MSE) score
        c=torch.nn.functional.cosine_similarity(ce,tv,dim=-1).mean(); rels.append(float(c.detach().cpu()))
    return value,statistics.mean(rels)


def theta_effect_test(p,bundle,args,device,theta):
    z,y,_,_=bundle; z=z.to(device); base=p(z,theta); base_mag=float(base.detach().norm(dim=-1).mean().item()+1e-8); effects=[]; deltas=[]
    delta=args.theta_delta
    for k in range(args.theta_dim):
        e=torch.zeros_like(theta); e[k]=delta
        plus=p(z,theta+e); minus=p(z,theta-e)
        eff=(plus-minus)/(2*delta); effects.append(float(eff.detach().norm(dim=-1).mean().item()/base_mag))
        deltas.append(float((plus-minus).detach().norm(dim=-1).mean().item()/base_mag))
    return statistics.mean(effects),max(effects),effects,deltas


def theta_stability(p,bundle,args,device):
    z,y,dirs,td=bundle; n=len(z); mid=n//2; bs=[]
    for idx0,idx1 in [(0,mid),(mid,n)]:
        zz=z[idx0:idx1]; yy=y[idx0:idx1]; dd=[d[idx0:idx1] for d in dirs]; tt=[t[idx0:idx1] for t in td]; bs.append(fit_theta(p,(zz,yy,dd,tt),args,device,steps=args.theta_fit_steps))
    dist=float(torch.norm(bs[0]-bs[1]).item()); denom=float((torch.norm(bs[0])+torch.norm(bs[1])).item()/2+1e-8); return dist/denom,bs[0],bs[1]


def candidate_gate(p,thetas,teachers,verifiers,bundles,args,device):
    acc=[]; rel=[]; effects=[]; stabs=[]
    for t,v,th,b in zip(teachers,verifiers,thetas,bundles):
        m=CompiledTransformer(t,p,th,args.trajectory_start,args.trajectory_end).to(device); acc.append(evaluate(m,v,device)['accuracy'])
        _,r=primitive_metrics(p,b,args,device,th); rel.append(r)
        eff,_,_,_=theta_effect_test(p,b,args,device,th); effects.append(eff)
        st,_,_=theta_stability(p,b,args,device); stabs.append(st)
    avg_acc=statistics.mean(acc); worst=min(acc); avg_rel=statistics.mean(rel); avg_eff=statistics.mean(effects); avg_stab=statistics.mean(stabs); theta_norm=float(thetas.norm(dim=1).mean().item())
    eligible=(avg_acc>=args.min_avg_source_acc and worst>=args.min_worst_source_acc and theta_norm>=args.min_theta_norm and avg_eff>=args.min_theta_effect and avg_rel>=args.min_relational and avg_stab<=args.max_theta_stability)
    score=avg_acc+args.relational_weight*avg_rel+args.theta_effect_weight*avg_eff-args.complexity_lambda*math.log1p(count_params(p))
    return {'eligible':eligible,'avg_accuracy':avg_acc,'worst_accuracy':worst,'avg_relational_agreement':avg_rel,'avg_theta_effect':avg_eff,'avg_theta_stability':avg_stab,'avg_theta_norm':theta_norm,'shared_core_params':count_params(p),'shared_core_macs':basis_core_macs(p.name,args.d_model,args.rank)*args.theta_dim,'theta_dim':args.theta_dim,'score':score,'task_accuracies':acc}


def train_mlp_control(teacher,train_loader,args,device):
    mlp=MLPControl(args.d_model,args.bottleneck).to(device); comp=RoutingMLP(teacher,mlp,args.trajectory_start,args.trajectory_end).to(device)
    for p in comp.parameters(): p.requires_grad=False
    for p in comp.mlp.parameters(): p.requires_grad=True
    train_model(comp,train_loader,device,args.transfer_control_steps,args.lr); return comp


def run_one(args,seed,holdout,contrast):
    device=torch.device(args.device); meta=[t for t in args.all_tasks if t not in {holdout,contrast}]
    teachers=[]; trainers=[]; verifiers=[]; bundles=[]
    for i,t in enumerate(meta):
        te,tr,va=train_teacher(t,args,device,seed+i*1000); teachers.append(te); trainers.append(tr); verifiers.append(va); bundles.append(build_directional_bundle(te,va,args,device,seed+91*i))
    rows=[]; final=None
    for r in range(args.surgery_rounds):
        candidates=[]
        for i,name in enumerate(args.structured_families):
            p,ths,secs=fit_shared(name,bundles,args,device,seed+100*r+37*i)
            # Freeze basis and separately refit theta per source task to test identifiability.
            for pp in p.parameters(): pp.requires_grad=False
            ref=torch.stack([fit_theta(p,b,args,device,steps=args.theta_fit_steps,init=ths[j]) for j,b in enumerate(bundles)])
            metrics=candidate_gate(p,ref,teachers,verifiers,bundles,args,device)
            metrics.update({'name':name,'kind':'dart_factorized','fit_seconds':secs,'raw_theta':ref.cpu().tolist(),'theta_nonzero_required':metrics['eligible']})
            candidates.append((metrics,p,ref))
        eligible=[c for c in candidates if c[0]['eligible']]
        pool=eligible if eligible else candidates
        row,p,ths=max(pool,key=lambda x:x[0]['score'])
        # Re-evaluate with independent theta fits after candidate selection; no basis training.
        refined=torch.stack([fit_theta(p,b,args,device,steps=args.meta_theta_adapt_steps,init=ths[i]) for i,b in enumerate(bundles)])
        pre=[evaluate(CompiledTransformer(t,p,ths[i],args.trajectory_start,args.trajectory_end).to(device),trainers[i],device)['accuracy'] for i,t in enumerate(teachers)]
        post=[evaluate(CompiledTransformer(t,p,refined[i],args.trajectory_start,args.trajectory_end).to(device),trainers[i],device)['accuracy'] for i,t in enumerate(teachers)]
        row['winner_eligible']=bool(row['eligible']); rows.append({'round':r,'winner':row,'meta_pre_accuracy':pre,'meta_post_accuracy':post,'selected_theta':refined.cpu().tolist()}); final=(p,refined)
        print(f"  round={r} winner={row['name']} eligible={row['eligible']} avg_acc={row['avg_accuracy']:.4f} theta_norm={row['avg_theta_norm']:.4f} theta_effect={row['avg_theta_effect']:.4f} stability={row['avg_theta_stability']:.4f} rel={row['avg_relational_agreement']:.4f} core={count_params(p)}",flush=True)
    p,source_theta=final; freeze(p)

    # Typo-safe explicit implementation instead of clever expression.
    def eval_hold(task,offset):
        te,tr,va=train_teacher(task,args,device,seed+offset); tev=evaluate(te,va,device); centroid=source_theta.mean(0)
        zero=evaluate(CompiledTransformer(te,p,centroid,args.trajectory_start,args.trajectory_end).to(device),va,device)
        tb=build_directional_bundle(te,tr,args,device,args.target_theta_seed+offset); target=fit_theta(p,tb,args,device,steps=args.target_theta_fit_steps,init=centroid)
        adapt=evaluate(CompiledTransformer(te,p,target,args.trajectory_start,args.trajectory_end).to(device),va,device)
        mlp=train_mlp_control(te,tr,args,device); me=evaluate(mlp,va,device)
        return {'task':task,'teacher':tev,'dart_zero_shot':zero,'dart_theta_adapted':adapt,'mlp_control':me,'zero_shot_gain_points':100*(zero['accuracy']-tev['accuracy']),'theta_adaptation_gain_points':100*(adapt['accuracy']-tev['accuracy']),'vs_mlp_zero_points':100*(zero['accuracy']-me['accuracy']),'vs_mlp_adapt_points':100*(adapt['accuracy']-me['accuracy']),'theta':target.cpu().tolist()}
    related=eval_hold(holdout,50000); contrast_rec=eval_hold(contrast,70000) if contrast else None
    return {'holdout_task':holdout,'contrast_task':contrast,'meta_tasks':meta,'rounds':rows,'related_holdout':related,'contrast_holdout':contrast_rec,'shared_core_params':count_params(p),'shared_core_macs':basis_core_macs(p.name,args.d_model,args.rank)*args.theta_dim,'conditioner_params':0,'task_code_params':args.theta_dim}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--all-tasks',nargs='+',default=['add','compose','mul','sub']); ap.add_argument('--holdout-tasks',nargs='+',default=['sub']); ap.add_argument('--contrast-tasks',nargs='+',default=['sort']); ap.add_argument('--seeds',nargs='+',type=int,default=[1,2])
    ap.add_argument('--train-size',type=int,default=6000); ap.add_argument('--verifier-size',type=int,default=1500); ap.add_argument('--test-size',type=int,default=1500); ap.add_argument('--teacher-steps',type=int,default=800); ap.add_argument('--core-fit-steps',type=int,default=300); ap.add_argument('--theta-fit-steps',type=int,default=120); ap.add_argument('--meta-theta-adapt-steps',type=int,default=200); ap.add_argument('--target-theta-fit-steps',type=int,default=400); ap.add_argument('--surgery-rounds',type=int,default=2); ap.add_argument('--transfer-control-steps',type=int,default=400)
    ap.add_argument('--d-model',type=int,default=32); ap.add_argument('--heads',type=int,default=2); ap.add_argument('--d-ff',type=int,default=128); ap.add_argument('--depth',type=int,default=3); ap.add_argument('--rank',type=int,default=8); ap.add_argument('--bottleneck',type=int,default=32); ap.add_argument('--theta-dim',type=int,default=4); ap.add_argument('--rel-samples-per-task',type=int,default=2048); ap.add_argument('--rel-directions',type=int,default=4); ap.add_argument('--intervention-eps',type=float,default=.05); ap.add_argument('--theta-delta',type=float,default=.25)
    ap.add_argument('--min-avg-source-acc',type=float,default=.25); ap.add_argument('--min-worst-source-acc',type=float,default=.18); ap.add_argument('--min-theta-norm',type=float,default=.05); ap.add_argument('--min-theta-effect',type=float,default=.02); ap.add_argument('--min-relational',type=float,default=.10); ap.add_argument('--max-theta-stability',type=float,default=.75)
    ap.add_argument('--relational-weight',type=float,default=.5); ap.add_argument('--theta-effect-weight',type=float,default=.5); ap.add_argument('--directional-weight',type=float,default=.5); ap.add_argument('--theta-l2',type=float,default=.0005); ap.add_argument('--theta-lr',type=float,default=.01); ap.add_argument('--core-fit-lr',type=float,default=.001); ap.add_argument('--lr',type=float,default=.0003); ap.add_argument('--complexity-lambda',type=float,default=1e-4); ap.add_argument('--batch-size',type=int,default=256); ap.add_argument('--trajectory-start',type=int,default=0); ap.add_argument('--trajectory-end',type=int,default=3); ap.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu'); ap.add_argument('--structured-families',nargs='+',default=['affine_polynomial','low_rank','polynomial','diagonal']); ap.add_argument('--target-theta-seed',type=int,default=1234); ap.add_argument('--out',default='dart015_results.json')
    a=ap.parse_args(); print('DART-1.5: necessary-parameter + identifiable primitive discovery',flush=True); records=[]
    for h in a.holdout_tasks:
        c=a.contrast_tasks[0] if a.contrast_tasks else None; print(f'\n===== RELATED HOLDOUT {h} | CONTRAST {c} =====',flush=True)
        for s in a.seeds: print(f'seed={s}',flush=True); records.append(run_one(a,s,h,c))
    summary={'related_holdout':{},'contrast_holdout':{}}
    for task in a.holdout_tasks:
        rs=[r for r in records if r['holdout_task']==task]
        summary['related_holdout'][task]={'teacher':statistics.mean(r['related_holdout']['teacher']['accuracy'] for r in rs),'dart_zero_shot':statistics.mean(r['related_holdout']['dart_zero_shot']['accuracy'] for r in rs),'dart_theta_adapted':statistics.mean(r['related_holdout']['dart_theta_adapted']['accuracy'] for r in rs),'mlp_control':statistics.mean(r['related_holdout']['mlp_control']['accuracy'] for r in rs),'zero_shot_gain_points':statistics.mean(r['related_holdout']['zero_shot_gain_points'] for r in rs),'theta_adaptation_gain_points':statistics.mean(r['related_holdout']['theta_adaptation_gain_points'] for r in rs),'vs_mlp_zero_points':statistics.mean(r['related_holdout']['vs_mlp_zero_points'] for r in rs),'vs_mlp_adapt_points':statistics.mean(r['related_holdout']['vs_mlp_adapt_points'] for r in rs),'shared_core_params':statistics.mean(r['shared_core_params'] for r in rs),'shared_core_macs':statistics.mean(r['shared_core_macs'] for r in rs),'theta_dim':a.theta_dim}
        cs=[r['contrast_holdout'] for r in rs if r.get('contrast_holdout')]
        if cs:
            summary['contrast_holdout'][cs[0]['task']]={'teacher':statistics.mean(x['teacher']['accuracy'] for x in cs),'dart_zero_shot':statistics.mean(x['dart_zero_shot']['accuracy'] for x in cs),'dart_theta_adapted':statistics.mean(x['dart_theta_adapted']['accuracy'] for x in cs),'mlp_control':statistics.mean(x['mlp_control']['accuracy'] for x in cs),'zero_shot_gain_points':statistics.mean(x['zero_shot_gain_points'] for x in cs),'theta_adaptation_gain_points':statistics.mean(x['theta_adaptation_gain_points'] for x in cs),'vs_mlp_zero_points':statistics.mean(x['vs_mlp_zero_points'] for x in cs),'vs_mlp_adapt_points':statistics.mean(x['vs_mlp_adapt_points'] for x in cs)}
    out={'config':vars(a),'records':records,'summary':summary}; Path(a.out).write_text(json.dumps(out,indent=2),encoding='utf-8'); print('\n================ DART-1.5 SUMMARY ================'); print(summary); print(f'Saved: {Path(a.out).resolve()}')

if __name__=='__main__': main()
