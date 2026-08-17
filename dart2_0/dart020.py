#!/usr/bin/env python3
"""DART-2.0: computational rule-graph discovery + frozen rule transfer.

Core hypothesis:
  DART-1.x tried to transfer states/representations.
  DART-2.0 instead discovers a shared structured transformation rule graph G
  and only changes tiny explicit task parameters theta.

Experiments:
  - rule motifs: sequential, parallel_sum, residual_parallel
  - structured nodes: diagonal, polynomial, affine_polynomial, low_rank
  - source-task joint fitting of one shared graph G + per-task theta
  - rule intervention: perturb/disable one graph node and compare teacher/DART
  - theta permutation control: correct task->theta pairing vs permuted pairing
  - random-graph control with same parameter budget
  - frozen G transfer to related holdout and contrast task
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
    def __init__(self,d):
        super().__init__(); self.s=nn.Parameter(torch.randn(d)*.02+.2); self.b=nn.Parameter(torch.zeros(d))
    def forward(self,x): return x*self.s+self.b
class PolynomialCore(nn.Module):
    def __init__(self,d):
        super().__init__(); self.a=nn.Parameter(torch.randn(d)*.02+.2); self.b=nn.Parameter(torch.randn(d)*.01); self.c=nn.Parameter(torch.zeros(d))
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

class RuleGraph(nn.Module):
    """Structured rule graph with explicit task parameters.

    Each node is a structured core. theta gates each node. The graph motif
    controls how node outputs interact, while G itself is shared across tasks.
    """
    def __init__(self,motif,nodes,d,r):
        super().__init__(); self.motif=motif; self.node_names=tuple(nodes); self.nodes=nn.ModuleList([make_core(n,d,r) for n in nodes]); self.theta_dim=len(nodes)
        self.scale=nn.Parameter(torch.ones(len(nodes))*0.25)
    def forward(self,x,theta,disable=None,perturb=None):
        outs=[]
        base=x
        for i,node in enumerate(self.nodes):
            v=node(base)
            if disable is not None and i==disable:
                v=torch.zeros_like(v)
            if perturb is not None and i==perturb[0]:
                v=v + perturb[1]*torch.ones_like(v)
            outs.append(theta[i].view(1,1)*v)
        if self.motif=='sequential':
            h=x
            for i,node in enumerate(self.nodes):
                if disable is not None and i==disable: continue
                v=node(h)
                if perturb is not None and i==perturb[0]: v=v+perturb[1]*torch.ones_like(v)
                h=h+theta[i].view(1,1)*v
            return h
        if self.motif=='parallel_sum':
            return x + torch.stack(outs,dim=0).sum(0)
        if self.motif=='residual_parallel':
            return x + outs[0] + outs[1] + (outs[2] if len(outs)>2 else 0.) + 0.1*x*outs[0]
        raise ValueError(self.motif)

class RoutingCompiled(nn.Module):
    def __init__(self,teacher,graph,theta,start,end):
        super().__init__(); self.emb=copy.deepcopy(teacher.emb); self.pos=copy.deepcopy(teacher.pos); self.head=copy.deepcopy(teacher.head); self.blocks=nn.ModuleList([RoutingRuleBlock(b,graph,theta) if start<=i<end else copy.deepcopy(b) for i,b in enumerate(teacher.blocks)])
    def forward(self,x):
        h=self.emb(x)+self.pos[:,:x.size(1)]
        for b in self.blocks: h=b(h)
        return self.head(h[:,0])
class RoutingRuleBlock(nn.Module):
    def __init__(self,b,graph,theta):
        super().__init__(); self.norm1=copy.deepcopy(b.norm1); self.attn=copy.deepcopy(b.attn); self.norm2=copy.deepcopy(b.norm2); self.graph=graph; self.register_buffer('theta_fixed',theta.detach().clone())
    def forward(self,x):
        n=self.norm1(x); a,_=self.attn(n,n,n,need_weights=False); u=x+a; z=self.norm2(u); return u+self.graph(z,self.theta_fixed)
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
            x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True); z=model(x); correct+=int((z.argmax(-1)==y).sum()); total+=y.numel()
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
    zs=[]; ys=[]
    with torch.no_grad():
        for x,y in loader:
            have=sum(t.shape[0] for t in zs)
            if have>=maxn: break
            x=x.to(device); h=teacher.emb(x)+teacher.pos[:,:x.size(1)]
            for i,b in enumerate(teacher.blocks):
                n=b.norm1(h); a,_=b.attn(n,n,n,need_weights=False); u=h+a; z=b.norm2(u)
                if i==layer:
                    take=min(maxn-have,z.reshape(-1,z.shape[-1]).shape[0]); zs.append(z.reshape(-1,z.shape[-1])[:take].cpu()); ys.append(y[:take].cpu()); break
                h=u+b.ff(z)
    return torch.cat(zs), torch.cat(ys)

def source_bundle(teacher,loader,args,device):
    z,_=capture_ff(teacher,loader,device,args.rel_samples_per_task,args.trajectory_start)
    with torch.no_grad(): y=teacher.blocks[args.trajectory_start].ff(z.to(device)).detach().cpu()
    return z,y

def theta_fit(graph,z,y,args,device,steps,init=None):
    th=nn.Parameter(init.clone().to(device) if init is not None else torch.full((graph.theta_dim,),.5,device=device))
    opt=torch.optim.Adam([th],lr=args.theta_lr); z=z.to(device); y=y.to(device)
    for _ in range(steps):
        pred=graph(z,th); loss=((pred-y)**2).mean()+args.theta_l2*th.square().mean(); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    return th.detach()

def fit_graph(motif,nodes,teachers,args,device):
    g=RuleGraph(motif,nodes,args.d_model,args.rank).to(device)
    bundles={t:source_bundle(v[0],v[1],args,device) for t,v in teachers.items()}; tasks=list(bundles)
    theta_bank=nn.Parameter(torch.full((len(tasks),g.theta_dim),.5,device=device))
    opt=torch.optim.AdamW(list(g.parameters())+[theta_bank],lr=args.core_fit_lr,weight_decay=1e-4)
    for _ in range(args.core_fit_steps):
        total=0.
        for i,t in enumerate(tasks):
            z,y=bundles[t]; idx=torch.randperm(len(z))[:min(args.fit_batch_samples,len(z))]; total += ((g(z[idx].to(device),theta_bank[i])-y[idx].to(device))**2).mean()+args.theta_l2*theta_bank[i].square().mean()
        opt.zero_grad(set_to_none=True); total.backward(); nn.utils.clip_grad_norm_(list(g.parameters())+[theta_bank],1.); opt.step()
    for p in g.parameters(): p.requires_grad=False
    ths=[]; st=[]
    for i,t in enumerate(tasks):
        z,y=bundles[t]; th=theta_fit(g,z,y,args,device,args.theta_fit_steps,theta_bank[i]); ths.append(th)
        n=len(z); ta=theta_fit(g,z[:n//2],y[:n//2],args,device,max(10,args.theta_fit_steps//2),th); tb=theta_fit(g,z[n//2:],y[n//2:],args,device,max(10,args.theta_fit_steps//2),th); st.append(float((ta-tb).norm().item()))
    return g,torch.stack(ths),sum(st)/len(st),bundles

def rule_intervention_fidelity(graph,theta,teacher,loader,args,device,layer):
    x,_=next(iter(loader)); x=x.to(device)
    with torch.no_grad():
        h=teacher.emb(x)+teacher.pos[:,:x.size(1)]
        for i,b in enumerate(teacher.blocks):
            n=b.norm1(h); a,_=b.attn(n,n,n,need_weights=False); u=h+a; z=b.norm2(u)
            if i==layer:
                base=u+graph(z,theta); disabled=u+graph(z,theta,disable=0); perturbed=u+graph(z,theta,perturb=(0,args.rule_delta))
                def downstream(hh):
                    for j in range(i+1,len(teacher.blocks)): hh=teacher.blocks[j](hh)
                    return teacher.head(hh[:,0])
                y0=downstream(base); yd=downstream(disabled); yp=downstream(perturbed)
                d_disable=(yd-y0); d_pert=(yp-y0)
                local_disable=teacher.head(disabled[:,0])-teacher.head(base[:,0]); local_pert=teacher.head(perturbed[:,0])-teacher.head(base[:,0])
                c1=float(nn.functional.cosine_similarity(d_disable,local_disable,dim=-1).mean().item()); c2=float(nn.functional.cosine_similarity(d_pert,local_pert,dim=-1).mean().item()); return (c1+c2)/2
            h=b(h)
    return 0.

def theta_permutation_gap(graph,thetas,teachers,args,device):
    tasks=list(teachers); correct=[]; perm=[]
    order=list(range(len(tasks))); rng=random.Random(12345); rng.shuffle(order)
    for i,t in enumerate(tasks):
        tea,_,va=teachers[t]; cm=RoutingCompiled(tea,graph,thetas[i],args.trajectory_start,args.trajectory_end).to(device); cp=RoutingCompiled(tea,graph,thetas[order[i]],args.trajectory_start,args.trajectory_end).to(device); correct.append(evaluate(cm,va,device)); perm.append(evaluate(cp,va,device))
    return sum(correct)/len(correct)-sum(perm)/len(perm)

def run_seed(args,seed):
    device=torch.device(args.device if args.device=='cpu' or torch.cuda.is_available() else 'cpu'); seed_all(seed)
    meta=[t for t in args.all_tasks if t not in args.holdout_tasks and t not in args.contrast_tasks]
    teachers={t:train_teacher(t,args,device,seed*1000+i) for i,t in enumerate(meta)}
    contrast=args.contrast_tasks[0]; c_tea,c_tr,c_va=train_teacher(contrast,args,device,seed+70000)
    motifs=['sequential','parallel_sum','residual_parallel']
    node_families=[('affine_polynomial','polynomial'),('low_rank','diagonal'),('polynomial','diagonal')]
    scored=[]
    for motif in motifs:
        for nodes in node_families:
            g,th,stab,bundles=fit_graph(motif,nodes,teachers,args,device)
            acc=[evaluate(RoutingCompiled(teachers[t][0],g,th[i],args.trajectory_start,args.trajectory_end).to(device),teachers[t][2],device) for i,t in enumerate(meta)]
            rif=sum(rule_intervention_fidelity(g,th[i],teachers[t][0],teachers[t][1],args,device,args.trajectory_start) for i,t in enumerate(meta))/len(meta)
            perm_gap=theta_permutation_gap(g,th,teachers,args,device)
            # random graph: same motif/nodes, freshly initialized, same theta
            rg=RuleGraph(motif,nodes,args.d_model,args.rank).to(device)
            racc=[evaluate(RoutingCompiled(teachers[t][0],rg,th[i],args.trajectory_start,args.trajectory_end).to(device),teachers[t][2],device) for i,t in enumerate(meta)]
            random_graph_gap=sum(acc)/len(acc)-sum(racc)/len(racc)
            eligible=(sum(acc)/len(acc)>=args.min_avg_source_acc and min(acc)>=args.min_worst_source_acc and perm_gap>=args.min_theta_permutation_gap and rif>=args.min_rule_causal_fidelity and random_graph_gap>=args.min_random_graph_gap)
            score=sum(acc)/len(acc)+args.rule_weight*rif+args.permutation_weight*perm_gap+args.random_graph_weight*random_graph_gap-args.complexity_lambda*count_params(g)
            scored.append({'motif':motif,'nodes':nodes,'graph':g,'theta':th,'source_avg':sum(acc)/len(acc),'source_worst':min(acc),'theta_stability':stab,'rule_causal_fidelity':rif,'theta_permutation_gap':perm_gap,'random_graph_gap':random_graph_gap,'task_acc':acc,'eligible':eligible,'score':score})
    elig=[c for c in scored if c['eligible']]; best=max(elig,key=lambda c:c['score']) if elig else max(scored,key=lambda c:c['score'])
    results={}
    for label,t in [('related',args.holdout_tasks[0]),('contrast',contrast)]:
        tea,tr,va=(train_teacher(t,args,device,seed+50000) if label=='related' else (c_tea,c_tr,c_va))
        te=DataLoader(TaskDataset(args.test_size,t,seed+60000+(0 if label=='related' else 1)),batch_size=args.batch_size,shuffle=False,pin_memory=device.type=='cuda')
        z,y=source_bundle(tea,tr,args,device); th0=best['theta'].mean(0); a0=evaluate(RoutingCompiled(tea,best['graph'],th0,args.trajectory_start,args.trajectory_end).to(device),te,device); tht=theta_fit(best['graph'],z,y,args,device,args.target_theta_fit_steps,th0); a1=evaluate(RoutingCompiled(tea,best['graph'],tht,args.trajectory_start,args.trajectory_end).to(device),te,device)
        # permutation control at transfer: a random source theta instead of matched mean
        perm_theta=best['theta'][0]
        ap=evaluate(RoutingCompiled(tea,best['graph'],perm_theta,args.trajectory_start,args.trajectory_end).to(device),te,device)
        mlp=MLPControl(tea,args.trajectory_start,args.trajectory_end,args.mlp_width).to(device); train(mlp,tr,device,args.transfer_control_steps,args.lr); am=evaluate(mlp,te,device)
        results[label+'_holdout']={'task':t,'teacher':evaluate(tea,te,device),'dart_zero':a0,'dart_adapted':a1,'theta_permutation_control':ap,'mlp_control':am,'gain_zero':(a0-results[label+'_holdout']['teacher'] if False else a0-evaluate(tea,te,device))*100,'gain_adapt':(a1-evaluate(tea,te,device))*100,'vs_mlp_adapt':(a1-am)*100,'theta':tht.cpu().tolist()}
    clean=lambda c:{k:v for k,v in c.items() if k not in ('graph','theta')}
    return {'seed':seed,'winner':clean(best),'candidates':[clean(c) for c in scored],**results}

def main():
    ap=argparse.ArgumentParser(description='DART-2.0 computational rule-graph discovery')
    ap.add_argument('--seeds',nargs='+',type=int,default=[1,2]); ap.add_argument('--all-tasks',nargs='+',default=['add','compose','mul','sub']); ap.add_argument('--holdout-tasks',nargs='+',default=['sub']); ap.add_argument('--contrast-tasks',nargs='+',default=['sort'])
    for k,v in [('teacher-steps',800),('core-fit-steps',300),('theta-fit-steps',120),('target-theta-fit-steps',400),('transfer-control-steps',400),('train-size',6000),('verifier-size',1500),('test-size',1500),('rel-samples-per-task',2048),('fit-batch-samples',512),('d-model',32),('heads',2),('d-ff',128),('depth',3),('rank',8),('batch-size',256),('mlp-width',64)]: ap.add_argument('--'+k,type=int,default=v)
    ap.add_argument('--theta-delta',type=float,default=.25); ap.add_argument('--theta-l2',type=float,default=.0005); ap.add_argument('--theta-lr',type=float,default=.01); ap.add_argument('--rule-delta',type=float,default=.10); ap.add_argument('--min-avg-source-acc',type=float,default=.30); ap.add_argument('--min-worst-source-acc',type=float,default=.22); ap.add_argument('--max-theta-stability',type=float,default=.75); ap.add_argument('--min-theta-permutation-gap',type=float,default=.01); ap.add_argument('--min-rule-causal-fidelity',type=float,default=.20); ap.add_argument('--min-random-graph-gap',type=float,default=.02); ap.add_argument('--rule-weight',type=float,default=.35); ap.add_argument('--permutation-weight',type=float,default=.30); ap.add_argument('--random-graph-weight',type=float,default=.30); ap.add_argument('--complexity-lambda',type=float,default=1e-5); ap.add_argument('--trajectory-start',type=int,default=0); ap.add_argument('--trajectory-end',type=int,default=1); ap.add_argument('--lr',type=float,default=.0003); ap.add_argument('--core-fit-lr',type=float,default=.001); ap.add_argument('--device',default='cuda'); ap.add_argument('--out',default='dart020_results.json')
    args=ap.parse_args(); rec=[run_seed(args,s) for s in args.seeds]
    def av(k,sec): return sum(r[sec][k] for r in rec)/len(rec)
    summary={'related_holdout':{args.holdout_tasks[0]:{k:av(k,'related_holdout') for k in ['teacher','dart_zero','dart_adapted','theta_permutation_control','mlp_control','gain_zero','gain_adapt','vs_mlp_adapt']}},'contrast_holdout':{args.contrast_tasks[0]:{k:av(k,'contrast_holdout') for k in ['teacher','dart_zero','dart_adapted','theta_permutation_control','mlp_control','gain_zero','gain_adapt','vs_mlp_adapt']}},'source':{'avg_accuracy':sum(r['winner']['source_avg'] for r in rec)/len(rec),'avg_theta_stability':sum(r['winner']['theta_stability'] for r in rec)/len(rec),'avg_rule_causal_fidelity':sum(r['winner']['rule_causal_fidelity'] for r in rec)/len(rec),'avg_theta_permutation_gap':sum(r['winner']['theta_permutation_gap'] for r in rec)/len(rec),'avg_random_graph_gap':sum(r['winner']['random_graph_gap'] for r in rec)/len(rec)}}
    payload={'config':vars(args),'records':rec,'summary':summary}; Path(args.out).write_text(json.dumps(payload,indent=2)); print('DART-2.0: computational rule-graph discovery'); print(json.dumps(summary,indent=2)); print('Saved:',Path(args.out).resolve())
if __name__=='__main__': main()
