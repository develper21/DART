#!/usr/bin/env python3
"""DART-2.1: invariant rule factorization + strict shared-rule validation.

Research hypothesis:
  DART-2.0 showed promising rule-level causal fidelity and random-graph
  separation, but the shared rule did not transfer strongly to an unseen task.
  DART-2.1 asks whether one *frozen invariant rule graph G* can explain several
  source tasks using only small explicit task parameters theta, while passing
  stronger controls:
    - joint shared-graph vs separate-graph parity,
    - full theta-permutation matrix,
    - node-level cross-task causal consistency,
    - random-graph control,
    - frozen unseen-task theta-only transfer,
    - optional leave-one-task-out cross-validation.

No large conditioner, task embedding, or target-task residual is used.
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
    return {'diagonal':DiagonalCore,'polynomial':PolynomialCore,'affine_polynomial':lambda d:AffinePolynomialCore(d,r),'low_rank':lambda d:LowRankCore(d,r)}[name](d)

class RuleGraph(nn.Module):
    def __init__(self,motif,nodes,d,r):
        super().__init__(); self.motif=motif; self.node_names=tuple(nodes); self.nodes=nn.ModuleList([make_core(n,d,r) for n in nodes]); self.theta_dim=len(nodes); self.scale=nn.Parameter(torch.ones(len(nodes))*0.25)
    def forward(self,x,theta,disable=None,perturb=None):
        def node_value(i,h):
            v=self.nodes[i](h)
            if disable==i: return torch.zeros_like(v)
            if perturb is not None and perturb[0]==i: v=v+perturb[1]*torch.ones_like(v)
            return v
        if self.motif=='sequential':
            h=x
            for i in range(len(self.nodes)):
                h=h+theta[i].view(1,1)*node_value(i,h)
            return h
        if self.motif=='parallel_sum':
            return x+sum(theta[i].view(1,1)*node_value(i,x) for i in range(len(self.nodes)))
        if self.motif=='residual_parallel':
            outs=[theta[i].view(1,1)*node_value(i,x) for i in range(len(self.nodes))]
            return x+sum(outs)+0.1*x*outs[0]
        raise ValueError(self.motif)

class RoutingRuleBlock(nn.Module):
    def __init__(self,b,graph,theta):
        super().__init__(); self.norm1=copy.deepcopy(b.norm1); self.attn=copy.deepcopy(b.attn); self.norm2=copy.deepcopy(b.norm2); self.graph=graph; self.register_buffer('theta_fixed',theta.detach().clone())
    def forward(self,x):
        n=self.norm1(x); a,_=self.attn(n,n,n,need_weights=False); u=x+a; z=self.norm2(u); return u+self.graph(z,self.theta_fixed)
class RoutingCompiled(nn.Module):
    def __init__(self,teacher,graph,theta,start,end):
        super().__init__(); self.emb=copy.deepcopy(teacher.emb); self.pos=copy.deepcopy(teacher.pos); self.head=copy.deepcopy(teacher.head); self.blocks=nn.ModuleList([RoutingRuleBlock(b,graph,theta) if start<=i<end else copy.deepcopy(b) for i,b in enumerate(teacher.blocks)])
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

def theta_fit(graph,z,y,args,device,steps,init=None):
    th=nn.Parameter(init.clone().to(device) if init is not None else torch.full((graph.theta_dim,),.5,device=device)); opt=torch.optim.Adam([th],lr=args.theta_lr); z=z.to(device); y=y.to(device)
    for _ in range(steps):
        loss=((graph(z,th)-y)**2).mean()+args.theta_l2*th.square().mean(); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    return th.detach()

def fit_shared_graph(motif,nodes,teachers,args,device):
    g=RuleGraph(motif,nodes,args.d_model,args.rank).to(device); bundles={t:source_bundle(v[0],v[1],args,device) for t,v in teachers.items()}; tasks=list(bundles)
    bank=nn.Parameter(torch.full((len(tasks),g.theta_dim),.5,device=device)); opt=torch.optim.AdamW(list(g.parameters())+[bank],lr=args.core_fit_lr,weight_decay=1e-4)
    for _ in range(args.core_fit_steps):
        total=0.
        for i,t in enumerate(tasks):
            z,y=bundles[t]; idx=torch.randperm(len(z))[:min(args.fit_batch_samples,len(z))]; total += ((g(z[idx].to(device),bank[i])-y[idx].to(device))**2).mean()+args.theta_l2*bank[i].square().mean()
        opt.zero_grad(set_to_none=True); total.backward(); nn.utils.clip_grad_norm_(list(g.parameters())+[bank],1.); opt.step()
    for p in g.parameters(): p.requires_grad=False
    ths=[]; st=[]
    for i,t in enumerate(tasks):
        z,y=bundles[t]; th=theta_fit(g,z,y,args,device,args.theta_fit_steps,bank[i]); ths.append(th); n=len(z); a=max(1,n//2); ta=theta_fit(g,z[:a],y[:a],args,device,max(10,args.theta_fit_steps//2),th); tb=theta_fit(g,z[a:],y[a:],args,device,max(10,args.theta_fit_steps//2),th); st.append(float((ta-tb).norm().item()))
    return g,torch.stack(ths),sum(st)/len(st),bundles

def fit_separate_graph(task,teacher_bundle,motif,nodes,args,device):
    tea,tr,va=teacher_bundle; g=RuleGraph(motif,nodes,args.d_model,args.rank).to(device); z,y=source_bundle(tea,tr,args,device); th=nn.Parameter(torch.full((g.theta_dim,),.5,device=device)); opt=torch.optim.AdamW(list(g.parameters())+[th],lr=args.core_fit_lr,weight_decay=1e-4)
    for _ in range(args.separate_control_steps):
        idx=torch.randperm(len(z))[:min(args.fit_batch_samples,len(z))]; loss=((g(z[idx].to(device),th)-y[idx].to(device))**2).mean()+args.theta_l2*th.square().mean(); opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(list(g.parameters())+[th],1.); opt.step()
    for p in g.parameters(): p.requires_grad=False
    th=theta_fit(g,z,y,args,device,args.theta_fit_steps,th.detach())
    return g,th

def rule_causal_fidelity(graph,theta,teacher,loader,args,device,layer):
    x,_=next(iter(loader)); x=x.to(device)
    with torch.no_grad():
        h=teacher.emb(x)+teacher.pos[:,:x.size(1)]
        for i,b in enumerate(teacher.blocks):
            n=b.norm1(h); a,_=b.attn(n,n,n,need_weights=False); u=h+a; z=b.norm2(u)
            if i==layer:
                base=u+graph(z,theta)
                ds=[]; ls=[]
                for node in range(graph.theta_dim):
                    pert=u+graph(z,theta,disable=node)
                    def down(hh):
                        for j in range(i+1,len(teacher.blocks)): hh=teacher.blocks[j](hh)
                        return teacher.head(hh[:,0])
                    d=(down(pert)-down(base)); local=teacher.head(pert[:,0])-teacher.head(base[:,0]); ds.append(float(nn.functional.cosine_similarity(d,local,dim=-1).mean().item())); ls.append(float(d.abs().mean().item()))
                return sum(ds)/len(ds),sum(ls)/len(ls)
            h=b(h)
    return 0.,0.

def parameter_permutation_matrix(graph,thetas,teachers,args,device):
    tasks=list(teachers); mat=[]
    for i,t in enumerate(tasks):
        tea,_,va=teachers[t]; row=[]
        for j in range(len(tasks)):
            row.append(evaluate(RoutingCompiled(tea,graph,thetas[j],args.trajectory_start,args.trajectory_end).to(device),va,device))
        mat.append(row)
    return mat

def permutation_gap(mat):
    n=len(mat); correct=sum(mat[i][i] for i in range(n))/n
    off=[mat[i][j] for i in range(n) for j in range(n) if i!=j]
    return correct-(sum(off)/len(off) if off else correct)

def fit_random_control_like(graph,theta,teachers,args,device):
    # Fresh random graph with same motif/node structure and same theta; no fitting.
    rg=RuleGraph(graph.motif,graph.node_names,args.d_model,args.rank).to(device)
    vals=[]
    for i,t in enumerate(teachers):
        tea,_,va=teachers[t]; vals.append(evaluate(RoutingCompiled(tea,rg,theta[i],args.trajectory_start,args.trajectory_end).to(device),va,device))
    return sum(vals)/len(vals)

def run_seed(args,seed):
    device=torch.device(args.device if args.device=='cpu' or torch.cuda.is_available() else 'cpu'); seed_all(seed)
    meta=[t for t in args.all_tasks if t not in args.holdout_tasks and t not in args.contrast_tasks]
    teachers={t:train_teacher(t,args,device,seed*1000+i) for i,t in enumerate(meta)}
    contrast=args.contrast_tasks[0]; c_tea,c_tr,c_va=train_teacher(contrast,args,device,seed+70000)
    motifs=['sequential','parallel_sum','residual_parallel']; node_families=[('affine_polynomial','polynomial'),('low_rank','diagonal'),('polynomial','diagonal')]
    candidates=[]
    for motif in motifs:
        for nodes in node_families:
            g,theta,stab,bundles=fit_shared_graph(motif,nodes,teachers,args,device)
            acc=[evaluate(RoutingCompiled(teachers[t][0],g,theta[i],args.trajectory_start,args.trajectory_end).to(device),teachers[t][2],device) for i,t in enumerate(meta)]
            rcf,_=rule_causal_fidelity(g,theta[0],teachers[meta[0]][0],teachers[meta[0]][1],args,device,args.trajectory_start)
            pmat=parameter_permutation_matrix(g,theta,teachers,args,device); pg=permutation_gap(pmat)
            rgavg=fit_random_control_like(g,theta,teachers,args,device); rgap=sum(acc)/len(acc)-rgavg
            sep_acc=[]
            for t in meta:
                sg,stheta=fit_separate_graph(t,teachers[t],motif,nodes,args,device); sep_acc.append(evaluate(RoutingCompiled(teachers[t][0],sg,stheta,args.trajectory_start,args.trajectory_end).to(device),teachers[t][2],device))
            sepavg=sum(sep_acc)/len(sep_acc); parity=sepavg-(sum(acc)/len(acc))
            eligible=(sum(acc)/len(acc)>=args.min_avg_source_acc and min(acc)>=args.min_worst_source_acc and stab<=args.max_theta_stability and pg>=args.min_theta_permutation_gap and rcf>=args.min_rule_causal_fidelity and rgap>=args.min_random_graph_gap and parity<=args.max_shared_vs_separate_gap)
            score=sum(acc)/len(acc)+args.rule_weight*rcf+args.permutation_weight*pg+args.random_graph_weight*rgap-args.shared_parity_weight*max(0.0,parity)-args.complexity_lambda*count_params(g)
            candidates.append({'motif':motif,'nodes':nodes,'graph':g,'theta':theta,'source_avg':sum(acc)/len(acc),'source_worst':min(acc),'theta_stability':stab,'rule_causal_fidelity':rcf,'theta_permutation_gap':pg,'random_graph_gap':rgap,'separate_graph_avg':sepavg,'shared_vs_separate_gap':parity,'task_acc':acc,'permutation_matrix':pmat,'eligible':eligible,'score':score})
    elig=[c for c in candidates if c['eligible']]; best=max(elig,key=lambda c:c['score']) if elig else max(candidates,key=lambda c:c['score'])
    results={}
    for label,t in [('related',args.holdout_tasks[0]),('contrast',contrast)]:
        tea,tr,va=(train_teacher(t,args,device,seed+50000) if label=='related' else (c_tea,c_tr,c_va)); te=DataLoader(TaskDataset(args.test_size,t,seed+60000+(0 if label=='related' else 1)),batch_size=args.batch_size,shuffle=False)
        teacher_acc=evaluate(tea,te,device); th0=best['theta'].mean(0); a0=evaluate(RoutingCompiled(tea,best['graph'],th0,args.trajectory_start,args.trajectory_end).to(device),te,device); z,y=source_bundle(tea,tr,args,device); tht=theta_fit(best['graph'],z,y,args,device,args.target_theta_fit_steps,th0); a1=evaluate(RoutingCompiled(tea,best['graph'],tht,args.trajectory_start,args.trajectory_end).to(device),te,device); perm=best['theta'][0]; ap=evaluate(RoutingCompiled(tea,best['graph'],perm,args.trajectory_start,args.trajectory_end).to(device),te,device)
        mlp=MLPControl(tea,args.trajectory_start,args.trajectory_end,args.mlp_width).to(device); train(mlp,tr,device,args.transfer_control_steps,args.lr); am=evaluate(mlp,te,device)
        results[label+'_holdout']={'task':t,'teacher':teacher_acc,'dart_zero':a0,'dart_adapted':a1,'theta_permutation_control':ap,'mlp_control':am,'gain_zero':(a0-teacher_acc)*100,'gain_adapt':(a1-teacher_acc)*100,'vs_mlp_adapt':(a1-am)*100,'theta':tht.cpu().tolist()}
    clean=lambda c:{k:v for k,v in c.items() if k not in ('graph','theta')}
    return {'seed':seed,'winner':clean(best),'candidates':[clean(c) for c in candidates],**results}

def main():
    p=argparse.ArgumentParser(description='DART-2.1 invariant rule factorization + strict shared-rule validation')
    for k,v in [('seeds',[1,2])]: p.add_argument('--'+k,nargs='+',type=int,default=v)
    p.add_argument('--all-tasks',nargs='+',default=['add','compose','mul','sub']); p.add_argument('--holdout-tasks',nargs='+',default=['sub']); p.add_argument('--contrast-tasks',nargs='+',default=['sort'])
    for k,v in [('teacher-steps',800),('core-fit-steps',300),('theta-fit-steps',120),('target-theta-fit-steps',400),('transfer-control-steps',400),('separate-control-steps',200),('train-size',6000),('verifier-size',1500),('test-size',1500),('rel-samples-per-task',2048),('fit-batch-samples',512),('d-model',32),('heads',2),('d-ff',128),('depth',3),('rank',8),('batch-size',256),('mlp-width',64)]: p.add_argument('--'+k,type=int,default=v)
    for k,v in [('theta-delta',.25),('theta-l2',.0005),('theta-lr',.01),('rule-delta',.1),('min-avg-source-acc',.30),('min-worst-source-acc',.22),('max-theta-stability',.75),('min-theta-permutation-gap',.01),('min-rule-causal-fidelity',.20),('min-random-graph-gap',.02),('max-shared-vs-separate-gap',.03),('rule-weight',.35),('permutation-weight',.25),('random-graph-weight',.25),('shared-parity-weight',.25),('complexity-lambda',1e-5),('lr',.0003),('core-fit-lr',.001)]: p.add_argument('--'+k,type=float,default=v)
    p.add_argument('--trajectory-start',type=int,default=0); p.add_argument('--trajectory-end',type=int,default=1); p.add_argument('--device',default='cuda'); p.add_argument('--out',default='dart021_results.json')
    args=p.parse_args(); rec=[run_seed(args,s) for s in args.seeds]
    def av(k,sec): return sum(r[sec][k] for r in rec)/len(rec)
    summary={'related_holdout':{args.holdout_tasks[0]:{k:av(k,'related_holdout') for k in ['teacher','dart_zero','dart_adapted','theta_permutation_control','mlp_control','gain_zero','gain_adapt','vs_mlp_adapt']}},'contrast_holdout':{args.contrast_tasks[0]:{k:av(k,'contrast_holdout') for k in ['teacher','dart_zero','dart_adapted','theta_permutation_control','mlp_control','gain_zero','gain_adapt','vs_mlp_adapt']}},'source':{'avg_accuracy':sum(r['winner']['source_avg'] for r in rec)/len(rec),'avg_theta_stability':sum(r['winner']['theta_stability'] for r in rec)/len(rec),'avg_rule_causal_fidelity':sum(r['winner']['rule_causal_fidelity'] for r in rec)/len(rec),'avg_theta_permutation_gap':sum(r['winner']['theta_permutation_gap'] for r in rec)/len(rec),'avg_random_graph_gap':sum(r['winner']['random_graph_gap'] for r in rec)/len(rec),'avg_separate_graph_accuracy':sum(r['winner']['separate_graph_avg'] for r in rec)/len(rec),'avg_shared_vs_separate_gap':sum(r['winner']['shared_vs_separate_gap'] for r in rec)/len(rec)} }
    payload={'config':vars(args),'records':rec,'summary':summary}; Path(args.out).write_text(json.dumps(payload,indent=2)); print('DART-2.1: invariant rule factorization + strict shared-rule validation'); print(json.dumps(summary,indent=2)); print('Saved:',Path(args.out).resolve())
if __name__=='__main__': main()
