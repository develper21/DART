#!/usr/bin/env python3
"""DART-2.2: structured task-operator discovery over a shared rule graph.

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
DART-2.2 replaces weak scalar theta configuration with a tiny structured task operator grammar.
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
        super().__init__(); self.motif=motif; self.node_names=tuple(nodes); self.nodes=nn.ModuleList([make_core(n,d,r) for n in nodes])
    def node_values(self,x):
        return [node(x) for node in self.nodes]
    def base(self,x):
        vals=self.node_values(x)
        if self.motif=='sequential':
            h=x
            for v in vals: h=h+v
            return h
        if self.motif=='parallel_sum': return x+sum(vals)
        if self.motif=='residual_parallel': return x+sum(vals)+0.1*x*vals[0]
        raise ValueError(self.motif)
    def forward(self,x): return self.base(x)

class TaskOperator(nn.Module):
    """Tiny structured operator that configures a shared rule graph.
    The graph is frozen before final source/holdout evaluation; only these
    operator parameters are task-specific. The operator family is explicit,
    not a learned embedding or conditioner."""
    MODES=('identity','scale','negate','difference','product','mix')
    def __init__(self,mode,d):
        super().__init__(); self.mode=mode; self.d=d
        self.raw=nn.Parameter(torch.tensor([0.25,0.25,0.0,0.0]))
    @property
    def num_params(self): return 4
    def forward(self,vals):
        x=vals[0]
        if self.mode=='identity': return x
        if self.mode=='scale': return vals[0]*(1.0+self.raw[0]) + self.raw[1]
        if self.mode=='negate': return -vals[0]*(1.0+self.raw[0]) + self.raw[1]
        if self.mode=='difference':
            if len(vals)<2: return vals[0]
            return vals[0]*(1.0+self.raw[0]) - vals[1]*(1.0+self.raw[1])
        if self.mode=='product':
            if len(vals)<2: return vals[0]
            g=torch.tanh(self.raw[0]); return vals[0]*(1.0+g*vals[1]) + self.raw[1]
        if self.mode=='mix':
            if len(vals)<2: return vals[0]
            a=torch.sigmoid(self.raw[0]); return a*vals[0] + (1-a)*vals[1] + self.raw[1]
        raise ValueError(self.mode)

class SharedTaskRule(nn.Module):
    def __init__(self,motif,nodes,operator_mode,d,r):
        super().__init__(); self.motif=motif; self.node_names=tuple(nodes); self.graph=RuleGraph(motif,nodes,d,r); self.operator=TaskOperator(operator_mode,d)
    def forward(self,x): return self.operator(self.graph.node_values(x))

class RoutingRuleBlock(nn.Module):
    def __init__(self,b,rule):
        super().__init__(); self.norm1=copy.deepcopy(b.norm1); self.attn=copy.deepcopy(b.attn); self.norm2=copy.deepcopy(b.norm2); self.rule=rule
    def forward(self,x):
        n=self.norm1(x); a,_=self.attn(n,n,n,need_weights=False); u=x+a; z=self.norm2(u); return u+self.rule(z)
class RoutingCompiled(nn.Module):
    def __init__(self,teacher,rule,start,end):
        super().__init__(); self.emb=copy.deepcopy(teacher.emb); self.pos=copy.deepcopy(teacher.pos); self.head=copy.deepcopy(teacher.head); self.blocks=nn.ModuleList([RoutingRuleBlock(b,rule) if start<=i<end else copy.deepcopy(b) for i,b in enumerate(teacher.blocks)])
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

def operator_fit(graph, mode, z, y, args, device, steps, init_state=None):
    op=TaskOperator(mode,args.d_model).to(device)
    if init_state is not None: op.load_state_dict(init_state)
    opt=torch.optim.Adam(op.parameters(),lr=args.operator_lr); z=z.to(device); y=y.to(device)
    graph.eval()
    for p in graph.parameters(): p.requires_grad=False
    for _ in range(steps):
        with torch.no_grad(): vals=graph.node_values(z)
        pred=op(vals); loss=((pred-y)**2).mean()+args.operator_l2*sum(p.square().mean() for p in op.parameters())
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    return op

def operator_stability(graph, mode, z, y, op, args, device):
    n=len(z); a=max(1,n//2)
    oa=operator_fit(graph,mode,z[:a],y[:a],args,device,max(10,args.operator_fit_steps//2),operator_state(op))
    ob=operator_fit(graph,mode,z[a:],y[a:],args,device,max(10,args.operator_fit_steps//2),operator_state(op))
    return sum(float((pa-pb).norm().item()) for pa,pb in zip(oa.parameters(),ob.parameters()))

def freeze_rule(rule):
    for p in rule.graph.parameters(): p.requires_grad=False

def operator_state(op): return {k:v.detach().clone() for k,v in op.state_dict().items()}

def fit_shared_graph(motif,nodes,op_mode,teachers,args,device):
    graph=RuleGraph(motif,nodes,args.d_model,args.rank).to(device)
    bundles={t:source_bundle(v[0],v[1],args,device) for t,v in teachers.items()}; tasks=list(bundles)
    ops=nn.ModuleList([TaskOperator(op_mode,args.d_model) for _ in tasks]).to(device)
    params=list(graph.parameters())+list(ops.parameters()); opt=torch.optim.AdamW(params,lr=args.core_fit_lr,weight_decay=1e-4)
    for _ in range(args.core_fit_steps):
        total=0.
        for i,t in enumerate(tasks):
            z,y=bundles[t]; idx=torch.randperm(len(z))[:min(args.fit_batch_samples,len(z))]; zz=z[idx].to(device); yy=y[idx].to(device)
            vals=graph.node_values(zz); pred=ops[i](vals); total += ((pred-yy)**2).mean()+args.operator_l2*sum(p.square().mean() for p in ops[i].parameters())
        opt.zero_grad(set_to_none=True); total.backward(); nn.utils.clip_grad_norm_(params,1.); opt.step()
    for p in graph.parameters(): p.requires_grad=False
    final_ops=[]; st=[]
    for i,t in enumerate(tasks):
        z,y=bundles[t]; op=operator_fit(graph,op_mode,z,y,args,device,args.operator_fit_steps,operator_state(ops[i])); final_ops.append(op)
        st.append(operator_stability(graph,op_mode,z,y,op,args,device))
    return graph,final_ops,sum(st)/len(st),bundles

def make_rule(graph,op):
    r=SharedTaskRule(graph.motif,graph.node_names,op.mode,graph.nodes[0].s.shape[0] if hasattr(graph.nodes[0],'s') else graph.nodes[0].down.out_features if hasattr(graph.nodes[0],'down') and hasattr(graph.nodes[0].down,'out_features') else graph.nodes[0].a.shape[0] if hasattr(graph.nodes[0],'a') else graph.nodes[0].up.out_features, 8)
    r.graph=graph; r.operator=copy.deepcopy(op); return r

def fit_separate_graph(task,teacher_bundle,motif,nodes,op_mode,args,device):
    tea,tr,va=teacher_bundle; graph=RuleGraph(motif,nodes,args.d_model,args.rank).to(device); op=TaskOperator(op_mode,args.d_model).to(device); z,y=source_bundle(tea,tr,args,device)
    params=list(graph.parameters())+list(op.parameters()); opt=torch.optim.AdamW(params,lr=args.core_fit_lr,weight_decay=1e-4)
    for _ in range(args.separate_control_steps):
        idx=torch.randperm(len(z))[:min(args.fit_batch_samples,len(z))]; zz=z[idx].to(device); yy=y[idx].to(device); pred=op(graph.node_values(zz)); loss=((pred-yy)**2).mean()+args.operator_l2*sum(p.square().mean() for p in op.parameters()); opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(params,1.); opt.step()
    for p in graph.parameters(): p.requires_grad=False
    op=operator_fit(graph,op_mode,z,y,args,device,args.operator_fit_steps,operator_state(op)); r=SharedTaskRule(motif,nodes,op_mode,args.d_model,args.rank); r.graph=graph; r.operator=op
    return r

def rule_param_effect(rule,z,args,device):
    base={k:v.clone() for k,v in rule.operator.state_dict().items()}; vals=[]
    with torch.no_grad(): y0=rule(z.to(device));
    for k,v in list(base.items()):
        if v.numel()==0: continue
        pert={kk:vv.clone() for kk,vv in base.items()}; pert[k]=pert[k]+args.operator_delta
        rule.operator.load_state_dict(pert); withp=rule(z.to(device)); vals.append(float((withp-y0).norm().item()/max(z.shape[0],1)))
    rule.operator.load_state_dict(base); return sum(vals)/len(vals) if vals else 0.

def operator_permutation_matrix(graph,ops,teachers,args,device):
    tasks=list(teachers); mat=[]
    for i,t in enumerate(tasks):
        tea,_,va=teachers[t]; row=[]
        for j in range(len(tasks)):
            rule=SharedTaskRule(graph.motif,graph.node_names,ops[j].mode,args.d_model,args.rank).to(device); rule.graph=graph; rule.operator=copy.deepcopy(ops[j]).to(device); row.append(evaluate(RoutingCompiled(tea,rule,args.trajectory_start,args.trajectory_end).to(device),va,device))
        mat.append(row)
    return mat

def operator_permutation_gap(mat):
    n=len(mat); correct=sum(mat[i][i] for i in range(n))/n; off=[mat[i][j] for i in range(n) for j in range(n) if i!=j]; return correct-(sum(off)/len(off) if off else correct)

def rule_causal_fidelity(rule,teacher,loader,args,device,layer):
    x,_=next(iter(loader)); x=x.to(device)
    with torch.no_grad():
        h=teacher.emb(x)+teacher.pos[:,:x.size(1)]
        for i,b in enumerate(teacher.blocks):
            n=b.norm1(h); a,_=b.attn(n,n,n,need_weights=False); u=h+a; z=b.norm2(u)
            if i==layer:
                base=u+rule(z); sims=[]
                for node in range(len(rule.graph.nodes)):
                    vals=rule.graph.node_values(z); vals2=[v.clone() for v in vals]; vals2[node]=torch.zeros_like(vals2[node]); pert=rule.operator(vals2); h0=base; h1=u+pert
                    def down(hh):
                        for j in range(i+1,len(teacher.blocks)): hh=teacher.blocks[j](hh)
                        return teacher.head(hh[:,0])
                    actual=down(h1)-down(h0); local=teacher.head(h1[:,0])-teacher.head(h0[:,0]); sims.append(float(nn.functional.cosine_similarity(actual,local,dim=-1).mean().item()))
                return sum(sims)/len(sims)
            h=b(h)
    return 0.

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
    motifs=['sequential','parallel_sum','residual_parallel']; node_families=[('affine_polynomial','polynomial'),('low_rank','diagonal'),('polynomial','diagonal')]; op_modes=['identity','scale','negate','difference','product','mix']
    candidates=[]
    for motif in motifs:
        for nodes in node_families:
            for mode in op_modes:
                graph,ops,stab,bundles=fit_shared_graph(motif,nodes,mode,teachers,args,device)
                rules=[SharedTaskRule(motif,nodes,mode,args.d_model,args.rank).to(device) for _ in meta]
                for i,r in enumerate(rules): r.graph=graph; r.operator=ops[i].to(device)
                acc=[evaluate(RoutingCompiled(teachers[t][0],rules[i],args.trajectory_start,args.trajectory_end).to(device),teachers[t][2],device) for i,t in enumerate(meta)]
                pmat=operator_permutation_matrix(graph,ops,teachers,args,device); pg=operator_permutation_gap(pmat)
                rcf=rule_causal_fidelity(rules[0],teachers[meta[0]][0],teachers[meta[0]][1],args,device,args.trajectory_start)
                # random graph control, same operator family and params
                rg=RuleGraph(motif,nodes,args.d_model,args.rank).to(device); rg_acc=[]
                for i,t in enumerate(meta):
                    rr=SharedTaskRule(motif,nodes,mode,args.d_model,args.rank).to(device); rr.graph=rg; rr.operator=ops[i].to(device); rg_acc.append(evaluate(RoutingCompiled(teachers[t][0],rr,args.trajectory_start,args.trajectory_end).to(device),teachers[t][2],device))
                rgap=sum(acc)/len(acc)-sum(rg_acc)/len(rg_acc)
                sep=[]
                for t in meta:
                    sr=fit_separate_graph(t,teachers[t],motif,nodes,mode,args,device); sep.append(evaluate(RoutingCompiled(teachers[t][0],sr,args.trajectory_start,args.trajectory_end).to(device),teachers[t][2],device))
                sepavg=sum(sep)/len(sep); parity=sepavg-sum(acc)/len(acc)
                op_eff=[]
                for i,t in enumerate(meta):
                    z,_=bundles[t]; base=ops[i].state_dict(); base_rule=rules[i]; base_out=base_rule(z.to(device)); diffs=[]
                    for key,val in base.items():
                        pert={k:v.clone() for k,v in base.items()}; pert[key]=pert[key]+args.operator_delta; base_rule.operator.load_state_dict(pert); diffs.append(float((base_rule(z.to(device))-base_out).norm().item()/max(len(z),1)))
                    base_rule.operator.load_state_dict(base); op_eff.append(sum(diffs)/len(diffs))
                op_effect=sum(op_eff)/len(op_eff)
                eligible=(sum(acc)/len(acc)>=args.min_avg_source_acc and min(acc)>=args.min_worst_source_acc and stab<=args.max_operator_stability and pg>=args.min_operator_specificity and rcf>=args.min_rule_causal_fidelity and rgap>=args.min_random_graph_gap and parity<=args.max_shared_vs_separate_gap and op_effect>=args.min_operator_effect)
                score=sum(acc)/len(acc)+args.rule_weight*rcf+args.operator_weight*pg+args.random_graph_weight*rgap+args.operator_effect_weight*op_effect-args.shared_parity_weight*max(0,parity)-args.complexity_lambda*(count_params(graph)+sum(count_params(o) for o in ops))
                candidates.append({'motif':motif,'nodes':nodes,'operator_mode':mode,'graph':graph,'ops':ops,'source_avg':sum(acc)/len(acc),'source_worst':min(acc),'operator_stability':stab,'operator_effect':op_effect,'rule_causal_fidelity':rcf,'operator_specificity':pg,'random_graph_gap':rgap,'separate_graph_avg':sepavg,'shared_vs_separate_gap':parity,'task_acc':acc,'operator_permutation_matrix':pmat,'eligible':eligible,'score':score})
    elig=[c for c in candidates if c['eligible']]; best=max(elig,key=lambda c:c['score']) if elig else max(candidates,key=lambda c:c['score'])
    results={}
    for label,t in [('related',args.holdout_tasks[0]),('contrast',contrast)]:
        tea,tr,va=(train_teacher(t,args,device,seed+50000) if label=='related' else (c_tea,c_tr,c_va)); te=DataLoader(TaskDataset(args.test_size,t,seed+60000+(0 if label=='related' else 1)),batch_size=args.batch_size,shuffle=False); teacher_acc=evaluate(tea,te,device)
        mean_op=copy.deepcopy(best['ops'][0]).to(device); mean_state={k:sum(best['ops'][i].state_dict()[k] for i in range(len(best['ops'])))/len(best['ops']) for k in best['ops'][0].state_dict()}; mean_op.load_state_dict(mean_state); rule=SharedTaskRule(best['motif'],best['nodes'],best['operator_mode'],args.d_model,args.rank).to(device); rule.graph=best['graph']; rule.operator=mean_op
        a0=evaluate(RoutingCompiled(tea,rule,args.trajectory_start,args.trajectory_end).to(device),te,device); z,y=source_bundle(tea,tr,args,device); op_t=operator_fit(rule.graph,best['operator_mode'],z,y,args,device,args.target_operator_fit_steps,operator_state(mean_op)); rule.operator=op_t; a1=evaluate(RoutingCompiled(tea,rule,args.trajectory_start,args.trajectory_end).to(device),te,device)
        perm_op=copy.deepcopy(best['ops'][0]).to(device); rule.operator=perm_op; ap=evaluate(RoutingCompiled(tea,rule,args.trajectory_start,args.trajectory_end).to(device),te,device)
        mlp=MLPControl(tea,args.trajectory_start,args.trajectory_end,args.mlp_width).to(device); train(mlp,tr,device,args.transfer_control_steps,args.lr); am=evaluate(mlp,te,device)
        results[label+'_holdout']={'task':t,'teacher':teacher_acc,'dart_zero':a0,'dart_adapted':a1,'operator_permutation_control':ap,'mlp_control':am,'gain_zero':(a0-teacher_acc)*100,'gain_adapt':(a1-teacher_acc)*100,'vs_mlp_adapt':(a1-am)*100,'operator_state':{k:v.cpu().tolist() for k,v in op_t.state_dict().items()}}
    clean=lambda c:{k:v for k,v in c.items() if k not in ('graph','ops')}
    return {'seed':seed,'winner':clean(best),'candidates':[clean(c) for c in candidates],**results}

def main():
    p=argparse.ArgumentParser(description='DART-2.2 structured task-operator discovery over shared rule graph')
    p.add_argument('--seeds',nargs='+',type=int,default=[1,2]); p.add_argument('--all-tasks',nargs='+',default=['add','compose','mul','sub']); p.add_argument('--holdout-tasks',nargs='+',default=['sub']); p.add_argument('--contrast-tasks',nargs='+',default=['sort'])
    for k,v in [('teacher-steps',800),('core-fit-steps',300),('operator-fit-steps',120),('target-operator-fit-steps',400),('transfer-control-steps',400),('separate-control-steps',200),('train-size',6000),('verifier-size',1500),('test-size',1500),('rel-samples-per-task',2048),('fit-batch-samples',512),('d-model',32),('heads',2),('d-ff',128),('depth',3),('rank',8),('batch-size',256),('mlp-width',64)]: p.add_argument('--'+k,type=int,default=v)
    for k,v in [('operator-delta',.1),('operator-l2',.0005),('operator-lr',.01),('min-avg-source-acc',.30),('min-worst-source-acc',.22),('max-operator-stability',.75),('min-operator-specificity',.01),('min-operator-effect',.02),('min-rule-causal-fidelity',.20),('min-random-graph-gap',.02),('max-shared-vs-separate-gap',.03),('rule-weight',.30),('operator-weight',.25),('random-graph-weight',.25),('operator-effect-weight',.15),('shared-parity-weight',.25),('complexity-lambda',1e-5),('lr',.0003),('core-fit-lr',.001)]: p.add_argument('--'+k,type=float,default=v)
    p.add_argument('--trajectory-start',type=int,default=0); p.add_argument('--trajectory-end',type=int,default=1); p.add_argument('--device',default='cuda'); p.add_argument('--out',default='dart022_results.json')
    args=p.parse_args(); rec=[run_seed(args,s) for s in args.seeds]
    def av(k,sec): return sum(r[sec][k] for r in rec)/len(rec)
    summary={'related_holdout':{args.holdout_tasks[0]:{k:av(k,'related_holdout') for k in ['teacher','dart_zero','dart_adapted','operator_permutation_control','mlp_control','gain_zero','gain_adapt','vs_mlp_adapt']}},'contrast_holdout':{args.contrast_tasks[0]:{k:av(k,'contrast_holdout') for k in ['teacher','dart_zero','dart_adapted','operator_permutation_control','mlp_control','gain_zero','gain_adapt','vs_mlp_adapt']}},'source':{'avg_accuracy':sum(r['winner']['source_avg'] for r in rec)/len(rec),'avg_operator_stability':sum(r['winner']['operator_stability'] for r in rec)/len(rec),'avg_operator_effect':sum(r['winner']['operator_effect'] for r in rec)/len(rec),'avg_rule_causal_fidelity':sum(r['winner']['rule_causal_fidelity'] for r in rec)/len(rec),'avg_operator_specificity':sum(r['winner']['operator_specificity'] for r in rec)/len(rec),'avg_random_graph_gap':sum(r['winner']['random_graph_gap'] for r in rec)/len(rec),'avg_separate_graph_accuracy':sum(r['winner']['separate_graph_avg'] for r in rec)/len(rec),'avg_shared_vs_separate_gap':sum(r['winner']['shared_vs_separate_gap'] for r in rec)/len(rec)} }
    payload={'config':vars(args),'records':rec,'summary':summary}; Path(args.out).write_text(json.dumps(payload,indent=2)); print('DART-2.2: structured task-operator discovery over shared rule graph'); print(json.dumps(summary,indent=2)); print('Saved:',Path(args.out).resolve())
if __name__=='__main__': main()
