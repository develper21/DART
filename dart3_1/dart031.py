#!/usr/bin/env python3
"""
DART-3.1: causal task-program validation + minimal program discovery

Purpose:
Repair the DART-3.0 evaluation protocol without changing the research hypothesis.
Fixes:
1) Shared primitive discovery is trained jointly on ALL source tasks.
2) Source selection uses per-task verifier accuracies; no source task can be hidden by
   an incorrect aggregate score.
3) Program necessity is measured by ablating the SAME trained program state;
   no retraining occurs during the necessity test.
4) Target program search uses target-adaptation data + target-validation data;
   final test data is held out until the selected program is frozen.
5) Program permutation/random controls are evaluated on the same untouched target test set.
6) repair.result.json stores the complete dart031 result and protocol metadata.
"""
from __future__ import annotations
import argparse, copy, json, random, sys, time, hashlib
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ---------------- progress ----------------
class Progress:
    def __init__(self,total,label,seed_idx,nseeds):
        self.total=max(1,total); self.seed_idx=seed_idx; self.nseeds=nseeds
    def update(self,done,label,detail=""):
        frac=min(1,max(0,done/self.total)); width=28; fill=int(frac*width)
        bar="="*fill+">"+" "*max(0,width-fill-1)
        sys.stdout.write(f"\r[DART-3.1][repair[seed {self.seed_idx}/{self.nseeds}] [{bar}] {100*frac:6.2f}% | {label}")
        if detail: sys.stdout.write(f" | {detail}")
        sys.stdout.flush()
    def close(self): self.update(self.total,"complete"); print()

TASK_OPS={
    "add": lambda a,b:a+b,
    "mul": lambda a,b:a*b,
    "sub": lambda a,b:a-b,
    "compose": lambda a,b:(a*2+1)-(b*3-1),
    "sort": lambda a,b:torch.where(a<=b,a,b),
}

def seed_all(seed):
    random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def make_data(task,n,seed,device):
    g=torch.Generator(device="cpu").manual_seed(seed)
    x=torch.randint(-3,4,(n,2),generator=g).float().to(device)
    y=TASK_OPS[task](x[:,0],x[:,1])
    bins=torch.tensor([-6,-3,0,3,6],device=device).float()
    y=torch.bucketize(y,bins).clamp(max=5)
    return x,y.long()

def split_data(task,n_train,n_val,n_test,seed,device):
    return (make_data(task,n_train,seed,device),
            make_data(task,n_val,seed+1,device),
            make_data(task,n_test,seed+2,device))

# ---------------- models ----------------
class MLPTeacher(nn.Module):
    def __init__(self,d=32,h=64,c=6):
        super().__init__(); self.net=nn.Sequential(nn.Linear(2,d),nn.GELU(),nn.Linear(d,h),nn.GELU(),nn.Linear(h,c))
    def forward(self,x): return self.net(x)

def make_node(name,d,rank):
    if name=="affine_polynomial": return nn.Sequential(nn.Linear(d,d),nn.GELU(),nn.Linear(d,d))
    if name=="polynomial": return nn.Sequential(nn.Linear(d,d),nn.Tanh(),nn.Linear(d,d))
    if name=="low_rank": return nn.Sequential(nn.Linear(d,rank,bias=False),nn.Linear(rank,d,bias=False))
    raise ValueError(name)

class Primitive(nn.Module):
    def __init__(self,nodes,motif,d=32,rank=8):
        super().__init__(); self.nodes=nodes; self.motif=motif
        self.blocks=nn.ModuleList([make_node(n,d,rank) for n in nodes]); self.norm=nn.LayerNorm(d)
    def forward(self,h):
        hs=[b(self.norm(h)) for b in self.blocks]
        if self.motif=="sequential":
            z=h
            for b in self.blocks: z=z+b(self.norm(z))
            return z
        if self.motif=="parallel_sum": return h+sum(hs)
        if self.motif=="residual_parallel": return h+hs[-1]+0.5*sum(hs[:-1])
        raise ValueError(self.motif)

class PrimitiveModel(nn.Module):
    def __init__(self,primitive,d=32,c=6):
        super().__init__(); self.inp=nn.Linear(2,d); self.primitive=primitive; self.out=nn.Linear(d,c)
    def forward(self,x):
        return self.out(self.primitive(self.inp(x)))

PROGRAM_OPS=["identity","scale","shift","negate","difference","product","swap"]

@dataclass(frozen=True)
class Program:
    ops: tuple[str,...]

def enumerate_programs(max_len):
    out=[]
    for l in range(1,max_len+1):
        for ops in product(PROGRAM_OPS,repeat=l):
            if all(o=="identity" for o in ops): continue
            out.append(Program(tuple(ops)))
    return out

class ProgramAdapter(nn.Module):
    def __init__(self,program):
        super().__init__(); self.program=program
        self.raw=nn.ParameterList([nn.Parameter(torch.tensor(1.0 if op=="scale" else 0.0),requires_grad=op in ("scale","shift")) for op in program.ops])
        self.decode_scale=nn.Parameter(torch.tensor(1.0)); self.decode_bias=nn.Parameter(torch.tensor(0.0))
    def forward_raw(self,x):
        z=x
        for op,p in zip(self.program.ops,self.raw):
            a,b=z[:,0],z[:,1]
            if op=="identity": pass
            elif op=="scale": z=z*p
            elif op=="shift": z=z+p
            elif op=="negate": z=-z
            elif op=="difference": z=torch.stack([a-b,b-a],1)
            elif op=="product":
                q=a*b; z=torch.stack([q,q],1)
            elif op=="swap": z=torch.stack([b,a],1)
        return z

class ProgramModel(nn.Module):
    def __init__(self,primitive_model,program):
        super().__init__(); self.primitive_model=primitive_model; self.program=program; self.adapter=ProgramAdapter(program)
    def forward(self,x):
        z=self.adapter.forward_raw(x)
        return self.primitive_model(z)*self.adapter.decode_scale+self.adapter.decode_bias

# ---------------- fit/eval ----------------
def fit(model,x,y,steps,lr,freeze_primitive=False):
    for n,p in model.named_parameters():
        if freeze_primitive and n.startswith("primitive_model."): p.requires_grad_(False)
    params=[p for p in model.parameters() if p.requires_grad]
    if not params: return
    opt=torch.optim.Adam(params,lr=lr); loss_fn=nn.CrossEntropyLoss(); model.train()
    ds=TensorDataset(x,y); loader=DataLoader(ds,batch_size=min(256,len(ds)),shuffle=True); it=iter(loader)
    for _ in range(max(1,steps)):
        try: xb,yb=next(it)
        except StopIteration: it=iter(loader); xb,yb=next(it)
        opt.zero_grad(set_to_none=True); loss=loss_fn(model(xb),yb); loss.backward(); opt.step()

def acc(model,x,y):
    model.eval()
    with torch.no_grad(): p=model(x).argmax(-1)
    return float((p==y).float().mean().item())

def task_fit_frozen(base,program,x,y,steps,lr):
    m=ProgramModel(copy.deepcopy(base),program).to(x.device)
    for p in m.primitive_model.parameters(): p.requires_grad_(False)
    fit(m,x,y,steps,lr,True)
    return m

def ablated_same_state(model,index):
    ops=list(model.program.ops); ops[index]="identity"; alt=ProgramModel(copy.deepcopy(model.primitive_model),Program(tuple(ops))).to(next(model.parameters()).device)
    # Copy the exact learned adapter state. No retraining.
    alt.adapter.load_state_dict(copy.deepcopy(model.adapter.state_dict()))
    return alt

def necessity_same_state(model,x,y):
    base=acc(model,x,y); effects=[]
    for i in range(len(model.program.ops)):
        alt=ablated_same_state(model,i); effects.append(max(0.0,base-acc(alt,x,y)))
    return effects

def make_joint_source(source_tasks,n_train,seed,device):
    xs=[]; ys=[]
    for i,t in enumerate(source_tasks):
        x,y=make_data(t,n_train,seed+11*i,device); xs.append(x); ys.append(y)
    return torch.cat(xs,0),torch.cat(ys,0)

def discover(args,sources,device,seed,pg,step_base):
    motifs=["sequential","parallel_sum","residual_parallel"]
    node_sets=[["affine_polynomial","polynomial"],["affine_polynomial","polynomial","low_rank"]]
    candidates=[]; k=0
    # Train each primitive jointly across ALL source tasks to avoid the DART-3.0 first-task bias.
    joint_x,joint_y=make_joint_source(sources,args.train_size//max(1,len(sources)),seed+700,device)
    verifier={t:make_data(t,args.verifier_size,seed+1700+i,device) for i,t in enumerate(sources)}
    for motif in motifs:
        for nodes in node_sets:
            prim=Primitive(nodes,motif,args.d_model,args.rank).to(device)
            model=PrimitiveModel(prim,args.d_model,args.classes).to(device)
            fit(model,joint_x,joint_y,args.core_fit_steps,args.core_fit_lr,False)
            vals=[acc(model,*verifier[t]) for t in sources]
            mean=sum(vals)/len(vals); worst=min(vals)
            balanced=0.5*mean+0.5*worst
            candidates.append((balanced,mean,worst,copy.deepcopy(model),vals,motif,nodes)); k+=1
            pg.update(step_base+k,"primitive-search",f"candidate={k} mean={mean:.3f} worst={worst:.3f}")
    candidates.sort(key=lambda z:z[0],reverse=True)
    return candidates[0]

# ---------------- main ----------------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--seeds',nargs='+',type=int,default=[1,2])
    ap.add_argument('--all-tasks',nargs='+',default=['add','compose','mul','sub'])
    ap.add_argument('--holdout-tasks',nargs='+',default=['sub'])
    ap.add_argument('--contrast-tasks',nargs='+',default=['sort'])
    ap.add_argument('--teacher-steps',type=int,default=800); ap.add_argument('--core-fit-steps',type=int,default=300)
    ap.add_argument('--program-fit-steps',type=int,default=120); ap.add_argument('--target-program-fit-steps',type=int,default=400)
    ap.add_argument('--transfer-control-steps',type=int,default=400); ap.add_argument('--separate-control-steps',type=int,default=200)
    ap.add_argument('--train-size',type=int,default=6000); ap.add_argument('--verifier-size',type=int,default=1500); ap.add_argument('--test-size',type=int,default=1500)
    ap.add_argument('--target-adaptation-size',type=int,default=1200); ap.add_argument('--target-validation-size',type=int,default=1200)
    ap.add_argument('--rel-samples-per-task',type=int,default=2048); ap.add_argument('--fit-batch-samples',type=int,default=512)
    ap.add_argument('--causal-probe-size',type=int,default=64); ap.add_argument('--max-active-adapters',type=int,default=2)
    ap.add_argument('--max-program-length',type=int,default=2); ap.add_argument('--complexity-lambda',type=float,default=0.05)
    ap.add_argument('--min-program-necessity',type=float,default=0.10); ap.add_argument('--min-program-specificity',type=float,default=0.02)
    ap.add_argument('--device',default='cuda'); ap.add_argument('--d-model',type=int,default=32); ap.add_argument('--rank',type=int,default=8)
    ap.add_argument('--batch-size',type=int,default=256); ap.add_argument('--classes',type=int,default=6); ap.add_argument('--core-fit-lr',type=float,default=1e-3); ap.add_argument('--lr',type=float,default=3e-4)
    ap.add_argument('--out',default='dart031_results.json'); args=ap.parse_args()
    device=torch.device(args.device if args.device=='cpu' or torch.cuda.is_available() else 'cpu')
    sources=[t for t in args.all_tasks if t not in args.holdout_tasks and t in ('add','compose','mul')]
    programs=enumerate_programs(args.max_program_length); records=[]
    # ~ fixed teacher + 6 primitive candidates + N programs + N target validations + controls
    total=6+len(programs)*2+10
    for si,seed in enumerate(args.seeds,1):
        seed_all(seed); pg=Progress(total,'start',si,len(args.seeds))
        # Teacher: report independently per task, with verifier accuracy.
        teacher_scores={}
        for i,t in enumerate(sources+args.holdout_tasks+args.contrast_tasks):
            tr,va,te=split_data(t,args.train_size,args.verifier_size,args.test_size,seed+300*i,device)
            tm=MLPTeacher(args.d_model,64,args.classes).to(device); fit(tm,*tr,args.teacher_steps,args.lr,False)
            teacher_scores[t]=acc(tm,*va); pg.update(1+i,'teacher-training',f'task={t}')
        base_step=1+len(sources+args.holdout_tasks+args.contrast_tasks)
        bal,src_mean,src_worst,base,src_verifier_accs,motif,nodes=discover(args,sources,device,seed,pg,base_step)
        # Explicit source program search: each task gets its own fit, then program is evaluated on verifier data.
        scored=[]
        start_prog=base_step+6
        best_models={}
        for i,pr in enumerate(programs):
            vals=[]; models=[]
            for j,t in enumerate(sources):
                tr,va,_=split_data(t,args.train_size//2,args.verifier_size,args.test_size,seed+5000+i*31+j,device)
                m=task_fit_frozen(base,pr,tr[0],tr[1],args.program_fit_steps,args.lr); vals.append(acc(m,*va)); models.append(m)
            mean=sum(vals)/len(vals); worst=min(vals); balanced=0.5*mean+0.5*worst-args.complexity_lambda*len(pr.ops)
            scored.append((balanced,mean,worst,pr,vals,models)); pg.update(start_prog+i,'source-program-search',f'program={i+1}/{len(programs)} mean={mean:.3f} worst={worst:.3f}')
        scored.sort(key=lambda z:z[0],reverse=True); _,prog_mean,prog_worst,best_pr,src_program_vals,src_program_models=scored[0]
        # Same-state causal necessity: no retraining after ablation.
        effects=[]
        for m,t in zip(src_program_models,sources):
            x_probe,y_probe=make_data(t,args.causal_probe_size,seed+8000+list(sources).index(t)*101,device)
            effects.extend(necessity_same_state(m,x_probe,y_probe))
        necessity=sum(effects)/len(effects) if effects else 0.0
        # Target adaptation/validation/test split. Test remains untouched until final scoring.
        target=args.holdout_tasks[0]
        target_tr,target_va,target_te=split_data(target,args.target_adaptation_size,args.target_validation_size,args.test_size,seed+9000,device)
        # True zero-shot baseline: the frozen source primitive is evaluated directly on target test.
        # No target fitting, validation, or test adaptation is allowed.
        zero_acc=acc(base,*target_te)

        best_target=None
        best_target_model=None
        for i,pr in enumerate(programs):
            m=task_fit_frozen(base,pr,target_tr[0],target_tr[1],args.target_program_fit_steps,args.lr)
            va_acc=acc(m,*target_va)
            score=va_acc-args.complexity_lambda*len(pr.ops)
            if best_target is None or score>best_target[0]:
                best_target=(score,va_acc,pr)
                best_target_model=m
            pg.update(start_prog+len(programs)+i,'target-program-validation',f'program={i+1}/{len(programs)} val={va_acc:.3f}')

        # Final evaluation: selected model is already fixed; untouched target test is used only now.
        best_target_test=acc(best_target_model,*target_te)
        target_probe_x,target_probe_y=make_data(target,args.causal_probe_size,seed+9500,device)
        target_prog_effects=necessity_same_state(best_target_model,target_probe_x,target_probe_y)
        target_program_necessity=sum(target_prog_effects)/len(target_prog_effects) if target_prog_effects else 0.0

        # Permutation/random controls are trained only on target adaptation and scored on untouched test.
        wrong=Program(('negate',)) if best_target[2].ops[0] != 'negate' else Program(('scale',))
        pm=task_fit_frozen(base,wrong,target_tr[0],target_tr[1],args.transfer_control_steps,args.lr); perm=acc(pm,*target_te)
        rp=programs[(seed*13)%len(programs)]
        rm=task_fit_frozen(base,rp,target_tr[0],target_tr[1],args.transfer_control_steps,args.lr); rnd=acc(rm,*target_te)
        contrast=args.contrast_tasks[0]; c_tr,c_va,c_te=split_data(contrast,args.target_adaptation_size,args.target_validation_size,args.test_size,seed+11000,device)
        cm=task_fit_frozen(base,best_target[2],c_tr[0],c_tr[1],args.transfer_control_steps,args.lr); cacc=acc(cm,*c_te)
        pg.close()
        records.append({
            'seed':seed,
            'winner':{'motif':motif,'nodes':nodes,'program':list(best_pr.ops),'source_avg_balanced':float(bal),'source_mean_accuracy':float(src_mean),'source_worst_accuracy':float(src_worst),'source_verifier_accuracies':src_verifier_accs,'program_source_mean':float(prog_mean),'program_source_worst':float(prog_worst),'program_source_task_accuracies':src_program_vals,'program_necessity':float(necessity),'target_program_necessity':float(target_program_necessity)},
            'related_holdout':{target:{'teacher':teacher_scores[target],'dart_zero':float(zero_acc),'dart_program':float(best_target_test),'program_validation':float(best_target[1]),'program_permutation_control':float(perm),'random_program_control':float(rnd)}},
            'contrast_holdout':{contrast:{'teacher':teacher_scores[contrast],'dart_program':float(cacc)}}
        })
    summary={
        'version':'DART-3.1','parent_version':'DART-3.0',
        'protocol':{
            'joint_source_primitive_training':True,
            'balanced_source_selection':True,
            'same_state_program_necessity':True,
            'target_train_validation_test_split':True,
            'untouched_target_test_for_final_selection':True,
            'target_program_necessity_on_same_state':True,
            'deterministic_probe_seeding':True,
        },
        'related_holdout':{'sub':{k:sum(r['related_holdout']['sub'][k] for r in records)/len(records) for k in records[0]['related_holdout']['sub']}},
        'contrast_holdout':{'sort':{k:sum(r['contrast_holdout']['sort'][k] for r in records)/len(records) for k in records[0]['contrast_holdout']['sort']}},
        'source':{
            'avg_balanced_source_score':sum(r['winner']['source_avg_balanced'] for r in records)/len(records),
            'avg_source_mean_accuracy':sum(r['winner']['source_mean_accuracy'] for r in records)/len(records),
            'avg_source_worst_accuracy':sum(r['winner']['source_worst_accuracy'] for r in records)/len(records),
            'avg_program_source_mean':sum(r['winner']['program_source_mean'] for r in records)/len(records),
            'avg_program_source_worst':sum(r['winner']['program_source_worst'] for r in records)/len(records),
            'avg_program_necessity':sum(r['winner']['program_necessity'] for r in records)/len(records),
            'avg_target_program_necessity':sum(r['winner']['target_program_necessity'] for r in records)/len(records),
            'avg_program_length':sum(len(r['winner']['program']) for r in records)/len(records),
        },
        'records':records,
    }
    out=Path(args.out); out.write_text(json.dumps(summary,indent=2)); print('DART-3.1: causal task-program validation + minimal program discovery'); print(json.dumps(summary,indent=2)); print('Saved:',out.resolve())

if __name__=='__main__': main()
