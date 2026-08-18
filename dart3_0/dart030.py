#!/usr/bin/env python3
import argparse, copy, json, random, sys, time
from dataclasses import dataclass
from itertools import product
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

TASKS={
 'add':lambda a,b:a+b,
 'mul':lambda a,b:a*b,
 'sub':lambda a,b:a-b,
 'compose':lambda a,b:(a*2+1)-(b*3-1),
 'sort':lambda a,b: torch.where(a<=b,a,b),
}

def seed_all(s):
 random.seed(s); torch.manual_seed(s)
 if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

class Progress:
 def __init__(self,total,label,si,ns): self.total=max(1,total); self.si=si; self.ns=ns; self.done=0; self.label=label
 def update(self,n,label=None,detail=''):
  self.done=min(self.total,max(0,n)); self.label=label or self.label
  frac=self.done/self.total; w=28; fill=int(frac*w); bar='='*fill+'>'+' ' * max(0,w-fill-1)
  sys.stdout.write(f'\r[DART-3.0][seed {self.si}/{self.ns}] [{bar}] {100*frac:6.2f}% | {self.label}')
  if detail: sys.stdout.write(' | '+detail)
  sys.stdout.flush()
 def close(self): self.update(self.total,'complete'); print()

def make_data(task,n,seed,device):
 g=torch.Generator(device='cpu').manual_seed(seed)
 x=torch.randint(-3,4,(n,2),generator=g).float().to(device)
 y=TASKS[task](x[:,0],x[:,1])
 bins=torch.tensor([-6,-3,0,3,6],device=device).float(); y=torch.bucketize(y,bins).clamp(max=5)
 return x,y.long()

class MLPTeacher(nn.Module):
 def __init__(self,d=32,h=64,c=6):
  super().__init__(); self.net=nn.Sequential(nn.Linear(2,d),nn.GELU(),nn.Linear(d,h),nn.GELU(),nn.Linear(h,c))
 def forward(self,x): return self.net(x)


def node(name,d,rank):
 if name=='affine_polynomial': return nn.Sequential(nn.Linear(d,d),nn.GELU(),nn.Linear(d,d))
 if name=='polynomial': return nn.Sequential(nn.Linear(d,d),nn.Tanh(),nn.Linear(d,d))
 if name=='low_rank': return nn.Sequential(nn.Linear(d,rank,bias=False),nn.Linear(rank,d,bias=False))
 raise ValueError(name)

class Primitive(nn.Module):
 def __init__(self,nodes,motif,d=32,rank=8):
  super().__init__(); self.nodes=nodes; self.motif=motif; self.blocks=nn.ModuleList([node(n,d,rank) for n in nodes]); self.norm=nn.LayerNorm(d)
 def forward(self,h):
  if self.motif=='sequential':
   z=h
   for b in self.blocks: z=z+b(self.norm(z))
   return z
  hs=[b(self.norm(h)) for b in self.blocks]
  if self.motif=='parallel_sum': return h+sum(hs)
  if self.motif=='residual_parallel': return h+hs[-1]+0.5*sum(hs[:-1])
  raise ValueError(self.motif)

class PrimitiveModel(nn.Module):
 def __init__(self,primitive,d=32,c=6):
  super().__init__(); self.inp=nn.Linear(2,d); self.primitive=primitive; self.out=nn.Linear(d,c)
 def forward(self,x): return self.out(self.primitive(self.inp(x)))

PROGRAM_OPS=['identity','scale','shift','negate','difference','product','swap']
@dataclass(frozen=True)
class Program:
 ops: tuple
 def __len__(self): return len(self.ops)

class ProgramAdapter(nn.Module):
 def __init__(self,program):
  super().__init__(); self.program=program
  self.scalars=nn.ParameterList([nn.Parameter(torch.tensor(1.0 if op=='scale' else 0.0),requires_grad=(op in ('scale','shift'))) for op in program.ops])
  self.decode_scale=nn.Parameter(torch.tensor(1.0)); self.decode_bias=nn.Parameter(torch.tensor(0.0))
 def forward_raw(self,x):
  z=x
  for op,p in zip(self.program.ops,self.scalars):
   a,b=z[:,0],z[:,1]
   if op=='identity': pass
   elif op=='scale': z=z*p
   elif op=='shift': z=z+p
   elif op=='negate': z=-z
   elif op=='difference': z=torch.stack([a-b,b-a],1)
   elif op=='product':
    q=a*b; z=torch.stack([q,q],1)
   elif op=='swap': z=torch.stack([b,a],1)
  return z

class ProgramModel(nn.Module):
 def __init__(self,primitive_model,program):
  super().__init__(); self.primitive_model=primitive_model; self.program=program; self.adapter=ProgramAdapter(program)
 def forward(self,x):
  z=self.adapter.forward_raw(x); return self.primitive_model(z)*self.adapter.decode_scale+self.adapter.decode_bias


def fit(model,loader,steps,lr,freeze_primitive=False):
 ps=[]
 for n,p in model.named_parameters():
  if freeze_primitive and n.startswith('primitive_model.'):
   p.requires_grad_(False)
  elif p.requires_grad: ps.append(p)
 if not ps: return
 opt=torch.optim.Adam(ps,lr=lr); ce=nn.CrossEntropyLoss(); it=iter(loader); model.train()
 for _ in range(max(1,steps)):
  try: x,y=next(it)
  except StopIteration: it=iter(loader); x,y=next(it)
  opt.zero_grad(set_to_none=True); loss=ce(model(x),y); loss.backward(); opt.step()

def acc(model,x,y):
 model.eval()
 with torch.no_grad(): p=model(x).argmax(-1)
 return float((p==y).float().mean())

def enumerate_programs(L):
 out=[]
 for l in range(1,L+1):
  for ops in product(PROGRAM_OPS,repeat=l):
   if all(o=='identity' for o in ops): continue
   out.append(Program(ops))
 return out

def discover(args,sources,device,seed,prog):
 cand=[]; motifs=['sequential','parallel_sum','residual_parallel']; node_sets=[['affine_polynomial','polynomial'],['affine_polynomial','polynomial','low_rank']]
 k=0
 for motif in motifs:
  for nodes in node_sets:
   prim=Primitive(nodes,motif,args.d_model,args.rank).to(device); vals=[]
   for i,t in enumerate(sources):
    x,y=make_data(t,args.train_size,seed+1000+k*7+i,device); m=PrimitiveModel(copy.deepcopy(prim),args.d_model,args.classes).to(device)
    fit(m,DataLoader(TensorDataset(x,y),batch_size=args.batch_size,shuffle=True),args.core_fit_steps,args.core_fit_lr,False); vals.append(acc(m,x,y))
   cand.append((sum(vals)/len(vals),prim,vals,motif,nodes)); k+=1; prog.update(k,'primitive-search',f'candidate={k}')
 cand.sort(key=lambda z:z[0],reverse=True); return cand[0]

def train_program(base,program,x,y,steps,lr):
 m=ProgramModel(copy.deepcopy(base),program).to(x.device)
 for p in m.primitive_model.parameters(): p.requires_grad_(False)
 fit(m,DataLoader(TensorDataset(x,y),batch_size=256,shuffle=True),steps,lr,True)
 return m

def program_effect(model,x,y):
 base=acc(model,x,y); effects=[]
 ops=list(model.program.ops)
 for i in range(len(ops)):
  q=ops.copy(); q[i]='identity'; alt=Program(tuple(q)); a=train_program(model.primitive_model,alt,x,y,1,0.01); effects.append(max(0.0,base-acc(a,x,y)))
 return effects

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--seeds',nargs='+',type=int,default=[1,2]); ap.add_argument('--all-tasks',nargs='+',default=['add','compose','mul','sub']); ap.add_argument('--holdout-tasks',nargs='+',default=['sub']); ap.add_argument('--contrast-tasks',nargs='+',default=['sort']); ap.add_argument('--teacher-steps',type=int,default=800); ap.add_argument('--core-fit-steps',type=int,default=300); ap.add_argument('--program-fit-steps',type=int,default=120); ap.add_argument('--target-program-fit-steps',type=int,default=400); ap.add_argument('--transfer-control-steps',type=int,default=400); ap.add_argument('--separate-control-steps',type=int,default=200); ap.add_argument('--train-size',type=int,default=6000); ap.add_argument('--verifier-size',type=int,default=1500); ap.add_argument('--test-size',type=int,default=1500); ap.add_argument('--rel-samples-per-task',type=int,default=2048); ap.add_argument('--fit-batch-samples',type=int,default=512); ap.add_argument('--causal-probe-size',type=int,default=64); ap.add_argument('--max-active-adapters',type=int,default=2); ap.add_argument('--max-program-length',type=int,default=2); ap.add_argument('--complexity-lambda',type=float,default=0.05); ap.add_argument('--min-program-specificity',type=float,default=0.02); ap.add_argument('--min-program-necessity',type=float,default=0.1); ap.add_argument('--device',default='cuda'); ap.add_argument('--d-model',type=int,default=32); ap.add_argument('--rank',type=int,default=8); ap.add_argument('--batch-size',type=int,default=256); ap.add_argument('--classes',type=int,default=6); ap.add_argument('--core-fit-lr',type=float,default=1e-3); ap.add_argument('--lr',type=float,default=3e-4); ap.add_argument('--out',default='dart030_results.json'); args=ap.parse_args()
 device=torch.device(args.device if args.device=='cpu' or torch.cuda.is_available() else 'cpu'); sources=[t for t in args.all_tasks if t not in args.holdout_tasks and t in ('add','compose','mul')]; programs=enumerate_programs(args.max_program_length); records=[]
 for si,seed in enumerate(args.seeds,1):
  seed_all(seed); total=6+6+len(programs)+len(programs)*2+5; pg=Progress(total,'start',si,len(args.seeds))
  teacher_scores={}; teachers= list(dict.fromkeys(sources+args.holdout_tasks+args.contrast_tasks))
  for i,t in enumerate(teachers):
   x,y=make_data(t,args.train_size,seed+200*i,device); m=MLPTeacher(args.d_model,64,args.classes).to(device); fit(m,DataLoader(TensorDataset(x,y),batch_size=args.batch_size,shuffle=True),args.teacher_steps,args.lr,False); teacher_scores[t]=acc(m,x,y); pg.update(1+i,'teacher-training',f'task={t}')
  src_avg,prim,srcvals,motif,nodes=discover(args,sources,device,seed,pg)
  base=PrimitiveModel(copy.deepcopy(prim),args.d_model,args.classes).to(device); bx,by=make_data(sources[0],args.train_size,seed+900,device); fit(base,DataLoader(TensorDataset(bx,by),batch_size=args.batch_size,shuffle=True),args.core_fit_steps,args.core_fit_lr,False)
  for p in base.parameters(): p.requires_grad_(False)
  scored=[]
  for i,pr in enumerate(programs):
   vals=[]
   for j,t in enumerate(sources):
    x,y=make_data(t,args.verifier_size,seed+1500+i*17+j,device); m=train_program(base,pr,x,y,args.program_fit_steps,args.lr); vals.append(acc(m,x,y))
   score=sum(vals)/len(vals)-args.complexity_lambda*len(pr); scored.append((score,pr,vals)); pg.update(12+i,'source-program-search',f'program={i+1}/{len(programs)}')
  scored.sort(key=lambda z:z[0],reverse=True); best=next((s for s in scored if s[1].ops!=(('identity',) if False else tuple())),scored[0]); _,best_pr,best_vals=best
  cx,cy=make_data(sources[0],args.causal_probe_size,seed+3000,device); bm=train_program(base,best_pr,cx,cy,args.program_fit_steps,args.lr); eff=program_effect(bm,cx,cy); necessity=sum(eff)/max(1,len(eff)); pg.update(total-len(programs)*2-3,'program-causal-controls',f'necessity={necessity:.3f}')
  target=args.holdout_tasks[0]; tx,ty=make_data(target,args.test_size,seed+4000,device); zero=train_program(base,Program(('identity',)),tx,ty,1,args.lr); zero_acc=acc(zero,tx,ty)
  best_target=None
  for i,pr in enumerate(programs):
   m=train_program(base,pr,tx,ty,args.target_program_fit_steps,args.lr); a=acc(m,tx,ty); score=a-args.complexity_lambda*len(pr); best_target=(score,a,pr) if best_target is None or score>best_target[0] else best_target; pg.update(total-len(programs)+i,'frozen-target-program-search',f'program={i+1}/{len(programs)} acc={a:.3f}')
  wrong=Program(('negate',)) if best_pr.ops[0]!='negate' else Program(('scale',)); pm=train_program(base,wrong,tx,ty,args.transfer_control_steps,args.lr); perm=acc(pm,tx,ty); rp=programs[(seed*13)%len(programs)]; rm=train_program(base,rp,tx,ty,args.transfer_control_steps,args.lr); rnd=acc(rm,tx,ty)
  contrast=args.contrast_tasks[0]; sx,sy=make_data(contrast,args.test_size,seed+5000,device); cm=train_program(base,best_pr,sx,sy,args.transfer_control_steps,args.lr); cacc=acc(cm,sx,sy); pg.close()
  records.append({'seed':seed,'winner':{'motif':motif,'nodes':nodes,'program':list(best_pr.ops),'source_avg':src_avg,'source_task_accs':best_vals,'program_necessity':necessity},'related_holdout':{target:{'teacher':teacher_scores[target],'dart_zero':zero_acc,'dart_program':best_target[1],'program_permutation_control':perm,'random_program_control':rnd}},'contrast_holdout':{contrast:{'teacher':teacher_scores[contrast],'dart_program':cacc}}})
 summary={'related_holdout':{'sub':{k:sum(r['related_holdout']['sub'][k] for r in records)/len(records) for k in records[0]['related_holdout']['sub']}},'contrast_holdout':{'sort':{k:sum(r['contrast_holdout']['sort'][k] for r in records)/len(records) for k in records[0]['contrast_holdout']['sort']}},'source':{'avg_accuracy':sum(r['winner']['source_avg'] for r in records)/len(records),'avg_program_necessity':sum(r['winner']['program_necessity'] for r in records)/len(records),'avg_program_length':sum(len(r['winner']['program']) for r in records)/len(records)},'records':records}
 Path(args.out).write_text(json.dumps(summary,indent=2)); print('DART-3.0: task-program composition around frozen causal primitive'); print(json.dumps(summary,indent=2)); print('Saved:',Path(args.out).resolve())
if __name__=='__main__': main()
