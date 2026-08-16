#!/usr/bin/env python3
from __future__ import annotations
import argparse, copy, json, random, time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

VOCAB = list('0123456789+= ')
STOI = {c:i for i,c in enumerate(VOCAB)}
PAD = STOI[' ']
BLOCK_SIZE = 12

def seed_everything(seed:int)->None:
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def make_example(a:int,b:int):
    text=f'{a}+{b}='
    ids=[STOI[c] for c in text]
    ids=(ids+[PAD]*BLOCK_SIZE)[:BLOCK_SIZE]
    target=(int(str(a)[0])+int(str(b)[-1]))%10
    return ids,target

class AddDataset(Dataset):
    def __init__(self,n:int,seed:int):
        rng=random.Random(seed); self.rows=[]
        for _ in range(n):
            a=rng.randint(0,999); b=rng.randint(0,999)
            x,y=make_example(a,b)
            self.rows.append((torch.tensor(x,dtype=torch.long),torch.tensor(y,dtype=torch.long)))
    def __len__(self): return len(self.rows)
    def __getitem__(self,i): return self.rows[i]

class SmallFF(nn.Module):
    def __init__(self,d,b):
        super().__init__(); self.net=nn.Sequential(nn.Linear(d,b),nn.GELU(),nn.Linear(b,d))
    def forward(self,x): return self.net(x)

class Block(nn.Module):
    def __init__(self,d,h,ff):
        super().__init__(); self.norm1=nn.LayerNorm(d); self.attn=nn.MultiheadAttention(d,h,batch_first=True,dropout=0.0); self.norm2=nn.LayerNorm(d); self.ff=nn.Sequential(nn.Linear(d,ff),nn.GELU(),nn.Linear(ff,d))
    def forward(self,x):
        h=self.norm1(x); a,_=self.attn(h,h,h,need_weights=False); x=x+a; return x+self.ff(self.norm2(x))

class TinyTransformer(nn.Module):
    def __init__(self,vocab,d=32,h=2,ff=128,small_bottleneck:Optional[int]=None):
        super().__init__(); self.d=d
        self.emb=nn.Embedding(vocab,d); self.pos=nn.Parameter(torch.randn(1,BLOCK_SIZE,d)*0.02)
        self.blocks=nn.ModuleList([Block(d,h,ff) for _ in range(3)]); self.head=nn.Linear(d,10)
        if small_bottleneck is not None: self.blocks[1].ff=SmallFF(d,small_bottleneck)
    def forward(self,x):
        h=self.emb(x)+self.pos[:,:x.size(1)]
        for b in self.blocks: h=b(h)
        return self.head(h[:,0])

def params(m): return sum(p.numel() for p in m.parameters())

def ff_params(m): return params(m.blocks[1].ff)

def ff_macs(ff):
    if hasattr(ff,'net'): mods=list(ff.net)
    elif isinstance(ff,nn.Sequential): mods=list(ff)
    elif hasattr(ff,'repl'): return ff_macs(ff.repl)
    else: raise TypeError(type(ff))
    ls=[m for m in mods if isinstance(m,nn.Linear)]
    return sum(x.in_features*x.out_features for x in ls)

def evaluate(m,loader,device):
    m.eval(); ce=nn.CrossEntropyLoss(reduction='sum'); total=correct=0; loss=0.0
    with torch.no_grad():
        for x,y in loader:
            x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True); z=m(x); loss+=float(ce(z,y)); correct+=int((z.argmax(-1)==y).sum()); total+=y.numel()
    return {'accuracy':correct/max(total,1),'loss':loss/max(total,1),'params':params(m),'ff_params':ff_params(m)}

def train(m,loader,device,steps,lr):
    if steps<=0:return 0.0
    m.train(); opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=1e-4); ce=nn.CrossEntropyLoss(); it=iter(loader)
    if device.type=='cuda': torch.cuda.synchronize()
    t=time.perf_counter()
    for _ in range(steps):
        try:x,y=next(it)
        except StopIteration: it=iter(loader); x,y=next(it)
        x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True)
        loss=ce(m(x),y)
        if not torch.isfinite(loss): raise RuntimeError('non-finite task loss')
        opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
    if device.type=='cuda': torch.cuda.synchronize()
    return time.perf_counter()-t

def collect_traces(teacher,loader,device,max_batches):
    xs=[]; ys=[]; block=teacher.blocks[1]
    def hook(_m,inputs,out): xs.append(inputs[0].detach().reshape(-1,inputs[0].shape[-1]).cpu()); ys.append(out.detach().reshape(-1,out.shape[-1]).cpu())
    h=block.ff.register_forward_hook(hook)
    try:
        with torch.no_grad():
            for i,(x,_) in enumerate(loader):
                if i>=max_batches: break
                teacher(x.to(device,non_blocking=True))
    finally: h.remove()
    return torch.cat(xs),torch.cat(ys)

def fit_replacement(tx,ty,d,b,device,steps,lr):
    cand=SmallFF(d,b).to(device); ds=torch.utils.data.TensorDataset(tx,ty); dl=DataLoader(ds,batch_size=512,shuffle=True); it=iter(dl); opt=torch.optim.AdamW(cand.parameters(),lr=lr); mse=nn.MSELoss()
    if device.type=='cuda': torch.cuda.synchronize()
    t=time.perf_counter(); cand.train()
    for _ in range(steps):
        try:x,y=next(it)
        except StopIteration:it=iter(dl); x,y=next(it)
        x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True); loss=mse(cand(x),y)
        if not torch.isfinite(loss): raise RuntimeError('non-finite replacement loss')
        opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(cand.parameters(),1.0); opt.step()
    if device.type=='cuda': torch.cuda.synchronize()
    return cand,time.perf_counter()-t

def cf_mse(teacher,repl,loader,device,noise,batches):
    teacher.eval(); repl.eval(); mse=nn.MSELoss(); total=n=0; block=teacher.blocks[1]
    with torch.no_grad():
        for i,(x,_) in enumerate(loader):
            if i>=batches: break
            x=x.to(device,non_blocking=True); box={}
            def ph(_m,inp): box['h']=inp[0].detach()
            hh=block.ff.register_forward_pre_hook(ph); teacher(x); hh.remove(); h=box['h']; hp=h+torch.randn_like(h)*noise
            total+=float(mse(repl(hp),block.ff(hp))); n+=1
    return total/max(n,1)

def replace_ff(m,repl):
    class Wrap(nn.Module):
        def __init__(self,r): super().__init__(); self.repl=r
        def forward(self,x): return self.repl(x)
    m.blocks[1].ff=Wrap(repl)

@torch.no_grad()
def latency(m,loader,device,warm,iters):
    if device.type!='cuda': return None
    m.eval(); x=next(iter(loader))[0].to(device)
    for _ in range(warm): m(x)
    torch.cuda.synchronize(); s=torch.cuda.Event(enable_timing=True); e=torch.cuda.Event(enable_timing=True); s.record()
    for _ in range(iters): m(x)
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e)/iters

def peak_mem(m,loader,device):
    if device.type!='cuda': return None
    m.eval(); x=next(iter(loader))[0].to(device); torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad(): m(x)
    torch.cuda.synchronize(); return torch.cuda.max_memory_allocated(device)/(1024**2)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--seed',type=int,default=7); ap.add_argument('--train-size',type=int,default=12000); ap.add_argument('--test-size',type=int,default=3000)
    ap.add_argument('--baseline-steps',type=int,default=1200); ap.add_argument('--scratch-steps',type=int,default=1200); ap.add_argument('--replacement-steps',type=int,default=400); ap.add_argument('--adaptation-steps',type=int,default=400)
    ap.add_argument('--d-model',type=int,default=32); ap.add_argument('--heads',type=int,default=2); ap.add_argument('--d-ff',type=int,default=128); ap.add_argument('--bottleneck',type=int,default=32); ap.add_argument('--batch-size',type=int,default=256)
    ap.add_argument('--lr',type=float,default=3e-4); ap.add_argument('--replacement-lr',type=float,default=1e-3); ap.add_argument('--trace-batches',type=int,default=50); ap.add_argument('--cf-batches',type=int,default=30); ap.add_argument('--cf-noise',type=float,default=0.05)
    ap.add_argument('--latency-warmup',type=int,default=30); ap.add_argument('--latency-iters',type=int,default=100); ap.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu'); ap.add_argument('--out',default='dart01_results.json')
    a=ap.parse_args(); seed_everything(a.seed); device=torch.device(a.device)
    train_ds=AddDataset(a.train_size,a.seed); test_ds=AddDataset(a.test_size,a.seed+1)
    train_dl=DataLoader(train_ds,batch_size=a.batch_size,shuffle=True,pin_memory=device.type=='cuda'); test_dl=DataLoader(test_ds,batch_size=a.batch_size,pin_memory=device.type=='cuda')
    print(f'device={device}')

    # Common teacher
    seed_everything(a.seed); teacher=TinyTransformer(len(VOCAB),a.d_model,a.heads,a.d_ff).to(device)
    print('\n[A] common teacher'); t_teacher=train(teacher,train_dl,device,a.baseline_steps,a.lr); e_teacher=evaluate(teacher,test_dl,device)
    r_teacher={'latency_ms':latency(teacher,test_dl,device,a.latency_warmup,a.latency_iters),'peak_cuda_mb':peak_mem(teacher,test_dl,device),'ff_macs_per_token':ff_macs(teacher.blocks[1].ff)}
    print(e_teacher); print(r_teacher)
    tx,ty=collect_traces(teacher,train_dl,device,a.trace_batches); print(f'trace_rows={len(tx):,}')

    # Scratch small
    seed_everything(a.seed+101); scratch=TinyTransformer(len(VOCAB),a.d_model,a.heads,a.d_ff,a.bottleneck).to(device)
    print('\n[B] scratch-small'); t_scratch=train(scratch,train_dl,device,a.scratch_steps,a.lr); e_scratch=evaluate(scratch,test_dl,device); r_scratch={'latency_ms':latency(scratch,test_dl,device,a.latency_warmup,a.latency_iters),'peak_cuda_mb':peak_mem(scratch,test_dl,device),'ff_macs_per_token':ff_macs(scratch.blocks[1].ff)}; print(e_scratch); print(r_scratch)

    # Distill replacement
    seed_everything(a.seed+202); repl, t_repl=fit_replacement(tx,ty,a.d_model,a.bottleneck,device,a.replacement_steps,a.replacement_lr); cf=cf_mse(teacher,repl,test_dl,device,a.cf_noise,a.cf_batches); print(f'\nreplacement_cf_mse={cf:.6f}')
    distill=copy.deepcopy(teacher).to(device); replace_ff(distill,copy.deepcopy(repl).to(device)); e_distill=evaluate(distill,test_dl,device); r_distill={'latency_ms':latency(distill,test_dl,device,a.latency_warmup,a.latency_iters),'peak_cuda_mb':peak_mem(distill,test_dl,device),'ff_macs_per_token':ff_macs(distill.blocks[1].ff)}; print('\n[C] distill-small'); print(e_distill); print(r_distill)

    # DART no adaptation
    dart=copy.deepcopy(teacher).to(device); replace_ff(dart,copy.deepcopy(repl).to(device)); e_dart=evaluate(dart,test_dl,device); r_dart={'latency_ms':latency(dart,test_dl,device,a.latency_warmup,a.latency_iters),'peak_cuda_mb':peak_mem(dart,test_dl,device),'ff_macs_per_token':ff_macs(dart.blocks[1].ff)}; print('\n[D] DART'); print(e_dart); print(r_dart)

    # DART + adaptation
    dart_ad=copy.deepcopy(dart).to(device); print('\n[E] DART+Adaptation'); t_ad=train(dart_ad,train_dl,device,a.adaptation_steps,a.lr); e_ad=evaluate(dart_ad,test_dl,device); r_ad={'latency_ms':latency(dart_ad,test_dl,device,a.latency_warmup,a.latency_iters),'peak_cuda_mb':peak_mem(dart_ad,test_dl,device),'ff_macs_per_token':ff_macs(dart_ad.blocks[1].ff)}; print(e_ad); print(r_ad)

    print('\n================ DART-0.1 ================')
    for name,e,r in [('Original',e_teacher,r_teacher),('Scratch',e_scratch,r_scratch),('Distill',e_distill,r_distill),('DART',e_dart,r_dart),('DART+Adapt',e_ad,r_ad)]:
        print(f"{name:12s} acc={e['accuracy']:.4f} loss={e['loss']:.4f} params={e['params']:,} ffparams={e['ff_params']:,} ffMAC/tok={r['ff_macs_per_token']:,} latency_ms={r['latency_ms']}")

    out={'config':vars(a),'teacher':{'eval':e_teacher,'runtime':r_teacher,'train_seconds':t_teacher},'scratch':{'eval':e_scratch,'runtime':r_scratch,'train_seconds':t_scratch},'distill':{'eval':e_distill,'runtime':r_distill,'replacement_seconds':t_repl,'cf_mse':cf},'dart':{'eval':e_dart,'runtime':r_dart,'replacement_seconds':t_repl,'cf_mse':cf},'dart_adapt':{'eval':e_ad,'runtime':r_ad,'adapt_seconds':t_ad,'adapt_steps':a.adaptation_steps,'cf_mse':cf}}
    Path(a.out).write_text(json.dumps(out,indent=2),encoding='utf-8'); print(f'\nSaved: {Path(a.out).resolve()}')

if __name__=='__main__': main()
