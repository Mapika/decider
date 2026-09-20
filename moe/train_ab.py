"""One arm of the 35B-A3B optimizer A/B (runs/moe_optimizer_ab_v1): the public supervised recipe (cross-entropy on the slot readout,
LR 1e-5, warm-up 150, cosine, 16,384-token micro-batches, two micro-batches per optimizer step) on Qwen3.5-35B-A3B-Base with the
routed experts frozen, data parallel over two GPUs, AdamW or Muon with FP32 masters. Both arms read the same pre-tokenized items and
the same batch order, so at every step they have seen the same examples. Saves the trainable parameters (an overlay on the base
weights) at the marks, and a training curve.

Run from the public checkout so `decider` is the public package:
    cd . && PYTHONPATH=. CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node 2 --master_port 29511 \
        moe/train_ab.py --optimizer adamw --out runs/moe_optimizer_ab_v1/adamw
"""
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import pickle
import random
import time

import torch
import torch.distributed as dist

from decider.model import collate
from decider.train import batches_by_tokens, loss_fn

HERE=Path(__file__).resolve().parent
def load(name):
    spec=importlib.util.spec_from_file_location(name,HERE/f'{name}.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
ab_optim=load('optim');ab_model=load('model')


def main():
    p=argparse.ArgumentParser();p.add_argument('--optimizer',choices=['adamw','muon'],required=True);p.add_argument('--out',required=True)
    p.add_argument('--model',default='data/models/Qwen3.5-35B-A3B-Base');p.add_argument('--items',default='runs/moe_optimizer_ab_v1/items.pkl')
    p.add_argument('--lr',type=float,default=1e-5);p.add_argument('--warmup',type=int,default=150);p.add_argument('--max_tokens',type=int,default=16384);p.add_argument('--batch_seed',type=int,default=1)
    p.add_argument('--marks',default='0.25,0.5,1.0');p.add_argument('--smoke_steps',type=int,default=0);p.add_argument('--log_every',type=int,default=20)
    a=p.parse_args()
    dist.init_process_group('nccl');rank=dist.get_rank();world=dist.get_world_size();local=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(local);device=torch.device('cuda',local)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);logf=open(out/'train.log','a') if rank==0 else None
    def log(*s):
        if rank==0:msg=' '.join(str(x) for x in s);print(msg,flush=True);logf.write(msg+'\n');logf.flush()
    log('[args]',json.dumps(vars(a)),'world',world)
    torch.manual_seed(0);items=pickle.load(open(a.items,'rb'));batches=batches_by_tokens(items,a.max_tokens,random.Random(a.batch_seed))
    total=len(batches)//world
    if a.smoke_steps:total=a.smoke_steps
    mine=[batches[i*world+rank] for i in range(total)];marks=sorted({max(1,int(round(float(m)*total))) for m in a.marks.split(',')})
    log(f'[data] {len(items)} items, {len(batches)} micro-batches, {total} optimizer steps ({world} micro-batches each), marks {marks}')
    model=ab_model.MoEDecisionModel(a.model).to(device);tok=model.tok
    frozen=0
    for n,q in model.lm.named_parameters():
        if '.experts.' in n and 'shared' not in n:q.requires_grad_(False);frozen+=q.numel()
    trainable=[(n,q) for n,q in model.lm.named_parameters() if q.requires_grad];ntr=sum(q.numel() for _,q in trainable)
    if a.optimizer=='adamw':
        opts=[ab_optim.MasterAdamW([q for _,q in trainable],lr=a.lr)];groups={'adamw':ntr,'muon':0}
    else:
        mu=[(n,q) for n,q in trainable if ab_optim.is_muon_param(n,q)];ad=[(n,q) for n,q in trainable if not ab_optim.is_muon_param(n,q)]
        opts=[ab_optim.MasterMuon([q for _,q in mu],lr=a.lr),ab_optim.MasterAdamW([q for _,q in ad],lr=a.lr)];groups={'muon':sum(q.numel() for _,q in mu),'adamw':sum(q.numel() for _,q in ad)}
        log('[muon] matrices:',len(mu),'adamw params:',[n for n,_ in ad][:40],'...' if len(ad)>40 else '')
    log(f'[params] trainable {ntr} frozen experts {frozen} groups {json.dumps(groups)}')
    def lr_at(s):return a.lr*s/a.warmup if s<a.warmup else a.lr*0.5*(1+math.cos(math.pi*min(1.,(s-a.warmup)/max(1,total-a.warmup))))
    params=[q for _,q in trainable];curve=[];model.train();t0=time.time();t_last=t0;ce_acc=torch.zeros((),device=device);n_acc=0;tok_acc=0;rej=0
    for step,bidx in enumerate(mine,1):
        b=collate([items[i] for i in bidx],tok.pad_token_id);b={k:(v.to(device,non_blocking=True) if torch.is_tensor(v) else v) for k,v in b.items()}
        logits=model(b);loss,ce=loss_fn(logits,b['golds'],b['nopts']);loss.backward()
        handles=[dist.all_reduce(q.grad,op=dist.ReduceOp.AVG,async_op=True) for q in params]
        for h in handles:h.wait()
        gn=torch.nn.utils.clip_grad_norm_(params,1.0)
        for o in opts:
            for g in o.param_groups:g['lr']=lr_at(step)
            o.step()
        for q in params:q.grad=None
        ce_acc+=ce;n_acc+=1;tok_acc+=b['input_ids'].numel()*world
        if step%a.log_every==0 or step==total:
            c=ce_acc.clone();dist.all_reduce(c,op=dist.ReduceOp.AVG);now=time.time();tps=tok_acc/(now-t_last)
            rec=dict(step=step,tokens_seen=None,ce=float(c)/n_acc,grad_norm=float(gn),lr=lr_at(step),minutes=(now-t0)/60,tokens_per_second=tps,peak_gb=torch.cuda.max_memory_allocated()/2**30)
            curve.append(rec);log(f"[train] step {step}/{total} ce {rec['ce']:.4f} gn {rec['grad_norm']:.2f} lr {rec['lr']:.2e} {rec['minutes']:.1f}min eta {(total-step)*(now-t_last)/max(1,n_acc)/60:.0f}min {tps:.0f}tok/s mem {rec['peak_gb']:.0f}GB")
            ce_acc.zero_();n_acc=0;tok_acc=0;t_last=now
            if rank==0:json.dump(curve,open(out/'curve.json','w'))
        if step in marks:
            dist.barrier()
            if rank==0:
                overlay={n:q.detach().to(torch.bfloat16).cpu() for n,q in trainable};torch.save(overlay,out/f'overlay_{step:06d}.pt');log('[save]',f'overlay_{step:06d}.pt',f'{sum(v.numel() for v in overlay.values())*2/2**30:.1f} GB')
            dist.barrier()
    if rank==0:json.dump(dict(status='complete',optimizer=a.optimizer,steps=total,marks=marks,minutes=(time.time()-t0)/60,trainable=ntr,frozen=frozen,groups=groups,world=world),open(out/'done.json','w'),indent=1)
    log('[done]',(time.time()-t0)/60,'min');dist.destroy_process_group()


if __name__=='__main__':main()
