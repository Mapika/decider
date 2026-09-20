"""Evaluate a base model plus a trainable-parameter overlay saved by train_ab.py on the public regression set (the eval half of the
mixture), with the public evaluate.py protocol (state-first layout, max_ctx 1536, batch 32, temperature 1; the temperature is fitted
afterwards by decider.report). `--overlay none` evaluates the base model itself (the 0% mark).
    cd . && PYTHONPATH=. CUDA_VISIBLE_DEVICES=6 python moe/eval_overlay.py \
        --overlay .../adamw/overlay_003000.pt --out .../adamw/eval_003000"""
import argparse
import json
import os
import pickle
import time

import torch

from decider import data as D
from decider.evaluate import aggregate, run_eval
import importlib.util
from pathlib import Path
spec=importlib.util.spec_from_file_location('ab_model',Path(__file__).resolve().parent/'model.py');ab_model=importlib.util.module_from_spec(spec);spec.loader.exec_module(ab_model)

p=argparse.ArgumentParser();p.add_argument('--overlay',required=True);p.add_argument('--out',required=True);p.add_argument('--model',default='data/models/Qwen3.5-35B-A3B-Base')
p.add_argument('--data',default='data/mixture_full.pkl');p.add_argument('--limit',type=int,default=0);p.add_argument('--bs',type=int,default=32);p.add_argument('--max_ctx',type=int,default=1536)
a=p.parse_args();t0=time.time();_,evals=D.load_cache(a.data)
if a.limit:evals={k:v[:a.limit] for k,v in evals.items()}
m=ab_model.MoEDecisionModel(a.model,grad_ckpt=False).cuda().eval()
if a.overlay!='none':
    overlay=torch.load(a.overlay,map_location='cpu');missing,unexpected=m.lm.load_state_dict(overlay,strict=False);assert not unexpected,unexpected[:5];assert len(missing)>0
    applied=len(overlay)
else:applied=0
with torch.no_grad():res,dump=run_eval(m,evals,bs=a.bs,temperature=1.0,max_ctx=a.max_ctx,layout='state_first')
agg=aggregate(res);os.makedirs(a.out,exist_ok=True)
json.dump(dict(results=res,agg=agg,model=a.model,overlay=a.overlay,overlay_tensors=applied,engine='',seconds=time.time()-t0),open(f'{a.out}/eval.json','w'),indent=1);pickle.dump(dump,open(f'{a.out}/preds.pkl','wb'))
print('[agg]',json.dumps(agg),'seconds',round(time.time()-t0))
