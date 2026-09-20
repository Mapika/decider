"""Tokenize the public full mixture once, with the public recipe's settings (max_ctx 16384, none_prob 0.1, max_options 255,
schema_first_prob 0.5, seed 0), so both arms of the optimizer A/B train on byte-identical items in the same order.
Run from the public checkout with PYTHONPATH pointing at it:
    python moe/prepare_items.py data/mixture_full.pkl <tokenizer path> <out.pkl>"""
import hashlib
import json
import pickle
import random
import sys
import time

import numpy as np
from transformers import AutoTokenizer

from decider import data as D
from decider.train import make_items

data,tok_path,out=sys.argv[1:4];t0=time.time();train,evals=D.load_cache(data);tok=AutoTokenizer.from_pretrained(tok_path)
items=make_items(train,tok,random.Random(0),16384,0.1,255,0.5)
for it in items:it['ids']=np.asarray(it['ids'],dtype=np.int32)
ntok=int(sum(len(it['ids']) for it in items));nq=sum(len(it['slots']) for it in items)
pickle.dump(items,open(out,'wb'),protocol=5);h=hashlib.sha256(open(out,'rb').read()).hexdigest()
meta=dict(items=len(items),questions=nq,tokens=ntok,seconds=time.time()-t0,sha256=h,data=data,tokenizer=tok_path,settings=dict(max_ctx=16384,none_prob=0.1,max_options=255,schema_first_prob=0.5,seed=0))
json.dump(meta,open(out+'.json','w'),indent=1);print(json.dumps(meta))
