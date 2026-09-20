"""Post-training NVFP4 quantization of the merged 35B-A3B decider with NVIDIA ModelOpt (runs/moe_optimizer_ab_v1/muon/model_nvfp4).
Calibrates the default NVFP4 recipe (weights and activations FP4 with FP8 block scales; ModelOpt's preset already leaves the delta-net
projections, routers, shared-expert gate and head in bf16) on 512 training prompts, measures fake-quant accuracy on a fixed subset of the
public regression set against the bf16 model on the same subset, and exports an HF checkpoint for vLLM / TensorRT-LLM.
    cd decider-public && PYTHONPATH=. CUDA_VISIBLE_DEVICES=3 python moe/ptq_nvfp4.py"""
import json,pickle,random,shutil,time
from pathlib import Path
import torch
import modelopt.torch.quantization as mtq
from modelopt.torch.export import export_hf_checkpoint
from decider import data as D
from decider.evaluate import aggregate,run_eval
from decider.model import DecisionModel,collate

R=Path('.');M=R/'runs/moe_optimizer_ab_v1/muon/model';OUT=R/'runs/moe_optimizer_ab_v1/muon/model_nvfp4';REP=R/'runs/moe_optimizer_ab_v1/nvfp4_ptq.json'
CAL=512;LIMIT=150;t0=time.time();rep=dict(config='NVFP4_DEFAULT_CFG',calibration_prompts=CAL,eval_limit_per_task=LIMIT)
m=DecisionModel(str(M),grad_ckpt=False).cuda().eval();tok=m.tok
_,evals=D.load_cache('data/mixture_full.pkl');sub={k:v[:LIMIT] for k,v in evals.items()}
with torch.no_grad():res_bf16,_=run_eval(m,sub,bs=32,max_ctx=1536,layout='state_first',log=lambda *a:None)
rep['bf16']=aggregate(res_bf16);print('bf16 subset',json.dumps(rep['bf16']),flush=True)
items=pickle.load(open(R/'runs/moe_optimizer_ab_v1/items.pkl','rb'));rng=random.Random(11);cal=[items[i] for i in rng.sample(range(len(items)),CAL)];cal=[it for it in cal if len(it['ids'])<=2048]
def forward_loop(model):
    with torch.no_grad():
        for s in range(0,len(cal),8):
            b=collate(cal[s:s+8],tok.pad_token_id);model(input_ids=b['input_ids'].cuda(),attention_mask=b['attention_mask'].cuda())
t=time.time();mtq.quantize(m.lm,mtq.NVFP4_DEFAULT_CFG,forward_loop);rep['calibration_seconds']=time.time()-t;print('quantized in',round(rep['calibration_seconds']),'s',flush=True)
summary=[];mtq.print_quant_summary(m.lm)
with torch.no_grad():res_q,_=run_eval(m,sub,bs=32,max_ctx=1536,layout='state_first',log=lambda *a:None)
rep['nvfp4_fake_quant']=aggregate(res_q);rep['per_task']={k:dict(bf16=res_bf16[k]['acc'],nvfp4=res_q[k]['acc'],heldout=res_q[k]['heldout']) for k in res_q};print('nvfp4 subset',json.dumps(rep['nvfp4_fake_quant']),flush=True)
OUT.mkdir(parents=True,exist_ok=True);export_hf_checkpoint(m.lm,export_dir=str(OUT),max_shard_size='5GB');tok.save_pretrained(str(OUT))
cfg=json.load(open(M/'decider_config.json'));cfg['version']=cfg['version']+'-nvfp4';cfg['quantization']='NVFP4 (ModelOpt 0.46.1, NVFP4_DEFAULT_CFG, 512 calibration prompts); for vLLM / TensorRT-LLM on Blackwell; the bf16 checkpoint is the reference';json.dump(cfg,open(OUT/'decider_config.json','w'),indent=1)
rep['export_dir']=str(OUT);rep['export_files']=sorted(p.name for p in OUT.iterdir());rep['export_gb']=sum(p.stat().st_size for p in OUT.iterdir())/2**30;rep['seconds']=time.time()-t0
json.dump(rep,open(REP,'w'),indent=1);print('done',json.dumps({k:v for k,v in rep.items() if k not in ('per_task','export_files')}))
