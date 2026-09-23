"""Serve a decider checkpoint through vLLM and read the slot the way the decider package does: the prompt ends at "Answer: (",
one token is sampled with the label tokens as the only allowed ids, and the processed logits vLLM returns for them are
softmaxed at the stored temperature. Used to check the NVFP4 export (ModelOpt hf_quant_config) against the bf16 predictions
of the same rows: accuracy, NLL, argmax agreement, mean total variation, and rows per second.
    .venv_vllm/bin/python moe/vllm_check.py <model dir> <predictions dir of the bf16 run> <out.json> [fixture ...]
Fixtures are runs/head_to_head_v1/requests/<name>.jsonl rows (id, context, question, options, gold)."""
import json,math,sys,time
from pathlib import Path

from vllm import LLM,SamplingParams

sys.path.insert(0,'.')
from decider.infer import Example,Q
from decider.prompt import build,label_table,chat_for

model=Path(sys.argv[1]);ref_dir=Path(sys.argv[2]);out_path=Path(sys.argv[3]);fixtures=sys.argv[4:] or ['typesafe','general_validation']
cfg=json.load(open(model/'decider_config.json'));T=float(cfg['temperature'])
class _NoShuffle:
    def shuffle(self,x):pass
    def sample(self,xs,k):return xs[:k]
llm=LLM(model=str(model),max_model_len=34816,gpu_memory_utilization=0.9,logprobs_mode='processed_logits',enable_prefix_caching=False,max_num_seqs=64,max_logprobs=256)
tok=llm.get_tokenizer();_,lab_ids,_=label_table(tok);chat=chat_for(tok,cfg)  # chat-layout models (decider_config.json "layout": "chat")
report=dict(model=str(model),temperature=T,quantization=cfg.get('quantization'),fixtures={})
for fx in fixtures:
    rows=[json.loads(l) for l in open(f'runs/head_to_head_v1/requests/{fx}.jsonl',encoding='utf-8')]
    ref={r['id']:(r.get('answer',r)) for r in (json.loads(l) for l in open(ref_dir/f'{fx}.jsonl',encoding='utf-8'))}
    items=[build(Example(r['context'],[Q(r['question'],list(r['options']),0)],'infer'),tok,_NoShuffle(),max_options=255,max_ctx_tokens=32768,chat=chat) for r in rows]
    prompts=[dict(prompt_token_ids=it['ids']) for it in items]
    params=[SamplingParams(max_tokens=1,temperature=1.0,logprobs=it['nopts'][0],allowed_token_ids=lab_ids[:it['nopts'][0]]) for it in items]
    t=time.time();outs=llm.generate(prompts,params,use_tqdm=False);sec=time.time()-t
    acc=nll=tv=agree=0.
    for r,it,o in zip(rows,items,outs):
        lp=o.outputs[0].logprobs[0];n=it['nopts'][0];logits=[lp[lab_ids[j]].logprob if lab_ids[j] in lp else -1e9 for j in range(n)]
        m=max(logits);z=[math.exp((x-m)/T) for x in logits];s=sum(z);p=[x/s for x in z]
        gold=r['gold'];j=max(range(n),key=p.__getitem__);acc+=j==gold;nll+=-math.log(max(p[gold],1e-12))
        rp=ref[r['id']]['probabilities'];refp=[rp.get(o_,0.) for o_ in r['options']];tv+=.5*sum(abs(a-b) for a,b in zip(p,refp));agree+=(r['options'][j]==ref[r['id']]['choice'])
    n=len(rows);report['fixtures'][fx]=dict(rows=n,accuracy=acc/n,nll=nll/n,argmax_agreement_with_bf16=agree/n,mean_total_variation_to_bf16=tv/n,seconds=sec,rows_per_second=n/sec)
    print(fx,json.dumps({k:(round(v,4) if isinstance(v,float) else v) for k,v in report['fixtures'][fx].items()}),flush=True)
out_path.parent.mkdir(parents=True,exist_ok=True);out_path.write_text(json.dumps(report,indent=1));print('written',out_path)
