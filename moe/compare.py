"""Compare the two arms of runs/moe_optimizer_ab_v1: training cross-entropy against steps (same examples at every step), and the
public regression set at each mark (accuracy / NLL / ECE, in-task and held-out, at temperature 1 and at the in-task-fitted
temperature), with a paired bootstrap over tasks between the arms at the same mark and for Muon at 50% against AdamW at 100%.
    cd . && PYTHONPATH=. python moe/compare.py"""
import json
from pathlib import Path
import pickle

import numpy as np

from decider.metrics import summarize
from decider.report import apply_temperature, fit_temperature

ROOT=Path('runs/moe_optimizer_ab_v1');ARMS=('adamw','muon');KEYS=('acc','nll','ece')


def per_task(run,T):
    res=json.load(open(run/'eval.json'))['results'];dump=pickle.load(open(run/'preds.pkl','rb'));out={}
    for t in res:
        d=dump[t];ok=d['golds']>=0;s=summarize(apply_temperature(d['probs'][ok],T),d['golds'][ok],d['nopts'][ok]);s['heldout']=res[t]['heldout'];out[t]=s
    return out


def agg(pt,split):return {k:float(np.mean([v[k] for v in pt.values() if v['heldout']==(split=='heldout')])) for k in KEYS}


def paired(pa,pb,split,key,seed=7,n=4000):
    tasks=sorted(t for t in pa if pa[t]['heldout']==(split=='heldout') and t in pb);d=np.array([pa[t][key]-pb[t][key] for t in tasks]);rng=np.random.default_rng(seed)
    idx=rng.integers(0,len(d),size=(n,len(d)));return dict(tasks=len(tasks),delta=float(d.mean()),ci95=np.quantile(d[idx].mean(1),[.025,.975]).tolist())


def main():
    out=dict(curves={},evals={},paired={});
    for arm in ARMS:
        c=json.load(open(ROOT/arm/'curve.json'));out['curves'][arm]=dict(steps=[r['step'] for r in c],ce=[r['ce'] for r in c],tokens_per_second=float(np.mean([r['tokens_per_second'] for r in c[1:]])) if len(c)>1 else None)
        for ev in sorted((ROOT/arm).glob('eval_*')):
            if not (ev/'preds.pkl').exists():continue
            res=json.load(open(ev/'eval.json'))['results'];dump=pickle.load(open(ev/'preds.pkl','rb'));T=fit_temperature(dump,[t for t in res if not res[t]['heldout']])
            p1=per_task(ev,1.0);pT=per_task(ev,T);out['evals'][f'{arm}/{ev.name}']=dict(T=float(T),t1=dict(in_task=agg(p1,'in'),heldout=agg(p1,'heldout')),fitted=dict(in_task=agg(pT,'in'),heldout=agg(pT,'heldout')))
    base=ROOT/'base'/'eval_000000'
    if (base/'preds.pkl').exists():
        res=json.load(open(base/'eval.json'))['results'];dump=pickle.load(open(base/'preds.pkl','rb'));T=fit_temperature(dump,[t for t in res if not res[t]['heldout']]);p1=per_task(base,1.0);pT=per_task(base,T)
        out['evals']['base/eval_000000']=dict(T=float(T),t1=dict(in_task=agg(p1,'in'),heldout=agg(p1,'heldout')),fitted=dict(in_task=agg(pT,'in'),heldout=agg(pT,'heldout')))
    marks=sorted({e.split('/')[1] for e in out['evals'] if e.startswith('adamw/')})
    for m in marks:
        a=ROOT/'adamw'/m;b=ROOT/'muon'/m
        if (a/'preds.pkl').exists() and (b/'preds.pkl').exists():
            pa=per_task(a,1.0);pb=per_task(b,1.0)
            for split in ('in','heldout'):
                for key in KEYS:out['paired'][f'muon_minus_adamw|{m}|{split}|{key}']=paired(pb,pa,split,key)
    if len(marks)>=2:
        half=marks[len(marks)//2-1] if len(marks)%2==0 else marks[len(marks)//2];full=marks[-1];a=ROOT/'adamw'/full;b=ROOT/'muon'/half
        if (a/'preds.pkl').exists() and (b/'preds.pkl').exists():
            pa=per_task(a,1.0);pb=per_task(b,1.0)
            for split in ('in','heldout'):
                for key in KEYS:out['paired'][f'muon@{half}_minus_adamw@{full}|{split}|{key}']=paired(pb,pa,split,key)
    (ROOT/'comparison.json').write_text(json.dumps(out,indent=1))
    for k,v in out['evals'].items():print(f"{k:28s} T={v['T']:.2f} in-task acc/nll/ece {v['fitted']['in_task']['acc']:.3f}/{v['fitted']['in_task']['nll']:.3f}/{v['fitted']['in_task']['ece']:.3f}  held-out {v['fitted']['heldout']['acc']:.3f}/{v['fitted']['heldout']['nll']:.3f}/{v['fitted']['heldout']['ece']:.3f}")
    for k,v in out['paired'].items():
        if k.endswith('|acc') or k.endswith('|nll'):print(f"{k:60s} {v['delta']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}] over {v['tasks']} tasks")


if __name__=='__main__':main()
