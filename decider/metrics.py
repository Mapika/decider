import numpy as np


def ece(conf, correct, bins=15):
    conf = np.asarray(conf); correct = np.asarray(correct, dtype=float)
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(e)


def aurc(conf, correct):
    """Area under risk-coverage curve (lower is better)."""
    order = np.argsort(-np.asarray(conf)); c = np.asarray(correct, dtype=float)[order]
    risk = np.cumsum(1 - c) / np.arange(1, len(c) + 1)
    return float(risk.mean())


def sel_acc(conf, correct, coverage):
    order = np.argsort(-np.asarray(conf)); c = np.asarray(correct, dtype=float)[order]
    n = max(1, int(round(coverage * len(c))))
    return float(c[:n].mean())


def summarize(probs, golds, nopts):
    """probs [N,K] (masked entries 0), golds [N], nopts [N]."""
    probs = np.asarray(probs); golds = np.asarray(golds); nopts = np.asarray(nopts)
    pred = probs.argmax(1); conf = probs.max(1); correct = (pred == golds)
    p_gold = probs[np.arange(len(golds)), golds]
    nll = -np.log(np.clip(p_gold, 1e-12, 1)).mean()
    onehot = np.zeros_like(probs); onehot[np.arange(len(golds)), golds] = 1
    brier = ((probs - onehot) ** 2).sum(1).mean()
    return dict(n=int(len(golds)), acc=float(correct.mean()), nll=float(nll), brier=float(brier), ece=ece(conf, correct),
                aurc=aurc(conf, correct), acc_at_80=sel_acc(conf, correct, 0.8), acc_at_50=sel_acc(conf, correct, 0.5),
                chance=float((1.0 / nopts).mean()), mean_conf=float(conf.mean()))
