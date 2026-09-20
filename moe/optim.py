"""Optimizers for the 35B-A3B optimizer A/B (runs/moe_optimizer_ab_v1): AdamW and Muon, both with FP32 master weights over BF16 parameters.

MasterAdamW is the class used by every calibrated-RL run (decider/optim.py), copied so the public package's import path can be used
for everything else. MasterMuon follows Jordan et al.'s Muon (momentum, Nesterov, five Newton-Schulz iterations to an orthogonal
update) with Moonlight's RMS matching (update scaled by 0.2 * sqrt(max(A, B))) so the AdamW learning-rate schedule carries over.
Parameters with more than two dimensions are treated as stacks of matrices over the leading dimensions (routed experts, if trained)."""
import math

import torch

NS_COEFFICIENTS=(3.4445,-4.7750,2.0315)


class MasterAdamW(torch.optim.Optimizer):
    def __init__(self,params,lr=1e-5,betas=(0.9,0.95),weight_decay=0.0,eps=1e-8):
        super().__init__(params,dict(lr=lr,betas=betas,weight_decay=weight_decay,eps=eps))

    @torch.no_grad()
    def step(self,closure=None):
        for group in self.param_groups:
            beta1,beta2=group['betas']
            for p in group['params']:
                if p.grad is None:continue
                state=self.state[p]
                if not state:
                    state['step']=0;state['master']=p.detach().float().clone();state['exp_avg']=torch.zeros_like(state['master']);state['exp_avg_sq']=torch.zeros_like(state['master'])
                state['step']+=1;grad=p.grad.float();m,v,master=state['exp_avg'],state['exp_avg_sq'],state['master']
                m.lerp_(grad,1-beta1);v.mul_(beta2).addcmul_(grad,grad,value=1-beta2)
                bc1=1-beta1**state['step'];bc2=1-beta2**state['step'];denom=(v/bc2).sqrt_().add_(group['eps'])
                if group['weight_decay']:master.mul_(1-group['lr']*group['weight_decay'])
                master.addcdiv_(m,denom,value=-group['lr']/bc1);p.copy_(master.to(p.dtype))

    def reset_first_moment(self):
        for state in self.state.values():
            if 'exp_avg' in state:state['exp_avg'].zero_()


def newton_schulz(G,steps=5,eps=1e-7):
    """Orthogonalize the last two dimensions of G (any leading batch dimensions), in bfloat16 as in the reference implementation."""
    a,b,c=NS_COEFFICIENTS;X=G.to(torch.bfloat16);X=X/(X.norm(dim=(-2,-1),keepdim=True)+eps);transposed=X.shape[-2]>X.shape[-1]
    if transposed:X=X.mT
    for _ in range(steps):
        A=X@X.mT;B=b*A+c*(A@A);X=a*X+B@X
    return X.mT if transposed else X


class MasterMuon(torch.optim.Optimizer):
    """Muon with FP32 master weights. lr is the AdamW-equivalent rate: the orthogonal update is multiplied by 0.2 * sqrt(max(A, B))."""
    def __init__(self,params,lr=1e-5,momentum=0.95,nesterov=True,weight_decay=0.0,ns_steps=5):
        super().__init__(params,dict(lr=lr,momentum=momentum,nesterov=nesterov,weight_decay=weight_decay,ns_steps=ns_steps))

    @torch.no_grad()
    def step(self,closure=None):
        for group in self.param_groups:
            mu=group['momentum']
            for p in group['params']:
                if p.grad is None:continue
                assert p.ndim>=2,'Muon needs matrices'
                state=self.state[p]
                if not state:state['step']=0;state['master']=p.detach().float().clone();state['momentum']=torch.zeros_like(state['master'])
                state['step']+=1;grad=p.grad.float();buf,master=state['momentum'],state['master'];buf.mul_(mu).add_(grad)
                g=grad.add(buf,alpha=mu) if group['nesterov'] else buf
                update=newton_schulz(g,group['ns_steps']).float()*(0.2*math.sqrt(max(p.shape[-2],p.shape[-1])))
                if group['weight_decay']:master.mul_(1-group['lr']*group['weight_decay'])
                master.add_(update,alpha=-group['lr']);p.copy_(master.to(p.dtype))

    def reset_first_moment(self):
        for state in self.state.values():
            if 'momentum' in state:state['momentum'].zero_()


def is_muon_param(name,p):
    """Matrices of the transformer blocks. Embeddings, the head, norms, routers, the shared-expert gate and every 1-D or conv parameter stay on AdamW."""
    if p.ndim<2 or min(p.shape[-2:])<64:return False
    return not any(k in name for k in ('embed_tokens','lm_head','mlp.gate.','shared_expert_gate','norm','conv1d'))
