"""The public DecisionModel with the MoE expert kernel selected: identical slot readout, weights loaded with experts_implementation='grouped_mm'
(the eager expert loop is 13x slower, see MOE_THROUGHPUT_PROBE.md) and non-reentrant gradient checkpointing (safe under data-parallel training)."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from decider.model import DecisionModel
from decider.prompt import letter_ids


class MoEDecisionModel(DecisionModel):
    def __init__(self,name,grad_ckpt=True,experts_implementation='grouped_mm'):
        torch.nn.Module.__init__(self)
        if grad_ckpt:torch.backends.cuda.enable_cudnn_sdp(False)
        self.tok=AutoTokenizer.from_pretrained(name);self.lm=AutoModelForCausalLM.from_pretrained(name,dtype=torch.bfloat16,experts_implementation=experts_implementation);self.lm.config.use_cache=False
        if grad_ckpt:self.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs=dict(use_reentrant=False))
        self.register_buffer('letters',torch.tensor(letter_ids(self.tok)),persistent=False)
