"""Imported inside every vLLM worker process by decider.serve_vllm (worker_extension_cls), before the model runner is built.

* Turns the cuDNN scaled-dot-product-attention backend off in the worker, as every decider model process does
  (decider.engine.set_attention_backend_policy: on Blackwell with torch 2.13/2.14 it returns wrong output for masked
  attention).  vLLM's own attention layers do not use SDPA; this covers any torch SDPA call inside the worker.
* With DECIDER_VLLM_WIDE_LOGPROBS=1 (the default) raises vLLM's cap on `logprob_token_ids` per request from 128 to
  WIDE_LOGPROB_IDS (256), so a question with up to 255 options is read in one request.  The cap is a module constant of
  vllm.sampling_params that the model runner copies at import (vllm/v1/worker/gpu/sample/logprob.py sizes its per-request
  id table with it); both are set here, before the runner allocates that table.  serve_vllm asks every worker for the cap
  it ended up with (`logprob_ids_cap`) and splits wider questions into several requests when a worker still has 128."""
import os

import torch

if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
    torch.backends.cuda.enable_cudnn_sdp(False)

WIDE_LOGPROB_IDS = 256


def raise_logprob_cap(n=WIDE_LOGPROB_IDS):
    """Set vLLM's per-request `logprob_token_ids` cap to at least n in this process.  -> the cap now in force."""
    import vllm.sampling_params as VSP
    VSP.MAX_LOGPROB_TOKEN_IDS = max(int(VSP.MAX_LOGPROB_TOKEN_IDS), n)
    try:
        import vllm.v1.worker.gpu.sample.logprob as L
        L.MAX_LOGPROB_TOKEN_IDS = max(int(L.MAX_LOGPROB_TOKEN_IDS), n)
    except ImportError:                     # another vLLM layout: the worker reports what it has, serve_vllm splits
        pass
    return logprob_ids_cap()


def logprob_ids_cap():
    import vllm.sampling_params as VSP
    cap = int(VSP.MAX_LOGPROB_TOKEN_IDS)
    try:
        import vllm.v1.worker.gpu.sample.logprob as L
        cap = min(cap, int(L.MAX_LOGPROB_TOKEN_IDS))
    except ImportError:
        pass
    return cap


if os.environ.get("DECIDER_VLLM_WIDE_LOGPROBS", "1") == "1":
    raise_logprob_cap()


class WorkerExtension:
    def logprob_ids_cap(self):
        return logprob_ids_cap()

    def cudnn_sdp_enabled(self):
        return bool(torch.backends.cuda.cudnn_sdp_enabled())

    def cache_block_size(self):
        """The prefix-cache block size the engine settled on (for hybrid models it is raised until one attention page holds
        one recurrent-state page, e.g. 784 tokens for Qwen3.6-27B with a bf16 KV cache)."""
        return int(self.vllm_config.cache_config.block_size)
