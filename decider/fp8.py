"""FP8 (e4m3) linear layers for Hopper via torch._scaled_mm.
Weights: per-output-channel scales, quantised once.  Activations: per-token dynamic scales.
Under torch.compile the quantisation ops fuse into the surrounding elementwise work."""
import torch, torch.nn as nn

E4M3_MAX = 448.0


def _quant_rowwise(x):
    s = x.abs().amax(dim=-1, keepdim=True).float().clamp(min=1e-12) / E4M3_MAX
    return (x.float() / s).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn), s


class FP8Linear(nn.Module):
    def __init__(self, lin: nn.Linear):
        super().__init__()
        wq, sw = _quant_rowwise(lin.weight.detach())          # [N,K] fp8, [N,1]
        self.register_buffer("wq", wq.contiguous())               # [N,K]; passed as wq.t() -> [K,N] column-major, as _scaled_mm wants
        self.register_buffer("sw_t", sw.t().contiguous())        # [1,N]
        self.bias = None if lin.bias is None else nn.Parameter(lin.bias.detach().clone(), requires_grad=False)
        self.in_features, self.out_features = lin.in_features, lin.out_features
        self.out_dtype = lin.weight.dtype

    def forward(self, x):
        shp = x.shape[:-1]
        x2 = x.reshape(-1, self.in_features)
        xq, sx = _quant_rowwise(x2)
        y = torch._scaled_mm(xq, self.wq.t(), scale_a=sx, scale_b=self.sw_t, bias=self.bias, out_dtype=self.out_dtype)
        return y.reshape(*shp, self.out_features)


def convert_to_fp8(model, skip=("lm_head",), min_dim=1024):
    """Replace nn.Linear (with in/out >= min_dim) by FP8Linear in place. Returns count."""
    n = 0
    for name, mod in list(model.named_modules()):
        for cname, child in list(mod.named_children()):
            full = f"{name}.{cname}" if name else cname
            if isinstance(child, nn.Linear) and not any(s in full for s in skip) and min(child.in_features, child.out_features) >= min_dim:
                setattr(mod, cname, FP8Linear(child)); n += 1
    return n


if __name__ == "__main__":
    import time
    lin = nn.Linear(2048, 6144, bias=False).cuda().to(torch.bfloat16)
    f8 = FP8Linear(lin)
    x = torch.randn(8192, 2048, device="cuda", dtype=torch.bfloat16)
    ref = lin(x); got = f8(x)
    print("rel err", ((ref.float() - got.float()).abs().mean() / ref.float().abs().mean()).item())
    for f, name in [(lin, "bf16 linear"), (f8, "fp8 linear (eager)"), (torch.compile(f8), "fp8 linear (compiled)")]:
        for _ in range(3): f(x)
        torch.cuda.synchronize(); t = time.time()
        for _ in range(20): f(x)
        torch.cuda.synchronize(); print(f"{name:24s} {(time.time()-t)/20*1000:.3f} ms")
