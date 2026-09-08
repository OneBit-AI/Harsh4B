"""PackedLinear: an nn.Module that keeps E2M-ATQ buffers packed in VRAM.

Replaces TWLA's QLinear. Same semantics (run_twla.py:167):
    x = L @ x @ R          KOTMS rotation on the activation
    return x @ W.T + b     with W reconstructed from the packed buffers

Decode (1 token)  -> Triton GEMV, W never materialised.
Prefill (n tokens)-> batched Triton GEMM (packed_gemm) or dequantise ONE layer into scratch.
"""
import torch, torch.nn as nn, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gemv_triton import packed_gemv
from gemv_triton_v4 import packed_gemv_v4, packed_gemv_opt, packed_gemv_dense, unpack_t1_dense
from gemm_triton import packed_gemm

def _v4_w1(x, p, b=None): return packed_gemv_v4(x, p, b, num_warps=1)
def _v4_w2(x, p, b=None): return packed_gemv_v4(x, p, b, num_warps=2)
def _v4_dense(x, p, b=None): return packed_gemv_dense(x, p, b, num_warps=1, num_stages=2)

KERNELS = {
    "v1": packed_gemv,
    "v4_w1": _v4_w1,
    "v4_w2": _v4_w2,
    "v4_opt": _v4_dense,
    "dense": _v4_dense,
}
KERNEL = [_v4_dense]
def set_kernel(name): KERNEL[0] = KERNELS[name]


def _unpack2_rows(c, k):
    n, pk = c.shape
    t = torch.stack([c & 3, (c >> 2) & 3, (c >> 4) & 3, (c >> 6) & 3], -1).reshape(n, pk * 4)
    return t[:, :k]


class PackedLinear(nn.Module):
    def __init__(self, p, bias=None, L=None, R=None, dim_l=0, dim_r=0):
        super().__init__()
        self.OC, self.IC = p["shape"]
        self.NB, self.BS = p["nblocks"], p["blocksize"]
        self.n_T1 = p["n_T1"]
        for n in ("maskid", "T0", "T1c", "base", "mu", "a0", "a1"):
            self.register_buffer(n, p[n], persistent=False)
        
        # Dense 2-bit unpacked T1 for ultra-fast single-token decode (avoids scattered gather)
        if "T1_dense" in p:
            self.register_buffer("T1_dense", p["T1_dense"], persistent=False)
        else:
            self.register_buffer("T1_dense", unpack_t1_dense(self), persistent=False)

        self.register_buffer("bias", bias, persistent=False) if bias is not None else setattr(self, "bias", None)
        if L is not None:
            self.register_buffer("L", L, persistent=False)
            self.register_buffer("R", R, persistent=False)
            self.dim_l, self.dim_r = int(dim_l), int(dim_r)
        else:
            self.L = None

    def _p(self):
        return dict(shape=(self.OC, self.IC), nblocks=self.NB, blocksize=self.BS,
                    maskid=self.maskid, T0=self.T0, T1c=self.T1c, T1_dense=self.T1_dense, base=self.base,
                    mu=self.mu, a0=self.a0, a1=self.a1)

    @torch.no_grad()
    def dequant(self):
        """Reconstruct W [OC, IC] fp16 on device. Used for testing."""
        mid = _unpack2_rows(self.maskid, self.IC)
        t0c = _unpack2_rows(self.T0, self.IC)
        T0 = torch.where(t0c == 1, 1.0, torch.where(t0c == 2, -1.0, 0.0)).to(torch.float32)
        is_o2 = mid < 2
        pad = self.NB * self.BS - self.IC
        o2p = torch.nn.functional.pad(is_o2, (0, pad), value=False) if pad else is_o2
        blk = o2p.view(self.OC, self.NB, self.BS).to(torch.int64)
        local = torch.cumsum(blk, -1) - blk
        idx = (self.base.to(torch.int64)[:, :, None] + local).view(self.OC, -1)[:, :self.IC]
        byte = self.T1c[(idx >> 2).clamp_(0, self.T1c.numel() - 1)]
        code = (byte >> ((idx & 3) * 2).to(torch.uint8)) & 3
        T1 = torch.where(is_o2, torch.where(code == 1, 1.0, torch.where(code == 2, -1.0, 0.0)), 0.0).to(torch.float32)
        mu = self.mu.float(); a0 = self.a0.float(); a1 = self.a1.float()
        e = torch.arange(self.NB, device=mid.device).repeat_interleave(self.BS)[:self.IC]
        MU = torch.zeros(self.OC, self.IC, device=mid.device)
        A0 = torch.zeros_like(MU); A1 = torch.zeros_like(MU)
        for m in range(mu.shape[0]):
            sel = (mid == m)
            MU = torch.where(sel, mu[m][:, e], MU)
            A0 = torch.where(sel, a0[m][:, e], A0)
            if m < a1.shape[0]:
                A1 = torch.where(sel, a1[m][:, e], A1)
        return (MU + A0 * T0 + A1 * T1).half()

    def forward(self, x):
        in_dtype = x.dtype
        init_shape = x.shape
        xh = x.half() if in_dtype != torch.float16 else x
        if self.L is not None:
            xh = xh.reshape(-1, self.dim_l, self.dim_r)
            xh = self.L @ xh @ self.R
            xh = xh.reshape(init_shape)
        ntok = xh.numel() // self.IC
        if ntok == 1:
            out = KERNEL[0](xh, self._p(), self.bias)
        else:
            out = packed_gemm(xh, self._p(), self.bias, NT=16)
        out = out.reshape(*init_shape[:-1], self.OC)
        return out.to(in_dtype) if in_dtype != torch.float16 else out
