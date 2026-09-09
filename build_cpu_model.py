#!/usr/bin/env python3
"""
OneBit AI - Standalone Mac CPU Model Builder for Qwen3-4B LATTICE.
Reconstructs and loads the packed ternary weights with KOTMS activation rotations
directly on CPU using PyTorch bfloat16.
Caches reconstructed weights to disk for instant subsequent startups.
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM
from accelerate import init_empty_weights

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def unpack2(c: torch.Tensor, k: int) -> torch.Tensor:
    n, pk = c.shape
    t = torch.stack([c & 3, (c >> 2) & 3, (c >> 4) & 3, (c >> 6) & 3], -1).reshape(n, pk * 4)
    return t[:, :k]


def unpack2_flat(c: torch.Tensor, n: int) -> torch.Tensor:
    t = torch.stack([c & 3, (c >> 2) & 3, (c >> 4) & 3, (c >> 6) & 3], -1).reshape(-1)
    return t[:n]


def trit(t: torch.Tensor) -> torch.Tensor:
    T = torch.zeros_like(t, dtype=torch.float32)
    T = torch.where(t == 1, torch.ones_like(T), T)
    T = torch.where(t == 2, torch.full_like(T, -1), T)
    return T


class CPURotatedLinear(nn.Module):
    """CPU implementation of Rotated Linear layer for KOTMS activations."""
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor = None,
                 L: torch.Tensor = None, R: torch.Tensor = None,
                 dim_l: int = 0, dim_r: int = 0):
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.bias = nn.Parameter(bias, requires_grad=False) if bias is not None else None
        if L is not None and R is not None:
            self.register_buffer("L", L)
            self.register_buffer("R", R)
            self.dim_l = int(dim_l)
            self.dim_r = int(dim_r)
        else:
            self.L = None
            self.R = None
            self.dim_l = 0
            self.dim_r = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.L is not None:
            orig_shape = x.shape
            x = (self.L @ x.reshape(-1, self.dim_l, self.dim_r) @ self.R).reshape(orig_shape)
        return F.linear(x, self.weight, self.bias)


def build_cpu(model_id="Qwen/Qwen3-4B",
              packed_path=os.path.join(SCRIPT_DIR, "Qwen3-4B-LATTICE_pd0.01.lat.pt"),
              rot_path=os.path.join(SCRIPT_DIR, "rot_4b_LR.pt"),
              cache_path=os.path.join(SCRIPT_DIR, "Qwen3-4B-LATTICE_cpu_bf16_absorbed.pt"),
              dtype=torch.bfloat16,
              verbose=True):
    
    # Configure optimal threading for Intel CPU (avoid hyperthreading contention)
    torch.set_num_threads(4)
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass
    os.environ["KMP_BLOCKTIME"] = "0"

    cfg = AutoConfig.from_pretrained(model_id)
    if verbose:
        print("Creating skeleton model on CPU...", flush=True)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(cfg, torch_dtype=dtype)

    # 1. Check if absorbed weights cache exists (fastest path: 100% native PyTorch C++ nn.Linear)
    if os.path.exists(cache_path):
        if verbose:
            print(f"Loading absorbed CPU weights from {os.path.basename(cache_path)}...", flush=True)
        t0 = time.time()
        cpu_weights = torch.load(cache_path, map_location="cpu", weights_only=False)
        model.load_state_dict(cpu_weights, strict=False, assign=True)
        model.eval()
        if verbose:
            print(f"Model fully assembled on CPU with absorbed rotations in {time.time()-t0:.1f}s!", flush=True)
        return model

    # 2. Check if unabsorbed cache exists, and absorb on the fly
    unabs_cache = os.path.join(SCRIPT_DIR, "Qwen3-4B-LATTICE_cpu_bf16.pt")
    if os.path.exists(unabs_cache) and os.path.exists(rot_path):
        if verbose:
            print(f"Absorbing KOTMS rotations into weights from {os.path.basename(unabs_cache)}...", flush=True)
        t0 = time.time()
        sd = torch.load(unabs_cache, map_location="cpu", weights_only=False)
        rot = torch.load(rot_path, map_location="cpu", weights_only=False)
        for k in list(sd.keys()):
            if not k.endswith(".weight"):
                continue
            mod_name = k[:-len(".weight")]
            lookup_name = mod_name.replace("model.", "", 1) if mod_name.startswith("model.") else mod_name
            rd = rot.get(lookup_name) or rot.get(mod_name)
            if rd is not None and "L" in rd and "R" in rd:
                W = sd[k]
                oc, ic = W.shape
                dim_l, dim_r = rd["dim_l"], rd["dim_r"]
                L = rd["L"].float()
                R = rd["R"].float()
                W_mat = W.float().reshape(oc, dim_l, dim_r)
                W_eff = torch.matmul(torch.matmul(L.T, W_mat), R.T).reshape(oc, ic).to(dtype)
                sd[k] = W_eff

        if verbose:
            print(f"Absorbed all rotations in {time.time()-t0:.1f}s. Saving to {os.path.basename(cache_path)}...", flush=True)
        torch.save(sd, cache_path)
        model.load_state_dict(sd, strict=False, assign=True)
        model.eval()
        return model

    # 3. Reconstruct from packed checkpoint
    if verbose:
        print(f"Reconstructing weights from packed checkpoint {os.path.basename(packed_path)}...", flush=True)
    t_start = time.time()
    pk = torch.load(packed_path, map_location="cpu", weights_only=False)
    pk.pop("__latmeta__", None)
    rot = torch.load(rot_path, map_location="cpu", weights_only=False) if os.path.exists(rot_path) else {}

    lin_keys = {k for k, v in pk.items() if isinstance(v, dict) and "maskid" in v}
    plain = {k: v for k, v in pk.items() if k not in lin_keys and torch.is_tensor(v)}

    # Plain tensors
    for k in plain:
        if plain[k].is_floating_point():
            plain[k] = plain[k].to(dtype)
    model.load_state_dict(plain, strict=False, assign=True)

    cache_dict = dict(plain)
    total_lin = len(lin_keys)
    if verbose:
        print(f"Reconstructing {total_lin} linear layers to {dtype}...", flush=True)

    for idx, k in enumerate(sorted(lin_keys)):
        v = pk[k]
        mod_name = k[:-len(".weight")]
        oc, ic = v["shape"]
        nb, bs = v["nblocks"], v["blocksize"]

        maskid = unpack2(v["maskid"], ic)
        T0 = trit(unpack2(v["T0"], ic))
        mu, a0 = v["mu"].float(), v["a0"].float()
        o2 = v["o2_idx"].tolist()
        a1c = v["a1"].float()

        T1 = torch.zeros_like(T0)
        if v["n_T1"] > 0 and len(o2):
            o2_sel = torch.zeros_like(maskid, dtype=torch.bool)
            for mid in o2:
                o2_sel |= (maskid == mid)
            T1[o2_sel] = trit(unpack2_flat(v["T1c"], v["n_T1"]))

        a1 = torch.zeros_like(a0)
        for j, mid in enumerate(o2):
            a1[mid] = a1c[j]

        W = torch.zeros(oc, ic, dtype=torch.float32)
        for b in range(nb):
            st, ed = b * bs, min((b + 1) * bs, ic)
            for mid in range(mu.shape[0]):
                sel = (maskid[:, st:ed] == mid)
                if not sel.any():
                    continue
                blk = (mu[mid, :, b:b+1] + a0[mid, :, b:b+1] * T0[:, st:ed]
                       + a1[mid, :, b:b+1] * T1[:, st:ed])
                W[:, st:ed][sel] = blk[sel]

        lookup_name = mod_name.replace("model.", "", 1) if mod_name.startswith("model.") else mod_name
        rd = rot.get(lookup_name) or rot.get(mod_name)
        if rd is not None and "L" in rd and "R" in rd:
            dim_l, dim_r = rd["dim_l"], rd["dim_r"]
            L = rd["L"].float()
            R = rd["R"].float()
            W_mat = W.reshape(oc, dim_l, dim_r)
            W_eff = torch.matmul(torch.matmul(L.T, W_mat), R.T).reshape(oc, ic)
            W = W_eff

        W_dtype = W.to(dtype)
        cache_dict[k] = W_dtype

        if verbose and (idx + 1) % 25 == 0:
            print(f"  Reconstructed [{idx+1}/{total_lin}] layers ({(idx+1)/total_lin*100:.0f}%)...", flush=True)

    elapsed = time.time() - t_start
    if verbose:
        print(f"Reconstruction completed in {elapsed:.1f}s! Saving to cache {os.path.basename(cache_path)}...", flush=True)
    torch.save(cache_dict, cache_path)
    if verbose:
        print(f"Saved CPU cache to {cache_path}! Subsequent startups will take ~5 seconds.", flush=True)

    model.load_state_dict(cache_dict, strict=False, assign=True)
    model.eval()
    return model


if __name__ == "__main__":
    m = build_cpu()
    print("Test build complete!")
