#!/usr/bin/env python3
"""
OneBit AI - Standalone Mac CPU Model Builder for Qwen3-4B LATTICE.
Reconstructs and loads the packed ternary weights with KOTMS activation rotations
directly on CPU using PyTorch bfloat16.
Caches reconstructed weights to disk for instant subsequent startups.
"""

import os
import hashlib
import json
from pathlib import Path
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


def reconstruct_weight(packed, rotation=None, dtype=torch.bfloat16, chunk_rows=64):
    """Vectorized reconstruction and existing KOTMS absorption, in bounded chunks."""
    oc, ic = map(int, packed["shape"])
    bs = int(packed["blocksize"])
    modes_o2 = [int(x) for x in packed["o2_idx"]]
    expected = int(packed["n_T1"])
    source = packed["T1c"]
    weight = torch.empty((oc, ic), dtype=dtype)
    blocks = (torch.arange(ic) // bs)[None, :]
    offset = 0
    if rotation:
        left, right = rotation["L"].float(), rotation["R"].float()
        dl, dr = left.shape[0], right.shape[0]
        if dl * dr != ic or left.shape != (dl, dl) or right.shape != (dr, dr):
            raise ValueError("Rotation dimensions do not match projection")
    for start in range(0, oc, chunk_rows):
        end = min(oc, start + chunk_rows)
        mid = unpack2(packed["maskid"][start:end], ic).long()
        t0 = trit(unpack2(packed["T0"][start:end], ic))
        selected = torch.zeros_like(mid, dtype=torch.bool)
        for mode in modes_o2:
            selected |= mid == mode
        count = int(selected.sum())
        if offset + count > expected:
            raise ValueError("Compact T1 count does not match masks")
        indices = torch.arange(offset, offset + count, dtype=torch.int64)
        t1 = torch.zeros_like(t0)
        t1[selected] = trit((source[indices >> 2] >> ((indices & 3) * 2)) & 3)
        offset += count
        rows = torch.arange(start, end)[:, None]
        mu = packed["mu"][mid, rows, blocks].float()
        a0 = packed["a0"][mid, rows, blocks].float()
        a1 = torch.zeros_like(t0)
        for index, mode in enumerate(modes_o2):
            scales = packed["a1"][index, rows, blocks].float()
            a1 = torch.where(mid == mode, scales, a1)
        dense = mu + a0 * t0 + a1 * t1
        if rotation:
            dense = (left.T @ dense.reshape(-1, dl, dr) @ right.T).reshape(end-start, ic)
        weight[start:end] = dense.to(dtype)
    if offset != expected:
        raise ValueError("Compact T1 count does not match masks")
    return weight


def _source_identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    return dict(path=str(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns)


def _atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    os.replace(temporary, path)


def build_cpu(model_id="Qwen/Qwen3-4B",
              packed_path=os.path.join(SCRIPT_DIR, "Qwen3-4B-LATTICE_pd0.01.lat.pt"),
              rot_path=os.path.join(SCRIPT_DIR, "rot_4b_LR.pt"),
              cache_path=None, dtype=torch.bfloat16, verbose=True, threads=4, config=None):
    """Reuse native PyTorch Linear + absorbed rotations with file-backed weights.

    New caches are source/dtype-specific shard directories. Explicit legacy .pt
    caches remain readable; no implicit reuse across different checkpoints.
    """
    if threads < 1:
        raise ValueError("threads must be positive")
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    os.environ["KMP_BLOCKTIME"] = "0"
    if config is None:
        local_config = Path(SCRIPT_DIR) / "configs/qwen3-4b.json"
        cfg = AutoConfig.from_pretrained(str(local_config) if model_id == "Qwen/Qwen3-4B" else model_id)
    else:
        cfg = config
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(cfg, dtype=dtype)
    expected_shapes = {key: tuple(value.shape) for key, value in model.state_dict().items()}

    def assemble(weights):
        if set(weights) != set(expected_shapes):
            raise ValueError(f"CPU cache keys mismatch: missing={set(expected_shapes)-set(weights)}, extra={set(weights)-set(expected_shapes)}")
        for key, tensor in weights.items():
            if tuple(tensor.shape) != expected_shapes[key] or tensor.dtype != dtype:
                raise ValueError(f"CPU cache shape/dtype mismatch for {key}")
        model.load_state_dict(weights, strict=True, assign=True)
        model.tie_weights()
        if any(p.is_meta for p in model.parameters()):
            raise ValueError("CPU model still has uninitialized parameters")
        model.eval()
        return model

    if cache_path and Path(cache_path).is_file():
        if verbose:
            print(f"Loading explicitly selected legacy CPU cache: {cache_path}", flush=True)
        weights = torch.load(cache_path, map_location="cpu", mmap=True, weights_only=True)
        return assemble(weights)

    source = _source_identity(packed_path)
    rotation_source = _source_identity(rot_path) if rot_path and Path(rot_path).is_file() else None
    identity = dict(version=2, source=source, rotations=rotation_source, dtype=str(dtype),
                    config=cfg.to_dict())
    # Remove version-only metadata from the cache identity.
    identity["config"].pop("transformers_version", None)
    identity = json.loads(json.dumps(identity, sort_keys=True))
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    directory = Path(cache_path) if cache_path else Path(SCRIPT_DIR) / ".cache" / "cpu" / digest
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if manifest and manifest["identity"] != identity:
        raise ValueError(f"CPU cache does not match the checkpoint/config/dtype: {directory}")
    if not manifest or not manifest.get("complete"):
        if directory.exists() and any(directory.iterdir()) and manifest is None:
            raise ValueError(f"CPU cache directory is not empty and has no manifest: {directory}")
        directory.mkdir(parents=True, exist_ok=True)
        manifest = dict(identity=identity, complete=False, shards=[], aliases={})
        _atomic_json(manifest_path, manifest)
        if verbose:
            print(f"Building memory-mapped CPU cache in {directory}", flush=True)
        # Mapping the 4 GB source avoids the eager anonymous copy in the old loader.
        packed_state = torch.load(packed_path, map_location="cpu", mmap=True, weights_only=False)
        rotations = torch.load(rot_path, map_location="cpu", mmap=True, weights_only=False) if rotation_source else {}
        storages = {}
        started = time.perf_counter()
        for index, key in enumerate(expected_shapes):
            if key not in packed_state:
                if key == "lm_head.weight" and cfg.tie_word_embeddings:
                    manifest["aliases"][key] = "model.embed_tokens.weight"
                    continue
                raise ValueError(f"Missing checkpoint parameter: {key}")
            value = packed_state[key]
            if torch.is_tensor(value):
                alias = (value.untyped_storage().data_ptr(), value.storage_offset(), tuple(value.shape), value.stride())
                if alias in storages:
                    manifest["aliases"][key] = storages[alias]
                    continue
                storages[alias] = key
                weight = value.to(dtype)
            else:
                module = key.removesuffix(".weight")
                short = module.removeprefix("model.")
                rotation = rotations.get(short) or rotations.get(module)
                # Preserve support for rotations embedded beside packed weights.
                if rotation is None and module + ".L" in packed_state and module + ".R" in packed_state:
                    rotation = {"L": packed_state[module + ".L"], "R": packed_state[module + ".R"]}
                if rotation is None:
                    raise ValueError(f"Missing KOTMS rotations for {module}")
                weight = reconstruct_weight(value, rotation, dtype)
            if tuple(weight.shape) != expected_shapes[key]:
                raise ValueError(f"Checkpoint shape mismatch for {key}")
            filename = f"weight-{index:04d}.pt"
            temporary = directory / (filename + ".tmp")
            torch.save({key: weight}, temporary)
            os.replace(temporary, directory / filename)
            manifest["shards"].append(filename)
            del weight
            if verbose and (index + 1) % 25 == 0:
                print(f"  Cached {index+1}/{len(expected_shapes)} tensors", flush=True)
        del packed_state, rotations
        manifest["complete"] = True
        _atomic_json(manifest_path, manifest)
        if verbose:
            print(f"CPU cache built in {time.perf_counter()-started:.1f}s", flush=True)
    weights = {}
    for filename in manifest["shards"]:
        # Manifest entries are generated local basenames, never arbitrary paths.
        if Path(filename).name != filename:
            raise ValueError("Invalid shard filename in CPU cache")
        shard = torch.load(directory / filename, map_location="cpu", mmap=True, weights_only=True)
        weights.update(shard)
    for key, target in manifest["aliases"].items():
        weights[key] = weights[target]
    if verbose:
        print("CPU model ready (native Linear, absorbed rotations, mmap weights)", flush=True)
    return assemble(weights)


if __name__ == "__main__":
    build_cpu()
