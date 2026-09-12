"""Low-memory Mac CPU execution using the existing packed LATTICE weights."""
from __future__ import annotations

import ctypes
import hashlib
import os
from pathlib import Path
import platform
import shutil
import subprocess
import warnings
from functools import lru_cache

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent


@lru_cache(None)
def native_kernel():
    if platform.system() != "Darwin":
        raise RuntimeError("Packed CPU execution currently requires macOS; use --cpu-layout absorbed.")
    source = ROOT / "cpu_kernels/lattice.cpp"
    compiler = shutil.which("clang++")
    if compiler is None:
        raise RuntimeError("Packed CPU needs Apple command-line tools (clang++); use --cpu-layout absorbed if unavailable.")
    digest = hashlib.sha256(source.read_bytes() + platform.machine().encode()).hexdigest()[:16]
    output = ROOT / ".cache" / "cpu-kernels" / f"lattice-{digest}.dylib"
    if not output.exists():
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(f".{os.getpid()}.tmp.dylib")
        subprocess.run([compiler, "-O3", "-ffast-math", "-std=c++17", "-dynamiclib",
                        str(source), "-o", str(temporary)], check=True)
        os.replace(temporary, output)
    library = ctypes.CDLL(str(output))
    function = library.lattice_cpu
    function.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 5 + [ctypes.c_void_p]
    function.restype = None
    return function


def _mapped_array(directory, name, make):
    if directory is None:
        return make()
    path = directory / (name + ".npy")
    if not path.exists():
        temporary = directory / (name + f".{os.getpid()}.tmp.npy")
        np.save(temporary, make(), allow_pickle=False)
        os.replace(temporary, path)
    return np.load(path, mmap_mode="r", allow_pickle=False)


def _readonly_tensor(array):
    # Native kernels only read these mapped weight buffers.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
        return torch.from_numpy(array)


class PackedCPULinear(nn.Module):
    def __init__(self, packed, left, right, dtype=torch.bfloat16, threads=4, cache_dir=None, compact=True):
        super().__init__()
        from MLX.lattice_model import expand_t1
        self.rows, self.width = map(int, packed["shape"])
        self.bs, self.threads = int(packed["blocksize"]), threads
        if min(self.rows, self.width, self.bs, self.threads) < 1:
            raise ValueError("Packed CPU dimensions and thread count must be positive")
        for name, source in (("mid", "maskid"), ("t0", "T0")):
            # These buffers are read-only kernel inputs. Keep checkpoint pages
            # mapped rather than duplicating ~1.7 GiB into anonymous RAM.
            tensor = _readonly_tensor(packed[source].numpy())
            if tensor.shape != (self.rows, (self.width+3)//4) or tensor.dtype != torch.uint8 or not tensor.is_contiguous():
                raise ValueError("Invalid packed CPU code buffer")
            self.register_buffer(name, tensor)
        if compact:
            self.register_buffer("t1", _readonly_tensor(packed["T1c"].numpy()))
            if self.t1.ndim != 1 or self.t1.numel() < (int(packed["n_T1"])+3)//4 or not self.t1.is_contiguous():
                raise ValueError("Invalid compact T1 buffer")
            modes = packed["maskid"].numpy()
            # Count only real columns (also supports non-aligned test fixtures).
            from MLX.lattice_model import unpack_codes
            counts = np.empty(self.rows, dtype=np.uint32)
            for start in range(0, self.rows, 64):
                counts[start:start+64] = (unpack_codes(modes[start:start+64], self.width) < 2).sum(axis=1)
            starts = np.zeros(self.rows, dtype=np.uint32)
            starts[1:] = np.cumsum(counts[:-1], dtype=np.uint32)
            if int(counts.sum()) != int(packed["n_T1"]):
                raise ValueError("Compact T1 count does not match masks")
            self.register_buffer("row_starts", torch.from_numpy(starts))
        else:
            self.register_buffer("t1", _readonly_tensor(_mapped_array(cache_dir, "t1", lambda: expand_t1(packed))))
            self.row_starts = None
        # Preserve all deployed FP16 scales; do not introduce another quantizer.
        def interleaved():
            scales = np.concatenate([packed[n].numpy() for n in ("mu", "a0", "a1")], axis=0)
            return np.ascontiguousarray(scales.transpose(1, 2, 0))
        self.register_buffer("scales", _readonly_tensor(_mapped_array(cache_dir, "scales", interleaved)))
        if self.scales.dtype != torch.float16:
            raise ValueError("Packed CPU kernels require deployed FP16 scales")
        if ((not compact and self.t1.shape != (self.rows, (self.width + 3) // 4))
                or self.t1.dtype != torch.uint8
                or self.scales.shape != (self.rows, (self.width+self.bs-1)//self.bs, 10)):
            raise ValueError("Packed CPU cache dimensions do not match the checkpoint")
        self.register_buffer("left", torch.tensor(left.numpy()).to(dtype))
        self.register_buffer("right", torch.tensor(right.numpy()).to(dtype))
        self.kernel = native_kernel()

    def forward(self, x):
        shape, dtype = x.shape, x.dtype
        rotated = (self.left @ x.reshape(-1, self.left.shape[0], self.right.shape[0]) @ self.right).reshape(-1, self.width)
        vector = rotated.shape[0] == 1
        inp = rotated.float().contiguous()
        if vector:
            output = torch.empty((self.rows,), dtype=torch.float32)
            self.kernel(inp.data_ptr(), self.mid.data_ptr(), self.t0.data_ptr(), self.t1.data_ptr(),
                        self.scales.data_ptr(), output.data_ptr(), self.rows, self.width, self.bs,
                        self.threads, 0, self.row_starts.data_ptr() if self.row_starts is not None else None)
            result = output.to(dtype)
        else:
            # Bound CPU prefill scratch to <=256 output rows. A full float32
            # MLP matrix plus its cast easily triggers pressure on a 16 GB Mac.
            result = torch.empty((rotated.shape[0], self.rows), dtype=dtype)
            stride = (self.width + 3) // 4
            scale_stride = ((self.width + self.bs - 1) // self.bs) * 10 * 2
            for start in range(0, self.rows, 256):
                count = min(256, self.rows - start)
                output = torch.empty((count, self.width), dtype=torch.float32)
                compact = self.row_starts is not None
                self.kernel(inp.data_ptr(), self.mid.data_ptr()+start*stride,
                            self.t0.data_ptr()+start*stride, self.t1.data_ptr()+(0 if compact else start*stride),
                            self.scales.data_ptr()+start*scale_stride, output.data_ptr(), count,
                            self.width, self.bs, self.threads, 1,
                            self.row_starts.data_ptr()+start*4 if compact else None)
                result[:, start:start+count] = F.linear(rotated, output.to(dtype))
        return result.reshape(*shape[:-1], self.rows)


def build_packed_cpu(packed_path, dtype=torch.bfloat16, threads=4, verbose=True):
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM
    from MLX.lattice_model import load_checkpoint
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    native_kernel()
    config = AutoConfig.from_pretrained(str(ROOT / "configs/qwen3-4b.json"))
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, dtype=dtype)
    state, archive = load_checkpoint(packed_path)
    source = Path(packed_path).resolve()
    stat = source.stat()
    identity = hashlib.sha256(f"v1:{source}:{stat.st_size}:{stat.st_mtime_ns}".encode()).hexdigest()[:16]
    cache_root = ROOT / ".cache" / "packed-cpu" / identity
    cache_root.mkdir(parents=True, exist_ok=True)
    try:
        # Assign small ordinary parameters and the tied embedding once.
        plain = {}
        for key in model.state_dict():
            value = state[key]
            if isinstance(value, dict) or key == "lm_head.weight":
                continue
            host = value.numpy()
            cache_file = cache_root / (key + "." + str(dtype) + ".pt")
            if host.ndim == 2 and cache_file.exists():
                plain[key] = torch.load(cache_file, mmap=True, weights_only=True)[key]
                continue
            tensor = torch.empty(host.shape, dtype=dtype)
            # The source embedding is FP32. Cast it in slices to avoid a
            # second whole-embedding allocation during startup.
            for start in range(0, host.shape[0], 1024):
                tensor[start:start+1024] = torch.tensor(np.array(host[start:start+1024], copy=True)).to(dtype)
            if host.ndim == 2:
                temporary = cache_file.with_suffix(f".{os.getpid()}.tmp")
                torch.save({key: tensor}, temporary)
                os.replace(temporary, cache_file)
                plain[key] = torch.load(cache_file, mmap=True, weights_only=True)[key]
                del tensor
            else:
                plain[key] = tensor
        model.load_state_dict(plain, strict=False, assign=True)
        del plain
        count = 0
        for name, module in list(model.named_modules()):
            if not isinstance(module, nn.Linear) or name == "lm_head":
                continue
            directory = cache_root / name
            directory.mkdir(exist_ok=True)
            projection = PackedCPULinear(state[name + ".weight"], state[name + ".L"], state[name + ".R"], dtype, threads, directory)
            parent_name, child = name.rsplit(".", 1)
            setattr(model.get_submodule(parent_name), child, projection)
            count += 1
            if verbose and count % 63 == 0:
                print(f"  Loaded {count}/252 packed CPU projections", flush=True)
        model.tie_weights()
        if any(p.is_meta for p in model.parameters()):
            raise ValueError("Packed CPU model has uninitialized parameters")
        model.eval()
        model._lattice_packed_cpu = True
    finally:
        archive.close()
    if verbose:
        print("CPU model ready (packed LATTICE, native CPU kernels; no GPU)", flush=True)
    return model
