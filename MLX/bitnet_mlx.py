"""Faithful MLX port of 1bitLLM/bitnet_b1_58-large (BitnetForCausalLM).

Mirrors MLX/models/bitnet-700m/modeling_bitnet.py and utils_quant.py:
T5-style RMSNorm, RoPE, per-token 8-bit absmax activation quantization inside
every BitLinear, absmean ternary weights, inner_attn_ln / ffn_layernorm, tied
embeddings. Architecture is unchanged; only the linear weight arithmetic is
served by the packed Metal GEMV kernel during single-token decode.
"""
import math

import mlx.core as mx
import numpy as np

from .kernels import TernaryLinear, pack_ternary


def activation_quant(x, num_bits=8):
    """utils_quant.activation_quant: per-token absmax symmetric quantization."""
    x32 = x.astype(mx.float32)
    qp = 2 ** (num_bits - 1) - 1
    qn = -(2 ** (num_bits - 1))
    s = qp / mx.maximum(mx.abs(x32).max(axis=-1, keepdims=True), 1e-5)
    return (mx.clip(mx.round(x32 * s), qn, qp) / s).astype(x.dtype)


def ternarize_weight(weight_f32):
    """utils_quant.weight_quant at load time. Returns (signs int8, scale float)."""
    w = np.asarray(weight_f32, dtype=np.float32)
    s = 1 / max(float(np.abs(w).mean()), 1e-5)
    signs = np.clip(np.rint(w * s), -1, 1).astype(np.int8)
    return signs, float(np.float16(np.float32(1) / s))


class PackedBitLinear:
    """BitLinear with deployed ternary codes. Decode uses the packed Metal GEMV;
    prefill unpacks weights transiently and never keeps a dense resident copy.
    dense_decode=True keeps a resident dense copy and uses dense GEMV instead,
    as the baseline arm for end-to-end comparison."""

    def __init__(self, weight_f32, dense_decode=False):
        signs, scale = ternarize_weight(weight_f32)
        rows, width = signs.shape
        self.linear = TernaryLinear(mx.array(pack_ternary(signs)), mx.array([scale]), width)
        mx.eval(self.linear.packed, self.linear.scale)
        self.width, self.rows = width, rows
        self.dense = self.linear.dense_weight(mx.float16) if dense_decode else None
        if self.dense is not None:
            mx.eval(self.dense)

    def __call__(self, x):
        xq = activation_quant(x)
        if xq.size == self.width:
            if self.dense is not None:
                return xq @ self.dense.T
            return self.linear(xq)
        dense = self.dense if self.dense is not None else self.linear.dense_weight(mx.float16)
        return xq @ dense.T


def rms_norm(x, weight, eps):
    x32 = x.astype(mx.float32)
    var = mx.mean(x32 * x32, axis=-1, keepdims=True)
    out = x32 * mx.rsqrt(var + eps)
    return (weight * out.astype(x.dtype)).astype(x.dtype)


def rope_cos_sin(positions, head_dim, base=10000.0):
    inv = 1.0 / (base ** (mx.arange(0, head_dim, 2, dtype=mx.float32) / head_dim))
    freqs = positions[:, None].astype(mx.float32) * inv[None, :]
    emb = mx.concatenate([freqs, freqs], axis=-1)
    return mx.cos(emb), mx.sin(emb)


def apply_rope(q, k, cos, sin):
    half = q.shape[-1] // 2

    def rot(t):
        t1, t2 = t[..., :half], t[..., half:]
        return mx.concatenate([-t2, t1], axis=-1)

    return q * cos + rot(q) * sin, k * cos + rot(k) * sin


class Attention:
    def __init__(self, cfg, params, prefix, dense_decode=False):
        self.num_heads = cfg["num_attention_heads"]
        self.num_kv_heads = cfg["num_key_value_heads"]
        self.head_dim = cfg["hidden_size"] // self.num_heads
        self.eps = cfg["rms_norm_eps"]
        self.q_proj = PackedBitLinear(params[f"{prefix}.q_proj.weight"], dense_decode)
        self.k_proj = PackedBitLinear(params[f"{prefix}.k_proj.weight"], dense_decode)
        self.v_proj = PackedBitLinear(params[f"{prefix}.v_proj.weight"], dense_decode)
        self.o_proj = PackedBitLinear(params[f"{prefix}.o_proj.weight"], dense_decode)
        self.ln_w = mx.array(params[f"{prefix}.inner_attn_ln.weight"])
        self.cos_cache = None

    def __call__(self, x, cache, start_pos):
        bsz, q_len, _ = x.shape
        q = self.q_proj(x).reshape(bsz, q_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        positions = mx.arange(start_pos, start_pos + q_len)
        cos, sin = rope_cos_sin(positions, self.head_dim)
        cos = cos[None, None].astype(x.dtype)
        sin = sin[None, None].astype(x.dtype)
        q, k = apply_rope(q, k, cos, sin)
        if cache.get(self) is None:
            cache.set(self, k, v)
        else:
            pk, pv = cache.get(self)
            cache.set(self, mx.concatenate([pk, k], axis=2), mx.concatenate([pv, v], axis=2))
        kk, vv = cache.get(self)
        scores = (q @ kk.transpose(0, 1, 3, 2)) / math.sqrt(self.head_dim)
        if q_len > 1:
            total = kk.shape[2]
            mask = mx.triu(mx.full((q_len, total), -mx.inf), k=1 + total - q_len)
            scores = scores + mask
        weights = mx.softmax(scores.astype(mx.float32), axis=-1).astype(x.dtype)
        out = (weights @ vv).transpose(0, 2, 1, 3).reshape(bsz, q_len, -1)
        out = rms_norm(out, self.ln_w, self.eps)
        return self.o_proj(out)


class MLP:
    def __init__(self, cfg, params, prefix, dense_decode=False):
        self.gate_proj = PackedBitLinear(params[f"{prefix}.gate_proj.weight"], dense_decode)
        self.up_proj = PackedBitLinear(params[f"{prefix}.up_proj.weight"], dense_decode)
        self.down_proj = PackedBitLinear(params[f"{prefix}.down_proj.weight"], dense_decode)
        self.ln_w = mx.array(params[f"{prefix}.ffn_layernorm.weight"])
        self.eps = cfg["rms_norm_eps"]

    def __call__(self, x):
        x = (self.gate_proj(x) * mx.sigmoid(self.gate_proj(x))) * self.up_proj(x)
        x = rms_norm(x, self.ln_w, self.eps)
        return self.down_proj(x)


class DecoderLayer:
    def __init__(self, cfg, params, idx, dense_decode=False):
        prefix = f"model.layers.{idx}"
        self.attn = Attention(cfg, params, f"{prefix}.self_attn", dense_decode)
        self.mlp = MLP(cfg, params, f"{prefix}.mlp", dense_decode)
        self.input_ln = mx.array(params[f"{prefix}.input_layernorm.weight"])
        self.post_ln = mx.array(params[f"{prefix}.post_attention_layernorm.weight"])
        self.eps = cfg["rms_norm_eps"]

    def __call__(self, x, cache, start_pos):
        r = x
        x = rms_norm(x, self.input_ln, self.eps)
        x = r + self.attn(x, cache, start_pos)
        r = x
        x = rms_norm(x, self.post_ln, self.eps)
        return r + self.mlp(x)


class KVCache:
    def __init__(self):
        self.store = {}

    def get(self, layer):
        return self.store.get(id(layer))

    def set(self, layer, k, v):
        self.store[id(layer)] = (k, v)

    def length(self):
        entry = next(iter(self.store.values()), None)
        return 0 if entry is None else entry[0].shape[2]


class BitnetForCausalLM:
    def __init__(self, cfg, params, dense_decode=False):
        self.cfg = cfg
        self.embed = mx.array(params["model.embed_tokens.weight"]).astype(mx.float16)
        self.layers = [DecoderLayer(cfg, params, i, dense_decode) for i in range(cfg["num_hidden_layers"])]
        self.final_ln = mx.array(params["model.norm.weight"])
        self.eps = cfg["rms_norm_eps"]

    def __call__(self, tokens, cache=None, start_pos=0):
        if cache is None:
            cache = KVCache()
        prefill = tokens.ndim == 2 and tokens.shape[1] > 1
        x = self.embed[tokens]
        for layer in self.layers:
            x = layer(x, cache, start_pos)
            if prefill:
                # Bound peak memory: transient prefill dense weights are freed
                # as soon as each layer output is materialized.
                mx.eval(x)
        x = rms_norm(x, self.final_ln, self.eps)
        return x @ self.embed.T

    def generate(self, tokens, max_new_tokens, temperature=0.0, eos_id=2):
        import time
        cache = KVCache()
        out = []
        pos = len(tokens)
        t0 = time.perf_counter()
        logits = self(tokens[None], cache, 0)
        mx.eval(logits)
        prefill_time = time.perf_counter() - t0
        token = int(mx.argmax(logits[0, -1]))
        decode_time = 0.0
        while len(out) < max_new_tokens:
            if token == eos_id:
                break
            out.append(token)
            step_start = time.perf_counter()
            logits = self(mx.array([token])[None], cache, pos)
            pos += 1
            if temperature > 0:
                probs = mx.softmax(logits[0, -1].astype(mx.float32) / temperature)
                token = int(mx.random.categorical(mx.log(probs)))
            else:
                token = int(mx.argmax(logits[0, -1]))
            decode_time += time.perf_counter() - step_start
        return out, prefill_time, decode_time
