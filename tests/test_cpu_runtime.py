"""CPU checks are independent of the CUDA scripts in this directory."""
import json
import pytest

torch = pytest.importorskip("torch")
from build_cpu_model import build_cpu, reconstruct_weight


def packed_fixture(rows=5, width=12):
    generator = torch.Generator().manual_seed(17)
    modes = torch.randint(0, 4, (rows, width), generator=generator, dtype=torch.uint8)
    c0 = torch.randint(0, 4, (rows, width), generator=generator, dtype=torch.uint8)
    c1 = torch.randint(0, 4, (rows, width), generator=generator, dtype=torch.uint8)
    def pack(c):
        c = torch.nn.functional.pad(c, (0, (-c.shape[-1]) % 4))
        return c[..., ::4] | c[..., 1::4] << 2 | c[..., 2::4] << 4 | c[..., 3::4] << 6
    selected = c1[modes < 2]
    mu = torch.randn((4, rows, 3), generator=generator) * 0.05
    a0 = torch.randn((4, rows, 3), generator=generator) * 0.05
    a1 = torch.randn((2, rows, 3), generator=generator) * 0.05
    packed = dict(shape=(rows, width), blocksize=4, nblocks=3, maskid=pack(modes),
                  T0=pack(c0), T1c=pack(selected), n_T1=selected.numel(), o2_idx=torch.tensor([0, 1]),
                  mu=mu, a0=a0, a1=a1)
    reference = torch.empty((rows, width))
    for r in range(rows):
        for col in range(width):
            m, b = int(modes[r, col]), col // 4
            t0 = int(c0[r, col] == 1) - int(c0[r, col] == 2)
            t1 = int(c1[r, col] == 1) - int(c1[r, col] == 2)
            reference[r, col] = mu[m, r, b] + a0[m, r, b] * t0 + (a1[m, r, b] * t1 if m < 2 else 0)
    return packed, reference


def test_chunked_cpu_reconstruction_and_absorption():
    packed, reference = packed_fixture()
    actual = reconstruct_weight(packed, dtype=torch.float32, chunk_rows=2)
    torch.testing.assert_close(actual, reference)
    left = torch.linalg.qr(torch.randn(3, 3))[0]
    right = torch.linalg.qr(torch.randn(4, 4))[0]
    absorbed = reconstruct_weight(packed, {"L": left, "R": right}, torch.float32, chunk_rows=2)
    x = torch.randn(7, 12)
    expected = (left @ x.reshape(-1, 3, 4) @ right).reshape(7, 12) @ reference.T
    torch.testing.assert_close(x @ absorbed.T, expected, rtol=1e-5, atol=1e-6)


def test_cpu_shards_roundtrip_and_cache_identity(tmp_path):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    cfg = Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                      num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                      head_dim=8, tie_word_embeddings=True)
    original = Qwen3ForCausalLM(cfg).eval()
    checkpoint, cache = tmp_path / "model.pt", tmp_path / "cache"
    torch.save(original.state_dict(), checkpoint)
    kwargs = dict(packed_path=checkpoint, cache_path=cache, rot_path=None,
                  config=cfg, dtype=torch.float32, verbose=False)
    first = build_cpu(**kwargs)
    second = build_cpu(**kwargs)
    ids = torch.tensor([[1, 2, 3]])
    with torch.inference_mode():
        torch.testing.assert_close(first(ids).logits, original(ids).logits)
        torch.testing.assert_close(second(ids).logits, original(ids).logits)
    assert second.lm_head.weight is second.model.embed_tokens.weight
    manifest = json.loads((cache / "manifest.json").read_text())
    assert manifest["complete"]
    assert manifest["aliases"]["lm_head.weight"] == "model.embed_tokens.weight"
    with pytest.raises(ValueError, match="does not match"):
        build_cpu(**dict(kwargs, dtype=torch.bfloat16))


@pytest.mark.parametrize("compact", [False, True])
def test_native_packed_cpu_matches_dense(compact):
    import platform
    if platform.system() != "Darwin":
        pytest.skip("Native CPU kernel uses macOS libdispatch")
    from packed_cpu import PackedCPULinear
    packed, dense = packed_fixture(rows=9)
    for key in ("mu", "a0", "a1"):
        packed[key] = packed[key].half()
    dense = reconstruct_weight(packed, dtype=torch.float32)
    class Tensor:
        def __init__(self, value): self.value = value.numpy()
        def numpy(self): return self.value
    mapped = {k: Tensor(v) if torch.is_tensor(v) else v for k, v in packed.items()}
    left, right = torch.eye(3), torch.eye(4)
    layer = PackedCPULinear(mapped, Tensor(left), Tensor(right), torch.float32, threads=2, compact=compact)
    for batch in (1, 5):
        x = torch.randn(batch, 12)
        torch.testing.assert_close(layer(x), x @ dense.T, atol=1e-5, rtol=1e-5)


def test_cpu_chunked_prefill_matches_full_context():
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from inference_runtime import TorchRuntime
    cfg = Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                      num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=8)
    model = Qwen3ForCausalLM(cfg).eval()
    model._lattice_packed_cpu = True
    runtime = TorchRuntime.__new__(TorchRuntime)
    runtime.torch, runtime.model, runtime.device, runtime.graphs = torch, model, "cpu", {}
    ids = list(range(19))
    logits, session = runtime.prefill(ids, 4)
    with torch.inference_mode():
        expected = model(torch.tensor([ids]), logits_to_keep=1).logits
    torch.testing.assert_close(logits, expected, rtol=1e-5, atol=1e-6)
    assert session["cache"].get_seq_length() == 19
