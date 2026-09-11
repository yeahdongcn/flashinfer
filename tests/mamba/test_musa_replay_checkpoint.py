"""MUSA coverage for FlashInfer's cache-replay public APIs."""

import os

import pytest

torch = pytest.importorskip("torch")
if os.environ.get("FLASHINFER_MAMBA_TEST_DEVICE") != "musa":
    pytest.skip("set FLASHINFER_MAMBA_TEST_DEVICE=musa", allow_module_level=True)

from flashinfer.mamba import checkpointing_ssu, replayssm_materialize  # noqa: E402


def test_checkpointing_ssu_musa_ring_cache():
    d = torch.device("musa")
    batch, steps, heads, dim, dstate, groups, ring = 1, 2, 2, 3, 4, 1, 8
    state = torch.zeros(4, heads, dim, dstate, device=d, dtype=torch.float32)
    x_cache = torch.zeros(4, heads, ring, dim, device=d, dtype=torch.bfloat16)
    b_cache = torch.zeros(4, groups, ring, dstate, device=d, dtype=torch.bfloat16)
    dt_cache = torch.zeros(4, heads, ring, device=d, dtype=torch.float32)
    x = torch.randn(batch, steps, heads, dim, device=d, dtype=torch.bfloat16)
    dt = torch.randn(batch, steps, heads, device=d, dtype=torch.float32)
    a = -torch.rand(heads, dim, dstate, device=d, dtype=torch.float32) - 1
    b = torch.randn(batch, steps, groups, dstate, device=d, dtype=torch.bfloat16)
    c = torch.randn_like(b)
    out = torch.empty_like(x)
    result = checkpointing_ssu(
        state,
        x_cache,
        b_cache,
        dt_cache,
        torch.zeros(batch, device=d, dtype=torch.int32),
        torch.zeros(batch, device=d, dtype=torch.int32),
        x,
        dt,
        a,
        b,
        c,
        out,
        state_batch_indices=torch.zeros(batch, device=d, dtype=torch.int32),
    )
    assert result.shape == x.shape
    assert torch.isfinite(result).all()
    assert torch.isfinite(state).all()


def test_replayssm_materialize_musa_dependency_path():
    d = torch.device("musa")
    layers, slots, heads, dim, dstate, groups, ring = 1, 4, 2, 3, 4, 1, 8
    state = torch.zeros(slots, heads, dim, dstate, device=d, dtype=torch.float32)
    x_cache = torch.randn(slots, heads, ring, dim, device=d, dtype=torch.bfloat16)
    b_cache = torch.randn(slots, groups, ring, dstate, device=d, dtype=torch.bfloat16)
    dt_cache = torch.randn(slots, heads, ring, device=d, dtype=torch.float32)
    a = -torch.rand(heads, dim, dstate, device=d, dtype=torch.float32) - 1
    replayssm_materialize(
        torch.empty(layers, device=d, dtype=torch.int64),
        torch.empty(layers, device=d, dtype=torch.int64),
        torch.empty(layers, device=d, dtype=torch.int64),
        torch.empty(layers, device=d, dtype=torch.int64),
        torch.empty(layers, device=d, dtype=torch.int64),
        torch.empty(layers, device=d, dtype=torch.int64),
        torch.empty(layers, device=d, dtype=torch.int64),
        torch.empty(layers, device=d, dtype=torch.int64),
        torch.empty(layers, device=d, dtype=torch.int64),
        torch.empty(layers, device=d, dtype=torch.int64),
        torch.empty(layers, device=d, dtype=torch.int64),
        torch.tensor([[0]], device=d, dtype=torch.int32),
        torch.tensor([[1]], device=d, dtype=torch.int32),
        torch.zeros(1, device=d, dtype=torch.int32),
        torch.tensor([1], device=d, dtype=torch.int32),
        dependency_inputs=[x_cache, b_cache, dt_cache, a],
        dependency_outputs=[state],
        heads_per_group=2,
        max_window=ring,
        ring_buffer_len=ring,
        pad_slot_id=-1,
        rand_seed=None,
        philox_rounds=10,
        state_scale_ptrs=None,
        state_dtype=torch.float32,
    )
    assert torch.isfinite(state).all()
