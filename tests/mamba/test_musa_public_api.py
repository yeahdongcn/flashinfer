"""Public Mamba API tests executed on a MUSA device."""

import os

import pytest

torch = pytest.importorskip("torch")
if os.environ.get("FLASHINFER_MAMBA_TEST_DEVICE") != "musa":
    pytest.skip("set FLASHINFER_MAMBA_TEST_DEVICE=musa", allow_module_level=True)

from flashinfer.mamba import (  # noqa: E402
    mamba_chunk_scan_combined_varlen,
    selective_state_update,
)


DEVICE = torch.device("musa")
H, D, N, G = 2, 3, 4, 1


def _ssu_inputs(batch=1, steps=None):
    x_shape = (batch, H, D) if steps is None else (batch, steps, H, D)
    b_shape = (batch, G, N) if steps is None else (batch, steps, G, N)
    x = torch.randn(*x_shape, device=DEVICE, dtype=torch.bfloat16)
    dt = torch.randn(*x_shape, device=DEVICE, dtype=torch.float32)
    A = -torch.rand(H, D, N, device=DEVICE, dtype=torch.float32) - 1
    B = torch.randn(*b_shape, device=DEVICE, dtype=torch.bfloat16)
    C = torch.randn_like(B)
    D_skip = torch.randn(H, D, device=DEVICE, dtype=torch.float32)
    return x, dt, A, B, C, D_skip


@pytest.mark.parametrize("algorithm", ["auto", "simple", "vertical", "horizontal"])
def test_selective_state_update_stochastic_rounding(algorithm):
    x, dt, A, B, C, D_skip = _ssu_inputs()
    state = torch.zeros(8, H, D, N, device=DEVICE, dtype=torch.float16)
    slot = torch.tensor([3], device=DEVICE, dtype=torch.int32)
    seed = torch.tensor([123], device=DEVICE, dtype=torch.int64)

    y = selective_state_update(
        state,
        x,
        dt,
        A,
        B,
        C,
        D=D_skip,
        state_batch_indices=slot,
        rand_seed=seed,
        philox_rounds=5,
        algorithm=algorithm,
    )

    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    assert torch.any(state[slot] != 0)


def test_selective_state_update_mtp_replay_with_intermediate_states():
    steps = 3
    x, dt, A, B, C, D_skip = _ssu_inputs(steps=steps)
    state = torch.zeros(8, H, D, N, device=DEVICE, dtype=torch.float32)
    read = torch.tensor([[2, 3, 4]], device=DEVICE, dtype=torch.int32)
    intermediate = torch.empty(1, steps, H, D, N, device=DEVICE, dtype=torch.float32)

    y = selective_state_update(
        state,
        x,
        dt,
        A,
        B,
        C,
        D=D_skip,
        state_batch_indices=read,
        intermediate_states_buffer=intermediate,
        num_accepted_tokens=torch.tensor([1], device=DEVICE, dtype=torch.int32),
        cache_steps=steps,
    )

    assert y.shape == x.shape
    assert intermediate.shape == (1, steps, H, D, N)
    assert torch.isfinite(y).all()
    assert torch.isfinite(intermediate).all()


def test_mamba_chunk_scan_combined_varlen_public_api():
    tokens = 5
    x = torch.randn(tokens, H, D, device=DEVICE, dtype=torch.bfloat16)
    dt = torch.randn(tokens, H, device=DEVICE, dtype=torch.float32)
    A = -torch.rand(H, device=DEVICE, dtype=torch.float32) - 1
    B = torch.randn(tokens, G, N, device=DEVICE, dtype=torch.bfloat16)
    C = torch.randn_like(B)
    initial = torch.randn(2, H, D, N, device=DEVICE, dtype=torch.float16)
    cu_seqlens = torch.tensor([0, 3, 5], device=DEVICE, dtype=torch.int32)
    cu_chunk_seqlens = torch.tensor([0, 2, 3, 5], device=DEVICE, dtype=torch.int32)
    last = torch.tensor([1, 2], device=DEVICE, dtype=torch.int32)
    seq_idx = torch.tensor([0, 0, 1], device=DEVICE, dtype=torch.int32)

    states = mamba_chunk_scan_combined_varlen(
        x,
        dt,
        A,
        B,
        C,
        3,
        cu_seqlens,
        cu_chunk_seqlens,
        last,
        seq_idx,
        torch.empty_like(x),
        initial_states=initial,
        return_intermediate_states=True,
        state_dtype=torch.float16,
    )

    assert states.shape == (3, H, D, N)
    assert torch.isfinite(states).all()
