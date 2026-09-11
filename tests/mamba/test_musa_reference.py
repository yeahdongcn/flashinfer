"""Correctness tests for the temporary FlashInfer-MUSA SSU provider."""

import pytest

torch = pytest.importorskip("torch")

from flashinfer.mamba.musa_reference import (  # noqa: E402
    selective_state_update_musa_reference,
)


def _inputs(*, batch: int = 2, steps: int | None = None):
    torch.manual_seed(0)
    heads, dim, dstate, groups = 2, 3, 4, 1
    state = torch.randn(8, heads, dim, dstate, dtype=torch.float32)
    x_shape = (batch, heads, dim) if steps is None else (batch, steps, heads, dim)
    b_shape = (batch, groups, dstate) if steps is None else (batch, steps, groups, dstate)
    x = torch.randn(*x_shape, dtype=torch.bfloat16)
    dt = torch.randn(*x_shape, dtype=torch.float32)
    A = -torch.rand(heads, dim, dstate, dtype=torch.float32) - 1
    B = torch.randn(*b_shape, dtype=torch.bfloat16)
    C = torch.randn(*b_shape, dtype=torch.bfloat16)
    D = torch.randn(heads, dim, dtype=torch.bfloat16)
    bias = torch.randn(heads, dim, dtype=torch.float32)
    return state, x, dt, A, B, C, D, bias


def test_musa_reference_single_token_updates_selected_slot():
    state, x, dt, A, B, C, D, bias = _inputs()
    original = state.clone()
    out = torch.empty_like(x)
    selected = torch.tensor([3, 5], dtype=torch.int32)

    result = selective_state_update_musa_reference(
        state,
        x,
        dt,
        A,
        B,
        C,
        D,
        None,
        bias,
        True,
        selected,
        None,
        -1,
        out,
        False,
        None,
        None,
        None,
        None,
        None,
        0,
        None,
        None,
    )

    assert result.shape == x.shape
    assert torch.isfinite(result).all()
    assert not torch.equal(state[selected], original[selected])
    untouched = torch.tensor([0, 1, 2, 4], dtype=torch.int64)
    assert torch.equal(state[untouched], original[untouched])


def test_musa_reference_mtp_writes_destination_slots_and_intermediates():
    state, x, dt, A, B, C, D, bias = _inputs(batch=1, steps=3)
    out = torch.empty_like(x)
    read = torch.tensor([2], dtype=torch.int32)
    destinations = torch.tensor([[4, 5, 6]], dtype=torch.int32)
    intermediate = torch.empty(1, 3, *state.shape[1:], dtype=state.dtype)

    result = selective_state_update_musa_reference(
        state,
        x,
        dt,
        A,
        B,
        C,
        D,
        None,
        bias,
        True,
        read,
        destinations,
        -1,
        out,
        False,
        intermediate,
        None,
        None,
        None,
        None,
        3,
        None,
        None,
    )

    assert result.shape == x.shape
    assert torch.isfinite(result).all()
    assert torch.isfinite(intermediate).all()
    assert not torch.equal(state[4], state[5])
    assert not torch.equal(state[5], state[6])
