"""Runtime gate for the opt-in native S5000 Simple-STP extension."""

import os

import pytest
import torch

from flashinfer.mamba.musa_ssu_native import musa_ssu_one_token_native
from flashinfer.mamba.musa_ssu_triton import ssu_one_token_musa_triton
from flashinfer.mamba.selective_state_update import selective_state_update

pytestmark = [
    pytest.mark.skipif(
        os.environ.get("FLASHINFER_MAMBA_TEST_DEVICE") != "musa",
        reason="requires a MUSA runtime",
    ),
    pytest.mark.skipif(
        os.environ.get("FLASHINFER_MUSA_SIMPLE_STP_NATIVE") != "1",
        reason="native extension is opt-in",
    ),
]


def test_native_simple_stp_matches_triton():
    torch.manual_seed(23)
    state = torch.randn((2, 64, 64, 128), device="musa", dtype=torch.float16)
    x = torch.randn((1, 64, 64), device="musa", dtype=torch.bfloat16)
    dt = torch.randn((1, 64, 1), device="musa", dtype=torch.float32).expand(1, 64, 64)
    a = -torch.rand((64, 1, 1), device="musa", dtype=torch.float32).expand(64, 64, 128)
    b = torch.randn((1, 8, 128), device="musa", dtype=torch.bfloat16)
    c = torch.randn_like(b)
    d = torch.randn((64,), device="musa", dtype=torch.bfloat16)
    slot = torch.zeros((1,), device="musa", dtype=torch.int32)
    native_state, triton_state = state.clone(), state.clone()
    native_out, triton_out = torch.empty_like(x), torch.empty_like(x)
    args = (native_state, x, dt, a, b, c, d, slot, slot, None, None, True, -1, native_out, None, 0)
    musa_ssu_one_token_native(*args)
    ssu_one_token_musa_triton(
        triton_state, x, dt, a, b, c, d, slot,
        dt_softplus=True, out=triton_out,
    )
    torch.testing.assert_close(native_out, triton_out, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(native_state, triton_state, atol=3e-2, rtol=3e-2)


def test_native_simple_stp_stochastic_matches_triton():
    torch.manual_seed(29)
    state = torch.randn((2, 64, 64, 128), device="musa", dtype=torch.float16)
    x = torch.randn((1, 64, 64), device="musa", dtype=torch.bfloat16)
    dt = torch.randn((1, 64, 1), device="musa", dtype=torch.float32).expand(1, 64, 64)
    a = -torch.rand((64, 1, 1), device="musa", dtype=torch.float32).expand(64, 64, 128)
    b = torch.randn((1, 8, 128), device="musa", dtype=torch.bfloat16)
    c = torch.randn_like(b)
    d = torch.randn((64,), device="musa", dtype=torch.bfloat16)
    slot = torch.zeros((1,), device="musa", dtype=torch.int32)
    seed = torch.tensor([9123], device="musa", dtype=torch.int64)
    native_state, triton_state = state.clone(), state.clone()
    native_out, triton_out = torch.empty_like(x), torch.empty_like(x)
    musa_ssu_one_token_native(
        native_state, x, dt, a, b, c, d, slot, slot, None, None, True, -1,
        native_out, seed, 5,
    )
    ssu_one_token_musa_triton(
        triton_state, x, dt, a, b, c, d, slot, dt_softplus=True,
        out=triton_out, rand_seed=seed, philox_rounds=5,
    )
    torch.testing.assert_close(native_out, triton_out, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(native_state, triton_state, atol=3e-2, rtol=3e-2)


def test_public_selective_state_update_routes_native():
    torch.manual_seed(31)
    state = torch.randn((2, 64, 64, 128), device="musa", dtype=torch.float16)
    x = torch.randn((1, 64, 64), device="musa", dtype=torch.bfloat16)
    dt = torch.randn((1, 64, 1), device="musa", dtype=torch.float32).expand(1, 64, 64)
    a = -torch.rand((64, 1, 1), device="musa", dtype=torch.float32).expand(64, 64, 128)
    b = torch.randn((1, 8, 128), device="musa", dtype=torch.bfloat16)
    c = torch.randn_like(b)
    d = torch.randn((64, 1), device="musa", dtype=torch.float32).expand(64, 64)
    slot = torch.zeros((1,), device="musa", dtype=torch.int32)
    result = selective_state_update(
        state,
        x,
        dt,
        a,
        b,
        c,
        d,
        dt_softplus=True,
        state_batch_indices=slot,
        backend="flashinfer",
    )
    assert result.shape == x.shape
    assert result.dtype == x.dtype
