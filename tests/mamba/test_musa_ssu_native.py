"""Runtime gate for the opt-in native S5000 Simple-STP extension."""

import os

import pytest
import torch

from flashinfer.mamba.musa_ssu_native import musa_ssu_one_token_native
from flashinfer.mamba.musa_ssu_triton import ssu_one_token_musa_triton

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
