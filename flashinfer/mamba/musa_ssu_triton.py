"""Fused MUSA Triton selective state update kernel.

This kernel intentionally covers the hot decode contract first: one token per
request, tied-dimension dt, floating-point state, and no bias/softplus/z gate.
Other contracts continue to use the correctness reference provider.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _ssu_one_token_kernel(
    state_ptr,
    x_ptr,
    dt_ptr,
    a_ptr,
    b_ptr,
    c_ptr,
    d_ptr,
    slot_ptr,
    out_ptr,
    H: tl.constexpr,
    D: tl.constexpr,
    N: tl.constexpr,
    G: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    hd = H * D
    batch = pid // hd
    rem = pid % hd
    head = rem // D
    dim = rem % D
    group = head * G // H
    slot = tl.load(slot_ptr + batch).to(tl.int64)
    n = tl.arange(0, BLOCK_N)
    mask = n < N
    state_offset = ((slot * H + head) * D + dim) * N
    a_offset = (head * D + dim) * N
    bc_offset = (batch * G + group) * N
    s = tl.load(state_ptr + state_offset + n, mask=mask, other=0.0).to(tl.float32)
    a = tl.load(a_ptr + a_offset + n, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + bc_offset + n, mask=mask, other=0.0).to(tl.float32)
    c = tl.load(c_ptr + bc_offset + n, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(x_ptr + batch * H * D + head * D + dim).to(tl.float32)
    dt = tl.load(dt_ptr + batch * H * D + head * D + dim).to(tl.float32)
    updated = s * tl.exp(a * dt) + (dt * x) * b
    tl.store(state_ptr + state_offset + n, updated, mask=mask)
    y = tl.sum(c * updated, axis=0)
    d = tl.load(d_ptr + head * D + dim).to(tl.float32)
    tl.store(out_ptr + batch * H * D + head * D + dim, y + d * x)


def ssu_one_token_musa_triton(
    state: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    state_batch_indices: torch.Tensor,
) -> torch.Tensor:
    """Run the fused MUSA decode kernel for the supported contract."""
    batch, heads, dim = x.shape
    groups = B.shape[1]
    dstate = A.shape[-1]
    block_n = triton.next_power_of_2(dstate)
    out = torch.empty_like(x)
    _ssu_one_token_kernel[(batch * heads * dim,)](
        state,
        x,
        dt,
        A,
        B,
        C,
        D,
        state_batch_indices,
        out,
        H=heads,
        D=dim,
        N=dstate,
        G=groups,
        BLOCK_N=block_n,
    )
    return out


__all__ = ["ssu_one_token_musa_triton"]
