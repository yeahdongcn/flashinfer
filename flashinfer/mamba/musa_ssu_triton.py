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
    dt_bias_ptr,
    z_ptr,
    slot_ptr,
    out_ptr,
    H: tl.constexpr,
    D: tl.constexpr,
    N: tl.constexpr,
    G: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BIAS_IS_MATRIX: tl.constexpr,
    HAS_Z: tl.constexpr,
    SOFTPLUS: tl.constexpr,
    D_IS_VECTOR: tl.constexpr,
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
    if HAS_BIAS:
        bias_offset = head * D + dim if BIAS_IS_MATRIX else head
        dt += tl.load(dt_bias_ptr + bias_offset).to(tl.float32)
    if SOFTPLUS:
        dt = tl.log(1.0 + tl.exp(dt))
    updated = s * tl.exp(a * dt) + (dt * x) * b
    tl.store(state_ptr + state_offset + n, updated, mask=mask)
    y = tl.sum(c * updated, axis=0)
    d_offset = head * D + dim if not D_IS_VECTOR else head
    d = tl.load(d_ptr + d_offset).to(tl.float32)
    y += d * x
    if HAS_Z:
        z = tl.load(z_ptr + batch * H * D + head * D + dim).to(tl.float32)
        y *= z / (1.0 + tl.exp(-z))
    tl.store(out_ptr + batch * H * D + head * D + dim, y)


def ssu_one_token_musa_triton(
    state: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    state_batch_indices: torch.Tensor,
    dt_bias: torch.Tensor | None = None,
    z: torch.Tensor | None = None,
    dt_softplus: bool = False,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the fused MUSA decode kernel for the supported contract."""
    batch, heads, dim = x.shape
    groups = B.shape[1]
    dstate = A.shape[-1]
    block_n = triton.next_power_of_2(dstate)
    if out is None:
        out = torch.empty_like(x)
    elif out.shape != x.shape or out.dtype != x.dtype:
        raise ValueError("MUSA fused SSU out must match x shape and dtype")
    _ssu_one_token_kernel[(batch * heads * dim,)](
        state,
        x,
        dt,
        A,
        B,
        C,
        D,
        dt_bias if dt_bias is not None else x,
        z if z is not None else x,
        state_batch_indices,
        out,
        H=heads,
        D=dim,
        N=dstate,
        G=groups,
        HAS_BIAS=dt_bias is not None,
        BIAS_IS_MATRIX=dt_bias is not None and dt_bias.dim() == 2,
        HAS_Z=z is not None,
        SOFTPLUS=dt_softplus,
        D_IS_VECTOR=D.dim() == 1,
        BLOCK_N=block_n,
    )
    return out


__all__ = ["ssu_one_token_musa_triton"]
