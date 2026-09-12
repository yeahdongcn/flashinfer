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
    src_slot_ptr,
    dst_slot_ptr,
    out_ptr,
    state_slot_stride,
    state_h_stride,
    state_d_stride,
    state_n_stride,
    x_b_stride,
    x_h_stride,
    x_d_stride,
    dt_b_stride,
    dt_h_stride,
    dt_d_stride,
    a_h_stride,
    a_d_stride,
    a_n_stride,
    b_b_stride,
    b_g_stride,
    b_n_stride,
    c_b_stride,
    c_g_stride,
    c_n_stride,
    d_h_stride,
    d_d_stride,
    bias_h_stride,
    bias_d_stride,
    out_b_stride,
    out_h_stride,
    out_d_stride,
    z_b_stride,
    z_h_stride,
    z_d_stride,
    pad_slot_id,
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
    HAS_PAD_SLOT: tl.constexpr,
):
    pid = tl.program_id(0)
    hd = H * D
    batch = pid // hd
    rem = pid % hd
    head = rem // D
    dim = rem % D
    group = head * G // H
    src_slot = tl.load(src_slot_ptr + batch).to(tl.int64)
    dst_slot = tl.load(dst_slot_ptr + batch).to(tl.int64)
    src_valid = True if not HAS_PAD_SLOT else src_slot != pad_slot_id
    dst_valid = True if not HAS_PAD_SLOT else dst_slot != pad_slot_id
    n = tl.arange(0, BLOCK_N)
    mask = n < N
    state_offset = src_slot * state_slot_stride + head * state_h_stride + dim * state_d_stride
    dst_state_offset = dst_slot * state_slot_stride + head * state_h_stride + dim * state_d_stride
    a_offset = head * a_h_stride + dim * a_d_stride
    bc_offset = batch * b_b_stride + group * b_g_stride
    s = tl.load(state_ptr + state_offset + n * state_n_stride, mask=mask & src_valid, other=0.0).to(tl.float32)
    a = tl.load(a_ptr + a_offset + n * a_n_stride, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + bc_offset + n * b_n_stride, mask=mask, other=0.0).to(tl.float32)
    c = tl.load(c_ptr + batch * c_b_stride + group * c_g_stride + n * c_n_stride, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(x_ptr + batch * x_b_stride + head * x_h_stride + dim * x_d_stride).to(tl.float32)
    dt = tl.load(dt_ptr + batch * dt_b_stride + head * dt_h_stride + dim * dt_d_stride).to(tl.float32)
    if HAS_BIAS:
        bias_offset = head * bias_h_stride + dim * bias_d_stride if BIAS_IS_MATRIX else head * bias_h_stride
        dt += tl.load(dt_bias_ptr + bias_offset).to(tl.float32)
    if SOFTPLUS:
        dt = tl.log(1.0 + tl.exp(dt))
    updated = s * tl.exp(a * dt) + (dt * x) * b
    tl.store(state_ptr + dst_state_offset + n * state_n_stride, updated, mask=mask & dst_valid)
    y = tl.sum(c * updated, axis=0)
    d_offset = head * d_h_stride + dim * d_d_stride if not D_IS_VECTOR else head * d_h_stride
    d = tl.load(d_ptr + d_offset).to(tl.float32)
    y += d * x
    if HAS_Z:
        z = tl.load(z_ptr + batch * z_b_stride + head * z_h_stride + dim * z_d_stride).to(tl.float32)
        y *= z / (1.0 + tl.exp(-z))
    tl.store(out_ptr + batch * out_b_stride + head * out_h_stride + dim * out_d_stride, y)


def ssu_one_token_musa_triton(
    state: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    state_batch_indices: torch.Tensor,
    dst_state_batch_indices: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    z: torch.Tensor | None = None,
    dt_softplus: bool = False,
    pad_slot_id: int = -1,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the fused MUSA decode kernel for the supported contract."""
    batch, heads, dim = x.shape
    groups = B.shape[1]
    dstate = A.shape[-1]
    block_n = triton.next_power_of_2(dstate)
    if dst_state_batch_indices is None:
        dst_state_batch_indices = state_batch_indices
    if dst_state_batch_indices.shape != state_batch_indices.shape:
        raise ValueError("MUSA fused SSU source/destination slot shapes differ")
    if out is None:
        out = torch.empty_like(x)
    elif out.shape != x.shape or out.dtype != x.dtype:
        raise ValueError("MUSA fused SSU out must match x shape and dtype")
    if any(t.dim() != 3 for t in (x, dt, B, C, out)) or state.dim() != 4 or A.dim() != 3:
        raise ValueError("MUSA fused SSU expects x/dt/B/C/out rank 3, state rank 4, A rank 3")
    if D.dim() not in (1, 2) or (dt_bias is not None and dt_bias.dim() not in (1, 2)):
        raise ValueError("MUSA fused SSU D and dt_bias must be rank 1 or 2")
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
        dst_state_batch_indices,
        out,
        state_slot_stride=state.stride(0),
        state_h_stride=state.stride(1),
        state_d_stride=state.stride(2),
        state_n_stride=state.stride(3),
        x_b_stride=x.stride(0),
        x_h_stride=x.stride(1),
        x_d_stride=x.stride(2),
        dt_b_stride=dt.stride(0),
        dt_h_stride=dt.stride(1),
        dt_d_stride=dt.stride(2),
        a_h_stride=A.stride(0),
        a_d_stride=A.stride(1),
        a_n_stride=A.stride(2),
        b_b_stride=B.stride(0),
        b_g_stride=B.stride(1),
        b_n_stride=B.stride(2),
        c_b_stride=C.stride(0),
        c_g_stride=C.stride(1),
        c_n_stride=C.stride(2),
        d_h_stride=D.stride(0),
        d_d_stride=D.stride(1) if D.dim() == 2 else 0,
        bias_h_stride=dt_bias.stride(0) if dt_bias is not None else 0,
        bias_d_stride=dt_bias.stride(1) if dt_bias is not None and dt_bias.dim() == 2 else 0,
        out_b_stride=out.stride(0),
        out_h_stride=out.stride(1),
        out_d_stride=out.stride(2),
        z_b_stride=z.stride(0) if z is not None else 0,
        z_h_stride=z.stride(1) if z is not None else 0,
        z_d_stride=z.stride(2) if z is not None else 0,
        pad_slot_id=pad_slot_id,
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
        HAS_PAD_SLOT=pad_slot_id >= 0,
    )
    return out


__all__ = ["ssu_one_token_musa_triton"]
