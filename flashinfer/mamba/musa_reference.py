"""Device-independent Mamba bring-up path used by the MUSA backend.

This module intentionally uses regular PyTorch operators.  It is a correctness
scaffold for the first FlashInfer-MUSA port: it preserves the public SSU
contract while the native MUSA kernel is being developed.  It must not be used
as a performance implementation; the MUSA kernel will replace this function
behind the same dispatch boundary.
"""

from __future__ import annotations

from typing import Optional

import torch


def _softplus(x: torch.Tensor) -> torch.Tensor:
    # Match the selective-scan kernels' numerically stable branch at large x.
    return torch.where(x <= 20, torch.nn.functional.softplus(x), x)


def _as_head_dim(tensor: Optional[torch.Tensor], nheads: int, dim: int) -> Optional[torch.Tensor]:
    if tensor is None:
        return None
    if tensor.dim() == 1:
        return tensor.view(1, -1).expand(nheads, dim)
    if tensor.dim() == 2 and tensor.shape == (nheads, 1):
        return tensor.expand(nheads, dim)
    return tensor


def _index_for(
    indices: Optional[torch.Tensor], batch: int, token: int, default: int
) -> int:
    if indices is None:
        return default
    if indices.dim() == 1:
        return int(indices[batch].item())
    return int(indices[batch, token].item())


def selective_state_update_musa_reference(
    state: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: Optional[torch.Tensor],
    z: Optional[torch.Tensor],
    dt_bias: Optional[torch.Tensor],
    dt_softplus: bool,
    state_batch_indices: Optional[torch.Tensor],
    dst_state_batch_indices: Optional[torch.Tensor],
    pad_slot_id: int,
    out: torch.Tensor,
    disable_state_update: bool,
    intermediate_states_buffer: Optional[torch.Tensor],
    intermediate_state_indices: Optional[torch.Tensor],
    state_scale: Optional[torch.Tensor],
    intermediate_state_scales: Optional[torch.Tensor],
    rand_seed: Optional[torch.Tensor],
    cache_steps: int,
    cu_seqlens: Optional[torch.Tensor],
    num_accepted_tokens: Optional[torch.Tensor],
) -> torch.Tensor:
    """Reference SSU implementation for MUSA bring-up.

    The function supports the unquantized state paths used by Mamba2 and keeps
    all recurrence arithmetic in fp32.  Quantized state and stochastic
    rounding are rejected until their native MUSA kernels are available.
    """
    if state.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise NotImplementedError("MUSA SSU bring-up supports fp16/bf16/fp32 state only")
    if state_scale is not None or intermediate_state_scales is not None:
        raise NotImplementedError("MUSA SSU bring-up does not support scaled state yet")
    if rand_seed is not None:
        raise NotImplementedError("MUSA SSU stochastic rounding is not implemented yet")
    if num_accepted_tokens is not None:
        # The native MTP kernel will add replay-specific initial-token selection.
        raise NotImplementedError("MUSA SSU bring-up does not support speculative replay yet")

    if state.dim() != 4:
        raise ValueError(f"state must be [slots, heads, dim, dstate], got {state.shape}")
    slots, nheads, dim, dstate = state.shape

    is_varlen = cu_seqlens is not None and x.dim() == 3 and dt.dim() == 3
    is_mtp = x.dim() == 4
    if is_varlen:
        batch = int(cu_seqlens.numel() - 1)
        steps = [
            (int(cu_seqlens[b].item()), int(cu_seqlens[b + 1].item()))
            for b in range(batch)
        ]
    elif is_mtp:
        batch, nsteps = x.shape[:2]
        steps = [(0, nsteps) for _ in range(batch)]
    else:
        batch = x.shape[0]
        steps = [(0, 1) for _ in range(batch)]

    if is_varlen:
        if B.dim() != 3 or C.dim() != 3:
            raise ValueError("varlen B/C must be [tokens, groups, dstate]")
        ngroups = B.shape[1]
    elif is_mtp:
        if B.dim() != 4 or C.dim() != 4:
            raise ValueError("MTP B/C must be [batch, T, groups, dstate]")
        ngroups = B.shape[2]
    else:
        if B.dim() != 3 or C.dim() != 3:
            raise ValueError("single-token B/C must be [batch, groups, dstate]")
        ngroups = B.shape[1]
    if nheads % ngroups != 0:
        raise ValueError("nheads must be divisible by ngroups")
    group_ratio = nheads // ngroups

    A_h = A if A.dim() == 3 else A.unsqueeze(0)
    if A_h.shape != (nheads, dim, dstate):
        raise ValueError(f"A must be [{nheads}, {dim}, {dstate}], got {A.shape}")
    D_h = _as_head_dim(D, nheads, dim)
    bias_h = _as_head_dim(dt_bias, nheads, dim)
    if D_h is not None and D_h.shape != (nheads, dim):
        raise ValueError(f"D must broadcast to [{nheads}, {dim}], got {D.shape}")
    if bias_h is not None and bias_h.shape != (nheads, dim):
        raise ValueError(
            f"dt_bias must broadcast to [{nheads}, {dim}], got {dt_bias.shape}"
        )

    def update_one(batch_idx: int, token_idx: int, state_slot: int) -> torch.Tensor:
        if state_slot == pad_slot_id:
            running = torch.zeros(
                (nheads, dim, dstate), dtype=torch.float32, device=state.device
            )
        else:
            if state_slot < 0 or state_slot >= slots:
                raise IndexError(f"state slot {state_slot} is outside [0, {slots})")
            running = state[state_slot].to(torch.float32).clone()

        for h in range(nheads):
            g = h // group_ratio
            if is_varlen:
                x_t = x[token_idx, h].to(torch.float32)
                dt_t = dt[token_idx, h].to(torch.float32)
                b_t = B[token_idx, g].to(torch.float32)
                c_t = C[token_idx, g].to(torch.float32)
                z_t = z[token_idx, h].to(torch.float32) if z is not None else None
            elif is_mtp:
                x_t = x[batch_idx, token_idx, h].to(torch.float32)
                dt_t = dt[batch_idx, token_idx, h].to(torch.float32)
                b_t = B[batch_idx, token_idx, g].to(torch.float32)
                c_t = C[batch_idx, token_idx, g].to(torch.float32)
                z_t = z[batch_idx, token_idx, h].to(torch.float32) if z is not None else None
            else:
                x_t = x[batch_idx, h].to(torch.float32)
                dt_t = dt[batch_idx, h].to(torch.float32)
                b_t = B[batch_idx, g].to(torch.float32)
                c_t = C[batch_idx, g].to(torch.float32)
                z_t = z[batch_idx, h].to(torch.float32) if z is not None else None

            if bias_h is not None:
                dt_t = dt_t + bias_h[h]
            if dt_softplus:
                dt_t = _softplus(dt_t)
            d_a = torch.exp(A_h[h] * dt_t)
            running[h] = running[h] * d_a + (dt_t * b_t)[None, :] * x_t[:, None]
            y_t = torch.sum(c_t[None, :] * running[h], dim=-1)
            if D_h is not None:
                y_t = y_t + D_h[h].to(torch.float32) * x_t
            if z_t is not None:
                y_t = y_t * z_t * torch.sigmoid(z_t)

            if is_varlen:
                out[token_idx, h].copy_(y_t.to(out.dtype))
            elif is_mtp:
                out[batch_idx, token_idx, h].copy_(y_t.to(out.dtype))
            else:
                out[batch_idx, h].copy_(y_t.to(out.dtype))
        return running

    for b, (start, end) in enumerate(steps):
        read_slot = _index_for(state_batch_indices, b, 0, b)
        running = None
        for token in range(end - start):
            token_idx = start + token if is_varlen else token
            running = update_one(b, token_idx, read_slot if token == 0 else read_slot)
            if is_mtp and dst_state_batch_indices is not None:
                write_slot = _index_for(dst_state_batch_indices, b, token, read_slot)
            else:
                write_slot = _index_for(dst_state_batch_indices, b, 0, read_slot)
            if not disable_state_update and write_slot != pad_slot_id:
                state[write_slot].copy_(running.to(state.dtype))
                read_slot = write_slot
            if intermediate_states_buffer is not None:
                cache_slot = (
                    int(intermediate_state_indices[b].item())
                    if intermediate_state_indices is not None
                    else b
                )
                intermediate_states_buffer[cache_slot, token].copy_(
                    running.to(intermediate_states_buffer.dtype)
                )
    return out


def ssd_combined_fwd_musa_reference(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: Optional[torch.Tensor] = None,
    z: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    dt_softplus: bool = False,
    dt_limit: tuple[float, float] = (0.0, float("inf")),
    initial_states: Optional[torch.Tensor] = None,
    seq_idx: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    return_final_states: bool = True,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Reference Mamba2 SSD forward for the initial MUSA port.

    This implements the same recurrence as SSDCombined without materializing
    the chunk matrices.  It is intentionally a correctness path; a native
    MUSA implementation will replace it after the API and reference tests are
    established.
    """
    if x.dim() != 4 or dt.dim() != 3:
        raise ValueError("x must be [batch, seqlen, heads, headdim] and dt [batch, seqlen, heads]")
    batch, seqlen, nheads, headdim = x.shape
    if dt.shape != (batch, seqlen, nheads):
        raise ValueError("dt shape does not match x")
    if B.shape != C.shape or B.shape[:2] != (batch, seqlen):
        raise ValueError("B and C must have shape [batch, seqlen, groups, dstate]")
    ngroups, dstate = B.shape[2:]
    if nheads % ngroups:
        raise ValueError("nheads must be divisible by ngroups")
    if A.shape != (nheads,):
        raise ValueError(f"A must have shape [{nheads}]")
    if z is not None and z.shape != x.shape:
        raise ValueError("z must have the same shape as x")

    state_dtype = initial_states.dtype if initial_states is not None else torch.bfloat16
    if state_dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise NotImplementedError("MUSA SSD bring-up supports fp16/bf16/fp32 state only")
    if initial_states is None:
        num_sequences = batch
        state = torch.zeros(
            batch, nheads, headdim, dstate, dtype=torch.float32, device=x.device
        )
        initial = None
    else:
        if initial_states.dim() != 4 or initial_states.shape[1:] != (
            nheads,
            headdim,
            dstate,
        ):
            raise ValueError("initial_states must be [num_sequences, heads, headdim, dstate]")
        num_sequences = initial_states.shape[0]
        initial = initial_states.to(torch.float32)
        state = initial[:batch].clone()
        if state.shape[0] != batch:
            state = torch.zeros(
                batch, nheads, headdim, dstate, dtype=torch.float32, device=x.device
            )

    if out is None:
        out = torch.empty_like(x)
    if out.shape != x.shape:
        raise ValueError("out must have the same shape as x")

    bias = None if dt_bias is None else dt_bias.reshape(1, 1, nheads).to(torch.float32)
    d_head = None
    if D is not None:
        d_head = D.to(torch.float32)
        if d_head.dim() == 1:
            d_head = d_head[:, None].expand(nheads, headdim)
        if d_head.shape != (nheads, headdim):
            raise ValueError("D must have shape [heads] or [heads, headdim]")

    # Direct recurrence over tokens.  All state arithmetic remains fp32; only
    # the externally visible output and final state use the requested dtypes.
    final = torch.empty(
        num_sequences, nheads, headdim, dstate, dtype=state_dtype, device=x.device
    )
    seen = torch.zeros(num_sequences, dtype=torch.bool, device=x.device)
    previous_seq = None
    ratio = nheads // ngroups
    dt_f = dt.to(torch.float32)
    for token in range(seqlen):
        if seq_idx is not None:
            ids = seq_idx[:, token].to(torch.int64)
            if ids.numel() != batch:
                raise ValueError("seq_idx must have shape [batch, seqlen]")
            changed = torch.ones(batch, dtype=torch.bool, device=x.device)
            if previous_seq is not None:
                changed = ids != previous_seq
            if initial is not None:
                replacement = initial.index_select(0, ids.clamp_min(0))
                state = torch.where(changed[:, None, None, None], replacement, state)
            else:
                state = torch.where(
                    changed[:, None, None, None],
                    torch.zeros_like(state),
                    state,
                )
            previous_seq = ids

        delta = dt_f[:, token]
        if bias is not None:
            delta = delta + bias[:, 0]
        if dt_softplus:
            delta = _softplus(delta)
        delta = delta.clamp(dt_limit[0], dt_limit[1])
        decay = torch.exp(A.to(torch.float32)[None, :, None, None] * delta[:, :, None, None])
        state = state * decay
        for head in range(nheads):
            group = head // ratio
            state[:, head] += (
                delta[:, head, None, None]
                * x[:, token, head].to(torch.float32)[:, :, None]
                * B[:, token, group].to(torch.float32)[:, None, :]
            )
        y = torch.empty(batch, nheads, headdim, dtype=torch.float32, device=x.device)
        for head in range(nheads):
            group = head // ratio
            y[:, head] = torch.sum(
                C[:, token, group].to(torch.float32)[:, None, :]
                * state[:, head],
                dim=-1,
            )
        if d_head is not None:
            y = y + x[:, token].to(torch.float32) * d_head[None]
        if z is not None:
            z_t = z[:, token].to(torch.float32)
            y = y * z_t * torch.sigmoid(z_t)
        out[:, token].copy_(y.to(out.dtype))

        if seq_idx is None:
            final.copy_(state.to(state_dtype))
        else:
            for sequence in torch.unique(ids).tolist():
                final[sequence].copy_(state[ids == sequence][0].to(state_dtype))
                seen[sequence] = True

    if seq_idx is not None and not bool(seen.all()):
        missing = (~seen).nonzero(as_tuple=False).flatten().tolist()
        if initial is not None:
            final[missing] = initial[missing].to(state_dtype)
        else:
            final[missing] = 0
    return out, final if return_final_states else None
