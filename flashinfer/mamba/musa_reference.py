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
    philox_rounds: int,
    cache_steps: int,
    cu_seqlens: Optional[torch.Tensor],
    num_accepted_tokens: Optional[torch.Tensor],
) -> torch.Tensor:
    """Reference SSU implementation for MUSA bring-up.

    The function keeps recurrence arithmetic in fp32 and mirrors the public
    state/index/replay contract.  Native MUSA kernels will replace this slow
    reference implementation behind the same boundary.
    """
    if state.dtype not in (torch.int16, torch.float16, torch.bfloat16, torch.float32):
        raise NotImplementedError("MUSA SSU supports int16/fp16/bf16/fp32 state")
    if D is not None and D.dtype != dt.dtype:
        raise ValueError("D must have the same dtype as dt")
    if state.dtype == torch.int16 and state_scale is None:
        raise ValueError("int16 state requires state_scale")
    if state.dtype != torch.int16 and state_scale is not None:
        raise ValueError("state_scale is only valid for int16 state")
    if state_scale is not None and state_scale.dim() == 4 and state_scale.shape[-1] == 1:
        state_scale = state_scale.squeeze(-1)
    if (
        intermediate_state_scales is not None
        and intermediate_state_scales.dim() == 5
        and intermediate_state_scales.shape[-1] == 1
    ):
        intermediate_state_scales = intermediate_state_scales.squeeze(-1)
    if (
        state_batch_indices is not None
        and intermediate_state_indices is not None
        and state_batch_indices.dtype != intermediate_state_indices.dtype
    ):
        raise ValueError("state and intermediate index tensors must have the same dtype")

    if state.dim() != 4:
        raise ValueError(f"state must be [slots, heads, dim, dstate], got {state.shape}")
    slots, nheads, dim, dstate = state.shape

    def cast_state(value: torch.Tensor, target_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        target_dtype = state.dtype if target_dtype is None else target_dtype
        if rand_seed is None or target_dtype not in (torch.float16, torch.bfloat16):
            return value.to(target_dtype)
        # Reference stochastic cast.  Native kernels will use the exact
        # device Philox sequence; this preserves the public seed/rounds
        # contract while keeping the bring-up path device-independent.
        mantissa_bits = 10 if target_dtype == torch.float16 else 7
        magnitude = value.abs().clamp_min(torch.finfo(torch.float32).tiny)
        exponent = torch.floor(torch.log2(magnitude))
        step = torch.pow(2.0, exponent - mantissa_bits)
        lower = torch.floor(value / step) * step
        probability = ((value - lower) / step).clamp(0, 1)
        try:
            generator = torch.Generator(device=value.device)
            generator.manual_seed(int(rand_seed.reshape(-1)[0].item()) + philox_rounds)
            random = torch.rand(value.shape, device=value.device, generator=generator)
        except (RuntimeError, TypeError):
            random = torch.rand(value.shape, device=value.device)
        rounded = torch.where(random < probability, lower + step, lower)
        return rounded.to(target_dtype)

    def round_integer(value: torch.Tensor) -> torch.Tensor:
        if rand_seed is None:
            return value.round()
        try:
            generator = torch.Generator(device=value.device)
            generator.manual_seed(int(rand_seed.reshape(-1)[0].item()) + philox_rounds)
            random = torch.rand(value.shape, device=value.device, generator=generator)
        except (RuntimeError, TypeError):
            random = torch.rand(value.shape, device=value.device)
        lower = torch.floor(value)
        return lower + (random < (value - lower)).to(value.dtype)

    def read_state(state_slot: int) -> torch.Tensor:
        if state_slot == pad_slot_id:
            return torch.zeros((nheads, dim, dstate), dtype=torch.float32, device=state.device)
        if state_slot < 0 or state_slot >= slots:
            raise IndexError(f"state slot {state_slot} is outside [0, {slots})")
        value = state[state_slot].to(torch.float32).clone()
        if state.dtype == torch.int16:
            value = value * state_scale[state_slot].to(torch.float32)[..., None]
        return value

    def write_state(state_slot: int, value: torch.Tensor) -> None:
        if disable_state_update or state_slot == pad_slot_id:
            return
        if state_slot < 0 or state_slot >= slots:
            raise IndexError(f"state slot {state_slot} is outside [0, {slots})")
        if state.dtype == torch.int16:
            scale = value.abs().amax(dim=-1).clamp_min(torch.finfo(torch.float32).tiny) / 32767
            state[state_slot].copy_(round_integer(value / scale[..., None]).clamp(-32768, 32767).to(torch.int16))
            state_scale[state_slot].copy_(scale.to(state_scale.dtype))
        else:
            state[state_slot].copy_(cast_state(value))

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

    def update_one(
        batch_idx: int,
        token_idx: int,
        state_slot: int,
        running_override: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        running = read_state(state_slot) if running_override is None else running_override

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
            d_a = torch.exp(A_h[h] * dt_t[:, None])
            running[h] = (
                running[h]
                * d_a
                + (dt_t * x_t)[:, None] * b_t[None, :]
            )
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
        accepted = (
            int(num_accepted_tokens[b].item())
            if num_accepted_tokens is not None
            else 0
        )
        read_slot = _index_for(state_batch_indices, b, accepted, b)
        running = None
        for token in range(end - start):
            token_idx = start + token if is_varlen else token
            running = update_one(
                b,
                token_idx,
                read_slot,
                running_override=running if token > 0 else None,
            )
            if is_mtp and dst_state_batch_indices is not None:
                write_slot = _index_for(dst_state_batch_indices, b, token, read_slot)
                write_state(write_slot, running)
                read_slot = write_slot
            elif not is_mtp and not is_varlen:
                write_slot = _index_for(dst_state_batch_indices, b, 0, read_slot)
                write_state(write_slot, running)
                read_slot = write_slot
            if intermediate_states_buffer is not None:
                cache_slot = (
                    int(intermediate_state_indices[b].item())
                    if intermediate_state_indices is not None
                    else b
                )
                if intermediate_states_buffer.dtype == torch.int16:
                    if intermediate_state_scales is None:
                        raise ValueError("int16 intermediate state requires scales")
                    scale = running.abs().amax(dim=-1).clamp_min(torch.finfo(torch.float32).tiny) / 32767
                    intermediate_states_buffer[cache_slot, token].copy_(
                        round_integer(running / scale[..., None]).clamp(-32768, 32767).to(torch.int16)
                    )
                    intermediate_state_scales[cache_slot, token].copy_(scale.to(intermediate_state_scales.dtype))
                else:
                    intermediate_states_buffer[cache_slot, token].copy_(
                        cast_state(running, intermediate_states_buffer.dtype)
                    )
        if (
            not disable_state_update
            and not is_mtp
            and is_varlen
            or (
                not disable_state_update
                and is_mtp
                and dst_state_batch_indices is None
                and intermediate_states_buffer is None
            )
        ):
            write_state(read_slot, running)
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


def ssd_combined_fwd_varlen_musa_reference(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    chunk_size: int,
    cu_seqlens: torch.Tensor,
    cu_chunk_seqlens: torch.Tensor,
    last_chunk_indices: torch.Tensor,
    seq_idx: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    D: Optional[torch.Tensor] = None,
    z: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    dt_softplus: bool = False,
    dt_limit: tuple[float, float] = (0.0, float("inf")),
    initial_states: Optional[torch.Tensor] = None,
    return_intermediate_states: bool = False,
    state_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Packed/varlen SSD API matching vLLM's Mamba2 prefill contract."""
    if x.dim() != 3 or dt.dim() != 2 or B.dim() != 3 or C.dim() != 3:
        raise ValueError("varlen SSD expects packed x/dt/B/C tensors")
    tokens, nheads, headdim = x.shape
    nchunks = cu_chunk_seqlens.numel() - 1
    if dt.shape != (tokens, nheads) or B.shape != C.shape:
        raise ValueError("packed SSD tensor shapes do not match")
    ngroups, dstate = B.shape[1:]
    if nheads % ngroups or A.shape != (nheads,):
        raise ValueError("A/groups are incompatible with the packed head count")
    if cu_seqlens.numel() != last_chunk_indices.numel() + 1:
        raise ValueError("cu_seqlens and last_chunk_indices do not describe the same batch")
    if seq_idx.numel() != nchunks:
        raise ValueError("seq_idx must contain one sequence id per physical chunk")
    if torch.any(cu_chunk_seqlens[1:] < cu_chunk_seqlens[:-1]) or int(cu_chunk_seqlens[-1]) != tokens:
        raise ValueError("cu_chunk_seqlens must be monotonic and end at x.shape[0]")

    if out is None:
        out = torch.empty_like(x)
    if initial_states is not None:
        if initial_states.dim() != 4 or initial_states.shape[1:] != (
            nheads,
            headdim,
            dstate,
        ):
            raise ValueError("initial_states must be [num_sequences, heads, headdim, dstate]")
        state_dtype = state_dtype or initial_states.dtype
        initial = initial_states.to(torch.float32)
    else:
        state_dtype = state_dtype or torch.float32
        initial = None

    if D is not None:
        D_f = D.to(torch.float32)
        if D_f.dim() == 1:
            D_f = D_f[:, None].expand(nheads, headdim)
        if D_f.shape != (nheads, headdim):
            raise ValueError("D must have shape [heads] or [heads, headdim]")
    else:
        D_f = None
    bias = None if dt_bias is None else dt_bias.to(torch.float32).reshape(1, nheads)
    ratio = nheads // ngroups
    A_f = A.to(torch.float32)
    states = torch.empty(
        nchunks, nheads, headdim, dstate, dtype=state_dtype, device=x.device
    )
    state_by_sequence: dict[int, torch.Tensor] = {}
    seen_sequences: set[int] = set()

    for chunk in range(nchunks):
        sequence = int(seq_idx[chunk].item())
        start = int(cu_chunk_seqlens[chunk].item())
        end = int(cu_chunk_seqlens[chunk + 1].item())
        if end - start > chunk_size:
            raise ValueError("a physical chunk exceeds chunk_size")
        if sequence not in seen_sequences:
            if initial is None:
                running = torch.zeros(
                    nheads, headdim, dstate, dtype=torch.float32, device=x.device
                )
            else:
                running = initial[sequence].clone()
            seen_sequences.add(sequence)
        else:
            running = state_by_sequence[sequence]

        for token in range(start, end):
            delta = dt[token].to(torch.float32)
            if bias is not None:
                delta = delta + bias[0]
            if dt_softplus:
                delta = _softplus(delta)
            delta = delta.clamp(dt_limit[0], dt_limit[1])
            for head in range(nheads):
                group = head // ratio
                running[head] = running[head] * torch.exp(A_f[head] * delta[head])
                running[head] += (
                    (delta[head] * x[token, head].to(torch.float32))[:, None]
                    * B[token, group].to(torch.float32)[None, :]
                )
                y = torch.sum(
                    C[token, group].to(torch.float32)[None, :]
                    * running[head],
                    dim=-1,
                )
                if D_f is not None:
                    y = y + D_f[head] * x[token, head].to(torch.float32)
                if z is not None:
                    z_t = z[token, head].to(torch.float32)
                    y = y * z_t * torch.sigmoid(z_t)
                out[token, head].copy_(y.to(out.dtype))
        state_by_sequence[sequence] = running
        states[chunk].copy_(running.to(state_dtype))

    if return_intermediate_states:
        return states
    return states.index_select(0, last_chunk_indices.to(torch.int64))
