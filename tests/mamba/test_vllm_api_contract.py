"""Keep the FlashInfer Mamba entry points aligned with vLLM call sites."""

import inspect

import pytest

torch = pytest.importorskip("torch")

from flashinfer.mamba import (  # noqa: E402
    mamba_chunk_scan_combined_varlen,
    selective_state_update,
    ssd_combined_fwd_varlen,
)


def test_ssu_has_vllm_dispatch_keywords():
    names = list(inspect.signature(selective_state_update).parameters)
    for name in (
        "state_batch_indices",
        "dst_state_batch_indices",
        "intermediate_states_buffer",
        "rand_seed",
        "philox_rounds",
        "cache_steps",
        "cu_seqlens",
        "num_accepted_tokens",
    ):
        assert name in names


def test_varlen_ssd_matches_vllm_positional_prefix():
    expected = [
        "x",
        "dt",
        "A",
        "B",
        "C",
        "chunk_size",
        "cu_seqlens",
        "cu_chunk_seqlens",
        "last_chunk_indices",
        "seq_idx",
        "out",
        "D",
        "z",
        "dt_bias",
        "initial_states",
        "dt_softplus",
        "dt_limit",
        "return_intermediate_states",
        "state_dtype",
    ]
    names = list(inspect.signature(ssd_combined_fwd_varlen).parameters)
    assert names == expected
    assert mamba_chunk_scan_combined_varlen is ssd_combined_fwd_varlen
