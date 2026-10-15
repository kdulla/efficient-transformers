# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
"""
Unit tests for the KV-block gather inside QEfficient.blocking.blocked_attention_forwards.

Covers, for `blocked_qkv_attention_forward`, `blocked_hqkv_attention_forward` and
`blocked_bhqkv_attention_forward` with `paged_attention=True`:
  1. The per-KV-block gather (`_read_kv_block`) runs exactly once per KV block per
     forward call, not once per (KV block x head/q/batch block) combination.
  2. The gathered output is still numerically correct (matches plain causal
     attention over the same paged cache).
  3. `skip_kv`'s early break out of the KV-block loop still stops the gather as
     soon as a block lies entirely in the future.

All tests run on CPU only, using hand-built paged KV caches (no full model needed).
"""

from unittest.mock import patch

import pytest
import torch

from QEfficient.blocking.blocked_attention_forwards import (
    _read_kv_block,
    blocked_bhqkv_attention_forward,
    blocked_hqkv_attention_forward,
    blocked_qkv_attention_forward,
    paged_gqa_attention_forward,
    repeat_kv,
)
from QEfficient.transformers.cache_utils import QEffDynamicCache, QEffDynamicLayer

HEAD_DIM = 4
BLOCK_SIZE = 2
NUM_PHYS_BLOCKS = 4
CTX_LEN = BLOCK_SIZE * NUM_PHYS_BLOCKS  # 8
SCALING = 1.0 / (HEAD_DIM**0.5)


class _DummyModule:
    """Minimal attention-module double: no `config` attribute (forces the
    value.dtype fallback for the mask) and an explicit `num_key_value_groups`
    so `_get_kv_states` takes its no-op path (num_kv_heads == num_heads)."""

    def __init__(self, num_key_value_groups=1):
        self.num_key_value_groups = num_key_value_groups


def _make_paged_cache(heads, block_size=BLOCK_SIZE, num_phys_blocks=NUM_PHYS_BLOCKS, head_dim=HEAD_DIM):
    cache = QEffDynamicCache()
    layer = QEffDynamicLayer()
    layer.keys = torch.randn(num_phys_blocks, heads, block_size, head_dim)
    layer.values = torch.randn(num_phys_blocks, heads, block_size, head_dim)
    layer.is_initialized = True
    cache.layers.append(layer)
    return cache


def _full_kv(cache, layer_idx=0):
    """Concatenate all physical blocks into one contiguous (heads, CTX_LEN, head_dim)
    tensor, matching what an identity block_table exposes to every batch row."""
    layer = cache.layers[layer_idx]
    k_full = torch.cat(list(layer.keys), dim=1)
    v_full = torch.cat(list(layer.values), dim=1)
    return k_full, v_full


def _manual_causal_attention(query, k_full, v_full, query_positions, scaling):
    """Reference: plain (unblocked) causal attention over `k_full`/`v_full`."""
    batch, _heads, q_len, _ = query.shape
    ctx_len = k_full.shape[1]
    k = k_full.unsqueeze(0).expand(batch, -1, -1, -1)
    v = v_full.unsqueeze(0).expand(batch, -1, -1, -1)
    attn_weights = torch.matmul(query, k.transpose(2, 3)) * scaling
    kv_indices = torch.arange(ctx_len).view(1, 1, 1, -1)
    q_pos = query_positions.view(batch, 1, q_len, 1)
    causal_mask = kv_indices > q_pos
    attn_weights = attn_weights.masked_fill(causal_mask, float("-inf"))
    attn_weights = torch.softmax(attn_weights, dim=-1)
    return torch.matmul(attn_weights, v)


def _manual_batched_causal_attention(query, key, value, query_positions, scaling):
    """Reference causal attention when each batch row owns distinct K/V states."""
    batch, _heads, query_len, _ = query.shape
    ctx_len = key.shape[2]
    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
    kv_indices = torch.arange(ctx_len).view(1, 1, 1, -1)
    causal_mask = kv_indices > query_positions.view(batch, 1, query_len, 1)
    attn_weights = attn_weights.masked_fill(causal_mask, float("-inf"))
    return torch.matmul(torch.softmax(attn_weights, dim=-1), value)


def _identity_block_table(batch, num_kv_blocks):
    return torch.arange(num_kv_blocks).unsqueeze(0).expand(batch, -1).clone()


def _cache_kwargs(position_ids, block_table):
    return {"position_ids": position_ids, "block_table": block_table}


@pytest.mark.cache
def test_paged_gqa_decode_matches_manual_causal_attention():
    num_kv_heads = 1
    num_query_heads = 2
    cache = _make_paged_cache(heads=num_kv_heads)
    module = _DummyModule(num_key_value_groups=num_query_heads // num_kv_heads)
    query = torch.randn(1, num_query_heads, 1, HEAD_DIM)
    key = torch.randn(1, num_kv_heads, 1, HEAD_DIM)
    value = torch.randn(1, num_kv_heads, 1, HEAD_DIM)
    position_ids = torch.tensor([[CTX_LEN - 1]])
    block_table = torch.arange(NUM_PHYS_BLOCKS).view(1, 1, NUM_PHYS_BLOCKS)

    output, weights = paged_gqa_attention_forward(
        module=module,
        query=query,
        key=key,
        value=value,
        attention_mask=None,
        scaling=SCALING,
        num_kv_blocks=2,
        cache_kwargs={"position_ids": position_ids, "block_table": block_table},
        layer_idx=0,
        past_key_value=cache,
        ctx_len=CTX_LEN,
        attn_dp=1,
        attn_cp=1,
        page_block_size=BLOCK_SIZE,
        num_cores_per_device=1,
    )

    key_cache = cache.layers[0].keys
    value_cache = cache.layers[0].values
    key_full = key_cache[:, 0].reshape(CTX_LEN, HEAD_DIM).unsqueeze(0)
    value_full = value_cache[:, 0].reshape(CTX_LEN, HEAD_DIM).unsqueeze(0)
    key_full = repeat_kv(key_full.unsqueeze(0), module.num_key_value_groups).squeeze(0)
    value_full = repeat_kv(value_full.unsqueeze(0), module.num_key_value_groups).squeeze(0)
    expected = _manual_causal_attention(query, key_full, value_full, position_ids, SCALING)

    assert weights is None
    assert torch.allclose(output, expected.transpose(1, 2), atol=1e-4, rtol=1e-4)


@pytest.mark.cache
def test_paged_gqa_prefill_matches_manual_causal_attention():
    num_kv_heads = 1
    num_query_heads = 4
    cache = _make_paged_cache(heads=num_kv_heads)
    module = _DummyModule(num_key_value_groups=num_query_heads // num_kv_heads)
    query = torch.randn(1, num_query_heads, CTX_LEN, HEAD_DIM)
    key = torch.randn(1, num_kv_heads, CTX_LEN, HEAD_DIM)
    value = torch.randn(1, num_kv_heads, CTX_LEN, HEAD_DIM)
    position_ids = torch.arange(CTX_LEN).unsqueeze(0)
    block_table = torch.arange(NUM_PHYS_BLOCKS).view(1, 1, NUM_PHYS_BLOCKS)

    output, weights = paged_gqa_attention_forward(
        module=module,
        query=query,
        key=key,
        value=value,
        attention_mask=None,
        scaling=SCALING,
        num_kv_blocks=2,
        cache_kwargs={"position_ids": position_ids, "block_table": block_table},
        layer_idx=0,
        past_key_value=cache,
        ctx_len=CTX_LEN,
        attn_dp=1,
        attn_cp=1,
        page_block_size=BLOCK_SIZE,
        num_cores_per_device=2,
        num_q_blocks=2,
        q_blocks_per_outer=1,
        head_block_size=1,
    )

    key_full = repeat_kv(key, module.num_key_value_groups).squeeze(0)
    value_full = repeat_kv(value, module.num_key_value_groups).squeeze(0)
    expected = _manual_causal_attention(query, key_full, value_full, position_ids, SCALING)

    assert weights is None
    assert torch.allclose(output, expected.transpose(1, 2), atol=1e-4, rtol=1e-4)


@pytest.mark.accuracy
@pytest.mark.cache
def test_paged_gqa_batch_two_dp2_cp2_prefill_repack_decode_accuracy():
    """Match the qwen3_moe_paged_gqa example's prefill-to-decode topology at CPU scale."""
    from examples.text_generation.qwen3_moe_paged_gqa import _block_table, _repack_paged_gqa_cache

    batch = 2
    attn_dp = 2
    attn_cp = 2
    num_query_heads = 8
    num_kv_heads = 1
    num_cores = 4
    num_kv_blocks = 4
    page_block_size = 2
    prefill_seq_len = 8
    ctx_len = 64
    num_logical_pages = ctx_len // page_block_size
    prefill_physical_pages = batch * num_logical_pages
    decode_physical_pages = batch * (num_logical_pages // attn_cp)
    generator = torch.Generator().manual_seed(42)
    module = _DummyModule(num_key_value_groups=num_query_heads // num_kv_heads)

    prefill_layer = QEffDynamicLayer.from_tensors(
        torch.zeros(prefill_physical_pages, num_kv_heads, page_block_size, HEAD_DIM),
        torch.zeros(prefill_physical_pages, num_kv_heads, page_block_size, HEAD_DIM),
    )
    prefill_cache = QEffDynamicCache()
    prefill_cache.layers.append(prefill_layer)
    prefill_table_np = _block_table(1, batch, num_logical_pages, 1, prefill_physical_pages)
    prefill_table = torch.from_numpy(prefill_table_np)
    prefill_positions = torch.arange(prefill_seq_len).unsqueeze(0).expand(batch, -1)
    prefill_query = torch.randn(batch, num_query_heads, prefill_seq_len, HEAD_DIM, generator=generator)
    prefill_key = torch.randn(batch, num_kv_heads, prefill_seq_len, HEAD_DIM, generator=generator)
    prefill_value = torch.randn(batch, num_kv_heads, prefill_seq_len, HEAD_DIM, generator=generator)

    prefill_output, prefill_weights = paged_gqa_attention_forward(
        module=module,
        query=prefill_query,
        key=prefill_key,
        value=prefill_value,
        attention_mask=None,
        scaling=SCALING,
        num_kv_blocks=num_kv_blocks,
        cache_kwargs={"position_ids": prefill_positions, "block_table": prefill_table},
        layer_idx=0,
        past_key_value=prefill_cache,
        ctx_len=ctx_len,
        skip_kv=True,
        attn_dp=1,
        attn_cp=1,
        page_block_size=page_block_size,
        num_cores_per_device=num_cores,
        num_q_blocks=2,
        q_blocks_per_outer=1,
        head_block_size=1,
    )

    expected_prefill = _manual_batched_causal_attention(
        prefill_query,
        repeat_kv(prefill_key, module.num_key_value_groups),
        repeat_kv(prefill_value, module.num_key_value_groups),
        prefill_positions,
        SCALING,
    )
    assert prefill_weights is None
    assert torch.allclose(prefill_output, expected_prefill.transpose(1, 2), atol=1e-4, rtol=1e-4)

    decode_table_np = _block_table(attn_dp, batch, num_logical_pages, attn_cp, decode_physical_pages)
    decode_cache_shape = (decode_physical_pages, attn_dp * attn_cp * num_kv_heads, page_block_size, HEAD_DIM)
    decode_key_cache = _repack_paged_gqa_cache(
        prefill_cache.layers[0].keys.numpy(),
        prefill_table_np,
        decode_table_np,
        decode_cache_shape,
        attn_dp,
        attn_cp,
    )
    decode_value_cache = _repack_paged_gqa_cache(
        prefill_cache.layers[0].values.numpy(),
        prefill_table_np,
        decode_table_np,
        decode_cache_shape,
        attn_dp,
        attn_cp,
    )
    decode_cache = QEffDynamicCache()
    decode_cache.layers.append(
        QEffDynamicLayer.from_tensors(torch.from_numpy(decode_key_cache), torch.from_numpy(decode_value_cache))
    )
    decode_query = torch.randn(batch, num_query_heads, 1, HEAD_DIM, generator=generator)
    decode_key = torch.randn(batch, num_kv_heads, 1, HEAD_DIM, generator=generator)
    decode_value = torch.randn(batch, num_kv_heads, 1, HEAD_DIM, generator=generator)
    decode_positions = torch.full((batch, 1), prefill_seq_len)

    decode_output, decode_weights = paged_gqa_attention_forward(
        module=module,
        query=decode_query,
        key=decode_key,
        value=decode_value,
        attention_mask=None,
        scaling=SCALING,
        num_kv_blocks=num_kv_blocks,
        cache_kwargs={"position_ids": decode_positions, "block_table": torch.from_numpy(decode_table_np)},
        layer_idx=0,
        past_key_value=decode_cache,
        ctx_len=ctx_len,
        skip_kv=True,
        attn_dp=attn_dp,
        attn_cp=attn_cp,
        page_block_size=page_block_size,
        num_cores_per_device=num_cores,
    )

    expected_decode = _manual_batched_causal_attention(
        decode_query,
        repeat_kv(torch.cat((prefill_key, decode_key), dim=2), module.num_key_value_groups),
        repeat_kv(torch.cat((prefill_value, decode_value), dim=2), module.num_key_value_groups),
        decode_positions,
        SCALING,
    )
    assert decode_weights is None
    assert torch.allclose(decode_output, expected_decode.transpose(1, 2), atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# Tests: hoisted gather is called once per KV block, not once per KV block per
# head/q/batch block
# ---------------------------------------------------------------------------


@pytest.mark.cache
class TestHoistedGatherCallCount:
    """After hoisting, `_read_kv_block` must be called exactly once per KV
    block per forward call, regardless of how many head/q/batch blocks wrap
    around the KV-block loop."""

    def test_qkv_gathers_each_kv_block_once(self):
        heads = 2
        cache = _make_paged_cache(heads=heads)
        module = _DummyModule()
        query = torch.randn(1, heads, 2, HEAD_DIM)
        cache_kwargs = _cache_kwargs(torch.tensor([[6, 7]]), _identity_block_table(1, NUM_PHYS_BLOCKS))

        with patch("QEfficient.blocking.blocked_attention_forwards._read_kv_block", wraps=_read_kv_block) as mocked:
            blocked_qkv_attention_forward(
                module=module,
                query=query,
                key=None,
                value=torch.zeros(1, heads, 1, HEAD_DIM),
                attention_mask=None,
                scaling=SCALING,
                num_kv_blocks=NUM_PHYS_BLOCKS,
                num_q_blocks=2,
                cache_kwargs=cache_kwargs,
                layer_idx=0,
                past_key_value=cache,
                paged_attention=True,
                use_causal_mask=True,
                skip_kv=False,
                ctx_len=CTX_LEN,
            )
        assert mocked.call_count == NUM_PHYS_BLOCKS, (
            f"Expected the KV gather to run once per KV block ({NUM_PHYS_BLOCKS}), got "
            f"{mocked.call_count} calls -- the per-q-block gather was not hoisted."
        )

    def test_hqkv_gathers_each_kv_block_once(self):
        heads = 4
        cache = _make_paged_cache(heads=heads)
        module = _DummyModule()
        query = torch.randn(1, heads, 2, HEAD_DIM)
        cache_kwargs = _cache_kwargs(torch.tensor([[6, 7]]), _identity_block_table(1, NUM_PHYS_BLOCKS))

        with patch("QEfficient.blocking.blocked_attention_forwards._read_kv_block", wraps=_read_kv_block) as mocked:
            blocked_hqkv_attention_forward(
                module=module,
                query=query,
                key=None,
                value=torch.zeros(1, heads, 1, HEAD_DIM),
                attention_mask=None,
                scaling=SCALING,
                num_kv_blocks=NUM_PHYS_BLOCKS,
                num_q_blocks=2,
                head_block_size=2,
                cache_kwargs=cache_kwargs,
                layer_idx=0,
                past_key_value=cache,
                paged_attention=True,
                use_causal_mask=True,
                skip_kv=False,
                ctx_len=CTX_LEN,
            )
        assert mocked.call_count == NUM_PHYS_BLOCKS, (
            f"Expected the KV gather to run once per KV block ({NUM_PHYS_BLOCKS}) regardless of "
            f"head/q blocking, got {mocked.call_count} calls."
        )

    def test_bhqkv_gathers_each_kv_block_once(self):
        heads = 4
        batch = 2
        cache = _make_paged_cache(heads=heads)
        module = _DummyModule()
        query = torch.randn(batch, heads, 2, HEAD_DIM)
        cache_kwargs = _cache_kwargs(torch.tensor([[6, 7], [6, 7]]), _identity_block_table(batch, NUM_PHYS_BLOCKS))

        with patch("QEfficient.blocking.blocked_attention_forwards._read_kv_block", wraps=_read_kv_block) as mocked:
            blocked_bhqkv_attention_forward(
                module=module,
                query=query,
                key=None,
                value=torch.zeros(batch, heads, 1, HEAD_DIM),
                attention_mask=None,
                scaling=SCALING,
                num_kv_blocks=NUM_PHYS_BLOCKS,
                num_q_blocks=2,
                num_batch_blocks=2,
                head_block_size=2,
                cache_kwargs=cache_kwargs,
                layer_idx=0,
                past_key_value=cache,
                paged_attention=True,
                use_causal_mask=True,
                skip_kv=False,
                ctx_len=CTX_LEN,
            )
        assert mocked.call_count == NUM_PHYS_BLOCKS, (
            f"Expected the KV gather to run once per KV block ({NUM_PHYS_BLOCKS}) regardless of "
            f"head/q/batch blocking, got {mocked.call_count} calls."
        )


# ---------------------------------------------------------------------------
# Tests: hoisted gather output is still numerically correct
# ---------------------------------------------------------------------------


@pytest.mark.cache
class TestHoistedGatherCorrectness:
    """The hoisted gather must still produce numerically identical output to a
    plain (unblocked) causal-attention computation over the same paged cache."""

    def test_qkv_paged_matches_manual_causal_attention(self):
        heads = 2
        cache = _make_paged_cache(heads=heads)
        module = _DummyModule()
        k_full, v_full = _full_kv(cache)

        query = torch.randn(1, heads, 2, HEAD_DIM)
        position_ids = torch.tensor([[6, 7]])
        cache_kwargs = _cache_kwargs(position_ids, _identity_block_table(1, NUM_PHYS_BLOCKS))

        attn_output, _ = blocked_qkv_attention_forward(
            module=module,
            query=query,
            key=None,
            value=torch.zeros(1, heads, 1, HEAD_DIM),
            attention_mask=None,
            scaling=SCALING,
            num_kv_blocks=NUM_PHYS_BLOCKS,
            num_q_blocks=2,
            cache_kwargs=cache_kwargs,
            layer_idx=0,
            past_key_value=cache,
            paged_attention=True,
            use_causal_mask=True,
            skip_kv=False,
            ctx_len=CTX_LEN,
        )

        expected = _manual_causal_attention(query, k_full, v_full, position_ids, SCALING)
        expected = expected.transpose(1, 2).contiguous()

        assert torch.allclose(attn_output, expected, atol=1e-4, rtol=1e-4), (
            "Hoisted-gather qkv output diverged from manual causal attention "
            f"(max abs diff={(attn_output - expected).abs().max().item()})"
        )

    def test_hqkv_paged_matches_manual_causal_attention(self):
        """Exercises the per-head slicing (k_block_states[:, h_start:h_end])
        that now reads from the hoisted/cached kv_blocks tuple instead of a
        freshly gathered k_block -- this is the line most likely to break if
        hoisting introduced a stale-block or index bug."""
        heads = 4
        cache = _make_paged_cache(heads=heads)
        module = _DummyModule()
        k_full, v_full = _full_kv(cache)

        query = torch.randn(1, heads, 3, HEAD_DIM)
        position_ids = torch.tensor([[5, 6, 7]])
        cache_kwargs = _cache_kwargs(position_ids, _identity_block_table(1, NUM_PHYS_BLOCKS))

        attn_output, _ = blocked_hqkv_attention_forward(
            module=module,
            query=query,
            key=None,
            value=torch.zeros(1, heads, 1, HEAD_DIM),
            attention_mask=None,
            scaling=SCALING,
            num_kv_blocks=NUM_PHYS_BLOCKS,
            num_q_blocks=2,
            head_block_size=2,
            cache_kwargs=cache_kwargs,
            layer_idx=0,
            past_key_value=cache,
            paged_attention=True,
            use_causal_mask=True,
            skip_kv=False,
            ctx_len=CTX_LEN,
        )

        expected = _manual_causal_attention(query, k_full, v_full, position_ids, SCALING)
        expected = expected.transpose(1, 2).contiguous()

        assert torch.allclose(attn_output, expected, atol=1e-4, rtol=1e-4), (
            "Hoisted-gather hqkv output diverged from manual causal attention "
            f"(max abs diff={(attn_output - expected).abs().max().item()})"
        )

    def test_bhqkv_paged_matches_manual_causal_attention_per_row(self):
        """Exercises both the per-head and per-batch slicing of the cached
        kv_blocks tuple, with two batch rows at *different* positions so a
        row/index mixup in the hoisted cache would show up as cross-row
        contamination rather than cancelling out."""
        heads = 4
        batch = 2
        cache = _make_paged_cache(heads=heads)
        module = _DummyModule()
        k_full, v_full = _full_kv(cache)

        query = torch.randn(batch, heads, 2, HEAD_DIM)
        position_ids = torch.tensor([[5, 6], [3, 4]])
        cache_kwargs = _cache_kwargs(position_ids, _identity_block_table(batch, NUM_PHYS_BLOCKS))

        attn_output, _ = blocked_bhqkv_attention_forward(
            module=module,
            query=query,
            key=None,
            value=torch.zeros(batch, heads, 1, HEAD_DIM),
            attention_mask=None,
            scaling=SCALING,
            num_kv_blocks=NUM_PHYS_BLOCKS,
            num_q_blocks=2,
            num_batch_blocks=2,
            head_block_size=2,
            cache_kwargs=cache_kwargs,
            layer_idx=0,
            past_key_value=cache,
            paged_attention=True,
            use_causal_mask=True,
            skip_kv=False,
            ctx_len=CTX_LEN,
        )

        expected = _manual_causal_attention(query, k_full, v_full, position_ids, SCALING)
        expected = expected.transpose(1, 2).contiguous()

        assert torch.allclose(attn_output, expected, atol=1e-4, rtol=1e-4), (
            "Hoisted-gather bhqkv output diverged from manual causal attention "
            f"(max abs diff={(attn_output - expected).abs().max().item()})"
        )

    def test_hqkv_paged_gqa_matches_manual_causal_attention(self):
        """GQA case (num_key_value_groups > 1): the hoisted pre-pass calls
        _get_kv_states (repeat_kv expansion) once per KV block, same as
        before hoisting -- verify that expansion is still correct when read
        back out of the cached kv_blocks tuple."""
        num_kv_heads = 2
        num_q_heads = 4
        cache = _make_paged_cache(heads=num_kv_heads)
        module = _DummyModule(num_key_value_groups=num_q_heads // num_kv_heads)
        k_full, v_full = _full_kv(cache)
        k_full_rep = repeat_kv(k_full.unsqueeze(0), module.num_key_value_groups).squeeze(0)
        v_full_rep = repeat_kv(v_full.unsqueeze(0), module.num_key_value_groups).squeeze(0)

        query = torch.randn(1, num_q_heads, 3, HEAD_DIM)
        position_ids = torch.tensor([[5, 6, 7]])
        cache_kwargs = _cache_kwargs(position_ids, _identity_block_table(1, NUM_PHYS_BLOCKS))

        attn_output, _ = blocked_hqkv_attention_forward(
            module=module,
            query=query,
            key=None,
            value=torch.zeros(1, num_q_heads, 1, HEAD_DIM),
            attention_mask=None,
            scaling=SCALING,
            num_kv_blocks=NUM_PHYS_BLOCKS,
            num_q_blocks=1,
            head_block_size=2,
            cache_kwargs=cache_kwargs,
            layer_idx=0,
            past_key_value=cache,
            paged_attention=True,
            use_causal_mask=True,
            skip_kv=False,
            ctx_len=CTX_LEN,
        )

        expected = _manual_causal_attention(query, k_full_rep, v_full_rep, position_ids, SCALING)
        expected = expected.transpose(1, 2).contiguous()

        assert torch.allclose(attn_output, expected, atol=1e-4, rtol=1e-4), (
            "Hoisted-gather GQA hqkv output diverged from manual causal attention "
            f"(max abs diff={(attn_output - expected).abs().max().item()})"
        )


# ---------------------------------------------------------------------------
# Tests: skip_kv early break is preserved by the hoisted pre-pass
# ---------------------------------------------------------------------------


@pytest.mark.cache
class TestSkipKvBreakPreserved:
    """The hoisted pre-pass must still break out of the KV-block loop as soon
    as a block lies entirely in the future, exactly as the un-hoisted loop did."""

    def test_hqkv_break_stops_gather_early(self):
        heads = 4
        cache = _make_paged_cache(heads=heads)
        module = _DummyModule()
        query = torch.randn(1, heads, 1, HEAD_DIM)
        # current position 3 falls in logical block 1 (start_index=2); block 2
        # (start_index=4) and block 3 (start_index=6) are entirely in the future.
        cache_kwargs = _cache_kwargs(torch.tensor([[3]]), _identity_block_table(1, NUM_PHYS_BLOCKS))

        with patch("QEfficient.blocking.blocked_attention_forwards._read_kv_block", wraps=_read_kv_block) as mocked:
            blocked_hqkv_attention_forward(
                module=module,
                query=query,
                key=None,
                value=torch.zeros(1, heads, 1, HEAD_DIM),
                attention_mask=None,
                scaling=SCALING,
                num_kv_blocks=NUM_PHYS_BLOCKS,
                num_q_blocks=1,
                head_block_size=2,
                cache_kwargs=cache_kwargs,
                layer_idx=0,
                past_key_value=cache,
                paged_attention=True,
                use_causal_mask=True,
                skip_kv=True,
                ctx_len=CTX_LEN,
            )
        expected_calls = 2  # blocks j=0,1 cover position 3; j=2 (start_index=4) is entirely future
        assert mocked.call_count == expected_calls, (
            f"Expected skip_kv to stop gathering after {expected_calls} KV blocks once a block "
            f"lies entirely in the future, got {mocked.call_count} calls."
        )

    def test_hqkv_skip_kv_matches_manual_causal_attention(self):
        """The call-count test above only proves the pre-pass gathers the right
        *number* of blocks once skip_kv breaks early -- it never checks that the
        surviving kv_blocks still produce numerically correct output. Verify
        that separately here: with skip_kv=True the blocks lying entirely in
        the future are never gathered at all (rather than gathered and then
        masked to -inf), so this must still match plain causal attention over
        the full cache."""
        heads = 4
        cache = _make_paged_cache(heads=heads)
        module = _DummyModule()
        k_full, v_full = _full_kv(cache)

        query = torch.randn(1, heads, 1, HEAD_DIM)
        position_ids = torch.tensor([[3]])
        cache_kwargs = _cache_kwargs(position_ids, _identity_block_table(1, NUM_PHYS_BLOCKS))

        attn_output, _ = blocked_hqkv_attention_forward(
            module=module,
            query=query,
            key=None,
            value=torch.zeros(1, heads, 1, HEAD_DIM),
            attention_mask=None,
            scaling=SCALING,
            num_kv_blocks=NUM_PHYS_BLOCKS,
            num_q_blocks=1,
            head_block_size=2,
            cache_kwargs=cache_kwargs,
            layer_idx=0,
            past_key_value=cache,
            paged_attention=True,
            use_causal_mask=True,
            skip_kv=True,
            ctx_len=CTX_LEN,
        )

        expected = _manual_causal_attention(query, k_full, v_full, position_ids, SCALING)
        expected = expected.transpose(1, 2).contiguous()

        assert torch.allclose(attn_output, expected, atol=1e-4, rtol=1e-4), (
            "Hoisted-gather hqkv output with skip_kv=True diverged from manual causal attention "
            f"(max abs diff={(attn_output - expected).abs().max().item()})"
        )
