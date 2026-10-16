# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
"""Compile and run Qwen3-30B-A3B with paged-GQA prefill and decode."""

import argparse
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer

from QEfficient import QEFFAutoModelForCausalLM
from QEfficient.generation.cloud_infer import QAICInferenceSession

MODEL_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507"


def _device_ids(value: str) -> list[int]:
    return [int(device_id) for device_id in value.strip("[]").split(",") if device_id]


def _qpc_path(result: Any, key: str) -> Path:
    if isinstance(result, dict):
        return Path(result[key])
    return Path(result)


def _block_table(
    dp: int,
    batch_size: int,
    num_logical_pages: int,
    cp: int,
    physical_pages: int | None,
) -> np.ndarray:
    batch_local = batch_size // dp
    page_groups = -(-num_logical_pages // cp)
    required_pages = dp * batch_local * page_groups
    if physical_pages is not None and physical_pages < required_pages:
        raise ValueError(f"Physical page capacity {physical_pages} is smaller than the required {required_pages}.")
    return np.arange(required_pages, dtype=np.int64).reshape(dp, batch_local, page_groups)


def _pad_prompt_for_prefill(
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
    prefill_seq_len: int,
    pad_token_id: int,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Right-pad a prompt to complete prefill chunks and mark padding positions invalid."""
    prompt_len = int(attention_mask.sum(axis=1).max())
    padded_len = -(-prompt_len // prefill_seq_len) * prefill_seq_len
    pad_len = padded_len - input_ids.shape[1]
    padded_input_ids = np.pad(input_ids, ((0, 0), (0, pad_len)), constant_values=pad_token_id)
    padded_attention_mask = np.pad(attention_mask, ((0, 0), (0, pad_len)), constant_values=0)
    position_ids = np.where(padded_attention_mask, np.arange(padded_len, dtype=np.int64), -1)
    return padded_input_ids, position_ids, prompt_len, padded_len


def _update_retained_inputs(state: dict[str, np.ndarray], output: dict[str, np.ndarray], layers: int) -> None:
    for layer_idx in range(layers):
        for prefix in ("past_key", "past_value"):
            name = f"{prefix}.{layer_idx}"
            retained_name = f"{name}_RetainedState"
            if retained_name in output:
                state[name] = output[retained_name]


def _binding_shape(session: QAICInferenceSession, name: str) -> tuple[int, ...]:
    return tuple(int(dim) for dim in session.bindings[session.binding_index_map[name]].dims)


def _repack_paged_gqa_cache(
    cache: np.ndarray,
    prefill_table: np.ndarray,
    decode_table: np.ndarray,
    decode_shape: tuple[int, ...],
    dp: int,
    cp: int,
) -> np.ndarray:
    """Convert a DP=CP=1 paged cache into the decode [DP, CP, Hkv] row layout."""
    if cache.ndim != 4 or len(decode_shape) != 4:
        raise ValueError("Paged GQA retained caches must be rank four.")
    if prefill_table.ndim != 3 or prefill_table.shape[0] != 1:
        raise ValueError("Prefill block_table must have shape [1, batch_size, num_pages].")
    if decode_table.ndim != 3 or decode_table.shape[0] != dp:
        raise ValueError("Decode block_table must have shape [attn_dp, batch_local, page_groups].")

    batch_size = prefill_table.shape[1]
    if batch_size % dp or decode_table.shape[1] != batch_size // dp:
        raise ValueError("Prefill and decode block tables are inconsistent with attn_dp.")
    num_kv_heads = cache.shape[1]
    if decode_shape[1] != dp * cp * num_kv_heads or tuple(decode_shape[2:]) != tuple(cache.shape[2:]):
        raise ValueError("Decode cache shape is incompatible with the requested DP/CP layout.")

    result = np.zeros(decode_shape, dtype=cache.dtype)
    batch_local = batch_size // dp
    for batch_idx in range(batch_size):
        dp_idx, local_idx = divmod(batch_idx, batch_local)
        for logical_page in range(prefill_table.shape[2]):
            source_page = int(prefill_table[0, batch_idx, logical_page])
            page_group, cp_idx = divmod(logical_page, cp)
            destination_page = int(decode_table[dp_idx, local_idx, page_group])
            row_start = (dp_idx * cp + cp_idx) * num_kv_heads
            if not 0 <= source_page < cache.shape[0]:
                raise ValueError(f"Prefill physical page {source_page} is outside the retained cache pool.")
            if not 0 <= destination_page < decode_shape[0]:
                raise ValueError(f"Decode physical page {destination_page} is outside the retained cache pool.")
            result[destination_page, row_start : row_start + num_kv_heads] = cache[source_page]
    return result


def _repack_retained_inputs(
    prefill_state: dict[str, np.ndarray],
    decode_inputs: dict[str, np.ndarray],
    decode_session: QAICInferenceSession,
    prefill_table: np.ndarray,
    decode_table: np.ndarray,
    layers: int,
    dp: int,
    cp: int,
) -> None:
    for layer_idx in range(layers):
        for prefix in ("past_key", "past_value"):
            name = f"{prefix}.{layer_idx}"
            if name not in prefill_state:
                raise ValueError(f"Prefill output did not return retained cache input {name}.")
            if name not in decode_session.binding_index_map:
                raise ValueError(f"Decode QPC does not expose retained cache input {name}.")
            decode_inputs[name] = _repack_paged_gqa_cache(
                prefill_state[name],
                prefill_table,
                decode_table,
                _binding_shape(decode_session, name),
                dp,
                cp,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--prompt", default="Explain paged attention in three concise paragraphs.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--ctx-len", type=int, default=32768)
    parser.add_argument("--generation-len", type=int, default=64)
    parser.add_argument(
        "--decode-only",
        action="store_true",
        help="Compile only decode and consume the prompt token-by-token through the decode QPC.",
    )
    parser.add_argument(
        "--prefill-seq-len",
        type=int,
        default=None,
        help="Paged prefill chunk length; defaults to --page-block-size.",
    )
    parser.add_argument(
        "--num-hidden-layers",
        type=int,
        default=None,
        help="Compile only this many transformer layers; defaults to the full model.",
    )
    parser.add_argument("--num-kv-blocks", type=int, default=4)
    parser.add_argument("--page-block-size", type=int, default=128)
    parser.add_argument(
        "--gqa-num-pages",
        type=int,
        default=None,
        help="Logical pages per sequence; defaults to ctx_len / page_block_size.",
    )
    parser.add_argument(
        "--gqa-physical-pages",
        type=int,
        default=None,
        help="Physical page capacity of the paged GQA decode KV pool.",
    )
    parser.add_argument("--attn-dp", type=int, default=1)
    parser.add_argument("--attn-cp", type=int, default=1)
    parser.add_argument("--num-cores", type=int, default=16)
    parser.add_argument(
        "--expert-parallel-chunk-size",
        type=int,
        default=256,
        help="MoE expert-parallel chunk size (expert_parallel_chunk_size in moe_config).",
    )
    parser.add_argument(
        "--cores-per-expert",
        type=int,
        default=2,
        help="Number of NSP cores assigned to each expert during decode.",
    )
    parser.add_argument(
        "--no-tree-reduce",
        dest="tree_reduce",
        action="store_false",
        default=True,
        help="Disable tree-reduce for MoE expert-parallel dispatch.",
    )
    parser.add_argument("--device-group", type=_device_ids, default=[0])
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/qwen3-30b-a3b-paged-gqa"))
    args = parser.parse_args()

    if "HF_HUB_CACHE" not in os.environ:
        parser.error("Set HF_HUB_CACHE before loading the model.")
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    prefill_seq_len = args.page_block_size if args.prefill_seq_len is None else args.prefill_seq_len
    if args.batch_size < 1 or args.attn_dp < 1 or args.attn_cp < 1:
        parser.error("--batch-size, --attn-dp, and --attn-cp must be positive")
    if args.batch_size % args.attn_dp:
        parser.error("--batch-size must be divisible by --attn-dp")
    if args.num_hidden_layers is not None and args.num_hidden_layers < 1:
        parser.error("--num-hidden-layers must be positive")
    if args.ctx_len % args.num_kv_blocks:
        parser.error("--ctx-len must be divisible by --num-kv-blocks")
    if args.ctx_len % (args.attn_cp * args.page_block_size):
        parser.error("--ctx-len must be divisible by --attn-cp * --page-block-size")
    if not args.decode_only and (
        prefill_seq_len <= 1 or prefill_seq_len > args.ctx_len or prefill_seq_len % args.page_block_size
    ):
        parser.error("--prefill-seq-len must be > 1, <= --ctx-len, and divisible by --page-block-size")
    if args.generation_len < 1:
        parser.error("--generation-len must be positive")
    if args.expert_parallel_chunk_size < 1:
        parser.error("--expert-parallel-chunk-size must be positive")
    if args.cores_per_expert < 1:
        parser.error("--cores-per-expert must be positive")

    minimum_gqa_pages = args.ctx_len // args.page_block_size
    gqa_num_pages = minimum_gqa_pages if args.gqa_num_pages is None else args.gqa_num_pages
    if gqa_num_pages < 1:
        parser.error("--gqa-num-pages must be positive")
    if gqa_num_pages * args.page_block_size < args.ctx_len:
        parser.error("GQA logical-page capacity must be at least --ctx-len")
    gqa_page_divisor = args.num_kv_blocks * args.attn_cp
    if gqa_num_pages % gqa_page_divisor:
        parser.error("--gqa-num-pages must be divisible by --num-kv-blocks times --attn-cp")
    if (gqa_num_pages // gqa_page_divisor) % args.num_cores:
        parser.error("GQA page groups per KV block must be divisible by --num-cores")
    required_decode_physical_pages = args.batch_size * -(-gqa_num_pages // args.attn_cp)
    if args.gqa_physical_pages is not None and args.gqa_physical_pages < required_decode_physical_pages:
        parser.error(f"--gqa-physical-pages must be at least {required_decode_physical_pages}")

    logical_kv_block_size = args.ctx_len // args.num_kv_blocks
    if logical_kv_block_size % (args.attn_cp * args.page_block_size):
        parser.error("ctx_len / num_kv_blocks must be divisible by --attn-cp * --page-block-size")
    pages_per_decode_kv_block = logical_kv_block_size // args.attn_cp // args.page_block_size
    if pages_per_decode_kv_block % args.num_cores:
        parser.error("ctx_len / num_kv_blocks / attn_cp / page_block_size must be divisible by --num-cores")

    decode_qaic_config = {
        "blocking_mode": "kv_headpar_paged",
        "num_kv_blocks": args.num_kv_blocks,
        "headpar_split": args.num_cores,
        "attn_dp": args.attn_dp,
        "attn_cp": args.attn_cp,
        "page_block_size": args.page_block_size,
        "gqa_page_block_size": args.page_block_size,
        "gqa_num_logical_pages": gqa_num_pages,
        "gqa_physical_pages": args.gqa_physical_pages,
        "num_cores_per_device": args.num_cores,
        "skip_kv": True,
        "moe_config": {
            "flavour": "expert_parallel",
            "expert_parallel_chunk_size": args.expert_parallel_chunk_size,
            "cores_per_expert": args.cores_per_expert,
            "tree_reduce": args.tree_reduce,
        },
    }
    prefill_physical_pages = args.batch_size * gqa_num_pages
    prefill_qaic_config = {
        **decode_qaic_config,
        "attn_dp": 1,
        "attn_cp": 1,
        "gqa_physical_pages": prefill_physical_pages,
    }

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    prompt_inputs = tokenizer([args.prompt] * args.batch_size, return_tensors="np", padding=True)
    attention_mask = prompt_inputs.pop("attention_mask")
    prompt_len = int(attention_mask.sum(axis=1).max())
    prompt_inputs.pop("token_type_ids", None)
    if prompt_len == 0:
        parser.error("The prompt must contain at least one token")
    if prompt_len + args.generation_len > args.ctx_len:
        parser.error("Prompt length plus generation length exceeds --ctx-len")
    padded_prefill_len = 0
    if not args.decode_only:
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
        prompt_inputs["input_ids"], prompt_inputs["position_ids"], prompt_len, padded_prefill_len = (
            _pad_prompt_for_prefill(
                prompt_inputs["input_ids"],
                attention_mask,
                prefill_seq_len,
                pad_token_id,
            )
        )
        if padded_prefill_len > args.ctx_len:
            parser.error(f"Padded prompt length ({padded_prefill_len}) exceeds --ctx-len ({args.ctx_len}).")

    model_config = AutoConfig.from_pretrained(args.model_id)
    if args.num_hidden_layers is not None:
        model_config.num_hidden_layers = args.num_hidden_layers
    layer_count = model_config.num_hidden_layers
    model = QEFFAutoModelForCausalLM.from_pretrained(
        args.model_id,
        config=model_config,
        dtype=torch.float16,
    )
    compile_kwargs = {
        "batch_size": args.batch_size,
        "ctx_len": args.ctx_len,
        "num_cores": args.num_cores,
        "num_devices": len(args.device_group),
        "mxfp6_matmul": True,
        "use_onnx_subfunctions": True,
        "user_tiled": True,
        "retain_full_kv": True,
    }

    prefill_qpc_path = None
    if not args.decode_only:
        print("Compiling paged prefill QPC with DP=CP=1...")
        prefill_result = model.compile(
            compile_dir=args.output_dir / "prefill",
            prefill_seq_len=prefill_seq_len,
            prefill_only=True,
            enable_chunking=True,
            offload_pt_weights=False,
            qaic_config=prefill_qaic_config,
            **compile_kwargs,
        )
        prefill_qpc_path = _qpc_path(prefill_result, "lang_prefill_qpc_path")
        print(f"Prefill QPC: {prefill_qpc_path}")

    print("Compiling decode QPC with requested DP/CP...")
    decode_result = model.compile(
        compile_dir=args.output_dir if args.decode_only else args.output_dir / "decode",
        prefill_seq_len=1,
        prefill_only=False,
        qaic_config=decode_qaic_config,
        **compile_kwargs,
    )
    decode_qpc_path = _qpc_path(decode_result, "lang_decode_qpc_path")
    print(f"Decode QPC: {decode_qpc_path}")

    decode_table = _block_table(
        args.attn_dp,
        args.batch_size,
        gqa_num_pages,
        args.attn_cp,
        args.gqa_physical_pages,
    )
    prefill_table = None
    prefill_state: dict[str, np.ndarray] = {}
    prompt_output = None
    prompt_start = time.perf_counter()
    prefill_seconds = 0.0

    if not args.decode_only:
        assert prefill_qpc_path is not None
        prefill_table = _block_table(1, args.batch_size, gqa_num_pages, 1, prefill_physical_pages)
        prefill_state["block_table"] = prefill_table
        prefill_session = QAICInferenceSession(prefill_qpc_path, args.device_group)
        if prefill_table.shape != _binding_shape(prefill_session, "block_table"):
            raise ValueError(
                f"Prefill block table shape {prefill_table.shape} does not match "
                f"QPC shape {_binding_shape(prefill_session, 'block_table')}."
            )
        try:
            prefill_start = time.perf_counter()
            for start in range(0, padded_prefill_len, prefill_seq_len):
                prefill_state["input_ids"] = prompt_inputs["input_ids"][:, start : start + prefill_seq_len]
                prefill_state["position_ids"] = prompt_inputs["position_ids"][:, start : start + prefill_seq_len]
                prompt_output = prefill_session.run(prefill_state)
                _update_retained_inputs(prefill_state, prompt_output, layer_count)
            prefill_seconds = time.perf_counter() - prefill_start
        finally:
            prefill_session.deactivate()
        prompt_decode_start = prompt_len
    else:
        prompt_decode_start = 0

    decode_session = QAICInferenceSession(decode_qpc_path, args.device_group)
    if decode_table.shape != _binding_shape(decode_session, "block_table"):
        raise ValueError(
            f"Decode block table shape {decode_table.shape} does not match "
            f"QPC shape {_binding_shape(decode_session, 'block_table')}."
        )
    decode_inputs: dict[str, np.ndarray] = {"block_table": decode_table}
    if prefill_table is not None:
        _repack_retained_inputs(
            prefill_state,
            decode_inputs,
            decode_session,
            prefill_table,
            decode_table,
            layer_count,
            args.attn_dp,
            args.attn_cp,
        )

    try:
        for position in range(prompt_decode_start, prompt_len):
            decode_inputs["input_ids"] = prompt_inputs["input_ids"][:, position : position + 1]
            decode_inputs["position_ids"] = np.full((args.batch_size, 1), position, dtype=np.int64)
            prompt_output = decode_session.run(decode_inputs)
            _update_retained_inputs(decode_inputs, prompt_output, layer_count)
        prompt_seconds = time.perf_counter() - prompt_start
        assert prompt_output is not None

        next_tokens = np.argmax(prompt_output["logits"][:, -1, :], axis=-1, keepdims=True).astype(np.int64)
        decode_inputs["input_ids"] = next_tokens
        decode_inputs["position_ids"] = np.full((args.batch_size, 1), prompt_len, dtype=np.int64)

        generated = [next_tokens]
        decode_start = time.perf_counter()
        for _ in range(args.generation_len - 1):
            decode_output = decode_session.run(decode_inputs)
            next_tokens = np.argmax(decode_output["logits"][:, -1, :], axis=-1, keepdims=True).astype(np.int64)
            generated.append(next_tokens)
            decode_inputs["input_ids"] = next_tokens
            decode_inputs["position_ids"] += 1
            _update_retained_inputs(decode_inputs, decode_output, layer_count)
        decode_seconds = time.perf_counter() - decode_start
    finally:
        decode_session.deactivate()

    generated_ids = np.concatenate(generated, axis=1)
    for index, generated_text in enumerate(tokenizer.batch_decode(generated_ids, skip_special_tokens=True)):
        print(f"\n[{index}] {generated_text}")
    if args.decode_only:
        print(f"\nPrompt decode: {prompt_seconds:.3f}s")
    else:
        print(f"\nPrefill: {prefill_seconds:.3f}s")
        print(f"TTFT: {prompt_seconds:.3f}s")
    decode_tokens = (args.generation_len - 1) * args.batch_size
    if decode_tokens:
        print(f"Decode: {decode_tokens / decode_seconds:.2f} token/s")


if __name__ == "__main__":
    main()
