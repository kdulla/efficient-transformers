# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------


import argparse

from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

from QEfficient import QEFFAutoModelForCausalLM


def main():
    parser = argparse.ArgumentParser(description="MoE model inference")
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen3.8-2.4T-A95B",
        help="HuggingFace MoE model ID",
    )
    parser.add_argument("--prompt", type=str, default="Explain quantum computing", help="Input prompt")
    parser.add_argument("--prefill-seq-len", type=int, default=32, help="Prefill sequence length")
    parser.add_argument("--ctx-len", type=int, default=4096, help="Context length")
    parser.add_argument("--generation-len", type=int, default=None, help="Number of tokens to generate")
    parser.add_argument("--num-cores", type=int, default=16, help="Number of cores")
    parser.add_argument("--num-devices", type=int, default=4, help="Number of devices")
    args = parser.parse_args()

    print(f"Skipping Load of MoE model: {args.model_name}")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    config = AutoConfig.from_pretrained(args.model_name)
    config.num_hidden_layers = 4
    hf_model = AutoModelForCausalLM.from_config(config)
    model = QEFFAutoModelForCausalLM(hf_model)

    # Compile the model
    qpc_path = model.compile(
        prefill_seq_len=args.prefill_seq_len,
        ctx_len=args.ctx_len,
        num_cores=args.num_cores,
        num_devices=args.num_devices,
        mxfp6_matmul=True,
        mxint8_kv_cache=True,
        aic_enable_depth_first=False,
        mos=1,
        split_model_io=True,
        use_onnx_subfunctions=True,
    )
    print(f"Model compiled to: {qpc_path}")

    # Generate text
    exec_info = model.generate(
        tokenizer=tokenizer,
        prompts=[args.prompt],
        device_id=args.device_group,
        generation_len=args.generation_len,
    )

    print(f"\nPrompt: {args.prompt}")
    print(f"Generated: {exec_info.generated_texts[0]}")


if __name__ == "__main__":
    main()
