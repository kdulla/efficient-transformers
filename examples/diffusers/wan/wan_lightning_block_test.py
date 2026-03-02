# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
import safetensors.torch
import torch
import os
from diffusers.loaders.lora_conversion_utils import _convert_non_diffusers_wan_lora_to_diffusers
from diffusers.utils import export_to_video
from huggingface_hub import hf_hub_download

from QEfficient import QEffWanPipeline

# Load the pipeline

pipeline = QEffWanPipeline.from_pretrained("Wan-AI/Wan2.2-T2V-A14B-Diffusers")

# Download the LoRAs
high_noise_lora_path = hf_hub_download(
    repo_id="lightx2v/Wan2.2-Lightning",
    filename="Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1/high_noise_model.safetensors",
)
low_noise_lora_path = hf_hub_download(
    repo_id="lightx2v/Wan2.2-Lightning",
    filename="Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1/low_noise_model.safetensors",
)


# LoRA conversion
def load_wan_lora(path: str):
    return _convert_non_diffusers_wan_lora_to_diffusers(safetensors.torch.load_file(path))


# Load into the transformers
pipeline.transformer.model.transformer_high.load_lora_adapter(
    load_wan_lora(high_noise_lora_path), adapter_name="high_noise"
)
pipeline.transformer.model.transformer_high.set_adapters(["high_noise"], weights=[1.0])
pipeline.transformer.model.transformer_low.load_lora_adapter(
    load_wan_lora(low_noise_lora_path), adapter_name="low_noise"
)
pipeline.transformer.model.transformer_low.set_adapters(["low_noise"], weights=[1.0])


prompt = "In a warmly lit living room, an elderly man with gray hair sits in a wooden armchair adorned with a blue cushion. He wears a gray cardigan over a white shirt, engrossed in reading a book. As he turns the pages, he subtly adjusts his posture, ensuring his glasses stay in place. He then removes his glasses, holding them in his hand, and turns his head to the right, maintaining his grip on the book. The soft glow of a bedside lamp bathes the scene, creating a calm and serene atmosphere, with gentle shadows enhancing the intimate setting."

# blocking variables
os.environ["ATTENTION_BLOCKING_MODE"] = "qkv"
os.environ["head_block_size"] = "1"
os.environ["num_kv_blocks"] = "16"
os.environ["num_q_blocks"] = "2"
os.environ["skip_threshold"] = "100.0"

block_configs = [[1, 8, 1, 0.1],
                 [1, 4, 1, 0.1],
                 [1, 8, 1, 0.001],
                 [1, 4, 1, 0.001]]
                #  [1, 24, 2, 0.5],
                #  [1, 16, 1, 0.5],
                #  [1, 24, 2, 10.0],
                #  [1, 16, 1, 10.0],
                #  [1, 24, 2, 1000.0],
                #  [1, 16, 1, 1000.0]]

for config in block_configs:
    os.environ["head_block_size"] = str(config[0])
    os.environ["num_kv_blocks"] = str(config[1])
    os.environ["num_q_blocks"] = str(config[2])
    os.environ["skip_threshold"] = str(config[3])

    os.environ["num_blocks_total"] = "0"
    os.environ["num_blocks_skipped"] = "0"

    output = pipeline(
        prompt=prompt,
        num_frames=29,
        guidance_scale=1.0,
        guidance_scale_2=1.0,
        num_inference_steps=4,
        generator=torch.manual_seed(0),
        custom_config_path="examples/diffusers/wan/wan_config.json",
        height=256,
        width=384,
        run_on_gpu=True,
    )
    frames = output.images[0]
    # export_to_video(frames, "output_t2v.mp4", fps=16)
    
    print(f"Config: {config[0]} head block size, {config[1]} kv blocks, {config[2]} q blocks, {config[3]} skip threshold")

    print(output)

    if int(os.environ.get("num_blocks_total", 0)) != 0:
        print(f"Ratio of blocks skipped for {config[1]} kv blocks and skip threshold {config[3]} is {int(os.environ.get('num_blocks_skipped', 0)) / int(os.environ.get('num_blocks_total', 0))}")
