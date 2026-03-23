# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import transformers
from transformers import AutoConfig, AutoProcessor, TextStreamer

from QEfficient import QEFFAutoModelForImageTextToText

model_id = "Qwen/Qwen3-VL-32B-Instruct"
config = AutoConfig.from_pretrained(model_id)
config.vision_config.depth = 9
config.text_config.num_hidden_layers = 1
config.vision_config.deepstack_visual_indexes = [8]

HEAD_BLOCK_SIZE = 8
NUM_KV_BLOCKS = 2
NUM_Q_BLOCKS = 2
NUM_BATCH_BLOCKS = 2
# head qkv blocking
qaic_config = dict(
    enable_blocking=True,
    blocking_mode="bhqkv",
    head_block_size=HEAD_BLOCK_SIZE,
    num_kv_blocks=NUM_KV_BLOCKS,
    num_q_blocks=NUM_Q_BLOCKS,
    num_batch_blocks=NUM_BATCH_BLOCKS,
)

qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
    model_id,
    attn_implementation="eager",
    kv_offload=True,
    config=config,
    continuous_batching=True,
    qaic_config=qaic_config,
)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id)

batch_size = 1

qeff_model.transform(
    ctx_len=4096, seq_len=1, batch_size=4, num_devices=1, qaic_config=qaic_config
)

## Vision + Text ##
qeff_model.compile(
    batch_size=batch_size,
    full_batch_size=4,
    prefill_seq_len=1,
    ctx_len=4096,
    num_cores=16,
    num_devices=4,
    height=354,
    width=536,
    mxfp6_matmul=True,
    mxint8_kv_cache=True,
    aic_enable_depth_first=True,
    mos=1,
    qaic_config=qaic_config,
)

image_urls = [
    "https://picsum.photos/id/237/536/354",
    "https://picsum.photos/id/237/536/354",
    "https://picsum.photos/id/237/536/354",
    "https://picsum.photos/id/237/536/354",
]

prompts = [
    "Can you describe the image in detail?",
    "What are the objects in the image?",
    "What is the main subject of the image?",
    "What colors are predominant in the image?",
]

streamer = TextStreamer(tokenizer)
output = qeff_model.generate(
    tokenizer=tokenizer,
    prompts=prompts,
    processor=processor,
    images=image_urls,
    generation_len=100,
)
print(output.generated_ids)
print(tokenizer.batch_decode(output.generated_ids))
print(output)
