# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

__compat__ = "sglang>=0.5.10"

import pkg_resources
import torch

from collector.helper import benchmark_with_power, log_perf
from collector.sglang.wan_common import (
    SingleRankGroup,
    WAN_CLIP_HEAD_DIM,
    WAN_CLIP_HIDDEN_SIZE,
    WAN_CLIP_IMAGE_SIZE,
    WAN_CLIP_INTERMEDIATE_SIZE,
    WAN_CLIP_NUM_CHANNELS,
    WAN_CLIP_NUM_HEADS,
    WAN_CLIP_PATCH_SIZE,
)


def get_wan_clip_test_cases():
    seq_len = (WAN_CLIP_IMAGE_SIZE // WAN_CLIP_PATCH_SIZE) ** 2 + 1
    return [
        ["vision_patch_embed", 1, seq_len, WAN_CLIP_HIDDEN_SIZE, WAN_CLIP_NUM_HEADS, WAN_CLIP_HEAD_DIM, WAN_CLIP_INTERMEDIATE_SIZE],
        ["layernorm", 1, seq_len, WAN_CLIP_HIDDEN_SIZE, WAN_CLIP_NUM_HEADS, WAN_CLIP_HEAD_DIM, WAN_CLIP_INTERMEDIATE_SIZE],
        ["attention_compute", 1, seq_len, WAN_CLIP_HIDDEN_SIZE, WAN_CLIP_NUM_HEADS, WAN_CLIP_HEAD_DIM, WAN_CLIP_INTERMEDIATE_SIZE],
        ["mlp_activation", 1, seq_len, WAN_CLIP_HIDDEN_SIZE, WAN_CLIP_NUM_HEADS, WAN_CLIP_HEAD_DIM, WAN_CLIP_INTERMEDIATE_SIZE],
    ]


def run_wan_clip(
    op_name,
    batch_size,
    seq_len,
    hidden_size,
    num_heads,
    head_dim,
    intermediate_size,
    *,
    perf_filename,
    device="cuda:0",
):
    torch.cuda.set_device(device)
    torch.set_default_dtype(torch.bfloat16)
    device_obj = torch.device(device)

    if op_name == "vision_patch_embed":
        from sglang.multimodal_gen.runtime.models.encoders.clip import CLIPVisionEmbeddings
        from sglang.multimodal_gen.configs.models.encoders import CLIPVisionConfig

        config = CLIPVisionConfig().arch_config
        module = CLIPVisionEmbeddings(config).to(device_obj, dtype=torch.bfloat16)
        pixel_values = torch.randn(
            batch_size,
            WAN_CLIP_NUM_CHANNELS,
            WAN_CLIP_IMAGE_SIZE,
            WAN_CLIP_IMAGE_SIZE,
            dtype=torch.bfloat16,
            device=device_obj,
        )

        def kernel_func():
            return module(pixel_values)

        kernel_source = "sglang_clip_vision_embeddings"
    elif op_name == "layernorm":
        module = torch.nn.LayerNorm(hidden_size, eps=1e-5).to(device_obj, dtype=torch.bfloat16)
        hidden_states = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.bfloat16, device=device_obj)

        def kernel_func():
            return module(hidden_states)

        kernel_source = "torch_layernorm"
    elif op_name == "attention_compute":
        q = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.bfloat16, device=device_obj)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        def kernel_func():
            return torch.nn.functional.scaled_dot_product_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                is_causal=True,
                scale=head_dim**-0.5,
            ).transpose(1, 2)

        kernel_source = "torch_sdpa"
    elif op_name == "mlp_activation":
        from sglang.multimodal_gen.runtime.layers.activation import get_act_fn

        activation = get_act_fn("quick_gelu")
        hidden_states = torch.randn(batch_size, seq_len, intermediate_size, dtype=torch.bfloat16, device=device_obj)

        def kernel_func():
            return activation(hidden_states)

        kernel_source = "sglang_quick_gelu"
    else:
        raise ValueError(f"Unsupported op_name: {op_name}")

    with benchmark_with_power(
        device=device_obj,
        kernel_func=kernel_func,
        num_warmups=3,
        num_runs=10,
        repeat_n=1,
        allow_graph_fail=True,
    ) as results:
        pass

    log_perf(
        item_list=[
            {
                "batch_size": batch_size,
                "seq_len": seq_len,
                "hidden_size": hidden_size,
                "num_heads": num_heads,
                "head_dim": head_dim,
                "intermediate_size": intermediate_size,
                "latency": results["latency_ms"],
            }
        ],
        framework="SGLang",
        version=pkg_resources.get_distribution("sglang").version,
        device_name=torch.cuda.get_device_name(device_obj),
        op_name=f"wan_clip_{op_name}",
        kernel_source=kernel_source,
        perf_filename=perf_filename,
        power_stats=results["power_stats"],
    )


if __name__ == "__main__":
    from collector.registry_types import PerfFile

    for test_case in get_wan_clip_test_cases():
        run_wan_clip(*test_case, perf_filename=PerfFile.WAN_CLIP)
