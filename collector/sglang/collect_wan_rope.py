# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

__compat__ = "sglang>=0.5.10"

import pkg_resources
import torch

from collector.helper import benchmark_with_power, log_perf
from collector.sglang.wan_common import (
    iter_wan_video_cases,
    local_attention_shape,
    valid_parallel_cases,
)


def get_wan_rope_test_cases():
    test_cases = []
    seen = set()
    for profile, num_frames, height, width in iter_wan_video_cases():
        from collector.sglang.wan_common import seq_len_from_video

        global_seq_len = seq_len_from_video(
            num_frames,
            height,
            width,
            latent_stride=profile.latent_prepare_stride,
        )
        for tp_size, sp_size, ulysses_degree, ring_degree, sp_algorithm in valid_parallel_cases(profile.num_heads):
            seq_len, num_heads = local_attention_shape(
                global_seq_len=global_seq_len,
                tp_size=tp_size,
                sp_size=sp_size,
                ulysses_degree=ulysses_degree,
                ring_degree=ring_degree,
                num_heads=profile.num_heads,
            )
            key = (1, seq_len, num_heads, profile.head_dim, tp_size, sp_size, ulysses_degree, ring_degree, sp_algorithm)
            if num_heads <= 0 or key in seen:
                continue
            seen.add(key)
            test_cases.append(list(key))
    return test_cases


def run_wan_rope(
    batch_size,
    seq_len,
    num_heads,
    head_dim,
    tp_size,
    sp_size,
    ulysses_degree,
    ring_degree,
    sp_algorithm,
    *,
    perf_filename,
    device="cuda:0",
):
    torch.cuda.set_device(device)
    device_obj = torch.device(device)
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.bfloat16, device=device_obj)
    k = torch.randn_like(q)
    cos = torch.randn(seq_len, head_dim // 2, dtype=torch.float32, device=device_obj)
    sin = torch.randn(seq_len, head_dim // 2, dtype=torch.float32, device=device_obj)
    cos_sin_cache = torch.cat([cos, sin], dim=-1).contiguous()

    from sglang.multimodal_gen.runtime.layers.rotary_embedding import apply_flashinfer_rope_qk_inplace

    kernel_source = "flashinfer_rope"

    def kernel_func():
        apply_flashinfer_rope_qk_inplace(q, k, cos_sin_cache, is_neox=False)

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
                "num_heads": num_heads,
                "head_dim": head_dim,
                "tp_size": tp_size,
                "sp_size": sp_size,
                "ulysses_degree": ulysses_degree,
                "ring_degree": ring_degree,
                "sp_algorithm": sp_algorithm,
                "latency": results["latency_ms"],
            }
        ],
        framework="SGLang",
        version=pkg_resources.get_distribution("sglang").version,
        device_name=torch.cuda.get_device_name(device_obj),
        op_name="wan_rope",
        kernel_source=kernel_source,
        perf_filename=perf_filename,
        power_stats=results["power_stats"],
    )


if __name__ == "__main__":
    from collector.registry_types import PerfFile

    for test_case in get_wan_rope_test_cases():
        run_wan_rope(*test_case, perf_filename=PerfFile.WAN_ROPE)
