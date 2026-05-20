# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

__compat__ = "sglang>=0.5.10"

import pkg_resources
import torch

from collector.helper import benchmark_with_power, log_perf
from collector.sglang.wan_common import iter_wan_video_cases, seq_len_from_video, sequence_shard_len, valid_parallel_cases


def get_wan_elementwise_test_cases():
    ops = [
        "fp32_layernorm_modulation",
        "rmsnorm_qk",
        "scale_residual_layernorm_scale_shift",
        "scale_residual",
    ]
    tp_list = [1, 2, 4, 8]
    video_cases = list(iter_wan_video_cases())
    test_cases = []
    seen = set()
    for op_name in ops:
        for profile, frames, height, width in video_cases:
            global_seq_len = seq_len_from_video(
                frames,
                height,
                width,
                latent_stride=profile.latent_prepare_stride,
            )
            sp_sizes = {sp_size for _, sp_size, _, _, _ in valid_parallel_cases(profile.num_heads)}
            for tp_size in tp_list:
                if profile.hidden_size % tp_size != 0:
                    continue
                hidden_size = profile.hidden_size // tp_size if op_name == "rmsnorm_qk" else profile.hidden_size
                for sp_size in sp_sizes:
                    seq_len = sequence_shard_len(global_seq_len, sp_size)
                    key = (op_name, 1, seq_len, hidden_size, tp_size)
                    if key in seen:
                        continue
                    seen.add(key)
                    test_cases.append(list(key))
    return test_cases


def run_wan_elementwise(
    op_name,
    batch_size,
    seq_len,
    hidden_size,
    tp_size,
    *,
    perf_filename,
    device="cuda:0",
):
    torch.cuda.set_device(device)
    device_obj = torch.device(device)
    x = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.bfloat16, device=device_obj)

    if op_name == "fp32_layernorm_modulation":
        from sglang.multimodal_gen.runtime.layers.layernorm import LayerNormScaleShift

        norm = LayerNormScaleShift(
            hidden_size,
            eps=1e-6,
            elementwise_affine=False,
            dtype=torch.float32,
        ).to(device_obj)
        shift = torch.randn(batch_size, 1, hidden_size, dtype=torch.float32, device=device_obj)
        scale = torch.randn(batch_size, 1, hidden_size, dtype=torch.float32, device=device_obj)

        def kernel_func():
            return norm(x, shift, scale)

        kernel_source = "sglang_layernorm_scale_shift"
    elif op_name == "rmsnorm_qk":
        from sglang.multimodal_gen.runtime.layers.layernorm import RMSNorm

        norm = RMSNorm(hidden_size, eps=1e-6).to(device=device_obj, dtype=torch.bfloat16)

        def kernel_func():
            return norm(x)

        kernel_source = "sglang_rmsnorm"
    elif op_name == "scale_residual_layernorm_scale_shift":
        from sglang.multimodal_gen.runtime.layers.layernorm import ScaleResidualLayerNormScaleShift

        module = ScaleResidualLayerNormScaleShift(
            hidden_size,
            eps=1e-6,
            elementwise_affine=False,
            dtype=torch.float32,
        ).to(device_obj)
        residual = torch.randn_like(x)
        gate = torch.randn(batch_size, 1, hidden_size, dtype=torch.float32, device=device_obj)
        shift = torch.randn(batch_size, 1, hidden_size, dtype=torch.float32, device=device_obj)
        scale = torch.randn(batch_size, 1, hidden_size, dtype=torch.float32, device=device_obj)

        def kernel_func():
            return module(residual, x, gate, shift, scale)

        kernel_source = "sglang_cutedsl_scale_residual_layernorm_scale_shift"
    elif op_name == "scale_residual":
        from sglang.multimodal_gen.runtime.layers.elementwise import MulAdd

        module = MulAdd().to(device_obj)

        residual = torch.randn_like(x)
        gate = torch.randn(batch_size, 1, hidden_size, dtype=torch.float32, device=device_obj)

        def kernel_func():
            return module(x, gate, residual)

        kernel_source = "sglang_mul_add"
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
                "tp_size": tp_size,
                "latency": results["latency_ms"],
            }
        ],
        framework="SGLang",
        version=pkg_resources.get_distribution("sglang").version,
        device_name=torch.cuda.get_device_name(device_obj),
        op_name=op_name,
        kernel_source=kernel_source,
        perf_filename=perf_filename,
        power_stats=results["power_stats"],
    )


if __name__ == "__main__":
    from collector.registry_types import PerfFile

    for test_case in get_wan_elementwise_test_cases():
        run_wan_elementwise(*test_case, perf_filename=PerfFile.WAN_ELEMENTWISE)
