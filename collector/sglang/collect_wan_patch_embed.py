# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

__compat__ = "sglang>=0.5.10"

import pkg_resources
import torch

from collector.helper import benchmark_with_power, log_perf
from collector.sglang.wan_common import (
    WAN_HIDDEN_SIZE,
    WAN_IN_CHANNELS,
    WAN_PATCH_SIZE,
    iter_wan_video_cases,
    latent_shape,
    seq_len_from_video,
)


def get_wan_patch_embed_test_cases():
    test_cases = []
    for profile, num_frames, height, width in iter_wan_video_cases():
        latent_t, latent_h, latent_w = latent_shape(num_frames, height, width)
        seq_len = seq_len_from_video(num_frames, height, width)
        test_cases.append(
            [
                profile.model,
                profile.task,
                1,
                WAN_IN_CHANNELS,
                latent_t,
                latent_h,
                latent_w,
                *WAN_PATCH_SIZE,
                seq_len,
            ]
        )
    return test_cases


def run_wan_patch_embed(
    model,
    task,
    batch_size,
    in_channels,
    frames,
    height,
    width,
    patch_t,
    patch_h,
    patch_w,
    seq_len,
    *,
    perf_filename,
    device="cuda:0",
):
    torch.cuda.set_device(device)
    torch.set_default_device(device)
    torch.set_default_dtype(torch.bfloat16)
    device_obj = torch.device(device)

    conv = torch.nn.Conv3d(
        in_channels,
        WAN_HIDDEN_SIZE,
        kernel_size=(patch_t, patch_h, patch_w),
        stride=(patch_t, patch_h, patch_w),
        bias=True,
        dtype=torch.bfloat16,
        device=device_obj,
    )
    hidden_states = torch.randn(
        batch_size,
        in_channels,
        frames,
        height,
        width,
        dtype=torch.bfloat16,
        device=device_obj,
    )

    def kernel_func():
        return conv(hidden_states)

    with benchmark_with_power(
        device=device_obj,
        kernel_func=kernel_func,
        num_warmups=3,
        num_runs=10,
        repeat_n=1,
    ) as results:
        pass

    log_perf(
        item_list=[
            {
                "model": model,
                "task": task,
                "batch_size": batch_size,
                "in_channels": in_channels,
                "frames": frames,
                "height": height,
                "width": width,
                "patch_t": patch_t,
                "patch_h": patch_h,
                "patch_w": patch_w,
                "seq_len": seq_len,
                "latency": results["latency_ms"],
            }
        ],
        framework="SGLang",
        version=pkg_resources.get_distribution("sglang").version,
        device_name=torch.cuda.get_device_name(device_obj),
        op_name="wan_patch_embed",
        kernel_source="torch_conv3d",
        perf_filename=perf_filename,
        power_stats=results["power_stats"],
    )


if __name__ == "__main__":
    from collector.registry_types import PerfFile

    for test_case in get_wan_patch_embed_test_cases():
        run_wan_patch_embed(*test_case, perf_filename=PerfFile.WAN_PATCH_EMBED)
