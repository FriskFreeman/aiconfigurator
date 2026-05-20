# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

__compat__ = "sglang>=0.5.10"

import os

import pkg_resources
import torch
import torch.nn.functional as F

from collector.helper import benchmark_with_power, log_perf
from collector.sglang.wan_common import (
    WAN_PROFILES,
    WAN_VAE_BASE_DIM,
    WAN_VAE_DIM_MULT,
    WAN_VAE_STRIDE,
    WAN_VAE_TEMPORAL_DOWNSAMPLE,
    WAN_VAE_Z_DIM,
    ceil_div,
)


VAE_SP_SIZES = (1, 2, 4, 8, 16, 32)


def _vae_latent_frames(num_frames: int) -> int:
    return (num_frames - 1) // WAN_VAE_STRIDE[0] + 1


def _video_cases():
    seen = set()
    for profile in WAN_PROFILES:
        for num_frames in profile.frames:
            for height, width in profile.resolutions:
                key = (profile.model, profile.task, num_frames, height, width)
                if key in seen:
                    continue
                seen.add(key)
                yield profile.model, profile.task, num_frames, height, width


def _vae_parallel_height(height: int, sp_size: int) -> int:
    return ceil_div(height, max(1, sp_size))


def _align_down_to_multiple(value: int, multiple: int) -> int:
    if multiple <= 1:
        return max(1, value)
    aligned = value - value % multiple
    return max(multiple, aligned)


def _vae_encode_input_local_height(height: int, sp_size: int) -> int:
    downsample_count = max(len(WAN_VAE_DIM_MULT) - 1, 0)
    factor = max(1, sp_size) * (2**downsample_count)
    padded = height + ((factor - height % factor) % factor)
    return max(1, padded // max(1, sp_size))


def _decode_shapes(num_frames: int, height: int, width: int, sp_size: int):
    frames = _vae_latent_frames(num_frames)
    h = _vae_parallel_height(height // WAN_VAE_STRIDE[1], sp_size)
    w = width // WAN_VAE_STRIDE[2]
    h = _align_down_to_multiple(h, 2)
    w = _align_down_to_multiple(w, 2)
    dims = [WAN_VAE_BASE_DIM * u for u in [WAN_VAE_DIM_MULT[-1]] + list(WAN_VAE_DIM_MULT[::-1])]
    yield "decode_post_quant_conv", WAN_VAE_Z_DIM, WAN_VAE_Z_DIM, frames, h, w, "1x1"
    yield "decode_conv_in", WAN_VAE_Z_DIM, dims[0], 1, h, w, "3x3x3"
    for idx, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:], strict=True)):
        yield f"decode_resblock_{idx}_conv1", in_dim, out_dim, 1, h, w, "3x3x3"
        yield f"decode_resblock_{idx}_conv2", out_dim, out_dim, 1, h, w, "3x3x3"
        if idx != len(WAN_VAE_DIM_MULT) - 1:
            h *= 2
            w *= 2
            if WAN_VAE_TEMPORAL_DOWNSAMPLE[::-1][idx]:
                yield f"decode_upsample3d_{idx}_time_conv", out_dim, out_dim * 2, 1, h // 2, w // 2, "3x1x1"
            yield f"decode_upsample2d_{idx}_conv", out_dim, out_dim, 1, h, w, "2d_3x3"
    yield "decode_conv_out", dims[-1], 3, 1, h, w, "3x3x3"


def _encode_shapes(num_frames: int, height: int, width: int, sp_size: int):
    frames = 1
    h = _vae_encode_input_local_height(height, sp_size)
    w = width
    dims = [WAN_VAE_BASE_DIM * u for u in [1] + list(WAN_VAE_DIM_MULT)]
    yield "encode_conv_in", 3, dims[0], frames, h, w, "3x3x3"
    for idx, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:], strict=True)):
        yield f"encode_resblock_{idx}_conv1", in_dim, out_dim, frames, h, w, "3x3x3"
        yield f"encode_resblock_{idx}_conv2", out_dim, out_dim, frames, h, w, "3x3x3"
        if idx != len(WAN_VAE_DIM_MULT) - 1:
            if WAN_VAE_TEMPORAL_DOWNSAMPLE[idx]:
                yield f"encode_downsample3d_{idx}_time_conv", out_dim, out_dim, frames, ceil_div(h, 2), ceil_div(w, 2), "3x1x1"
            yield f"encode_downsample2d_{idx}_conv", out_dim, out_dim, frames, ceil_div(h, 2), ceil_div(w, 2), "2d_3x3"
            h = ceil_div(h, 2)
            w = ceil_div(w, 2)
    yield "encode_quant_conv", WAN_VAE_Z_DIM * 2, WAN_VAE_Z_DIM * 2, frames, h, w, "1x1"


def get_wan_vae_test_cases():
    max_elems = int(os.environ.get("COLLECTOR_WAN_VAE_MAX_INPUT_ELEMS", "600000000"))
    test_cases = []
    seen = set()
    for model, task, num_frames, height, width in _video_cases():
        for sp_size in VAE_SP_SIZES:
            for path, iterator in (("decode", _decode_shapes), ("encode", _encode_shapes)):
                for stage, in_channels, out_channels, frames, local_height, local_width, conv_kind in iterator(num_frames, height, width, sp_size):
                    if in_channels * max(1, frames) * local_height * local_width > max_elems:
                        continue
                    key = (
                        model,
                        task,
                        path,
                        stage,
                        1,
                        in_channels,
                        out_channels,
                        frames,
                        local_height,
                        local_width,
                        conv_kind,
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    test_cases.append(list(key))
    return test_cases


def get_wan_vae_attention_test_cases():
    test_cases = []
    seen = set()
    for model, task, num_frames, height, width in _video_cases():
        frames = 1
        for sp_size in VAE_SP_SIZES:
            for channels, spatial_div in ((384, 16), (192, 8), (96, 4)):
                local_height = _vae_parallel_height(max(1, height // spatial_div), sp_size)
                local_width = max(1, width // spatial_div)
                key = (model, task, 1, channels, frames, local_height, local_width)
                if key not in seen:
                    seen.add(key)
                    test_cases.append(list(key))
    return test_cases


def get_wan_vae_elementwise_test_cases():
    test_cases = []
    seen = set()
    for model, task, num_frames, height, width in _video_cases():
        frames = _vae_latent_frames(num_frames)
        for sp_size in VAE_SP_SIZES:
            for op_name in ("rms_norm_5d", "silu", "avg_down3d", "dup_up3d"):
                for channels, spatial_div in ((96, 4), (192, 8), (384, 16)):
                    local_height = _vae_parallel_height(max(1, height // spatial_div), sp_size)
                    local_width = max(1, width // spatial_div)
                    if op_name == "avg_down3d":
                        # SGLang AvgDown3D only pads time; spatial dims must already be even.
                        local_height = _align_down_to_multiple(local_height, 2)
                        local_width = _align_down_to_multiple(local_width, 2)
                    case_frames = 1 if op_name in ("avg_down3d", "dup_up3d") else frames
                    key = (model, task, op_name, 1, channels, case_frames, local_height, local_width)
                    if key not in seen:
                        seen.add(key)
                        test_cases.append(list(key))
    return test_cases


def _make_conv(in_channels, out_channels, conv_kind, device_obj):
    from sglang.multimodal_gen.runtime.models.vaes.parallel.wan_common_utils import WanCausalConv3d

    if conv_kind == "1x1":
        return WanCausalConv3d(in_channels, out_channels, 1).to(device_obj, dtype=torch.float32)
    if conv_kind == "3x1x1":
        return WanCausalConv3d(in_channels, out_channels, (3, 1, 1), padding=(1, 0, 0)).to(device_obj, dtype=torch.float32)
    if conv_kind == "2d_3x3":
        return torch.nn.Conv2d(in_channels, out_channels, 3, padding=1).to(device_obj, dtype=torch.float32)
    return WanCausalConv3d(in_channels, out_channels, 3, padding=1).to(device_obj, dtype=torch.float32)


def run_wan_vae(
    model,
    task,
    path,
    stage,
    batch_size,
    in_channels,
    out_channels,
    frames,
    height,
    width,
    conv_kind,
    *,
    perf_filename,
    device="cuda:0",
):
    torch.cuda.set_device(device)
    device_obj = torch.device(device)
    module = _make_conv(in_channels, out_channels, conv_kind, device_obj)
    hidden_states = torch.randn(batch_size, in_channels, frames, height, width, dtype=torch.float32, device=device_obj)

    if conv_kind == "2d_3x3":
        hidden_2d = hidden_states.permute(0, 2, 1, 3, 4).reshape(batch_size * frames, in_channels, height, width)

        def kernel_func():
            return module(hidden_2d)

        kernel_source = "torch_conv2d"
    else:
        cache_x = torch.randn(
            batch_size,
            in_channels,
            max(0, min(2, module._padding[4])),
            height,
            width,
            dtype=torch.float32,
            device=device_obj,
        )

        def kernel_func():
            return module(hidden_states, cache_x if cache_x.shape[2] > 0 else None)

        kernel_source = "sglang_wan_causal_conv3d"

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
                "model": model,
                "task": task,
                "path": path,
                "stage": stage,
                "batch_size": batch_size,
                "in_channels": in_channels,
                "out_channels": out_channels,
                "frames": frames,
                "height": height,
                "width": width,
                "conv_kind": conv_kind,
                "latency": results["latency_ms"],
            }
        ],
        framework="SGLang",
        version=pkg_resources.get_distribution("sglang").version,
        device_name=torch.cuda.get_device_name(device_obj),
        op_name="wan_vae_conv",
        kernel_source=kernel_source,
        perf_filename=perf_filename,
        power_stats=results["power_stats"],
    )


def run_wan_vae_attention(
    model,
    task,
    batch_size,
    channels,
    frames,
    height,
    width,
    *,
    perf_filename,
    device="cuda:0",
):
    torch.cuda.set_device(device)
    device_obj = torch.device(device)
    q = torch.randn(batch_size * frames, 1, height * width, channels, dtype=torch.float32, device=device_obj)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    def kernel_func():
        return F.scaled_dot_product_attention(q, k, v)

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
                "model": model,
                "task": task,
                "batch_size": batch_size,
                "channels": channels,
                "frames": frames,
                "height": height,
                "width": width,
                "tokens_per_frame": height * width,
                "latency": results["latency_ms"],
            }
        ],
        framework="SGLang",
        version=pkg_resources.get_distribution("sglang").version,
        device_name=torch.cuda.get_device_name(device_obj),
        op_name="wan_vae_attention",
        kernel_source="torch_sdpa",
        perf_filename=perf_filename,
        power_stats=results["power_stats"],
    )


def run_wan_vae_elementwise(
    model,
    task,
    op_name,
    batch_size,
    channels,
    frames,
    height,
    width,
    *,
    perf_filename,
    device="cuda:0",
):
    torch.cuda.set_device(device)
    device_obj = torch.device(device)
    x = torch.randn(batch_size, channels, frames, height, width, dtype=torch.float32, device=device_obj)

    if op_name == "rms_norm_5d":
        from sglang.multimodal_gen.runtime.models.vaes import wanvae
        from sglang.multimodal_gen.runtime.models.vaes.parallel.wan_common_utils import WanRMS_norm

        module = WanRMS_norm(channels, images=False).to(device_obj, dtype=torch.float32)

        def kernel_func():
            with wanvae.forward_context():
                return module(x)

        kernel_source = "sglang_wan_rms_norm"
    elif op_name == "silu":
        module = torch.nn.SiLU().to(device_obj)

        def kernel_func():
            return module(x)

        kernel_source = "torch_silu"
    elif op_name == "avg_down3d":
        from sglang.multimodal_gen.runtime.models.vaes.parallel.wan_common_utils import AvgDown3D

        module = AvgDown3D(channels, channels * 2, factor_t=1, factor_s=2).to(device_obj)

        def kernel_func():
            return module(x)

        kernel_source = "sglang_avg_down3d"
    elif op_name == "dup_up3d":
        from sglang.multimodal_gen.runtime.models.vaes import wanvae
        from sglang.multimodal_gen.runtime.models.vaes.parallel.wan_common_utils import DupUp3D

        module = DupUp3D(channels, max(channels // 2, 1), factor_t=1, factor_s=2).to(device_obj)

        def kernel_func():
            with wanvae.forward_context(first_chunk_arg=False):
                return module(x)

        kernel_source = "sglang_dup_up3d"
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
                "model": model,
                "task": task,
                "batch_size": batch_size,
                "channels": channels,
                "frames": frames,
                "height": height,
                "width": width,
                "latency": results["latency_ms"],
            }
        ],
        framework="SGLang",
        version=pkg_resources.get_distribution("sglang").version,
        device_name=torch.cuda.get_device_name(device_obj),
        op_name=f"wan_vae_{op_name}",
        kernel_source=kernel_source,
        perf_filename=perf_filename,
        power_stats=results["power_stats"],
    )


if __name__ == "__main__":
    from collector.registry_types import PerfFile

    for test_case in get_wan_vae_test_cases():
        run_wan_vae(*test_case, perf_filename=PerfFile.WAN_VAE)
    for test_case in get_wan_vae_attention_test_cases():
        run_wan_vae_attention(*test_case, perf_filename=PerfFile.WAN_VAE_ATTENTION)
    for test_case in get_wan_vae_elementwise_test_cases():
        run_wan_vae_elementwise(*test_case, perf_filename=PerfFile.WAN_VAE_ELEMENTWISE)
