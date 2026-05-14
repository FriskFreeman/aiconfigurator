# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

__compat__ = "sglang>=0.5.10"

import os
import random

import pkg_resources
import torch
import torch.nn.functional as F
from sgl_kernel import (
    fp8_scaled_mm,
    sgl_per_token_quant_fp8,
)

from collector.common_test_cases import get_gemm_common_test_cases

try:
    from flashinfer import fp4_quantize as flashinfer_fp4_quantize
    from flashinfer import mm_fp4 as flashinfer_mm_fp4
    from flashinfer import shuffle_matrix_sf_a as flashinfer_shuffle_sf_a

    HAS_FLASHINFER_FP4 = True
except ImportError:
    HAS_FLASHINFER_FP4 = False

from sglang.srt.layers.deep_gemm_wrapper import (
    DEEPGEMM_SCALE_UE8M0,
    gemm_nt_f8f8bf16,
)
from sglang.srt.layers.quantization.fp8_kernel import sglang_per_token_group_quant_fp8

from collector.helper import benchmark_with_power, get_sm_version, log_perf

# Disable DeepGEMM JIT precompilation (compiles ALL M values per unique N,K pair).
# The collector only needs the specific M being tested.
os.environ.setdefault("SGLANG_JIT_DEEPGEMM_PRECOMPILE", "0")


def _wan_latent_seq_lens() -> list[int]:
    """Return representative Wan2.2 DiT latent token counts after patching."""
    resolutions = [(704, 1280), (720, 1280), (480, 832)]
    frames = [121, 81, 49]
    vae_stride_ti2v = (4, 16, 16)
    patch_size = (1, 2, 2)
    seq_lens = set()
    for num_frames in frames:
        for height, width in resolutions:
            latent_t = num_frames
            latent_h = height // vae_stride_ti2v[1]
            latent_w = width // vae_stride_ti2v[2]
            seq_lens.add((latent_t // patch_size[0]) * (latent_h // patch_size[1]) * (latent_w // patch_size[2]))
    return sorted(seq_lens)


def _get_wan_gemm_common_test_cases() -> list[tuple[int, int, int]]:
    """Generate Wan2.2 DiT-specific (m, n, k) GEMM shapes.

    SGLang Wan uses ColumnParallelLinear / RowParallelLinear for QKV, output,
    MLP and condition embedders.  TP affects the per-rank output/input
    dimensions, while SP affects attention shapes but not these GEMM M values
    after local latent sharding is modeled by the caller.  We include both
    global and representative SP-local token counts to keep the GEMM table
    useful for single-card compute modeling.
    """
    hidden = 5120
    ffn = 13824
    text_dim = 4096
    freq_dim = 256
    image_tokens = 257
    text_tokens = 512
    t5_d_model = 4096
    t5_d_ff = 10240
    t5_heads = 64
    t5_d_kv = 64
    clip_hidden = 768
    clip_intermediate = 3072
    clip_tokens = 50
    vae_conv_tokens = [64, 256, 1024, 4096]
    vae_channels = [96, 192, 384]
    tp_list = [1, 2, 4, 8]
    sp_list = [1, 2, 4, 8]

    shapes: set[tuple[int, int, int]] = set()
    latent_seq_lens = _wan_latent_seq_lens()

    for seq_len in latent_seq_lens:
        for sp_size in sp_list:
            local_seq_len = (seq_len + sp_size - 1) // sp_size
            for tp_size in tp_list:
                hidden_per_tp = hidden // tp_size
                ffn_per_tp = ffn // tp_size

                # Wan block self-attention projections.
                shapes.add((local_seq_len, hidden_per_tp, hidden))  # to_q/k/v
                shapes.add((local_seq_len, hidden, hidden_per_tp))  # to_out

                # Wan block FFN.
                shapes.add((local_seq_len, ffn_per_tp, hidden))  # fc_in
                shapes.add((local_seq_len, hidden, ffn_per_tp))  # fc_out

    for tp_size in tp_list:
        hidden_per_tp = hidden // tp_size
        ffn_per_tp = ffn // tp_size

        # Cross attention projections: q uses latent tokens, k/v use text/image.
        for seq_len in latent_seq_lens:
            shapes.add((seq_len, hidden_per_tp, hidden))
            shapes.add((seq_len, hidden, hidden_per_tp))
        shapes.add((text_tokens, hidden_per_tp, hidden))
        shapes.add((image_tokens, hidden_per_tp, hidden))

        # Condition embedders.
        shapes.add((text_tokens, hidden_per_tp, text_dim))
        shapes.add((text_tokens, hidden, hidden_per_tp))
        shapes.add((image_tokens, hidden_per_tp, hidden))
        shapes.add((image_tokens, hidden, ffn_per_tp))
        shapes.add((1, hidden_per_tp, freq_dim))
        shapes.add((1, hidden, hidden_per_tp))
        shapes.add((1, (hidden * 6) // tp_size, hidden))  # time modulation

        # Wan T5 text encoder parallel Linear shapes.
        t5_qkv_per_tp = (3 * t5_heads * t5_d_kv) // tp_size
        t5_ff_per_tp = t5_d_ff // tp_size
        for seq_len in (128, 256, text_tokens):
            shapes.add((seq_len, t5_qkv_per_tp, t5_d_model))
            shapes.add((seq_len, t5_d_model, (t5_heads * t5_d_kv) // tp_size))
            shapes.add((seq_len, t5_ff_per_tp, t5_d_model))
            shapes.add((seq_len, t5_d_model, t5_ff_per_tp))

        # Wan I2V CLIP vision encoder Linear shapes.
        clip_qkv_per_tp = (3 * clip_hidden) // tp_size
        clip_intermediate_per_tp = clip_intermediate // tp_size
        shapes.add((clip_tokens, clip_qkv_per_tp, clip_hidden))
        shapes.add((clip_tokens, clip_hidden, clip_hidden // tp_size))
        shapes.add((clip_tokens, clip_intermediate_per_tp, clip_hidden))
        shapes.add((clip_tokens, clip_hidden, clip_intermediate_per_tp))

        # Wan VAE 1x1 Conv2d/QKV and projection map to small GEMM-like shapes.
        for tokens in vae_conv_tokens:
            for channels in vae_channels:
                shapes.add((tokens, channels * 3, channels))
                shapes.add((tokens, channels, channels))

    return sorted(shapes, reverse=True)


def get_gemm_test_cases():
    test_cases = []

    sm_version = get_sm_version()
    if sm_version < 89:
        gemm_list = ["bfloat16"]
    elif sm_version < 90:
        # SM89 (L40S) and earlier don't have TMA - skip fp8_block
        gemm_list = ["bfloat16", "fp8"]
    elif sm_version < 100:
        # Hopper supports fp8_block
        # fp8_block (DeepGEMM) requires SM90+ for TMA support
        gemm_list = ["fp8_block", "bfloat16", "fp8"]
    elif sm_version < 110:
        # SM100/SM103 (B100/B200 datacenter Blackwell): fp8_block + nvfp4
        gemm_list = ["fp8_block", "bfloat16", "fp8", "nvfp4"]
    else:
        # SM120+ (RTX PRO 6000 Blackwell workstation): no DeepGEMM recipe for fp8_block
        gemm_list = ["bfloat16", "fp8", "nvfp4"]

    if os.environ.get("COLLECTOR_GEMM_ONLY_WAN", "0").lower() not in ("1", "true", "yes"):
        for gemm_common_testcase in get_gemm_common_test_cases():
            x = gemm_common_testcase.x
            n = gemm_common_testcase.n
            k = gemm_common_testcase.k
            for gemm_type in gemm_list:
                if (gemm_type == "nvfp4" or gemm_type == "fp8_block") and (n < 128 or k < 128):
                    continue
                if gemm_type == "fp8_block" and k % 128 != 0:
                    continue

                test_cases.append([gemm_type, x, n, k])

    seen = {(gemm_type, m, n, k) for gemm_type, m, n, k in test_cases}
    for m, n, k in _get_wan_gemm_common_test_cases():
        for gemm_type in gemm_list:
            if (gemm_type == "nvfp4" or gemm_type == "fp8_block") and (n < 128 or k < 128):
                continue
            if gemm_type == "fp8_block" and k % 128 != 0:
                continue
            item = (gemm_type, m, n, k)
            if item not in seen:
                test_cases.append([gemm_type, m, n, k])
                seen.add(item)

    # Try to optimize number of JIT precompile cache hits by shuffling test cases.
    random.seed(42)
    random.shuffle(test_cases)

    return test_cases


def cdiv(a: int, b: int) -> int:
    """Ceiling division."""
    return -(a // -b)


def fp8_gemm_deepgemm(
    x_fp8: torch.Tensor,
    x_scale: torch.Tensor,
    y_fp8: torch.Tensor,
    y_scale: torch.Tensor,
    out: torch.Tensor,
    m: int,
    n: int,
    k: int,
):
    """
    DeepGEMM implementation of FP8 GEMM
    It maps to a specific commit for each SGLang release.
    Check the commit tag in sglang/sgl-kernel/CMakeLists.txt, repo-deepgemm
    """

    # Run DeepGEMM kernel
    gemm_nt_f8f8bf16((x_fp8, x_scale), (y_fp8, y_scale), out)
    return out


def scale_shape(shape, group_shape):
    assert len(shape) == len(group_shape)
    return tuple(cdiv(shape[i], group_shape[i]) for i in range(len(group_shape)))


def per_token_quant_int8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize fp32/fp16/bf16 tensor to int8 with per-token scaling"""
    # Calculate per-row (per-token) scaling factor
    x_fp32 = x.to(torch.float32) if x.dtype != torch.float32 else x
    absmax = torch.max(torch.abs(x_fp32), dim=-1, keepdim=True)[0].clamp(min=1e-10)
    scale = absmax / 127.0

    # Quantize to int8
    x_scaled = x_fp32 / scale
    x_int8 = torch.round(x_scaled).clamp(-128, 127).to(torch.int8)

    # Return int8 tensor and scale (squeeze the last dimension for scale)
    return x_int8, scale.squeeze(-1)


def run_gemm(gemm_type, batch_size, N, K, *, perf_filename, device="cuda:0"):  # noqa: N803
    assert gemm_type in [
        "fp8_block",
        "fp8",
        "bfloat16",
        "int8_wo",
        "nvfp4",
    ], "not support gemm type"
    torch.cuda.set_device(device)
    M = batch_size  # noqa: N806

    def round_up(x, m):
        return (x + m - 1) // m * m

    def create_gemm():
        dtype = torch.bfloat16
        if gemm_type == "nvfp4":
            if not HAS_FLASHINFER_FP4:
                return None

            # Prepare source data: Activation A [M, K] in BF16
            a_bf16 = torch.randn((M, K), device=device, dtype=dtype)

            # Prepare Weight B [N, K] and its dummy scale
            b_bf16_dummy = torch.randn((N, K), device=device, dtype=dtype)

            # Global scales
            a_global_scale = torch.tensor([1.0], device=device, dtype=torch.float32)
            b_global_scale = torch.tensor([1.0], device=device, dtype=torch.float32)
            alpha = 1.0 / (a_global_scale * b_global_scale)

            # Pre-quantize and Shuffle Weight B (load time process)
            b_fp4_linear, b_sf_linear = flashinfer_fp4_quantize(
                b_bf16_dummy, b_global_scale, is_sf_swizzled_layout=False
            )

            from flashinfer import shuffle_matrix_a as flashinfer_shuffle_a

            epilogue_tile_m = 128
            del b_bf16_dummy
            b_fp4_shuffled = flashinfer_shuffle_a(b_fp4_linear, epilogue_tile_m)
            del b_fp4_linear
            b_sf_shuffled = flashinfer_shuffle_sf_a(b_sf_linear.view(torch.uint8), epilogue_tile_m).view(
                torch.float8_e4m3fn
            )
            del b_sf_linear
            b_fp4_final = b_fp4_shuffled.t()
            del b_fp4_shuffled
            b_sf_final = b_sf_shuffled.t()
            del b_sf_shuffled

            out = torch.empty((M, round_up(N, 128)), device=device, dtype=dtype)

            def gemm_op():
                # Dynamic Quantization of Activation + GEMM
                a_fp4_dynamic, a_sf_dynamic = flashinfer_fp4_quantize(
                    a_bf16, a_global_scale, is_sf_swizzled_layout=True
                )
                return flashinfer_mm_fp4(
                    a_fp4_dynamic, b_fp4_final, a_sf_dynamic, b_sf_final, alpha, dtype, backend="cutlass", out=out
                )

            return gemm_op

        elif gemm_type == "fp8_block":
            fp8_info = torch.finfo(torch.float8_e4m3fn)
            a_bf16 = torch.randn(M, K, dtype=dtype, device=device)
            b_fp32 = (torch.rand(N, K, device=device) - 0.5) * 2 * fp8_info.max
            b_fp8 = b_fp32.clamp(min=fp8_info.min, max=fp8_info.max).to(torch.float8_e4m3fn)
            del b_fp32
            scale_b = torch.randn(scale_shape(b_fp8.shape, (128, 128)), device=device, dtype=torch.float32)
            out = torch.empty((M, N), device=device, dtype=dtype)

            def gemm_op():
                a_fp8, scale_a = sglang_per_token_group_quant_fp8(
                    a_bf16,
                    group_size=128,
                    column_major_scales=True,
                    scale_tma_aligned=True,
                    scale_ue8m0=DEEPGEMM_SCALE_UE8M0,
                )
                return fp8_gemm_deepgemm(a_fp8, scale_a, b_fp8, scale_b, out, M, N, K)

            return gemm_op

        elif gemm_type == "fp8":
            fp8_info = torch.finfo(torch.float8_e4m3fn)
            a_bf16 = torch.randn(M, K, dtype=dtype, device=device)
            b_fp32 = (torch.rand(N, K, device=device) - 0.5) * 2 * fp8_info.max
            b_fp8 = b_fp32.clamp(min=fp8_info.min, max=fp8_info.max).to(torch.float8_e4m3fn).t()
            del b_fp32
            scale_b = torch.randn((N,), device=device, dtype=torch.float32)
            output_fp8 = torch.empty_like(a_bf16, dtype=torch.float8_e4m3fn)
            scale_a = torch.empty((M, 1), device=device, dtype=torch.float32)

            def gemm_op():
                sgl_per_token_quant_fp8(a_bf16, output_fp8, scale_a)
                return fp8_scaled_mm(output_fp8, b_fp8, scale_a, scale_b, dtype)

            return gemm_op

        elif gemm_type == "bfloat16":
            a_bfloat16 = torch.randn(M, K, dtype=torch.bfloat16, device=device)
            b_bfloat16 = torch.randn(N, K, dtype=torch.bfloat16, device=device)

            def gemm_op():
                return F.linear(a_bfloat16, b_bfloat16, None)

            return gemm_op

    # Scale loop count down for large matrices to avoid OOM.
    # Each iteration holds persistent tensors (~M*K*2 + N*K + M*N*2 bytes for fp8).
    # Cap at 6 for small shapes (L2 thrashing mitigation), reduce for large ones.
    _bytes_per_elem = {"bfloat16": 2, "fp8": 1, "fp8_block": 1, "nvfp4": 0.5}
    _weight_bytes = int(N * K * _bytes_per_elem.get(gemm_type, 2))
    _gpu_mem = torch.cuda.get_device_properties(device).total_memory
    # Estimate per-iteration footprint: weight + activation + output
    _iter_bytes = _weight_bytes + M * K * 2 + M * N * 2
    outside_loop_count = max(1, min(6, int(_gpu_mem * 0.7 / _iter_bytes)))
    op_list = []
    for _ in range(outside_loop_count):
        op = create_gemm()
        if op is not None:
            op_list.append(op)

    if not op_list:
        print(f"No ops created for {gemm_type}, skipping.")
        return

    def kernel_func():
        for op in op_list:
            op()

    # Use benchmark_with_power context manager
    nvtx_tag = f"{gemm_type}_m{M}_n{N}_k{K}"
    torch.cuda.nvtx.range_push(nvtx_tag)

    with benchmark_with_power(
        device=device,
        kernel_func=kernel_func,
        num_warmups=3,
        num_runs=6,
        repeat_n=1,
    ) as results:
        pass

    torch.cuda.nvtx.range_pop()

    log_perf(
        item_list=[{"gemm_dtype": gemm_type, "m": M, "n": N, "k": K, "latency": results["latency_ms"] / len(op_list)}],
        framework="SGLang",
        version=pkg_resources.get_distribution("sglang").version,
        device_name=torch.cuda.get_device_name(device),
        op_name="gemm",
        kernel_source="sglang",
        perf_filename=perf_filename,
        power_stats=results["power_stats"],
    )
