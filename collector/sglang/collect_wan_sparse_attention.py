# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

__compat__ = "sglang>=0.5.10"

import os

import pkg_resources
import torch

from collector.helper import benchmark_with_power, log_perf
from collector.sglang.wan_common import (
    SingleRankGroup,
    WAN_PATCH_SIZE,
    iter_wan_video_cases,
    latent_shape,
    seq_len_from_video,
)


def _parse_float_list(value: str, default: tuple[float, ...]) -> tuple[float, ...]:
    if not value:
        return default
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def _enable_sagesla() -> bool:
    return os.environ.get("COLLECTOR_WAN_ENABLE_SAGESLA", "0").lower() in ("1", "true", "yes")


def _enable_vsa() -> bool:
    return os.environ.get("COLLECTOR_WAN_ENABLE_VSA", "1").lower() in ("1", "true", "yes")


def _sparse_case_is_reasonable(seq_len: int, num_heads: int, head_dim: int) -> bool:
    max_output_elems = int(os.environ.get("COLLECTOR_WAN_MAX_SPARSE_ATTN_OUTPUT_ELEMS", "80000000"))
    return seq_len * max(1, num_heads) * head_dim <= max_output_elems


def _a2a_parallel_cases(num_heads: int) -> list[tuple[int, int, str]]:
    cases = []
    for tp_size in (1, 2, 4, 8):
        if num_heads % tp_size != 0:
            continue
        heads_after_tp = num_heads // tp_size
        for sp_size in (1, 2, 4, 8):
            if sp_size == 1:
                cases.append((tp_size, sp_size, "none"))
            elif heads_after_tp % sp_size == 0:
                cases.append((tp_size, sp_size, "a2a"))
    return cases


def _post_a2a_shape(global_seq_len: int, tp_size: int, sp_size: int, num_heads: int) -> tuple[int, int]:
    return global_seq_len, (num_heads // tp_size) // sp_size


def get_wan_sparse_attention_test_cases():
    test_cases = []
    seen = set()
    sla_attention_types = ["sla"]
    if _enable_sagesla():
        sla_attention_types.append("sagesla")
    vsa_sparsities = _parse_float_list(
        os.environ.get("COLLECTOR_WAN_VSA_SPARSITIES", ""),
        default=(0.0, 0.3, 0.5),
    )
    for profile, num_frames, height, width in iter_wan_video_cases():
        global_seq_len = seq_len_from_video(
            num_frames,
            height,
            width,
            latent_stride=profile.latent_prepare_stride,
        )
        raw_latent_t, raw_latent_h, raw_latent_w = latent_shape(
            num_frames, height, width, stride=profile.latent_prepare_stride
        )
        for tp_size, sp_size, sp_algorithm in _a2a_parallel_cases(profile.num_heads):
            q_seq_len, local_heads = _post_a2a_shape(global_seq_len, tp_size, sp_size, profile.num_heads)
            if local_heads <= 0 or not _sparse_case_is_reasonable(q_seq_len, local_heads, profile.head_dim):
                continue
            for attention_type in sla_attention_types:
                key = (
                    profile.model,
                    profile.task,
                    "minimal_a2a_sla",
                    attention_type,
                    1,
                    q_seq_len,
                    local_heads,
                    profile.head_dim,
                    tp_size,
                    sp_size,
                    sp_algorithm,
                    raw_latent_t,
                    raw_latent_h,
                    raw_latent_w,
                    0.1,
                    0.0,
                )
                if key not in seen:
                    seen.add(key)
                    test_cases.append(list(key))
            if _enable_vsa():
                for vsa_sparsity in vsa_sparsities:
                    key = (
                        profile.model,
                        profile.task,
                        "ulysses_vsa",
                        "video_sparse_attn",
                        1,
                        q_seq_len,
                        local_heads,
                        profile.head_dim,
                        tp_size,
                        sp_size,
                        "vsa_ulysses" if sp_size > 1 else "none",
                        raw_latent_t,
                        raw_latent_h,
                        raw_latent_w,
                        0.0,
                        vsa_sparsity,
                    )
                    if key not in seen:
                        seen.add(key)
                        test_cases.append(list(key))
    return test_cases


def _ensure_single_rank_sequence_state():
    from sglang.multimodal_gen.runtime.distributed import parallel_state

    group = SingleRankGroup()
    for name in ("_SP", "_WORLD", "_TP", "_DP", "_PP", "_CFG"):
        if getattr(parallel_state, name, None) is None:
            setattr(parallel_state, name, group)


def _run_minimal_a2a_sla(
    attention_type: str,
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    device_obj: torch.device,
):
    from sglang.multimodal_gen.runtime.layers.attention import MinimalA2AAttnOp
    from sglang.multimodal_gen.runtime.layers.attention.selector import global_force_attn_backend_context_manager
    from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
    from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum

    forced_backend = AttentionBackendEnum.SLA_ATTN
    if attention_type == "sagesla":
        forced_backend = AttentionBackendEnum.SAGE_SLA_ATTN
    with global_force_attn_backend_context_manager(forced_backend):
        module = MinimalA2AAttnOp(
            num_heads=num_heads,
            head_size=head_dim,
            attention_type=attention_type,
            topk=0.1,
            supported_attention_backends={forced_backend},
            prefix="collector.wan_sparse_attention.minimal_a2a",
        ).to(device_obj)
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.bfloat16, device=device_obj)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    def kernel_func():
        with set_forward_context(current_timestep=0, attn_metadata=None):
            return module(q, k, v)

    return kernel_func, f"sglang_minimal_a2a_{attention_type}"


def _run_ulysses_vsa(
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    raw_latent_t: int,
    raw_latent_h: int,
    raw_latent_w: int,
    vsa_sparsity: float,
    device_obj: torch.device,
):
    from sglang.multimodal_gen.runtime.layers.attention import UlyssesAttention_VSA
    from sglang.multimodal_gen.runtime.layers.attention.backends.video_sparse_attn import VideoSparseAttentionMetadataBuilder
    from sglang.multimodal_gen.runtime.layers.attention.selector import global_force_attn_backend_context_manager
    from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
    from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum

    _ensure_single_rank_sequence_state()
    attn_metadata = VideoSparseAttentionMetadataBuilder().build(
        current_timestep=0,
        raw_latent_shape=(raw_latent_t, raw_latent_h, raw_latent_w),
        patch_size=WAN_PATCH_SIZE,
        VSA_sparsity=vsa_sparsity,
        device=device_obj,
    )
    with global_force_attn_backend_context_manager(AttentionBackendEnum.VIDEO_SPARSE_ATTN):
        module = UlyssesAttention_VSA(
            num_heads=num_heads,
            head_size=head_dim,
            causal=False,
            supported_attention_backends={AttentionBackendEnum.VIDEO_SPARSE_ATTN},
            prefix="collector.wan_sparse_attention.vsa",
        ).to(device_obj)
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.bfloat16, device=device_obj)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    gate_compress = torch.randn_like(q)

    def kernel_func():
        with set_forward_context(current_timestep=0, attn_metadata=attn_metadata):
            return module(q, k, v, gate_compress=gate_compress)

    return kernel_func, "sglang_ulysses_video_sparse_attn"


def run_wan_sparse_attention(
    model,
    task,
    op_variant,
    backend,
    batch_size,
    seq_len,
    num_heads,
    head_dim,
    tp_size,
    sp_size,
    sp_algorithm,
    raw_latent_t,
    raw_latent_h,
    raw_latent_w,
    sla_topk,
    vsa_sparsity,
    *,
    perf_filename,
    device="cuda:0",
):
    torch.cuda.set_device(device)
    torch.set_default_dtype(torch.bfloat16)
    device_obj = torch.device(device)
    if op_variant == "minimal_a2a_sla":
        kernel_func, kernel_source = _run_minimal_a2a_sla(
            backend,
            batch_size,
            seq_len,
            num_heads,
            head_dim,
            device_obj,
        )
    elif op_variant == "ulysses_vsa":
        kernel_func, kernel_source = _run_ulysses_vsa(
            batch_size,
            seq_len,
            num_heads,
            head_dim,
            raw_latent_t,
            raw_latent_h,
            raw_latent_w,
            vsa_sparsity,
            device_obj,
        )
    else:
        raise ValueError(f"Unsupported Wan sparse attention variant: {op_variant}")

    with benchmark_with_power(
        device=device_obj,
        kernel_func=kernel_func,
        num_warmups=3,
        num_runs=10,
        repeat_n=1,
        allow_graph_fail=True,
        use_cuda_graph=False,
    ) as results:
        pass

    log_perf(
        item_list=[
            {
                "model": model,
                "task": task,
                "op_variant": op_variant,
                "backend": backend,
                "batch_size": batch_size,
                "seq_len": seq_len,
                "num_heads": num_heads,
                "head_dim": head_dim,
                "tp_size": tp_size,
                "sp_size": sp_size,
                "sp_algorithm": sp_algorithm,
                "raw_latent_t": raw_latent_t,
                "raw_latent_h": raw_latent_h,
                "raw_latent_w": raw_latent_w,
                "patch_t": WAN_PATCH_SIZE[0],
                "patch_h": WAN_PATCH_SIZE[1],
                "patch_w": WAN_PATCH_SIZE[2],
                "sla_topk": sla_topk,
                "vsa_sparsity": vsa_sparsity,
                "latency": results["latency_ms"],
            }
        ],
        framework="SGLang",
        version=pkg_resources.get_distribution("sglang").version,
        device_name=torch.cuda.get_device_name(device_obj),
        op_name="wan_sparse_attention",
        kernel_source=kernel_source,
        perf_filename=perf_filename,
        power_stats=results["power_stats"],
    )


if __name__ == "__main__":
    from collector.registry_types import PerfFile

    for test_case in get_wan_sparse_attention_test_cases():
        run_wan_sparse_attention(*test_case, perf_filename=PerfFile.WAN_SPARSE_ATTENTION)
