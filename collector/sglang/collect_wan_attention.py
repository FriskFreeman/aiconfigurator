# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

__compat__ = "sglang>=0.5.10"

import os

import pkg_resources
import torch

from collector.helper import benchmark_with_power, log_perf
from collector.sglang.wan_common import (
    WAN_HEAD_DIM,
    WAN_IMAGE_TOKENS,
    WAN_TEXT_TOKENS,
    iter_wan_video_cases,
    local_attention_shape,
    seq_len_from_video,
    valid_parallel_cases,
)


def _backend_names() -> list[str]:
    names = ["auto"]
    if os.environ.get("COLLECTOR_WAN_ENABLE_FLASH_ATTN", "0").lower() in ("1", "true", "yes"):
        names.append("flash_attn")
    if os.environ.get("COLLECTOR_WAN_ENABLE_SDPA", "0").lower() in ("1", "true", "yes"):
        names.append("sdpa")
    if os.environ.get("COLLECTOR_WAN_ENABLE_SAGE_ATTN", "0").lower() in ("1", "true", "yes"):
        names.append("sage_attn")
    if os.environ.get("COLLECTOR_WAN_ENABLE_SLA", "0").lower() in ("1", "true", "yes"):
        names.extend(["sla", "sagesla"])
    return names


def _attention_case_is_reasonable(q_seq_len: int, kv_seq_len: int, num_heads: int, head_dim: int) -> bool:
    max_work = int(os.environ.get("COLLECTOR_WAN_MAX_ATTN_WORK", "2000000000"))
    max_output_elems = int(os.environ.get("COLLECTOR_WAN_MAX_ATTN_OUTPUT_ELEMS", "80000000"))
    if q_seq_len * kv_seq_len * max(1, num_heads) > max_work:
        return False
    if q_seq_len * max(1, num_heads) * head_dim > max_output_elems:
        return False
    return True


def get_wan_attention_test_cases():
    test_cases = []
    seen = set()
    for profile, num_frames, height, width in iter_wan_video_cases():
        global_seq_len = seq_len_from_video(num_frames, height, width)
        for tp_size, sp_size, sp_algorithm in valid_parallel_cases():
            self_q_seq_len, local_heads = local_attention_shape(
                global_seq_len=global_seq_len,
                tp_size=tp_size,
                sp_size=sp_size,
                sp_algorithm=sp_algorithm,
            )
            for backend in _backend_names():
                if backend in ("sla", "sagesla"):
                    attn_kinds = ["self_attn_sla"]
                else:
                    attn_kinds = ["self_attn", "cross_attn_text"]
                    if profile.has_image_context:
                        attn_kinds.append("cross_attn_image")

                for attn_kind in attn_kinds:
                    if attn_kind == "cross_attn_text":
                        q_seq_len, kv_seq_len = self_q_seq_len, WAN_TEXT_TOKENS
                    elif attn_kind == "cross_attn_image":
                        q_seq_len, kv_seq_len = self_q_seq_len, WAN_IMAGE_TOKENS
                    else:
                        q_seq_len = kv_seq_len = self_q_seq_len

                    if not _attention_case_is_reasonable(q_seq_len, kv_seq_len, local_heads, WAN_HEAD_DIM):
                        continue

                    key = (
                        profile.model,
                        profile.task,
                        attn_kind,
                        backend,
                        1,
                        q_seq_len,
                        kv_seq_len,
                        local_heads,
                        WAN_HEAD_DIM,
                        tp_size,
                        sp_size,
                        sp_algorithm,
                    )
                    if local_heads <= 0 or key in seen:
                        continue
                    seen.add(key)
                    test_cases.append(list(key))
    return test_cases


def _get_forced_backend_set(backend):
    from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum

    if backend == "auto":
        return None
    if backend == "flash_attn":
        return {AttentionBackendEnum.FA}
    if backend == "sdpa":
        return {AttentionBackendEnum.TORCH_SDPA}
    if backend == "sage_attn":
        return {AttentionBackendEnum.SAGE_ATTN}
    if backend == "sla":
        return {AttentionBackendEnum.SLA_ATTN}
    if backend == "sagesla":
        return {AttentionBackendEnum.SAGE_SLA_ATTN}
    raise ValueError(f"Unsupported Wan attention backend: {backend}")


def _build_impl(backend, num_heads, head_dim):
    from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
        wrap_attention_impl_forward,
    )
    from sglang.multimodal_gen.runtime.platforms import current_platform
    from sglang.multimodal_gen.utils import resolve_obj_by_qualname

    forced_backend_set = _get_forced_backend_set(backend)
    selected_backend = None if forced_backend_set is None else next(iter(forced_backend_set))
    backend_cls = resolve_obj_by_qualname(
        current_platform.get_attn_backend_cls_str(selected_backend, head_dim, torch.bfloat16)
    )
    impl_cls = backend_cls.get_impl_cls()
    impl = impl_cls(
        num_heads=num_heads,
        head_size=head_dim,
        causal=False,
        softmax_scale=head_dim**-0.5,
        num_kv_heads=num_heads,
        prefix="collector.wan_attention.impl",
    )
    wrap_attention_impl_forward(impl)
    return impl, backend_cls.get_enum().name.lower()


def run_wan_attention(
    model,
    task,
    attn_kind,
    backend,
    batch_size,
    q_seq_len,
    kv_seq_len,
    num_heads,
    head_dim,
    tp_size,
    sp_size,
    sp_algorithm,
    *,
    perf_filename,
    device="cuda:0",
):
    torch.cuda.set_device(device)
    torch.set_default_dtype(torch.bfloat16)
    device_obj = torch.device(device)

    q = torch.randn(batch_size, q_seq_len, num_heads, head_dim, dtype=torch.bfloat16, device=device_obj)
    k = torch.randn(batch_size, kv_seq_len, num_heads, head_dim, dtype=torch.bfloat16, device=device_obj)
    v = torch.randn(batch_size, kv_seq_len, num_heads, head_dim, dtype=torch.bfloat16, device=device_obj)
    impl, actual_backend = _build_impl(backend, num_heads, head_dim)

    def kernel_func():
        return impl.forward(q, k, v, attn_metadata=None)

    from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context

    with set_forward_context(current_timestep=0, attn_metadata=None):
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
                "attn_kind": attn_kind,
                "backend": actual_backend,
                "requested_backend": backend,
                "batch_size": batch_size,
                "q_seq_len": q_seq_len,
                "kv_seq_len": kv_seq_len,
                "num_heads": num_heads,
                "head_dim": head_dim,
                "tp_size": tp_size,
                "sp_size": sp_size,
                "sp_algorithm": sp_algorithm,
                "latency": results["latency_ms"],
            }
        ],
        framework="SGLang",
        version=pkg_resources.get_distribution("sglang").version,
        device_name=torch.cuda.get_device_name(device_obj),
        op_name="wan_attention",
        kernel_source=f"sglang_{actual_backend}",
        perf_filename=perf_filename,
        power_stats=results["power_stats"],
    )


if __name__ == "__main__":
    from collector.registry_types import PerfFile

    for test_case in get_wan_attention_test_cases():
        run_wan_attention(*test_case, perf_filename=PerfFile.WAN_ATTENTION)
