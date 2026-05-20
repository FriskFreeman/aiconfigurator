# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

__compat__ = "sglang>=0.5.10"

import pkg_resources
import torch
import torch.nn.functional as F

from collector.helper import benchmark_with_power, log_perf
from collector.sglang.wan_common import (
    SingleRankGroup,
    WAN_T5_D_FF,
    WAN_T5_D_KV,
    WAN_T5_D_MODEL,
    WAN_T5_NUM_HEADS,
    WAN_T5_VOCAB_SIZE,
    WAN_TEXT_TOKENS,
)


def _valid_t5_group_sizes() -> tuple[int, ...]:
    group_sizes = []
    for group_size in (1, 2, 4, 8, 16, 32):
        if WAN_T5_NUM_HEADS % group_size == 0 and WAN_T5_D_FF % group_size == 0:
            group_sizes.append(group_size)
    return tuple(group_sizes)


def _ensure_single_rank_parallel_state():
    from sglang.multimodal_gen.runtime.distributed import parallel_state

    group = SingleRankGroup()
    for name in ("_TP", "_SP", "_DP", "_PP", "_CFG", "_WORLD"):
        if getattr(parallel_state, name, None) is None:
            setattr(parallel_state, name, group)


def get_wan_t5_test_cases():
    seq_lens = [128, 256, WAN_TEXT_TOKENS]
    test_cases = []
    seen = set()
    for group_size in _valid_t5_group_sizes():
        local_heads = WAN_T5_NUM_HEADS // group_size
        local_d_ff = WAN_T5_D_FF // group_size
        for seq_len in seq_lens:
            cases = [
                ["embedding", 1, seq_len, WAN_T5_D_MODEL, WAN_T5_NUM_HEADS, WAN_T5_D_KV, WAN_T5_D_FF],
                ["rmsnorm", 1, seq_len, WAN_T5_D_MODEL, WAN_T5_NUM_HEADS, WAN_T5_D_KV, WAN_T5_D_FF],
                ["attention_compute_bias_softmax", 1, seq_len, WAN_T5_D_MODEL, local_heads, WAN_T5_D_KV, WAN_T5_D_FF],
                ["ffn_gated_act_mul", 1, seq_len, WAN_T5_D_MODEL, WAN_T5_NUM_HEADS, WAN_T5_D_KV, local_d_ff],
            ]
            for case in cases:
                key = tuple(case)
                if key in seen:
                    continue
                seen.add(key)
                test_cases.append(case)
    return test_cases


def run_wan_t5(
    op_name,
    batch_size,
    seq_len,
    d_model,
    num_heads,
    d_kv,
    d_ff,
    *,
    perf_filename,
    device="cuda:0",
):
    torch.cuda.set_device(device)
    torch.set_default_dtype(torch.bfloat16)
    device_obj = torch.device(device)
    _ensure_single_rank_parallel_state()

    if op_name == "embedding":
        from sglang.multimodal_gen.runtime.layers.vocab_parallel_embedding import VocabParallelEmbedding

        module = VocabParallelEmbedding(
            WAN_T5_VOCAB_SIZE,
            d_model,
            org_num_embeddings=WAN_T5_VOCAB_SIZE,
            tp_group=SingleRankGroup(),
        ).to(device_obj, dtype=torch.bfloat16)
        input_ids = torch.randint(0, WAN_T5_VOCAB_SIZE, (batch_size, seq_len), device=device_obj)

        def kernel_func():
            return module(input_ids)

        kernel_source = "sglang_vocab_parallel_embedding"
    elif op_name == "rmsnorm":
        from sglang.multimodal_gen.runtime.layers.layernorm import RMSNorm

        module = RMSNorm(d_model, eps=1e-6).to(device_obj, dtype=torch.bfloat16)
        hidden_states = torch.randn(batch_size, seq_len, d_model, dtype=torch.bfloat16, device=device_obj)

        def kernel_func():
            return module(hidden_states)

        kernel_source = "sglang_rmsnorm"
    elif op_name == "attention_compute_bias_softmax":
        from sglang.multimodal_gen.runtime.models.encoders.t5 import T5MultiHeadAttention

        module = T5MultiHeadAttention().to(device_obj)
        q = torch.randn(batch_size, seq_len, num_heads, d_kv, dtype=torch.bfloat16, device=device_obj)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        attn_bias = torch.randn(batch_size, num_heads, seq_len, seq_len, dtype=torch.bfloat16, device=device_obj)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device_obj)
        masked_bias = attn_bias.masked_fill(
            attention_mask.view(batch_size, 1, 1, seq_len) == 0,
            torch.finfo(q.dtype).min,
        )

        def kernel_func():
            return module(q, k, v, masked_bias)

        kernel_source = "sglang_t5_einsum_softmax"
    elif op_name == "ffn_gated_act_mul":
        hidden_gelu = torch.randn(batch_size, seq_len, d_ff, dtype=torch.bfloat16, device=device_obj)
        hidden_linear = torch.randn_like(hidden_gelu)

        def kernel_func():
            return F.gelu(hidden_gelu, approximate="tanh") * hidden_linear

        kernel_source = "torch_gelu_tanh_mul"
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
                "d_model": d_model,
                "num_heads": num_heads,
                "d_kv": d_kv,
                "d_ff": d_ff,
                "latency": results["latency_ms"],
            }
        ],
        framework="SGLang",
        version=pkg_resources.get_distribution("sglang").version,
        device_name=torch.cuda.get_device_name(device_obj),
        op_name=f"wan_t5_{op_name}",
        kernel_source=kernel_source,
        perf_filename=perf_filename,
        power_stats=results["power_stats"],
    )


if __name__ == "__main__":
    from collector.registry_types import PerfFile

    for test_case in get_wan_t5_test_cases():
        run_wan_t5(*test_case, perf_filename=PerfFile.WAN_T5)
