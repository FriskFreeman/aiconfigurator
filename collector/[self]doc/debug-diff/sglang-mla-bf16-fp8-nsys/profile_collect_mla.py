#!/usr/bin/env python3
"""Profile the exact RadixAttention boundary constructed by collect_mla.py."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from collector.sglang import collect_mla


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("prefill", "decode"), required=True)
    parser.add_argument("--kv-cache-dtype", choices=("bf16", "fp8"), required=True)
    parser.add_argument("--input-len", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=128)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--execution", choices=("eager", "graph"), default="graph")
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = quantile * (len(ordered) - 1)
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    fraction = pos - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(0)

    result: dict[str, object] = {
        "stage": args.stage,
        "kv_cache_dtype": args.kv_cache_dtype,
        "input_len": args.input_len,
        "batch_size": args.batch_size,
        "num_heads": args.num_heads,
        "tp_size": args.tp_size,
        "execution": args.execution,
        "warmups": args.warmups,
        "iterations": args.iterations,
        "profile": args.profile,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
    }

    def profiled_benchmark_layer(layer, forward_batch, q, k, v, q_rope, k_rope, **kwargs):
        def kernel_func():
            extra_kwargs = dict(kwargs)
            if q_rope is not None:
                extra_kwargs["q_rope"] = q_rope
            if k_rope is not None:
                extra_kwargs["k_rope"] = k_rope
            return layer(q, k, v, forward_batch, **extra_kwargs)

        torch.cuda.synchronize()
        for _ in range(args.warmups):
            kernel_func()
        torch.cuda.synchronize()

        graph = None
        if args.execution == "graph":
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                kernel_func()
            torch.cuda.synchronize()
            for _ in range(args.warmups):
                graph.replay()
            torch.cuda.synchronize()

        def invoke():
            if graph is None:
                kernel_func()
            else:
                graph.replay()

        starts = [torch.cuda.Event(enable_timing=True) for _ in range(args.iterations)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(args.iterations)]
        tag_base = (
            f"aic_collect_mla::{args.stage}::{args.kv_cache_dtype}::"
            f"b{args.batch_size}_s{args.input_len}_h{args.num_heads}_tp{args.tp_size}::"
            f"{args.execution}"
        )

        if args.profile:
            torch.cuda.profiler.start()
        torch.cuda.nvtx.range_push(tag_base)
        for index in range(args.iterations):
            torch.cuda.nvtx.range_push(f"{tag_base}::iter_{index:03d}")
            starts[index].record()
            invoke()
            ends[index].record()
            torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()
        if args.profile:
            torch.cuda.profiler.stop()

        samples_ms = [start.elapsed_time(end) for start, end in zip(starts, ends)]
        result.update(
            {
                "samples_ms": samples_ms,
                "latency_mean_ms": statistics.fmean(samples_ms),
                "latency_median_ms": statistics.median(samples_ms),
                "latency_p10_ms": percentile(samples_ms, 0.10),
                "latency_p90_ms": percentile(samples_ms, 0.90),
                "memory_allocated_bytes": torch.cuda.memory_allocated(0),
                "memory_reserved_bytes": torch.cuda.memory_reserved(0),
                "input_tensors": {
                    "q": {"shape": list(q.shape), "dtype": str(q.dtype)},
                    "k": {"shape": list(k.shape), "dtype": str(k.dtype)},
                    "v": {"shape": list(v.shape), "dtype": str(v.dtype)},
                    "q_rope": None
                    if q_rope is None
                    else {"shape": list(q_rope.shape), "dtype": str(q_rope.dtype)},
                    "k_rope": None
                    if k_rope is None
                    else {"shape": list(k_rope.shape), "dtype": str(k_rope.dtype)},
                },
                "save_kv_cache": kwargs.get("save_kv_cache", True),
                "forward_mode": str(forward_batch.forward_mode),
                "kv_pool_dtype": str(forward_batch.token_to_kv_pool.dtype),
            }
        )
        return statistics.fmean(samples_ms), None

    collect_mla.benchmark_layer = profiled_benchmark_layer
    dtype = torch.bfloat16 if args.kv_cache_dtype == "bf16" else torch.float8_e4m3fn
    collect_mla.run_mla(
        input_len=args.input_len,
        batch_size=args.batch_size,
        output_len=1,
        kv_cache_dtype=dtype,
        num_heads=args.num_heads,
        world_size=args.tp_size,
        tp_size=args.tp_size,
        tokens_per_block=64,
        warming_up=args.warmups,
        test_ite=args.iterations,
        is_context_phase=args.stage == "prefill",
        perf_filename=str(output_dir / "collector_boundary_perf.csv"),
        device="cuda:0",
    )

    result["selected_backend"] = collect_mla._select_default_mla_backend()
    (output_dir / "measurement.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
