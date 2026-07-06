#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "analysis/wideep_module_mla_makespan_inspect"
RAW_RUNS = ROOT / "bench_data/h100_sxm/sglang/v0.5.9/deepseek_v3_mla/raw_runs"

CASES = [
    (
        "target",
        "req1_equal_fresh_regular_b1_f8192_p0",
        "20260618_105655_prefill_stage1_req1_equal_fresh_regular_b1_f8192_p0_b1_fvar8192_pvar0_layers5_backend_auto_cg_off_pcg_off_tp1_marker_on_profile_nsys",
    ),
    (
        "target",
        "req1_equal_fresh_irregular_b1_f6000_p0",
        "20260618_110607_prefill_stage1_req1_equal_fresh_irregular_b1_f6000_p0_b1_fvar6000_pvar0_layers5_backend_auto_cg_off_pcg_off_tp1_marker_on_profile_nsys",
    ),
    (
        "target",
        "req1_equal_fresh_irregular_b1_f4500_p0",
        "20260618_110344_prefill_stage1_req1_equal_fresh_irregular_b1_f4500_p0_b1_fvar4500_pvar0_layers5_backend_auto_cg_off_pcg_off_tp1_marker_on_profile_nsys",
    ),
    (
        "reference",
        "req1_equal_fresh_regular_b2_f4096_p0",
        "20260618_105438_prefill_stage1_req1_equal_fresh_regular_b2_f4096_p0_b2_fvar4096-4096_pvar0-0_layers5_backend_auto_cg_off_pcg_off_tp1_marker_on_profile_nsys",
    ),
    (
        "reference",
        "req1_equal_fresh_regular_b4_f2048_p0",
        "20260618_105215_prefill_stage1_req1_equal_fresh_regular_b4_f2048_p0_b4_fvar2048-2048-2048-2048_pvar0-0-0-0_layers5_backend_auto_cg_off_pcg_off_tp1_marker_on_profile_nsys",
    ),
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def string_map(conn: sqlite3.Connection) -> dict[int, str]:
    return {int(row[0]): str(row[1]) for row in conn.execute("SELECT id, value FROM StringIds")}


def thread_name_map(conn: sqlite3.Connection, strings: dict[int, str]) -> dict[int, str]:
    return {
        int(tid): strings.get(int(name_id), str(name_id))
        for name_id, _prio, tid in conn.execute("SELECT nameId, priority, globalTid FROM ThreadNames")
    }


def table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
        ).fetchone()
        is not None
    )


def get_self_attn_window(conn: sqlite3.Connection) -> tuple[int, int]:
    candidates = []
    for start, end, text, json_text in conn.execute(
        """
        SELECT start, end, text, jsonText
        FROM NVTX_EVENTS
        WHERE end IS NOT NULL
        ORDER BY start
        """
    ):
        payload = str(text or json_text or "")
        if (
            "model.model.layers.0.self_attn" in payload
            and all(
                marker not in payload
                for marker in [
                    "fused_qkv",
                    "q_a_layernorm",
                    "q_b_proj",
                    "kv_a_layernorm",
                    "rotary_emb",
                    "kv_b_proj",
                    "attn_mha",
                    "attn_mqa",
                    "o_proj",
                ]
            )
        ):
            candidates.append((int(start), int(end)))
    if not candidates:
        raise RuntimeError("layer0 self_attn NVTX range not found")
    return candidates[0]


def kernels_in_window(
    conn: sqlite3.Connection,
    strings: dict[int, str],
    start_ns: int,
    end_ns: int,
) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT start, end, shortName, correlationId, gridX, gridY, gridZ, blockX, blockY, blockZ
        FROM CUPTI_ACTIVITY_KIND_KERNEL
        WHERE start >= ? AND start <= ?
        ORDER BY start
        """,
        (start_ns - 2_000_000, end_ns + 5_000_000),
    ).fetchall()
    out = []
    for start, end, name_id, corr, gx, gy, gz, bx, by, bz in rows:
        out.append(
            {
                "start_ns": int(start),
                "end_ns": int(end),
                "duration_ms": (int(end) - int(start)) / 1e6,
                "name": strings.get(int(name_id), str(name_id)),
                "correlation_id": corr,
                "grid": [gx, gy, gz],
                "block": [bx, by, bz],
            }
        )
    return out


def runtime_events(
    conn: sqlite3.Connection,
    strings: dict[int, str],
    start_ns: int,
    end_ns: int,
) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT start, end, nameId, globalTid, correlationId
        FROM CUPTI_ACTIVITY_KIND_RUNTIME
        WHERE start < ? AND end > ?
        ORDER BY start
        """,
        (end_ns, start_ns),
    ).fetchall()
    return [
        {
            "start_ns": int(start),
            "end_ns": int(end),
            "duration_ms": (int(end) - int(start)) / 1e6,
            "name": strings.get(int(name_id), str(name_id)),
            "global_tid": tid,
            "correlation_id": corr,
        }
        for start, end, name_id, tid, corr in rows
    ]


def profiler_events(
    conn: sqlite3.Connection,
    strings: dict[int, str],
    threads: dict[int, str],
    start_ns: int,
    end_ns: int,
) -> list[dict[str, Any]]:
    if not table_exists(conn, "PROFILER_OVERHEAD"):
        return []
    rows = conn.execute(
        """
        SELECT start, end, nameId, globalTid
        FROM PROFILER_OVERHEAD
        WHERE start < ? AND end > ?
        ORDER BY start
        """,
        (end_ns, start_ns),
    ).fetchall()
    return [
        {
            "start_ns": int(start),
            "end_ns": int(end),
            "duration_ms": (int(end) - int(start)) / 1e6,
            "name": strings.get(int(name_id), str(name_id)),
            "global_tid": tid,
            "thread_name": threads.get(int(tid), ""),
        }
        for start, end, name_id, tid in rows
    ]


def find_first(events: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    for event in events:
        if event["name"] == name:
            return event
    return None


def summarize_case(group: str, tag: str, run_name: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    run_dir = RAW_RUNS / run_name
    conn = sqlite3.connect(run_dir / "nsys/report.sqlite")
    strings = string_map(conn)
    threads = thread_name_map(conn, strings)
    self_start, self_end = get_self_attn_window(conn)
    kernels = kernels_in_window(conn, strings, self_start, self_end)
    rope = find_first(kernels, "BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel")
    set_mla = find_first(kernels, "set_mla_kv_buffer_kernel")
    if rope is None or set_mla is None:
        raise RuntimeError(f"missing rope/set_mla kernel for {tag}")

    gap_start = rope["end_ns"]
    gap_end = set_mla["start_ns"]
    gap_ms = (gap_end - gap_start) / 1e6
    runtimes = runtime_events(conn, strings, gap_start, gap_end)
    profilers = profiler_events(conn, strings, threads, gap_start, gap_end)
    module_loads = [event for event in runtimes if event["name"] == "cuModuleLoadData"]
    launch_events = [
        event
        for event in runtimes
        if event["name"] in {"cuLaunchKernelEx", "cudaLaunchKernel_v7000", "cuLaunchKernel"}
    ]
    compiler_threads = sorted(
        {
            event["thread_name"]
            for event in profilers
            if event["thread_name"] in {"ptxas", "llvm-worker-0", "llvm-worker-1"}
        }
    )

    meta = json.loads((run_dir / "run_meta.json").read_text())
    summary = {
        "group": group,
        "tag": tag,
        "run_instance_name": run_name,
        "batch_size": meta.get("batch_size"),
        "fresh_lens": json.dumps(meta.get("fresh_lens"), separators=(",", ":")),
        "prefix_lens": json.dumps(meta.get("prefix_lens"), separators=(",", ":")),
        "warmup_runs": meta.get("warmup_runs"),
        "context_length": meta.get("context_length"),
        "attention_backend": meta.get("attention_backend"),
        "chunked_prefix_cache_threshold": meta.get("effective_chunked_prefix_cache_threshold"),
        "self_attn_host_ms": (self_end - self_start) / 1e6,
        "rope_to_set_mla_gap_ms": gap_ms,
        "rope_kernel_ms": rope["duration_ms"],
        "set_mla_kernel_ms": set_mla["duration_ms"],
        "set_mla_grid": json.dumps(set_mla["grid"], separators=(",", ":")),
        "module_load_count_in_gap": len(module_loads),
        "first_module_load_offset_ms": (
            (module_loads[0]["start_ns"] - gap_start) / 1e6 if module_loads else ""
        ),
        "first_launch_offset_ms": (
            (launch_events[0]["start_ns"] - gap_start) / 1e6 if launch_events else ""
        ),
        "profiler_event_count_in_gap": len(profilers),
        "compiler_threads_in_gap": ",".join(compiler_threads),
        "has_ptxas_or_llvm_profiler_events": bool(compiler_threads),
        "nsys_has_cpu_sampling": table_exists(conn, "SAMPLING_CALLCHAINS")
        or table_exists(conn, "SAMPLING_EVENTS"),
    }

    seq_rows = []
    for kernel in kernels:
        if self_start - 1_000_000 <= kernel["start_ns"] <= set_mla["end_ns"] + 1_000_000:
            seq_rows.append(
                {
                    "group": group,
                    "tag": tag,
                    "kernel_name": kernel["name"],
                    "offset_start_ms": (kernel["start_ns"] - self_start) / 1e6,
                    "offset_end_ms": (kernel["end_ns"] - self_start) / 1e6,
                    "duration_ms": kernel["duration_ms"],
                    "grid": json.dumps(kernel["grid"], separators=(",", ":")),
                    "block": json.dumps(kernel["block"], separators=(",", ":")),
                    "correlation_id": kernel["correlation_id"],
                }
            )
    return summary, seq_rows


def render_report(rows: list[dict[str, Any]]) -> str:
    def fmt(value: Any, digits: int = 3) -> str:
        if value == "":
            return ""
        if isinstance(value, float):
            return f"{value:.{digits}f}"
        return str(value)

    lines = [
        "# Prefill 首层 Rope 到 set_mla_kv_buffer 大 Gap 溯源报告",
        "",
        "## 结论",
        "",
        "- 异常 gap 发生在 DeepSeek MHA prefill prepare 阶段的 `rotary_emb` 之后、`set_mla_kv_buffer_kernel` 之前；最终的 `set_mla_kv_buffer_kernel` 本身只有约 0.02ms，不是耗时来源。",
        "- `b1_f4500/b1_f6000/b1_f8192` 的 gap 中可见 `ptxas`、`llvm-worker-*`、`cuModuleLoadData` 和 profiler/TLS 初始化事件；正常的 `b2_f4096/b4_f2048` 对照没有 `cuModuleLoadData`，rope 后几乎立即 launch `set_mla_kv_buffer_kernel`。",
        "- 因此当前证据更支持：首层大 gap 是 Triton/CUDA module 的 lazy JIT 编译或模块加载污染，触发点是 SGLang 写 MLA latent KV cache 的 Triton kernel，而不是 KV cache prefix 加载、FA attention kernel 或 DeepGEMM kernel 慢。",
        "- 这份 nsys 是 `--sample=none --cpuctxsw=none`，没有 CPU sampling/backtrace；所以还不能从 sqlite 直接证明 100% CPU 的完整函数栈。若要把“CPU 忙在 ptxas/Triton 编译”从高置信推断升级为直接证明，需要补采 `--sample=cpu` 或打开更完整 CPU stack。",
        "",
        "## 关键数据",
        "",
        "| group | tag | batch/fresh/prefix | warmup | self_attn host ms | rope->set_mla gap ms | set_mla kernel ms | module load in gap | first module load offset ms | compiler threads |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        shape = f"b{row['batch_size']} fresh={row['fresh_lens']} prefix={row['prefix_lens']}"
        lines.append(
            "| {group} | {tag} | {shape} | {warmup} | {host} | {gap} | {kernel} | {loads} | {load_offset} | {threads} |".format(
                group=row["group"],
                tag=row["tag"],
                shape=shape,
                warmup=row["warmup_runs"],
                host=fmt(row["self_attn_host_ms"]),
                gap=fmt(row["rope_to_set_mla_gap_ms"]),
                kernel=fmt(row["set_mla_kernel_ms"], 6),
                loads=row["module_load_count_in_gap"],
                load_offset=fmt(row["first_module_load_offset_ms"]),
                threads=row["compiler_threads_in_gap"] or "",
            )
        )
    lines.extend(
        [
            "",
            "## SGLang 0.5.9 执行路径对照",
            "",
            "- `deepseek_common/attention_backend_handler.py` 中，FA3/FlashInfer 等后端在 extend prefill 且 `sum_extend_prefix_lens == 0` 时走 `MHA_ONE_SHOT`，本次 `prefix_lens=[0]` 正落在这个分支。",
            "- 源码依据：`attention_backend_handler.py:81-94` 在 `forward_mode.is_extend_without_speculative()` 且 `(sum_extend_prefix_lens >= threshold or sum_extend_prefix_lens == 0)` 时选择 MHA one-shot 或 chunked-kv；本批次 prefix 为 0，因此进入 MHA prefill 路径。",
            "- `deepseek_common/attention_forward_methods/forward_mha.py:209-214` 的 `forward_normal_prepare` 顺序是：`rotary_emb` -> `_set_mla_kv_buffer(...)`；随后 `forward_mha.py:235-246` 才执行 `kv_b_proj` 和拼接 MHA K。",
            "- `_set_mla_kv_buffer` 的 CUDA 分支在 `forward_mha.py:389-400` 调用 `forward_batch.token_to_kv_pool.set_mla_kv_buffer(self.attn_mha, out_cache_loc, kv_a.unsqueeze(1), k_pe)`。",
            "- `mem_cache/memory_pool.py:1509-1548` 的 `set_mla_kv_buffer` 最终调用 `mem_cache/utils.py` 里的 `set_mla_kv_buffer_triton(...)`。",
            "- `mem_cache/utils.py:25-109` 定义了 `@triton.jit` 的 `set_mla_kv_buffer_kernel`；`set_mla_kv_buffer_triton` 的 grid 为 `(loc.numel(), cdiv(nope_dim + rope_dim, 128))`。DeepSeek V3 BF16 KV cache 下 `total_dim=512+64=576`，所以异常 b1_f8192 和正常 b2_f4096/b4_f2048 都看到 `grid=(8192,5,1)`。",
            "",
            "## 为什么不是 KV cache 加载",
            "",
            "- 这些异常 case 的 `prefix_lens=[0]`，没有 prefix KV 命中加载；当前 `_set_mla_kv_buffer` 是把本轮新 token 的 latent cache 写入 KV pool。",
            "- 真正写入的 GPU kernel `set_mla_kv_buffer_kernel` 只有约 0.019ms 到 0.026ms；如果是 KV 写入/加载本身慢，耗时应体现在该 kernel 或 memcpy 上，而不是 kernel 启动前 200ms 级 host 空窗。",
            "- gap 内没有 CUDA memcpy 或长 kernel，只有 gap 尾部出现 `cuModuleLoadData` 和后续 launch。",
            "",
            "## 为什么 warmup 后仍可能出现",
            "",
            "- 当前脚本确实在 `start_profile` 前执行了同长度 warmup；`stage.txt` 记录为 `before_warmup_generate_0 -> after_warmup_generate_0 -> before_start_profile`。",
            "- 但正式 profile run 中仍出现 `cuModuleLoadData`，说明 warmup 没有覆盖或没有复用正式 run 的这个 CUDA module 实例。可疑因素包括：Triton specialization/module load 的 lazy 行为、profile/CUPTI attach 后触发的新模块加载、或不同 session/request 生命周期导致的首次 module load 重现。",
            "- 目前证据能证明 `cuModuleLoadData` 发生在正式 run 的 gap 尾部，不能单靠现有 nsys 证明 warmup 未触发编译的精确原因；这需要 CPU sampling 或增加应用侧 Triton cache/compile 日志进一步确认。",
            "",
            "## OSRT Threads 与 CPU 100% 的解释",
            "",
            "- `b1_f8192` 的 gap 窗口内，OSRT 表能看到很长的 `epoll_wait/epoll_pwait/pthread_cond_timedwait/clock_nanosleep`，对应线程名包括 `ZMQbg/IO/*`、`python`、`pt_nccl_*`、`pt_tcpstore_uv` 等；这些事件覆盖窗口，但调用栈多是 Gloo/ZMQ/NCCL watchdog 或 TCPStore 后台等待，不是 scheduler 主线程下一 kernel launch 的直接原因。",
            "- scheduler 主线程在 gap 起点附近只有极短的 `stat64/mmap64/munmap` 等 OSRT 调用；随后直到 gap 尾部的 `cuModuleLoadData/cuLaunchKernelEx`，OSRT 表并不记录 CPU busy 的函数栈。这与 `--sample=none --cpuctxsw=none` 的采集配置一致：CPU 忙等或编译计算不会以 sampling callchain 形式出现在 sqlite 中。",
            "- `PROFILER_OVERHEAD` 表在异常用例 gap 中记录到 `ptxas`、`llvm-worker-0/1`、`TLS allocation`、`OS runtime libraries profiling initialization`、`In-process plugins initialization`；正常 b2/b4 对照没有同样的 `cuModuleLoadData` 和 compiler thread 组合。因此，你在 Nsight UI 中看到的 CPU 高占用，更可能对应 Triton/ptxas 编译或 module load 前的 host 侧工作，而不是 OSRT wait 事件本身。",
            "- 因为当前 sqlite 没有 CPU sampling，报告把这一点定性为高置信推断，而不是已经精确定位到某个 Python/C++ 函数的直接栈证据。",
            "",
            "## 为什么只在部分用例出现",
            "",
            "- 异常集中于单请求长序列 `b1_f4500/b1_f6000/b1_f8192`；正常对照 `b2_f4096/b4_f2048` 虽然总 fresh token 同为 8192，但没有 module load，也没有大 gap。",
            "- 这排除了“总 token 数大必然导致 gap”的解释，也排除了 `set_mla_kv_buffer_kernel` 运行时长随总 token 增大造成的解释。",
            "- 更可能是单请求长序列路径在正式 profile run 中触发了某个 Triton/CUDA module lazy load；由于现有 nsys 没有 CPU sample，暂不能确定触发差异来自 Triton cache key、session 生命周期、CUPTI profiler attach，还是 SGLang 内部请求路径差异。",
            "",
            "## 建议验证",
            "",
            "- 对 `b1_f8192_p0` 重跑一次 nsys，打开 `--sample=cpu --cpuctxsw=process-tree` 或至少 CPU sampling，确认 gap 内 scheduler 线程/ptxas/llvm-worker 的调用栈。",
            "- 显式设置并持久化 `TRITON_CACHE_DIR`，在 warmup 后检查正式 run 是否仍有 `cuModuleLoadData`。",
            "- 增加一个 profile 前的 `torch.cuda.synchronize()`，并可选对 `set_mla_kv_buffer_triton` 做一次显式预触发，以验证是否能消除 rope->set_mla gap。",
            "",
            "## 产物",
            "",
            "- `prefill_first_layer_gap_summary.csv`：异常/对照 case 的首层 gap 汇总。",
            "- `prefill_first_layer_kernel_sequence.csv`：首层 self_attn 起点到 `set_mla_kv_buffer_kernel` 附近的 kernel 序列。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    summaries: list[dict[str, Any]] = []
    sequences: list[dict[str, Any]] = []
    for group, tag, run_name in CASES:
        summary, sequence = summarize_case(group, tag, run_name)
        summaries.append(summary)
        sequences.extend(sequence)

    write_csv(
        OUT_DIR / "prefill_first_layer_gap_summary.csv",
        summaries,
        [
            "group",
            "tag",
            "run_instance_name",
            "batch_size",
            "fresh_lens",
            "prefix_lens",
            "warmup_runs",
            "context_length",
            "attention_backend",
            "chunked_prefix_cache_threshold",
            "self_attn_host_ms",
            "rope_to_set_mla_gap_ms",
            "rope_kernel_ms",
            "set_mla_kernel_ms",
            "set_mla_grid",
            "module_load_count_in_gap",
            "first_module_load_offset_ms",
            "first_launch_offset_ms",
            "profiler_event_count_in_gap",
            "compiler_threads_in_gap",
            "has_ptxas_or_llvm_profiler_events",
            "nsys_has_cpu_sampling",
        ],
    )
    write_csv(
        OUT_DIR / "prefill_first_layer_kernel_sequence.csv",
        sequences,
        [
            "group",
            "tag",
            "kernel_name",
            "offset_start_ms",
            "offset_end_ms",
            "duration_ms",
            "grid",
            "block",
            "correlation_id",
        ],
    )
    (OUT_DIR / "Prefill首层大Gap溯源报告.md").write_text(
        render_report(summaries)
    )


if __name__ == "__main__":
    main()
