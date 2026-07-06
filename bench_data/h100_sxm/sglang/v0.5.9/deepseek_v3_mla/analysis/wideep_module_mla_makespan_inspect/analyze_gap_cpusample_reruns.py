#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "analysis/wideep_module_mla_makespan_inspect"
TARGET_ROOT = ROOT / ".self/task-localbench/tmp/tmp-prefill-gap-cpusample-cap"
REF_ROOT = ROOT / ".self/task-localbench/tmp/tmp-prefill-gap-cpusample-cap-ref"


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
        ).fetchone()
        is not None
    )


def string_map(conn: sqlite3.Connection) -> dict[int, str]:
    return {int(k): v for k, v in conn.execute("SELECT id, value FROM StringIds")}


def thread_map(conn: sqlite3.Connection, strings: dict[int, str]) -> dict[int, str]:
    if not table_exists(conn, "ThreadNames"):
        return {}
    return {
        int(tid): strings.get(int(name_id), "")
        for name_id, _priority, tid in conn.execute(
            "SELECT nameId, priority, globalTid FROM ThreadNames"
        )
    }


def first_layer_self_attn_window(conn: sqlite3.Connection) -> tuple[int, int]:
    for start, end, text, json_text in conn.execute(
        """
        SELECT start, end, text, jsonText
        FROM NVTX_EVENTS
        WHERE end IS NOT NULL
        ORDER BY start
        """
    ):
        payload = str(text or json_text or "")
        if "model.model.layers.0.self_attn" not in payload:
            continue
        if any(
            marker in payload
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
        ):
            continue
        return int(start), int(end)
    raise RuntimeError("layer0 self_attn NVTX range not found")


def kernels_in_window(
    conn: sqlite3.Connection,
    strings: dict[int, str],
    start_ns: int,
    end_ns: int,
) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT start, end, shortName, correlationId, gridX, gridY, gridZ
        FROM CUPTI_ACTIVITY_KIND_KERNEL
        WHERE start >= ? AND start <= ?
        ORDER BY start
        """,
        (start_ns - 2_000_000, end_ns + 5_000_000),
    ).fetchall()
    return [
        {
            "start_ns": int(start),
            "end_ns": int(end),
            "duration_ms": (int(end) - int(start)) / 1e6,
            "name": strings.get(int(name_id), str(name_id)),
            "correlation_id": corr,
            "grid": [gx, gy, gz],
        }
        for start, end, name_id, corr, gx, gy, gz in rows
    ]


def first_named_kernel(kernels: list[dict[str, Any]], name: str) -> dict[str, Any]:
    for kernel in kernels:
        if kernel["name"] == name:
            return kernel
    raise RuntimeError(f"kernel not found: {name}")


def runtime_events(
    conn: sqlite3.Connection,
    strings: dict[int, str],
    threads: dict[int, str],
    start_ns: int,
    end_ns: int,
) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT start, end, nameId, correlationId, globalTid
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
            "offset_start_ms": (int(start) - start_ns) / 1e6,
            "offset_end_ms": (int(end) - start_ns) / 1e6,
            "duration_ms": (int(end) - int(start)) / 1e6,
            "name": strings.get(int(name_id), str(name_id)),
            "correlation_id": corr,
            "thread": threads.get(int(tid), ""),
        }
        for start, end, name_id, corr, tid in rows
    ]


def sampling_summary(
    conn: sqlite3.Connection,
    strings: dict[int, str],
    threads: dict[int, str],
    start_ns: int,
    end_ns: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not table_exists(conn, "COMPOSITE_EVENTS") or not table_exists(
        conn, "SAMPLING_CALLCHAINS"
    ):
        return (
            {
                "sampling_table_present": False,
                "sample_count": 0,
                "scheduler_samples": 0,
                "ptxas_samples": 0,
                "llvm_worker_samples": 0,
                "triton_module_frame_count": 0,
                "ptxas_module_frame_count": 0,
                "top_threads": "",
                "top_modules": "",
                "top_symbols": "",
            },
            [],
        )

    samples = conn.execute(
        """
        SELECT id, start, globalTid
        FROM COMPOSITE_EVENTS
        WHERE start >= ? AND start <= ?
        ORDER BY start
        """,
        (start_ns, end_ns),
    ).fetchall()
    thread_counts: Counter[str] = Counter()
    module_counts: Counter[str] = Counter()
    symbol_counts: Counter[str] = Counter()
    sample_rows: list[dict[str, Any]] = []
    for sample_id, sample_start, tid in samples:
        thread_name = threads.get(int(tid), "")
        thread_counts[thread_name] += 1
        frames = conn.execute(
            """
            SELECT symbol, module, stackDepth
            FROM SAMPLING_CALLCHAINS
            WHERE id = ?
            ORDER BY stackDepth
            """,
            (sample_id,),
        ).fetchall()
        for symbol_id, module_id, depth in frames:
            symbol = strings.get(int(symbol_id), str(symbol_id))
            module = strings.get(int(module_id), str(module_id))
            module_counts[module] += 1
            symbol_counts[symbol] += 1
            if len(sample_rows) < 80 and (
                "triton" in module
                or "ptxas" in module
                or "llvm" in symbol
                or "mlir" in symbol
                or "parseIR" in symbol
            ):
                sample_rows.append(
                    {
                        "sample_id": sample_id,
                        "offset_ms": (int(sample_start) - start_ns) / 1e6,
                        "thread": thread_name,
                        "stack_depth": depth,
                        "symbol": symbol[:500],
                        "module": module,
                    }
                )

    def top(counter: Counter[str], n: int = 8) -> str:
        return json.dumps(counter.most_common(n), ensure_ascii=False)

    return (
        {
            "sampling_table_present": True,
            "sample_count": len(samples),
            "scheduler_samples": sum(
                count for name, count in thread_counts.items() if "sglang::schedul" in name
            ),
            "ptxas_samples": thread_counts.get("ptxas", 0),
            "llvm_worker_samples": sum(
                count for name, count in thread_counts.items() if name.startswith("llvm-worker")
            ),
            "triton_module_frame_count": sum(
                count for name, count in module_counts.items() if "triton/_C/libtriton.so" in name
            ),
            "ptxas_module_frame_count": sum(
                count
                for name, count in module_counts.items()
                if "triton/backends/nvidia/bin/ptxas" in name
            ),
            "top_threads": top(thread_counts),
            "top_modules": top(module_counts),
            "top_symbols": top(symbol_counts),
        },
        sample_rows,
    )


def set_mla_cache_variants(run_dir: Path) -> dict[str, Any]:
    variants = sorted(
        p.parent for p in (run_dir / "triton_cache").glob("*/set_mla_kv_buffer_kernel.ttir")
    )
    names = [p.name for p in variants]
    diff_hint = ""
    if len(variants) >= 2:
        a = (variants[0] / "set_mla_kv_buffer_kernel.ttir").read_text()
        b = (variants[1] / "set_mla_kv_buffer_kernel.ttir").read_text()
        if "{tt.divisibility = 16 : i32} loc(\"loc_ptr\"" in a and (
            "{tt.divisibility = 16 : i32} loc(\"loc_ptr\"" not in b
        ):
            diff_hint = "loc_ptr_divisible16_to_unannotated"
        elif "{tt.divisibility = 16 : i32} loc(\"loc_ptr\"" in b and (
            "{tt.divisibility = 16 : i32} loc(\"loc_ptr\"" not in a
        ):
            diff_hint = "loc_ptr_unannotated_to_divisible16"
        else:
            diff_hint = "multiple_set_mla_specializations"
    elif len(variants) == 1:
        text = (variants[0] / "set_mla_kv_buffer_kernel.ttir").read_text()
        diff_hint = (
            "single_loc_ptr_divisible16"
            if "{tt.divisibility = 16 : i32} loc(\"loc_ptr\"" in text
            else "single_loc_ptr_unannotated"
        )
    return {
        "set_mla_cache_variant_count": len(variants),
        "set_mla_cache_variants": json.dumps(names, ensure_ascii=False),
        "set_mla_cache_diff_hint": diff_hint,
    }


def analyze_run(run_dir: Path, group: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    conn = sqlite3.connect(run_dir / "nsys/report.sqlite")
    strings = string_map(conn)
    threads = thread_map(conn, strings)
    self_start, self_end = first_layer_self_attn_window(conn)
    kernels = kernels_in_window(conn, strings, self_start, self_end)
    rope = first_named_kernel(
        kernels, "BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel"
    )
    set_mla = first_named_kernel(kernels, "set_mla_kv_buffer_kernel")
    gap_start = rope["end_ns"]
    gap_end = set_mla["start_ns"]
    gap_ms = (gap_end - gap_start) / 1e6
    runtimes = runtime_events(conn, strings, threads, gap_start, gap_end)
    sample_summary, sample_rows = sampling_summary(
        conn, strings, threads, gap_start, gap_end
    )
    module_loads = [event for event in runtimes if event["name"] == "cuModuleLoadData"]
    launches = [event for event in runtimes if event["name"].startswith("cuLaunch")]
    meta = json.loads((run_dir / "run_meta.json").read_text())
    row = {
        "group": group,
        "run_name": run_dir.name,
        "tag": meta.get("tag", run_dir.name),
        "batch_size": meta.get("batch_size"),
        "fresh_lens": json.dumps(meta.get("fresh_lens"), separators=(",", ":")),
        "prefix_lens": json.dumps(meta.get("prefix_lens"), separators=(",", ":")),
        "warmup_runs": meta.get("warmup_runs"),
        "context_length": meta.get("context_length"),
        "gap_ms": gap_ms,
        "self_attn_host_ms": (self_end - self_start) / 1e6,
        "rope_kernel_ms": rope["duration_ms"],
        "set_mla_kernel_ms": set_mla["duration_ms"],
        "set_mla_grid": json.dumps(set_mla["grid"], separators=(",", ":")),
        "runtime_events": json.dumps(
            [
                {
                    "name": event["name"],
                    "offset_start_ms": event["offset_start_ms"],
                    "duration_ms": event["duration_ms"],
                    "thread": event["thread"],
                }
                for event in runtimes
            ],
            ensure_ascii=False,
        ),
        "module_load_count": len(module_loads),
        "first_module_load_offset_ms": (
            module_loads[0]["offset_start_ms"] if module_loads else ""
        ),
        "first_module_load_duration_ms": (
            module_loads[0]["duration_ms"] if module_loads else ""
        ),
        "first_launch_offset_ms": launches[0]["offset_start_ms"] if launches else "",
        **sample_summary,
        **set_mla_cache_variants(run_dir),
    }
    for sample in sample_rows:
        sample["group"] = group
        sample["run_name"] = run_dir.name
    return row, sample_rows


def find_runs() -> list[tuple[Path, str]]:
    runs: list[tuple[Path, str]] = []
    for run_dir in sorted(TARGET_ROOT.glob("20260623_*")):
        runs.append((run_dir, "target_single_long"))
    for run_dir in sorted(REF_ROOT.glob("20260623_*")):
        runs.append((run_dir, "reference_multi_same_total"))
    return runs


def render_report(rows: list[dict[str, Any]]) -> str:
    def fmt(value: Any, digits: int = 3) -> str:
        if value == "":
            return ""
        if isinstance(value, float):
            return f"{value:.{digits}f}"
        return str(value)

    lines = [
        "# Prefill 首层大 Gap CPU 采样复跑分析",
        "",
        "## 结论",
        "",
        "- 三个单请求长序列目标用例均复现 `rotary_emb -> set_mla_kv_buffer_kernel` 前的大 gap：约 478-540ms；`set_mla_kv_buffer_kernel` 自身仍只有约 0.02ms。",
        "- 这次打开了 Nsight CPU sampling。gap 内 CPU 样本直接落在 `sglang::scheduler` 主线程、`ptxas`、Triton `_C/libtriton.so` 的 LLVM/MLIR 编译栈上；这把上一轮“疑似 Triton/CUDA lazy JIT/module load”的判断升级为直接证据。",
        "- `cuModuleLoadData` 只出现在 gap 尾部，耗时约 0.23-0.25ms；因此几百毫秒的大头不是 CUDA module load API 本身，而是它之前的 Triton/LLVM/ptxas 编译与 materialize 工作。",
        "- 参考用例 `b2_f4096_p0` 同样总 fresh token 为 8192，但 gap 只有 0.001ms，且 Triton cache 中只有 warmup 已生成的 `set_mla` specialization；它没有触发第二个 formal specialization。",
        "- 单请求长序列目标用例的 Triton cache 中均出现两个 `set_mla_kv_buffer_kernel` specialization，差异集中在 `loc_ptr` 参数是否带 `tt.divisibility = 16`。这解释了为什么 warmup 后 formal 仍可能出现 gap：warmup 编译了一个 specialization，formal 的 `out_cache_loc/loc_ptr` 触发了另一个 specialization。",
        "",
        "## 汇总",
        "",
        "| group | shape | gap ms | set_mla grid | samples | scheduler/ptxas/llvm samples | Triton frames | ptxas frames | module load offset ms | set_mla variants | diff hint |",
        "| --- | --- | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        shape = f"b{row['batch_size']} fresh={row['fresh_lens']} prefix={row['prefix_lens']}"
        lines.append(
            "| {group} | {shape} | {gap} | `{grid}` | {samples} | {sched}/{ptxas}/{llvm} | {triton} | {ptxas_frames} | {load_offset} | {variants} | `{hint}` |".format(
                group=row["group"],
                shape=shape,
                gap=fmt(row["gap_ms"]),
                grid=row["set_mla_grid"],
                samples=row["sample_count"],
                sched=row["scheduler_samples"],
                ptxas=row["ptxas_samples"],
                llvm=row["llvm_worker_samples"],
                triton=row["triton_module_frame_count"],
                ptxas_frames=row["ptxas_module_frame_count"],
                load_offset=fmt(row["first_module_load_offset_ms"]),
                variants=row["set_mla_cache_variant_count"],
                hint=row["set_mla_cache_diff_hint"],
            )
        )
    lines.extend(
        [
            "",
            "## 为什么 warmup 后仍然有 gap",
            "",
            "- `stage.txt` 显示每个目标 run 都执行了 `before_warmup_generate_0 -> after_warmup_generate_0 -> before_start_profile -> after_start_profile -> before_generate`，所以不是完全没有 warmup。",
            "- 但目标 run 的 `/out/triton_cache` 中有两个 `set_mla_kv_buffer_kernel` 目录；从 mtime 和 TTIR 差异看，warmup 先生成 `loc_ptr` 带 16-byte divisibility 的 specialization，formal profile 段又生成了 `loc_ptr` 未标注 divisibility 的 specialization。",
            "- Nsight gap 内采样显示 scheduler 主线程在 Triton `_C/libtriton.so` 的 LLVM/MLIR 路径中，同时有独立 `ptxas` 线程/进程样本。因此 formal 段不是简单读取 warmup 产物，而是在编译另一个 Triton specialization。",
            "",
            "## 为什么主要是这些单请求长序列 shape",
            "",
            "- 对照 `b2_f4096_p0` 和目标 `b1_f8192_p0` 的 `set_mla` grid 都是 `[8192,5,1]`，但前者 gap 为 0.001ms，后者为 509ms；所以不是总 token 或 kernel grid 本身导致。",
            "- 更合理的触发条件是 formal run 中 `out_cache_loc/loc_ptr` 的指针属性或张量构造路径不同，导致 Triton cache key 变化。单请求长序列在 formal session 中更容易走到 `loc_ptr` 非 16-byte divisibility specialization；多请求同总 token 的参考形状复用了 warmup specialization。",
            "- 这里仍保留一点不确定性：Nsight 数据能证明差异体现在 Triton specialization 与 `loc_ptr` 属性，不能仅从 trace 反推出 SGLang allocator 选择该 `loc_ptr` 属性的全部内部条件。若要最终闭环，需要在 SGLang `set_mla_kv_buffer_triton` 调用前打印 `out_cache_loc.data_ptr()`、`stride/storage_offset/is_contiguous` 与 Triton cache key。",
            "",
            "## 可复现性与随机性判断",
            "",
            "- 从当前证据看，这不是普通运行随机抖动：三个目标 shape 在独立复跑中全部稳定出现同类 gap、同类 CPU 编译栈、同类第二个 `set_mla` specialization；参考 shape 在同一采样配置下没有出现 gap。",
            "- gap 绝对值会受机器负载、Nsight sampling、Triton/ptxas 编译耗时影响，所以 478ms/509ms/540ms 这些数值不应视作严格常数；但“是否触发第二个 `loc_ptr` specialization 编译”是当前更固定、更可复现的判据。",
            "- 若复用持久化 Triton cache 或提前显式预触发 formal 所需 specialization，这类 gap 预期会消失或显著缩小；因此它更像 cold-specialization 编译污染，而不是模型计算本身的稳定耗时。",
            "",
            "## 运行与环境注意",
            "",
            "- 普通容器下 `nsys status --environment` 显示 `perf_event_open` 和 sampling trigger 失败；加 `--privileged --pid=host` 虽可打开 CPU sampling，但会暴露全部 GPU，破坏 `--gpus device=7` 隔离。",
            "- 本次最终采用 `--cap-add SYS_ADMIN --cap-add SYS_PTRACE --security-opt seccomp=unconfined`，CPU profiling OK，且容器内仍只看到指定 GPU。",
            "",
            "## 产物",
            "",
            "- `gap_cpusample_rerun_summary.csv`：四个复跑用例的 gap、runtime、CPU sampling、Triton cache specialization 汇总。",
            "- `gap_cpusample_rerun_sample_frames.csv`：gap 内截取的 Triton/LLVM/ptxas 相关 CPU sample 栈帧。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    summaries: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    for run_dir, group in find_runs():
        summary, samples = analyze_run(run_dir, group)
        summaries.append(summary)
        sample_rows.extend(samples)

    write_csv(
        OUT_DIR / "gap_cpusample_rerun_summary.csv",
        summaries,
        [
            "group",
            "run_name",
            "batch_size",
            "fresh_lens",
            "prefix_lens",
            "warmup_runs",
            "context_length",
            "gap_ms",
            "self_attn_host_ms",
            "rope_kernel_ms",
            "set_mla_kernel_ms",
            "set_mla_grid",
            "runtime_events",
            "module_load_count",
            "first_module_load_offset_ms",
            "first_module_load_duration_ms",
            "first_launch_offset_ms",
            "sampling_table_present",
            "sample_count",
            "scheduler_samples",
            "ptxas_samples",
            "llvm_worker_samples",
            "triton_module_frame_count",
            "ptxas_module_frame_count",
            "top_threads",
            "top_modules",
            "top_symbols",
            "set_mla_cache_variant_count",
            "set_mla_cache_variants",
            "set_mla_cache_diff_hint",
        ],
    )
    write_csv(
        OUT_DIR / "gap_cpusample_rerun_sample_frames.csv",
        sample_rows,
        [
            "group",
            "run_name",
            "sample_id",
            "offset_ms",
            "thread",
            "stack_depth",
            "symbol",
            "module",
        ],
    )
    (OUT_DIR / "Prefill首层大Gap-CPU采样复跑分析.md").write_text(
        render_report(summaries),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
