#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "analysis/wideep_module_mla_makespan_inspect"
RAW_RUNS = ROOT / "bench_data/h100_sxm/sglang/v0.5.9/deepseek_v3_mla/raw_runs"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def short_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def string_map(conn: sqlite3.Connection) -> dict[int, str]:
    if not table_exists(conn, "StringIds"):
        return {}
    return {int(row[0]): str(row[1]) for row in conn.execute("SELECT id, value FROM StringIds")}


def table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
        ).fetchone()
        is not None
    )


def as_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def kernel_name(strings: dict[int, str], name_id: int | None) -> str:
    if name_id is None:
        return ""
    return strings.get(int(name_id), str(name_id))


def runtime_name(strings: dict[int, str], name_id: int | None) -> str:
    if name_id is None:
        return ""
    return strings.get(int(name_id), str(name_id))


def boundary_kernel(
    conn: sqlite3.Connection,
    strings: dict[int, str],
    timestamp_ns: int,
    *,
    side: str,
) -> dict[str, Any] | None:
    if not table_exists(conn, "CUPTI_ACTIVITY_KIND_KERNEL"):
        return None
    if side == "prev":
        row = conn.execute(
            """
            SELECT start, end, shortName, correlationId, streamId
            FROM CUPTI_ACTIVITY_KIND_KERNEL
            WHERE end <= ?
            ORDER BY end DESC
            LIMIT 1
            """,
            (timestamp_ns,),
        ).fetchone()
    else:
        row = conn.execute(
            """
            SELECT start, end, shortName, correlationId, streamId
            FROM CUPTI_ACTIVITY_KIND_KERNEL
            WHERE start >= ?
            ORDER BY start ASC
            LIMIT 1
            """,
            (timestamp_ns,),
        ).fetchone()
    if row is None:
        return None
    start, end, name_id, corr, stream_id = row
    return {
        "start_ns": int(start),
        "end_ns": int(end),
        "duration_ms": (int(end) - int(start)) / 1e6,
        "name": kernel_name(strings, name_id),
        "correlation_id": corr,
        "stream_id": stream_id,
    }


def runtimes_for_correlation(
    conn: sqlite3.Connection,
    strings: dict[int, str],
    correlation_id: int | None,
) -> list[dict[str, Any]]:
    if correlation_id is None or not table_exists(conn, "CUPTI_ACTIVITY_KIND_RUNTIME"):
        return []
    rows = conn.execute(
        """
        SELECT start, end, nameId, globalTid
        FROM CUPTI_ACTIVITY_KIND_RUNTIME
        WHERE correlationId = ?
        ORDER BY start
        """,
        (correlation_id,),
    ).fetchall()
    return [
        {
            "start_ns": int(start),
            "end_ns": int(end),
            "duration_ms": (int(end) - int(start)) / 1e6,
            "name": runtime_name(strings, name_id),
            "global_tid": global_tid,
        }
        for start, end, name_id, global_tid in rows
    ]


def runtime_events_in_window(
    conn: sqlite3.Connection,
    strings: dict[int, str],
    start_ns: int,
    end_ns: int,
    *,
    min_duration_ns: int = 0,
) -> list[dict[str, Any]]:
    if not table_exists(conn, "CUPTI_ACTIVITY_KIND_RUNTIME"):
        return []
    rows = conn.execute(
        """
        SELECT start, end, nameId, correlationId, globalTid
        FROM CUPTI_ACTIVITY_KIND_RUNTIME
        WHERE start < ? AND end > ? AND end - start >= ?
        ORDER BY start
        """,
        (end_ns, start_ns, min_duration_ns),
    ).fetchall()
    return [
        {
            "start_ns": int(start),
            "end_ns": int(end),
            "duration_ms": (int(end) - int(start)) / 1e6,
            "name": runtime_name(strings, name_id),
            "correlation_id": corr,
            "global_tid": tid,
            "offset_start_ms": (int(start) - start_ns) / 1e6,
            "offset_end_ms": (int(end) - start_ns) / 1e6,
        }
        for start, end, name_id, corr, tid in rows
    ]


def named_events_in_window(
    conn: sqlite3.Connection,
    strings: dict[int, str],
    table: str,
    start_ns: int,
    end_ns: int,
    *,
    name_col: str = "nameId",
    limit: int = 8,
) -> list[dict[str, Any]]:
    if not table_exists(conn, table):
        return []
    cols = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    if name_col not in cols:
        return []
    rows = conn.execute(
        f"""
        SELECT start, end, {name_col}
        FROM {table}
        WHERE start < ? AND end > ?
        ORDER BY (MIN(end, ?) - MAX(start, ?)) DESC
        LIMIT ?
        """,
        (end_ns, start_ns, end_ns, start_ns, limit),
    ).fetchall()
    events = []
    for start, end, name_id in rows:
        overlap = max(0, min(int(end), end_ns) - max(int(start), start_ns))
        events.append(
            {
                "name": strings.get(int(name_id), str(name_id)),
                "start_ns": int(start),
                "end_ns": int(end),
                "duration_ms": (int(end) - int(start)) / 1e6,
                "overlap_ms": overlap / 1e6,
                "offset_start_ms": (int(start) - start_ns) / 1e6,
                "offset_end_ms": (int(end) - start_ns) / 1e6,
            }
        )
    return events


def nvtx_events_in_window(
    conn: sqlite3.Connection,
    start_ns: int,
    end_ns: int,
    *,
    limit: int = 10,
) -> list[dict[str, Any]]:
    if not table_exists(conn, "NVTX_EVENTS"):
        return []
    rows = conn.execute(
        """
        SELECT start, end, COALESCE(text, jsonText, CAST(textId AS TEXT))
        FROM NVTX_EVENTS
        WHERE end IS NOT NULL AND start < ? AND end > ?
        ORDER BY (MIN(end, ?) - MAX(start, ?)) DESC
        LIMIT ?
        """,
        (end_ns, start_ns, end_ns, start_ns, limit),
    ).fetchall()
    out = []
    for start, end, text in rows:
        overlap = max(0, min(int(end), end_ns) - max(int(start), start_ns))
        out.append(
            {
                "name": str(text)[:300],
                "start_ns": int(start),
                "end_ns": int(end),
                "duration_ms": (int(end) - int(start)) / 1e6,
                "overlap_ms": overlap / 1e6,
                "offset_start_ms": (int(start) - start_ns) / 1e6,
                "offset_end_ms": (int(end) - start_ns) / 1e6,
            }
        )
    return out


def idle_segments(
    start_ns: int,
    end_ns: int,
    events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    ranges = sorted(
        (
            max(start_ns, int(event["start_ns"])),
            min(end_ns, int(event["end_ns"])),
        )
        for event in events
        if int(event["end_ns"]) > start_ns and int(event["start_ns"]) < end_ns
    )
    merged: list[tuple[int, int]] = []
    for start, end in ranges:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    cursor = start_ns
    gaps = []
    for start, end in merged:
        if start > cursor:
            gaps.append(
                {
                    "start_ns": cursor,
                    "end_ns": start,
                    "duration_ms": (start - cursor) / 1e6,
                    "offset_start_ms": (cursor - start_ns) / 1e6,
                    "offset_end_ms": (start - start_ns) / 1e6,
                }
            )
        cursor = max(cursor, end)
    if cursor < end_ns:
        gaps.append(
            {
                "start_ns": cursor,
                "end_ns": end_ns,
                "duration_ms": (end_ns - cursor) / 1e6,
                "offset_start_ms": (cursor - start_ns) / 1e6,
                "offset_end_ms": (end_ns - start_ns) / 1e6,
            }
        )
    return sorted(gaps, key=lambda item: item["duration_ms"], reverse=True)


def inspect_gap(row: dict[str, str]) -> dict[str, Any]:
    run_name = row["run_instance_name"]
    sqlite_path = RAW_RUNS / run_name / "nsys/report.sqlite"
    gap_start = as_int(row["gap_window_start_ns"])
    gap_end = as_int(row["gap_window_end_ns"])
    conn = sqlite3.connect(sqlite_path)
    try:
        strings = string_map(conn)
        prev_kernel = boundary_kernel(conn, strings, gap_start, side="prev")
        next_kernel = boundary_kernel(conn, strings, gap_end, side="next")
        prev_runtime = runtimes_for_correlation(
            conn, strings, prev_kernel.get("correlation_id") if prev_kernel else None
        )
        next_runtime = runtimes_for_correlation(
            conn, strings, next_kernel.get("correlation_id") if next_kernel else None
        )
        runtime_events = runtime_events_in_window(
            conn, strings, gap_start, gap_end, min_duration_ns=0
        )
        runtime_events_10us = [
            event for event in runtime_events if event["duration_ms"] >= 0.01
        ]
        runtime_idle = idle_segments(gap_start, gap_end, runtime_events)
        host_first_runtime_start_ns = (
            min(event["start_ns"] for event in runtime_events)
            if runtime_events
            else None
        )
        next_launch_start_ns = next_runtime[0]["start_ns"] if next_runtime else None
        profiler_events = named_events_in_window(
            conn, strings, "PROFILER_OVERHEAD", gap_start, gap_end, limit=8
        )
        osrt_events = named_events_in_window(
            conn, strings, "OSRT_API", gap_start, gap_end, limit=8
        )
        nvtx_events = nvtx_events_in_window(conn, gap_start, gap_end, limit=10)
    finally:
        conn.close()

    gap_ms = as_float(row["gpu_gap_ms"])
    host_silence_before_first_runtime_ms = (
        (host_first_runtime_start_ns - gap_start) / 1e6
        if host_first_runtime_start_ns is not None
        else gap_ms
    )
    host_silence_before_next_launch_ms = (
        (next_launch_start_ns - gap_start) / 1e6
        if next_launch_start_ns is not None
        else gap_ms
    )
    first_runtime_event = min(runtime_events, key=lambda event: event["start_ns"]) if runtime_events else None
    next_launch_event = next_runtime[0] if next_runtime else None
    runtime_total_ms = sum(event["duration_ms"] for event in runtime_events)
    profiler_overlap_ms = sum(event["overlap_ms"] for event in profiler_events)
    longest_runtime_idle_ms = runtime_idle[0]["duration_ms"] if runtime_idle else gap_ms

    cause = classify_gap(
        stage=row["stage"],
        gap_ms=gap_ms,
        host_silence_before_first_runtime_ms=host_silence_before_first_runtime_ms,
        longest_runtime_idle_ms=longest_runtime_idle_ms,
        profiler_overlap_ms=profiler_overlap_ms,
        runtime_total_ms=runtime_total_ms,
    )

    return {
        **row,
        "prev_kernel_json": prev_kernel or {},
        "next_kernel_json": next_kernel or {},
        "prev_runtime_json": prev_runtime,
        "next_runtime_json": next_runtime,
        "runtime_events_json": runtime_events_10us[:12],
        "first_runtime_name": first_runtime_event["name"] if first_runtime_event else "",
        "first_runtime_offset_ms": first_runtime_event["offset_start_ms"] if first_runtime_event else gap_ms,
        "next_launch_runtime_name": next_launch_event["name"] if next_launch_event else "",
        "next_launch_runtime_offset_ms": (
            (next_launch_event["start_ns"] - gap_start) / 1e6
            if next_launch_event
            else gap_ms
        ),
        "runtime_event_count": len(runtime_events),
        "runtime_total_ms": runtime_total_ms,
        "host_silence_before_first_runtime_ms": host_silence_before_first_runtime_ms,
        "host_silence_before_next_launch_ms": host_silence_before_next_launch_ms,
        "longest_runtime_idle_ms": longest_runtime_idle_ms,
        "longest_runtime_idle_json": runtime_idle[:3],
        "profiler_events_json": profiler_events,
        "profiler_overlap_ms": profiler_overlap_ms,
        "osrt_events_json": osrt_events,
        "nvtx_events_json": nvtx_events,
        "root_cause_class": cause,
    }


def classify_gap(
    *,
    stage: str,
    gap_ms: float,
    host_silence_before_first_runtime_ms: float,
    longest_runtime_idle_ms: float,
    profiler_overlap_ms: float,
    runtime_total_ms: float,
) -> str:
    if gap_ms >= 50 and host_silence_before_first_runtime_ms / gap_ms > 0.8:
        return "host_no_launch_long_prefill_first_layer_gap"
    if stage == "decode" and gap_ms >= 0.1 and runtime_total_ms < gap_ms * 0.3:
        return "short_decode_fixed_launch_interval_dominates"
    if profiler_overlap_ms > gap_ms * 0.3:
        return "profiler_overhead_overlap"
    if longest_runtime_idle_ms > gap_ms * 0.5:
        return "host_runtime_idle_gap"
    return "mixed_or_small_gap"


def select_gap_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    selected = []
    for row in rows:
        if row["inspect_group"] != "target" or row["gap_rank"] != "1":
            continue
        gap_ms = as_float(row["gpu_gap_ms"])
        if row["stage"] == "prefill" and gap_ms >= 50:
            selected.append(row)
        elif row["stage"] == "decode" and gap_ms >= 0.1:
            selected.append(row)
    return selected


def summarize_cases(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["stage"], row["tag"])].append(row)
    out = []
    for (stage, tag), items in sorted(grouped.items()):
        max_gap = max(items, key=lambda item: as_float(item["gpu_gap_ms"]))
        out.append(
            {
                "stage": stage,
                "tag": tag,
                "gap_count": len(items),
                "max_gpu_gap_ms": max(as_float(item["gpu_gap_ms"]) for item in items),
                "mean_gpu_gap_ms": sum(as_float(item["gpu_gap_ms"]) for item in items)
                / len(items),
                "max_gap_layer_id": max_gap["layer_id"],
                "max_gap_stage_instance_index": max_gap.get("stage_instance_index", ""),
                "max_gap_edge": f"{max_gap['prev_child']}->{max_gap['next_child']}",
                "max_gap_root_cause_class": max_gap["root_cause_class"],
                "max_gap_host_silence_before_first_runtime_ms": max_gap[
                    "host_silence_before_first_runtime_ms"
                ],
                "max_gap_child_host_gap_ms": as_float(max_gap.get("host_gap_ms")),
                "max_gap_first_runtime_name": max_gap["first_runtime_name"],
                "max_gap_first_runtime_offset_ms": max_gap["first_runtime_offset_ms"],
                "max_gap_next_launch_runtime_name": max_gap["next_launch_runtime_name"],
                "max_gap_next_launch_runtime_offset_ms": max_gap[
                    "next_launch_runtime_offset_ms"
                ],
                "max_gap_runtime_total_ms": max_gap["runtime_total_ms"],
                "max_gap_profiler_overlap_ms": max_gap["profiler_overlap_ms"],
            }
        )
    return out


def render_report(rows: list[dict[str, Any]], summary: list[dict[str, Any]]) -> str:
    prefill = [row for row in summary if row["stage"] == "prefill"]
    decode = [row for row in summary if row["stage"] == "decode"]

    def table(items: list[dict[str, Any]]) -> str:
        lines = [
            "| tag | max gap ms | mean gap ms | layer | edge | cause | child host gap ms | first runtime offset ms | next launch offset ms | runtime total ms |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for item in items:
            lines.append(
                "| {tag} | {max_gap:.3f} | {mean_gap:.3f} | {layer} | {edge} | {cause} | {host_gap:.3f} | {first_runtime:.3f} | {next_launch:.3f} | {runtime:.3f} |".format(
                    tag=item["tag"],
                    max_gap=item["max_gpu_gap_ms"],
                    mean_gap=item["mean_gpu_gap_ms"],
                    layer=item["max_gap_layer_id"],
                    edge=item["max_gap_edge"],
                    cause=item["max_gap_root_cause_class"],
                    host_gap=item["max_gap_child_host_gap_ms"],
                    first_runtime=item["max_gap_first_runtime_offset_ms"],
                    next_launch=item["max_gap_next_launch_runtime_offset_ms"],
                    runtime=item["max_gap_runtime_total_ms"],
                )
            )
        return "\n".join(lines)

    examples = sorted(rows, key=lambda item: as_float(item["gpu_gap_ms"]), reverse=True)[:8]
    lines = [
        "# GPU gap host-root-cause 深挖报告",
        "",
        "## 核心结论",
        "",
        "- 大 gap 的边界 kernel 与 CUDA runtime `correlationId` 能够对上；多数异常 gap 不是 GPU 上有长 kernel 占用，而是前一个 kernel 结束后 host 很久没有发起下一段相关 runtime launch。",
        "- Prefill 的 200ms 级 gap 发生在 layer0 `rotary_emb -> kv_b_proj`，`host_silence_before_first_runtime_ms` 基本等于整个 gap，下一段 `cuModuleLoadData/cuLaunchKernel*` 都出现在 gap 尾部。这指向首层 lazy 初始化/host 调度等待，而不是 MLA/FA/GEMM kernel 本身慢。",
        "- Decode 的 gap 绝对值较小但相对 kernel sum 很大，典型是 `attn_mqa -> o_proj` 或 `rotary_emb -> o_proj`。runtime 总时长只有几十微秒，gap 主要是短 workload 下固定 host launch 间隔吞掉 makespan。",
        "- OSRT 长等待事件经常与 gap 窗口重叠，但它们多为后台线程等待；本报告只把它们作为旁证，不作为直接根因。",
        "",
        "## Prefill gap 根因分类",
        "",
        table(prefill),
        "",
        "## Decode gap 根因分类",
        "",
        table(decode),
        "",
        "## 最大 gap 边界样例",
        "",
        "| stage | tag | layer | edge | gap ms | prev kernel | next kernel | prev runtime | next runtime |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in examples:
        prev_kernel = row["prev_kernel_json"]
        next_kernel = row["next_kernel_json"]
        prev_runtime = row["prev_runtime_json"][0]["name"] if row["prev_runtime_json"] else ""
        next_runtime = row["next_runtime_json"][0]["name"] if row["next_runtime_json"] else ""
        lines.append(
            "| {stage} | {tag} | {layer} | {edge} | {gap:.3f} | {prev_kernel} | {next_kernel} | {prev_runtime} | {next_runtime} |".format(
                stage=row["stage"],
                tag=row["tag"],
                layer=row["layer_id"],
                edge=f"{row['prev_child']}->{row['next_child']}",
                gap=as_float(row["gpu_gap_ms"]),
                prev_kernel=prev_kernel.get("name", ""),
                next_kernel=next_kernel.get("name", ""),
                prev_runtime=prev_runtime,
                next_runtime=next_runtime,
            )
        )
    lines.extend(
        [
            "",
            "## 字段说明",
            "",
            "- `host_silence_before_first_runtime_ms`：gap 起点到 gap 内第一条 CUDA runtime 调用开始的时间；若接近 `gpu_gap_ms`，说明 host 在大部分 gap 中没有发起 CUDA 调用。",
            "- `host_silence_before_next_launch_ms`：gap 起点到下一边界 kernel 对应 runtime launch 开始的时间。",
            "- `host_gap_ms`：gap 两侧子模块 NVTX/host marker 的结束到开始间隔，用来判断 Python/module 层是否也出现同向空窗。",
            "- `first_runtime_offset_ms/next_launch_runtime_offset_ms`：从 gap 起点算起，第一条 CUDA runtime 调用和下一边界 kernel launch 出现的位置。",
            "- `runtime_total_ms`：gap 窗口内 CUDA runtime 调用时长总和；它远小于 gap 时，说明 runtime 自身不是主要耗时。",
            "- `longest_runtime_idle_ms`：只看 CUDA runtime 时间线时，gap 内最长无 runtime 区间。",
            "- `prev_runtime_json/next_runtime_json`：边界 kernel 通过 `correlationId` 对应到的 host runtime launch。",
            "",
            "## 产物",
            "",
            "- `gap_root_cause_summary.csv`：按 case 汇总最大 gap、host silence、runtime 总时长和分类。",
            "- `gap_root_cause_details.csv`：每个入选 gap 的边界 kernel、runtime、NVTX/OSRT/Profiler 事件摘要。",
            "- `gap_root_cause_details.json`：完整结构化明细。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    gap_rows = select_gap_rows(read_csv(OUT_DIR / "child_gap_events.csv"))
    inspected = [inspect_gap(row) for row in gap_rows]
    summary = summarize_cases(inspected)

    detail_rows = []
    for row in inspected:
        detail_rows.append(
            {
                **{
                    key: row.get(key, "")
                    for key in [
                        "inspect_group",
                        "tag",
                        "stage",
                        "run_instance_name",
                        "layer_id",
                        "event_order_index",
                        "stage_instance_index",
                        "stage_instance_count",
                        "gap_rank",
                        "prev_child",
                        "next_child",
                        "gpu_gap_ms",
                        "host_gap_ms",
                        "gap_window_start_ns",
                        "gap_window_end_ns",
                        "root_cause_class",
                        "first_runtime_name",
                        "first_runtime_offset_ms",
                        "next_launch_runtime_name",
                        "next_launch_runtime_offset_ms",
                        "host_silence_before_first_runtime_ms",
                        "host_silence_before_next_launch_ms",
                        "longest_runtime_idle_ms",
                        "runtime_event_count",
                        "runtime_total_ms",
                        "profiler_overlap_ms",
                    ]
                },
                "prev_kernel_json": short_json(row["prev_kernel_json"]),
                "next_kernel_json": short_json(row["next_kernel_json"]),
                "prev_runtime_json": short_json(row["prev_runtime_json"]),
                "next_runtime_json": short_json(row["next_runtime_json"]),
                "runtime_events_json": short_json(row["runtime_events_json"]),
                "longest_runtime_idle_json": short_json(row["longest_runtime_idle_json"]),
                "profiler_events_json": short_json(row["profiler_events_json"]),
                "osrt_events_json": short_json(row["osrt_events_json"]),
                "nvtx_events_json": short_json(row["nvtx_events_json"]),
            }
        )

    write_csv(
        OUT_DIR / "gap_root_cause_details.csv",
        detail_rows,
        [
            "inspect_group",
            "tag",
            "stage",
            "run_instance_name",
            "layer_id",
            "event_order_index",
            "stage_instance_index",
            "stage_instance_count",
            "gap_rank",
            "prev_child",
            "next_child",
            "gpu_gap_ms",
            "host_gap_ms",
            "gap_window_start_ns",
            "gap_window_end_ns",
            "root_cause_class",
            "first_runtime_name",
            "first_runtime_offset_ms",
            "next_launch_runtime_name",
            "next_launch_runtime_offset_ms",
            "host_silence_before_first_runtime_ms",
            "host_silence_before_next_launch_ms",
            "longest_runtime_idle_ms",
            "runtime_event_count",
            "runtime_total_ms",
            "profiler_overlap_ms",
            "prev_kernel_json",
            "next_kernel_json",
            "prev_runtime_json",
            "next_runtime_json",
            "runtime_events_json",
            "longest_runtime_idle_json",
            "profiler_events_json",
            "osrt_events_json",
            "nvtx_events_json",
        ],
    )
    write_csv(
        OUT_DIR / "gap_root_cause_summary.csv",
        summary,
        [
            "stage",
            "tag",
            "gap_count",
            "max_gpu_gap_ms",
            "mean_gpu_gap_ms",
            "max_gap_layer_id",
            "max_gap_stage_instance_index",
            "max_gap_edge",
            "max_gap_root_cause_class",
            "max_gap_host_silence_before_first_runtime_ms",
            "max_gap_child_host_gap_ms",
            "max_gap_first_runtime_name",
            "max_gap_first_runtime_offset_ms",
            "max_gap_next_launch_runtime_name",
            "max_gap_next_launch_runtime_offset_ms",
            "max_gap_runtime_total_ms",
            "max_gap_profiler_overlap_ms",
        ],
    )
    (OUT_DIR / "gap_root_cause_details.json").write_text(
        json.dumps(inspected, ensure_ascii=False, indent=2)
    )
    (OUT_DIR / "GPU-gap根因补充分析.md").write_text(render_report(inspected, summary))


if __name__ == "__main__":
    main()
