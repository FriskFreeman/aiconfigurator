#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "bench_data/h100_sxm/sglang/v0.5.9/deepseek_v3_mla"
COMPARE_CSV = (
    DATA_ROOT
    / "analysis/wideep_module_mla/wideep_module_mla_aic_compare__comparison.csv"
)
RAW_RUNS = DATA_ROOT / "raw_runs"
OUT_DIR = ROOT / "analysis/wideep_module_mla_makespan_inspect"


TARGET_TAGS = {
    "prefill": {
        "req1_equal_fresh_irregular_b1_f4500_p0",
        "req1_equal_fresh_irregular_b1_f6000_p0",
        "req1_equal_fresh_regular_b1_f8192_p0",
        "req1_equal_prefix_regular_b1_f8192_p0",
    },
    "decode": {
        "req1_equal_fresh_regular_b1_f1_p0",
        "req1_equal_fresh_irregular_b1_f4500_p0",
        "req1_equal_fresh_irregular_b1_f6000_p0",
        "req1_equal_fresh_regular_b4_f512_p0",
        "req1_equal_prefix_regular_b4_f512_p512",
        "req1_equal_fresh_regular_b8_f1_p0",
        "req1_equal_fresh_irregular_b12_f300_p0",
    },
}


REFERENCE_TAGS = {
    "prefill": {
        "req1_equal_fresh_regular_b4_f2048_p0",
        "req1_equal_fresh_regular_b2_f4096_p0",
        "req1_equal_prefix_regular_b4_f2048_p1024",
    },
    "decode": {
        "req1_equal_fresh_regular_b4_f2048_p0",
        "req1_equal_fresh_regular_b2_f4096_p0",
        "req1_equal_prefix_regular_b4_f2048_p1024",
    },
}


CHILD_NAMES_WIDEEP = {
    "prefill": {
        "attn_mha",
        "attn_mqa",
        "kv_a_layernorm",
        "kv_b_proj",
        "o_proj",
        "q_a_layernorm",
        "q_b_proj",
        "rotary_emb",
    },
    "decode": {
        "attn_mqa",
        "kv_a_layernorm",
        "o_proj",
        "q_a_layernorm",
        "q_b_proj",
        "rotary_emb",
    },
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def to_float(value: Any, default: float = math.nan) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def to_int(value: Any, default: int = 0) -> int:
    if value is None or value == "":
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def load_json_field(value: str) -> Any:
    if not value:
        return None
    return json.loads(value)


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


def top_overlapping_named_events(
    conn: sqlite3.Connection,
    strings: dict[int, str],
    table: str,
    start_ns: int,
    end_ns: int,
    *,
    name_col: str = "nameId",
    limit: int = 5,
) -> list[dict[str, Any]]:
    if not table_exists(conn, table) or end_ns <= start_ns:
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
            }
        )
    return events


def top_overlapping_kernels(
    conn: sqlite3.Connection,
    strings: dict[int, str],
    start_ns: int,
    end_ns: int,
    limit: int = 5,
) -> list[dict[str, Any]]:
    table = "CUPTI_ACTIVITY_KIND_KERNEL"
    if not table_exists(conn, table) or end_ns <= start_ns:
        return []
    rows = conn.execute(
        """
        SELECT start, end, shortName, streamId
        FROM CUPTI_ACTIVITY_KIND_KERNEL
        WHERE start < ? AND end > ?
        ORDER BY (MIN(end, ?) - MAX(start, ?)) DESC
        LIMIT ?
        """,
        (end_ns, start_ns, end_ns, start_ns, limit),
    ).fetchall()
    events = []
    for start, end, name_id, stream_id in rows:
        overlap = max(0, min(int(end), end_ns) - max(int(start), start_ns))
        events.append(
            {
                "name": strings.get(int(name_id), str(name_id)),
                "stream_id": int(stream_id),
                "start_ns": int(start),
                "end_ns": int(end),
                "duration_ms": (int(end) - int(start)) / 1e6,
                "overlap_ms": overlap / 1e6,
            }
        )
    return events


def top_overlapping_nvtx(
    conn: sqlite3.Connection,
    start_ns: int,
    end_ns: int,
    limit: int = 5,
) -> list[dict[str, Any]]:
    table = "NVTX_EVENTS"
    if not table_exists(conn, table) or end_ns <= start_ns:
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
    events = []
    for start, end, text in rows:
        overlap = max(0, min(int(end), end_ns) - max(int(start), start_ns))
        events.append(
            {
                "name": str(text)[:240],
                "start_ns": int(start),
                "end_ns": int(end),
                "duration_ms": (int(end) - int(start)) / 1e6,
                "overlap_ms": overlap / 1e6,
            }
        )
    return events


def gap_events_for_window(sqlite_path: Path, start_ns: int, end_ns: int) -> dict[str, Any]:
    if not sqlite_path.exists() or end_ns <= start_ns:
        return {}
    conn = sqlite3.connect(sqlite_path)
    try:
        strings = string_map(conn)
        return {
            "kernels": top_overlapping_kernels(conn, strings, start_ns, end_ns),
            "cuda_runtime": top_overlapping_named_events(
                conn, strings, "CUPTI_ACTIVITY_KIND_RUNTIME", start_ns, end_ns
            ),
            "cuda_sync": top_overlapping_named_events(
                conn,
                strings,
                "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION",
                start_ns,
                end_ns,
                name_col="syncType",
            ),
            "osrt": top_overlapping_named_events(
                conn, strings, "OSRT_API", start_ns, end_ns
            ),
            "profiler_overhead": top_overlapping_named_events(
                conn, strings, "PROFILER_OVERHEAD", start_ns, end_ns
            ),
            "nvtx": top_overlapping_nvtx(conn, start_ns, end_ns),
        }
    finally:
        conn.close()


def find_run_csv(run_instance_name: str) -> Path:
    path = RAW_RUNS / run_instance_name / "nsys/MLA时延拆解.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def find_sqlite(run_instance_name: str) -> Path:
    return RAW_RUNS / run_instance_name / "nsys/report.sqlite"


def selected_compare_rows(compare_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    selected = []
    for row in compare_rows:
        stage = row["stage"]
        tag = row["tag"]
        if tag in TARGET_TAGS.get(stage, set()):
            row = dict(row)
            row["inspect_group"] = "target"
            selected.append(row)
        elif tag in REFERENCE_TAGS.get(stage, set()):
            row = dict(row)
            row["inspect_group"] = "reference"
            selected.append(row)
    return selected


def compare_index(rows: list[dict[str, str]]) -> dict[tuple[str, str, str, str], dict[str, str]]:
    indexed = {}
    for row in rows:
        key = (
            row["run_instance_name"],
            row["stage"],
            row["layer_id"],
            row["event_order_index"],
        )
        indexed[key] = row
    return indexed


def load_needed_decomposition_rows(
    selected_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    by_run = defaultdict(list)
    for row in selected_rows:
        by_run[row["run_instance_name"]].append(row)

    needed = []
    selected_keys = {
        (
            row["run_instance_name"],
            row["stage"],
            row["layer_id"],
            row["event_order_index"],
        )
        for row in selected_rows
    }
    for run_name in sorted(by_run):
        csv_path = find_run_csv(run_name)
        with csv_path.open(newline="") as f:
            for raw in csv.DictReader(f):
                key = (
                    raw["run_instance_name"],
                    raw["stage"],
                    raw["layer_id"],
                    raw["event_order_index"],
                )
                if key in selected_keys:
                    raw["_source_csv"] = str(csv_path)
                    needed.append(raw)
    return needed


def largest_child_gaps(
    children: list[dict[str, Any]],
    *,
    stage: str,
    sqlite_path: Path,
) -> list[dict[str, Any]]:
    filtered = [
        child
        for child in children
        if child.get("canonical_name") in CHILD_NAMES_WIDEEP.get(stage, set())
    ]
    filtered.sort(key=lambda item: to_int(item.get("order_index")))
    gaps = []
    for previous, current in zip(filtered, filtered[1:]):
        prev_name = previous.get("canonical_name")
        cur_name = current.get("canonical_name")
        host_start = to_int(previous.get("host_end_ns"))
        host_end = to_int(current.get("host_start_ns"))
        gpu_start = to_int(previous.get("last_kernel_end_ns"))
        gpu_end = to_int(current.get("first_kernel_start_ns"))
        host_gap_ns = max(0, host_end - host_start)
        gpu_gap_ns = max(0, gpu_end - gpu_start)
        # GPU makespan bubbles are defined by adjacent child kernel boundaries.
        # Host-side gaps are useful context, but they can include unrelated CPU work
        # outside the GPU execution window and should not drive the main diagnosis.
        gap_start = gpu_start
        gap_end = gpu_end
        events = gap_events_for_window(sqlite_path, gap_start, gap_end)
        gaps.append(
            {
                "prev_child": prev_name,
                "next_child": cur_name,
                "host_gap_ns": host_gap_ns,
                "host_gap_ms": host_gap_ns / 1e6,
                "gpu_gap_ns": gpu_gap_ns,
                "gpu_gap_ms": gpu_gap_ns / 1e6,
                "gap_window_start_ns": gap_start,
                "gap_window_end_ns": gap_end,
                "gap_window_ms": max(0, gap_end - gap_start) / 1e6,
                "top_events": events,
            }
        )
    return sorted(gaps, key=lambda item: item["gpu_gap_ms"], reverse=True)


def summarize_decomposition_rows(
    compare_rows: list[dict[str, str]],
    decomposition_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    cindex = compare_index(compare_rows)
    summaries = []
    gap_rows = []
    details: dict[str, Any] = {"cases": []}

    for row in decomposition_rows:
        key = (
            row["run_instance_name"],
            row["stage"],
            row["layer_id"],
            row["event_order_index"],
        )
        comp = cindex[key]
        stage = row["stage"]
        run_name = row["run_instance_name"]
        sqlite_path = find_sqlite(run_name)
        children = load_json_field(row.get("child_timing_json", "")) or []
        gaps = largest_child_gaps(children, stage=stage, sqlite_path=sqlite_path)
        max_gap = gaps[0] if gaps else {}

        module_makespan = to_float(row.get("module_gpu_makespan_ms"))
        module_kernel_sum = to_float(row.get("module_gpu_kernel_time_sum_ms"))
        wideep_makespan = to_float(comp.get("real_wideep_aligned_gpu_makespan_ms"))
        wideep_kernel_sum = to_float(comp.get("real_wideep_aligned_gpu_kernel_time_sum_ms"))

        summary = {
            "inspect_group": comp["inspect_group"],
            "tag": comp["tag"],
            "stage": stage,
            "run_instance_name": run_name,
            "layer_id": row["layer_id"],
            "event_order_index": row["event_order_index"],
            "stage_instance_index": row.get("stage_instance_index", ""),
            "stage_instance_count": row.get("stage_instance_count", ""),
            "batch_size": comp.get("batch_size"),
            "fresh_len": comp.get("fresh_len"),
            "prefix_len": comp.get("prefix_len"),
            "attention_module": comp.get("attention_module"),
            "real_trace_attention_backend": comp.get("real_trace_attention_backend"),
            "aic_wideep_mla_latency_ms": comp.get("aic_wideep_mla_latency_ms"),
            "aic_error_pct_vs_module_gpu_makespan_ms": comp.get(
                "aic_error_pct_vs_module_gpu_makespan_ms"
            ),
            "aic_error_pct_vs_wideep_aligned_gpu_makespan_ms": comp.get(
                "aic_error_pct_vs_wideep_aligned_gpu_makespan_ms"
            ),
            "module_gpu_makespan_ms": module_makespan,
            "module_gpu_kernel_time_sum_ms": module_kernel_sum,
            "module_gpu_bubble_ms": module_makespan - module_kernel_sum,
            "module_bubble_ratio": (
                (module_makespan - module_kernel_sum) / module_makespan
                if module_makespan and module_makespan > 0
                else math.nan
            ),
            "wideep_gpu_makespan_ms": wideep_makespan,
            "wideep_gpu_kernel_time_sum_ms": wideep_kernel_sum,
            "wideep_gpu_bubble_ms": wideep_makespan - wideep_kernel_sum,
            "wideep_bubble_ratio": (
                (wideep_makespan - wideep_kernel_sum) / wideep_makespan
                if wideep_makespan and wideep_makespan > 0
                else math.nan
            ),
            "module_host_duration_ms": row.get("module_host_duration_ms"),
            "module_host_to_first_kernel_gap_ms": row.get(
                "module_host_to_first_kernel_gap_ms"
            ),
            "module_host_end_to_last_kernel_tail_ms": row.get(
                "module_host_end_to_last_kernel_tail_ms"
            ),
            "max_child_gap_prev": max_gap.get("prev_child", ""),
            "max_child_gap_next": max_gap.get("next_child", ""),
            "max_child_host_gap_ms": max_gap.get("host_gap_ms", 0.0),
            "max_child_gpu_gap_ms": max_gap.get("gpu_gap_ms", 0.0),
            "max_child_gap_window_ms": max_gap.get("gap_window_ms", 0.0),
            "max_child_gap_top_events_json": short_json(max_gap.get("top_events", {})),
            "source_csv": row["_source_csv"],
        }
        summaries.append(summary)

        for rank, gap in enumerate(gaps[:4], start=1):
            gap_rows.append(
                {
                    "inspect_group": comp["inspect_group"],
                    "tag": comp["tag"],
                    "stage": stage,
                    "run_instance_name": run_name,
                    "layer_id": row["layer_id"],
                    "event_order_index": row["event_order_index"],
                    "stage_instance_index": row.get("stage_instance_index", ""),
                    "stage_instance_count": row.get("stage_instance_count", ""),
                    "gap_rank": rank,
                    "prev_child": gap["prev_child"],
                    "next_child": gap["next_child"],
                    "host_gap_ms": gap["host_gap_ms"],
                    "gpu_gap_ms": gap["gpu_gap_ms"],
                    "gap_window_ms": gap["gap_window_ms"],
                    "gap_window_start_ns": gap["gap_window_start_ns"],
                    "gap_window_end_ns": gap["gap_window_end_ns"],
                    "top_events_json": short_json(gap["top_events"]),
                }
            )

        details["cases"].append(
            {
                "summary": summary,
                "top_child_gaps": gaps[:8],
                "child_timing": children,
            }
        )

    summaries.sort(
        key=lambda item: (
            item["stage"],
            item["inspect_group"],
            item["tag"],
            int(item["layer_id"]),
        )
    )
    gap_rows.sort(
        key=lambda item: (
            item["stage"],
            item["inspect_group"],
            item["tag"],
            int(item["layer_id"]),
            item["gap_rank"],
        )
    )
    return summaries, gap_rows, details


def summarize_by_tag(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["inspect_group"], row["stage"], row["tag"])].append(row)

    out = []
    for (group, stage, tag), items in sorted(grouped.items()):
        makespans = [to_float(item["module_gpu_makespan_ms"]) for item in items]
        bubbles = [to_float(item["module_gpu_bubble_ms"]) for item in items]
        max_item = max(items, key=lambda item: to_float(item["module_gpu_makespan_ms"]))
        out.append(
            {
                "inspect_group": group,
                "stage": stage,
                "tag": tag,
                "layer_count": len(items),
                "mean_module_gpu_makespan_ms": sum(makespans) / len(makespans),
                "max_module_gpu_makespan_ms": max(makespans),
                "max_makespan_layer_id": max_item["layer_id"],
                "mean_module_gpu_bubble_ms": sum(bubbles) / len(bubbles),
                "max_module_gpu_bubble_ms": max(bubbles),
                "max_child_gap_window_ms": max(
                    to_float(item["max_child_gap_window_ms"]) for item in items
                ),
                "max_child_gap_edge": f"{max_item['max_child_gap_prev']}->{max_item['max_child_gap_next']}",
                "mean_abs_error_pct_vs_module_makespan": sum(
                    abs(to_float(item["aic_error_pct_vs_module_gpu_makespan_ms"]))
                    for item in items
                )
                / len(items),
                "mean_abs_error_pct_vs_wideep_makespan": sum(
                    abs(to_float(item["aic_error_pct_vs_wideep_aligned_gpu_makespan_ms"]))
                    for item in items
                )
                / len(items),
            }
        )
    return out


def render_report(tag_rows: list[dict[str, Any]], layer_rows: list[dict[str, Any]]) -> str:
    target_prefill = [
        row for row in tag_rows if row["inspect_group"] == "target" and row["stage"] == "prefill"
    ]
    target_decode = [
        row for row in tag_rows if row["inspect_group"] == "target" and row["stage"] == "decode"
    ]

    def markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
        lines = [
            "| " + " | ".join(title for title, _ in columns) + " |",
            "| " + " | ".join("---" for _ in columns) + " |",
        ]
        for row in rows:
            values = []
            for _, key in columns:
                value = row.get(key, "")
                if isinstance(value, float):
                    value = f"{value:.3f}"
                values.append(str(value))
            lines.append("| " + " | ".join(values) + " |")
        return "\n".join(lines)

    prefill_big = [
        row
        for row in layer_rows
        if row["stage"] == "prefill" and row["inspect_group"] == "target"
    ]
    prefill_big = sorted(
        prefill_big, key=lambda item: to_float(item["module_gpu_makespan_ms"]), reverse=True
    )[:8]

    decode_big = [
        row
        for row in layer_rows
        if row["stage"] == "decode" and row["inspect_group"] == "target"
    ]
    decode_big = sorted(
        decode_big, key=lambda item: to_float(item["module_gpu_makespan_ms"]), reverse=True
    )[:12]

    lines = [
        "# WideEP MLA GPU makespan 离群剖析报告",
        "",
        "## 结论摘要",
        "",
        "- Prefill 的最显著离群来自单请求长 prefill 的第 0 层：`b1f4500p0`、`b1f6000p0`、`b1f8192p0` 都在 layer0 出现 230 到 290 ms 级别的 GPU makespan。子模块净 kernel 加总只有数毫秒，主要差异来自 `rotary_emb -> kv_b_proj` 之间的巨大空窗。",
        "- 同一批 prefill case 的后续层回到 3 到 7 ms 区间，说明 AIC 的数值本身并不是系统性错几个数量级，而是实机首层存在一次性 bubble/host 侧等待。",
        "- Decode 的离群主要发生在短小 decode workload：真实 kernel 加总约 0.11 到 0.12 ms，但 GPU makespan 常被 1 到 2.6 ms 的跨子模块间隔放大。此时 makespan 被固定调度开销主导，对 AIC 的纯模块计算时延不公平。",
        "- `b1f1p0` 在 trace 中包含两个 decode stage instance：第一轮 marker 被解析为 `attn_mha`，第二轮为 `attn_mqa`，因此该 case 有 10 行 layer 记录；这不是重复运行，而是极短输入下的特殊调度形态。",
        "- 正常参考 case 的 kernel sum 与 AIC 更接近；当 makespan 接近 kernel sum 时误差可回到 20% 左右。",
        "",
        "## Target Case 汇总",
        "",
        "### Prefill",
        markdown_table(
            target_prefill,
            [
                ("tag", "tag"),
                ("mean makespan ms", "mean_module_gpu_makespan_ms"),
                ("max makespan ms", "max_module_gpu_makespan_ms"),
                ("max layer", "max_makespan_layer_id"),
                ("mean bubble ms", "mean_module_gpu_bubble_ms"),
                ("max child gap ms", "max_child_gap_window_ms"),
                ("mean err %", "mean_abs_error_pct_vs_wideep_makespan"),
            ],
        ),
        "",
        "### Decode",
        markdown_table(
            target_decode,
            [
                ("tag", "tag"),
                ("mean makespan ms", "mean_module_gpu_makespan_ms"),
                ("max makespan ms", "max_module_gpu_makespan_ms"),
                ("max layer", "max_makespan_layer_id"),
                ("mean bubble ms", "mean_module_gpu_bubble_ms"),
                ("max child gap ms", "max_child_gap_window_ms"),
                ("mean err %", "mean_abs_error_pct_vs_wideep_makespan"),
            ],
        ),
        "",
        "## Prefill 最大层级离群",
        "",
        markdown_table(
            prefill_big,
            [
                ("tag", "tag"),
                ("layer", "layer_id"),
                ("makespan ms", "module_gpu_makespan_ms"),
                ("kernel sum ms", "module_gpu_kernel_time_sum_ms"),
                ("bubble ms", "module_gpu_bubble_ms"),
                ("gap edge", "max_child_gap_prev"),
                ("gap next", "max_child_gap_next"),
                ("gap ms", "max_child_gap_window_ms"),
            ],
        ),
        "",
        "## Decode 最大层级离群",
        "",
        markdown_table(
            decode_big,
            [
                ("tag", "tag"),
                ("layer", "layer_id"),
                ("makespan ms", "module_gpu_makespan_ms"),
                ("kernel sum ms", "module_gpu_kernel_time_sum_ms"),
                ("bubble ms", "module_gpu_bubble_ms"),
                ("gap edge", "max_child_gap_prev"),
                ("gap next", "max_child_gap_next"),
                ("gap ms", "max_child_gap_window_ms"),
            ],
        ),
        "",
        "## 原始 nsys 证据",
        "",
        "- Prefill 长单请求 layer0 的最大 gap 都位于 `rotary_emb -> kv_b_proj`：`b1f4500p0` 为 274.305 ms，`b1f6000p0` 为 283.861 ms，`b1f8192p0` 为 252.021 ms。gap 内只看到极短 `set_mla_kv_buffer_kernel`（约 0.015 到 0.026 ms）和少量 `cuModuleLoadData/cuLaunchKernel*`，但 NVTX 范围仍处在 `model.model.layers.0.self_attn` 内，说明 makespan 被首层 self_attn 内部的大空窗主导。",
        "- Prefill layer0 同时伴随 Nsight/运行时初始化痕迹，例如 `OS runtime libraries profiling initialization` 与多次 `TLS allocation`，这些事件只解释数毫秒级开销，不能覆盖 200ms 以上的大 gap；更合理的判断是首层存在一次性 host/GPU 调度等待或 lazy 初始化链路。",
        "- Decode target case 的 kernel sum 通常约 0.10 到 0.12 ms，但子模块间 GPU gap 可达 0.13 到 0.90 ms。典型边界是 `attn_mqa -> o_proj` 或 `rotary_emb -> o_proj`，gap 内常见的是很短的 `cudaLaunchKernel/cuLaunchKernelEx` 与小型 GEMM/FA kernel，说明 makespan 对固定调度间隔非常敏感。",
        "- `child_gap_events.csv` 中的 OSRT `epoll_wait/pthread_cond_timedwait` 常常是后台线程长等待，只表示时间窗口重叠，不直接作为 causation；报告主判断以 GPU kernel 边界 gap 为准。",
        "",
        "## 产物说明",
        "",
        "- `case_layer_summary.csv`：每个 case/layer 一行，包含 AIC 误差、GPU makespan、kernel sum、bubble、最大子模块 gap。",
        "- `child_gap_events.csv`：每个 case/layer 的前 4 个最大子模块 gap，并附带 gap 窗口内 top CUDA kernel/runtime、OSRT、NVTX 事件。",
        "- `case_summary.csv`：按 case 汇总的均值和最大值。",
        "- `case_diagnosis.json`：包含每层完整 child timing 与 gap 事件，便于继续深挖。",
        "",
        "## 判断",
        "",
        "这些离群点的共同特征不是 attention/GEMM kernel 本身变慢，而是 `GPU makespan` 口径把子模块之间的空窗也计入了模块时间。对于 prefill 长单请求，空窗集中在首层；对于 decode 小 workload，空窗在所有层都容易超过真实 kernel 时间一个数量级。因此后续对 AIC 的 MLA 模块表校验时，建议同时报告 `gpu_kernel_time_sum`、`wideep aligned makespan` 和 gap/bubble 指标，并把首层一次性 bubble 单独标注。",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    compare_rows = selected_compare_rows(read_csv(COMPARE_CSV))
    decomposition_rows = load_needed_decomposition_rows(compare_rows)
    layer_rows, gap_rows, details = summarize_decomposition_rows(
        compare_rows, decomposition_rows
    )
    case_rows = summarize_by_tag(layer_rows)

    write_csv(
        OUT_DIR / "case_layer_summary.csv",
        layer_rows,
        [
            "inspect_group",
            "tag",
            "stage",
            "run_instance_name",
            "layer_id",
            "event_order_index",
            "stage_instance_index",
            "stage_instance_count",
            "batch_size",
            "fresh_len",
            "prefix_len",
            "attention_module",
            "real_trace_attention_backend",
            "aic_wideep_mla_latency_ms",
            "aic_error_pct_vs_module_gpu_makespan_ms",
            "aic_error_pct_vs_wideep_aligned_gpu_makespan_ms",
            "module_gpu_makespan_ms",
            "module_gpu_kernel_time_sum_ms",
            "module_gpu_bubble_ms",
            "module_bubble_ratio",
            "wideep_gpu_makespan_ms",
            "wideep_gpu_kernel_time_sum_ms",
            "wideep_gpu_bubble_ms",
            "wideep_bubble_ratio",
            "module_host_duration_ms",
            "module_host_to_first_kernel_gap_ms",
            "module_host_end_to_last_kernel_tail_ms",
            "max_child_gap_prev",
            "max_child_gap_next",
            "max_child_host_gap_ms",
            "max_child_gpu_gap_ms",
            "max_child_gap_window_ms",
            "max_child_gap_top_events_json",
            "source_csv",
        ],
    )
    write_csv(
        OUT_DIR / "child_gap_events.csv",
        gap_rows,
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
            "host_gap_ms",
            "gpu_gap_ms",
            "gap_window_ms",
            "gap_window_start_ns",
            "gap_window_end_ns",
            "top_events_json",
        ],
    )
    write_csv(
        OUT_DIR / "case_summary.csv",
        case_rows,
        [
            "inspect_group",
            "stage",
            "tag",
            "layer_count",
            "mean_module_gpu_makespan_ms",
            "max_module_gpu_makespan_ms",
            "max_makespan_layer_id",
            "mean_module_gpu_bubble_ms",
            "max_module_gpu_bubble_ms",
            "max_child_gap_window_ms",
            "max_child_gap_edge",
            "mean_abs_error_pct_vs_module_makespan",
            "mean_abs_error_pct_vs_wideep_makespan",
        ],
    )
    (OUT_DIR / "case_diagnosis.json").write_text(
        json.dumps(details, ensure_ascii=False, indent=2)
    )
    (OUT_DIR / "问题报告.md").write_text(render_report(case_rows, layer_rows))


if __name__ == "__main__":
    main()
