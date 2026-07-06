#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "analysis/wideep_module_mla_makespan_inspect"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


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


def load_json(value: str) -> Any:
    if not value:
        return None
    return json.loads(value)


def load_child_timing_index() -> dict[tuple[str, str, str, str], list[dict[str, Any]]]:
    data = json.loads((OUT_DIR / "case_diagnosis.json").read_text())
    index: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for case in data.get("cases", []):
        summary = case.get("summary", {})
        key = (
            str(summary.get("run_instance_name", "")),
            str(summary.get("stage", "")),
            str(summary.get("layer_id", "")),
            str(summary.get("event_order_index", "")),
        )
        index[key] = case.get("child_timing", [])
    return index


def child_by_name(children: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    for child in children:
        if child.get("canonical_name") == name:
            return child
    return None


def event_row(
    row: dict[str, Any],
    *,
    event_kind: str,
    event_name: str,
    start_ns: int,
    end_ns: int | None = None,
    relation: str = "",
    source: str = "",
) -> dict[str, Any]:
    if end_ns is None:
        end_ns = start_ns
    gap_start = as_int(row["gap_window_start_ns"])
    return {
        "stage": row["stage"],
        "tag": row["tag"],
        "run_instance_name": row["run_instance_name"],
        "layer_id": row["layer_id"],
        "event_order_index": row["event_order_index"],
        "stage_instance_index": row.get("stage_instance_index", ""),
        "stage_instance_count": row.get("stage_instance_count", ""),
        "gap_rank": row["gap_rank"],
        "gap_edge": f"{row['prev_child']}->{row['next_child']}",
        "gpu_gap_ms": row["gpu_gap_ms"],
        "root_cause_class": row.get("root_cause_class", ""),
        "event_kind": event_kind,
        "event_name": event_name,
        "relation": relation,
        "source": source,
        "start_ns": start_ns,
        "end_ns": end_ns,
        "offset_start_ms": (start_ns - gap_start) / 1e6,
        "offset_end_ms": (end_ns - gap_start) / 1e6,
        "duration_ms": (end_ns - start_ns) / 1e6,
    }


def add_boundary_runtime(rows: list[dict[str, Any]], row: dict[str, Any], field: str, relation: str) -> None:
    for event in load_json(row.get(field, "")) or []:
        rows.append(
            event_row(
                row,
                event_kind="cuda_runtime_boundary",
                event_name=event.get("name", ""),
                start_ns=as_int(event.get("start_ns")),
                end_ns=as_int(event.get("end_ns")),
                relation=relation,
                source="CUPTI runtime correlationId",
            )
        )


def add_boundary_kernel(rows: list[dict[str, Any]], row: dict[str, Any], field: str, relation: str) -> None:
    event = load_json(row.get(field, "")) or {}
    if event:
        rows.append(
            event_row(
                row,
                event_kind="gpu_kernel_boundary",
                event_name=event.get("name", ""),
                start_ns=as_int(event.get("start_ns")),
                end_ns=as_int(event.get("end_ns")),
                relation=relation,
                source="CUPTI kernel",
            )
        )


def build_timeline() -> list[dict[str, Any]]:
    details = read_csv(OUT_DIR / "gap_root_cause_details.csv")
    child_index = load_child_timing_index()
    rows: list[dict[str, Any]] = []
    for row in details:
        key = (
            row["run_instance_name"],
            row["stage"],
            row["layer_id"],
            row["event_order_index"],
        )
        children = child_index.get(key, [])
        prev_child = child_by_name(children, row["prev_child"])
        next_child = child_by_name(children, row["next_child"])
        if prev_child:
            rows.append(
                event_row(
                    row,
                    event_kind="host_child_marker",
                    event_name=f"{row['prev_child']}.host_end",
                    start_ns=as_int(prev_child.get("host_end_ns")),
                    relation="prev_child_host_end",
                    source="child_timing_json/NVTX",
                )
            )
        add_boundary_runtime(rows, row, "prev_runtime_json", "prev_kernel_launch_runtime")
        add_boundary_kernel(rows, row, "prev_kernel_json", "prev_kernel")
        rows.append(
            event_row(
                row,
                event_kind="gpu_gap_boundary",
                event_name="gap_start_prev_kernel_end",
                start_ns=as_int(row["gap_window_start_ns"]),
                relation="gap_start",
                source="kernel boundary",
            )
        )
        for event in load_json(row.get("runtime_events_json", "")) or []:
            rows.append(
                event_row(
                    row,
                    event_kind="cuda_runtime_in_gap",
                    event_name=event.get("name", ""),
                    start_ns=as_int(event.get("start_ns")),
                    end_ns=as_int(event.get("end_ns")),
                    relation="inside_gap",
                    source="CUPTI runtime",
                )
            )
        for event in (load_json(row.get("profiler_events_json", "")) or [])[:5]:
            rows.append(
                event_row(
                    row,
                    event_kind="profiler_overhead_in_gap",
                    event_name=event.get("name", ""),
                    start_ns=as_int(event.get("start_ns")),
                    end_ns=as_int(event.get("end_ns")),
                    relation="inside_gap",
                    source="PROFILER_OVERHEAD",
                )
            )
        for event in (load_json(row.get("osrt_events_json", "")) or [])[:5]:
            rows.append(
                event_row(
                    row,
                    event_kind="osrt_overlap_in_gap",
                    event_name=event.get("name", ""),
                    start_ns=as_int(event.get("start_ns")),
                    end_ns=as_int(event.get("end_ns")),
                    relation="overlaps_gap",
                    source="OSRT_API",
                )
            )
        for event in (load_json(row.get("nvtx_events_json", "")) or [])[:5]:
            rows.append(
                event_row(
                    row,
                    event_kind="nvtx_covering_gap",
                    event_name=event.get("name", ""),
                    start_ns=as_int(event.get("start_ns")),
                    end_ns=as_int(event.get("end_ns")),
                    relation="covers_or_overlaps_gap",
                    source="NVTX_EVENTS",
                )
            )
        add_boundary_runtime(rows, row, "next_runtime_json", "next_kernel_launch_runtime")
        rows.append(
            event_row(
                row,
                event_kind="gpu_gap_boundary",
                event_name="gap_end_next_kernel_start",
                start_ns=as_int(row["gap_window_end_ns"]),
                relation="gap_end",
                source="kernel boundary",
            )
        )
        add_boundary_kernel(rows, row, "next_kernel_json", "next_kernel")
        if next_child:
            rows.append(
                event_row(
                    row,
                    event_kind="host_child_marker",
                    event_name=f"{row['next_child']}.host_start",
                    start_ns=as_int(next_child.get("host_start_ns")),
                    relation="next_child_host_start",
                    source="child_timing_json/NVTX",
                )
            )
    return sorted(
        rows,
        key=lambda item: (
            item["stage"],
            item["tag"],
            int(item["layer_id"]),
            int(item["event_order_index"]),
            float(item["offset_start_ms"]),
            item["event_kind"],
        ),
    )


def main() -> None:
    rows = build_timeline()
    write_csv(
        OUT_DIR / "gap_timeline_events.csv",
        rows,
        [
            "stage",
            "tag",
            "run_instance_name",
            "layer_id",
            "event_order_index",
            "stage_instance_index",
            "stage_instance_count",
            "gap_rank",
            "gap_edge",
            "gpu_gap_ms",
            "root_cause_class",
            "event_kind",
            "event_name",
            "relation",
            "source",
            "start_ns",
            "end_ns",
            "offset_start_ms",
            "offset_end_ms",
            "duration_ms",
        ],
    )


if __name__ == "__main__":
    main()
