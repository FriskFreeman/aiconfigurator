#!/usr/bin/env python3
"""Analyze kernels launched from per-iteration collect_mla NVTX ranges."""

from __future__ import annotations

import csv
import json
import math
import sqlite3
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


IMAGE = "booleimg.myaddr.io/lmsysorg/sglang:v0.5.9"
CATEGORY_ORDER = (
    "activation_bf16_to_fp8",
    "current_kv_bf16_to_fp8",
    "current_kv_cache_write",
    "full_kv_cache_fp8_to_bf16",
    "fa3_prepare",
    "fa3_attention_main",
    "fa3_attention_combine",
    "other",
)
CATEGORY_COLORS = {
    "activation_bf16_to_fp8": "#d95f02",
    "current_kv_bf16_to_fp8": "#e6ab02",
    "current_kv_cache_write": "#7570b3",
    "full_kv_cache_fp8_to_bf16": "#e7298a",
    "fa3_prepare": "#66a61e",
    "fa3_attention_main": "#1b9e77",
    "fa3_attention_combine": "#1f78b4",
    "other": "#8c8c8c",
}
CASE_ORDER = (
    "prefill_b1_s256",
    "prefill_b1_s2048",
    "decode_b1_step255",
    "decode_b1_step2047",
    "decode_b64_step32767",
)


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", newline="") as handle:
        if not fields:
            return
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def export_sqlite(case_dir: Path) -> Path:
    sqlite_path = case_dir / "report.sqlite"
    if sqlite_path.exists() and sqlite_path.stat().st_size > 0:
        return sqlite_path
    cmd = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{case_dir.resolve()}:/out",
        IMAGE,
        "nsys",
        "export",
        "--force-overwrite=true",
        "--type",
        "sqlite",
        "--output",
        "/out/report.sqlite",
        "/out/report.nsys-rep",
    ]
    completed = subprocess.run(cmd, text=True, capture_output=True)
    (case_dir / "export.stdout.log").write_text(completed.stdout)
    (case_dir / "export.stderr.log").write_text(completed.stderr)
    if completed.returncode != 0 or not sqlite_path.exists():
        raise RuntimeError(f"Nsight SQLite export failed for {case_dir}")
    return sqlite_path


def shorten_name(name: str) -> str:
    if "float8_copy_kernel_cuda" in name:
        return "at::native::float8_copy_kernel_cuda"
    if "direct_copy_kernel_cuda" in name:
        return "at::native::direct_copy_kernel_cuda"
    if "set_mla_kv_buffer_kernel" in name:
        return "set_mla_kv_buffer_kernel"
    if "prepare_varlen_num_blocks_kernel" in name:
        return "flash::prepare_varlen_num_blocks_kernel"
    if "FlashAttnFwdCombine" in name:
        return "flash::FlashAttnFwdCombine"
    if "FlashAttnFwdSm90" in name:
        return "flash::FlashAttnFwdSm90"
    return name[:180]


def classify_kernel(name: str, stage: str, dtype: str) -> str:
    if "float8_copy_kernel_cuda" in name:
        return "activation_bf16_to_fp8" if stage == "prefill" else "current_kv_bf16_to_fp8"
    if "set_mla_kv_buffer_kernel" in name:
        return "current_kv_cache_write"
    if "direct_copy_kernel_cuda" in name and stage == "decode" and dtype == "fp8":
        return "full_kv_cache_fp8_to_bf16"
    if "prepare_varlen_num_blocks_kernel" in name:
        return "fa3_prepare"
    if "FlashAttnFwdCombine" in name:
        return "fa3_attention_combine"
    if "FlashAttnFwdSm90" in name:
        return "fa3_attention_main"
    return "other"


def extract_profile(case_dir: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    measurement = json.loads((case_dir / "measurement.json").read_text())
    stage = str(measurement["stage"])
    dtype = str(measurement["kv_cache_dtype"])
    case = case_dir.parent.name
    sqlite_path = export_sqlite(case_dir)
    conn = sqlite3.connect(sqlite_path)
    try:
        strings = {int(key): str(value) for key, value in conn.execute("SELECT id, value FROM StringIds")}
        ranges = list(
            conn.execute(
                """
                SELECT start, end, text
                FROM NVTX_EVENTS
                WHERE text LIKE 'aic_collect_mla%::iter_%'
                  AND start IS NOT NULL AND end IS NOT NULL
                ORDER BY start
                """
            )
        )
        raw_rows: list[dict[str, object]] = []
        iteration_totals: list[float] = []
        for iteration, (range_start, range_end, text) in enumerate(ranges):
            correlation_ids = {
                int(row[0])
                for row in conn.execute(
                    """
                    SELECT correlationId
                    FROM CUPTI_ACTIVITY_KIND_RUNTIME
                    WHERE start >= ? AND start <= ? AND correlationId IS NOT NULL
                    """,
                    (range_start, range_end),
                )
            }
            if not correlation_ids:
                iteration_totals.append(0.0)
                continue
            placeholders = ",".join("?" for _ in correlation_ids)
            kernel_rows = conn.execute(
                f"""
                SELECT start, end, demangledName, correlationId,
                       gridX, gridY, gridZ, blockX, blockY, blockZ
                FROM CUPTI_ACTIVITY_KIND_KERNEL
                WHERE correlationId IN ({placeholders})
                ORDER BY start
                """,
                tuple(sorted(correlation_ids)),
            ).fetchall()
            total_us = 0.0
            for launch_index, row in enumerate(kernel_rows):
                start, end, name_id, correlation_id, *launch_shape = row
                name = strings.get(int(name_id), str(name_id))
                duration_us = (int(end) - int(start)) / 1000.0
                total_us += duration_us
                raw_rows.append(
                    {
                        "case": case,
                        "stage": stage,
                        "kv_cache_dtype": dtype,
                        "iteration": iteration,
                        "nvtx_text": text,
                        "launch_index": launch_index,
                        "duration_us": duration_us,
                        "category": classify_kernel(name, stage, dtype),
                        "short_name": shorten_name(name),
                        "full_name": name,
                        "correlation_id": correlation_id,
                        "grid_x": launch_shape[0],
                        "grid_y": launch_shape[1],
                        "grid_z": launch_shape[2],
                        "block_x": launch_shape[3],
                        "block_y": launch_shape[4],
                        "block_z": launch_shape[5],
                    }
                )
            iteration_totals.append(total_us)
    finally:
        conn.close()

    profile_summary = {
        "case": case,
        "stage": stage,
        "kv_cache_dtype": dtype,
        "batch_size": measurement["batch_size"],
        "input_len": measurement["input_len"],
        "num_heads": measurement["num_heads"],
        "tp_size": measurement["tp_size"],
        "iteration_count": len(iteration_totals),
        "kernel_total_mean_us": statistics.fmean(iteration_totals),
        "kernel_total_median_us": statistics.median(iteration_totals),
        "nsys_eager_event_mean_us": float(measurement["latency_mean_ms"]) * 1000.0,
    }
    return raw_rows, profile_summary


def summarize_categories(raw_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    per_iter: dict[tuple[str, str, str, int], list[float]] = defaultdict(list)
    launch_counts: dict[tuple[str, str, str, int], int] = defaultdict(int)
    iteration_ids: dict[tuple[str, str], set[int]] = defaultdict(set)
    for row in raw_rows:
        case = str(row["case"])
        dtype = str(row["kv_cache_dtype"])
        category = str(row["category"])
        iteration = int(row["iteration"])
        key = (case, dtype, category, iteration)
        per_iter[key].append(float(row["duration_us"]))
        launch_counts[key] += 1
        iteration_ids[(case, dtype)].add(iteration)

    rows: list[dict[str, object]] = []
    for case, dtype in sorted(iteration_ids):
        all_iters = sorted(iteration_ids[(case, dtype)])
        for category in CATEGORY_ORDER:
            durations = [sum(per_iter.get((case, dtype, category, i), [])) for i in all_iters]
            counts = [launch_counts.get((case, dtype, category, i), 0) for i in all_iters]
            if not any(counts):
                continue
            rows.append(
                {
                    "case": case,
                    "kv_cache_dtype": dtype,
                    "category": category,
                    "launches_per_iteration": statistics.median(counts),
                    "duration_mean_us": statistics.fmean(durations),
                    "duration_median_us": statistics.median(durations),
                    "duration_min_us": min(durations),
                    "duration_max_us": max(durations),
                }
            )
    return rows


def load_graph_latency(result_root: Path) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    sample_meta: dict[tuple[str, str], dict[str, object]] = {}
    for path in result_root.glob("latency/*/*/repeat_*/measurement.json"):
        measurement = json.loads(path.read_text())
        case = path.parents[2].name
        dtype = str(measurement["kv_cache_dtype"])
        grouped[(case, dtype)].append(float(measurement["latency_mean_ms"]))
        sample_meta[(case, dtype)] = measurement
    rows = []
    for (case, dtype), values in sorted(grouped.items()):
        meta = sample_meta[(case, dtype)]
        rows.append(
            {
                "case": case,
                "stage": meta["stage"],
                "batch_size": meta["batch_size"],
                "input_len": meta["input_len"],
                "num_heads": meta["num_heads"],
                "tp_size": meta["tp_size"],
                "kv_cache_dtype": dtype,
                "process_repeats": len(values),
                "latency_process_mean_ms": statistics.fmean(values),
                "latency_process_median_ms": statistics.median(values),
                "latency_process_min_ms": min(values),
                "latency_process_max_ms": max(values),
            }
        )
    return rows


def add_database_reference(root: Path, latency_rows: list[dict[str, object]]) -> None:
    tables: dict[str, dict[tuple[str, ...], float]] = {}
    for stage, filename in (
        ("prefill", "context_mla_perf.txt"),
        ("decode", "generation_mla_perf.txt"),
    ):
        path = root / "src/aiconfigurator/systems/data/h100_sxm/sglang/0.5.9" / filename
        mapping: dict[tuple[str, ...], float] = {}
        with path.open() as handle:
            for row in csv.DictReader(handle):
                key = (
                    row["mla_dtype"],
                    row["kv_cache_dtype"],
                    row["num_heads"],
                    row["batch_size"],
                    row["isl"],
                    row["tp_size"],
                    row["step"],
                )
                mapping[key] = float(row["latency"])
        tables[stage] = mapping
    for row in latency_rows:
        dtype = "bfloat16" if row["kv_cache_dtype"] == "bf16" else "fp8"
        stage = str(row["stage"])
        isl = row["input_len"] if stage == "prefill" else 1
        step = 0 if stage == "prefill" else row["input_len"]
        key = (
            "bfloat16",
            dtype,
            str(row["num_heads"]),
            str(row["batch_size"]),
            str(isl),
            str(row["tp_size"]),
            str(step),
        )
        row["database_latency_ms"] = tables[stage].get(key, math.nan)


def build_pair_summary(latency_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped = {(str(row["case"]), str(row["kv_cache_dtype"])): row for row in latency_rows}
    rows = []
    for case in CASE_ORDER:
        bf16 = grouped.get((case, "bf16"))
        fp8 = grouped.get((case, "fp8"))
        if bf16 is None or fp8 is None:
            continue
        rows.append(
            {
                "case": case,
                "stage": bf16["stage"],
                "batch_size": bf16["batch_size"],
                "input_len": bf16["input_len"],
                "num_heads": bf16["num_heads"],
                "tp_size": bf16["tp_size"],
                "bf16_graph_ms": bf16["latency_process_median_ms"],
                "fp8_graph_ms": fp8["latency_process_median_ms"],
                "fp8_over_bf16_graph": float(fp8["latency_process_median_ms"])
                / float(bf16["latency_process_median_ms"]),
                "bf16_database_ms": bf16["database_latency_ms"],
                "fp8_database_ms": fp8["database_latency_ms"],
                "fp8_over_bf16_database": float(fp8["database_latency_ms"])
                / float(bf16["database_latency_ms"]),
            }
        )
    return rows


def plot_latency(pair_rows: list[dict[str, object]], figure_dir: Path) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    for axis, stage in zip(axes, ("prefill", "decode")):
        rows = [row for row in pair_rows if row["stage"] == stage]
        x = np.arange(len(rows))
        width = 0.34
        bf16 = [float(row["bf16_graph_ms"]) for row in rows]
        fp8 = [float(row["fp8_graph_ms"]) for row in rows]
        axis.bar(x - width / 2, bf16, width, label="BF16/BF16", color="#377eb8")
        axis.bar(x + width / 2, fp8, width, label="BF16/FP8", color="#e41a1c")
        for index, row in enumerate(rows):
            top = max(bf16[index], fp8[index])
            axis.text(index, top * 1.08, f"{row['fp8_over_bf16_graph']:.2f}x", ha="center", fontsize=9)
        axis.set_xticks(x, [str(row["case"]).replace("_", "\n", 1) for row in rows])
        axis.set_yscale("log")
        axis.set_ylabel("CUDA Graph latency (ms, log scale)")
        axis.set_title(stage.capitalize())
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
    fig.suptitle("SGLang 0.5.9 collect_mla on H100: matched BF16 vs FP8 KV cache")
    fig.savefig(figure_dir / "graph_latency_comparison.png", dpi=180)
    fig.savefig(figure_dir / "graph_latency_comparison.pdf")
    plt.close(fig)


def plot_kernel_breakdown(category_rows: list[dict[str, object]], figure_dir: Path) -> None:
    values = {
        (str(row["case"]), str(row["kv_cache_dtype"]), str(row["category"])): float(
            row["duration_mean_us"]
        )
        for row in category_rows
    }
    fig, axes = plt.subplots(len(CASE_ORDER), 1, figsize=(13, 13.5), constrained_layout=False)
    legend_handles = {}
    for axis, case in zip(axes, CASE_ORDER):
        left = np.zeros(2)
        for category in CATEGORY_ORDER:
            width = np.array([values.get((case, dtype, category), 0.0) for dtype in ("bf16", "fp8")])
            if not np.any(width):
                continue
            bars = axis.barh(
                ["BF16/BF16", "BF16/FP8"],
                width,
                left=left,
                color=CATEGORY_COLORS[category],
                label=category,
            )
            legend_handles[category] = bars[0]
            left += width
        axis.set_title(case)
        axis.set_xlabel("Mean CUDA kernel duration per iteration (us)")
        axis.grid(axis="x", alpha=0.25)
        for y, total in enumerate(left):
            axis.text(total * 1.01 if total else 0, y, f"{total:.1f}", va="center", fontsize=8)
    fig.legend(
        [legend_handles[key] for key in CATEGORY_ORDER if key in legend_handles],
        [key for key in CATEGORY_ORDER if key in legend_handles],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=4,
    )
    fig.suptitle("NVTX-scoped CUDA kernel breakdown", y=0.945)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(figure_dir / "nsys_kernel_breakdown.png", dpi=180)
    fig.savefig(figure_dir / "nsys_kernel_breakdown.pdf")
    plt.close(fig)


def capture_source_evidence(result_root: Path) -> None:
    source_dir = result_root / "source_evidence"
    source_dir.mkdir(parents=True, exist_ok=True)
    snippets = {
        "flashattention_prefill_0_5_9.py.txt": (
            "/sgl-workspace/sglang/python/sglang/srt/layers/attention/flashattention_backend.py",
            "735,995p",
        ),
        "flashattention_decode_0_5_9.py.txt": (
            "/sgl-workspace/sglang/python/sglang/srt/layers/attention/flashattention_backend.py",
            "1075,1370p",
        ),
        "memory_pool_mla_0_5_9.py.txt": (
            "/sgl-workspace/sglang/python/sglang/srt/mem_cache/memory_pool.py",
            "1495,1555p",
        ),
    }
    for output_name, (source_path, sed_range) in snippets.items():
        completed = subprocess.run(
            ["docker", "run", "--rm", IMAGE, "sed", "-n", sed_range, source_path],
            text=True,
            capture_output=True,
            check=True,
        )
        (source_dir / output_name).write_text(completed.stdout)


def main() -> int:
    work_dir = Path(__file__).resolve().parent
    root = Path(__file__).resolve().parents[4]
    result_root = work_dir / "results"
    summary_dir = result_root / "summary"
    figure_dir = result_root / "figures"

    raw_rows: list[dict[str, object]] = []
    profile_rows: list[dict[str, object]] = []
    for report in sorted(result_root.glob("nsys/*/*/report.nsys-rep")):
        case_rows, profile_summary = extract_profile(report.parent)
        raw_rows.extend(case_rows)
        profile_rows.append(profile_summary)
    if not raw_rows:
        raise RuntimeError("No Nsight reports found")

    category_rows = summarize_categories(raw_rows)
    latency_rows = load_graph_latency(result_root)
    add_database_reference(root, latency_rows)
    pair_rows = build_pair_summary(latency_rows)

    write_csv(summary_dir / "kernel_launches.csv", raw_rows)
    write_csv(summary_dir / "kernel_category_summary.csv", category_rows)
    write_csv(summary_dir / "nsys_profile_summary.csv", profile_rows)
    write_csv(summary_dir / "graph_latency_summary.csv", latency_rows)
    write_csv(summary_dir / "bf16_fp8_pair_summary.csv", pair_rows)
    plot_latency(pair_rows, figure_dir)
    plot_kernel_breakdown(category_rows, figure_dir)
    capture_source_evidence(result_root)

    summary = {
        "profile_count": len(profile_rows),
        "kernel_launch_rows": len(raw_rows),
        "category_rows": len(category_rows),
        "latency_rows": len(latency_rows),
        "pair_rows": pair_rows,
    }
    (summary_dir / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
