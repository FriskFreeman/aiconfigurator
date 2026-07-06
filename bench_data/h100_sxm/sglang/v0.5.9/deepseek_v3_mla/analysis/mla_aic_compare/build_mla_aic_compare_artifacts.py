#!/usr/bin/env python3
"""Build prefill MLA real-vs-AIC summary tables and static plots.

This script is intentionally self-contained so the comparison can be
regenerated from the checked-in manifest and copied nsys parser CSV files.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[7]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from aiconfigurator.sdk import common  # noqa: E402
from aiconfigurator.sdk import operations as ops  # noqa: E402
from aiconfigurator.sdk.perf_database import PerfDataNotAvailableError, PerfDatabase  # noqa: E402

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)


TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
}

COLORS = {
    "blue": "#A3BEFA",
    "blue_mid": "#5477C4",
    "blue_dark": "#2E4780",
    "gold": "#FFE15B",
    "gold_mid": "#B8A037",
    "orange": "#F0986E",
    "orange_mid": "#CC6F47",
    "olive": "#A3D576",
    "olive_mid": "#71B436",
    "pink": "#F390CA",
    "pink_mid": "#BD569B",
    "neutral": "#C5CAD3",
    "neutral_mid": "#7A828F",
    "neutral_dark": "#464C55",
}

REAL_CHILD_ORDER = [
    "fused_qkv_a_proj_with_mqa",
    "q_a_layernorm",
    "kv_a_layernorm",
    "q_b_proj",
    "rotary_emb",
    "kv_b_proj",
    "attn_mha",
    "attn_mqa",
    "o_proj",
]

REAL_CHILD_LABELS = {
    "fused_qkv_a_proj_with_mqa": "qkv_a_proj",
    "q_a_layernorm": "q_a_norm",
    "kv_a_layernorm": "kv_a_norm",
    "q_b_proj": "q_b_proj",
    "rotary_emb": "rotary",
    "kv_b_proj": "kv_b_proj",
    "attn_mha": "attn_mha",
    "attn_mqa": "attn_mqa",
    "o_proj": "o_proj",
}

REAL_CHILD_COLORS = {
    "fused_qkv_a_proj_with_mqa": COLORS["blue"],
    "q_a_layernorm": "#CEDFFE",
    "kv_a_layernorm": "#EAF1FE",
    "q_b_proj": COLORS["gold"],
    "rotary_emb": "#FFEDDE",
    "kv_b_proj": COLORS["orange"],
    "attn_mha": COLORS["olive"],
    "attn_mqa": COLORS["olive_mid"],
    "o_proj": COLORS["pink"],
}

DEEPSEEK_FALLBACK_ORDER = [
    "context_downscale_gemm",
    "context_q_b_proj_gemm",
    "context_kv_b_proj_gemm",
    "context_attention",
    "context_proj_gemm",
]

DEEPSEEK_FALLBACK_COLORS = {
    "context_downscale_gemm": COLORS["blue"],
    "context_q_b_proj_gemm": COLORS["gold"],
    "context_kv_b_proj_gemm": COLORS["orange"],
    "context_attention": COLORS["olive"],
    "context_proj_gemm": COLORS["pink"],
}

WIDEEP_ORDER = [
    "context_qkv_a_proj_gemm",
    "wideep_context_mla",
]

WIDEEP_COLORS = {
    "context_qkv_a_proj_gemm": COLORS["blue"],
    "wideep_context_mla": COLORS["olive"],
}


@dataclass(frozen=True)
class ManifestEntry:
    tag: str
    status: str
    csv_path: Path
    batch_size: int
    fresh_lens: list[int]
    prefix_lens: list[int]
    tp_size: int
    attention_backend: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bench-root",
        default="bench_data/h100_sxm/sglang/v0.5.9/deepseek_v3_mla",
    )
    parser.add_argument("--systems-root", default="src/aiconfigurator/systems")
    parser.add_argument("--system", default="h100_sxm")
    parser.add_argument("--backend", default="sglang")
    parser.add_argument("--version", default="0.5.9")
    return parser.parse_args()


def add_chart_header(fig, ax, title: str, subtitle: str) -> None:
    title = textwrap.fill(title, width=92, break_long_words=False)
    subtitle = textwrap.fill(subtitle, width=128, break_long_words=False)
    ax.set_title("")
    fig.subplots_adjust(top=0.84)
    left = ax.get_position().x0
    fig.text(left, 0.965, title, ha="left", va="top", fontsize=16, fontweight="bold", color=TOKENS["ink"])
    fig.text(left, 0.915, subtitle, ha="left", va="top", fontsize=10.5, color=TOKENS["muted"])


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": TOKENS["surface"],
            "axes.facecolor": TOKENS["panel"],
            "axes.edgecolor": TOKENS["axis"],
            "axes.labelcolor": TOKENS["ink"],
            "xtick.color": TOKENS["muted"],
            "ytick.color": TOKENS["muted"],
            "grid.color": TOKENS["grid"],
            "grid.linewidth": 0.8,
            "font.family": ["DejaVu Sans", "sans-serif"],
            "font.size": 10,
            "axes.grid": True,
        }
    )


def parse_int_list_csv(text: str | None) -> list[int]:
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return []
    text = str(text).strip()
    if not text:
        return []
    return [int(part) for part in text.split(",") if part.strip()]


def safe_json_loads(text: Any) -> Any:
    if text is None:
        return None
    if isinstance(text, float) and math.isnan(text):
        return None
    text = str(text).strip()
    if not text:
        return None
    return json.loads(text)


def source_of(result: Any) -> str:
    return str(getattr(result, "source", "silicon"))


def pct_gap(sim: float | None, real: float | None) -> float | None:
    if sim is None or real in (None, 0, 0.0) or pd.isna(real):
        return None
    return (sim - float(real)) / float(real) * 100.0


def load_manifest(path: Path) -> list[ManifestEntry]:
    entries: list[ManifestEntry] = []
    with path.open() as f:
        for row in csv.DictReader(f):
            entries.append(
                ManifestEntry(
                    tag=row["tag"],
                    status=row["status"],
                    csv_path=Path(row["csv_path"]),
                    batch_size=int(row["batch_size"]),
                    fresh_lens=parse_int_list_csv(row.get("fresh_lens")),
                    prefix_lens=parse_int_list_csv(row.get("prefix_lens")),
                    tp_size=int(row["tp_size"]),
                    attention_backend=row["attention_backend"],
                )
            )
    return entries


def uniform_shape_from_row(row: pd.Series, entry: ManifestEntry) -> tuple[bool, str, dict[str, Any]]:
    prefix_lens = safe_json_loads(row.get("chunk_req_prefix_lens_json")) or list(entry.prefix_lens)
    fresh_lens = safe_json_loads(row.get("chunk_req_fresh_lens_json")) or list(entry.fresh_lens)
    total_lens = safe_json_loads(row.get("chunk_req_total_seq_lens_json"))
    if not total_lens:
        total_lens = [p + f for p, f in zip(prefix_lens, fresh_lens)]
    prefix_lens = [int(x) for x in prefix_lens]
    fresh_lens = [int(x) for x in fresh_lens]
    total_lens = [int(x) for x in total_lens]
    if not prefix_lens or not fresh_lens or not total_lens:
        return False, "missing_shape_lists", {}
    if not (len(prefix_lens) == len(fresh_lens) == len(total_lens)):
        return False, "shape_list_length_mismatch", {
            "prefix_lens": prefix_lens,
            "fresh_lens": fresh_lens,
            "total_lens": total_lens,
        }
    if len(set(prefix_lens)) > 1 or len(set(fresh_lens)) > 1 or len(set(total_lens)) > 1:
        return False, "non_uniform_shape_lists", {
            "prefix_lens": prefix_lens,
            "fresh_lens": fresh_lens,
            "total_lens": total_lens,
        }
    return True, "ok", {
        "batch_size": len(prefix_lens),
        "prefix_len": prefix_lens[0],
        "fresh_len": fresh_lens[0],
        "total_seq_len": total_lens[0],
        "chunk_req_prefix_lens_json": json.dumps(prefix_lens),
        "chunk_req_fresh_lens_json": json.dumps(fresh_lens),
        "chunk_req_total_seq_lens_json": json.dumps(total_lens),
    }


def read_real_layer_rows(entries: list[ManifestEntry]) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    for entry in entries:
        if entry.status != "ok":
            skipped_rows.append({"tag": entry.tag, "reason": f"manifest_status_{entry.status}", "csv_path": str(entry.csv_path)})
            continue
        if not entry.csv_path.exists():
            skipped_rows.append({"tag": entry.tag, "reason": "missing_csv", "csv_path": str(entry.csv_path)})
            continue
        df = pd.read_csv(entry.csv_path)
        for _, row in df.iterrows():
            if row.get("stage") != "prefill":
                continue
            if int(row.get("layer_id", -1)) == 0:
                skipped_rows.append({
                    "tag": entry.tag,
                    "run_instance_name": row.get("run_instance_name"),
                    "layer_id": row.get("layer_id"),
                    "reason": "first_layer_ignored",
                    "csv_path": str(entry.csv_path),
                })
                continue
            ok, reason, shape = uniform_shape_from_row(row, entry)
            if not ok:
                skipped_rows.append({
                    "tag": entry.tag,
                    "run_instance_name": row.get("run_instance_name"),
                    "layer_id": row.get("layer_id"),
                    "reason": reason,
                    "shape_info": json.dumps(shape, ensure_ascii=False),
                    "csv_path": str(entry.csv_path),
                })
                continue
            children = safe_json_loads(row.get("child_timing_json")) or []
            kernels = safe_json_loads(row.get("kernel_summary_json")) or []
            detail_rows.append(
                {
                    "tag": entry.tag,
                    "run_instance_name": row["run_instance_name"],
                    "layer_id": int(row["layer_id"]),
                    "event_order_index": int(row["event_order_index"]),
                    "tp_size": int(entry.tp_size),
                    "attention_backend": entry.attention_backend,
                    "attention_module": row.get("attention_module"),
                    **shape,
                    "module_total_to_last_kernel_ms": float(row.get("module_total_to_last_kernel_ms") or 0.0),
                    "module_host_duration_ms": float(row.get("module_host_duration_ms") or 0.0),
                    "module_gpu_makespan_ms": float(row.get("module_gpu_makespan_ms") or 0.0),
                    "module_gpu_kernel_time_sum_ms": float(row.get("module_gpu_kernel_time_sum_ms") or 0.0),
                    "module_gpu_kernel_count": int(row.get("module_gpu_kernel_count") or 0),
                    "module_first_kernel_name": row.get("module_first_kernel_name"),
                    "module_last_kernel_name": row.get("module_last_kernel_name"),
                    "child_timing_json": json.dumps(children, ensure_ascii=False),
                    "kernel_summary_json": json.dumps(kernels, ensure_ascii=False),
                }
            )
    return pd.DataFrame(detail_rows), pd.DataFrame(skipped_rows)


def build_real_kernel_stack(real_detail: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in real_detail.iterrows():
        children = safe_json_loads(row["child_timing_json"]) or []
        summaries = {x.get("canonical_name"): x for x in (safe_json_loads(row["kernel_summary_json"]) or [])}
        for child in children:
            canonical = child.get("canonical_name")
            if canonical not in REAL_CHILD_ORDER:
                continue
            summary = summaries.get(canonical) or {}
            top_kernels = summary.get("top_gpu_kernels") or []
            if top_kernels:
                for kernel_order, kernel in enumerate(top_kernels, start=1):
                    duration_ms = float(kernel.get("gpu_time_ms") or 0.0)
                    kernel_name = kernel.get("short_name")
                    duration_source = "kernel_summary_json"
                    rows.append(
                        {
                            "tag": row["tag"],
                            "run_instance_name": row["run_instance_name"],
                            "layer_id": row["layer_id"],
                            "child_name": canonical,
                            "child_label": REAL_CHILD_LABELS.get(canonical, canonical),
                            "child_order": REAL_CHILD_ORDER.index(canonical) + 1,
                            "kernel_order": kernel_order,
                            "kernel_name": kernel_name,
                            "stack_component": REAL_CHILD_LABELS.get(canonical, canonical),
                            "duration_ms": duration_ms,
                            "duration_source": duration_source,
                            "kernel_detail_json": json.dumps(kernel, ensure_ascii=False),
                        }
                    )
            else:
                rows.append(
                    {
                        "tag": row["tag"],
                        "run_instance_name": row["run_instance_name"],
                        "layer_id": row["layer_id"],
                        "child_name": canonical,
                        "child_label": REAL_CHILD_LABELS.get(canonical, canonical),
                        "child_order": REAL_CHILD_ORDER.index(canonical) + 1,
                        "kernel_order": 1,
                        "kernel_name": child.get("first_kernel_name"),
                        "stack_component": REAL_CHILD_LABELS.get(canonical, canonical),
                        "duration_ms": float(child.get("gpu_kernel_time_sum_ms") or 0.0),
                        "duration_source": "child_timing_json_gpu_kernel_time_sum",
                        "kernel_detail_json": json.dumps(child, ensure_ascii=False),
                    }
                )
    return pd.DataFrame(rows)


def query_result(op: ops.Operation, db: PerfDatabase, *, x: int, batch_size: int, s: int, prefix: int) -> tuple[float, str]:
    result = op.query(db, x=x, batch_size=batch_size, s=s, prefix=prefix, beam_width=1, model_name="DeepSeek-V3")
    return float(result), source_of(result)


def query_deepseek_fallback_stack(real_cases: pd.DataFrame, db: PerfDatabase) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    for _, case in real_cases.iterrows():
        batch_size = int(case["batch_size"])
        fresh_len = int(case["fresh_len"])
        prefix_len = int(case["prefix_len"])
        tp_size = int(case["tp_size"])
        token_count = batch_size * fresh_len
        num_heads = 128 // tp_size
        h = 7168
        gemm_quant_mode = common.GEMMQuantMode.fp8_block
        kvcache_quant_mode = common.KVCacheQuantMode.fp8
        fmha_quant_mode = common.FMHAQuantMode.bfloat16

        try:
            primary = ops.MLAModule(
                "context_mla_module",
                1,
                True,
                num_heads,
                kvcache_quant_mode,
                fmha_quant_mode,
                gemm_quant_mode,
            ).query(db, batch_size=batch_size, s=fresh_len, prefix=prefix_len, x=token_count)
            validation_rows.append(
                {
                    "tag": case["tag"],
                    "primary_op": "context_mla_module",
                    "status": "unexpected_success",
                    "latency_ms": float(primary),
                    "source": source_of(primary),
                    "message": "",
                }
            )
        except Exception as exc:
            validation_rows.append(
                {
                    "tag": case["tag"],
                    "primary_op": "context_mla_module",
                    "status": "miss_expected",
                    "latency_ms": None,
                    "source": None,
                    "message": f"{type(exc).__name__}: {exc}",
                    "is_perf_data_not_available": isinstance(exc, PerfDataNotAvailableError),
                }
            )

        stack_ops: list[tuple[str, ops.Operation]] = [
            ("context_downscale_gemm", ops.GEMM("context_downscale_gemm", 1, 2112, h, gemm_quant_mode)),
            ("context_q_b_proj_gemm", ops.GEMM("context_q_b_proj_gemm", 1, 24576 // tp_size, 1536, gemm_quant_mode)),
            ("context_kv_b_proj_gemm", ops.GEMM("context_kv_b_proj_gemm", 1, 32768 // tp_size, 512, gemm_quant_mode)),
            ("context_attention", ops.ContextMLA("context_attention", 1, num_heads, kvcache_quant_mode, fmha_quant_mode)),
            ("context_proj_gemm", ops.GEMM("context_proj_gemm", 1, h, 128 * 128 // tp_size, gemm_quant_mode)),
        ]
        for order, (op_name, op) in enumerate(stack_ops, start=1):
            latency_ms, source = query_result(op, db, x=token_count, batch_size=batch_size, s=fresh_len, prefix=prefix_len)
            rows.append(
                {
                    "tag": case["tag"],
                    "aic_model": "DeepSeekModel",
                    "op_name": op_name,
                    "op_order": order,
                    "latency_ms": latency_ms,
                    "source": source,
                    "batch_size": batch_size,
                    "fresh_len": fresh_len,
                    "prefix_len": prefix_len,
                    "tp_size": tp_size,
                    "x_tokens": token_count,
                    "kvcache_quant_mode": kvcache_quant_mode.name,
                    "fmha_quant_mode": fmha_quant_mode.name,
                    "gemm_quant_mode": gemm_quant_mode.name,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(validation_rows)


def query_wideep_stack(real_cases: pd.DataFrame, db: PerfDatabase) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, case in real_cases.iterrows():
        batch_size = int(case["batch_size"])
        fresh_len = int(case["fresh_len"])
        prefix_len = int(case["prefix_len"])
        tp_size = int(case["tp_size"])
        token_count = batch_size * fresh_len
        h = 7168
        gemm_quant_mode = common.GEMMQuantMode.fp8_block
        kvcache_quant_mode = common.KVCacheQuantMode.fp8
        fmha_quant_mode = common.FMHAQuantMode.fp8_block

        qkv_op = ops.GEMM(
            "context_qkv_a_proj_gemm",
            1,
            1536 + 512 + 64,
            h,
            gemm_quant_mode,
            scale_num_tokens=tp_size,
        )
        qkv_latency, qkv_source = query_result(qkv_op, db, x=token_count, batch_size=batch_size, s=fresh_len, prefix=prefix_len)
        rows.append(
            {
                "tag": case["tag"],
                "aic_model": "WideEPDeepSeekModel",
                "op_name": "context_qkv_a_proj_gemm",
                "op_order": 1,
                "latency_ms": qkv_latency,
                "source": qkv_source,
                "included_policy": "include_one_qkv_a_gemm_only",
                "batch_size": batch_size,
                "fresh_len": fresh_len,
                "prefix_len": prefix_len,
                "tp_size": tp_size,
                "x_tokens": token_count,
                "kvcache_quant_mode": kvcache_quant_mode.name,
                "fmha_quant_mode": fmha_quant_mode.name,
                "gemm_quant_mode": gemm_quant_mode.name,
                "attention_backend": "fa3",
            }
        )

        wideep = ops.WideEPContextMLA(
            "wideep_context_mla",
            1,
            tp_size,
            kvcache_quant_mode,
            fmha_quant_mode,
            "fa3",
        )
        mla_latency, mla_source = query_result(wideep, db, x=token_count, batch_size=batch_size, s=fresh_len, prefix=prefix_len)
        rows.append(
            {
                "tag": case["tag"],
                "aic_model": "WideEPDeepSeekModel",
                "op_name": "wideep_context_mla",
                "op_order": 2,
                "latency_ms": mla_latency,
                "source": mla_source,
                "included_policy": "include_wideep_context_mla_module",
                "batch_size": batch_size,
                "fresh_len": fresh_len,
                "prefix_len": prefix_len,
                "tp_size": tp_size,
                "x_tokens": token_count,
                "kvcache_quant_mode": kvcache_quant_mode.name,
                "fmha_quant_mode": fmha_quant_mode.name,
                "gemm_quant_mode": gemm_quant_mode.name,
                "attention_backend": "fa3",
            }
        )
    return pd.DataFrame(rows)


def summarize_cases(real_detail: pd.DataFrame, real_stack: pd.DataFrame, deepseek_stack: pd.DataFrame, wideep_stack: pd.DataFrame) -> pd.DataFrame:
    real_summary = (
        real_detail.groupby("tag", as_index=False)
        .agg(
            layer_sample_count=("layer_id", "count"),
            batch_size=("batch_size", "first"),
            fresh_len=("fresh_len", "first"),
            prefix_len=("prefix_len", "first"),
            total_seq_len=("total_seq_len", "first"),
            tp_size=("tp_size", "first"),
            real_module_gpu_makespan_mean_ms=("module_gpu_makespan_ms", "mean"),
            real_module_gpu_makespan_median_ms=("module_gpu_makespan_ms", "median"),
            real_module_gpu_kernel_time_sum_mean_ms=("module_gpu_kernel_time_sum_ms", "mean"),
            real_module_gpu_kernel_time_sum_median_ms=("module_gpu_kernel_time_sum_ms", "median"),
            real_module_total_to_last_kernel_mean_ms=("module_total_to_last_kernel_ms", "mean"),
        )
    )
    real_stack_total = real_stack.groupby("tag", as_index=False).agg(real_kernel_stack_mean_ms=("duration_ms", "sum"))
    # The row above sums every layer. Convert to mean per layer for case-level comparison.
    counts = real_detail.groupby("tag", as_index=False).agg(layer_sample_count_for_stack=("layer_id", "count"))
    real_stack_total = real_stack_total.merge(counts, on="tag", how="left")
    real_stack_total["real_kernel_stack_mean_ms"] = (
        real_stack_total["real_kernel_stack_mean_ms"] / real_stack_total["layer_sample_count_for_stack"]
    )
    summary = real_summary.merge(real_stack_total[["tag", "real_kernel_stack_mean_ms"]], on="tag", how="left")

    for name, stack_df in [("deepseek_fallback", deepseek_stack), ("wideep", wideep_stack)]:
        total = stack_df.groupby("tag", as_index=False).agg(**{f"aic_{name}_total_ms": ("latency_ms", "sum")})
        summary = summary.merge(total, on="tag", how="left")
        summary[f"aic_{name}_gap_pct_vs_real_makespan_mean"] = summary.apply(
            lambda r: pct_gap(r.get(f"aic_{name}_total_ms"), r.get("real_module_gpu_makespan_mean_ms")), axis=1
        )
        summary[f"aic_{name}_gap_pct_vs_real_kernel_sum_mean"] = summary.apply(
            lambda r: pct_gap(r.get(f"aic_{name}_total_ms"), r.get("real_module_gpu_kernel_time_sum_mean_ms")), axis=1
        )
    summary = summary.sort_values(["total_seq_len", "batch_size", "fresh_len", "prefix_len", "tag"]).reset_index(drop=True)
    return summary


def aggregate_real_stack(real_stack: pd.DataFrame) -> pd.DataFrame:
    # A child module can launch multiple CUDA kernels. Aggregate at
    # case/layer/child first, then average layers; otherwise taking the mean of
    # individual kernel rows would undercount multi-kernel children.
    per_layer_child = (
        real_stack.groupby(
            ["tag", "run_instance_name", "layer_id", "child_name", "child_label", "child_order"],
            as_index=False,
        )
        .agg(
            child_duration_ms=("duration_ms", "sum"),
            kernel_row_count=("duration_ms", "count"),
            kernel_names=("kernel_name", lambda s: ",".join(sorted({str(x) for x in s if pd.notna(x)}))),
        )
    )
    return (
        per_layer_child.groupby(["tag", "child_name", "child_label", "child_order"], as_index=False)
        .agg(
            mean_duration_ms=("child_duration_ms", "mean"),
            median_duration_ms=("child_duration_ms", "median"),
            sample_count=("child_duration_ms", "count"),
            kernel_row_count_mean=("kernel_row_count", "mean"),
            kernel_names=("kernel_names", lambda s: ",".join(sorted({str(x) for x in s if pd.notna(x)}))),
        )
        .sort_values(["tag", "child_order"])
    )


def make_case_labels(summary: pd.DataFrame) -> dict[str, str]:
    return {
        row.tag: f"b{int(row.batch_size)} f{int(row.fresh_len)} p{int(row.prefix_len)}"
        for row in summary.itertuples(index=False)
    }


def save_boxplot(real_detail: pd.DataFrame, summary: pd.DataFrame, out_dir: Path) -> None:
    apply_style()
    labels = make_case_labels(summary)
    plot_df = real_detail.copy()
    plot_df["case_label"] = plot_df["tag"].map(labels)
    order = [labels[tag] for tag in summary["tag"]]
    fig, ax = plt.subplots(figsize=(max(12, len(order) * 0.58), 6.4))
    grouped = [
        plot_df.loc[plot_df["case_label"] == label, "module_gpu_kernel_time_sum_ms"].astype(float).to_numpy()
        for label in order
    ]
    positions = np.arange(len(order))
    bp = ax.boxplot(
        grouped,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        showfliers=True,
        medianprops={"color": COLORS["blue_dark"], "linewidth": 1.4},
        boxprops={"facecolor": COLORS["blue"], "edgecolor": COLORS["neutral_dark"], "linewidth": 0.9},
        whiskerprops={"color": COLORS["neutral_dark"], "linewidth": 0.9},
        capprops={"color": COLORS["neutral_dark"], "linewidth": 0.9},
        flierprops={"marker": "o", "markersize": 2.5, "markerfacecolor": COLORS["blue_dark"], "markeredgewidth": 0.0, "alpha": 0.7},
    )
    # Keep linters quiet and make the intended Matplotlib ownership explicit.
    _ = bp
    for pos, values in zip(positions, grouped):
        jitter = np.linspace(-0.08, 0.08, len(values)) if len(values) > 1 else np.array([0.0])
        ax.scatter(
            np.full(len(values), pos) + jitter,
            values,
            s=13,
            color=COLORS["blue_dark"],
            alpha=0.65,
            zorder=3,
        )
    ax.set_xticks(positions)
    ax.set_xticklabels(order)
    ax.set_xlabel("Prefill case")
    ax.set_ylabel("Real self_attn GPU kernel sum per layer (ms)")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    add_chart_header(
        fig,
        ax,
        "Real prefill self_attn layer distribution",
        "Each box uses layer samples after excluding layer 0; values are nsys GPU kernel-time sums inside the module.",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    for ext in ["png", "svg"]:
        fig.savefig(out_dir / f"prefill_real_layer_boxplot.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_stacked_compare(
    summary: pd.DataFrame,
    real_stack_case: pd.DataFrame,
    aic_stack: pd.DataFrame,
    *,
    aic_total_col: str,
    title: str,
    subtitle: str,
    output_stem: str,
    component_order: list[str],
    component_colors: dict[str, str],
    out_dir: Path,
    sort_panels: list[tuple[str, str | None]] | None = None,
) -> None:
    apply_style()
    base_tags = list(summary["tag"])
    labels = make_case_labels(summary)
    width = 0.24

    def case_values(tag: str) -> tuple[float, float, float]:
        real_ks = float(summary.loc[summary["tag"] == tag, "real_module_gpu_kernel_time_sum_mean_ms"].iloc[0])
        real_ms = float(summary.loc[summary["tag"] == tag, "real_module_gpu_makespan_mean_ms"].iloc[0])
        aic_total = float(summary.loc[summary["tag"] == tag, aic_total_col].iloc[0])
        return real_ms, real_ks, aic_total

    def gap_value(tag: str, gap_key: str) -> float | None:
        real_ms, real_ks, aic_total = case_values(tag)
        if gap_key == "M/K":
            return pct_gap(real_ms, real_ks)
        if gap_key == "A/K":
            return pct_gap(aic_total, real_ks)
        if gap_key == "A/M":
            return pct_gap(aic_total, real_ms)
        raise ValueError(f"Unknown gap key: {gap_key}")

    def ordered_tags(gap_key: str | None) -> list[str]:
        if gap_key is None:
            return list(base_tags)
        return sorted(
            base_tags,
            key=lambda tag: (
                -abs(float(gap_value(tag, gap_key))) if gap_value(tag, gap_key) is not None and not pd.isna(gap_value(tag, gap_key)) else float("inf"),
                labels[tag],
            ),
        )

    panels = sort_panels or [("Original order", None)]
    panel_count = len(panels)
    fig_height = 7.6 if panel_count == 1 else max(10.5, 5.8 * panel_count)
    fig, axes = plt.subplots(
        panel_count,
        1,
        figsize=(max(14.5, len(base_tags) * 0.78), fig_height),
        squeeze=False,
    )

    global_aic_totals = np.array([case_values(tag)[2] for tag in base_tags], dtype=float)
    global_makespan = np.array([case_values(tag)[0] for tag in base_tags], dtype=float)
    global_kernel_sum = np.array([case_values(tag)[1] for tag in base_tags], dtype=float)
    y_max = max(np.nanmax(np.r_[global_aic_totals, global_makespan, global_kernel_sum]), 0.001)
    gap_label_specs = [
        ("M/K", COLORS["neutral_dark"]),
        ("A/K", COLORS["orange_mid"]),
        ("A/M", COLORS["blue_mid"]),
    ]

    def draw_panel(ax, tags: list[str], panel_title: str, show_legend: bool) -> None:
        x = np.arange(len(tags))
        makespan = np.array([case_values(tag)[0] for tag in tags], dtype=float)
        kernel_sum = np.array([case_values(tag)[1] for tag in tags], dtype=float)
        aic_totals = np.array([case_values(tag)[2] for tag in tags], dtype=float)
        ax.bar(
            x - width,
            makespan,
            width,
            color="#E2E5EA",
            edgecolor=COLORS["neutral_dark"],
            linewidth=0.45,
            label="real GPU makespan mean",
        )

        real_bottom = np.zeros(len(tags))
        for child in REAL_CHILD_ORDER:
            values = []
            for tag in tags:
                match = real_stack_case[(real_stack_case["tag"] == tag) & (real_stack_case["child_name"] == child)]
                values.append(float(match["mean_duration_ms"].iloc[0]) if not match.empty else 0.0)
            ax.bar(
                x,
                values,
                width,
                bottom=real_bottom,
                color=REAL_CHILD_COLORS.get(child, COLORS["neutral"]),
                edgecolor=COLORS["neutral_dark"],
                linewidth=0.35,
                label=f"real {REAL_CHILD_LABELS.get(child, child)}",
            )
            real_bottom += np.array(values)

        aic_bottom = np.zeros(len(tags))
        for op_name in component_order:
            values = []
            for tag in tags:
                match = aic_stack[(aic_stack["tag"] == tag) & (aic_stack["op_name"] == op_name)]
                values.append(float(match["latency_ms"].iloc[0]) if not match.empty else 0.0)
            ax.bar(
                x + width,
                values,
                width,
                bottom=aic_bottom,
                color=component_colors.get(op_name, COLORS["neutral"]),
                edgecolor=COLORS["neutral_dark"],
                linewidth=0.35,
                hatch="//" if len(component_order) <= 2 else "",
                label=f"AIC {op_name}",
            )
            aic_bottom += np.array(values)

        for i, tag in enumerate(tags):
            real_ms, real_ks, aic_total = case_values(tag)
            gap_values = [
                pct_gap(real_ms, real_ks),
                pct_gap(aic_total, real_ks),
                pct_gap(aic_total, real_ms),
            ]
            group_top = max(real_ms, real_ks, aic_total)
            for xpos, value in [
                (x[i] - width, real_ms),
                (x[i], real_ks),
                (x[i] + width, aic_total),
            ]:
                ax.text(
                    xpos,
                    float(value) + y_max * 0.018,
                    f"{float(value):.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=6.8,
                    color=TOKENS["ink"],
                    rotation=90,
                )
            for offset, ((label, color), value) in enumerate(zip(gap_label_specs, gap_values)):
                gap_text = "n/a" if value is None or pd.isna(value) else f"{value:+.0f}%"
                ax.text(
                    x[i],
                    group_top + y_max * (0.105 + offset * 0.052),
                    f"{label} {gap_text}",
                    ha="center",
                    va="bottom",
                    fontsize=8.3,
                    fontweight="bold",
                    color=color,
                )

        ax.set_title(panel_title, loc="left", fontsize=11, fontweight="bold", color=TOKENS["ink"], pad=9)
        ax.set_xticks(x)
        ax.set_xticklabels([labels[tag] for tag in tags], rotation=45, ha="right")
        ax.set_xlabel("Prefill case")
        ax.set_ylabel("Latency per layer (ms)")
        ax.grid(axis="y", linestyle="--", alpha=0.6)
        ax.set_ylim(0, y_max * 1.55)
        if show_legend:
            handles, labels_ = ax.get_legend_handles_labels()
            # Keep legend compact: show all AIC components and only the most important real components.
            keep = []
            for h, l in zip(handles, labels_):
                if l.startswith("AIC") or l in {"real qkv_a_proj", "real q_b_proj", "real kv_b_proj", "real attn_mha", "real o_proj", "real GPU makespan mean"}:
                    keep.append((h, l))
            ax.legend(
                [h for h, _ in keep],
                [l for _, l in keep],
                loc="upper left",
                bbox_to_anchor=(0.0, 1.03),
                ncol=3,
                frameon=False,
                fontsize=8,
            )

    for panel_idx, (panel_title, gap_key) in enumerate(panels):
        draw_panel(axes[panel_idx][0], ordered_tags(gap_key), panel_title, show_legend=(panel_idx == 0))
    add_chart_header(fig, axes[0][0], title, subtitle)
    fig.tight_layout(rect=(0, 0, 1, 0.84), h_pad=2.4)
    for ext in ["png", "svg"]:
        fig.savefig(out_dir / f"{output_stem}.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_summary_md(
    out_dir: Path,
    real_detail: pd.DataFrame,
    skipped: pd.DataFrame,
    summary: pd.DataFrame,
    validation: pd.DataFrame,
) -> None:
    def mape(col: str, real_col: str) -> float:
        vals = []
        for _, row in summary.iterrows():
            gap = pct_gap(row.get(col), row.get(real_col))
            if gap is not None:
                vals.append(abs(gap))
        return float(np.mean(vals)) if vals else float("nan")

    expected_primary_miss = (
        validation["status"].eq("miss_expected").all() if not validation.empty else False
    )
    lines = [
        "# MLA AIC Compare 汇总说明",
        "",
        "## 范围",
        "",
        "- 仅包含 prefill；所有 decode 行已排除。",
        "- 实机侧每个 case/layer 为一条样本，并按要求排除 `layer_id=0`。",
        "- 实机主口径保留 `module_gpu_makespan_ms` 和 `module_gpu_kernel_time_sum_ms`，同时展开 `kernel_summary_json` 记录每个子模块底层 CUDA kernel 明细。",
        "- 普通 `DeepSeekModel` AIC 侧验证 `context_mla_module` primary miss 后，按 SDK `FallbackOp` 的 5 个 fallback op 求和。",
        "- `WideEPDeepSeekModel` AIC 侧纳入一个 `context_qkv_a_proj_gemm` 与 `WideEPContextMLA`；源码中重复出现的 `context_downscale_gemm` 不再重复计入。",
        "",
        "## 规模",
        "",
        f"- 实机 layer 样本数: `{len(real_detail)}`",
        f"- case 数: `{summary['tag'].nunique()}`",
        f"- 跳过记录数: `{len(skipped)}`，其中包含按规则忽略的首层。",
        f"- `context_mla_module` primary miss 验证: `{'通过' if expected_primary_miss else '存在非预期结果'}`。",
        "",
        "## 平均绝对百分比误差",
        "",
        f"- 普通 DeepSeek fallback vs real GPU makespan mean: `{mape('aic_deepseek_fallback_total_ms', 'real_module_gpu_makespan_mean_ms'):.2f}%`",
        f"- 普通 DeepSeek fallback vs real GPU kernel-sum mean: `{mape('aic_deepseek_fallback_total_ms', 'real_module_gpu_kernel_time_sum_mean_ms'):.2f}%`",
        f"- WideEP aligned stack vs real GPU makespan mean: `{mape('aic_wideep_total_ms', 'real_module_gpu_makespan_mean_ms'):.2f}%`",
        f"- WideEP aligned stack vs real GPU kernel-sum mean: `{mape('aic_wideep_total_ms', 'real_module_gpu_kernel_time_sum_mean_ms'):.2f}%`",
        "",
        "## 输出文件",
        "",
        "- `prefill_real_layer_detail.csv`: 实机每 case/layer 明细，保留原始 child/kernel JSON。",
        "- `prefill_real_kernel_stack.csv`: 实机每 layer 的子模块/kernel 展开明细。",
        "- `prefill_real_kernel_stack_case_mean.csv`: 实机每 case 的子模块均值堆叠输入。",
        "- `prefill_aic_deepseek_fallback_stack.csv`: 普通 DeepSeek fallback 小算子查表明细。",
        "- `prefill_aic_deepseek_primary_validation.csv`: `context_mla_module` primary miss 验证。",
        "- `prefill_aic_wideep_stack.csv`: WideEP 对齐 op 查表明细。",
        "- `prefill_compare_summary.csv`: case 级总表，含 gap 百分比。",
        "- `prefill_real_layer_boxplot.png/svg`: 实机跨层重复样本箱线图。",
        "- `prefill_deepseek_fallback_stacked_compare.png/svg`: 普通 DeepSeek fallback 堆叠对比。",
        "- `prefill_wideep_stacked_compare.png/svg`: WideEP 对齐堆叠对比。",
        "",
        "## 注意",
        "",
        "- AIC 普通 DeepSeek fallback 使用 `kvcache_quant_mode=fp8`、`fmha_quant_mode=bfloat16`、`gemm_quant_mode=fp8_block`，与当前 SDK `DeepSeekModel` 及现有 H100/SGLang 0.5.9 数据可查口径一致。",
        "- WideEP 模块表使用 `kvcache_quant_mode=fp8`、`fmha_quant_mode=fp8_block`、`attention_backend=fa3`。",
        "- 图中每个 case 使用三根柱：real GPU makespan mean、real GPU kernel-sum mean、AIC total；其中 kernel-sum 与 AIC total 继续保留组成堆叠。",
    ]
    (out_dir / "README.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    bench_root = Path(args.bench_root)
    manifest_path = bench_root / "manifest.csv"
    out_dir = bench_root / "analysis" / "mla_aic_compare"
    out_dir.mkdir(parents=True, exist_ok=True)

    entries = load_manifest(manifest_path)
    real_detail, skipped = read_real_layer_rows(entries)
    if real_detail.empty:
        raise SystemExit("No prefill real detail rows after filtering layer 0.")

    db = PerfDatabase(
        system=args.system,
        backend=args.backend,
        version=args.version,
        systems_root=args.systems_root,
    )

    real_stack = build_real_kernel_stack(real_detail)
    real_stack_case = aggregate_real_stack(real_stack)
    real_cases = (
        real_detail.sort_values(["tag", "layer_id"])
        .groupby("tag", as_index=False)
        .first()[
            [
                "tag",
                "batch_size",
                "fresh_len",
                "prefix_len",
                "total_seq_len",
                "tp_size",
                "attention_backend",
            ]
        ]
    )

    deepseek_stack, deepseek_validation = query_deepseek_fallback_stack(real_cases, db)
    wideep_stack = query_wideep_stack(real_cases, db)
    compare_summary = summarize_cases(real_detail, real_stack, deepseek_stack, wideep_stack)

    real_detail.to_csv(out_dir / "prefill_real_layer_detail.csv", index=False)
    skipped.to_csv(out_dir / "prefill_skipped_rows.csv", index=False)
    real_stack.to_csv(out_dir / "prefill_real_kernel_stack.csv", index=False)
    real_stack_case.to_csv(out_dir / "prefill_real_kernel_stack_case_mean.csv", index=False)
    deepseek_stack.to_csv(out_dir / "prefill_aic_deepseek_fallback_stack.csv", index=False)
    deepseek_validation.to_csv(out_dir / "prefill_aic_deepseek_primary_validation.csv", index=False)
    wideep_stack.to_csv(out_dir / "prefill_aic_wideep_stack.csv", index=False)
    compare_summary.to_csv(out_dir / "prefill_compare_summary.csv", index=False)

    save_boxplot(real_detail, compare_summary, out_dir)
    plot_stacked_compare(
        compare_summary,
        real_stack_case,
        deepseek_stack,
        aic_total_col="aic_deepseek_fallback_total_ms",
        title="DeepSeekModel fallback vs real self_attn",
        subtitle=(
            "Prefill only; layer 0 excluded. Real bars show nsys GPU kernel-time decomposition, "
            "AIC bars sum SDK FallbackOp components. Percent labels: M/K=makespan vs kernel_sum, "
            "A/K=AIC vs kernel_sum, A/M=AIC vs makespan."
        ),
        output_stem="prefill_deepseek_fallback_stacked_compare",
        component_order=DEEPSEEK_FALLBACK_ORDER,
        component_colors=DEEPSEEK_FALLBACK_COLORS,
        out_dir=out_dir,
    )
    plot_stacked_compare(
        compare_summary,
        real_stack_case,
        wideep_stack,
        aic_total_col="aic_wideep_total_ms",
        title="WideEPDeepSeekModel aligned ops vs real self_attn",
        subtitle=(
            "Prefill only; layer 0 excluded. AIC includes one qkv_a GEMM plus WideEPContextMLA "
            "and intentionally avoids the duplicate qkv/downscale GEMM bug. Percent labels: "
            "M/K=makespan vs kernel_sum, A/K=AIC vs kernel_sum, A/M=AIC vs makespan."
        ),
        output_stem="prefill_wideep_stacked_compare",
        component_order=WIDEEP_ORDER,
        component_colors=WIDEEP_COLORS,
        out_dir=out_dir,
        sort_panels=[
            ("Sorted by |A/M| gap descending", "A/M"),
            ("Sorted by |A/K| gap descending", "A/K"),
            ("Sorted by |M/K| gap descending", "M/K"),
        ],
    )
    write_summary_md(out_dir, real_detail, skipped, compare_summary, deepseek_validation)

    files = sorted(p.name for p in out_dir.iterdir() if p.is_file())
    if "artifact_manifest.json" not in files:
        files.append("artifact_manifest.json")
    manifest = {
        "output_dir": str(out_dir),
        "real_layer_rows": int(len(real_detail)),
        "case_count": int(compare_summary["tag"].nunique()),
        "skipped_rows": int(len(skipped)),
        "files": sorted(files),
    }
    (out_dir / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
