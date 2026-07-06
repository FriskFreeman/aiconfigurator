#!/usr/bin/env python3
"""Analyze decode MLA single-op AIC fallback error against real kernels."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import sys
import textwrap
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[7]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from aiconfigurator.sdk import common  # noqa: E402
from aiconfigurator.sdk.perf_database import PerfDatabase  # noqa: E402

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
    "blue_dark": "#2E4780",
    "gold": "#FFE15B",
    "gold_dark": "#736422",
    "orange": "#F0986E",
    "orange_dark": "#804126",
    "olive": "#A3D576",
    "olive_dark": "#4D7F26",
    "pink": "#F390CA",
    "pink_dark": "#7D2C63",
    "neutral": "#C5CAD3",
    "neutral_dark": "#464C55",
}

FALLBACK_OP_ORDER = [
    "generation_downscale_gemm",
    "generation_q_b_proj_gemm",
    "generation_bmm_pre",
    "generation_attention",
    "generation_bmm_post",
    "generation_proj_gemm",
]

FALLBACK_COLORS = {
    "generation_downscale_gemm": COLORS["blue"],
    "generation_q_b_proj_gemm": COLORS["gold"],
    "generation_bmm_pre": "#F6C36E",
    "generation_attention": COLORS["olive"],
    "generation_bmm_post": "#C9A4E8",
    "generation_proj_gemm": COLORS["pink"],
}

OP_LABELS = {
    "generation_downscale_gemm": "downscale GEMM",
    "generation_q_b_proj_gemm": "q_b GEMM",
    "generation_bmm_pre": "q_w_kc BMM",
    "generation_attention": "attention",
    "generation_bmm_post": "s_w_vc BMM",
    "generation_proj_gemm": "o_proj GEMM",
}

REAL_MAPPING = {
    # collect_gemm fp8_block includes per-token-group quant before DeepGEMM.
    "generation_downscale_gemm": ["qkv_downscale_quant", "qkv_downscale_gemm"],
    "generation_q_b_proj_gemm": ["q_b_proj_quant", "q_b_proj_gemm"],
    # collect_mla_bmm fp8 includes per_tensor_quant_mla_fp8 + bmm_fp8. The
    # default FA3 real trace uses torch nvjet BMM without visible fp8 bmm
    # quant kernels in the module body, so compare against the visible BMM.
    "generation_bmm_pre": ["q_w_kc_bmm"],
    # collect_mla generation covers the attention backend proper. We keep two
    # real columns: pure FA3 kernels and FA3 plus cache/rotary support.
    "generation_attention": ["fa3_prepare", "fa3_attention", "fa3_combine"],
    "generation_bmm_post": ["s_w_vc_bmm"],
    "generation_proj_gemm": ["o_proj_quant", "o_proj_gemm"],
}

ATTENTION_SUPPORT_COMPONENTS = ["rotary_emb", "set_mla_kv_buffer"]

HIDDEN_SIZE = 7168
DEEPSEEK_DECODE_OPROJ_INPUT_DIM = 128 * 128


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bench-root",
        type=Path,
        default=Path("bench_data/h100_sxm/sglang/v0.5.9/deepseek_v3_mla"),
    )
    parser.add_argument("--systems-root", type=Path, default=Path("src/aiconfigurator/systems"))
    parser.add_argument("--system", default="h100_sxm")
    parser.add_argument("--backend", default="sglang")
    parser.add_argument("--version", default="0.5.9")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


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
            "font.family": ["DejaVu Sans", "sans-serif"],
            "font.size": 10,
            "axes.grid": True,
        }
    )


def add_chart_header(fig, ax, title: str, subtitle: str) -> None:
    title = textwrap.fill(title, width=92, break_long_words=False)
    subtitle = textwrap.fill(subtitle, width=132, break_long_words=False)
    ax.set_title("")
    fig.subplots_adjust(top=0.84)
    left = ax.get_position().x0
    fig.text(left, 0.965, title, ha="left", va="top", fontsize=16, fontweight="bold", color=TOKENS["ink"])
    fig.text(left, 0.915, subtitle, ha="left", va="top", fontsize=10.5, color=TOKENS["muted"])


def pct_gap(sim: float | None, real: float | None) -> float | None:
    if sim is None or real in (None, 0, 0.0) or pd.isna(sim) or pd.isna(real):
        return None
    return (float(sim) - float(real)) / float(real) * 100.0


def nearest_pair(value: int, points: list[int], *, inner_only: bool = False) -> tuple[int | None, int | None, str]:
    points = sorted(set(int(x) for x in points))
    if not points:
        return None, None, "no_data"
    if value in points:
        return value, value, "exact"
    left = max((x for x in points if x < value), default=None)
    right = min((x for x in points if x > value), default=None)
    if left is None:
        return (points[0], points[1] if len(points) > 1 else points[0], "left_extrapolate")
    if right is None:
        return (points[-2] if len(points) > 1 else points[-1], points[-1], "right_extrapolate")
    return left, right, "interpolate"


def leaf_latency(value: Any) -> float:
    if isinstance(value, dict):
        return float(value["latency"])
    return float(value)


def trace_gemm(db: PerfDatabase, *, m: int, n: int, k: int, quant_mode: common.GEMMQuantMode) -> dict[str, Any]:
    data = db._gemm_data.data[quant_mode]  # noqa: SLF001
    exact = m in data and n in data[m] and k in data[m][n]
    if exact:
        latency = leaf_latency(data[m][n][k])
        return {
            "query_kind": "gemm",
            "database_file": db._gemm_data.filepath,  # noqa: SLF001
            "lookup_method": "exact",
            "axis_x": "m",
            "axis_y": "n",
            "axis_z": "k",
            "x": m,
            "y": n,
            "z": k,
            "x_left": m,
            "x_right": m,
            "neighbor_latencies_json": json.dumps({str(m): latency}),
        }
    m_values = sorted(m_key for m_key in data if n in data[m_key] and k in data[m_key][n])
    if len(m_values) >= 2:
        left, right, mode = nearest_pair(m, m_values)
        neighbor = {str(left): leaf_latency(data[left][n][k]), str(right): leaf_latency(data[right][n][k])}
        method = f"1d_m_{mode}"
        return {
            "query_kind": "gemm",
            "database_file": db._gemm_data.filepath,  # noqa: SLF001
            "lookup_method": method,
            "axis_x": "m",
            "axis_y": "n",
            "axis_z": "k",
            "x": m,
            "y": n,
            "z": k,
            "x_left": left,
            "x_right": right,
            "neighbor_latencies_json": json.dumps(neighbor),
        }
    m_axis = sorted(data.keys())
    left, right, mode = nearest_pair(m, m_axis)
    return {
        "query_kind": "gemm",
        "database_file": db._gemm_data.filepath,  # noqa: SLF001
        "lookup_method": f"3d_cubic_{mode}",
        "axis_x": "m",
        "axis_y": "n",
        "axis_z": "k",
        "x": m,
        "y": n,
        "z": k,
        "x_left": left,
        "x_right": right,
        "neighbor_latencies_json": "{}",
    }


def trace_generation_mla(
    db: PerfDatabase,
    *,
    b: int,
    s: int,
    num_heads: int,
    kvcache_quant_mode: common.KVCacheQuantMode,
) -> dict[str, Any]:
    data = db._generation_mla_data.data[kvcache_quant_mode]  # noqa: SLF001
    exact = num_heads in data and b in data[num_heads] and s in data[num_heads][b]
    heads_axis = sorted(data.keys())
    b_axis = sorted(data.get(num_heads, {}).keys())
    s_axis = sorted(data.get(num_heads, {}).get(b, {}).keys())
    h_l, h_r, h_mode = nearest_pair(num_heads, heads_axis)
    b_l, b_r, b_mode = nearest_pair(b, b_axis)
    s_l, s_r, s_mode = nearest_pair(s, s_axis)
    return {
        "query_kind": "generation_mla",
        "database_file": db._generation_mla_data.filepath,  # noqa: SLF001
        "lookup_method": "exact" if exact else f"3d_bilinear_h:{h_mode}_b:{b_mode}_s:{s_mode}",
        "axis_x": "num_heads",
        "axis_y": "batch_size",
        "axis_z": "s",
        "x": num_heads,
        "y": b,
        "z": s,
        "x_left": h_l,
        "x_right": h_r,
        "y_left": b_l,
        "y_right": b_r,
        "z_left": s_l,
        "z_right": s_r,
        "neighbor_latencies_json": "{}",
    }


def trace_mla_bmm(
    db: PerfDatabase,
    *,
    num_tokens: int,
    num_heads: int,
    quant_mode: common.GEMMQuantMode,
    if_pre: bool,
) -> dict[str, Any]:
    loaded = db._mla_bmm_data  # noqa: SLF001
    data = loaded.data
    quant_mode_lookup = quant_mode if quant_mode in data else common.GEMMQuantMode.bfloat16
    op_name = "mla_gen_pre" if if_pre else "mla_gen_post"
    token_dict = data[quant_mode_lookup][op_name][num_heads]
    left, right, mode = nearest_pair(num_tokens, list(token_dict.keys()))
    neighbor = {str(left): leaf_latency(token_dict[left]), str(right): leaf_latency(token_dict[right])}
    return {
        "query_kind": "mla_bmm",
        "database_file": loaded.filepath,
        "lookup_method": f"1d_tokens_{mode}",
        "axis_x": "num_tokens",
        "axis_y": "num_heads",
        "axis_z": "if_pre",
        "x": num_tokens,
        "y": num_heads,
        "z": int(if_pre),
        "x_left": left,
        "x_right": right,
        "quant_mode_lookup": quant_mode_lookup.name,
        "neighbor_latencies_json": json.dumps(neighbor),
    }


def query_db_value(db: PerfDatabase, op_name: str, *, batch_size: int, kv_len: int, tp_size: int) -> tuple[float, str, dict[str, Any]]:
    s = kv_len + 1
    num_heads = 128 // tp_size
    h = HIDDEN_SIZE
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        if op_name == "generation_downscale_gemm":
            result = db.query_gemm(batch_size, 2112, h, common.GEMMQuantMode.fp8_block)
            trace = trace_gemm(db, m=batch_size, n=2112, k=h, quant_mode=common.GEMMQuantMode.fp8_block)
        elif op_name == "generation_q_b_proj_gemm":
            result = db.query_gemm(batch_size, 24576 // tp_size, 1536, common.GEMMQuantMode.fp8_block)
            trace = trace_gemm(
                db,
                m=batch_size,
                n=24576 // tp_size,
                k=1536,
                quant_mode=common.GEMMQuantMode.fp8_block,
            )
        elif op_name == "generation_bmm_pre":
            result = db.query_mla_bmm(batch_size, num_heads, common.GEMMQuantMode.fp8, True)
            trace = trace_mla_bmm(
                db,
                num_tokens=batch_size,
                num_heads=num_heads,
                quant_mode=common.GEMMQuantMode.fp8,
                if_pre=True,
            )
        elif op_name == "generation_attention":
            result = db.query_generation_mla(batch_size, s, num_heads, common.KVCacheQuantMode.fp8)
            trace = trace_generation_mla(
                db,
                b=batch_size,
                s=s,
                num_heads=num_heads,
                kvcache_quant_mode=common.KVCacheQuantMode.fp8,
            )
        elif op_name == "generation_bmm_post":
            result = db.query_mla_bmm(batch_size, num_heads, common.GEMMQuantMode.fp8, False)
            trace = trace_mla_bmm(
                db,
                num_tokens=batch_size,
                num_heads=num_heads,
                quant_mode=common.GEMMQuantMode.fp8,
                if_pre=False,
            )
        elif op_name == "generation_proj_gemm":
            result = db.query_gemm(batch_size, h, h // tp_size, common.GEMMQuantMode.fp8_block)
            trace = trace_gemm(db, m=batch_size, n=h, k=h // tp_size, quant_mode=common.GEMMQuantMode.fp8_block)
        else:
            raise ValueError(op_name)
    return float(result), str(getattr(result, "source", "silicon")), trace


def query_gemm_value(
    db: PerfDatabase,
    *,
    batch_size: int,
    n: int,
    k: int,
    quant_mode: common.GEMMQuantMode,
) -> tuple[float, str, dict[str, Any]]:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        result = db.query_gemm(batch_size, n, k, quant_mode)
        trace = trace_gemm(db, m=batch_size, n=n, k=k, quant_mode=quant_mode)
    return float(result), str(getattr(result, "source", "silicon")), trace


def query_generation_mla_value(
    db: PerfDatabase,
    *,
    batch_size: int,
    s: int,
    num_heads: int,
    kvcache_quant_mode: common.KVCacheQuantMode,
) -> tuple[float, str, dict[str, Any]]:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        result = db.query_generation_mla(batch_size, s, num_heads, kvcache_quant_mode)
        trace = trace_generation_mla(
            db,
            b=batch_size,
            s=s,
            num_heads=num_heads,
            kvcache_quant_mode=kvcache_quant_mode,
        )
    return float(result), str(getattr(result, "source", "silicon")), trace


def real_value_for_op(summary_row: pd.Series, op_name: str, *, attention_support: bool = False) -> float:
    components = REAL_MAPPING[op_name]
    value = sum(float(summary_row[f"real_semantic_{component}_sum_ms"]) for component in components)
    if op_name == "generation_attention" and attention_support:
        value += sum(float(summary_row[f"real_semantic_{component}_sum_ms"]) for component in ATTENTION_SUPPORT_COMPONENTS)
    return value


def build_op_comparison(summary: pd.DataFrame, db: PerfDatabase) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    for case in summary.itertuples(index=False):
        case_dict = case._asdict()
        for order, op_name in enumerate(FALLBACK_OP_ORDER, start=1):
            aic_latency, source, trace = query_db_value(
                db,
                op_name,
                batch_size=int(case_dict["batch_size"]),
                kv_len=int(case_dict["kv_len"]),
                tp_size=int(case_dict["tp_size"]),
            )
            real_kernel_ms = real_value_for_op(pd.Series(case_dict), op_name)
            real_with_support_ms = (
                real_value_for_op(pd.Series(case_dict), op_name, attention_support=True)
                if op_name == "generation_attention"
                else real_kernel_ms
            )
            row = {
                "tag": case_dict["tag"],
                "batch_size": int(case_dict["batch_size"]),
                "kv_len": int(case_dict["kv_len"]),
                "s": int(case_dict["aic_query_s"]),
                "tp_size": int(case_dict["tp_size"]),
                "num_heads": 128 // int(case_dict["tp_size"]),
                "op_name": op_name,
                "op_label": OP_LABELS[op_name],
                "op_order": order,
                "real_kernel_ms": real_kernel_ms,
                "real_with_support_ms": real_with_support_ms,
                "real_components_json": json.dumps(REAL_MAPPING[op_name], ensure_ascii=False),
                "aic_latency_ms": aic_latency,
                "aic_source": source,
                "delta_ms": aic_latency - real_kernel_ms,
                "gap_pct": pct_gap(aic_latency, real_kernel_ms),
                "delta_vs_support_ms": aic_latency - real_with_support_ms,
                "gap_vs_support_pct": pct_gap(aic_latency, real_with_support_ms),
                "aic_corrected_op_latency_ms": aic_latency,
                "corrected_delta_ms": aic_latency - real_kernel_ms,
                "corrected_gap_pct": pct_gap(aic_latency, real_kernel_ms),
                "aic_attention_s_eq_kv_len_ms": math.nan,
                "attention_s_eq_kv_len_delta_vs_aic_ms": math.nan,
                "attention_s_eq_kv_len_gap_pct": math.nan,
                "aic_combined_diagnostic_latency_ms": aic_latency,
                "diagnostic_note": "",
            }
            if op_name == "generation_proj_gemm":
                corrected_k = DEEPSEEK_DECODE_OPROJ_INPUT_DIM // int(case_dict["tp_size"])
                corrected_latency, corrected_source, corrected_trace = query_gemm_value(
                    db,
                    batch_size=int(case_dict["batch_size"]),
                    n=HIDDEN_SIZE,
                    k=corrected_k,
                    quant_mode=common.GEMMQuantMode.fp8_block,
                )
                row.update(
                    {
                        "aic_corrected_op_latency_ms": corrected_latency,
                        "aic_corrected_source": corrected_source,
                        "corrected_delta_ms": corrected_latency - real_kernel_ms,
                        "corrected_gap_pct": pct_gap(corrected_latency, real_kernel_ms),
                        "aic_combined_diagnostic_latency_ms": corrected_latency,
                        "diagnostic_note": "o_proj_shape_corrected_to_7168x16384_over_tp",
                        "corrected_trace_lookup_method": corrected_trace.get("lookup_method"),
                        "corrected_trace_x": corrected_trace.get("x"),
                        "corrected_trace_y": corrected_trace.get("y"),
                        "corrected_trace_z": corrected_trace.get("z"),
                        "corrected_trace_neighbor_latencies_json": corrected_trace.get("neighbor_latencies_json"),
                    }
                )
            elif op_name == "generation_attention":
                alt_latency, alt_source, alt_trace = query_generation_mla_value(
                    db,
                    batch_size=int(case_dict["batch_size"]),
                    s=int(case_dict["kv_len"]),
                    num_heads=128 // int(case_dict["tp_size"]),
                    kvcache_quant_mode=common.KVCacheQuantMode.fp8,
                )
                row.update(
                    {
                        "aic_attention_s_eq_kv_len_ms": alt_latency,
                        "aic_attention_s_eq_kv_len_source": alt_source,
                        "attention_s_eq_kv_len_delta_vs_aic_ms": alt_latency - aic_latency,
                        "attention_s_eq_kv_len_gap_pct": pct_gap(alt_latency, real_kernel_ms),
                        "aic_combined_diagnostic_latency_ms": alt_latency,
                        "diagnostic_note": "attention_query_s_changed_from_kv_len_plus_1_to_kv_len",
                        "attention_s_eq_kv_len_trace_lookup_method": alt_trace.get("lookup_method"),
                        "attention_s_eq_kv_len_trace_x": alt_trace.get("x"),
                        "attention_s_eq_kv_len_trace_y": alt_trace.get("y"),
                        "attention_s_eq_kv_len_trace_z": alt_trace.get("z"),
                    }
                )
            row.update({f"trace_{k}": v for k, v in trace.items()})
            rows.append(row)
            trace_rows.append(
                {
                    "tag": case_dict["tag"],
                    "op_name": op_name,
                    "aic_latency_ms": aic_latency,
                    "aic_source": source,
                    **trace,
                }
            )
            if op_name == "generation_proj_gemm":
                corrected_trace_row = {
                    "tag": case_dict["tag"],
                    "op_name": f"{op_name}__diagnostic_correct_shape",
                    "aic_latency_ms": row["aic_corrected_op_latency_ms"],
                    "aic_source": row.get("aic_corrected_source", ""),
                    **corrected_trace,
                }
                trace_rows.append(corrected_trace_row)
            elif op_name == "generation_attention":
                alt_trace_row = {
                    "tag": case_dict["tag"],
                    "op_name": f"{op_name}__diagnostic_s_eq_kv_len",
                    "aic_latency_ms": row["aic_attention_s_eq_kv_len_ms"],
                    "aic_source": row.get("aic_attention_s_eq_kv_len_source", ""),
                    **alt_trace,
                }
                trace_rows.append(alt_trace_row)
    return pd.DataFrame(rows), pd.DataFrame(trace_rows)


def summarize_errors(op_compare: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    totals = (
        op_compare.groupby("tag", as_index=False)
        .agg(
            aic_single_op_total_ms=("aic_latency_ms", "sum"),
            aic_corrected_o_proj_total_ms=("aic_corrected_op_latency_ms", "sum"),
            aic_combined_diagnostic_total_ms=("aic_combined_diagnostic_latency_ms", "sum"),
            real_single_op_mapped_total_ms=("real_kernel_ms", "sum"),
            total_delta_ms=("delta_ms", "sum"),
            corrected_total_delta_ms=("corrected_delta_ms", "sum"),
        )
        .merge(
            summary[
                [
                    "tag",
                    "batch_size",
                    "kv_len",
                    "real_mla_makespan_mean_ms",
                    "real_mla_kernel_sum_mean_ms",
                    "aic_deepseek_fallback_total_ms",
                    "aic_wideep_dedup_total_ms",
                ]
            ],
            on="tag",
            how="left",
        )
    )
    abs_error = (
        op_compare.groupby("tag", as_index=False)
        .agg(
            op_abs_delta_sum_ms=("delta_ms", lambda s: s.abs().sum()),
            op_corrected_abs_delta_sum_ms=("corrected_delta_ms", lambda s: s.abs().sum()),
        )
    )
    totals = totals.merge(abs_error, on="tag", how="left")
    totals["aic_vs_real_kernel_sum_gap_pct"] = totals.apply(
        lambda r: pct_gap(r["aic_single_op_total_ms"], r["real_mla_kernel_sum_mean_ms"]),
        axis=1,
    )
    totals["aic_vs_real_makespan_gap_pct"] = totals.apply(
        lambda r: pct_gap(r["aic_single_op_total_ms"], r["real_mla_makespan_mean_ms"]),
        axis=1,
    )
    totals["aic_corrected_o_proj_gap_pct_vs_real_kernel_sum"] = totals.apply(
        lambda r: pct_gap(r["aic_corrected_o_proj_total_ms"], r["real_mla_kernel_sum_mean_ms"]),
        axis=1,
    )
    totals["aic_corrected_o_proj_gap_pct_vs_real_makespan"] = totals.apply(
        lambda r: pct_gap(r["aic_corrected_o_proj_total_ms"], r["real_mla_makespan_mean_ms"]),
        axis=1,
    )
    totals["aic_combined_diagnostic_gap_pct_vs_real_kernel_sum"] = totals.apply(
        lambda r: pct_gap(r["aic_combined_diagnostic_total_ms"], r["real_mla_kernel_sum_mean_ms"]),
        axis=1,
    )
    totals["aic_combined_diagnostic_gap_pct_vs_real_makespan"] = totals.apply(
        lambda r: pct_gap(r["aic_combined_diagnostic_total_ms"], r["real_mla_makespan_mean_ms"]),
        axis=1,
    )
    contribution = op_compare.copy()
    contribution["abs_delta_ms"] = contribution["delta_ms"].abs()
    top = contribution.sort_values(["tag", "abs_delta_ms"], ascending=[True, False]).groupby("tag").head(1)
    top = top[["tag", "op_name", "op_label", "delta_ms", "gap_pct"]].rename(
        columns={
            "op_name": "largest_abs_delta_op",
            "op_label": "largest_abs_delta_op_label",
            "delta_ms": "largest_abs_delta_ms",
            "gap_pct": "largest_abs_delta_gap_pct",
        }
    )
    totals = totals.merge(top, on="tag", how="left")
    return totals.sort_values(["kv_len", "batch_size", "tag"]).reset_index(drop=True)


def plot_op_delta(op_compare: pd.DataFrame, out_dir: Path) -> None:
    apply_style()
    plot_df = op_compare.copy()
    tags = (
        plot_df[["tag", "batch_size", "kv_len"]]
        .drop_duplicates()
        .sort_values(["kv_len", "batch_size", "tag"])
    )
    ordered_tags = tags["tag"].tolist()
    labels = {r.tag: f"b{int(r.batch_size)} p{int(r.kv_len)}" for r in tags.itertuples(index=False)}
    x = np.arange(len(ordered_tags))
    fig, ax = plt.subplots(figsize=(max(13.0, len(ordered_tags) * 0.82), 7.0))
    pos_bottom = np.zeros(len(ordered_tags))
    neg_bottom = np.zeros(len(ordered_tags))
    for op_name in FALLBACK_OP_ORDER:
        values = []
        for tag in ordered_tags:
            match = plot_df[(plot_df["tag"] == tag) & (plot_df["op_name"] == op_name)]
            values.append(float(match["delta_ms"].iloc[0]) if not match.empty else 0.0)
        values_arr = np.array(values)
        bottoms = np.where(values_arr >= 0, pos_bottom, neg_bottom)
        ax.bar(
            x,
            values_arr,
            bottom=bottoms,
            color=FALLBACK_COLORS[op_name],
            edgecolor=COLORS["neutral_dark"],
            linewidth=0.35,
            label=OP_LABELS[op_name],
        )
        pos_bottom += np.where(values_arr >= 0, values_arr, 0.0)
        neg_bottom += np.where(values_arr < 0, values_arr, 0.0)
    ax.axhline(0, color=COLORS["neutral_dark"], linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([labels[tag] for tag in ordered_tags], rotation=45, ha="right")
    ax.set_ylabel("AIC - real mapped kernel time (ms)")
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.04), ncol=3, frameon=False, fontsize=8)
    add_chart_header(
        fig,
        ax,
        "Decode DeepSeek fallback single-op error contribution",
        "Each stacked bar decomposes AIC single-op total error by fallback op. Positive means AIC is slower than the mapped real kernels.",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    for ext in ["png", "svg"]:
        fig.savefig(out_dir / f"decode_single_op_error_contribution.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_real_vs_aic_per_op(op_compare: pd.DataFrame, out_dir: Path) -> None:
    apply_style()
    fig, axes = plt.subplots(2, 3, figsize=(16.0, 8.6), sharex=False)
    axes = axes.ravel()
    tags = (
        op_compare[["tag", "batch_size", "kv_len"]]
        .drop_duplicates()
        .sort_values(["kv_len", "batch_size", "tag"])
    )
    ordered_tags = tags["tag"].tolist()
    labels = {r.tag: f"b{int(r.batch_size)}\np{int(r.kv_len)}" for r in tags.itertuples(index=False)}
    x = np.arange(len(ordered_tags))
    width = 0.35
    for ax, op_name in zip(axes, FALLBACK_OP_ORDER):
        values_real = []
        values_aic = []
        for tag in ordered_tags:
            match = op_compare[(op_compare["tag"] == tag) & (op_compare["op_name"] == op_name)]
            values_real.append(float(match["real_kernel_ms"].iloc[0]))
            values_aic.append(float(match["aic_latency_ms"].iloc[0]))
        ax.bar(x - width / 2, values_real, width, color="#E2E5EA", edgecolor=COLORS["neutral_dark"], linewidth=0.4, label="real")
        ax.bar(x + width / 2, values_aic, width, color=FALLBACK_COLORS[op_name], edgecolor=COLORS["neutral_dark"], linewidth=0.4, label="AIC")
        ax.set_title(OP_LABELS[op_name], fontsize=10, color=TOKENS["ink"])
        ax.set_xticks(x)
        ax.set_xticklabels([labels[tag] for tag in ordered_tags], fontsize=7)
        ax.grid(axis="y", linestyle="--", alpha=0.55)
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="upper left", bbox_to_anchor=(0.065, 0.90), ncol=2, frameon=False)
    fig.text(0.06, 0.975, "Real mapped kernels vs AIC single-op lookup", fontsize=16, fontweight="bold", color=TOKENS["ink"], va="top")
    fig.text(0.06, 0.945, "Real GEMM mappings include visible quant kernels to match collect_gemm fp8_block timing.", fontsize=10.5, color=TOKENS["muted"], va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    for ext in ["png", "svg"]:
        fig.savefig(out_dir / f"decode_real_vs_aic_per_single_op.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_markdown(out_dir: Path, op_compare: pd.DataFrame, case_summary: pd.DataFrame, query_trace: pd.DataFrame) -> None:
    op_mape = (
        op_compare.assign(abs_gap=lambda d: d["gap_pct"].abs())
        .groupby(["op_order", "op_name", "op_label"], as_index=False)
        .agg(
            mean_real_ms=("real_kernel_ms", "mean"),
            mean_aic_ms=("aic_latency_ms", "mean"),
            mean_delta_ms=("delta_ms", "mean"),
            mean_corrected_aic_ms=("aic_corrected_op_latency_ms", "mean"),
            mean_corrected_delta_ms=("corrected_delta_ms", "mean"),
            mape_pct=("abs_gap", "mean"),
            corrected_mape_pct=("corrected_gap_pct", lambda s: s.abs().mean()),
        )
        .sort_values("op_order")
    )
    total_mape_kernel = case_summary["aic_vs_real_kernel_sum_gap_pct"].abs().mean()
    total_mape_makespan = case_summary["aic_vs_real_makespan_gap_pct"].abs().mean()
    corrected_o_proj_mape_kernel = case_summary["aic_corrected_o_proj_gap_pct_vs_real_kernel_sum"].abs().mean()
    combined_diag_mape_kernel = case_summary["aic_combined_diagnostic_gap_pct_vs_real_kernel_sum"].abs().mean()
    original_l1_ms = case_summary["op_abs_delta_sum_ms"].mean()
    corrected_l1_ms = case_summary["op_corrected_abs_delta_sum_ms"].mean()
    largest_counts = case_summary["largest_abs_delta_op_label"].value_counts()
    oprow = op_mape[op_mape["op_name"].eq("generation_proj_gemm")].iloc[0]
    attn_rows = op_compare[op_compare["op_name"].eq("generation_attention")].copy()
    attn_s_delta_mean = attn_rows["attention_s_eq_kv_len_delta_vs_aic_ms"].mean()
    attn_s_gap_mape = attn_rows["attention_s_eq_kv_len_gap_pct"].abs().mean()
    attention_exact_count = int(query_trace[query_trace["op_name"].eq("generation_attention")]["lookup_method"].eq("exact").sum())
    attention_diag_exact_count = int(
        query_trace[query_trace["op_name"].eq("generation_attention__diagnostic_s_eq_kv_len")]["lookup_method"]
        .eq("exact")
        .sum()
    )
    lines = [
        "# Decode Single-Op Kernel AIC Error Analysis",
        "",
        "## 结论摘要",
        "",
        f"- DeepSeek fallback 单算子累加相对实机 MLA kernel_sum 的 MAPE 为 `{total_mape_kernel:.2f}%`，相对 makespan 的 MAPE 为 `{total_mape_makespan:.2f}%`。",
        f"- 仅把 `generation_proj_gemm` 从 SDK 当前 `7168 x 7168/tp` 诊断性改为实机 `7168 x 16384/tp` 后，case 级带符号总和 MAPE 从 `{total_mape_kernel:.2f}%` 变为 `{corrected_o_proj_mape_kernel:.2f}%`；它变大是因为原先 o_proj 低估抵消了 attention 高估。",
        f"- 按每个 op 的绝对误差加总看，修正 o_proj 后平均 L1 error 从 `{original_l1_ms:.6f}ms` 降到 `{corrected_l1_ms:.6f}ms`，说明该 shape 问题确实是单算子误差的重要来源。",
        f"- 若同时把 attention 查询从 SDK 当前 `s=kv_len+1` 诊断性改成 `s=kv_len`，case 级 kernel_sum MAPE 为 `{combined_diag_mape_kernel:.2f}%`；这不是建议直接改 SDK，只用于分离插值/坐标误差。",
        "- 误差主要不是 CUDA Graph module 边界问题；WideEP module 已与 makespan 高度吻合，单算子误差集中在 fallback op 的采集/查表口径。",
        "- GEMM 对照已按 collect_gemm 的 `fp8_block` 口径包含前置 `per_token_group_quant_8bit_kernel + deepgemm`，否则会低估实机 GEMM 对齐口径。",
        "- BMM 对照按可见 `nvjet_tst_*` kernel 进行；collector 的 fp8 BMM 口径包含 `per_tensor_quant_mla_fp8 + bmm_fp8`，而当前默认 FA3 实机 trace 中 MLA body 内可见的是 torch/nvjet BMM。",
        "- `generation_attention` 的 AIC 查询来自 `query_generation_mla(b, s, num_heads, fp8)`，对实机侧主要对应 FA3 attention kernels；rotary 与 KV write 是 SGLang wrapper 支持 kernel，未计入主 attention 单算子口径。",
        "- 根因排序：`generation_proj_gemm` 是明确的 SDK 普通 DeepSeekModel shape 口径错误；`generation_attention` 还叠加了 `s=kv_len+1` 插值坐标与 collect_mla/RadixAttention 计时口径差异。",
        "",
        "## 最大误差来源",
        "",
    ]
    for label, count in largest_counts.items():
        lines.append(f"- `{label}` 是 `{int(count)}` 个 case 的最大绝对误差来源。")
    lines.extend(["", "## Op 级均值", ""])
    lines.append("| op | real mean ms | AIC mean ms | AIC-real mean ms | MAPE | diagnostic AIC mean ms | diagnostic MAPE |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in op_mape.itertuples(index=False):
        lines.append(
            f"| {row.op_label} | {row.mean_real_ms:.6f} | {row.mean_aic_ms:.6f} | "
            f"{row.mean_delta_ms:+.6f} | {row.mape_pct:.2f}% | "
            f"{row.mean_corrected_aic_ms:.6f} | {row.corrected_mape_pct:.2f}% |"
        )
    lines.extend(
        [
            "",
            "## 关键诊断",
            "",
            f"- `generation_proj_gemm` 当前 SDK 查询 `n=7168, k=7168/tp`，而实机 CUDA Graph 边界中的最终 o_proj DeepGEMM 是 `deepgemm(7168,16384)`；诊断性改成 `k=16384/tp` 后，该 op 平均 AIC 从 `{oprow.mean_aic_ms:.6f}ms` 变为 `{oprow.mean_corrected_aic_ms:.6f}ms`，MAPE 从 `{oprow.mape_pct:.2f}%` 降到 `{oprow.corrected_mape_pct:.2f}%`。",
            f"- `generation_attention` 当前 `s=kv_len+1` 查询在 `{attention_exact_count}` 个原始 case 精确命中；诊断性 `s=kv_len` 在 `{attention_diag_exact_count}` 个 case 精确命中，平均改变 `{attn_s_delta_mean:+.6f}ms`，attention op MAPE 变为 `{attn_s_gap_mape:.2f}%`。",
            "- `attention_s_eq_kv_len` 诊断不能单独证明 SDK 应改成 `kv_len`：decode 语义里当前 token 也参与 query，`s=kv_len+1` 有语义合理性；它主要说明 generation_mla_perf 的采集网格与 decode case 坐标错开时会放大插值误差。",
            "- 默认 FA3 实机 trace 中两段 `nvjet_tst_*` 已按 SGLang `forward_mla.py` 的 `torch.bmm` 语义归入 BMM；这修正了旧数据中将部分 `nvjet_tst_*` 错标成 flashmla attention 的问题。",
            "",
            "## Collector 口径证据",
            "",
            "- `collector/sglang/collect_gemm.py` 的 `fp8_block` kernel_func 先调用 `sglang_per_token_group_quant_fp8()`，再调用 `fp8_gemm_deepgemm()`；因此实机 GEMM 对照必须包含 `quant + deepgemm`。",
            "- `collector/sglang/collect_mla.py` 用 `RadixAttention` 构造 `mla_generation`，decode 采集记录 `isl=1, step=input_len`，PerfDatabase 加载后按 `s=isl+step` 查询。",
            "- `collector/sglang/collect_mla_bmm.py` 的 fp8 pre/post BMM 会把 `per_tensor_quant_mla_fp8()` 与 `bmm_fp8()` 放进同一次 `benchmark_with_power()`；默认 FA3 实机路径当前可见的是 torch/nvjet BMM，没有显式的 fp8 BMM quant kernel。",
            "",
            "## AIC 查询链路",
            "",
            "- `generation_downscale_gemm`: `ops.GEMM(..., n=2112, k=7168, fp8_block)` -> `PerfDatabase.query_gemm(m=batch, n=2112, k=7168, fp8_block)` -> `gemm_perf.txt`。",
            "- `generation_q_b_proj_gemm`: `ops.GEMM(..., n=24576/tp, k=1536, fp8_block)` -> `query_gemm()` -> `gemm_perf.txt`。",
            "- `generation_bmm_pre`: `ops.MLABmm(if_pre=True, fp8)` -> `query_mla_bmm(num_tokens=batch, num_heads=128/tp, fp8, pre)` -> `mla_bmm_perf.txt`。",
            "- `generation_attention`: `ops.GenerationMLA(kvcache=fp8)` -> `query_generation_mla(b=batch, s=kv_len+1, num_heads=128/tp, fp8)` -> `generation_mla_perf.txt`。",
            "- `generation_bmm_post`: `ops.MLABmm(if_pre=False, fp8)` -> `query_mla_bmm(..., post)` -> `mla_bmm_perf.txt`。",
            "- `generation_proj_gemm`: `ops.GEMM(..., n=7168, k=7168/tp, fp8_block)` -> `query_gemm()` -> `gemm_perf.txt`。",
            "",
            "## 插值与外推",
            "",
            "- `query_gemm()` 优先精确命中 `(m,n,k)`；若同一 `n,k` 下有多个 `m` 点，则沿 `m` 做 1D 插值/外推；否则进入 3D cubic 插值。",
            "- `query_mla_bmm()` 在固定 `num_heads/op/type` 后按 `num_tokens` 做 1D 插值/外推。",
            "- `query_generation_mla()` 在 `(num_heads,b,s)` 上使用 3D bilinear 插值；本分析的 `s=kv_len+1`，因此很多点虽然接近采集网格，但不一定精确命中。",
            "",
            "## 输出文件",
            "",
            "- `decode_single_op_kernel_vs_aic.csv`: 每个 case/op 的实机 kernel 映射、AIC 查询值、误差和查询追踪。",
            "- `decode_single_op_query_trace.csv`: AIC 查表函数、输入坐标、数据文件、命中/插值状态；包含 `__diagnostic_correct_shape` 与 `__diagnostic_s_eq_kv_len` 追踪行。",
            "- `decode_single_op_case_error_summary.csv`: case 级总误差、修正 o_proj 诊断总误差、组合诊断总误差与最大误差来源。",
            "- `decode_single_op_error_contribution.png/svg`: 总误差按 op 分解图。",
            "- `decode_real_vs_aic_per_single_op.png/svg`: 每个 op 的实机 vs AIC 对比小图。",
        ]
    )
    out_dir.joinpath("README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    source_dir = args.bench_root / "analysis" / "decode_mla_kernel_envelope_aic_compare"
    out_dir = args.output_dir or args.bench_root / "analysis" / "decode_single_op_kernel_aic_error_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(source_dir / "decode_compare_summary.csv")
    db = PerfDatabase(
        system=args.system,
        backend=args.backend,
        version=args.version,
        systems_root=str(args.systems_root),
    )
    op_compare, query_trace = build_op_comparison(summary, db)
    case_summary = summarize_errors(op_compare, summary)

    op_compare.to_csv(out_dir / "decode_single_op_kernel_vs_aic.csv", index=False)
    query_trace.to_csv(out_dir / "decode_single_op_query_trace.csv", index=False)
    case_summary.to_csv(out_dir / "decode_single_op_case_error_summary.csv", index=False)
    plot_op_delta(op_compare, out_dir)
    plot_real_vs_aic_per_op(op_compare, out_dir)
    write_markdown(out_dir, op_compare, case_summary, query_trace)
    manifest = {
        "source_dir": str(source_dir),
        "case_count": int(len(case_summary)),
        "op_row_count": int(len(op_compare)),
        "files": sorted(path.name for path in out_dir.iterdir() if path.is_file()),
    }
    out_dir.joinpath("artifact_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote single-op error analysis to {out_dir}")
    print(case_summary[["tag", "aic_vs_real_kernel_sum_gap_pct", "largest_abs_delta_op_label", "largest_abs_delta_ms"]].to_string(index=False))


if __name__ == "__main__":
    main()
