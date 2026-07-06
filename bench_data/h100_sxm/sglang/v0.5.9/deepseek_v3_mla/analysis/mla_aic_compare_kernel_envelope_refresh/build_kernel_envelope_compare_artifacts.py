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
import matplotlib.colors as mcolors
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
    "context_mla_concat_k",
    "context_attention",
    "context_proj_gemm",
]

DEEPSEEK_FALLBACK_COLORS = {
    "context_downscale_gemm": COLORS["blue"],
    "context_q_b_proj_gemm": COLORS["gold"],
    "context_kv_b_proj_gemm": COLORS["orange"],
    "context_mla_concat_k": "#FFBDA1",
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
    parser.add_argument("--tag-prefix", default=None)
    parser.add_argument("--output-dir", default=None)
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
            # DeepSeek MLA prefill projects latent KV for both fresh and cached
            # prefix tokens; the SDK op expands GEMM m from fresh to fresh+prefix.
            ("context_kv_b_proj_gemm", ops.ContextKVBProjGEMM("context_kv_b_proj_gemm", 1, 32768 // tp_size, 512, gemm_quant_mode)),
            ("context_mla_concat_k", ops.MLAConcatK("context_mla_concat_k", 1, num_heads)),
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


def summarize_cases(real_detail: pd.DataFrame, deepseek_stack: pd.DataFrame, wideep_stack: pd.DataFrame) -> pd.DataFrame:
    real_summary = (
        real_detail.groupby("tag", as_index=False)
        .agg(
            layer_sample_count=("layer_id", "count"),
            batch_size=("batch_size", "first"),
            fresh_len=("fresh_len", "first"),
            prefix_len=("prefix_len", "first"),
            total_seq_len=("total_seq_len", "first"),
            tp_size=("tp_size", "first"),
            real_kernel_envelope_mean_ms=("module_gpu_makespan_ms", "mean"),
            real_kernel_envelope_median_ms=("module_gpu_makespan_ms", "median"),
            real_module_total_to_last_kernel_mean_ms=("module_total_to_last_kernel_ms", "mean"),
        )
    )
    summary = real_summary.copy()

    for name, stack_df in [("deepseek_fallback", deepseek_stack), ("wideep", wideep_stack)]:
        total = stack_df.groupby("tag", as_index=False).agg(**{f"aic_{name}_total_ms": ("latency_ms", "sum")})
        summary = summary.merge(total, on="tag", how="left")
        summary[f"aic_{name}_gap_pct_vs_real_kernel_envelope"] = summary.apply(
            lambda r: pct_gap(r.get(f"aic_{name}_total_ms"), r.get("real_kernel_envelope_mean_ms")), axis=1
        )

    summary["best_aic_model"] = summary.apply(
        lambda r: (
            "DeepSeek fallback"
            if abs(float(r["aic_deepseek_fallback_gap_pct_vs_real_kernel_envelope"]))
            <= abs(float(r["aic_wideep_gap_pct_vs_real_kernel_envelope"]))
            else "WideEP module"
        ),
        axis=1,
    )
    summary["best_aic_total_ms"] = summary.apply(
        lambda r: (
            float(r["aic_deepseek_fallback_total_ms"])
            if r["best_aic_model"] == "DeepSeek fallback"
            else float(r["aic_wideep_total_ms"])
        ),
        axis=1,
    )
    summary["best_aic_gap_pct_vs_real_kernel_envelope"] = summary.apply(
        lambda r: pct_gap(r.get("best_aic_total_ms"), r.get("real_kernel_envelope_mean_ms")), axis=1
    )
    summary["rule_aic_model"] = summary.apply(
        lambda r: "WideEP module" if int(r["prefix_len"]) == 0 else "DeepSeek fallback",
        axis=1,
    )
    summary["rule_aic_total_ms"] = summary.apply(
        lambda r: (
            float(r["aic_wideep_total_ms"])
            if r["rule_aic_model"] == "WideEP module"
            else float(r["aic_deepseek_fallback_total_ms"])
        ),
        axis=1,
    )
    summary["rule_aic_gap_pct_vs_real_kernel_envelope"] = summary.apply(
        lambda r: pct_gap(r.get("rule_aic_total_ms"), r.get("real_kernel_envelope_mean_ms")), axis=1
    )
    summary["rule_is_best"] = summary.apply(
        lambda r: str(r["rule_aic_model"]) == str(r["best_aic_model"]),
        axis=1,
    )
    summary = summary.sort_values(["total_seq_len", "batch_size", "fresh_len", "prefix_len", "tag"]).reset_index(drop=True)
    return summary


def make_case_labels(summary: pd.DataFrame) -> dict[str, str]:
    return {
        row.tag: f"b{int(row.batch_size)}\nf{int(row.fresh_len)}\np{int(row.prefix_len)}"
        for row in summary.itertuples(index=False)
    }


def alpha_color(color: str, alpha: float) -> tuple[float, float, float, float]:
    return mcolors.to_rgba(color, alpha)


def plot_kernel_envelope_compare(
    summary: pd.DataFrame,
    deepseek_stack: pd.DataFrame,
    wideep_stack: pd.DataFrame,
    out_dir: Path,
) -> None:
    apply_style()
    tags = list(summary["tag"])
    labels = make_case_labels(summary)
    x = np.arange(len(tags))
    width = 0.24
    fig, ax = plt.subplots(figsize=(max(14.5, len(tags) * 0.78), 8.1))

    envelope = summary["real_kernel_envelope_mean_ms"].to_numpy(dtype=float)
    ax.bar(
        x - width,
        envelope,
        width,
        color="#E2E5EA",
        edgecolor=COLORS["neutral_dark"],
        linewidth=0.45,
        label="real GPU kernel envelope mean",
    )

    def plot_stack(
        stack_df: pd.DataFrame,
        component_order: list[str],
        component_colors: dict[str, str],
        xpos: np.ndarray,
        label_prefix: str,
        hatch: str = "",
    ) -> np.ndarray:
        bottom = np.zeros(len(tags))
        for op_name in component_order:
            values = []
            for tag in tags:
                match = stack_df[(stack_df["tag"] == tag) & (stack_df["op_name"] == op_name)]
                values.append(float(match["latency_ms"].iloc[0]) if not match.empty else 0.0)
            ax.bar(
                xpos,
                values,
                width,
                bottom=bottom,
                color=component_colors.get(op_name, COLORS["neutral"]),
                edgecolor=COLORS["neutral_dark"],
                linewidth=0.35,
                hatch=hatch,
                label=f"{label_prefix} {op_name}",
            )
            bottom += np.array(values)
        return bottom

    deepseek_total = plot_stack(
        deepseek_stack,
        DEEPSEEK_FALLBACK_ORDER,
        DEEPSEEK_FALLBACK_COLORS,
        x,
        "DeepSeek",
    )
    wideep_total = plot_stack(
        wideep_stack,
        WIDEEP_ORDER,
        WIDEEP_COLORS,
        x + width,
        "WideEP",
        hatch="//",
    )
    y_max = max(np.nanmax(np.r_[envelope, deepseek_total, wideep_total]), 0.001)

    for i, tag in enumerate(tags):
        real_value = float(summary.loc[summary["tag"] == tag, "real_kernel_envelope_mean_ms"].iloc[0])
        ds_value = float(summary.loc[summary["tag"] == tag, "aic_deepseek_fallback_total_ms"].iloc[0])
        we_value = float(summary.loc[summary["tag"] == tag, "aic_wideep_total_ms"].iloc[0])
        ds_gap = pct_gap(ds_value, real_value)
        we_gap = pct_gap(we_value, real_value)
        best_model = "DS" if abs(float(ds_gap)) <= abs(float(we_gap)) else "WE"
        best_value = ds_value if best_model == "DS" else we_value
        best_gap = pct_gap(best_value, real_value)
        group_top = max(real_value, ds_value, we_value)
        for xpos, value in [
            (x[i] - width, real_value),
            (x[i], ds_value),
            (x[i] + width, we_value),
        ]:
            ax.text(
                xpos,
                value + y_max * 0.018,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=6.8,
                color=TOKENS["ink"],
                rotation=90,
            )
        label_specs = [
            (f"DS {ds_gap:+.0f}%", COLORS["orange_mid"]),
            (f"WE {we_gap:+.0f}%", COLORS["blue_mid"]),
            (f"Best {best_gap:+.0f}%", COLORS["olive_mid"]),
        ]
        for offset, (label, color) in enumerate(label_specs):
            ax.text(
                x[i],
                group_top + y_max * (0.105 + offset * 0.052),
                label,
                ha="center",
                va="bottom",
                fontsize=8.3,
                fontweight="bold",
                color=color,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([labels[tag] for tag in tags], rotation=0, ha="center")
    ax.set_xlabel("Prefill case")
    ax.set_ylabel("Latency per layer (ms)")
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    ax.set_ylim(0, y_max * 1.55)
    handles, labels_ = ax.get_legend_handles_labels()
    keep = []
    for h, l in zip(handles, labels_):
        if l == "real GPU kernel envelope mean" or l.startswith("DeepSeek") or l.startswith("WideEP"):
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
    add_chart_header(
        fig,
        ax,
        "Prefill AIC refreshed comparison vs real GPU kernel envelope",
        (
            "Real baseline is nsys self_attn kernel envelope mean only. AIC bars show DeepSeekModel "
            "FallbackOp kernel-time stack and WideEP module-level stack. Labels show AIC gap vs envelope "
            "and best-of-two."
        ),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    for ext in ["png", "svg"]:
        fig.savefig(out_dir / f"prefill_kernel_envelope_aic_stacked_compare.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_kernel_envelope_rule_compare(
    summary: pd.DataFrame,
    deepseek_stack: pd.DataFrame,
    wideep_stack: pd.DataFrame,
    out_dir: Path,
) -> None:
    apply_style()
    tags = list(summary["tag"])
    labels = make_case_labels(summary)
    x = np.arange(len(tags))
    width = 0.24
    fig, ax = plt.subplots(figsize=(max(14.5, len(tags) * 0.78), 8.4))

    envelope = summary["real_kernel_envelope_mean_ms"].to_numpy(dtype=float)
    ax.bar(
        x - width,
        envelope,
        width,
        color="#E2E5EA",
        edgecolor=COLORS["neutral_dark"],
        linewidth=0.45,
        label="real GPU kernel envelope mean",
    )

    ds_selected = {tag: model == "DeepSeek fallback" for tag, model in zip(summary["tag"], summary["rule_aic_model"])}
    we_selected = {tag: model == "WideEP module" for tag, model in zip(summary["tag"], summary["rule_aic_model"])}

    def plot_stack_with_rule_alpha(
        stack_df: pd.DataFrame,
        component_order: list[str],
        component_colors: dict[str, str],
        xpos: np.ndarray,
        label_prefix: str,
        selected_by_tag: dict[str, bool],
        hatch: str = "",
    ) -> np.ndarray:
        bottom = np.zeros(len(tags))
        for op_name in component_order:
            values = []
            facecolors = []
            for tag in tags:
                match = stack_df[(stack_df["tag"] == tag) & (stack_df["op_name"] == op_name)]
                values.append(float(match["latency_ms"].iloc[0]) if not match.empty else 0.0)
                alpha = 0.98 if selected_by_tag[tag] else 0.35
                facecolors.append(alpha_color(component_colors.get(op_name, COLORS["neutral"]), alpha))
            bars = ax.bar(
                xpos,
                values,
                width,
                bottom=bottom,
                color=facecolors,
                edgecolor=COLORS["neutral_dark"],
                linewidth=0.35,
                hatch=hatch,
                label=f"{label_prefix} {op_name}",
            )
            for idx, bar in enumerate(bars):
                if not selected_by_tag[tags[idx]]:
                    bar.set_edgecolor(alpha_color(COLORS["neutral_dark"], 0.45))
            bottom += np.array(values)
        return bottom

    deepseek_total = plot_stack_with_rule_alpha(
        deepseek_stack,
        DEEPSEEK_FALLBACK_ORDER,
        DEEPSEEK_FALLBACK_COLORS,
        x,
        "DeepSeek",
        ds_selected,
    )
    wideep_total = plot_stack_with_rule_alpha(
        wideep_stack,
        WIDEEP_ORDER,
        WIDEEP_COLORS,
        x + width,
        "WideEP",
        we_selected,
        hatch="//",
    )
    y_max = max(np.nanmax(np.r_[envelope, deepseek_total, wideep_total]), 0.001)

    for i, tag in enumerate(tags):
        real_value = float(summary.loc[summary["tag"] == tag, "real_kernel_envelope_mean_ms"].iloc[0])
        ds_value = float(summary.loc[summary["tag"] == tag, "aic_deepseek_fallback_total_ms"].iloc[0])
        we_value = float(summary.loc[summary["tag"] == tag, "aic_wideep_total_ms"].iloc[0])
        ds_gap = pct_gap(ds_value, real_value)
        we_gap = pct_gap(we_value, real_value)
        rule_gap = float(summary.loc[summary["tag"] == tag, "rule_aic_gap_pct_vs_real_kernel_envelope"].iloc[0])
        rule_is_best = bool(summary.loc[summary["tag"] == tag, "rule_is_best"].iloc[0])
        best_gap = float(summary.loc[summary["tag"] == tag, "best_aic_gap_pct_vs_real_kernel_envelope"].iloc[0])
        group_top = max(real_value, ds_value, we_value)
        for xpos, value in [
            (x[i] - width, real_value),
            (x[i], ds_value),
            (x[i] + width, we_value),
        ]:
            ax.text(
                xpos,
                value + y_max * 0.018,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=6.8,
                color=TOKENS["ink"],
                rotation=90,
            )
        label_specs = [
            (f"DS {ds_gap:+.0f}%", COLORS["orange_mid"]),
            (f"WE {we_gap:+.0f}%", COLORS["blue_mid"]),
            (f"Rule {rule_gap:+.0f}%", COLORS["olive_mid"]),
        ]
        if not rule_is_best:
            label_specs.append((f"Best {best_gap:+.0f}%", "#C93C37"))
        for offset, (label, color) in enumerate(label_specs):
            ax.text(
                x[i],
                group_top + y_max * (0.105 + offset * 0.052),
                label,
                ha="center",
                va="bottom",
                fontsize=8.3,
                fontweight="bold",
                color=color,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([labels[tag] for tag in tags], rotation=0, ha="center")
    ax.set_xlabel("Prefill case")
    ax.set_ylabel("Latency per layer (ms)")
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    ax.set_ylim(0, y_max * 1.60)
    handles, labels_ = ax.get_legend_handles_labels()
    keep = []
    for h, l in zip(handles, labels_):
        if l == "real GPU kernel envelope mean" or l.startswith("DeepSeek") or l.startswith("WideEP"):
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
    add_chart_header(
        fig,
        ax,
        "Prefill AIC rule-based comparison vs real GPU kernel envelope",
        (
            "Rule uses WideEP module-level stack when prefix=0 and DeepSeekModel fallback kernel stack when prefix>0. "
            "Non-selected AIC bars are faded; red Best labels mark cases where this rule is not the lowest-error choice."
        ),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    for ext in ["png", "svg"]:
        fig.savefig(out_dir / f"prefill_kernel_envelope_aic_rule_compare.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_envelope_boxplot(real_detail: pd.DataFrame, summary: pd.DataFrame, out_dir: Path) -> None:
    apply_style()
    labels = make_case_labels(summary)
    plot_df = real_detail.copy()
    plot_df["case_label"] = plot_df["tag"].map(labels)
    order = [labels[tag] for tag in summary["tag"]]
    fig, ax = plt.subplots(figsize=(max(12, len(order) * 0.58), 6.4))
    grouped = [
        plot_df.loc[plot_df["case_label"] == label, "module_gpu_makespan_ms"].astype(float).to_numpy()
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
    ax.set_ylabel("Real self_attn GPU kernel envelope per layer (ms)")
    ax.tick_params(axis="x", rotation=0)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    add_chart_header(
        fig,
        ax,
        "Real prefill self_attn kernel-envelope distribution",
        "Each box uses layer samples after excluding layer 0; values are nsys first-kernel-start to last-kernel-end envelope.",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    for ext in ["png", "svg"]:
        fig.savefig(out_dir / f"prefill_real_kernel_envelope_boxplot.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_summary_md(
    out_dir: Path,
    real_detail: pd.DataFrame,
    skipped: pd.DataFrame,
    summary: pd.DataFrame,
    validation: pd.DataFrame,
) -> None:
    def mape(col: str) -> float:
        vals = []
        for _, row in summary.iterrows():
            gap = pct_gap(row.get(col), row.get("real_kernel_envelope_mean_ms"))
            if gap is not None:
                vals.append(abs(gap))
        return float(np.mean(vals)) if vals else float("nan")

    expected_primary_miss = validation["status"].eq("miss_expected").all() if not validation.empty else False
    lines = [
        "# Prefill Kernel Envelope AIC Compare",
        "",
        "## 范围",
        "",
        "- 仅包含 prefill；所有 decode 行已排除。",
        "- 实机侧只采用 `module_gpu_makespan_ms`，即 self_attn 内首个 GPU kernel 启动到最后一个 GPU kernel 结束的包络时间。",
        "- AIC 侧比较两个方案：普通 `DeepSeekModel` fallback 小算子 kernel 时间累加，以及 `WideEPDeepSeekModel` 的 `qkv_a_proj + WideEPContextMLA` 模块级路径。",
        "- 本批 DeepSeek fallback 的 `context_kv_b_proj_gemm` 使用 prefix-aware 查询：DeepSeek MLA prefill 会对 fresh 与 prefix latent KV 一并执行 `kv_b_proj`，因此 GEMM `m` 维按 `fresh_tokens + prefix_tokens` 查询。",
        "- 本批 DeepSeek fallback 同步纳入 `context_mla_concat_k`，对应 SGLang MLA prefill 中 `kv_b_proj` 后、attention 前的 K 拼接 kernel。",
        "- 每个 case/layer 为一个实机样本，并按既有规则排除 `layer_id=0`。",
        "",
        "## 规模",
        "",
        f"- 实机 layer 样本数: `{len(real_detail)}`",
        f"- case 数: `{summary['tag'].nunique()}`",
        f"- 跳过记录数: `{len(skipped)}`，其中包含按规则忽略的首层。",
        f"- `context_mla_module` primary miss 验证: `{'通过' if expected_primary_miss else '存在非预期结果'}`。",
        "",
        "## 平均绝对百分比误差（相对实机 kernel 包络时间）",
        "",
        f"- DeepSeek fallback stack: `{mape('aic_deepseek_fallback_total_ms'):.2f}%`",
        f"- WideEP module stack: `{mape('aic_wideep_total_ms'):.2f}%`",
        f"- Rule(prefix=0 -> WideEP, prefix>0 -> DeepSeek): `{summary['rule_aic_gap_pct_vs_real_kernel_envelope'].abs().mean():.2f}%`",
        f"- Best-of-two by absolute error: `{summary['best_aic_gap_pct_vs_real_kernel_envelope'].abs().mean():.2f}%`",
        "",
        "## 输出文件",
        "",
        "- `prefill_real_layer_detail.csv`: 实机每 case/layer 明细，保留原始 child/kernel JSON。",
        "- `prefill_aic_deepseek_fallback_stack.csv`: 普通 DeepSeek fallback 小算子查表明细。",
        "- `prefill_aic_wideep_stack.csv`: WideEP 对齐 op 查表明细。",
        "- `prefill_compare_summary.csv`: case 级总表，含两个 AIC 方案、rule 方案和 best-of-two 的 gap。",
        "- `prefill_kernel_envelope_aic_stacked_compare.png/svg`: 本轮主图。",
        "- `prefill_kernel_envelope_aic_rule_compare.png/svg`: 按 prefix 规则选择最终方案的新图。",
        "- `prefill_real_kernel_envelope_boxplot.png/svg`: 实机包络时间跨层分布箱线图。",
        "",
        "## 图中标注",
        "",
        "- 灰色柱: 实机 kernel 包络时间均值。",
        "- DeepSeek 堆叠柱: 普通 DeepSeekModel fallback 路径的小算子时间累加。",
        "- WideEP 斜线堆叠柱: `context_qkv_a_proj_gemm + wideep_context_mla`。",
        "- `DS` 和 `WE`: 分别表示两个 AIC 方案相对实机 kernel 包络时间的误差百分比。",
        "- `Rule`: 应用规则后最终采用方案的误差百分比；规则为 `prefix=0 -> WideEP`、`prefix>0 -> DeepSeek`。",
        "- 红色 `Best`: 仅在 Rule 不是最优时出现，表示两个 AIC 方案中误差绝对值更小者。",
    ]
    (out_dir / "README.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    bench_root = Path(args.bench_root)
    manifest_path = bench_root / "manifest.csv"
    out_dir = Path(args.output_dir) if args.output_dir else bench_root / "analysis" / "mla_aic_compare_kernel_envelope_refresh"
    out_dir.mkdir(parents=True, exist_ok=True)

    entries = load_manifest(manifest_path)
    if args.tag_prefix:
        entries = [entry for entry in entries if entry.tag.startswith(args.tag_prefix)]
        if not entries:
            raise SystemExit(f"No manifest entries match --tag-prefix {args.tag_prefix!r}.")
    real_detail, skipped = read_real_layer_rows(entries)
    if real_detail.empty:
        raise SystemExit("No prefill real detail rows after filtering layer 0.")

    db = PerfDatabase(
        system=args.system,
        backend=args.backend,
        version=args.version,
        systems_root=args.systems_root,
    )

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
    compare_summary = summarize_cases(real_detail, deepseek_stack, wideep_stack)

    real_detail.to_csv(out_dir / "prefill_real_layer_detail.csv", index=False)
    skipped.to_csv(out_dir / "prefill_skipped_rows.csv", index=False)
    deepseek_stack.to_csv(out_dir / "prefill_aic_deepseek_fallback_stack.csv", index=False)
    deepseek_validation.to_csv(out_dir / "prefill_aic_deepseek_primary_validation.csv", index=False)
    wideep_stack.to_csv(out_dir / "prefill_aic_wideep_stack.csv", index=False)
    compare_summary.to_csv(out_dir / "prefill_compare_summary.csv", index=False)

    save_envelope_boxplot(real_detail, compare_summary, out_dir)
    plot_kernel_envelope_compare(compare_summary, deepseek_stack, wideep_stack, out_dir)
    plot_kernel_envelope_rule_compare(compare_summary, deepseek_stack, wideep_stack, out_dir)
    write_summary_md(out_dir, real_detail, skipped, compare_summary, deepseek_validation)

    files = sorted(p.name for p in out_dir.iterdir() if p.is_file())
    if "artifact_manifest.json" not in files:
        files.append("artifact_manifest.json")
    manifest = {
        "output_dir": str(out_dir),
        "real_layer_rows": int(len(real_detail)),
        "case_count": int(compare_summary["tag"].nunique()),
        "skipped_rows": int(len(skipped)),
        "real_baseline": "module_gpu_makespan_ms",
        "aic_series": ["DeepSeekModel fallback stack", "WideEP qkv_a + WideEPContextMLA"],
        "kv_b_proj_policy": "context_kv_b_proj_gemm uses ContextKVBProjGEMM, querying GEMM m=fresh_tokens+prefix_tokens",
        "concat_k_policy": "DeepSeek fallback stack includes MLAConcatK(context_mla_concat_k) between context_kv_b_proj_gemm and context_attention",
        "files": sorted(files),
    }
    (out_dir / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
