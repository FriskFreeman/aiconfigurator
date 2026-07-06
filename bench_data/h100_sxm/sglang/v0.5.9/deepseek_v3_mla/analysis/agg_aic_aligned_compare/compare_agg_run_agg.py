#!/usr/bin/env python3
"""Compare formal SGLang mixed/agg runs with AIC SGLang run_agg estimates.

The real-machine side uses Nsight-derived GPU kernel envelopes from the MLA
module timing CSVs.  The AIC side uses SDK InferenceSession.run_agg and exports
both the final end-to-end result dict and the per-step/per-op breakdown that
run_agg attaches to the InferenceSummary.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import sys
import textwrap
import traceback
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


REPO_ROOT = Path(__file__).resolve().parents[7]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from aiconfigurator.sdk import common, config, models, perf_database
from aiconfigurator.sdk.backends.factory import get_backend
from aiconfigurator.sdk.inference_session import InferenceSession


DATA_ROOT = REPO_ROOT / "bench_data/h100_sxm/sglang/v0.5.9/deepseek_v3_mla"
DEFAULT_SUMMARY = DATA_ROOT / "agg_csv/formal_aic_aligned_agg_summary.csv"
DEFAULT_OUT_DIR = DATA_ROOT / "analysis/agg_aic_aligned_compare"


TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
}
COLORS = {
    "blue_base": "#A3BEFA",
    "blue_dark": "#2E4780",
    "orange_base": "#F0986E",
    "orange_dark": "#804126",
    "olive_base": "#A3D576",
    "olive_dark": "#386411",
    "gold_base": "#FFE15B",
    "gold_dark": "#736422",
    "neutral_base": "#C5CAD3",
    "neutral_dark": "#464C55",
}


def _jsonify(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if hasattr(value, "name") and hasattr(value, "value"):
        return value.name
    return repr(value)


def _parse_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    try:
        parsed = ast.literal_eval(text)
    except Exception:
        return []
    return parsed if isinstance(parsed, list) else []


def _first_or_zero(values: list[Any]) -> int:
    for value in values:
        if value is not None:
            try:
                return int(value)
            except Exception:
                continue
    return 0


def use_chart_theme() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": TOKENS["surface"],
            "figure.edgecolor": "none",
            "savefig.facecolor": TOKENS["panel"],
            "savefig.edgecolor": "none",
            "axes.facecolor": TOKENS["panel"],
            "axes.edgecolor": TOKENS["axis"],
            "axes.labelcolor": TOKENS["ink"],
            "axes.grid": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": TOKENS["grid"],
            "grid.linewidth": 0.8,
            "font.family": "sans-serif",
            "font.sans-serif": ["Aptos", "Inter", "Segoe UI", "DejaVu Sans", "Arial", "sans-serif"],
            "font.monospace": ["SF Mono", "Menlo", "Consolas", "DejaVu Sans Mono", "monospace"],
            "patch.linewidth": 1.0,
        }
    )


def add_chart_header(fig, ax, title: str, subtitle: str) -> None:
    title = textwrap.fill(title.strip(), width=80, break_long_words=False)
    subtitle = textwrap.fill(subtitle.strip(), width=118, break_long_words=False)
    title_lines = title.count("\n") + 1
    subtitle_lines = subtitle.count("\n") + 1
    fig.subplots_adjust(top=max(0.62, 0.86 - 0.045 * (title_lines - 1) - 0.032 * (subtitle_lines - 1)))
    left = ax.get_position().x0
    fig.text(left, 0.985, title, ha="left", va="top", fontsize=13, fontweight="semibold", color=TOKENS["ink"])
    fig.text(
        left,
        0.93 - 0.045 * (title_lines - 1),
        subtitle,
        ha="left",
        va="top",
        fontsize=9,
        color=TOKENS["muted"],
        linespacing=1.18,
    )
    ax.set_title("")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_chart(fig, out_dir: Path, stem: str) -> None:
    fig.savefig(out_dir / f"{stem}.png", dpi=180, bbox_inches="tight", facecolor=TOKENS["panel"], transparent=False)
    fig.savefig(out_dir / f"{stem}.svg", bbox_inches="tight", facecolor=TOKENS["panel"], transparent=False)
    plt.close(fig)


def build_model_config(args: argparse.Namespace) -> config.ModelConfig:
    return config.ModelConfig(
        tp_size=args.tp_size,
        pp_size=args.pp_size,
        attention_dp_size=args.attention_dp_size,
        moe_tp_size=args.moe_tp_size,
        moe_ep_size=args.moe_ep_size,
        gemm_quant_mode=common.GEMMQuantMode[args.gemm_quant_mode],
        moe_quant_mode=common.MoEQuantMode[args.moe_quant_mode],
        kvcache_quant_mode=common.KVCacheQuantMode[args.kvcache_quant_mode],
        fmha_quant_mode=common.FMHAQuantMode[args.fmha_quant_mode],
        comm_quant_mode=common.CommQuantMode[args.comm_quant_mode],
        overwrite_num_layers=args.num_layers,
        attention_backend=args.attention_backend,
        moe_backend=args.moe_backend,
        enable_wideep=args.enable_wideep,
    )


def build_session(args: argparse.Namespace) -> InferenceSession:
    model_config = build_model_config(args)
    model = models.get_model(args.model_path, model_config, args.backend)
    db = perf_database.get_database(args.system, args.backend, args.version, database_mode=args.database_mode)
    if db is None:
        raise RuntimeError(f"failed to load database for {args.system}/{args.backend}/{args.version}")
    db.set_default_database_mode(common.DatabaseMode[args.database_mode])
    backend = get_backend(args.backend)
    return InferenceSession(model, db, backend)


def extract_real_metrics(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in summary_df.iterrows():
        csv_path = Path(str(row["csv_path"]))
        metric_row: dict[str, Any] = {
            "tag": row["tag"],
            "alignment_class": row["alignment_class"],
            "status": row["status"],
            "csv_path": str(csv_path),
        }
        try:
            detail = pd.read_csv(csv_path)
            first_kernel = detail["module_first_kernel_start_ns"].min()
            last_kernel = detail["module_last_kernel_end_ns"].max()
            metric_row.update(
                {
                    "real_csv_rows": int(len(detail)),
                    "real_layer_count": int(detail["layer_id"].nunique()) if "layer_id" in detail else int(len(detail)),
                    "real_stage_values": ",".join(sorted(map(str, detail["stage"].dropna().unique()))),
                    "real_mla_gpu_envelope_all_layers_ms": (last_kernel - first_kernel) / 1e6,
                    "real_mla_module_gpu_makespan_sum_ms": float(detail["module_gpu_makespan_ms"].sum()),
                    "real_mla_gpu_kernel_time_sum_ms": float(detail["module_gpu_kernel_time_sum_ms"].sum()),
                    "real_layer0_gpu_makespan_ms": float(
                        detail.sort_values("layer_id").iloc[0]["module_gpu_makespan_ms"]
                    ),
                    "real_nonfirst_layer_gpu_makespan_avg_ms": float(
                        detail.loc[detail["layer_id"] != detail["layer_id"].min(), "module_gpu_makespan_ms"].mean()
                    ),
                    "real_attention_token_count_first_layer": float(
                        detail.sort_values("layer_id").iloc[0].get("attention_token_count", np.nan)
                    ),
                }
            )
        except Exception as exc:
            metric_row.update(
                {
                    "real_error": f"{type(exc).__name__}: {exc}",
                    "real_traceback": traceback.format_exc(),
                }
            )
        rows.append(metric_row)
    return pd.DataFrame(rows)


def derive_aic_params(row: pd.Series, mapping: str, osl: int) -> dict[str, Any]:
    intended_total_isl = int(row["prefill_prefix_len"]) + int(row["prefill_fresh_len"])
    intended_total_isl = max(intended_total_isl, int(row["decode_prefix_len"]))
    if mapping == "intended_case":
        return {
            "mapping": mapping,
            "aic_batch_size": int(row["decode_batch_size"]) + int(row["prefill_batch_size"]),
            "aic_isl": intended_total_isl,
            "aic_prefix": int(row["prefill_prefix_len"]),
            "aic_ctx_tokens": int(row["prefill_batch_size"]) * intended_total_isl,
            "aic_osl": osl,
            "mapping_note": "case design: decode_batch + prefill_batch; ctx_tokens=prefill_batch*total_isl",
        }

    fresh_lens = [int(x) for x in _parse_list(row.get("first_formal_prefill_fresh_lens"))]
    prefix_lens = [int(x) for x in _parse_list(row.get("first_formal_prefill_prefix_lens"))]
    decode_kv_lens = [int(x) for x in _parse_list(row.get("first_formal_decode_kv_lens"))]
    observed_isl_candidates = fresh_lens + decode_kv_lens + [intended_total_isl]
    observed_isl = max(observed_isl_candidates) if observed_isl_candidates else intended_total_isl
    observed_prefix = _first_or_zero(prefix_lens)
    observed_ctx_req_count = int(row.get("first_formal_prefill_req_count", 0))
    observed_decode_req_count = int(row.get("first_formal_decode_req_count", 0))
    return {
        "mapping": mapping,
        "aic_batch_size": max(1, observed_ctx_req_count + observed_decode_req_count),
        "aic_isl": max(1, observed_isl),
        "aic_prefix": max(0, observed_prefix),
        "aic_ctx_tokens": max(1, observed_ctx_req_count * max(1, observed_isl)),
        "aic_osl": osl,
        "mapping_note": "observed first formal mixed batch: batch=req_count; ctx_tokens=prefill_req_count*observed_isl",
    }


def run_aic_rows(summary_df: pd.DataFrame, session: InferenceSession, args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, case in summary_df.iterrows():
        for mapping in args.mappings:
            params = derive_aic_params(case, mapping, args.osl)
            out: dict[str, Any] = {
                "tag": case["tag"],
                "alignment_class": case["alignment_class"],
                "decode_batch_size": int(case["decode_batch_size"]),
                "decode_prefix_len": int(case["decode_prefix_len"]),
                "prefill_batch_size": int(case["prefill_batch_size"]),
                "prefill_fresh_len": int(case["prefill_fresh_len"]),
                "prefill_prefix_len": int(case["prefill_prefix_len"]),
                "first_formal_prefill_req_count": int(case.get("first_formal_prefill_req_count", 0)),
                "first_formal_decode_req_count": int(case.get("first_formal_decode_req_count", 0)),
                "first_formal_sum_prefill_fresh_tokens": int(case.get("first_formal_sum_prefill_fresh_tokens", 0)),
                "first_formal_sum_decode_kv_tokens": int(case.get("first_formal_sum_decode_kv_tokens", 0)),
                **params,
            }
            try:
                runtime = config.RuntimeConfig(
                    batch_size=params["aic_batch_size"],
                    isl=params["aic_isl"],
                    osl=params["aic_osl"],
                    prefix=params["aic_prefix"],
                    engine_step_backend=args.engine_step_backend,
                )
                summary = session.run_agg(
                    runtime,
                    ctx_tokens=params["aic_ctx_tokens"],
                    max_seq_len=params["aic_isl"] + params["aic_osl"],
                )
                result = summary.get_result_dict() or {}
                per_ops = summary.get_per_ops_data() or {}
                scheduling = per_ops.get("scheduling", {})
                mix_step = per_ops.get("mix_step", {})
                genonly_step = per_ops.get("genonly_step", {})
                out.update({f"aic_{k}": v for k, v in result.items()})
                out.update(
                    {
                        "aic_oom": bool(summary.check_oom()),
                        "aic_kv_cache_oom": bool(summary.check_kv_cache_oom()),
                        "aic_mix_step_latency_ms": scheduling.get("mix_step_latency_ms"),
                        "aic_genonly_step_latency_ms": scheduling.get("genonly_step_latency_ms"),
                        "aic_num_mix_steps": scheduling.get("num_mix_steps"),
                        "aic_num_genonly_steps": scheduling.get("num_genonly_steps"),
                        "aic_mix_context_mla_block_ms": mix_step.get("context_mla_block"),
                        "aic_mix_context_mla_block_per_layer_ms": (
                            mix_step.get("context_mla_block") / args.num_layers
                            if mix_step.get("context_mla_block") is not None and args.num_layers
                            else np.nan
                        ),
                        "aic_mix_generation_mla_block_ms": mix_step.get("generation_mla_block"),
                        "aic_genonly_generation_mla_block_ms": genonly_step.get("generation_mla_block"),
                        "aic_per_ops_json": json.dumps(_jsonify(per_ops), ensure_ascii=False, sort_keys=True),
                        "aic_status": "ok",
                    }
                )
            except Exception as exc:
                out.update(
                    {
                        "aic_status": "error",
                        "aic_error": f"{type(exc).__name__}: {exc}",
                        "aic_traceback": traceback.format_exc(),
                    }
                )
            rows.append(out)
    return pd.DataFrame(rows)


def add_comparison_columns(joined: pd.DataFrame) -> pd.DataFrame:
    joined = joined.copy()
    joined["ratio_real_envelope_to_aic_mla_block"] = (
        joined["real_mla_gpu_envelope_all_layers_ms"] / joined["aic_mix_context_mla_block_ms"]
    )
    joined["ratio_real_nonfirst_layer_to_aic_mla_per_layer"] = (
        joined["real_nonfirst_layer_gpu_makespan_avg_ms"] / joined["aic_mix_context_mla_block_per_layer_ms"]
    )
    joined["ratio_real_envelope_to_aic_mix_step"] = (
        joined["real_mla_gpu_envelope_all_layers_ms"] / joined["aic_mix_step_latency_ms"]
    )
    joined["real_minus_aic_mla_block_ms"] = (
        joined["real_mla_gpu_envelope_all_layers_ms"] - joined["aic_mix_context_mla_block_ms"]
    )
    return joined


def make_case_label(row: pd.Series, *, observed: bool = False) -> str:
    """Compact, multiline label for mixed P/D cases."""
    if observed:
        p_b = int(row.get("first_formal_prefill_req_count", 0))
        d_b = int(row.get("first_formal_decode_req_count", 0))
        ctx = int(row.get("first_formal_sum_prefill_fresh_tokens", 0))
        d_kv_total = int(row.get("first_formal_sum_decode_kv_tokens", 0))
        d_kv = int(round(d_kv_total / d_b)) if d_b > 0 else int(row.get("decode_prefix_len", 0))
        p_isl = int(row.get("aic_isl", 0))
        prefix = int(row.get("aic_prefix", 0))
        return f"Obs P b{p_b} ctx{ctx}\nisl{p_isl} pre{prefix}\nD b{d_b} kv{d_kv}"

    p_b = int(row.get("prefill_batch_size", 0))
    p_fresh = int(row.get("prefill_fresh_len", 0))
    prefix = int(row.get("prefill_prefix_len", 0))
    p_isl = prefix + p_fresh
    ctx = p_b * p_isl
    d_b = int(row.get("decode_batch_size", 0))
    d_kv = int(row.get("decode_prefix_len", 0))
    return f"P b{p_b} ctx{ctx}\nisl{p_isl} pre{prefix}\nD b{d_b} kv{d_kv}"


def plot_mla_block_comparison(joined: pd.DataFrame, out_dir: Path) -> None:
    plot_df = joined[(joined["mapping"] == "observed_first_batch") & (joined["aic_status"] == "ok")].copy()
    if plot_df.empty:
        return
    plot_df["case_label"] = plot_df.apply(lambda row: make_case_label(row, observed=False), axis=1)
    long_df = plot_df.melt(
        id_vars=["case_label", "alignment_class"],
        value_vars=["real_mla_gpu_envelope_all_layers_ms", "aic_mix_context_mla_block_ms"],
        var_name="metric",
        value_name="latency_ms",
    )
    label_map = {
        "real_mla_gpu_envelope_all_layers_ms": "Real MLA GPU envelope",
        "aic_mix_context_mla_block_ms": "AIC run_agg MLA block",
    }
    long_df["metric"] = long_df["metric"].map(label_map)
    fig, ax = plt.subplots(figsize=(max(15.5, len(plot_df) * 1.28), 6.8))
    sns.barplot(
        data=long_df,
        x="case_label",
        y="latency_ms",
        hue="metric",
        palette={
            "Real MLA GPU envelope": COLORS["orange_base"],
            "AIC run_agg MLA block": COLORS["blue_base"],
        },
        edgecolor=COLORS["neutral_dark"],
        linewidth=0.8,
        ax=ax,
    )
    ax.set_xlabel("Case")
    ax.set_ylabel("Latency (ms)")
    ax.tick_params(axis="x", rotation=0, labelsize=6.5)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), frameon=False, ncol=2, borderaxespad=0)
    add_chart_header(
        fig,
        ax,
        "Observed mixed batch MLA GPU envelope versus AIC run_agg MLA block",
        "Real metric is Nsight GPU kernel envelope across six MLA layers; AIC metric is context_mla_block from run_agg per-op data under observed-first-batch mapping.",
    )
    save_chart(fig, out_dir, "mla_envelope_vs_aic_mla_block_observed")


def plot_ratio(joined: pd.DataFrame, out_dir: Path) -> None:
    plot_df = joined[(joined["mapping"] == "observed_first_batch") & (joined["aic_status"] == "ok")].copy()
    if plot_df.empty:
        return
    plot_df["case_label"] = plot_df.apply(lambda row: make_case_label(row, observed=False), axis=1)
    plot_df = plot_df.sort_values("ratio_real_envelope_to_aic_mla_block", ascending=True)
    fig, ax = plt.subplots(figsize=(12.5, 6.7))
    colors = np.where(plot_df["alignment_class"].eq("strict_prefix0"), COLORS["olive_base"], COLORS["gold_base"])
    bars = ax.barh(
        plot_df["case_label"],
        plot_df["ratio_real_envelope_to_aic_mla_block"],
        color=colors,
        edgecolor=COLORS["neutral_dark"],
        linewidth=0.8,
    )
    ax.axvline(1.0, color=COLORS["neutral_dark"], linewidth=1.0, linestyle="--")
    for bar, value in zip(bars, plot_df["ratio_real_envelope_to_aic_mla_block"]):
        ax.text(value + 0.02, bar.get_y() + bar.get_height() / 2, f"{value:.2f}x", va="center", fontsize=8)
    ax.set_xlabel("Real / AIC ratio")
    ax.set_ylabel("Case")
    add_chart_header(
        fig,
        ax,
        "Real MLA envelope often exceeds the AIC run_agg MLA block estimate",
        "Ratio uses observed-first-batch mapping. Dashed line marks parity; strict_prefix0 and prefix_probe are colored separately.",
    )
    save_chart(fig, out_dir, "real_to_aic_mla_block_ratio_observed")


def plot_aic_e2e(joined: pd.DataFrame, out_dir: Path) -> None:
    plot_df = joined[(joined["mapping"] == "intended_case") & (joined["aic_status"] == "ok")].copy()
    if plot_df.empty:
        return
    plot_df["case_label"] = plot_df.apply(lambda row: make_case_label(row, observed=False), axis=1)
    long_df = plot_df.melt(
        id_vars=["case_label", "alignment_class"],
        value_vars=["aic_ttft", "aic_tpot", "aic_request_latency"],
        var_name="metric",
        value_name="latency_ms",
    )
    long_df["metric"] = long_df["metric"].map(
        {
            "aic_ttft": "AIC TTFT",
            "aic_tpot": "AIC TPOT",
            "aic_request_latency": "AIC request latency",
        }
    )
    fig, ax = plt.subplots(figsize=(max(15.5, len(plot_df) * 1.28), 6.8))
    sns.barplot(
        data=long_df,
        x="case_label",
        y="latency_ms",
        hue="metric",
        palette={
            "AIC TTFT": COLORS["blue_base"],
            "AIC TPOT": COLORS["olive_base"],
            "AIC request latency": COLORS["orange_base"],
        },
        edgecolor=COLORS["neutral_dark"],
        linewidth=0.8,
        ax=ax,
    )
    ax.set_xlabel("Case")
    ax.set_ylabel("Latency (ms)")
    ax.tick_params(axis="x", rotation=0, labelsize=6.5)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), frameon=False, ncol=3, borderaxespad=0)
    add_chart_header(
        fig,
        ax,
        "AIC run_agg final end-to-end metrics under intended case mapping",
        "These are full run_agg service-level outputs, not single-module GPU envelopes; use them as the requested coarse endpoint baseline.",
    )
    save_chart(fig, out_dir, "aic_run_agg_e2e_metrics_intended")


def write_report(joined: pd.DataFrame, real_df: pd.DataFrame, aic_df: pd.DataFrame, args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    ok = joined[joined["aic_status"] == "ok"].copy()
    observed = ok[ok["mapping"] == "observed_first_batch"]
    intended = ok[ok["mapping"] == "intended_case"]
    by_class = (
        observed.groupby("alignment_class", dropna=False)
        .agg(
            cases=("tag", "count"),
            real_mla_envelope_ms_avg=("real_mla_gpu_envelope_all_layers_ms", "mean"),
            aic_mla_block_ms_avg=("aic_mix_context_mla_block_ms", "mean"),
            ratio_real_to_aic_mla_avg=("ratio_real_envelope_to_aic_mla_block", "mean"),
            real_to_mix_step_share_avg=("ratio_real_envelope_to_aic_mix_step", "mean"),
        )
        .reset_index()
    )
    by_class.to_csv(out_dir / "summary_by_alignment_class.csv", index=False)

    report = f"""# AGG Mixed 实机与 AIC run_agg 对比摘要

## 输入与口径

- 实机输入：`{Path(args.summary_csv).relative_to(REPO_ROOT)}`，共 {len(real_df)} 个 case。
- 实机主指标：每个 case 的 `MLA时延拆解.csv` 中，六层 MLA 模块的 GPU kernel 时间包络 `real_mla_gpu_envelope_all_layers_ms`；同时保留 makespan 求和、kernel 净时延求和、首层/非首层均值。
- AIC 主入口：`InferenceSession.run_agg(...)`，导出最终端到端结果字段 `ttft/tpot/request_latency/tokens/s`，并保留 run_agg 内部 `per_ops_data`。
- AIC 配置：`model={args.model_path}`，`system/backend/version={args.system}/{args.backend}/{args.version}`，`tp/pp/dp={args.tp_size}/{args.pp_size}/{args.attention_dp_size}`，`layers={args.num_layers}`，`gemm={args.gemm_quant_mode}`，`moe={args.moe_quant_mode}`，`kvcache={args.kvcache_quant_mode}`，`fmha={args.fmha_quant_mode}`，`database_mode={args.database_mode}`。

## 映射方式

- `intended_case`：使用用例设计参数，`batch_size=decode_batch_size+prefill_batch_size`，`isl=prefill_prefix_len+prefill_fresh_len`，`prefix=prefill_prefix_len`，`ctx_tokens=prefill_batch_size*isl`。
- `observed_first_batch`：使用 Nsight/meta 解析到的首个 formal mixed batch 实际组成，`batch_size=first_formal_prefill_req_count+first_formal_decode_req_count`，`ctx_tokens=prefill_req_count*observed_isl`。

## 关键结果

- AIC 成功行数：{int((aic_df["aic_status"] == "ok").sum())}/{len(aic_df)}。
- `observed_first_batch` 平均 `real/AIC context_mla_block`：{observed["ratio_real_envelope_to_aic_mla_block"].mean():.3f}x。
- `strict_prefix0` case 数：{int((real_df["alignment_class"] == "strict_prefix0").sum())}；`prefix_probe` case 数：{int((real_df["alignment_class"] == "prefix_probe").sum())}。
- 注意：AIC `ttft/tpot/request_latency` 是 run_agg 服务级端到端结果，不应直接解释为单个 mixed batch 的 MLA 模块时间；更可比的细粒度列是 `aic_mix_context_mla_block_ms` 与实机的 MLA GPU 包络。

## 产物

- `real_metrics.csv`：实机侧从每个 Nsight CSV 复算的 MLA GPU 指标。
- `aic_run_agg_results.csv`：AIC `run_agg` 结果、调度字段和 per-op JSON。
- `real_aic_joined.csv`：实机和 AIC 合并后的对比表。
- `summary_by_alignment_class.csv`：按 `strict_prefix0/prefix_probe` 聚合的摘要。
- `mla_envelope_vs_aic_mla_block_observed.png/svg`：实机 MLA 六层 GPU 包络 vs AIC run_agg MLA block。
- `real_to_aic_mla_block_ratio_observed.png/svg`：实机/AIC MLA block 比值。
- `aic_run_agg_e2e_metrics_intended.png/svg`：AIC run_agg 端到端输出指标。

## 风险与下一步

- `prefix_probe` 中，SGLang 实际首个 formal batch 可能因 prefix cache/chunked prefill 调度而与设计值不同；因此报告同时保留 intended 与 observed 两套映射。
- 当前对比先满足“run_agg 最终端到端结果”需求，但真正与 Nsight 单 batch GPU 包络最接近的是 run_agg 的 `mix_step.context_mla_block` 分解项。
- 后续若要做严谨误差归因，应固定一个 case，进一步对齐 AIC `mix_step_latency_ms` 的非 attention/MLA/MoE 子项与 Nsight 对应模块包络。
"""
    (out_dir / "agg_aic_aligned_compare_summary.md").write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--model-path", default="deepseek-ai/DeepSeek-V3")
    parser.add_argument("--system", default="h100_sxm")
    parser.add_argument("--backend", default="sglang")
    parser.add_argument("--version", default="0.5.9")
    parser.add_argument("--database-mode", choices=[mode.name for mode in common.DatabaseMode], default="SILICON")
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--pp-size", type=int, default=1)
    parser.add_argument("--attention-dp-size", type=int, default=1)
    parser.add_argument("--moe-tp-size", type=int, default=1)
    parser.add_argument("--moe-ep-size", type=int, default=1)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--gemm-quant-mode", default="fp8_block")
    parser.add_argument("--moe-quant-mode", default="fp8_block")
    parser.add_argument("--kvcache-quant-mode", default="fp8")
    parser.add_argument("--fmha-quant-mode", default="bfloat16")
    parser.add_argument("--comm-quant-mode", default="half")
    parser.add_argument("--attention-backend", default="fa3")
    parser.add_argument("--moe-backend", default=None)
    parser.add_argument("--enable-wideep", action="store_true")
    parser.add_argument("--engine-step-backend", default=None)
    parser.add_argument("--osl", type=int, default=16)
    parser.add_argument("--mappings", nargs="+", default=["intended_case", "observed_first_batch"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    use_chart_theme()

    summary_df = pd.read_csv(args.summary_csv)
    real_df = extract_real_metrics(summary_df)
    session = build_session(args)
    aic_df = run_aic_rows(summary_df, session, args)
    joined = add_comparison_columns(real_df.merge(aic_df, on=["tag", "alignment_class"], how="left"))

    real_df.to_csv(args.out_dir / "real_metrics.csv", index=False)
    aic_df.to_csv(args.out_dir / "aic_run_agg_results.csv", index=False)
    joined.to_csv(args.out_dir / "real_aic_joined.csv", index=False)
    (args.out_dir / "analysis_metadata.json").write_text(
        json.dumps(
            _jsonify(
                {
                    "summary_csv": str(args.summary_csv),
                    "out_dir": str(args.out_dir),
                    "args": vars(args),
                    "real_rows": len(real_df),
                    "aic_rows": len(aic_df),
                    "joined_rows": len(joined),
                }
            ),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    plot_mla_block_comparison(joined, args.out_dir)
    plot_ratio(joined, args.out_dir)
    plot_aic_e2e(joined, args.out_dir)
    write_report(joined, real_df, aic_df, args)

    print(f"Wrote analysis to {args.out_dir}")
    print(f"real rows={len(real_df)} aic rows={len(aic_df)} joined rows={len(joined)}")
    print(f"aic ok={(aic_df['aic_status'] == 'ok').sum()} / {len(aic_df)}")


if __name__ == "__main__":
    main()
