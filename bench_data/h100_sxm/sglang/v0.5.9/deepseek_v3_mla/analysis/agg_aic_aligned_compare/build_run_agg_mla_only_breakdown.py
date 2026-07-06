#!/usr/bin/env python3
"""Build MLA-only AGG breakdown tables from real Nsight CSVs and AIC run_agg data.

This intentionally excludes MoE/FFN/embedding/logits layer-level components.
The AIC side keeps two views:

1. actual run_agg block values exported in aic_per_ops_json, e.g.
   context_mla_block and generation_mla_block;
2. forced fallback decomposition of the model's MLA FallbackOp internals, used
   for attribution only. It does not claim to be the exact path run_agg used
   when module-level tables are available.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import textwrap
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[6]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from aiconfigurator.sdk import config  # noqa: E402
from compare_agg_run_agg import build_session, make_case_label  # noqa: E402


DEFAULT_JOINED = SCRIPT_DIR / "real_aic_joined.csv"
DEFAULT_OUT_DIR = SCRIPT_DIR / "run_agg_mla_only_breakdown"

TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
}

COLORS = {
    "real": "#F0986E",
    "aic_block": "#A3BEFA",
    "aic_fallback": "#A3D576",
    "neutral": "#C5CAD3",
    "edge": "#464C55",
    "context_downscale_gemm": "#A3BEFA",
    "context_q_b_proj_gemm": "#CEDFFE",
    "context_kv_b_proj_gemm": "#5477C4",
    "context_mla_concat_k": "#BEEB96",
    "context_attention": "#FFE15B",
    "context_proj_gemm": "#F0986E",
    "generation_downscale_gemm": "#A3BEFA",
    "generation_q_b_proj_gemm": "#CEDFFE",
    "generation_bmm_pre": "#BEEB96",
    "generation_attention": "#FFE15B",
    "generation_bmm_post": "#F390CA",
    "generation_proj_gemm": "#F0986E",
    "other": "#E2E5EA",
}

REAL_CHILD_TO_AIC_HINT = {
    "fused_qkv_a_proj_with_mqa": "context_downscale_gemm / generation_downscale_gemm",
    "q_a_layernorm": "not modeled in DeepSeekModel FallbackOp",
    "kv_a_layernorm": "not modeled in DeepSeekModel FallbackOp",
    "q_b_proj": "context_q_b_proj_gemm / generation_q_b_proj_gemm",
    "rotary_emb": "partly inside context_attention / generation bmm_rope handling",
    "attn_mha": "context_attention",
    "attn_mqa": "generation_attention",
    "o_proj": "context_proj_gemm / generation_proj_gemm",
}


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
        }
    )


def add_chart_header(fig, ax, title: str, subtitle: str) -> None:
    title = textwrap.fill(title.strip(), width=88, break_long_words=False)
    subtitle = textwrap.fill(subtitle.strip(), width=120, break_long_words=False)
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


def save_chart(fig, out_dir: Path, stem: str) -> None:
    fig.savefig(out_dir / f"{stem}.png", dpi=180, bbox_inches="tight", facecolor=TOKENS["panel"], transparent=False)
    fig.savefig(out_dir / f"{stem}.svg", bbox_inches="tight", facecolor=TOKENS["panel"], transparent=False)
    plt.close(fig)


def safe_json_loads(value: Any, default: Any) -> Any:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return default
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return default
    return parsed


def find_fallback_op(model: Any, op_name: str, *, phase: str) -> Any:
    ops = model.context_ops if phase == "context" else model.generation_ops
    for op in ops:
        if getattr(op, "_name", None) == op_name and hasattr(op, "_fallback"):
            return op
    raise KeyError(f"could not find {phase} fallback op {op_name}")


def query_op(op: Any, database: Any, *, batch_size: int, s: int, prefix: int, model_name: str = "") -> dict[str, Any]:
    kwargs = {
        "x": batch_size * s,
        "batch_size": batch_size,
        "beam_width": 1,
        "s": s,
        "prefix": prefix,
        "model_name": model_name,
    }
    result = op.query(database, **kwargs)
    return {
        "latency_ms": float(result),
        "energy_wms": float(getattr(result, "energy", 0.0) or 0.0),
        "source": getattr(result, "source", "silicon"),
    }


def derive_run_agg_step_shape(row: pd.Series) -> dict[str, Any]:
    b = int(row["aic_batch_size"])
    isl = int(row["aic_isl"])
    osl = int(row["aic_osl"])
    prefix = int(row["aic_prefix"])
    ctx_tokens = int(row["aic_ctx_tokens"])
    steps_to_finish_ctx = math.ceil(isl * b / ctx_tokens)
    if b > 1:
        if steps_to_finish_ctx >= osl:
            mix_ctx_tokens = ctx_tokens
            mix_gen_tokens = max(1, int(b // (steps_to_finish_ctx / osl)))
            genonly_tokens = 0
        else:
            mix_ctx_tokens = ctx_tokens
            mix_gen_tokens = int(b - math.ceil(ctx_tokens / isl))
            genonly_tokens = b
    else:
        mix_ctx_tokens = ctx_tokens
        mix_gen_tokens = 0
        genonly_tokens = 1
    return {
        "steps_to_finish_ctx": steps_to_finish_ctx,
        "mix_ctx_tokens": int(mix_ctx_tokens),
        "mix_gen_tokens": int(max(0, mix_gen_tokens)),
        "genonly_tokens": int(max(0, genonly_tokens)),
        "mix_first_pass_batch_size": 1,
        "mix_first_pass_isl": int(mix_ctx_tokens + max(0, mix_gen_tokens)),
        "mix_first_pass_prefix": int(prefix * math.floor(ctx_tokens / isl)),
        "mix_context_second_pass_batch_size": int(math.ceil(ctx_tokens / isl)),
        "mix_context_second_pass_isl": int(isl),
        "mix_context_second_pass_prefix": int(prefix),
        "generation_lookup_s": int(isl + osl // 2 + 1),
    }


def append_aic_block_rows(rows: list[dict[str, Any]], row: pd.Series, per_ops: dict[str, Any], num_layers: int) -> None:
    components = [
        ("mix_step", "context_mla_block", "context", "actual_run_agg_block"),
        ("mix_step", "context_attention (scaled)", "context", "actual_run_agg_scaled_attention"),
        ("mix_step", "generation_attention", "generation", "actual_run_agg_generation_attention"),
        ("mix_step", "generation_mla_block", "generation", "actual_run_agg_generation_mla_block"),
        ("genonly_step", "generation_mla_block", "generation", "actual_run_agg_block"),
    ]
    for step, op_name, phase, query_path in components:
        op_map = per_ops.get(step, {})
        if not isinstance(op_map, dict) or op_name not in op_map:
            continue
        latency = float(op_map.get(op_name) or 0.0)
        rows.append(
            {
                "tag": row["tag"],
                "mapping": row["mapping"],
                "alignment_class": row["alignment_class"],
                "step": step,
                "phase": phase,
                "aic_query_path": query_path,
                "op_name": op_name,
                "latency_ms": latency,
                "per_layer_latency_ms": latency / num_layers if num_layers else np.nan,
                "step_count": float(row.get("aic_num_mix_steps" if step == "mix_step" else "aic_num_genonly_steps", 0.0) or 0.0),
                "weighted_latency_ms": latency
                * float(row.get("aic_num_mix_steps" if step == "mix_step" else "aic_num_genonly_steps", 0.0) or 0.0),
                "source": "aic_per_ops_json",
            }
        )


def expand_aic_mla(joined: pd.DataFrame, session: Any, num_layers: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    model = session._model
    database = session._database
    model_name = str(getattr(model, "model_name", "") or "")
    context_mla = find_fallback_op(model, "context_mla_block", phase="context")
    generation_mla = find_fallback_op(model, "generation_mla_block", phase="generation")

    for _, row in joined.drop_duplicates(["tag", "mapping"]).iterrows():
        per_ops = safe_json_loads(row.get("aic_per_ops_json"), {})
        append_aic_block_rows(rows, row, per_ops, num_layers)
        shape = derive_run_agg_step_shape(row)

        context_s = shape["mix_first_pass_isl"] - shape["mix_first_pass_prefix"]
        if context_s > 0:
            for op in context_mla._fallback:
                op_name = getattr(op, "_name", type(op).__name__)
                try:
                    result = query_op(
                        op,
                        database,
                        batch_size=shape["mix_first_pass_batch_size"],
                        s=context_s,
                        prefix=shape["mix_first_pass_prefix"],
                        model_name=model_name,
                    )
                    rows.append(
                        {
                            "tag": row["tag"],
                            "mapping": row["mapping"],
                            "alignment_class": row["alignment_class"],
                            "step": "mix_step",
                            "phase": "context",
                            "aic_query_path": "forced_fallback_decomposition",
                            "op_name": op_name,
                            "latency_ms": result["latency_ms"],
                            "per_layer_latency_ms": result["latency_ms"] / num_layers if num_layers else np.nan,
                            "step_count": float(row.get("aic_num_mix_steps", 0.0) or 0.0),
                            "weighted_latency_ms": result["latency_ms"] * float(row.get("aic_num_mix_steps", 0.0) or 0.0),
                            "source": result["source"],
                            **shape,
                        }
                    )
                except Exception as exc:  # keep the table useful even when one lookup misses.
                    rows.append(
                        {
                            "tag": row["tag"],
                            "mapping": row["mapping"],
                            "alignment_class": row["alignment_class"],
                            "step": "mix_step",
                            "phase": "context",
                            "aic_query_path": "forced_fallback_decomposition",
                            "op_name": op_name,
                            "latency_ms": np.nan,
                            "per_layer_latency_ms": np.nan,
                            "step_count": float(row.get("aic_num_mix_steps", 0.0) or 0.0),
                            "weighted_latency_ms": np.nan,
                            "source": f"error: {type(exc).__name__}: {exc}",
                            **shape,
                        }
                    )

        for step, batch_size, step_count_col in (
            ("mix_step", shape["mix_gen_tokens"], "aic_num_mix_steps"),
            ("genonly_step", shape["genonly_tokens"], "aic_num_genonly_steps"),
        ):
            if batch_size <= 0:
                continue
            for op in generation_mla._fallback:
                op_name = getattr(op, "_name", type(op).__name__)
                try:
                    result = query_op(
                        op,
                        database,
                        batch_size=batch_size,
                        s=shape["generation_lookup_s"],
                        prefix=0,
                        model_name=model_name,
                    )
                    rows.append(
                        {
                            "tag": row["tag"],
                            "mapping": row["mapping"],
                            "alignment_class": row["alignment_class"],
                            "step": step,
                            "phase": "generation",
                            "aic_query_path": "forced_fallback_decomposition",
                            "op_name": op_name,
                            "latency_ms": result["latency_ms"],
                            "per_layer_latency_ms": result["latency_ms"] / num_layers if num_layers else np.nan,
                            "step_count": float(row.get(step_count_col, 0.0) or 0.0),
                            "weighted_latency_ms": result["latency_ms"] * float(row.get(step_count_col, 0.0) or 0.0),
                            "source": result["source"],
                            **shape,
                        }
                    )
                except Exception as exc:
                    rows.append(
                        {
                            "tag": row["tag"],
                            "mapping": row["mapping"],
                            "alignment_class": row["alignment_class"],
                            "step": step,
                            "phase": "generation",
                            "aic_query_path": "forced_fallback_decomposition",
                            "op_name": op_name,
                            "latency_ms": np.nan,
                            "per_layer_latency_ms": np.nan,
                            "step_count": float(row.get(step_count_col, 0.0) or 0.0),
                            "weighted_latency_ms": np.nan,
                            "source": f"error: {type(exc).__name__}: {exc}",
                            **shape,
                        }
                    )
    return pd.DataFrame(rows)


def expand_real_mla(joined: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    child_rows: list[dict[str, Any]] = []
    kernel_rows: list[dict[str, Any]] = []
    real_cases = joined.drop_duplicates("tag")
    for _, case in real_cases.iterrows():
        csv_path = Path(str(case["csv_path"]))
        detail = pd.read_csv(csv_path)
        for _, layer in detail.iterrows():
            base = {
                "tag": case["tag"],
                "alignment_class": case["alignment_class"],
                "csv_path": str(csv_path),
                "layer_id": int(layer.get("layer_id", -1)),
                "stage": layer.get("stage"),
                "stage_instance_index": int(layer.get("stage_instance_index", 0)),
                "attention_module": layer.get("attention_module"),
                "attention_token_count": layer.get("attention_token_count"),
                "module_gpu_makespan_ms": layer.get("module_gpu_makespan_ms"),
                "module_gpu_kernel_time_sum_ms": layer.get("module_gpu_kernel_time_sum_ms"),
            }
            children = safe_json_loads(layer.get("child_timing_json"), [])
            if isinstance(children, list):
                for child in children:
                    if not isinstance(child, dict):
                        continue
                    child_rows.append(
                        {
                            **base,
                            "real_component": child.get("canonical_name"),
                            "aic_mapping_hint": REAL_CHILD_TO_AIC_HINT.get(str(child.get("canonical_name")), ""),
                            "host_duration_ms": child.get("host_duration_ms"),
                            "total_to_last_kernel_ms": child.get("module_total_to_last_kernel_ms"),
                            "gpu_makespan_ms": child.get("gpu_makespan_ms"),
                            "gpu_kernel_time_sum_ms": child.get("gpu_kernel_time_sum_ms"),
                            "gpu_kernel_count": child.get("gpu_kernel_count"),
                            "first_kernel_name": child.get("first_kernel_name"),
                            "last_kernel_name": child.get("last_kernel_name"),
                        }
                    )
            kernels = safe_json_loads(layer.get("kernel_summary_json"), [])
            if isinstance(kernels, list):
                for component in kernels:
                    if not isinstance(component, dict):
                        continue
                    for kernel in component.get("top_gpu_kernels", []):
                        kernel_rows.append(
                            {
                                **base,
                                "real_component": component.get("canonical_name"),
                                "kernel_short_name": kernel.get("short_name"),
                                "kernel_gpu_time_ms": kernel.get("gpu_time_ms"),
                                "kernel_gpu_time_ns": kernel.get("gpu_time_ns"),
                                "component_kernel_count": component.get("gpu_kernel_count"),
                            }
                        )
    return pd.DataFrame(child_rows), pd.DataFrame(kernel_rows)


def summarize_real_children(real_child: pd.DataFrame) -> pd.DataFrame:
    if real_child.empty:
        return real_child
    return (
        real_child.groupby(["tag", "alignment_class", "real_component", "aic_mapping_hint"], dropna=False)
        .agg(
            layers=("layer_id", "nunique"),
            avg_gpu_makespan_ms=("gpu_makespan_ms", "mean"),
            nonfirst_avg_gpu_makespan_ms=(
                "gpu_makespan_ms",
                lambda s: float(s[real_child.loc[s.index, "layer_id"] != real_child.loc[s.index, "layer_id"].min()].mean()),
            ),
            sum_gpu_makespan_ms=("gpu_makespan_ms", "sum"),
            avg_gpu_kernel_time_sum_ms=("gpu_kernel_time_sum_ms", "mean"),
            sum_gpu_kernel_time_sum_ms=("gpu_kernel_time_sum_ms", "sum"),
            avg_host_duration_ms=("host_duration_ms", "mean"),
        )
        .reset_index()
    )


def summarize_aic(aic_long: pd.DataFrame) -> pd.DataFrame:
    if aic_long.empty:
        return aic_long
    return (
        aic_long.groupby(["tag", "mapping", "alignment_class", "step", "phase", "aic_query_path", "op_name"], dropna=False)
        .agg(
            latency_ms=("latency_ms", "sum"),
            per_layer_latency_ms=("per_layer_latency_ms", "sum"),
            weighted_latency_ms=("weighted_latency_ms", "sum"),
            step_count=("step_count", "first"),
            source=("source", "first"),
        )
        .reset_index()
    )


def make_real_case_label(row: pd.Series) -> str:
    try:
        return make_case_label(row, observed=False)
    except Exception:
        return str(row.get("tag", "case"))


def plot_total_comparison(joined: pd.DataFrame, aic_summary: pd.DataFrame, out_dir: Path) -> None:
    observed = joined[(joined["mapping"] == "observed_first_batch") & (joined["aic_status"] == "ok")].copy()
    if observed.empty:
        return
    block = aic_summary[
        aic_summary["mapping"].eq("observed_first_batch")
        & aic_summary["step"].eq("mix_step")
        & aic_summary["op_name"].eq("context_mla_block")
        & aic_summary["aic_query_path"].eq("actual_run_agg_block")
    ][["tag", "per_layer_latency_ms"]].rename(columns={"per_layer_latency_ms": "aic_context_mla_block_per_layer_ms"})
    fallback = (
        aic_summary[
            aic_summary["mapping"].eq("observed_first_batch")
            & aic_summary["step"].eq("mix_step")
            & aic_summary["phase"].eq("context")
            & aic_summary["aic_query_path"].eq("forced_fallback_decomposition")
        ]
        .groupby("tag")["per_layer_latency_ms"]
        .sum()
        .reset_index(name="aic_forced_fallback_context_mla_per_layer_ms")
    )
    plot_df = observed.merge(block, on="tag", how="left").merge(fallback, on="tag", how="left")
    plot_df["case_label"] = plot_df.apply(make_real_case_label, axis=1)
    long_df = plot_df.melt(
        id_vars=["case_label"],
        value_vars=[
            "real_nonfirst_layer_gpu_makespan_avg_ms",
            "aic_context_mla_block_per_layer_ms",
            "aic_forced_fallback_context_mla_per_layer_ms",
        ],
        var_name="metric",
        value_name="latency_ms",
    )
    labels = {
        "real_nonfirst_layer_gpu_makespan_avg_ms": "Real MLA non-first layer GPU makespan",
        "aic_context_mla_block_per_layer_ms": "AIC actual run_agg context_mla_block / layer",
        "aic_forced_fallback_context_mla_per_layer_ms": "AIC forced fallback MLA internals / layer",
    }
    long_df["metric"] = long_df["metric"].map(labels)
    x = np.arange(plot_df.shape[0])
    width = 0.26
    fig, ax = plt.subplots(figsize=(max(15.5, len(plot_df) * 1.25), 6.8))
    for idx, metric in enumerate(labels.values()):
        vals = long_df[long_df["metric"].eq(metric)]["latency_ms"].to_numpy(dtype=float)
        ax.bar(
            x + (idx - 1) * width,
            vals,
            width=width,
            label=metric,
            color=[COLORS["real"], COLORS["aic_block"], COLORS["aic_fallback"]][idx],
            edgecolor=COLORS["edge"],
            linewidth=0.7,
        )
    ax.set_xticks(x, plot_df["case_label"])
    ax.tick_params(axis="x", labelsize=6.4)
    ax.set_xlabel("Case")
    ax.set_ylabel("Single-layer latency (ms)")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), frameon=False, ncol=2, borderaxespad=0)
    add_chart_header(
        fig,
        ax,
        "MLA-only single-layer comparison excludes MoE and whole-layer effects",
        "Real value is Nsight non-first-layer MLA GPU makespan; AIC values are run_agg context_mla_block/layer and forced fallback sub-op sum/layer.",
    )
    save_chart(fig, out_dir, "mla_only_single_layer_total_comparison_observed")


def plot_real_child_breakdown(real_summary: pd.DataFrame, joined: pd.DataFrame, out_dir: Path) -> None:
    observed_tags = joined[joined["mapping"].eq("observed_first_batch")]["tag"].drop_duplicates()
    df = real_summary[real_summary["tag"].isin(observed_tags)].copy()
    if df.empty:
        return
    label_map = joined.drop_duplicates("tag").set_index("tag").apply(make_real_case_label, axis=1).to_dict()
    df["case_label"] = df["tag"].map(label_map)
    pivot = df.pivot_table(
        index="case_label",
        columns="real_component",
        values="nonfirst_avg_gpu_makespan_ms",
        aggfunc="sum",
        fill_value=0.0,
    )
    preferred = ["fused_qkv_a_proj_with_mqa", "q_a_layernorm", "kv_a_layernorm", "q_b_proj", "rotary_emb", "attn_mha", "attn_mqa", "o_proj"]
    columns = [c for c in preferred if c in pivot.columns] + [c for c in pivot.columns if c not in preferred]
    pivot = pivot[columns]
    fig, ax = plt.subplots(figsize=(13.8, max(6.2, len(pivot) * 0.48)))
    y = np.arange(len(pivot))
    left = np.zeros(len(pivot))
    for component in pivot.columns:
        vals = pivot[component].to_numpy(dtype=float)
        ax.barh(
            y,
            vals,
            left=left,
            label=component,
            color=COLORS.get(component, COLORS["other"]),
            edgecolor=COLORS["edge"],
            linewidth=0.45,
        )
        left += vals
    ax.set_yticks(y, pivot.index)
    ax.invert_yaxis()
    ax.set_xlabel("Non-first-layer average GPU makespan (ms)")
    ax.set_ylabel("Case")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), frameon=False, ncol=4, borderaxespad=0)
    add_chart_header(
        fig,
        ax,
        "Real Nsight MLA internal breakdown by child module",
        "Only self_attn child modules are included; MoE/FFN/embedding/logits are excluded.",
    )
    save_chart(fig, out_dir, "real_mla_child_breakdown_observed")


def plot_aic_fallback_breakdown(aic_summary: pd.DataFrame, joined: pd.DataFrame, out_dir: Path) -> None:
    df = aic_summary[
        aic_summary["mapping"].eq("observed_first_batch")
        & aic_summary["step"].eq("mix_step")
        & aic_summary["phase"].eq("context")
        & aic_summary["aic_query_path"].eq("forced_fallback_decomposition")
    ].copy()
    if df.empty:
        return
    label_map = joined.drop_duplicates("tag").set_index("tag").apply(make_real_case_label, axis=1).to_dict()
    df["case_label"] = df["tag"].map(label_map)
    pivot = df.pivot_table(index="case_label", columns="op_name", values="per_layer_latency_ms", aggfunc="sum", fill_value=0.0)
    preferred = [
        "context_downscale_gemm",
        "context_q_b_proj_gemm",
        "context_kv_b_proj_gemm",
        "context_mla_concat_k",
        "context_attention",
        "context_proj_gemm",
    ]
    columns = [c for c in preferred if c in pivot.columns] + [c for c in pivot.columns if c not in preferred]
    pivot = pivot[columns]
    fig, ax = plt.subplots(figsize=(13.8, max(6.2, len(pivot) * 0.48)))
    y = np.arange(len(pivot))
    left = np.zeros(len(pivot))
    for op_name in pivot.columns:
        vals = pivot[op_name].to_numpy(dtype=float)
        ax.barh(
            y,
            vals,
            left=left,
            label=op_name,
            color=COLORS.get(op_name, COLORS["other"]),
            edgecolor=COLORS["edge"],
            linewidth=0.45,
        )
        left += vals
    ax.set_yticks(y, pivot.index)
    ax.invert_yaxis()
    ax.set_xlabel("Forced fallback single-layer latency (ms)")
    ax.set_ylabel("Case")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), frameon=False, ncol=3, borderaxespad=0)
    add_chart_header(
        fig,
        ax,
        "AIC MLA forced fallback decomposition shows the modeled context sub-ops",
        "This is an attribution view from DeepSeekModel FallbackOp internals, not necessarily the actual module-table path used by run_agg.",
    )
    save_chart(fig, out_dir, "aic_forced_fallback_context_mla_breakdown_observed")


def write_report(out_dir: Path, joined: pd.DataFrame, real_summary: pd.DataFrame, aic_summary: pd.DataFrame) -> None:
    observed = joined[(joined["mapping"].eq("observed_first_batch")) & (joined["aic_status"].eq("ok"))].copy()
    actual_block = aic_summary[
        aic_summary["mapping"].eq("observed_first_batch")
        & aic_summary["step"].eq("mix_step")
        & aic_summary["op_name"].eq("context_mla_block")
        & aic_summary["aic_query_path"].eq("actual_run_agg_block")
    ].groupby("tag")["per_layer_latency_ms"].sum()
    fallback = aic_summary[
        aic_summary["mapping"].eq("observed_first_batch")
        & aic_summary["step"].eq("mix_step")
        & aic_summary["phase"].eq("context")
        & aic_summary["aic_query_path"].eq("forced_fallback_decomposition")
    ].groupby("tag")["per_layer_latency_ms"].sum()
    observed["aic_actual_context_mla_per_layer_ms"] = observed["tag"].map(actual_block)
    observed["aic_forced_fallback_context_mla_per_layer_ms"] = observed["tag"].map(fallback)
    observed["real_to_aic_actual_block_per_layer"] = (
        observed["real_nonfirst_layer_gpu_makespan_avg_ms"] / observed["aic_actual_context_mla_per_layer_ms"]
    )
    observed["real_to_aic_forced_fallback_per_layer"] = (
        observed["real_nonfirst_layer_gpu_makespan_avg_ms"] / observed["aic_forced_fallback_context_mla_per_layer_ms"]
    )
    observed[
        [
            "tag",
            "alignment_class",
            "real_nonfirst_layer_gpu_makespan_avg_ms",
            "aic_actual_context_mla_per_layer_ms",
            "aic_forced_fallback_context_mla_per_layer_ms",
            "real_to_aic_actual_block_per_layer",
            "real_to_aic_forced_fallback_per_layer",
        ]
    ].to_csv(out_dir / "mla_only_observed_single_layer_comparison.csv", index=False)

    report = f"""# run_agg MLA-only Breakdown

## 这次修正的口径

本目录只分析单层 MLA / self_attn，不再把 `context_moe`、shared FFN、embedding、logits 或整层端到端项纳入 breakdown。上一轮 `run_agg_ops_breakdown` 适合观察 run_agg 整体 step 构成，但粒度包含 MLA + MoE，因此不能回答“MLA 内部细粒度算子”这个问题。

## AIC 侧如何拆

- `actual_run_agg_block`：直接来自已有 `aic_per_ops_json`，例如 `mix_step.context_mla_block`、`genonly_step.generation_mla_block`。这是当前 `SGLANGBackend.run_agg()` 实际暴露的块级字段。
- `forced_fallback_decomposition`：旁路读取 `DeepSeekModel` 中 `context_mla_block` / `generation_mla_block` 的 `FallbackOp._fallback` 子 op 并逐个查询，用来解释 MLA block 背后可能由哪些子算子拼成。
- 由于 `FallbackOp` 命中 module 表时不会在 `run_static` summary 中自然展开，`forced_fallback_decomposition` 是归因视图，不等同于实际 run_agg 命中的数据路径。
- 所有对比以单层为单位：AIC op 的 latency 已除以 `num_layers`，实机侧使用非首层平均 `real_nonfirst_layer_gpu_makespan_avg_ms`，避免首层冷启动和端到端 scale 混入。

## 关键均值

- observed case 数：{len(observed)}
- 平均 real / AIC actual context_mla_block per-layer：{observed["real_to_aic_actual_block_per_layer"].mean():.3f}x
- 平均 real / AIC forced fallback context MLA per-layer：{observed["real_to_aic_forced_fallback_per_layer"].mean():.3f}x

## 产物

- `real_mla_child_long.csv`：实机 Nsight 每个 layer 的 MLA child module 细分。
- `real_mla_kernel_long.csv`：实机 child module 下的 top CUDA kernel 细分。
- `real_mla_child_by_case.csv`：实机 child module 按 case 聚合。
- `aic_mla_only_breakdown_long.csv`：AIC actual block 与 forced fallback 的逐 op 长表。
- `aic_mla_only_by_component.csv`：AIC MLA-only 逐组件聚合表。
- `mla_only_observed_single_layer_comparison.csv`：observed mapping 下单层总量对比。
- `mla_only_single_layer_total_comparison_observed.png/svg`：单层 MLA 总量对比。
- `real_mla_child_breakdown_observed.png/svg`：实机 MLA 内部 child module 堆积。
- `aic_forced_fallback_context_mla_breakdown_observed.png/svg`：AIC context MLA forced fallback 子 op 堆积。

## 结论性提示

当前 `run_agg` 的实际输出不适合直接做“MLA 内部子算子”breakdown，因为它在 `run_static` summary 中保留的是 `context_mla_block` 这种外层 FallbackOp 名。要比较 MLA 内部，必须像本脚本一样显式展开模型 op，或者在 SDK 中为 `FallbackOp` 增加可选的 inner-op provenance 输出。
"""
    (out_dir / "run_agg_mla_only_breakdown_report.md").write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    from compare_agg_run_agg import parse_args as parse_compare_args

    defaults = parse_compare_args([])
    parser = argparse.ArgumentParser()
    parser.add_argument("--joined", type=Path, default=DEFAULT_JOINED)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--model-path", default=defaults.model_path)
    parser.add_argument("--system", default=defaults.system)
    parser.add_argument("--backend", default=defaults.backend)
    parser.add_argument("--version", default=defaults.version)
    parser.add_argument("--database-mode", default=defaults.database_mode)
    parser.add_argument("--tp-size", type=int, default=defaults.tp_size)
    parser.add_argument("--pp-size", type=int, default=defaults.pp_size)
    parser.add_argument("--attention-dp-size", type=int, default=defaults.attention_dp_size)
    parser.add_argument("--moe-tp-size", type=int, default=defaults.moe_tp_size)
    parser.add_argument("--moe-ep-size", type=int, default=defaults.moe_ep_size)
    parser.add_argument("--num-layers", type=int, default=defaults.num_layers)
    parser.add_argument("--gemm-quant-mode", default=defaults.gemm_quant_mode)
    parser.add_argument("--moe-quant-mode", default=defaults.moe_quant_mode)
    parser.add_argument("--kvcache-quant-mode", default=defaults.kvcache_quant_mode)
    parser.add_argument("--fmha-quant-mode", default=defaults.fmha_quant_mode)
    parser.add_argument("--comm-quant-mode", default=defaults.comm_quant_mode)
    parser.add_argument("--attention-backend", default=defaults.attention_backend)
    parser.add_argument("--moe-backend", default=defaults.moe_backend)
    parser.add_argument("--enable-wideep", action="store_true", default=defaults.enable_wideep)
    parser.add_argument("--engine-step-backend", default=defaults.engine_step_backend)
    parser.add_argument("--osl", type=int, default=defaults.osl)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    use_chart_theme()

    joined = pd.read_csv(args.joined)
    session = build_session(args)
    real_child, real_kernel = expand_real_mla(joined)
    real_summary = summarize_real_children(real_child)
    aic_long = expand_aic_mla(joined, session, args.num_layers)
    aic_summary = summarize_aic(aic_long)

    real_child.to_csv(args.out_dir / "real_mla_child_long.csv", index=False)
    real_kernel.to_csv(args.out_dir / "real_mla_kernel_long.csv", index=False)
    real_summary.to_csv(args.out_dir / "real_mla_child_by_case.csv", index=False)
    aic_long.to_csv(args.out_dir / "aic_mla_only_breakdown_long.csv", index=False)
    aic_summary.to_csv(args.out_dir / "aic_mla_only_by_component.csv", index=False)

    plot_total_comparison(joined, aic_summary, args.out_dir)
    plot_real_child_breakdown(real_summary, joined, args.out_dir)
    plot_aic_fallback_breakdown(aic_summary, joined, args.out_dir)
    write_report(args.out_dir, joined, real_summary, aic_summary)

    print(f"Wrote MLA-only run_agg breakdown to {args.out_dir}")
    print(f"real child rows={len(real_child)} real kernel rows={len(real_kernel)}")
    print(f"aic rows={len(aic_long)} aic summary rows={len(aic_summary)}")


if __name__ == "__main__":
    main()
