#!/usr/bin/env python3
"""Expand AIC run_agg per-op data into case-level breakdown tables and charts."""

from __future__ import annotations

import argparse
import json
import math
import textwrap
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_AIC_RESULTS = SCRIPT_DIR / "aic_run_agg_results.csv"
DEFAULT_JOINED = SCRIPT_DIR / "real_aic_joined.csv"
DEFAULT_OUT_DIR = SCRIPT_DIR / "run_agg_ops_breakdown"

TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
}

COLORS = {
    "context_mla_block": "#A3BEFA",
    "context_moe": "#F0986E",
    "context_shared_ffn": "#A3D576",
    "context_other": "#E2E5EA",
    "generation_mla_block": "#FFE15B",
    "generation_moe": "#F390CA",
    "generation_other": "#CEDFFE",
    "attention_scaled": "#BEEB96",
    "other": "#C5CAD3",
}


def use_chart_theme() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": TOKENS["surface"],
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
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_chart(fig, out_dir: Path, stem: str) -> None:
    fig.savefig(out_dir / f"{stem}.png", dpi=180, bbox_inches="tight", facecolor=TOKENS["panel"], transparent=False)
    fig.savefig(out_dir / f"{stem}.svg", bbox_inches="tight", facecolor=TOKENS["panel"], transparent=False)
    plt.close(fig)


def safe_json_loads(value: Any) -> dict[str, Any]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return {}
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def op_category(step: str, op_name: str) -> str:
    if step == "mix_step":
        if op_name == "context_mla_block":
            return "context_mla_block"
        if op_name == "context_moe":
            return "context_moe"
        if op_name in {"context_shared_gate_up_gemm", "context_shared_ffn2_gemm", "context_shared_act_gate"}:
            return "context_shared_ffn"
        if "attention" in op_name:
            return "attention_scaled"
        if op_name.startswith("generation_"):
            if "mla" in op_name or "attention" in op_name:
                return "generation_mla_block"
            if "moe" in op_name:
                return "generation_moe"
            return "generation_other"
        if op_name.startswith("context_"):
            return "context_other"
    if step == "genonly_step":
        if op_name == "generation_mla_block":
            return "generation_mla_block"
        if "moe" in op_name:
            return "generation_moe"
        return "generation_other"
    return "other"


def make_case_label(row: pd.Series, *, observed: bool) -> str:
    if observed:
        p_b = int(row.get("first_formal_prefill_req_count", 0))
        d_b = int(row.get("first_formal_decode_req_count", 0))
        ctx = int(row.get("first_formal_sum_prefill_fresh_tokens", 0))
        d_kv_total = int(row.get("first_formal_sum_decode_kv_tokens", 0))
        d_kv = int(round(d_kv_total / d_b)) if d_b > 0 else int(row.get("decode_prefix_len", 0))
        return f"Obs P b{p_b} ctx{ctx}\nisl{int(row.get('aic_isl', 0))} pre{int(row.get('aic_prefix', 0))}\nD b{d_b} kv{d_kv}"

    p_b = int(row.get("prefill_batch_size", 0))
    p_fresh = int(row.get("prefill_fresh_len", 0))
    prefix = int(row.get("prefill_prefix_len", 0))
    p_isl = p_fresh + prefix
    d_b = int(row.get("decode_batch_size", 0))
    d_kv = int(row.get("decode_prefix_len", 0))
    return f"P b{p_b} ctx{p_b * p_isl}\nisl{p_isl} pre{prefix}\nD b{d_b} kv{d_kv}"


def expand_ops(aic: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in aic.iterrows():
        per_ops = safe_json_loads(row.get("aic_per_ops_json"))
        step_counts = {
            "mix_step": float(row.get("aic_num_mix_steps", 0.0) or 0.0),
            "genonly_step": float(row.get("aic_num_genonly_steps", 0.0) or 0.0),
        }
        total_weighted = 0.0
        flat: list[dict[str, Any]] = []
        for step_name in ("mix_step", "genonly_step"):
            op_map = per_ops.get(step_name, {})
            if not isinstance(op_map, dict):
                continue
            step_latency = sum(float(v) for v in op_map.values())
            for op_name, latency in op_map.items():
                latency = float(latency)
                weighted = latency * step_counts[step_name]
                total_weighted += weighted
                flat.append(
                    {
                        "tag": row["tag"],
                        "mapping": row["mapping"],
                        "alignment_class": row["alignment_class"],
                        "step": step_name,
                        "op_name": op_name,
                        "op_category": op_category(step_name, op_name),
                        "latency_ms": latency,
                        "step_latency_ms": step_latency,
                        "step_count": step_counts[step_name],
                        "weighted_latency_ms": weighted,
                        "aic_mix_step_latency_ms": row.get("aic_mix_step_latency_ms"),
                        "aic_genonly_step_latency_ms": row.get("aic_genonly_step_latency_ms"),
                        "aic_ttft": row.get("aic_ttft"),
                        "aic_tpot": row.get("aic_tpot"),
                        "aic_request_latency": row.get("aic_request_latency"),
                        "decode_batch_size": row.get("decode_batch_size"),
                        "decode_prefix_len": row.get("decode_prefix_len"),
                        "prefill_batch_size": row.get("prefill_batch_size"),
                        "prefill_fresh_len": row.get("prefill_fresh_len"),
                        "prefill_prefix_len": row.get("prefill_prefix_len"),
                        "first_formal_prefill_req_count": row.get("first_formal_prefill_req_count"),
                        "first_formal_decode_req_count": row.get("first_formal_decode_req_count"),
                        "first_formal_sum_prefill_fresh_tokens": row.get("first_formal_sum_prefill_fresh_tokens"),
                        "first_formal_sum_decode_kv_tokens": row.get("first_formal_sum_decode_kv_tokens"),
                        "aic_isl": row.get("aic_isl"),
                        "aic_prefix": row.get("aic_prefix"),
                    }
                )
        for item in flat:
            item["pct_of_weighted_total"] = item["weighted_latency_ms"] / total_weighted if total_weighted else 0.0
            item["pct_of_step"] = item["latency_ms"] / item["step_latency_ms"] if item["step_latency_ms"] else 0.0
            rows.append(item)
    return pd.DataFrame(rows)


def summarize_categories(breakdown: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["tag", "mapping", "alignment_class", "step", "op_category"]
    keep_cols = [
        "aic_mix_step_latency_ms",
        "aic_genonly_step_latency_ms",
        "aic_ttft",
        "aic_tpot",
        "aic_request_latency",
        "decode_batch_size",
        "decode_prefix_len",
        "prefill_batch_size",
        "prefill_fresh_len",
        "prefill_prefix_len",
        "first_formal_prefill_req_count",
        "first_formal_decode_req_count",
        "first_formal_sum_prefill_fresh_tokens",
        "first_formal_sum_decode_kv_tokens",
        "aic_isl",
        "aic_prefix",
    ]
    firsts = {col: (col, "first") for col in keep_cols if col in breakdown.columns}
    out = (
        breakdown.groupby(group_cols, dropna=False)
        .agg(
            latency_ms=("latency_ms", "sum"),
            step_latency_ms=("step_latency_ms", "first"),
            step_count=("step_count", "first"),
            weighted_latency_ms=("weighted_latency_ms", "sum"),
            **firsts,
        )
        .reset_index()
    )
    total_step = out.groupby(["tag", "mapping", "step"])["latency_ms"].transform("sum")
    total_weighted = out.groupby(["tag", "mapping"])["weighted_latency_ms"].transform("sum")
    out["pct_of_step"] = np.where(total_step > 0, out["latency_ms"] / total_step, 0.0)
    out["pct_of_weighted_total"] = np.where(total_weighted > 0, out["weighted_latency_ms"] / total_weighted, 0.0)
    return out


def pivot_for_plot(summary: pd.DataFrame, *, mapping: str, value_col: str, step: str | None = None) -> pd.DataFrame:
    df = summary[summary["mapping"].eq(mapping)].copy()
    if step is not None:
        df = df[df["step"].eq(step)].copy()
    if df.empty:
        return df
    df["case_label"] = df.apply(lambda row: make_case_label(row, observed=mapping == "observed_first_batch"), axis=1)
    return (
        df.pivot_table(index="case_label", columns="op_category", values=value_col, aggfunc="sum", fill_value=0.0)
        .reset_index()
    )


def stacked_bar(df: pd.DataFrame, out_dir: Path, stem: str, title: str, subtitle: str, xlabel: str) -> None:
    if df.empty:
        return
    category_order = [
        "context_mla_block",
        "attention_scaled",
        "context_moe",
        "context_shared_ffn",
        "context_other",
        "generation_mla_block",
        "generation_moe",
        "generation_other",
    ]
    categories = [c for c in category_order if c in df.columns and df[c].sum() > 0]
    labels = list(df["case_label"])
    y = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(13.8, max(6.2, len(df) * 0.48)))
    left = np.zeros(len(df))
    for category in categories:
        values = df[category].to_numpy(dtype=float)
        ax.barh(
            y,
            values,
            left=left,
            label=category,
            color=COLORS.get(category, COLORS["other"]),
            edgecolor="#464C55",
            linewidth=0.45,
        )
        left += values
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Case")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), frameon=False, ncol=min(4, len(categories)), borderaxespad=0)
    add_chart_header(fig, ax, title, subtitle)
    save_chart(fig, out_dir, stem)


def write_report(out_dir: Path, breakdown: pd.DataFrame, category_summary: pd.DataFrame) -> None:
    observed_mix = category_summary[
        category_summary["mapping"].eq("observed_first_batch") & category_summary["step"].eq("mix_step")
    ]
    by_category = (
        observed_mix.groupby("op_category")
        .agg(avg_latency_ms=("latency_ms", "mean"), avg_pct_of_step=("pct_of_step", "mean"))
        .sort_values("avg_latency_ms", ascending=False)
        .reset_index()
    )
    by_category.to_csv(out_dir / "observed_mix_step_category_summary.csv", index=False)
    top_rows = "\n".join(
        f"- `{row.op_category}`: avg {row.avg_latency_ms:.3f} ms, avg share {row.avg_pct_of_step:.1%}"
        for row in by_category.itertuples(index=False)
    )
    report = f"""# run_agg Ops Breakdown 与 WideEP Attention 字段核查

## 产物

- `run_agg_ops_breakdown_long.csv`：从 `aic_per_ops_json` 展开的逐 op 长表。
- `run_agg_ops_breakdown_by_category.csv`：按 `op_category` 聚合后的 per-step 与 weighted-total 贡献。
- `observed_mix_step_category_summary.csv`：`observed_first_batch` 映射下 mixed step 的类别平均贡献。
- `observed_mix_step_ops_breakdown.png/svg`：每个 observed mixed step 的类别堆积。
- `observed_weighted_total_ops_breakdown.png/svg`：每个 observed case 在 run_agg 总累加时延中的类别堆积。
- `intended_weighted_total_ops_breakdown.png/svg`：每个 intended case 在 run_agg 总累加时延中的类别堆积。

## observed_first_batch mixed step 平均组成

{top_rows}

## WideEP 字段语义核查结论

确认存在语义风险：`SGLANGBackend.run_agg()` 通过硬编码字段名 `context_attention` / `generation_attention` 将 attention 从 `run_static` 结果中单独取出并做缩放/混合处理；但 `WideEPDeepSeekModel` 中这两个名字对应的实际 op 类并不是普通 attention 单算子，而是：

- `context_attention` -> `ops.WideEPContextMLA`
- `generation_attention` -> `ops.WideEPGenerationMLA`

因此代码运行层面通常不会 KeyError，因为字段名仍然存在；但语义层面会把 WideEP MLA module 级查询结果当作 “attention 单算子” 处理。对于当前非 WideEP 的这批对比，AIC 结果里主要表现为 `context_mla_block` 已在 first pass 中被当作 non-attention 累加，而 `context_attention (scaled)` 为 0；若切到 SGLang DeepEP/WideEP 模型，`context_attention (scaled)` 将实际代表 `wideep_context_mla_perf.txt` 的模块级 WideEP MLA，而不是普通 FA/attention kernel。

代码证据：

- `src/aiconfigurator/sdk/backends/sglang_backend.py:147-177`：first pass 只排除 `layer_name != "context_attention"`，second pass 直接读取 `latency_dict["context_attention"]` 并按 `scale_factor` 折算。
- `src/aiconfigurator/sdk/backends/sglang_backend.py:194-198`：decode attention 也直接读取 `latency_dict["generation_attention"]`。
- `src/aiconfigurator/sdk/models/deepseek.py:1078-1085`：`WideEPDeepSeekModel` 将 `ops.WideEPContextMLA` 命名为 `"context_attention"`。
- `src/aiconfigurator/sdk/models/deepseek.py:1209-1216`：`WideEPDeepSeekModel` 将 `ops.WideEPGenerationMLA` 命名为 `"generation_attention"`。
- `src/aiconfigurator/sdk/operations.py:1269-1276` 与 `src/aiconfigurator/sdk/operations.py:1315-1323`：这两个 WideEP op 实际查询的是 `query_wideep_generation_mla` / `query_wideep_context_mla`，即 WideEP MLA 表，而不是普通 attention 表。

建议后续修复方向不是简单改字段名，而是让 `run_agg` 显式从模型/op 类型判断 “需要特殊折算的 prefill attention 项”：

- 普通 attention：`ContextAttention` / `GenerationAttention`
- DeepSeek MLA 单算子或模块：`ContextMLA` / `GenerationMLA` / `MLAModule`
- WideEP module：`WideEPContextMLA` / `WideEPGenerationMLA`

然后在 `per_ops_data` 中使用语义化字段，例如 `context_prefill_attn_like_scaled`，同时保留原始 op 名和 op class，避免 `context_attention` 这个名字在不同模型中承担不同含义。
"""
    (out_dir / "run_agg_ops_breakdown_report.md").write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aic-results", type=Path, default=DEFAULT_AIC_RESULTS)
    parser.add_argument("--joined", type=Path, default=DEFAULT_JOINED)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    use_chart_theme()

    aic = pd.read_csv(args.aic_results)
    breakdown = expand_ops(aic)
    category_summary = summarize_categories(breakdown)
    breakdown.to_csv(args.out_dir / "run_agg_ops_breakdown_long.csv", index=False)
    category_summary.to_csv(args.out_dir / "run_agg_ops_breakdown_by_category.csv", index=False)

    observed_mix = pivot_for_plot(
        category_summary,
        mapping="observed_first_batch",
        step="mix_step",
        value_col="latency_ms",
    )
    stacked_bar(
        observed_mix,
        args.out_dir,
        "observed_mix_step_ops_breakdown",
        "AIC run_agg observed mixed step is dominated by MoE and MLA-like blocks",
        "Per-step latency from run_agg per_ops_data, using observed-first-batch mapping. Values are ms within one mixed step.",
        "Per mixed step latency (ms)",
    )

    observed_total = pivot_for_plot(
        category_summary,
        mapping="observed_first_batch",
        value_col="weighted_latency_ms",
    )
    stacked_bar(
        observed_total,
        args.out_dir,
        "observed_weighted_total_ops_breakdown",
        "Weighted run_agg total latency composition under observed-first-batch mapping",
        "Each op category is multiplied by run_agg's scheduled step count, then stacked by case.",
        "Weighted total latency contribution (ms)",
    )

    intended_total = pivot_for_plot(
        category_summary,
        mapping="intended_case",
        value_col="weighted_latency_ms",
    )
    stacked_bar(
        intended_total,
        args.out_dir,
        "intended_weighted_total_ops_breakdown",
        "Weighted run_agg total latency composition under intended case mapping",
        "Uses case design parameters rather than the first formal mixed batch observed in Nsight.",
        "Weighted total latency contribution (ms)",
    )

    write_report(args.out_dir, breakdown, category_summary)
    print(f"Wrote run_agg ops breakdown to {args.out_dir}")
    print(f"breakdown rows={len(breakdown)} category rows={len(category_summary)}")


if __name__ == "__main__":
    main()
