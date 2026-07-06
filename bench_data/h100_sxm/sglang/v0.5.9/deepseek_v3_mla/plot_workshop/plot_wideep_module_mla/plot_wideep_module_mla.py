#!/usr/bin/env python3
"""Plot real-vs-AIC comparisons for WideEP module-level MLA timing.

This script generates plots for 6 real-time metrics:
1. Module total to last kernel
2. Module GPU makespan
3. Module GPU kernel time sum
4. WideEP-aligned GPU makespan
5. WideEP-aligned GPU kernel time sum
6. WideEP-aligned host duration sum

For each metric and stage (prefill/decode), generates a boxplot with real data,
AIC reference line, and percent-error annotations.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MetricConfig:
    real_column: str
    title_suffix: str
    output_suffix: str


METRICS = [
    MetricConfig(
        real_column="real_module_total_to_last_kernel_ms",
        title_suffix="Module Total to Last Kernel",
        output_suffix="module_total_to_last_kernel",
    ),
    MetricConfig(
        real_column="real_module_gpu_makespan_ms",
        title_suffix="Module GPU Makespan",
        output_suffix="module_gpu_makespan",
    ),
    MetricConfig(
        real_column="real_module_gpu_kernel_time_sum_ms",
        title_suffix="Module GPU Kernel Time Sum",
        output_suffix="module_gpu_kernel_time_sum",
    ),
    MetricConfig(
        real_column="real_wideep_aligned_gpu_makespan_ms",
        title_suffix="WideEP-Aligned GPU Makespan",
        output_suffix="wideep_aligned_gpu_makespan",
    ),
    MetricConfig(
        real_column="real_wideep_aligned_gpu_kernel_time_sum_ms",
        title_suffix="WideEP-Aligned GPU Kernel Time Sum",
        output_suffix="wideep_aligned_gpu_kernel_time_sum",
    ),
    MetricConfig(
        real_column="real_wideep_aligned_host_duration_sum_ms",
        title_suffix="WideEP-Aligned Host Duration Sum",
        output_suffix="wideep_aligned_host_duration_sum",
    ),
]


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comparison-csv",
        default=Path(__file__).resolve().parents[2] / "analysis" / "wideep_module_mla" / "wideep_module_mla_aic_compare__comparison.csv",
        type=Path,
        help="Path to wideep_module_mla comparison CSV",
    )
    parser.add_argument(
        "--output-dir",
        default=Path(__file__).resolve().parent,
        type=Path,
        help="Directory to store figures and summary tables.",
    )
    parser.add_argument(
        "--stage",
        choices=["prefill", "decode", "all"],
        default="prefill",
        help="Which stage rows to plot.",
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Prefix for output files. Defaults to the selected stage.",
    )
    return parser.parse_args(argv)


def _case_key(df: pd.DataFrame) -> pd.Series:
    return (
        df["tag"].astype(str)
        + "|"
        + df["stage"].astype(str)
        + "|b"
        + df["batch_size"].astype(int).astype(str)
        + "|f"
        + df["fresh_len"].astype(str)
        + "|p"
        + df["prefix_len"].astype(str)
    )


def _display_label(row: pd.Series) -> str:
    return f"b{int(row['batch_size'])}\nf{row['fresh_len']}\np{int(row['prefix_len'])}"


def load_stage_data(path: Path, stage: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if stage != "all":
        df = df[df["stage"] == stage].copy()
    if df.empty:
        raise ValueError(f"No rows left after filtering: {path}")

    for column in [
        "real_module_total_to_last_kernel_ms",
        "real_module_gpu_makespan_ms",
        "real_module_gpu_kernel_time_sum_ms",
        "real_wideep_aligned_gpu_makespan_ms",
        "real_wideep_aligned_gpu_kernel_time_sum_ms",
        "real_wideep_aligned_host_duration_sum_ms",
        "aic_wideep_mla_latency_ms",
        "batch_size",
        "fresh_len",
        "prefix_len",
    ]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=["aic_wideep_mla_latency_ms", "batch_size", "fresh_len", "prefix_len"])
    df["case_key"] = _case_key(df)
    df["display_label"] = df.apply(_display_label, axis=1)
    return df


def aggregate_case_statistics(df: pd.DataFrame, real_column: str) -> pd.DataFrame:
    grouped = df.groupby(["case_key", "display_label", "batch_size", "fresh_len", "prefix_len"], as_index=False)
    summary = grouped.agg(
        real_mean_ms=(real_column, "mean"),
        real_std_ms=(real_column, "std"),
        real_count=(real_column, "count"),
        aic_mean_ms=("aic_wideep_mla_latency_ms", "mean"),
        aic_std_ms=("aic_wideep_mla_latency_ms", "std"),
        aic_count=("aic_wideep_mla_latency_ms", "count"),
    )
    summary["real_std_ms"] = summary["real_std_ms"].fillna(0.0)
    summary["aic_std_ms"] = summary["aic_std_ms"].fillna(0.0)
    summary["pct_increase_vs_aic_mean"] = (
        (summary["real_mean_ms"] - summary["aic_mean_ms"]) / summary["aic_mean_ms"] * 100.0
    )
    return summary.sort_values(["batch_size", "fresh_len", "prefix_len", "case_key"]).reset_index(drop=True)


def _jittered_points(center: float, count: int, scale: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return center + rng.uniform(-scale, scale, size=count)


def plot_comparison(df: pd.DataFrame, real_column: str, title: str, output_path: Path) -> None:
    summary = aggregate_case_statistics(df, real_column)
    case_keys = list(summary["case_key"])
    labels = list(summary["display_label"])
    x_positions = np.arange(len(case_keys), dtype=float)
    real_offset = -0.18
    aic_offset = 0.20
    width = 0.30

    fig_width = max(14, len(case_keys) * 0.62)
    fig, ax = plt.subplots(figsize=(fig_width, 7.6), dpi=210)

    real_values = []
    aic_values = []
    for case_key in case_keys:
        case_df = df[df["case_key"] == case_key]
        real_values.append(case_df[real_column].to_numpy(dtype=float))
        aic_values.append(case_df["aic_wideep_mla_latency_ms"].to_numpy(dtype=float))

    all_values = [values for values in real_values if len(values)] + [values for values in aic_values if len(values)]

    real_pos = x_positions + real_offset
    real_box = ax.boxplot(
        real_values,
        positions=real_pos,
        widths=width,
        patch_artist=True,
        showfliers=False,
        whis=(5, 95),
        medianprops={"color": "#222222", "linewidth": 1.8},
        boxprops={"linewidth": 1.4, "edgecolor": "#1f77b4"},
        whiskerprops={"linewidth": 1.2, "color": "#1f77b4"},
        capprops={"linewidth": 1.2, "color": "#1f77b4"},
    )
    for patch in real_box["boxes"]:
        patch.set_facecolor("#1f77b4")
        patch.set_alpha(0.18)

    real_means = np.array([np.mean(values) for values in real_values], dtype=float)
    real_stds = np.array([np.std(values, ddof=1) if len(values) > 1 else 0.0 for values in real_values], dtype=float)
    ax.errorbar(
        real_pos,
        real_means,
        yerr=real_stds,
        fmt="o",
        color="#1f77b4",
        ecolor="#1f77b4",
        elinewidth=1.3,
        capsize=4,
        markersize=5,
        markeredgecolor="white",
        markeredgewidth=0.8,
        zorder=4,
        label="Real mean ± std",
    )
    for idx, values in enumerate(real_values):
        if len(values) == 0:
            continue
        points_x = _jittered_points(real_pos[idx], len(values), scale=0.045, seed=idx)
        ax.scatter(points_x, values, s=16, color="#1f77b4", alpha=0.42, linewidths=0, zorder=3)

    aic_means = np.array([np.mean(values) for values in aic_values], dtype=float)
    aic_stds = np.array([np.std(values, ddof=1) if len(values) > 1 else 0.0 for values in aic_values], dtype=float)
    aic_pos = x_positions + aic_offset
    ax.errorbar(
        aic_pos,
        aic_means,
        yerr=aic_stds,
        fmt="s",
        color="#d62728",
        ecolor="#d62728",
        elinewidth=1.3,
        capsize=4,
        markersize=5,
        markeredgecolor="white",
        markeredgewidth=0.8,
        zorder=5,
        label="AIC mean ± std",
    )
    for idx, values in enumerate(aic_values):
        if len(values) == 0:
            continue
        points_x = _jittered_points(aic_pos[idx], len(values), scale=0.02, seed=idx + 5000)
        ax.scatter(points_x, values, s=12, color="#d62728", alpha=0.55, marker="x", linewidths=0.8, zorder=4)
        ax.hlines(np.mean(values), real_pos[idx] - 0.10, aic_pos[idx] + 0.10, colors="#d62728", linestyles="dashed", linewidth=1.0, alpha=0.85)

    y_span = max(max(v) for v in all_values) - min(min(v) for v in all_values)
    label_offset = max(y_span * 0.03, max(max(v) for v in all_values) * 0.012, 0.0008)
    top = max(max(v) for v in all_values) * 1.12
    for idx, row in summary.iterrows():
        real_mean = float(row["real_mean_ms"])
        upper = max(np.max(real_values[idx]), np.max(aic_values[idx]))
        lower = min(np.min(real_values[idx]), np.min(aic_values[idx]))
        if real_mean + label_offset <= top * 0.97:
            label_y = real_mean + label_offset
            valign = "bottom"
        else:
            label_y = max(lower - label_offset, 0.0)
            valign = "top"
        pct = row['pct_increase_vs_aic_mean']
        sign = "+" if pct >= 0 else ""
        ax.text(
            x_positions[idx],
            label_y,
            f"{sign}{pct:.1f}%",
            ha="center",
            va=valign,
            fontsize=9,
            color="#7f1d1d",
            fontweight="bold",
            clip_on=True,
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Latency (ms)", fontsize=11)
    ax.set_xlabel("Case (batch / fresh / prefix)", fontsize=11)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=True, fontsize=9)

    ax.set_ylim(bottom=0, top=top)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = load_stage_data(args.comparison_csv, stage=args.stage)
    output_prefix = args.output_prefix or args.stage

    summary_paths = []
    for metric in METRICS:
        plot_df = df.dropna(subset=[metric.real_column]).copy()
        if plot_df.empty:
            print(f"Warning: No rows available for {metric.output_suffix}, skipping")
            continue
        
        title = f"{metric.title_suffix} (WideEP vs Real)"
        output_path = args.output_dir / f"{output_prefix}__{metric.output_suffix}__real_vs_aic.png"
        summary_path = args.output_dir / f"{output_prefix}__{metric.output_suffix}__summary.csv"
        
        plot_comparison(plot_df, metric.real_column, title, output_path)
        aggregate_case_statistics(plot_df, metric.real_column).to_csv(summary_path, index=False)
        summary_paths.append(str(summary_path))
        print(f"Wrote plot to {output_path}")
        print(f"Wrote summary to {summary_path}")

    print(f"Loaded {len(df)} {args.stage} rows from {args.comparison_csv}")
    print(f"Generated {len(summary_paths)} plots")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
