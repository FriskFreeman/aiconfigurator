#!/usr/bin/env python3
"""Plot real-vs-AIC comparisons for MLA kernel timing.

This script only uses prefill rows by default. It generates four figures:
- MLA / bfloat16
- MLA / fp8
- Attention / bfloat16
- Attention / fp8

Each case is one x-axis category. The real measurements are shown as boxplots
with raw data points and mean +/- std error bars. The AIC value is shown as a
horizontal reference line per case. A percent label annotates the mean real
latency increase versus the mean AIC latency.
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
class PlotConfig:
    name: str
    sim_column: str
    title: str
    output_suffix: str


PLOT_CONFIGS = [
    PlotConfig(
        name="mla_bfloat16",
        sim_column="aic_mla_bfloat16_latency_ms",
        title="MLA vs Real Attention Kernel Time (bf16)",
        output_suffix="mla_bfloat16",
    ),
    PlotConfig(
        name="mla_fp8",
        sim_column="aic_mla_fp8_latency_ms",
        title="MLA vs Real Attention Kernel Time (fp8)",
        output_suffix="mla_fp8",
    ),
    PlotConfig(
        name="attention_bfloat16",
        sim_column="aic_attention_bfloat16_latency_ms",
        title="Attention vs Real Attention Kernel Time (bf16)",
        output_suffix="attention_bfloat16",
    ),
    PlotConfig(
        name="attention_fp8",
        sim_column="aic_attention_fp8_latency_ms",
        title="Attention vs Real Attention Kernel Time (fp8)",
        output_suffix="attention_fp8",
    ),
]


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comparison-csv",
        default=Path(__file__).with_name("mla_kernel_aic_compare__comparison.csv"),
        type=Path,
        help="Path to the comparison CSV produced by compare_mla_kernel_with_aic.py",
    )
    parser.add_argument(
        "--output-dir",
        default=Path(__file__).with_name("plots"),
        type=Path,
        help="Directory to store figures and summary tables.",
    )
    parser.add_argument(
        "--prefill-only",
        action="store_true",
        default=True,
        help="Keep only prefill rows. Enabled by default to avoid mixing decode rows.",
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
    prefix = f"p{int(row['prefix_len'])}"
    family = str(row["tag"]).split("_")[3] if "_" in str(row["tag"]) else "case"
    return f"b{int(row['batch_size'])}\nf{row['fresh_len']}\n{prefix}\n{family}"


def load_prefill_data(path: Path, prefill_only: bool = True) -> pd.DataFrame:
    df = pd.read_csv(path)
    if prefill_only:
        df = df[df["stage"] == "prefill"].copy()
    if df.empty:
        raise ValueError(f"No rows left after filtering: {path}")

    for column in [
        "real_attention_kernel_time_ms",
        "real_gpu_makespan_ms",
        "real_gpu_kernel_time_sum_ms",
        "batch_size",
        "fresh_len",
        "prefix_len",
    ]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=["real_attention_kernel_time_ms", "batch_size", "fresh_len", "prefix_len"])
    df["case_key"] = _case_key(df)
    df["display_label"] = df.apply(_display_label, axis=1)
    return df


def aggregate_case_statistics(df: pd.DataFrame, sim_column: str) -> pd.DataFrame:
    grouped = df.groupby(["case_key", "display_label", "batch_size", "fresh_len", "prefix_len"], as_index=False)
    summary = grouped.agg(
        real_mean_ms=("real_attention_kernel_time_ms", "mean"),
        real_std_ms=("real_attention_kernel_time_ms", "std"),
        real_count=("real_attention_kernel_time_ms", "count"),
        real_makespan_mean_ms=("real_gpu_makespan_ms", "mean"),
        real_kernel_sum_mean_ms=("real_gpu_kernel_time_sum_ms", "mean"),
        sim_mean_ms=(sim_column, "mean"),
        sim_std_ms=(sim_column, "std"),
        sim_count=(sim_column, "count"),
    )
    summary["real_std_ms"] = summary["real_std_ms"].fillna(0.0)
    summary["sim_std_ms"] = summary["sim_std_ms"].fillna(0.0)
    summary["pct_increase_vs_sim_mean"] = (
        (summary["real_mean_ms"] - summary["sim_mean_ms"]) / summary["sim_mean_ms"] * 100.0
    )
    summary["pct_real_vs_kernel_sum"] = (
        (summary["real_mean_ms"] - summary["real_kernel_sum_mean_ms"])
        / summary["real_kernel_sum_mean_ms"]
        * 100.0
    )
    return summary.sort_values(["batch_size", "fresh_len", "prefix_len", "case_key"]).reset_index(drop=True)


def _jittered_points(center: float, count: int, scale: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return center + rng.uniform(-scale, scale, size=count)


def plot_comparison(df: pd.DataFrame, sim_column: str, title: str, output_path: Path) -> None:
    summary = aggregate_case_statistics(df, sim_column)
    case_keys = list(summary["case_key"])
    labels = list(summary["display_label"])
    x_positions = np.arange(len(case_keys), dtype=float)
    real_offset = -0.18
    sim_offset = 0.20
    width = 0.30

    fig_width = max(16, len(case_keys) * 0.72)
    fig, ax = plt.subplots(figsize=(fig_width, 8.6), dpi=220)

    real_values = []
    sim_values = []
    for case_key in case_keys:
        case_df = df[df["case_key"] == case_key]
        real_values.append(case_df["real_attention_kernel_time_ms"].to_numpy(dtype=float))
        sim_values.append(case_df[sim_column].to_numpy(dtype=float))

    all_values = [values for values in real_values if len(values)] + [values for values in sim_values if len(values)]

    # Real measurements as boxplot + raw points + mean/std.
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

    # AIC as reference line + optional raw marker.
    sim_means = np.array([np.mean(values) for values in sim_values], dtype=float)
    sim_stds = np.array([np.std(values, ddof=1) if len(values) > 1 else 0.0 for values in sim_values], dtype=float)
    sim_pos = x_positions + sim_offset
    ax.errorbar(
        sim_pos,
        sim_means,
        yerr=sim_stds,
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
    for idx, values in enumerate(sim_values):
        if len(values) == 0:
            continue
        points_x = _jittered_points(sim_pos[idx], len(values), scale=0.02, seed=idx + 5000)
        ax.scatter(points_x, values, s=12, color="#d62728", alpha=0.55, marker="x", linewidths=0.8, zorder=4)
        ax.hlines(np.mean(values), real_pos[idx] - 0.10, sim_pos[idx] + 0.10, colors="#d62728", linestyles="dashed", linewidth=1.0, alpha=0.85)

    # Percent labels: real mean increase vs sim mean.
    y_span = max(max(v) for v in all_values) - min(min(v) for v in all_values)
    label_offset = max(y_span * 0.025, max(max(v) for v in all_values) * 0.012, 0.15)
    for idx, row in summary.iterrows():
        upper = max(np.max(real_values[idx]), np.max(sim_values[idx]))
        ax.text(
            x_positions[idx],
            upper + label_offset,
            f"+{row['pct_increase_vs_sim_mean']:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            color="#7f1d1d",
            fontweight="bold",
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Latency (ms)")
    ax.set_xlabel("Case (batch / fresh / prefix)")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=True)

    top = max(max(v) for v in all_values) * 1.16
    ax.set_ylim(bottom=0, top=top)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = load_prefill_data(args.comparison_csv, prefill_only=args.prefill_only)

    summary_paths = []
    for cfg in PLOT_CONFIGS:
        plot_df = df.dropna(subset=[cfg.sim_column]).copy()
        if plot_df.empty:
            raise ValueError(f"No rows available for {cfg.name}")
        output_path = args.output_dir / f"{cfg.output_suffix}__real_vs_aic.png"
        summary_path = args.output_dir / f"{cfg.output_suffix}__summary.csv"
        plot_comparison(plot_df, cfg.sim_column, cfg.title, output_path)
        aggregate_case_statistics(plot_df, cfg.sim_column).to_csv(summary_path, index=False)
        summary_paths.append(str(summary_path))
        print(f"Wrote plot to {output_path}")
        print(f"Wrote summary to {summary_path}")

    print(f"Loaded {len(df)} prefill rows from {args.comparison_csv}")
    print("Summary CSVs:")
    for path in summary_paths:
        print(f"- {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
