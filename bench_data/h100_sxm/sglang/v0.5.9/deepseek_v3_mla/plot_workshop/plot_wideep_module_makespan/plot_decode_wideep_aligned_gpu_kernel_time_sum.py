#!/usr/bin/env python3
"""Plot WideEP-aligned GPU kernel time sum for decode stage as standalone.

Produces a boxplot comparing real vs MLA Sim (AIC) for the
`real_wideep_aligned_gpu_kernel_time_sum_ms` metric. Legend shows "MLA Sim mean ± std".
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comparison-csv",
        default=Path(__file__).resolve().parents[2] / "analysis" / "wideep_module_mla" / "wideep_module_mla_aic_compare__comparison.csv",
        type=Path,
    )
    parser.add_argument(
        "--output-dir",
        default=Path(__file__).resolve().parent,
        type=Path,
    )
    parser.add_argument(
        "--stage",
        choices=["prefill", "decode", "all"],
        default="decode",
    )
    parser.add_argument("--output-prefix", default=None)
    return parser.parse_args(argv)


def _case_key(df: pd.DataFrame) -> pd.Series:
    return (
        df["tag"].astype(str)
        + "|"
        + df["stage"].astype(str)
        + "|b"
        + df["batch_size"].astype(int).astype(str)
        + "|f"
        + df["fresh_len"].astype(int).astype(str)
        + "|p"
        + df["prefix_len"].astype(int).astype(str)
    )


def _display_label(row: pd.Series) -> str:
    return f"b{int(row['batch_size'])}\nf{int(row['fresh_len'])}\np{int(row['prefix_len'])}"


def load_stage_data(path: Path, stage: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if stage != "all":
        df = df[df["stage"] == stage].copy()
    if df.empty:
        raise ValueError(f"No rows left after filtering: {path}")

    cols = [
        "real_wideep_aligned_gpu_kernel_time_sum_ms",
        "aic_wideep_mla_latency_ms",
        "batch_size",
        "fresh_len",
        "prefix_len",
    ]
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["real_wideep_aligned_gpu_kernel_time_sum_ms", "aic_wideep_mla_latency_ms", "batch_size", "fresh_len", "prefix_len"]) 
    df["case_key"] = _case_key(df)
    df["display_label"] = df.apply(_display_label, axis=1)
    return df


def aggregate_case_statistics(df: pd.DataFrame, real_column: str) -> pd.DataFrame:
    grouped = df.groupby(["case_key", "display_label", "batch_size", "fresh_len", "prefix_len"], as_index=False)
    summary = grouped.agg(
        real_mean_ms=(real_column, "mean"),
        real_std_ms=(real_column, "std"),
        real_count=(real_column, "count"),
        sim_mean_ms=("aic_wideep_mla_latency_ms", "mean"),
        sim_std_ms=("aic_wideep_mla_latency_ms", "std"),
        sim_count=("aic_wideep_mla_latency_ms", "count"),
    )
    summary["real_std_ms"] = summary["real_std_ms"].fillna(0.0)
    summary["sim_std_ms"] = summary["sim_std_ms"].fillna(0.0)
    summary["pct_increase_vs_sim_mean"] = (
        (summary["real_mean_ms"] - summary["sim_mean_ms"]) / summary["sim_mean_ms"] * 100.0
    )
    return summary.sort_values(["batch_size", "fresh_len", "prefix_len", "case_key"]).reset_index(drop=True)


def _jittered_points(center: float, count: int, scale: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return center + rng.uniform(-scale, scale, size=count)


def plot_comparison(df: pd.DataFrame, output_path: Path) -> None:
    real_col = "real_wideep_aligned_gpu_kernel_time_sum_ms"
    summary = aggregate_case_statistics(df, real_col)
    case_keys = list(summary["case_key"])
    labels = list(summary["display_label"])
    x_positions = np.arange(len(case_keys), dtype=float)
    real_offset = -0.18
    sim_offset = 0.20
    width = 0.30

    fig_width = max(14, len(case_keys) * 0.62)
    fig, ax = plt.subplots(figsize=(fig_width, 7.6), dpi=210)

    real_values = []
    sim_values = []
    for case_key in case_keys:
        case_df = df[df["case_key"] == case_key]
        real_values.append(case_df[real_col].to_numpy(dtype=float))
        sim_values.append(case_df["aic_wideep_mla_latency_ms"].to_numpy(dtype=float))

    real_pos = x_positions + real_offset
    real_box = ax.boxplot(real_values, positions=real_pos, widths=width, patch_artist=True, showfliers=False, whis=(5,95))
    for patch in real_box["boxes"]:
        patch.set_facecolor("#1f77b4")
        patch.set_alpha(0.18)

    real_means = np.array([np.mean(v) for v in real_values], dtype=float)
    real_stds = np.array([np.std(v, ddof=1) if len(v)>1 else 0.0 for v in real_values], dtype=float)
    ax.errorbar(real_pos, real_means, yerr=real_stds, fmt="o", color="#1f77b4", label="Real mean ± std")

    sim_means = np.array([np.mean(v) for v in sim_values], dtype=float)
    sim_stds = np.array([np.std(v, ddof=1) if len(v)>1 else 0.0 for v in sim_values], dtype=float)
    sim_pos = x_positions + sim_offset
    ax.errorbar(sim_pos, sim_means, yerr=sim_stds, fmt="s", color="#d62728", label="MLA Sim mean ± std")

    # percent labels
    y_span = max(max(v) for v in real_values+sim_values) - min(min(v) for v in real_values+sim_values)
    label_offset = max(y_span * 0.03, max(max(v) for v in real_values+sim_values) * 0.012, 0.0008)
    top = max(max(v) for v in real_values+sim_values) * 1.12
    for idx, row in summary.iterrows():
        real_mean = float(row["real_mean_ms"])
        upper = max(np.max(real_values[idx]), np.max(sim_values[idx]))
        lower = min(np.min(real_values[idx]), np.min(sim_values[idx]))
        if real_mean + label_offset <= top * 0.97:
            label_y = real_mean + label_offset
            valign = "bottom"
        else:
            label_y = max(lower - label_offset, 0.0)
            valign = "top"
        pct = row["pct_increase_vs_sim_mean"]
        sign = "+" if pct >= 0 else ""
        ax.text(x_positions[idx], label_y, f"{sign}{pct:.1f}%", ha="center", va=valign, fontsize=9, color="#7f1d1d", fontweight="bold")

    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Latency (ms)")
    ax.set_xlabel("Case (batch / fresh / prefix)")
    ax.set_title("Decode GPU Kernel Time Sum (Real vs MLA Sim)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper left")

    ax.set_ylim(bottom=0, top=top)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = load_stage_data(args.comparison_csv, stage=args.stage)
    output_prefix = args.output_prefix or f"{args.stage}_wideep_aligned_gpu_kernel_time_sum"
    output_path = args.output_dir / f"{output_prefix}__real_vs_mla_sim.png"
    summary_path = args.output_dir / f"{output_prefix}__summary.csv"
    plot_comparison(df, output_path)
    aggregate_case_statistics(df, "real_wideep_aligned_gpu_kernel_time_sum_ms").to_csv(summary_path, index=False)
    print(f"Wrote plot to {output_path}")
    print(f"Wrote summary to {summary_path}")
    print(f"Loaded {len(df)} rows from {args.comparison_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
