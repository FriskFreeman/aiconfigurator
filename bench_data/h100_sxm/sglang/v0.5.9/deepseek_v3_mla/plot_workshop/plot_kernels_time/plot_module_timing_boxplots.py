#!/usr/bin/env python3
"""Plot module timing boxplots for stage1 prefill runs.

The script scans the benchmark CSV directory, groups files by case, and plots
per-case distributions for:
- module_gpu_makespan_ms
- module_gpu_kernel_time_sum_ms

Each case is represented by the batch size, fresh length, and prefix length
parsed from the CSV file name. The plot includes:
- boxplots
- raw data points
- mean +/- std error bars
- a percent annotation for mean makespan increase vs mean kernel time sum
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


CSV_NAME_RE = re.compile(
    r"_req1_equal_(?P<family>fresh|prefix)_(?P<variant>regular|irregular)_b(?P<batch>\d+)_f(?P<fresh>[^_]+)_p(?P<prefix>[^_]+)_b"
)


def _case_label_from_name(name: str) -> tuple[str, str, str, int, str, str]:
    match = CSV_NAME_RE.search(name)
    if not match:
        raise ValueError(f"Unable to parse case label from file name: {name}")
    family = match.group("family")
    batch = int(match.group("batch"))
    fresh = match.group("fresh")
    prefix = match.group("prefix")
    label = f"b{batch}\nf{fresh}\np{prefix}"
    case_key = f"{family}:{batch}:{fresh}:{prefix}"
    return case_key, family, label, batch, fresh, prefix


def _pick_numeric_series(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame.columns:
        raise KeyError(f"Missing required column: {column}")
    series = pd.to_numeric(frame[column], errors="coerce").dropna()
    if series.empty:
        raise ValueError(f"No numeric values found in {column}")
    return series.to_numpy(dtype=float)


def collect_case_rows(csv_dir: Path) -> pd.DataFrame:
    rows = []
    skipped_files = []
    for csv_path in sorted(csv_dir.glob("*__MLA时延拆解.csv")):
        case_key, family, label, batch, fresh, prefix = _case_label_from_name(csv_path.name)
        frame = pd.read_csv(csv_path)
        if "stage" not in frame.columns:
            raise KeyError(f"Missing required column: stage in {csv_path.name}")
        frame = frame.loc[frame["stage"] == "prefill"].copy()
        if frame.empty:
            skipped_files.append(csv_path.name)
            continue
        makespan = _pick_numeric_series(frame, "module_gpu_makespan_ms")
        kernel_sum = _pick_numeric_series(frame, "module_gpu_kernel_time_sum_ms")
        if len(makespan) != len(kernel_sum):
            raise ValueError(
                f"Length mismatch in {csv_path.name}: {len(makespan)} vs {len(kernel_sum)}"
            )
        base = pd.DataFrame(
            {
                "case_key": case_key,
                "case_family": family,
                "case_label": label,
                "batch_size": batch,
                "fresh_len": fresh,
                "prefix_len": prefix,
                "row_index": np.arange(len(makespan)),
                "source_csv": csv_path.name,
            }
        )
        rows.append(
            pd.concat(
                [
                    base.assign(metric="makespan_ms", value_ms=makespan),
                    base.assign(metric="kernel_time_sum_ms", value_ms=kernel_sum),
                ],
                ignore_index=True,
            )
        )
    if not rows:
        raise FileNotFoundError(f"No CSV files found under {csv_dir}")
    data = pd.concat(rows, ignore_index=True)
    data.attrs["skipped_files"] = skipped_files
    return data


def _filter_stage_rows(frame: pd.DataFrame, stage: str, csv_name: str) -> pd.DataFrame:
    if "stage" not in frame.columns:
        raise KeyError(f"Missing required column: stage in {csv_name}")
    filtered = frame.loc[frame["stage"] == stage].copy()
    if filtered.empty:
        raise ValueError(f"No {stage} rows found in {csv_name}")
    return filtered


def filter_outliers_iqr(data: pd.DataFrame) -> pd.DataFrame:
    filtered_groups = []
    for (case_key, metric), group in data.groupby(["case_key", "metric"], sort=False):
        values = group["value_ms"].to_numpy(dtype=float)
        if len(values) < 4:
            filtered_groups.append(group)
            continue
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        filtered_groups.append(group[(group["value_ms"] >= lower) & (group["value_ms"] <= upper)])
    return pd.concat(filtered_groups, ignore_index=True)


def build_metric_summary(data: pd.DataFrame) -> pd.DataFrame:
    summary = (
        data.groupby(["case_key", "case_family", "case_label", "batch_size", "fresh_len", "prefix_len", "metric"], as_index=False)
        .agg(mean_ms=("value_ms", "mean"), std_ms=("value_ms", "std"), count=("value_ms", "count"))
        .pivot(index=["case_key", "case_family", "case_label", "batch_size", "fresh_len", "prefix_len"], columns="metric")
        .reset_index()
    )
    summary.columns = [
        "case_key",
        "case_family",
        "case_label",
        "batch_size",
        "fresh_len",
        "prefix_len",
        "mean_makespan_ms",
        "mean_kernel_time_sum_ms",
        "std_makespan_ms",
        "std_kernel_time_sum_ms",
        "n_samples_makespan",
        "n_samples_kernel_sum",
    ]
    summary["pct_increase_vs_kernel_sum"] = (
        (summary["mean_makespan_ms"] - summary["mean_kernel_time_sum_ms"])
        / summary["mean_kernel_time_sum_ms"]
        * 100.0
    )
    summary["display_label"] = summary["case_label"]
    duplicate_labels = summary["display_label"].duplicated(keep=False)
    summary.loc[duplicate_labels, "display_label"] = (
        summary.loc[duplicate_labels, "case_family"] + "\n" + summary.loc[duplicate_labels, "display_label"]
    )
    return summary.sort_values(
        ["pct_increase_vs_kernel_sum", "batch_size", "fresh_len", "prefix_len", "case_label"]
    ).reset_index(drop=True)


def build_simple_summary(data: pd.DataFrame) -> pd.DataFrame:
    summary = (
        data.groupby(["case_key", "case_family", "case_label", "batch_size", "fresh_len", "prefix_len", "metric"], as_index=False)
        .agg(mean_ms=("value_ms", "mean"), std_ms=("value_ms", "std"), count=("value_ms", "count"))
        .pivot(index=["case_key", "case_family", "case_label", "batch_size", "fresh_len", "prefix_len"], columns="metric")
        .reset_index()
    )
    summary.columns = [
        "case_key",
        "case_family",
        "case_label",
        "batch_size",
        "fresh_len",
        "prefix_len",
        "mean_primary_ms",
        "mean_secondary_ms",
        "std_primary_ms",
        "std_secondary_ms",
        "n_samples_primary",
        "n_samples_secondary",
    ]
    summary["pct_increase_vs_kernel_sum"] = (
        (summary["mean_primary_ms"] - summary["mean_secondary_ms"])
        / summary["mean_secondary_ms"]
        * 100.0
    )
    summary["display_label"] = summary["case_label"]
    duplicate_labels = summary["display_label"].duplicated(keep=False)
    summary.loc[duplicate_labels, "display_label"] = (
        summary.loc[duplicate_labels, "case_family"] + "\n" + summary.loc[duplicate_labels, "display_label"]
    )
    return summary.sort_values(
        ["batch_size", "fresh_len", "prefix_len", "case_label"]
    ).reset_index(drop=True)


def _jittered_points(center: float, count: int, scale: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return center + rng.uniform(-scale, scale, size=count)


def plot_boxplots(summary: pd.DataFrame, raw: pd.DataFrame, output_path: Path, primary_metric: str, secondary_metric: str, title: str, y_label: str) -> None:
    sns.set_theme(style="whitegrid", context="talk")

    cases = list(summary["case_key"])
    display_labels = list(summary["display_label"])
    x_positions = np.arange(len(cases), dtype=float)
    offsets = {primary_metric: -0.18, secondary_metric: 0.18}
    colors = {primary_metric: "#4C78A8", secondary_metric: "#F58518"}
    widths = 0.30

    fig_width = max(16, len(cases) * 0.72)
    fig, ax = plt.subplots(figsize=(fig_width, 8.5), dpi=220)

    all_values = []

    for metric in [primary_metric, secondary_metric]:
        metric_raw = raw[raw["metric"] == metric]
        grouped_values = [
            metric_raw.loc[metric_raw["case_key"] == case, "value_ms"].to_numpy(dtype=float)
            for case in cases
        ]
        all_values.extend([values for values in grouped_values if len(values)])

        positions = x_positions + offsets[metric]
        box = ax.boxplot(
            grouped_values,
            positions=positions,
            widths=widths,
            patch_artist=True,
            showfliers=False,
            whis=(5, 95),
            medianprops={"color": "#222222", "linewidth": 1.8},
            boxprops={"linewidth": 1.5, "edgecolor": colors[metric]},
            whiskerprops={"linewidth": 1.3, "color": colors[metric]},
            capprops={"linewidth": 1.3, "color": colors[metric]},
        )
        for patch in box["boxes"]:
            patch.set_facecolor(colors[metric])
            patch.set_alpha(0.20)

        means = np.array([np.mean(vals) for vals in grouped_values], dtype=float)
        stds = np.array([np.std(vals, ddof=1) if len(vals) > 1 else 0.0 for vals in grouped_values], dtype=float)
        ax.errorbar(
            positions,
            means,
            yerr=stds,
            fmt="o",
            color=colors[metric],
            ecolor=colors[metric],
            elinewidth=1.4,
            capsize=4,
            markersize=5,
            markeredgecolor="white",
            markeredgewidth=0.8,
            zorder=4,
            label=f"{metric} mean ± std",
        )

        for idx, vals in enumerate(grouped_values):
            if len(vals) == 0:
                continue
            points_x = _jittered_points(positions[idx], len(vals), scale=0.045, seed=idx + (0 if metric == "makespan_ms" else 1000))
            ax.scatter(
                points_x,
                vals,
                s=16,
                color=colors[metric],
                alpha=0.45,
                linewidths=0,
                zorder=3,
            )

    summary = summary.copy()
    base_max = raw["value_ms"].max()
    y_range = raw["value_ms"].max() - raw["value_ms"].min()
    offset = max(y_range * 0.025, base_max * 0.015, 0.2)

    for idx, row in summary.iterrows():
        case_mask = raw["case_key"] == row["case_key"]
        case_max = raw.loc[case_mask, "value_ms"].max()
        ax.text(
            x_positions[idx],
            case_max + offset,
            f"{row['pct_increase_vs_kernel_sum']:+.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            color="#7F1D1D",
            fontweight="bold",
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(display_labels, fontsize=10)
    ax.set_ylabel(y_label)
    ax.set_xlabel("Case (batch / fresh / prefix)")
    ax.set_title(title)
    ax.legend(loc="upper left", ncols=2, frameon=True)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.25)

    ylim_top = max(max(v) for v in all_values) * 1.18
    ax.set_ylim(bottom=0, top=ylim_top)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "csv",
        help="Directory containing module timing CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory to write plots and summaries.",
    )
    parser.add_argument(
        "--output-name",
        default="req1_equal_len_module_gpu_timing_boxplot.png",
        help="Output PNG file name.",
    )
    parser.add_argument(
        "--summary-name",
        default="req1_equal_len_module_gpu_timing_summary.csv",
        help="Output summary CSV file name.",
    )
    parser.add_argument(
        "--stage",
        choices=["prefill", "decode"],
        default="prefill",
        help="Which stage to extract from the CSV files.",
    )
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw = collect_case_rows(args.csv_dir)
    stage_rows = []
    skipped_files = raw.attrs.get("skipped_files", [])
    for csv_path in sorted(args.csv_dir.glob("*__MLA时延拆解.csv")):
        frame = pd.read_csv(csv_path)
        try:
            stage_frame = _filter_stage_rows(frame, args.stage, csv_path.name)
        except ValueError:
            if csv_path.name not in skipped_files:
                skipped_files.append(csv_path.name)
            continue
        case_key, family, label, batch, fresh, prefix = _case_label_from_name(csv_path.name)
        primary_metric_name = "makespan_ms"
        secondary_metric_name = "kernel_time_sum_ms"
        primary_values = _pick_numeric_series(stage_frame, "module_gpu_makespan_ms")
        secondary_values = _pick_numeric_series(stage_frame, "module_gpu_kernel_time_sum_ms")
        if len(primary_values) != len(secondary_values):
            raise ValueError(f"Length mismatch in {csv_path.name}")
        base = pd.DataFrame(
            {
                "case_key": case_key,
                "case_family": family,
                "case_label": label,
                "batch_size": batch,
                "fresh_len": fresh,
                "prefix_len": prefix,
                "row_index": np.arange(len(primary_values)),
                "source_csv": csv_path.name,
            }
        )
        stage_rows.append(
            pd.concat(
                [
                    base.assign(metric=primary_metric_name, value_ms=primary_values),
                    base.assign(metric=secondary_metric_name, value_ms=secondary_values),
                ],
                ignore_index=True,
            )
        )
    if not stage_rows:
        raise FileNotFoundError(f"No {args.stage} rows found under {args.csv_dir}")
    stage_raw = pd.concat(stage_rows, ignore_index=True)
    stage_raw.attrs["skipped_files"] = skipped_files
    filtered_raw = filter_outliers_iqr(stage_raw)
    if args.stage == "prefill":
        summary = build_metric_summary(filtered_raw)
        primary_metric = "makespan_ms"
        secondary_metric = "kernel_time_sum_ms"
        title = "Module GPU Timing Distribution for req1_equal_len (prefill)"
    else:
        summary = build_simple_summary(filtered_raw)
        primary_metric = "makespan_ms"
        secondary_metric = "kernel_time_sum_ms"
        title = "Module GPU Timing Distribution for req1_equal_len (decode)"

    summary_path = args.output_dir / args.summary_name
    summary.to_csv(summary_path, index=False)

    output_path = args.output_dir / args.output_name
    plot_boxplots(summary, filtered_raw, output_path, primary_metric, secondary_metric, title, "Latency (ms)")

    print(f"Wrote plot to {output_path}")
    print(f"Wrote summary to {summary_path}")
    print(f"Loaded {summary.shape[0]} cases from {args.csv_dir}")
    skipped_files = stage_raw.attrs.get("skipped_files", [])
    if skipped_files:
        print(f"Skipped {len(skipped_files)} pure-decode files")
        for file_name in skipped_files:
            print(f"  - {file_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
