#!/usr/bin/env python3
"""Build SILICON versus ANALYTICAL low/standard/high comparison artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from aiconfigurator.sdk.pareto_analysis import get_pareto_front


ROOT = Path(__file__).resolve().parent / "analytical-levels"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
FRONTIER_FIGURES = FIGURES / "pareto_frontiers"
MODEL_ORDER = ["llama31_70b", "qwen3_32b", "deepseek_v3"]
SYSTEM_ORDER = ["a100_sxm", "h100_sxm", "h200_sxm", "b200_sxm", "b300_sxm", "rtx_pro_6000_server"]
SERIES_ORDER = ["SILICON", "ANALYTICAL-low", "ANALYTICAL-standard", "ANALYTICAL-high"]
COLORS = {
    "SILICON": "#386FA4",
    "ANALYTICAL-low": "#E89A9A",
    "ANALYTICAL-standard": "#C94C55",
    "ANALYTICAL-high": "#8F1D2C",
}
MODEL_LABELS = {
    "llama31_70b": "Llama 3.1 70B",
    "qwen3_32b": "Qwen3 32B",
    "deepseek_v3": "DeepSeek V3",
}
SYSTEM_LABELS = {
    "a100_sxm": "A100 SXM",
    "h100_sxm": "H100 SXM",
    "h200_sxm": "H200 SXM",
    "b200_sxm": "B200 SXM",
    "b300_sxm": "B300 SXM",
    "rtx_pro_6000_server": "RTX PRO 6000 Server",
}


def series_name(status: dict) -> str:
    if status["mode"] == "SILICON":
        return "SILICON"
    return f"ANALYTICAL-{status['analytical_level']}"


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    statuses = []
    frontiers = []
    for path in sorted(RESULTS.glob("*/status.json")):
        status = json.loads(path.read_text(encoding="utf-8"))
        status["series"] = series_name(status)
        statuses.append(status)
        if status["status"] != "success":
            continue
        candidates = pd.read_csv(path.parent / "pareto_df.csv")
        total_gpus = int(status["total_gpus"])
        candidates["tokens/s/gpu_cluster"] = (
            candidates["tokens/s/gpu"]
            * (total_gpus // candidates["num_total_gpus"])
            * candidates["num_total_gpus"]
            / total_gpus
        )
        frontier = get_pareto_front(
            candidates,
            "tokens/s/user",
            "tokens/s/gpu_cluster",
            maximize_x=True,
            maximize_y=True,
        ).sort_values("tokens/s/user")
        frontier.insert(0, "series", status["series"])
        frontier.insert(0, "case_id", status["case_id"])
        frontier.insert(0, "system", status["system"])
        frontier.insert(0, "model_key", status["model_key"])
        frontiers.append(frontier)

    status_df = pd.DataFrame(statuses).sort_values(["model_key", "system", "series"])
    frontier_df = pd.concat(frontiers, ignore_index=True)
    status_df.to_csv(RESULTS / "run_summary.csv", index=False)
    frontier_df.to_csv(RESULTS / "pareto_frontiers.csv", index=False)
    best = frontier_df.loc[
        frontier_df.groupby(["model_key", "system", "series"])["tokens/s/gpu_cluster"].idxmax()
    ].sort_values(["model_key", "system", "series"])
    best.to_csv(RESULTS / "best_frontier_points.csv", index=False)
    return status_df, frontier_df


def plot_frontiers(status: pd.DataFrame, frontiers: pd.DataFrame) -> None:
    FRONTIER_FIGURES.mkdir(parents=True, exist_ok=True)
    for model in MODEL_ORDER:
        for system in SYSTEM_ORDER:
            fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
            unavailable = []
            for series in SERIES_ORDER:
                data = frontiers[
                    (frontiers.model_key == model)
                    & (frontiers.system == system)
                    & (frontiers.series == series)
                ]
                if data.empty:
                    unavailable.append(series)
                    continue
                ax.plot(
                    data["tokens/s/user"],
                    data["tokens/s/gpu_cluster"],
                    marker="o",
                    markersize=4,
                    linewidth=2,
                    color=COLORS[series],
                    label=f"{series} (n={len(data)})",
                )
            if unavailable:
                ax.text(
                    0.02,
                    0.98,
                    "Unavailable: " + ", ".join(unavailable),
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=9,
                    color="#8B2E2E",
                )
            ax.set_title(f"{MODEL_LABELS[model]} on {SYSTEM_LABELS[system]}")
            ax.set_xlabel("Per-user generation rate (tokens/s/user, higher is better)")
            ax.set_ylabel("Cluster-normalized throughput (tokens/s/GPU, higher is better)")
            ax.grid(alpha=0.25)
            ax.legend(loc="best")
            fig.savefig(FRONTIER_FIGURES / f"{model}__{system}.png", dpi=180)
            plt.close(fig)


def plot_best(best: pd.DataFrame) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(15, 14), constrained_layout=True)
    for ax, model in zip(axes, MODEL_ORDER):
        subset = best[best.model_key == model]
        x = np.arange(len(SYSTEM_ORDER))
        width = 0.2
        for index, series in enumerate(SERIES_ORDER):
            values = []
            for system in SYSTEM_ORDER:
                rows = subset[(subset.system == system) & (subset.series == series)]
                values.append(rows["tokens/s/gpu_cluster"].iloc[0] if len(rows) else np.nan)
            bars = ax.bar(
                x + (index - 1.5) * width,
                values,
                width,
                label=series,
                color=COLORS[series],
            )
            ax.bar_label(bars, fmt="%.0f", fontsize=7, padding=2)
        ax.set_title(MODEL_LABELS[model])
        ax.set_ylabel("Best cluster-normalized tokens/s/GPU")
        ax.set_xticks(x, [SYSTEM_LABELS[s] for s in SYSTEM_ORDER], rotation=15)
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(ncol=4, fontsize=9)
    fig.savefig(FIGURES / "best_throughput_by_estimate_level.png", dpi=180)
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    status, frontiers = load()
    plot_frontiers(status, frontiers)
    best = pd.read_csv(RESULTS / "best_frontier_points.csv")
    plot_best(best)


if __name__ == "__main__":
    main()
