#!/usr/bin/env python3
"""Aggregate acceptance artifacts and render comparison plots."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from aiconfigurator.sdk.pareto_analysis import get_pareto_front


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
MODEL_ORDER = ["llama31_70b", "qwen3_32b", "deepseek_v3"]
SYSTEM_ORDER = ["a100_sxm", "h100_sxm", "h200_sxm", "b200_sxm", "b300_sxm", "rtx_pro_6000_server"]
MODE_ORDER = ["SILICON", "EMPIRICAL", "ANALYTICAL"]
COLORS = {"SILICON": "#386FA4", "EMPIRICAL": "#D98E32", "ANALYTICAL": "#2A9D6F"}
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


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    statuses = []
    best_rows = []
    for path in RESULTS.glob("*/status.json"):
        status = json.loads(path.read_text(encoding="utf-8"))
        statuses.append(status)
        if status["status"] != "success":
            continue
        pareto = pd.read_csv(path.parent / "pareto_df.csv")
        best = pareto.loc[pareto["tokens/s/gpu"].idxmax()].to_dict()
        best_rows.append({**status, **{f"best_{key}": value for key, value in best.items()}})
    status_df = pd.DataFrame(statuses).sort_values(["model_key", "system", "mode"])
    best_df = pd.DataFrame(best_rows).sort_values(["model_key", "system", "mode"])
    status_df.to_csv(RESULTS / "run_summary.csv", index=False)
    status_df.to_json(RESULTS / "run_summary.json", orient="records", force_ascii=False, indent=2)
    best_df.to_csv(RESULTS / "best_points.csv", index=False)
    return status_df, best_df


def plot_best(best: pd.DataFrame) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(14, 13), constrained_layout=True)
    for ax, model in zip(axes, MODEL_ORDER):
        subset = best[best.model_key == model]
        x = np.arange(len(SYSTEM_ORDER))
        width = 0.25
        for offset, mode in enumerate(MODE_ORDER):
            values = []
            for system in SYSTEM_ORDER:
                rows = subset[(subset.system == system) & (subset["mode"] == mode)]
                values.append(rows.best_tokens_per_s_gpu.iloc[0] if len(rows) else np.nan)
            bars = ax.bar(x + (offset - 1) * width, values, width, label=mode, color=COLORS[mode])
            ax.bar_label(bars, fmt="%.0f", fontsize=8, padding=2)
        ax.set_title(model)
        ax.set_ylabel("Best output tokens/s/GPU")
        ax.set_xticks(x, SYSTEM_ORDER, rotation=15)
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(ncol=3)
    fig.savefig(FIGURES / "best_throughput_by_mode.png", dpi=180)
    plt.close(fig)


def plot_ratios(best: pd.DataFrame) -> None:
    rows = []
    for (model, system), group in best.groupby(["model_key", "system"]):
        values = group.set_index("mode").best_tokens_per_s_gpu
        if "SILICON" not in values:
            continue
        for mode in ["EMPIRICAL", "ANALYTICAL"]:
            if mode in values:
                rows.append({"model": model, "system": system, "mode": mode, "ratio": values[mode] / values.SILICON})
    data = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(14, 6), constrained_layout=True)
    labels = [f"{row.model}\n{row.system}" for row in data.itertuples() if row.mode == "EMPIRICAL"]
    pairs = data.pivot(index=["model", "system"], columns="mode", values="ratio")
    x = np.arange(len(pairs))
    for index, mode in enumerate(["EMPIRICAL", "ANALYTICAL"]):
        bars = ax.bar(x + (index - 0.5) * 0.36, pairs[mode], 0.36, label=mode, color=COLORS[mode])
        ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
    ax.axhline(1.0, color="black", linewidth=1)
    ax.set_xticks(x, labels, rotation=35, ha="right")
    ax.set_ylabel("Best throughput / SILICON")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(FIGURES / "throughput_ratio_vs_silicon.png", dpi=180)
    plt.close(fig)


def plot_status(status: pd.DataFrame) -> None:
    labels = [f"{m}\n{s}" for m in MODEL_ORDER for s in SYSTEM_ORDER]
    matrix = np.zeros((len(labels), len(MODE_ORDER)))
    mapping = {"failed": 0, "skipped": 1, "success": 2}
    for i, (model, system) in enumerate((m, s) for m in MODEL_ORDER for s in SYSTEM_ORDER):
        for j, mode in enumerate(MODE_ORDER):
            row = status[(status.model_key == model) & (status.system == system) & (status["mode"] == mode)]
            matrix[i, j] = mapping[row.status.iloc[0]]
    fig, ax = plt.subplots(figsize=(8, 10), constrained_layout=True)
    ax.imshow(matrix, cmap=plt.matplotlib.colors.ListedColormap(["#C94C4C", "#D9B44A", "#2A9D6F"]), vmin=0, vmax=2)
    ax.set_xticks(range(3), MODE_ORDER)
    ax.set_yticks(range(len(labels)), labels)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, ["FAIL", "SKIP", "PASS"][int(matrix[i, j])], ha="center", va="center", color="white", fontsize=8)
    ax.set_title("PD-disaggregated Pareto acceptance status")
    fig.savefig(FIGURES / "acceptance_status.png", dpi=180)
    plt.close(fig)


def plot_native_pareto_frontiers(status: pd.DataFrame) -> None:
    """Overlay CLI-equivalent per-user/per-GPU Pareto fronts per case."""
    output_dir = FIGURES / "pareto_frontiers"
    output_dir.mkdir(parents=True, exist_ok=True)
    frontier_rows = []

    for model in MODEL_ORDER:
        for system in SYSTEM_ORDER:
            fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
            unavailable = []
            for mode in MODE_ORDER:
                row = status[
                    (status.model_key == model)
                    & (status.system == system)
                    & (status["mode"] == mode)
                ]
                if row.empty or row.status.iloc[0] != "success":
                    unavailable.append(mode)
                    continue

                case_id = row.case_id.iloc[0]
                candidates = pd.read_csv(RESULTS / case_id / "pareto_df.csv")
                total_gpus = int(row.total_gpus.iloc[0])
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
                if frontier.empty:
                    unavailable.append(mode)
                    continue

                ax.plot(
                    frontier["tokens/s/user"],
                    frontier["tokens/s/gpu_cluster"],
                    marker="o",
                    markersize=4,
                    linewidth=2,
                    color=COLORS[mode],
                    label=f"{mode} (n={len(frontier)})",
                )
                exported = frontier.copy()
                exported.insert(0, "mode", mode)
                exported.insert(0, "system", system)
                exported.insert(0, "model_key", model)
                frontier_rows.append(exported)

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
            fig.savefig(output_dir / f"{model}__{system}.png", dpi=180)
            plt.close(fig)

    pd.concat(frontier_rows, ignore_index=True).to_csv(
        RESULTS / "pareto_frontiers.csv", index=False
    )


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    status, best = load()
    plot_best(best)
    plot_ratios(best)
    plot_status(status)
    plot_native_pareto_frontiers(status)


if __name__ == "__main__":
    main()
