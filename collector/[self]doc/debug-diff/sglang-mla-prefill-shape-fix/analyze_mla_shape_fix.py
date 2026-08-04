#!/usr/bin/env python3
"""Compare refreshed SGLang H100 MLA data with original hardware tables.

The refreshed reference is H100 SGLang 0.5.9.  Its prefill collector uses the
DeepSeek MHA shape (Q/K=192, V=128, local KV heads).  H100 0.5.10 and H200
0.5.10 are retained as original FA3 tables and are expected to contain the old
absorbed MLA shape.  Other backends are compared as observations, with their
kernel source reported separately rather than treated as pure shape effects.
"""

from __future__ import annotations

import argparse
import gzip
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


WORK_DIR = Path(__file__).resolve().parent
REPO_ROOT = WORK_DIR.parents[3]
DEFAULT_DATA_ROOT = REPO_ROOT / "src/aiconfigurator/systems/data"
DEFAULT_OUTPUT_DIR = WORK_DIR / "results"

REFERENCE = {"label": "H100 refreshed", "hardware": "h100_sxm", "version": "0.5.9"}
TARGETS = [
    {"label": "H100 original", "hardware": "h100_sxm", "version": "0.5.10", "status": "old FA3 table; prefill shape bug"},
    {"label": "H200 original", "hardware": "h200_sxm", "version": "0.5.10", "status": "old FA3 table; prefill shape bug"},
    {"label": "B200 original", "hardware": "b200_sxm", "version": "0.5.10", "status": "TRT-LLM MLA path; prefill branch was already dimension-specialized"},
    {"label": "B300 original", "hardware": "b300_sxm", "version": "0.5.10", "status": "TRT-LLM MLA path; prefill branch was already dimension-specialized"},
    {"label": "GB200 original", "hardware": "gb200", "version": "0.5.10", "status": "TRT-LLM MLA path; prefill branch was already dimension-specialized"},
    {"label": "GB300 original", "hardware": "gb300", "version": "0.5.10", "status": "TRT-LLM MLA path; prefill branch was already dimension-specialized"},
    {"label": "RTX PRO 6000 original", "hardware": "rtx_pro_6000_server", "version": "0.5.10", "status": "old Triton table; no corrected prefill refresh"},
]

STAGES = {
    "prefill": "context_mla_perf.txt",
    "decode": "generation_mla_perf.txt",
}
QUANT_CONFIGS = [("bfloat16", "bfloat16"), ("bfloat16", "fp8")]
QUANT_LABELS = {("bfloat16", "bfloat16"): "BF16/BF16", ("bfloat16", "fp8"): "BF16/FP8"}
QUANT_COLORS = {"BF16/BF16": "#2563A6", "BF16/FP8": "#C2413B"}
MATCH_KEYS = ["mla_dtype", "kv_cache_dtype", "num_heads", "batch_size", "isl", "tp_size", "step"]
SHAPE_KEYS = ["num_heads", "batch_size", "isl", "tp_size", "step"]


def path_for(root: Path, spec: dict[str, str], filename: str) -> Path:
    return root / spec["hardware"] / "sglang" / spec["version"] / filename


def load_table(root: Path, spec: dict[str, str], stage: str) -> pd.DataFrame:
    path = path_for(root, spec, STAGES[stage])
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    frame["source_label"] = spec["label"]
    frame["source_hardware"] = spec["hardware"]
    frame["source_version"] = spec["version"]
    frame["source_status"] = spec.get("status", "refreshed corrected MHA prefill")
    return frame


def logical_work(frame: pd.DataFrame, stage: str) -> pd.Series:
    if stage == "prefill":
        return frame["batch_size"] * frame["num_heads"] * frame["isl"]
    return frame["batch_size"] * frame["num_heads"] * frame["step"]


def first_wins(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    return frame.drop_duplicates(keys, keep="first").copy()


def match_pair(reference: pd.DataFrame, target: pd.DataFrame, stage: str, target_spec: dict[str, str]) -> tuple[pd.DataFrame, dict[str, object]]:
    ref = first_wins(reference, MATCH_KEYS)
    tgt = first_wins(target, MATCH_KEYS)
    merged = ref.merge(tgt, on=MATCH_KEYS, suffixes=("_ref", "_target"), how="inner")
    merged["stage"] = stage
    merged["target_label"] = target_spec["label"]
    merged["target_hardware"] = target_spec["hardware"]
    merged["target_version"] = target_spec["version"]
    merged["latency_ratio"] = merged["latency_target"] / merged["latency_ref"]
    merged["logical_work"] = logical_work(merged.rename(columns={"batch_size": "batch_size", "num_heads": "num_heads", "isl": "isl", "step": "step"}), stage)

    ref_keys = set(map(tuple, ref[MATCH_KEYS].to_numpy()))
    tgt_keys = set(map(tuple, tgt[MATCH_KEYS].to_numpy()))
    summary = {
        "stage": stage,
        "target_label": target_spec["label"],
        "target_hardware": target_spec["hardware"],
        "target_version": target_spec["version"],
        "target_status": target_spec.get("status", ""),
        "reference_rows": len(ref),
        "target_rows": len(tgt),
        "matched_rows": len(merged),
        "reference_only_rows": len(ref_keys - tgt_keys),
        "target_only_rows": len(tgt_keys - ref_keys),
        "reference_kernel_sources": "; ".join(f"{k}:{v}" for k, v in ref["kernel_source"].value_counts().items()),
        "target_kernel_sources": "; ".join(f"{k}:{v}" for k, v in tgt["kernel_source"].value_counts().items()),
        "kernel_source_changed_pct": (merged["kernel_source_ref"] != merged["kernel_source_target"]).mean() * 100 if len(merged) else np.nan,
    }
    return merged, summary


def add_size_buckets(points: pd.DataFrame) -> pd.DataFrame:
    result = points.copy()
    result["size_bucket"] = "medium"
    for stage, idx in result.groupby("stage", sort=False).groups.items():
        part = result.loc[idx]
        q25, q75 = part["logical_work"].quantile([0.25, 0.75])
        result.loc[idx, "size_bucket"] = np.select(
            [part["logical_work"] <= q25, part["logical_work"] >= q75],
            ["small", "large"],
            default="medium",
        )
    return result


def summarize(points: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_cols = ["stage", "target_label", "target_hardware", "target_version"]
    for group_key, group in points.groupby(group_cols, sort=False):
        values = dict(zip(group_cols, group_key, strict=True))
        ratios = group["latency_ratio"]
        row = {
            **values,
            "matched_rows": len(group),
            "ratio_p10": ratios.quantile(0.10),
            "ratio_p50": ratios.quantile(0.50),
            "ratio_p90": ratios.quantile(0.90),
            "ratio_mean": ratios.mean(),
            "target_faster_pct": (ratios < 1).mean() * 100,
            "kernel_source_changed_pct": (group["kernel_source_ref"] != group["kernel_source_target"]).mean() * 100,
            "reference_kernel_sources": "; ".join(group["kernel_source_ref"].value_counts().index),
            "target_kernel_sources": "; ".join(group["kernel_source_target"].value_counts().index),
        }
        for combo_name, combo in [("bf16_bf16", QUANT_CONFIGS[0]), ("bf16_fp8", QUANT_CONFIGS[1])]:
            combo_part = group[(group["mla_dtype"] == combo[0]) & (group["kv_cache_dtype"] == combo[1])]
            row[f"{combo_name}_matched"] = len(combo_part)
            row[f"{combo_name}_p50"] = combo_part["latency_ratio"].median() if len(combo_part) else np.nan
        for bucket in ["small", "large"]:
            part = group[group["size_bucket"] == bucket]
            row[f"{bucket}_matched"] = len(part)
            row[f"{bucket}_p50"] = part["latency_ratio"].median() if len(part) else np.nan
            row[f"{bucket}_faster_pct"] = (part["latency_ratio"] < 1).mean() * 100 if len(part) else np.nan
        row["large_minus_small_p50"] = row["large_p50"] - row["small_p50"]
        row["size_pattern"] = (
            "small_slower_large_faster" if row["small_p50"] > 1 and row["large_p50"] < 1 else
            "small_faster_large_slower" if row["small_p50"] < 1 and row["large_p50"] > 1 else
            "target_faster_both" if row["small_p50"] <= 1 and row["large_p50"] <= 1 else
            "target_slower_both" if row["small_p50"] >= 1 and row["large_p50"] >= 1 else "mixed"
        )
        ranked_work = group["logical_work"].rank(method="average")
        ranked_ratio = group["latency_ratio"].rank(method="average")
        row["workload_ratio_spearman"] = ranked_work.corr(ranked_ratio)
        rows.append(row)
    return pd.DataFrame(rows)


def plot_scatter(points: pd.DataFrame, stage: str, output: Path) -> None:
    selected = points[points["stage"] == stage]
    fig, axes = plt.subplots(3, 3, figsize=(15.5, 15.5), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.91, bottom=0.09, wspace=0.14, hspace=0.22)
    fig.suptitle(f"SGLang base MLA {stage}: refreshed H100 vs original hardware tables", fontsize=17, fontweight="bold")
    fig.supxlabel("H100 refreshed latency (ms, log scale)", fontsize=13)
    fig.supylabel("Original-table latency (ms, log scale)", fontsize=13)
    valid = selected[["latency_ref", "latency_target"]].to_numpy().ravel()
    valid = valid[np.isfinite(valid) & (valid > 0)]
    lower = 10 ** (np.floor(np.log10(valid.min()) * 4) / 4)
    upper = 10 ** (np.ceil(np.log10(valid.max()) * 4) / 4)
    for ax, spec in zip(axes.flat, TARGETS):
        panel = selected[selected["target_label"] == spec["label"]]
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lower, upper); ax.set_ylim(lower, upper)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, which="major", color="#D4D4D4", linewidth=0.6); ax.grid(True, which="minor", color="#EEEEEE", linewidth=0.35)
        ax.plot([lower, upper], [lower, upper], "--", color="#303030", linewidth=1)
        if panel.empty:
            ax.text(0.5, 0.5, "No matched data", transform=ax.transAxes, ha="center", va="center", color="#777777")
            ax.set_title(spec["label"], fontsize=10)
            continue
        for combo in QUANT_CONFIGS:
            part = panel[(panel["mla_dtype"] == combo[0]) & (panel["kv_cache_dtype"] == combo[1])]
            if part.empty: continue
            label = QUANT_LABELS[combo]
            ax.scatter(part["latency_ref"], part["latency_target"], s=8, alpha=0.32, color=QUANT_COLORS[label], edgecolors="none", rasterized=True, label=label)
        ratio = panel["latency_ratio"]
        sources = f"{panel.kernel_source_ref.iloc[0]} -> {panel.kernel_source_target.iloc[0]}"
        ax.set_title(f"{spec['label']}\n{sources}", fontsize=9)
        ax.text(0.03, 0.97, f"n={len(panel):,}, p10/50/90={ratio.quantile(.1):.3f}/{ratio.median():.3f}/{ratio.quantile(.9):.3f}x", transform=ax.transAxes, va="top", fontsize=8, bbox={"facecolor":"white", "edgecolor":"#CCC", "alpha":0.82, "pad":3})
    for ax in list(axes.flat)[len(TARGETS):]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.025))
    fig.text(0.5, 0.057, "Each point is an exact shape + quantization-config match; below diagonal means original table is faster.", ha="center", fontsize=10, color="#444444")
    fig.savefig(output, dpi=180, bbox_inches="tight"); fig.savefig(output.with_suffix(".pdf"), dpi=180, bbox_inches="tight"); plt.close(fig)


def plot_fix_factor(points: pd.DataFrame, output: Path) -> None:
    selected = points[points["target_label"] == "H100 original"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), sharey=False)
    for ax, stage in zip(axes, ["prefill", "decode"], strict=True):
        part = selected[selected["stage"] == stage]
        values = []
        labels = []
        colors = []
        for combo in QUANT_CONFIGS:
            q = part[(part["mla_dtype"] == combo[0]) & (part["kv_cache_dtype"] == combo[1])]["latency_ratio"]
            values.append(q.to_numpy()); labels.append(QUANT_LABELS[combo]); colors.append(QUANT_COLORS[QUANT_LABELS[combo]])
        bp = ax.boxplot(values, tick_labels=labels, showfliers=False, patch_artist=True)
        for patch, color in zip(bp["boxes"], colors, strict=True): patch.set_facecolor(color); patch.set_alpha(0.6)
        ax.axhline(1, color="#303030", linestyle="--", linewidth=1)
        ax.set_yscale("log"); ax.set_ylabel("old H100 / refreshed H100 latency ratio"); ax.set_title(f"{stage}: same H100 shape/config")
        ax.grid(True, axis="y", which="both", color="#E5E5E5", linewidth=0.6)
        for i, arr in enumerate(values, 1): ax.text(i, np.median(arr), f"median {np.median(arr):.3f}x", ha="center", va="bottom", fontsize=9)
    fig.suptitle("Direct shape-fix effect: old H100 table vs refreshed H100 table", fontsize=15, fontweight="bold")
    fig.tight_layout(); fig.savefig(output, dpi=180, bbox_inches="tight"); fig.savefig(output.with_suffix(".pdf"), dpi=180, bbox_inches="tight"); plt.close(fig)


def plot_hardware_summary(summary: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8), sharey=True)
    for ax, stage in zip(axes, ["prefill", "decode"], strict=True):
        part = summary[summary["stage"] == stage].copy()
        labels = part["target_label"].tolist()
        x = np.arange(len(labels)); width = 0.36
        for offset, combo_name, label in [(-width / 2, "bf16_bf16_p50", "BF16/BF16"), (width / 2, "bf16_fp8_p50", "BF16/FP8")]:
            vals = part[combo_name].to_numpy(dtype=float)
            mask = np.isfinite(vals)
            ax.bar(x[mask] + offset, vals[mask], width, color=QUANT_COLORS[label], label=label)
        ax.axhline(1, color="#303030", linestyle="--", linewidth=1)
        ax.set_xticks(x, labels, rotation=35, ha="right"); ax.set_yscale("log"); ax.set_title(f"{stage}: target / refreshed H100 P50")
        ax.grid(True, axis="y", which="both", color="#E5E5E5", linewidth=0.6)
    axes[0].set_ylabel("latency ratio (log scale)"); axes[1].legend(frameon=False)
    fig.suptitle("Cross-hardware latency after exact shape/config matching", fontsize=15, fontweight="bold")
    fig.tight_layout(); fig.savefig(output, dpi=180, bbox_inches="tight"); fig.savefig(output.with_suffix(".pdf"), dpi=180, bbox_inches="tight"); plt.close(fig)


def write_gzip(frame: pd.DataFrame, path: Path) -> None:
    with gzip.open(path, "wt", encoding="utf-8", newline="") as stream: frame.to_csv(stream, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    all_points = []; pair_rows = []
    for stage in STAGES:
        reference = load_table(args.data_root, REFERENCE, stage)
        for target_spec in TARGETS:
            target = load_table(args.data_root, target_spec, stage)
            points, pair = match_pair(reference, target, stage, target_spec)
            all_points.append(points); pair_rows.append(pair)
    points = add_size_buckets(pd.concat(all_points, ignore_index=True))
    pair_summary = pd.DataFrame(pair_rows)
    points = points.sort_values(["stage", "target_label", *MATCH_KEYS]).reset_index(drop=True)
    write_gzip(points, args.output_dir / "matched_shape_config_points.csv.gz")
    pair_summary.to_csv(args.output_dir / "pair_coverage_summary.csv", index=False)
    summary = summarize(points)
    stage_order = {"prefill": 0, "decode": 1}
    target_order = {spec["label"]: index for index, spec in enumerate(TARGETS)}
    summary = summary.sort_values(
        ["stage", "target_label"],
        key=lambda column: column.map(stage_order if column.name == "stage" else target_order),
    ).reset_index(drop=True)
    summary.to_csv(args.output_dir / "hardware_latency_summary.csv", index=False)
    plot_scatter(points, "prefill", args.output_dir / "prefill_h100_ref_vs_original.png")
    plot_scatter(points, "decode", args.output_dir / "decode_h100_ref_vs_original.png")
    plot_fix_factor(points, args.output_dir / "h100_old_vs_refreshed_fix_factor.png")
    plot_hardware_summary(summary, args.output_dir / "hardware_p50_ratio_summary.png")
    print(f"Wrote {len(points):,} matched points and {len(summary)} summary rows to {args.output_dir}")


if __name__ == "__main__":
    main()
