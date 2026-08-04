#!/usr/bin/env python3
"""Analyze dtype effects in AIC Attention and base MLA silicon data.

The script intentionally excludes MLA module and WideEP module tables.  It
matches rows by the same keys used by the SDK query paths, then compares dtype
combinations within the same engine, hardware, phase, and exact shape.
"""

from __future__ import annotations

import argparse
import gzip
from dataclasses import dataclass
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


@dataclass(frozen=True)
class Platform:
    directory: str
    label: str


PLATFORMS = [
    Platform("a100_sxm", "A100 SXM"),
    Platform("l40s", "L40S"),
    Platform("h100_sxm", "H100 SXM"),
    Platform("h200_sxm", "H200 SXM"),
    Platform("b200_sxm", "B200 SXM"),
    Platform("b300_sxm", "B300 SXM"),
    Platform("gb200", "GB200"),
    Platform("gb300", "GB300"),
    Platform("rtx_pro_6000_server", "RTX PRO 6000"),
]

ENGINES = ["sglang", "trtllm", "vllm"]
ENGINE_LABELS = {"sglang": "SGLang", "trtllm": "TensorRT-LLM", "vllm": "vLLM"}
ENGINE_COLORS = {"sglang": "#15803D", "trtllm": "#C2413B", "vllm": "#2563A6"}
ENGINE_MARKERS = {"sglang": "o", "trtllm": "s", "vllm": "^"}

BASE_COMBO = ("bfloat16", "bfloat16")
COMBO_STYLES = {
    ("bfloat16", "fp8"): {"label": "BF16/FP8", "color": "#2563A6", "marker": "o"},
    ("fp8", "fp8"): {"label": "FP8/FP8", "color": "#C2413B", "marker": "s"},
    ("fp8", "bfloat16"): {"label": "FP8/BF16", "color": "#15803D", "marker": "^"},
}

# Explicit selection avoids silently changing results when a new data directory
# lands.  Base vLLM MLA was retired after 0.14.0, so MLA intentionally falls
# back to 0.14.0 where those files exist.
ATTENTION_VERSIONS = {
    "a100_sxm": {"sglang": "0.5.10", "trtllm": "1.0.0", "vllm": "0.14.0"},
    "l40s": {"sglang": "0.5.10", "trtllm": "1.0.0", "vllm": "0.14.0"},
    "h100_sxm": {"sglang": "0.5.10", "trtllm": "1.3.0rc10", "vllm": "0.19.0"},
    "h200_sxm": {"sglang": "0.5.10", "trtllm": "1.3.0rc10", "vllm": "0.19.0"},
    "b200_sxm": {"sglang": "0.5.10", "trtllm": "1.3.0rc10", "vllm": "0.19.0"},
    "b300_sxm": {"sglang": "0.5.10", "trtllm": "1.3.0rc10", "vllm": "0.19.0"},
    "gb200": {"sglang": "0.5.10", "trtllm": "1.3.0rc10", "vllm": "0.19.0"},
    "gb300": {"sglang": "0.5.10", "trtllm": "1.3.0rc10", "vllm": "0.19.0"},
    "rtx_pro_6000_server": {"sglang": "0.5.10", "trtllm": "1.3.0rc10", "vllm": "0.19.0"},
}

MLA_VERSIONS = {
    "a100_sxm": {"trtllm": "1.0.0", "vllm": "0.14.0"},
    "l40s": {"trtllm": "1.0.0", "vllm": "0.14.0"},
    "h100_sxm": {"sglang": "0.5.10", "trtllm": "1.3.0rc10", "vllm": "0.14.0"},
    "h200_sxm": {"sglang": "0.5.10", "trtllm": "1.3.0rc10", "vllm": "0.14.0"},
    "b200_sxm": {"sglang": "0.5.10", "trtllm": "1.3.0rc10"},
    "b300_sxm": {"sglang": "0.5.10", "trtllm": "1.3.0rc10"},
    "gb200": {"sglang": "0.5.10", "trtllm": "1.3.0rc10"},
    "gb300": {"sglang": "0.5.10", "trtllm": "1.3.0rc10"},
    "rtx_pro_6000_server": {"sglang": "0.5.10", "trtllm": "1.3.0rc10"},
}

FILE_SPECS = {
    ("attention", "prefill"): "context_attention_perf.txt",
    ("attention", "decode"): "generation_attention_perf.txt",
    ("mla", "prefill"): "context_mla_perf.txt",
    ("mla", "decode"): "generation_mla_perf.txt",
}

ATTENTION_PREFILL_KEYS = [
    "batch_size",
    "isl",
    "num_heads",
    "num_key_value_heads",
    "head_dim",
    "window_size",
]
ATTENTION_DECODE_KEYS = [
    "batch_size",
    "total_seq_len",
    "num_heads",
    "num_key_value_heads",
    "head_dim",
    "window_size",
]
# The SDK loader uses num_heads directly when that column exists.  tp_size is
# then provenance, not a query axis, and duplicate SDK keys are first-row-wins.
MLA_PREFILL_KEYS = ["num_heads", "batch_size", "isl"]
MLA_DECODE_KEYS = ["num_heads", "batch_size", "total_seq_len"]


@dataclass(frozen=True)
class Transition:
    name: str
    operator: str
    phase: str
    base_attn: str
    base_kv: str
    target_attn: str
    target_kv: str
    title: str
    short_title: str
    base_label: str
    target_label: str


TRANSITIONS = [
    Transition(
        "attention_prefill_bf16_fp8kv_vs_bf16_bf16kv",
        "attention",
        "prefill",
        "bfloat16",
        "bfloat16",
        "bfloat16",
        "fp8",
        "Attention prefill: BF16 input tag + FP8 KV vs BF16/BF16",
        "Attn prefill BF16/FP8KV vs BF16/BF16KV",
        "BF16 attn / BF16 KV",
        "BF16 attn / FP8 KV",
    ),
    Transition(
        "attention_prefill_fp8_fp8kv_vs_bf16_bf16kv",
        "attention",
        "prefill",
        "bfloat16",
        "bfloat16",
        "fp8",
        "fp8",
        "Attention prefill: FP8/FP8 vs BF16/BF16",
        "Attn prefill FP8/FP8KV vs BF16/BF16KV",
        "BF16 attn / BF16 KV",
        "FP8 attn / FP8 KV",
    ),
    Transition(
        "attention_prefill_fp8attn_vs_bf16attn_same_fp8kv",
        "attention",
        "prefill",
        "bfloat16",
        "fp8",
        "fp8",
        "fp8",
        "Attention prefill: FP8 attn tag vs BF16 attn tag (same FP8 KV)",
        "Attn prefill FP8 vs BF16 attn, same FP8KV",
        "BF16 attn / FP8 KV",
        "FP8 attn / FP8 KV",
    ),
    Transition(
        "attention_decode_fp8kv_vs_bf16kv",
        "attention",
        "decode",
        "bfloat16",
        "bfloat16",
        "bfloat16",
        "fp8",
        "Attention decode: FP8 KV vs BF16 KV",
        "Attn decode FP8KV vs BF16KV",
        "BF16 KV",
        "FP8 KV",
    ),
    Transition(
        "mla_prefill_fp8kv_vs_bf16kv",
        "mla",
        "prefill",
        "bfloat16",
        "bfloat16",
        "bfloat16",
        "fp8",
        "Base MLA prefill: FP8 KV vs BF16 KV",
        "Base MLA prefill FP8KV vs BF16KV",
        "BF16 KV",
        "FP8 KV",
    ),
    Transition(
        "mla_decode_fp8kv_vs_bf16kv",
        "mla",
        "decode",
        "bfloat16",
        "bfloat16",
        "bfloat16",
        "fp8",
        "Base MLA decode: FP8 KV vs BF16 KV",
        "Base MLA decode FP8KV vs BF16KV",
        "BF16 KV",
        "FP8 KV",
    ),
]


def normalize_dtype(value: object) -> str:
    value = str(value).strip().lower()
    aliases = {
        "bf16": "bfloat16",
        "torch.bfloat16": "bfloat16",
        "fp8_e4m3": "fp8",
        "float8_e4m3fn": "fp8",
        "torch.float8_e4m3fn": "fp8",
    }
    return aliases.get(value, value)


def query_keys(operator: str, phase: str) -> list[str]:
    if operator == "attention":
        return ATTENTION_PREFILL_KEYS if phase == "prefill" else ATTENTION_DECODE_KEYS
    return MLA_PREFILL_KEYS if phase == "prefill" else MLA_DECODE_KEYS


def selected_version(platform: str, engine: str, operator: str) -> str | None:
    versions = ATTENTION_VERSIONS if operator == "attention" else MLA_VERSIONS
    return versions.get(platform, {}).get(engine)


def read_selected_data(data_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    inventory_rows: list[dict[str, object]] = []
    kernel_rows: list[dict[str, object]] = []

    for platform in PLATFORMS:
        for engine in ENGINES:
            for (operator, phase), filename in FILE_SPECS.items():
                version = selected_version(platform.directory, engine, operator)
                if version is None:
                    continue
                path = data_root / platform.directory / engine / version / filename
                if not path.exists():
                    continue
                frame = pd.read_csv(path)
                frame["window_size"] = frame.get("window_size", 0)
                frame["beam_width"] = frame.get("beam_width", 1)
                frame["total_seq_len"] = frame["isl"] + frame["step"]
                dtype_col = "attn_dtype" if operator == "attention" else "mla_dtype"
                frame["compute_tag"] = frame[dtype_col].map(normalize_dtype)
                frame["kv_cache_dtype"] = frame["kv_cache_dtype"].map(normalize_dtype)
                frame["hardware"] = platform.directory
                frame["hardware_label"] = platform.label
                frame["engine"] = engine
                frame["engine_label"] = ENGINE_LABELS[engine]
                frame["selected_version"] = version
                frame["operator"] = operator
                frame["phase"] = phase
                frame["source_file"] = str(path.relative_to(REPO_ROOT))
                frames.append(frame)

                combo_counts = (
                    frame.groupby(["compute_tag", "kv_cache_dtype"], dropna=False).size().sort_index()
                )
                inventory_rows.append(
                    {
                        "hardware": platform.directory,
                        "hardware_label": platform.label,
                        "engine": engine,
                        "version": version,
                        "operator": operator,
                        "phase": phase,
                        "rows": len(frame),
                        "dtype_combinations": "; ".join(
                            f"{attn}/{kv}:{count}" for (attn, kv), count in combo_counts.items()
                        ),
                        "kernel_sources": "; ".join(
                            f"{kernel}:{count}" for kernel, count in frame["kernel_source"].value_counts().items()
                        ),
                        "source_file": str(path.relative_to(REPO_ROOT)),
                    }
                )

                grouped = (
                    frame.groupby(["compute_tag", "kv_cache_dtype", "kernel_source"], dropna=False)
                    .size()
                    .reset_index(name="rows")
                )
                for row in grouped.to_dict("records"):
                    kernel_rows.append(
                        {
                            "hardware": platform.directory,
                            "hardware_label": platform.label,
                            "engine": engine,
                            "version": version,
                            "operator": operator,
                            "phase": phase,
                            **row,
                        }
                    )

    if not frames:
        raise RuntimeError(f"No data files found below {data_root}")
    return pd.concat(frames, ignore_index=True), pd.DataFrame(inventory_rows), pd.DataFrame(kernel_rows)


def first_wins(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    # This mirrors the SDK loader's first-row-wins behavior for duplicate keys.
    return frame.drop_duplicates(keys, keep="first")


def match_transition(data: pd.DataFrame, transition: Transition) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    relevant = data[(data["operator"] == transition.operator) & (data["phase"] == transition.phase)]
    keys = query_keys(transition.operator, transition.phase)
    matches: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []

    for platform in PLATFORMS:
        for engine in ENGINES:
            group = relevant[(relevant["hardware"] == platform.directory) & (relevant["engine"] == engine)]
            if group.empty:
                continue
            base_raw = group[
                (group["compute_tag"] == transition.base_attn)
                & (group["kv_cache_dtype"] == transition.base_kv)
            ]
            target_raw = group[
                (group["compute_tag"] == transition.target_attn)
                & (group["kv_cache_dtype"] == transition.target_kv)
            ]
            base = first_wins(base_raw, keys)
            target = first_wins(target_raw, keys)
            merged = base.merge(target, on=keys, how="inner", suffixes=("_base", "_target"))
            merged["latency_ratio"] = merged["latency_target"] / merged["latency_base"]
            merged["transition"] = transition.name
            merged["operator"] = transition.operator
            merged["phase"] = transition.phase
            merged["hardware"] = platform.directory
            merged["hardware_label"] = platform.label
            merged["engine"] = engine
            merged["engine_label"] = ENGINE_LABELS[engine]
            merged["base_compute_tag"] = transition.base_attn
            merged["base_kv_cache_dtype"] = transition.base_kv
            merged["target_compute_tag"] = transition.target_attn
            merged["target_kv_cache_dtype"] = transition.target_kv
            if not merged.empty:
                matches.append(merged)

            base_key_set = set(map(tuple, base[keys].to_numpy()))
            target_key_set = set(map(tuple, target[keys].to_numpy()))
            ratios = merged["latency_ratio"]
            summaries.append(
                {
                    "transition": transition.name,
                    "operator": transition.operator,
                    "phase": transition.phase,
                    "hardware": platform.directory,
                    "hardware_label": platform.label,
                    "engine": engine,
                    "version": group["selected_version"].iloc[0],
                    "base_rows": len(base_raw),
                    "base_unique_shapes": len(base),
                    "target_rows": len(target_raw),
                    "target_unique_shapes": len(target),
                    "matched_shapes": len(merged),
                    "base_only_shapes": len(base_key_set - target_key_set),
                    "target_only_shapes": len(target_key_set - base_key_set),
                    "ratio_p10": ratios.quantile(0.10) if len(ratios) else np.nan,
                    "ratio_p50": ratios.quantile(0.50) if len(ratios) else np.nan,
                    "ratio_p90": ratios.quantile(0.90) if len(ratios) else np.nan,
                    "target_faster_pct": (ratios < 1).mean() * 100 if len(ratios) else np.nan,
                    "latency_exact_equal_pct": (
                        (merged["latency_base"] == merged["latency_target"]).mean() * 100
                        if len(merged)
                        else np.nan
                    ),
                    "latency_within_1pct_pct": (
                        (np.abs(ratios - 1.0) <= 0.01).mean() * 100 if len(ratios) else np.nan
                    ),
                    "kernel_source_changed_pct": (
                        (merged["kernel_source_base"] != merged["kernel_source_target"]).mean() * 100
                        if len(merged)
                        else np.nan
                    ),
                    "base_kernel_sources": "; ".join(
                        f"{name}:{count}" for name, count in base["kernel_source"].value_counts().items()
                    ),
                    "target_kernel_sources": "; ".join(
                        f"{name}:{count}" for name, count in target["kernel_source"].value_counts().items()
                    ),
                }
            )

    if matches:
        return pd.concat(matches, ignore_index=True), summaries
    return pd.DataFrame(), summaries


def log_limits(frame: pd.DataFrame) -> tuple[float, float]:
    values = frame[["latency_base", "latency_target"]].to_numpy(dtype=float).ravel()
    values = values[np.isfinite(values) & (values > 0)]
    lower = 10 ** (np.floor(np.log10(values.min()) * 4) / 4)
    upper = 10 ** (np.ceil(np.log10(values.max()) * 4) / 4)
    return float(lower), float(upper)


def workload_proxy(frame: pd.DataFrame) -> pd.Series:
    """Return a monotonic shape-size proxy independent of measured latency."""
    if frame["operator"].iloc[0] == "attention":
        if frame["phase"].iloc[0] == "prefill":
            attended_s = np.where(
                frame["window_size"].to_numpy() > 0,
                np.minimum(frame["isl"].to_numpy(), frame["window_size"].to_numpy()),
                frame["isl"].to_numpy(),
            )
            return (
                frame["batch_size"]
                * frame["num_heads"]
                * frame["head_dim"]
                * frame["isl"]
                * attended_s
            ).astype(float)
        return (
            frame["batch_size"]
            * frame["num_heads"]
            * frame["head_dim"]
            * frame["total_seq_len"]
        ).astype(float)
    if frame["phase"].iloc[0] == "prefill":
        return (frame["batch_size"] * frame["num_heads"] * frame["isl"] ** 2).astype(float)
    return (frame["batch_size"] * frame["num_heads"] * frame["total_seq_len"]).astype(float)


def add_size_buckets(frame: pd.DataFrame) -> pd.DataFrame:
    """Classify the bottom/top workload quartiles within each source table."""
    result = frame.copy()
    result["workload_proxy"] = np.nan
    result["size_bucket"] = "medium"
    table_cols = ["operator", "phase", "hardware", "engine"]
    for _, index in result.groupby(table_cols, sort=False).groups.items():
        table = result.loc[index]
        proxy = workload_proxy(table)
        result.loc[index, "workload_proxy"] = proxy
        keys = query_keys(table["operator"].iloc[0], table["phase"].iloc[0])
        unique_shapes = table.assign(_workload=proxy).drop_duplicates(keys, keep="first")
        q25, q75 = unique_shapes["_workload"].quantile([0.25, 0.75])
        result.loc[index, "size_bucket"] = np.select(
            [proxy <= q25, proxy >= q75],
            ["small", "large"],
            default="medium",
        )
    return result


def combo_label(compute_tag: str, kv_cache_dtype: str) -> str:
    style = COMBO_STYLES.get((compute_tag, kv_cache_dtype))
    if style is not None:
        return str(style["label"])
    return f"{compute_tag.upper()}/{kv_cache_dtype.upper()}"


def size_pattern(small_p50: float, large_p50: float) -> str:
    if not np.isfinite(small_p50) or not np.isfinite(large_p50):
        return "insufficient"
    if small_p50 > 1 and large_p50 < 1:
        return "small_slower_large_faster"
    if small_p50 < 1 and large_p50 > 1:
        return "small_faster_large_slower"
    if small_p50 <= 1 and large_p50 <= 1:
        return "target_faster_both"
    if small_p50 >= 1 and large_p50 >= 1:
        return "target_slower_both"
    return "near_equal_or_mixed"


def summarize_by_size(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_cols = ["transition", "operator", "phase", "hardware", "hardware_label", "engine"]
    for keys, group in frame.groupby(group_cols, sort=False):
        values = dict(zip(group_cols, keys, strict=True))
        ratios = group["latency_ratio"]
        row: dict[str, object] = {
            **values,
            "version": group["selected_version_base"].iloc[0],
            "target_combination": combo_label(
                group["target_compute_tag"].iloc[0], group["target_kv_cache_dtype"].iloc[0]
            ),
            "matched_shapes": len(group),
            "ratio_p10": ratios.quantile(0.10),
            "ratio_p50": ratios.quantile(0.50),
            "ratio_p90": ratios.quantile(0.90),
            "target_faster_pct": (ratios < 1).mean() * 100,
            "kernel_source_changed_pct": (
                group["kernel_source_base"] != group["kernel_source_target"]
            ).mean()
            * 100,
            "base_kernel_sources": "; ".join(
                f"{name}:{count}" for name, count in group["kernel_source_base"].value_counts().items()
            ),
            "target_kernel_sources": "; ".join(
                f"{name}:{count}" for name, count in group["kernel_source_target"].value_counts().items()
            ),
        }
        for bucket in ("small", "large"):
            bucket_ratios = group.loc[group["size_bucket"] == bucket, "latency_ratio"]
            row[f"{bucket}_shapes"] = len(bucket_ratios)
            row[f"{bucket}_ratio_p10"] = bucket_ratios.quantile(0.10)
            row[f"{bucket}_ratio_p50"] = bucket_ratios.quantile(0.50)
            row[f"{bucket}_ratio_p90"] = bucket_ratios.quantile(0.90)
            row[f"{bucket}_target_faster_pct"] = (bucket_ratios < 1).mean() * 100
        row["large_minus_small_p50"] = row["large_ratio_p50"] - row["small_ratio_p50"]
        row["size_pattern"] = size_pattern(row["small_ratio_p50"], row["large_ratio_p50"])
        ranked_workload = group["workload_proxy"].rank(method="average")
        ranked_ratio = group["latency_ratio"].rank(method="average")
        row["workload_ratio_spearman"] = ranked_workload.corr(ranked_ratio)
        rows.append(row)
    return pd.DataFrame(rows)


def plot_engine_dtype_combinations(
    frame: pd.DataFrame,
    summary: pd.DataFrame,
    operator: str,
    phase: str,
    engine: str,
    output_dir: Path,
    dpi: int,
) -> None:
    selected = frame[
        (frame["operator"] == operator) & (frame["phase"] == phase) & (frame["engine"] == engine)
    ]
    if selected.empty:
        return
    lower, upper = log_limits(selected)
    fig, axes = plt.subplots(3, 3, figsize=(15.5, 15.5), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.91, bottom=0.09, wspace=0.15, hspace=0.22)
    operator_label = "Attention" if operator == "attention" else "Base MLA"
    fig.suptitle(
        f"{ENGINE_LABELS[engine]} {operator_label} {phase}: dtype combinations vs BF16/BF16",
        fontsize=17,
        fontweight="bold",
    )
    fig.supxlabel("Baseline latency: BF16/BF16 (ms, log scale)", fontsize=13)
    fig.supylabel("Other dtype-combination latency (ms, log scale)", fontsize=13)

    legend_handles = []
    for ax, platform in zip(axes.flat, PLATFORMS, strict=True):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lower, upper)
        ax.set_ylim(lower, upper)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, which="major", color="#D4D4D4", linewidth=0.6)
        ax.grid(True, which="minor", color="#EEEEEE", linewidth=0.35)
        ax.plot([lower, upper], [lower, upper], color="#303030", linestyle="--", linewidth=1.0, zorder=1)
        panel = selected[selected["hardware"] == platform.directory]
        panel_notes = []
        kernel_notes = []
        target_combos = (
            panel[["target_compute_tag", "target_kv_cache_dtype"]]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )
        for target_combo in target_combos:
            points = panel[
                (panel["target_compute_tag"] == target_combo[0])
                & (panel["target_kv_cache_dtype"] == target_combo[1])
            ]
            if points.empty:
                continue
            style = COMBO_STYLES.get(
                target_combo,
                {"label": combo_label(*target_combo), "color": "#555555", "marker": "D"},
            )
            handle = ax.scatter(
                points["latency_base"],
                points["latency_target"],
                s=8,
                alpha=0.32,
                marker=style["marker"],
                color=style["color"],
                edgecolors="none",
                rasterized=True,
                label=style["label"],
                zorder=2,
            )
            if not any(item.get_label() == handle.get_label() for item in legend_handles):
                legend_handles.append(handle)
            stats = summary[
                (summary["operator"] == operator)
                & (summary["phase"] == phase)
                & (summary["engine"] == engine)
                & (summary["hardware"] == platform.directory)
                & (summary["target_combination"] == style["label"])
            ].iloc[0]
            panel_notes.append(
                f"{style['label']} n={len(points):,}\n"
                f"  all p10/50/90={stats['ratio_p10']:.3f}/{stats['ratio_p50']:.3f}/{stats['ratio_p90']:.3f}x\n"
                f"  small/large p50={stats['small_ratio_p50']:.3f}/{stats['large_ratio_p50']:.3f}x"
            )
            base_sources = "/".join(sorted(points["kernel_source_base"].unique()))
            target_sources = "/".join(sorted(points["kernel_source_target"].unique()))
            kernel_notes.append(base_sources if base_sources == target_sources else f"{base_sources}->{target_sources}")
        version = panel["selected_version_base"].iloc[0] if not panel.empty else ""
        title = f"{platform.label} | {version}" if version else platform.label
        if kernel_notes:
            title += "\n" + "; ".join(dict.fromkeys(kernel_notes))
        ax.set_title(title, fontsize=10)
        if panel_notes:
            ax.text(
                0.03,
                0.97,
                "\n".join(panel_notes),
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=8,
                bbox={"facecolor": "white", "edgecolor": "#CCCCCC", "alpha": 0.82, "pad": 3},
            )
        else:
            ax.text(0.5, 0.5, "No matched dtype pairs", transform=ax.transAxes, ha="center", va="center", color="#777777")

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=max(1, len(legend_handles)),
        frameon=False,
        bbox_to_anchor=(0.5, 0.025),
    )
    fig.text(
        0.5,
        0.057,
        "Each panel is one selected .txt table. Small/large are bottom/top workload quartiles; below diagonal: target is faster.",
        ha="center",
        fontsize=10,
        color="#444444",
    )
    stem = f"{operator}_{phase}_{engine}_dtype_combinations"
    fig.savefig(output_dir / f"{stem}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def write_gzip_csv(frame: pd.DataFrame, path: Path) -> None:
    with gzip.open(path, "wt", encoding="utf-8", newline="") as stream:
        frame.to_csv(stream, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data, inventory, kernels = read_selected_data(args.data_root.resolve())
    inventory.to_csv(args.output_dir / "data_inventory.csv", index=False)
    kernels.to_csv(args.output_dir / "kernel_source_by_dtype.csv", index=False)

    all_matches = []
    summary_rows = []
    for transition in TRANSITIONS:
        matched, summaries = match_transition(data, transition)
        summary_rows.extend(summaries)
        if not matched.empty:
            all_matches.append(matched)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.output_dir / "dtype_pair_summary.csv", index=False)
    if all_matches:
        combined_matches = pd.concat(all_matches, ignore_index=True)
        write_gzip_csv(combined_matches, args.output_dir / "matched_dtype_points.csv.gz")
        baseline_matches = combined_matches[
            (combined_matches["base_compute_tag"] == BASE_COMBO[0])
            & (combined_matches["base_kv_cache_dtype"] == BASE_COMBO[1])
        ]
        baseline_matches = add_size_buckets(baseline_matches)
        write_gzip_csv(
            baseline_matches,
            args.output_dir / "baseline_comparison_points.csv.gz",
        )
        size_summary = summarize_by_size(baseline_matches)
        size_summary.to_csv(args.output_dir / "size_stratified_summary.csv", index=False)
        for operator, phase in FILE_SPECS:
            for engine in ENGINES:
                plot_engine_dtype_combinations(
                    baseline_matches,
                    size_summary,
                    operator,
                    phase,
                    engine,
                    args.output_dir,
                    args.dpi,
                )

    print(f"Read {len(data):,} selected rows")
    print(f"Wrote results to {args.output_dir}")


if __name__ == "__main__":
    main()
