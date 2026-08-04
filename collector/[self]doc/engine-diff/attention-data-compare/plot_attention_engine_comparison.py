#!/usr/bin/env python3
"""Plot exact matched SGLang/TRT-LLM attention shapes, without sampling."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


@dataclass(frozen=True)
class Platform:
    data_dir: str
    display_name: str
    sglang_version: str
    trtllm_version: str


@dataclass(frozen=True)
class Category:
    name: str
    phase: str
    attn_dtype: str | None
    kv_cache_dtype: str
    title: str


PLATFORMS = (
    Platform("a100_sxm", "A100 SXM", "0.5.10", "1.0.0"),
    Platform("l40s", "L40S", "0.5.10", "1.0.0"),
    Platform("h100_sxm", "H100 SXM", "0.5.10", "1.3.0rc10"),
    Platform("h200_sxm", "H200 SXM", "0.5.10", "1.3.0rc10"),
    Platform("b200_sxm", "B200 SXM", "0.5.10", "1.3.0rc10"),
    Platform("b300_sxm", "B300 SXM", "0.5.10", "1.3.0rc10"),
    Platform("gb200", "GB200", "0.5.10", "1.3.0rc10"),
    Platform("gb300", "GB300", "0.5.10", "1.3.0rc10"),
    Platform("rtx_pro_6000_server", "RTX PRO 6000", "0.5.10", "1.3.0rc10"),
)

CATEGORIES = (
    Category(
        "context_bfloat16_kv_bfloat16",
        "context",
        "bfloat16",
        "bfloat16",
        "Context attention | BF16 compute, BF16 KV cache",
    ),
    Category(
        "context_bfloat16_kv_fp8",
        "context",
        "bfloat16",
        "fp8",
        "Context attention | BF16 input tag, FP8 KV cache",
    ),
    Category(
        "context_fp8_kv_fp8",
        "context",
        "fp8",
        "fp8",
        "Context attention | FP8 compute, FP8 KV cache",
    ),
    Category(
        "generation_kv_bfloat16",
        "generation",
        None,
        "bfloat16",
        "Generation attention | BF16 KV cache",
    ),
    Category(
        "generation_kv_fp8",
        "generation",
        None,
        "fp8",
        "Generation attention | FP8 KV cache",
    ),
)

COMMON_DIMENSIONS = [
    "batch_size",
    "num_heads",
    "num_key_value_heads",
    "head_dim",
    "window_size",
]
CONTEXT_KEY = ["attn_dtype", "kv_cache_dtype", "isl", *COMMON_DIMENSIONS]
# The SDK generation loader intentionally ignores attn_dtype and indexes by
# total sequence length (isl + step). Current collectors always log isl=1.
GENERATION_KEY = ["kv_cache_dtype", "total_seq_len", *COMMON_DIMENSIONS]

SGLANG_COLOR = "#087F8C"
TRTLLM_COLOR = "#C14953"
EQUAL_COLOR = "#202124"


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_root = script_dir.parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=default_root)
    parser.add_argument("--output-dir", type=Path, default=script_dir / "results")
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def load_perf_file(path: Path, phase: str) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Return normalized raw rows, SDK-compatible first-wins rows, and key."""
    raw = pd.read_csv(path)
    if "window_size" not in raw.columns:
        raw["window_size"] = 0

    numeric_columns = [*COMMON_DIMENSIONS, "isl", "step", "latency"]
    for column in numeric_columns:
        raw[column] = pd.to_numeric(raw[column], errors="coerce")
    required = ["attn_dtype", "kv_cache_dtype", *numeric_columns]
    raw = raw.dropna(subset=required)
    raw = raw[raw["latency"] > 0].copy()
    for column in [*COMMON_DIMENSIONS, "isl", "step"]:
        raw[column] = raw[column].astype(np.int64)
    raw["total_seq_len"] = raw["isl"] + raw["step"]

    key = CONTEXT_KEY if phase == "context" else GENERATION_KEY
    deduplicated = raw.drop_duplicates(key, keep="first").copy()
    return raw, deduplicated, key


def load_platform_data(data_root: Path) -> dict[tuple[str, str], dict[str, object]]:
    loaded: dict[tuple[str, str], dict[str, object]] = {}
    for platform in PLATFORMS:
        for phase, filename in (
            ("context", "context_attention_perf.txt"),
            ("generation", "generation_attention_perf.txt"),
        ):
            sglang_path = data_root / platform.data_dir / "sglang" / platform.sglang_version / filename
            trtllm_path = data_root / platform.data_dir / "trtllm" / platform.trtllm_version / filename
            for path in (sglang_path, trtllm_path):
                if not path.is_file():
                    raise FileNotFoundError(f"Required attention data file does not exist: {path}")

            sglang_raw, sglang, key = load_perf_file(sglang_path, phase)
            trtllm_raw, trtllm, trt_key = load_perf_file(trtllm_path, phase)
            if key != trt_key:
                raise AssertionError(f"Internal key mismatch for {platform.data_dir}/{phase}")
            loaded[(platform.data_dir, phase)] = {
                "platform": platform,
                "phase": phase,
                "key": key,
                "sglang_path": sglang_path,
                "trtllm_path": trtllm_path,
                "sglang_raw": sglang_raw,
                "trtllm_raw": trtllm_raw,
                "sglang": sglang,
                "trtllm": trtllm,
            }
    return loaded


def select_category(frame: pd.DataFrame, category: Category) -> pd.DataFrame:
    selected = frame[frame["kv_cache_dtype"] == category.kv_cache_dtype]
    if category.attn_dtype is not None:
        selected = selected[selected["attn_dtype"] == category.attn_dtype]
    return selected


def match_category(data: dict[str, object], category: Category) -> tuple[pd.DataFrame, dict[str, object]]:
    key = data["key"]
    sglang_raw = data["sglang_raw"]
    trtllm_raw = data["trtllm_raw"]
    sglang = data["sglang"]
    trtllm = data["trtllm"]
    platform = data["platform"]
    assert isinstance(key, list)
    assert isinstance(sglang_raw, pd.DataFrame)
    assert isinstance(trtllm_raw, pd.DataFrame)
    assert isinstance(sglang, pd.DataFrame)
    assert isinstance(trtllm, pd.DataFrame)
    assert isinstance(platform, Platform)

    sglang_raw_cat = select_category(sglang_raw, category)
    trtllm_raw_cat = select_category(trtllm_raw, category)
    sglang_cat = select_category(sglang, category)
    trtllm_cat = select_category(trtllm, category)

    merged = sglang_cat.merge(
        trtllm_cat,
        on=key,
        how="inner",
        suffixes=("_sglang", "_trtllm"),
        validate="one_to_one",
    )
    merged["trtllm_over_sglang"] = merged["latency_trtllm"] / merged["latency_sglang"]

    sglang_keys = pd.MultiIndex.from_frame(sglang_cat[key])
    trtllm_keys = pd.MultiIndex.from_frame(trtllm_cat[key])
    ratios = merged["trtllm_over_sglang"]
    summary: dict[str, object] = {
        "hardware": platform.data_dir,
        "hardware_label": platform.display_name,
        "category": category.name,
        "phase": category.phase,
        "attn_dtype": category.attn_dtype or "not_a_query_axis",
        "kv_cache_dtype": category.kv_cache_dtype,
        "sglang_version": platform.sglang_version,
        "trtllm_version": platform.trtllm_version,
        "sglang_rows": len(sglang_raw_cat),
        "sglang_unique_shapes": len(sglang_cat),
        "sglang_duplicate_rows": len(sglang_raw_cat) - len(sglang_cat),
        "trtllm_rows": len(trtllm_raw_cat),
        "trtllm_unique_shapes": len(trtllm_cat),
        "trtllm_duplicate_rows": len(trtllm_raw_cat) - len(trtllm_cat),
        "matched_shapes": len(merged),
        "sglang_only_shapes": len(sglang_keys.difference(trtllm_keys)),
        "trtllm_only_shapes": len(trtllm_keys.difference(sglang_keys)),
        "ratio_p10": ratios.quantile(0.10) if len(ratios) else np.nan,
        "ratio_p50": ratios.quantile(0.50) if len(ratios) else np.nan,
        "ratio_p90": ratios.quantile(0.90) if len(ratios) else np.nan,
        "trtllm_faster_pct": (ratios < 1).mean() * 100 if len(ratios) else np.nan,
    }
    return merged, summary


def padded_log_limits(matched_frames: list[pd.DataFrame]) -> tuple[float, float]:
    values = np.concatenate(
        [
            frame[["latency_sglang", "latency_trtllm"]].to_numpy(dtype=float).ravel()
            for frame in matched_frames
            if not frame.empty
        ]
    )
    lower = 10 ** (np.floor(np.log10(values.min()) * 4) / 4)
    upper = 10 ** (np.ceil(np.log10(values.max()) * 4) / 4)
    return float(lower), float(upper)


def plot_category(
    category: Category,
    platform_matches: dict[str, pd.DataFrame],
    summaries: dict[str, dict[str, object]],
    output_dir: Path,
    dpi: int,
) -> None:
    nonempty = [frame for frame in platform_matches.values() if not frame.empty]
    if not nonempty:
        return
    lower, upper = padded_log_limits(nonempty)

    fig, axes = plt.subplots(3, 3, figsize=(15.5, 15.5), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.915, bottom=0.125, wspace=0.15, hspace=0.22)
    fig.suptitle(f"SGLang vs TensorRT-LLM | {category.title}", fontsize=18, fontweight="bold")
    fig.supxlabel("SGLang latency (ms, log scale)", fontsize=14, y=0.080)
    fig.supylabel("TensorRT-LLM latency (ms, log scale)", fontsize=14)

    total_points = 0
    for ax, platform in zip(axes.flat, PLATFORMS, strict=True):
        frame = platform_matches[platform.data_dir]
        summary = summaries[platform.data_dir]
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lower, upper)
        ax.set_ylim(lower, upper)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, which="major", color="#D9D9D9", linewidth=0.6)
        ax.grid(True, which="minor", color="#EEEEEE", linewidth=0.35)
        ax.plot([lower, upper], [lower, upper], color=EQUAL_COLOR, linestyle="--", linewidth=1.0, zorder=1)
        ax.set_title(
            f"{platform.display_name}\nSGL {platform.sglang_version} | TRT {platform.trtllm_version}",
            fontsize=11,
        )

        if frame.empty:
            ax.text(0.5, 0.5, "No matched shapes", transform=ax.transAxes, ha="center", va="center", color="#666666")
            continue

        trtllm_faster = frame["latency_trtllm"] < frame["latency_sglang"]
        for mask, color in ((trtllm_faster, TRTLLM_COLOR), (~trtllm_faster, SGLANG_COLOR)):
            subset = frame.loc[mask]
            ax.scatter(
                subset["latency_sglang"],
                subset["latency_trtllm"],
                s=4.0,
                alpha=0.22,
                color=color,
                edgecolors="none",
                rasterized=True,
                zorder=2,
            )
        plotted_points = int(trtllm_faster.sum() + (~trtllm_faster).sum())
        if plotted_points != len(frame):
            raise AssertionError(f"Not every matched shape was plotted for {platform.data_dir}/{category.name}")
        total_points += plotted_points
        ax.text(
            0.03,
            0.97,
            f"n={len(frame):,}\nmedian TRT/SGL={summary['ratio_p50']:.3f}\nTRT faster={summary['trtllm_faster_pct']:.1f}%",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.88, "pad": 3},
        )

    legend_handles = [
        Line2D([], [], marker="o", linestyle="None", markersize=6, color=TRTLLM_COLOR, label="TRT-LLM faster (y < x)"),
        Line2D([], [], marker="o", linestyle="None", markersize=6, color=SGLANG_COLOR, label="SGLang faster or equal (y >= x)"),
        Line2D([], [], color=EQUAL_COLOR, linestyle="--", linewidth=1.2, label="Equal latency (y = x)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, 0.035), ncol=3, frameon=False, fontsize=11)
    shape_label = "(B,ISL,H,Hkv,D,window)" if category.phase == "context" else "(B,total_seq,H,Hkv,D,window)"
    fig.text(
        0.995,
        0.005,
        f"Every exact matched {shape_label} is plotted; no sampling or aggregation. Total points: {total_points:,}",
        ha="right",
        va="bottom",
        fontsize=8,
        color="#555555",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(output_dir / f"{category.name}.{extension}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    data_root = args.repo_root / "src" / "aiconfigurator" / "systems" / "data"
    loaded = load_platform_data(data_root)
    all_summaries: list[dict[str, object]] = []

    for category in CATEGORIES:
        platform_matches: dict[str, pd.DataFrame] = {}
        category_summaries: dict[str, dict[str, object]] = {}
        for platform in PLATFORMS:
            data = loaded[(platform.data_dir, category.phase)]
            matched, summary = match_category(data, category)
            platform_matches[platform.data_dir] = matched
            category_summaries[platform.data_dir] = summary
            all_summaries.append(summary)
        plot_category(category, platform_matches, category_summaries, args.output_dir, args.dpi)

    summary_frame = pd.DataFrame(all_summaries)
    summary_frame.to_csv(args.output_dir / "matched_shape_summary.csv", index=False, float_format="%.6f")
    print(summary_frame.to_string(index=False))


if __name__ == "__main__":
    main()
