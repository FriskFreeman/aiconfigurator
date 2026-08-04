#!/usr/bin/env python3
"""Plot matched SGLang/TRT-LLM GEMM latency, one point per (M, N, K)."""

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

QUANTIZATIONS = ("bfloat16", "fp8", "fp8_block", "nvfp4")
KEY_COLUMNS = ["gemm_dtype", "m", "n", "k"]
READ_COLUMNS = KEY_COLUMNS + ["latency"]
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


def load_perf_file(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return raw selected columns and database-compatible first-wins rows."""
    raw = pd.read_csv(path, usecols=READ_COLUMNS)
    raw["latency"] = pd.to_numeric(raw["latency"], errors="coerce")
    for column in ("m", "n", "k"):
        raw[column] = pd.to_numeric(raw[column], errors="coerce")
    raw = raw.dropna(subset=READ_COLUMNS)
    raw = raw[(raw["latency"] > 0) & (raw[["m", "n", "k"]] >= 0).all(axis=1)].copy()
    raw[["m", "n", "k"]] = raw[["m", "n", "k"]].astype(np.int64)
    deduplicated = raw.drop_duplicates(KEY_COLUMNS, keep="first").copy()
    return raw, deduplicated


def load_platform_data(data_root: Path) -> dict[str, dict[str, object]]:
    loaded: dict[str, dict[str, object]] = {}
    for platform in PLATFORMS:
        sglang_path = data_root / platform.data_dir / "sglang" / platform.sglang_version / "gemm_perf.txt"
        trtllm_path = data_root / platform.data_dir / "trtllm" / platform.trtllm_version / "gemm_perf.txt"
        for path in (sglang_path, trtllm_path):
            if not path.is_file():
                raise FileNotFoundError(f"Required GEMM data file does not exist: {path}")

        sglang_raw, sglang = load_perf_file(sglang_path)
        trtllm_raw, trtllm = load_perf_file(trtllm_path)
        loaded[platform.data_dir] = {
            "platform": platform,
            "sglang_path": sglang_path,
            "trtllm_path": trtllm_path,
            "sglang_raw": sglang_raw,
            "trtllm_raw": trtllm_raw,
            "sglang": sglang,
            "trtllm": trtllm,
        }
    return loaded


def match_dtype(data: dict[str, object], dtype: str) -> tuple[pd.DataFrame, dict[str, object]]:
    sglang_raw = data["sglang_raw"]
    trtllm_raw = data["trtllm_raw"]
    sglang = data["sglang"]
    trtllm = data["trtllm"]
    assert isinstance(sglang_raw, pd.DataFrame)
    assert isinstance(trtllm_raw, pd.DataFrame)
    assert isinstance(sglang, pd.DataFrame)
    assert isinstance(trtllm, pd.DataFrame)

    sglang_raw_dtype = sglang_raw[sglang_raw["gemm_dtype"] == dtype]
    trtllm_raw_dtype = trtllm_raw[trtllm_raw["gemm_dtype"] == dtype]
    sglang_dtype = sglang[sglang["gemm_dtype"] == dtype]
    trtllm_dtype = trtllm[trtllm["gemm_dtype"] == dtype]

    merged = sglang_dtype.merge(
        trtllm_dtype,
        on=KEY_COLUMNS,
        how="inner",
        suffixes=("_sglang", "_trtllm"),
        validate="one_to_one",
    )
    merged["trtllm_over_sglang"] = merged["latency_trtllm"] / merged["latency_sglang"]

    sglang_keys = pd.MultiIndex.from_frame(sglang_dtype[KEY_COLUMNS])
    trtllm_keys = pd.MultiIndex.from_frame(trtllm_dtype[KEY_COLUMNS])
    ratios = merged["trtllm_over_sglang"]
    platform = data["platform"]
    assert isinstance(platform, Platform)

    summary: dict[str, object] = {
        "hardware": platform.data_dir,
        "hardware_label": platform.display_name,
        "gemm_dtype": dtype,
        "sglang_version": platform.sglang_version,
        "trtllm_version": platform.trtllm_version,
        "sglang_dtype_rows": len(sglang_raw_dtype),
        "sglang_duplicate_rows": len(sglang_raw_dtype) - len(sglang_dtype),
        "trtllm_dtype_rows": len(trtllm_raw_dtype),
        "trtllm_duplicate_rows": len(trtllm_raw_dtype) - len(trtllm_dtype),
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


def plot_dtype(
    dtype: str,
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
    fig.suptitle(
        f"GEMM latency: SGLang vs TensorRT-LLM | {dtype}",
        fontsize=18,
        fontweight="bold",
    )
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
            ax.text(
                0.5,
                0.5,
                "No matched shapes",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=12,
                color="#666666",
            )
            continue

        trtllm_faster = frame["latency_trtllm"] < frame["latency_sglang"]
        for mask, color in ((trtllm_faster, TRTLLM_COLOR), (~trtllm_faster, SGLANG_COLOR)):
            subset = frame.loc[mask]
            ax.scatter(
                subset["latency_sglang"],
                subset["latency_trtllm"],
                s=3.0,
                alpha=0.20,
                color=color,
                edgecolors="none",
                rasterized=True,
                zorder=2,
            )
        plotted_points = int(trtllm_faster.sum() + (~trtllm_faster).sum())
        if plotted_points != len(frame):
            raise AssertionError(f"Not every matched shape was plotted for {platform.data_dir}/{dtype}")
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
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.035),
        ncol=3,
        frameon=False,
        fontsize=11,
    )
    fig.text(
        0.995,
        0.005,
        f"Every matched (M,N,K) is plotted; no sampling or aggregation. Total points: {total_points:,}",
        ha="right",
        va="bottom",
        fontsize=8,
        color="#555555",
    )

    stem = f"gemm_{dtype}_sglang_x_trtllm_y"
    fig.savefig(output_dir / f"{stem}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data_root = repo_root / "src" / "aiconfigurator" / "systems" / "data"

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.linewidth": 0.8,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "savefig.facecolor": "white",
        }
    )

    loaded = load_platform_data(data_root)
    all_summaries: list[dict[str, object]] = []
    for dtype in QUANTIZATIONS:
        matches: dict[str, pd.DataFrame] = {}
        summaries: dict[str, dict[str, object]] = {}
        for platform in PLATFORMS:
            matched, summary = match_dtype(loaded[platform.data_dir], dtype)
            matches[platform.data_dir] = matched
            summaries[platform.data_dir] = summary
            all_summaries.append(summary)
        plot_dtype(dtype, matches, summaries, output_dir, args.dpi)

    summary_frame = pd.DataFrame(all_summaries)
    summary_frame.to_csv(output_dir / "matched_shape_summary.csv", index=False, float_format="%.6f")
    print(f"Wrote {len(QUANTIZATIONS)} PNG files, {len(QUANTIZATIONS)} PDF files, and summary CSV to {output_dir}")


if __name__ == "__main__":
    main()
