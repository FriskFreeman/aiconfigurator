#!/usr/bin/env python3
"""Re-query AIC GEMM data with real mixed-batch projection shapes.

The run_agg MLA diagnostic path uses a synthetic first-pass `query_s`.
For qkv_a/q_b/o projection GEMMs, the real SGLang mixed batch executes on the
actual formal batch token count captured by Nsight.  This script rebuilds the
projection-only comparison using that real token count as GEMM M.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def find_repo_root(start: Path) -> Path:
    for parent in [start, *start.parents]:
        if (parent / "src" / "aiconfigurator").is_dir():
            return parent
    raise RuntimeError(f"Cannot find repo root from {start}")


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = find_repo_root(SCRIPT_PATH)
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from aiconfigurator.sdk import common  # noqa: E402
from aiconfigurator.sdk.perf_database import PerfDatabase  # noqa: E402


OP_SPECS = {
    "qkv_a_proj": {
        "component": "fused_qkv_a_proj_with_mqa",
        "n": 2112,
        "k": 7168,
        "aic_op": "context_downscale_gemm",
    },
    "q_b_proj": {
        "component": "q_b_proj",
        "n": 24576,
        "k": 1536,
        "aic_op": "context_q_b_proj_gemm",
    },
    "o_proj": {
        "component": "o_proj",
        "n": 7168,
        "k": 16384,
        "aic_op": "context_proj_gemm",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=SCRIPT_PATH.parent,
        help="Directory containing run_agg MLA operator comparison CSVs.",
    )
    parser.add_argument("--system", default="h100_sxm")
    parser.add_argument("--backend", default="sglang")
    parser.add_argument("--version", default="0.5.9")
    parser.add_argument(
        "--systems-root",
        type=Path,
        default=REPO_ROOT / "src" / "aiconfigurator" / "systems",
    )
    return parser.parse_args()


def safe_float(value: object) -> float:
    if value is None:
        return math.nan
    try:
        return float(value)
    except Exception:
        return math.nan


def load_gemm_table(systems_root: Path, system: str, backend: str, version: str) -> pd.DataFrame:
    path = systems_root / "data" / system / backend / version / "gemm_perf.txt"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def nearest_m_values(gemm_df: pd.DataFrame, m: int, n: int, k: int, limit: int = 4) -> str:
    sub = gemm_df[(gemm_df["gemm_dtype"] == "fp8_block") & (gemm_df["n"] == n) & (gemm_df["k"] == k)]
    if sub.empty:
        return ""
    vals = sorted(int(v) for v in sub["m"].dropna().unique())
    vals = sorted(vals, key=lambda x: (abs(x - m), x))[:limit]
    return ",".join(str(v) for v in vals)


def exact_in_gemm_perf(gemm_df: pd.DataFrame, m: int, n: int, k: int) -> bool:
    sub = gemm_df[
        (gemm_df["gemm_dtype"] == "fp8_block")
        & (gemm_df["m"] == m)
        & (gemm_df["n"] == n)
        & (gemm_df["k"] == k)
    ]
    return not sub.empty


def extract_first_query_s(row: pd.Series) -> float:
    raw = row.get("aic_query_shapes_json")
    if not isinstance(raw, str) or not raw:
        return math.nan
    try:
        outer = json.loads(raw)
        if not outer:
            return math.nan
        inner = json.loads(outer[0]) if isinstance(outer[0], str) else outer[0]
        if not inner:
            return math.nan
        return safe_float(inner[0].get("s"))
    except Exception:
        return math.nan


def build_compare(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    analysis_dir = args.analysis_dir
    real_child = pd.read_csv(analysis_dir / "real_mla_child_long.csv")
    real_kernel = pd.read_csv(analysis_dir / "real_mla_kernel_long.csv")
    compare = pd.read_csv(analysis_dir / "mla_operator_compare_long.csv")
    cases = pd.read_csv(analysis_dir / "case_label_map.csv")
    gemm_df = load_gemm_table(args.systems_root, args.system, args.backend, args.version)
    db = PerfDatabase(
        system=args.system,
        backend=args.backend,
        version=args.version,
        systems_root=str(args.systems_root),
    )

    rows: list[dict[str, object]] = []
    for _, case in cases.sort_values("case_id").iterrows():
        tag = case["tag"]
        case_id = case["case_id"]
        child_case = real_child[real_child["tag"] == tag]
        kernel_case = real_kernel[real_kernel["tag"] == tag]
        compare_case = compare[compare["tag"] == tag]

        for op_name, spec in OP_SPECS.items():
            component = spec["component"]
            n = int(spec["n"])
            k = int(spec["k"])
            child_rows = child_case[child_case["real_component"] == component]
            kernel_rows = kernel_case[kernel_case["real_component"] == component]
            if child_rows.empty:
                continue

            token_values = child_rows["attention_token_count"].dropna().astype(int)
            corrected_m = int(token_values.mode().iloc[0]) if not token_values.empty else math.nan

            gemm_rows = kernel_rows[kernel_rows["kernel_short_name"] == "sm90_fp8_gemm_1d2d_impl"]
            quant_rows = kernel_rows[kernel_rows["kernel_short_name"] == "per_token_group_quant_8bit_kernel"]
            original_rows = compare_case[compare_case["operator_family"] == op_name]
            original = original_rows.iloc[0] if not original_rows.empty else pd.Series(dtype=object)

            corrected_aic = float(db.query_gemm(corrected_m, n, k, common.GEMMQuantMode.fp8_block))
            real_child_ms = float(child_rows["gpu_makespan_ms"].mean())
            real_gemm_ms = float(gemm_rows["kernel_gpu_time_ms"].mean()) if not gemm_rows.empty else math.nan
            real_quant_ms = float(quant_rows["kernel_gpu_time_ms"].mean()) if not quant_rows.empty else 0.0
            original_aic_ms = safe_float(original.get("aic_per_layer_latency_ms"))

            rows.append(
                {
                    "case_id": case_id,
                    "tag": tag,
                    "case_label": case.get("case_label"),
                    "stage": child_rows["stage"].mode().iloc[0],
                    "operator_family": op_name,
                    "component": component,
                    "corrected_m": corrected_m,
                    "n": n,
                    "k": k,
                    "original_aic_query_s": extract_first_query_s(original),
                    "actual_attention_token_count_min": int(token_values.min()),
                    "actual_attention_token_count_max": int(token_values.max()),
                    "first_prefill_req_count": case.get("first_formal_prefill_req_count"),
                    "first_decode_req_count": case.get("first_formal_decode_req_count"),
                    "sum_prefill_fresh_tokens": case.get("first_formal_sum_prefill_fresh_tokens"),
                    "sum_decode_kv_tokens": case.get("first_formal_sum_decode_kv_tokens"),
                    "real_child_ms": real_child_ms,
                    "real_gemm_kernel_ms": real_gemm_ms,
                    "real_quant_kernel_ms": real_quant_ms,
                    "original_aic_ms": original_aic_ms,
                    "corrected_aic_gemm_ms": corrected_aic,
                    "original_ratio_child_to_aic": real_child_ms / original_aic_ms if original_aic_ms else math.nan,
                    "corrected_ratio_child_to_aic": real_child_ms / corrected_aic if corrected_aic else math.nan,
                    "corrected_ratio_gemm_to_aic": real_gemm_ms / corrected_aic if corrected_aic else math.nan,
                    "child_minus_corrected_aic_ms": real_child_ms - corrected_aic,
                    "gemm_minus_corrected_aic_ms": real_gemm_ms - corrected_aic,
                    "corrected_shape_exact_in_gemm_perf": exact_in_gemm_perf(gemm_df, corrected_m, n, k),
                    "nearest_sampled_m_same_nk": nearest_m_values(gemm_df, corrected_m, n, k),
                }
            )

    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby("operator_family", as_index=False)
        .agg(
            cases=("case_id", "nunique"),
            avg_original_ratio_child_to_aic=("original_ratio_child_to_aic", "mean"),
            avg_corrected_ratio_child_to_aic=("corrected_ratio_child_to_aic", "mean"),
            avg_corrected_ratio_gemm_to_aic=("corrected_ratio_gemm_to_aic", "mean"),
            median_corrected_ratio_gemm_to_aic=("corrected_ratio_gemm_to_aic", "median"),
            avg_abs_gemm_minus_corrected_aic_ms=("gemm_minus_corrected_aic_ms", lambda s: s.abs().mean()),
            max_abs_gemm_minus_corrected_aic_ms=("gemm_minus_corrected_aic_ms", lambda s: s.abs().max()),
            avg_quant_kernel_ms=("real_quant_kernel_ms", "mean"),
            exact_shape_cases=("corrected_shape_exact_in_gemm_perf", "sum"),
        )
        .sort_values("operator_family")
    )
    return detail, summary


def plot_compare(detail: pd.DataFrame, out_dir: Path) -> None:
    ops = ["qkv_a_proj", "q_b_proj", "o_proj"]
    fig, axes = plt.subplots(len(ops), 1, figsize=(13, 8.5), sharex=True)
    fig.patch.set_facecolor("white")
    width = 0.25
    for ax, op in zip(axes, ops, strict=True):
        sub = detail[detail["operator_family"] == op].sort_values("case_id")
        x = np.arange(len(sub))
        ax.set_facecolor("white")
        ax.bar(x - width, sub["real_gemm_kernel_ms"], width, label="Real GEMM kernel", color="#78A6F7")
        ax.bar(x, sub["real_child_ms"], width, label="Real child range", color="#F2C15E")
        ax.bar(x + width, sub["corrected_aic_gemm_ms"], width, label="AIC corrected GEMM", color="#7BC47F")
        ax.set_ylabel(f"{op}\nms")
        ax.grid(axis="y", color="#E6E8F0", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xticks(x)
        ax.set_xticklabels(sub["case_id"].tolist(), rotation=0)
    axes[0].legend(loc="upper left", ncols=3, frameon=False)
    axes[-1].set_xlabel("AGG case")
    fig.suptitle("Non-attention projection GEMMs: real trace vs AIC with corrected M", y=0.995)
    fig.tight_layout()
    fig.savefig(out_dir / "nonattn_proj_corrected_shape_compare.png", dpi=180, facecolor="white")
    fig.savefig(out_dir / "nonattn_proj_corrected_shape_compare.svg", facecolor="white")
    plt.close(fig)


def write_markdown(detail: pd.DataFrame, summary: pd.DataFrame, out_dir: Path) -> None:
    sample_cases = ["C01", "C03", "C06", "C12"]
    sample_cols = [
        "case_id",
        "operator_family",
        "corrected_m",
        "n",
        "k",
        "real_gemm_kernel_ms",
        "corrected_aic_gemm_ms",
        "corrected_ratio_gemm_to_aic",
        "corrected_shape_exact_in_gemm_perf",
    ]
    sample = detail[detail["case_id"].isin(sample_cases)][sample_cols].sort_values(["case_id", "operator_family"])
    lines = [
        "# AGG Mixed 非 Attn Projection 修正形状查表分析",
        "",
        "本文只覆盖 `qkv_a_proj / q_b_proj / o_proj` 三个 projection GEMM；`kv_b_proj` 和 `mla_concat_k` 的实机边界另有问题，不在本次修正形状口径内。",
        "",
        "## 形状推导",
        "",
        "SGLang DeepSeek-V3 TP=1 下三类 projection 的 GEMM 形状为：",
        "",
        "| op | 实机 component | AIC GEMM 形状 | M 的来源 |",
        "|---|---|---|---|",
        "| `qkv_a_proj` | `fused_qkv_a_proj_with_mqa` | `(M, N=2112, K=7168)` | Nsight `attention_token_count` |",
        "| `q_b_proj` | `q_b_proj` | `(M, N=24576, K=1536)` | Nsight `attention_token_count` |",
        "| `o_proj` | `o_proj` | `(M, N=7168, K=16384)` | Nsight `attention_token_count` |",
        "",
        "`run_agg` 外层 diagnostic 原先使用 first-pass `query_s` 查询这些 GEMM；但 mixed batch 中这些 projection 实际处理的是当前 formal batch 的真实 token 数。因此本分析把 AIC 查表的 `M` 修正为实机 trace 捕获的 `attention_token_count`。",
        "",
        "实机侧同时给出两个口径：`real_gemm_kernel_ms` 是纯 `sm90_fp8_gemm_1d2d_impl` kernel；`real_child_ms` 是该子模块 NVTX range 的 GPU makespan，通常包含 `per_token_group_quant_8bit_kernel + DeepGEMM`。AIC `GEMM(fp8_block)` 更接近前者，不包含 runtime activation quant。",
        "",
        "## 汇总结论",
        "",
        summary.to_markdown(index=False, floatfmt=".6f"),
        "",
        "结论：",
        "",
        "- `q_b_proj` 在修正 `M` 后基本对齐，纯 GEMM kernel / AIC 的均值约为 `0.895`，剩余误差主要来自 perf 表插值与 collector/实机环境差异。",
        "- `qkv_a_proj` 修正后仍偏离，纯 GEMM kernel / AIC 均值约为 `0.759`；AIC 对该形状没有同 `(N,K)` 采样族，属于 3D 插值结果。",
        "- `o_proj` 修正后仍偏离，纯 GEMM kernel / AIC 均值约为 `0.798`；其中 `M=8192` 有 exact `fp8_block` 行但仍明显慢于实机 kernel，说明不只是插值问题，也可能包含 collector DeepGEMM microbench 与 engine 内实际 DeepGEMM 调用路径/输入布局/调度环境差异。",
        "- 若使用 `real_child_ms` 与 AIC 比较，`qkv_a_proj` 会显得更接近甚至偏大，这是因为 child range 包含 activation quant；但 AIC GEMM 表按当前实现不覆盖该 quant kernel，不应把这部分当成 GEMM 表误差。",
        "",
        "## 代表用例明细",
        "",
        sample.to_markdown(index=False, floatfmt=".6f"),
        "",
        "完整 12 个 case 的逐算子明细见 `nonattn_proj_corrected_shape_compare.csv`。",
        "",
        "## 产物",
        "",
        "- `nonattn_proj_corrected_shape_compare.csv`: case/op 级明细，含修正后 GEMM 形状、实机 child/GEMM/quant 时延、AIC 原始与修正查表值。",
        "- `nonattn_proj_corrected_shape_summary.csv`: op 级汇总。",
        "- `nonattn_proj_corrected_shape_compare.png/svg`: 每个 op 的实机与 AIC 修正查表对比图。",
    ]
    out_dir.joinpath("非Attn投影算子修正形状查表分析.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    logging.getLogger("aiconfigurator.sdk.interpolation").setLevel(logging.ERROR)
    args.analysis_dir.mkdir(parents=True, exist_ok=True)
    detail, summary = build_compare(args)
    detail.to_csv(args.analysis_dir / "nonattn_proj_corrected_shape_compare.csv", index=False)
    summary.to_csv(args.analysis_dir / "nonattn_proj_corrected_shape_summary.csv", index=False)
    plot_compare(detail, args.analysis_dir)
    write_markdown(detail, summary, args.analysis_dir)
    print(f"Wrote corrected-shape non-attn projection analysis to {args.analysis_dir}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
