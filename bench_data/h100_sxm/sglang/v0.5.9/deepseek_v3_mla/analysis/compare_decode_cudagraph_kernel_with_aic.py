#!/usr/bin/env python3
"""Compare formal decode CUDA Graph attention kernels with AIC lookups.

This script is intentionally scoped to the legacy `attention_backend=auto`
decode-only cases where SGLang selected FA3 on Hopper/H100. Those cases were
previously misread as FlashMLA because `nvjet_tst_*` kernels were treated as
FlashMLA kernels. We now keep the analysis aligned to the actual execution
pattern in trace order:

- The first `num_layers * 5` attention kernels form the stable per-layer core
  pattern.
- Any remaining attention kernels are preserved as graph-tail diagnostics.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import logging
import sys
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[6]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from aiconfigurator.sdk import common
from aiconfigurator.sdk.perf_database import PerfDatabase


TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
}
BLUE = {"base": "#A3BEFA", "mid": "#5477C4", "dark": "#2E4780"}
GOLD = {"base": "#FFE15B", "mid": "#B8A037", "dark": "#736422"}
ORANGE = {"base": "#F0986E", "mid": "#CC6F47", "dark": "#804126"}

SKIPPED_COLUMNS = ["tag", "reason", "csv_path", "expected", "actual"]
CORE_ATTENTION_KERNELS_PER_LAYER = 3
NVJET_BMM_KERNEL_NAMES = {
    "nvjet_tst_256x8_64x6_2x1_v_bz_TNT",
    "nvjet_tst_128x8_64x12_1x1_h_bz_TNT",
    "nvjet_tst_256x16_64x6_2x1_v_bz_TNT",
    "nvjet_tst_128x16_64x11_1x1_h_bz_TNT",
    "nvjet_tst_256x32_64x5_2x1_v_bz_TNT",
    "nvjet_tst_128x32_64x10_1x1_h_bz_TNT",
    "nvjet_tst_64x32_64x16_4x1_v_bz_splitK_TNT",
    "nvjet_tst_256x64_64x5_2x1_v_bz_TNT",
    "nvjet_tst_128x64_64x8_1x1_h_bz_TNT",
    "nvjet_tst_64x32_64x16_4x2_h_bz_splitK_TNT",
}


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bench-root",
        type=Path,
        default=Path("bench_data/h100_sxm/sglang/v0.5.9/deepseek_v3_mla"),
    )
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--systems-root", type=Path, default=Path("src/aiconfigurator/systems"))
    parser.add_argument("--system", default="h100_sxm")
    parser.add_argument("--backend", default="sglang")
    parser.add_argument("--version", default="0.5.9")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--num-heads", type=int, default=128)
    parser.add_argument("--head-size", type=int, default=128)
    parser.add_argument(
        "--show-interpolation-warnings",
        action="store_true",
        help="Show SDK interpolation warnings emitted while building/querying PerfDatabase.",
    )
    return parser.parse_args(argv)


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _kernel_rows(csv_path: Path) -> list[dict]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        name = row.get("kernel_short_name") or row.get("kernel_name") or ""
        if name in NVJET_BMM_KERNEL_NAMES:
            row["operator_category"] = "bmm"
            row["implementation"] = "torch"
    return rows


def _sum_ms(rows: list[dict]) -> float:
    return sum(float(row.get("kernel_duration_ms", 0.0) or 0.0) for row in rows)


def _top_names(rows: list[dict], limit: int = 8) -> str:
    counts: dict[str, int] = {}
    for row in rows:
        name = row.get("kernel_short_name") or row.get("kernel_name") or ""
        counts[name] = counts.get(name, 0) + 1
    top = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]
    return json.dumps([{"name": name, "count": count} for name, count in top], ensure_ascii=False)


def _kernel_name(row: dict) -> str:
    return row.get("kernel_short_name") or row.get("kernel_name") or ""


def _is_nvjet_kernel(row: dict) -> bool:
    return _kernel_name(row) in NVJET_BMM_KERNEL_NAMES


def _is_flashmla_aux_kernel(row: dict) -> bool:
    lowered = _kernel_name(row).lower()
    return (
        "flash_mla" in lowered
        or "flashmla" in lowered
        or "flash_fwd_mla" in lowered
        or "get_mla_metadata" in lowered
        or "create_flashmla" in lowered
    )


def _is_fa3_kernel(row: dict) -> bool:
    lowered = _kernel_name(row).lower()
    return "flashattn" in lowered or "flash::" in lowered or row.get("implementation") == "fa3"


def _query_decode_aic(
    database: PerfDatabase,
    *,
    family: str,
    batch_size: int,
    kv_len: int,
    num_heads: int,
    head_size: int,
    kv_cache_dtype: common.KVCacheQuantMode,
) -> tuple[float | None, str | None]:
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            if family == "mla":
                result = database.query_generation_mla(
                    b=batch_size,
                    s=kv_len + 1,
                    num_heads=num_heads,
                    kvcache_quant_mode=kv_cache_dtype,
                )
            else:
                result = database.query_generation_attention(
                    b=batch_size,
                    s=kv_len + 1,
                    n=num_heads,
                    n_kv=1,
                    kvcache_quant_mode=kv_cache_dtype,
                    head_size=head_size,
                )
        return float(result), getattr(result, "source", "silicon")
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _query_wideep_decode_aic(
    database: PerfDatabase,
    *,
    batch_size: int,
    kv_len: int,
    tp_size: int,
    kv_cache_dtype: common.KVCacheQuantMode,
    attention_backend: str,
) -> tuple[float | None, str | None]:
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result = database.query_wideep_generation_mla(
                b=batch_size,
                s=kv_len + 1,
                tp_size=tp_size,
                kvcache_quant_mode=kv_cache_dtype,
                fmha_quant_mode=common.FMHAQuantMode.fp8
                if kv_cache_dtype == common.KVCacheQuantMode.fp8
                else common.FMHAQuantMode.bfloat16,
                attention_backend=attention_backend,
            )
        return float(result), getattr(result, "source", "silicon")
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _pct_error(sim: float | None, real: float | None) -> float | None:
    if sim is None or real in (None, 0.0):
        return None
    return (sim - real) / real * 100.0


def _should_include_case(entry: dict, run_meta: dict) -> bool:
    if entry.get("status") != "ok":
        return False
    decode_backend = run_meta.get("decode_attention_backend")
    if decode_backend not in (None, "", "auto", "default"):
        return False
    return True


def build_comparison(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    bench_root = args.bench_root
    manifest_path = args.manifest or bench_root / "decode_manifest.csv"
    if not args.show_interpolation_warnings:
        logging.getLogger("aiconfigurator.sdk.interpolation").setLevel(logging.ERROR)
    database = PerfDatabase(
        system=args.system,
        backend=args.backend,
        version=args.version,
        systems_root=str(args.systems_root),
    )

    rows_out: list[dict] = []
    skipped: list[dict] = []
    with manifest_path.open(newline="", encoding="utf-8") as f:
        for entry in csv.DictReader(f):
            raw_run_dir = Path(entry["raw_run_dir"]) if entry.get("raw_run_dir") else None
            run_meta = _read_json(raw_run_dir / "run_meta.json") if raw_run_dir else {}
            if not _should_include_case(entry, run_meta):
                continue

            csv_path = Path(entry["csv_path"])
            num_layers = int(run_meta.get("num_layers") or 5)
            batch_size = int(entry["batch_size"])
            kv_len = int(entry["kv_len"])
            tp_size = int(entry.get("tp_size") or 1)
            local_num_heads = args.num_heads // tp_size

            kernels = _kernel_rows(csv_path)
            if not kernels:
                skipped.append({"tag": entry["tag"], "reason": "empty_kernel_csv", "csv_path": str(csv_path)})
                continue

            attention_rows = [row for row in kernels if row.get("operator_category") == "attention"]
            expected_core_attention = num_layers * CORE_ATTENTION_KERNELS_PER_LAYER
            core_attention = attention_rows[:expected_core_attention]
            tail_attention = attention_rows[expected_core_attention:]

            if len(core_attention) < expected_core_attention:
                skipped.append(
                    {
                        "tag": entry["tag"],
                        "reason": "insufficient_core_attention_kernels",
                        "expected": expected_core_attention,
                        "actual": len(core_attention),
                        "csv_path": str(csv_path),
                    }
                )
                continue

            core_fa3 = [row for row in core_attention if _is_fa3_kernel(row)]
            core_nvjet = [row for row in core_attention if _is_nvjet_kernel(row)]
            core_flashmla = [row for row in core_attention if _is_flashmla_aux_kernel(row)]

            real_core_attention_sum_ms = _sum_ms(core_attention)
            real_core_fa3_sum_ms = _sum_ms(core_fa3)
            real_core_nvjet_sum_ms = _sum_ms(core_nvjet)
            real_core_flashmla_sum_ms = _sum_ms(core_flashmla)
            real_all_attention_sum_ms = _sum_ms(attention_rows)
            tail_attention_sum_ms = _sum_ms(tail_attention)
            real_core_attention_per_layer_ms = real_core_attention_sum_ms / num_layers
            real_tail_attention_per_layer_ms = tail_attention_sum_ms / num_layers if num_layers else None

            row_out = {
                "tag": entry["tag"],
                "run_instance_name": csv_path.name.split("__", 1)[0],
                "stage": "decode",
                "batch_size": batch_size,
                "fresh_len": 1,
                "kv_len": kv_len,
                "prefix_len": kv_len,
                "total_seq_len": kv_len + 1,
                "tp_size": tp_size,
                "local_num_heads": local_num_heads,
                "num_layers": num_layers,
                "cuda_graph_mode": entry.get("cuda_graph_mode", "on"),
                "real_core_attention_kernel_count": len(core_attention),
                "real_core_fa3_kernel_count": len(core_fa3),
                "real_core_nvjet_kernel_count": len(core_nvjet),
                "real_core_flashmla_aux_kernel_count": len(core_flashmla),
                "real_tail_attention_kernel_count": len(tail_attention),
                "real_all_attention_kernel_count": len(attention_rows),
                "real_core_attention_sum_ms": real_core_attention_sum_ms,
                "real_core_fa3_sum_ms": real_core_fa3_sum_ms,
                "real_core_nvjet_sum_ms": real_core_nvjet_sum_ms,
                "real_core_flashmla_aux_sum_ms": real_core_flashmla_sum_ms,
                "real_all_attention_sum_ms": real_all_attention_sum_ms,
                "real_tail_attention_sum_ms": tail_attention_sum_ms,
                "real_tail_attention_per_layer_ms": real_tail_attention_per_layer_ms,
                "real_core_attention_per_layer_ms": real_core_attention_per_layer_ms,
                "real_core_attention_names_json": _top_names(core_attention),
                "real_core_fa3_names_json": _top_names(core_fa3),
                "real_core_nvjet_names_json": _top_names(core_nvjet),
                "real_core_flashmla_aux_names_json": _top_names(core_flashmla),
                "real_tail_attention_names_json": _top_names(tail_attention),
                "aic_query_s": kv_len + 1,
                "aic_attention_n_kv": 1,
                "aic_attention_head_size": args.head_size,
            }

            for kv_dtype in [common.KVCacheQuantMode.bfloat16, common.KVCacheQuantMode.fp8]:
                for family in ["mla", "attention"]:
                    latency, source = _query_decode_aic(
                        database,
                        family=family,
                        batch_size=batch_size,
                        kv_len=kv_len,
                        num_heads=local_num_heads,
                        head_size=args.head_size,
                        kv_cache_dtype=kv_dtype,
                    )
                    key = f"aic_{family}_{kv_dtype.name}"
                    row_out[f"{key}_latency_ms"] = latency
                    row_out[f"{key}_source"] = source
                    row_out[f"{key}_error_pct_vs_core_attention_per_layer"] = _pct_error(
                        latency, real_core_attention_per_layer_ms
                    )
                if kv_dtype == common.KVCacheQuantMode.fp8:
                    for backend in ["flashinfer", "fa3"]:
                        wideep_latency, wideep_source = _query_wideep_decode_aic(
                            database,
                            batch_size=batch_size,
                            kv_len=kv_len,
                            tp_size=tp_size,
                            kv_cache_dtype=kv_dtype,
                            attention_backend=backend,
                        )
                        wideep_key = f"aic_wideep_mla_{backend}_{kv_dtype.name}"
                        row_out[f"{wideep_key}_latency_ms"] = wideep_latency
                        row_out[f"{wideep_key}_source"] = wideep_source
                        row_out[f"{wideep_key}_error_pct_vs_tail_attention_per_layer"] = _pct_error(
                            wideep_latency, real_tail_attention_per_layer_ms
                        )

            rows_out.append(row_out)

    return pd.DataFrame(rows_out), pd.DataFrame(skipped, columns=SKIPPED_COLUMNS)


def add_chart_header(fig, title: str, subtitle: str) -> None:
    fig.text(0.06, 0.965, title, ha="left", va="top", fontsize=15, fontweight="bold", color=TOKENS["ink"])
    fig.text(0.06, 0.925, subtitle, ha="left", va="top", fontsize=10.5, color=TOKENS["muted"])


def _case_label(row: pd.Series) -> str:
    return f"b{int(row['batch_size'])}\np{int(row['kv_len'])}"


def plot_real_vs_aic(
    df: pd.DataFrame,
    *,
    sim_column: str,
    real_column: str,
    title: str,
    subtitle: str,
    output_path: Path,
) -> None:
    plot_df = df.sort_values(["kv_len", "batch_size"]).copy()
    plot_df["case_label"] = plot_df.apply(_case_label, axis=1)
    x = np.arange(len(plot_df))
    width = 0.34

    fig, ax = plt.subplots(figsize=(max(10.5, len(plot_df) * 0.95), 6.2), dpi=220)
    fig.patch.set_facecolor(TOKENS["surface"])
    ax.set_facecolor(TOKENS["panel"])
    add_chart_header(fig, title, subtitle)

    ax.bar(
        x - width / 2,
        plot_df[real_column],
        width=width,
        color=BLUE["base"],
        edgecolor=BLUE["dark"],
        linewidth=0.8,
        label="Real H100",
    )
    ax.bar(
        x + width / 2,
        plot_df[sim_column],
        width=width,
        color=GOLD["base"] if "mla" in sim_column else ORANGE["base"],
        edgecolor=GOLD["dark"] if "mla" in sim_column else ORANGE["dark"],
        linewidth=0.8,
        label="AIC",
    )

    for idx, row in plot_df.reset_index(drop=True).iterrows():
        sim = row[sim_column]
        real = row[real_column]
        if pd.notna(sim) and pd.notna(real) and real:
            err = (sim - real) / real * 100.0
            top = max(sim, real)
            ax.text(idx, top * 1.03, f"{err:+.0f}%", ha="center", va="bottom", fontsize=8.5, color=TOKENS["ink"])

    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["case_label"], fontsize=9)
    ax.set_ylabel("Latency per layer (ms)", color=TOKENS["ink"])
    ax.grid(axis="y", color=TOKENS["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(TOKENS["axis"])
    ax.spines["bottom"].set_color(TOKENS["axis"])
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.04), ncols=2, frameon=False)
    ax.set_ylim(0, max(plot_df[[real_column, sim_column]].max()) * 1.22)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(output_path)
    fig.savefig(output_path.with_suffix(".svg"))
    plt.close(fig)


def plot_kernel_breakdown(df: pd.DataFrame, output_path: Path) -> None:
    plot_df = df.sort_values(["kv_len", "batch_size"]).copy()
    plot_df["case_label"] = plot_df.apply(_case_label, axis=1)
    x = np.arange(len(plot_df))

    fig, ax = plt.subplots(figsize=(max(10.5, len(plot_df) * 0.95), 6.2), dpi=220)
    fig.patch.set_facecolor(TOKENS["surface"])
    ax.set_facecolor(TOKENS["panel"])
    add_chart_header(
        fig,
        "Decode CUDA Graph attention kernel decomposition",
        "Core attention keeps the stable 5-kernel-per-layer pattern; tail attention is preserved as a graph-tail diagnostic.",
    )

    bottom = np.zeros(len(plot_df))
    stacks = [
        ("Core nvjet_tst", "real_core_nvjet_sum_ms", BLUE["base"], BLUE["dark"]),
        ("Core FA3 attention", "real_core_fa3_sum_ms", GOLD["base"], GOLD["dark"]),
        ("Tail attention", "real_tail_attention_sum_ms", ORANGE["base"], ORANGE["dark"]),
    ]
    for label, column, color, edge in stacks:
        values = plot_df[column].to_numpy(dtype=float)
        ax.bar(x, values, bottom=bottom, color=color, edgecolor=edge, linewidth=0.8, label=label)
        bottom += values

    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["case_label"], fontsize=9)
    ax.set_ylabel("Kernel time in profiled decode graph (ms)", color=TOKENS["ink"])
    ax.grid(axis="y", color=TOKENS["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(TOKENS["axis"])
    ax.spines["bottom"].set_color(TOKENS["axis"])
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.04), ncols=3, frameon=False)
    ax.set_ylim(0, bottom.max() * 1.16)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(output_path)
    fig.savefig(output_path.with_suffix(".svg"))
    plt.close(fig)


def summarize(comparison: pd.DataFrame, skipped: pd.DataFrame) -> str:
    lines = [
        "# Decode CUDA Graph Kernel 与 AIC 对比摘要",
        "",
        "## 口径",
        "",
        "- 实机侧输入为正式归档的 `DecodeCudaGraphKernel汇总.csv`，这里只保留旧的 `attention_backend=auto` decode-only case，也就是 SGLang 在 H100 上自动选择 `fa3` 的那一批。",
        "- 主实机指标为 `real_core_attention_per_layer_ms`：按时间顺序取前 `num_layers*3` 个纯 attention kernels，也就是每层稳定重复的 3 个 FA3 kernels，再按层数归一。",
        "- `real_core_nvjet_sum_ms` 和 `real_core_fa3_sum_ms` 仅用于解释 core attention 的内部构成，不再把 `nvjet_tst_*` 直接称为 FlashMLA。",
        "- `tail_attention_*` 单独保留，用于标记 CUDA Graph 窗口中的尾部 attention 类 kernel，不纳入主对齐口径。",
        "- AIC 侧同时查询 `query_generation_mla()` 和 `query_generation_attention()`，查询长度为 `kv_len + 1`。",
        "- 另附 `query_wideep_generation_mla()` 的 flashinfer/fa3 查询列，用于观察尾部 attention 与模块级融合口径的量级关系；该列不作为主对齐口径。",
        "",
        "## 统计",
        "",
        f"- 可比较 case 数: `{len(comparison)}`",
        f"- 跳过行数: `{len(skipped)}`",
    ]
    if comparison.empty:
        return "\n".join(lines) + "\n"

    metrics = [
        ("MLA / bfloat16", "aic_mla_bfloat16_latency_ms", "real_core_attention_per_layer_ms"),
        ("MLA / fp8", "aic_mla_fp8_latency_ms", "real_core_attention_per_layer_ms"),
        ("Attention / bfloat16", "aic_attention_bfloat16_latency_ms", "real_core_attention_per_layer_ms"),
        ("Attention / fp8", "aic_attention_fp8_latency_ms", "real_core_attention_per_layer_ms"),
    ]
    lines.extend(["", "## MAPE（对 core attention per-layer）", ""])
    for label, sim_col, real_col in metrics:
        valid = comparison[[sim_col, real_col]].dropna()
        valid = valid[valid[real_col] != 0]
        if valid.empty:
            lines.append(f"- `{label}`: 无有效值")
            continue
        mape = ((valid[sim_col] - valid[real_col]).abs() / valid[real_col]).mean() * 100.0
        lines.append(f"- `{label}`: `{mape:.2f}%`")

    best = []
    for _, row in comparison.iterrows():
        candidates = []
        for label, sim_col, real_col in metrics:
            sim = row.get(sim_col)
            real = row.get(real_col)
            if pd.notna(sim) and pd.notna(real) and real:
                candidates.append((abs((sim - real) / real) * 100.0, label))
        if candidates:
            best.append(min(candidates)[1])
    if best:
        counts = pd.Series(best).value_counts()
        lines.extend(["", "## 每 case 最接近口径", ""])
        for label, count in counts.items():
            lines.append(f"- `{label}`: `{int(count)}` / `{len(best)}`")

    wideep_metrics = [
        ("WideEP MLA flashinfer / fp8", "aic_wideep_mla_flashinfer_fp8_latency_ms"),
        ("WideEP MLA fa3 / fp8", "aic_wideep_mla_fa3_fp8_latency_ms"),
    ]
    lines.extend(["", "## WideEP 诊断（对 tail attention per-layer）", ""])
    for label, sim_col in wideep_metrics:
        if sim_col not in comparison:
            continue
        valid = comparison[[sim_col, "real_tail_attention_per_layer_ms"]].dropna()
        valid = valid[valid["real_tail_attention_per_layer_ms"] != 0]
        if valid.empty:
            lines.append(f"- `{label}`: 无有效值")
            continue
        mape = (
            (valid[sim_col] - valid["real_tail_attention_per_layer_ms"]).abs()
            / valid["real_tail_attention_per_layer_ms"]
        ).mean() * 100.0
        lines.append(f"- `{label}`: `{mape:.2f}%`")

    lines.extend(
        [
            "",
            "## 注意事项",
            "",
            "- 这批旧 case 的 `nvjet_tst_*` 只能说明 trace 中存在一类 attention 子 kernel，不能单凭名字把它直接等同为 FlashMLA backend。",
            "- 当前 Nsight CUDA Graph trace 中，前 `num_layers*3` 个纯 attention kernels 呈稳定层级重复结构；两类 `nvjet_tst` 已改归到 `bmm/torch`，尾部仍会附带一个大 `nvjet_tst_*` 节点，因此 tail 部分继续单独保留。",
            "- 因此主图只比较 core attention per-layer；完整 decode graph 的端到端或模块级对齐仍需要额外 NVTX 边界或更细粒度 graph-node 语义确认。",
        ]
    )
    return "\n".join(lines) + "\n"


def build_readme(comparison: pd.DataFrame, skipped: pd.DataFrame) -> str:
    lines = [
        "# Decode CUDA Graph AIC Compare 汇总说明",
        "",
        "## 范围",
        "",
        "- 仅包含旧的正式 decode-only CUDA Graph 实机结果；即 `attention_backend=auto`、`decode_attention_backend` 未显式指定，SGLang 在 H100/Hopper 上自动选择 `fa3` 的那一批 case。",
        "- 实机侧输入来自 `decode_manifest.csv` 指向的 `DecodeCudaGraphKernel汇总.csv`，每行是 Nsight CUDA Graph node 展开后的 GPU kernel。",
        "- 当前 decode trace 没有稳定 Python/module NVTX 层级边界，因此主比较口径使用 kernel 级 attention 分类，而不是 module 级 host 事件。",
        "- 主实机指标 `real_core_attention_per_layer_ms`：按时间顺序取前 `num_layers*3` 个纯 attention kernels 求和后除以层数。",
        "- `real_core_nvjet_sum_ms` 与 `real_core_fa3_sum_ms` 只用于解释 attn 周边拆解；两类重复出现的 `nvjet_tst_*` 现已改归到 `bmm/torch`。",
        "- 尾部 `tail_attention_*` 单独保留为 CUDA Graph 尾部诊断，不自动归入每层 attention。",
        "",
        "## 规模",
        "",
        f"- 可比较 decode case 数: `{len(comparison)}`",
        f"- 跳过记录数: `{len(skipped)}`",
    ]
    if not comparison.empty:
        tags = ", ".join(str(tag) for tag in comparison["tag"].tolist())
        lines.extend(
            [
                f"- 覆盖 case: `{tags}`",
                f"- batch 范围: `{int(comparison['batch_size'].min())}` - `{int(comparison['batch_size'].max())}`",
                f"- kv_len 范围: `{int(comparison['kv_len'].min())}` - `{int(comparison['kv_len'].max())}`",
            ]
        )

    lines.extend(
        [
            "",
            "## 输出文件",
            "",
            "- `decode_cudagraph_kernel_aic_compare__comparison.csv`: case 级主表，包含实机 kernel 汇总、AIC MLA/Attention 查表值和误差百分比。",
            "- `decode_cudagraph_kernel_aic_compare__skipped.csv`: 跳过记录；即使为空也保留固定表头，便于脚本化读取。",
            "- `decode_cudagraph_kernel_aic_compare__summary.md`: 面向阅读的结论摘要和 MAPE 汇总。",
            "- `decode_core_attention_vs_aic_mla_bfloat16.png/svg`: core attention 与 AIC `query_generation_mla` bf16 对比图。",
            "- `decode_core_attention_vs_aic_mla_fp8.png/svg`: core attention 与 AIC `query_generation_mla` fp8 对比图。",
            "- `decode_core_attention_vs_aic_attention_bfloat16.png/svg`: core attention 与 AIC `query_generation_attention` bf16 对比图。",
            "- `decode_core_attention_vs_aic_attention_fp8.png/svg`: core attention 与 AIC `query_generation_attention` fp8 对比图。",
            "- `decode_attention_kernel_decomposition.png/svg`: core `nvjet_tst`、core FA3 attention、tail attention 的 kernel 时间分解图。",
            "- `artifact_manifest.json`: 本目录归档清单和行数摘要。",
            "",
            "## 注意",
            "",
            "- 这批旧 case 的正确结论是：SGLang 实际后端选择为 `fa3`，而不是 FlashMLA；trace 中出现 `nvjet_tst_*` 不能直接推翻这一点。",
            "- 当前主口径下，正式 CUDA Graph decode 的 core attention 更接近 AIC `generation_attention(n_kv=1)`；`generation_mla` 列仍保留作对照，但不应建立在旧的 FlashMLA 误读上。",
            "- `query_wideep_generation_mla()` 仅作为 fp8 尾部 attention 诊断列保留，不作为主对齐口径。",
            "- 若后续需要模块级端到端对齐，需要补充可靠的 decode CUDA Graph node 语义或 module NVTX 边界，不能只凭当前 tail attention 节点强行摊到每层。",
        ]
    )
    return "\n".join(lines) + "\n"


def build_artifact_manifest(output_dir: Path, comparison: pd.DataFrame, skipped: pd.DataFrame) -> dict:
    files = sorted(path.name for path in output_dir.iterdir() if path.is_file())
    return {
        "output_dir": str(output_dir),
        "case_count": int(len(comparison)),
        "skipped_rows": int(len(skipped)),
        "batch_sizes": sorted(int(value) for value in comparison["batch_size"].dropna().unique())
        if "batch_size" in comparison
        else [],
        "kv_lens": sorted(int(value) for value in comparison["kv_len"].dropna().unique())
        if "kv_len" in comparison
        else [],
        "files": files,
    }


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir or args.bench_root / "analysis" / "decode_cudagraph_aic_compare"
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison, skipped = build_comparison(args)
    comparison_csv = output_dir / "decode_cudagraph_kernel_aic_compare__comparison.csv"
    skipped_csv = output_dir / "decode_cudagraph_kernel_aic_compare__skipped.csv"
    summary_md = output_dir / "decode_cudagraph_kernel_aic_compare__summary.md"
    readme_md = output_dir / "README.md"
    artifact_manifest_json = output_dir / "artifact_manifest.json"
    comparison.to_csv(comparison_csv, index=False)
    skipped.to_csv(skipped_csv, index=False)
    summary_md.write_text(summarize(comparison, skipped), encoding="utf-8")
    readme_md.write_text(build_readme(comparison, skipped), encoding="utf-8")

    if not comparison.empty:
        plot_real_vs_aic(
            comparison,
            sim_column="aic_mla_bfloat16_latency_ms",
            real_column="real_core_attention_per_layer_ms",
            title="Decode core attention: H100 real vs AIC MLA bf16",
            subtitle="Per-layer latency, formal decode-only CUDA Graph runs with legacy auto->fa3 backend selection",
            output_path=output_dir / "decode_core_attention_vs_aic_mla_bfloat16.png",
        )
        plot_real_vs_aic(
            comparison,
            sim_column="aic_mla_fp8_latency_ms",
            real_column="real_core_attention_per_layer_ms",
            title="Decode core attention: H100 real vs AIC MLA fp8",
            subtitle="Per-layer latency, formal decode-only CUDA Graph runs with legacy auto->fa3 backend selection",
            output_path=output_dir / "decode_core_attention_vs_aic_mla_fp8.png",
        )
        plot_real_vs_aic(
            comparison,
            sim_column="aic_attention_bfloat16_latency_ms",
            real_column="real_core_attention_per_layer_ms",
            title="Decode core attention: H100 real vs AIC Attention bf16",
            subtitle="Per-layer latency, formal decode-only CUDA Graph runs with legacy auto->fa3 backend selection",
            output_path=output_dir / "decode_core_attention_vs_aic_attention_bfloat16.png",
        )
        plot_real_vs_aic(
            comparison,
            sim_column="aic_attention_fp8_latency_ms",
            real_column="real_core_attention_per_layer_ms",
            title="Decode core attention: H100 real vs AIC Attention fp8",
            subtitle="Per-layer latency, formal decode-only CUDA Graph runs with legacy auto->fa3 backend selection",
            output_path=output_dir / "decode_core_attention_vs_aic_attention_fp8.png",
        )
        plot_kernel_breakdown(
            comparison,
            output_path=output_dir / "decode_attention_kernel_decomposition.png",
        )

    artifact_manifest_json.write_text(
        json.dumps(build_artifact_manifest(output_dir, comparison, skipped), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "comparison_csv": str(comparison_csv),
                "skipped_csv": str(skipped_csv),
                "summary_md": str(summary_md),
                "readme_md": str(readme_md),
                "artifact_manifest_json": str(artifact_manifest_json),
                "rows": len(comparison),
                "skipped": len(skipped),
                "output_dir": str(output_dir),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
