#!/usr/bin/env python3
"""Compare real kv_b_proj timing with prefix-aware AIC GEMM queries.

This analysis is intentionally isolated from the existing prefill comparison
artifacts. It focuses only on DeepSeek MLA's context kv_b_proj GEMM and tests
three AIC query semantics:

1. old_fresh_only: current generic GEMM path, m = sum(fresh_len)
2. direct_total_tokens: one GEMM query with m = sum(prefix_len + fresh_len)
3. sglang_chunk_policy: mimic SGLang 0.5.9 MHA prefix-cache policy for kv_b_proj
   query granularity. Existing cases usually collapse to direct_total_tokens
   because MHA_ONE_SHOT applies when sum(seq_lens) <= 128K.
"""

from __future__ import annotations

import json
import logging
import math
import sys
import textwrap
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[7]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from aiconfigurator.sdk import common  # noqa: E402
from aiconfigurator.sdk.perf_database import PerfDatabase  # noqa: E402

logging.getLogger().setLevel(logging.ERROR)
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

ANALYSIS_ROOT = (
    REPO_ROOT
    / "bench_data/h100_sxm/sglang/v0.5.9/deepseek_v3_mla/analysis"
)
OUTPUT_DIR = ANALYSIS_ROOT / "kv_b_proj_prefix_token_sensitivity"
SOURCE_DIRS = [
    ANALYSIS_ROOT / "mla_aic_compare_kernel_envelope_refresh",
    ANALYSIS_ROOT / "mla_aic_compare_large_prefix",
]

SYSTEMS_ROOT = REPO_ROOT / "src/aiconfigurator/systems"
SYSTEM = "h100_sxm"
BACKEND = "sglang"
VERSION = "0.5.9"
GEMM_QUANT_MODE = common.GEMMQuantMode.fp8_block
K_DIM = 512
SGLANG_MAX_CHUNK_CAPACITY = 128 * 1024

TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
}
COLORS = {
    "real": "#5477C4",
    "old": "#C5CAD3",
    "direct": "#F0986E",
    "sglang": "#A3D576",
    "gemm_only": "#B8A037",
}


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": TOKENS["surface"],
            "axes.facecolor": TOKENS["panel"],
            "axes.edgecolor": TOKENS["axis"],
            "axes.labelcolor": TOKENS["ink"],
            "xtick.color": TOKENS["muted"],
            "ytick.color": TOKENS["muted"],
            "grid.color": TOKENS["grid"],
            "grid.linewidth": 0.8,
            "font.family": ["DejaVu Sans", "sans-serif"],
            "font.size": 9,
            "axes.grid": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def add_chart_header(fig, ax, title: str, subtitle: str) -> None:
    title = textwrap.fill(title, width=90, break_long_words=False)
    subtitle = textwrap.fill(subtitle, width=128, break_long_words=False)
    ax.set_title("")
    fig.subplots_adjust(top=0.84)
    left = ax.get_position().x0
    fig.text(
        left,
        0.975,
        title,
        ha="left",
        va="top",
        fontsize=14,
        fontweight="semibold",
        color=TOKENS["ink"],
    )
    fig.text(
        left,
        0.925,
        subtitle,
        ha="left",
        va="top",
        fontsize=9,
        color=TOKENS["muted"],
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def safe_json_loads(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, float) and math.isnan(value):
        return default
    text = str(value).strip()
    if not text:
        return default
    return json.loads(text)


def extract_child(row: pd.Series, canonical_name: str) -> dict[str, Any] | None:
    children = safe_json_loads(row.get("child_timing_json"), [])
    for child in children:
        if child.get("canonical_name") == canonical_name:
            return child
    return None


def extract_kernel_component(row: pd.Series, canonical_name: str, short_name: str) -> float:
    summaries = safe_json_loads(row.get("kernel_summary_json"), [])
    for summary in summaries:
        if summary.get("canonical_name") != canonical_name:
            continue
        total = 0.0
        for kernel in summary.get("top_gpu_kernels", []):
            if kernel.get("short_name") == short_name:
                total += float(kernel.get("gpu_time_ms", 0.0))
        return total
    return 0.0


def get_int_list(row: pd.Series, column: str, fallback: list[int]) -> list[int]:
    values = safe_json_loads(row.get(column), fallback)
    return [int(v) for v in values]


def shape_for_kv_b(tp_size: int) -> tuple[int, int]:
    return 32768 // int(tp_size), K_DIM


def query_route(gemm_table: pd.DataFrame, *, m: int, n: int, k: int) -> str:
    sub = gemm_table[
        (gemm_table["gemm_dtype"] == GEMM_QUANT_MODE.name)
        & (gemm_table["n"] == n)
        & (gemm_table["k"] == k)
    ]
    if sub.empty:
        return "interp_3d_no_same_nk"
    if int(m) in set(int(v) for v in sub["m"].unique()):
        return "exact"
    return "interp_1d_same_nk"


def query_gemm_ms(db: PerfDatabase, gemm_table: pd.DataFrame, *, m: int, n: int, k: int) -> dict[str, Any]:
    result = db.query_gemm(int(m), int(n), int(k), GEMM_QUANT_MODE)
    return {
        "latency_ms": float(result),
        "source": str(getattr(result, "source", "silicon")),
        "route": query_route(gemm_table, m=int(m), n=int(n), k=int(k)),
    }


def prefix_chunk_token_counts(prefix_lens: list[int], batch_size: int) -> list[int]:
    if not prefix_lens or max(prefix_lens) <= 0:
        return []
    prefix_chunk_len = SGLANG_MAX_CHUNK_CAPACITY // int(batch_size)
    num_chunks = (max(prefix_lens) + prefix_chunk_len - 1) // prefix_chunk_len
    chunk_ms: list[int] = []
    for idx in range(num_chunks):
        start = idx * prefix_chunk_len
        end = start + prefix_chunk_len
        chunk_tokens = sum(max(min(prefix_len, end) - start, 0) for prefix_len in prefix_lens)
        if chunk_tokens > 0:
            chunk_ms.append(int(chunk_tokens))
    return chunk_ms


def sglang_policy_query_ms(
    db: PerfDatabase,
    gemm_table: pd.DataFrame,
    *,
    fresh_lens: list[int],
    prefix_lens: list[int],
    n: int,
    k: int,
) -> dict[str, Any]:
    fresh_tokens = int(sum(fresh_lens))
    prefix_tokens = int(sum(prefix_lens))
    total_tokens = fresh_tokens + prefix_tokens
    batch_size = len(fresh_lens)

    if prefix_tokens == 0:
        query_ms = [fresh_tokens]
        policy = "no_prefix_one_query"
    elif total_tokens <= SGLANG_MAX_CHUNK_CAPACITY:
        # SGLang 0.5.9 MHA_ONE_SHOT path fetches fresh+prefix latent cache and
        # applies kv_b_proj once to the complete sequence set.
        query_ms = [total_tokens]
        policy = "mha_one_shot_total_query"
    else:
        query_ms = [fresh_tokens] + prefix_chunk_token_counts(prefix_lens, batch_size)
        policy = "mha_chunked_kv_fresh_plus_prefix_chunks"

    parts = [query_gemm_ms(db, gemm_table, m=m, n=n, k=k) for m in query_ms if m > 0]
    return {
        "latency_ms": sum(part["latency_ms"] for part in parts),
        "query_m_tokens": query_ms,
        "query_routes": [part["route"] for part in parts],
        "policy": policy,
    }


def read_real_rows(source_dir: Path) -> pd.DataFrame:
    path = source_dir / "prefill_real_layer_detail.csv"
    df = pd.read_csv(path)
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        child = extract_child(row, "kv_b_proj")
        if not child:
            continue
        fresh_lens = get_int_list(row, "chunk_req_fresh_lens_json", [int(row["fresh_len"])] * int(row["batch_size"]))
        prefix_lens = get_int_list(row, "chunk_req_prefix_lens_json", [int(row["prefix_len"])] * int(row["batch_size"]))
        total_lens = get_int_list(
            row,
            "chunk_req_total_seq_lens_json",
            [p + f for p, f in zip(prefix_lens, fresh_lens)],
        )
        rows.append(
            {
                "source_name": source_dir.name,
                "tag": row["tag"],
                "run_instance_name": row["run_instance_name"],
                "layer_id": int(row["layer_id"]),
                "tp_size": int(row["tp_size"]),
                "attention_backend": row.get("attention_backend", ""),
                "attention_module": row.get("attention_module", ""),
                "batch_size": int(row["batch_size"]),
                "fresh_len": int(row["fresh_len"]),
                "prefix_len": int(row["prefix_len"]),
                "total_seq_len": int(row["total_seq_len"]),
                "fresh_lens": fresh_lens,
                "prefix_lens": prefix_lens,
                "total_seq_lens": total_lens,
                "fresh_tokens": int(sum(fresh_lens)),
                "prefix_tokens": int(sum(prefix_lens)),
                "total_tokens": int(sum(total_lens)),
                "real_kv_b_proj_host_ms": float(child.get("host_duration_ms", 0.0)),
                "real_kv_b_proj_gpu_makespan_ms": float(child.get("gpu_makespan_ms", 0.0)),
                "real_kv_b_proj_kernel_sum_ms": float(child.get("gpu_kernel_time_sum_ms", 0.0)),
                "real_kv_b_proj_kernel_count": int(child.get("gpu_kernel_count", 0)),
                "real_kv_b_proj_gemm_only_ms": extract_kernel_component(
                    row, "kv_b_proj", "sm90_fp8_gemm_1d2d_impl"
                ),
                "real_kv_b_proj_quant_only_ms": extract_kernel_component(
                    row, "kv_b_proj", "per_token_group_quant_8bit_kernel"
                ),
            }
        )
    return pd.DataFrame(rows)


def build_comparison() -> tuple[pd.DataFrame, pd.DataFrame]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    db = PerfDatabase(
        system=SYSTEM,
        backend=BACKEND,
        version=VERSION,
        systems_root=str(SYSTEMS_ROOT),
    )
    gemm_table = pd.read_csv(
        REPO_ROOT / "src/aiconfigurator/systems/data/h100_sxm/sglang/0.5.9/gemm_perf.txt"
    )

    real = pd.concat([read_real_rows(source_dir) for source_dir in SOURCE_DIRS], ignore_index=True)
    if real.empty:
        raise SystemExit("No kv_b_proj rows found in source prefill_real_layer_detail.csv files.")

    case_rows: list[dict[str, Any]] = []
    for key, part in real.groupby(["source_name", "tag"], sort=True):
        source_name, tag = key
        first = part.sort_values("layer_id").iloc[0]
        n, k = shape_for_kv_b(int(first["tp_size"]))
        fresh_lens = list(first["fresh_lens"])
        prefix_lens = list(first["prefix_lens"])
        fresh_tokens = int(sum(fresh_lens))
        prefix_tokens = int(sum(prefix_lens))
        total_tokens = fresh_tokens + prefix_tokens

        old = query_gemm_ms(db, gemm_table, m=fresh_tokens, n=n, k=k)
        direct = query_gemm_ms(db, gemm_table, m=total_tokens, n=n, k=k)
        sglang = sglang_policy_query_ms(
            db,
            gemm_table,
            fresh_lens=fresh_lens,
            prefix_lens=prefix_lens,
            n=n,
            k=k,
        )

        real_kernel_mean = float(part["real_kv_b_proj_kernel_sum_ms"].mean())
        real_gemm_mean = float(part["real_kv_b_proj_gemm_only_ms"].mean())

        row = {
            "source_name": source_name,
            "tag": tag,
            "num_layers": int(part["layer_id"].nunique()),
            "tp_size": int(first["tp_size"]),
            "n": n,
            "k": k,
            "batch_size": int(first["batch_size"]),
            "fresh_len": int(first["fresh_len"]),
            "prefix_len": int(first["prefix_len"]),
            "fresh_tokens": fresh_tokens,
            "prefix_tokens": prefix_tokens,
            "total_tokens": total_tokens,
            "real_kernel_sum_mean_ms": real_kernel_mean,
            "real_kernel_sum_std_ms": float(part["real_kv_b_proj_kernel_sum_ms"].std(ddof=0)),
            "real_gemm_only_mean_ms": real_gemm_mean,
            "real_quant_only_mean_ms": float(part["real_kv_b_proj_quant_only_ms"].mean()),
            "real_gpu_makespan_mean_ms": float(part["real_kv_b_proj_gpu_makespan_ms"].mean()),
            "aic_old_fresh_only_ms": old["latency_ms"],
            "aic_old_query_m_tokens": json.dumps([fresh_tokens]),
            "aic_old_route": old["route"],
            "aic_direct_total_tokens_ms": direct["latency_ms"],
            "aic_direct_query_m_tokens": json.dumps([total_tokens]),
            "aic_direct_route": direct["route"],
            "aic_sglang_chunk_policy_ms": sglang["latency_ms"],
            "aic_sglang_query_m_tokens": json.dumps(sglang["query_m_tokens"]),
            "aic_sglang_query_routes": json.dumps(sglang["query_routes"]),
            "aic_sglang_policy": sglang["policy"],
        }
        for estimate_col in [
            "aic_old_fresh_only_ms",
            "aic_direct_total_tokens_ms",
            "aic_sglang_chunk_policy_ms",
        ]:
            row[f"{estimate_col}_vs_kernel_sum_error_pct"] = (
                (row[estimate_col] - real_kernel_mean) / real_kernel_mean * 100.0
                if real_kernel_mean
                else np.nan
            )
            row[f"{estimate_col}_vs_gemm_only_error_pct"] = (
                (row[estimate_col] - real_gemm_mean) / real_gemm_mean * 100.0
                if real_gemm_mean
                else np.nan
            )
        case_rows.append(row)

    case_df = pd.DataFrame(case_rows).sort_values(["source_name", "prefix_len", "fresh_len", "batch_size"])
    return real, case_df


def summarize(case_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for source_name, part in case_df.groupby("source_name", sort=True):
        for label, col in [
            ("old_fresh_only", "aic_old_fresh_only_ms"),
            ("direct_total_tokens", "aic_direct_total_tokens_ms"),
            ("sglang_chunk_policy", "aic_sglang_chunk_policy_ms"),
        ]:
            err = (part[col] - part["real_kernel_sum_mean_ms"]) / part["real_kernel_sum_mean_ms"] * 100.0
            rows.append(
                {
                    "source_name": source_name,
                    "estimate": label,
                    "case_count": len(part),
                    "mape_vs_kernel_sum_pct": float(err.abs().mean()),
                    "mean_error_vs_kernel_sum_pct": float(err.mean()),
                    "median_abs_error_vs_kernel_sum_pct": float(err.abs().median()),
                    "max_abs_error_vs_kernel_sum_pct": float(err.abs().max()),
                }
            )
    return pd.DataFrame(rows)


def label_for_case(row: pd.Series) -> str:
    return f"b{int(row.batch_size)}\nf{int(row.fresh_len)}\np{int(row.prefix_len)}"


def plot_values(case_df: pd.DataFrame, source_name: str) -> None:
    part = case_df[case_df["source_name"] == source_name].copy()
    part = part.sort_values(["prefix_len", "fresh_len", "batch_size"])
    labels = [label_for_case(row) for _, row in part.iterrows()]
    x = np.arange(len(part))
    width = 0.18

    fig, ax = plt.subplots(figsize=(max(10, len(part) * 0.55), 6.2))
    series = [
        ("Real kernel sum", "real_kernel_sum_mean_ms", COLORS["real"]),
        ("AIC old fresh-only", "aic_old_fresh_only_ms", COLORS["old"]),
        ("AIC total tokens", "aic_direct_total_tokens_ms", COLORS["direct"]),
    ]
    width = 0.22
    for idx, (name, col, color) in enumerate(series):
        ax.bar(
            x + (idx - 1) * width,
            part[col],
            width=width,
            label=name,
            color=color,
            edgecolor=TOKENS["ink"],
            linewidth=0.7,
        )
    ax.set_xticks(x, labels)
    ax.tick_params(axis="x", labelrotation=0)
    ax.set_ylabel("Latency (ms)")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), frameon=False, ncol=3, borderaxespad=0)
    add_chart_header(
        fig,
        ax,
        "kv_b_proj real vs AIC query semantics",
        f"Source: {source_name}. Real is per-layer mean GPU kernel sum for the kv_b_proj child; AIC variants compare current fresh-only lookup with the prefix-aware total-token lookup.",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(OUTPUT_DIR / f"{source_name}_kv_b_proj_values.png", dpi=180)
    fig.savefig(OUTPUT_DIR / f"{source_name}_kv_b_proj_values.svg")
    plt.close(fig)


def plot_error(case_df: pd.DataFrame, source_name: str) -> None:
    part = case_df[case_df["source_name"] == source_name].copy()
    part = part.sort_values(["prefix_len", "fresh_len", "batch_size"])
    labels = [label_for_case(row) for _, row in part.iterrows()]
    x = np.arange(len(part))
    width = 0.22
    fig, ax = plt.subplots(figsize=(max(10, len(part) * 0.58), 6.4))
    series = [
        ("Old fresh-only", "aic_old_fresh_only_ms_vs_kernel_sum_error_pct", COLORS["old"]),
        ("Total tokens", "aic_direct_total_tokens_ms_vs_kernel_sum_error_pct", COLORS["direct"]),
    ]
    width = 0.26
    for idx, (name, col, color) in enumerate(series):
        ax.bar(
            x + (idx - 0.5) * width,
            part[col],
            width=width,
            label=name,
            color=color,
            edgecolor=TOKENS["ink"],
            linewidth=0.7,
        )
    ax.axhline(0, color=TOKENS["ink"], linewidth=1.0)
    ax.set_xticks(x, labels)
    ax.set_xlabel("")
    ax.set_ylabel("AIC vs real kernel sum error (%)")
    ax.tick_params(axis="x", labelrotation=0)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), frameon=False, ncol=2, borderaxespad=0)
    add_chart_header(
        fig,
        ax,
        "kv_b_proj AIC error after prefix-aware token fixes",
        f"Source: {source_name}. Negative values mean AIC underestimates real kv_b_proj GPU kernel-sum time; current cases do not trigger chunked-kv, so the chart omits the identical SGLang-policy series.",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(OUTPUT_DIR / f"{source_name}_kv_b_proj_error_pct.png", dpi=180)
    fig.savefig(OUTPUT_DIR / f"{source_name}_kv_b_proj_error_pct.svg")
    plt.close(fig)


def write_readme(summary_df: pd.DataFrame, case_df: pd.DataFrame) -> None:
    summary_cols = [
        "source_name",
        "estimate",
        "case_count",
        "mape_vs_kernel_sum_pct",
        "mean_error_vs_kernel_sum_pct",
        "max_abs_error_vs_kernel_sum_pct",
    ]
    summary_lines = dataframe_to_markdown(summary_df[summary_cols])
    lines = [
        "# kv_b_proj prefix token sensitivity",
        "",
        "This isolated analysis compares real H100/SGLang `kv_b_proj` timing with three AIC GEMM query semantics.",
        "",
        "## Query Semantics",
        "",
        "- `old_fresh_only`: current generic AIC `GEMM` behavior for context ops, `m=sum(fresh_lens)`.",
        "- `direct_total_tokens`: direct prefix-aware fix, `m=sum(prefix_lens + fresh_lens)`.",
        "- `sglang_chunk_policy`: SGLang 0.5.9 policy. If `sum(seq_lens) <= 128K`, MHA_ONE_SHOT queries `m=sum(prefix+fresh)` once; otherwise it queries fresh once plus prefix chunks with `prefix_chunk_len=(128K // batch_size)`.",
        "",
        "## Summary vs real kv_b_proj kernel-sum",
        "",
        summary_lines,
        "",
        "## Important Caveats",
        "",
        "- Real main baseline is `kv_b_proj` child `gpu_kernel_time_sum_ms`, which includes the per-token FP8 quant kernel plus the DeepGEMM kernel.",
        "- `real_gemm_only_mean_ms` is also exported for checking the narrower GEMM-only kernel口径.",
        "- For TP1, `(n=32768,k=512)` is absent from `gemm_perf.txt`, so all three AIC variants still rely on 3D interpolation. Prefix-aware token counts fix semantics but not the missing-shape data gap.",
        "- In the current real prefill cases, SGLang policy mostly equals `direct_total_tokens` because `sum(prefix+fresh)` stays below `128K` tokens, causing MHA_ONE_SHOT rather than multi-chunk prefix processing.",
        "",
        "## Files",
        "",
        "- `kv_b_proj_layer_detail.csv`: parsed per-layer real kv_b_proj timings.",
        "- `kv_b_proj_case_comparison.csv`: case-level real mean and AIC estimates.",
        "- `kv_b_proj_summary.csv`: aggregate error summary.",
        "- `*_kv_b_proj_values.png/svg`: values by case.",
        "- `*_kv_b_proj_error_pct.png/svg`: error percentages by case.",
    ]
    (OUTPUT_DIR / "README.md").write_text("\n".join(lines) + "\n")


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    rendered = []
    for _, row in df.iterrows():
        values = []
        for col in cols:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.2f}")
            else:
                values.append(str(value))
        rendered.append(values)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = ["| " + " | ".join(values) + " |" for values in rendered]
    return "\n".join([header, sep, *body])


def main() -> None:
    apply_style()
    real_df, case_df = build_comparison()
    summary_df = summarize(case_df)

    real_df.to_csv(OUTPUT_DIR / "kv_b_proj_layer_detail.csv", index=False)
    case_df.to_csv(OUTPUT_DIR / "kv_b_proj_case_comparison.csv", index=False)
    summary_df.to_csv(OUTPUT_DIR / "kv_b_proj_summary.csv", index=False)

    for source_name in sorted(case_df["source_name"].unique()):
        plot_values(case_df, source_name)
        plot_error(case_df, source_name)

    write_readme(summary_df, case_df)
    manifest = {
        "output_dir": str(OUTPUT_DIR.relative_to(REPO_ROOT)),
        "source_dirs": [str(path.relative_to(REPO_ROOT)) for path in SOURCE_DIRS],
        "real_layer_rows": int(len(real_df)),
        "case_rows": int(len(case_df)),
        "query_semantics": [
            "old_fresh_only",
            "direct_total_tokens",
            "sglang_chunk_policy",
        ],
        "sglang_max_chunk_capacity": SGLANG_MAX_CHUNK_CAPACITY,
        "files": sorted(path.name for path in OUTPUT_DIR.iterdir() if path.is_file()),
    }
    (OUTPUT_DIR / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
