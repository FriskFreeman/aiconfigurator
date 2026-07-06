#!/usr/bin/env python3
"""Build decode CUDA Graph MLA-module real-vs-AIC comparison artifacts.

CUDA Graph decode traces do not expose stable Python/NVTX module ranges, so the
real MLA range is inferred from the repeated kernel sequence:

  quant -> deepgemm(2112,7168) ... FA3/nvjet ... quant -> deepgemm(7168,16384)

The final 7168x16384 GEMM is the self_attn output projection. Kernels after it
belong to residual/MoE and are intentionally excluded.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import logging
import math
import re
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[7]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from aiconfigurator.sdk import common  # noqa: E402
from aiconfigurator.sdk import operations as ops  # noqa: E402
from aiconfigurator.sdk.perf_database import PerfDatabase  # noqa: E402

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
}

COLORS = {
    "blue": "#A3BEFA",
    "blue_mid": "#5477C4",
    "blue_dark": "#2E4780",
    "gold": "#FFE15B",
    "orange": "#F0986E",
    "orange_mid": "#CC6F47",
    "olive": "#A3D576",
    "olive_mid": "#71B436",
    "pink": "#F390CA",
    "neutral": "#C5CAD3",
    "neutral_mid": "#7A828F",
    "neutral_dark": "#464C55",
}

REAL_COMPONENT_ORDER = [
    "norm",
    "quant",
    "gemm",
    "bmm",
    "support",
    "attention_support",
    "attention",
    "other",
]

SEMANTIC_COMPONENT_ORDER = [
    "qkv_downscale_quant",
    "qkv_downscale_gemm",
    "q_a_layernorm",
    "kv_a_layernorm",
    "q_b_proj_quant",
    "q_b_proj_gemm",
    "q_w_kc_bmm",
    "rotary_emb",
    "set_mla_kv_buffer",
    "fa3_prepare",
    "fa3_attention",
    "fa3_combine",
    "s_w_vc_bmm",
    "o_proj_quant",
    "o_proj_gemm",
    "other",
]

REAL_COMPONENT_COLORS = {
    "norm": "#CEDFFE",
    "quant": "#EAF1FE",
    "gemm": COLORS["blue"],
    "bmm": COLORS["gold"],
    "support": "#FFEDDE",
    "attention_support": COLORS["orange"],
    "attention": COLORS["olive"],
    "other": COLORS["neutral"],
}

SEMANTIC_COMPONENT_LABELS = {
    "qkv_downscale_quant": "qkv/down q",
    "qkv_downscale_gemm": "qkv/down GEMM",
    "q_a_layernorm": "q_a norm",
    "kv_a_layernorm": "kv_a norm",
    "q_b_proj_quant": "q_b quant",
    "q_b_proj_gemm": "q_b GEMM",
    "q_w_kc_bmm": "q_w_kc BMM",
    "rotary_emb": "rotary",
    "set_mla_kv_buffer": "KV write",
    "fa3_prepare": "FA3 prep",
    "fa3_attention": "FA3 attn",
    "fa3_combine": "FA3 combine",
    "s_w_vc_bmm": "s_w_vc BMM",
    "o_proj_quant": "o_proj quant",
    "o_proj_gemm": "o_proj GEMM",
    "other": "other",
}

SEMANTIC_COMPONENT_COLORS = {
    "qkv_downscale_quant": "#DCE8FF",
    "qkv_downscale_gemm": COLORS["blue"],
    "q_a_layernorm": "#F0F4FF",
    "kv_a_layernorm": "#D7DFF2",
    "q_b_proj_quant": "#FFF4B8",
    "q_b_proj_gemm": COLORS["gold"],
    "q_w_kc_bmm": "#F6C36E",
    "rotary_emb": "#F8D8C4",
    "set_mla_kv_buffer": COLORS["orange"],
    "fa3_prepare": "#DDEFC5",
    "fa3_attention": COLORS["olive"],
    "fa3_combine": "#7CCB8A",
    "s_w_vc_bmm": "#C9A4E8",
    "o_proj_quant": "#F8CBE6",
    "o_proj_gemm": COLORS["pink"],
    "other": COLORS["neutral"],
}

DEEPSEEK_FALLBACK_ORDER = [
    "generation_downscale_gemm",
    "generation_q_b_proj_gemm",
    "generation_bmm_pre",
    "generation_attention",
    "generation_bmm_post",
    "generation_proj_gemm",
]

DEEPSEEK_FALLBACK_COLORS = {
    "generation_downscale_gemm": COLORS["blue"],
    "generation_q_b_proj_gemm": COLORS["gold"],
    "generation_bmm_pre": "#FFE9A8",
    "generation_attention": COLORS["olive"],
    "generation_bmm_post": "#D7EFB8",
    "generation_proj_gemm": COLORS["pink"],
}

WIDEEP_DEDUP_ORDER = [
    "generation_downscale_gemm",
    "wideep_generation_mla",
]

WIDEEP_SDK_ORDER = [
    "generation_qkv_a_proj_gemm",
    "generation_downscale_gemm",
    "wideep_generation_mla",
]

WIDEEP_COLORS = {
    "generation_qkv_a_proj_gemm": "#CEDFFE",
    "generation_downscale_gemm": COLORS["blue"],
    "wideep_generation_mla": COLORS["olive"],
}


@dataclass(frozen=True)
class ManifestEntry:
    tag: str
    status: str
    csv_path: Path
    raw_run_dir: Path | None
    batch_size: int
    kv_len: int
    tp_size: int
    attention_backend: str
    decode_attention_backend: str
    cuda_graph_mode: str


def parse_args() -> argparse.Namespace:
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
    parser.add_argument("--hidden-size", type=int, default=7168)
    return parser.parse_args()


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
            "font.size": 10,
            "axes.grid": True,
        }
    )


def add_chart_header(fig, ax, title: str, subtitle: str) -> None:
    title = textwrap.fill(title, width=90, break_long_words=False)
    subtitle = textwrap.fill(subtitle, width=130, break_long_words=False)
    ax.set_title("")
    fig.subplots_adjust(top=0.84)
    left = ax.get_position().x0
    fig.text(left, 0.965, title, ha="left", va="top", fontsize=16, fontweight="bold", color=TOKENS["ink"])
    fig.text(left, 0.915, subtitle, ha="left", va="top", fontsize=10.5, color=TOKENS["muted"])


def read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def load_manifest(path: Path) -> list[ManifestEntry]:
    entries: list[ManifestEntry] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            raw_run_dir = Path(row["raw_run_dir"]) if row.get("raw_run_dir") else None
            entries.append(
                ManifestEntry(
                    tag=row["tag"],
                    status=row["status"],
                    csv_path=Path(row["csv_path"]) if row.get("csv_path") else Path(),
                    raw_run_dir=raw_run_dir,
                    batch_size=int(row["batch_size"]),
                    kv_len=int(row["kv_len"]),
                    tp_size=int(row.get("tp_size") or 1),
                    attention_backend=row.get("attention_backend", ""),
                    decode_attention_backend=row.get("decode_attention_backend", ""),
                    cuda_graph_mode=row.get("cuda_graph_mode", ""),
                )
            )
    return entries


def is_default_fa3_decode(entry: ManifestEntry, run_meta: dict[str, Any]) -> bool:
    if entry.status != "ok":
        return False
    if not entry.csv_path.exists():
        return False
    if entry.attention_backend not in ("", "auto", "default"):
        return False
    raw_backend = str(run_meta.get("decode_attention_backend") or entry.decode_attention_backend or "").lower()
    if "flashmla" in raw_backend:
        return False
    if "flashmla" in entry.tag.lower() or "decodebackend_flashmla" in str(entry.csv_path).lower():
        return False
    if "_cg_on" not in entry.tag and entry.cuda_graph_mode not in ("on", ""):
        return False
    # In this manifest the legacy CUDA-Graph-on field was stored as "on" in
    # decode_attention_backend. Treat it as a CG flag, not an explicit backend.
    return raw_backend in ("", "on", "auto", "default")


def kernel_name(row: dict[str, Any]) -> str:
    return str(row.get("kernel_name") or "")


def kernel_short_name(row: dict[str, Any]) -> str:
    return str(row.get("kernel_short_name") or row.get("kernel_name") or "")


def is_deepgemm_shape(row: dict[str, Any], n: int, k: int) -> bool:
    name = kernel_name(row)
    if row.get("operator_category") != "gemm" or "deep_gemm" not in name:
        return False
    return re.search(rf"\(unsigned int\){n}.*\(unsigned int\){k}", name) is not None


def kernel_duration_ms(row: dict[str, Any]) -> float:
    return float(row.get("kernel_duration_ms") or 0.0)


def kernel_start_ns(row: dict[str, Any]) -> int:
    return int(float(row.get("kernel_start_ns") or 0))


def kernel_end_ns(row: dict[str, Any]) -> int:
    return int(float(row.get("kernel_end_ns") or 0))


def component_key(row: dict[str, Any]) -> str:
    category = str(row.get("operator_category") or "")
    implementation = str(row.get("implementation") or "")
    if category in REAL_COMPONENT_ORDER:
        return category
    if category == "triton_misc" or category == "torch_misc":
        return "other"
    if implementation == "fa3":
        return "attention"
    return "other"


def semantic_component_keys(segment_rows: list[dict[str, Any]]) -> list[str]:
    """Classify kernels by MLA execution semantics in trace order.

    The decode MLA path has a stable 15-kernel module body for these CUDA Graph
    traces. Shape checks are used for GEMM anchors and ordering is used for the
    duplicated norm/quant/nvjet/device kernels.
    """

    keys = ["other"] * len(segment_rows)
    norm_seen = 0
    quant_seen = 0
    nvjet_seen = 0
    fa3_device_seen = 0
    for idx, row in enumerate(segment_rows):
        short = kernel_short_name(row)
        if is_deepgemm_shape(row, 2112, 7168):
            keys[idx] = "qkv_downscale_gemm"
        elif is_deepgemm_shape(row, 24576, 1536):
            keys[idx] = "q_b_proj_gemm"
        elif is_deepgemm_shape(row, 7168, 16384):
            keys[idx] = "o_proj_gemm"
        elif short == "RMSNormKernel":
            norm_seen += 1
            keys[idx] = "q_a_layernorm" if norm_seen == 1 else "kv_a_layernorm"
        elif short.startswith("per_token_group_quant_8bit_kernel"):
            quant_seen += 1
            if quant_seen == 1:
                keys[idx] = "qkv_downscale_quant"
            elif quant_seen == 2:
                keys[idx] = "q_b_proj_quant"
            elif quant_seen == 3:
                keys[idx] = "o_proj_quant"
        elif short.startswith("nvjet_tst_"):
            nvjet_seen += 1
            keys[idx] = "q_w_kc_bmm" if nvjet_seen == 1 else "s_w_vc_bmm"
        elif short == "BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel":
            keys[idx] = "rotary_emb"
        elif short == "set_mla_kv_buffer_kernel":
            keys[idx] = "set_mla_kv_buffer"
        elif short == "prepare_varlen_num_blocks_kernel":
            keys[idx] = "fa3_prepare"
        elif short == "device_kernel" and str(row.get("implementation") or "") == "fa3":
            fa3_device_seen += 1
            keys[idx] = "fa3_attention" if fa3_device_seen == 1 else "fa3_combine"
    return keys


def summarize_names(rows: list[dict[str, Any]], limit: int = 10) -> str:
    counts: dict[str, int] = {}
    for row in rows:
        name = kernel_short_name(row)
        counts[name] = counts.get(name, 0) + 1
    top = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]
    return json.dumps([{"name": name, "count": count} for name, count in top], ensure_ascii=False)


def segment_decode_mla_layers(
    rows: list[dict[str, Any]],
    *,
    expected_layers: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    starts: list[int] = []
    for idx, row in enumerate(rows):
        if idx == 0:
            continue
        if is_deepgemm_shape(row, 2112, 7168) and rows[idx - 1].get("operator_category") == "quant":
            starts.append(idx - 1)

    segments: list[dict[str, Any]] = []
    warnings: list[str] = []
    for layer_id, start_idx in enumerate(starts):
        end_idx = None
        for idx in range(start_idx + 1, min(start_idx + 90, len(rows))):
            if is_deepgemm_shape(rows[idx], 7168, 16384) and rows[idx - 1].get("operator_category") == "quant":
                end_idx = idx
                break
        if end_idx is None:
            warnings.append(f"layer_start_index_{start_idx}_missing_o_proj_deepgemm")
            continue
        segment_rows = rows[start_idx : end_idx + 1]
        components = {key: 0.0 for key in REAL_COMPONENT_ORDER}
        counts = {key: 0 for key in REAL_COMPONENT_ORDER}
        semantic_keys = semantic_component_keys(segment_rows)
        semantic_components = {key: 0.0 for key in SEMANTIC_COMPONENT_ORDER}
        semantic_counts = {key: 0 for key in SEMANTIC_COMPONENT_ORDER}
        for row in segment_rows:
            key = component_key(row)
            components[key] += kernel_duration_ms(row)
            counts[key] += 1
        for row, key in zip(segment_rows, semantic_keys):
            semantic_components[key] += kernel_duration_ms(row)
            semantic_counts[key] += 1

        start_ns = min(kernel_start_ns(row) for row in segment_rows)
        end_ns = max(kernel_end_ns(row) for row in segment_rows)
        attention_rows = [row for row in segment_rows if component_key(row) == "attention"]
        bmm_rows = [row for row in segment_rows if component_key(row) == "bmm"]
        gemm_rows = [row for row in segment_rows if component_key(row) == "gemm"]
        segments.append(
            {
                "layer_id": layer_id,
                "start_kernel_order_index": int(segment_rows[0]["kernel_order_index"]),
                "end_kernel_order_index": int(segment_rows[-1]["kernel_order_index"]),
                "start_kernel_short_name": kernel_short_name(segment_rows[0]),
                "end_kernel_short_name": kernel_short_name(segment_rows[-1]),
                "kernel_count": len(segment_rows),
                "kernel_start_ns": start_ns,
                "kernel_end_ns": end_ns,
                "real_mla_makespan_ms": (end_ns - start_ns) / 1_000_000.0,
                "real_mla_kernel_sum_ms": sum(kernel_duration_ms(row) for row in segment_rows),
                "real_attention_kernel_sum_ms": sum(kernel_duration_ms(row) for row in attention_rows),
                "real_bmm_kernel_sum_ms": sum(kernel_duration_ms(row) for row in bmm_rows),
                "real_gemm_kernel_sum_ms": sum(kernel_duration_ms(row) for row in gemm_rows),
                "component_time_json": json.dumps(components, ensure_ascii=False),
                "component_count_json": json.dumps(counts, ensure_ascii=False),
                "semantic_time_json": json.dumps(semantic_components, ensure_ascii=False),
                "semantic_count_json": json.dumps(semantic_counts, ensure_ascii=False),
                "semantic_order_json": json.dumps(semantic_keys, ensure_ascii=False),
                "kernel_names_json": summarize_names(segment_rows),
                "attention_kernel_names_json": summarize_names(attention_rows),
            }
        )

    if len(segments) != expected_layers:
        warnings.append(f"segment_count_{len(segments)}_expected_{expected_layers}")
    return segments[:expected_layers], warnings


def query_result(op: ops.Operation, db: PerfDatabase, *, x: int, batch_size: int, s: int) -> tuple[float, str]:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        result = op.query(db, x=x, batch_size=batch_size, s=s, prefix=s - 1, beam_width=1, model_name="DeepSeek-V3")
    return float(result), str(getattr(result, "source", "silicon"))


def query_deepseek_fallback_stack(real_cases: pd.DataFrame, db: PerfDatabase) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, case in real_cases.iterrows():
        batch_size = int(case["batch_size"])
        s = int(case["aic_query_s"])
        tp_size = int(case["tp_size"])
        hidden_size = 7168
        num_heads = 128 // tp_size
        x_tokens = batch_size
        gemm_quant_mode = common.GEMMQuantMode.fp8_block
        bmm_quant_mode = common.GEMMQuantMode.fp8
        kvcache_quant_mode = common.KVCacheQuantMode.fp8
        stack_ops: list[tuple[str, ops.Operation]] = [
            ("generation_downscale_gemm", ops.GEMM("generation_downscale_gemm", 1, 2112, hidden_size, gemm_quant_mode)),
            ("generation_q_b_proj_gemm", ops.GEMM("generation_q_b_proj_gemm", 1, 24576 // tp_size, 1536, gemm_quant_mode)),
            ("generation_bmm_pre", ops.MLABmm("generation_bmm_pre", 1, num_heads, bmm_quant_mode, if_pre=True)),
            ("generation_attention", ops.GenerationMLA("generation_attention", 1, num_heads, kvcache_quant_mode)),
            ("generation_bmm_post", ops.MLABmm("generation_bmm_post", 1, num_heads, bmm_quant_mode, if_pre=False)),
            ("generation_proj_gemm", ops.GEMM("generation_proj_gemm", 1, hidden_size, hidden_size // tp_size, gemm_quant_mode)),
        ]
        for order, (op_name, op) in enumerate(stack_ops, start=1):
            latency_ms, source = query_result(op, db, x=x_tokens, batch_size=batch_size, s=s)
            rows.append(
                {
                    "tag": case["tag"],
                    "aic_model": "DeepSeekModel_fallback",
                    "op_name": op_name,
                    "op_order": order,
                    "latency_ms": latency_ms,
                    "source": source,
                    "batch_size": batch_size,
                    "kv_len": int(case["kv_len"]),
                    "s": s,
                    "tp_size": tp_size,
                    "num_heads": num_heads,
                    "x_tokens": x_tokens,
                    "gemm_quant_mode": gemm_quant_mode.name,
                    "bmm_quant_mode": bmm_quant_mode.name,
                    "kvcache_quant_mode": kvcache_quant_mode.name,
                }
            )
    return pd.DataFrame(rows)


def query_wideep_stack(real_cases: pd.DataFrame, db: PerfDatabase) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, case in real_cases.iterrows():
        batch_size = int(case["batch_size"])
        s = int(case["aic_query_s"])
        tp_size = int(case["tp_size"])
        hidden_size = 7168
        x_tokens = batch_size
        gemm_quant_mode = common.GEMMQuantMode.fp8_block
        kvcache_quant_mode = common.KVCacheQuantMode.fp8
        fmha_quant_mode = common.FMHAQuantMode.fp8_block
        stack_ops: list[tuple[str, str, ops.Operation]] = [
            (
                "generation_qkv_a_proj_gemm",
                "sdk_current_duplicate_diagnostic",
                ops.GEMM("generation_qkv_a_proj_gemm", 1, 2112, hidden_size, gemm_quant_mode),
            ),
            (
                "generation_downscale_gemm",
                "dedup_module_aligned",
                ops.GEMM("generation_downscale_gemm", 1, 2112, hidden_size, gemm_quant_mode),
            ),
            (
                "wideep_generation_mla",
                "dedup_module_aligned",
                ops.WideEPGenerationMLA(
                    "wideep_generation_mla",
                    1,
                    tp_size,
                    kvcache_quant_mode,
                    fmha_quant_mode,
                    "fa3",
                ),
            ),
        ]
        for order, (op_name, include_policy, op) in enumerate(stack_ops, start=1):
            latency_ms, source = query_result(op, db, x=x_tokens, batch_size=batch_size, s=s)
            rows.append(
                {
                    "tag": case["tag"],
                    "aic_model": "WideEPDeepSeekModel",
                    "op_name": op_name,
                    "op_order": order,
                    "include_policy": include_policy,
                    "latency_ms": latency_ms,
                    "source": source,
                    "batch_size": batch_size,
                    "kv_len": int(case["kv_len"]),
                    "s": s,
                    "tp_size": tp_size,
                    "x_tokens": x_tokens,
                    "gemm_quant_mode": gemm_quant_mode.name,
                    "kvcache_quant_mode": kvcache_quant_mode.name,
                    "fmha_quant_mode": fmha_quant_mode.name,
                    "attention_backend": "fa3",
                }
            )
    return pd.DataFrame(rows)


def query_standard_module_validation(real_cases: pd.DataFrame, db: PerfDatabase) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, case in real_cases.iterrows():
        batch_size = int(case["batch_size"])
        s = int(case["aic_query_s"])
        tp_size = int(case["tp_size"])
        num_heads = 128 // tp_size
        op = ops.MLAModule(
            "generation_mla_module",
            1,
            False,
            num_heads,
            common.KVCacheQuantMode.fp8,
            common.FMHAQuantMode.fp8_block,
            common.GEMMQuantMode.fp8_block,
        )
        try:
            latency_ms, source = query_result(op, db, x=batch_size, batch_size=batch_size, s=s)
            status = "success"
            message = ""
        except Exception as exc:
            latency_ms = math.nan
            source = ""
            status = "miss"
            message = f"{type(exc).__name__}: {exc}"
        rows.append(
            {
                "tag": case["tag"],
                "op_name": "generation_mla_module",
                "status": status,
                "latency_ms": latency_ms,
                "source": source,
                "message": message,
                "batch_size": batch_size,
                "kv_len": int(case["kv_len"]),
                "s": s,
                "tp_size": tp_size,
                "num_heads": num_heads,
            }
        )
    return pd.DataFrame(rows)


def read_real_segments(entries: list[ManifestEntry]) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    for entry in entries:
        run_meta = read_json(entry.raw_run_dir / "run_meta.json" if entry.raw_run_dir else None)
        if not is_default_fa3_decode(entry, run_meta):
            continue
        df = pd.read_csv(entry.csv_path)
        rows = df.sort_values("kernel_order_index").to_dict("records")
        num_layers = int(run_meta.get("num_layers") or 5)
        segments, warnings = segment_decode_mla_layers(rows, expected_layers=num_layers)
        if warnings:
            skipped_rows.append(
                {
                    "tag": entry.tag,
                    "reason": ";".join(warnings),
                    "csv_path": str(entry.csv_path),
                    "expected_layers": num_layers,
                    "actual_segments": len(segments),
                }
            )
        for segment in segments:
            detail_rows.append(
                {
                    "tag": entry.tag,
                    "run_instance_name": entry.csv_path.name.split("__", 1)[0],
                    "stage": "decode",
                    "batch_size": entry.batch_size,
                    "fresh_len": 1,
                    "kv_len": entry.kv_len,
                    "prefix_len": entry.kv_len,
                    "total_seq_len": entry.kv_len + 1,
                    "tp_size": entry.tp_size,
                    "num_layers": num_layers,
                    "attention_backend": entry.attention_backend,
                    "decode_attention_backend": "default_fa3",
                    "cuda_graph_mode": "on",
                    "aic_query_s": entry.kv_len + 1,
                    **segment,
                }
            )
    return pd.DataFrame(detail_rows), pd.DataFrame(skipped_rows)


def pct_gap(sim: float | None, real: float | None) -> float | None:
    if sim is None or real in (None, 0, 0.0) or pd.isna(sim) or pd.isna(real):
        return None
    return (float(sim) - float(real)) / float(real) * 100.0


def summarize_cases(real_detail: pd.DataFrame, deepseek_stack: pd.DataFrame, wideep_stack: pd.DataFrame) -> pd.DataFrame:
    component_rows: list[dict[str, Any]] = []
    semantic_rows: list[dict[str, Any]] = []
    for _, row in real_detail.iterrows():
        components = json.loads(row["component_time_json"])
        component_rows.append({"tag": row["tag"], **{f"real_component_{k}_sum_ms": v for k, v in components.items()}})
        semantic_components = json.loads(row["semantic_time_json"])
        semantic_rows.append(
            {"tag": row["tag"], **{f"real_semantic_{k}_sum_ms": v for k, v in semantic_components.items()}}
        )
    component_df = pd.DataFrame(component_rows)
    real_summary = (
        real_detail.groupby("tag", as_index=False)
        .agg(
            layer_sample_count=("layer_id", "count"),
            batch_size=("batch_size", "first"),
            kv_len=("kv_len", "first"),
            prefix_len=("prefix_len", "first"),
            total_seq_len=("total_seq_len", "first"),
            tp_size=("tp_size", "first"),
            real_mla_makespan_mean_ms=("real_mla_makespan_ms", "mean"),
            real_mla_makespan_median_ms=("real_mla_makespan_ms", "median"),
            real_mla_kernel_sum_mean_ms=("real_mla_kernel_sum_ms", "mean"),
            real_attention_kernel_sum_mean_ms=("real_attention_kernel_sum_ms", "mean"),
            real_bmm_kernel_sum_mean_ms=("real_bmm_kernel_sum_ms", "mean"),
            real_gemm_kernel_sum_mean_ms=("real_gemm_kernel_sum_ms", "mean"),
            aic_query_s=("aic_query_s", "first"),
        )
    )
    if not component_df.empty:
        component_summary = component_df.groupby("tag", as_index=False).mean(numeric_only=True)
        real_summary = real_summary.merge(component_summary, on="tag", how="left")
    semantic_df = pd.DataFrame(semantic_rows)
    if not semantic_df.empty:
        semantic_summary = semantic_df.groupby("tag", as_index=False).mean(numeric_only=True)
        real_summary = real_summary.merge(semantic_summary, on="tag", how="left")
    summary = real_summary.copy()
    wideep_dedup_stack = wideep_stack[wideep_stack["include_policy"].eq("dedup_module_aligned")].copy()
    for name, stack_df in [
        ("deepseek_fallback", deepseek_stack),
        ("wideep_dedup", wideep_dedup_stack),
        ("wideep_sdk_current", wideep_stack),
    ]:
        total = stack_df.groupby("tag", as_index=False).agg(**{f"aic_{name}_total_ms": ("latency_ms", "sum")})
        summary = summary.merge(total, on="tag", how="left")
        summary[f"aic_{name}_gap_pct_vs_real_makespan"] = summary.apply(
            lambda r: pct_gap(r.get(f"aic_{name}_total_ms"), r.get("real_mla_makespan_mean_ms")),
            axis=1,
        )
        summary[f"aic_{name}_gap_pct_vs_real_kernel_sum"] = summary.apply(
            lambda r: pct_gap(r.get(f"aic_{name}_total_ms"), r.get("real_mla_kernel_sum_mean_ms")),
            axis=1,
        )
    summary["real_makespan_gap_pct_vs_kernel_sum"] = summary.apply(
        lambda r: pct_gap(r.get("real_mla_makespan_mean_ms"), r.get("real_mla_kernel_sum_mean_ms")),
        axis=1,
    )
    summary["best_aic_model_vs_makespan"] = summary.apply(
        lambda r: (
            "DeepSeek fallback"
            if abs(float(r["aic_deepseek_fallback_gap_pct_vs_real_makespan"]))
            <= abs(float(r["aic_wideep_dedup_gap_pct_vs_real_makespan"]))
            else "WideEP module"
        ),
        axis=1,
    )
    summary["best_aic_total_ms_vs_makespan"] = summary.apply(
        lambda r: (
            float(r["aic_deepseek_fallback_total_ms"])
            if r["best_aic_model_vs_makespan"] == "DeepSeek fallback"
            else float(r["aic_wideep_dedup_total_ms"])
        ),
        axis=1,
    )
    summary["best_aic_gap_pct_vs_real_makespan"] = summary.apply(
        lambda r: pct_gap(r.get("best_aic_total_ms_vs_makespan"), r.get("real_mla_makespan_mean_ms")),
        axis=1,
    )
    return summary.sort_values(["kv_len", "batch_size", "tag"]).reset_index(drop=True)


def make_case_labels(summary: pd.DataFrame) -> dict[str, str]:
    return {row.tag: f"b{int(row.batch_size)} p{int(row.kv_len)}" for row in summary.itertuples(index=False)}


def stack_values(stack_df: pd.DataFrame, tags: list[str], component_order: list[str]) -> dict[str, list[float]]:
    values: dict[str, list[float]] = {}
    for op_name in component_order:
        series = []
        for tag in tags:
            match = stack_df[(stack_df["tag"] == tag) & (stack_df["op_name"] == op_name)]
            series.append(float(match["latency_ms"].iloc[0]) if not match.empty else 0.0)
        values[op_name] = series
    return values


def plot_compare(summary: pd.DataFrame, deepseek_stack: pd.DataFrame, wideep_stack: pd.DataFrame, out_dir: Path) -> None:
    apply_style()
    tags = list(summary["tag"])
    labels = make_case_labels(summary)
    x = np.arange(len(tags))
    width = 0.22
    fig, ax = plt.subplots(figsize=(max(14.5, len(tags) * 0.9), 8.1))

    real_makespan = summary["real_mla_makespan_mean_ms"].to_numpy(dtype=float)
    kernel_sum = summary["real_mla_kernel_sum_mean_ms"].to_numpy(dtype=float)
    ax.bar(
        x - 1.5 * width,
        real_makespan,
        width,
        color="#E2E5EA",
        edgecolor=COLORS["neutral_dark"],
        linewidth=0.45,
        label="real MLA makespan mean",
    )
    ax.bar(
        x - 0.5 * width,
        kernel_sum,
        width,
        color="#F3F4F7",
        edgecolor=COLORS["neutral_mid"],
        linewidth=0.45,
        hatch="..",
        label="real MLA kernel sum mean",
    )

    def plot_stack(
        stack_df: pd.DataFrame,
        component_order: list[str],
        component_colors: dict[str, str],
        xpos: np.ndarray,
        label_prefix: str,
        hatch: str = "",
    ) -> np.ndarray:
        bottom = np.zeros(len(tags))
        values = stack_values(stack_df, tags, component_order)
        for op_name in component_order:
            component_values = np.array(values[op_name])
            ax.bar(
                xpos,
                component_values,
                width,
                bottom=bottom,
                color=component_colors.get(op_name, COLORS["neutral"]),
                edgecolor=COLORS["neutral_dark"],
                linewidth=0.35,
                hatch=hatch,
                label=f"{label_prefix} {op_name}",
            )
            bottom += component_values
        return bottom

    deepseek_total = plot_stack(
        deepseek_stack,
        DEEPSEEK_FALLBACK_ORDER,
        DEEPSEEK_FALLBACK_COLORS,
        x + 0.5 * width,
        "DS",
    )
    wideep_dedup_stack = wideep_stack[wideep_stack["include_policy"].eq("dedup_module_aligned")].copy()
    wideep_total = plot_stack(
        wideep_dedup_stack,
        WIDEEP_DEDUP_ORDER,
        WIDEEP_COLORS,
        x + 1.5 * width,
        "WE",
        hatch="//",
    )
    y_max = max(np.nanmax(np.r_[real_makespan, kernel_sum, deepseek_total, wideep_total]), 0.001)

    for i, tag in enumerate(tags):
        real_value = float(summary.loc[summary["tag"] == tag, "real_mla_makespan_mean_ms"].iloc[0])
        ksum_value = float(summary.loc[summary["tag"] == tag, "real_mla_kernel_sum_mean_ms"].iloc[0])
        ds_value = float(summary.loc[summary["tag"] == tag, "aic_deepseek_fallback_total_ms"].iloc[0])
        we_value = float(summary.loc[summary["tag"] == tag, "aic_wideep_dedup_total_ms"].iloc[0])
        ds_gap = pct_gap(ds_value, real_value)
        we_gap = pct_gap(we_value, real_value)
        best_label = "DS" if abs(float(ds_gap)) <= abs(float(we_gap)) else "WE"
        best_gap = ds_gap if best_label == "DS" else we_gap
        group_top = max(real_value, ksum_value, ds_value, we_value)
        for xpos, value in [
            (x[i] - 1.5 * width, real_value),
            (x[i] - 0.5 * width, ksum_value),
            (x[i] + 0.5 * width, ds_value),
            (x[i] + 1.5 * width, we_value),
        ]:
            ax.text(
                xpos,
                value + y_max * 0.016,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=6.8,
                color=TOKENS["ink"],
                rotation=90,
            )
        label_specs = [
            (f"DS {ds_gap:+.0f}%", COLORS["orange_mid"]),
            (f"WE {we_gap:+.0f}%", COLORS["blue_mid"]),
            (f"Best {best_gap:+.0f}%", COLORS["olive_mid"]),
        ]
        for offset, (label, color) in enumerate(label_specs):
            ax.text(
                x[i],
                group_top + y_max * (0.105 + offset * 0.052),
                label,
                ha="center",
                va="bottom",
                fontsize=8.3,
                fontweight="bold",
                color=color,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([labels[tag] for tag in tags], rotation=45, ha="right")
    ax.set_xlabel("Decode case")
    ax.set_ylabel("Latency per layer (ms)")
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    ax.set_ylim(0, y_max * 1.56)
    handles, labels_ = ax.get_legend_handles_labels()
    seen: set[str] = set()
    keep = []
    for h, label in zip(handles, labels_):
        if label in seen:
            continue
        seen.add(label)
        if (
            label.startswith("real")
            or label.startswith("DS")
            or label.startswith("WE")
        ):
            keep.append((h, label))
    ax.legend(
        [h for h, _ in keep],
        [label for _, label in keep],
        loc="upper left",
        bbox_to_anchor=(0.0, 1.03),
        ncol=4,
        frameon=False,
        fontsize=7.7,
    )
    add_chart_header(
        fig,
        ax,
        "Decode CUDA Graph MLA module envelope vs AIC",
        (
            "Real baseline is inferred from CUDA kernel sequence: quant before deepgemm(2112,7168) "
            "through deepgemm(7168,16384). AIC bars show DeepSeek fallback stack and WideEP generation MLA path."
        ),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    for ext in ["png", "svg"]:
        fig.savefig(out_dir / f"decode_mla_kernel_envelope_aic_compare.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_real_components(summary: pd.DataFrame, out_dir: Path) -> None:
    apply_style()
    tags = list(summary["tag"])
    labels = make_case_labels(summary)
    x = np.arange(len(tags))
    fig, ax = plt.subplots(figsize=(max(14.0, len(tags) * 0.88), 7.6))
    bottom = np.zeros(len(tags))
    for component in SEMANTIC_COMPONENT_ORDER:
        col = f"real_semantic_{component}_sum_ms"
        if col not in summary:
            continue
        values = summary[col].fillna(0.0).to_numpy(dtype=float)
        if not np.any(values):
            continue
        ax.bar(
            x,
            values,
            bottom=bottom,
            color=SEMANTIC_COMPONENT_COLORS.get(component, COLORS["neutral"]),
            edgecolor=COLORS["neutral_dark"],
            linewidth=0.45,
            label=SEMANTIC_COMPONENT_LABELS.get(component, component),
        )
        bottom += values
    ax.plot(x, summary["real_mla_makespan_mean_ms"].to_numpy(dtype=float), color=COLORS["neutral_dark"], marker="o", label="makespan")
    ax.set_xticks(x)
    ax.set_xticklabels([labels[tag] for tag in tags], rotation=45, ha="right")
    ax.set_xlabel("Decode case")
    ax.set_ylabel("Latency per layer (ms)")
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.05), ncol=4, frameon=False, fontsize=7.5)
    add_chart_header(
        fig,
        ax,
        "Real decode MLA kernel-sum decomposition by execution semantics",
        "Bars are ordered by the CUDA kernel sequence inside the inferred MLA range; the line is first-kernel-start to last-kernel-end makespan.",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.82))
    for stem in ["decode_mla_real_kernel_components", "decode_mla_real_kernel_semantic_components"]:
        for ext in ["png", "svg"]:
            fig.savefig(out_dir / f"{stem}.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def summarize_mapes(summary: pd.DataFrame) -> dict[str, float]:
    metrics = {
        "deepseek_fallback_vs_makespan_mape_pct": "aic_deepseek_fallback_gap_pct_vs_real_makespan",
        "wideep_dedup_vs_makespan_mape_pct": "aic_wideep_dedup_gap_pct_vs_real_makespan",
        "wideep_sdk_current_vs_makespan_mape_pct": "aic_wideep_sdk_current_gap_pct_vs_real_makespan",
        "deepseek_fallback_vs_kernel_sum_mape_pct": "aic_deepseek_fallback_gap_pct_vs_real_kernel_sum",
        "wideep_dedup_vs_kernel_sum_mape_pct": "aic_wideep_dedup_gap_pct_vs_real_kernel_sum",
        "wideep_sdk_current_vs_kernel_sum_mape_pct": "aic_wideep_sdk_current_gap_pct_vs_real_kernel_sum",
    }
    out: dict[str, float] = {}
    for key, column in metrics.items():
        values = summary[column].dropna().abs()
        out[key] = float(values.mean()) if not values.empty else math.nan
    return out


def write_readme(out_dir: Path, summary: pd.DataFrame, validation: pd.DataFrame, skipped: pd.DataFrame) -> None:
    mapes = summarize_mapes(summary)
    validation_miss = validation["status"].eq("miss").all() if not validation.empty else False
    lines = [
        "# Decode CUDA Graph MLA Kernel Envelope AIC Compare",
        "",
        "## 口径",
        "",
        "- 仅保留 `decode_manifest.csv` 中 `attention_backend=auto/default`、未显式指定 `flashmla`、且 CUDA Graph 开启的 decode case。",
        "- 实机侧没有可用的 host/module NVTX 边界，因此按 kernel 语义切 self_attn/MLA module：从 `quant` 后接 `deepgemm(2112,7168)` 的位置开始，到后续 `quant -> deepgemm(7168,16384)` 的 `o_proj` 结束。",
        "- `FusedAddRMSNorm`、MoE gate/up/down、activation 等后续 kernel 不计入 MLA module。",
        "- 主实机指标是 `real_mla_makespan_mean_ms`：每层首 kernel start 到尾 kernel end 的包络时间；同时保留 `real_mla_kernel_sum_mean_ms` 和组件拆分。",
        "- AIC 侧比较 `DeepSeekModel` fallback 小算子累加，以及去重后的 WideEP module 路径：一个 `2112x7168` GEMM 加 `wideep_generation_mla(fa3)`。",
        "- 当前 SDK 原样 WideEP 路径会额外包含一次 `generation_qkv_a_proj_gemm`，该值保留为 `aic_wideep_sdk_current_*` 诊断列，但主图和 `aic_wideep_dedup_*` 采用去重对齐口径。",
        "- 标准 `generation_mla_module` 查询结果单独放在 `decode_aic_generation_mla_module_validation.csv`；当前 0.5.9 数据目录没有对应 silicon 文件时应为 miss。",
        "",
        "## 结果概览",
        "",
        f"- 可比较 case 数: `{len(summary)}`",
        f"- 跳过/告警记录数: `{len(skipped)}`",
        f"- `generation_mla_module` 全部 miss: `{validation_miss}`",
        f"- DeepSeek fallback vs makespan MAPE: `{mapes['deepseek_fallback_vs_makespan_mape_pct']:.2f}%`",
        f"- WideEP module dedup vs makespan MAPE: `{mapes['wideep_dedup_vs_makespan_mape_pct']:.2f}%`",
        f"- WideEP SDK-current vs makespan MAPE: `{mapes['wideep_sdk_current_vs_makespan_mape_pct']:.2f}%`",
        f"- DeepSeek fallback vs kernel_sum MAPE: `{mapes['deepseek_fallback_vs_kernel_sum_mape_pct']:.2f}%`",
        f"- WideEP module dedup vs kernel_sum MAPE: `{mapes['wideep_dedup_vs_kernel_sum_mape_pct']:.2f}%`",
        f"- WideEP SDK-current vs kernel_sum MAPE: `{mapes['wideep_sdk_current_vs_kernel_sum_mape_pct']:.2f}%`",
        "",
        "## 输出文件",
        "",
        "- `decode_real_mla_layer_segments.csv`: 每个 case 每层的实机 MLA kernel range、makespan、kernel_sum、组件拆分和 kernel 名摘要。",
        "- `decode_compare_summary.csv`: case 级汇总，含实机均值、AIC 总时延和误差。",
        "- `decode_aic_deepseek_fallback_stack.csv`: DeepSeek fallback 小算子查表明细。",
        "- `decode_aic_wideep_stack.csv`: WideEP generation module 路径查表明细，含 `include_policy` 标识去重主口径与 SDK 当前重复诊断行。",
        "- `decode_aic_generation_mla_module_validation.csv`: 标准 `generation_mla_module` 查询校验。",
        "- `decode_mla_kernel_envelope_aic_compare.png/svg`: 主对比图。",
        "- `decode_mla_real_kernel_components.png/svg`: 实机 MLA range 内 kernel_sum 细粒度语义组件拆分图。",
        "- `decode_mla_real_kernel_semantic_components.png/svg`: 同上，保留一个语义明确的文件名副本。",
        "- `artifact_manifest.json`: 产物清单。",
        "",
        "## 边界说明",
        "",
        "本目录采用完整 `o_proj` 结束边界。较早只截到第二个 `nvjet_tst` 的口径会漏掉最后的 `quant + deepgemm(7168,16384)`，因此不是完整 MLA module。",
        "如果后续 SGLang kernel 名或 DeepGEMM template 参数发生变化，需要优先检查 `segment_decode_mla_layers()` 的两个 shape 匹配条件。",
    ]
    out_dir.joinpath("README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_artifact_manifest(out_dir: Path, summary: pd.DataFrame, real_detail: pd.DataFrame) -> None:
    manifest = {
        "output_dir": str(out_dir),
        "case_count": int(len(summary)),
        "real_layer_segment_count": int(len(real_detail)),
        "files": sorted(path.name for path in out_dir.iterdir() if path.is_file()),
        "boundary": "quant_before_deepgemm_2112x7168_to_deepgemm_7168x16384_end",
    }
    out_dir.joinpath("artifact_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    bench_root = args.bench_root
    out_dir = args.output_dir or bench_root / "analysis" / "decode_mla_kernel_envelope_aic_compare"
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.getLogger("aiconfigurator.sdk.interpolation").setLevel(logging.ERROR)
    db = PerfDatabase(
        system=args.system,
        backend=args.backend,
        version=args.version,
        systems_root=str(args.systems_root),
    )
    manifest_path = args.manifest or bench_root / "decode_manifest.csv"
    entries = load_manifest(manifest_path)
    real_detail, skipped = read_real_segments(entries)
    if real_detail.empty:
        raise RuntimeError("No default FA3 CUDA Graph decode MLA segments were found.")
    real_cases = (
        real_detail.groupby("tag", as_index=False)
        .agg(
            batch_size=("batch_size", "first"),
            kv_len=("kv_len", "first"),
            tp_size=("tp_size", "first"),
            aic_query_s=("aic_query_s", "first"),
        )
        .sort_values(["kv_len", "batch_size", "tag"])
        .reset_index(drop=True)
    )
    deepseek_stack = query_deepseek_fallback_stack(real_cases, db)
    wideep_stack = query_wideep_stack(real_cases, db)
    validation = query_standard_module_validation(real_cases, db)
    summary = summarize_cases(real_detail, deepseek_stack, wideep_stack)

    real_detail.to_csv(out_dir / "decode_real_mla_layer_segments.csv", index=False)
    skipped.to_csv(out_dir / "decode_skipped_or_warnings.csv", index=False)
    deepseek_stack.to_csv(out_dir / "decode_aic_deepseek_fallback_stack.csv", index=False)
    wideep_stack.to_csv(out_dir / "decode_aic_wideep_stack.csv", index=False)
    validation.to_csv(out_dir / "decode_aic_generation_mla_module_validation.csv", index=False)
    summary.to_csv(out_dir / "decode_compare_summary.csv", index=False)

    plot_compare(summary, deepseek_stack, wideep_stack, out_dir)
    plot_real_components(summary, out_dir)
    write_readme(out_dir, summary, validation, skipped)
    write_artifact_manifest(out_dir, summary, real_detail)
    print(f"Wrote decode MLA comparison artifacts to {out_dir}")
    print(
        summary[
            [
                "tag",
                "real_mla_makespan_mean_ms",
                "real_mla_kernel_sum_mean_ms",
                "aic_deepseek_fallback_total_ms",
                "aic_wideep_dedup_total_ms",
                "aic_wideep_sdk_current_total_ms",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
