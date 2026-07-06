#!/usr/bin/env python3
"""Compare real Nsight MLA child operators with AIC run_agg MLA operator ops.

The comparison is single-layer and MLA-only:

- Real side: non-first-layer average GPU makespan from MLA时延拆解.csv child_timing_json.
- AIC side: persisted run_agg block values are preserved, and MLA granular
  operator attribution is reconstructed from the current SDK PrefixConditionalOp
  prefix path with the same run_agg observed_first_batch shapes.

Some real child ranges have no one-to-one AIC op and vice versa; those rows are
kept with boundary notes instead of being silently dropped.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[6]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from compare_agg_run_agg import build_session, make_case_label  # noqa: E402
from build_run_agg_mla_only_breakdown import derive_run_agg_step_shape, expand_real_mla, safe_json_loads  # noqa: E402


DEFAULT_JOINED = SCRIPT_DIR / "real_aic_joined.csv"
DEFAULT_OUT_DIR = SCRIPT_DIR / "run_agg_mla_operator_compare"

TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
}

COLORS = {
    "real": "#F0986E",
    "aic": "#A3BEFA",
    "ratio": "#A3D576",
    "missing": "#E2E5EA",
    "edge": "#464C55",
    "parity": "#7A828F",
}

OPERATOR_SPECS = [
    {
        "operator_family": "qkv_a_proj",
        "display_name": "qkv_a_proj",
        "real_components": ["fused_qkv_a_proj_with_mqa"],
        "aic_ops": ["context_downscale_gemm"],
        "boundary_note": (
            "Real fused_qkv_a_proj_with_mqa includes q_a/kv_a projection path and fp8 activation quant kernel. "
            "AIC context_downscale_gemm is a standalone SDK GEMM op before context_mla_block; collector fp8 GEMM "
            "uses pre-quantized inputs and usually excludes runtime activation quantization."
        ),
    },
    {
        "operator_family": "q_a_layernorm",
        "display_name": "q_a_layernorm",
        "real_components": ["q_a_layernorm"],
        "aic_ops": [],
        "boundary_note": "Real has an explicit q_a RMSNorm child range; ordinary DeepSeekModel MLA path does not model it as a separate SDK op.",
    },
    {
        "operator_family": "kv_a_layernorm",
        "display_name": "kv_a_layernorm",
        "real_components": ["kv_a_layernorm"],
        "aic_ops": [],
        "boundary_note": "Real has an explicit kv_a RMSNorm child range; ordinary DeepSeekModel MLA path does not model it as a separate SDK op.",
    },
    {
        "operator_family": "q_b_proj",
        "display_name": "q_b_proj",
        "real_components": ["q_b_proj"],
        "aic_ops": ["context_q_b_proj_gemm"],
        "boundary_note": "Both sides represent q_b projection, but real fp8 path includes activation quantization plus DeepGEMM kernels.",
    },
    {
        "operator_family": "kv_b_proj",
        "display_name": "kv_b_proj",
        "real_components": ["kv_b_proj"],
        "aic_ops": ["context_kv_b_proj_gemm"],
        "boundary_note": "Present as a separate real child mainly for attn_mha/prefill cases; MQA cases often fold or omit this child.",
    },
    {
        "operator_family": "mla_concat_k",
        "display_name": "mla_concat_k",
        "real_components": [],
        "aic_ops": ["context_mla_concat_k"],
        "boundary_note": "AIC prefix/granular path models SGLang MLA K concat as a separate op; real trace usually exposes concat_mla_k_kernel inside surrounding child ranges.",
    },
    {
        "operator_family": "rotary_emb",
        "display_name": "rotary_emb",
        "real_components": ["rotary_emb"],
        "aic_ops": [],
        "boundary_note": "Real has an explicit RoPE child range; AIC context MLA path does not expose a separate RoPE SDK op.",
    },
    {
        "operator_family": "attention",
        "display_name": "attention",
        "real_components": ["attn_mha", "attn_mqa"],
        "aic_ops": ["context_attention"],
        "boundary_note": (
            "Real attention may be attn_mha or attn_mqa depending on SGLang scheduling/backend choice. "
            "AIC context_attention is queried with the second-pass run_agg context-attention shape and scale correction."
        ),
    },
    {
        "operator_family": "o_proj",
        "display_name": "o_proj",
        "real_components": ["o_proj"],
        "aic_ops": ["context_proj_gemm"],
        "boundary_note": "Both sides represent output projection, but real fp8 path includes activation quantization plus DeepGEMM kernels.",
    },
    {
        "operator_family": "mla_total",
        "display_name": "MLA total",
        "real_components": [
            "fused_qkv_a_proj_with_mqa",
            "q_a_layernorm",
            "kv_a_layernorm",
            "q_b_proj",
            "kv_b_proj",
            "rotary_emb",
            "attn_mha",
            "attn_mqa",
            "o_proj",
        ],
        "aic_ops": [
            "context_downscale_gemm",
            "context_q_b_proj_gemm",
            "context_kv_b_proj_gemm",
            "context_mla_concat_k",
            "context_attention",
            "context_proj_gemm",
        ],
        "boundary_note": "Roll-up of MLA child/operator pieces only; excludes MoE/FFN/embedding/logits and excludes generation-only steps.",
    },
]


def use_chart_theme() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": TOKENS["surface"],
            "figure.edgecolor": "none",
            "savefig.facecolor": TOKENS["panel"],
            "savefig.edgecolor": "none",
            "axes.facecolor": TOKENS["panel"],
            "axes.edgecolor": TOKENS["axis"],
            "axes.labelcolor": TOKENS["ink"],
            "axes.grid": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": TOKENS["grid"],
            "grid.linewidth": 0.8,
            "font.family": "sans-serif",
            "font.sans-serif": ["Aptos", "Inter", "Segoe UI", "DejaVu Sans", "Arial", "sans-serif"],
            "font.monospace": ["SF Mono", "Menlo", "Consolas", "DejaVu Sans Mono", "monospace"],
        }
    )


def add_chart_header(fig, ax, title: str, subtitle: str, *, width: int = 112) -> None:
    title = textwrap.fill(title.strip(), width=88, break_long_words=False)
    subtitle = textwrap.fill(subtitle.strip(), width=width, break_long_words=False)
    title_lines = title.count("\n") + 1
    subtitle_lines = subtitle.count("\n") + 1
    fig.subplots_adjust(top=max(0.72, 0.88 - 0.045 * (title_lines - 1) - 0.03 * (subtitle_lines - 1)))
    left = ax.get_position().x0
    fig.text(left, 0.985, title, ha="left", va="top", fontsize=14, fontweight="semibold", color=TOKENS["ink"])
    fig.text(
        left,
        0.932 - 0.045 * (title_lines - 1),
        subtitle,
        ha="left",
        va="top",
        fontsize=9.5,
        color=TOKENS["muted"],
        linespacing=1.16,
    )


def save_chart(fig, out_dir: Path, stem: str) -> None:
    fig.savefig(out_dir / f"{stem}.png", dpi=180, bbox_inches="tight", facecolor=TOKENS["panel"], transparent=False)
    fig.savefig(out_dir / f"{stem}.svg", bbox_inches="tight", facecolor=TOKENS["panel"], transparent=False)
    plt.close(fig)


def build_session_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        model_path=args.model_path,
        system=args.system,
        backend=args.backend,
        version=args.version,
        database_mode=args.database_mode,
        tp_size=args.tp_size,
        pp_size=args.pp_size,
        attention_dp_size=args.attention_dp_size,
        moe_tp_size=args.moe_tp_size,
        moe_ep_size=args.moe_ep_size,
        num_layers=args.num_layers,
        gemm_quant_mode=args.gemm_quant_mode,
        moe_quant_mode=args.moe_quant_mode,
        kvcache_quant_mode=args.kvcache_quant_mode,
        fmha_quant_mode=args.fmha_quant_mode,
        comm_quant_mode=args.comm_quant_mode,
        attention_backend=args.attention_backend,
        moe_backend=args.moe_backend,
        enable_wideep=args.enable_wideep,
    )


def short_case_label(row: pd.Series, case_index: int) -> str:
    try:
        designed = make_case_label(row, observed=False).replace("\n", " | ")
    except Exception:
        designed = str(row.get("tag", "case"))
    p_b = int(row.get("prefill_batch_size", 0))
    p_fresh = int(row.get("prefill_fresh_len", 0))
    p_prefix = int(row.get("prefill_prefix_len", 0))
    d_b = int(row.get("decode_batch_size", 0))
    d_kv = int(row.get("decode_prefix_len", 0))
    return f"C{case_index:02d}\nP{p_b}x{p_fresh}+{p_prefix}\nD{d_b}x{d_kv}", designed


def summarize_real_by_component(real_child: pd.DataFrame) -> pd.DataFrame:
    if real_child.empty:
        return real_child
    df = real_child.copy()
    df["first_layer_id"] = df.groupby("tag")["layer_id"].transform("min")
    nonfirst = df[df["layer_id"] != df["first_layer_id"]].copy()
    if nonfirst.empty:
        nonfirst = df
    return (
        nonfirst.groupby(["tag", "real_component"], dropna=False)
        .agg(
            real_layer_count=("layer_id", "nunique"),
            real_gpu_makespan_ms=("gpu_makespan_ms", "mean"),
            real_gpu_kernel_time_sum_ms=("gpu_kernel_time_sum_ms", "mean"),
            real_host_duration_ms=("host_duration_ms", "mean"),
            real_first_kernel_name=("first_kernel_name", "first"),
            real_last_kernel_name=("last_kernel_name", "first"),
        )
        .reset_index()
    )


def summarize_real_kernels(real_kernel: pd.DataFrame) -> pd.DataFrame:
    if real_kernel.empty:
        return real_kernel
    df = real_kernel.copy()
    df["first_layer_id"] = df.groupby("tag")["layer_id"].transform("min")
    nonfirst = df[df["layer_id"] != df["first_layer_id"]].copy()
    if nonfirst.empty:
        nonfirst = df
    return (
        nonfirst.groupby(["tag", "real_component", "kernel_short_name"], dropna=False)
        .agg(avg_kernel_gpu_time_ms=("kernel_gpu_time_ms", "mean"), rows=("kernel_gpu_time_ms", "size"))
        .reset_index()
        .sort_values(["tag", "real_component", "avg_kernel_gpu_time_ms"], ascending=[True, True, False])
    )


def _find_op_by_name(ops: list[Any], name: str) -> Any | None:
    for op in ops:
        if getattr(op, "_name", None) == name:
            return op
    return None


def _query_sdk_op(
    op: Any,
    database: Any,
    *,
    batch_size: int,
    s: int,
    prefix: int,
    model_name: str,
    scale_divisor: float = 1.0,
) -> dict[str, Any]:
    kwargs = {
        "x": batch_size * s,
        "batch_size": batch_size,
        "beam_width": 1,
        "s": s,
        "prefix": prefix,
        "model_name": model_name,
    }
    result = op.query(database, **kwargs)
    latency = float(result) / scale_divisor
    energy = float(getattr(result, "energy", 0.0) or 0.0) / scale_divisor
    return {
        "latency_ms": latency,
        "energy_wms": energy,
        "source": getattr(result, "source", "silicon"),
    }


def _append_aic_row(
    rows: list[dict[str, Any]],
    row: pd.Series,
    shape: dict[str, Any],
    *,
    op_name: str,
    latency_ms: float,
    energy_wms: float = 0.0,
    source: str = "silicon",
    query_path: str,
    query_batch_size: int | None = None,
    query_s: int | None = None,
    query_prefix: int | None = None,
    num_layers: int,
    error: str = "",
) -> None:
    step_count = float(row.get("aic_num_mix_steps", 0.0) or 0.0)
    rows.append(
        {
            "tag": row["tag"],
            "mapping": row["mapping"],
            "alignment_class": row["alignment_class"],
            "step": "mix_step",
            "phase": "context",
            "aic_query_path": query_path,
            "op_name": op_name,
            "latency_ms": latency_ms,
            "per_layer_latency_ms": latency_ms / num_layers if num_layers and pd.notna(latency_ms) else np.nan,
            "step_count": step_count,
            "weighted_latency_ms": latency_ms * step_count if pd.notna(latency_ms) else np.nan,
            "energy_wms": energy_wms,
            "source": source,
            "query_batch_size": query_batch_size,
            "query_s": query_s,
            "query_prefix": query_prefix,
            "query_error": error,
            **shape,
        }
    )


def expand_aic_mla_operator_ops(joined: pd.DataFrame, session: Any, num_layers: int) -> pd.DataFrame:
    """Reconstruct MLA-relevant AIC op timings for the AGG mixed context step.

    The persisted run_agg CSV exposes `context_mla_block` as a block value. For
    operator attribution, this function also queries the SDK's granular prefix
    path directly:

    - qkv/downscale and q_b/kv_b/concat/o_proj use run_agg's first static_ctx
      pass shape (non-attention path).
    - context_attention uses run_agg's second static_ctx pass shape and the same
      scale correction used in SGLangBackend.run_agg.

    Rows are tagged by `aic_query_path` so the report can distinguish actual
    persisted block values from diagnostic granular reconstruction.
    """

    rows: list[dict[str, Any]] = []
    model = session._model
    database = session._database
    model_name = str(getattr(model, "model_name", "") or "")
    context_downscale = _find_op_by_name(model.context_ops, "context_downscale_gemm")
    context_mla_block = _find_op_by_name(model.context_ops, "context_mla_block")
    prefix_ops = list(getattr(context_mla_block, "_prefix_ops", []) or [])
    granular_ops = {getattr(op, "_name", type(op).__name__): op for op in prefix_ops}

    for _, row in joined.drop_duplicates(["tag", "mapping"]).iterrows():
        shape = derive_run_agg_step_shape(row)
        per_ops = safe_json_loads(row.get("aic_per_ops_json"), {})
        mix_ops = per_ops.get("mix_step", {}) if isinstance(per_ops, dict) else {}
        if isinstance(mix_ops, dict) and "context_mla_block" in mix_ops:
            _append_aic_row(
                rows,
                row,
                shape,
                op_name="context_mla_block",
                latency_ms=float(mix_ops["context_mla_block"]),
                source="persisted_run_agg_csv",
                query_path="run_agg_snapshot_block",
                num_layers=num_layers,
            )

        first_pass_prefix = int(shape["mix_first_pass_prefix"])
        first_pass_s = int(shape["mix_first_pass_isl"] - first_pass_prefix)
        first_pass_batch = int(shape["mix_first_pass_batch_size"])
        first_pass_ops = [("context_downscale_gemm", context_downscale)] + [
            (name, granular_ops.get(name))
            for name in (
                "context_q_b_proj_gemm",
                "context_kv_b_proj_gemm",
                "context_mla_concat_k",
                "context_proj_gemm",
            )
        ]

        for op_name, op in first_pass_ops:
            if op is None:
                _append_aic_row(
                    rows,
                    row,
                    shape,
                    op_name=op_name,
                    latency_ms=np.nan,
                    source="missing_sdk_op",
                    query_path="diagnostic_forced_prefix_ops",
                    query_batch_size=first_pass_batch,
                    query_s=first_pass_s,
                    query_prefix=first_pass_prefix,
                    num_layers=num_layers,
                    error=f"{op_name} not found in current model ops",
                )
                continue
            try:
                result = _query_sdk_op(
                    op,
                    database,
                    batch_size=first_pass_batch,
                    s=first_pass_s,
                    prefix=first_pass_prefix,
                    model_name=model_name,
                )
                _append_aic_row(
                    rows,
                    row,
                    shape,
                    op_name=op_name,
                    latency_ms=result["latency_ms"],
                    energy_wms=result["energy_wms"],
                    source=result["source"],
                    query_path="diagnostic_forced_prefix_ops",
                    query_batch_size=first_pass_batch,
                    query_s=first_pass_s,
                    query_prefix=first_pass_prefix,
                    num_layers=num_layers,
                )
            except Exception as exc:
                _append_aic_row(
                    rows,
                    row,
                    shape,
                    op_name=op_name,
                    latency_ms=np.nan,
                    source="query_error",
                    query_path="diagnostic_forced_prefix_ops",
                    query_batch_size=first_pass_batch,
                    query_s=first_pass_s,
                    query_prefix=first_pass_prefix,
                    num_layers=num_layers,
                    error=f"{type(exc).__name__}: {exc}",
                )

        attn_op = granular_ops.get("context_attention")
        second_pass_batch = int(shape["mix_context_second_pass_batch_size"])
        second_pass_prefix = int(shape["mix_context_second_pass_prefix"])
        second_pass_s = int(shape["mix_context_second_pass_isl"] - second_pass_prefix)
        # This mirrors SGLangBackend.run_agg: ctx_attention / ceil(isl / ctx_tokens).
        scale_factor = float(math.ceil(float(row["aic_isl"]) / float(row["aic_ctx_tokens"])))
        scale_factor = max(scale_factor, 1.0)
        if attn_op is None:
            _append_aic_row(
                rows,
                row,
                shape,
                op_name="context_attention",
                latency_ms=np.nan,
                source="missing_sdk_op",
                query_path="diagnostic_forced_prefix_ops_attention_second_pass",
                query_batch_size=second_pass_batch,
                query_s=second_pass_s,
                query_prefix=second_pass_prefix,
                num_layers=num_layers,
                error="context_attention not found in current context_mla_block prefix ops",
            )
        else:
            try:
                result = _query_sdk_op(
                    attn_op,
                    database,
                    batch_size=second_pass_batch,
                    s=second_pass_s,
                    prefix=second_pass_prefix,
                    model_name=model_name,
                    scale_divisor=scale_factor,
                )
                _append_aic_row(
                    rows,
                    row,
                    shape,
                    op_name="context_attention",
                    latency_ms=result["latency_ms"],
                    energy_wms=result["energy_wms"],
                    source=result["source"],
                    query_path="diagnostic_forced_prefix_ops_attention_second_pass",
                    query_batch_size=second_pass_batch,
                    query_s=second_pass_s,
                    query_prefix=second_pass_prefix,
                    num_layers=num_layers,
                )
            except Exception as exc:
                _append_aic_row(
                    rows,
                    row,
                    shape,
                    op_name="context_attention",
                    latency_ms=np.nan,
                    source="query_error",
                    query_path="diagnostic_forced_prefix_ops_attention_second_pass",
                    query_batch_size=second_pass_batch,
                    query_s=second_pass_s,
                    query_prefix=second_pass_prefix,
                    num_layers=num_layers,
                    error=f"{type(exc).__name__}: {exc}",
                )

    return pd.DataFrame(rows)


def summarize_aic_context_ops(aic_long: pd.DataFrame) -> pd.DataFrame:
    df = aic_long[
        aic_long["mapping"].eq("observed_first_batch")
        & aic_long["step"].eq("mix_step")
        & aic_long["phase"].eq("context")
        & aic_long["aic_query_path"].str.startswith("diagnostic_forced_prefix_ops", na=False)
    ].copy()
    if df.empty:
        return df
    grouped = (
        df.groupby(["tag", "op_name"], dropna=False)
        .agg(
            aic_latency_ms=("latency_ms", "sum"),
            aic_per_layer_latency_ms=("per_layer_latency_ms", "sum"),
            aic_weighted_latency_ms=("weighted_latency_ms", "sum"),
            aic_source=("source", "first"),
            aic_query_path=("aic_query_path", "first"),
        )
        .reset_index()
    )
    shape_rows = []
    for (tag, op_name), sub in df.groupby(["tag", "op_name"], dropna=False):
        shape_rows.append(
            {
                "tag": tag,
                "op_name": op_name,
                "aic_query_shapes_json": json.dumps(
                    [
                        {
                            "batch_size": int(r.query_batch_size) if pd.notna(r.query_batch_size) else None,
                            "s": int(r.query_s) if pd.notna(r.query_s) else None,
                            "prefix": int(r.query_prefix) if pd.notna(r.query_prefix) else None,
                            "path": r.aic_query_path,
                        }
                        for r in sub.itertuples(index=False)
                    ],
                    ensure_ascii=False,
                ),
            }
        )
    return grouped.merge(pd.DataFrame(shape_rows), on=["tag", "op_name"], how="left")


def build_case_index(joined: pd.DataFrame) -> pd.DataFrame:
    rows = []
    observed = joined[joined["mapping"].eq("observed_first_batch")].drop_duplicates("tag").copy()
    for idx, (_, row) in enumerate(observed.iterrows(), start=1):
        label, full_label = short_case_label(row, idx)
        rows.append(
            {
                "tag": row["tag"],
                "case_id": f"C{idx:02d}",
                "case_label": label,
                "case_full_label": full_label,
                "alignment_class": row["alignment_class"],
                "real_stage_values": row.get("real_stage_values"),
                "real_attention_token_count_first_layer": row.get("real_attention_token_count_first_layer"),
                "aic_batch_size": row.get("aic_batch_size"),
                "aic_isl": row.get("aic_isl"),
                "aic_prefix": row.get("aic_prefix"),
                "aic_ctx_tokens": row.get("aic_ctx_tokens"),
                "first_formal_prefill_req_count": row.get("first_formal_prefill_req_count"),
                "first_formal_decode_req_count": row.get("first_formal_decode_req_count"),
                "first_formal_sum_prefill_fresh_tokens": row.get("first_formal_sum_prefill_fresh_tokens"),
                "first_formal_sum_decode_kv_tokens": row.get("first_formal_sum_decode_kv_tokens"),
            }
        )
    return pd.DataFrame(rows)


def build_operator_compare(
    joined: pd.DataFrame,
    real_component: pd.DataFrame,
    real_kernel_summary: pd.DataFrame,
    aic_component: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    case_index = build_case_index(joined)
    rows = []
    for _, case in case_index.iterrows():
        tag = case["tag"]
        for spec in OPERATOR_SPECS:
            real_parts = real_component[
                real_component["tag"].eq(tag) & real_component["real_component"].isin(spec["real_components"])
            ]
            aic_parts = aic_component[aic_component["tag"].eq(tag) & aic_component["op_name"].isin(spec["aic_ops"])]
            real_ms = float(real_parts["real_gpu_makespan_ms"].sum()) if not real_parts.empty else 0.0
            real_kernel_ms = float(real_parts["real_gpu_kernel_time_sum_ms"].sum()) if not real_parts.empty else 0.0
            aic_ms = float(aic_parts["aic_per_layer_latency_ms"].sum()) if not aic_parts.empty else 0.0
            real_components = sorted(real_parts["real_component"].dropna().unique().tolist())
            aic_ops = sorted(aic_parts["op_name"].dropna().unique().tolist())
            kernel_rows = real_kernel_summary[
                real_kernel_summary["tag"].eq(tag)
                & real_kernel_summary["real_component"].isin(spec["real_components"])
            ]
            top_kernels = (
                kernel_rows.sort_values("avg_kernel_gpu_time_ms", ascending=False)
                .head(5)[["real_component", "kernel_short_name", "avg_kernel_gpu_time_ms"]]
                .to_dict("records")
                if not kernel_rows.empty
                else []
            )
            status = "both"
            if real_ms <= 0 and aic_ms <= 0:
                status = "neither"
            elif real_ms <= 0:
                status = "aic_only"
            elif aic_ms <= 0:
                status = "real_only"
            rows.append(
                {
                    **case.to_dict(),
                    "operator_family": spec["operator_family"],
                    "display_name": spec["display_name"],
                    "real_components_json": json.dumps(real_components, ensure_ascii=False),
                    "aic_ops_json": json.dumps(aic_ops, ensure_ascii=False),
                    "real_gpu_makespan_ms": real_ms,
                    "real_gpu_kernel_time_sum_ms": real_kernel_ms,
                    "aic_per_layer_latency_ms": aic_ms,
                    "diff_real_minus_aic_ms": real_ms - aic_ms if status == "both" else np.nan,
                    "ratio_real_to_aic": real_ms / aic_ms if aic_ms > 0 and real_ms > 0 else np.nan,
                    "coverage_status": status,
                    "boundary_note": spec["boundary_note"],
                    "aic_query_path_json": json.dumps(
                        sorted(aic_parts["aic_query_path"].dropna().unique().tolist())
                        if "aic_query_path" in aic_parts and not aic_parts.empty
                        else [],
                        ensure_ascii=False,
                    ),
                    "aic_query_shapes_json": json.dumps(
                        aic_parts["aic_query_shapes_json"].dropna().tolist()
                        if "aic_query_shapes_json" in aic_parts and not aic_parts.empty
                        else [],
                        ensure_ascii=False,
                    ),
                    "top_real_kernels_json": json.dumps(top_kernels, ensure_ascii=False),
                }
            )
    compare = pd.DataFrame(rows)
    return compare, case_index


def plot_operator_facets(compare: pd.DataFrame, out_dir: Path) -> None:
    df = compare[~compare["operator_family"].eq("mla_total")].copy()
    operators = [spec["operator_family"] for spec in OPERATOR_SPECS if spec["operator_family"] != "mla_total"]
    ncols = 3
    nrows = math.ceil(len(operators) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(22, 4.2 * nrows), sharex=False)
    axes_flat = np.array(axes).reshape(-1)
    for ax, operator in zip(axes_flat, operators):
        sub = df[df["operator_family"].eq(operator)].copy()
        x = np.arange(len(sub))
        width = 0.38
        ax.bar(
            x - width / 2,
            sub["real_gpu_makespan_ms"],
            width=width,
            label="Real GPU makespan" if operator == operators[0] else None,
            color=COLORS["real"],
            edgecolor=COLORS["edge"],
            linewidth=0.5,
        )
        ax.bar(
            x + width / 2,
            sub["aic_per_layer_latency_ms"],
            width=width,
            label="AIC diagnostic op" if operator == operators[0] else None,
            color=COLORS["aic"],
            edgecolor=COLORS["edge"],
            linewidth=0.5,
        )
        ax.set_title(sub["display_name"].iloc[0], fontsize=11, fontweight="semibold", color=TOKENS["ink"])
        ax.set_xticks(x, sub["case_id"], rotation=0, fontsize=8)
        ax.set_ylabel("ms / layer")
        ax.grid(True, axis="y")
        ax.grid(False, axis="x")
    for ax in axes_flat[len(operators) :]:
        ax.axis("off")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.07, 0.955), frameon=False, ncol=2)
    fig.text(
        0.07,
        0.995,
        "MLA operator-level Real vs AIC breakdown by case",
        ha="left",
        va="top",
        fontsize=15,
        fontweight="semibold",
        color=TOKENS["ink"],
    )
    fig.text(
        0.07,
        0.972,
        "Each subplot is one MLA internal operator family. Real uses non-first-layer Nsight GPU makespan; AIC uses current-SDK diagnostic granular queries with run_agg mixed-step shapes.",
        ha="left",
        va="top",
        fontsize=9.5,
        color=TOKENS["muted"],
    )
    fig.tight_layout(rect=(0.04, 0.03, 0.995, 0.93))
    save_chart(fig, out_dir, "mla_operator_facets_real_vs_aic")


def plot_ratio_heatmap(compare: pd.DataFrame, out_dir: Path) -> None:
    df = compare[~compare["operator_family"].eq("mla_total")].copy()
    pivot = df.pivot_table(index="display_name", columns="case_id", values="ratio_real_to_aic", aggfunc="first")
    operators = [spec["display_name"] for spec in OPERATOR_SPECS if spec["operator_family"] != "mla_total"]
    pivot = pivot.reindex([op for op in operators if op in pivot.index])
    fig, ax = plt.subplots(figsize=(13.5, 7.2))
    masked = np.ma.masked_invalid(pivot.to_numpy(dtype=float))
    im = ax.imshow(masked, aspect="auto", cmap="YlOrBr", vmin=0, vmax=np.nanpercentile(pivot.to_numpy(dtype=float), 90))
    ax.set_xticks(np.arange(pivot.shape[1]), pivot.columns, fontsize=8)
    ax.set_yticks(np.arange(pivot.shape[0]), pivot.index, fontsize=9)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j]
            text = "NA" if pd.isna(val) else f"{val:.1f}x"
            ax.text(j, i, text, ha="center", va="center", fontsize=7, color=TOKENS["ink"])
    cbar = fig.colorbar(im, ax=ax, shrink=0.78)
    cbar.set_label("Real / AIC")
    ax.set_xlabel("Case")
    ax.set_ylabel("Operator family")
    add_chart_header(
        fig,
        ax,
        "Real-to-AIC ratio highlights which MLA operators drive the gap",
        "NA means one side has no directly comparable operator boundary. Ratios use single-layer real GPU makespan divided by AIC diagnostic per-layer latency.",
    )
    save_chart(fig, out_dir, "mla_operator_ratio_heatmap")


def plot_per_case(compare: pd.DataFrame, out_dir: Path) -> None:
    case_dir = out_dir / "per_case_operator_charts"
    case_dir.mkdir(parents=True, exist_ok=True)
    for tag, sub in compare[~compare["operator_family"].eq("mla_total")].groupby("tag", sort=False):
        sub = sub.copy()
        x = np.arange(len(sub))
        width = 0.38
        fig, ax = plt.subplots(figsize=(13.5, 6.5))
        ax.bar(
            x - width / 2,
            sub["real_gpu_makespan_ms"],
            width=width,
            label="Real GPU makespan",
            color=COLORS["real"],
            edgecolor=COLORS["edge"],
            linewidth=0.6,
        )
        ax.bar(
            x + width / 2,
            sub["aic_per_layer_latency_ms"],
            width=width,
            label="AIC diagnostic op",
            color=COLORS["aic"],
            edgecolor=COLORS["edge"],
            linewidth=0.6,
        )
        ax.set_xticks(x, sub["display_name"], rotation=25, ha="right", fontsize=8)
        ax.set_ylabel("ms / layer")
        ax.set_xlabel("MLA operator family")
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), frameon=False, ncol=2, borderaxespad=0)
        label = sub["case_full_label"].iloc[0]
        stage = sub["real_stage_values"].iloc[0]
        add_chart_header(
            fig,
            ax,
            f"{sub['case_id'].iloc[0]} MLA operator comparison",
            f"{tag} | {label} | real stage={stage}. Single-layer comparison; missing bars indicate no directly comparable boundary.",
            width=130,
        )
        safe_tag = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in tag)
        save_chart(fig, case_dir, safe_tag)


def write_report(out_dir: Path, compare: pd.DataFrame, case_index: pd.DataFrame) -> None:
    rollup = compare[compare["operator_family"].eq("mla_total")].copy()
    by_operator = (
        compare[~compare["operator_family"].eq("mla_total")]
        .groupby(["operator_family", "display_name", "coverage_status"], dropna=False)
        .agg(
            cases=("tag", "nunique"),
            avg_real_ms=("real_gpu_makespan_ms", "mean"),
            avg_aic_ms=("aic_per_layer_latency_ms", "mean"),
            avg_ratio=("ratio_real_to_aic", "mean"),
            max_ratio=("ratio_real_to_aic", "max"),
        )
        .reset_index()
        .sort_values(["display_name", "coverage_status"])
    )
    by_operator.to_csv(out_dir / "mla_operator_compare_by_operator_summary.csv", index=False)
    rollup.to_csv(out_dir / "mla_operator_compare_total_rollup.csv", index=False)

    top_gap = (
        compare[compare["coverage_status"].eq("both") & ~compare["operator_family"].eq("mla_total")]
        .sort_values("diff_real_minus_aic_ms", ascending=False)
        .head(10)
    )
    top_gap_lines = "\n".join(
        f"- `{r.case_id}` `{r.display_name}`: real {r.real_gpu_makespan_ms:.3f} ms vs AIC {r.aic_per_layer_latency_ms:.3f} ms, ratio {r.ratio_real_to_aic:.2f}x"
        for r in top_gap.itertuples(index=False)
    )
    op_notes = "\n".join(
        f"- `{spec['display_name']}`: {spec['boundary_note']}"
        for spec in OPERATOR_SPECS
        if spec["operator_family"] != "mla_total"
    )
    report = f"""# MLA 算子级 Real vs AIC run_agg Breakdown

## 口径

- 实机侧：读取 `MLA时延拆解.csv` 的 `child_timing_json`，使用非首层平均 `gpu_makespan_ms` 作为主时延；同时在 CSV 中保留 `gpu_kernel_time_sum_ms` 和 top kernel。
- AIC 侧：保留既有 `run_agg` 快照中的 `context_mla_block` block 值，同时基于当前 SDK 的 `PrefixConditionalOp(context_mla_block)` 子结构，用同一批 `observed_first_batch` 形状诊断性重放细粒度 prefix-path 子 op 查询。
- 单层比较：AIC 子 op latency 除以 `num_layers=6`；实机取非首层均值，避免首层冷启动。
- 当前图只比较 mixed step 中的 context MLA 相关算子；不混入 MoE/FFN/embedding/logits，也不混入 genonly step。
- 注意：若 `run_agg` 快照实际只暴露 `context_mla_block`，则 q_b/kv_b/attention/o_proj 等 AIC 细粒度柱是 `diagnostic_forced_prefix_ops*` 口径，用于定位误差来源，不等价于原始快照直接导出的 per-op 字段。

## 输出

- `mla_operator_compare_long.csv`：每个 case x operator_family 的 Real/AIC 对比长表。
- `aic_mla_operator_ops_long.csv`：AIC 侧 block 快照和细粒度诊断查询长表。
- `aic_mla_operator_context_ops.csv`：AIC 侧用于画图的 context MLA 细粒度聚合表。
- `mla_operator_compare_by_operator_summary.csv`：按算子聚合的误差摘要。
- `mla_operator_compare_total_rollup.csv`：MLA total roll-up。
- `case_label_map.csv`：图中 `C01...C12` 与原始 case 的映射。
- `mla_operator_facets_real_vs_aic.png/svg`：每个算子一个子图的 Real vs AIC 对比。
- `mla_operator_ratio_heatmap.png/svg`：Real/AIC ratio 热力图。
- `per_case_operator_charts/`：每个用例一张算子级对比图。

## 最大正向误差项

{top_gap_lines}

## 算子边界说明

{op_notes}

## 源码与 collector 证据

- `src/aiconfigurator/sdk/models/deepseek.py` 中普通 DeepSeek context 路径将 `context_downscale_gemm` 放在 `context_mla_block` 之前，注释说明 qkv/downscale 不在 SGLang MLA module collector boundary 内，而是 shared op 单独计一次。
- `src/aiconfigurator/sdk/operations.py` 的 `PrefixConditionalOp` 按 `prefix` 选择路径：`prefix == 0` 走 `_no_prefix_ops`，即 `context_mla_module`；`prefix > 0` 走 `_prefix_ops`，即 `context_q_b_proj_gemm/context_kv_b_proj_gemm/context_mla_concat_k/context_attention/context_proj_gemm`。
- 本批 `real_aic_joined.csv` 的 `aic_prefix` 均为 0，因此既有 `run_agg` 快照主要暴露 `context_mla_block` block 值；本报告的细粒度 AIC 子算子柱是按当前 SDK prefix-path 强制拆解出的 diagnostic attribution，用来定位误差来源，而不是声称原始快照直接导出了这些字段。
- `src/aiconfigurator/sdk/operations.py` 的 `ContextMLA.query()` 仅调用 `database.query_context_mla(...)`，源码注释写明 “Context MLA operation. now only contains MHA part.”；因此图中的 AIC `attention` 对应 context MHA/FA 类 attention 表，而不是整个 MLA 模块。
- `src/aiconfigurator/sdk/operations.py` 的 `MLAModule.query()` 调用 `query_context_mla_module/query_generation_mla_module`，其模块边界注释为 context 替代 `q_b_proj + kv_b_proj + ContextMLA + proj`，generation 替代 `MLABmm(pre) + GenerationMLA + MLABmm(post)`。
- `collector/sglang/collect_gemm.py` 的 fp8 DeepGEMM collector 直接调用 `gemm_nt_f8f8bf16((x_fp8, x_scale), (y_fp8, y_scale), out)`，输入和 scale 已预先构造。因此该表主要覆盖 DeepGEMM kernel 本身，不覆盖实机 child range 中可能出现的运行时 activation quantization kernel。
- `collector/sglang/collect_mla.py` 构造 `RadixAttention`/`ForwardBatch` 并校验 prefill 走 DeepSeek MHA 分支；它覆盖的是 attention/RadixAttention 这一类 kernel 边界，不包含 q_a/kv_a layernorm、RoPE child range 或 qkv/downscale GEMM。

## 重要解释

fp8 GEMM 的 AIC `GEMM` 查询来自 `collector/sglang/collect_gemm.py` 的 `fp8_block` DeepGEMM microbench；该 collector 预先构造 fp8 输入和 scale，并调用 `gemm_nt_f8f8bf16`，通常不包含实机 trace 中 child range 前置的 `per_token_group_quant_8bit_kernel`。因此 qkv/q_b/o_proj 一类 GEMM 对比中，实机侧常会比 AIC 侧多出激活动态量化 kernel 开销。

`q_a_layernorm`、`kv_a_layernorm`、`rotary_emb` 在实机中是显式 child range，但普通 DeepSeekModel 的 MLA path 未单独建模；这些算子若在图中只出现 Real bar，代表 AIC 当前口径没有独立对应项，而不是实机异常。

`attention` 的实机子模块可能是 `attn_mha` 或 `attn_mqa`。这与 SGLang mixed/prefix 调度相关；AIC 诊断拆解里的 `context_attention` 固定查询 `ContextMLA/query_context_mla` 并按 `run_agg` 第二段 attention pass 做 scale 修正，因此小 batch/prefix 命中 MQA 的 case 会天然存在 backend 口径差异。
"""
    (out_dir / "MLA算子级实机-AIC对比报告.md").write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--joined", type=Path, default=DEFAULT_JOINED)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--model-path", default="deepseek-ai/DeepSeek-V3")
    parser.add_argument("--system", default="h100_sxm")
    parser.add_argument("--backend", default="sglang")
    parser.add_argument("--version", default="0.5.9")
    parser.add_argument("--database-mode", default="SILICON")
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--pp-size", type=int, default=1)
    parser.add_argument("--attention-dp-size", type=int, default=1)
    parser.add_argument("--moe-tp-size", type=int, default=1)
    parser.add_argument("--moe-ep-size", type=int, default=1)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--gemm-quant-mode", default="fp8_block")
    parser.add_argument("--moe-quant-mode", default="fp8_block")
    parser.add_argument("--kvcache-quant-mode", default="fp8")
    parser.add_argument("--fmha-quant-mode", default="bfloat16")
    parser.add_argument("--comm-quant-mode", default="half")
    parser.add_argument("--attention-backend", default="fa3")
    parser.add_argument("--moe-backend", default=None)
    parser.add_argument("--enable-wideep", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    use_chart_theme()

    joined = pd.read_csv(args.joined)
    observed = joined[joined["mapping"].eq("observed_first_batch")].copy()
    session = build_session(build_session_args(args))

    real_child, real_kernel = expand_real_mla(observed)
    real_component = summarize_real_by_component(real_child)
    real_kernel_summary = summarize_real_kernels(real_kernel)
    aic_long = expand_aic_mla_operator_ops(observed, session, args.num_layers)
    aic_component = summarize_aic_context_ops(aic_long)

    compare, case_index = build_operator_compare(observed, real_component, real_kernel_summary, aic_component)

    real_child.to_csv(args.out_dir / "real_mla_child_long.csv", index=False)
    real_kernel.to_csv(args.out_dir / "real_mla_kernel_long.csv", index=False)
    real_component.to_csv(args.out_dir / "real_mla_child_operator_summary.csv", index=False)
    real_kernel_summary.to_csv(args.out_dir / "real_mla_kernel_operator_summary.csv", index=False)
    aic_long.to_csv(args.out_dir / "aic_mla_operator_ops_long.csv", index=False)
    aic_component.to_csv(args.out_dir / "aic_mla_operator_context_ops.csv", index=False)
    compare.to_csv(args.out_dir / "mla_operator_compare_long.csv", index=False)
    case_index.to_csv(args.out_dir / "case_label_map.csv", index=False)

    plot_operator_facets(compare, args.out_dir)
    plot_ratio_heatmap(compare, args.out_dir)
    plot_per_case(compare, args.out_dir)
    write_report(args.out_dir, compare, case_index)

    print(f"Wrote MLA operator comparison to {args.out_dir}")
    print(f"cases={case_index.shape[0]} compare_rows={compare.shape[0]} real_child_rows={real_child.shape[0]} aic_rows={aic_long.shape[0]}")


if __name__ == "__main__":
    main()
