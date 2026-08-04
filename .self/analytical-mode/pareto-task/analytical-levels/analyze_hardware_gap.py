#!/usr/bin/env python3
"""Explain H100/H200/RTX PRO 6000 Pareto gaps from saved best points."""

from __future__ import annotations

import copy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from aiconfigurator.sdk import config, models
from aiconfigurator.sdk.backends.factory import get_backend
from aiconfigurator.sdk.inference_session import InferenceSession
from aiconfigurator.sdk.task import TaskRunner

from run_pareto import make_task


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures" / "hardware_gap"
SYSTEM_DIR = Path(__file__).resolve().parents[4] / "src" / "aiconfigurator" / "systems"
SYSTEMS = ["h100_sxm", "h200_sxm", "rtx_pro_6000_server"]
LABELS = {"h100_sxm": "H100", "h200_sxm": "H200", "rtx_pro_6000_server": "RTX PRO 6000"}
COLORS = {"h100_sxm": "#3676A8", "h200_sxm": "#55A868", "rtx_pro_6000_server": "#C44E52"}


def load_systems() -> pd.DataFrame:
    rows = []
    for system in SYSTEMS:
        data = yaml.safe_load((SYSTEM_DIR / f"{system}.yaml").read_text(encoding="utf-8"))
        gpu, node = data["gpu"], data["node"]
        rows.append(
            {
                "system": system,
                "memory_gib": gpu["mem_capacity"] / 2**30,
                "memory_bw_tb_s": gpu["mem_bw"] / 1e12,
                "bf16_tflops": gpu["bfloat16_tc_flops"] / 1e12,
                "fp8_tflops": gpu["fp8_tc_flops"] / 1e12,
                "gpus_per_node": node["num_gpus_per_node"],
                "intra_node_gb_s": node["intra_node_bw"] / 1e9,
                "inter_node_gb_s": node["inter_node_bw"] / 1e9,
            }
        )
    return pd.DataFrame(rows)


def load_best() -> pd.DataFrame:
    best = pd.read_csv(RESULTS / "best_frontier_points.csv")
    return best[
        best.model_key.eq("deepseek_v3")
        & best.system.isin(SYSTEMS)
        & best.series.eq("ANALYTICAL-standard")
    ].copy()


def model_config(task, phase: str, row: pd.Series) -> config.ModelConfig:
    wc = getattr(task.config, f"{phase}_worker_config")
    result = config.ModelConfig(
        gemm_quant_mode=wc.gemm_quant_mode,
        kvcache_quant_mode=wc.kvcache_quant_mode,
        fmha_quant_mode=wc.fmha_quant_mode,
        moe_quant_mode=wc.moe_quant_mode,
        comm_quant_mode=wc.comm_quant_mode,
        nextn=task.config.nextn,
        nextn_accept_rates=task.config.nextn_accept_rates,
        moe_backend=getattr(wc, "moe_backend", None) or task.config.moe_backend,
        attention_backend=getattr(wc, "attention_backend", None) or task.config.attention_backend,
        enable_wideep=getattr(wc, "enable_wideep", None)
        if getattr(wc, "enable_wideep", None) is not None
        else task.config.enable_wideep,
        enable_eplb=getattr(wc, "enable_eplb", False),
    )
    result.tp_size = int(row[f"({phase[0]})tp"])
    result.pp_size = int(row[f"({phase[0]})pp"])
    result.attention_dp_size = int(row[f"({phase[0]})dp"])
    result.moe_tp_size = int(row[f"({phase[0]})moe_tp"])
    result.moe_ep_size = int(row[f"({phase[0]})moe_ep"])
    return result


def replay_phase(system: str, row: pd.Series, phase: str) -> tuple[list[dict], dict]:
    task = make_task("deepseek_v3", system, "ANALYTICAL", analytical_level="standard")
    wc = getattr(task.config, f"{phase}_worker_config")
    database = TaskRunner._get_database(
        system=wc.system_name,
        backend=wc.backend_name,
        version=wc.backend_version,
        database_mode="ANALYTICAL",
    )
    TaskRunner._configure_analytical(database, task.config)
    mc = model_config(task, phase, row)
    model = models.get_model(task.config.model_path, mc, wc.backend_name)
    session = InferenceSession(model, database, get_backend(wc.backend_name))
    runtime = config.RuntimeConfig(
        batch_size=int(row[f"({phase[0]})bs"]),
        isl=int(row.isl),
        osl=1 if phase == "prefill" else int(row.osl),
        prefix=int(row.prefix),
        ttft=float(row.ttft),
        tpot=float(row.tpot),
    )
    summary = session.run_static(runtime, "static_ctx" if phase == "prefill" else "static_gen", stride=32)
    values = (
        summary.get_context_latency_dict() if phase == "prefill" else summary.get_generation_latency_dict()
    )
    total = sum(values.values())
    records = [
        {
            "system": system,
            "phase": phase,
            "operation": operation,
            "latency_ms": latency,
            "share_pct": 100.0 * latency / total if total else 0.0,
        }
        for operation, latency in values.items()
    ]
    reference_ms = float(row.ttft if phase == "prefill" else row.tpot * (int(row.osl) - 1))
    metadata = {
        "system": system,
        "phase": phase,
        "replay_latency_ms": total,
        "pareto_latency_ms": reference_ms,
        "replay_ratio": total / reference_ms if reference_ms else np.nan,
        "batch_size": int(runtime.batch_size),
        "tp": mc.tp_size,
        "pp": mc.pp_size,
        "dp": mc.attention_dp_size,
        "moe_tp": mc.moe_tp_size,
        "moe_ep": mc.moe_ep_size,
        "enable_wideep": bool(mc.enable_wideep),
    }
    return records, metadata


def plot_resources(hardware: pd.DataFrame) -> None:
    metrics = ["memory_gib", "memory_bw_tb_s", "bf16_tflops", "intra_node_gb_s"]
    titles = ["Memory capacity (GiB)", "Memory bandwidth (TB/s)", "BF16 Tensor peak (TFLOPS)", "Intra-node bandwidth (GB/s)"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, metric, title in zip(axes.flat, metrics, titles):
        bars = ax.bar([LABELS[s] for s in hardware.system], hardware[metric], color=[COLORS[s] for s in hardware.system])
        ax.bar_label(bars, fmt="%.1f", fontsize=9)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    fig.savefig(FIGURES / "hardware_resources.png", dpi=180)
    plt.close(fig)


def plot_throughput_decomposition(best: pd.DataFrame) -> None:
    data = best.set_index("system").loc[SYSTEMS]
    values = {
        "Request rate (seq/s)": data.request_rate,
        "Replica GPUs": data.num_total_gpus,
        "Throughput (tokens/s/GPU)": data["tokens/s/gpu_cluster"],
    }
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    for ax, (title, series) in zip(axes, values.items()):
        bars = ax.bar([LABELS[s] for s in SYSTEMS], series, color=[COLORS[s] for s in SYSTEMS])
        ax.bar_label(bars, fmt="%.1f", fontsize=9)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    fig.savefig(FIGURES / "throughput_decomposition.png", dpi=180)
    plt.close(fig)


def operation_group(name: str) -> str:
    lowered = name.lower()
    if "moe" in lowered or "expert" in lowered or "dispatch" in lowered:
        return "MoE / dispatch"
    if "mla" in lowered or "bmm" in lowered or "attention" in lowered:
        return "MLA / attention"
    if "gemm" in lowered or "linear" in lowered or "logits" in lowered:
        return "Other GEMM"
    if "norm" in lowered or "add" in lowered:
        return "Norm / elementwise"
    if "comm" in lowered or "p2p" in lowered or "reduce" in lowered or "gather" in lowered:
        return "Communication"
    return "Other"


def plot_breakdown(breakdown: pd.DataFrame) -> pd.DataFrame:
    grouped = breakdown.assign(group=breakdown.operation.map(operation_group)).groupby(
        ["system", "phase", "group"], as_index=False
    ).latency_ms.sum()
    totals = grouped.groupby(["system", "phase"]).latency_ms.transform("sum")
    grouped["share_pct"] = grouped.latency_ms / totals * 100
    order = ["MoE / dispatch", "MLA / attention", "Other GEMM", "Norm / elementwise", "Communication", "Other"]
    colors = ["#C44E52", "#8172B2", "#DD8452", "#55A868", "#4C72B0", "#999999"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for ax, phase in zip(axes, ["prefill", "decode"]):
        pivot = grouped[grouped.phase.eq(phase)].pivot(index="system", columns="group", values="share_pct").fillna(0).reindex(SYSTEMS)
        bottom = np.zeros(len(pivot))
        for group, color in zip(order, colors):
            values = pivot[group].to_numpy() if group in pivot else np.zeros(len(pivot))
            ax.bar([LABELS[s] for s in pivot.index], values, bottom=bottom, label=group, color=color)
            bottom += values
        ax.set_title(f"DeepSeek {phase} latency composition")
        ax.set_ylabel("Latency share (%)")
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.2)
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.savefig(FIGURES / "deepseek_operator_breakdown.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return grouped


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    hardware = load_systems()
    best = load_best()
    hardware.to_csv(RESULTS / "hardware_gap_resources.csv", index=False)
    best.to_csv(RESULTS / "hardware_gap_best_points.csv", index=False)
    plot_resources(hardware)
    plot_throughput_decomposition(best)

    records, metadata = [], []
    for system in SYSTEMS:
        row = best[best.system.eq(system)].iloc[0]
        for phase in ("prefill", "decode"):
            phase_records, phase_metadata = replay_phase(system, row, phase)
            records.extend(phase_records)
            metadata.append(phase_metadata)
    breakdown = pd.DataFrame(records)
    replay = pd.DataFrame(metadata)
    breakdown.to_csv(RESULTS / "hardware_gap_operator_breakdown.csv", index=False)
    replay.to_csv(RESULTS / "hardware_gap_replay_validation.csv", index=False)
    grouped = plot_breakdown(breakdown)
    grouped.to_csv(RESULTS / "hardware_gap_operator_groups.csv", index=False)


if __name__ == "__main__":
    main()
