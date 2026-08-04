#!/usr/bin/env python3
"""Decompose the RTX PRO 6000 DeepSeek EP8 decode MoE bottleneck."""

from __future__ import annotations

import copy
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from aiconfigurator.sdk import models
from aiconfigurator.sdk.task import TaskRunner
from aiconfigurator.sdk.kernelsim.moe.sglang_moe_empirical_final import estimate_sglang_moe

from analyze_hardware_gap import load_best, make_task, model_config


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures" / "hardware_gap"
SYSTEM = "rtx_pro_6000_server"


def build():
    row = load_best().loc[lambda frame: frame.system.eq(SYSTEM)].iloc[0]
    task = make_task("deepseek_v3", SYSTEM, "ANALYTICAL", analytical_level="standard")
    wc = task.config.decode_worker_config
    database = TaskRunner._get_database(
        wc.system_name, wc.backend_name, wc.backend_version, "ANALYTICAL"
    )
    TaskRunner._configure_analytical(database, task.config)
    model = models.get_model(
        task.config.model_path,
        model_config(task, "decode", row),
        wc.backend_name,
    )
    overlap = next(op for op in model.generation_ops if op._name == "generation_moe_overlap")
    return row, database, model, overlap


def counterfactual(
    database,
    *,
    h100_topology: bool = False,
    h100_memory_bw: bool = False,
    h100_compute: bool = False,
):
    result = copy.deepcopy(database)
    if h100_topology:
        result.system_spec["node"].update(
            num_gpus_per_node=8,
            intra_node_bw=450_000_000_000,
            pcie_bw=64_000_000_000,
            inter_node_bw=50_000_000_000,
        )
    if h100_memory_bw:
        result.system_spec["gpu"]["mem_bw"] = 3_350_000_000_000
    if h100_compute:
        result.system_spec["gpu"].update(
            bfloat16_tc_flops=989_000_000_000_000,
            fp8_tc_flops=1_978_000_000_000_000,
            int8_tc_flops=1_978_000_000_000_000,
        )
    # System-spec changes must not reuse cached query results from the baseline.
    for attr in ("query_cache", "_query_cache"):
        cache = getattr(result, attr, None)
        if hasattr(cache, "clear"):
            cache.clear()
    return result


def query_group(database, overlap, batch: int, scenario: str) -> list[dict]:
    records = []
    groups = {"routed": overlap._group_a, "shared": overlap._group_b}
    for group_name, operations in groups.items():
        for operation in operations:
            latency_ms = float(operation.query(database, x=batch))
            records.append(
                {
                    "scenario": scenario,
                    "group": group_name,
                    "operation": operation._name,
                    "latency_ms_per_step": latency_ms,
                }
            )
    return records


def summarize(details: pd.DataFrame) -> pd.DataFrame:
    groups = details.groupby(["scenario", "group"], as_index=False).latency_ms_per_step.sum()
    pivot = groups.pivot(index="scenario", columns="group", values="latency_ms_per_step").fillna(0)
    pivot["overlap_latency_ms_per_step"] = pivot.max(axis=1)
    pivot["critical_group"] = pivot[["routed", "shared"]].idxmax(axis=1)
    baseline = pivot.loc["PRO6000 baseline", "overlap_latency_ms_per_step"]
    pivot["speedup_vs_baseline"] = baseline / pivot["overlap_latency_ms_per_step"]
    return pivot.reset_index()


def plot_details(details: pd.DataFrame) -> None:
    baseline = details[details.scenario.eq("PRO6000 baseline")].copy()
    baseline["label"] = baseline.operation.str.replace("generation_", "", regex=False)
    baseline = baseline.sort_values("latency_ms_per_step", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    bars = ax.barh(
        baseline.label,
        baseline.latency_ms_per_step,
        color=baseline.group.map({"routed": "#C44E52", "shared": "#4C72B0"}),
    )
    ax.bar_label(bars, fmt="%.3f ms", padding=3, fontsize=9)
    ax.set_xlabel("Latency per decode step (ms, includes all MoE layers)")
    ax.set_title("RTX PRO 6000 DeepSeek EP8 MoE sub-operation latency")
    ax.grid(axis="x", alpha=0.25)
    fig.savefig(FIGURES / "pro6000_moe_ep8_suboperations.png", dpi=180)
    plt.close(fig)


def plot_counterfactual(summary: pd.DataFrame) -> None:
    order = [
        "PRO6000 baseline",
        "H100 compute only",
        "H100 topology only",
        "H100 memory BW only",
        "H100 topology + resources",
    ]
    summary = summary.set_index("scenario").loc[order].reset_index()
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    bars = ax.bar(
        summary.scenario,
        summary.overlap_latency_ms_per_step,
        color=["#C44E52", "#937860", "#DD8452", "#55A868", "#4C72B0"],
    )
    ax.bar_label(bars, fmt="%.2f ms", padding=3, fontsize=9)
    ax.set_ylabel("MoE overlap latency per decode step (ms)")
    ax.set_title("Topology and GPU-resource counterfactuals")
    ax.tick_params(axis="x", rotation=12)
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(FIGURES / "pro6000_moe_ep8_counterfactuals.png", dpi=180)
    plt.close(fig)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    row, database, model, overlap = build()
    # DeepSeek MTP evaluates the accepted-token path together with one draft token.
    batch = int(row["(d)bs"]) * (int(getattr(model, "_nextn", 0)) + 1)
    scenarios = {
        "PRO6000 baseline": database,
        "H100 topology only": counterfactual(database, h100_topology=True),
        "H100 memory BW only": counterfactual(database, h100_memory_bw=True),
        "H100 compute only": counterfactual(database, h100_compute=True),
        "H100 topology + resources": counterfactual(
            database, h100_topology=True, h100_memory_bw=True, h100_compute=True
        ),
    }
    records = []
    for scenario, scenario_database in scenarios.items():
        records.extend(query_group(scenario_database, overlap, batch, scenario))
    details = pd.DataFrame(records)
    summary = summarize(details)
    details.to_csv(RESULTS / "pro6000_moe_ep8_suboperations.csv", index=False)
    summary.to_csv(RESULTS / "pro6000_moe_ep8_counterfactuals.csv", index=False)
    moe_op = next(op for op in overlap._group_a if op._name == "generation_moe")
    breakdown = estimate_sglang_moe(
        recipe="fp8_block_triton",
        num_tokens=batch * moe_op._attention_dp_size,
        hidden_size=moe_op._hidden_size,
        inter_size=moe_op._inter_size,
        topk=moe_op._topk,
        num_experts=moe_op._num_experts,
        moe_tp_size=moe_op._moe_tp_size,
        moe_ep_size=moe_op._moe_ep_size,
        peak_flops_s=database.system_spec["gpu"]["fp8_tc_flops"],
        mem_bandwidth_bytes_s=database.system_spec["gpu"]["mem_bw"],
        parameter_level="standard",
    )
    pd.DataFrame([asdict(breakdown)]).to_csv(
        RESULTS / "pro6000_moe_ep8_analytical_terms.csv", index=False
    )
    plot_details(details)
    plot_counterfactual(summary)


if __name__ == "__main__":
    main()
