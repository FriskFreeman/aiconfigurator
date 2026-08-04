#!/usr/bin/env python3
"""Run disaggregated Pareto acceptance cases with resumable artifacts."""

from __future__ import annotations

import argparse
import json
import logging
import time
import traceback
from pathlib import Path

import pandas as pd

from aiconfigurator.sdk.task import TaskConfig, TaskRunner


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
LOGS = ROOT / "logs"

MODELS = {
    "llama31_70b": "meta-llama/Meta-Llama-3.1-70B",
    "qwen3_32b": "Qwen/Qwen3-32B",
    "deepseek_v3": "deepseek-ai/DeepSeek-V3",
}
SYSTEMS = ["a100_sxm", "h100_sxm", "h200_sxm", "b200_sxm", "b300_sxm", "rtx_pro_6000_server"]
MODES = ["SILICON", "EMPIRICAL", "ANALYTICAL"]


def quant_patch(model_key: str, system: str) -> dict:
    if model_key != "deepseek_v3":
        return {}
    worker_quant = {
        "kvcache_quant_mode": "bfloat16",
        "fmha_quant_mode": "bfloat16",
        "gemm_quant_mode": "bfloat16" if system == "a100_sxm" else "fp8",
        "moe_quant_mode": "bfloat16" if system == "a100_sxm" else "fp8_block",
    }
    return {
        "prefill_worker_config": dict(worker_quant),
        "decode_worker_config": dict(worker_quant),
    }


def make_task(model_key: str, system: str, mode: str, analytical_level: str = "standard") -> TaskConfig:
    total_gpus = 64 if model_key == "deepseek_v3" and system == "a100_sxm" else 32
    enable_wideep = model_key == "deepseek_v3" and system in {"a100_sxm", "h100_sxm"}
    # H100 has ordinary MLA/MoE silicon tables but lacks the WideEP DeepEP module table.
    # Keep the calibrated modes unchanged and use the ordinary path for this silicon baseline.
    if system == "h100_sxm" and mode == "SILICON":
        enable_wideep = False
    config_patch = quant_patch(model_key, system)
    if model_key == "deepseek_v3" and system == "h100_sxm" and mode == "SILICON":
        # The ordinary model needs >80 GB/GPU at PP1. PP2/PP4 keeps the model
        # on physical H100 capacity while continuing to use ordinary MoE tables.
        ordinary_h100_parallel = {
            "num_gpu_per_worker": [16, 32],
            "tp_list": [8],
            "pp_list": [2, 4],
            "dp_list": [1],
            "moe_tp_list": [1, 2, 4, 8],
            "moe_ep_list": [1, 2, 4, 8],
        }
        for worker_key in ("prefill_worker_config", "decode_worker_config"):
            config_patch[worker_key].update(ordinary_h100_parallel)
    return TaskConfig(
        serving_mode="disagg",
        model_path=MODELS[model_key],
        system_name=system,
        decode_system_name=system,
        backend_name="sglang",
        backend_version="0.5.10",
        database_mode=mode,
        analytical_level=analytical_level,
        analytical_fp8_gemm_recipe="sglang",
        analytical_attention_algorithm="fa2",
        analytical_communication_mode="empirical",
        total_gpus=total_gpus,
        isl=4096,
        osl=1024,
        ttft=5000.0,
        tpot=100.0,
        free_gpu_memory_fraction=0.9,
        enable_wideep=enable_wideep,
        yaml_config={"mode": "patch", "config": config_patch},
    )


def run_case(
    model_key: str,
    system: str,
    mode: str,
    force: bool,
    analytical_level: str = "standard",
    label_analytical_level: bool = False,
) -> dict:
    level_suffix = f"__{analytical_level}" if mode == "ANALYTICAL" and label_analytical_level else ""
    case_id = f"{model_key}__{system}__{mode.lower()}{level_suffix}"
    case_dir = RESULTS / case_id
    status_path = case_dir / "status.json"
    if status_path.exists() and not force:
        return json.loads(status_path.read_text(encoding="utf-8"))

    case_dir.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    started = time.time()
    status = {
        "case_id": case_id,
        "model_key": model_key,
        "model_path": MODELS[model_key],
        "system": system,
        "mode": mode,
        "analytical_level": analytical_level if mode == "ANALYTICAL" else "",
        "backend": "sglang",
        "backend_version": "0.5.10",
        "serving_mode": "disagg",
        "isl": 4096,
        "osl": 1024,
        "total_gpus": 64 if model_key == "deepseek_v3" and system == "a100_sxm" else 32,
        "status": "running",
    }
    try:
        task = make_task(model_key, system, mode, analytical_level=analytical_level)
        (case_dir / "task.yaml").write_text(task.to_yaml(), encoding="utf-8")
        result = TaskRunner().run(task)
        if result is None or result.get("pareto_df") is None or result["pareto_df"].empty:
            raise RuntimeError("TaskRunner returned no Pareto points")
        for key, value in result.items():
            if isinstance(value, pd.DataFrame):
                value.to_csv(case_dir / f"{key}.csv", index=False)
        pareto = result["pareto_df"]
        throughput_column = (
            "tokens/s/gpu_cluster" if "tokens/s/gpu_cluster" in pareto else "tokens/s/gpu"
        )
        request_rate_column = (
            "cluster_request_rate" if "cluster_request_rate" in pareto else "request_rate"
        )
        status.update(
            status="success",
            pareto_points=len(pareto),
            best_tokens_per_s_gpu=float(pareto[throughput_column].max()),
            best_cluster_request_rate=float(pareto[request_rate_column].max()),
        )
    except BaseException as error:
        status.update(
            status="skipped" if mode == "SILICON" else "failed",
            error_type=type(error).__name__,
            error=str(error),
        )
        (case_dir / "traceback.log").write_text(traceback.format_exc(), encoding="utf-8")
    status["elapsed_s"] = time.time() - started
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    return status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--systems", nargs="+", choices=SYSTEMS, default=SYSTEMS)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=MODES)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    RESULTS.mkdir(parents=True, exist_ok=True)

    statuses = []
    for model_key in args.models:
        for system in args.systems:
            for mode in args.modes:
                logging.info("Running %s / %s / %s", model_key, system, mode)
                status = run_case(model_key, system, mode, args.force)
                statuses.append(status)
                logging.info("Completed %s: %s", status["case_id"], status["status"])
                pd.DataFrame(statuses).to_csv(RESULTS / "run_summary.csv", index=False)
    (RESULTS / "run_summary.json").write_text(
        json.dumps(statuses, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
