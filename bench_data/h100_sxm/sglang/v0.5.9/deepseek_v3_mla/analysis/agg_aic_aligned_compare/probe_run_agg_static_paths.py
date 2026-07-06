#!/usr/bin/env python3
"""Probe which model/op paths SGLangBackend.run_agg uses behind run_static."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[6]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from aiconfigurator.sdk import common, config, models, perf_database  # noqa: E402
from aiconfigurator.sdk.backends.factory import get_backend  # noqa: E402


DEFAULT_OUT_DIR = SCRIPT_DIR / "run_agg_static_path_probe"


def jsonify(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonify(v) for v in value]
    if hasattr(value, "name"):
        return value.name
    return repr(value)


def op_descriptor(op: Any) -> dict[str, Any]:
    item = {
        "name": getattr(op, "_name", None),
        "class": type(op).__name__,
        "scale_factor": getattr(op, "_scale_factor", None),
    }
    if hasattr(op, "_primary"):
        item["primary"] = {
            "name": getattr(op._primary, "_name", None),
            "class": type(op._primary).__name__,
        }
        item["fallback"] = [
            {
                "name": getattr(child, "_name", None),
                "class": type(child).__name__,
            }
            for child in getattr(op, "_fallback", [])
        ]
        item["primary_unavailable"] = getattr(op, "_primary_unavailable", None)
    for attr in ("_tp_size", "_num_heads", "_kvcache_quant_mode", "_fmha_quant_mode", "_attn_backend"):
        if hasattr(op, attr):
            item[attr.removeprefix("_")] = jsonify(getattr(op, attr))
    return item


def list_ops(model: Any, phase: str) -> list[dict[str, Any]]:
    ops = model.context_ops if phase == "context" else model.generation_ops
    return [op_descriptor(op) for op in ops]


def find_fallback_state(model: Any) -> dict[str, Any]:
    states = {}
    for phase, ops in (("context", model.context_ops), ("generation", model.generation_ops)):
        for op in ops:
            if hasattr(op, "_primary"):
                states[f"{phase}.{op._name}"] = {
                    "primary": f"{type(op._primary).__name__}:{op._primary._name}",
                    "fallback": [f"{type(child).__name__}:{child._name}" for child in op._fallback],
                    "primary_unavailable": getattr(op, "_primary_unavailable", None),
                }
    return states


def make_model_config(case: dict[str, Any]) -> config.ModelConfig:
    return config.ModelConfig(
        tp_size=case["tp_size"],
        pp_size=1,
        attention_dp_size=case["attention_dp_size"],
        moe_tp_size=case["moe_tp_size"],
        moe_ep_size=case["moe_ep_size"],
        gemm_quant_mode=common.GEMMQuantMode[case.get("gemm_quant_mode", "fp8_block")],
        moe_quant_mode=common.MoEQuantMode[case.get("moe_quant_mode", "fp8_block")],
        kvcache_quant_mode=common.KVCacheQuantMode[case.get("kvcache_quant_mode", "fp8")],
        fmha_quant_mode=common.FMHAQuantMode[case.get("fmha_quant_mode", "bfloat16")],
        comm_quant_mode=common.CommQuantMode[case.get("comm_quant_mode", "half")],
        overwrite_num_layers=case["num_layers"],
        moe_backend=case.get("moe_backend"),
        attention_backend=case.get("attention_backend", "fa3"),
        enable_wideep=case.get("enable_wideep", False),
        enable_eplb=case.get("enable_eplb", False),
    )


def run_case(case: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"case": case}
    try:
        model_config = make_model_config(case)
        model = models.get_model(case["model_path"], model_config, case["backend"])
        database = perf_database.get_database(
            case["system"],
            case["backend"],
            case["version"],
            database_mode=case["database_mode"],
        )
        if database is None:
            raise RuntimeError("database load returned None")
        database.set_default_database_mode(common.DatabaseMode[case["database_mode"]])
        backend = get_backend(case["backend"])
        runtime = config.RuntimeConfig(
            batch_size=case["batch_size"],
            isl=case["isl"],
            osl=case["osl"],
            prefix=case["prefix"],
            engine_step_backend=case.get("engine_step_backend"),
        )

        out["model_class"] = type(model).__name__
        out["context_ops_before"] = list_ops(model, "context")
        out["generation_ops_before"] = list_ops(model, "generation")
        out["fallback_state_before"] = find_fallback_state(model)

        summary = backend.run_agg(model, database, runtime, ctx_tokens=case["ctx_tokens"])
        per_ops = summary.get_per_ops_data() or {}
        per_ops_source = summary.get_per_ops_source() or {}
        result = summary.get_result_dict() or {}
        out["status"] = "ok"
        out["run_agg_result"] = jsonify(result)
        out["per_ops_data"] = jsonify(per_ops)
        out["per_ops_source"] = jsonify(per_ops_source)
        out["fallback_state_after_run_agg"] = find_fallback_state(model)
        out["context_ops_after"] = list_ops(model, "context")
        out["generation_ops_after"] = list_ops(model, "generation")

        # Reproduce the three run_static calls that run_agg uses and capture the
        # raw summary dictionaries. This makes the hidden run_static path explicit.
        b = case["batch_size"]
        isl = case["isl"]
        osl = case["osl"]
        prefix = case["prefix"]
        ctx_tokens = case["ctx_tokens"]
        import numpy as np

        steps_to_finish_ctx = int(np.ceil(isl * b / ctx_tokens))
        if b > 1 and steps_to_finish_ctx >= osl:
            mix_ctx_tokens = ctx_tokens
            mix_gen_tokens = max(1, int(b // (steps_to_finish_ctx / osl)))
            genonly_tokens = 0
        elif b > 1:
            mix_ctx_tokens = ctx_tokens
            mix_gen_tokens = int(b - np.ceil(ctx_tokens / isl))
            genonly_tokens = b
        else:
            mix_ctx_tokens = ctx_tokens
            mix_gen_tokens = 0
            genonly_tokens = 1

        out["derived_run_agg_shape"] = {
            "steps_to_finish_ctx": steps_to_finish_ctx,
            "mix_ctx_tokens": int(mix_ctx_tokens),
            "mix_gen_tokens": int(mix_gen_tokens),
            "genonly_tokens": int(genonly_tokens),
        }

        static_summaries: dict[str, Any] = {}
        static_1 = backend.run_static(
            model,
            database,
            config.RuntimeConfig(
                batch_size=1,
                isl=int(mix_ctx_tokens + mix_gen_tokens),
                osl=1,
                prefix=int(prefix * np.floor(ctx_tokens / isl)),
            ),
            mode="static_ctx",
        )
        static_summaries["mix_non_attention_static_ctx"] = {
            "runtime": {
                "batch_size": 1,
                "isl": int(mix_ctx_tokens + mix_gen_tokens),
                "osl": 1,
                "prefix": int(prefix * np.floor(ctx_tokens / isl)),
            },
            "context_latency_dict": jsonify(static_1.get_context_latency_dict()),
            "context_source_dict": jsonify(static_1.get_context_source_dict()),
        }
        static_2 = backend.run_static(
            model,
            database,
            config.RuntimeConfig(
                batch_size=int(np.ceil(ctx_tokens / isl)),
                isl=isl,
                osl=1,
                prefix=prefix,
            ),
            mode="static_ctx",
        )
        static_summaries["context_attention_static_ctx"] = {
            "runtime": {
                "batch_size": int(np.ceil(ctx_tokens / isl)),
                "isl": isl,
                "osl": 1,
                "prefix": prefix,
            },
            "context_latency_dict": jsonify(static_2.get_context_latency_dict()),
            "context_source_dict": jsonify(static_2.get_context_source_dict()),
        }
        if mix_gen_tokens > 0:
            static_3 = backend.run_static(
                model,
                database,
                config.RuntimeConfig(
                    batch_size=int(mix_gen_tokens),
                    isl=int(isl + osl // 2),
                    osl=2,
                ),
                mode="static_gen",
            )
            static_summaries["mix_generation_attention_static_gen"] = {
                "runtime": {
                    "batch_size": int(mix_gen_tokens),
                    "isl": int(isl + osl // 2),
                    "osl": 2,
                },
                "generation_latency_dict": jsonify(static_3.get_generation_latency_dict()),
                "generation_source_dict": jsonify(static_3.get_generation_source_dict()),
            }
        if genonly_tokens > 0:
            static_4 = backend.run_static(
                model,
                database,
                config.RuntimeConfig(
                    batch_size=int(genonly_tokens),
                    isl=int(isl + osl // 2),
                    osl=2,
                ),
                mode="static_gen",
            )
            static_summaries["genonly_static_gen"] = {
                "runtime": {
                    "batch_size": int(genonly_tokens),
                    "isl": int(isl + osl // 2),
                    "osl": 2,
                },
                "generation_latency_dict": jsonify(static_4.get_generation_latency_dict()),
                "generation_source_dict": jsonify(static_4.get_generation_source_dict()),
            }
        out["reproduced_run_static_summaries"] = static_summaries
        out["fallback_state_after_reproduced_static"] = find_fallback_state(model)
    except Exception as exc:
        out["status"] = "error"
        out["error"] = f"{type(exc).__name__}: {exc}"
        out["traceback"] = traceback.format_exc()
    return out


def default_cases() -> list[dict[str, Any]]:
    base = {
        "model_path": "deepseek-ai/DeepSeek-V3",
        "system": "h100_sxm",
        "backend": "sglang",
        "version": "0.5.9",
        "database_mode": "SILICON",
        "num_layers": 6,
        "batch_size": 8,
        "isl": 512,
        "osl": 16,
        "prefix": 0,
        "ctx_tokens": 2048,
        "attention_backend": "fa3",
        "gemm_quant_mode": "fp8_block",
        "moe_quant_mode": "fp8_block",
        "kvcache_quant_mode": "fp8",
        "fmha_quant_mode": "bfloat16",
        "comm_quant_mode": "half",
        "tp_size": 1,
        "attention_dp_size": 1,
        "moe_tp_size": 1,
        "moe_ep_size": 1,
    }
    return [
        {
            **base,
            "name": "normal_sglang_deepseek_tp1",
            "moe_backend": None,
            "enable_wideep": False,
        },
        {
            **base,
            "name": "enable_wideep_flag_only_tp1",
            "moe_backend": None,
            "enable_wideep": True,
        },
        {
            **base,
            "name": "normal_sglang_deepseek_tp2_shape",
            "tp_size": 2,
            "attention_dp_size": 1,
            "moe_tp_size": 2,
            "moe_ep_size": 1,
            "moe_backend": None,
            "enable_wideep": False,
        },
        {
            **base,
            "name": "wideep_deepep_moe_tp1_adp16_ep16",
            "tp_size": 1,
            "attention_dp_size": 16,
            "moe_tp_size": 1,
            "moe_ep_size": 16,
            "moe_backend": "deepep_moe",
            "enable_wideep": False,
        },
        {
            **base,
            "name": "wideep_deepep_moe_tp1_grid_b1_s512_silicon",
            "database_mode": "SILICON",
            "fmha_quant_mode": "bfloat16",
            "batch_size": 1,
            "isl": 512,
            "osl": 16,
            "prefix": 0,
            "ctx_tokens": 512,
            "tp_size": 1,
            "attention_dp_size": 16,
            "moe_tp_size": 1,
            "moe_ep_size": 16,
            "moe_backend": "deepep_moe",
            "enable_wideep": False,
        },
        {
            **base,
            "name": "wideep_deepep_moe_tp1_grid_b1_s512_silicon_fp8block",
            "database_mode": "SILICON",
            "fmha_quant_mode": "fp8_block",
            "batch_size": 1,
            "isl": 512,
            "osl": 16,
            "prefix": 0,
            "ctx_tokens": 512,
            "tp_size": 1,
            "attention_dp_size": 16,
            "moe_tp_size": 1,
            "moe_ep_size": 16,
            "moe_backend": "deepep_moe",
            "enable_wideep": False,
        },
        {
            **base,
            "name": "wideep_deepep_moe_tp1_grid_b1_s512_sol_fp8block",
            "database_mode": "SOL",
            "fmha_quant_mode": "fp8_block",
            "batch_size": 1,
            "isl": 512,
            "osl": 16,
            "prefix": 0,
            "ctx_tokens": 512,
            "tp_size": 1,
            "attention_dp_size": 16,
            "moe_tp_size": 1,
            "moe_ep_size": 16,
            "moe_backend": "deepep_moe",
            "enable_wideep": False,
        },
        {
            **base,
            "name": "wideep_deepep_moe_tp1_adp16_ep16_hybrid",
            "database_mode": "HYBRID",
            "fmha_quant_mode": "fp8_block",
            "tp_size": 1,
            "attention_dp_size": 16,
            "moe_tp_size": 1,
            "moe_ep_size": 16,
            "moe_backend": "deepep_moe",
            "enable_wideep": False,
        },
        {
            **base,
            "name": "wideep_deepep_moe_tp2_adp8_ep16",
            "tp_size": 2,
            "attention_dp_size": 8,
            "moe_tp_size": 1,
            "moe_ep_size": 16,
            "moe_backend": "deepep_moe",
            "enable_wideep": True,
        },
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    results = [run_case(case) for case in default_cases()]
    for result in results:
        path = args.out_dir / f"{result['case']['name']}.json"
        path.write_text(json.dumps(jsonify(result), indent=2, ensure_ascii=False), encoding="utf-8")

    summary_rows = []
    for result in results:
        per_ops = result.get("per_ops_data") or {}
        mix = per_ops.get("mix_step", {}) if isinstance(per_ops, dict) else {}
        gen = per_ops.get("genonly_step", {}) if isinstance(per_ops, dict) else {}
        summary_rows.append(
            {
                "name": result["case"]["name"],
                "status": result.get("status"),
                "model_class": result.get("model_class"),
                "moe_backend": result["case"].get("moe_backend"),
                "enable_wideep": result["case"].get("enable_wideep"),
                "tp_size": result["case"].get("tp_size"),
                "attention_dp_size": result["case"].get("attention_dp_size"),
                "moe_tp_size": result["case"].get("moe_tp_size"),
                "moe_ep_size": result["case"].get("moe_ep_size"),
                "mix_step_keys": ",".join(mix.keys()) if isinstance(mix, dict) else "",
                "genonly_step_keys": ",".join(gen.keys()) if isinstance(gen, dict) else "",
                "fallback_state_after_run_agg": json.dumps(
                    jsonify(result.get("fallback_state_after_run_agg", {})), ensure_ascii=False, sort_keys=True
                ),
                "error": result.get("error", ""),
            }
        )
    import pandas as pd

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.out_dir / "probe_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"Wrote probe outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
