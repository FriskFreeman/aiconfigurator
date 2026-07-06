#!/usr/bin/env python3
"""Run a lightweight AIC origin probe for SGLang DeepSeek-V3 P/D static paths.

The script intentionally does not patch SDK source files. It wraps operation
instances in memory to capture which ops are queried, their kwargs, returned
latency/energy/source, and any handled fallback exceptions.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
import traceback
import types
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from aiconfigurator.sdk import common, config, models, perf_database
from aiconfigurator.sdk.backends.factory import get_backend
from aiconfigurator.sdk.inference_session import DisaggInferenceSession, InferenceSession


def _jsonify(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {k: _jsonify(v) for k, v in dataclasses.asdict(value).items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if hasattr(value, "name") and hasattr(value, "value"):
        return {"enum": value.__class__.__name__, "name": value.name, "value": value.value}
    return repr(value)


def _enum_name(value: Any) -> str | None:
    if hasattr(value, "name") and hasattr(value, "value"):
        return str(value.name)
    return None


def _enum(enum_cls, name: str | None):
    if name is None:
        return None
    return enum_cls[name]


def build_model_config(args: argparse.Namespace, *, role: str) -> config.ModelConfig:
    tp = getattr(args, f"{role}_tp_size")
    pp = getattr(args, f"{role}_pp_size")
    dp = getattr(args, f"{role}_attention_dp_size")
    moe_tp = getattr(args, f"{role}_moe_tp_size")
    moe_ep = getattr(args, f"{role}_moe_ep_size")
    return config.ModelConfig(
        tp_size=tp,
        pp_size=pp,
        attention_dp_size=dp,
        moe_tp_size=moe_tp,
        moe_ep_size=moe_ep,
        gemm_quant_mode=_enum(common.GEMMQuantMode, args.gemm_quant_mode),
        moe_quant_mode=_enum(common.MoEQuantMode, args.moe_quant_mode),
        kvcache_quant_mode=_enum(common.KVCacheQuantMode, args.kvcache_quant_mode),
        fmha_quant_mode=_enum(common.FMHAQuantMode, args.fmha_quant_mode),
        comm_quant_mode=_enum(common.CommQuantMode, args.comm_quant_mode),
        overwrite_num_layers=args.overwrite_num_layers,
        moe_backend=args.moe_backend,
        attention_backend=args.attention_backend,
        enable_wideep=args.enable_wideep,
    )


def build_runtime_config(args: argparse.Namespace, *, role: str) -> config.RuntimeConfig:
    batch_size = args.prefill_batch_size if role == "prefill" else args.decode_batch_size
    return config.RuntimeConfig(
        batch_size=batch_size,
        beam_width=args.beam_width,
        isl=args.isl,
        osl=args.osl,
        prefix=args.prefix,
        seq_imbalance_correction_scale=args.seq_imbalance_correction_scale,
        gen_seq_imbalance_correction_scale=args.gen_seq_imbalance_correction_scale,
        engine_step_backend=args.engine_step_backend,
    )


def summarize_database(db) -> dict[str, Any]:
    loaded = []
    for attr_name in sorted(dir(db)):
        if not attr_name.startswith("_"):
            continue
        attr = getattr(db, attr_name)
        items = attr if isinstance(attr, tuple) else (attr,)
        for idx, item in enumerate(items):
            if hasattr(item, "loaded") and hasattr(item, "filepath"):
                loaded.append(
                    {
                        "attribute": attr_name if len(items) == 1 else f"{attr_name}[{idx}]",
                        "op_file": getattr(getattr(item, "op_name_enum", None), "value", repr(getattr(item, "op_name_enum", None))),
                        "loaded": bool(item.loaded),
                        "filepath": item.filepath,
                        "exists": os.path.exists(item.filepath),
                    }
                )
    return {
        "system": db.system,
        "backend": db.backend,
        "version": db.version,
        "systems_root": db.systems_root,
        "default_database_mode": db.get_default_database_mode().name,
        "enable_shared_layer": bool(getattr(db, "enable_shared_layer", False)),
        "loaded_op_data": loaded,
    }


def _compact_key(value: Any) -> str:
    enum_name = _enum_name(value)
    if enum_name is not None:
        return enum_name
    if isinstance(value, (str, int, float, bool)) or value is None:
        return str(value)
    return repr(value)


def _nested_key_summary(data: Any, *, max_depth: int = 4, max_keys: int = 24) -> dict[str, Any]:
    """Summarize nested dict axes without dumping the performance table itself."""
    if not isinstance(data, dict):
        return {"type": type(data).__name__}

    keys = list(data.keys())
    key_names = [_compact_key(key) for key in keys[:max_keys]]
    summary: dict[str, Any] = {
        "type": "dict",
        "key_count": len(keys),
        "keys_sample": key_names,
        "truncated": len(keys) > max_keys,
    }
    if max_depth <= 1 or not keys:
        return summary

    first_key = keys[0]
    summary["first_key"] = _compact_key(first_key)
    summary["first_child"] = _nested_key_summary(data[first_key], max_depth=max_depth - 1, max_keys=max_keys)
    return summary


def summarize_quant_tables(db) -> list[dict[str, Any]]:
    """Expose the dtype/quant axes currently loaded in the perf database."""
    interesting_fragments = (
        "gemm",
        "moe",
        "attention",
        "mla",
        "compute_scale",
        "scale_matrix",
        "custom_allreduce",
        "nccl",
    )
    tables = []
    for attr_name in sorted(dir(db)):
        if not attr_name.startswith("_"):
            continue
        attr = getattr(db, attr_name)
        items = attr if isinstance(attr, tuple) else (attr,)
        for idx, item in enumerate(items):
            if not (hasattr(item, "loaded") and hasattr(item, "filepath")):
                continue
            op_file = getattr(getattr(item, "op_name_enum", None), "value", repr(getattr(item, "op_name_enum", None)))
            if not any(fragment in str(op_file) for fragment in interesting_fragments):
                continue
            table = {
                "attribute": attr_name if len(items) == 1 else f"{attr_name}[{idx}]",
                "op_file": op_file,
                "loaded": bool(item.loaded),
                "filepath": item.filepath,
                "exists": os.path.exists(item.filepath),
            }
            if item.loaded:
                table["nested_axes_sample"] = _nested_key_summary(item.data, max_depth=5)
            tables.append(table)
    return tables


def extract_quant_fields(op: Any) -> dict[str, Any]:
    fields = {}
    for key, value in sorted(vars(op).items()):
        key_lower = key.lower()
        if (
            "quant" in key_lower
            or "dtype" in key_lower
            or "attn_backend" in key_lower
            or key in {"_n", "_n_kv", "_num_heads", "_head_size", "_tp_size"}
        ):
            fields[key] = _jsonify(value)
    return fields


def flatten_operation_quant_fields(ops: list[Any], *, phase: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    visited: set[int] = set()

    def visit(op: Any, path: str) -> None:
        if id(op) in visited:
            return
        visited.add(id(op))
        quant_fields = extract_quant_fields(op)
        if quant_fields:
            rows.append(
                {
                    "phase": phase,
                    "path": path,
                    "op_name": getattr(op, "_name", None),
                    "op_class": op.__class__.__name__,
                    "quant_fields": quant_fields,
                }
            )
        if hasattr(op, "_primary"):
            visit(op._primary, f"{path}.primary")
        if hasattr(op, "_fallback"):
            for child_idx, child in enumerate(op._fallback):
                visit(child, f"{path}.fallback[{child_idx}]")
        if hasattr(op, "_group_a"):
            for child_idx, child in enumerate(op._group_a):
                visit(child, f"{path}.group_a[{child_idx}]")
        if hasattr(op, "_group_b"):
            for child_idx, child in enumerate(op._group_b):
                visit(child, f"{path}.group_b[{child_idx}]")

    for idx, op in enumerate(ops):
        visit(op, f"{phase}_ops[{idx}]")
    return rows


def build_quant_summary(model, db, trace: list[dict[str, Any]], *, role: str, mode: str) -> dict[str, Any]:
    config_fields = {
        "gemm_quant_mode": model.config.gemm_quant_mode,
        "moe_quant_mode": model.config.moe_quant_mode,
        "kvcache_quant_mode": model.config.kvcache_quant_mode,
        "fmha_quant_mode": model.config.fmha_quant_mode,
        "comm_quant_mode": model.config.comm_quant_mode,
    }
    op_rows = flatten_operation_quant_fields(model.context_ops, phase="context") + flatten_operation_quant_fields(
        model.generation_ops, phase="generation"
    )
    query_rows = [
        {
            "phase": event.get("phase"),
            "path": event.get("path"),
            "op_name": event.get("op_name"),
            "op_class": event.get("op_class"),
            "status": event.get("status"),
            "source": event.get("source"),
            "latency_ms": event.get("latency_ms"),
            "op_quant_fields": event.get("op_quant_fields", {}),
            "kwargs": event.get("kwargs", {}),
        }
        for event in trace
    ]
    return {
        "role": role,
        "mode": mode,
        "model_class": model.__class__.__name__,
        "backend_quant_defaults_after_model_build": _jsonify(config_fields),
        "notes": {
            "gemm_and_moe": "Quant modes here are used by SDK ops for weight memory scaling, compute throughput assumptions, and perf-table lookup axes.",
            "kvcache": "KV cache quant mode models stored/read KV cache precision; live q/k/v activations can remain BF16 unless a collector/backend explicitly casts them.",
            "fmha": "FMHA quant mode models attention compute/table axis. For DeepSeek V3 SGLang defaults in this run it is BF16 even when KV cache is FP8.",
        },
        "operation_quant_fields": op_rows,
        "query_quant_events": query_rows,
        "database_quant_tables": summarize_quant_tables(db),
    }


def _axis_summary(values: list[Any]) -> dict[str, Any]:
    values = sorted(v for v in values if isinstance(v, (int, float)))
    if not values:
        return {"count": 0, "min": None, "max": None, "contains": False}
    return {"count": len(values), "min": values[0], "max": values[-1]}


def _interp_3d_probe(data: dict, x: int, y: int, z: int) -> dict[str, Any]:
    """Best-effort provenance from the nested 3D table passed into PerfDatabase.

    The SDK's interpolation helpers return only metrics. This probe does not
    change the query result; it records whether the requested point is present
    in the table and whether it falls inside the observed global axis ranges.
    Sparse grids can still fail inside interpolation even when the global ranges
    contain the point, so the status is intentionally marked as best-effort.
    """
    x_keys = [k for k in data.keys() if isinstance(k, (int, float))]
    y_keys = []
    z_keys = []
    exact_hit = False
    leaf_value = None
    if x in data and isinstance(data[x], dict):
        y_keys_for_x = [k for k in data[x].keys() if isinstance(k, (int, float))]
        if y in data[x] and isinstance(data[x][y], dict):
            z_keys_for_xy = [k for k in data[x][y].keys() if isinstance(k, (int, float))]
            exact_hit = z in data[x][y]
            if exact_hit:
                leaf_value = data[x][y][z]
        else:
            z_keys_for_xy = []
    else:
        y_keys_for_x = []
        z_keys_for_xy = []

    for x_value in x_keys:
        child = data.get(x_value, {})
        if isinstance(child, dict):
            y_keys.extend(k for k in child.keys() if isinstance(k, (int, float)))
            for y_value, grandchild in child.items():
                if isinstance(y_value, (int, float)) and isinstance(grandchild, dict):
                    z_keys.extend(k for k in grandchild.keys() if isinstance(k, (int, float)))

    def in_range(value: int, values: list[Any]) -> bool:
        values = sorted(v for v in values if isinstance(v, (int, float)))
        return bool(values) and values[0] <= value <= values[-1]

    if exact_hit:
        classification = "direct_hit"
    elif in_range(x, x_keys) and in_range(y, y_keys) and in_range(z, z_keys):
        classification = "interpolation_or_sparse_grid"
    else:
        classification = "extrapolation_or_out_of_grid"

    return {
        "classification_best_effort": classification,
        "exact_hit": exact_hit,
        "requested": {"x": x, "y": y, "z": z},
        "global_axes": {
            "x": _axis_summary(x_keys),
            "y": _axis_summary(y_keys),
            "z": _axis_summary(z_keys),
        },
        "local_axes": {
            "y_at_x": _axis_summary(y_keys_for_x),
            "z_at_xy": _axis_summary(z_keys_for_xy),
        },
        "exact_leaf_value": _jsonify(leaf_value) if exact_hit else None,
        "note": (
            "Best-effort wrapper around PerfDatabase._interp_3d. Native AIC returns "
            "metrics only; it does not expose interpolation neighbor points or row ids."
        ),
    }


def install_database_tracing(db, interp_trace: list[dict[str, Any]]) -> None:
    original_interp_3d = db._interp_3d

    def traced_interp_3d(self, x: int, y: int, z: int, data: dict, method: str):
        event = _interp_3d_probe(data, x, y, z)
        event["method"] = method
        event["data_id"] = id(data)
        event["current_op"] = _jsonify(getattr(self, "_aic_probe_current_op", None))
        event["status"] = "pending"
        interp_trace.append(event)
        start_ns = time.perf_counter_ns()
        try:
            result = original_interp_3d(x, y, z, data, method)
            event["status"] = "ok"
            event["result"] = _jsonify(result)
            return result
        except Exception as exc:
            event["status"] = "exception"
            event["exception_type"] = type(exc).__name__
            event["exception"] = str(exc)
            event["traceback_tail"] = traceback.format_exc().splitlines()[-8:]
            raise
        finally:
            event["elapsed_wall_ms"] = (time.perf_counter_ns() - start_ns) / 1e6

    db._interp_3d = types.MethodType(traced_interp_3d, db)


def operation_tree(ops: list[Any]) -> list[dict[str, Any]]:
    return [_describe_op(op) for op in ops]


def _describe_op(op: Any) -> dict[str, Any]:
    node = {
        "class": op.__class__.__name__,
        "name": getattr(op, "_name", None),
        "scale_factor": getattr(op, "_scale_factor", None),
        "fields": {
            key: _jsonify(value)
            for key, value in sorted(vars(op).items())
            if key not in {"_primary", "_fallback", "_group_a", "_group_b"}
            and not callable(value)
            and not key.startswith("__")
        },
    }
    if hasattr(op, "_primary"):
        node["primary"] = _describe_op(op._primary)
    if hasattr(op, "_fallback"):
        node["fallback"] = [_describe_op(child) for child in op._fallback]
    if hasattr(op, "_group_a"):
        node["group_a"] = [_describe_op(child) for child in op._group_a]
    if hasattr(op, "_group_b"):
        node["group_b"] = [_describe_op(child) for child in op._group_b]
    return node


def install_op_tracing(ops: list[Any], trace: list[dict[str, Any]], *, phase: str) -> None:
    visited: set[int] = set()

    def wrap(op: Any, path: str) -> None:
        if id(op) in visited:
            return
        visited.add(id(op))

        original_query = op.query

        def traced_query(self, database, **kwargs):
            event = {
                "phase": phase,
                "path": path,
                "op_name": getattr(self, "_name", None),
                "op_class": self.__class__.__name__,
                "op_quant_fields": extract_quant_fields(self),
                "kwargs": _jsonify(kwargs),
                "database_mode_before": database.get_default_database_mode().name,
            }
            trace.append(event)
            start_ns = time.perf_counter_ns()
            previous_current_op = getattr(database, "_aic_probe_current_op", None)
            database._aic_probe_current_op = {
                "phase": phase,
                "path": path,
                "op_name": getattr(self, "_name", None),
                "op_class": self.__class__.__name__,
            }
            try:
                result = original_query(database, **kwargs)
                event.update(
                    {
                        "status": "ok",
                        "latency_ms": float(result),
                        "energy_wms": float(getattr(result, "energy", 0.0)),
                        "power_w": float(getattr(result, "power", 0.0)),
                        "source": getattr(result, "source", "silicon"),
                    }
                )
                return result
            except Exception as exc:
                event.update(
                    {
                        "status": "exception",
                        "exception_type": type(exc).__name__,
                        "exception": str(exc),
                        "traceback_tail": traceback.format_exc().splitlines()[-8:],
                    }
                )
                raise
            finally:
                event["elapsed_wall_ms"] = (time.perf_counter_ns() - start_ns) / 1e6
                event["database_mode_after"] = database.get_default_database_mode().name
                database._aic_probe_current_op = previous_current_op

        op.query = types.MethodType(traced_query, op)

        if hasattr(op, "_primary"):
            wrap(op._primary, f"{path}.primary")
        if hasattr(op, "_fallback"):
            for idx, child in enumerate(op._fallback):
                wrap(child, f"{path}.fallback[{idx}]")
        if hasattr(op, "_group_a"):
            for idx, child in enumerate(op._group_a):
                wrap(child, f"{path}.group_a[{idx}]")
        if hasattr(op, "_group_b"):
            for idx, child in enumerate(op._group_b):
                wrap(child, f"{path}.group_b[{idx}]")

    for idx, op in enumerate(ops):
        wrap(op, f"{phase}_ops[{idx}]")


def export_summary(summary, out_dir: Path, stem: str) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_df = summary.get_summary_df()
    if summary_df is not None:
        summary_df.to_csv(out_dir / f"{stem}_summary_df.csv", index=False)
        (out_dir / f"{stem}_summary_df.json").write_text(
            summary_df.to_json(orient="records", indent=2), encoding="utf-8"
        )

    perf_info, mem_info, context_info, generation_info = summary.get_static_info()
    (out_dir / f"{stem}_static_info.txt").write_text(
        "\n".join([perf_info, mem_info, context_info, generation_info]), encoding="utf-8"
    )
    payload = {
        "check_oom": summary.check_oom(),
        "check_kv_cache_oom": summary.check_kv_cache_oom(),
        "memory": summary.get_memory(),
        "context_latency_dict": summary.get_context_latency_dict(),
        "generation_latency_dict": summary.get_generation_latency_dict(),
        "context_energy_wms_dict": summary.get_context_energy_wms_dict(),
        "generation_energy_wms_dict": summary.get_generation_energy_wms_dict(),
        "context_source_dict": summary.get_context_source_dict(),
        "generation_source_dict": summary.get_generation_source_dict(),
        "context_power_avg": summary.get_context_power_avg(),
        "generation_power_avg": summary.get_generation_power_avg(),
        "e2e_power_avg": summary.get_e2e_power_avg(),
        "result_dict": summary.get_result_dict(),
    }
    (out_dir / f"{stem}_native_summary.json").write_text(
        json.dumps(_jsonify(payload), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return payload


def run_static_probe(args: argparse.Namespace, out_dir: Path, *, role: str, mode: str) -> dict[str, Any]:
    model_config = build_model_config(args, role=role)
    runtime_config = build_runtime_config(args, role=role)
    db = perf_database.get_database(
        args.system,
        args.backend,
        args.version,
        systems_paths=args.systems_paths,
        database_mode=args.database_mode,
    )
    if db is None:
        raise RuntimeError(f"failed to load database: system={args.system} backend={args.backend} version={args.version}")
    db.set_default_database_mode(common.DatabaseMode[args.database_mode])
    interp_trace: list[dict[str, Any]] = []
    install_database_tracing(db, interp_trace)
    backend = get_backend(args.backend)
    model = models.get_model(args.model_path, model_config, args.backend)

    trace: list[dict[str, Any]] = []
    install_op_tracing(model.context_ops, trace, phase="context")
    install_op_tracing(model.generation_ops, trace, phase="generation")

    session = InferenceSession(model=model, database=db, backend=backend)
    summary = session.run_static(runtime_config=runtime_config, mode=mode, stride=args.stride)

    role_dir = out_dir / role
    export_summary(summary, role_dir, role)
    (role_dir / "query_trace.json").write_text(json.dumps(trace, indent=2, ensure_ascii=False), encoding="utf-8")
    (role_dir / "interp_trace.json").write_text(
        json.dumps(interp_trace, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    meta = {
        "role": role,
        "mode": mode,
        "model_class": model.__class__.__name__,
        "backend_class": backend.__class__.__name__,
        "runtime_config": _jsonify(runtime_config),
        "model_config_after_defaults": _jsonify(model.config),
        "model_info": {
            "model_path": model.model_path,
            "model_name": getattr(model, "model_name", ""),
            "model_family": getattr(model, "model_family", ""),
            "architecture": getattr(model, "_architecture", ""),
            "num_context_ops": len(model.context_ops),
            "num_generation_ops": len(model.generation_ops),
        },
        "context_ops": operation_tree(model.context_ops),
        "generation_ops": operation_tree(model.generation_ops),
        "database": summarize_database(db),
        "interp_trace_events": len(interp_trace),
    }
    quant_summary = build_quant_summary(model, db, trace, role=role, mode=mode)
    (role_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    (role_dir / "quant_summary.json").write_text(
        json.dumps(_jsonify(quant_summary), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return {"meta": meta, "trace": trace, "summary": summary.get_result_dict(), "quant_summary": quant_summary}


def run_disagg_native(args: argparse.Namespace, out_dir: Path) -> None:
    prefill_config = build_model_config(args, role="prefill")
    decode_config = build_model_config(args, role="decode")
    runtime_config = config.RuntimeConfig(
        isl=args.isl,
        osl=args.osl,
        prefix=args.prefix,
        engine_step_backend=args.engine_step_backend,
    )
    prefill_db = perf_database.get_database(args.system, args.backend, args.version, systems_paths=args.systems_paths)
    decode_db = perf_database.get_database(args.system, args.backend, args.version, systems_paths=args.systems_paths)
    if prefill_db is None or decode_db is None:
        raise RuntimeError("failed to load database for disagg native run")
    prefill_db.set_default_database_mode(common.DatabaseMode[args.database_mode])
    decode_db.set_default_database_mode(common.DatabaseMode[args.database_mode])
    session = DisaggInferenceSession(
        prefill_database=prefill_db,
        prefill_backend=get_backend(args.backend),
        decode_database=decode_db,
        decode_backend=get_backend(args.backend),
    )
    summary = session.run_disagg(
        model_path=args.model_path,
        runtime_config=runtime_config,
        prefill_model_config=prefill_config,
        prefill_batch_size=args.prefill_batch_size,
        prefill_num_worker=args.prefill_num_workers,
        decode_model_config=decode_config,
        decode_batch_size=args.decode_batch_size,
        decode_num_worker=args.decode_num_workers,
    )
    disagg_dir = out_dir / "disagg_native"
    export_summary(summary, disagg_dir, "disagg")
    (disagg_dir / "run_meta.json").write_text(
        json.dumps(
            _jsonify(
                {
                    "runtime_config": runtime_config,
                    "prefill_model_config": prefill_config,
                    "decode_model_config": decode_config,
                    "prefill_num_workers": args.prefill_num_workers,
                    "decode_num_workers": args.decode_num_workers,
                    "note": "This is the unwrapped DisaggInferenceSession.run_disagg result. Detailed per-inner-query trace is collected in prefill/ and decode/ static runs.",
                }
            ),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(REPO_ROOT / ".self/task-aic-origin/runs"))
    parser.add_argument("--model-path", default="deepseek-ai/DeepSeek-V3")
    parser.add_argument("--system", default="h100_sxm")
    parser.add_argument("--backend", default="sglang")
    parser.add_argument("--version", default="0.5.9")
    parser.add_argument("--systems-paths", default=None)
    parser.add_argument("--database-mode", default="SILICON", choices=[m.name for m in common.DatabaseMode])
    parser.add_argument("--isl", type=int, default=1024)
    parser.add_argument("--osl", type=int, default=16)
    parser.add_argument("--prefix", type=int, default=0)
    parser.add_argument("--beam-width", type=int, default=1)
    parser.add_argument("--prefill-batch-size", type=int, default=1)
    parser.add_argument("--decode-batch-size", type=int, default=4)
    parser.add_argument("--prefill-num-workers", type=int, default=1)
    parser.add_argument("--decode-num-workers", type=int, default=1)
    parser.add_argument("--prefill-tp-size", type=int, default=8)
    parser.add_argument("--decode-tp-size", type=int, default=8)
    parser.add_argument("--prefill-pp-size", type=int, default=1)
    parser.add_argument("--decode-pp-size", type=int, default=1)
    parser.add_argument("--prefill-attention-dp-size", type=int, default=1)
    parser.add_argument("--decode-attention-dp-size", type=int, default=1)
    parser.add_argument("--prefill-moe-tp-size", type=int, default=1)
    parser.add_argument("--decode-moe-tp-size", type=int, default=1)
    parser.add_argument("--prefill-moe-ep-size", type=int, default=8)
    parser.add_argument("--decode-moe-ep-size", type=int, default=8)
    parser.add_argument("--gemm-quant-mode", default=None)
    parser.add_argument("--moe-quant-mode", default=None)
    parser.add_argument("--kvcache-quant-mode", default=None)
    parser.add_argument("--fmha-quant-mode", default=None)
    parser.add_argument("--comm-quant-mode", default="half")
    parser.add_argument("--overwrite-num-layers", type=int, default=0)
    parser.add_argument("--moe-backend", default=None)
    parser.add_argument("--attention-backend", default="flashinfer")
    parser.add_argument("--enable-wideep", action="store_true")
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--seq-imbalance-correction-scale", type=float, default=1.0)
    parser.add_argument("--gen-seq-imbalance-correction-scale", type=float, default=1.0)
    parser.add_argument("--engine-step-backend", default=None)
    parser.add_argument("--skip-disagg-native", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_root) / f"{timestamp}_aic_origin_sglang_deepseek_v3_pd_static"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "args.json").write_text(json.dumps(_jsonify(vars(args)), indent=2, ensure_ascii=False), encoding="utf-8")

    results = {
        "prefill": run_static_probe(args, out_dir, role="prefill", mode="static_ctx"),
        "decode": run_static_probe(args, out_dir, role="decode", mode="static_gen"),
    }
    if not args.skip_disagg_native:
        run_disagg_native(args, out_dir)

    compact = {
        "output_dir": str(out_dir),
        "prefill_summary": results["prefill"]["summary"],
        "decode_summary": results["decode"]["summary"],
        "prefill_trace_events": len(results["prefill"]["trace"]),
        "decode_trace_events": len(results["decode"]["trace"]),
        "prefill_interp_trace_events": results["prefill"]["meta"]["interp_trace_events"],
        "decode_interp_trace_events": results["decode"]["meta"]["interp_trace_events"],
    }
    (out_dir / "run_complete.json").write_text(json.dumps(_jsonify(compact), indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(_jsonify(compact), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
