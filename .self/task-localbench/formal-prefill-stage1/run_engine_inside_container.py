#!/usr/bin/env python3
import argparse
import copy
import inspect
import json
import math
import os
import traceback
from contextlib import contextmanager
from pathlib import Path

import sglang as sgl
from sglang.srt.server_args import ServerArgs


def install_generate_req_batch_session_patch() -> None:
    from sglang.srt.managers.io_struct import GenerateReqInput

    if getattr(GenerateReqInput.__getitem__, "_aic_batch_session_patch", False):
        return

    original_getitem = GenerateReqInput.__getitem__

    def patched_getitem(self, i):
        item = original_getitem(self, i)
        session_params = getattr(self, "session_params", None)
        if isinstance(session_params, list):
            item.session_params = session_params[i]
        return item

    patched_getitem._aic_batch_session_patch = True
    GenerateReqInput.__getitem__ = patched_getitem


def _to_jsonable_list(value) -> list:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    try:
        return list(value)
    except TypeError:
        return []


def _safe_len(value) -> int:
    try:
        return len(value)
    except TypeError:
        return 0


def _build_prefill_chunk_payload(scheduler, batch) -> dict[str, object]:
    reqs = list(getattr(batch, "reqs", []) or [])
    prefix_lens = _to_jsonable_list(getattr(batch, "prefix_lens", None))
    if not prefix_lens:
        prefix_lens = [_safe_len(getattr(req, "prefix_indices", [])) for req in reqs]

    extend_lens = _to_jsonable_list(getattr(batch, "extend_lens", None))
    if not extend_lens:
        extend_lens = [getattr(req, "extend_input_len", None) for req in reqs]

    seq_lens = _to_jsonable_list(getattr(batch, "seq_lens_cpu", None))
    if not seq_lens:
        seq_lens = [_safe_len(getattr(req, "fill_ids", [])) for req in reqs]

    chunked_req = getattr(batch, "chunked_req", None)
    if chunked_req is None:
        chunked_req = getattr(scheduler, "chunked_req", None)

    chunked_req_prefix_len = 0
    chunked_req_extend_input_len = None
    chunked_req_seq_len = None
    chunked_req_rid = None
    if chunked_req is not None:
        chunked_req_prefix_len = _safe_len(getattr(chunked_req, "prefix_indices", []))
        chunked_req_extend_input_len = getattr(chunked_req, "extend_input_len", None)
        chunked_req_seq_len = _safe_len(getattr(chunked_req, "fill_ids", []))
        chunked_req_rid = getattr(chunked_req, "rid", None)

    extend_num_tokens = getattr(batch, "extend_num_tokens", None)
    if extend_num_tokens is None:
        extend_num_tokens = sum(x for x in extend_lens if isinstance(x, int))

    return {
        "kind": "prefill_chunk",
        "forward_ct": getattr(scheduler, "forward_ct", None),
        "batch_size": len(reqs),
        "extend_num_tokens": extend_num_tokens,
        "prefix_lens": prefix_lens,
        "extend_lens": extend_lens,
        "seq_lens": seq_lens,
        "chunked_req_prefix_len": chunked_req_prefix_len,
        "chunked_req_extend_input_len": chunked_req_extend_input_len,
        "chunked_req_seq_len": chunked_req_seq_len,
        "chunked_req_rid": chunked_req_rid,
    }


def install_prefill_chunk_nvtx_patch() -> None:
    from sglang.srt.managers.scheduler import Scheduler

    if getattr(Scheduler.run_batch, "_aic_prefill_chunk_nvtx_patch", False):
        return

    original_run_batch = Scheduler.run_batch

    def patched_run_batch(self, batch, *args, **kwargs):
        is_extend = False
        forward_mode = getattr(batch, "forward_mode", None)
        if forward_mode is not None and hasattr(forward_mode, "is_extend"):
            is_extend = bool(forward_mode.is_extend())
        if not is_extend:
            return original_run_batch(self, batch, *args, **kwargs)

        pushed = False
        try:
            import torch

            payload = {"AICPrefillChunk": _build_prefill_chunk_payload(self, batch)}
            torch.cuda.nvtx.range_push(repr(payload))
            pushed = True
        except Exception:
            pushed = False

        try:
            return original_run_batch(self, batch, *args, **kwargs)
        finally:
            if pushed:
                try:
                    import torch

                    torch.cuda.nvtx.range_pop()
                except Exception:
                    pass

    patched_run_batch._aic_prefill_chunk_nvtx_patch = True
    Scheduler.run_batch = patched_run_batch


def install_decode_only_profile_patch() -> None:
    """Capture CUDA profiler ranges only around real decode batches.

    SGLang's profile_by_stage starts profiling at the first prefill stage and
    restarts at decode. That is fine for chrome traces, but for nsys
    capture-range=cudaProfilerApi the first cudaProfilerStop can terminate the
    capture before decode. This patch keeps the profiler dormant through
    prefill and wraps the configured number of decode batches directly.
    """
    from sglang.srt.managers.scheduler import Scheduler

    if getattr(Scheduler.run_batch, "_aic_decode_only_profile_patch", False):
        return

    original_profile_predicate = Scheduler._profile_batch_predicate
    original_run_batch = Scheduler.run_batch

    def _enabled() -> bool:
        return os.getenv("AIC_PROFILE_DECODE_ONLY_STAGE") == "1"

    def patched_profile_predicate(self, batch):
        if _enabled() and getattr(self, "profile_by_stage", False):
            return
        return original_profile_predicate(self, batch)

    def patched_run_batch(self, batch, *args, **kwargs):
        if not (_enabled() and getattr(self, "profile_by_stage", False)):
            return original_run_batch(self, batch, *args, **kwargs)

        forward_mode = getattr(batch, "forward_mode", None)
        is_decode = bool(
            forward_mode is not None
            and hasattr(forward_mode, "is_decode")
            and forward_mode.is_decode()
        )
        if not is_decode:
            return original_run_batch(self, batch, *args, **kwargs)

        target_decode_ct = getattr(self, "profiler_target_decode_ct", None) or 1
        captured_decode_ct = getattr(self, "_aic_profiled_decode_ct", 0)
        if captured_decode_ct >= target_decode_ct:
            return original_run_batch(self, batch, *args, **kwargs)

        if not getattr(self, "profile_in_progress", False):
            self.start_profile(forward_mode)
        self._aic_profiled_decode_ct = captured_decode_ct + 1
        try:
            return original_run_batch(self, batch, *args, **kwargs)
        finally:
            if (
                getattr(self, "_aic_profiled_decode_ct", 0) >= target_decode_ct
                and getattr(self, "profile_in_progress", False)
            ):
                self.stop_profile(stage=forward_mode)

    patched_profile_predicate._aic_decode_only_profile_patch = True
    patched_run_batch._aic_decode_only_profile_patch = True
    Scheduler._profile_batch_predicate = patched_profile_predicate
    Scheduler.run_batch = patched_run_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--request-json", required=True)
    parser.add_argument("--profile-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-layers", type=int, default=5)
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument("--attention-backend", default="auto")
    parser.add_argument("--decode-attention-backend", default="auto")
    parser.add_argument("--mem-fraction-static", type=float, default=0.5)
    parser.add_argument("--cuda-graph-mode", choices=["off", "on"], default="off")
    parser.add_argument("--pcg-mode", choices=["off", "on"], default="off")
    parser.add_argument("--layerwise-marker", action="store_true")
    parser.add_argument("--warmup-runs", type=int, default=0)
    parser.add_argument("--chunked-prefill-size", type=int, default=None)
    parser.add_argument("--max-prefill-tokens", type=int, default=None)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--allow-chunked-kv", action="store_true")
    parser.add_argument("--chunked-prefix-cache-threshold", type=int, default=8192)
    parser.add_argument(
        "--limit-cuda-graph-bs-to-batch",
        action="store_true",
        help="Capture CUDA graph only for the formal request batch size.",
    )
    parser.add_argument(
        "--decode-empty-input-continuation",
        action="store_true",
        help=(
            "After preparing the session prefix, profile generation with empty "
            "input_ids and max_new_tokens. This keeps the formal request on the "
            "decode path instead of appending fresh user tokens via extend/prefill."
        ),
    )
    parser.add_argument(
        "--profile-decode-only-stage",
        action="store_true",
        help=(
            "With profile_by_stage=true, skip prefill profiler ranges and wrap "
            "only real ForwardMode.DECODE batches. Intended for nsys decode+CG."
        ),
    )
    parser.add_argument(
        "--stream-profile-after-first-token",
        action="store_true",
        help=(
            "Run formal generation as stream=True, consume the first streamed "
            "token outside the profile window, then profile the remaining decode steps."
        ),
    )
    return parser.parse_args()


def _validate_request_schema(request: dict[str, object]) -> None:
    fresh_input_ids = request.get("fresh_input_ids")
    prefix_input_ids = request.get("prefix_input_ids")
    if not isinstance(fresh_input_ids, list) or not fresh_input_ids:
        raise ValueError("request-json must contain non-empty fresh_input_ids")
    if not isinstance(prefix_input_ids, list):
        raise ValueError("request-json must contain prefix_input_ids list")
    if len(fresh_input_ids) != len(prefix_input_ids):
        raise ValueError("fresh_input_ids and prefix_input_ids must have the same batch size")
    warmup_fresh_input_ids = request.get("warmup_fresh_input_ids")
    warmup_prefix_input_ids = request.get("warmup_prefix_input_ids")
    if warmup_fresh_input_ids is not None:
        if not isinstance(warmup_fresh_input_ids, list) or len(warmup_fresh_input_ids) != len(fresh_input_ids):
            raise ValueError("warmup_fresh_input_ids must match fresh_input_ids batch size")
    if warmup_prefix_input_ids is not None:
        if not isinstance(warmup_prefix_input_ids, list) or len(warmup_prefix_input_ids) != len(prefix_input_ids):
            raise ValueError("warmup_prefix_input_ids must match prefix_input_ids batch size")


def _request_lengths(req_tokens: list[list[int]]) -> list[int]:
    return [len(tokens) for tokens in req_tokens]


def _estimate_max_kv_chunk_capacity(context_length: int, tp_size: int) -> int:
    del tp_size
    return max(1, 128 * 1024)


def _precheck_chunked_kv(
    request: dict[str, object],
    context_length: int,
    tp_size: int,
    allow_chunked_kv: bool,
    chunked_prefix_cache_threshold: int,
) -> dict[str, object]:
    prefix_input_ids = request["prefix_input_ids"]
    fresh_input_ids = request["fresh_input_ids"]
    prefix_lens = _request_lengths(prefix_input_ids)
    fresh_lens = _request_lengths(fresh_input_ids)
    sum_prefix = sum(prefix_lens)
    max_kv_chunk_capacity = _estimate_max_kv_chunk_capacity(context_length, tp_size)

    chunked_kv_risk = (
        sum_prefix >= chunked_prefix_cache_threshold
        and sum_prefix > max_kv_chunk_capacity
    )

    summary = {
        "prefix_lens": prefix_lens,
        "fresh_lens": fresh_lens,
        "sum_prefix_lens": sum_prefix,
        "sum_fresh_lens": sum(fresh_lens),
        "chunked_prefix_cache_threshold": chunked_prefix_cache_threshold,
        "max_kv_chunk_capacity_estimate": max_kv_chunk_capacity,
        "chunked_kv_risk": chunked_kv_risk,
        "allow_chunked_kv": allow_chunked_kv,
    }
    if chunked_kv_risk and not allow_chunked_kv:
        raise RuntimeError(
            "Chunked-KV precheck failed: sum(prefix_lens) exceeds DeepSeek MHA one-shot capacity; "
            "this run would enter MHA_CHUNKED_KV. "
            f"sum_prefix_lens={sum_prefix}, "
            f"chunked_prefix_cache_threshold={chunked_prefix_cache_threshold}, "
            f"max_kv_chunk_capacity_estimate={max_kv_chunk_capacity}."
        )
    return summary


@contextmanager
def _tp_load_model_patch(tp_size: int, output_dir: Path | None = None):
    if tp_size <= 1:
        yield
        return

    import sglang.srt.distributed.parallel_state as ps
    from sglang.srt.model_executor.model_runner import ModelRunner

    original_load = ModelRunner.load_model

    def _patch_group(group, *, world_size: int, rank: int):
        if group is None:
            return None
        original_state = {
            "world_size": getattr(group, "world_size", None),
            "rank_in_group": getattr(group, "rank_in_group", None),
        }
        group.world_size = world_size
        group.rank_in_group = rank
        return original_state

    def _restore_group(group, original_state):
        if group is None or original_state is None:
            return
        group.world_size = original_state["world_size"]
        group.rank_in_group = original_state["rank_in_group"]

    def _shape_of(value):
        if value is None:
            return None
        shape = getattr(value, "shape", None)
        if shape is None:
            return None
        try:
            return [int(dim) for dim in shape]
        except Exception:
            return repr(shape)

    def _inspect_linear_module(module):
        if module is None:
            return None
        fields = {
            "class_name": module.__class__.__name__,
            "tp_size": getattr(module, "tp_size", None),
            "tp_rank": getattr(module, "tp_rank", None),
            "input_size": getattr(module, "input_size", None),
            "output_size": getattr(module, "output_size", None),
            "output_size_per_partition": getattr(
                module, "output_size_per_partition", None
            ),
            "gather_output": getattr(module, "gather_output", None),
            "reduce_results": getattr(module, "reduce_results", None),
            "weight_shape": _shape_of(getattr(module, "weight", None)),
            "weight_packed_shape": _shape_of(getattr(module, "weight_packed", None)),
            "bias_shape": _shape_of(getattr(module, "bias", None)),
        }
        return fields

    def _dump_tp_model_inspection(model_runner):
        if output_dir is None:
            return
        try:
            model = getattr(model_runner, "model", None)
            model_root = getattr(model, "model", None)
            layers = getattr(model_root, "layers", None)
            if not layers:
                return
            layer0 = layers[0]
            self_attn = getattr(layer0, "self_attn", None)
            if self_attn is None:
                return
            inspection = {
                "requested_tp_size": tp_size,
                "self_attn_class_name": self_attn.__class__.__name__,
                "num_heads": getattr(self_attn, "num_heads", None),
                "num_local_heads": getattr(self_attn, "num_local_heads", None),
                "qk_head_dim": getattr(self_attn, "qk_head_dim", None),
                "qk_nope_head_dim": getattr(self_attn, "qk_nope_head_dim", None),
                "qk_rope_head_dim": getattr(self_attn, "qk_rope_head_dim", None),
                "v_head_dim": getattr(self_attn, "v_head_dim", None),
                "kv_lora_rank": getattr(self_attn, "kv_lora_rank", None),
                "q_lora_rank": getattr(self_attn, "q_lora_rank", None),
                "q_proj": _inspect_linear_module(getattr(self_attn, "q_proj", None)),
                "q_b_proj": _inspect_linear_module(getattr(self_attn, "q_b_proj", None)),
                "kv_b_proj": _inspect_linear_module(getattr(self_attn, "kv_b_proj", None)),
                "o_proj": _inspect_linear_module(getattr(self_attn, "o_proj", None)),
                "lm_head": _inspect_linear_module(getattr(model, "lm_head", None)),
            }
            (output_dir / "tp_model_inspection.json").write_text(
                json.dumps(inspection, indent=2)
            )
        except Exception as exc:
            (output_dir / "tp_model_inspection_error.txt").write_text(
                traceback.format_exc() + f"\ninspection_error={exc!r}\n"
            )

    def patched_load(self):
        tp_group = ps._TP
        attn_tp_group = getattr(ps, "_ATTN_TP", None)
        if tp_group is None:
            raise RuntimeError("_TP group is not initialized before ModelRunner.load_model")
        tp_group_state = _patch_group(tp_group, world_size=tp_size, rank=0)
        attn_tp_group_state = _patch_group(attn_tp_group, world_size=tp_size, rank=0)
        try:
            result = original_load(self)
            _dump_tp_model_inspection(self)
            return result
        finally:
            _restore_group(attn_tp_group, attn_tp_group_state)
            _restore_group(tp_group, tp_group_state)

    ModelRunner.load_model = patched_load
    try:
        yield
    finally:
        ModelRunner.load_model = original_load


def _build_generate_payload(
    *,
    input_ids: list[list[int]],
    sampling_params: dict[str, object],
    rids: list[str],
    session_id: str | None = None,
    session_rids: list[str | None] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "input_ids": input_ids,
        "sampling_params": sampling_params,
        "return_logprob": False,
        "rid": rids,
    }
    if session_id is not None:
        if session_rids is None or len(session_rids) != len(input_ids):
            raise ValueError("session_rids must match batch size when session_id is provided")
        session_params = []
        for rid in session_rids:
            entry = {"id": session_id}
            if rid is not None:
                entry["rid"] = rid
            session_params.append(entry)
        payload["session_params"] = session_params
    return payload


def _single_from_batched_payload(payload: dict[str, object]) -> dict[str, object]:
    single = {
        "input_ids": payload["input_ids"][0],
        "sampling_params": payload["sampling_params"],
        "return_logprob": payload.get("return_logprob", False),
        "rid": payload["rid"][0],
    }
    session_params = payload.get("session_params")
    if isinstance(session_params, list) and session_params:
        single["session_params"] = session_params[0]
    return single


def _close_sessions(engine, session_ids: list[str | None]) -> None:
    for session_id in session_ids:
        if session_id is None:
            continue
        engine.close_session(session_id)


def _select_phase_request(
    request: dict[str, object],
    *,
    use_warmup_inputs: bool,
) -> dict[str, object]:
    if not use_warmup_inputs:
        return request

    warmup_prefix_input_ids = request.get("warmup_prefix_input_ids")
    warmup_fresh_input_ids = request.get("warmup_fresh_input_ids")
    if not isinstance(warmup_prefix_input_ids, list) or not isinstance(
        warmup_fresh_input_ids, list
    ):
        return request

    warmup_request = dict(request)
    warmup_request["prefix_input_ids"] = warmup_prefix_input_ids
    warmup_request["fresh_input_ids"] = warmup_fresh_input_ids
    return warmup_request


def _run_prefix_cache_pass(
    engine,
    request: dict[str, object],
    output_dir: Path,
    stage_file: Path,
    phase_name: str,
) -> dict[str, object]:
    prefix_input_ids = request["prefix_input_ids"]
    fresh_input_ids = request["fresh_input_ids"]
    session_ids: list[str] = []
    continuation_rids: list[str | None] = []
    prefix_payloads: list[dict[str, object]] = []

    for idx, prefix_ids in enumerate(prefix_input_ids):
        fresh_ids = fresh_input_ids[idx]
        session_id = engine.open_session(
            capacity_of_str_len=len(prefix_ids) + len(fresh_ids) + 32
        )
        session_ids.append(session_id)
        if prefix_ids:
            prefix_rid = f"stage1-prefix-{idx}"
            continuation_rids.append(prefix_rid)
            prefix_payloads.append(
                _build_generate_payload(
                    input_ids=[prefix_ids],
                    sampling_params={"temperature": 0.0, "max_new_tokens": 0},
                    rids=[prefix_rid],
                    session_id=session_id,
                    session_rids=[None],
                )
            )
        else:
            continuation_rids.append(None)
    prefix_results: list[dict[str, object]] = []
    for idx, payload in enumerate(prefix_payloads):
        stage_file.write_text(
            stage_file.read_text() + f"before_{phase_name}_prefix_cache_{idx}\n"
        )
        prefix_results.append(engine.generate(**_single_from_batched_payload(payload)))
        stage_file.write_text(
            stage_file.read_text() + f"after_{phase_name}_prefix_cache_{idx}\n"
        )

    prefix_meta = {
        "phase_name": phase_name,
        "session_ids": session_ids,
        "continuation_rids": continuation_rids,
        "results": prefix_results,
    }
    phase_path = output_dir / f"{phase_name}_prefix_cache_response.json"
    phase_path.write_text(json.dumps(prefix_meta, indent=2))
    if phase_name == "formal":
        (output_dir / "prefix_cache_response.json").write_text(
            json.dumps(prefix_meta, indent=2)
        )
    return prefix_meta


def _build_profiled_request(
    request: dict[str, object],
    prefix_meta: dict[str, object],
    *,
    decode_empty_input_continuation: bool = False,
) -> dict[str, object]:
    fresh_input_ids = request["fresh_input_ids"]
    sampling_params = copy.deepcopy(request["sampling_params"])
    rids = request["rid"]
    session_ids = prefix_meta["session_ids"]
    continuation_rids = prefix_meta["continuation_rids"]

    session_params = []
    for idx, _fresh_ids in enumerate(fresh_input_ids):
        entry = {"id": session_ids[idx]}
        if continuation_rids[idx] is not None:
            entry["rid"] = continuation_rids[idx]
        session_params.append(entry)

    payload = {
        "input_ids": (
            [[] for _ in fresh_input_ids]
            if decode_empty_input_continuation
            else fresh_input_ids
        ),
        "sampling_params": sampling_params,
        "return_logprob": False,
        "rid": rids,
        "session_params": session_params,
    }
    return payload


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    install_prefill_chunk_nvtx_patch()
    if args.profile_decode_only_stage:
        os.environ["AIC_PROFILE_DECODE_ONLY_STAGE"] = "1"
        install_decode_only_profile_patch()
    install_generate_req_batch_session_patch()

    request = json.loads(Path(args.request_json).read_text())
    profile_cfg = json.loads(Path(args.profile_json).read_text())
    _validate_request_schema(request)
    precheck_summary = _precheck_chunked_kv(
        request,
        args.context_length,
        args.tp_size,
        args.allow_chunked_kv,
        args.chunked_prefix_cache_threshold,
    )
    (output_dir / "chunked_kv_precheck.json").write_text(
        json.dumps(precheck_summary, indent=2)
    )

    server_arg_names = set(inspect.signature(ServerArgs).parameters)
    override_args = {
        "num_hidden_layers": args.num_layers,
    }

    engine_kwargs = {
        "model_path": args.model_path,
        "load_format": "dummy",
        "skip_tokenizer_init": True,
        "mem_fraction_static": args.mem_fraction_static,
        "context_length": args.context_length,
        "disable_cuda_graph": args.cuda_graph_mode != "on",
        "disable_radix_cache": False,
        "json_model_override_args": json.dumps(override_args),
        "log_level": "info",
        "piecewise_cuda_graph_compiler": "eager",
        "tp_size": 1,
    }
    if (
        args.limit_cuda_graph_bs_to_batch
        and args.cuda_graph_mode == "on"
        and "cuda_graph_bs" in server_arg_names
    ):
        engine_kwargs["cuda_graph_bs"] = [len(request["fresh_input_ids"])]
    if args.attention_backend not in ("", "auto", "default"):
        engine_kwargs["attention_backend"] = args.attention_backend
    if args.decode_attention_backend not in ("", "auto", "default"):
        engine_kwargs["decode_attention_backend"] = args.decode_attention_backend
    if "enable_layerwise_nvtx_marker" in server_arg_names:
        engine_kwargs["enable_layerwise_nvtx_marker"] = args.layerwise_marker
    if args.chunked_prefill_size is not None and "chunked_prefill_size" in server_arg_names:
        engine_kwargs["chunked_prefill_size"] = args.chunked_prefill_size
    if args.max_prefill_tokens is not None and "max_prefill_tokens" in server_arg_names:
        engine_kwargs["max_prefill_tokens"] = args.max_prefill_tokens
    if "enable_dynamic_chunking" in server_arg_names:
        engine_kwargs["enable_dynamic_chunking"] = False

    if args.pcg_mode == "on":
        if "enable_piecewise_cuda_graph" in server_arg_names:
            engine_kwargs["enable_piecewise_cuda_graph"] = True
        elif "disable_piecewise_cuda_graph" in server_arg_names:
            engine_kwargs["disable_piecewise_cuda_graph"] = False
            if "enforce_piecewise_cuda_graph" in server_arg_names:
                engine_kwargs["enforce_piecewise_cuda_graph"] = True
    else:
        if "enable_piecewise_cuda_graph" in server_arg_names:
            engine_kwargs["enable_piecewise_cuda_graph"] = False
        elif "disable_piecewise_cuda_graph" in server_arg_names:
            engine_kwargs["disable_piecewise_cuda_graph"] = True

    if args.tp_size > 1 and "disable_custom_all_reduce" in server_arg_names:
        engine_kwargs["disable_custom_all_reduce"] = True
    if (
        args.stream_profile_after_first_token or args.profile_decode_only_stage
    ) and "decode_log_interval" in server_arg_names:
        engine_kwargs["decode_log_interval"] = 1

    (output_dir / "engine_kwargs.json").write_text(
        json.dumps(engine_kwargs, indent=2, sort_keys=True)
    )
    stage_file = output_dir / "stage.txt"
    stage_file.write_text("before_engine_init\n")

    try:
        with _tp_load_model_patch(args.tp_size, output_dir):
            with sgl.Engine(**engine_kwargs) as engine:
                stage_file.write_text(stage_file.read_text() + "after_engine_init\n")

                warmup_results = []
                for warmup_idx in range(args.warmup_runs):
                    warmup_request_source = _select_phase_request(
                        request,
                        use_warmup_inputs=True,
                    )
                    warmup_prefix_meta = _run_prefix_cache_pass(
                        engine,
                        warmup_request_source,
                        output_dir,
                        stage_file,
                        phase_name=f"warmup_{warmup_idx}",
                    )
                    try:
                        warmup_request = _build_profiled_request(
                            warmup_request_source,
                            warmup_prefix_meta,
                            decode_empty_input_continuation=(
                                args.decode_empty_input_continuation
                            ),
                        )
                        (output_dir / f"warmup_{warmup_idx}_generate_request.json").write_text(
                            json.dumps(warmup_request, indent=2)
                        )
                        stage_file.write_text(
                            stage_file.read_text()
                            + f"before_warmup_generate_{warmup_idx}\n"
                        )
                        warmup_result = engine.generate(**warmup_request)
                        warmup_results.append(warmup_result)
                        stage_file.write_text(
                            stage_file.read_text()
                            + f"after_warmup_generate_{warmup_idx}\n"
                        )
                    finally:
                        _close_sessions(engine, warmup_prefix_meta["session_ids"])
                if warmup_results:
                    (output_dir / "warmup_generate_response.json").write_text(
                        json.dumps(warmup_results, indent=2)
                    )

                prefix_meta = _run_prefix_cache_pass(
                    engine,
                    request,
                    output_dir,
                    stage_file,
                    phase_name="formal",
                )
                profiled_request = _build_profiled_request(
                    request,
                    prefix_meta,
                    decode_empty_input_continuation=args.decode_empty_input_continuation,
                )
                (output_dir / "profiled_generate_request.json").write_text(
                    json.dumps(profiled_request, indent=2)
                )

                if args.stream_profile_after_first_token:
                    stream_request = dict(profiled_request)
                    stream_request["stream"] = True
                    (output_dir / "stream_generate_request.json").write_text(
                        json.dumps(stream_request, indent=2)
                    )
                    stage_file.write_text(
                        stage_file.read_text() + "before_stream_generate\n"
                    )
                    stream_iter = engine.generate(**stream_request)
                    stream_chunks = []
                    first_chunk = next(stream_iter)
                    stream_chunks.append(first_chunk)
                    (output_dir / "stream_first_chunk.json").write_text(
                        json.dumps(first_chunk, indent=2)
                    )
                    stage_file.write_text(
                        stage_file.read_text() + "after_first_stream_chunk\n"
                    )

                    stage_file.write_text(
                        stage_file.read_text() + "before_start_profile\n"
                    )
                    engine.start_profile(**profile_cfg)
                    (output_dir / "start_profile_response.txt").write_text(
                        "Start profiling via Engine API after first streamed token.\n"
                    )
                    stage_file.write_text(
                        stage_file.read_text() + "after_start_profile\n"
                    )

                    for chunk in stream_iter:
                        stream_chunks.append(chunk)
                    (output_dir / "generate_response.json").write_text(
                        json.dumps(stream_chunks, indent=2)
                    )
                    stage_file.write_text(
                        stage_file.read_text() + "after_stream_generate\n"
                    )

                    stage_file.write_text(
                        stage_file.read_text() + "before_stop_profile\n"
                    )
                    engine.stop_profile()
                    (output_dir / "stop_profile_response.txt").write_text(
                        "Stop profiling via Engine API after stream completion.\n"
                    )
                    stage_file.write_text(
                        stage_file.read_text() + "after_stop_profile\n"
                    )
                else:
                    stage_file.write_text(
                        stage_file.read_text() + "before_start_profile\n"
                    )
                    engine.start_profile(**profile_cfg)
                    (output_dir / "start_profile_response.txt").write_text(
                        "Start profiling via Engine API.\n"
                    )
                    stage_file.write_text(
                        stage_file.read_text() + "after_start_profile\n"
                    )

                    stage_file.write_text(stage_file.read_text() + "before_generate\n")
                    result = engine.generate(**profiled_request)
                    (output_dir / "generate_response.json").write_text(
                        json.dumps(result, indent=2)
                    )
                    stage_file.write_text(stage_file.read_text() + "after_generate\n")

                    if args.profile_decode_only_stage and profile_cfg.get(
                        "profile_by_stage"
                    ):
                        (output_dir / "stop_profile_response.txt").write_text(
                            "Skip explicit stop_profile; stage profiling stops itself.\n"
                        )
                    else:
                        stage_file.write_text(
                            stage_file.read_text() + "before_stop_profile\n"
                        )
                        engine.stop_profile()
                        (output_dir / "stop_profile_response.txt").write_text(
                            "Stop profiling via Engine API.\n"
                        )
                        stage_file.write_text(
                            stage_file.read_text() + "after_stop_profile\n"
                        )

                _close_sessions(engine, prefix_meta["session_ids"])

        (output_dir / "status.json").write_text(
            json.dumps(
                {
                    "status": "ok",
                    "cuda_graph_mode": args.cuda_graph_mode,
                    "pcg_mode": args.pcg_mode,
                    "layerwise_marker": args.layerwise_marker,
                    "warmup_runs": args.warmup_runs,
                    "tp_size": args.tp_size,
                    "allow_chunked_kv": args.allow_chunked_kv,
                    "decode_empty_input_continuation": args.decode_empty_input_continuation,
                    "profile_decode_only_stage": args.profile_decode_only_stage,
                    "stream_profile_after_first_token": args.stream_profile_after_first_token,
                },
                indent=2,
            )
        )
        return 0
    except Exception:
        (output_dir / "engine_exception.txt").write_text(traceback.format_exc())
        (output_dir / "status.json").write_text(
            json.dumps(
                {
                    "status": "fail",
                    "cuda_graph_mode": args.cuda_graph_mode,
                    "pcg_mode": args.pcg_mode,
                    "layerwise_marker": args.layerwise_marker,
                    "warmup_runs": args.warmup_runs,
                    "tp_size": args.tp_size,
                    "allow_chunked_kv": args.allow_chunked_kv,
                    "decode_empty_input_continuation": args.decode_empty_input_continuation,
                    "profile_decode_only_stage": args.profile_decode_only_stage,
                    "stream_profile_after_first_token": args.stream_profile_after_first_token,
                },
                indent=2,
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
