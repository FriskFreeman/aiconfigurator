#!/usr/bin/env python3
import argparse
import asyncio
import copy
import inspect
import json
import os
import time
import traceback
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


def _safe_len(value) -> int:
    try:
        return len(value)
    except TypeError:
        return 0


def _to_jsonable_list(value) -> list:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    try:
        return list(value)
    except TypeError:
        return []


def _forward_mode_name(batch) -> str | None:
    forward_mode = getattr(batch, "forward_mode", None)
    return getattr(forward_mode, "name", str(forward_mode) if forward_mode is not None else None)


def _is_mixed_batch(batch) -> bool:
    forward_mode = getattr(batch, "forward_mode", None)
    return bool(
        forward_mode is not None
        and hasattr(forward_mode, "is_mixed")
        and forward_mode.is_mixed()
    )


def _build_mixed_batch_payload(scheduler, batch) -> dict[str, object]:
    reqs = list(getattr(batch, "reqs", []) or [])
    decoding_reqs = list(getattr(batch, "decoding_reqs", []) or [])
    decode_ids = {id(req) for req in decoding_reqs}
    prefill_count = max(0, len(reqs) - len(decoding_reqs))

    prefix_lens = _to_jsonable_list(getattr(batch, "prefix_lens", None))
    if not prefix_lens:
        prefix_lens = [_safe_len(getattr(req, "prefix_indices", [])) for req in reqs]
    extend_lens = _to_jsonable_list(getattr(batch, "extend_lens", None))
    if not extend_lens:
        extend_lens = [getattr(req, "extend_input_len", None) for req in reqs]
    seq_lens = _to_jsonable_list(getattr(batch, "seq_lens_cpu", None))
    if not seq_lens:
        seq_lens = [_safe_len(getattr(req, "fill_ids", [])) for req in reqs]

    prefill_items = []
    decode_items = []
    for idx, req in enumerate(reqs):
        item = {
            "rid": getattr(req, "rid", None),
            "req_index_in_batch": idx,
            "prefix_len": int(prefix_lens[idx]) if idx < len(prefix_lens) and prefix_lens[idx] is not None else None,
            "extend_len": int(extend_lens[idx]) if idx < len(extend_lens) and extend_lens[idx] is not None else None,
            "seq_len_after": int(seq_lens[idx]) if idx < len(seq_lens) and seq_lens[idx] is not None else None,
            "origin_input_len": _safe_len(getattr(req, "origin_input_ids", [])),
            "output_len": _safe_len(getattr(req, "output_ids", [])),
            "is_chunked": getattr(req, "is_chunked", None),
        }
        if id(req) in decode_ids or idx >= prefill_count:
            decode_items.append(item)
        else:
            prefill_items.append(item)

    chunked_req = getattr(batch, "chunked_req", None)
    if chunked_req is None:
        chunked_req = getattr(scheduler, "chunked_req", None)

    payload = {
        "kind": "mixed_batch",
        "forward_ct": getattr(scheduler, "forward_ct", None),
        "forward_mode": _forward_mode_name(batch),
        "batch_size": len(reqs),
        "prefill": {
            "req_count": len(prefill_items),
            "prefix_lens": [item["prefix_len"] for item in prefill_items],
            "fresh_lens": [item["extend_len"] for item in prefill_items],
            "seq_lens_after": [item["seq_len_after"] for item in prefill_items],
            "sum_fresh_tokens": sum(item["extend_len"] or 0 for item in prefill_items),
            "sum_prefill_kv_tokens": sum(item["prefix_len"] or 0 for item in prefill_items),
            "items": prefill_items,
        },
        "decode": {
            "req_count": len(decode_items),
            "kv_lens": [item["prefix_len"] for item in decode_items],
            "extend_lens": [item["extend_len"] for item in decode_items],
            "seq_lens_after": [item["seq_len_after"] for item in decode_items],
            "output_lens": [item["output_len"] for item in decode_items],
            "sum_decode_kv_tokens": sum(item["prefix_len"] or 0 for item in decode_items),
            "items": decode_items,
        },
        "scheduler": {
            "chunked_prefill_size": getattr(scheduler, "chunked_prefill_size", None),
            "max_prefill_tokens": getattr(scheduler, "max_prefill_tokens", None),
            "enable_mixed_chunk": getattr(scheduler, "is_mixed_chunk", None),
            "chunked_req_rid": getattr(chunked_req, "rid", None) if chunked_req is not None else None,
            "running_batch_size_before_prefill": getattr(scheduler, "running_bs", None),
        },
    }
    return payload


def install_mixed_batch_nvtx_and_profile_patch() -> None:
    from sglang.srt.managers.scheduler import Scheduler

    if getattr(Scheduler.run_batch, "_aic_mixed_batch_patch", False):
        return

    original_profile_predicate = Scheduler._profile_batch_predicate
    original_run_batch = Scheduler.run_batch

    def _enabled() -> bool:
        return os.getenv("AIC_PROFILE_MIXED_ONLY_STAGE") == "1"

    def patched_profile_predicate(self, batch):
        if _enabled() and getattr(self, "profile_by_stage", False):
            return
        return original_profile_predicate(self, batch)

    def _append_payload(payload: dict[str, object]) -> None:
        meta_path = os.getenv("AIC_MIXED_META_PATH")
        if not meta_path:
            return
        with open(meta_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _matches_profile_target(payload: dict[str, object] | None) -> bool:
        target_prefix = os.getenv("AIC_PROFILE_PREFILL_RID_PREFIX")
        if not target_prefix or payload is None:
            return True
        prefill = payload.get("prefill")
        if not isinstance(prefill, dict):
            return False
        items = prefill.get("items")
        if not isinstance(items, list):
            return False
        for item in items:
            if isinstance(item, dict) and str(item.get("rid", "")).startswith(target_prefix):
                return True
        return False

    def patched_run_batch(self, batch, *args, **kwargs):
        is_mixed = _is_mixed_batch(batch)
        pushed = False
        payload = None
        if is_mixed:
            payload = _build_mixed_batch_payload(self, batch)
            _append_payload(payload)
            try:
                import torch

                torch.cuda.nvtx.range_push(repr({"AICMixedBatch": payload}))
                pushed = True
            except Exception:
                pushed = False

        should_wrap_profile = bool(
            _enabled()
            and getattr(self, "profile_by_stage", False)
            and is_mixed
            and _matches_profile_target(payload)
        )
        if should_wrap_profile:
            target = getattr(self, "profiler_target_prefill_ct", None) or 1
            captured = getattr(self, "_aic_profiled_mixed_ct", 0)
            if captured >= target:
                should_wrap_profile = False
            else:
                if not getattr(self, "profile_in_progress", False):
                    self.start_profile(getattr(batch, "forward_mode", None))
                self._aic_profiled_mixed_ct = captured + 1

        try:
            return original_run_batch(self, batch, *args, **kwargs)
        finally:
            if should_wrap_profile:
                target = getattr(self, "profiler_target_prefill_ct", None) or 1
                if (
                    getattr(self, "_aic_profiled_mixed_ct", 0) >= target
                    and getattr(self, "profile_in_progress", False)
                ):
                    self.stop_profile(stage=getattr(batch, "forward_mode", None))
            if pushed:
                try:
                    import torch

                    torch.cuda.nvtx.range_pop()
                except Exception:
                    pass

    patched_profile_predicate._aic_mixed_batch_patch = True
    patched_run_batch._aic_mixed_batch_patch = True
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
    parser.add_argument("--chunked-prefill-size", type=int, default=8192)
    parser.add_argument("--max-prefill-tokens", type=int, default=16384)
    parser.add_argument("--mixed-profile-steps", type=int, default=1)
    parser.add_argument("--decode-consume-delay-s", type=float, default=0.0)
    parser.add_argument("--decode-start-timeout-s", type=float, default=120.0)
    parser.add_argument("--decode-finish-timeout-s", type=float, default=30.0)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--chunked-prefix-cache-threshold", type=int, default=8192)
    return parser.parse_args()


def _validate_request_schema(request: dict[str, object]) -> None:
    required = [
        "decode_input_ids",
        "prefill_prefix_input_ids",
        "prefill_fresh_input_ids",
        "decode_sampling_params",
        "prefill_sampling_params",
    ]
    for key in required:
        if key not in request:
            raise ValueError(f"request-json missing {key}")
    if not isinstance(request["decode_input_ids"], list) or not request["decode_input_ids"]:
        raise ValueError("decode_input_ids must be a non-empty list")
    if not isinstance(request["prefill_fresh_input_ids"], list) or not request["prefill_fresh_input_ids"]:
        raise ValueError("prefill_fresh_input_ids must be a non-empty list")
    if not isinstance(request["prefill_prefix_input_ids"], list):
        raise ValueError("prefill_prefix_input_ids must be a list")
    if len(request["prefill_prefix_input_ids"]) != len(request["prefill_fresh_input_ids"]):
        raise ValueError("prefill prefix/fresh batch sizes differ")


def _request_lengths(rows: list[list[int]]) -> list[int]:
    return [len(row) for row in rows]


def _shift_token_rows(rows: list[list[int]], delta: int, vocab_size: int) -> list[list[int]]:
    usable_vocab = max(1, vocab_size - 10)
    return [[10 + ((int(token) - 10 + delta) % usable_vocab) for token in row] for row in rows]


def _build_warmup_request(request: dict[str, object], warmup_idx: int) -> dict[str, object]:
    warmup_request = copy.deepcopy(request)
    meta = warmup_request.get("meta")
    vocab_size = int(meta.get("vocab_size", 129280)) if isinstance(meta, dict) else 129280
    delta = 31_337 * (warmup_idx + 1)
    for key in ["decode_input_ids", "prefill_prefix_input_ids", "prefill_fresh_input_ids"]:
        warmup_request[key] = _shift_token_rows(warmup_request[key], delta, vocab_size)
    return warmup_request


def _build_generate_payload(
    *,
    input_ids: list[list[int]],
    sampling_params: dict[str, object],
    rids: list[str],
    stream: bool = False,
    session_id: str | None = None,
    session_rids: list[str | None] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "input_ids": input_ids,
        "sampling_params": sampling_params,
        "return_logprob": False,
        "rid": rids,
        "stream": stream,
    }
    if session_id is not None:
        if session_rids is None or len(session_rids) != len(input_ids):
            raise ValueError("session_rids must match input batch size")
        payload["session_params"] = [
            {"id": session_id, **({"rid": rid} if rid is not None else {})}
            for rid in session_rids
        ]
    return payload


def _single_from_batched_payload(payload: dict[str, object]) -> dict[str, object]:
    single = {
        "input_ids": payload["input_ids"][0],
        "sampling_params": payload["sampling_params"],
        "return_logprob": payload.get("return_logprob", False),
        "rid": payload["rid"][0],
        "stream": payload.get("stream", False),
    }
    session_params = payload.get("session_params")
    if isinstance(session_params, list) and session_params:
        single["session_params"] = session_params[0]
    return single


def _close_sessions(engine, session_ids: list[str]) -> None:
    for session_id in session_ids:
        try:
            engine.close_session(session_id)
        except Exception:
            pass


def _run_prefill_prefix_cache_pass(engine, request: dict[str, object], output_dir: Path, phase_name: str) -> dict[str, object]:
    prefix_rows = request["prefill_prefix_input_ids"]
    fresh_rows = request["prefill_fresh_input_ids"]
    session_ids: list[str] = []
    continuation_rids: list[str | None] = []
    results = []
    for idx, prefix_ids in enumerate(prefix_rows):
        session_id = engine.open_session(capacity_of_str_len=len(prefix_ids) + len(fresh_rows[idx]) + 32)
        session_ids.append(session_id)
        if prefix_ids:
            rid = f"{phase_name}-prefill-prefix-{idx}"
            payload = _build_generate_payload(
                input_ids=[prefix_ids],
                sampling_params={"temperature": 0.0, "max_new_tokens": 0},
                rids=[rid],
                session_id=session_id,
                session_rids=[None],
            )
            results.append(engine.generate(**_single_from_batched_payload(payload)))
            continuation_rids.append(rid)
        else:
            continuation_rids.append(None)
    meta = {"session_ids": session_ids, "continuation_rids": continuation_rids, "results": results}
    (output_dir / f"{phase_name}_prefill_prefix_cache_response.json").write_text(json.dumps(meta, indent=2))
    return meta


def _build_profiled_prefill_payload(request: dict[str, object], prefix_meta: dict[str, object], phase_name: str) -> dict[str, object]:
    fresh_rows = request["prefill_fresh_input_ids"]
    session_ids = prefix_meta["session_ids"]
    continuation_rids = prefix_meta["continuation_rids"]
    return {
        "input_ids": fresh_rows,
        "sampling_params": copy.deepcopy(request["prefill_sampling_params"]),
        "return_logprob": False,
        "rid": [f"{phase_name}-prefill-{i}" for i in range(len(fresh_rows))],
        "stream": False,
        "session_params": [
            {"id": session_ids[i], **({"rid": continuation_rids[i]} if continuation_rids[i] is not None else {})}
            for i in range(len(fresh_rows))
        ],
    }


async def _run_mixed_workload(
    engine,
    request: dict[str, object],
    profile_cfg: dict[str, object],
    output_dir: Path,
    args: argparse.Namespace,
    phase_name: str,
    prefix_meta: dict[str, object],
    profile_enabled: bool,
) -> dict[str, object]:
    decode_rows = request["decode_input_ids"]
    decode_sampling = copy.deepcopy(request["decode_sampling_params"])
    decode_sampling.setdefault("ignore_eos", True)

    first_chunks: dict[int, object] = {}
    decode_results: dict[int, list[object]] = {}
    first_ready = asyncio.Event()

    async def consume_decode(idx: int, input_ids: list[int]):
        chunks: list[object] = []
        agen = await engine.async_generate(
            input_ids=input_ids,
            sampling_params=decode_sampling,
            rid=f"{phase_name}-decode-{idx}",
            stream=True,
        )
        async for chunk in agen:
            chunks.append(chunk)
            if idx not in first_chunks:
                first_chunks[idx] = chunk
                if len(first_chunks) >= len(decode_rows):
                    first_ready.set()
            if args.decode_consume_delay_s > 0:
                await asyncio.sleep(args.decode_consume_delay_s)
        decode_results[idx] = chunks

    tasks = [asyncio.create_task(consume_decode(i, row)) for i, row in enumerate(decode_rows)]
    await asyncio.wait_for(first_ready.wait(), timeout=args.decode_start_timeout_s)
    (output_dir / f"{phase_name}_decode_first_chunks.json").write_text(json.dumps(first_chunks, indent=2, default=str))

    prefill_payload = _build_profiled_prefill_payload(request, prefix_meta, phase_name)
    (output_dir / f"{phase_name}_prefill_generate_request.json").write_text(json.dumps(prefill_payload, indent=2))

    if profile_enabled:
        await engine.tokenizer_manager.start_profile(**profile_cfg)
        (output_dir / f"{phase_name}_start_profile_response.txt").write_text(
            "Configured mixed-only profiling via tokenizer_manager.start_profile.\n"
        )
    else:
        (output_dir / f"{phase_name}_start_profile_response.txt").write_text(
            "Mixed warmup ran without tokenizer_manager.start_profile.\n"
        )
    prefill_result = await engine.async_generate(**prefill_payload)
    (output_dir / f"{phase_name}_prefill_generate_response.json").write_text(json.dumps(prefill_result, indent=2))

    try:
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=args.decode_finish_timeout_s)
    except asyncio.TimeoutError:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        pass

    ordered_decode_results = [decode_results.get(i, []) for i in range(len(decode_rows))]
    (output_dir / f"{phase_name}_decode_stream_chunks.json").write_text(json.dumps(ordered_decode_results, indent=2, default=str))
    return {
        "phase_name": phase_name,
        "decode_first_chunks": first_chunks,
        "prefill_result": prefill_result,
        "decode_result_lengths": [len(chunks) for chunks in ordered_decode_results],
    }


def _tp_load_model_patch(tp_size: int):
    from contextlib import contextmanager

    @contextmanager
    def _noop():
        yield

    if tp_size <= 1:
        return _noop()

    import sglang.srt.distributed.parallel_state as ps
    from sglang.srt.model_executor.model_runner import ModelRunner

    @contextmanager
    def _patch():
        original_load = ModelRunner.load_model

        def _patch_group(group):
            if group is None:
                return None
            state = {"world_size": getattr(group, "world_size", None), "rank_in_group": getattr(group, "rank_in_group", None)}
            group.world_size = tp_size
            group.rank_in_group = 0
            return state

        def _restore_group(group, state):
            if group is None or state is None:
                return
            group.world_size = state["world_size"]
            group.rank_in_group = state["rank_in_group"]

        def patched_load(self):
            tp_state = _patch_group(ps._TP)
            attn_state = _patch_group(getattr(ps, "_ATTN_TP", None))
            try:
                return original_load(self)
            finally:
                _restore_group(getattr(ps, "_ATTN_TP", None), attn_state)
                _restore_group(ps._TP, tp_state)

        ModelRunner.load_model = patched_load
        try:
            yield
        finally:
            ModelRunner.load_model = original_load

    return _patch()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["AIC_PROFILE_MIXED_ONLY_STAGE"] = "1"
    os.environ["AIC_MIXED_META_PATH"] = str(output_dir / "mixed_batch_meta.jsonl")
    install_generate_req_batch_session_patch()
    install_mixed_batch_nvtx_and_profile_patch()

    request = json.loads(Path(args.request_json).read_text())
    profile_cfg = json.loads(Path(args.profile_json).read_text())
    profile_cfg["profile_by_stage"] = True
    profile_cfg["num_steps"] = args.mixed_profile_steps
    _validate_request_schema(request)

    server_arg_names = set(inspect.signature(ServerArgs).parameters)
    engine_kwargs = {
        "model_path": args.model_path,
        "load_format": "dummy",
        "skip_tokenizer_init": True,
        "mem_fraction_static": args.mem_fraction_static,
        "context_length": args.context_length,
        "disable_cuda_graph": args.cuda_graph_mode != "on",
        "disable_radix_cache": False,
        "json_model_override_args": json.dumps({"num_hidden_layers": args.num_layers}),
        "log_level": "info",
        "piecewise_cuda_graph_compiler": "eager",
        "tp_size": 1,
    }
    if args.attention_backend not in ("", "auto", "default"):
        engine_kwargs["attention_backend"] = args.attention_backend
    if args.decode_attention_backend not in ("", "auto", "default"):
        engine_kwargs["decode_attention_backend"] = args.decode_attention_backend
    if "enable_layerwise_nvtx_marker" in server_arg_names:
        engine_kwargs["enable_layerwise_nvtx_marker"] = args.layerwise_marker
    if "chunked_prefill_size" in server_arg_names:
        engine_kwargs["chunked_prefill_size"] = args.chunked_prefill_size
    if "max_prefill_tokens" in server_arg_names:
        engine_kwargs["max_prefill_tokens"] = args.max_prefill_tokens
    if "enable_mixed_chunk" in server_arg_names:
        engine_kwargs["enable_mixed_chunk"] = True
    if "enable_dynamic_chunking" in server_arg_names:
        engine_kwargs["enable_dynamic_chunking"] = False
    if args.pcg_mode == "on":
        if "disable_piecewise_cuda_graph" in server_arg_names:
            engine_kwargs["disable_piecewise_cuda_graph"] = False
    else:
        if "disable_piecewise_cuda_graph" in server_arg_names:
            engine_kwargs["disable_piecewise_cuda_graph"] = True

    (output_dir / "engine_kwargs.json").write_text(json.dumps(engine_kwargs, indent=2, sort_keys=True))
    (output_dir / "run_request_summary.json").write_text(
        json.dumps(
            {
                "decode_lens": _request_lengths(request["decode_input_ids"]),
                "prefill_prefix_lens": _request_lengths(request["prefill_prefix_input_ids"]),
                "prefill_fresh_lens": _request_lengths(request["prefill_fresh_input_ids"]),
                "chunked_prefill_size": args.chunked_prefill_size,
                "max_prefill_tokens": args.max_prefill_tokens,
                "mixed_profile_steps": args.mixed_profile_steps,
            },
            indent=2,
        )
    )
    stage_file = output_dir / "stage.txt"
    stage_file.write_text("before_engine_init\n")

    try:
        with _tp_load_model_patch(args.tp_size):
            with sgl.Engine(**engine_kwargs) as engine:
                stage_file.write_text(stage_file.read_text() + "after_engine_init\n")
                mixed_warmup_results = []
                for warmup_idx in range(args.warmup_runs):
                    phase_name = f"warmup{warmup_idx}"
                    warmup_request = _build_warmup_request(request, warmup_idx)
                    stage_file.write_text(stage_file.read_text() + f"before_mixed_warmup_{warmup_idx}\n")
                    warmup_prefix_meta = _run_prefill_prefix_cache_pass(engine, warmup_request, output_dir, phase_name)
                    try:
                        mixed_warmup_results.append(
                            engine.loop.run_until_complete(
                                _run_mixed_workload(
                                    engine,
                                    warmup_request,
                                    profile_cfg,
                                    output_dir,
                                    args,
                                    phase_name,
                                    warmup_prefix_meta,
                                    profile_enabled=False,
                                )
                            )
                        )
                    finally:
                        _close_sessions(engine, warmup_prefix_meta["session_ids"])
                    stage_file.write_text(stage_file.read_text() + f"after_mixed_warmup_{warmup_idx}\n")
                if mixed_warmup_results:
                    (output_dir / "mixed_warmup_response.json").write_text(
                        json.dumps(mixed_warmup_results, indent=2, default=str)
                    )

                prefix_meta = _run_prefill_prefix_cache_pass(engine, request, output_dir, "formal")
                stage_file.write_text(stage_file.read_text() + "before_formal_mixed\n")
                try:
                    workload_result = engine.loop.run_until_complete(
                        _run_mixed_workload(
                            engine,
                            request,
                            profile_cfg,
                            output_dir,
                            args,
                            "formal",
                            prefix_meta,
                            profile_enabled=True,
                        )
                    )
                    (output_dir / "generate_response.json").write_text(json.dumps(workload_result, indent=2, default=str))
                    stage_file.write_text(stage_file.read_text() + "after_formal_mixed\n")
                finally:
                    _close_sessions(engine, prefix_meta["session_ids"])

        mixed_meta_path = output_dir / "mixed_batch_meta.jsonl"
        mixed_count = 0
        if mixed_meta_path.exists():
            mixed_count = sum(1 for line in mixed_meta_path.read_text().splitlines() if line.strip())
        status = "ok" if mixed_count > 0 else "no_mixed_batch"
        (output_dir / "status.json").write_text(
            json.dumps({"status": status, "mixed_batch_count": mixed_count}, indent=2)
        )
        return 0 if mixed_count > 0 else 2
    except Exception:
        (output_dir / "engine_exception.txt").write_text(traceback.format_exc())
        (output_dir / "status.json").write_text(json.dumps({"status": "fail"}, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
