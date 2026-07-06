#!/usr/bin/env python3
import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="booleimg.myaddr.io/lmsysorg/sglang:v0.5.9")
    parser.add_argument("--gpu-id", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=5)
    parser.add_argument("--attention-backend", default="auto")
    parser.add_argument("--decode-attention-backend", default="auto")
    parser.add_argument("--mem-fraction-static", type=float, default=0.5)
    parser.add_argument("--cuda-graph-mode", choices=["off", "on"], default="off")
    parser.add_argument("--pcg-mode", choices=["off", "on"], default="off")
    parser.add_argument("--profile-mode", choices=["chrome", "nsys"], default="nsys")
    parser.add_argument("--layerwise-marker", action="store_true", default=True)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--deepgemm-precompile", action="store_true")
    parser.add_argument("--no-deepgemm-fast-warmup", action="store_true")
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument("--chunked-prefill-size", type=int, default=8192)
    parser.add_argument("--max-prefill-tokens", type=int, default=16384)
    parser.add_argument("--mixed-profile-steps", type=int, default=1)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--tag", default="smoke")
    parser.add_argument("--run-root", default=".self/task-localbench/formal-agg-stage2")

    parser.add_argument("--decode-batch-size", type=int, required=True)
    parser.add_argument("--decode-prefix-len", type=int)
    parser.add_argument("--decode-prefix-lens-csv")
    parser.add_argument("--decode-max-new-tokens", type=int, default=16)
    parser.add_argument("--decode-consume-delay-s", type=float, default=0.0)
    parser.add_argument("--decode-start-timeout-s", type=float, default=120.0)
    parser.add_argument("--decode-finish-timeout-s", type=float, default=30.0)

    parser.add_argument("--prefill-batch-size", type=int, required=True)
    parser.add_argument("--prefill-fresh-len", type=int)
    parser.add_argument("--prefill-fresh-lens-csv")
    parser.add_argument("--prefill-prefix-len", type=int, default=0)
    parser.add_argument("--prefill-prefix-lens-csv")
    parser.add_argument("--prefill-max-new-tokens", type=int, default=1)

    parser.add_argument("--nsys-sample", choices=["process-tree", "system-wide", "none"], default="none")
    parser.add_argument("--nsys-cpuctxsw", choices=["process-tree", "system-wide", "none"], default="none")
    parser.add_argument("--extra-env", action="append", default=[], metavar="KEY=VALUE")
    return parser.parse_args()


def _parse_csv_ints(raw: str | None) -> list[int] | None:
    if raw is None:
        return None
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values or any(value < 0 for value in values):
        raise ValueError("CSV lists must contain non-negative integers")
    return values


def _resolve_equal_or_csv(*, csv_raw: str | None, scalar: int | None, batch_size: int, name: str, positive: bool) -> list[int]:
    values = _parse_csv_ints(csv_raw)
    if values is None:
        if scalar is None:
            raise ValueError(f"{name}: either scalar or csv is required")
        values = [scalar for _ in range(batch_size)]
    if len(values) != batch_size:
        raise ValueError(f"{name}: length {len(values)} does not match batch_size {batch_size}")
    if positive and any(value <= 0 for value in values):
        raise ValueError(f"{name}: values must be positive")
    if not positive and any(value < 0 for value in values):
        raise ValueError(f"{name}: values must be non-negative")
    return values


def _load_vocab_size(root_dir: Path) -> int:
    cfg = json.loads((root_dir / "src/aiconfigurator/model_configs/deepseek-ai--DeepSeek-V3_config.json").read_text())
    return int(cfg["vocab_size"])


def _build_token_rows(lengths: list[int], base_seed: int, vocab_size: int) -> list[list[int]]:
    rows = []
    cursor = base_seed
    usable_vocab = vocab_size - 10
    for length in lengths:
        rows.append([10 + ((cursor + offset) % usable_vocab) for offset in range(length)])
        cursor += length + 17
    return rows


def build_local_model(root_dir: Path, local_model_dir: Path, tp_size: int) -> None:
    src_cfg = root_dir / "src/aiconfigurator/model_configs/deepseek-ai--DeepSeek-V3_config.json"
    cfg = json.loads(src_cfg.read_text())
    cfg["model_type"] = "deepseek_v3"
    cfg["architectures"] = ["DeepseekV3ForCausalLM"]
    cfg.pop("auto_map", None)
    if tp_size > 1:
        for key in ["num_attention_heads", "num_key_value_heads"]:
            value = int(cfg[key])
            if value % tp_size != 0:
                raise ValueError(f"{key}={value} is not divisible by tp_size={tp_size}")
            cfg[key] = value // tp_size
        cfg["_aic_tp_shape_only_rank0"] = {
            "tp_size": tp_size,
            "note": "Single-card TP shape-only override; no TP communication.",
        }
    local_model_dir.mkdir(parents=True, exist_ok=True)
    (local_model_dir / "config.json").write_text(json.dumps(cfg, indent=2))


def _lens_name(prefix: str, values: list[int], scalar: int | None, csv_raw: str | None) -> str:
    if csv_raw is None and scalar is not None and len(set(values)) == 1:
        return f"{prefix}{scalar}"
    if len(values) > 6:
        return f"{prefix}var{len(values)}_sum{sum(values)}_min{min(values)}_max{max(values)}"
    return f"{prefix}var" + "-".join(str(v) for v in values)


def build_request(path: Path, args: argparse.Namespace, vocab_size: int) -> dict[str, object]:
    decode_lens = _resolve_equal_or_csv(
        csv_raw=args.decode_prefix_lens_csv,
        scalar=args.decode_prefix_len,
        batch_size=args.decode_batch_size,
        name="decode_prefix_lens",
        positive=True,
    )
    prefill_fresh_lens = _resolve_equal_or_csv(
        csv_raw=args.prefill_fresh_lens_csv,
        scalar=args.prefill_fresh_len,
        batch_size=args.prefill_batch_size,
        name="prefill_fresh_lens",
        positive=True,
    )
    prefill_prefix_lens = _resolve_equal_or_csv(
        csv_raw=args.prefill_prefix_lens_csv,
        scalar=args.prefill_prefix_len,
        batch_size=args.prefill_batch_size,
        name="prefill_prefix_lens",
        positive=False,
    )

    request = {
        "decode_input_ids": _build_token_rows(decode_lens, 1_000, vocab_size),
        "prefill_prefix_input_ids": _build_token_rows(prefill_prefix_lens, 100_000, vocab_size),
        "prefill_fresh_input_ids": _build_token_rows(prefill_fresh_lens, 200_000, vocab_size),
        "decode_sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": args.decode_max_new_tokens,
            "ignore_eos": True,
        },
        "prefill_sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": args.prefill_max_new_tokens,
        },
        "meta": {
            "decode_prefix_lens": decode_lens,
            "prefill_prefix_lens": prefill_prefix_lens,
            "prefill_fresh_lens": prefill_fresh_lens,
            "vocab_size": vocab_size,
        },
    }
    path.write_text(json.dumps(request, indent=2))
    return request


def build_profile(path: Path, profile_mode: str, mixed_profile_steps: int) -> None:
    path.write_text(
        json.dumps(
            {
                "output_dir": "/out/profile",
                "activities": ["CUDA_PROFILER"] if profile_mode == "nsys" else ["CPU", "GPU"],
                "with_stack": False,
                "record_shapes": True,
                "profile_by_stage": True,
                "num_steps": mixed_profile_steps,
                "profile_prefix": "deepseek-v3-agg-stage2",
            },
            indent=2,
        )
    )


def write_sitecustomize_patch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        r'''
import json
import os


def _safe_len(value):
    try:
        return len(value)
    except TypeError:
        return 0


def _to_list(value):
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    try:
        return list(value)
    except TypeError:
        return []


def _is_mixed_batch(batch):
    forward_mode = getattr(batch, "forward_mode", None)
    return bool(
        forward_mode is not None
        and hasattr(forward_mode, "is_mixed")
        and forward_mode.is_mixed()
    )


def _build_payload(scheduler, batch):
    reqs = list(getattr(batch, "reqs", []) or [])
    decoding_reqs = list(getattr(batch, "decoding_reqs", []) or [])
    decode_ids = {id(req) for req in decoding_reqs}
    prefill_count = max(0, len(reqs) - len(decoding_reqs))
    prefix_lens = _to_list(getattr(batch, "prefix_lens", None)) or [
        _safe_len(getattr(req, "prefix_indices", [])) for req in reqs
    ]
    extend_lens = _to_list(getattr(batch, "extend_lens", None)) or [
        getattr(req, "extend_input_len", None) for req in reqs
    ]
    seq_lens = _to_list(getattr(batch, "seq_lens_cpu", None)) or [
        _safe_len(getattr(req, "fill_ids", [])) for req in reqs
    ]

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

    chunked_req = getattr(batch, "chunked_req", None) or getattr(scheduler, "chunked_req", None)
    return {
        "kind": "mixed_batch",
        "forward_ct": getattr(scheduler, "forward_ct", None),
        "forward_mode": getattr(getattr(batch, "forward_mode", None), "name", None),
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


def _install_aic_mixed_patch():
    try:
        from sglang.srt.managers.scheduler import Scheduler
    except Exception:
        return
    if getattr(Scheduler.run_batch, "_aic_site_mixed_batch_patch", False):
        return

    original_profile_predicate = Scheduler._profile_batch_predicate
    original_run_batch = Scheduler.run_batch

    def _enabled():
        return os.getenv("AIC_PROFILE_MIXED_ONLY_STAGE") == "1"

    def patched_profile_predicate(self, batch):
        if _enabled() and getattr(self, "profile_by_stage", False):
            return
        return original_profile_predicate(self, batch)

    def _append_payload(payload):
        meta_path = os.getenv("AIC_MIXED_META_PATH")
        if not meta_path:
            return
        with open(meta_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _matches_profile_target(payload):
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
            payload = _build_payload(self, batch)
            _append_payload(payload)
            try:
                import torch

                torch.cuda.nvtx.range_push(repr({"AICMixedBatch": payload}))
                pushed = True
            except Exception:
                pushed = False

        wrap = bool(
            _enabled()
            and getattr(self, "profile_by_stage", False)
            and is_mixed
            and _matches_profile_target(payload)
        )
        if wrap:
            target = getattr(self, "profiler_target_prefill_ct", None) or 1
            captured = getattr(self, "_aic_profiled_mixed_ct", 0)
            if captured >= target:
                wrap = False
            else:
                if not getattr(self, "profile_in_progress", False):
                    self.start_profile(getattr(batch, "forward_mode", None))
                self._aic_profiled_mixed_ct = captured + 1
        try:
            return original_run_batch(self, batch, *args, **kwargs)
        finally:
            if wrap:
                target = getattr(self, "profiler_target_prefill_ct", None) or 1
                if getattr(self, "_aic_profiled_mixed_ct", 0) >= target and getattr(self, "profile_in_progress", False):
                    self.stop_profile(stage=getattr(batch, "forward_mode", None))
            if pushed:
                try:
                    import torch

                    torch.cuda.nvtx.range_pop()
                except Exception:
                    pass

    patched_profile_predicate._aic_site_mixed_batch_patch = True
    patched_run_batch._aic_site_mixed_batch_patch = True
    Scheduler._profile_batch_predicate = patched_profile_predicate
    Scheduler.run_batch = patched_run_batch


_install_aic_mixed_patch()
'''.lstrip(),
        encoding="utf-8",
    )


def run_export_if_needed(root_dir: Path, run_dir: Path, image: str) -> None:
    export_script = root_dir / ".self/task-localbench/formal-prefill-stage1/export_nsys_profile.py"
    if not (run_dir / "nsys" / "report.nsys-rep").exists():
        return
    completed = subprocess.run(
        ["python", str(export_script), "--run-dir", str(run_dir), "--image", image],
        text=True,
        capture_output=True,
    )
    (run_dir / "nsys_export.stdout.log").write_text(completed.stdout)
    (run_dir / "nsys_export.stderr.log").write_text(completed.stderr)
    if completed.returncode != 0:
        raise RuntimeError(f"nsys export failed for {run_dir}")


def main() -> int:
    args = parse_args()
    root_dir = Path(__file__).resolve().parents[3]
    vocab_size = _load_vocab_size(root_dir)
    run_root = root_dir / args.run_root
    if run_root.name != "output":
        run_root = run_root / "output"
    run_root.mkdir(parents=True, exist_ok=True)

    prefill_fresh_values = _resolve_equal_or_csv(
        csv_raw=args.prefill_fresh_lens_csv,
        scalar=args.prefill_fresh_len,
        batch_size=args.prefill_batch_size,
        name="prefill_fresh_lens",
        positive=True,
    )
    prefill_prefix_values = _resolve_equal_or_csv(
        csv_raw=args.prefill_prefix_lens_csv,
        scalar=args.prefill_prefix_len,
        batch_size=args.prefill_batch_size,
        name="prefill_prefix_lens",
        positive=False,
    )
    decode_values = _resolve_equal_or_csv(
        csv_raw=args.decode_prefix_lens_csv,
        scalar=args.decode_prefix_len,
        batch_size=args.decode_batch_size,
        name="decode_prefix_lens",
        positive=True,
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = (
        f"{stamp}_agg_stage2_{args.tag}_"
        f"d{args.decode_batch_size}_{_lens_name('dctx', decode_values, args.decode_prefix_len, args.decode_prefix_lens_csv)}_"
        f"p{args.prefill_batch_size}_{_lens_name('pf', prefill_fresh_values, args.prefill_fresh_len, args.prefill_fresh_lens_csv)}_"
        f"{_lens_name('pp', prefill_prefix_values, args.prefill_prefix_len, args.prefill_prefix_lens_csv)}_"
        f"layers{args.num_layers}_backend_{args.attention_backend}_cg_{args.cuda_graph_mode}_"
        f"pcg_{args.pcg_mode}_tp{args.tp_size}_profile_{args.profile_mode}"
    )
    run_dir = run_root / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    local_model_dir = run_dir / "local_model"
    sitecustomize_dir = run_dir / "python_site"
    nsys_dir = run_dir / "nsys"
    nsys_dir.mkdir(exist_ok=True)

    request_json = run_dir / "generate_request.json"
    profile_json = run_dir / "start_profile.json"
    build_local_model(root_dir, local_model_dir, args.tp_size)
    write_sitecustomize_patch(sitecustomize_dir / "sitecustomize.py")
    request = build_request(request_json, args, vocab_size)
    build_profile(profile_json, args.profile_mode, args.mixed_profile_steps)

    helper_script = Path(__file__).resolve().parent / "run_agg_inside_container.py"
    inner_cmd = [
        "python",
        "/runner/run_agg_inside_container.py",
        "--model-path",
        "/model",
        "--request-json",
        "/out/generate_request.json",
        "--profile-json",
        "/out/start_profile.json",
        "--output-dir",
        "/out",
        "--num-layers",
        str(args.num_layers),
        "--context-length",
        str(args.context_length),
        "--mem-fraction-static",
        str(args.mem_fraction_static),
        "--cuda-graph-mode",
        args.cuda_graph_mode,
        "--pcg-mode",
        args.pcg_mode,
        "--warmup-runs",
        str(args.warmup_runs),
        "--chunked-prefill-size",
        str(args.chunked_prefill_size),
        "--max-prefill-tokens",
        str(args.max_prefill_tokens),
        "--mixed-profile-steps",
        str(args.mixed_profile_steps),
        "--decode-consume-delay-s",
        str(args.decode_consume_delay_s),
        "--decode-start-timeout-s",
        str(args.decode_start_timeout_s),
        "--decode-finish-timeout-s",
        str(args.decode_finish_timeout_s),
        "--tp-size",
        str(args.tp_size),
    ]
    if args.attention_backend not in ("", "auto", "default"):
        inner_cmd.extend(["--attention-backend", args.attention_backend])
    if args.decode_attention_backend not in ("", "auto", "default"):
        inner_cmd.extend(["--decode-attention-backend", args.decode_attention_backend])
    if args.layerwise_marker:
        inner_cmd.append("--layerwise-marker")

    if args.profile_mode == "nsys":
        inner_cmd = [
            "nsys",
            "profile",
            "--force-overwrite=true",
            "--trace=cuda,nvtx,osrt",
            f"--sample={args.nsys_sample}",
            f"--cpuctxsw={args.nsys_cpuctxsw}",
            "--cuda-graph-trace=node",
            "--capture-range=cudaProfilerApi",
            "--capture-range-end=stop",
            "--output=/out/nsys/report",
        ] + inner_cmd

    envs = {
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPYCACHEPREFIX": "/tmp/pycache",
        "PYTHONPATH": "/runner_site",
        "TORCHINDUCTOR_CACHE_DIR": "/tmp/torchinductor",
        "SGLANG_JIT_DEEPGEMM_PRECOMPILE": "1" if args.deepgemm_precompile else "0",
        "SGLANG_JIT_DEEPGEMM_FAST_WARMUP": "0" if args.no_deepgemm_fast_warmup else "1",
        "AIC_PROFILE_MIXED_ONLY_STAGE": "1",
        "AIC_MIXED_META_PATH": "/out/mixed_batch_meta.jsonl",
        "AIC_PROFILE_PREFILL_RID_PREFIX": "formal-prefill",
    }
    for item in args.extra_env:
        key, value = item.split("=", 1)
        envs[key] = value

    docker_cmd = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        f"device={args.gpu_id}",
        "--ipc=host",
        "--shm-size",
        "32g",
        *[part for key, value in envs.items() for part in ("-e", f"{key}={value}")],
        "-v",
        f"{run_dir}:/out",
        "-v",
        f"{local_model_dir}:/model:ro",
        "-v",
        f"{helper_script}:/runner/run_agg_inside_container.py:ro",
        "-v",
        f"{sitecustomize_dir}:/runner_site:ro",
        args.image,
    ] + inner_cmd

    (run_dir / "docker_command.txt").write_text(" ".join(docker_cmd) + "\n")
    (run_dir / "run_meta.json").write_text(
        json.dumps(
            {
                "run_name": run_name,
                "tag": args.tag,
                "gpu_id": args.gpu_id,
                "profile_mode": args.profile_mode,
                "num_layers": args.num_layers,
                "attention_backend": args.attention_backend,
                "decode_attention_backend": args.decode_attention_backend,
                "context_length": args.context_length,
                "chunked_prefill_size": args.chunked_prefill_size,
                "max_prefill_tokens": args.max_prefill_tokens,
                "mixed_profile_steps": args.mixed_profile_steps,
                "mixed_warmup_runs": args.warmup_runs,
                "deepgemm_precompile": args.deepgemm_precompile,
                "deepgemm_fast_warmup": not args.no_deepgemm_fast_warmup,
                **request["meta"],
            },
            indent=2,
        )
    )

    completed = subprocess.run(docker_cmd, text=True, capture_output=True)
    (run_dir / "container.stdout.log").write_text(completed.stdout)
    (run_dir / "container.stderr.log").write_text(completed.stderr)
    (run_dir / "docker_returncode.txt").write_text(str(completed.returncode) + "\n")
    if completed.returncode != 0:
        print(run_dir)
        return completed.returncode

    if args.profile_mode == "nsys":
        run_export_if_needed(root_dir, run_dir, args.image)

    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
