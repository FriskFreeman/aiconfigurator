#!/usr/bin/env python3
import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        default="booleimg.myaddr.io/lmsysorg/sglang:v0.5.9",
    )
    parser.add_argument("--gpu-id", type=int, default=7)
    parser.add_argument("--num-layers", type=int, default=5)
    parser.add_argument("--attention-backend", default="auto")
    parser.add_argument("--decode-attention-backend", default="auto")
    parser.add_argument(
        "--force-prefill-mha-for-prefix",
        action="store_true",
        help="Set SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD=0 so DeepSeek prefix-prefill "
        "prefers MHA on supported backends without changing decode.",
    )
    parser.add_argument(
        "--chunked-prefix-cache-threshold",
        type=int,
        default=8192,
        help="Value for SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD; only affects DeepSeek "
        "prefix-prefill backend selection on supported backends.",
    )
    parser.add_argument("--mem-fraction-static", type=float, default=0.5)
    parser.add_argument("--cuda-graph-mode", choices=["off", "on"], default="off")
    parser.add_argument("--pcg-mode", choices=["off", "on"], default="off")
    marker_group = parser.add_mutually_exclusive_group()
    marker_group.add_argument(
        "--layerwise-marker",
        dest="layerwise_marker",
        action="store_true",
    )
    marker_group.add_argument(
        "--no-layerwise-marker",
        dest="layerwise_marker",
        action="store_false",
    )
    parser.set_defaults(layerwise_marker=None)
    parser.add_argument(
        "--profile-mode",
        choices=["chrome", "nsys"],
        default="nsys",
    )
    parser.add_argument(
        "--nsys-sample",
        choices=["process-tree", "system-wide", "none"],
        default="none",
        help="Value passed to nsys profile --sample. Use process-tree for CPU IP/backtrace sampling.",
    )
    parser.add_argument(
        "--nsys-cpuctxsw",
        choices=["process-tree", "system-wide", "none"],
        default="none",
        help="Value passed to nsys profile --cpuctxsw.",
    )
    parser.add_argument(
        "--nsys-sampling-period",
        type=int,
        default=None,
        help="Optional value passed to nsys profile --sampling-period.",
    )
    parser.add_argument(
        "--nsys-samples-per-backtrace",
        type=int,
        default=None,
        help="Optional value passed to nsys profile --samples-per-backtrace.",
    )
    parser.add_argument(
        "--triton-cache-dir",
        default=None,
        help="Optional container path for TRITON_CACHE_DIR. Example: /out/triton_cache.",
    )
    parser.add_argument(
        "--triton-dump-dir",
        default=None,
        help="Optional container path for TRITON_DUMP_DIR. Example: /out/triton_dump.",
    )
    parser.add_argument(
        "--triton-debug",
        action="store_true",
        help="Set TRITON_DEBUG=1 inside the container.",
    )
    parser.add_argument(
        "--triton-print-autotuning",
        action="store_true",
        help="Set TRITON_PRINT_AUTOTUNING=1 inside the container.",
    )
    parser.add_argument(
        "--extra-env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Additional environment variable passed to docker. Can be repeated.",
    )
    parser.add_argument(
        "--docker-privileged",
        action="store_true",
        help="Run docker with --privileged. Useful when nsys CPU sampling is blocked by perf/capability limits.",
    )
    parser.add_argument(
        "--docker-cap-add",
        action="append",
        default=[],
        metavar="CAP",
        help="Additional docker --cap-add value. Can be repeated.",
    )
    parser.add_argument(
        "--docker-security-opt",
        action="append",
        default=[],
        metavar="OPT",
        help="Additional docker --security-opt value. Can be repeated.",
    )
    parser.add_argument(
        "--docker-pid-host",
        action="store_true",
        help="Run docker with --pid=host. Sometimes needed for process-tree profiler visibility.",
    )
    parser.add_argument("--profile-with-stack", action="store_true")
    parser.add_argument("--profile-record-shapes", action="store_true", default=True)
    parser.add_argument("--profile-by-stage", action="store_true")
    parser.add_argument(
        "--profile-num-steps",
        type=int,
        default=None,
        help="Optional SGLang profiler num_steps. With --profile-by-stage this is per stage.",
    )
    parser.add_argument(
        "--profile-decode-only-stage",
        action="store_true",
        help=(
            "For nsys decode runs, initialize SGLang stage profiling but capture "
            "only real decode batches inside the container runner."
        ),
    )
    parser.add_argument(
        "--stream-profile-after-first-token",
        action="store_true",
        help=(
            "For decode runs, consume the first streamed token outside profiling "
            "and profile remaining decode steps."
        ),
    )
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument(
        "--align-collector-prefill",
        action="store_true",
        default=False,
        help="Experimentally override SGLang chunk budgets to batch_size * fresh_len. "
        "Disabled by default to preserve native SGLang scheduling behavior.",
    )

    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--fresh-len", type=int)
    parser.add_argument("--prefix-len", type=int, default=0)
    parser.add_argument("--fresh-lens-csv")
    parser.add_argument("--prefix-lens-csv")

    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--tag", default="pilot")
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--allow-chunked-kv", action="store_true")
    parser.add_argument(
        "--limit-cuda-graph-bs-to-batch",
        action="store_true",
        help="For decode CUDA graph runs, capture only the formal batch size.",
    )
    parser.add_argument(
        "--decode-empty-input-continuation",
        action="store_true",
        help=(
            "After session prefix preparation, profile empty-input continuation "
            "generation. Intended for decode-only runs reusing this runner."
        ),
    )
    parser.add_argument(
        "--run-root",
        default=".self/task-localbench/formal-prefill-stage1",
    )
    return parser.parse_args()


def resolve_layerwise_marker(args: argparse.Namespace) -> bool:
    if args.layerwise_marker is not None:
        return bool(args.layerwise_marker)
    return args.profile_mode == "nsys"


def build_local_model(root_dir: Path, local_model_dir: Path, tp_size: int) -> None:
    src_cfg = (
        root_dir
        / "src/aiconfigurator/model_configs/deepseek-ai--DeepSeek-V3_config.json"
    )
    cfg = json.loads(src_cfg.read_text())
    cfg["model_type"] = "deepseek_v3"
    cfg["architectures"] = ["DeepseekV3ForCausalLM"]
    cfg.pop("auto_map", None)
    if tp_size > 1:
        for key in ["num_attention_heads", "num_key_value_heads"]:
            value = cfg.get(key)
            if not isinstance(value, int) or value <= 0 or value % tp_size != 0:
                raise ValueError(
                    f"{key}={value} is not divisible by tp_size={tp_size}"
                )
            cfg[key] = value // tp_size
        cfg["_aic_tp_shape_only_rank0"] = {
            "tp_size": tp_size,
            "note": (
                "Single-card TP shape-only override for MLA-aligned Engine runs. "
                "This is not real multi-rank TP and does not include TP communication."
            ),
        }
    local_model_dir.mkdir(parents=True, exist_ok=True)
    (local_model_dir / "config.json").write_text(json.dumps(cfg, indent=2))


def _parse_csv_ints(raw: str | None) -> list[int] | None:
    if raw is None:
        return None
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values or any(value < 0 for value in values):
        raise ValueError("CSV length lists must contain non-negative integers")
    return values


def resolve_lengths(args: argparse.Namespace) -> tuple[list[int], list[int]]:
    fresh_lens = _parse_csv_ints(args.fresh_lens_csv)
    prefix_lens = _parse_csv_ints(args.prefix_lens_csv)

    if fresh_lens is None:
        if args.batch_size is None or args.fresh_len is None:
            raise ValueError(
                "Either --fresh-lens-csv or both --batch-size/--fresh-len are required"
            )
        if args.batch_size <= 0 or args.fresh_len <= 0:
            raise ValueError("--batch-size and --fresh-len must be positive")
        fresh_lens = [args.fresh_len for _ in range(args.batch_size)]

    if prefix_lens is None:
        if args.prefix_len < 0:
            raise ValueError("--prefix-len must be non-negative")
        prefix_lens = [args.prefix_len for _ in range(len(fresh_lens))]

    if len(prefix_lens) != len(fresh_lens):
        raise ValueError("prefix lengths and fresh lengths must have the same batch size")
    if any(length <= 0 for length in fresh_lens):
        raise ValueError("All fresh lengths must be positive")
    return fresh_lens, prefix_lens


def _load_vocab_size(root_dir: Path) -> int:
    src_cfg = (
        root_dir
        / "src/aiconfigurator/model_configs/deepseek-ai--DeepSeek-V3_config.json"
    )
    cfg = json.loads(src_cfg.read_text())
    vocab_size = int(cfg["vocab_size"])
    if vocab_size <= 32:
        raise ValueError(f"Invalid vocab_size: {vocab_size}")
    return vocab_size


def _build_token_rows(
    lengths: list[int],
    base_seed: int,
    vocab_size: int,
) -> list[list[int]]:
    rows = []
    cursor = base_seed
    usable_vocab = vocab_size - 10
    for length in lengths:
        row = [10 + ((cursor + offset) % usable_vocab) for offset in range(length)]
        rows.append(row)
        cursor += length + 17
    return rows


def build_request(
    path: Path,
    fresh_lens: list[int],
    prefix_lens: list[int],
    max_new_tokens: int,
    vocab_size: int,
) -> None:
    prefix_input_ids = _build_token_rows(prefix_lens, 1_000, vocab_size)
    fresh_input_ids = _build_token_rows(fresh_lens, 10_000, vocab_size)
    warmup_prefix_input_ids = _build_token_rows(prefix_lens, 31_000, vocab_size)
    warmup_fresh_input_ids = _build_token_rows(fresh_lens, 51_000, vocab_size)
    payload = {
        "prefix_input_ids": prefix_input_ids,
        "fresh_input_ids": fresh_input_ids,
        "warmup_prefix_input_ids": warmup_prefix_input_ids,
        "warmup_fresh_input_ids": warmup_fresh_input_ids,
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": max_new_tokens,
        },
        "return_logprob": False,
        "rid": [f"stage1-req-{i}" for i in range(len(fresh_lens))],
    }
    path.write_text(json.dumps(payload, indent=2))


def build_profile(
    path: Path,
    profile_prefix: str,
    profile_mode: str,
    profile_with_stack: bool,
    profile_record_shapes: bool,
    profile_by_stage: bool,
    profile_num_steps: int | None,
) -> None:
    if profile_mode == "chrome":
        activities = ["CPU", "GPU"]
    else:
        activities = ["CUDA_PROFILER"]
    payload = {
        "output_dir": "/out/profile",
        "activities": activities,
        "with_stack": profile_with_stack,
        "record_shapes": profile_record_shapes,
        "profile_by_stage": profile_by_stage,
        "profile_prefix": profile_prefix,
    }
    if profile_num_steps is not None:
        payload["num_steps"] = profile_num_steps
    path.write_text(json.dumps(payload, indent=2))


def write_sitecustomize_patch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        r'''
import os


def _install_aic_decode_only_profile_patch():
    if os.getenv("AIC_PROFILE_DECODE_ONLY_STAGE") != "1":
        return
    try:
        from sglang.srt.managers.scheduler import Scheduler
    except Exception:
        return

    if getattr(Scheduler.run_batch, "_aic_site_decode_only_profile_patch", False):
        return

    original_profile_predicate = Scheduler._profile_batch_predicate
    original_run_batch = Scheduler.run_batch

    def _enabled():
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

    patched_profile_predicate._aic_site_decode_only_profile_patch = True
    patched_run_batch._aic_site_decode_only_profile_patch = True
    Scheduler._profile_batch_predicate = patched_profile_predicate
    Scheduler.run_batch = patched_run_batch


_install_aic_decode_only_profile_patch()
'''.lstrip(),
        encoding="utf-8",
    )


def _lens_name(prefix: str, values: list[int], scalar: int | None, csv_raw: str | None) -> str:
    if csv_raw is None and scalar is not None and len(set(values)) == 1:
        return f"{prefix}{scalar}"
    return f"{prefix}var" + "-".join(str(v) for v in values)


def main() -> int:
    args = parse_args()
    resolved_layerwise_marker = resolve_layerwise_marker(args)
    fresh_lens, prefix_lens = resolve_lengths(args)
    batch_size = len(fresh_lens)
    effective_chunked_prefix_cache_threshold = (
        0 if args.force_prefill_mha_for_prefix else args.chunked_prefix_cache_threshold
    )
    fresh_name = _lens_name("f", fresh_lens, args.fresh_len, args.fresh_lens_csv)
    prefix_name = _lens_name("p", prefix_lens, args.prefix_len, args.prefix_lens_csv)

    root_dir = Path(__file__).resolve().parents[3]
    vocab_size = _load_vocab_size(root_dir)
    run_root = root_dir / args.run_root
    if run_root.name != "output":
        run_root = run_root / "output"
    helper_script = Path(__file__).resolve().parent / "run_engine_inside_container.py"
    export_script = Path(__file__).resolve().parent / "export_nsys_profile.py"

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = (
        f"{stamp}_prefill_stage1_{args.tag}_"
        f"b{batch_size}_{fresh_name}_{prefix_name}_layers{args.num_layers}_"
        f"backend_{args.attention_backend}_decodebackend_{args.decode_attention_backend}_"
        f"cg_{args.cuda_graph_mode}_"
        f"pcg_{args.pcg_mode}_tp{args.tp_size}_"
        f"marker_{'on' if resolved_layerwise_marker else 'off'}_"
        f"profile_{args.profile_mode}"
    )
    run_dir = run_root / run_name
    local_model_dir = run_dir / "local_model"
    profile_dir = run_dir / "profile"
    nsys_dir = run_dir / "nsys"
    sitecustomize_dir = run_dir / "python_site"
    request_json = run_dir / "generate_request.json"
    profile_json = run_dir / "start_profile.json"
    stdout_log = run_dir / "container.stdout.log"
    stderr_log = run_dir / "container.stderr.log"
    command_txt = run_dir / "docker_command.txt"

    run_dir.mkdir(parents=True, exist_ok=True)
    profile_dir.mkdir(parents=True, exist_ok=True)
    nsys_dir.mkdir(parents=True, exist_ok=True)
    write_sitecustomize_patch(sitecustomize_dir / "sitecustomize.py")
    build_local_model(root_dir, local_model_dir, args.tp_size)
    build_request(
        request_json,
        fresh_lens,
        prefix_lens,
        args.max_new_tokens,
        vocab_size,
    )
    build_profile(
        profile_json,
        "deepseek-v3-prefill-stage1",
        args.profile_mode,
        args.profile_with_stack,
        args.profile_record_shapes,
        args.profile_by_stage,
        args.profile_num_steps,
    )

    inner_cmd = [
        "python",
        "/runner/run_engine_inside_container.py",
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
        "--tp-size",
        str(args.tp_size),
        "--chunked-prefix-cache-threshold",
        str(effective_chunked_prefix_cache_threshold),
    ]
    if args.allow_chunked_kv:
        inner_cmd.append("--allow-chunked-kv")
    if args.decode_empty_input_continuation:
        inner_cmd.append("--decode-empty-input-continuation")
    if args.limit_cuda_graph_bs_to_batch:
        inner_cmd.append("--limit-cuda-graph-bs-to-batch")
    if args.profile_decode_only_stage:
        inner_cmd.append("--profile-decode-only-stage")
    if args.stream_profile_after_first_token:
        inner_cmd.append("--stream-profile-after-first-token")
    if args.align_collector_prefill:
        total_fresh_tokens = sum(fresh_lens)
        inner_cmd.extend(
            [
                "--chunked-prefill-size",
                str(total_fresh_tokens),
                "--max-prefill-tokens",
                str(total_fresh_tokens),
            ]
        )
    if args.attention_backend not in ("", "auto", "default"):
        inner_cmd.extend(["--attention-backend", args.attention_backend])
    if args.decode_attention_backend not in ("", "auto", "default"):
        inner_cmd.extend(["--decode-attention-backend", args.decode_attention_backend])
    if resolved_layerwise_marker:
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
        if args.nsys_sampling_period is not None:
            inner_cmd.insert(6, f"--sampling-period={args.nsys_sampling_period}")
        if args.nsys_samples_per_backtrace is not None:
            inner_cmd.insert(6, f"--samples-per-backtrace={args.nsys_samples_per_backtrace}")

    extra_envs = {
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPYCACHEPREFIX": "/tmp/pycache",
        "PYTHONPATH": "/runner_site",
        "TORCHINDUCTOR_CACHE_DIR": "/tmp/torchinductor",
        "SGLANG_JIT_DEEPGEMM_PRECOMPILE": "0",
        "SGLANG_JIT_DEEPGEMM_FAST_WARMUP": "1",
        "SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD": str(
            effective_chunked_prefix_cache_threshold
        ),
    }
    if args.triton_cache_dir:
        extra_envs["TRITON_CACHE_DIR"] = args.triton_cache_dir
    if args.triton_dump_dir:
        extra_envs["TRITON_DUMP_DIR"] = args.triton_dump_dir
    if args.triton_debug:
        extra_envs["TRITON_DEBUG"] = "1"
    if args.triton_print_autotuning:
        extra_envs["TRITON_PRINT_AUTOTUNING"] = "1"
    if args.profile_decode_only_stage:
        extra_envs["AIC_PROFILE_DECODE_ONLY_STAGE"] = "1"
    for item in args.extra_env:
        if "=" not in item:
            raise ValueError(f"--extra-env must be KEY=VALUE, got: {item!r}")
        key, value = item.split("=", 1)
        if not key:
            raise ValueError(f"--extra-env key is empty: {item!r}")
        extra_envs[key] = value

    docker_profile_args = []
    if args.docker_privileged:
        docker_profile_args.append("--privileged")
    if args.docker_pid_host:
        docker_profile_args.append("--pid=host")
    for cap in args.docker_cap_add:
        docker_profile_args.extend(["--cap-add", cap])
    for opt in args.docker_security_opt:
        docker_profile_args.extend(["--security-opt", opt])

    docker_cmd = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        f"device={args.gpu_id}",
        "--ipc=host",
        "--shm-size",
        "32g",
        *docker_profile_args,
        *[
            part
            for key, value in extra_envs.items()
            for part in ("-e", f"{key}={value}")
        ],
        "-v",
        f"{run_dir}:/out",
        "-v",
        f"{local_model_dir}:/model:ro",
        "-v",
        f"{helper_script}:/runner/run_engine_inside_container.py:ro",
        "-v",
        f"{sitecustomize_dir}:/runner_site:ro",
        args.image,
    ] + inner_cmd

    command_txt.write_text(" ".join(docker_cmd) + "\n")
    completed = subprocess.run(docker_cmd, text=True, capture_output=True)
    stdout_log.write_text(completed.stdout)
    stderr_log.write_text(completed.stderr)

    status_json_path = run_dir / "status.json"
    inner_status = None
    if status_json_path.exists():
        try:
            inner_status = json.loads(status_json_path.read_text())
        except json.JSONDecodeError:
            inner_status = None

    run_ok = completed.returncode == 0 and inner_status is not None and inner_status.get("status") == "ok"

    meta = {
        "image": args.image,
        "gpu_id": args.gpu_id,
        "num_layers": args.num_layers,
        "attention_backend": args.attention_backend,
        "decode_attention_backend": args.decode_attention_backend,
        "force_prefill_mha_for_prefix": args.force_prefill_mha_for_prefix,
        "chunked_prefix_cache_threshold": args.chunked_prefix_cache_threshold,
        "effective_chunked_prefix_cache_threshold": effective_chunked_prefix_cache_threshold,
        "mem_fraction_static": args.mem_fraction_static,
        "cuda_graph_mode": args.cuda_graph_mode,
        "pcg_mode": args.pcg_mode,
        "layerwise_marker": resolved_layerwise_marker,
        "layerwise_marker_requested": args.layerwise_marker,
        "profile_mode": args.profile_mode,
        "nsys_sample": args.nsys_sample,
        "nsys_cpuctxsw": args.nsys_cpuctxsw,
        "nsys_sampling_period": args.nsys_sampling_period,
        "nsys_samples_per_backtrace": args.nsys_samples_per_backtrace,
        "triton_cache_dir": args.triton_cache_dir,
        "triton_dump_dir": args.triton_dump_dir,
        "triton_debug": args.triton_debug,
        "triton_print_autotuning": args.triton_print_autotuning,
        "extra_env": args.extra_env,
        "docker_env": extra_envs,
        "docker_privileged": args.docker_privileged,
        "docker_cap_add": args.docker_cap_add,
        "docker_security_opt": args.docker_security_opt,
        "docker_pid_host": args.docker_pid_host,
        "profile_with_stack": args.profile_with_stack,
        "profile_record_shapes": args.profile_record_shapes,
        "profile_by_stage": args.profile_by_stage,
        "profile_num_steps": args.profile_num_steps,
        "profile_decode_only_stage": args.profile_decode_only_stage,
        "stream_profile_after_first_token": args.stream_profile_after_first_token,
        "warmup_runs": args.warmup_runs,
        "align_collector_prefill": args.align_collector_prefill,
        "fresh_len": args.fresh_len,
        "prefix_len": args.prefix_len,
        "fresh_lens": fresh_lens,
        "prefix_lens": prefix_lens,
        "batch_size": batch_size,
        "context_length": args.context_length,
        "max_new_tokens": args.max_new_tokens,
        "tp_size": args.tp_size,
        "tp_shape_only_rank0": args.tp_size > 1,
        "tp_shape_impl": (
            "local_model_config_head_shard"
            if args.tp_size > 1
            else "none"
        ),
        "allow_chunked_kv": args.allow_chunked_kv,
        "limit_cuda_graph_bs_to_batch": args.limit_cuda_graph_bs_to_batch,
        "decode_empty_input_continuation": args.decode_empty_input_continuation,
        "vocab_size": vocab_size,
        "docker_returncode": completed.returncode,
        "inner_status": inner_status,
        "nsys_export_returncode": None,
        "status": "ok" if run_ok else "fail",
    }
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))

    export_returncode = None
    if run_ok and args.profile_mode == "nsys":
        export_log = run_dir / "nsys_export.stdout.log"
        export_err = run_dir / "nsys_export.stderr.log"
        export_cmd = [
            "python",
            str(export_script),
            "--run-dir",
            str(run_dir),
            "--image",
            args.image,
        ]
        export_completed = subprocess.run(export_cmd, text=True, capture_output=True)
        export_log.write_text(export_completed.stdout)
        export_err.write_text(export_completed.stderr)
        export_returncode = export_completed.returncode

    meta["nsys_export_returncode"] = export_returncode
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))
    print(run_dir)
    return 0 if run_ok else max(1, completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
