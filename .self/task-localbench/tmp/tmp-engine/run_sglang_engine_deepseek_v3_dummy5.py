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
    parser.add_argument("--prompt-len", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--attention-backend", default="auto")
    parser.add_argument("--mem-fraction-static", type=float, default=0.5)
    parser.add_argument("--cuda-graph-mode", choices=["off", "on"], default="off")
    parser.add_argument("--pcg-mode", choices=["off", "on", "both"], default="both")
    parser.add_argument(
        "--run-root",
        default=".self/task-localbench/tmp/tmp-engine",
    )
    return parser.parse_args()


def build_local_model(root_dir: Path, local_model_dir: Path) -> None:
    src_cfg = (
        root_dir
        / "src/aiconfigurator/model_configs/deepseek-ai--DeepSeek-V3_config.json"
    )
    cfg = json.loads(src_cfg.read_text())
    cfg["model_type"] = "deepseek_v3"
    cfg["architectures"] = ["DeepseekV3ForCausalLM"]
    cfg.pop("auto_map", None)
    local_model_dir.mkdir(parents=True, exist_ok=True)
    (local_model_dir / "config.json").write_text(json.dumps(cfg, indent=2))


def build_request(path: Path, prompt_len: int, max_new_tokens: int) -> None:
    payload = {
        "input_ids": list(range(1, prompt_len + 1)),
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": max_new_tokens,
        },
        "return_logprob": False,
    }
    path.write_text(json.dumps(payload, indent=2))


def build_profile(path: Path) -> None:
    payload = {
        "output_dir": "/out/profile",
        "activities": ["CPU", "GPU"],
        "with_stack": False,
        "record_shapes": True,
        "profile_by_stage": False,
        "profile_prefix": "deepseek-v3-dummy5-engine",
    }
    path.write_text(json.dumps(payload, indent=2))


def run_case(
    args: argparse.Namespace,
    root_dir: Path,
    helper_script: Path,
    pcg_mode: str,
):
    run_root = root_dir / args.run_root
    if run_root.name != "output":
        run_root = run_root / "output"
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = (
        run_root
        / (
            f"{run_stamp}_sglang_engine_deepseek_v3_dummy5_"
            f"cg_{args.cuda_graph_mode}_pcg_{pcg_mode}"
        )
    )
    local_model_dir = run_dir / "local_model"
    profile_dir = run_dir / "profile"
    request_json = run_dir / "generate_request.json"
    profile_json = run_dir / "start_profile.json"
    stdout_log = run_dir / "container.stdout.log"
    stderr_log = run_dir / "container.stderr.log"
    command_txt = run_dir / "docker_command.txt"

    run_dir.mkdir(parents=True, exist_ok=True)
    profile_dir.mkdir(parents=True, exist_ok=True)
    build_local_model(root_dir, local_model_dir)
    build_request(request_json, args.prompt_len, args.max_new_tokens)
    build_profile(profile_json)

    docker_cmd = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        f"device={args.gpu_id}",
        "--ipc=host",
        "--shm-size",
        "32g",
        "-e",
        "PYTHONDONTWRITEBYTECODE=1",
        "-e",
        "PYTHONPYCACHEPREFIX=/tmp/pycache",
        "-e",
        "TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor",
        "-e",
        "SGLANG_JIT_DEEPGEMM_PRECOMPILE=0",
        "-e",
        "SGLANG_JIT_DEEPGEMM_FAST_WARMUP=1",
        "-v",
        f"{run_dir}:/out",
        "-v",
        f"{local_model_dir}:/model:ro",
        "-v",
        f"{helper_script}:/runner/run_engine_inside_container.py:ro",
        args.image,
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
        pcg_mode,
    ]
    if args.attention_backend not in ("", "auto", "default"):
        docker_cmd.extend(["--attention-backend", args.attention_backend])
    command_txt.write_text(" ".join(docker_cmd) + "\n")

    completed = subprocess.run(docker_cmd, text=True, capture_output=True)
    stdout_log.write_text(completed.stdout)
    stderr_log.write_text(completed.stderr)

    status = "ok" if completed.returncode == 0 else "fail"
    (run_dir / "run_meta.txt").write_text(
        "\n".join(
            [
                f"image={args.image}",
                f"gpu_id={args.gpu_id}",
                f"num_layers={args.num_layers}",
                f"prompt_len={args.prompt_len}",
                f"max_new_tokens={args.max_new_tokens}",
                f"context_length={args.context_length}",
                f"attention_backend={args.attention_backend}",
                f"mem_fraction_static={args.mem_fraction_static}",
                f"cuda_graph_mode={args.cuda_graph_mode}",
                f"pcg_mode={pcg_mode}",
                f"docker_returncode={completed.returncode}",
                f"status={status}",
            ]
        )
        + "\n"
    )
    return status, run_dir


def main() -> int:
    args = parse_args()
    root_dir = Path(__file__).resolve().parents[3]
    helper_script = Path(__file__).resolve().parent / "run_engine_inside_container.py"

    modes = [args.pcg_mode] if args.pcg_mode != "both" else ["off", "on"]
    summary_path = root_dir / args.run_root / "engine_pcg_summary.tsv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_lines = ["cuda_graph_mode\tpcg_mode\tstatus\trun_dir"]

    exit_code = 0
    for mode in modes:
        status, run_dir = run_case(args, root_dir, helper_script, mode)
        summary_lines.append(
            f"{args.cuda_graph_mode}\t{mode}\t{status}\t{run_dir}"
        )
        if status != "ok":
            exit_code = 1

    summary_path.write_text("\n".join(summary_lines) + "\n")
    print(summary_path)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
