#!/usr/bin/env python3
import argparse
import json
import os
import shlex
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
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--attention-backend", default="auto")
    parser.add_argument("--mem-fraction-static", type=float, default=0.5)
    parser.add_argument("--bootstrap-port", type=int, default=28999)
    parser.add_argument("--engine-info-bootstrap-port", type=int, default=36790)
    parser.add_argument("--run-root", default=".self/task-localbench/tmp/tmp-decode")
    return parser.parse_args()


def shell_join(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


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


def main() -> int:
    args = parse_args()
    root_dir = Path(__file__).resolve().parents[3]
    helper_script = Path(__file__).resolve().parent / "run_decode_inside_container.py"
    run_root = root_dir / args.run_root
    if run_root.name != "output":
        run_root = run_root / "output"
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = run_root / (
        f"{run_stamp}_sglang_decode_instance_dummy5_"
        f"cg_on_backend_{args.attention_backend}"
    )
    local_model_dir = run_dir / "local_model"
    stdout_log = run_dir / "container.stdout.log"
    stderr_log = run_dir / "container.stderr.log"
    command_txt = run_dir / "docker_command.txt"

    run_dir.mkdir(parents=True, exist_ok=True)
    build_local_model(root_dir, local_model_dir)

    docker_cmd = [
        "docker",
        "run",
        "--rm",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--gpus",
        f"device={args.gpu_id}",
        "--ipc=host",
        "--shm-size",
        "32g",
        "-e",
        "PYTHONDONTWRITEBYTECODE=1",
        "-e",
        "HOME=/tmp",
        "-e",
        "XDG_CACHE_HOME=/tmp/.cache",
        "-e",
        "PYTHONPYCACHEPREFIX=/tmp/pycache",
        "-e",
        "TRITON_CACHE_DIR=/tmp/triton",
        "-e",
        "TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor",
        "-e",
        "SGLANG_JIT_DEEPGEMM_PRECOMPILE=0",
        "-e",
        "SGLANG_JIT_DEEPGEMM_FAST_WARMUP=1",
        "-v",
        "/etc/passwd:/etc/passwd:ro",
        "-v",
        "/etc/group:/etc/group:ro",
        "-v",
        f"{run_dir}:/out",
        "-v",
        f"{local_model_dir}:/model:ro",
        "-v",
        f"{helper_script}:/runner/run_decode_inside_container.py:ro",
        args.image,
        "python",
        "/runner/run_decode_inside_container.py",
        "--model-path",
        "/model",
        "--output-dir",
        "/out",
        "--num-layers",
        str(args.num_layers),
        "--context-length",
        str(args.context_length),
        "--mem-fraction-static",
        str(args.mem_fraction_static),
        "--bootstrap-port",
        str(args.bootstrap_port),
        "--engine-info-bootstrap-port",
        str(args.engine_info_bootstrap_port),
    ]
    if args.attention_backend not in ("", "auto", "default"):
        docker_cmd.extend(["--attention-backend", args.attention_backend])

    command_txt.write_text(shell_join(docker_cmd) + "\n")
    completed = subprocess.run(docker_cmd, text=True, capture_output=True)
    stdout_log.write_text(completed.stdout)
    stderr_log.write_text(completed.stderr)

    status_file = run_dir / "status.json"
    stage_file = run_dir / "stage.txt"
    engine_status = None
    if status_file.exists():
        try:
            engine_status = json.loads(status_file.read_text()).get("status")
        except json.JSONDecodeError:
            engine_status = "invalid_status_json"
    stage_trace = []
    if stage_file.exists():
        stage_trace = [
            line.strip() for line in stage_file.read_text().splitlines() if line.strip()
        ]
    final_status = (
        "ok"
        if completed.returncode == 0 and engine_status == "ok"
        else "fail"
    )

    run_meta = {
        "status": final_status,
        "image": args.image,
        "gpu_id": args.gpu_id,
        "num_layers": args.num_layers,
        "context_length": args.context_length,
        "attention_backend": args.attention_backend,
        "mem_fraction_static": args.mem_fraction_static,
        "disaggregation_mode": "decode",
        "disaggregation_transfer_backend": "nixl",
        "cuda_graph_mode": "on",
        "docker_returncode": completed.returncode,
        "engine_status": engine_status,
        "stage_trace": stage_trace,
        "run_dir": str(run_dir),
    }
    (run_dir / "run_meta.json").write_text(
        json.dumps(run_meta, indent=2, sort_keys=True)
    )
    print(run_dir)
    return 0 if final_status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
