#!/usr/bin/env python3
"""Run matched BF16/FP8 collect_mla experiments in the SGLang 0.5.9 image."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from datetime import datetime
from pathlib import Path


IMAGE = "booleimg.myaddr.io/lmsysorg/sglang:v0.5.9"
CASES = (
    {"name": "prefill_b1_s256", "stage": "prefill", "input_len": 256, "batch_size": 1},
    {"name": "prefill_b1_s2048", "stage": "prefill", "input_len": 2048, "batch_size": 1},
    {"name": "decode_b1_step255", "stage": "decode", "input_len": 255, "batch_size": 1},
    {"name": "decode_b1_step2047", "stage": "decode", "input_len": 2047, "batch_size": 1},
    {
        "name": "decode_b64_step32767",
        "stage": "decode",
        "input_len": 32767,
        "batch_size": 64,
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default="4")
    parser.add_argument("--image", default=IMAGE)
    parser.add_argument("--mode", choices=("latency", "nsys", "all"), default="all")
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--latency-repeats", type=int, default=3)
    parser.add_argument("--latency-iterations", type=int, default=100)
    parser.add_argument("--nsys-iterations", type=int, default=10)
    return parser.parse_args()


def run_logged(cmd: list[str], run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "command.txt").write_text(shlex.join(cmd) + "\n")
    completed = subprocess.run(cmd, text=True, capture_output=True)
    (run_dir / "stdout.log").write_text(completed.stdout)
    (run_dir / "stderr.log").write_text(completed.stderr)
    (run_dir / "status.json").write_text(
        json.dumps({"returncode": completed.returncode}, indent=2) + "\n"
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed ({completed.returncode}): {shlex.join(cmd)}")


def docker_prefix(args: argparse.Namespace, root: Path, out_dir: Path) -> list[str]:
    return [
        "docker",
        "run",
        "--rm",
        "--gpus",
        f"device={args.gpu}",
        "--ipc=host",
        "--shm-size",
        "8g",
        "-e",
        "PYTHONPATH=/workspace",
        "-e",
        "PYTHONDONTWRITEBYTECODE=1",
        "-e",
        "PYTHONPYCACHEPREFIX=/tmp/pycache",
        "-v",
        f"{root}:/workspace:ro",
        "-v",
        f"{out_dir}:/out",
        "-w",
        "/workspace",
        args.image,
    ]


def runner_args(case: dict[str, object], dtype: str, execution: str, iterations: int, profile: bool) -> list[str]:
    args = [
        "python",
        "/workspace/collector/[self]doc/debug-diff/sglang-mla-bf16-fp8-nsys/profile_collect_mla.py",
        "--stage",
        str(case["stage"]),
        "--kv-cache-dtype",
        dtype,
        "--input-len",
        str(case["input_len"]),
        "--batch-size",
        str(case.get("batch_size", 1)),
        "--num-heads",
        str(case.get("num_heads", 128)),
        "--tp-size",
        str(case.get("tp_size", 1)),
        "--execution",
        execution,
        "--warmups",
        "3",
        "--iterations",
        str(iterations),
        "--output-dir",
        "/out",
    ]
    if profile:
        args.append("--profile")
    return args


def check_gpu_free(gpu: str) -> dict[str, object]:
    cmd = [
        "nvidia-smi",
        "-i",
        gpu,
        "--query-gpu=index,name,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(cmd, text=True, capture_output=True, check=True)
    fields = [item.strip() for item in completed.stdout.strip().split(",")]
    state = {
        "index": fields[0],
        "name": fields[1],
        "memory_used_mib": int(fields[2]),
        "memory_free_mib": int(fields[3]),
        "utilization_percent": int(fields[4]),
    }
    if state["memory_free_mib"] < 70_000:
        raise RuntimeError(f"GPU {gpu} is not sufficiently free: {state}")
    return state


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[4]
    work_dir = Path(__file__).resolve().parent
    result_root = work_dir / "results"
    selected = [case for case in CASES if not args.case or case["name"] in args.case]
    unknown = set(args.case) - {str(case["name"]) for case in CASES}
    if unknown:
        raise ValueError(f"Unknown cases: {sorted(unknown)}")

    preflight = check_gpu_free(args.gpu)
    result_root.mkdir(parents=True, exist_ok=True)
    (result_root / "run_metadata.json").write_text(
        json.dumps(
            {
                "timestamp": datetime.now().astimezone().isoformat(),
                "image": args.image,
                "gpu": args.gpu,
                "preflight": preflight,
                "cases": selected,
            },
            indent=2,
        )
        + "\n"
    )

    if args.mode in ("latency", "all"):
        for case in selected:
            for dtype in ("bf16", "fp8"):
                for repeat in range(1, args.latency_repeats + 1):
                    out_dir = result_root / "latency" / str(case["name"]) / dtype / f"repeat_{repeat}"
                    cmd = docker_prefix(args, root, out_dir) + runner_args(
                        case, dtype, "graph", args.latency_iterations, False
                    )
                    run_logged(cmd, out_dir)

    if args.mode in ("nsys", "all"):
        for case in selected:
            for dtype in ("bf16", "fp8"):
                out_dir = result_root / "nsys" / str(case["name"]) / dtype
                nsys_cmd = [
                    "nsys",
                    "profile",
                    "--force-overwrite=true",
                    "--trace=cuda,nvtx,osrt",
                    "--sample=none",
                    "--cpuctxsw=none",
                    "--cuda-graph-trace=node",
                    "--capture-range=cudaProfilerApi",
                    "--capture-range-end=stop",
                    "--output=/out/report",
                ] + runner_args(case, dtype, "eager", args.nsys_iterations, True)
                cmd = docker_prefix(args, root, out_dir) + nsys_cmd
                run_logged(cmd, out_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
