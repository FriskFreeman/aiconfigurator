#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="booleimg.myaddr.io/lmsysorg/sglang:v0.5.9")
    parser.add_argument("--gpu-id", type=int, default=7)
    parser.add_argument("--run-root", default=".self/task-localbench/mla-module-quant-exp")
    parser.add_argument("--tag", default="fp8gemm_fp8kv_bf16attn_fa3_smoke")
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--num-heads", type=int, default=128)
    parser.add_argument("--attention-backend", default="fa3")
    parser.add_argument("--kv-cache-dtype", default="fp8")
    parser.add_argument("--compute-dtype", default="bfloat16")
    parser.add_argument("--gemm-type", default="fp8_block")
    parser.add_argument("--prefill-cases", default="4x512,1x8192")
    parser.add_argument("--decode-cases", default="4x512,8x2048")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--mem-fraction-static", type=float, default=0.5)
    parser.add_argument("--nsys-sample", choices=["none", "process-tree", "system-wide"], default="none")
    parser.add_argument("--nsys-cpuctxsw", choices=["none", "process-tree", "system-wide"], default="none")
    return parser.parse_args()


def _parse_cases(raw: str, is_prefill: bool) -> list[tuple[int, int, bool]]:
    cases: list[tuple[int, int, bool]] = []
    if not raw:
        return cases
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "x" not in item:
            raise ValueError(f"case must be BxS, got {item!r}")
        b_raw, s_raw = item.split("x", 1)
        b = int(b_raw)
        s = int(s_raw)
        if b <= 0 or s <= 0:
            raise ValueError(f"case values must be positive, got {item!r}")
        cases.append((b, s, is_prefill))
    return cases


def write_inner_script(path: Path) -> None:
    path.write_text(
        r'''
import argparse
import json
import os
from pathlib import Path

import torch

from collector.sglang import collect_mla_module as cmm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--num-layers", type=int, required=True)
    parser.add_argument("--num-heads", type=int, required=True)
    parser.add_argument("--attention-backend", required=True)
    parser.add_argument("--kv-cache-dtype", required=True)
    parser.add_argument("--compute-dtype", required=True)
    parser.add_argument("--gemm-type", required=True)
    parser.add_argument("--cases-json", required=True)
    parser.add_argument("--warmup", type=int, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    os.environ["SGLANG_TEST_NUM_LAYERS"] = str(args.num_layers)
    os.environ.setdefault("SGLANG_LOAD_FORMAT", "dummy")
    os.environ.setdefault("SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD", "0")
    os.environ.setdefault("SGLANG_JIT_DEEPGEMM_PRECOMPILE", "0")
    os.environ.setdefault("SGLANG_JIT_DEEPGEMM_FAST_WARMUP", "1")

    cases = [tuple(item) for item in json.loads(args.cases_json)]
    runner = cmm.load_model_runner(
        model_path="deepseek-ai/DeepSeek-V3",
        head_num=args.num_heads,
        kv_cache_dtype=args.kv_cache_dtype,
        attention_backend=args.attention_backend,
        device="cuda:0",
        gemm_type=args.gemm_type,
    )

    meta = {
        "source": "collect_mla_module_direct_import",
        "model_path": "deepseek-ai/DeepSeek-V3",
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "attention_backend": args.attention_backend,
        "kv_cache_dtype": args.kv_cache_dtype,
        "compute_dtype": args.compute_dtype,
        "gemm_type": args.gemm_type,
        "cases": cases,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "server_args_attention_backend": runner.server_args.attention_backend,
        "server_args_kv_cache_dtype": getattr(runner.server_args, "kv_cache_dtype", None),
        "server_args_quantization": getattr(runner.server_args, "quantization", None),
    }
    (output_path / "experiment_meta.json").write_text(json.dumps(meta, indent=2, default=str))

    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    torch.cuda.nvtx.range_push("aic_collect_mla_module_quant_experiment")
    try:
        cmm.run_attention_torch(
            model_runner=runner,
            test_cases=cases,
            head_num=args.num_heads,
            test_layer=0,
            num_warmup=args.warmup,
            num_iterations=args.iterations,
            device="cuda:0",
            output_path=str(output_path),
            attn_type="mla",
            model_path="deepseek-ai/DeepSeek-V3",
            kv_cache_dtype=args.kv_cache_dtype,
            compute_dtype=args.compute_dtype,
            gemm_type=args.gemm_type,
        )
        torch.cuda.synchronize()
    finally:
        torch.cuda.nvtx.range_pop()
        torch.cuda.cudart().cudaProfilerStop()


if __name__ == "__main__":
    main()
'''.lstrip()
    )


def main() -> int:
    args = parse_args()
    root_dir = Path(__file__).resolve().parents[3]
    run_root = root_dir / args.run_root
    if run_root.name != "output":
        run_root = run_root / "output"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = (
        f"{stamp}_{args.tag}_heads{args.num_heads}_backend_{args.attention_backend}_"
        f"kv_{args.kv_cache_dtype}_gemm_{args.gemm_type}_compute_{args.compute_dtype}"
    )
    run_dir = run_root / run_name
    nsys_dir = run_dir / "nsys"
    run_dir.mkdir(parents=True, exist_ok=True)
    nsys_dir.mkdir(parents=True, exist_ok=True)

    inner_script = run_dir / "run_inside_container.py"
    write_inner_script(inner_script)

    cases = _parse_cases(args.prefill_cases, True) + _parse_cases(args.decode_cases, False)
    (run_dir / "cases.json").write_text(json.dumps(cases, indent=2))

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
        "python",
        "/out/run_inside_container.py",
        "--output-path",
        "/out",
        "--num-layers",
        str(args.num_layers),
        "--num-heads",
        str(args.num_heads),
        "--attention-backend",
        args.attention_backend,
        "--kv-cache-dtype",
        args.kv_cache_dtype,
        "--compute-dtype",
        args.compute_dtype,
        "--gemm-type",
        args.gemm_type,
        "--cases-json",
        json.dumps(cases),
        "--warmup",
        str(args.warmup),
        "--iterations",
        str(args.iterations),
    ]

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
        "PYTHONPATH=/workspace",
        "-e",
        "PYTHONDONTWRITEBYTECODE=1",
        "-e",
        "PYTHONPYCACHEPREFIX=/tmp/pycache",
        "-e",
        f"SGLANG_TEST_NUM_LAYERS={args.num_layers}",
        "-e",
        "SGLANG_LOAD_FORMAT=dummy",
        "-e",
        "SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD=0",
        "-e",
        "SGLANG_JIT_DEEPGEMM_PRECOMPILE=0",
        "-e",
        "SGLANG_JIT_DEEPGEMM_FAST_WARMUP=1",
        "-v",
        f"{root_dir}:/workspace:ro",
        "-v",
        f"{run_dir}:/out",
        "-w",
        "/workspace",
        args.image,
    ] + inner_cmd

    (run_dir / "docker_command.txt").write_text(" ".join(docker_cmd) + "\n")
    (run_dir / "run_meta.json").write_text(
        json.dumps(
            {
                "run_name": run_name,
                "image": args.image,
                "gpu_id": args.gpu_id,
                "cases": cases,
                "num_layers": args.num_layers,
                "num_heads": args.num_heads,
                "attention_backend": args.attention_backend,
                "kv_cache_dtype": args.kv_cache_dtype,
                "compute_dtype": args.compute_dtype,
                "gemm_type": args.gemm_type,
            },
            indent=2,
        )
    )

    completed = subprocess.run(docker_cmd, text=True, capture_output=True)
    (run_dir / "container.stdout.log").write_text(completed.stdout)
    (run_dir / "container.stderr.log").write_text(completed.stderr)
    (run_dir / "status.json").write_text(
        json.dumps({"returncode": completed.returncode, "run_dir": str(run_dir)}, indent=2)
    )
    print(run_dir)
    if completed.returncode != 0:
        print(completed.stdout[-4000:])
        print(completed.stderr[-4000:])
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
