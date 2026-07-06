#!/usr/bin/env python3
import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="booleimg.myaddr.io/lmsysorg/sglang:v0.5.9")
    parser.add_argument("--gpu-id", type=int, default=7)
    parser.add_argument("--run-root", default=".self/task-localbench/mla-module-quant-exp")
    parser.add_argument("--tag", default="module_timing_smoke")
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--num-heads", type=int, default=128)
    parser.add_argument("--attention-backend", default="fa3")
    parser.add_argument("--kv-cache-dtype", default="fp8")
    parser.add_argument("--compute-dtype", default="bfloat16")
    parser.add_argument("--gemm-type", default="fp8_block")
    parser.add_argument("--prefill-cases", default="1x1,4x512,1x8192")
    parser.add_argument("--decode-cases", default="4x512,8x2048")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    return parser.parse_args()


def _parse_cases(raw: str, is_prefill: bool) -> list[tuple[int, int, bool]]:
    cases: list[tuple[int, int, bool]] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        b_raw, s_raw = item.split("x", 1)
        cases.append((int(b_raw), int(s_raw), is_prefill))
    return cases


def write_inner_script(path: Path) -> None:
    path.write_text(
        r'''
import argparse
import json
import os
import time
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
    out = Path(args.output_path)
    out.mkdir(parents=True, exist_ok=True)

    os.environ["SGLANG_TEST_NUM_LAYERS"] = str(args.num_layers)
    os.environ.setdefault("SGLANG_LOAD_FORMAT", "dummy")
    os.environ.setdefault("SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD", "0")
    os.environ.setdefault("SGLANG_JIT_DEEPGEMM_PRECOMPILE", "0")
    os.environ.setdefault("SGLANG_JIT_DEEPGEMM_FAST_WARMUP", "1")

    cases = [tuple(item) for item in json.loads(args.cases_json)]
    t0 = time.perf_counter()
    runner = cmm.load_model_runner(
        model_path="deepseek-ai/DeepSeek-V3",
        head_num=args.num_heads,
        kv_cache_dtype=args.kv_cache_dtype,
        attention_backend=args.attention_backend,
        device="cuda:0",
        gemm_type=args.gemm_type,
    )
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    per_case = []
    total_run_start = time.perf_counter()
    attention_module = runner.model.model.layers[0].self_attn
    q_lora_rank = getattr(attention_module, "q_lora_rank", 1536) or 1536
    kv_lora_rank = getattr(attention_module, "kv_lora_rank", 512)
    qk_rope_head_dim = getattr(attention_module, "qk_rope_head_dim", 64)
    qkv_latent_dim = q_lora_rank + kv_lora_rank + qk_rope_head_dim

    def dummy_qkv_latent_func(h, fb):
        return torch.randn(h.shape[0], qkv_latent_dim, dtype=h.dtype, device=h.device)

    for batch_size, seq_length, is_prefill in cases:
        start = time.perf_counter()
        if is_prefill:
            ok = cmm._run_prefill(
                model_runner=runner,
                attention_module=attention_module,
                batch_size=batch_size,
                seq_length=seq_length,
                head_num=args.num_heads,
                num_warmup=args.warmup,
                num_iterations=args.iterations,
                device="cuda:0",
                output_path=str(out),
                dummy_qkv_latent_func=dummy_qkv_latent_func,
                attn_type="mla",
                model_path="deepseek-ai/DeepSeek-V3",
                architecture="DeepseekV3ForCausalLM",
                backend_name=runner.server_args.attention_backend,
                version="timing",
                device_name=torch.cuda.get_device_name("cuda:0"),
                log_mla_dtype="fp8_block",
                log_kv_dtype="fp8",
                log_gemm_type="fp8_block",
            )
        else:
            ok = cmm._run_decode(
                model_runner=runner,
                attention_module=attention_module,
                batch_size=batch_size,
                seq_length=seq_length,
                head_num=args.num_heads,
                num_warmup=args.warmup,
                num_iterations=args.iterations,
                device="cuda:0",
                output_path=str(out),
                dummy_qkv_latent_func=dummy_qkv_latent_func,
                attn_type="mla",
                model_path="deepseek-ai/DeepSeek-V3",
                architecture="DeepseekV3ForCausalLM",
                backend_name=runner.server_args.attention_backend,
                version="timing",
                device_name=torch.cuda.get_device_name("cuda:0"),
                log_mla_dtype="fp8_block",
                log_kv_dtype="fp8",
                log_gemm_type="fp8_block",
            )
        torch.cuda.synchronize()
        end = time.perf_counter()
        per_case.append(
            {
                "batch_size": batch_size,
                "seq_length": seq_length,
                "phase": "prefill" if is_prefill else "decode",
                "ok": bool(ok),
                "wall_s": end - start,
            }
        )
    total_run_end = time.perf_counter()
    result = {
        "load_model_runner_s": t1 - t0,
        "case_run_s": total_run_end - total_run_start,
        "total_s": total_run_end - t0,
        "num_cases": len(cases),
        "warmup": args.warmup,
        "iterations": args.iterations,
        "server_args": {
            "attention_backend": runner.server_args.attention_backend,
            "kv_cache_dtype": getattr(runner.server_args, "kv_cache_dtype", None),
            "quantization": getattr(runner.server_args, "quantization", None),
        },
        "per_case": per_case,
    }
    (out / "timing_result.json").write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps(result, indent=2, default=str))


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
    run_dir.mkdir(parents=True, exist_ok=True)
    inner_script = run_dir / "run_inside_container.py"
    write_inner_script(inner_script)
    cases = _parse_cases(args.prefill_cases, True) + _parse_cases(args.decode_cases, False)
    (run_dir / "cases.json").write_text(json.dumps(cases, indent=2))

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
    (run_dir / "docker_command.txt").write_text(" ".join(docker_cmd) + "\n")
    completed = subprocess.run(docker_cmd, text=True, capture_output=True)
    (run_dir / "container.stdout.log").write_text(completed.stdout)
    (run_dir / "container.stderr.log").write_text(completed.stderr)
    (run_dir / "status.json").write_text(
        json.dumps({"returncode": completed.returncode, "run_dir": str(run_dir)}, indent=2)
    )
    print(run_dir)
    print(completed.stdout[-4000:])
    if completed.returncode != 0:
        print(completed.stderr[-4000:])
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
