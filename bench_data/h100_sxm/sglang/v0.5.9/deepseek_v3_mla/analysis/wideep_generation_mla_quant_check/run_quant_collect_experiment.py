#!/usr/bin/env python3
"""Run a tiny WideEP generation MLA collector sweep with explicit precision modes.

This script is intended to run inside the SGLang 0.5.9 Docker image with the
repository mounted at /workspace and an output directory mounted at /out.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def parse_case(text: str) -> tuple[int, int]:
    b_text, s_text = text.split("x", 1)
    return int(b_text), int(s_text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("/workspace"))
    parser.add_argument("--output-path", type=Path, default=Path("/out"))
    parser.add_argument("--cases", nargs="+", default=["1x512", "4x2048", "8x4096"])
    parser.add_argument("--backend", default="fa3")
    parser.add_argument("--num-heads", type=int, default=128)
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-V3")
    parser.add_argument("--run-bf16", action="store_true")
    parser.add_argument("--run-fp8", action="store_true")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(repo_root / "collector" / "sglang"))

    import collect_mla_module as cmm  # noqa: WPS433

    args.output_path.mkdir(parents=True, exist_ok=True)
    cases = [parse_case(item) for item in args.cases]

    def patched_generation_cases(attn_type: str):
        out = []
        for batch_size, seq_len in cases:
            out.append(
                [
                    seq_len,
                    batch_size,
                    args.num_heads,
                    patched_generation_cases.kv_cache_dtype,
                    "bfloat16",
                    patched_generation_cases.gemm_type,
                ]
            )
        return out

    cmm.get_generation_test_cases = patched_generation_cases

    modes: list[tuple[str, str, str]] = []
    if args.run_bf16 or not args.run_fp8:
        modes.append(("bf16_actual_legacy_logged_fp8", "bfloat16", "bfloat16"))
    if args.run_fp8:
        modes.append(("fp8_actual", "fp8", "fp8_block"))

    manifest = {
        "sglang_version": None,
        "cases": [{"batch_size": b, "seq_len": s} for b, s in cases],
        "backend": args.backend,
        "num_heads": args.num_heads,
        "model": args.model,
        "modes": [],
    }
    try:
        import sglang

        manifest["sglang_version"] = getattr(sglang, "__version__", None)
    except Exception as exc:  # pragma: no cover - diagnostic only
        manifest["sglang_version_error"] = repr(exc)

    for mode_name, kv_cache_dtype, gemm_type in modes:
        mode_out = args.output_path / mode_name
        mode_out.mkdir(parents=True, exist_ok=True)
        patched_generation_cases.kv_cache_dtype = kv_cache_dtype
        patched_generation_cases.gemm_type = gemm_type
        os.environ.setdefault("SGLANG_LOAD_FORMAT", "dummy")
        os.environ.setdefault("SGLANG_TEST_NUM_LAYERS", "2")
        cmm.run_mla_module(
            attn_type="mla",
            head_num=args.num_heads,
            model_path=args.model,
            kv_cache_dtype=kv_cache_dtype,
            compute_dtype="bfloat16",
            gemm_type=gemm_type,
            is_prefill=False,
            gpu_id=0,
            output_path=str(mode_out),
            attention_backend=args.backend,
        )
        manifest["modes"].append(
            {
                "mode_name": mode_name,
                "kv_cache_dtype_arg": kv_cache_dtype,
                "compute_dtype_arg": "bfloat16",
                "gemm_type_arg": gemm_type,
                "output_file": str(mode_out / "wideep_generation_mla_perf.txt"),
            }
        )

    (args.output_path / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
