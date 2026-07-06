#!/usr/bin/env python3
"""Summarize whether existing decode traces use fp8 DeepGEMM and fp8 KV writes."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


BENCH_ROOT = Path("bench_data/h100_sxm/sglang/v0.5.9/deepseek_v3_mla")
RAW_ROOT = BENCH_ROOT / "raw_runs"
OUT_DIR = BENCH_ROOT / "analysis" / "wideep_generation_mla_quant_check"


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    kernel_rows: list[dict] = []
    for csv_path in sorted(RAW_ROOT.glob("*decode*_backend_auto_cg_on*_profile_nsys/nsys/DecodeCudaGraphKernel汇总.csv")):
        if "flashmla" in str(csv_path):
            continue
        run_dir = csv_path.parents[1]
        df = pd.read_csv(csv_path)
        run_meta = read_json(run_dir / "run_meta.json")
        engine_kwargs = read_json(run_dir / "engine_kwargs.json")
        config = read_json(run_dir / "local_model" / "config.json")
        quant_config = config.get("quantization_config", {})

        deepgemm_fp8 = df["kernel_name"].str.contains("deep_gemm::sm90_fp8_gemm", regex=False, na=False)
        bf16_to_fp8_quant = df["kernel_name"].str.contains(
            "per_token_group_quant_8bit_kernel<__nv_bfloat16, __nv_fp8_e4m3", regex=False, na=False
        )
        kv_write = df["kernel_short_name"].eq("set_mla_kv_buffer_kernel")
        torch_bmm = df["kernel_short_name"].str.startswith("nvjet_tst_", na=False)

        shape_hits = {
            "qkv_downscale_2112x7168": df["kernel_name"].str.contains("2112.*7168", regex=True, na=False).sum(),
            "q_b_proj_24576x1536": df["kernel_name"].str.contains("24576.*1536", regex=True, na=False).sum(),
            "o_proj_7168x16384": df["kernel_name"].str.contains("7168.*16384", regex=True, na=False).sum(),
        }
        rows.append(
            {
                "run": run_dir.name,
                "batch_size": run_meta.get("batch_size"),
                "prefix_len": run_meta.get("prefix_len"),
                "attention_backend": run_meta.get("attention_backend"),
                "decode_attention_backend": run_meta.get("decode_attention_backend"),
                "cuda_graph_mode": run_meta.get("cuda_graph_mode"),
                "engine_kv_cache_dtype": engine_kwargs.get("kv_cache_dtype"),
                "engine_quantization": engine_kwargs.get("quantization"),
                "config_quant_method": quant_config.get("quant_method"),
                "config_quant_fmt": quant_config.get("fmt"),
                "config_weight_block_size": json.dumps(quant_config.get("weight_block_size"), ensure_ascii=False),
                "deepgemm_fp8_kernel_count": int(deepgemm_fp8.sum()),
                "bf16_to_fp8_quant_kernel_count": int(bf16_to_fp8_quant.sum()),
                "set_mla_kv_buffer_kernel_count": int(kv_write.sum()),
                "torch_nvjet_bmm_kernel_count": int(torch_bmm.sum()),
                **shape_hits,
            }
        )
        for _, krow in df[deepgemm_fp8 | bf16_to_fp8_quant | kv_write | torch_bmm].iterrows():
            kernel_rows.append(
                {
                    "run": run_dir.name,
                    "kernel_order_index": int(krow["kernel_order_index"]),
                    "kernel_duration_ms": float(krow["kernel_duration_ms"]),
                    "operator_category": krow["operator_category"],
                    "implementation": krow["implementation"],
                    "kernel_short_name": krow["kernel_short_name"],
                    "kernel_name": krow["kernel_name"],
                }
            )

    summary = pd.DataFrame(rows)
    detail = pd.DataFrame(kernel_rows)
    summary.to_csv(OUT_DIR / "real_decode_fp8_trace_summary.csv", index=False)
    detail.to_csv(OUT_DIR / "real_decode_fp8_kernel_evidence.csv", index=False)

    all_fp8 = bool(
        not summary.empty
        and summary["config_quant_method"].eq("fp8").all()
        and summary["deepgemm_fp8_kernel_count"].gt(0).all()
        and summary["bf16_to_fp8_quant_kernel_count"].gt(0).all()
        and summary["set_mla_kv_buffer_kernel_count"].gt(0).all()
    )
    lines = [
        "# Real Decode FP8 Trace Check",
        "",
        f"- Checked runs: `{len(summary)}`",
        f"- All checked runs have fp8 config + fp8 DeepGEMM + bf16-to-fp8 quant + MLA KV write kernels: `{all_fp8}`",
        "- `engine_kwargs.json` does not explicitly store `kv_cache_dtype`, so KV-cache fp8 is inferred from the decode MLA source path and `set_mla_kv_buffer_kernel` fed by fp8 quantized buffers, plus SGLang source code.",
        "",
        "## Evidence Files",
        "",
        "- `real_decode_fp8_trace_summary.csv`",
        "- `real_decode_fp8_kernel_evidence.csv`",
    ]
    (OUT_DIR / "real_decode_fp8_trace_check.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"Wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
