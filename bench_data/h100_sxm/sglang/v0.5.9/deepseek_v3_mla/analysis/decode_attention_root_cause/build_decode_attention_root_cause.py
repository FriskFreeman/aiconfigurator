#!/usr/bin/env python3
"""Build decode attention root-cause analysis artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd


BENCH_ROOT = Path("bench_data/h100_sxm/sglang/v0.5.9/deepseek_v3_mla")
SINGLE_OP_DIR = BENCH_ROOT / "analysis" / "decode_single_op_kernel_aic_error_analysis"
MODULE_DIR = BENCH_ROOT / "analysis" / "decode_mla_kernel_envelope_aic_compare"
OUT_DIR = BENCH_ROOT / "analysis" / "decode_attention_root_cause"
GEN_MLA_PERF = Path("src/aiconfigurator/systems/data/h100_sxm/sglang/0.5.9/generation_mla_perf.txt")


def load_perf_grid() -> pd.DataFrame:
    rows: list[dict] = []
    with GEN_MLA_PERF.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            isl = int(row["isl"])
            step = int(row["step"])
            rows.append(
                {
                    "kernel_source": row["kernel_source"],
                    "mla_dtype": row["mla_dtype"],
                    "kv_cache_dtype": row["kv_cache_dtype"],
                    "num_heads": int(row["num_heads"]) if row.get("num_heads") else 128 // int(row["tp_size"]),
                    "batch_size": int(row["batch_size"]),
                    "isl": isl,
                    "step": step,
                    "s_total": isl + step,
                    "latency_ms": float(row["latency"]),
                }
            )
    return pd.DataFrame(rows)


def build_case_compare() -> pd.DataFrame:
    op = pd.read_csv(SINGLE_OP_DIR / "decode_single_op_kernel_vs_aic.csv")
    attn = op[op["op_name"].eq("generation_attention")].copy()
    module_segments = pd.read_csv(MODULE_DIR / "decode_real_mla_layer_segments.csv")

    support_rows = []
    for _, row in module_segments.iterrows():
        semantic = json.loads(row["semantic_time_json"])
        support_rows.append(
            {
                "tag": row["tag"],
                "real_fa3_prepare_ms": semantic.get("fa3_prepare", 0.0),
                "real_fa3_attention_ms": semantic.get("fa3_attention", 0.0),
                "real_fa3_combine_ms": semantic.get("fa3_combine", 0.0),
                "real_rotary_ms": semantic.get("rotary_emb", 0.0),
                "real_set_mla_kv_buffer_ms": semantic.get("set_mla_kv_buffer", 0.0),
            }
        )
    support = pd.DataFrame(support_rows).groupby("tag", as_index=False).mean(numeric_only=True)

    out = attn.merge(support, on="tag", how="left")
    out["real_support_ms"] = out["real_rotary_ms"] + out["real_set_mla_kv_buffer_ms"]
    out["real_fa3_kernel_sum_rebuilt_ms"] = (
        out["real_fa3_prepare_ms"] + out["real_fa3_attention_ms"] + out["real_fa3_combine_ms"]
    )
    out["aic_over_real_fa3_ratio"] = out["aic_latency_ms"] / out["real_kernel_ms"]
    out["aic_over_real_with_support_ratio"] = out["aic_latency_ms"] / out["real_with_support_ms"]
    out["support_reduces_gap_pct_points"] = out["gap_pct"] - out["gap_vs_support_pct"]
    out["s_eq_kv_len_delta_pct_of_aic"] = (
        out["attention_s_eq_kv_len_delta_vs_aic_ms"] / out["aic_latency_ms"] * 100.0
    )
    return out[
        [
            "tag",
            "batch_size",
            "kv_len",
            "s",
            "num_heads",
            "real_fa3_prepare_ms",
            "real_fa3_attention_ms",
            "real_fa3_combine_ms",
            "real_kernel_ms",
            "real_rotary_ms",
            "real_set_mla_kv_buffer_ms",
            "real_with_support_ms",
            "aic_latency_ms",
            "aic_attention_s_eq_kv_len_ms",
            "gap_pct",
            "gap_vs_support_pct",
            "attention_s_eq_kv_len_gap_pct",
            "aic_over_real_fa3_ratio",
            "aic_over_real_with_support_ratio",
            "support_reduces_gap_pct_points",
            "s_eq_kv_len_delta_pct_of_aic",
            "trace_lookup_method",
            "trace_z_left",
            "trace_z_right",
        ]
    ]


def extract_relevant_grid(case_compare: pd.DataFrame, perf_grid: pd.DataFrame) -> pd.DataFrame:
    keys = set()
    for row in case_compare.itertuples(index=False):
        for s in {int(row.kv_len), int(row.s), int(row.trace_z_left), int(row.trace_z_right)}:
            keys.add((int(row.batch_size), int(row.num_heads), s))
    grid = perf_grid[
        perf_grid.apply(lambda r: (int(r["batch_size"]), int(r["num_heads"]), int(r["s_total"])) in keys, axis=1)
    ].copy()
    return grid.sort_values(["num_heads", "batch_size", "s_total", "kv_cache_dtype"]).reset_index(drop=True)


def write_readme(case_compare: pd.DataFrame, perf_grid_extract: pd.DataFrame) -> None:
    pure_mape = case_compare["gap_pct"].abs().mean()
    support_mape = case_compare["gap_vs_support_pct"].abs().mean()
    s_diag_mape = case_compare["attention_s_eq_kv_len_gap_pct"].abs().mean()
    mean_ratio = case_compare["aic_over_real_fa3_ratio"].mean()
    mean_support_ratio = case_compare["aic_over_real_with_support_ratio"].mean()
    mean_s_delta_pct = case_compare["s_eq_kv_len_delta_pct_of_aic"].abs().mean()
    max_gap = case_compare.sort_values("gap_pct", key=lambda s: s.abs(), ascending=False).iloc[0]
    kernel_sources = ", ".join(sorted(perf_grid_extract["kernel_source"].dropna().unique()))

    lines = [
        "# Decode Attention Root-Cause Analysis",
        "",
        "## 结论",
        "",
        f"- AIC `generation_attention` 与实机 pure FA3 attention 的 MAPE 为 `{pure_mape:.2f}%`，即平均约 `{mean_ratio:.2f}x`；把 rotary 与 KV write support 加入实机侧后 MAPE 仍有 `{support_mape:.2f}%`，平均约 `{mean_support_ratio:.2f}x`。",
        f"- 最大相对差异出现在 `{max_gap['tag']}`：实机 pure FA3 `{max_gap['real_kernel_ms']:.6f}ms`，AIC `{max_gap['aic_latency_ms']:.6f}ms`，gap `{max_gap['gap_pct']:.2f}%`。",
        f"- `s=kv_len+1` 插值不是主因：诊断性改查 `s=kv_len` 后 attention MAPE 为 `{s_diag_mape:.2f}%`，AIC latency 平均只改变 `{mean_s_delta_pct:.4f}%`。",
        "- 更可信的主因是采集/实机口径不同：AIC `generation_mla_perf.txt` 来自 collector 手工构造的 `RadixAttention` decode 微基准，实机对比取的是完整 SGLang CUDA Graph 中 MLA body 内的 FA3 backend kernel 三段。",
        "- 当前 AIC generation_mla 数据本身就是较大的数值，不是 SDK 额外放大；PerfDatabase 只是按 `(num_heads,b,s)` 读取或近邻插值这些 collector 数据。",
        "",
        "## 实机口径",
        "",
        "- 实机默认 FA3 decode 的 attention 被解析为 `prepare_varlen_num_blocks_kernel + device_kernel + device_kernel`。",
        "- 这三段分别在 CSV 中记录为 `fa3_prepare`、`fa3_attention`、`fa3_combine`；rotary 与 `set_mla_kv_buffer` 单独列出为 support，不计入主 attention 单算子口径。",
        "- SGLang 0.5.9 `forward_mla.py` 中 decode MLA 调用 `self.attn_mqa(...)` 后再做 `attn_output.view(...)` 与后续 BMM/o_proj；`flashattention_backend.py::forward_decode()` 在 `self.use_mla` 分支中调用 `flash_attn_with_kvcache(q=q_rope, k_cache=k_rope_cache, v_cache=c_kv_cache, qv=q_nope, ...)`。",
        "",
        "## Collector 口径",
        "",
        "- `collector/sglang/collect_mla.py` 的 generation 分支手工构造 `ForwardBatch(DECODE)`、预填 KV pool，然后调用 `RadixAttention.forward()` 并用 `benchmark_with_power()` 统计整次 layer 调用。",
        "- 采集输出 `op_name=mla_generation`，`kernel_source=flash_attention`，`isl=1`，`step=input_len`；PerfDatabase 加载时把 `s_total=isl+step` 作为查询坐标。",
        f"- 当前相关 perf 数据的 `kernel_source` 为 `{kernel_sources}`，说明 AIC 查询确实来自 flash_attention collector 数据。",
        "",
        "## SDK 查询",
        "",
        "- `ops.GenerationMLA.query()` 调用 `PerfDatabase.query_generation_mla(batch_size, s, num_heads, kvcache_quant_mode)`。",
        "- `query_generation_mla()` 在 silicon mode 下读取 `_generation_mla_data[kvcache_quant_mode][num_heads][b][s]`，通过 `_interp_3d(..., 'bilinear')` 查询。",
        "- 本组 case 的 `b` 与 `num_heads=128` 都精确命中；`s=kv_len+1` 位于相邻 grid 点之间，但诊断表明 +1 插值几乎不改变结果。",
        "",
        "## 判断",
        "",
        "- 已排除或弱化：module 边界、support kernel 漏计、`s=kv_len+1` 插值、SDK 额外缩放。",
        "- 仍需谨慎：collector 的 `RadixAttention.forward()` 计时是否包含比实机解析的三段 FA3 kernel 更多的 backend 内部工作，或者 collector 构造的 q/k/v/KV pool metadata 是否触发了比真实 CUDA Graph replay 更慢的 FA3 路径。仅靠现有 trace 不能完全拆开 collector 内部 kernel 组成。",
        "- 因此，若要修正 AIC 单算子 fallback，优先方案不是在 SDK query 端调参，而是重新定义/重采 `generation_mla_perf` 的 collector 口径，使其与实机 FA3 attention kernel 组件对齐，或者在 SDK 普通 DeepSeek fallback 中改用 module 级 `wideep_generation_mla` 数据。",
        "",
        "## 输出文件",
        "",
        "- `attention_case_compare.csv`: 每个 decode case 的实机 FA3/support/AIC attention 对照。",
        "- `generation_mla_perf_grid_extract.csv`: 与这些 case 相关的 generation_mla collector 原始格点。",
    ]
    (OUT_DIR / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    case_compare = build_case_compare()
    perf_grid = load_perf_grid()
    perf_grid_extract = extract_relevant_grid(case_compare, perf_grid)
    case_compare.to_csv(OUT_DIR / "attention_case_compare.csv", index=False)
    perf_grid_extract.to_csv(OUT_DIR / "generation_mla_perf_grid_extract.csv", index=False)
    write_readme(case_compare, perf_grid_extract)
    manifest = {
        "case_count": int(len(case_compare)),
        "perf_grid_extract_rows": int(len(perf_grid_extract)),
        "source_single_op_dir": str(SINGLE_OP_DIR),
        "source_module_dir": str(MODULE_DIR),
        "source_generation_mla_perf": str(GEN_MLA_PERF),
        "files": sorted(path.name for path in OUT_DIR.iterdir() if path.is_file()),
    }
    (OUT_DIR / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote attention root-cause artifacts to {OUT_DIR}")
    print(case_compare[["tag", "real_kernel_ms", "real_with_support_ms", "aic_latency_ms", "gap_pct"]].to_string(index=False))


if __name__ == "__main__":
    main()
