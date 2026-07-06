#!/usr/bin/env python3
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

from aiconfigurator.sdk import common
from aiconfigurator.sdk.perf_database import PerfDatabase


@dataclass(frozen=True)
class ManifestEntry:
    tag: str
    status: str
    run_dir: str
    raw_run_dir: str
    csv_path: Path
    json_path: Path
    md_path: Path
    batch_size: int
    fresh_lens: list[int]
    prefix_lens: list[int]
    tp_size: int
    attention_backend: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="bench_data/h100_sxm/sglang/v0.5.9/deepseek_v3_mla/manifest.csv",
    )
    parser.add_argument(
        "--bench-root",
        default="bench_data/h100_sxm/sglang/v0.5.9/deepseek_v3_mla",
    )
    parser.add_argument(
        "--systems-root",
        default="src/aiconfigurator/systems",
    )
    parser.add_argument("--system", default="h100_sxm")
    parser.add_argument("--backend", default="sglang")
    parser.add_argument("--version", default="0.5.9")
    parser.add_argument(
        "--output-prefix",
        default="mla_kernel_aic_compare",
        help="Output filename prefix under <bench-root>/analysis/",
    )
    parser.add_argument(
        "--include-decode",
        action="store_true",
        help="Include decode attn_mqa rows in addition to prefill attn_mha rows.",
    )
    return parser.parse_args()


def parse_int_list_csv(text: str) -> list[int]:
    if text is None:
        return []
    text = text.strip()
    if not text:
        return []
    return [int(part) for part in text.split(",") if part.strip()]


def load_manifest(path: Path) -> list[ManifestEntry]:
    entries: list[ManifestEntry] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            entries.append(
                ManifestEntry(
                    tag=row["tag"],
                    status=row["status"],
                    run_dir=row["run_dir"],
                    raw_run_dir=row["raw_run_dir"],
                    csv_path=Path(row["csv_path"]),
                    json_path=Path(row["json_path"]),
                    md_path=Path(row["md_path"]),
                    batch_size=int(row["batch_size"]),
                    fresh_lens=parse_int_list_csv(row["fresh_lens"]),
                    prefix_lens=parse_int_list_csv(row["prefix_lens"]),
                    tp_size=int(row["tp_size"]),
                    attention_backend=row["attention_backend"],
                )
            )
    return entries


def safe_json_loads(text: str) -> object:
    if text is None:
        return None
    text = text.strip()
    if not text:
        return None
    return json.loads(text)


def all_equal(values: list[int]) -> bool:
    return len(set(values)) <= 1


def find_child_timing(row: dict[str, str], canonical_name: str) -> dict | None:
    children = safe_json_loads(row.get("child_timing_json")) or []
    for child in children:
        if child.get("canonical_name") == canonical_name:
            return child
    return None


def find_kernel_summary(row: dict[str, str], canonical_name: str) -> dict | None:
    summaries = safe_json_loads(row.get("kernel_summary_json")) or []
    for summary in summaries:
        if summary.get("canonical_name") == canonical_name:
            return summary
    return None


def kv_dtype_candidates() -> list[common.KVCacheQuantMode]:
    return [
        common.KVCacheQuantMode.bfloat16,
        common.KVCacheQuantMode.fp8,
    ]


def default_fmha_dtype() -> common.FMHAQuantMode:
    return common.FMHAQuantMode.bfloat16


def default_attention_head_size() -> int:
    return 128


def mean_abs_percent_error(rows: list[dict], sim_key: str, real_key: str) -> float | None:
    errors: list[float] = []
    for row in rows:
        sim = row.get(sim_key)
        real = row.get(real_key)
        if sim is None or real in (None, 0, 0.0):
            continue
        errors.append(abs((sim - real) / real) * 100.0)
    if not errors:
        return None
    return sum(errors) / len(errors)


def attention_n_kv(attention_module: str, num_heads: int) -> int:
    if attention_module == "attn_mqa":
        return 1
    return num_heads


def extract_real_attention_kernel_metrics(
    row: dict[str, str],
    attention_module: str,
) -> dict[str, object] | None:
    child_timing = find_child_timing(row, attention_module)
    kernel_summary = find_kernel_summary(row, attention_module)
    if child_timing is None:
        return None

    top_gpu_kernels = (kernel_summary or {}).get("top_gpu_kernels", [])
    kernel_summary_count = (kernel_summary or {}).get("gpu_kernel_count")
    child_kernel_count = child_timing.get("gpu_kernel_count")
    top_kernel_sum_ms = sum(float(item.get("gpu_time_ms", 0.0) or 0.0) for item in top_gpu_kernels)
    top_kernel_sum_ns = sum(int(item.get("gpu_time_ns", 0) or 0) for item in top_gpu_kernels)

    if (
        top_gpu_kernels
        and isinstance(kernel_summary_count, int)
        and kernel_summary_count > 0
        and len(top_gpu_kernels) >= kernel_summary_count
    ):
        real_attention_kernel_time_ms = top_kernel_sum_ms
        real_attention_kernel_time_ns = top_kernel_sum_ns
        real_attention_kernel_time_source = "kernel_summary_top_gpu_kernels_sum_complete"
    elif child_timing.get("gpu_kernel_time_sum_ms") is not None:
        real_attention_kernel_time_ms = float(child_timing["gpu_kernel_time_sum_ms"])
        real_attention_kernel_time_ns = int(child_timing.get("gpu_kernel_time_sum_ns", 0) or 0)
        real_attention_kernel_time_source = "child_timing_gpu_kernel_time_sum_fallback"
    else:
        return None

    dominant_kernel_name = top_gpu_kernels[0]["short_name"] if top_gpu_kernels else None
    return {
        "child_timing": child_timing,
        "kernel_summary": kernel_summary,
        "top_gpu_kernels": top_gpu_kernels,
        "dominant_kernel_name": dominant_kernel_name,
        "real_attention_kernel_time_ms": real_attention_kernel_time_ms,
        "real_attention_kernel_time_ns": real_attention_kernel_time_ns,
        "real_attention_kernel_time_source": real_attention_kernel_time_source,
        "real_attention_kernel_count": (
            kernel_summary_count if isinstance(kernel_summary_count, int) else child_kernel_count
        ),
    }


def uniform_shape_from_row(
    row: dict[str, str], manifest_entry: ManifestEntry
) -> tuple[bool, str, dict[str, int | list[int]]]:
    stage = row["stage"]
    chunk_prefix_lens = safe_json_loads(row.get("chunk_req_prefix_lens_json")) or []
    chunk_fresh_lens = safe_json_loads(row.get("chunk_req_fresh_lens_json")) or []
    chunk_total_lens = safe_json_loads(row.get("chunk_req_total_seq_lens_json")) or []

    if stage == "prefill":
        prefix_lens = [int(x) for x in chunk_prefix_lens] if chunk_prefix_lens else list(manifest_entry.prefix_lens)
        fresh_lens = [int(x) for x in chunk_fresh_lens] if chunk_fresh_lens else list(manifest_entry.fresh_lens)
        total_lens = [int(x) for x in chunk_total_lens] if chunk_total_lens else [
            p + f for p, f in zip(prefix_lens, fresh_lens)
        ]
    else:
        prefix_lens = list(manifest_entry.prefix_lens)
        fresh_lens = list(manifest_entry.fresh_lens)
        total_lens = [p + f for p, f in zip(prefix_lens, fresh_lens)]

    if not prefix_lens or not fresh_lens or not total_lens:
        return False, "missing_shape_lists", {}
    if not (len(prefix_lens) == len(fresh_lens) == len(total_lens)):
        return False, "shape_list_length_mismatch", {}
    if not all_equal(prefix_lens):
        return False, "non_uniform_prefix_lens", {
            "prefix_lens": prefix_lens,
            "fresh_lens": fresh_lens,
            "total_lens": total_lens,
        }
    if not all_equal(fresh_lens):
        return False, "non_uniform_fresh_lens", {
            "prefix_lens": prefix_lens,
            "fresh_lens": fresh_lens,
            "total_lens": total_lens,
        }
    if not all_equal(total_lens):
        return False, "non_uniform_total_lens", {
            "prefix_lens": prefix_lens,
            "fresh_lens": fresh_lens,
            "total_lens": total_lens,
        }

    return True, "ok", {
        "prefix_lens": prefix_lens,
        "fresh_lens": fresh_lens,
        "total_lens": total_lens,
        "prefix_len": prefix_lens[0],
        "fresh_len": fresh_lens[0],
        "total_len": total_lens[0],
        "batch_size": len(prefix_lens),
    }


def simulate_mla_latency_ms(
    database: PerfDatabase,
    stage: str,
    batch_size: int,
    fresh_len: int,
    prefix_len: int,
    total_len: int,
    num_heads: int,
    kv_cache_dtype: common.KVCacheQuantMode,
) -> tuple[float | None, str | None]:
    try:
        if stage == "prefill":
            result = database.query_context_mla(
                b=batch_size,
                s=fresh_len,
                prefix=prefix_len,
                num_heads=num_heads,
                kvcache_quant_mode=kv_cache_dtype,
                fmha_quant_mode=default_fmha_dtype(),
            )
            return float(result), getattr(result, "source", "silicon")

        result = database.query_generation_mla(
            b=batch_size,
            s=total_len + 1,
            num_heads=num_heads,
            kvcache_quant_mode=kv_cache_dtype,
        )
        return float(result), getattr(result, "source", "silicon")
    except Exception:
        return None, None


def simulate_attention_latency_ms(
    database: PerfDatabase,
    stage: str,
    attention_module: str,
    batch_size: int,
    fresh_len: int,
    prefix_len: int,
    total_len: int,
    num_heads: int,
    kv_cache_dtype: common.KVCacheQuantMode,
) -> tuple[float | None, str | None]:
    n_kv = attention_n_kv(attention_module, num_heads)
    try:
        if stage == "prefill":
            result = database.query_context_attention(
                b=batch_size,
                s=fresh_len,
                prefix=prefix_len,
                n=num_heads,
                n_kv=n_kv,
                kvcache_quant_mode=kv_cache_dtype,
                fmha_quant_mode=default_fmha_dtype(),
                head_size=default_attention_head_size(),
            )
            return float(result), getattr(result, "source", "silicon")

        result = database.query_generation_attention(
            b=batch_size,
            s=total_len + 1,
            n=num_heads,
            n_kv=n_kv,
            kvcache_quant_mode=kv_cache_dtype,
            head_size=default_attention_head_size(),
        )
        return float(result), getattr(result, "source", "silicon")
    except Exception:
        return None, None


def real_rows_for_manifest_entry(
    entry: ManifestEntry,
    include_decode: bool,
) -> list[dict]:
    selected: list[dict] = []
    with entry.csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            stage = row["stage"]
            attention_module = row["attention_module"]
            if stage == "prefill" and attention_module not in {"attn_mha", "attn_mqa"}:
                continue
            if stage == "decode":
                if not include_decode:
                    continue
                if attention_module != "attn_mqa":
                    continue
            if stage not in {"prefill", "decode"}:
                continue
            selected.append(row)
    return selected


def build_comparison_rows(
    manifest_entries: list[ManifestEntry],
    database: PerfDatabase,
    include_decode: bool,
) -> tuple[list[dict], list[dict]]:
    comparison_rows: list[dict] = []
    skipped_rows: list[dict] = []

    for entry in manifest_entries:
        if entry.status != "ok":
            skipped_rows.append(
                {
                    "tag": entry.tag,
                    "reason": f"manifest_status_{entry.status}",
                    "csv_path": str(entry.csv_path),
                }
            )
            continue

        for row in real_rows_for_manifest_entry(entry, include_decode):
            attention_module = row["attention_module"]
            real_metrics = extract_real_attention_kernel_metrics(row, attention_module)
            ok, shape_reason, shape_info = uniform_shape_from_row(row, entry)
            if not ok or real_metrics is None:
                skipped_rows.append(
                    {
                        "tag": entry.tag,
                        "run_instance_name": row.get("run_instance_name"),
                        "stage": row.get("stage"),
                        "layer_id": row.get("layer_id"),
                        "attention_module": attention_module,
                        "reason": "missing_attention_kernel_metrics" if real_metrics is None else shape_reason,
                        "shape_info": json.dumps(shape_info, ensure_ascii=False) if shape_info else "",
                    }
                )
                continue

            batch_size = int(shape_info["batch_size"])
            fresh_len = int(shape_info["fresh_len"])
            prefix_len = int(shape_info["prefix_len"])
            total_len = int(shape_info["total_len"])
            local_num_heads = 128 // int(entry.tp_size)

            child_timing = real_metrics["child_timing"]
            top_gpu_kernels = real_metrics["top_gpu_kernels"]
            dominant_kernel_name = real_metrics["dominant_kernel_name"]
            real_attention_kernel_time_ms = real_metrics["real_attention_kernel_time_ms"]
            real_attention_kernel_time_ns = real_metrics["real_attention_kernel_time_ns"]
            real_attention_kernel_time_source = real_metrics["real_attention_kernel_time_source"]
            real_attention_kernel_count = real_metrics["real_attention_kernel_count"]
            real_gpu_makespan_ms = child_timing.get("gpu_makespan_ms")
            real_gpu_kernel_time_sum_ms = child_timing.get("gpu_kernel_time_sum_ms")

            row_out = {
                "tag": entry.tag,
                "run_instance_name": row["run_instance_name"],
                "stage": row["stage"],
                "layer_id": int(row["layer_id"]),
                "event_order_index": int(row["event_order_index"]),
                "attention_module": attention_module,
                "attention_backend": entry.attention_backend,
                "tp_size": entry.tp_size,
                "local_num_heads": local_num_heads,
                "batch_size": batch_size,
                "fresh_len": fresh_len,
                "prefix_len": prefix_len,
                "total_seq_len": total_len,
                "attention_token_count": int(row["attention_token_count"]),
                "attention_total_tokens": int(row["attention_total_tokens"]),
                "real_attention_kernel_time_ns": real_attention_kernel_time_ns,
                "real_attention_kernel_time_ms": real_attention_kernel_time_ms,
                "real_attention_kernel_time_source": real_attention_kernel_time_source,
                "real_attention_kernel_count": real_attention_kernel_count,
                "real_gpu_makespan_ms": real_gpu_makespan_ms,
                "real_gpu_kernel_time_sum_ms": real_gpu_kernel_time_sum_ms,
                "real_gpu_kernel_count": child_timing.get("gpu_kernel_count"),
                "real_first_kernel_name": child_timing.get("first_kernel_name"),
                "real_last_kernel_name": child_timing.get("last_kernel_name"),
                "real_dominant_kernel_name": dominant_kernel_name,
                "real_top_gpu_kernels_json": json.dumps(top_gpu_kernels, ensure_ascii=False),
                "aic_mla_query_path": "query_context_mla" if row["stage"] == "prefill" else "query_generation_mla",
                "aic_attention_query_path": "query_context_attention" if row["stage"] == "prefill" else "query_generation_attention",
                "aic_attention_n_kv": attention_n_kv(attention_module, local_num_heads),
                "aic_attention_head_size": default_attention_head_size(),
                "aic_fmha_quant_mode": default_fmha_dtype().name,
            }

            for kv_dtype in kv_dtype_candidates():
                mla_latency_ms, mla_source = simulate_mla_latency_ms(
                    database=database,
                    stage=row["stage"],
                    batch_size=batch_size,
                    fresh_len=fresh_len,
                    prefix_len=prefix_len,
                    total_len=total_len,
                    num_heads=local_num_heads,
                    kv_cache_dtype=kv_dtype,
                )
                attention_latency_ms, attention_source = simulate_attention_latency_ms(
                    database=database,
                    stage=row["stage"],
                    attention_module=attention_module,
                    batch_size=batch_size,
                    fresh_len=fresh_len,
                    prefix_len=prefix_len,
                    total_len=total_len,
                    num_heads=local_num_heads,
                    kv_cache_dtype=kv_dtype,
                )

                for family, latency_ms, source in [
                    ("mla", mla_latency_ms, mla_source),
                    ("attention", attention_latency_ms, attention_source),
                ]:
                    key_prefix = f"aic_{family}_{kv_dtype.name}"
                    row_out[f"{key_prefix}_latency_ms"] = latency_ms
                    row_out[f"{key_prefix}_source"] = source
                    if latency_ms is not None and real_attention_kernel_time_ms not in (None, 0, 0.0):
                        row_out[f"{key_prefix}_error_pct_vs_real_attention_kernel"] = (
                            (latency_ms - real_attention_kernel_time_ms) / real_attention_kernel_time_ms * 100.0
                        )
                    else:
                        row_out[f"{key_prefix}_error_pct_vs_real_attention_kernel"] = None
                    if latency_ms is not None and real_gpu_makespan_ms not in (None, 0, 0.0):
                        row_out[f"{key_prefix}_error_pct_vs_makespan"] = (
                            (latency_ms - real_gpu_makespan_ms) / real_gpu_makespan_ms * 100.0
                        )
                    else:
                        row_out[f"{key_prefix}_error_pct_vs_makespan"] = None
                    if latency_ms is not None and real_gpu_kernel_time_sum_ms not in (None, 0, 0.0):
                        row_out[f"{key_prefix}_error_pct_vs_kernel_sum"] = (
                            (latency_ms - real_gpu_kernel_time_sum_ms) / real_gpu_kernel_time_sum_ms * 100.0
                        )
                    else:
                        row_out[f"{key_prefix}_error_pct_vs_kernel_sum"] = None

            comparison_rows.append(row_out)

    comparison_rows.sort(
        key=lambda item: (
            item["tag"],
            0 if item["stage"] == "prefill" else 1,
            item["layer_id"],
            item["event_order_index"],
        )
    )
    return comparison_rows, skipped_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_rows(rows: list[dict], skipped_rows: list[dict]) -> str:
    prefill_rows = [row for row in rows if row["stage"] == "prefill"]
    decode_rows = [row for row in rows if row["stage"] == "decode"]

    lines = [
        "# MLA Kernel AIC 对比摘要",
        "",
        "## 范围",
        "",
        "- 对比对象是实机 `MLA时延拆解.csv` 中 attention 子模块对应的底层 CUDA kernel 时间，而不是整个 `self_attn/attn_mha` 模块包络时间。",
        "- 主指标是 `real_attention_kernel_time_ms`：优先取 `kernel_summary_json` 中 attention 子模块全部 kernel 时间之和；若摘要列表不完整，则回退到 `child_timing_json.gpu_kernel_time_sum_ms`。",
        "- AIC 同时输出两套查询口径：`query_*_mla()` 与 `query_*_attention()`。",
        "- `real_gpu_makespan_ms` 与 `real_gpu_kernel_time_sum_ms` 继续保留为辅助参考。",
        "- 由于实机 run 目录里没有直接暴露最终 `kv_cache_dtype`，本次同时输出 `bfloat16` 和 `fp8` 两种 AIC 仿真列。",
        "",
        "## 统计",
        "",
        f"- 可比较总行数: `{len(rows)}`",
        f"- Prefill 行数: `{len(prefill_rows)}`",
        f"- Decode 行数: `{len(decode_rows)}`",
        f"- 跳过行数: `{len(skipped_rows)}`",
        "",
    ]

    for stage_name, stage_rows in [("Prefill", prefill_rows), ("Decode", decode_rows)]:
        if not stage_rows:
            continue
        metrics = [
            ("MLA / bfloat16", "aic_mla_bfloat16_latency_ms"),
            ("MLA / fp8", "aic_mla_fp8_latency_ms"),
            ("Attention / bfloat16", "aic_attention_bfloat16_latency_ms"),
            ("Attention / fp8", "aic_attention_fp8_latency_ms"),
        ]
        lines.extend([f"## {stage_name}", ""])
        for label, sim_key in metrics:
            mape = mean_abs_percent_error(stage_rows, sim_key, "real_attention_kernel_time_ms")
            if mape is not None:
                lines.append(f"- `{label}` 对 `real_attention_kernel_time_ms` 的平均绝对百分比误差: `{mape:.2f}%`")
            else:
                lines.append(f"- `{label}` 暂无有效统计。")
        lines.append("")

    if skipped_rows:
        lines.extend(["## 跳过原因", ""])
        reason_counts: dict[str, int] = {}
        for row in skipped_rows:
            reason = row["reason"]
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        for reason, count in sorted(reason_counts.items()):
            lines.append(f"- `{reason}`: `{count}`")
        lines.append("")

    lines.extend(
        [
            "## 解释",
            "",
            "- `query_context_mla()` 与 `query_context_attention()` 都会先按 `full_s = fresh_len + prefix_len` 查表，再做 prefix 修正。",
            "- `query_generation_mla()` 与 `query_generation_attention()` 都按 `s = prompt_len + 1` 查询 decode 点。",
            "- `query_*_attention()` 中，`attn_mha` 按 `n_kv = n`，`attn_mqa` 按 `n_kv = 1` 解释。",
            "- 因为本批正式 case 大多是“每个请求长度一致”的设置，本次可以把每个 chunk/layer 映射到一个统一的 `(batch_size, fresh_len, prefix_len)` AIC attention 查询点。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    bench_root = Path(args.bench_root)
    analysis_dir = bench_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    manifest_entries = load_manifest(manifest_path)
    database = PerfDatabase(
        system=args.system,
        backend=args.backend,
        version=args.version,
        systems_root=args.systems_root,
    )

    comparison_rows, skipped_rows = build_comparison_rows(
        manifest_entries=manifest_entries,
        database=database,
        include_decode=args.include_decode,
    )

    comparison_csv = analysis_dir / f"{args.output_prefix}__comparison.csv"
    skipped_csv = analysis_dir / f"{args.output_prefix}__skipped.csv"
    summary_md = analysis_dir / f"{args.output_prefix}__summary.md"

    write_csv(comparison_csv, comparison_rows)
    write_csv(skipped_csv, skipped_rows)
    summary_md.write_text(summarize_rows(comparison_rows, skipped_rows))

    print(json.dumps(
        {
            "comparison_csv": str(comparison_csv),
            "skipped_csv": str(skipped_csv),
            "summary_md": str(summary_md),
            "comparison_row_count": len(comparison_rows),
            "skipped_row_count": len(skipped_rows),
        },
        indent=2,
        ensure_ascii=False,
    ))


if __name__ == "__main__":
    main()
