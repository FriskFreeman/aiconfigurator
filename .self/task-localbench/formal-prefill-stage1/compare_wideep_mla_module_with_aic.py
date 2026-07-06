#!/usr/bin/env python3
import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from aiconfigurator.sdk import common
from aiconfigurator.sdk.perf_database import PerfDatabase


@dataclass(frozen=True)
class ManifestEntry:
    tag: str
    status: str
    csv_path: Path
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
    parser.add_argument("--systems-root", default="src/aiconfigurator/systems")
    parser.add_argument("--system", default="h100_sxm")
    parser.add_argument("--backend", default="sglang")
    parser.add_argument("--version", default="0.5.9")
    parser.add_argument("--output-prefix", default="wideep_module_mla_aic_compare")
    parser.add_argument(
        "--attention-backend",
        default="fa3",
        choices=["fa3", "flashinfer"],
        help="WideEP MLA perf table backend to query.",
    )
    return parser.parse_args()


def parse_int_list_csv(text: str | None) -> list[int]:
    if text is None:
        return []
    text = text.strip()
    if not text:
        return []
    return [int(part) for part in text.split(",") if part.strip()]


def load_manifest(path: Path) -> list[ManifestEntry]:
    entries: list[ManifestEntry] = []
    with path.open() as f:
        for row in csv.DictReader(f):
            entries.append(
                ManifestEntry(
                    tag=row["tag"],
                    status=row["status"],
                    csv_path=Path(row["csv_path"]),
                    batch_size=int(row["batch_size"]),
                    fresh_lens=parse_int_list_csv(row["fresh_lens"]),
                    prefix_lens=parse_int_list_csv(row["prefix_lens"]),
                    tp_size=int(row["tp_size"]),
                    attention_backend=row["attention_backend"],
                )
            )
    return entries


def safe_json_loads(text: str | None) -> Any:
    if text is None:
        return None
    text = text.strip()
    if not text:
        return None
    return json.loads(text)


def all_equal(values: list[int]) -> bool:
    return len(set(values)) <= 1


def uniform_shape_from_row(
    row: dict[str, str],
    entry: ManifestEntry,
) -> tuple[bool, str, dict[str, Any]]:
    stage = row["stage"]
    chunk_prefix_lens = safe_json_loads(row.get("chunk_req_prefix_lens_json")) or []
    chunk_fresh_lens = safe_json_loads(row.get("chunk_req_fresh_lens_json")) or []
    chunk_total_lens = safe_json_loads(row.get("chunk_req_total_seq_lens_json")) or []

    if stage == "prefill":
        prefix_lens = [int(x) for x in chunk_prefix_lens] if chunk_prefix_lens else list(entry.prefix_lens)
        fresh_lens = [int(x) for x in chunk_fresh_lens] if chunk_fresh_lens else list(entry.fresh_lens)
        total_lens = (
            [int(x) for x in chunk_total_lens]
            if chunk_total_lens
            else [p + f for p, f in zip(prefix_lens, fresh_lens)]
        )
    elif stage == "decode":
        # Existing stage-1 decode rows are the one-token continuation after the
        # prefill request. The parser records empty chunk lists for decode, so
        # the manifest prompt shape is the stable source of the KV length.
        prefix_lens = list(entry.prefix_lens)
        fresh_lens = list(entry.fresh_lens)
        total_lens = [p + f for p, f in zip(prefix_lens, fresh_lens)]
    else:
        return False, "unsupported_stage", {}

    if not prefix_lens or not fresh_lens or not total_lens:
        return False, "missing_shape_lists", {}
    if not (len(prefix_lens) == len(fresh_lens) == len(total_lens)):
        return False, "shape_list_length_mismatch", {
            "prefix_lens": prefix_lens,
            "fresh_lens": fresh_lens,
            "total_lens": total_lens,
        }
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
        "batch_size": len(prefix_lens),
        "prefix_len": prefix_lens[0],
        "fresh_len": fresh_lens[0],
        "total_len": total_lens[0],
        "prefix_lens": prefix_lens,
        "fresh_lens": fresh_lens,
        "total_lens": total_lens,
    }


def child_timings(row: dict[str, str]) -> list[dict[str, Any]]:
    return safe_json_loads(row.get("child_timing_json")) or []


def kernel_summaries(row: dict[str, str]) -> list[dict[str, Any]]:
    return safe_json_loads(row.get("kernel_summary_json")) or []


def selected_child_names(stage: str) -> set[str]:
    if stage == "prefill":
        return {
            "q_a_layernorm",
            "q_b_proj",
            "kv_a_layernorm",
            "rotary_emb",
            "kv_b_proj",
            "attn_mha",
            "attn_mqa",
            "o_proj",
        }
    return {
        "q_a_layernorm",
        "kv_a_layernorm",
        "q_b_proj",
        "rotary_emb",
        "attn_mqa",
        "o_proj",
    }


def sum_child_gpu_ms(row: dict[str, str], names: set[str]) -> tuple[float, int]:
    total = 0.0
    count = 0
    for child in child_timings(row):
        if child.get("canonical_name") in names:
            total += float(child.get("gpu_kernel_time_sum_ms") or 0.0)
            count += int(child.get("gpu_kernel_count") or 0)
    return total, count


def sum_child_host_ms(row: dict[str, str], names: set[str]) -> float:
    return sum(
        float(child.get("host_duration_ms") or 0.0)
        for child in child_timings(row)
        if child.get("canonical_name") in names
    )


def sum_child_makespan_bounds_ms(row: dict[str, str], names: set[str]) -> float | None:
    selected = [child for child in child_timings(row) if child.get("canonical_name") in names]
    starts = [child.get("first_kernel_start_ns") for child in selected if child.get("first_kernel_start_ns") is not None]
    ends = [child.get("last_kernel_end_ns") for child in selected if child.get("last_kernel_end_ns") is not None]
    if not starts or not ends:
        return None
    return (max(int(x) for x in ends) - min(int(x) for x in starts)) / 1e6


def sum_kernel_summary_ms(row: dict[str, str], names: set[str]) -> tuple[float, int]:
    total = 0.0
    count = 0
    for summary in kernel_summaries(row):
        if summary.get("canonical_name") not in names:
            continue
        kernels = summary.get("top_gpu_kernels") or []
        total += sum(float(item.get("gpu_time_ms") or 0.0) for item in kernels)
        count += int(summary.get("gpu_kernel_count") or 0)
    return total, count


def query_wideep_mla_ms(
    database: PerfDatabase,
    stage: str,
    batch_size: int,
    fresh_len: int,
    prefix_len: int,
    total_len: int,
    tp_size: int,
    attention_backend: str,
) -> tuple[float | None, str | None]:
    try:
        if stage == "prefill":
            result = database.query_wideep_context_mla(
                b=batch_size,
                s=fresh_len,
                prefix=prefix_len,
                tp_size=tp_size,
                kvcache_quant_mode=common.KVCacheQuantMode.fp8,
                fmha_quant_mode=common.FMHAQuantMode.fp8_block,
                attention_backend=attention_backend,
            )
        else:
            result = database.query_wideep_generation_mla(
                b=batch_size,
                s=total_len + 1,
                tp_size=tp_size,
                kvcache_quant_mode=common.KVCacheQuantMode.fp8,
                fmha_quant_mode=common.FMHAQuantMode.fp8_block,
                attention_backend=attention_backend,
            )
        return float(result), getattr(result, "source", "silicon")
    except Exception:
        return None, None


def pct_error(sim: float | None, real: float | None) -> float | None:
    if sim is None or real in (None, 0, 0.0):
        return None
    return (sim - real) / real * 100.0


def abs_pct_error(sim: float | None, real: float | None) -> float | None:
    err = pct_error(sim, real)
    return abs(err) if err is not None else None


def real_rows_for_entry(entry: ManifestEntry) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with entry.csv_path.open() as f:
        for row in csv.DictReader(f):
            if row.get("stage") not in {"prefill", "decode"}:
                continue
            rows.append(row)
    return rows


def build_rows(
    entries: list[ManifestEntry],
    database: PerfDatabase,
    attention_backend: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    comparison: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for entry in entries:
        if entry.status != "ok":
            skipped.append({"tag": entry.tag, "reason": f"manifest_status_{entry.status}", "csv_path": str(entry.csv_path)})
            continue

        for row in real_rows_for_entry(entry):
            ok, reason, shape = uniform_shape_from_row(row, entry)
            if not ok:
                skipped.append(
                    {
                        "tag": entry.tag,
                        "run_instance_name": row.get("run_instance_name"),
                        "stage": row.get("stage"),
                        "layer_id": row.get("layer_id"),
                        "reason": reason,
                        "shape_info": json.dumps(shape, ensure_ascii=False),
                    }
                )
                continue

            stage = row["stage"]
            names = selected_child_names(stage)
            real_selected_gpu_sum_ms, real_selected_kernel_count = sum_child_gpu_ms(row, names)
            real_selected_host_sum_ms = sum_child_host_ms(row, names)
            real_selected_gpu_makespan_ms = sum_child_makespan_bounds_ms(row, names)
            real_selected_summary_sum_ms, real_selected_summary_kernel_count = sum_kernel_summary_ms(row, names)

            sim_ms, sim_source = query_wideep_mla_ms(
                database=database,
                stage=stage,
                batch_size=int(shape["batch_size"]),
                fresh_len=int(shape["fresh_len"]),
                prefix_len=int(shape["prefix_len"]),
                total_len=int(shape["total_len"]),
                tp_size=entry.tp_size,
                attention_backend=attention_backend,
            )

            out = {
                "tag": entry.tag,
                "run_instance_name": row["run_instance_name"],
                "stage": stage,
                "layer_id": int(row["layer_id"]),
                "event_order_index": int(row["event_order_index"]),
                "attention_module": row.get("attention_module"),
                "real_trace_attention_backend": entry.attention_backend,
                "aic_wideep_attention_backend": attention_backend,
                "tp_size": entry.tp_size,
                "local_num_heads": 128 // entry.tp_size,
                "batch_size": int(shape["batch_size"]),
                "fresh_len": int(shape["fresh_len"]),
                "prefix_len": int(shape["prefix_len"]),
                "total_seq_len": int(shape["total_len"]),
                "chunk_req_prefix_lens_json": json.dumps(shape["prefix_lens"], ensure_ascii=False),
                "chunk_req_fresh_lens_json": json.dumps(shape["fresh_lens"], ensure_ascii=False),
                "chunk_req_total_seq_lens_json": json.dumps(shape["total_lens"], ensure_ascii=False),
                "collector_prefill_aligned": row.get("collector_prefill_aligned"),
                "real_module_total_to_last_kernel_ms": float(row.get("module_total_to_last_kernel_ms") or 0.0),
                "real_module_host_duration_ms": float(row.get("module_host_duration_ms") or 0.0),
                "real_module_gpu_makespan_ms": float(row.get("module_gpu_makespan_ms") or 0.0),
                "real_module_gpu_kernel_time_sum_ms": float(row.get("module_gpu_kernel_time_sum_ms") or 0.0),
                "real_module_gpu_kernel_count": int(row.get("module_gpu_kernel_count") or 0),
                "real_module_first_kernel_name": row.get("module_first_kernel_name"),
                "real_module_last_kernel_name": row.get("module_last_kernel_name"),
                "real_wideep_aligned_child_names_json": json.dumps(sorted(names), ensure_ascii=False),
                "real_wideep_aligned_gpu_kernel_time_sum_ms": real_selected_gpu_sum_ms,
                "real_wideep_aligned_gpu_kernel_count": real_selected_kernel_count,
                "real_wideep_aligned_gpu_makespan_ms": real_selected_gpu_makespan_ms,
                "real_wideep_aligned_host_duration_sum_ms": real_selected_host_sum_ms,
                "real_wideep_aligned_kernel_summary_sum_ms": real_selected_summary_sum_ms,
                "real_wideep_aligned_kernel_summary_count": real_selected_summary_kernel_count,
                "aic_query_path": "query_wideep_context_mla" if stage == "prefill" else "query_wideep_generation_mla",
                "aic_kvcache_quant_mode": common.KVCacheQuantMode.fp8.name,
                "aic_fmha_quant_mode": common.FMHAQuantMode.fp8_block.name,
                "aic_wideep_mla_latency_ms": sim_ms,
                "aic_wideep_mla_source": sim_source,
            }

            for metric in [
                "real_module_total_to_last_kernel_ms",
                "real_module_host_duration_ms",
                "real_module_gpu_makespan_ms",
                "real_module_gpu_kernel_time_sum_ms",
                "real_wideep_aligned_gpu_kernel_time_sum_ms",
                "real_wideep_aligned_gpu_makespan_ms",
                "real_wideep_aligned_host_duration_sum_ms",
                "real_wideep_aligned_kernel_summary_sum_ms",
            ]:
                out[f"aic_error_pct_vs_{metric.removeprefix('real_')}"] = pct_error(sim_ms, out.get(metric))

            comparison.append(out)

    comparison.sort(
        key=lambda item: (
            item["tag"],
            0 if item["stage"] == "prefill" else 1,
            item["layer_id"],
            item["event_order_index"],
        )
    )
    return comparison, skipped


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def metric_mape(rows: list[dict[str, Any]], metric: str) -> float | None:
    values = [
        v
        for row in rows
        if (v := abs_pct_error(row.get("aic_wideep_mla_latency_ms"), row.get(metric))) is not None
    ]
    return mean(values)


def summarize(rows: list[dict[str, Any]], skipped: list[dict[str, Any]], attention_backend: str) -> str:
    lines = [
        "# WideEP MLA Module AIC 对比摘要",
        "",
        "## 范围",
        "",
        "- 本报告独立于先前的 `mla_kernel_aic_compare*` 单算子 attention 对比。",
        "- AIC 侧查询 `wideep_context_mla_perf.txt` / `wideep_generation_mla_perf.txt`，对应 SDK 的 `WideEPContextMLA` / `WideEPGenerationMLA`。",
        "- 实机侧复用已有 `MLA时延拆解.csv`，不重新跑实机。",
        "- 主表同时保留整块 module 口径与剔除 `fused_qkv_a_proj_with_mqa` 后的 WideEP 对齐子模块口径。",
        f"- 本次 AIC 查询使用 `attention_backend={attention_backend}`、`fmha_quant_mode=fp8_block`、`kvcache_quant_mode=fp8`。",
        "",
        "## 统计",
        "",
        f"- 可比较行数: `{len(rows)}`",
        f"- Prefill 行数: `{sum(1 for row in rows if row['stage'] == 'prefill')}`",
        f"- Decode 行数: `{sum(1 for row in rows if row['stage'] == 'decode')}`",
        f"- 跳过行数: `{len(skipped)}`",
        "",
    ]

    metrics = [
        ("整块 module 端到端到尾 kernel", "real_module_total_to_last_kernel_ms"),
        ("整块 module GPU makespan", "real_module_gpu_makespan_ms"),
        ("整块 module GPU kernel 加总", "real_module_gpu_kernel_time_sum_ms"),
        ("WideEP 对齐子模块 GPU makespan", "real_wideep_aligned_gpu_makespan_ms"),
        ("WideEP 对齐子模块 GPU kernel 加总", "real_wideep_aligned_gpu_kernel_time_sum_ms"),
        ("WideEP 对齐子模块 host duration 加总", "real_wideep_aligned_host_duration_sum_ms"),
    ]
    for stage in ["prefill", "decode"]:
        stage_rows = [row for row in rows if row["stage"] == stage]
        if not stage_rows:
            continue
        lines.extend([f"## {stage.capitalize()}", ""])
        for label, metric in metrics:
            mape = metric_mape(stage_rows, metric)
            if mape is None:
                lines.append(f"- `{label}`: 无有效统计")
            else:
                lines.append(f"- `{label}` MAPE: `{mape:.2f}%`")
        lines.append("")

    if skipped:
        counts: dict[str, int] = {}
        for row in skipped:
            counts[row["reason"]] = counts.get(row["reason"], 0) + 1
        lines.extend(["## 跳过原因", ""])
        for reason, count in sorted(counts.items()):
            lines.append(f"- `{reason}`: `{count}`")
        lines.append("")

    lines.extend(
        [
            "## 口径说明",
            "",
            "- `real_module_total_to_last_kernel_ms` 是 nsys 解析得到的每层 attention 模块 host 入口到最后一个关联 kernel 结束的时间。",
            "- `real_module_gpu_kernel_time_sum_ms` 是该模块下所有 GPU kernel 时长加总。",
            "- `real_wideep_aligned_*` 会从子模块中排除 `fused_qkv_a_proj_with_mqa`，因为 SDK 的 `WideEPDeepSeekModel` 将 qkv_a/downscale 作为单独 GEMM，而 `WideEPContextMLA` / `WideEPGenerationMLA` 本身覆盖 q/kv 后续投影、attention、o_proj 等部分。",
            "- 当前实机 trace 来自普通 SGLang engine 路径，不是 WideEP 运行时；因此该表是模块级 MLA 数据口径对照，而不是 WideEP 通信/调度保真验证。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    bench_root = Path(args.bench_root)
    analysis_dir = bench_root / "analysis" / "wideep_module_mla"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    database = PerfDatabase(
        system=args.system,
        backend=args.backend,
        version=args.version,
        systems_root=args.systems_root,
    )
    entries = load_manifest(Path(args.manifest))
    rows, skipped = build_rows(entries, database, args.attention_backend)

    comparison_csv = analysis_dir / f"{args.output_prefix}__comparison.csv"
    skipped_csv = analysis_dir / f"{args.output_prefix}__skipped.csv"
    summary_md = analysis_dir / f"{args.output_prefix}__summary.md"
    summary_json = analysis_dir / f"{args.output_prefix}__summary.json"

    write_csv(comparison_csv, rows)
    write_csv(skipped_csv, skipped)
    summary_md.write_text(summarize(rows, skipped, args.attention_backend))
    summary_json.write_text(
        json.dumps(
            {
                "comparison_csv": str(comparison_csv),
                "skipped_csv": str(skipped_csv),
                "summary_md": str(summary_md),
                "comparison_row_count": len(rows),
                "skipped_row_count": len(skipped),
                "attention_backend": args.attention_backend,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
    print(summary_json.read_text())


if __name__ == "__main__":
    main()
