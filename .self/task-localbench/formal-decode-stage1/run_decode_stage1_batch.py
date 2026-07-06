#!/usr/bin/env python3
import argparse
import csv
import json
import re
import shutil
import sqlite3
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path


@dataclass(frozen=True)
class DecodeCase:
    tag: str
    batch_size: int
    kv_len: int
    tp_size: int = 1
    num_layers: int = 5
    attention_backend: str = "auto"
    decode_attention_backend: str = "auto"
    warmup_runs: int = 1
    layerwise_marker: bool = True
    profile_mode: str = "nsys"
    nsys_sample: str = "none"
    nsys_cpuctxsw: str = "none"
    mem_fraction_static: float = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-id", type=int, default=7)
    parser.add_argument(
        "--run-root",
        default=".self/task-localbench/formal-decode-stage1",
    )
    parser.add_argument(
        "--bench-root",
        default="bench_data/h100_sxm/sglang/v0.5.9/deepseek_v3_mla",
    )
    parser.add_argument(
        "--case-group",
        choices=["smoke_decode_only", "decode_only", "smoke_flashmla", "flashmla_decode_only"],
        default="smoke_decode_only",
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--skip-existing-ok", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-chunked-kv", action="store_true")
    parser.add_argument(
        "--max-total-kv-tokens",
        type=int,
        default=131072,
        help="Reject cases with batch_size * kv_len above this value unless --allow-chunked-kv is set.",
    )
    parser.add_argument(
        "--nsys-sample",
        choices=["process-tree", "system-wide", "none"],
        default=None,
    )
    parser.add_argument(
        "--nsys-cpuctxsw",
        choices=["process-tree", "system-wide", "none"],
        default=None,
    )
    return parser.parse_args()


def build_case_group(case_group: str) -> list[DecodeCase]:
    if case_group == "smoke_decode_only":
        return [
            DecodeCase(
                tag="decode_smoke_b4p512_cg_on",
                batch_size=4,
                kv_len=512,
                num_layers=3,
            ),
            DecodeCase(
                tag="decode_smoke_b8p2048_cg_on",
                batch_size=8,
                kv_len=2048,
                num_layers=3,
            ),
        ]

    if case_group == "smoke_flashmla":
        return [
            DecodeCase(
                tag="decode_flashmla_smoke_b4p512_cg_on",
                batch_size=4,
                kv_len=512,
                num_layers=3,
                decode_attention_backend="flashmla",
            ),
            DecodeCase(
                tag="decode_flashmla_smoke_b8p2048_cg_on",
                batch_size=8,
                kv_len=2048,
                num_layers=3,
                decode_attention_backend="flashmla",
            ),
        ]

    if case_group == "decode_only":
        return [
            DecodeCase(tag="decode_b1p512_cg_on", batch_size=1, kv_len=512),
            DecodeCase(tag="decode_b4p512_cg_on", batch_size=4, kv_len=512),
            DecodeCase(tag="decode_b32p512_cg_on", batch_size=32, kv_len=512),
            DecodeCase(tag="decode_b4p2048_cg_on", batch_size=4, kv_len=2048),
            DecodeCase(tag="decode_b16p2048_cg_on", batch_size=16, kv_len=2048),
            DecodeCase(tag="decode_b4p4096_cg_on", batch_size=4, kv_len=4096),
            DecodeCase(tag="decode_b8p4096_cg_on", batch_size=8, kv_len=4096),
            DecodeCase(tag="decode_b4p8192_cg_on", batch_size=4, kv_len=8192),
            DecodeCase(tag="decode_b2p16384_cg_on", batch_size=2, kv_len=16384),
        ]

    if case_group == "flashmla_decode_only":
        return [
            DecodeCase(tag="decode_flashmla_b1p512_cg_on", batch_size=1, kv_len=512, decode_attention_backend="flashmla"),
            DecodeCase(tag="decode_flashmla_b4p512_cg_on", batch_size=4, kv_len=512, decode_attention_backend="flashmla"),
            DecodeCase(tag="decode_flashmla_b16p512_cg_on", batch_size=16, kv_len=512, decode_attention_backend="flashmla"),
            DecodeCase(tag="decode_flashmla_b32p512_cg_on", batch_size=32, kv_len=512, decode_attention_backend="flashmla"),
            DecodeCase(tag="decode_flashmla_b64p512_cg_on", batch_size=64, kv_len=512, decode_attention_backend="flashmla", mem_fraction_static=0.6),
            DecodeCase(tag="decode_flashmla_b4p2048_cg_on", batch_size=4, kv_len=2048, decode_attention_backend="flashmla"),
            DecodeCase(tag="decode_flashmla_b16p2048_cg_on", batch_size=16, kv_len=2048, decode_attention_backend="flashmla"),
            DecodeCase(tag="decode_flashmla_b32p2048_cg_on", batch_size=32, kv_len=2048, decode_attention_backend="flashmla", mem_fraction_static=0.6),
            DecodeCase(tag="decode_flashmla_b4p4096_cg_on", batch_size=4, kv_len=4096, decode_attention_backend="flashmla"),
            DecodeCase(tag="decode_flashmla_b8p4096_cg_on", batch_size=8, kv_len=4096, decode_attention_backend="flashmla"),
            DecodeCase(tag="decode_flashmla_b16p4096_cg_on", batch_size=16, kv_len=4096, decode_attention_backend="flashmla", mem_fraction_static=0.6),
            DecodeCase(tag="decode_flashmla_b4p8192_cg_on", batch_size=4, kv_len=8192, decode_attention_backend="flashmla"),
            DecodeCase(tag="decode_flashmla_b8p8192_cg_on", batch_size=8, kv_len=8192, decode_attention_backend="flashmla", mem_fraction_static=0.6),
            DecodeCase(tag="decode_flashmla_b2p16384_cg_on", batch_size=2, kv_len=16384, decode_attention_backend="flashmla"),
            DecodeCase(tag="decode_flashmla_b4p16384_cg_on", batch_size=4, kv_len=16384, decode_attention_backend="flashmla", mem_fraction_static=0.6),
            DecodeCase(tag="decode_flashmla_b1p32768_cg_on", batch_size=1, kv_len=32768, decode_attention_backend="flashmla"),
            DecodeCase(tag="decode_flashmla_b2p32768_cg_on", batch_size=2, kv_len=32768, decode_attention_backend="flashmla", mem_fraction_static=0.6),
        ]

    raise ValueError(f"Unsupported case group: {case_group}")


def _context_length(case: DecodeCase) -> int:
    # SGLang v0.5.9 keeps a few internal reserve tokens beyond visible input.
    return max(case.kv_len + 16, 64)


def validate_case(case: DecodeCase, max_total_kv_tokens: int, allow_chunked_kv: bool) -> None:
    if case.batch_size <= 0:
        raise ValueError(f"Invalid batch_size for {case.tag}: {case.batch_size}")
    if case.kv_len <= 0:
        raise ValueError(f"Invalid kv_len for {case.tag}: {case.kv_len}")
    total_kv_tokens = case.batch_size * case.kv_len
    if total_kv_tokens > max_total_kv_tokens and not allow_chunked_kv:
        raise ValueError(
            f"{case.tag} has total_kv_tokens={total_kv_tokens}, above max_total_kv_tokens={max_total_kv_tokens}. "
            "This may trigger SGLang chunked_kv; pass --allow-chunked-kv to override."
        )


def build_command(case: DecodeCase, gpu_id: int, run_root: str, allow_chunked_kv: bool) -> list[str]:
    cmd = [
        "python",
        ".self/task-localbench/formal-prefill-stage1/run_formal_prefill_stage1.py",
        "--gpu-id",
        str(gpu_id),
        "--num-layers",
        str(case.num_layers),
        "--batch-size",
        str(case.batch_size),
        "--fresh-len",
        "1",
        "--prefix-len",
        str(case.kv_len),
        "--context-length",
        str(_context_length(case)),
        "--max-new-tokens",
        "2",
        "--warmup-runs",
        str(case.warmup_runs),
        "--profile-mode",
        case.profile_mode,
        "--cuda-graph-mode",
        "on",
        "--pcg-mode",
        "off",
        "--tp-size",
        str(case.tp_size),
        "--tag",
        case.tag,
        "--run-root",
        run_root,
        "--mem-fraction-static",
        str(case.mem_fraction_static),
        "--chunked-prefix-cache-threshold",
        str(max(case.batch_size * case.kv_len + case.batch_size + 1, 8192)),
        "--decode-empty-input-continuation",
        "--profile-by-stage",
        "--profile-num-steps",
        "1",
        "--profile-decode-only-stage",
        "--limit-cuda-graph-bs-to-batch",
    ]
    if case.attention_backend not in ("", "auto", "default"):
        cmd.extend(["--attention-backend", case.attention_backend])
    if case.decode_attention_backend not in ("", "auto", "default"):
        cmd.extend(["--decode-attention-backend", case.decode_attention_backend])
    if case.layerwise_marker:
        cmd.append("--layerwise-marker")
    else:
        cmd.append("--no-layerwise-marker")
    if allow_chunked_kv:
        cmd.append("--allow-chunked-kv")
    if case.nsys_sample != "none":
        cmd.extend(["--nsys-sample", case.nsys_sample])
    if case.nsys_cpuctxsw != "none":
        cmd.extend(["--nsys-cpuctxsw", case.nsys_cpuctxsw])
    return cmd


def copy_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


_NVJET_BMM_KERNEL_NAMES = {
    "nvjet_tst_256x8_64x6_2x1_v_bz_TNT",
    "nvjet_tst_128x8_64x12_1x1_h_bz_TNT",
    "nvjet_tst_256x16_64x6_2x1_v_bz_TNT",
    "nvjet_tst_128x16_64x11_1x1_h_bz_TNT",
    "nvjet_tst_256x32_64x5_2x1_v_bz_TNT",
    "nvjet_tst_128x32_64x10_1x1_h_bz_TNT",
    "nvjet_tst_64x32_64x16_4x1_v_bz_splitK_TNT",
    "nvjet_tst_256x64_64x5_2x1_v_bz_TNT",
    "nvjet_tst_128x64_64x8_1x1_h_bz_TNT",
    "nvjet_tst_64x32_64x16_4x2_h_bz_splitK_TNT",
}


def _classify_kernel(name: str) -> tuple[str, str]:
    lowered = name.lower()
    if name in _NVJET_BMM_KERNEL_NAMES:
        return "bmm", "torch"
    if "deep_gemm" in lowered:
        return "gemm", "deepgemm"
    if "nvjet" in lowered:
        return "attention", "nvjet_tst"
    if (
        "flash_mla" in lowered
        or "flashmla" in lowered
        or "flash_fwd_mla" in lowered
        or "get_mla_metadata" in lowered
        or "create_flashmla" in lowered
    ):
        return "attention", "flashmla"
    if "flashattn" in name or "flash::" in lowered:
        return "attention", "fa3"
    if "flashinfer" in lowered:
        if "norm" in lowered:
            return "norm", "flashinfer"
        if "activation" in lowered:
            return "activation", "flashinfer"
        return "support", "flashinfer"
    if "set_mla" in lowered:
        return "attention_support", "sglang_mla"
    if "quant" in lowered:
        return "quant", "sglang"
    if "rmsnorm" in lowered or "norm" in lowered:
        return "norm", "torch_or_triton"
    if "reduce" in lowered or "argmax" in lowered:
        return "sampling", "torch"
    if "triton" in lowered:
        return "triton_misc", "triton"
    if "at::native" in lowered or "at_cuda" in lowered:
        return "torch_misc", "torch"
    return "other", "unknown"


def ensure_decode_kernel_csv(run_dir: Path) -> Path | None:
    csv_path = run_dir / "nsys" / "DecodeCudaGraphKernel汇总.csv"
    if csv_path.exists():
        return csv_path

    sqlite_path = run_dir / "nsys" / "report.sqlite"
    if not sqlite_path.exists():
        return None

    run_meta = {}
    run_meta_path = run_dir / "run_meta.json"
    if run_meta_path.exists():
        try:
            run_meta = json.loads(run_meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            run_meta = {}

    con = sqlite3.connect(sqlite_path)
    try:
        tables = {
            row[0]
            for row in con.execute(
                "select name from sqlite_master where type = 'table'"
            )
        }
        if "CUPTI_ACTIVITY_KIND_KERNEL" not in tables or "StringIds" not in tables:
            return None
        rows = con.execute(
            """
            select
                k.start,
                k.end,
                k.deviceId,
                k.contextId,
                k.streamId,
                k.correlationId,
                k.graphNodeId,
                k.graphId,
                k.gridX,
                k.gridY,
                k.gridZ,
                k.blockX,
                k.blockY,
                k.blockZ,
                coalesce(d.value, s.value, m.value, '') as kernel_name,
                coalesce(s.value, d.value, m.value, '') as short_name
            from CUPTI_ACTIVITY_KIND_KERNEL k
            left join StringIds d on k.demangledName = d.id
            left join StringIds s on k.shortName = s.id
            left join StringIds m on k.mangledName = m.id
            order by k.start, k.end
            """
        ).fetchall()
    finally:
        con.close()

    if not rows:
        return None

    fieldnames = [
        "kernel_order_index",
        "run_instance_name",
        "stage",
        "batch_size",
        "kv_len",
        "total_kv_tokens",
        "tp_size",
        "attention_backend",
        "decode_attention_backend",
        "cuda_graph_mode",
        "cuda_graph_validated",
        "kernel_start_ns",
        "kernel_end_ns",
        "kernel_duration_ns",
        "kernel_duration_ms",
        "device_id",
        "context_id",
        "stream_id",
        "correlation_id",
        "graph_node_id",
        "graph_id",
        "grid_xyz",
        "block_xyz",
        "operator_category",
        "implementation",
        "kernel_name",
        "kernel_short_name",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, row in enumerate(rows):
            (
                start_ns,
                end_ns,
                device_id,
                context_id,
                stream_id,
                correlation_id,
                graph_node_id,
                graph_id,
                grid_x,
                grid_y,
                grid_z,
                block_x,
                block_y,
                block_z,
                kernel_name,
                short_name,
            ) = row
            duration_ns = int(end_ns) - int(start_ns)
            category, implementation = _classify_kernel(kernel_name or short_name or "")
            writer.writerow(
                {
                    "kernel_order_index": idx,
                    "run_instance_name": run_dir.name,
                    "stage": "decode",
                    "batch_size": run_meta.get("batch_size", ""),
                    "kv_len": run_meta.get("prefix_len", ""),
                    "total_kv_tokens": (
                        int(run_meta.get("batch_size", 0))
                        * int(run_meta.get("prefix_len", 0))
                        if run_meta.get("batch_size") is not None
                        and run_meta.get("prefix_len") is not None
                        else ""
                    ),
                    "tp_size": run_meta.get("tp_size", ""),
                    "attention_backend": run_meta.get("attention_backend", ""),
                    "decode_attention_backend": run_meta.get(
                        "decode_attention_backend", ""
                    ),
                    "cuda_graph_mode": run_meta.get("cuda_graph_mode", ""),
                    "cuda_graph_validated": "true",
                    "kernel_start_ns": start_ns,
                    "kernel_end_ns": end_ns,
                    "kernel_duration_ns": duration_ns,
                    "kernel_duration_ms": duration_ns / 1_000_000,
                    "device_id": device_id,
                    "context_id": context_id,
                    "stream_id": stream_id,
                    "correlation_id": correlation_id,
                    "graph_node_id": graph_node_id,
                    "graph_id": graph_id,
                    "grid_xyz": json.dumps([grid_x, grid_y, grid_z]),
                    "block_xyz": json.dumps([block_x, block_y, block_z]),
                    "operator_category": category,
                    "implementation": implementation,
                    "kernel_name": kernel_name,
                    "kernel_short_name": short_name,
                }
            )
    return csv_path


def archive_run(run_dir: Path, bench_root: Path) -> dict[str, str]:
    run_name = run_dir.name
    raw_root = bench_root / "raw_runs" / run_name
    raw_root.parent.mkdir(parents=True, exist_ok=True)
    if raw_root.exists():
        shutil.rmtree(raw_root)
    shutil.copytree(run_dir, raw_root)

    csv_root = bench_root / "decode_csv"
    csv_root.mkdir(parents=True, exist_ok=True)
    kernel_csv = ensure_decode_kernel_csv(run_dir)
    csv_dst = csv_root / f"{run_name}__DecodeCudaGraphKernel汇总.csv"
    json_dst = csv_root / f"{run_name}__MLA对齐解析.json"
    md_dst = csv_root / f"{run_name}__MLA对齐解析.md"
    if kernel_csv is not None:
        copy_if_exists(kernel_csv, csv_dst)
    else:
        copy_if_exists(run_dir / "nsys" / "MLA时延拆解.csv", csv_dst)
    copy_if_exists(run_dir / "nsys" / "MLA对齐解析.json", json_dst)
    copy_if_exists(run_dir / "nsys" / "MLA对齐解析.md", md_dst)

    return {
        "raw_run_dir": str(raw_root),
        "csv_path": str(csv_dst) if csv_dst.exists() else "",
        "json_path": str(json_dst) if json_dst.exists() else "",
        "md_path": str(md_dst) if md_dst.exists() else "",
    }


def validate_decode_artifacts(
    run_dir: Path,
    expected_decode_attention_backend: str | None = None,
) -> tuple[bool, str]:
    stderr_path = run_dir / "container.stderr.log"
    if not stderr_path.exists():
        return False, f"missing scheduler stderr log: {stderr_path}"
    stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace")
    if "Profiling starts for DECODE" not in stderr_text:
        return False, "missing SGLang profile start for real DECODE stage"
    if "Stop profiling-DECODE" not in stderr_text:
        return False, "missing SGLang profile stop for real DECODE stage"
    decode_cg_matches = re.findall(r"Decode batch.*?cuda graph: True", stderr_text)
    if not decode_cg_matches:
        return False, "missing scheduler evidence: Decode batch ... cuda graph: True"
    prefill_cg_matches = re.findall(r"Prefill batch.*?cuda graph: True", stderr_text)

    csv_path = ensure_decode_kernel_csv(run_dir)
    if csv_path is None:
        csv_path = run_dir / "nsys" / "MLA时延拆解.csv"
    if not csv_path.exists():
        return False, f"missing decode CSV: {csv_path}"

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return False, f"empty decode CSV: {csv_path}"

    stages = sorted({row.get("stage", "") for row in rows})
    if stages != ["decode"]:
        return False, f"expected only stage=decode, got stages={stages}"
    if csv_path.name == "MLA时延拆解.csv":
        attn_modules = sorted({row.get("attention_module", "") for row in rows})
        if attn_modules != ["attn_mqa"]:
            return False, f"expected only attention_module=attn_mqa, got {attn_modules}"
    else:
        implementations = {row.get("implementation", "") for row in rows}
        if expected_decode_attention_backend == "flashmla" and "flashmla" not in implementations:
            return False, (
                "decode backend expected flashmla, but kernel CSV does not contain flashmla; "
                f"implementations={sorted(implementations)}"
            )
        if "flashmla" not in implementations and "fa3" not in implementations:
            return False, (
                "decode kernel CSV does not contain expected attention kernels; "
                f"implementations={sorted(implementations)}"
            )
    return (
        True,
        "validated "
        f"{len(rows)} decode kernel rows; "
        f"decode_cuda_graph_true_logs={len(decode_cg_matches)}; "
        f"prefill_cuda_graph_true_logs={len(prefill_cg_matches)}",
    )


def normalize_decode_run_dir(run_dir: Path) -> Path:
    """Keep decode artifacts from inheriting the reused prefill runner prefix."""
    old_marker = "_prefill_stage1_"
    new_marker = "_decode_stage1_"
    if old_marker not in run_dir.name:
        return run_dir

    dst = run_dir.with_name(run_dir.name.replace(old_marker, new_marker, 1))
    if dst == run_dir:
        return run_dir
    if dst.exists():
        raise FileExistsError(f"Cannot rename {run_dir} to existing decode run dir {dst}")
    run_dir.rename(dst)
    return dst


def append_log(log_path: Path, message: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(message.rstrip() + "\n")


def write_manifest_row(manifest_path: Path, row: dict[str, str]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "tag",
        "status",
        "run_dir",
        "raw_run_dir",
        "csv_path",
        "json_path",
        "md_path",
        "batch_size",
        "kv_len",
        "total_kv_tokens",
        "tp_size",
        "attention_backend",
        "decode_attention_backend",
        "cuda_graph_mode",
    ]
    exists = manifest_path.exists()
    with manifest_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def load_existing_ok_tags(manifest_path: Path) -> set[str]:
    if not manifest_path.exists():
        return set()
    ok_tags: set[str] = set()
    with manifest_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status") != "ok" or not row.get("tag"):
                continue
            raw_run_dir = Path(row.get("raw_run_dir") or "")
            if not raw_run_dir.exists():
                continue
            valid, _ = validate_decode_artifacts(
                raw_run_dir, row.get("decode_attention_backend") or None
            )
            if valid:
                ok_tags.add(row["tag"])
    return ok_tags


def main() -> int:
    args = parse_args()
    root_dir = Path(__file__).resolve().parents[3]
    run_root = str((root_dir / args.run_root).resolve())
    bench_root = (root_dir / args.bench_root).resolve()
    bench_root.mkdir(parents=True, exist_ok=True)

    cases = build_case_group(args.case_group)
    if args.start_index:
        cases = cases[args.start_index :]
    if args.max_cases is not None:
        cases = cases[: args.max_cases]
    cases = [
        replace(
            case,
            nsys_sample=args.nsys_sample or case.nsys_sample,
            nsys_cpuctxsw=args.nsys_cpuctxsw or case.nsys_cpuctxsw,
        )
        for case in cases
    ]
    for case in cases:
        validate_case(case, args.max_total_kv_tokens, args.allow_chunked_kv)

    log_path = root_dir / ".self/task-localbench/formal-decode-stage1/decode批量运行记录.md"
    manifest_path = bench_root / "decode_manifest.csv"
    existing_ok_tags = load_existing_ok_tags(manifest_path) if args.skip_existing_ok else set()
    if existing_ok_tags:
        cases = [case for case in cases if case.tag not in existing_ok_tags]

    if not log_path.exists():
        log_path.write_text(
            "# Decode批量运行记录\n\n"
            "## 说明\n\n"
            "- 本文档记录 decode-only 场景的实机运行过程。\n"
            "- 每个请求先在 session 中写入等长 KV prefix；正式 profile 只包住 1-token continuation decode。\n"
            "- 原始 run 目录归档到 `bench_data/.../deepseek_v3_mla/raw_runs/`。\n"
            "- Decode 核心 `nsys` 结果归档到 `bench_data/.../deepseek_v3_mla/decode_csv/`。\n\n"
            "## 运行日志\n\n",
            encoding="utf-8",
        )

    append_log(
        log_path,
        f"### 新批次\n- case_group: `{args.case_group}`\n- start_index: `{args.start_index}`\n- max_cases: `{args.max_cases}`\n- skip_existing_ok: `{args.skip_existing_ok}`\n- bench_root: `{bench_root}`\n",
    )

    failures = 0
    for idx, case in enumerate(cases, start=1):
        cmd = build_command(case, args.gpu_id, run_root, args.allow_chunked_kv)
        append_log(log_path, f"- [{idx}/{len(cases)}] 开始 `{case.tag}`\n  - cmd: `{' '.join(cmd)}`")
        if args.dry_run:
            write_manifest_row(
                manifest_path,
                {
                    "tag": case.tag,
                    "status": "dry_run",
                    "batch_size": str(case.batch_size),
                    "kv_len": str(case.kv_len),
                    "total_kv_tokens": str(case.batch_size * case.kv_len),
                    "tp_size": str(case.tp_size),
                    "attention_backend": case.attention_backend,
                    "decode_attention_backend": case.decode_attention_backend,
                    "cuda_graph_mode": "on",
                },
            )
            continue

        completed = subprocess.run(cmd, text=True, capture_output=True)
        stdout = completed.stdout.strip()
        run_dir = Path(stdout.splitlines()[-1]) if stdout else None
        status = "ok"
        archive_info = {"raw_run_dir": "", "csv_path": "", "json_path": "", "md_path": ""}

        if completed.returncode != 0 or run_dir is None or not run_dir.exists():
            status = "failed"
            failures += 1
            append_log(
                log_path,
                f"  - 失败: returncode={completed.returncode}\n"
                f"  - stdout: `{completed.stdout.strip()}`\n"
                f"  - stderr: `{completed.stderr.strip()}`",
            )
        else:
            original_run_dir = run_dir
            run_dir = normalize_decode_run_dir(run_dir)
            valid, validation_message = validate_decode_artifacts(
                run_dir, case.decode_attention_backend
            )
            if valid:
                archive_info = archive_run(run_dir, bench_root)
                append_log(
                    log_path,
                    f"  - 成功: `{run_dir}`\n"
                    f"  - 原始runner输出目录: `{original_run_dir}`\n"
                    f"  - 校验: `{validation_message}`\n"
                    f"  - CSV: `{archive_info['csv_path']}`",
                )
            else:
                status = "failed_decode_validation"
                failures += 1
                append_log(
                    log_path,
                    f"  - Decode校验失败: `{run_dir}`\n"
                    f"  - 原始runner输出目录: `{original_run_dir}`\n"
                    f"  - reason: `{validation_message}`",
                )

        write_manifest_row(
            manifest_path,
            {
                "tag": case.tag,
                "status": status,
                "run_dir": str(run_dir) if run_dir is not None else "",
                **archive_info,
                "batch_size": str(case.batch_size),
                "kv_len": str(case.kv_len),
                "total_kv_tokens": str(case.batch_size * case.kv_len),
                "tp_size": str(case.tp_size),
                "attention_backend": case.attention_backend,
                "decode_attention_backend": case.decode_attention_backend,
                "cuda_graph_mode": "on",
            },
        )

    if not args.dry_run:
        append_log(log_path, f"\n## 本批次总结\n- total_cases: `{len(cases)}`\n- failures: `{failures}`\n")
    print(bench_root)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
