#!/usr/bin/env python3
import argparse
import csv
import json
import shutil
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path


@dataclass(frozen=True)
class AggCase:
    tag: str
    decode_batch_size: int
    decode_prefix_len: int
    prefill_batch_size: int
    prefill_fresh_len: int
    prefill_prefix_len: int
    num_layers: int = 6
    context_length: int = 0
    chunked_prefill_size: int = 8192
    max_prefill_tokens: int = 16384
    mixed_profile_steps: int = 1
    warmup_runs: int = 1
    attention_backend: str = "auto"
    decode_attention_backend: str = "auto"
    profile_mode: str = "nsys"
    cuda_graph_mode: str = "off"
    pcg_mode: str = "off"
    mem_fraction_static: float = 0.5
    tp_size: int = 1
    decode_max_new_tokens: int = 16
    prefill_max_new_tokens: int = 1
    decode_consume_delay_s: float = 0.0
    decode_start_timeout_s: float = 900.0
    decode_finish_timeout_s: float = 120.0
    deepgemm_precompile: bool = False
    deepgemm_fast_warmup: bool = True
    extra_env: tuple[str, ...] = ()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-ids-csv", default="4,5")
    parser.add_argument("--run-root", default=".self/task-localbench/formal-agg-stage2")
    parser.add_argument(
        "--bench-root",
        default="bench_data/h100_sxm/sglang/v0.5.9/deepseek_v3_mla",
    )
    parser.add_argument(
        "--case-group",
        choices=["smoke", "formal_uniform_agg", "formal_aic_aligned_agg"],
        default="smoke",
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--skip-existing-ok", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--deepgemm-precompile", action="store_true")
    parser.add_argument("--no-deepgemm-fast-warmup", action="store_true")
    parser.add_argument("--warmup-runs", type=int)
    parser.add_argument("--num-layers", type=int)
    parser.add_argument("--extra-env", action="append", default=[])
    return parser.parse_args()


def build_case_group(case_group: str) -> list[AggCase]:
    if case_group == "smoke":
        return [
            AggCase(
                tag="agg_smoke_uniform_d2_ctx128_p1_isl256_prefix0",
                decode_batch_size=2,
                decode_prefix_len=128,
                prefill_batch_size=1,
                prefill_fresh_len=256,
                prefill_prefix_len=0,
                num_layers=3,
                context_length=1024,
                chunked_prefill_size=256,
                max_prefill_tokens=512,
                warmup_runs=1,
                mem_fraction_static=0.45,
            )
        ]

    if case_group == "formal_uniform_agg":
        return [
            AggCase(
                tag="agg_uniform_small_d4_ctx512_p4_isl512_prefix0",
                decode_batch_size=4,
                decode_prefix_len=512,
                prefill_batch_size=4,
                prefill_fresh_len=512,
                prefill_prefix_len=0,
            ),
            AggCase(
                tag="agg_uniform_small_prefix_d8_ctx512_p4_isl512_prefix512",
                decode_batch_size=8,
                decode_prefix_len=512,
                prefill_batch_size=4,
                prefill_fresh_len=512,
                prefill_prefix_len=512,
            ),
            AggCase(
                tag="agg_uniform_prod_short_d16_ctx1024_p8_isl256_prefix1024",
                decode_batch_size=16,
                decode_prefix_len=1024,
                prefill_batch_size=8,
                prefill_fresh_len=256,
                prefill_prefix_len=1024,
            ),
            AggCase(
                tag="agg_uniform_mid_d4_ctx4096_p4_isl2048_prefix1024",
                decode_batch_size=4,
                decode_prefix_len=4096,
                prefill_batch_size=4,
                prefill_fresh_len=2048,
                prefill_prefix_len=1024,
            ),
            AggCase(
                tag="agg_uniform_long_prefix_d8_ctx2048_p2_isl4096_prefix4096",
                decode_batch_size=8,
                decode_prefix_len=2048,
                prefill_batch_size=2,
                prefill_fresh_len=4096,
                prefill_prefix_len=4096,
            ),
            AggCase(
                tag="agg_uniform_chunk_boundary_d8_ctx8192_p1_isl8192_prefix0",
                decode_batch_size=8,
                decode_prefix_len=8192,
                prefill_batch_size=1,
                prefill_fresh_len=8192,
                prefill_prefix_len=0,
            ),
            AggCase(
                tag="agg_uniform_long_decode_d4_ctx16384_p2_isl2048_prefix2048",
                decode_batch_size=4,
                decode_prefix_len=16384,
                prefill_batch_size=2,
                prefill_fresh_len=2048,
                prefill_prefix_len=2048,
                context_length=24576,
            ),
            AggCase(
                tag="agg_uniform_dense_decode_d32_ctx512_p4_isl1024_prefix0",
                decode_batch_size=32,
                decode_prefix_len=512,
                prefill_batch_size=4,
                prefill_fresh_len=1024,
                prefill_prefix_len=0,
            ),
        ]

    if case_group == "formal_aic_aligned_agg":
        return [
            # Strict AIC run_agg alignment: prefix=0, decode KV length ~= total prompt length.
            AggCase(
                tag="agg_aic_p0_small_b8_isl512_ctx2048",
                decode_batch_size=4,
                decode_prefix_len=512,
                prefill_batch_size=4,
                prefill_fresh_len=512,
                prefill_prefix_len=0,
            ),
            AggCase(
                tag="agg_aic_p0_decode_heavy_b12_isl512_ctx2048",
                decode_batch_size=8,
                decode_prefix_len=512,
                prefill_batch_size=4,
                prefill_fresh_len=512,
                prefill_prefix_len=0,
            ),
            AggCase(
                tag="agg_aic_p0_prefill_heavy_b16_isl512_ctx4096",
                decode_batch_size=8,
                decode_prefix_len=512,
                prefill_batch_size=8,
                prefill_fresh_len=512,
                prefill_prefix_len=0,
            ),
            AggCase(
                tag="agg_aic_p0_mid_b8_isl2048_ctx8192",
                decode_batch_size=4,
                decode_prefix_len=2048,
                prefill_batch_size=4,
                prefill_fresh_len=2048,
                prefill_prefix_len=0,
            ),
            AggCase(
                tag="agg_aic_p0_decode_heavy_b17_isl2048_ctx2048",
                decode_batch_size=16,
                decode_prefix_len=2048,
                prefill_batch_size=1,
                prefill_fresh_len=2048,
                prefill_prefix_len=0,
            ),
            AggCase(
                tag="agg_aic_p0_chunk_b9_isl8192_ctx8192",
                decode_batch_size=8,
                decode_prefix_len=8192,
                prefill_batch_size=1,
                prefill_fresh_len=8192,
                prefill_prefix_len=0,
            ),
            AggCase(
                tag="agg_aic_p0_dense_b36_isl1024_ctx4096",
                decode_batch_size=32,
                decode_prefix_len=1024,
                prefill_batch_size=4,
                prefill_fresh_len=1024,
                prefill_prefix_len=0,
            ),
            AggCase(
                tag="agg_aic_p0_dense_prefill_b40_isl512_ctx4096",
                decode_batch_size=32,
                decode_prefix_len=512,
                prefill_batch_size=8,
                prefill_fresh_len=512,
                prefill_prefix_len=0,
            ),
            AggCase(
                tag="agg_aic_p0_longctx_b8_isl4096_ctx8192",
                decode_batch_size=6,
                decode_prefix_len=4096,
                prefill_batch_size=2,
                prefill_fresh_len=4096,
                prefill_prefix_len=0,
            ),
            # Prefix-cache cases are intentionally marked as AIC approximation probes:
            # AIC counts ctx_tokens against total isl, while SGLang schedules fresh tokens.
            AggCase(
                tag="agg_aic_prefix_probe_b12_isl1024_prefix512_ctx4096",
                decode_batch_size=8,
                decode_prefix_len=1024,
                prefill_batch_size=4,
                prefill_fresh_len=512,
                prefill_prefix_len=512,
            ),
            AggCase(
                tag="agg_aic_prefix_probe_b24_isl1280_prefix1024_ctx10240",
                decode_batch_size=16,
                decode_prefix_len=1280,
                prefill_batch_size=8,
                prefill_fresh_len=256,
                prefill_prefix_len=1024,
            ),
            AggCase(
                tag="agg_aic_prefix_probe_b12_isl4096_prefix2048_ctx8192",
                decode_batch_size=10,
                decode_prefix_len=4096,
                prefill_batch_size=2,
                prefill_fresh_len=2048,
                prefill_prefix_len=2048,
            ),
        ]

    raise ValueError(f"Unsupported case group: {case_group}")


def _context_length(case: AggCase) -> int:
    max_decode_seq = case.decode_prefix_len + case.decode_max_new_tokens + 8
    max_prefill_seq = case.prefill_prefix_len + case.prefill_fresh_len + case.prefill_max_new_tokens + 8
    return max(case.context_length, max_decode_seq, max_prefill_seq)


def build_command(case: AggCase, gpu_id: int, run_root: str) -> list[str]:
    cmd = [
        "python",
        ".self/task-localbench/formal-agg-stage2/run_formal_agg_stage2.py",
        "--gpu-id",
        str(gpu_id),
        "--num-layers",
        str(case.num_layers),
        "--attention-backend",
        case.attention_backend,
        "--decode-attention-backend",
        case.decode_attention_backend,
        "--mem-fraction-static",
        str(case.mem_fraction_static),
        "--cuda-graph-mode",
        case.cuda_graph_mode,
        "--pcg-mode",
        case.pcg_mode,
        "--profile-mode",
        case.profile_mode,
        "--warmup-runs",
        str(case.warmup_runs),
        "--context-length",
        str(_context_length(case)),
        "--chunked-prefill-size",
        str(case.chunked_prefill_size),
        "--max-prefill-tokens",
        str(case.max_prefill_tokens),
        "--mixed-profile-steps",
        str(case.mixed_profile_steps),
        "--tp-size",
        str(case.tp_size),
        "--tag",
        case.tag,
        "--run-root",
        run_root,
        "--decode-batch-size",
        str(case.decode_batch_size),
        "--decode-prefix-len",
        str(case.decode_prefix_len),
        "--decode-max-new-tokens",
        str(case.decode_max_new_tokens),
        "--decode-consume-delay-s",
        str(case.decode_consume_delay_s),
        "--decode-start-timeout-s",
        str(case.decode_start_timeout_s),
        "--decode-finish-timeout-s",
        str(case.decode_finish_timeout_s),
        "--prefill-batch-size",
        str(case.prefill_batch_size),
        "--prefill-fresh-len",
        str(case.prefill_fresh_len),
        "--prefill-prefix-len",
        str(case.prefill_prefix_len),
        "--prefill-max-new-tokens",
        str(case.prefill_max_new_tokens),
        "--layerwise-marker",
    ]
    if case.deepgemm_precompile:
        cmd.append("--deepgemm-precompile")
    if not case.deepgemm_fast_warmup:
        cmd.append("--no-deepgemm-fast-warmup")
    for item in case.extra_env:
        cmd.extend(["--extra-env", item])
    return cmd


def copy_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def archive_run(run_dir: Path, bench_root: Path) -> dict[str, str]:
    run_name = run_dir.name
    raw_root = bench_root / "raw_runs" / run_name
    raw_root.parent.mkdir(parents=True, exist_ok=True)
    if raw_root.exists():
        shutil.rmtree(raw_root)
    shutil.copytree(run_dir, raw_root)

    csv_root = bench_root / "agg_csv"
    csv_root.mkdir(parents=True, exist_ok=True)
    mla_csv_src = run_dir / "nsys" / "MLA时延拆解.csv"
    mla_json_src = run_dir / "nsys" / "MLA对齐解析.json"
    mla_md_src = run_dir / "nsys" / "MLA对齐解析.md"
    meta_src = run_dir / "mixed_batch_meta.jsonl"
    csv_dst = csv_root / f"{run_name}__MLA时延拆解.csv"
    json_dst = csv_root / f"{run_name}__MLA对齐解析.json"
    md_dst = csv_root / f"{run_name}__MLA对齐解析.md"
    meta_dst = csv_root / f"{run_name}__mixed_batch_meta.jsonl"
    copy_if_exists(mla_csv_src, csv_dst)
    copy_if_exists(mla_json_src, json_dst)
    copy_if_exists(mla_md_src, md_dst)
    copy_if_exists(meta_src, meta_dst)

    return {
        "raw_run_dir": str(raw_root),
        "csv_path": str(csv_dst) if csv_dst.exists() else "",
        "json_path": str(json_dst) if json_dst.exists() else "",
        "md_path": str(md_dst) if md_dst.exists() else "",
        "mixed_meta_path": str(meta_dst) if meta_dst.exists() else "",
    }


def append_log(log_path: Path, message: str) -> None:
    with log_path.open("a", encoding="utf-8") as f:
        f.write(message.rstrip() + "\n")


def write_manifest_row(manifest_path: Path, row: dict[str, str]) -> None:
    fieldnames = [
        "tag",
        "status",
        "run_dir",
        "raw_run_dir",
        "csv_path",
        "json_path",
        "md_path",
        "mixed_meta_path",
        "decode_batch_size",
        "decode_prefix_len",
        "prefill_batch_size",
        "prefill_fresh_len",
        "prefill_prefix_len",
        "num_layers",
        "chunked_prefill_size",
        "max_prefill_tokens",
        "mixed_warmup_runs",
        "deepgemm_precompile",
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
            if row.get("status") == "ok" and row.get("tag"):
                ok_tags.add(row["tag"])
    return ok_tags


def parse_gpu_ids(raw: str) -> list[int]:
    gpu_ids = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not gpu_ids:
        raise ValueError("--gpu-ids-csv must contain at least one gpu id")
    return gpu_ids


def main() -> int:
    args = parse_args()
    root_dir = Path(__file__).resolve().parents[3]
    run_root = str((root_dir / args.run_root).resolve())
    bench_root = (root_dir / args.bench_root).resolve()
    bench_root.mkdir(parents=True, exist_ok=True)
    log_path = root_dir / ".self/task-localbench/formal-agg-stage2/阶段2正式批量运行记录.md"
    manifest_path = bench_root / "manifest_agg.csv"
    gpu_ids = parse_gpu_ids(args.gpu_ids_csv)

    cases = build_case_group(args.case_group)
    if args.start_index:
        cases = cases[args.start_index :]
    if args.max_cases is not None:
        cases = cases[: args.max_cases]

    replacements = {}
    if args.deepgemm_precompile:
        replacements["deepgemm_precompile"] = True
    if args.no_deepgemm_fast_warmup:
        replacements["deepgemm_fast_warmup"] = False
    if args.warmup_runs is not None:
        replacements["warmup_runs"] = args.warmup_runs
    if args.num_layers is not None:
        replacements["num_layers"] = args.num_layers
    if args.extra_env:
        cases = [replace(case, extra_env=tuple([*case.extra_env, *args.extra_env])) for case in cases]
    if replacements:
        cases = [replace(case, **replacements) for case in cases]

    existing_ok = load_existing_ok_tags(manifest_path) if args.skip_existing_ok else set()
    if existing_ok:
        cases = [case for case in cases if case.tag not in existing_ok]

    if not log_path.exists():
        log_path.write_text(
            "# 阶段2正式批量运行记录\n\n"
            "## 说明\n\n"
            "- 本文档记录 PD mixed / agg 实机用例的批量运行。\n"
            "- 用例采用统一 `decode_prefix_len`、`prefill_fresh_len`、`prefill_prefix_len`，对齐 AIC `run_agg` 的单一 isl/prefix 口径。\n"
            "- 原始 run 归档到 `bench_data/.../deepseek_v3_mla/raw_runs/`，核心 mixed/MLA 解析归档到 `agg_csv/`。\n\n"
            "## 运行日志\n\n",
            encoding="utf-8",
        )

    append_log(
        log_path,
        f"### 新批次\n- case_group: `{args.case_group}`\n- cases: `{len(cases)}`\n- gpu_ids: `{gpu_ids}`\n- bench_root: `{bench_root}`\n",
    )

    failures = 0
    for idx, case in enumerate(cases):
        gpu_id = gpu_ids[idx % len(gpu_ids)]
        cmd = build_command(case, gpu_id, run_root)
        append_log(log_path, f"- [{idx + 1}/{len(cases)}] 开始 `{case.tag}` on GPU `{gpu_id}`\n  - cmd: `{' '.join(cmd)}`")
        if args.dry_run:
            write_manifest_row(
                manifest_path,
                {
                    "tag": case.tag,
                    "status": "dry_run",
                    "decode_batch_size": str(case.decode_batch_size),
                    "decode_prefix_len": str(case.decode_prefix_len),
                    "prefill_batch_size": str(case.prefill_batch_size),
                    "prefill_fresh_len": str(case.prefill_fresh_len),
                    "prefill_prefix_len": str(case.prefill_prefix_len),
                    "num_layers": str(case.num_layers),
                    "chunked_prefill_size": str(case.chunked_prefill_size),
                    "max_prefill_tokens": str(case.max_prefill_tokens),
                    "mixed_warmup_runs": str(case.warmup_runs),
                    "deepgemm_precompile": str(case.deepgemm_precompile),
                },
            )
            continue

        completed = subprocess.run(cmd, text=True, capture_output=True)
        stdout = completed.stdout.strip()
        run_dir = Path(stdout.splitlines()[-1]) if stdout else None
        status = "ok"
        archive_info = {
            "raw_run_dir": "",
            "csv_path": "",
            "json_path": "",
            "md_path": "",
            "mixed_meta_path": "",
        }

        if completed.returncode != 0 or run_dir is None or not run_dir.exists():
            status = "failed"
            failures += 1
            append_log(
                log_path,
                f"  - 失败: returncode=`{completed.returncode}`\n"
                f"  - stdout: `{completed.stdout.strip()}`\n"
                f"  - stderr: `{completed.stderr.strip()}`",
            )
        else:
            archive_info = archive_run(run_dir, bench_root)
            append_log(
                log_path,
                f"  - 成功: `{run_dir}`\n"
                f"  - CSV: `{archive_info['csv_path']}`\n"
                f"  - mixed_meta: `{archive_info['mixed_meta_path']}`",
            )

        write_manifest_row(
            manifest_path,
            {
                "tag": case.tag,
                "status": status,
                "run_dir": str(run_dir) if run_dir is not None else "",
                **archive_info,
                "decode_batch_size": str(case.decode_batch_size),
                "decode_prefix_len": str(case.decode_prefix_len),
                "prefill_batch_size": str(case.prefill_batch_size),
                "prefill_fresh_len": str(case.prefill_fresh_len),
                "prefill_prefix_len": str(case.prefill_prefix_len),
                "num_layers": str(case.num_layers),
                "chunked_prefill_size": str(case.chunked_prefill_size),
                "max_prefill_tokens": str(case.max_prefill_tokens),
                "mixed_warmup_runs": str(case.warmup_runs),
                "deepgemm_precompile": str(case.deepgemm_precompile),
            },
        )

    if not args.dry_run:
        append_log(log_path, f"\n## 本批次总结\n- total_cases: `{len(cases)}`\n- failures: `{failures}`\n")
    print(bench_root)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
