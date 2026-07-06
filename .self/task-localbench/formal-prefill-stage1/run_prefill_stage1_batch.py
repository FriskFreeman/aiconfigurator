#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path


@dataclass(frozen=True)
class BatchCase:
    tag: str
    fresh_lens: list[int]
    prefix_lens: list[int]
    attention_backend: str
    force_prefill_mha_for_prefix: bool
    tp_size: int = 1
    num_layers: int = 5
    context_length: int = 0
    max_new_tokens: int = 1
    warmup_runs: int = 1
    layerwise_marker: bool = True
    profile_mode: str = "nsys"
    nsys_sample: str = "none"
    nsys_cpuctxsw: str = "none"
    nsys_sampling_period: int | None = None
    nsys_samples_per_backtrace: int | None = None
    triton_cache_dir: str | None = None
    triton_dump_dir: str | None = None
    triton_debug: bool = False
    triton_print_autotuning: bool = False
    extra_env: tuple[str, ...] = ()
    docker_privileged: bool = False
    docker_cap_add: tuple[str, ...] = ()
    docker_security_opt: tuple[str, ...] = ()
    docker_pid_host: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-id", type=int, default=7)
    parser.add_argument(
        "--run-root",
        default=".self/task-localbench/formal-prefill-stage1",
    )
    parser.add_argument(
        "--bench-root",
        default="bench_data/h100_sxm/sglang/v0.5.9/deepseek_v3_mla",
    )
    parser.add_argument(
        "--case-group",
        choices=["req1_equal_len", "req1_large_prefix_mid_fresh", "smoke_req1_equal_len"],
        default="smoke_req1_equal_len",
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--skip-existing-ok", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--nsys-sample",
        choices=["process-tree", "system-wide", "none"],
        default=None,
        help="Override all cases' nsys --sample value.",
    )
    parser.add_argument(
        "--nsys-cpuctxsw",
        choices=["process-tree", "system-wide", "none"],
        default=None,
        help="Override all cases' nsys --cpuctxsw value.",
    )
    parser.add_argument("--nsys-sampling-period", type=int, default=None)
    parser.add_argument("--nsys-samples-per-backtrace", type=int, default=None)
    parser.add_argument("--triton-cache-dir", default=None)
    parser.add_argument("--triton-dump-dir", default=None)
    parser.add_argument("--triton-debug", action="store_true")
    parser.add_argument("--triton-print-autotuning", action="store_true")
    parser.add_argument("--extra-env", action="append", default=[])
    parser.add_argument("--docker-privileged", action="store_true")
    parser.add_argument("--docker-cap-add", action="append", default=[])
    parser.add_argument("--docker-security-opt", action="append", default=[])
    parser.add_argument("--docker-pid-host", action="store_true")
    return parser.parse_args()


def build_case_group(case_group: str) -> list[BatchCase]:
    if case_group == "smoke_req1_equal_len":
        return [
            BatchCase(
                tag="req1_smoke_fresh_regular_b4_f2048_p0",
                fresh_lens=[2048] * 4,
                prefix_lens=[0] * 4,
                attention_backend="auto",
                force_prefill_mha_for_prefix=False,
                num_layers=3,
            ),
            BatchCase(
                tag="req1_smoke_prefix_regular_b4_f2048_p512",
                fresh_lens=[2048] * 4,
                prefix_lens=[512] * 4,
                attention_backend="auto",
                force_prefill_mha_for_prefix=True,
                num_layers=3,
            ),
            BatchCase(
                tag="req1_smoke_tp2_rank0_fresh_b1_f32_p0",
                fresh_lens=[32],
                prefix_lens=[0],
                attention_backend="auto",
                force_prefill_mha_for_prefix=False,
                tp_size=2,
                num_layers=3,
            ),
        ]

    if case_group == "req1_equal_len":
        return [
            BatchCase(
                tag="req1_equal_fresh_regular_b1_f1_p0",
                fresh_lens=[1],
                prefix_lens=[0],
                attention_backend="auto",
                force_prefill_mha_for_prefix=True,
            ),
            BatchCase(
                tag="req1_equal_fresh_regular_b8_f1_p0",
                fresh_lens=[1] * 8,
                prefix_lens=[0] * 8,
                attention_backend="auto",
                force_prefill_mha_for_prefix=True,
            ),
            BatchCase(
                tag="req1_equal_fresh_regular_b4_f512_p0",
                fresh_lens=[512] * 4,
                prefix_lens=[0] * 4,
                attention_backend="auto",
                force_prefill_mha_for_prefix=True,
            ),
            BatchCase(
                tag="req1_equal_fresh_regular_b4_f2048_p0",
                fresh_lens=[2048] * 4,
                prefix_lens=[0] * 4,
                attention_backend="auto",
                force_prefill_mha_for_prefix=True,
            ),
            BatchCase(
                tag="req1_equal_fresh_regular_b2_f4096_p0",
                fresh_lens=[4096] * 2,
                prefix_lens=[0] * 2,
                attention_backend="auto",
                force_prefill_mha_for_prefix=True,
            ),
            BatchCase(
                tag="req1_equal_fresh_regular_b1_f8192_p0",
                fresh_lens=[8192],
                prefix_lens=[0],
                attention_backend="auto",
                force_prefill_mha_for_prefix=True,
            ),
            BatchCase(
                tag="req1_equal_fresh_irregular_b12_f300_p0",
                fresh_lens=[300] * 12,
                prefix_lens=[0] * 12,
                attention_backend="auto",
                force_prefill_mha_for_prefix=True,
            ),
            BatchCase(
                tag="req1_equal_fresh_irregular_b4_f1200_p0",
                fresh_lens=[1200] * 4,
                prefix_lens=[0] * 4,
                attention_backend="auto",
                force_prefill_mha_for_prefix=True,
            ),
            BatchCase(
                tag="req1_equal_fresh_irregular_b1_f4500_p0",
                fresh_lens=[4500],
                prefix_lens=[0],
                attention_backend="auto",
                force_prefill_mha_for_prefix=True,
            ),
            BatchCase(
                tag="req1_equal_fresh_irregular_b1_f6000_p0",
                fresh_lens=[6000],
                prefix_lens=[0],
                attention_backend="auto",
                force_prefill_mha_for_prefix=True,
            ),
            BatchCase(
                tag="req1_equal_prefix_regular_b4_f512_p512",
                fresh_lens=[512] * 4,
                prefix_lens=[512] * 4,
                attention_backend="auto",
                force_prefill_mha_for_prefix=True,
            ),
            BatchCase(
                tag="req1_equal_prefix_regular_b4_f2048_p1024",
                fresh_lens=[2048] * 4,
                prefix_lens=[1024] * 4,
                attention_backend="auto",
                force_prefill_mha_for_prefix=True,
            ),
            BatchCase(
                tag="req1_equal_prefix_regular_b2_f4096_p4096",
                fresh_lens=[4096] * 2,
                prefix_lens=[4096] * 2,
                attention_backend="auto",
                force_prefill_mha_for_prefix=True,
            ),
            BatchCase(
                tag="req1_equal_prefix_regular_b1_f8192_p0",
                fresh_lens=[8192],
                prefix_lens=[0],
                attention_backend="auto",
                force_prefill_mha_for_prefix=False,
            ),
            BatchCase(
                tag="req1_equal_prefix_regular_b1_f8192_p1024",
                fresh_lens=[8192],
                prefix_lens=[1024],
                attention_backend="auto",
                force_prefill_mha_for_prefix=True,
            ),
            BatchCase(
                tag="req1_equal_prefix_regular_b1_f8192_p4096",
                fresh_lens=[8192],
                prefix_lens=[4096],
                attention_backend="auto",
                force_prefill_mha_for_prefix=True,
            ),
            BatchCase(
                tag="req1_equal_prefix_regular_b1_f8192_p8192",
                fresh_lens=[8192],
                prefix_lens=[8192],
                attention_backend="auto",
                force_prefill_mha_for_prefix=True,
            ),
            BatchCase(
                tag="req1_equal_prefix_regular_b1_f8192_p10500",
                fresh_lens=[8192],
                prefix_lens=[10500],
                attention_backend="auto",
                force_prefill_mha_for_prefix=True,
            ),
            BatchCase(
                tag="req1_equal_prefix_regular_b1_f8192_p32000",
                fresh_lens=[8192],
                prefix_lens=[32000],
                attention_backend="auto",
                force_prefill_mha_for_prefix=True,
            ),
        ]

    if case_group == "req1_large_prefix_mid_fresh":
        return [
            BatchCase(
                tag="req1_large_prefix_regular_b8_f256_p4096",
                fresh_lens=[256] * 8,
                prefix_lens=[4096] * 8,
                attention_backend="auto",
                force_prefill_mha_for_prefix=True,
            ),
            BatchCase(
                tag="req1_large_prefix_regular_b8_f512_p4096",
                fresh_lens=[512] * 8,
                prefix_lens=[4096] * 8,
                attention_backend="auto",
                force_prefill_mha_for_prefix=True,
            ),
            BatchCase(
                tag="req1_large_prefix_regular_b4_f512_p8192",
                fresh_lens=[512] * 4,
                prefix_lens=[8192] * 4,
                attention_backend="auto",
                force_prefill_mha_for_prefix=True,
            ),
            BatchCase(
                tag="req1_large_prefix_regular_b4_f1024_p8192",
                fresh_lens=[1024] * 4,
                prefix_lens=[8192] * 4,
                attention_backend="auto",
                force_prefill_mha_for_prefix=True,
            ),
            BatchCase(
                tag="req1_large_prefix_regular_b4_f1024_p16384",
                fresh_lens=[1024] * 4,
                prefix_lens=[16384] * 4,
                attention_backend="auto",
                force_prefill_mha_for_prefix=True,
            ),
            BatchCase(
                tag="req1_large_prefix_regular_b2_f2048_p16384",
                fresh_lens=[2048] * 2,
                prefix_lens=[16384] * 2,
                attention_backend="auto",
                force_prefill_mha_for_prefix=True,
            ),
            BatchCase(
                tag="req1_large_prefix_regular_b2_f2048_p32000",
                fresh_lens=[2048] * 2,
                prefix_lens=[32000] * 2,
                attention_backend="auto",
                force_prefill_mha_for_prefix=True,
            ),
            BatchCase(
                tag="req1_large_prefix_regular_b1_f4096_p32000",
                fresh_lens=[4096],
                prefix_lens=[32000],
                attention_backend="auto",
                force_prefill_mha_for_prefix=True,
            ),
        ]

    raise ValueError(f"Unsupported case group: {case_group}")


def build_command(
    case: BatchCase,
    gpu_id: int,
    run_root: str,
) -> list[str]:
    # SGLang v0.5.9 applies multiple prompt-length guards. In practice a request of
    # length L may still fail when context_length is only L/L+1 because the worker
    # side also derives max_req_input_len ~= context_length - 6.
    max_request_seq_len = max(
        prefix + fresh for prefix, fresh in zip(case.prefix_lens, case.fresh_lens)
    )
    effective_context_length = max(
        case.context_length,
        max_request_seq_len + case.max_new_tokens + 2,
        max_request_seq_len + 8,
    )
    cmd = [
        "python",
        ".self/task-localbench/formal-prefill-stage1/run_formal_prefill_stage1.py",
        "--gpu-id",
        str(gpu_id),
        "--num-layers",
        str(case.num_layers),
        "--fresh-lens-csv",
        ",".join(str(x) for x in case.fresh_lens),
        "--prefix-lens-csv",
        ",".join(str(x) for x in case.prefix_lens),
        "--context-length",
        str(effective_context_length),
        "--max-new-tokens",
        str(case.max_new_tokens),
        "--warmup-runs",
        str(case.warmup_runs),
        "--profile-mode",
        case.profile_mode,
        "--cuda-graph-mode",
        "off",
        "--pcg-mode",
        "off",
        "--tp-size",
        str(case.tp_size),
        "--tag",
        case.tag,
        "--run-root",
        run_root,
    ]
    if case.attention_backend not in ("", "auto", "default"):
        cmd.extend(["--attention-backend", case.attention_backend])
    if case.layerwise_marker:
        cmd.append("--layerwise-marker")
    else:
        cmd.append("--no-layerwise-marker")
    if case.force_prefill_mha_for_prefix:
        cmd.append("--force-prefill-mha-for-prefix")
    if case.nsys_sample != "none":
        cmd.extend(["--nsys-sample", case.nsys_sample])
    if case.nsys_cpuctxsw != "none":
        cmd.extend(["--nsys-cpuctxsw", case.nsys_cpuctxsw])
    if case.nsys_sampling_period is not None:
        cmd.extend(["--nsys-sampling-period", str(case.nsys_sampling_period)])
    if case.nsys_samples_per_backtrace is not None:
        cmd.extend(
            ["--nsys-samples-per-backtrace", str(case.nsys_samples_per_backtrace)]
        )
    if case.triton_cache_dir:
        cmd.extend(["--triton-cache-dir", case.triton_cache_dir])
    if case.triton_dump_dir:
        cmd.extend(["--triton-dump-dir", case.triton_dump_dir])
    if case.triton_debug:
        cmd.append("--triton-debug")
    if case.triton_print_autotuning:
        cmd.append("--triton-print-autotuning")
    for item in case.extra_env:
        cmd.extend(["--extra-env", item])
    if case.docker_privileged:
        cmd.append("--docker-privileged")
    if case.docker_pid_host:
        cmd.append("--docker-pid-host")
    for cap in case.docker_cap_add:
        cmd.extend(["--docker-cap-add", cap])
    for opt in case.docker_security_opt:
        cmd.extend(["--docker-security-opt", opt])
    return cmd


def copy_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def safe_artifact_stem(run_name: str, max_len: int = 150) -> str:
    if len(run_name) <= max_len:
        return run_name
    digest = hashlib.sha1(run_name.encode("utf-8")).hexdigest()[:10]
    return f"{run_name[: max_len - 12]}__{digest}"


def archive_run(run_dir: Path, bench_root: Path) -> dict[str, str]:
    run_name = run_dir.name
    artifact_stem = safe_artifact_stem(run_name)
    raw_root = bench_root / "raw_runs" / run_name
    raw_root.parent.mkdir(parents=True, exist_ok=True)
    if raw_root.exists():
        shutil.rmtree(raw_root)
    shutil.copytree(run_dir, raw_root)

    csv_root = bench_root / "csv"
    csv_root.mkdir(parents=True, exist_ok=True)
    mla_csv_src = run_dir / "nsys" / "MLA时延拆解.csv"
    mla_json_src = run_dir / "nsys" / "MLA对齐解析.json"
    mla_md_src = run_dir / "nsys" / "MLA对齐解析.md"
    csv_dst = csv_root / f"{artifact_stem}__MLA时延拆解.csv"
    json_dst = csv_root / f"{artifact_stem}__MLA对齐解析.json"
    md_dst = csv_root / f"{artifact_stem}__MLA对齐解析.md"
    copy_if_exists(mla_csv_src, csv_dst)
    copy_if_exists(mla_json_src, json_dst)
    copy_if_exists(mla_md_src, md_dst)

    return {
        "raw_run_dir": str(raw_root),
        "csv_path": str(csv_dst) if csv_dst.exists() else "",
        "json_path": str(json_dst) if json_dst.exists() else "",
        "md_path": str(md_dst) if md_dst.exists() else "",
    }


def append_log(log_path: Path, message: str) -> None:
    with log_path.open("a", encoding="utf-8") as f:
        f.write(message.rstrip() + "\n")


def write_manifest_row(
    manifest_path: Path,
    row: dict[str, str],
) -> None:
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
        "fresh_lens",
        "prefix_lens",
        "tp_size",
        "attention_backend",
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


def main() -> int:
    args = parse_args()
    root_dir = Path(__file__).resolve().parents[3]
    run_root = str((root_dir / args.run_root).resolve())
    bench_root = (root_dir / args.bench_root).resolve()
    bench_root.mkdir(parents=True, exist_ok=True)

    log_path = root_dir / ".self/task-localbench/formal-prefill-stage1/需求1批量运行记录.md"
    manifest_path = bench_root / "manifest.csv"

    cases = build_case_group(args.case_group)
    if args.start_index:
        cases = cases[args.start_index :]
    if args.max_cases is not None:
        cases = cases[: args.max_cases]

    if any(
        value is not None
        for value in [
            args.nsys_sample,
            args.nsys_cpuctxsw,
            args.nsys_sampling_period,
            args.nsys_samples_per_backtrace,
            args.triton_cache_dir,
            args.triton_dump_dir,
        ]
    ) or args.triton_debug or args.triton_print_autotuning or args.extra_env or args.docker_privileged or args.docker_cap_add or args.docker_security_opt or args.docker_pid_host:
        cases = [
            replace(
                case,
                nsys_sample=args.nsys_sample or case.nsys_sample,
                nsys_cpuctxsw=args.nsys_cpuctxsw or case.nsys_cpuctxsw,
                nsys_sampling_period=(
                    args.nsys_sampling_period
                    if args.nsys_sampling_period is not None
                    else case.nsys_sampling_period
                ),
                nsys_samples_per_backtrace=(
                    args.nsys_samples_per_backtrace
                    if args.nsys_samples_per_backtrace is not None
                    else case.nsys_samples_per_backtrace
                ),
                triton_cache_dir=args.triton_cache_dir or case.triton_cache_dir,
                triton_dump_dir=args.triton_dump_dir or case.triton_dump_dir,
                triton_debug=args.triton_debug or case.triton_debug,
                triton_print_autotuning=(
                    args.triton_print_autotuning or case.triton_print_autotuning
                ),
                extra_env=tuple([*case.extra_env, *args.extra_env]),
                docker_privileged=args.docker_privileged or case.docker_privileged,
                docker_cap_add=tuple([*case.docker_cap_add, *args.docker_cap_add]),
                docker_security_opt=tuple(
                    [*case.docker_security_opt, *args.docker_security_opt]
                ),
                docker_pid_host=args.docker_pid_host or case.docker_pid_host,
            )
            for case in cases
        ]

    existing_ok_tags = load_existing_ok_tags(manifest_path) if args.skip_existing_ok else set()
    if existing_ok_tags:
        cases = [case for case in cases if case.tag not in existing_ok_tags]

    if not log_path.exists():
        log_path.write_text(
            "# 需求1批量运行记录\n\n"
            "## 说明\n\n"
            "- 本文档记录 `prompt` 中需求1纯 prefill 场景的批量运行过程。\n"
            "- 原始 run 目录归档到 `bench_data/.../deepseek_v3_mla/raw_runs/`。\n"
            "- 核心 `nsys` 结果归档到 `bench_data/.../deepseek_v3_mla/csv/`。\n\n"
            "## 运行日志\n\n",
            encoding="utf-8",
        )

    append_log(
        log_path,
        f"### 新批次\n- case_group: `{args.case_group}`\n- start_index: `{args.start_index}`\n- max_cases: `{args.max_cases}`\n- skip_existing_ok: `{args.skip_existing_ok}`\n- bench_root: `{bench_root}`\n",
    )

    failures = 0
    for idx, case in enumerate(cases, start=1):
        cmd = build_command(case, args.gpu_id, run_root)
        append_log(
            log_path,
            f"- [{idx}/{len(cases)}] 开始 `{case.tag}`\n  - cmd: `{ ' '.join(cmd) }`",
        )
        if args.dry_run:
            write_manifest_row(
                manifest_path,
                {
                    "tag": case.tag,
                    "status": "dry_run",
                    "batch_size": str(len(case.fresh_lens)),
                    "fresh_lens": ",".join(str(x) for x in case.fresh_lens),
                    "prefix_lens": ",".join(str(x) for x in case.prefix_lens),
                    "tp_size": str(case.tp_size),
                    "attention_backend": case.attention_backend,
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
        }

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
            archive_info = archive_run(run_dir, bench_root)
            append_log(
                log_path,
                f"  - 成功: `{run_dir}`\n"
                f"  - CSV: `{archive_info['csv_path']}`",
            )

        write_manifest_row(
            manifest_path,
            {
                "tag": case.tag,
                "status": status,
                "run_dir": str(run_dir) if run_dir is not None else "",
                **archive_info,
                "batch_size": str(len(case.fresh_lens)),
                "fresh_lens": ",".join(str(x) for x in case.fresh_lens),
                "prefix_lens": ",".join(str(x) for x in case.prefix_lens),
                "tp_size": str(case.tp_size),
                "attention_backend": case.attention_backend,
            },
        )

    if not args.dry_run:
        append_log(
            log_path,
            f"\n## 本批次总结\n- total_cases: `{len(cases)}`\n- failures: `{failures}`\n",
        )
    print(bench_root)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
