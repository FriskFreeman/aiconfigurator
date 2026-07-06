#!/usr/bin/env python3
import argparse
import csv
import json
import sqlite3
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--image", default="booleimg.myaddr.io/lmsysorg/sglang:v0.5.9")
    parser.add_argument(
        "--reference-run-dir",
        action="append",
        default=[],
        help="Existing stage1 run dir to compare against. Can be repeated.",
    )
    return parser.parse_args()


def run_cmd(cmd: list[str], log_path: Path) -> None:
    completed = subprocess.run(cmd, text=True, capture_output=True)
    log_path.write_text(
        "\n".join(
            [
                "COMMAND:",
                " ".join(cmd),
                "",
                "STDOUT:",
                completed.stdout,
                "",
                "STDERR:",
                completed.stderr,
                "",
                f"RETURNCODE: {completed.returncode}",
            ]
        )
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


def export_sqlite(image: str, run_dir: Path) -> Path:
    nsys_dir = run_dir / "nsys"
    sqlite_path = nsys_dir / "report.sqlite"
    alt_sqlite_path = nsys_dir / "report"
    if sqlite_path.exists() and sqlite_path.stat().st_size > 0:
        return sqlite_path
    if alt_sqlite_path.exists() and alt_sqlite_path.stat().st_size > 0:
        return alt_sqlite_path
    sqlite_path.unlink(missing_ok=True)
    alt_sqlite_path.unlink(missing_ok=True)
    reps = sorted(nsys_dir.glob("*.nsys-rep"))
    if not reps:
        raise FileNotFoundError(f"No .nsys-rep under {nsys_dir}")
    rep_rel = reps[0].relative_to(run_dir)
    sqlite_rel = sqlite_path.relative_to(run_dir).with_suffix("")
    run_cmd(
        [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{run_dir}:/work",
            "-w",
            "/work",
            image,
            "nsys",
            "export",
            "--force-overwrite=true",
            "--type",
            "sqlite",
            "--output",
            str(sqlite_rel),
            str(rep_rel),
        ],
        nsys_dir / "export_sqlite.log",
    )
    if sqlite_path.exists() and sqlite_path.stat().st_size > 0:
        return sqlite_path
    if alt_sqlite_path.exists() and alt_sqlite_path.stat().st_size > 0:
        return alt_sqlite_path
    raise FileNotFoundError(f"SQLite export did not create {sqlite_path} or {alt_sqlite_path}")


def _load_string_map(conn: sqlite3.Connection) -> dict[int, str]:
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    for table in ("StringIds", "STRINGS"):
        if table not in tables:
            continue
        cols = [row[1] for row in conn.execute(f"PRAGMA table_info({table})")]
        id_col = "id" if "id" in cols else cols[0]
        value_col = None
        for cand in ("value", "string", "name"):
            if cand in cols:
                value_col = cand
                break
        if value_col is None and len(cols) >= 2:
            value_col = cols[1]
        if value_col is None:
            continue
        return {int(i): str(v) for i, v in conn.execute(f"SELECT {id_col}, {value_col} FROM {table}")}
    return {}


def _resolve_name(raw, strings: dict[int, str]) -> str:
    if raw is None:
        return ""
    if isinstance(raw, int):
        return strings.get(raw, str(raw))
    text = str(raw)
    if text.isdigit():
        return strings.get(int(text), text)
    return text


def classify_kernel(name: str) -> tuple[str, str]:
    lowered = name.lower()
    if "deep_gemm" in lowered or "deepgemm" in lowered:
        return "gemm", "deepgemm"
    if "flash_mla" in lowered or "flashmla" in lowered or "flash_fwd_mla" in lowered:
        return "attention", "flashmla"
    if "flashattn" in lowered or "flash::" in lowered:
        return "attention", "fa3"
    if "concat_mla_k" in lowered or "concat_and_cache_mla" in lowered:
        return "attention_support", "concat_mla_k"
    if "prepare_varlen_num_blocks" in lowered:
        return "attention_support", "fa3_prepare"
    if "set_mla_kv_buffer" in lowered:
        return "attention_support", "set_mla_kv_buffer"
    if "quant" in lowered or "cast" in lowered:
        return "quant_or_cast", "sglang_or_triton"
    if "rmsnorm" in lowered or "norm" in lowered:
        return "norm", "torch_or_triton"
    if "triton" in lowered:
        return "triton_misc", "triton"
    if "void at::native" in lowered or "at_cuda" in lowered:
        return "torch_misc", "torch"
    return "other", "unknown"


def extract_kernels(sqlite_path: Path) -> list[dict[str, object]]:
    conn = sqlite3.connect(sqlite_path)
    try:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "CUPTI_ACTIVITY_KIND_KERNEL" not in tables:
            return []
        cols = [row[1] for row in conn.execute("PRAGMA table_info(CUPTI_ACTIVITY_KIND_KERNEL)")]
        name_col = "demangledName" if "demangledName" in cols else "shortName"
        strings = _load_string_map(conn)
        rows = conn.execute(
            f"""
            SELECT start, end, {name_col}
            FROM CUPTI_ACTIVITY_KIND_KERNEL
            WHERE start IS NOT NULL AND end IS NOT NULL
            ORDER BY start
            """
        ).fetchall()
        out = []
        for idx, (start, end, raw_name) in enumerate(rows):
            name = _resolve_name(raw_name, strings)
            category, source = classify_kernel(name)
            out.append(
                {
                    "index": idx,
                    "start_ns": int(start),
                    "end_ns": int(end),
                    "duration_us": (int(end) - int(start)) / 1000.0,
                    "name": name,
                    "category": category,
                    "source": source,
                }
            )
        return out
    finally:
        conn.close()


def summarize(kernels: list[dict[str, object]]) -> dict[str, object]:
    by_source: dict[str, dict[str, object]] = {}
    for kernel in kernels:
        key = f"{kernel['category']}::{kernel['source']}"
        item = by_source.setdefault(key, {"count": 0, "duration_us": 0.0, "examples": []})
        item["count"] = int(item["count"]) + 1
        item["duration_us"] = float(item["duration_us"]) + float(kernel["duration_us"])
        examples = item["examples"]
        if isinstance(examples, list) and len(examples) < 5:
            examples.append(kernel["name"])
    return {
        "kernel_count": len(kernels),
        "total_kernel_duration_us": sum(float(k["duration_us"]) for k in kernels),
        "by_source": by_source,
        "first_80_kernels": kernels[:80],
    }


def write_csv(path: Path, kernels: list[dict[str, object]]) -> None:
    if not kernels:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(kernels[0].keys()))
        writer.writeheader()
        writer.writerows(kernels)


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    sqlite_path = export_sqlite(args.image, run_dir)
    kernels = extract_kernels(sqlite_path)
    write_csv(run_dir / "nsys" / "kernel_timeline.csv", kernels)

    report: dict[str, object] = {
        "run_dir": str(run_dir),
        "sqlite_path": str(sqlite_path),
        "summary": summarize(kernels),
        "references": [],
    }
    for ref_raw in args.reference_run_dir:
        ref_dir = Path(ref_raw).resolve()
        ref_sqlite = export_sqlite(args.image, ref_dir)
        ref_kernels = extract_kernels(ref_sqlite)
        write_csv(ref_dir / "nsys" / "kernel_timeline_for_quant_compare.csv", ref_kernels)
        report["references"].append(
            {
                "run_dir": str(ref_dir),
                "sqlite_path": str(ref_sqlite),
                "summary": summarize(ref_kernels),
            }
        )

    (run_dir / "nsys" / "quant_kernel_compare.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))

    lines = [
        "# MLA Module 量化实验 Kernel 对照",
        "",
        f"实验目录: `{run_dir}`",
        "",
        "## Collector Module 实验汇总",
        "",
        f"- kernel 数: {report['summary']['kernel_count']}",
        f"- kernel 总时长(us): {report['summary']['total_kernel_duration_us']:.3f}",
        "",
        "### 来源分类",
        "",
    ]
    for key, item in sorted(report["summary"]["by_source"].items()):
        lines.append(f"- `{key}`: count={item['count']}, duration_us={item['duration_us']:.3f}")
    lines += ["", "## Reference 汇总", ""]
    for ref in report["references"]:
        lines.append(f"### `{Path(ref['run_dir']).name}`")
        lines.append(f"- kernel 数: {ref['summary']['kernel_count']}")
        lines.append(f"- kernel 总时长(us): {ref['summary']['total_kernel_duration_us']:.3f}")
        for key, item in sorted(ref["summary"]["by_source"].items()):
            lines.append(f"- `{key}`: count={item['count']}, duration_us={item['duration_us']:.3f}")
        lines.append("")
    (run_dir / "nsys" / "quant_kernel_compare.md").write_text("\n".join(lines) + "\n")
    print(run_dir / "nsys" / "quant_kernel_compare.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
