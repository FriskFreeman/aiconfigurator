#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-root",
        default=".self/task-localbench/formal-prefill-stage1",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_root = Path(args.run_root).resolve()
    if run_root.name != "output":
        run_root = run_root / "output"
    rows = []
    clean_rows = []
    recommended_rows = []
    split_rows = []
    for run_dir in sorted(run_root.glob("2026*_prefill_stage1_*")):
        meta_path = run_dir / "run_meta.json"
        mla_json_path = run_dir / "nsys" / "MLA对齐解析.json"
        if not meta_path.exists() or not mla_json_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        mla_rows = json.loads(mla_json_path.read_text())
        prompt_shape = (
            "x".join(str(x) for x in meta.get("prompt_lens", []))
            if meta.get("prompt_lens")
            else str(meta.get("prompt_len"))
        )
        for item in mla_rows:
            if item.get("stage") != "prefill":
                continue
            rows.append(
                {
                    "run_dir": str(run_dir),
                    "tag": run_dir.name,
                    "batch_size": meta.get("batch_size"),
                    "prompt_len": meta.get("prompt_len"),
                    "prompt_lens": meta.get("prompt_lens"),
                    "prompt_shape": prompt_shape,
                    "warmup_runs": meta.get("warmup_runs"),
                    "layer_id": item.get("layer_id"),
                    "stage_instance_index": item.get("stage_instance_index"),
                    "stage_instance_count": item.get("stage_instance_count"),
                    "collector_prefill_aligned": item.get("collector_prefill_aligned"),
                    "prefill_mla_duration_ms": item.get("duration_ms"),
                }
            )
            target = clean_rows if item.get("collector_prefill_aligned") else split_rows
            target.append(rows[-1])
            if item.get("collector_prefill_aligned") and (meta.get("warmup_runs") or 0) >= 1:
                recommended_rows.append(rows[-1])

    rows.sort(
        key=lambda x: (
            x["batch_size"],
            str(x["prompt_shape"]),
            x["tag"],
            x["layer_id"],
            x["stage_instance_index"],
        )
    )
    clean_rows.sort(
        key=lambda x: (
            x["batch_size"],
            str(x["prompt_shape"]),
            x["tag"],
            x["layer_id"],
            x["stage_instance_index"],
        )
    )
    split_rows.sort(
        key=lambda x: (
            x["batch_size"],
            str(x["prompt_shape"]),
            x["tag"],
            x["layer_id"],
            x["stage_instance_index"],
        )
    )
    out_json = run_root / "stage1_prefill_mla_summary.json"
    out_md = run_root / "stage1_prefill_mla_summary.md"
    clean_out_json = run_root / "stage1_prefill_mla_summary_clean.json"
    clean_out_md = run_root / "stage1_prefill_mla_summary_clean.md"
    recommended_out_json = run_root / "stage1_prefill_mla_summary_recommended.json"
    recommended_out_md = run_root / "stage1_prefill_mla_summary_recommended.md"
    split_out_json = run_root / "stage1_prefill_mla_summary_split.json"
    split_out_md = run_root / "stage1_prefill_mla_summary_split.md"
    stats_out_md = run_root / "stage1_prefill_mla_stats.md"
    out_json.write_text(json.dumps(rows, indent=2))
    clean_out_json.write_text(json.dumps(clean_rows, indent=2))
    recommended_out_json.write_text(json.dumps(recommended_rows, indent=2))
    split_out_json.write_text(json.dumps(split_rows, indent=2))

    lines = [
        "# Stage1 Prefill MLA Summary",
        "",
        "| batch_size | prompt_shape | warmup_runs | layer_id | instance_index | instance_count | collector_prefill_aligned | prefill_mla_duration_ms | run_dir |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['batch_size']} | {row['prompt_shape']} | {row['warmup_runs']} | "
            f"{row['layer_id']} | {row['stage_instance_index']} | {row['stage_instance_count']} | "
            f"{row['collector_prefill_aligned']} | {row['prefill_mla_duration_ms']:.6f} | {row['tag']} |"
        )
    out_md.write_text("\n".join(lines) + "\n")

    def _write_subset(title: str, subset: list[dict], path: Path) -> None:
        subset_lines = [
            f"# {title}",
            "",
            "| batch_size | prompt_shape | warmup_runs | layer_id | instance_index | prefill_mla_duration_ms | run_dir |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for row in subset:
            subset_lines.append(
                f"| {row['batch_size']} | {row['prompt_shape']} | {row['warmup_runs']} | "
                f"{row['layer_id']} | {row['stage_instance_index']} | "
                f"{row['prefill_mla_duration_ms']:.6f} | {row['tag']} |"
            )
        path.write_text("\n".join(subset_lines) + "\n")

    _write_subset("Stage1 Prefill MLA Summary Clean", clean_rows, clean_out_md)
    _write_subset(
        "Stage1 Prefill MLA Summary Recommended",
        recommended_rows,
        recommended_out_md,
    )
    _write_subset("Stage1 Prefill MLA Summary Split", split_rows, split_out_md)

    grouped: dict[tuple[int, int, int], list[float]] = {}
    for row in recommended_rows:
        key = (
            int(row["batch_size"]),
            str(row["prompt_shape"]),
            int(row["layer_id"]),
        )
        grouped.setdefault(key, []).append(float(row["prefill_mla_duration_ms"]))

    stats_lines = [
        "# Stage1 Prefill MLA Stats",
        "",
        "| batch_size | prompt_shape | layer_id | samples | mean_ms | std_ms | min_ms | max_ms |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for key in sorted(grouped):
        vals = grouped[key]
        mean = sum(vals) / len(vals)
        var = sum((x - mean) ** 2 for x in vals) / len(vals)
        std = math.sqrt(var)
        stats_lines.append(
            f"| {key[0]} | {key[1]} | {key[2]} | {len(vals)} | "
            f"{mean:.6f} | {std:.6f} | {min(vals):.6f} | {max(vals):.6f} |"
        )
    stats_out_md.write_text("\n".join(stats_lines) + "\n")
    print(out_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
