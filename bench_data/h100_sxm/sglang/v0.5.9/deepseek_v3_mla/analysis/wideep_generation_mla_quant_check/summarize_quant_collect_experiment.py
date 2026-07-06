#!/usr/bin/env python3
"""Compare tiny actual-bf16 vs actual-fp8 collector results with existing data."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


BENCH_ROOT = Path("bench_data/h100_sxm/sglang/v0.5.9/deepseek_v3_mla")
OUT_DIR = BENCH_ROOT / "analysis" / "wideep_generation_mla_quant_check"
SYSTEM_DATA = Path("src/aiconfigurator/systems/data/h100_sxm/sglang/0.5.9/wideep_generation_mla_perf.txt")


def load_perf(path: Path, source: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["source_file"] = str(path)
    df["source"] = source
    return df


def pct(new: float, base: float) -> float | None:
    if base == 0:
        return None
    return (new - base) / base * 100.0


def simple_markdown_table(df: pd.DataFrame) -> str:
    """Render a compact markdown table without depending on tabulate."""
    if df.empty:
        return "No matching rows were produced."
    render_df = df.copy()
    for col in render_df.columns:
        if pd.api.types.is_float_dtype(render_df[col]):
            render_df[col] = render_df[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
    headers = [str(col) for col in render_df.columns]
    rows = [[str(value) for value in row] for row in render_df.to_numpy().tolist()]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT_DIR / "collector_run" / "experiment_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}

    frames = [load_perf(SYSTEM_DATA, "existing_system_data")]
    for mode in manifest.get("modes", []):
        frames.append(load_perf(Path(mode["output_file"]), mode["mode_name"]))
    all_df = pd.concat([df for df in frames if not df.empty], ignore_index=True)

    if all_df.empty:
        raise RuntimeError("No perf data found to summarize.")

    all_df["effective_s"] = all_df["isl"].astype(int) + all_df["step"].astype(int)
    target_cases = {(c["batch_size"], c["seq_len"]) for c in manifest.get("cases", [])}
    if target_cases:
        all_df = all_df[all_df.apply(lambda r: (int(r["batch_size"]), int(r["effective_s"])) in target_cases, axis=1)]
    all_df = all_df[(all_df["kernel_source"] == manifest.get("backend", "fa3")) & (all_df["num_heads"].astype(int) == manifest.get("num_heads", 128))]

    pivot = all_df.pivot_table(
        index=["kernel_source", "num_heads", "batch_size", "effective_s"],
        columns="source",
        values="latency",
        aggfunc="first",
    ).reset_index()
    for col in ["bf16_actual_legacy_logged_fp8", "fp8_actual"]:
        if col in pivot and "existing_system_data" in pivot:
            pivot[f"{col}_gap_pct_vs_existing"] = [
                pct(float(row[col]), float(row["existing_system_data"]))
                if pd.notna(row.get(col)) and pd.notna(row.get("existing_system_data"))
                else None
                for _, row in pivot.iterrows()
            ]
    if {"bf16_actual_legacy_logged_fp8", "fp8_actual"}.issubset(pivot.columns):
        pivot["fp8_gap_pct_vs_bf16_actual"] = [
            pct(float(row["fp8_actual"]), float(row["bf16_actual_legacy_logged_fp8"]))
            if pd.notna(row.get("fp8_actual")) and pd.notna(row.get("bf16_actual_legacy_logged_fp8"))
            else None
            for _, row in pivot.iterrows()
        ]

    all_df.to_csv(OUT_DIR / "collector_quant_experiment_raw_rows.csv", index=False)
    pivot.to_csv(OUT_DIR / "collector_quant_experiment_compare.csv", index=False)

    lines = [
        "# WideEP Generation MLA Quantization Check",
        "",
        "## Scope",
        "",
        "- Existing AIC data file: `src/aiconfigurator/systems/data/h100_sxm/sglang/0.5.9/wideep_generation_mla_perf.txt`.",
        "- Temporary collector output: `collector_run/`.",
        "- This checks whether a real `gemm_type=fp8_block, kv_cache_dtype=fp8` collector run differs materially from the legacy bf16 run whose rows were logged as fp8.",
        "",
        "## Results",
        "",
    ]
    lines.append(simple_markdown_table(pivot))
    lines.extend(
        [
            "",
            "## Output Files",
            "",
            "- `collector_quant_experiment_raw_rows.csv`",
            "- `collector_quant_experiment_compare.csv`",
            "- `real_decode_fp8_trace_summary.csv`",
            "- `real_decode_fp8_kernel_evidence.csv`",
        ]
    )
    (OUT_DIR / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(pivot.to_string(index=False))
    print(f"Wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
