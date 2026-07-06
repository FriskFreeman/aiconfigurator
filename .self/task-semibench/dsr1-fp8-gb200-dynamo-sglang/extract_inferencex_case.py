#!/usr/bin/env python3
"""
Extract a normalized InferenceX case bundle for one config key.

This script is designed for AIC-vs-InferenceX alignment work:
1. Read one config key from an InferenceX master config YAML.
2. Expand fixed-seq-len search spaces into point-wise records.
3. Optionally resolve external srt-slurm recipes referenced via CONFIG_FILE=...
4. Optionally load/download benchmark artifacts and filter matching result rows.
5. Write a structured JSON bundle plus CSV views for downstream AIC ingestion.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import subprocess
import sys
import urllib.parse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() == "true"


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _load_yaml(path: Path) -> Any:
    with path.open() as handle:
        return yaml.safe_load(handle)


def _load_json(path: Path) -> Any:
    with path.open() as handle:
        return json.load(handle)


def _fetch_json(base_url: str, path: str, params: dict[str, Any] | None = None) -> Any:
    query = urllib.parse.urlencode(
        {key: value for key, value in (params or {}).items() if value is not None}
    )
    url = f"{base_url.rstrip('/')}{path}"
    if query:
        url = f"{url}?{query}"
    completed = subprocess.run(
        [
            "curl",
            "-fsSL",
            "--compressed",
            "--retry",
            "3",
            "--retry-all-errors",
            "-A",
            "aic-inferencex-extractor/1.0",
            url,
        ],
        check=True,
        capture_output=True,
    )
    payload = completed.stdout
    if payload[:2] == b"\x1f\x8b":
        payload = gzip.decompress(payload)
    return json.loads(payload.decode("utf-8"))


def _flatten_dict(prefix: str, value: Any, out: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            _flatten_dict(next_prefix, child, out)
        return
    if isinstance(value, list):
        out[prefix] = json.dumps(value, ensure_ascii=True)
        return
    out[prefix] = value


def _recipe_ref(additional_settings: list[str] | None) -> str | None:
    for item in additional_settings or []:
        if isinstance(item, str) and item.startswith("CONFIG_FILE="):
            return item.split("=", 1)[1]
    return None


@dataclass
class CaseIdentity:
    config_key: str
    runner: str
    hw: str
    model_prefix: str
    framework: str
    precision: str

    @classmethod
    def from_config(cls, config_key: str, config_entry: dict[str, Any]) -> "CaseIdentity":
        runner = str(config_entry["runner"])
        return cls(
            config_key=config_key,
            runner=runner,
            hw=runner,
            model_prefix=str(config_entry["model-prefix"]),
            framework=str(config_entry["framework"]),
            precision=str(config_entry["precision"]),
        )


def load_case(inferencex_repo: Path, config_key: str) -> tuple[dict[str, Any], Path]:
    config_paths = [
        inferencex_repo / ".github/configs/nvidia-master.yaml",
        inferencex_repo / ".github/configs/amd-master.yaml",
    ]
    for path in config_paths:
        if not path.exists():
            continue
        data = _load_yaml(path) or {}
        if config_key in data:
            return data[config_key], path
    raise KeyError(f"Config key '{config_key}' not found in known InferenceX master configs")


def expand_points(case_id: CaseIdentity, case_entry: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    scenarios = case_entry.get("scenarios", {}).get("fixed-seq-len", []) or []
    scenario_rows: list[dict[str, Any]] = []
    point_rows: list[dict[str, Any]] = []

    for scenario_index, scenario in enumerate(scenarios, start=1):
        isl = _int(scenario["isl"])
        osl = _int(scenario["osl"])
        search_space = scenario.get("search-space", []) or []
        scenario_rows.append(
            {
                "scenario_index": scenario_index,
                "isl": isl,
                "osl": osl,
                "recipe_group_count": len(search_space),
                "point_count": sum(len(item.get("conc-list", []) or []) for item in search_space),
            }
        )

        for group_index, group in enumerate(search_space, start=1):
            prefill = group.get("prefill", {}) or {}
            decode = group.get("decode", {}) or {}
            conc_list = group.get("conc-list", []) or []
            recipe_path = _recipe_ref(prefill.get("additional-settings"))
            base = {
                "config_key": case_id.config_key,
                "hw": case_id.hw,
                "runner": case_id.runner,
                "model_prefix": case_id.model_prefix,
                "framework": case_id.framework,
                "precision": case_id.precision,
                "model": case_entry.get("model"),
                "image": case_entry.get("image"),
                "multinode": _bool(case_entry.get("multinode", False)),
                "disagg": _bool(case_entry.get("disagg", False)),
                "scenario_index": scenario_index,
                "recipe_group_index": group_index,
                "isl": isl,
                "osl": osl,
                "spec_decoding": str(group.get("spec-decoding", "none")),
                "prefill_num_workers": _int(prefill.get("num-worker")),
                "prefill_tp": _int(prefill.get("tp")),
                "prefill_ep": _int(prefill.get("ep", 1)),
                "prefill_dp_attn": _bool(prefill.get("dp-attn", False)),
                "decode_num_workers": _int(decode.get("num-worker")),
                "decode_tp": _int(decode.get("tp")),
                "decode_ep": _int(decode.get("ep", 1)),
                "decode_dp_attn": _bool(decode.get("dp-attn", False)),
                "recipe_path": recipe_path,
            }
            for conc in conc_list:
                row = dict(base)
                row["conc"] = _int(conc)
                point_rows.append(row)

    return scenario_rows, point_rows


def resolve_recipe(recipe_root: Path | None, recipe_path: str | None) -> dict[str, Any] | None:
    if recipe_root is None or not recipe_path:
        return None
    path = recipe_root / recipe_path
    if not path.exists():
        return {
            "recipe_path": recipe_path,
            "resolved": False,
            "resolved_path": str(path),
        }
    data = _load_yaml(path) or {}
    return {
        "recipe_path": recipe_path,
        "resolved": True,
        "resolved_path": str(path),
        "name": data.get("name"),
        "dynamo_version": (data.get("dynamo") or {}).get("version"),
        "frontend": data.get("frontend"),
        "model": data.get("model"),
        "resources": data.get("resources"),
        "backend": data.get("backend"),
        "benchmark": data.get("benchmark"),
    }


def collect_recipes(recipe_root: Path | None, points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: dict[str, dict[str, Any]] = {}
    for point in points:
        recipe_path = point.get("recipe_path")
        if not recipe_path or recipe_path in seen:
            continue
        seen[recipe_path] = resolve_recipe(recipe_root, recipe_path) or {"recipe_path": recipe_path, "resolved": False}
    return [seen[key] for key in sorted(seen)]


def _artifact_json_candidates(artifact_dir: Path) -> list[Path]:
    if not artifact_dir.exists():
        return []
    patterns = [
        "**/agg_bmk.json",
        "**/agg_*.json",
        "**/results_bmk/*.json",
    ]
    seen: set[Path] = set()
    output: list[Path] = []
    for pattern in patterns:
        for path in artifact_dir.glob(pattern):
            if path in seen or not path.is_file():
                continue
            seen.add(path)
            output.append(path)
    return sorted(output)


def _matches_case(row: dict[str, Any], case_id: CaseIdentity) -> bool:
    return (
        str(row.get("infmax_model_prefix", row.get("model_prefix", ""))).lower() == case_id.model_prefix.lower()
        and str(row.get("framework", "")).lower() == case_id.framework.lower()
        and str(row.get("precision", "")).lower() == case_id.precision.lower()
        and str(row.get("hw", "")).lower() == case_id.hw.lower()
    )


def load_results(artifact_dir: Path | None, case_id: CaseIdentity) -> list[dict[str, Any]]:
    if artifact_dir is None:
        return []
    rows: list[dict[str, Any]] = []
    for path in _artifact_json_candidates(artifact_dir):
        payload = _load_json(path)
        for row in payload if isinstance(payload, list) else [payload]:
            if isinstance(row, dict) and _matches_case(row, case_id):
                row = dict(row)
                row["_source_file"] = str(path)
                rows.append(row)
    return rows


def _seq_pairs(points: list[dict[str, Any]]) -> set[tuple[int, int]]:
    return {(_int(point.get("isl")), _int(point.get("osl"))) for point in points}


def _api_case_match(row: dict[str, Any], case_id: CaseIdentity, seq_pairs: set[tuple[int, int]]) -> bool:
    return (
        str(row.get("model", "")).lower() == case_id.model_prefix.lower()
        and str(row.get("hardware", "")).lower() == case_id.hw.lower()
        and str(row.get("framework", "")).lower() == case_id.framework.lower()
        and str(row.get("precision", "")).lower() == case_id.precision.lower()
        and str(row.get("spec_method", "none")).lower() == "none"
        and (_int(row.get("isl")), _int(row.get("osl"))) in seq_pairs
    )


def load_public_api_context(
    base_url: str,
    case_entry: dict[str, Any],
    case_id: CaseIdentity,
    points: list[dict[str, Any]],
    benchmark_date: str | None,
) -> dict[str, Any]:
    seq_pairs = _seq_pairs(points)
    availability = _fetch_json(base_url, "/api/v1/availability")
    latest_images = _fetch_json(base_url, "/api/v1/latest-images")

    availability_rows = [
        row for row in availability if isinstance(row, dict) and _api_case_match(row, case_id, seq_pairs)
    ]
    latest_image_rows = [
        row for row in latest_images if isinstance(row, dict) and _api_case_match(row, case_id, seq_pairs)
    ]

    selected_date = benchmark_date
    if selected_date is None and latest_image_rows:
        selected_date = max(str(row.get("date", "")) for row in latest_image_rows)
    if selected_date is None and availability_rows:
        selected_date = max(str(row.get("date", "")) for row in availability_rows)

    workflow_info: dict[str, Any] | None = None
    benchmark_rows: list[dict[str, Any]] = []
    if selected_date:
        api_model_name = str(case_entry.get("model", "")).split("/")[-1]
        workflow_info = _fetch_json(base_url, "/api/v1/workflow-info", {"date": selected_date})
        benchmark_payload = _fetch_json(
            base_url,
            "/api/v1/benchmarks",
            {
                "model": api_model_name,
                "date": selected_date,
                "exact": "true",
            },
        )
        benchmark_rows = [
            row
            for row in benchmark_payload
            if isinstance(row, dict)
            and _api_case_match(row, case_id, seq_pairs)
            and _bool(row.get("is_multinode", False)) == _bool(case_entry.get("multinode", False))
            and _bool(row.get("disagg", False)) == _bool(case_entry.get("disagg", False))
        ]

    return {
        "selected_date": selected_date,
        "availability_rows": availability_rows,
        "latest_image_rows": latest_image_rows,
        "workflow_info": workflow_info,
        "benchmark_rows": benchmark_rows,
    }


def normalize_public_api_results(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        metrics = row.get("metrics") or {}
        normalized.append(
            {
                "hw": row.get("hardware"),
                "framework": row.get("framework"),
                "precision": row.get("precision"),
                "infmax_model_prefix": row.get("model"),
                "spec_decoding": row.get("spec_method", "none"),
                "disagg": _bool(row.get("disagg", False)),
                "is_multinode": _bool(row.get("is_multinode", False)),
                "isl": _int(row.get("isl")),
                "osl": _int(row.get("osl")),
                "conc": _int(row.get("conc")),
                "prefill_tp": _int(row.get("prefill_tp")),
                "prefill_ep": _int(row.get("prefill_ep", 1)),
                "prefill_dp_attention": _bool(row.get("prefill_dp_attention", False)),
                "prefill_num_workers": _int(row.get("prefill_num_workers", 0)),
                "decode_tp": _int(row.get("decode_tp")),
                "decode_ep": _int(row.get("decode_ep", 1)),
                "decode_dp_attention": _bool(row.get("decode_dp_attention", False)),
                "decode_num_workers": _int(row.get("decode_num_workers", 0)),
                "num_prefill_gpu": _int(row.get("num_prefill_gpu", 0)),
                "num_decode_gpu": _int(row.get("num_decode_gpu", 0)),
                "image": row.get("image"),
                "date": row.get("date"),
                "run_url": row.get("run_url"),
                "_source_type": "public_api",
                **{key: value for key, value in metrics.items() if value is not None},
            }
        )
    return normalized


def join_points_with_results(points: list[dict[str, Any]], results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result_index: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in results:
        key = (
            _int(row.get("isl")),
            _int(row.get("osl")),
            _int(row.get("conc")),
            _int(row.get("prefill_tp", row.get("tp"))),
            _int(row.get("prefill_ep", row.get("ep", 1))),
            _bool(row.get("prefill_dp_attention", row.get("dp_attention", False))),
            _int(row.get("prefill_num_workers", 0)),
            _int(row.get("decode_tp", 0)),
            _int(row.get("decode_ep", 0)),
            _bool(row.get("decode_dp_attention", False)),
            _int(row.get("decode_num_workers", 0)),
        )
        result_index[key] = row

    joined: list[dict[str, Any]] = []
    for point in points:
        key = (
            _int(point["isl"]),
            _int(point["osl"]),
            _int(point["conc"]),
            _int(point["prefill_tp"]),
            _int(point["prefill_ep"]),
            _bool(point["prefill_dp_attn"]),
            _int(point["prefill_num_workers"]),
            _int(point["decode_tp"]),
            _int(point["decode_ep"]),
            _bool(point["decode_dp_attn"]),
            _int(point["decode_num_workers"]),
        )
        joined_row = dict(point)
        result = result_index.get(key)
        joined_row["result_found"] = result is not None
        if result is not None:
            for metric in (
                "tput_per_gpu",
                "output_tput_per_gpu",
                "input_tput_per_gpu",
                "median_ttft",
                "p99_ttft",
                "median_intvty",
                "p99_intvty",
                "median_e2el",
                "p99_e2el",
                "avg_power_w",
                "joules_per_output_token",
                "joules_per_total_token",
            ):
                if metric in result:
                    joined_row[metric] = result[metric]
            joined_row["_source_file"] = result.get("_source_file")
            joined_row["_source_type"] = result.get("_source_type")
            joined_row["result_date"] = result.get("date")
            joined_row["result_run_url"] = result.get("run_url")
            joined_row["num_prefill_gpu"] = result.get("num_prefill_gpu")
            joined_row["num_decode_gpu"] = result.get("num_decode_gpu")
        joined.append(joined_row)
    return joined


def maybe_download_results(run_id: int | None, repo_slug: str, download_dir: Path) -> tuple[Path | None, dict[str, Any]]:
    if run_id is None:
        return None, {"mode": "not_requested"}

    download_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "gh",
        "run",
        "download",
        str(run_id),
        "--repo",
        repo_slug,
        "-n",
        "results_bmk",
        "-D",
        str(download_dir),
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError:
        return None, {"mode": "gh_missing", "command": command}
    except subprocess.CalledProcessError as exc:
        return None, {
            "mode": "download_failed",
            "command": command,
            "returncode": exc.returncode,
            "stderr": exc.stderr[-2000:],
            "stdout": exc.stdout[-2000:],
        }
    return download_dir, {
        "mode": "downloaded",
        "command": command,
        "stdout": completed.stdout[-2000:],
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if rows:
        header_keys: list[str] = []
        for row in rows:
            for key in row.keys():
                if key not in header_keys:
                    header_keys.append(key)
    else:
        header_keys = []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header_keys)
        if header_keys:
            writer.writeheader()
            writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inferencex-repo", type=Path, required=True)
    parser.add_argument("--config-key", required=True)
    parser.add_argument("--srt-slurm-repo", type=Path, default=None)
    parser.add_argument("--artifact-dir", type=Path, default=None)
    parser.add_argument("--github-run-id", type=int, default=None)
    parser.add_argument("--github-repo", default="SemiAnalysisAI/InferenceX")
    parser.add_argument("--use-public-api", action="store_true")
    parser.add_argument("--api-base-url", default="https://inferencex.semianalysis.com")
    parser.add_argument("--benchmark-date", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    case_entry, master_config_path = load_case(args.inferencex_repo, args.config_key)
    case_id = CaseIdentity.from_config(args.config_key, case_entry)
    scenario_rows, point_rows = expand_points(case_id, case_entry)
    recipe_rows = collect_recipes(args.srt_slurm_repo, point_rows)

    artifact_dir = args.artifact_dir
    download_status = {"mode": "not_requested"}
    public_api_context: dict[str, Any] | None = None
    if artifact_dir is None and args.github_run_id is not None:
        artifact_dir, download_status = maybe_download_results(
            args.github_run_id,
            args.github_repo,
            args.output_dir / "_downloaded_artifacts",
        )

    if artifact_dir is not None:
        result_rows = load_results(artifact_dir, case_id)
        result_source = "artifact"
    elif args.use_public_api:
        public_api_context = load_public_api_context(
            args.api_base_url,
            case_entry,
            case_id,
            point_rows,
            args.benchmark_date,
        )
        result_rows = normalize_public_api_results(public_api_context["benchmark_rows"])
        result_source = "public_api"
    else:
        result_rows = []
        result_source = "none"
    joined_rows = join_points_with_results(point_rows, result_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    bundle = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config_key": args.config_key,
        "master_config_path": str(master_config_path),
        "case_identity": {
            "config_key": case_id.config_key,
            "runner": case_id.runner,
            "hw": case_id.hw,
            "model_prefix": case_id.model_prefix,
            "framework": case_id.framework,
            "precision": case_id.precision,
        },
        "sources": {
            "inferencex_repo": str(args.inferencex_repo),
            "srt_slurm_repo": str(args.srt_slurm_repo) if args.srt_slurm_repo else None,
            "artifact_dir": str(artifact_dir) if artifact_dir else None,
            "github_run_id": args.github_run_id,
            "github_repo": args.github_repo if args.github_run_id is not None else None,
            "download_status": download_status,
            "result_source": result_source,
            "public_api_base_url": args.api_base_url if args.use_public_api else None,
        },
        "case_metadata": {
            "image": case_entry.get("image"),
            "model": case_entry.get("model"),
            "model_prefix": case_entry.get("model-prefix"),
            "runner": case_entry.get("runner"),
            "precision": case_entry.get("precision"),
            "framework": case_entry.get("framework"),
            "multinode": _bool(case_entry.get("multinode", False)),
            "disagg": _bool(case_entry.get("disagg", False)),
        },
        "scenario_summary": scenario_rows,
        "points": point_rows,
        "recipes": recipe_rows,
        "results": result_rows,
        "joined_points": joined_rows,
        "public_api": public_api_context,
        "status": {
            "point_count": len(point_rows),
            "recipe_count": len(recipe_rows),
            "result_row_count": len(result_rows),
            "matched_point_count": sum(1 for row in joined_rows if row.get("result_found")),
            "selected_benchmark_date": public_api_context.get("selected_date") if public_api_context else None,
        },
    }

    (args.output_dir / "case_bundle.json").write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    (args.output_dir / "case_metadata.json").write_text(
        json.dumps(bundle["case_metadata"], indent=2), encoding="utf-8"
    )
    (args.output_dir / "recipes.json").write_text(json.dumps(recipe_rows, indent=2), encoding="utf-8")
    (args.output_dir / "results.json").write_text(json.dumps(result_rows, indent=2), encoding="utf-8")
    if public_api_context is not None:
        (args.output_dir / "availability.json").write_text(
            json.dumps(public_api_context["availability_rows"], indent=2),
            encoding="utf-8",
        )
        (args.output_dir / "latest_images.json").write_text(
            json.dumps(public_api_context["latest_image_rows"], indent=2),
            encoding="utf-8",
        )
        (args.output_dir / "workflow_info.json").write_text(
            json.dumps(public_api_context["workflow_info"], indent=2),
            encoding="utf-8",
        )
        (args.output_dir / "benchmark_rows_raw.json").write_text(
            json.dumps(public_api_context["benchmark_rows"], indent=2),
            encoding="utf-8",
        )

    write_csv(args.output_dir / "scenario_summary.csv", scenario_rows)
    write_csv(args.output_dir / "points.csv", point_rows)
    write_csv(args.output_dir / "joined_points.csv", joined_rows)

    if recipe_rows:
        flattened_recipes: list[dict[str, Any]] = []
        for row in recipe_rows:
            flat: dict[str, Any] = {}
            _flatten_dict("", row, flat)
            flattened_recipes.append(flat)
        write_csv(args.output_dir / "recipes_flat.csv", flattened_recipes)
    else:
        write_csv(args.output_dir / "recipes_flat.csv", [])

    if result_rows:
        write_csv(args.output_dir / "results.csv", result_rows)
    else:
        write_csv(args.output_dir / "results.csv", [])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
