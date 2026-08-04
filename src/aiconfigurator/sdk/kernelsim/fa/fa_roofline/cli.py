"""Command-line interface for batch JSON prediction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

from .model import MODEL_VERSION, estimate_attention
from .profiles import PROFILE_VERSION, all_profiles
from .schema import AttentionShape, HardwareSpec, ModelOptions, SCHEMA_VERSION


def _split_value(value: str) -> str | int:
    if value.lower() == "auto":
        return "auto"
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("decode splits must be 'auto' or an integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("decode splits must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fa-roofline",
        description="Standalone FA2/FA3 analytical and generic-profile estimator",
    )
    parser.add_argument("--hardware", type=Path, help="hardware JSON configuration")
    parser.add_argument("--input", type=Path, help="attention shape JSON or shape list")
    parser.add_argument(
        "--mode",
        choices=("analytical", "profiled", "both"),
        default="profiled",
    )
    parser.add_argument("--algorithm", choices=("fa2", "fa3"), default="fa2")
    parser.add_argument(
        "--estimate-level",
        choices=("standard", "high", "low"),
        default="standard",
        help="profiled-mode latency level; high is conservative and low is optimistic",
    )
    parser.add_argument("--br", type=int)
    parser.add_argument("--bc", type=int)
    parser.add_argument("--decode-splits", type=_split_value, default="auto")
    parser.add_argument("--decode-query-threshold", type=int, default=16)
    parser.add_argument("--max-decode-splits", type=int, default=128)
    parser.add_argument("--min-kv-tiles-per-split", type=int, default=4)
    parser.add_argument("--overlap-fraction", type=float)
    parser.add_argument(
        "--no-parallelism-model", action="store_true", help="disable CTA/SM rate scaling"
    )
    parser.add_argument("--no-gqa-hbm-reuse", action="store_true")
    parser.add_argument("--no-gqa-l2-reuse", action="store_true")
    parser.add_argument(
        "--assume-query-tile-l2-reuse",
        action="store_true",
        help="serve repeated query-tile KV requests from L2 after one HBM load",
    )
    parser.add_argument("--store-lse", action="store_true")
    parser.add_argument("--no-kv-cache-update", action="store_true")
    parser.add_argument("--output", type=Path, help="write JSON instead of stdout")
    parser.add_argument("--compact", action="store_true", help="emit compact JSON")
    parser.add_argument(
        "--list-profiles", action="store_true", help="print embedded profiles and exit"
    )
    return parser


def _options(args: argparse.Namespace, mode: str) -> ModelOptions:
    return ModelOptions(
        algorithm=args.algorithm,
        mode=mode,
        estimate_level=args.estimate_level,
        br=args.br,
        bc=args.bc,
        decode_splits=args.decode_splits,
        decode_query_threshold=args.decode_query_threshold,
        max_decode_splits=args.max_decode_splits,
        min_kv_tiles_per_split=args.min_kv_tiles_per_split,
        account_for_parallelism=not args.no_parallelism_model,
        assume_gqa_hbm_reuse=not args.no_gqa_hbm_reuse,
        assume_gqa_l2_reuse=not args.no_gqa_l2_reuse,
        store_lse=args.store_lse,
        include_kv_cache_update=not args.no_kv_cache_update,
        overlap_fraction=args.overlap_fraction,
        assume_query_tile_l2_reuse=args.assume_query_tile_l2_reuse,
    )


def _dump(payload: object, compact: bool) -> str:
    return json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=compact,
        separators=(",", ":") if compact else None,
        indent=None if compact else 2,
    ) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list_profiles:
        sys.stdout.write(_dump([profile.to_dict() for profile in all_profiles()], args.compact))
        return 0
    if args.hardware is None or args.input is None:
        parser.error("--hardware and --input are required unless --list-profiles is used")

    try:
        hardware = HardwareSpec.from_json(args.hardware)
        shapes = AttentionShape.many_from_json(args.input)
        modes = (
            ("analytical", "profiled")
            if args.mode == "both"
            else (args.mode,)
        )
        results = [
            estimate_attention(hardware, shape, _options(args, mode)).to_dict()
            for shape in shapes
            for mode in modes
        ]
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        parser.exit(2, f"fa-roofline: error: {error}\n")

    payload = {
        "model_version": MODEL_VERSION,
        "profile_version": PROFILE_VERSION,
        "schema_version": SCHEMA_VERSION,
        "hardware_file": str(args.hardware),
        "input_file": str(args.input),
        "result_count": len(results),
        "results": results,
    }
    serialized = _dump(payload, args.compact)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    else:
        sys.stdout.write(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
