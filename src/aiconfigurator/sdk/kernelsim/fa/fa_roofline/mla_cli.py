"""Command-line interface for batch BF16 MLA prediction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

from .mla_model import MLA_MODEL_VERSION, estimate_mla
from .mla_profiles import (
    MLA_PROFILE_VERSION,
    MlaReferenceProfile,
    all_mla_profiles,
    get_mla_reference_profile,
)
from .mla_schema import MLA_SCHEMA_VERSION, MlaModelOptions, MlaRequest
from .schema import HardwareSpec


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
        prog="mla-roofline",
        description="BF16 DeepSeek-default MLA roofline estimator",
    )
    parser.add_argument("--hardware", type=Path, help="FA hardware JSON configuration")
    parser.add_argument("--input", type=Path, help="MLA request JSON or request list")
    parser.add_argument("--algorithm", choices=("fa2", "fa3"), default="fa2")
    parser.add_argument(
        "--estimate-level",
        choices=("standard", "high", "low"),
        default="standard",
    )
    parser.add_argument(
        "--profile-file",
        type=Path,
        help="explicit profile JSON; overrides --estimate-level",
    )
    parser.add_argument("--br", type=int)
    parser.add_argument("--bc", type=int)
    parser.add_argument("--decode-splits", type=_split_value, default="auto")
    parser.add_argument("--decode-query-threshold", type=int, default=16)
    parser.add_argument("--max-decode-splits", type=int, default=128)
    parser.add_argument("--min-kv-tiles-per-split", type=int, default=4)
    parser.add_argument("--overlap-fraction", type=float)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--compact", action="store_true")
    parser.add_argument("--list-profiles", action="store_true")
    return parser


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
        sys.stdout.write(_dump([profile.to_dict() for profile in all_mla_profiles()], args.compact))
        return 0
    if args.hardware is None or args.input is None:
        parser.error("--hardware and --input are required unless --list-profiles is used")
    try:
        hardware = HardwareSpec.from_json(args.hardware)
        requests = MlaRequest.many_from_json(args.input)
        options = MlaModelOptions(
            algorithm=args.algorithm,
            estimate_level=args.estimate_level,
            br=args.br,
            bc=args.bc,
            decode_splits=args.decode_splits,
            decode_query_threshold=args.decode_query_threshold,
            max_decode_splits=args.max_decode_splits,
            min_kv_tiles_per_split=args.min_kv_tiles_per_split,
            overlap_fraction=args.overlap_fraction,
        )
        profile = (
            MlaReferenceProfile.from_json(args.profile_file)
            if args.profile_file
            else get_mla_reference_profile(args.estimate_level)
        )
        results = [
            estimate_mla(hardware, request, options, profile).to_dict()
            for request in requests
        ]
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        parser.exit(2, f"mla-roofline: error: {error}\n")
    payload = {
        "model_version": MLA_MODEL_VERSION,
        "profile_version": MLA_PROFILE_VERSION,
        "schema_version": MLA_SCHEMA_VERSION,
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
