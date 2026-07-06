#!/usr/bin/env python3
import argparse
import inspect
import json
import traceback
from pathlib import Path

import sglang as sgl
from sglang.srt.server_args import ServerArgs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-layers", type=int, default=5)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--attention-backend", default="auto")
    parser.add_argument("--mem-fraction-static", type=float, default=0.5)
    parser.add_argument("--bootstrap-port", type=int, default=28998)
    parser.add_argument("--engine-info-bootstrap-port", type=int, default=36789)
    parser.add_argument("--pcg-mode", choices=["off", "on"], default="off")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stage_file = output_dir / "stage.txt"

    def append_stage(stage: str) -> None:
        with stage_file.open("a") as f:
            f.write(stage + "\n")

    engine_kwargs = {
        "model_path": args.model_path,
        "load_format": "dummy",
        "skip_tokenizer_init": True,
        "mem_fraction_static": args.mem_fraction_static,
        "context_length": args.context_length,
        "disable_cuda_graph": True,
        "disable_radix_cache": True,
        "disaggregation_mode": "prefill",
        "disaggregation_transfer_backend": "nixl",
        "disaggregation_bootstrap_port": args.bootstrap_port,
        "json_model_override_args": json.dumps({"num_hidden_layers": args.num_layers}),
        "log_level": "info",
    }
    server_arg_names = set(inspect.signature(ServerArgs).parameters)
    if "engine_info_bootstrap_port" in server_arg_names:
        engine_kwargs["engine_info_bootstrap_port"] = args.engine_info_bootstrap_port
    if "piecewise_cuda_graph_compiler" in server_arg_names:
        engine_kwargs["piecewise_cuda_graph_compiler"] = "eager"
    if args.pcg_mode == "on":
        if "enable_piecewise_cuda_graph" in server_arg_names:
            engine_kwargs["enable_piecewise_cuda_graph"] = True
        elif "disable_piecewise_cuda_graph" in server_arg_names:
            engine_kwargs["disable_piecewise_cuda_graph"] = False
            if "enforce_piecewise_cuda_graph" in server_arg_names:
                engine_kwargs["enforce_piecewise_cuda_graph"] = True
    else:
        if "enable_piecewise_cuda_graph" in server_arg_names:
            engine_kwargs["enable_piecewise_cuda_graph"] = False
        elif "disable_piecewise_cuda_graph" in server_arg_names:
            engine_kwargs["disable_piecewise_cuda_graph"] = True
    if args.attention_backend not in ("", "auto", "default"):
        engine_kwargs["attention_backend"] = args.attention_backend

    (output_dir / "engine_kwargs.json").write_text(
        json.dumps(engine_kwargs, indent=2, sort_keys=True)
    )
    append_stage("before_engine_init")

    try:
        with sgl.Engine(**engine_kwargs):
            append_stage("after_engine_init")

        append_stage("after_engine_shutdown")
        (output_dir / "status.json").write_text(
            json.dumps(
                {"status": "ok", "mode": "prefill", "pcg_mode": args.pcg_mode},
                indent=2,
            )
        )
        return 0
    except Exception:
        (output_dir / "engine_exception.txt").write_text(traceback.format_exc())
        (output_dir / "status.json").write_text(
            json.dumps(
                {"status": "fail", "mode": "prefill", "pcg_mode": args.pcg_mode},
                indent=2,
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
