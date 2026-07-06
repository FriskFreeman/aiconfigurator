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
    parser.add_argument("--request-json", required=True)
    parser.add_argument("--profile-json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-layers", type=int, default=5)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--attention-backend", default="auto")
    parser.add_argument("--mem-fraction-static", type=float, default=0.5)
    parser.add_argument("--cuda-graph-mode", choices=["off", "on"], default="off")
    parser.add_argument("--pcg-mode", choices=["off", "on"], required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    request = json.loads(Path(args.request_json).read_text())
    profile_cfg = None
    if args.profile_json:
        profile_cfg = json.loads(Path(args.profile_json).read_text())

    engine_kwargs = {
        "model_path": args.model_path,
        "load_format": "dummy",
        "skip_tokenizer_init": True,
        "mem_fraction_static": args.mem_fraction_static,
        "context_length": args.context_length,
        "disable_cuda_graph": args.cuda_graph_mode != "on",
        "disable_radix_cache": True,
        "json_model_override_args": json.dumps({"num_hidden_layers": args.num_layers}),
        "log_level": "info",
        "piecewise_cuda_graph_compiler": "eager",
    }
    if args.attention_backend not in ("", "auto", "default"):
        engine_kwargs["attention_backend"] = args.attention_backend
    server_arg_names = set(inspect.signature(ServerArgs).parameters)
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
    (output_dir / "engine_kwargs.json").write_text(
        json.dumps(engine_kwargs, indent=2, sort_keys=True)
    )
    stage_file = output_dir / "stage.txt"
    stage_file.write_text("before_engine_init\n")

    try:
        with sgl.Engine(**engine_kwargs) as engine:
            stage_file.write_text("after_engine_init\n")

            if profile_cfg is not None:
                stage_file.write_text("before_start_profile\n")
                engine.start_profile(**profile_cfg)
                (output_dir / "start_profile_response.txt").write_text(
                    "Start profiling via Engine API.\n"
                )
                stage_file.write_text("after_start_profile\n")

            stage_file.write_text("before_generate\n")
            result = engine.generate(**request)
            (output_dir / "generate_response.json").write_text(
                json.dumps(result, indent=2)
            )
            stage_file.write_text("after_generate\n")

            if profile_cfg is not None:
                stage_file.write_text("before_stop_profile\n")
                engine.stop_profile()
                (output_dir / "stop_profile_response.txt").write_text(
                    "Stop profiling via Engine API.\n"
                )
                stage_file.write_text("after_stop_profile\n")

        (output_dir / "status.json").write_text(
            json.dumps({"status": "ok", "pcg_mode": args.pcg_mode}, indent=2)
        )
        return 0
    except Exception:
        (output_dir / "engine_exception.txt").write_text(traceback.format_exc())
        (output_dir / "status.json").write_text(
            json.dumps({"status": "fail", "pcg_mode": args.pcg_mode}, indent=2)
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
