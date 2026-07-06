#!/usr/bin/env python3
import argparse
import json
import shlex
import shutil
import subprocess
import time
import traceback
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

NO_PROXY_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        default="booleimg.myaddr.io/lmsysorg/sglang:v0.5.12",
    )
    parser.add_argument("--prefill-gpu", type=int, default=6)
    parser.add_argument("--decode-gpu", type=int, default=7)
    parser.add_argument("--num-layers", type=int, default=5)
    parser.add_argument("--prompt-len", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--attention-backend", default="auto")
    parser.add_argument("--mem-fraction-static", type=float, default=0.5)
    parser.add_argument("--cuda-graph-mode", choices=["off", "on"], default="off")
    parser.add_argument("--pcg-mode", choices=["off", "on"], default="on")
    parser.add_argument("--skip-server-warmup", type=int, choices=[0, 1], default=1)
    parser.add_argument("--request-timeout-sec", type=int, default=300)
    parser.add_argument("--ready-timeout-sec", type=int, default=1800)
    parser.add_argument("--poll-interval-sec", type=int, default=5)
    parser.add_argument("--run-root", default=".self/task-localbench/tmp/tmp-pd")
    parser.add_argument("--prefill-port", type=int, default=31100)
    parser.add_argument("--decode-port", type=int, default=31101)
    parser.add_argument("--router-port", type=int, default=31102)
    parser.add_argument("--bootstrap-port", type=int, default=28998)
    parser.add_argument("--prefill-engine-info-port", type=int, default=36789)
    parser.add_argument("--decode-engine-info-port", type=int, default=36790)
    parser.add_argument(
        "--router-tokenizer-dir",
        default=".self/task-localbench/fake_tokenizer_min",
    )
    parser.add_argument(
        "--transfer-backend",
        default="fake",
        choices=["fake", "mooncake", "nixl", "ascend", "mori"],
    )
    return parser.parse_args()


def shell_join(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def build_local_model(root_dir: Path, local_model_dir: Path) -> None:
    src_cfg = (
        root_dir
        / "src/aiconfigurator/model_configs/deepseek-ai--DeepSeek-V3_config.json"
    )
    cfg = json.loads(src_cfg.read_text())
    cfg["model_type"] = "deepseek_v3"
    cfg["architectures"] = ["DeepseekV3ForCausalLM"]
    cfg.pop("auto_map", None)
    local_model_dir.mkdir(parents=True, exist_ok=True)
    (local_model_dir / "config.json").write_text(json.dumps(cfg, indent=2))


def inject_fake_tokenizer_into_model_dir(
    tokenizer_dir: Path, local_model_dir: Path
) -> None:
    for name in [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ]:
        src = tokenizer_dir / name
        if src.exists():
            shutil.copy2(src, local_model_dir / name)


def build_request(path: Path, prompt_len: int, max_new_tokens: int) -> None:
    payload = {
        "input_ids": list(range(1, prompt_len + 1)),
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": max_new_tokens,
        },
        "stream": False,
        "return_logprob": False,
    }
    path.write_text(json.dumps(payload, indent=2))


def ensure_fake_tokenizer(image: str, tokenizer_dir: Path) -> None:
    if (tokenizer_dir / "tokenizer.json").exists():
        return

    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    script = r"""
from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

out_dir = Path("/out")
vocab = {
    "<pad>": 0,
    "<unk>": 1,
    "<bos>": 2,
    "<eos>": 3,
}
for i in range(4, 4100):
    vocab[f"tok_{i}"] = i

tok = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
tok.pre_tokenizer = Whitespace()
fast = PreTrainedTokenizerFast(
    tokenizer_object=tok,
    unk_token="<unk>",
    pad_token="<pad>",
    bos_token="<bos>",
    eos_token="<eos>",
)
fast.save_pretrained(out_dir)
(out_dir / "README.fake_tokenizer.txt").write_text(
    "Minimal fake tokenizer for local PD/router experiments.\n"
)
"""
    cmd = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{tokenizer_dir}:/out",
        image,
        "python",
        "-c",
        script,
    ]
    subprocess.run(cmd, check=True, text=True, capture_output=True)


def http_get(url: str, timeout: int = 10) -> tuple[int, str]:
    with NO_PROXY_OPENER.open(url, timeout=timeout) as resp:
        return resp.status, resp.read().decode("utf-8", errors="replace")


def http_post_json(url: str, payload: dict, timeout: int) -> tuple[int, str]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with NO_PROXY_OPENER.open(req, timeout=timeout) as resp:
        return resp.status, resp.read().decode("utf-8", errors="replace")


def docker_run_detached(cmd: list[str]) -> str:
    completed = subprocess.run(cmd, text=True, capture_output=True, check=True)
    return completed.stdout.strip()


def docker_logs(name: str) -> str:
    completed = subprocess.run(
        ["docker", "logs", name],
        text=True,
        capture_output=True,
    )
    return completed.stdout + completed.stderr


def docker_inspect(name: str) -> str:
    completed = subprocess.run(
        ["docker", "inspect", name],
        text=True,
        capture_output=True,
    )
    return completed.stdout if completed.returncode == 0 else completed.stderr


def docker_state(name: str) -> str:
    completed = subprocess.run(
        [
            "docker",
            "inspect",
            "-f",
            "{{.State.Status}} exit={{.State.ExitCode}} oom={{.State.OOMKilled}}",
            name,
        ],
        text=True,
        capture_output=True,
    )
    if completed.returncode == 0:
        return completed.stdout.strip()
    return "missing"


def docker_rm_force(name: str) -> None:
    subprocess.run(
        ["docker", "rm", "-f", name],
        text=True,
        capture_output=True,
    )


def wait_http_ready(
    name: str,
    container_name: str,
    url: str,
    ready_timeout_sec: int,
    poll_interval_sec: int,
    poll_log_path: Path,
) -> None:
    start = time.time()
    while time.time() - start < ready_timeout_sec:
        try:
            status, body = http_get(url, timeout=10)
            with poll_log_path.open("a") as f:
                f.write(
                    f"[{datetime.now().isoformat(timespec='seconds')}] "
                    f"{name} ready status={status} url={url}\n"
                )
            return
        except Exception as exc:
            state = docker_state(container_name)
            with poll_log_path.open("a") as f:
                f.write(
                    f"[{datetime.now().isoformat(timespec='seconds')}] "
                    f"{name} not_ready url={url} state={state} "
                    f"error={type(exc).__name__}: {exc}\n"
                )
            if state.startswith("exited") or state.startswith("dead"):
                raise RuntimeError(
                    f"{name} container exited before ready: {container_name} {state}"
                )
            time.sleep(poll_interval_sec)
    raise TimeoutError(f"{name} did not become ready: {url}")


def main() -> int:
    args = parse_args()
    root_dir = Path(__file__).resolve().parents[3]
    run_root = root_dir / args.run_root
    if run_root.name != "output":
        run_root = run_root / "output"
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = (
        run_root
        / (
            f"{run_stamp}_sglang_pd_deepseek_v3_dummy5_"
            f"backend_{args.attention_backend}_cg_{args.cuda_graph_mode}_pcg_{args.pcg_mode}"
        )
    )
    local_model_dir = run_dir / "local_model"
    request_json = run_dir / "generate_request.json"
    poll_log = run_dir / "poll.log"
    router_tokenizer_dir = root_dir / args.router_tokenizer_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    build_local_model(root_dir, local_model_dir)
    build_request(request_json, args.prompt_len, args.max_new_tokens)
    ensure_fake_tokenizer(args.image, router_tokenizer_dir)
    inject_fake_tokenizer_into_model_dir(router_tokenizer_dir, local_model_dir)

    prefill_name = f"aic_pd_prefill_{run_stamp}_{int(time.time())}"
    decode_name = f"aic_pd_decode_{run_stamp}_{int(time.time())}"
    router_name = f"aic_pd_router_{run_stamp}_{int(time.time())}"
    container_names = [prefill_name, decode_name, router_name]

    def write_text(path: Path, text: str) -> None:
        path.write_text(text)

    extra_args = [
        "--load-format",
        "dummy",
        "--skip-tokenizer-init",
        "--host",
        "127.0.0.1",
        "--mem-fraction-static",
        str(args.mem_fraction_static),
        "--context-length",
        str(args.context_length),
        "--disable-radix-cache",
        "--json-model-override-args",
        json.dumps({"num_hidden_layers": args.num_layers}),
        "--piecewise-cuda-graph-compiler",
        "eager",
    ]
    if args.attention_backend not in ("", "auto", "default"):
        extra_args.extend(["--attention-backend", args.attention_backend])
    if args.skip_server_warmup == 1:
        extra_args.append("--skip-server-warmup")
    if args.cuda_graph_mode == "off":
        extra_args.append("--disable-cuda-graph")
    if args.pcg_mode == "off":
        extra_args.append("--disable-piecewise-cuda-graph")
    else:
        extra_args.extend(
            [
                "--enable-piecewise-cuda-graph",
                "--enforce-piecewise-cuda-graph",
            ]
        )

    prefill_cmd = [
        "docker",
        "run",
        "-d",
        "--name",
        prefill_name,
        "--gpus",
        f"device={args.prefill_gpu}",
        "--network",
        "host",
        "--ipc=host",
        "--shm-size",
        "32g",
        "-e",
        "PYTHONDONTWRITEBYTECODE=1",
        "-e",
        "PYTHONPYCACHEPREFIX=/tmp/pycache",
        "-e",
        "TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor",
        "-e",
        "SGLANG_JIT_DEEPGEMM_PRECOMPILE=0",
        "-e",
        "SGLANG_JIT_DEEPGEMM_FAST_WARMUP=1",
        "-v",
        f"{local_model_dir}:/model:ro",
        args.image,
        "sglang",
        "serve",
        "--model-path",
        "/model",
        "--port",
        str(args.prefill_port),
        "--engine-info-bootstrap-port",
        str(args.prefill_engine_info_port),
        "--disaggregation-mode",
        "prefill",
        "--disaggregation-transfer-backend",
        args.transfer_backend,
        "--disaggregation-bootstrap-port",
        str(args.bootstrap_port),
        *extra_args,
    ]
    decode_cmd = [
        "docker",
        "run",
        "-d",
        "--name",
        decode_name,
        "--gpus",
        f"device={args.decode_gpu}",
        "--network",
        "host",
        "--ipc=host",
        "--shm-size",
        "32g",
        "-e",
        "PYTHONDONTWRITEBYTECODE=1",
        "-e",
        "PYTHONPYCACHEPREFIX=/tmp/pycache",
        "-e",
        "TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor",
        "-e",
        "SGLANG_JIT_DEEPGEMM_PRECOMPILE=0",
        "-e",
        "SGLANG_JIT_DEEPGEMM_FAST_WARMUP=1",
        "-v",
        f"{local_model_dir}:/model:ro",
        args.image,
        "sglang",
        "serve",
        "--model-path",
        "/model",
        "--port",
        str(args.decode_port),
        "--engine-info-bootstrap-port",
        str(args.decode_engine_info_port),
        "--disaggregation-mode",
        "decode",
        "--disaggregation-transfer-backend",
        args.transfer_backend,
        "--disaggregation-bootstrap-port",
        str(args.bootstrap_port),
        *extra_args,
    ]
    router_cmd = [
        "docker",
        "run",
        "-d",
        "--name",
        router_name,
        "--network",
        "host",
        "-v",
        f"{router_tokenizer_dir}:/tokenizer:ro",
        "-v",
        f"{local_model_dir}:/model:ro",
        args.image,
        "python",
        "-m",
        "sglang_router.launch_router",
        "--host",
        "127.0.0.1",
        "--port",
        str(args.router_port),
        "--pd-disaggregation",
        "--model-path",
        "/tokenizer",
        "--tokenizer-path",
        "/tokenizer",
        "--prefill",
        f"http://127.0.0.1:{args.prefill_port}",
        str(args.bootstrap_port),
        "--decode",
        f"http://127.0.0.1:{args.decode_port}",
    ]

    write_text(run_dir / "docker_prefill_command.txt", shell_join(prefill_cmd) + "\n")
    write_text(run_dir / "docker_decode_command.txt", shell_join(decode_cmd) + "\n")
    write_text(run_dir / "docker_router_command.txt", shell_join(router_cmd) + "\n")

    result = {
        "status": "unknown",
        "image": args.image,
        "attention_backend": args.attention_backend,
        "cuda_graph_mode": args.cuda_graph_mode,
        "pcg_mode": args.pcg_mode,
        "transfer_backend": args.transfer_backend,
        "prefill_gpu": args.prefill_gpu,
        "decode_gpu": args.decode_gpu,
        "prefill_port": args.prefill_port,
        "decode_port": args.decode_port,
        "router_port": args.router_port,
        "bootstrap_port": args.bootstrap_port,
        "skip_server_warmup": args.skip_server_warmup,
    }

    try:
        docker_run_detached(prefill_cmd)
        docker_run_detached(decode_cmd)
        wait_http_ready(
            "prefill",
            prefill_name,
            f"http://127.0.0.1:{args.prefill_port}/health",
            args.ready_timeout_sec,
            args.poll_interval_sec,
            poll_log,
        )
        wait_http_ready(
            "decode",
            decode_name,
            f"http://127.0.0.1:{args.decode_port}/health",
            args.ready_timeout_sec,
            args.poll_interval_sec,
            poll_log,
        )
        write_text(
            run_dir / "prefill_server_info.json",
            http_get(f"http://127.0.0.1:{args.prefill_port}/server_info", timeout=30)[1],
        )
        write_text(
            run_dir / "decode_server_info.json",
            http_get(f"http://127.0.0.1:{args.decode_port}/server_info", timeout=30)[1],
        )

        docker_run_detached(router_cmd)
        wait_http_ready(
            "router",
            router_name,
            f"http://127.0.0.1:{args.router_port}/health_generate",
            args.ready_timeout_sec,
            args.poll_interval_sec,
            poll_log,
        )
        write_text(
            run_dir / "router_server_info.json",
            http_get(f"http://127.0.0.1:{args.router_port}/server_info", timeout=30)[1],
        )

        request_payload = json.loads(request_json.read_text())
        status, body = http_post_json(
            f"http://127.0.0.1:{args.router_port}/generate",
            request_payload,
            timeout=args.request_timeout_sec,
        )
        write_text(run_dir / "generate_response.json", body)
        result["generate_http_status"] = status
        result["status"] = "ok"
        return_code = 0
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        write_text(run_dir / "http_error_body.txt", body)
        write_text(run_dir / "exception.txt", traceback.format_exc())
        result["status"] = "fail"
        result["http_error"] = str(exc)
        return_code = 1
    except Exception as exc:
        write_text(run_dir / "exception.txt", traceback.format_exc())
        result["status"] = "fail"
        result["error"] = f"{type(exc).__name__}: {exc}"
        return_code = 1
    finally:
        for name, prefix in [
            (prefill_name, "prefill"),
            (decode_name, "decode"),
            (router_name, "router"),
        ]:
            write_text(run_dir / f"{prefix}.docker.logs.txt", docker_logs(name))
            write_text(run_dir / f"{prefix}.docker.inspect.json", docker_inspect(name))
        for name in container_names:
            docker_rm_force(name)
        write_text(run_dir / "run_meta.json", json.dumps(result, indent=2, sort_keys=True))
        print(run_dir)

    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
