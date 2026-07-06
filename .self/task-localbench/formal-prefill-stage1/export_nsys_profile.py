#!/usr/bin/env python3
import argparse
import ast
import collections
import csv
import json
import re
import sqlite3
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--image",
        default="booleimg.myaddr.io/lmsysorg/sglang:v0.5.9",
    )
    return parser.parse_args()


def find_nsys_rep(run_dir: Path) -> Path:
    matches = sorted(run_dir.glob("nsys/*.nsys-rep"))
    if not matches:
        raise FileNotFoundError(f"No .nsys-rep found under {run_dir / 'nsys'}")
    return matches[0]


def run_cmd(cmd: list[str], output_path: Path) -> None:
    completed = subprocess.run(cmd, text=True, capture_output=True)
    output_path.write_text(
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


def run_nsys_in_container(
    image: str,
    run_dir: Path,
    nsys_args: list[str],
    output_path: Path,
) -> None:
    cmd = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{run_dir}:/work",
        "-w",
        "/work",
        image,
    ] + nsys_args
    run_cmd(cmd, output_path)


def export_sqlite(
    image: str,
    run_dir: Path,
    rep_path: Path,
    sqlite_path: Path,
    log_path: Path,
) -> None:
    rep_rel = rep_path.relative_to(run_dir)
    sqlite_rel = sqlite_path.relative_to(run_dir)
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
            str(sqlite_rel.with_suffix("")),
            str(rep_rel),
        ],
        log_path,
    )


def export_stats(image: str, run_dir: Path, rep_path: Path, log_path: Path) -> None:
    rep_rel = rep_path.relative_to(run_dir)
    run_nsys_in_container(
        image,
        run_dir,
        [
            "nsys",
            "stats",
            "--report",
            "nvtx_pushpop_sum,cuda_api_sum,cuda_gpu_kern_sum",
            "--format",
            "csv",
            str(rep_rel),
        ],
        log_path,
    )


def choose_table(conn: sqlite3.Connection, preferred: list[str]) -> str | None:
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    names = {row[0] for row in cur.fetchall()}
    for name in preferred:
        if name in names:
            return name
    return None


def fetch_table_columns(conn: sqlite3.Connection, table_name: str) -> list[str]:
    cur = conn.execute(f"PRAGMA table_info({table_name})")
    return [row[1] for row in cur.fetchall()]


def summarize_sqlite(sqlite_path: Path, summary_path: Path) -> None:
    conn = sqlite3.connect(sqlite_path)
    try:
        tables = [
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
        ]

        nvtx_table = choose_table(
            conn,
            [
                "NVTX_EVENTS",
                "NVTX_EVENT",
                "NVTX_PUSHPOP_EVENTS",
                "NVTX_PUSHPOP_EVENT",
            ],
        )
        string_table = choose_table(conn, ["StringIds", "STRINGS"])

        summary: dict[str, object] = {
            "sqlite_path": str(sqlite_path),
            "tables": tables,
            "nvtx_table": nvtx_table,
            "string_table": string_table,
        }

        if nvtx_table is not None:
            nvtx_cols = fetch_table_columns(conn, nvtx_table)
            summary["nvtx_columns"] = nvtx_cols
            query = f"SELECT * FROM {nvtx_table} LIMIT 20"
            rows = conn.execute(query).fetchall()
            summary["nvtx_preview"] = rows

            # Try to recover text markers if a text/string id column exists.
            candidate_cols = [c for c in nvtx_cols if "text" in c.lower() or "string" in c.lower()]
            summary["nvtx_candidate_text_columns"] = candidate_cols

            module_rows = conn.execute(
                """
                SELECT text, start, end
                FROM NVTX_EVENTS
                WHERE text IS NOT NULL
                  AND text LIKE '%model.model.layers.%'
                ORDER BY start
                """
            ).fetchall()
            summary["layer_marker_count"] = len(module_rows)
            summary["layer_marker_preview"] = module_rows[:40]

            layer_stats: dict[str, dict[str, object]] = {}
            for text, start, end in module_rows:
                if start is None or end is None:
                    continue
                match = re.search(r"model\.model\.layers\.(\d+)(?:[,'}]|\.)", text)
                if not match:
                    continue
                layer_id = match.group(1)
                duration_ns = int(end) - int(start)
                entry = layer_stats.setdefault(
                    layer_id,
                    {
                        "count": 0,
                        "examples": [],
                        "max_duration_ns": 0,
                    },
                )
                entry["count"] = int(entry["count"]) + 1
                if len(entry["examples"]) < 8:
                    entry["examples"].append(
                        {
                            "text": text,
                            "duration_ns": duration_ns,
                            "duration_ms": duration_ns / 1e6,
                        }
                    )
                if duration_ns > int(entry["max_duration_ns"]):
                    entry["max_duration_ns"] = duration_ns

            summary["layer_stats"] = layer_stats

        summary_path.write_text(json.dumps(summary, indent=2, default=str))
    finally:
        conn.close()


def build_markdown_summary(sqlite_path: Path, output_path: Path) -> None:
    conn = sqlite3.connect(sqlite_path)
    try:
        rows = conn.execute(
            """
            SELECT text, start, end
            FROM NVTX_EVENTS
            WHERE text IS NOT NULL
              AND text LIKE '%model.model.layers.%'
              AND start IS NOT NULL
              AND end IS NOT NULL
            ORDER BY start
            """
        ).fetchall()

        all_events = [(str(text), int(start), int(end)) for text, start, end in rows]
        attn_events: list[tuple[str, int, int, str]] = []
        for text, start, end in all_events:
            match = re.fullmatch(
                r"\{'Module': 'model\.model\.layers\.(\d+)\.self_attn'\}",
                text,
            )
            if match:
                attn_events.append((match.group(1), start, end, text))

        lines = ["# Nsys解析摘要", "", f"源文件: `{sqlite_path.name}`", ""]
        if not attn_events:
            lines += ["未发现 `model.model.layers.X.self_attn` NVTX 标记。", ""]
        else:
            lines += [
                "## Layer级别观察",
                "",
                "- 已检测到 `model.model.layers.*` 级别 NVTX marker，说明 `enable_layerwise_nvtx_marker` 在 `nsys` 路径下是生效的。",
                "- 本摘要按 `self_attn` 实例展开，因此 chunked prefill 的每个 chunk 都会单独列出。",
                "- 子模块列表只保留 `self_attn` 的直接子模块，并按执行先后排序，不按耗时排序。",
                "- `rotary_emb` 的 marker 名可能显示为共享对象路径，不能只靠该名字判断真实层号。",
                "",
                "## Self-Attn实例",
                "",
            ]
            prefill_count_by_layer: dict[int, int] = {}
            decode_count_by_layer: dict[int, int] = {}
            for event_order_index, (layer_id, start, end, text) in enumerate(
                attn_events, start=1
            ):
                child_rows = _find_direct_child_module_events(
                    text, start, end, all_events
                )
                stage = _classify_attn_stage_from_children(child_rows)
                layer_int = int(layer_id)
                if stage == "prefill":
                    prefill_count_by_layer[layer_int] = (
                        prefill_count_by_layer.get(layer_int, 0) + 1
                    )
                    instance_index = prefill_count_by_layer[layer_int]
                elif stage == "decode":
                    decode_count_by_layer[layer_int] = (
                        decode_count_by_layer.get(layer_int, 0) + 1
                    )
                    instance_index = decode_count_by_layer[layer_int]
                else:
                    instance_index = 1

                shape_info = _extract_shape_info(child_rows)
                lines.append(
                    f"### Layer {layer_id} / {stage} / instance {instance_index}"
                )
                lines.append("")
                lines.append(f"- 执行序号: `{event_order_index}`")
                lines.append(f"- self_attn总时长: `{(end - start) / 1e6:.3f} ms`")
                if isinstance(shape_info, dict):
                    lines.append(
                        "- Attention形状信息: "
                        f"`module={shape_info.get('module')}`, "
                        f"`token_count={shape_info.get('token_count')}`, "
                        f"`total_tokens={shape_info.get('total_tokens')}`, "
                        f"`chunked_req_prefix_len={shape_info.get('chunked_req_prefix_len')}`, "
                        f"`inputs={shape_info.get('inputs')}`"
                    )
                lines.append("")
                for child_text, child_start, child_end, child_duration in child_rows:
                    lines.append(
                        f"- `{_canonical_child_name(child_text)}` -> "
                        f"{child_duration / 1e6:.3f} ms, "
                        f"start_ns=`{child_start}`"
                    )
                lines.append("")

        output_path.write_text("\n".join(lines))
    finally:
        conn.close()


def _classify_attn_stage_from_children(
    child_rows: list[tuple[str, int, int, int]]
) -> str:
    for text, _, _, _ in child_rows:
        if ".attn_mha" in text:
            return "prefill"
        if ".attn_mqa" in text:
            return "decode"
        if "[[512" in text or "[[1024" in text or "[[2048" in text or "[[4096" in text or "[[8192" in text or "[[16384" in text:
            return "prefill"
        if "[[1," in text or "[[1]" in text:
            return "decode"
    return "unknown"


def _extract_self_attn_suffix(text: str) -> str | None:
    match = re.search(r"self_attn\.([^'\"]+)", text)
    if match:
        return match.group(1)
    return None


def _is_module_event(text: str) -> bool:
    return text.startswith("{'Module': '") or text.startswith('{"Module": "')


def _find_direct_child_module_events(
    parent_text: str,
    parent_start: int,
    parent_end: int,
    all_events: list[tuple[str, int, int]],
) -> list[tuple[str, int, int, int]]:
    parent_path_match = re.search(r"model\.model\.layers\.(\d+)\.self_attn", parent_text)
    parent_layer = parent_path_match.group(1) if parent_path_match else None
    candidates: list[tuple[int, int, str]] = []
    for child_text, child_start, child_end in all_events:
        if child_text == parent_text:
            continue
        if child_start < parent_start or child_end > parent_end:
            continue
        if not _is_module_event(child_text):
            continue
        if "self_attn" not in child_text:
            continue
        suffix = _extract_self_attn_suffix(child_text)
        if suffix is None:
            continue
        # Exclude events from other layers that happen to be nested.
        layer_match = re.search(r"model\.model\.layers\.(\d+)\.self_attn", child_text)
        if layer_match and parent_layer is not None and layer_match.group(1) != parent_layer:
            # Shared objects such as rotary_emb may keep layer.0 in the marker name.
            # Keep only if the suffix is a leaf module name rather than a full nested block.
            if "." in suffix:
                continue
        candidates.append((child_start, child_end, child_text))

    direct_children: list[tuple[str, int, int, int]] = []
    for idx, (child_start, child_end, child_text) in enumerate(candidates):
        is_direct = True
        for jdx, (other_start, other_end, other_text) in enumerate(candidates):
            if idx == jdx:
                continue
            if other_text == child_text and other_start == child_start and other_end == child_end:
                continue
            if other_start <= child_start and child_end <= other_end:
                if (other_start, other_end) != (child_start, child_end):
                    is_direct = False
                    break
        if is_direct:
            direct_children.append(
                (child_text, child_start, child_end, child_end - child_start)
            )

    direct_children.sort(key=lambda item: item[1])
    return direct_children


def _canonical_child_name(text: str) -> str:
    suffix = _extract_self_attn_suffix(text)
    if suffix:
        return suffix
    return text


def _parse_module_event_payload(text: str) -> dict[str, object] | None:
    try:
        payload = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return None
    if isinstance(payload, dict):
        return payload
    return None


def _load_run_meta(sqlite_path: Path) -> dict[str, object]:
    run_dir = sqlite_path.parent.parent
    meta_path = run_dir / "run_meta.json"
    if not meta_path.exists():
        return {}
    try:
        meta = json.loads(meta_path.read_text())
    except json.JSONDecodeError:
        return {}
    if isinstance(meta, dict):
        return meta
    return {}


def _load_engine_kwargs(sqlite_path: Path) -> dict[str, object]:
    run_dir = sqlite_path.parent.parent
    kwargs_path = run_dir / "engine_kwargs.json"
    if not kwargs_path.exists():
        return {}
    try:
        kwargs = json.loads(kwargs_path.read_text())
    except json.JSONDecodeError:
        return {}
    if isinstance(kwargs, dict):
        return kwargs
    return {}


def _reconstruct_prefill_rounds(
    run_meta: dict[str, object],
    engine_kwargs: dict[str, object],
) -> list[dict[str, object]]:
    fresh_lens = run_meta.get("fresh_lens")
    prefix_lens = run_meta.get("prefix_lens")

    if isinstance(fresh_lens, list) and fresh_lens:
        prompt_lens = fresh_lens
    else:
        prompt_lens = run_meta.get("prompt_lens")
    if not isinstance(prompt_lens, list) or not prompt_lens:
        prompt_len = run_meta.get("fresh_len")
        if not isinstance(prompt_len, int):
            prompt_len = run_meta.get("prompt_len")
        batch_size = run_meta.get("batch_size")
        if isinstance(prompt_len, int) and isinstance(batch_size, int):
            prompt_lens = [prompt_len for _ in range(batch_size)]
        else:
            return []
    if not all(isinstance(length, int) and length > 0 for length in prompt_lens):
        return []
    if not isinstance(prefix_lens, list) or len(prefix_lens) != len(prompt_lens):
        prefix_lens = [0 for _ in range(len(prompt_lens))]
    if not all(isinstance(length, int) and length >= 0 for length in prefix_lens):
        prefix_lens = [0 for _ in range(len(prompt_lens))]

    chunked_prefill_size = engine_kwargs.get("chunked_prefill_size")
    if not isinstance(chunked_prefill_size, int) or chunked_prefill_size <= 0:
        chunked_prefill_size = 8192
    max_prefill_tokens = engine_kwargs.get("max_prefill_tokens")
    if not isinstance(max_prefill_tokens, int) or max_prefill_tokens <= 0:
        max_prefill_tokens = 16384

    waiting = [
        {
            "req_id": req_id,
            "remaining": int(prompt_len),
            "prefix_len": int(prefix_lens[req_id]),
            "prompt_len": int(prefix_lens[req_id]) + int(prompt_len),
        }
        for req_id, prompt_len in enumerate(prompt_lens)
    ]
    chunked_req: dict[str, int] | None = None
    rounds: list[dict[str, object]] = []

    while waiting or chunked_req is not None:
        rem_input_tokens = max_prefill_tokens
        rem_chunk_tokens = chunked_prefill_size
        batch_items: list[dict[str, int]] = []
        current_chunked_req_prefix_len = 0

        if chunked_req is not None:
            take = min(chunked_req["remaining"], rem_chunk_tokens)
            current_chunked_req_prefix_len = chunked_req["prefix_len"]
            batch_items.append(
                {
                    "req_id": chunked_req["req_id"],
                    "extend_len": take,
                    "prefix_len": chunked_req["prefix_len"],
                    "seq_len_after": chunked_req["prefix_len"] + take,
                    "prompt_len": chunked_req["prompt_len"],
                    "is_chunked_req": 1,
                }
            )
            chunked_req["prefix_len"] += take
            chunked_req["remaining"] -= take
            rem_input_tokens -= take
            rem_chunk_tokens -= take
            if chunked_req["remaining"] == 0:
                chunked_req = None

        next_waiting: list[dict[str, int]] = []
        for req in waiting:
            if chunked_req is not None:
                next_waiting.append(req)
                continue

            input_tokens = req["remaining"]
            if input_tokens >= rem_input_tokens and batch_items:
                next_waiting.append(req)
                continue

            if input_tokens <= rem_chunk_tokens:
                batch_items.append(
                    {
                        "req_id": req["req_id"],
                        "extend_len": input_tokens,
                        "prefix_len": req["prefix_len"],
                        "seq_len_after": req["prefix_len"] + input_tokens,
                        "prompt_len": req["prompt_len"],
                        "is_chunked_req": 0,
                    }
                )
                req["prefix_len"] += input_tokens
                req["remaining"] = 0
                rem_input_tokens -= input_tokens
                rem_chunk_tokens -= input_tokens
                continue

            take = rem_chunk_tokens
            if take <= 0:
                next_waiting.append(req)
                continue

            batch_items.append(
                {
                    "req_id": req["req_id"],
                    "extend_len": take,
                    "prefix_len": req["prefix_len"],
                    "seq_len_after": req["prefix_len"] + take,
                    "prompt_len": req["prompt_len"],
                    "is_chunked_req": 1,
                }
            )
            current_chunked_req_prefix_len = req["prefix_len"]
            req["prefix_len"] += take
            req["remaining"] -= take
            rem_input_tokens -= take
            rem_chunk_tokens = 0
            chunked_req = req

        waiting = [req for req in next_waiting if req["remaining"] > 0]
        rounds.append(
            {
                "round_index": len(rounds) + 1,
                "items": batch_items,
                "sum_extend_len": sum(item["extend_len"] for item in batch_items),
                "sum_seq_len_after": sum(item["seq_len_after"] for item in batch_items),
                "sum_prefix_len": sum(item["prefix_len"] for item in batch_items),
                "current_chunked_req_prefix_len": (
                    max(
                        [
                            item["prefix_len"]
                            for item in batch_items
                            if item["is_chunked_req"]
                        ]
                        or [current_chunked_req_prefix_len]
                    )
                ),
                "chunked_req_prefix_lens": [
                    item["prefix_len"] for item in batch_items if item["is_chunked_req"]
                ],
                "chunked_req_ids": [
                    item["req_id"] for item in batch_items if item["is_chunked_req"]
                ],
                "chunked_prefill_size": chunked_prefill_size,
                "max_prefill_tokens": max_prefill_tokens,
            }
        )

    return rounds


def _build_operator_info(
    child_text: str,
    canonical_name: str,
    run_meta: dict[str, object],
) -> dict[str, object]:
    payload = _parse_module_event_payload(child_text) or {}
    trainable_params = payload.get("TrainableParams")
    inputs = payload.get("Inputs")
    module_name = payload.get("Module")
    attention_backend = run_meta.get("attention_backend")

    info: dict[str, object] = {
        "module_name": module_name,
        "canonical_name": canonical_name,
        "inputs": inputs,
        "trainable_params": trainable_params,
        "operator_category": "unknown",
        "implementation": "unknown",
        "implementation_source": "not_inferred",
    }

    if canonical_name in {"attn_mha", "attn_mqa"}:
        info.update(
            {
                "operator_category": "attention",
                "implementation": attention_backend or "unknown",
                "implementation_source": "run_meta.attention_backend",
            }
        )
        return info

    if isinstance(trainable_params, dict) and "weight" in trainable_params:
        has_fp8_scale = "weight_scale_inv" in trainable_params
        info["operator_category"] = "linear"
        if has_fp8_scale:
            info.update(
                {
                    "implementation": "deepgemm_fp8_inferred",
                    "implementation_source": "fp8_weight_scale_inv_marker",
                }
            )
        else:
            info.update(
                {
                    "implementation": "linear_backend_unspecified",
                    "implementation_source": "module_marker",
                }
            )
        return info

    if "layernorm" in canonical_name:
        info.update(
            {
                "operator_category": "normalization",
                "implementation": "rmsnorm_or_layernorm",
                "implementation_source": "module_name",
            }
        )
        return info

    if canonical_name == "rotary_emb":
        info.update(
            {
                "operator_category": "position_embedding",
                "implementation": "rotary_embedding",
                "implementation_source": "module_name",
            }
        )
        return info

    return info


def _load_string_map(conn: sqlite3.Connection) -> dict[int, str]:
    table_name = choose_table(conn, ["StringIds", "STRINGS"])
    if table_name is None:
        return {}
    return {
        int(row[0]): str(row[1])
        for row in conn.execute(f"SELECT id, value FROM {table_name}")
    }


def _resolve_string(string_map: dict[int, str], value: object) -> object:
    if isinstance(value, int):
        return string_map.get(value, value)
    return value


def _classify_from_kernel_names(
    short_names: list[str],
    demangled_names: list[str],
) -> str:
    joined = " | ".join(short_names + demangled_names).lower()
    if not joined:
        return "unknown"
    if "gemm" in joined or "group_gemm" in joined or "matmul" in joined:
        return "linear_or_gemm"
    if "flashattn" in joined or "flashattnfwd" in joined or "device_kernel" in joined:
        return "attention_or_fused_attention"
    if "rotary" in joined:
        return "position_embedding"
    if "norm" in joined:
        return "normalization"
    if "quant" in joined:
        return "quantization"
    if "concat" in joined or "scan" in joined:
        return "tensor_transform"
    return "cuda_kernel_group"


def _fetch_module_gpu_trace(
    conn: sqlite3.Connection,
    string_map: dict[int, str],
    module_start_ns: int,
    module_end_ns: int,
    module_global_tid: int | None,
) -> dict[str, object]:
    runtime_query = """
        SELECT start, end, globalTid, correlationId, nameId
        FROM CUPTI_ACTIVITY_KIND_RUNTIME
        WHERE start >= ? AND end <= ?
    """
    params: list[object] = [module_start_ns, module_end_ns]
    if module_global_tid is not None:
        runtime_query += " AND globalTid = ?"
        params.append(module_global_tid)
    runtime_query += " ORDER BY start"

    runtime_rows = conn.execute(runtime_query, params).fetchall()
    runtime_calls: list[dict[str, object]] = []
    kernels: list[dict[str, object]] = []

    for runtime_start, runtime_end, global_tid, correlation_id, name_id in runtime_rows:
        launch_api = _resolve_string(string_map, name_id)
        runtime_calls.append(
            {
                "start_ns": int(runtime_start),
                "end_ns": int(runtime_end),
                "duration_ns": int(runtime_end) - int(runtime_start),
                "global_tid": int(global_tid),
                "correlation_id": int(correlation_id),
                "launch_api": launch_api,
            }
        )
        kernel_rows = conn.execute(
            """
            SELECT start, end, demangledName, shortName, streamId, correlationId
            FROM CUPTI_ACTIVITY_KIND_KERNEL
            WHERE correlationId = ?
            ORDER BY start
            """,
            (correlation_id,),
        ).fetchall()
        for (
            kernel_start,
            kernel_end,
            demangled_name,
            short_name,
            stream_id,
            kernel_corr,
        ) in kernel_rows:
            kernels.append(
                {
                    "start_ns": int(kernel_start),
                    "end_ns": int(kernel_end),
                    "duration_ns": int(kernel_end) - int(kernel_start),
                    "stream_id": int(stream_id),
                    "correlation_id": int(kernel_corr),
                    "short_name": _resolve_string(string_map, short_name),
                    "demangled_name": _resolve_string(string_map, demangled_name),
                    "launch_api": launch_api,
                    "launch_runtime_start_ns": int(runtime_start),
                    "launch_runtime_end_ns": int(runtime_end),
                }
            )

    kernels.sort(key=lambda item: int(item["start_ns"]))
    total_gpu_kernel_time_ns = sum(int(item["duration_ns"]) for item in kernels)

    kernel_time_by_short_name: dict[str, int] = collections.Counter()
    for kernel in kernels:
        short_name = str(kernel.get("short_name") or "unknown")
        kernel_time_by_short_name[short_name] += int(kernel["duration_ns"])

    top_kernels = [
        {
            "short_name": name,
            "gpu_time_ns": duration_ns,
            "gpu_time_ms": duration_ns / 1e6,
        }
        for name, duration_ns in sorted(
            kernel_time_by_short_name.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:5]
    ]
    dominant_kernel_name = top_kernels[0]["short_name"] if top_kernels else "unknown"
    short_names = [str(item.get("short_name") or "") for item in kernels]
    demangled_names = [str(item.get("demangled_name") or "") for item in kernels]

    return {
        "runtime_call_count": len(runtime_calls),
        "runtime_calls": runtime_calls,
        "gpu_kernel_count": len(kernels),
        "gpu_kernels": kernels,
        "gpu_kernel_time_ns_total": total_gpu_kernel_time_ns,
        "gpu_kernel_time_ms_total": total_gpu_kernel_time_ns / 1e6,
        "top_gpu_kernels": top_kernels,
        "dominant_kernel_name": dominant_kernel_name,
        "dominant_kernel_time_ns": top_kernels[0]["gpu_time_ns"] if top_kernels else 0,
        "trace_operator_category": _classify_from_kernel_names(short_names, demangled_names),
    }


def _build_operator_info_from_trace(
    child_text: str,
    canonical_name: str,
    trace_info: dict[str, object],
) -> dict[str, object]:
    payload = _parse_module_event_payload(child_text) or {}
    trainable_params = payload.get("TrainableParams")
    inputs = payload.get("Inputs")
    module_name = payload.get("Module")
    gpu_kernel_count = int(trace_info.get("gpu_kernel_count", 0) or 0)
    dominant_kernel_name = trace_info.get("dominant_kernel_name") or "unknown"

    info: dict[str, object] = {
        "module_name": module_name,
        "canonical_name": canonical_name,
        "inputs": inputs,
        "trainable_params": trainable_params,
        "operator_category": trace_info.get("trace_operator_category", "unknown"),
        "implementation": (
            dominant_kernel_name if gpu_kernel_count > 0 else "unknown"
        ),
        "implementation_source": (
            "cupti_kernel.shortName" if gpu_kernel_count > 0 else "not_inferred"
        ),
        "gpu_kernel_count": gpu_kernel_count,
        "gpu_kernel_time_ns_total": trace_info.get("gpu_kernel_time_ns_total", 0),
        "gpu_kernel_time_ms_total": trace_info.get("gpu_kernel_time_ms_total", 0.0),
        "runtime_call_count": trace_info.get("runtime_call_count", 0),
        "dominant_kernel_name": dominant_kernel_name,
        "dominant_kernel_time_ns": trace_info.get("dominant_kernel_time_ns", 0),
        "top_gpu_kernels": trace_info.get("top_gpu_kernels", []),
        "trace_backend_raw": {
            "runtime_calls": trace_info.get("runtime_calls", []),
            "gpu_kernels": trace_info.get("gpu_kernels", []),
        },
    }
    return info


def _dedupe_kernels(kernels: list[dict[str, object]]) -> list[dict[str, object]]:
    deduped: list[dict[str, object]] = []
    seen: set[tuple[object, ...]] = set()
    for kernel in kernels:
        key = (
            kernel.get("correlation_id"),
            kernel.get("start_ns"),
            kernel.get("end_ns"),
            kernel.get("stream_id"),
            kernel.get("short_name"),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(kernel)
    deduped.sort(key=lambda item: int(item.get("start_ns", 0)))
    return deduped


def _compute_timing_breakdown(
    host_start_ns: int,
    host_end_ns: int,
    kernels: list[dict[str, object]],
) -> dict[str, object]:
    deduped_kernels = _dedupe_kernels(kernels)
    host_duration_ns = host_end_ns - host_start_ns

    if not deduped_kernels:
        return {
            "host_start_ns": host_start_ns,
            "host_end_ns": host_end_ns,
            "host_duration_ns": host_duration_ns,
            "host_duration_ms": host_duration_ns / 1e6,
            "first_kernel_start_ns": None,
            "last_kernel_end_ns": None,
            "module_total_to_last_kernel_ns": host_duration_ns,
            "module_total_to_last_kernel_ms": host_duration_ns / 1e6,
            "host_to_first_kernel_gap_ns": None,
            "host_to_first_kernel_gap_ms": None,
            "host_end_to_last_kernel_tail_ns": 0,
            "host_end_to_last_kernel_tail_ms": 0.0,
            "host_end_to_last_kernel_tail_ns_signed": None,
            "host_end_to_last_kernel_tail_ms_signed": None,
            "gpu_makespan_ns": 0,
            "gpu_makespan_ms": 0.0,
            "gpu_kernel_time_sum_ns": 0,
            "gpu_kernel_time_sum_ms": 0.0,
            "gpu_kernel_count": 0,
            "first_kernel_name": None,
            "last_kernel_name": None,
        }

    first_kernel = deduped_kernels[0]
    last_kernel = max(deduped_kernels, key=lambda item: int(item["end_ns"]))
    first_kernel_start_ns = int(first_kernel["start_ns"])
    last_kernel_end_ns = int(last_kernel["end_ns"])
    gpu_kernel_time_sum_ns = sum(int(item["duration_ns"]) for item in deduped_kernels)
    host_to_first_kernel_gap_ns = first_kernel_start_ns - host_start_ns
    host_end_to_last_kernel_tail_ns_signed = last_kernel_end_ns - host_end_ns
    host_end_to_last_kernel_tail_ns = max(0, host_end_to_last_kernel_tail_ns_signed)
    module_total_to_last_kernel_ns = max(host_end_ns, last_kernel_end_ns) - host_start_ns
    gpu_makespan_ns = last_kernel_end_ns - first_kernel_start_ns

    return {
        "host_start_ns": host_start_ns,
        "host_end_ns": host_end_ns,
        "host_duration_ns": host_duration_ns,
        "host_duration_ms": host_duration_ns / 1e6,
        "first_kernel_start_ns": first_kernel_start_ns,
        "last_kernel_end_ns": last_kernel_end_ns,
        "module_total_to_last_kernel_ns": module_total_to_last_kernel_ns,
        "module_total_to_last_kernel_ms": module_total_to_last_kernel_ns / 1e6,
        "host_to_first_kernel_gap_ns": host_to_first_kernel_gap_ns,
        "host_to_first_kernel_gap_ms": host_to_first_kernel_gap_ns / 1e6,
        "host_end_to_last_kernel_tail_ns": host_end_to_last_kernel_tail_ns,
        "host_end_to_last_kernel_tail_ms": host_end_to_last_kernel_tail_ns / 1e6,
        "host_end_to_last_kernel_tail_ns_signed": host_end_to_last_kernel_tail_ns_signed,
        "host_end_to_last_kernel_tail_ms_signed": host_end_to_last_kernel_tail_ns_signed / 1e6,
        "gpu_makespan_ns": gpu_makespan_ns,
        "gpu_makespan_ms": gpu_makespan_ns / 1e6,
        "gpu_kernel_time_sum_ns": gpu_kernel_time_sum_ns,
        "gpu_kernel_time_sum_ms": gpu_kernel_time_sum_ns / 1e6,
        "gpu_kernel_count": len(deduped_kernels),
        "first_kernel_name": first_kernel.get("short_name"),
        "last_kernel_name": last_kernel.get("short_name"),
    }


def _build_child_timing_summary(child_entry: dict[str, object]) -> dict[str, object]:
    operator_info = child_entry.get("operator_info")
    trace_backend_raw = (
        operator_info.get("trace_backend_raw")
        if isinstance(operator_info, dict)
        else None
    )
    kernels = (
        trace_backend_raw.get("gpu_kernels", [])
        if isinstance(trace_backend_raw, dict)
        else []
    )
    timing = _compute_timing_breakdown(
        int(child_entry["start_ns"]),
        int(child_entry["end_ns"]),
        kernels if isinstance(kernels, list) else [],
    )
    return {
        "canonical_name": child_entry.get("canonical_name"),
        "order_index": child_entry.get("order_index"),
        **timing,
    }


def _build_module_timing_summary(
    module_start_ns: int,
    module_end_ns: int,
    child_entries: list[dict[str, object]],
) -> dict[str, object]:
    kernels: list[dict[str, object]] = []
    child_timing: list[dict[str, object]] = []
    for child in child_entries:
        child_timing.append(_build_child_timing_summary(child))
        operator_info = child.get("operator_info")
        trace_backend_raw = (
            operator_info.get("trace_backend_raw")
            if isinstance(operator_info, dict)
            else None
        )
        child_kernels = (
            trace_backend_raw.get("gpu_kernels", [])
            if isinstance(trace_backend_raw, dict)
            else []
        )
        if isinstance(child_kernels, list):
            kernels.extend(child_kernels)
    timing = _compute_timing_breakdown(module_start_ns, module_end_ns, kernels)
    timing["children"] = child_timing
    return timing


def _write_module_timing_csv(
    json_summary: list[dict[str, object]],
    csv_output_path: Path,
) -> None:
    fieldnames = [
        "event_order_index",
        "run_instance_name",
        "tp_size",
        "layer_id",
        "stage",
        "stage_instance_index",
        "stage_instance_count",
        "collector_prefill_aligned",
        "module_host_start_ns",
        "module_host_end_ns",
        "module_host_duration_ns",
        "module_host_duration_ms",
        "module_first_kernel_start_ns",
        "module_last_kernel_end_ns",
        "module_total_to_last_kernel_ns",
        "module_total_to_last_kernel_ms",
        "module_host_to_first_kernel_gap_ns",
        "module_host_to_first_kernel_gap_ms",
        "module_host_end_to_last_kernel_tail_ns",
        "module_host_end_to_last_kernel_tail_ms",
        "module_host_end_to_last_kernel_tail_ns_signed",
        "module_host_end_to_last_kernel_tail_ms_signed",
        "module_gpu_makespan_ns",
        "module_gpu_makespan_ms",
        "module_gpu_kernel_time_sum_ns",
        "module_gpu_kernel_time_sum_ms",
        "module_gpu_kernel_count",
        "module_first_kernel_name",
        "module_last_kernel_name",
        "attention_module",
        "attention_token_count",
        "attention_total_tokens",
        "current_chunked_req_prefix_len",
        "current_chunked_req_prefix_len_source",
        "chunk_req_ids_json",
        "chunk_req_prefix_lens_json",
        "chunk_req_fresh_lens_json",
        "chunk_req_total_seq_lens_json",
        "chunk_req_prompt_lens_json",
        "chunk_req_is_chunked_json",
        "chunk_req_items_json",
        "child_timing_json",
        "kernel_summary_json",
    ]
    with csv_output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        run_instance_name = csv_output_path.parent.parent.name
        run_meta = _load_run_meta(csv_output_path)
        tp_size = run_meta.get("tp_size")
        for item in json_summary:
            timing = item.get("timing_breakdown")
            if not isinstance(timing, dict):
                continue
            shape_info = item.get("shape_info") if isinstance(item.get("shape_info"), dict) else {}
            child_timing_json = json.dumps(timing.get("children", []), ensure_ascii=False)
            kernel_summary = []
            for child in item.get("children", []):
                operator_info = child.get("operator_info") if isinstance(child, dict) else None
                if not isinstance(operator_info, dict):
                    continue
                kernel_summary.append(
                    {
                        "canonical_name": child.get("canonical_name"),
                        "top_gpu_kernels": operator_info.get("top_gpu_kernels", []),
                        "gpu_kernel_count": operator_info.get("gpu_kernel_count"),
                    }
                )
            writer.writerow(
                {
                    "event_order_index": item.get("event_order_index"),
                    "run_instance_name": run_instance_name,
                    "tp_size": tp_size,
                    "layer_id": item.get("layer_id"),
                    "stage": item.get("stage"),
                    "stage_instance_index": item.get("stage_instance_index"),
                    "stage_instance_count": item.get("stage_instance_count"),
                    "collector_prefill_aligned": item.get("collector_prefill_aligned"),
                    "module_host_start_ns": timing.get("host_start_ns"),
                    "module_host_end_ns": timing.get("host_end_ns"),
                    "module_host_duration_ns": timing.get("host_duration_ns"),
                    "module_host_duration_ms": timing.get("host_duration_ms"),
                    "module_first_kernel_start_ns": timing.get("first_kernel_start_ns"),
                    "module_last_kernel_end_ns": timing.get("last_kernel_end_ns"),
                    "module_total_to_last_kernel_ns": timing.get("module_total_to_last_kernel_ns"),
                    "module_total_to_last_kernel_ms": timing.get("module_total_to_last_kernel_ms"),
                    "module_host_to_first_kernel_gap_ns": timing.get("host_to_first_kernel_gap_ns"),
                    "module_host_to_first_kernel_gap_ms": timing.get("host_to_first_kernel_gap_ms"),
                    "module_host_end_to_last_kernel_tail_ns": timing.get("host_end_to_last_kernel_tail_ns"),
                    "module_host_end_to_last_kernel_tail_ms": timing.get("host_end_to_last_kernel_tail_ms"),
                    "module_host_end_to_last_kernel_tail_ns_signed": timing.get("host_end_to_last_kernel_tail_ns_signed"),
                    "module_host_end_to_last_kernel_tail_ms_signed": timing.get("host_end_to_last_kernel_tail_ms_signed"),
                    "module_gpu_makespan_ns": timing.get("gpu_makespan_ns"),
                    "module_gpu_makespan_ms": timing.get("gpu_makespan_ms"),
                    "module_gpu_kernel_time_sum_ns": timing.get("gpu_kernel_time_sum_ns"),
                    "module_gpu_kernel_time_sum_ms": timing.get("gpu_kernel_time_sum_ms"),
                    "module_gpu_kernel_count": timing.get("gpu_kernel_count"),
                    "module_first_kernel_name": timing.get("first_kernel_name"),
                    "module_last_kernel_name": timing.get("last_kernel_name"),
                    "attention_module": shape_info.get("module"),
                    "attention_token_count": shape_info.get("token_count"),
                    "attention_total_tokens": shape_info.get("total_tokens"),
                    "current_chunked_req_prefix_len": shape_info.get("current_chunked_req_prefix_len"),
                    "current_chunked_req_prefix_len_source": shape_info.get("current_chunked_req_prefix_len_source"),
                    "chunk_req_ids_json": json.dumps(shape_info.get("chunk_req_ids", []), ensure_ascii=False),
                    "chunk_req_prefix_lens_json": json.dumps(shape_info.get("chunk_req_prefix_lens", []), ensure_ascii=False),
                    "chunk_req_fresh_lens_json": json.dumps(shape_info.get("chunk_req_fresh_lens", []), ensure_ascii=False),
                    "chunk_req_total_seq_lens_json": json.dumps(shape_info.get("chunk_req_total_seq_lens", []), ensure_ascii=False),
                    "chunk_req_prompt_lens_json": json.dumps(shape_info.get("chunk_req_prompt_lens", []), ensure_ascii=False),
                    "chunk_req_is_chunked_json": json.dumps(shape_info.get("chunk_req_is_chunked", []), ensure_ascii=False),
                    "chunk_req_items_json": json.dumps(
                        (
                            shape_info.get("reconstructed_prefill_round", {}).get("items", [])
                            if isinstance(shape_info.get("reconstructed_prefill_round"), dict)
                            else []
                        ),
                        ensure_ascii=False,
                    ),
                    "child_timing_json": child_timing_json,
                    "kernel_summary_json": json.dumps(kernel_summary, ensure_ascii=False),
                }
            )


def _extract_shape_info(
    child_rows: list[tuple[str, int, int, int]]
) -> dict[str, object] | None:
    for child_text, _, _, _ in child_rows:
        payload = _parse_module_event_payload(child_text)
        if payload is None:
            continue
        module_name = payload.get("Module")
        if not isinstance(module_name, str):
            continue
        if not (module_name.endswith(".attn_mha") or module_name.endswith(".attn_mqa")):
            continue
        inputs = payload.get("Inputs")
        token_count = None
        total_tokens = None
        if (
            isinstance(inputs, list)
            and inputs
            and isinstance(inputs[0], list)
            and inputs[0]
            and isinstance(inputs[0][0], int)
        ):
            token_count = inputs[0][0]
        if (
            isinstance(inputs, list)
            and len(inputs) >= 2
            and isinstance(inputs[1], list)
            and inputs[1]
            and isinstance(inputs[1][0], int)
        ):
            total_tokens = inputs[1][0]
        chunked_req_prefix_len = 0
        if token_count is not None and total_tokens is not None and total_tokens >= token_count:
            chunked_req_prefix_len = total_tokens - token_count
        return {
            "module": module_name.rsplit(".", 1)[-1],
            "inputs": inputs,
            "token_count": token_count,
            "total_tokens": total_tokens,
            "chunked_req_prefix_len": chunked_req_prefix_len,
        }
    return None


def _extract_prefill_chunk_events(
    all_events: list[tuple[str, int, int]]
) -> list[dict[str, object]]:
    chunk_events: list[dict[str, object]] = []
    for text, start, end in all_events:
        if "AICPrefillChunk" not in text:
            continue
        payload = _parse_module_event_payload(text)
        if payload is None:
            continue
        chunk_payload = payload.get("AICPrefillChunk")
        if not isinstance(chunk_payload, dict):
            continue
        chunk_events.append(
            {
                "start_ns": start,
                "end_ns": end,
                "duration_ns": end - start,
                "payload": chunk_payload,
                "raw_text": text,
            }
        )
    chunk_events.sort(key=lambda item: int(item["start_ns"]))
    return chunk_events


def _find_prefill_chunk_for_event(
    event_start: int,
    event_end: int,
    chunk_events: list[dict[str, object]],
) -> dict[str, object] | None:
    containing = [
        item
        for item in chunk_events
        if int(item["start_ns"]) <= event_start and event_end <= int(item["end_ns"])
    ]
    if containing:
        return min(containing, key=lambda item: int(item["duration_ns"]))

    overlapping = [
        item
        for item in chunk_events
        if int(item["start_ns"]) < event_end and event_start < int(item["end_ns"])
    ]
    if not overlapping:
        return None

    def overlap_ns(item: dict[str, object]) -> int:
        return min(event_end, int(item["end_ns"])) - max(event_start, int(item["start_ns"]))

    return max(overlapping, key=overlap_ns)


def _merge_scheduler_chunk_info(
    shape_info: dict[str, object] | None,
    chunk_event: dict[str, object] | None,
    reconstructed_round: dict[str, object] | None = None,
) -> dict[str, object] | None:
    if shape_info is None and chunk_event is None:
        return None

    merged = dict(shape_info or {})
    inferred_chunk_prefix_len = merged.get("chunked_req_prefix_len", 0)
    if reconstructed_round is not None:
        round_items = reconstructed_round.get("items", [])
        if not isinstance(round_items, list):
            round_items = []
        reconstructed_prefix_len = reconstructed_round.get(
            "current_chunked_req_prefix_len", 0
        )
        merged.update(
            {
                "current_chunked_req_prefix_len": reconstructed_prefix_len,
                "current_chunked_req_prefix_len_source": "reconstructed_scheduler",
                "reconstructed_prefill_round": reconstructed_round,
                "aggregate_prefix_len": reconstructed_round.get("sum_prefix_len"),
                "aggregate_extend_len": reconstructed_round.get("sum_extend_len"),
                "aggregate_seq_len_after": reconstructed_round.get(
                    "sum_seq_len_after"
                ),
                "chunk_req_ids": [
                    item.get("req_id") for item in round_items if isinstance(item, dict)
                ],
                "chunk_req_prefix_lens": [
                    item.get("prefix_len") for item in round_items if isinstance(item, dict)
                ],
                "chunk_req_fresh_lens": [
                    item.get("extend_len") for item in round_items if isinstance(item, dict)
                ],
                "chunk_req_total_seq_lens": [
                    item.get("seq_len_after")
                    for item in round_items
                    if isinstance(item, dict)
                ],
                "chunk_req_prompt_lens": [
                    item.get("prompt_len") for item in round_items if isinstance(item, dict)
                ],
                "chunk_req_is_chunked": [
                    item.get("is_chunked_req") for item in round_items if isinstance(item, dict)
                ],
                "attention_inputs_prefix_len": inferred_chunk_prefix_len,
            }
        )
        return merged

    if chunk_event is None:
        merged["current_chunked_req_prefix_len"] = inferred_chunk_prefix_len
        merged["current_chunked_req_prefix_len_source"] = "attention_inputs"
        merged.setdefault("scheduler_chunked_req_prefix_len", None)
        return merged

    payload = chunk_event.get("payload")
    if not isinstance(payload, dict):
        merged["current_chunked_req_prefix_len"] = inferred_chunk_prefix_len
        merged["current_chunked_req_prefix_len_source"] = "attention_inputs"
        merged.setdefault("scheduler_chunked_req_prefix_len", None)
        return merged

    scheduler_prefix_len = payload.get("chunked_req_prefix_len", 0)
    merged.update(
        {
            "current_chunked_req_prefix_len": scheduler_prefix_len,
            "current_chunked_req_prefix_len_source": "scheduler_nvtx",
            "scheduler_chunked_req_prefix_len": scheduler_prefix_len,
            "scheduler_chunked_req_extend_input_len": payload.get(
                "chunked_req_extend_input_len"
            ),
            "scheduler_chunked_req_seq_len": payload.get("chunked_req_seq_len"),
            "scheduler_chunked_req_rid": payload.get("chunked_req_rid"),
            "scheduler_batch_size": payload.get("batch_size"),
            "scheduler_extend_num_tokens": payload.get("extend_num_tokens"),
            "scheduler_prefix_lens": payload.get("prefix_lens"),
            "scheduler_extend_lens": payload.get("extend_lens"),
            "scheduler_seq_lens": payload.get("seq_lens"),
            "scheduler_forward_ct": payload.get("forward_ct"),
            "scheduler_chunk_range_ns": [
                chunk_event.get("start_ns"),
                chunk_event.get("end_ns"),
            ],
        }
    )
    return merged


def build_mla_aligned_summary(
    sqlite_path: Path,
    output_path: Path,
    json_output_path: Path,
    csv_output_path: Path,
) -> None:
    conn = sqlite3.connect(sqlite_path)
    try:
        string_map = _load_string_map(conn)
        run_meta = _load_run_meta(sqlite_path)
        engine_kwargs = _load_engine_kwargs(sqlite_path)
        reconstructed_rounds = _reconstruct_prefill_rounds(run_meta, engine_kwargs)
        rows = conn.execute(
            """
            SELECT text, start, end, globalTid
            FROM NVTX_EVENTS
            WHERE text IS NOT NULL
              AND start IS NOT NULL
              AND end IS NOT NULL
            ORDER BY start
            """
        ).fetchall()

        all_events = [(str(text), int(start), int(end)) for text, start, end, _ in rows]
        event_tid_map = {
            (str(text), int(start), int(end)): (
                int(global_tid) if global_tid is not None else None
            )
            for text, start, end, global_tid in rows
        }
        prefill_chunk_events = _extract_prefill_chunk_events(all_events)
        attn_events: list[tuple[str, int, int, str]] = []
        for text, start, end in all_events:
            match = re.fullmatch(r"\{'Module': 'model\.model\.layers\.(\d+)\.self_attn'\}", text)
            if match:
                attn_events.append((match.group(1), start, end, text))

        lines = [
            "# MLA对齐解析摘要",
            "",
            f"源文件: `{sqlite_path.name}`",
            "",
            "## 对齐原则",
            "",
            "- `collector/sglang/collect_mla_module.py` 的 MLA module 计时边界是 `model.model.layers[test_layer].self_attn(...)`。",
            "- 因此这里把 `nsys` 中每层的 `model.model.layers.X.self_attn` NVTX range 视为与 collector 对齐的 MLA-module 边界。",
            "- 该区间内部的 `.self_attn.*` 子模块用于做 MLA 内部 breakdown。",
            "",
        ]
        json_summary: list[dict[str, object]] = []

        if not attn_events:
            lines += ["未检测到 `model.model.layers.X.self_attn` NVTX 边界。", ""]
            output_path.write_text("\n".join(lines))
            json_output_path.write_text(json.dumps([], indent=2))
            return

        prefill_count_by_layer: dict[int, int] = {}
        decode_count_by_layer: dict[int, int] = {}

        for event_order_index, (layer_id, start, end, text) in enumerate(
            attn_events, start=1
        ):
            child_rows = _find_direct_child_module_events(text, start, end, all_events)
            stage = _classify_attn_stage_from_children(child_rows)
            duration_ms = (end - start) / 1e6
            layer_int = int(layer_id)
            if stage == "prefill":
                prefill_count_by_layer[layer_int] = prefill_count_by_layer.get(layer_int, 0) + 1
                stage_instance_index = prefill_count_by_layer[layer_int]
                stage_instance_count = None
            elif stage == "decode":
                decode_count_by_layer[layer_int] = decode_count_by_layer.get(layer_int, 0) + 1
                stage_instance_index = decode_count_by_layer[layer_int]
                stage_instance_count = None
            else:
                stage_instance_index = 1
                stage_instance_count = None
            raw_shape_info = _extract_shape_info(child_rows)
            chunk_event = (
                _find_prefill_chunk_for_event(start, end, prefill_chunk_events)
                if stage == "prefill"
                else None
            )
            reconstructed_round = None
            if stage == "prefill" and stage_instance_index <= len(reconstructed_rounds):
                reconstructed_round = reconstructed_rounds[stage_instance_index - 1]
            shape_info = _merge_scheduler_chunk_info(
                raw_shape_info, chunk_event, reconstructed_round
            )
            child_entries: list[dict[str, object]] = []
            for order_index, (
                child_text,
                child_start,
                child_end,
                child_duration,
            ) in enumerate(child_rows, start=1):
                canonical_name = _canonical_child_name(child_text)
                child_global_tid = event_tid_map.get((child_text, child_start, child_end))
                trace_info = _fetch_module_gpu_trace(
                    conn,
                    string_map,
                    child_start,
                    child_end,
                    child_global_tid,
                )
                operator_info = _build_operator_info_from_trace(
                    child_text,
                    canonical_name,
                    trace_info,
                )
                if int(operator_info.get("gpu_kernel_count", 0) or 0) == 0:
                    fallback_info = _build_operator_info(
                        child_text,
                        canonical_name,
                        run_meta,
                    )
                    for key, value in fallback_info.items():
                        if operator_info.get(key) in {None, "unknown", "not_inferred"}:
                            operator_info[key] = value
                child_timing_breakdown = _compute_timing_breakdown(
                    child_start,
                    child_end,
                    trace_info.get("gpu_kernels", []),
                )
                child_entries.append(
                    {
                        "order_index": order_index,
                        "text": child_text,
                        "canonical_name": canonical_name,
                        "operator_info": operator_info,
                        "timing_breakdown": child_timing_breakdown,
                        "start_ns": child_start,
                        "end_ns": child_end,
                        "duration_ns": child_duration,
                        "duration_ms": child_duration / 1e6,
                    }
                )
            module_timing_breakdown = _build_module_timing_summary(
                start,
                end,
                child_entries,
            )
            json_summary.append(
                {
                    "event_order_index": event_order_index,
                    "layer_id": layer_int,
                    "stage": stage,
                    "stage_instance_index": stage_instance_index,
                    "collector_aligned_boundary": text,
                    "start_ns": start,
                    "end_ns": end,
                    "duration_ns": end - start,
                    "duration_ms": duration_ms,
                    "shape_info": shape_info,
                    "timing_breakdown": module_timing_breakdown,
                    "prefill_chunk_info": (
                        {
                            "start_ns": chunk_event["start_ns"],
                            "end_ns": chunk_event["end_ns"],
                            "duration_ns": chunk_event["duration_ns"],
                            "payload": chunk_event["payload"],
                        }
                        if chunk_event is not None
                        else None
                    ),
                    "children": child_entries,
                }
            )
        stage_count_by_layer: dict[tuple[int, str], int] = {}
        for item in json_summary:
            key = (int(item["layer_id"]), str(item["stage"]))
            stage_count_by_layer[key] = stage_count_by_layer.get(key, 0) + 1

        for item in json_summary:
            item["stage_instance_count"] = stage_count_by_layer[
                (int(item["layer_id"]), str(item["stage"]))
            ]
            item["collector_prefill_aligned"] = (
                item["stage"] == "prefill" and item["stage_instance_count"] == 1
            )

        clean_prefill_layers = sorted(
            {
                int(item["layer_id"])
                for item in json_summary
                if item["stage"] == "prefill" and item["collector_prefill_aligned"]
            }
        )
        split_prefill_layers = sorted(
            {
                int(item["layer_id"])
                for item in json_summary
                if item["stage"] == "prefill" and not item["collector_prefill_aligned"]
            }
        )

        lines += [
            "## 运行摘要",
            "",
            f"- `prefill` 对齐成功层: `{clean_prefill_layers}`",
            f"- `prefill` 被切分层: `{split_prefill_layers}`",
            "- 若某层 `prefill` 出现多个 `self_attn` 实例，则说明 Engine 调度把一次前向切成了多块，已不再与 collector 的单次 MLA-module 采集严格一一对应。",
            "",
        ]

        for item in json_summary:
            layer_id = int(item["layer_id"])
            stage = str(item["stage"])
            duration_ms = float(item["duration_ms"])
            collector_aligned = bool(item["collector_prefill_aligned"])
            child_rows = [
                (
                    child["text"],
                    int(child["start_ns"]),
                    int(child["duration_ns"]),
                    child["canonical_name"],
                    child.get("operator_info") or {},
                    child.get("timing_breakdown") or {},
                )
                for child in item["children"]
            ]
            lines.append(
                f"## Layer {layer_id} / {stage} / instance {item['stage_instance_index']}"
            )
            lines.append("")
            lines.append(f"- 执行序号: `{item['event_order_index']}`")
            lines.append(f"- Collector对齐边界: `{item['collector_aligned_boundary']}`")
            lines.append(f"- 整块 MLA-module 时长: `{duration_ms:.3f} ms`")
            lines.append(f"- 内部子模块数量: `{len(child_rows)}`")
            lines.append(
                f"- 该层该阶段实例数: `{item['stage_instance_count']}`"
            )
            timing = item.get("timing_breakdown")
            if isinstance(timing, dict):
                lines.append(
                    "- 模块时延拆解: "
                    f"`host_total={timing.get('host_duration_ms', 0.0):.3f} ms`, "
                    f"`module_to_last_kernel={timing.get('module_total_to_last_kernel_ms', 0.0):.3f} ms`, "
                    f"`host_to_first_kernel_gap={timing.get('host_to_first_kernel_gap_ms')}`, "
                    f"`host_end_to_last_kernel_tail={timing.get('host_end_to_last_kernel_tail_ms', 0.0):.3f} ms`, "
                    f"`gpu_makespan={timing.get('gpu_makespan_ms', 0.0):.3f} ms`, "
                    f"`gpu_kernel_sum={timing.get('gpu_kernel_time_sum_ms', 0.0):.3f} ms`, "
                    f"`gpu_kernel_count={timing.get('gpu_kernel_count', 0)}`"
                )
            if stage == "prefill":
                lines.append(f"- 是否严格对齐 collector 单块边界: `{collector_aligned}`")
                shape_info = item.get("shape_info")
                if isinstance(shape_info, dict):
                    lines.append(
                        "- Attention形状信息: "
                        f"`module={shape_info.get('module')}`, "
                        f"`token_count={shape_info.get('token_count')}`, "
                        f"`total_tokens={shape_info.get('total_tokens')}`, "
                        f"`chunked_req_prefix_len={shape_info.get('chunked_req_prefix_len', 0)}`, "
                        f"`current_chunked_req_prefix_len={shape_info.get('current_chunked_req_prefix_len', 0)}`, "
                        f"`current_chunked_req_prefix_len_source={shape_info.get('current_chunked_req_prefix_len_source')}`, "
                        f"`inputs={shape_info.get('inputs')}`"
                    )
                    if shape_info.get("current_chunked_req_prefix_len_source") == "reconstructed_scheduler":
                        round_info = shape_info.get("reconstructed_prefill_round")
                        if isinstance(round_info, dict):
                            lines.append(
                                "- 重建调度信息: "
                                f"`round={round_info.get('round_index')}`, "
                                f"`sum_extend={round_info.get('sum_extend_len')}`, "
                                f"`sum_prefix={round_info.get('sum_prefix_len')}`, "
                                f"`sum_seq_after={round_info.get('sum_seq_len_after')}`, "
                                f"`chunked_req_ids={round_info.get('chunked_req_ids')}`, "
                                f"`chunked_req_prefix_lens={round_info.get('chunked_req_prefix_lens')}`, "
                                f"`items={round_info.get('items')}`"
                            )
                    if (
                        shape_info.get("current_chunked_req_prefix_len_source")
                        == "scheduler_nvtx"
                    ):
                        lines.append(
                            "- SGLang调度chunk信息: "
                            f"`scheduler_chunked_req_prefix_len={shape_info.get('scheduler_chunked_req_prefix_len')}`, "
                            f"`scheduler_extend_num_tokens={shape_info.get('scheduler_extend_num_tokens')}`, "
                            f"`scheduler_prefix_lens={shape_info.get('scheduler_prefix_lens')}`, "
                            f"`scheduler_extend_lens={shape_info.get('scheduler_extend_lens')}`, "
                            f"`scheduler_seq_lens={shape_info.get('scheduler_seq_lens')}`"
                        )
                    else:
                        lines.append(
                            "- SGLang调度chunk信息: "
                            "`scheduler_nvtx未捕获，当前prefix由attention输入形状推断`"
                        )
            lines.append("")
            if child_rows:
                lines.append("### 子模块Breakdown（按执行先后）")
                lines.append("")
                for (
                    child_text,
                    child_start,
                    child_duration,
                    canonical_name,
                    operator_info,
                    child_timing,
                ) in child_rows:
                    lines.append(
                        f"- `{canonical_name}` -> {child_duration / 1e6:.3f} ms"
                    )
                    gpu_ms = operator_info.get("gpu_kernel_time_ms_total")
                    gpu_count = operator_info.get("gpu_kernel_count")
                    dominant_kernel = operator_info.get("dominant_kernel_name")
                    lines.append(
                        f"  纯GPU kernel时间: `{(gpu_ms or 0.0):.3f} ms`, "
                        f"kernel数: `{gpu_count}`, dominant kernel: `{dominant_kernel}`"
                    )
                    lines.append(
                        "  时延拆解: "
                        f"`host_total={child_timing.get('host_duration_ms', 0.0):.3f} ms`, "
                        f"`host_to_first_kernel_gap={child_timing.get('host_to_first_kernel_gap_ms')}`, "
                        f"`host_end_to_last_kernel_tail={child_timing.get('host_end_to_last_kernel_tail_ms', 0.0):.3f} ms`, "
                        f"`gpu_makespan={child_timing.get('gpu_makespan_ms', 0.0):.3f} ms`, "
                        f"`gpu_kernel_sum={child_timing.get('gpu_kernel_time_sum_ms', 0.0):.3f} ms`"
                    )
                    lines.append(f"  开始时间(ns): `{child_start}`")
                    lines.append(f"  原始NVTX: `{child_text}`")
                lines.append("")

        output_path.write_text("\n".join(lines))
        json_output_path.write_text(json.dumps(json_summary, indent=2))
        _write_module_timing_csv(json_summary, csv_output_path)
    finally:
        conn.close()


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    nsys_dir = run_dir / "nsys"
    nsys_dir.mkdir(parents=True, exist_ok=True)

    rep_path = find_nsys_rep(run_dir)
    sqlite_path = nsys_dir / "report.sqlite"
    export_sqlite(
        args.image, run_dir, rep_path, sqlite_path, nsys_dir / "export_sqlite.log"
    )
    export_stats(args.image, run_dir, rep_path, nsys_dir / "stats_nvtx_cuda.log")
    summarize_sqlite(sqlite_path, nsys_dir / "sqlite_summary.json")
    build_markdown_summary(sqlite_path, nsys_dir / "解析摘要.md")
    build_mla_aligned_summary(
        sqlite_path,
        nsys_dir / "MLA对齐解析.md",
        nsys_dir / "MLA对齐解析.json",
        nsys_dir / "MLA时延拆解.csv",
    )
    mla_json_path = nsys_dir / "MLA对齐解析.json"
    sqlite_summary_path = nsys_dir / "sqlite_summary.json"
    try:
        mla_rows = json.loads(mla_json_path.read_text())
    except Exception:
        mla_rows = None
    try:
        sqlite_summary = json.loads(sqlite_summary_path.read_text())
    except Exception:
        sqlite_summary = {}
    layer_marker_count = sqlite_summary.get("layer_marker_count", 0)
    if isinstance(mla_rows, list) and not mla_rows and layer_marker_count == 0:
        raise RuntimeError(
            "Nsys export succeeded but no layerwise MLA NVTX markers were found. "
            "Re-run with --layerwise-marker (enabled by default for nsys profile mode)."
        )
    print(nsys_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
