#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
BASE_DIR="${ROOT_DIR}/.self/task-localbench/tmp/tmp(server)"
RUN_DIR="${BASE_DIR}/output/${RUN_STAMP}_sglang_deepseek_v3_dummy5_e2e"
LOCAL_MODEL_DIR="${RUN_DIR}/local_model"
PROFILE_DIR="${RUN_DIR}/profile"
SERVER_LOG="${RUN_DIR}/server.log"
WAIT_LOG="${RUN_DIR}/wait_progress.log"
GPU_MONITOR_LOG="${RUN_DIR}/gpu_monitor.log"
REQUEST_JSON="${RUN_DIR}/generate_request.json"
PROFILE_JSON="${RUN_DIR}/start_profile.json"
RESPONSE_JSON="${RUN_DIR}/generate_response.json"
STOP_PROFILE_TXT="${RUN_DIR}/stop_profile_response.txt"
RUN_META="${RUN_DIR}/run_meta.txt"
START_PROFILE_HTTP="${RUN_DIR}/start_profile.http"
GENERATE_HTTP="${RUN_DIR}/generate.http"
STOP_PROFILE_HTTP="${RUN_DIR}/stop_profile.http"
FAIL_STAGE_FILE="${RUN_DIR}/failed_stage.txt"

GPU_ID="${GPU_ID:-7}"
PORT="${PORT:-31080}"
SGLANG_IMAGE="${SGLANG_IMAGE:-booleimg.myaddr.io/lmsysorg/sglang:v0.5.9}"
PCG_MODE="${PCG_MODE:-auto}"
NUM_LAYERS="${NUM_LAYERS:-5}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8}"
PROMPT_LEN="${PROMPT_LEN:-32}"
ATTN_BACKEND="${ATTN_BACKEND:-fa3}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-512}"
READY_TIMEOUT_SEC="${READY_TIMEOUT_SEC:-900}"
POLL_INTERVAL_SEC="${POLL_INTERVAL_SEC:-5}"
SKIP_SERVER_WARMUP="${SKIP_SERVER_WARMUP:-1}"
SERVER_INFO_TIMEOUT_SEC="${SERVER_INFO_TIMEOUT_SEC:-10}"
START_PROFILE_TIMEOUT_SEC="${START_PROFILE_TIMEOUT_SEC:-30}"
GENERATE_TIMEOUT_SEC="${GENERATE_TIMEOUT_SEC:-60}"
STOP_PROFILE_TIMEOUT_SEC="${STOP_PROFILE_TIMEOUT_SEC:-30}"

mkdir -p "${RUN_DIR}" "${LOCAL_MODEL_DIR}" "${PROFILE_DIR}"

python - <<'PY' "${ROOT_DIR}" "${LOCAL_MODEL_DIR}"
import json
import pathlib
import sys

root_dir = pathlib.Path(sys.argv[1])
local_model_dir = pathlib.Path(sys.argv[2])
src_cfg = root_dir / "src/aiconfigurator/model_configs/deepseek-ai--DeepSeek-V3_config.json"
cfg = json.loads(src_cfg.read_text())
cfg["model_type"] = "deepseek_v3"
cfg["architectures"] = ["DeepseekV3ForCausalLM"]
cfg.pop("auto_map", None)
(local_model_dir / "config.json").write_text(json.dumps(cfg, indent=2))
PY

python - <<'PY' "${REQUEST_JSON}" "${PROMPT_LEN}" "${MAX_NEW_TOKENS}"
import json
import sys

request_path = sys.argv[1]
prompt_len = int(sys.argv[2])
max_new_tokens = int(sys.argv[3])
payload = {
    "input_ids": list(range(1, prompt_len + 1)),
    "sampling_params": {
        "temperature": 0.0,
        "max_new_tokens": max_new_tokens,
    },
    "stream": False,
    "return_logprob": False,
}
with open(request_path, "w") as f:
    json.dump(payload, f, indent=2)
PY

cat > "${PROFILE_JSON}" <<EOF
{
  "output_dir": "/out/profile",
  "activities": ["CPU", "GPU"],
  "with_stack": false,
  "record_shapes": true,
  "profile_by_stage": false,
  "profile_prefix": "deepseek-v3-dummy5"
}
EOF

IMAGE_TAG="${SGLANG_IMAGE##*:}"
SERVER_INFO_PATH="${SERVER_INFO_PATH:-}"
EXTRA_SGLANG_ARGS=()
if [[ "${PCG_MODE}" == "on" ]]; then
  if [[ "${IMAGE_TAG}" == "v0.5.9" ]]; then
    EXTRA_SGLANG_ARGS+=(--enable-piecewise-cuda-graph)
  fi
elif [[ "${PCG_MODE}" == "off" ]]; then
  if [[ "${IMAGE_TAG}" == "v0.5.9" ]]; then
    :
  else
    EXTRA_SGLANG_ARGS+=(--disable-piecewise-cuda-graph)
  fi
fi
if [[ "${SKIP_SERVER_WARMUP}" == "1" ]]; then
  EXTRA_SGLANG_ARGS+=(--skip-server-warmup)
fi

if [[ -z "${SERVER_INFO_PATH}" ]]; then
  if [[ "${IMAGE_TAG}" == "v0.5.9" ]]; then
    SERVER_INFO_PATH="/get_server_info"
  else
    SERVER_INFO_PATH="/server_info"
  fi
fi

CONTAINER_NAME="aic_sglang_dsv3_dummy5_${RUN_STAMP}_$$"

cleanup() {
  docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

docker run -d \
  --name "${CONTAINER_NAME}" \
  --gpus "device=${GPU_ID}" \
  --ipc=host \
  --shm-size 32g \
  -p "${PORT}:30000" \
  -v "${RUN_DIR}:/out" \
  -v "${LOCAL_MODEL_DIR}:/model:ro" \
  "${SGLANG_IMAGE}" \
  bash -lc "
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX=/tmp/pycache
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0
export SGLANG_JIT_DEEPGEMM_FAST_WARMUP=1
mkdir -p /tmp/pycache /tmp/torchinductor
sglang serve \
  --model-path /model \
  --load-format dummy \
  --skip-tokenizer-init \
  --port 30000 \
  --host 0.0.0.0 \
  --mem-fraction-static 0.5 \
  --context-length ${CONTEXT_LENGTH} \
  --attention-backend ${ATTN_BACKEND} \
  --disable-cuda-graph \
  --disable-radix-cache \
  ${EXTRA_SGLANG_ARGS[*]} \
  --json-model-override-args '{\"num_hidden_layers\": ${NUM_LAYERS}}'
" >"${SERVER_LOG}" 2>&1

{
  echo "wait_start=$(date --iso-8601=seconds)"
  echo "ready_timeout_sec=${READY_TIMEOUT_SEC}"
  echo "poll_interval_sec=${POLL_INTERVAL_SEC}"
} > "${WAIT_LOG}"

{
  echo "gpu_monitor_start=$(date --iso-8601=seconds)"
  echo "fields=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw"
} > "${GPU_MONITOR_LOG}"

LAST_LOG_SNAPSHOT=""
READY=0
MAX_POLLS=$((READY_TIMEOUT_SEC / POLL_INTERVAL_SEC))

for _ in $(seq 1 "${MAX_POLLS}"); do
  if curl -m "${SERVER_INFO_TIMEOUT_SEC}" -fsS "http://127.0.0.1:${PORT}${SERVER_INFO_PATH}" >/dev/null 2>&1; then
    READY=1
    printf '[%s] server_ready\n' "$(date --iso-8601=seconds)" >> "${WAIT_LOG}"
    break
  fi

  {
    printf '\n[%s] not_ready\n' "$(date --iso-8601=seconds)"
    docker inspect "${CONTAINER_NAME}" --format 'state={{.State.Status}} running={{.State.Running}} oom={{.State.OOMKilled}} pid={{.State.Pid}}' 2>/dev/null || true
  } >> "${WAIT_LOG}"

  CURRENT_LOG_SNAPSHOT="$(docker logs "${CONTAINER_NAME}" 2>&1 || true)"
  if [[ "${CURRENT_LOG_SNAPSHOT}" != "${LAST_LOG_SNAPSHOT}" ]]; then
    {
      printf '[%s] docker_logs_update_begin\n' "$(date --iso-8601=seconds)"
      printf '%s\n' "${CURRENT_LOG_SNAPSHOT}"
      printf '[%s] docker_logs_update_end\n' "$(date --iso-8601=seconds)"
    } >> "${WAIT_LOG}"
    LAST_LOG_SNAPSHOT="${CURRENT_LOG_SNAPSHOT}"
  fi

  {
    printf '[%s] ' "$(date --iso-8601=seconds)"
    nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw --format=csv,noheader,nounits | sed -n "$((GPU_ID + 1))p"
  } >> "${GPU_MONITOR_LOG}" || true

  sleep "${POLL_INTERVAL_SEC}"
done

docker logs "${CONTAINER_NAME}" > "${SERVER_LOG}" 2>&1 || true

if [[ "${READY}" -ne 1 ]] || ! curl -m "${SERVER_INFO_TIMEOUT_SEC}" -fsS "http://127.0.0.1:${PORT}${SERVER_INFO_PATH}" > "${RUN_DIR}/server_info.json"; then
  echo "server_info" > "${FAIL_STAGE_FILE}"
  echo "server failed to become ready within timeout" >&2
  exit 1
fi

if ! curl -m "${START_PROFILE_TIMEOUT_SEC}" -fsS -D "${START_PROFILE_HTTP}" -X POST "http://127.0.0.1:${PORT}/start_profile" \
  -H 'Content-Type: application/json' \
  --data @"${PROFILE_JSON}" \
  > "${RUN_DIR}/start_profile_response.txt"; then
  echo "start_profile" > "${FAIL_STAGE_FILE}"
  exit 1
fi

if ! curl -m "${GENERATE_TIMEOUT_SEC}" -fsS -D "${GENERATE_HTTP}" -X POST "http://127.0.0.1:${PORT}/generate" \
  -H 'Content-Type: application/json' \
  --data @"${REQUEST_JSON}" \
  > "${RESPONSE_JSON}"; then
  echo "generate" > "${FAIL_STAGE_FILE}"
  exit 1
fi

if ! curl -m "${STOP_PROFILE_TIMEOUT_SEC}" -fsS -D "${STOP_PROFILE_HTTP}" -X POST "http://127.0.0.1:${PORT}/stop_profile" \
  > "${STOP_PROFILE_TXT}"; then
  echo "stop_profile" > "${FAIL_STAGE_FILE}"
  exit 1
fi

sleep 3

docker logs "${CONTAINER_NAME}" > "${SERVER_LOG}" 2>&1 || true

cat > "${RUN_META}" <<EOF
gpu_id=${GPU_ID}
port=${PORT}
num_layers=${NUM_LAYERS}
prompt_len=${PROMPT_LEN}
max_new_tokens=${MAX_NEW_TOKENS}
attention_backend=${ATTN_BACKEND}
context_length=${CONTEXT_LENGTH}
ready_timeout_sec=${READY_TIMEOUT_SEC}
poll_interval_sec=${POLL_INTERVAL_SEC}
sglang_image=${SGLANG_IMAGE}
pcg_mode=${PCG_MODE}
skip_server_warmup=${SKIP_SERVER_WARMUP}
server_info_path=${SERVER_INFO_PATH}
server_info_timeout_sec=${SERVER_INFO_TIMEOUT_SEC}
start_profile_timeout_sec=${START_PROFILE_TIMEOUT_SEC}
generate_timeout_sec=${GENERATE_TIMEOUT_SEC}
stop_profile_timeout_sec=${STOP_PROFILE_TIMEOUT_SEC}
container_name=${CONTAINER_NAME}
EOF

echo "${RUN_DIR}"
