#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
BASE_SCRIPT="${ROOT_DIR}/.self/task-localbench/tmp/tmp(server)/run_sglang_deepseek_v3_dummy5_e2e.sh"
SUMMARY_PATH="${ROOT_DIR}/.self/task-localbench/tmp/tmp(server)/sglang_v059_pcg_matrix_summary.tsv"

cases=(
  "off 1 31180"
  "off 0 31181"
  "on 1 31182"
  "on 0 31183"
)

printf "pcg_mode\tskip_server_warmup\tport\tstatus\trun_dir\n" > "${SUMMARY_PATH}"

for case in "${cases[@]}"; do
  read -r pcg_mode skip_server_warmup port <<< "${case}"
  status="ok"
  run_dir=""

  if output="$(
    PCG_MODE="${pcg_mode}" \
    SKIP_SERVER_WARMUP="${skip_server_warmup}" \
    PORT="${port}" \
    READY_TIMEOUT_SEC=240 \
    POLL_INTERVAL_SEC=5 \
    SGLANG_IMAGE="booleimg.myaddr.io/lmsysorg/sglang:v0.5.9" \
    "${BASE_SCRIPT}" 2>&1
  )"; then
    run_dir="$(printf '%s\n' "${output}" | tail -n 1)"
  else
    status="fail"
    run_dir="$(printf '%s\n' "${output}" | tail -n 1)"
  fi

  printf "%s\t%s\t%s\t%s\t%s\n" "${pcg_mode}" "${skip_server_warmup}" "${port}" "${status}" "${run_dir}" >> "${SUMMARY_PATH}"
done

cat "${SUMMARY_PATH}"
