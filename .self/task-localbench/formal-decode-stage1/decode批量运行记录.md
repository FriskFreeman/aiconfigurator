# Decode批量运行记录

## 说明

- 本文档记录 decode-only 场景的实机运行过程。
- 每个请求先在 session 中写入等长 KV prefix；正式 profile 只包住 1-token continuation decode。
- 原始 run 目录归档到 `bench_data/.../deepseek_v3_mla/raw_runs/`。
- Decode 核心 `nsys` 结果归档到 `bench_data/.../deepseek_v3_mla/decode_csv/`。

## 运行日志

### 新批次
- case_group: `smoke_decode_only`
- start_index: `0`
- max_cases: `1`
- skip_existing_ok: `False`
- bench_root: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/bench_data/h100_sxm/sglang/v0.5.9/deepseek_v3_mla`
- [1/1] 开始 `decode_smoke_b4p512_cg_on`
  - cmd: `python .self/task-localbench/formal-prefill-stage1/run_formal_prefill_stage1.py --gpu-id 7 --num-layers 3 --fresh-lens-csv 1,1,1,1 --prefix-lens-csv 512,512,512,512 --context-length 528 --max-new-tokens 1 --warmup-runs 1 --profile-mode nsys --cuda-graph-mode on --pcg-mode off --tp-size 1 --tag decode_smoke_b4p512_cg_on --run-root /home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1 --mem-fraction-static 0.5 --layerwise-marker`
### 新批次
- case_group: `smoke_decode_only`
- start_index: `0`
- max_cases: `1`
- skip_existing_ok: `False`
- bench_root: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/bench_data/h100_sxm/sglang/v0.5.9/deepseek_v3_mla`
- [1/1] 开始 `decode_smoke_b4p512_cg_on`
  - cmd: `python .self/task-localbench/formal-prefill-stage1/run_formal_prefill_stage1.py --gpu-id 7 --num-layers 3 --fresh-lens-csv 1,1,1,1 --prefix-lens-csv 512,512,512,512 --context-length 528 --max-new-tokens 1 --warmup-runs 1 --profile-mode nsys --cuda-graph-mode on --pcg-mode off --tp-size 1 --tag decode_smoke_b4p512_cg_on --run-root /home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1 --mem-fraction-static 0.5 --layerwise-marker`
  - 成功: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1/20260625_110627_prefill_stage1_decode_smoke_b4p512_cg_on_b4_fvar1-1-1-1_pvar512-512-512-512_layers3_backend_auto_cg_on_pcg_off_tp1_marker_on_profile_nsys`
  - CSV: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/bench_data/h100_sxm/sglang/v0.5.9/deepseek_v3_mla/decode_csv/20260625_110627_prefill_stage1_decode_smoke_b4p512_cg_on_b4_fvar1-1-1-1_pvar512-512-512-512_layers3_backend_auto_cg_on_pcg_off_tp1_marker_on_profile_nsys__MLA时延拆解.csv`

## 本批次总结
- total_cases: `1`
- failures: `0`
### 新批次
- case_group: `smoke_decode_only`
- start_index: `0`
- max_cases: `1`
- skip_existing_ok: `False`
- bench_root: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/bench_data/h100_sxm/sglang/v0.5.9/deepseek_v3_mla`
- [1/1] 开始 `decode_smoke_b4p512_cg_on`
  - cmd: `python .self/task-localbench/formal-prefill-stage1/run_formal_prefill_stage1.py --gpu-id 7 --num-layers 3 --fresh-lens-csv 1,1,1,1 --prefix-lens-csv 512,512,512,512 --context-length 528 --max-new-tokens 1 --warmup-runs 1 --profile-mode nsys --cuda-graph-mode on --pcg-mode off --tp-size 1 --tag decode_smoke_b4p512_cg_on --run-root /home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1 --mem-fraction-static 0.5 --layerwise-marker`
### 新批次
- case_group: `smoke_decode_only`
- start_index: `1`
- max_cases: `1`
- skip_existing_ok: `False`
- bench_root: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/bench_data/h100_sxm/sglang/v0.5.9/deepseek_v3_mla`
- [1/1] 开始 `decode_smoke_b8p2048_cg_on`
  - cmd: `python .self/task-localbench/formal-prefill-stage1/run_formal_prefill_stage1.py --gpu-id 7 --num-layers 3 --fresh-lens-csv 1,1,1,1,1,1,1,1 --prefix-lens-csv 2048,2048,2048,2048,2048,2048,2048,2048 --context-length 2064 --max-new-tokens 1 --warmup-runs 1 --profile-mode nsys --cuda-graph-mode on --pcg-mode off --tp-size 1 --tag decode_smoke_b8p2048_cg_on --run-root /home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1 --mem-fraction-static 0.5 --layerwise-marker`
  - 失败: returncode=143
  - stdout: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1/20260625_111621_prefill_stage1_decode_smoke_b8p2048_cg_on_b8_fvar1-1-1-1-1-1-1-1_pvar2048-2048-2048-2048-2048-2048-2048-2048_layers3_backend_auto_cg_on_pcg_off_tp1_marker_on_profile_nsys`
  - stderr: ``

## 本批次总结
- total_cases: `1`
- failures: `1`
### 新批次
- case_group: `smoke_decode_only`
- start_index: `1`
- max_cases: `1`
- skip_existing_ok: `False`
- bench_root: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/bench_data/h100_sxm/sglang/v0.5.9/deepseek_v3_mla`
- [1/1] 开始 `decode_smoke_b8p2048_cg_on`
  - cmd: `python .self/task-localbench/formal-prefill-stage1/run_formal_prefill_stage1.py --gpu-id 5 --num-layers 3 --fresh-lens-csv 1,1,1,1,1,1,1,1 --prefix-lens-csv 2048,2048,2048,2048,2048,2048,2048,2048 --context-length 2064 --max-new-tokens 1 --warmup-runs 1 --profile-mode nsys --cuda-graph-mode on --pcg-mode off --tp-size 1 --tag decode_smoke_b8p2048_cg_on --run-root /home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1 --mem-fraction-static 0.5 --layerwise-marker`
  - 成功: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1/20260625_111840_decode_stage1_decode_smoke_b8p2048_cg_on_b8_fvar1-1-1-1-1-1-1-1_pvar2048-2048-2048-2048-2048-2048-2048-2048_layers3_backend_auto_cg_on_pcg_off_tp1_marker_on_profile_nsys`
  - 原始runner输出目录: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1/20260625_111840_prefill_stage1_decode_smoke_b8p2048_cg_on_b8_fvar1-1-1-1-1-1-1-1_pvar2048-2048-2048-2048-2048-2048-2048-2048_layers3_backend_auto_cg_on_pcg_off_tp1_marker_on_profile_nsys`
  - CSV: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/bench_data/h100_sxm/sglang/v0.5.9/deepseek_v3_mla/decode_csv/20260625_111840_decode_stage1_decode_smoke_b8p2048_cg_on_b8_fvar1-1-1-1-1-1-1-1_pvar2048-2048-2048-2048-2048-2048-2048-2048_layers3_backend_auto_cg_on_pcg_off_tp1_marker_on_profile_nsys__MLA时延拆解.csv`

## 本批次总结
- total_cases: `1`
- failures: `0`
### 新批次
- case_group: `smoke_decode_only`
- start_index: `0`
- max_cases: `1`
- skip_existing_ok: `False`
- bench_root: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/bench_data/h100_sxm/sglang/v0.5.9/deepseek_v3_mla`
- [1/1] 开始 `decode_smoke_b4p512_cg_on`
  - cmd: `python .self/task-localbench/formal-prefill-stage1/run_formal_prefill_stage1.py --gpu-id 5 --num-layers 3 --fresh-lens-csv 1,1,1,1 --prefix-lens-csv 512,512,512,512 --context-length 528 --max-new-tokens 1 --warmup-runs 1 --profile-mode nsys --cuda-graph-mode on --pcg-mode off --tp-size 1 --tag decode_smoke_b4p512_cg_on --run-root /home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1 --mem-fraction-static 0.5 --decode-empty-input-continuation --layerwise-marker`
  - 成功: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1/20260625_112756_decode_stage1_decode_smoke_b4p512_cg_on_b4_fvar1-1-1-1_pvar512-512-512-512_layers3_backend_auto_cg_on_pcg_off_tp1_marker_on_profile_nsys`
  - 原始runner输出目录: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1/20260625_112756_prefill_stage1_decode_smoke_b4p512_cg_on_b4_fvar1-1-1-1_pvar512-512-512-512_layers3_backend_auto_cg_on_pcg_off_tp1_marker_on_profile_nsys`
  - 校验: `validated 3 decode rows`
  - CSV: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/bench_data/h100_sxm/sglang/v0.5.9/deepseek_v3_mla/decode_csv/20260625_112756_decode_stage1_decode_smoke_b4p512_cg_on_b4_fvar1-1-1-1_pvar512-512-512-512_layers3_backend_auto_cg_on_pcg_off_tp1_marker_on_profile_nsys__MLA时延拆解.csv`

## 本批次总结
- total_cases: `1`
- failures: `0`
### 新批次
- case_group: `smoke_decode_only`
- start_index: `1`
- max_cases: `1`
- skip_existing_ok: `False`
- bench_root: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/bench_data/h100_sxm/sglang/v0.5.9/deepseek_v3_mla`
- [1/1] 开始 `decode_smoke_b8p2048_cg_on`
  - cmd: `python .self/task-localbench/formal-prefill-stage1/run_formal_prefill_stage1.py --gpu-id 5 --num-layers 3 --fresh-lens-csv 1,1,1,1,1,1,1,1 --prefix-lens-csv 2048,2048,2048,2048,2048,2048,2048,2048 --context-length 2064 --max-new-tokens 1 --warmup-runs 1 --profile-mode nsys --cuda-graph-mode on --pcg-mode off --tp-size 1 --tag decode_smoke_b8p2048_cg_on --run-root /home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1 --mem-fraction-static 0.5 --decode-empty-input-continuation --layerwise-marker`
  - Decode校验失败: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1/20260625_113314_decode_stage1_decode_smoke_b8p2048_cg_on_b8_fvar1-1-1-1-1-1-1-1_pvar2048-2048-2048-2048-2048-2048-2048-2048_layers3_backend_auto_cg_on_pcg_off_tp1_marker_on_profile_nsys`
  - 原始runner输出目录: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1/20260625_113314_prefill_stage1_decode_smoke_b8p2048_cg_on_b8_fvar1-1-1-1-1-1-1-1_pvar2048-2048-2048-2048-2048-2048-2048-2048_layers3_backend_auto_cg_on_pcg_off_tp1_marker_on_profile_nsys`
  - reason: `expected only stage=decode, got stages=['prefill']`

## 本批次总结
- total_cases: `1`
- failures: `1`
### 新批次
- case_group: `smoke_decode_only`
- start_index: `1`
- max_cases: `1`
- skip_existing_ok: `False`
- bench_root: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/bench_data/h100_sxm/sglang/v0.5.9/deepseek_v3_mla`
- [1/1] 开始 `decode_smoke_b8p2048_cg_on`
  - cmd: `python .self/task-localbench/formal-prefill-stage1/run_formal_prefill_stage1.py --gpu-id 5 --num-layers 3 --batch-size 8 --fresh-len 1 --prefix-len 2048 --context-length 2064 --max-new-tokens 1 --warmup-runs 1 --profile-mode nsys --cuda-graph-mode on --pcg-mode off --tp-size 1 --tag decode_smoke_b8p2048_cg_on --run-root /home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1 --mem-fraction-static 0.5 --chunked-prefix-cache-threshold 16393 --decode-empty-input-continuation --layerwise-marker`
### 新批次
- case_group: `smoke_decode_only`
- start_index: `1`
- max_cases: `1`
- skip_existing_ok: `False`
- bench_root: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/bench_data/h100_sxm/sglang/v0.5.9/deepseek_v3_mla`
- [1/1] 开始 `decode_smoke_b8p2048_cg_on`
  - cmd: `python .self/task-localbench/formal-prefill-stage1/run_formal_prefill_stage1.py --gpu-id 5 --num-layers 3 --batch-size 8 --fresh-len 1 --prefix-len 2048 --context-length 2064 --max-new-tokens 1 --warmup-runs 1 --profile-mode nsys --cuda-graph-mode on --pcg-mode off --tp-size 1 --tag decode_smoke_b8p2048_cg_on --run-root /home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1 --mem-fraction-static 0.5 --chunked-prefix-cache-threshold 16393 --decode-empty-input-continuation --layerwise-marker`
  - 成功: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1/20260625_113858_decode_stage1_decode_smoke_b8p2048_cg_on_b8_f1_p2048_layers3_backend_auto_cg_on_pcg_off_tp1_marker_on_profile_nsys`
  - 原始runner输出目录: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1/20260625_113858_prefill_stage1_decode_smoke_b8p2048_cg_on_b8_f1_p2048_layers3_backend_auto_cg_on_pcg_off_tp1_marker_on_profile_nsys`
  - 校验: `validated 3 decode rows`
  - CSV: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/bench_data/h100_sxm/sglang/v0.5.9/deepseek_v3_mla/decode_csv/20260625_113858_decode_stage1_decode_smoke_b8p2048_cg_on_b8_f1_p2048_layers3_backend_auto_cg_on_pcg_off_tp1_marker_on_profile_nsys__MLA时延拆解.csv`

## 本批次总结
- total_cases: `1`
- failures: `0`
### 新批次
- case_group: `decode_only`
- start_index: `0`
- max_cases: `None`
- skip_existing_ok: `True`
- bench_root: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/bench_data/h100_sxm/sglang/v0.5.9/deepseek_v3_mla`
- [1/9] 开始 `decode_b1p512_cg_on`
  - cmd: `python .self/task-localbench/formal-prefill-stage1/run_formal_prefill_stage1.py --gpu-id 5 --num-layers 5 --batch-size 1 --fresh-len 1 --prefix-len 512 --context-length 528 --max-new-tokens 1 --warmup-runs 1 --profile-mode nsys --cuda-graph-mode on --pcg-mode off --tp-size 1 --tag decode_b1p512_cg_on --run-root /home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1 --mem-fraction-static 0.5 --chunked-prefix-cache-threshold 8192 --decode-empty-input-continuation --layerwise-marker`
  - 成功: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1/20260625_114359_decode_stage1_decode_b1p512_cg_on_b1_f1_p512_layers5_backend_auto_cg_on_pcg_off_tp1_marker_on_profile_nsys`
  - 原始runner输出目录: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1/20260625_114359_prefill_stage1_decode_b1p512_cg_on_b1_f1_p512_layers5_backend_auto_cg_on_pcg_off_tp1_marker_on_profile_nsys`
  - 校验: `validated 5 decode rows`
  - CSV: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/bench_data/h100_sxm/sglang/v0.5.9/deepseek_v3_mla/decode_csv/20260625_114359_decode_stage1_decode_b1p512_cg_on_b1_f1_p512_layers5_backend_auto_cg_on_pcg_off_tp1_marker_on_profile_nsys__MLA时延拆解.csv`
- [2/9] 开始 `decode_b4p512_cg_on`
  - cmd: `python .self/task-localbench/formal-prefill-stage1/run_formal_prefill_stage1.py --gpu-id 5 --num-layers 5 --batch-size 4 --fresh-len 1 --prefix-len 512 --context-length 528 --max-new-tokens 1 --warmup-runs 1 --profile-mode nsys --cuda-graph-mode on --pcg-mode off --tp-size 1 --tag decode_b4p512_cg_on --run-root /home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1 --mem-fraction-static 0.5 --chunked-prefix-cache-threshold 8192 --decode-empty-input-continuation --layerwise-marker`
  - 成功: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1/20260625_114814_decode_stage1_decode_b4p512_cg_on_b4_f1_p512_layers5_backend_auto_cg_on_pcg_off_tp1_marker_on_profile_nsys`
  - 原始runner输出目录: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1/20260625_114814_prefill_stage1_decode_b4p512_cg_on_b4_f1_p512_layers5_backend_auto_cg_on_pcg_off_tp1_marker_on_profile_nsys`
  - 校验: `validated 5 decode rows`
  - CSV: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/bench_data/h100_sxm/sglang/v0.5.9/deepseek_v3_mla/decode_csv/20260625_114814_decode_stage1_decode_b4p512_cg_on_b4_f1_p512_layers5_backend_auto_cg_on_pcg_off_tp1_marker_on_profile_nsys__MLA时延拆解.csv`
- [3/9] 开始 `decode_b32p512_cg_on`
  - cmd: `python .self/task-localbench/formal-prefill-stage1/run_formal_prefill_stage1.py --gpu-id 5 --num-layers 5 --batch-size 32 --fresh-len 1 --prefix-len 512 --context-length 528 --max-new-tokens 1 --warmup-runs 1 --profile-mode nsys --cuda-graph-mode on --pcg-mode off --tp-size 1 --tag decode_b32p512_cg_on --run-root /home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1 --mem-fraction-static 0.5 --chunked-prefix-cache-threshold 16417 --decode-empty-input-continuation --layerwise-marker`
  - 成功: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1/20260625_115226_decode_stage1_decode_b32p512_cg_on_b32_f1_p512_layers5_backend_auto_cg_on_pcg_off_tp1_marker_on_profile_nsys`
  - 原始runner输出目录: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1/20260625_115226_prefill_stage1_decode_b32p512_cg_on_b32_f1_p512_layers5_backend_auto_cg_on_pcg_off_tp1_marker_on_profile_nsys`
  - 校验: `validated 5 decode rows`
  - CSV: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/bench_data/h100_sxm/sglang/v0.5.9/deepseek_v3_mla/decode_csv/20260625_115226_decode_stage1_decode_b32p512_cg_on_b32_f1_p512_layers5_backend_auto_cg_on_pcg_off_tp1_marker_on_profile_nsys__MLA时延拆解.csv`
- [4/9] 开始 `decode_b4p2048_cg_on`
  - cmd: `python .self/task-localbench/formal-prefill-stage1/run_formal_prefill_stage1.py --gpu-id 5 --num-layers 5 --batch-size 4 --fresh-len 1 --prefix-len 2048 --context-length 2064 --max-new-tokens 1 --warmup-runs 1 --profile-mode nsys --cuda-graph-mode on --pcg-mode off --tp-size 1 --tag decode_b4p2048_cg_on --run-root /home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1 --mem-fraction-static 0.5 --chunked-prefix-cache-threshold 8197 --decode-empty-input-continuation --layerwise-marker`
  - 成功: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1/20260625_115639_decode_stage1_decode_b4p2048_cg_on_b4_f1_p2048_layers5_backend_auto_cg_on_pcg_off_tp1_marker_on_profile_nsys`
  - 原始runner输出目录: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1/20260625_115639_prefill_stage1_decode_b4p2048_cg_on_b4_f1_p2048_layers5_backend_auto_cg_on_pcg_off_tp1_marker_on_profile_nsys`
  - 校验: `validated 5 decode rows`
  - CSV: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/bench_data/h100_sxm/sglang/v0.5.9/deepseek_v3_mla/decode_csv/20260625_115639_decode_stage1_decode_b4p2048_cg_on_b4_f1_p2048_layers5_backend_auto_cg_on_pcg_off_tp1_marker_on_profile_nsys__MLA时延拆解.csv`
- [5/9] 开始 `decode_b16p2048_cg_on`
  - cmd: `python .self/task-localbench/formal-prefill-stage1/run_formal_prefill_stage1.py --gpu-id 5 --num-layers 5 --batch-size 16 --fresh-len 1 --prefix-len 2048 --context-length 2064 --max-new-tokens 1 --warmup-runs 1 --profile-mode nsys --cuda-graph-mode on --pcg-mode off --tp-size 1 --tag decode_b16p2048_cg_on --run-root /home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1 --mem-fraction-static 0.5 --chunked-prefix-cache-threshold 32785 --decode-empty-input-continuation --layerwise-marker`
  - 成功: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1/20260625_120053_decode_stage1_decode_b16p2048_cg_on_b16_f1_p2048_layers5_backend_auto_cg_on_pcg_off_tp1_marker_on_profile_nsys`
  - 原始runner输出目录: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1/20260625_120053_prefill_stage1_decode_b16p2048_cg_on_b16_f1_p2048_layers5_backend_auto_cg_on_pcg_off_tp1_marker_on_profile_nsys`
  - 校验: `validated 5 decode rows`
  - CSV: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/bench_data/h100_sxm/sglang/v0.5.9/deepseek_v3_mla/decode_csv/20260625_120053_decode_stage1_decode_b16p2048_cg_on_b16_f1_p2048_layers5_backend_auto_cg_on_pcg_off_tp1_marker_on_profile_nsys__MLA时延拆解.csv`
- [6/9] 开始 `decode_b4p4096_cg_on`
  - cmd: `python .self/task-localbench/formal-prefill-stage1/run_formal_prefill_stage1.py --gpu-id 5 --num-layers 5 --batch-size 4 --fresh-len 1 --prefix-len 4096 --context-length 4112 --max-new-tokens 1 --warmup-runs 1 --profile-mode nsys --cuda-graph-mode on --pcg-mode off --tp-size 1 --tag decode_b4p4096_cg_on --run-root /home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1 --mem-fraction-static 0.5 --chunked-prefix-cache-threshold 16389 --decode-empty-input-continuation --layerwise-marker`
  - 成功: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1/20260625_120502_decode_stage1_decode_b4p4096_cg_on_b4_f1_p4096_layers5_backend_auto_cg_on_pcg_off_tp1_marker_on_profile_nsys`
  - 原始runner输出目录: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1/20260625_120502_prefill_stage1_decode_b4p4096_cg_on_b4_f1_p4096_layers5_backend_auto_cg_on_pcg_off_tp1_marker_on_profile_nsys`
  - 校验: `validated 5 decode rows`
  - CSV: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/bench_data/h100_sxm/sglang/v0.5.9/deepseek_v3_mla/decode_csv/20260625_120502_decode_stage1_decode_b4p4096_cg_on_b4_f1_p4096_layers5_backend_auto_cg_on_pcg_off_tp1_marker_on_profile_nsys__MLA时延拆解.csv`
- [7/9] 开始 `decode_b8p4096_cg_on`
  - cmd: `python .self/task-localbench/formal-prefill-stage1/run_formal_prefill_stage1.py --gpu-id 5 --num-layers 5 --batch-size 8 --fresh-len 1 --prefix-len 4096 --context-length 4112 --max-new-tokens 1 --warmup-runs 1 --profile-mode nsys --cuda-graph-mode on --pcg-mode off --tp-size 1 --tag decode_b8p4096_cg_on --run-root /home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1 --mem-fraction-static 0.5 --chunked-prefix-cache-threshold 32777 --decode-empty-input-continuation --layerwise-marker`
  - 成功: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1/20260625_120915_decode_stage1_decode_b8p4096_cg_on_b8_f1_p4096_layers5_backend_auto_cg_on_pcg_off_tp1_marker_on_profile_nsys`
  - 原始runner输出目录: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1/20260625_120915_prefill_stage1_decode_b8p4096_cg_on_b8_f1_p4096_layers5_backend_auto_cg_on_pcg_off_tp1_marker_on_profile_nsys`
  - 校验: `validated 5 decode rows`
  - CSV: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/bench_data/h100_sxm/sglang/v0.5.9/deepseek_v3_mla/decode_csv/20260625_120915_decode_stage1_decode_b8p4096_cg_on_b8_f1_p4096_layers5_backend_auto_cg_on_pcg_off_tp1_marker_on_profile_nsys__MLA时延拆解.csv`
- [8/9] 开始 `decode_b4p8192_cg_on`
  - cmd: `python .self/task-localbench/formal-prefill-stage1/run_formal_prefill_stage1.py --gpu-id 5 --num-layers 5 --batch-size 4 --fresh-len 1 --prefix-len 8192 --context-length 8208 --max-new-tokens 1 --warmup-runs 1 --profile-mode nsys --cuda-graph-mode on --pcg-mode off --tp-size 1 --tag decode_b4p8192_cg_on --run-root /home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1 --mem-fraction-static 0.5 --chunked-prefix-cache-threshold 32773 --decode-empty-input-continuation --layerwise-marker`
  - 成功: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1/20260625_121333_decode_stage1_decode_b4p8192_cg_on_b4_f1_p8192_layers5_backend_auto_cg_on_pcg_off_tp1_marker_on_profile_nsys`
  - 原始runner输出目录: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1/20260625_121333_prefill_stage1_decode_b4p8192_cg_on_b4_f1_p8192_layers5_backend_auto_cg_on_pcg_off_tp1_marker_on_profile_nsys`
  - 校验: `validated 5 decode rows`
  - CSV: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/bench_data/h100_sxm/sglang/v0.5.9/deepseek_v3_mla/decode_csv/20260625_121333_decode_stage1_decode_b4p8192_cg_on_b4_f1_p8192_layers5_backend_auto_cg_on_pcg_off_tp1_marker_on_profile_nsys__MLA时延拆解.csv`
- [9/9] 开始 `decode_b2p16384_cg_on`
  - cmd: `python .self/task-localbench/formal-prefill-stage1/run_formal_prefill_stage1.py --gpu-id 5 --num-layers 5 --batch-size 2 --fresh-len 1 --prefix-len 16384 --context-length 16400 --max-new-tokens 1 --warmup-runs 1 --profile-mode nsys --cuda-graph-mode on --pcg-mode off --tp-size 1 --tag decode_b2p16384_cg_on --run-root /home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1 --mem-fraction-static 0.5 --chunked-prefix-cache-threshold 32771 --decode-empty-input-continuation --layerwise-marker`
  - 成功: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1/20260625_121809_decode_stage1_decode_b2p16384_cg_on_b2_f1_p16384_layers5_backend_auto_cg_on_pcg_off_tp1_marker_on_profile_nsys`
  - 原始runner输出目录: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1/20260625_121809_prefill_stage1_decode_b2p16384_cg_on_b2_f1_p16384_layers5_backend_auto_cg_on_pcg_off_tp1_marker_on_profile_nsys`
  - 校验: `validated 5 decode rows`
  - CSV: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/bench_data/h100_sxm/sglang/v0.5.9/deepseek_v3_mla/decode_csv/20260625_121809_decode_stage1_decode_b2p16384_cg_on_b2_f1_p16384_layers5_backend_auto_cg_on_pcg_off_tp1_marker_on_profile_nsys__MLA时延拆解.csv`

## 本批次总结
- total_cases: `9`
- failures: `0`
### 新批次
- case_group: `smoke_decode_only`
- start_index: `0`
- max_cases: `1`
- skip_existing_ok: `False`
- bench_root: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/bench_data/h100_sxm/sglang/v0.5.9/deepseek_v3_mla`
- [1/1] 开始 `decode_smoke_b4p512_cg_on`
  - cmd: `python .self/task-localbench/formal-prefill-stage1/run_formal_prefill_stage1.py --gpu-id 5 --num-layers 3 --batch-size 4 --fresh-len 1 --prefix-len 512 --context-length 528 --max-new-tokens 2 --warmup-runs 1 --profile-mode nsys --cuda-graph-mode on --pcg-mode off --tp-size 1 --tag decode_smoke_b4p512_cg_on --run-root /home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1 --mem-fraction-static 0.5 --chunked-prefix-cache-threshold 8192 --decode-empty-input-continuation --profile-by-stage --profile-num-steps 1 --profile-decode-only-stage --layerwise-marker`
  - 失败: returncode=1
  - stdout: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1/20260625_134045_prefill_stage1_decode_smoke_b4p512_cg_on_b4_f1_p512_layers3_backend_auto_cg_on_pcg_off_tp1_marker_on_profile_nsys`
  - stderr: ``

## 本批次总结
- total_cases: `1`
- failures: `1`
### 新批次
- case_group: `smoke_decode_only`
- start_index: `0`
- max_cases: `1`
- skip_existing_ok: `False`
- bench_root: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/bench_data/h100_sxm/sglang/v0.5.9/deepseek_v3_mla`
- [1/1] 开始 `decode_smoke_b4p512_cg_on`
  - cmd: `python .self/task-localbench/formal-prefill-stage1/run_formal_prefill_stage1.py --gpu-id 5 --num-layers 3 --batch-size 4 --fresh-len 1 --prefix-len 512 --context-length 528 --max-new-tokens 3 --warmup-runs 1 --profile-mode nsys --cuda-graph-mode on --pcg-mode off --tp-size 1 --tag decode_smoke_b4p512_cg_on --run-root /home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1 --mem-fraction-static 0.5 --chunked-prefix-cache-threshold 8192 --decode-empty-input-continuation --stream-profile-after-first-token --layerwise-marker`
### 新批次
- case_group: `smoke_decode_only`
- start_index: `0`
- max_cases: `1`
- skip_existing_ok: `False`
- bench_root: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/bench_data/h100_sxm/sglang/v0.5.9/deepseek_v3_mla`
- [1/1] 开始 `decode_smoke_b4p512_cg_on`
  - cmd: `python .self/task-localbench/formal-prefill-stage1/run_formal_prefill_stage1.py --gpu-id 5 --num-layers 3 --batch-size 4 --fresh-len 1 --prefix-len 512 --context-length 528 --max-new-tokens 3 --warmup-runs 1 --profile-mode nsys --cuda-graph-mode on --pcg-mode off --tp-size 1 --tag decode_smoke_b4p512_cg_on --run-root /home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1 --mem-fraction-static 0.5 --chunked-prefix-cache-threshold 8192 --decode-empty-input-continuation --stream-profile-after-first-token --layerwise-marker`
  - Decode校验失败: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1/20260625_134834_decode_stage1_decode_smoke_b4p512_cg_on_b4_f1_p512_layers3_backend_auto_cg_on_pcg_off_tp1_marker_on_profile_nsys`
  - 原始runner输出目录: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1/20260625_134834_prefill_stage1_decode_smoke_b4p512_cg_on_b4_f1_p512_layers3_backend_auto_cg_on_pcg_off_tp1_marker_on_profile_nsys`
  - reason: `missing decode CSV: /home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1/20260625_134834_decode_stage1_decode_smoke_b4p512_cg_on_b4_f1_p512_layers3_backend_auto_cg_on_pcg_off_tp1_marker_on_profile_nsys/nsys/MLA时延拆解.csv`

## 本批次总结
- total_cases: `1`
- failures: `1`
### 新批次
- case_group: `smoke_decode_only`
- start_index: `0`
- max_cases: `1`
- skip_existing_ok: `False`
- bench_root: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/bench_data/h100_sxm/sglang/v0.5.9/deepseek_v3_mla`
- [1/1] 开始 `decode_smoke_b4p512_cg_on`
  - cmd: `python .self/task-localbench/formal-prefill-stage1/run_formal_prefill_stage1.py --gpu-id 5 --num-layers 3 --batch-size 4 --fresh-len 1 --prefix-len 512 --context-length 528 --max-new-tokens 3 --warmup-runs 1 --profile-mode nsys --cuda-graph-mode on --pcg-mode off --tp-size 1 --tag decode_smoke_b4p512_cg_on --run-root /home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1 --mem-fraction-static 0.5 --chunked-prefix-cache-threshold 8192 --decode-empty-input-continuation --stream-profile-after-first-token --layerwise-marker`
  - Decode校验失败: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1/20260625_135454_decode_stage1_decode_smoke_b4p512_cg_on_b4_f1_p512_layers3_backend_auto_cg_on_pcg_off_tp1_marker_on_profile_nsys`
  - 原始runner输出目录: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1/20260625_135454_prefill_stage1_decode_smoke_b4p512_cg_on_b4_f1_p512_layers3_backend_auto_cg_on_pcg_off_tp1_marker_on_profile_nsys`
  - reason: `missing decode CSV: /home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1/20260625_135454_decode_stage1_decode_smoke_b4p512_cg_on_b4_f1_p512_layers3_backend_auto_cg_on_pcg_off_tp1_marker_on_profile_nsys/nsys/MLA时延拆解.csv`

## 本批次总结
- total_cases: `1`
- failures: `1`
### 新批次
- case_group: `smoke_decode_only`
- start_index: `0`
- max_cases: `1`
- skip_existing_ok: `False`
- bench_root: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/bench_data/h100_sxm/sglang/v0.5.9/deepseek_v3_mla`
- [1/1] 开始 `decode_smoke_b4p512_cg_on`
  - cmd: `python .self/task-localbench/formal-prefill-stage1/run_formal_prefill_stage1.py --gpu-id 5 --num-layers 3 --batch-size 4 --fresh-len 1 --prefix-len 512 --context-length 528 --max-new-tokens 2 --warmup-runs 1 --profile-mode nsys --cuda-graph-mode on --pcg-mode off --tp-size 1 --tag decode_smoke_b4p512_cg_on --run-root /home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1 --mem-fraction-static 0.5 --chunked-prefix-cache-threshold 8192 --decode-empty-input-continuation --profile-by-stage --profile-num-steps 1 --profile-decode-only-stage --layerwise-marker`
### 新批次
- case_group: `smoke_decode_only`
- start_index: `0`
- max_cases: `1`
- skip_existing_ok: `False`
- bench_root: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/bench_data/h100_sxm/sglang/v0.5.9/deepseek_v3_mla`
- [1/1] 开始 `decode_smoke_b4p512_cg_on`
  - cmd: `python .self/task-localbench/formal-prefill-stage1/run_formal_prefill_stage1.py --gpu-id 5 --num-layers 3 --batch-size 4 --fresh-len 1 --prefix-len 512 --context-length 528 --max-new-tokens 2 --warmup-runs 1 --profile-mode nsys --cuda-graph-mode on --pcg-mode off --tp-size 1 --tag decode_smoke_b4p512_cg_on --run-root /home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1 --mem-fraction-static 0.5 --chunked-prefix-cache-threshold 8192 --decode-empty-input-continuation --profile-by-stage --profile-num-steps 1 --profile-decode-only-stage --layerwise-marker`
  - Decode校验失败: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1/20260625_140116_decode_stage1_decode_smoke_b4p512_cg_on_b4_f1_p512_layers3_backend_auto_cg_on_pcg_off_tp1_marker_on_profile_nsys`
  - 原始runner输出目录: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1/20260625_140116_prefill_stage1_decode_smoke_b4p512_cg_on_b4_f1_p512_layers3_backend_auto_cg_on_pcg_off_tp1_marker_on_profile_nsys`
  - reason: `missing decode CSV: /home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1/20260625_140116_decode_stage1_decode_smoke_b4p512_cg_on_b4_f1_p512_layers3_backend_auto_cg_on_pcg_off_tp1_marker_on_profile_nsys/nsys/MLA时延拆解.csv`

## 本批次总结
- total_cases: `1`
- failures: `1`
### 新批次
- case_group: `smoke_decode_only`
- start_index: `0`
- max_cases: `1`
- skip_existing_ok: `False`
- bench_root: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/bench_data/h100_sxm/sglang/v0.5.9/deepseek_v3_mla`
- [1/1] 开始 `decode_smoke_b4p512_cg_on`
  - cmd: `python .self/task-localbench/formal-prefill-stage1/run_formal_prefill_stage1.py --gpu-id 5 --num-layers 3 --batch-size 4 --fresh-len 1 --prefix-len 512 --context-length 528 --max-new-tokens 2 --warmup-runs 1 --profile-mode nsys --cuda-graph-mode on --pcg-mode off --tp-size 1 --tag decode_smoke_b4p512_cg_on --run-root /home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1 --mem-fraction-static 0.5 --chunked-prefix-cache-threshold 8192 --decode-empty-input-continuation --profile-by-stage --profile-num-steps 1 --profile-decode-only-stage --limit-cuda-graph-bs-to-batch --layerwise-marker`
### 新批次
- case_group: `smoke_decode_only`
- start_index: `0`
- max_cases: `1`
- skip_existing_ok: `False`
- bench_root: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/bench_data/h100_sxm/sglang/v0.5.9/deepseek_v3_mla`
- [1/1] 开始 `decode_smoke_b4p512_cg_on`
  - cmd: `python .self/task-localbench/formal-prefill-stage1/run_formal_prefill_stage1.py --gpu-id 5 --num-layers 3 --batch-size 4 --fresh-len 1 --prefix-len 512 --context-length 528 --max-new-tokens 2 --warmup-runs 1 --profile-mode nsys --cuda-graph-mode on --pcg-mode off --tp-size 1 --tag decode_smoke_b4p512_cg_on --run-root /home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1 --mem-fraction-static 0.5 --chunked-prefix-cache-threshold 8192 --decode-empty-input-continuation --profile-by-stage --profile-num-steps 1 --profile-decode-only-stage --limit-cuda-graph-bs-to-batch --layerwise-marker`
  - 成功: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1/20260625_140912_decode_stage1_decode_smoke_b4p512_cg_on_b4_f1_p512_layers3_backend_auto_cg_on_pcg_off_tp1_marker_on_profile_nsys`
  - 原始runner输出目录: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1/20260625_140912_prefill_stage1_decode_smoke_b4p512_cg_on_b4_f1_p512_layers3_backend_auto_cg_on_pcg_off_tp1_marker_on_profile_nsys`
  - 校验: `validated 86 decode kernel rows; decode_cuda_graph_true_logs=4; prefill_cuda_graph_true_logs=0`
  - CSV: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/bench_data/h100_sxm/sglang/v0.5.9/deepseek_v3_mla/decode_csv/20260625_140912_decode_stage1_decode_smoke_b4p512_cg_on_b4_f1_p512_layers3_backend_auto_cg_on_pcg_off_tp1_marker_on_profile_nsys__DecodeCudaGraphKernel汇总.csv`

## 本批次总结
- total_cases: `1`
- failures: `0`
### 新批次
- case_group: `decode_only`
- start_index: `0`
- max_cases: `None`
- skip_existing_ok: `False`
- bench_root: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/bench_data/h100_sxm/sglang/v0.5.9/deepseek_v3_mla`
- [1/9] 开始 `decode_b1p512_cg_on`
  - cmd: `python .self/task-localbench/formal-prefill-stage1/run_formal_prefill_stage1.py --gpu-id 5 --num-layers 5 --batch-size 1 --fresh-len 1 --prefix-len 512 --context-length 528 --max-new-tokens 2 --warmup-runs 1 --profile-mode nsys --cuda-graph-mode on --pcg-mode off --tp-size 1 --tag decode_b1p512_cg_on --run-root /home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1 --mem-fraction-static 0.5 --chunked-prefix-cache-threshold 8192 --decode-empty-input-continuation --profile-by-stage --profile-num-steps 1 --profile-decode-only-stage --limit-cuda-graph-bs-to-batch --layerwise-marker`
  - 成功: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1/20260625_141235_decode_stage1_decode_b1p512_cg_on_b1_f1_p512_layers5_backend_auto_cg_on_pcg_off_tp1_marker_on_profile_nsys`
  - 原始runner输出目录: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1/20260625_141235_prefill_stage1_decode_b1p512_cg_on_b1_f1_p512_layers5_backend_auto_cg_on_pcg_off_tp1_marker_on_profile_nsys`
  - 校验: `validated 138 decode kernel rows; decode_cuda_graph_true_logs=4; prefill_cuda_graph_true_logs=0`
  - CSV: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/bench_data/h100_sxm/sglang/v0.5.9/deepseek_v3_mla/decode_csv/20260625_141235_decode_stage1_decode_b1p512_cg_on_b1_f1_p512_layers5_backend_auto_cg_on_pcg_off_tp1_marker_on_profile_nsys__DecodeCudaGraphKernel汇总.csv`
- [2/9] 开始 `decode_b4p512_cg_on`
  - cmd: `python .self/task-localbench/formal-prefill-stage1/run_formal_prefill_stage1.py --gpu-id 5 --num-layers 5 --batch-size 4 --fresh-len 1 --prefix-len 512 --context-length 528 --max-new-tokens 2 --warmup-runs 1 --profile-mode nsys --cuda-graph-mode on --pcg-mode off --tp-size 1 --tag decode_b4p512_cg_on --run-root /home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1 --mem-fraction-static 0.5 --chunked-prefix-cache-threshold 8192 --decode-empty-input-continuation --profile-by-stage --profile-num-steps 1 --profile-decode-only-stage --limit-cuda-graph-bs-to-batch --layerwise-marker`
  - 成功: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1/20260625_141521_decode_stage1_decode_b4p512_cg_on_b4_f1_p512_layers5_backend_auto_cg_on_pcg_off_tp1_marker_on_profile_nsys`
  - 原始runner输出目录: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1/20260625_141521_prefill_stage1_decode_b4p512_cg_on_b4_f1_p512_layers5_backend_auto_cg_on_pcg_off_tp1_marker_on_profile_nsys`
  - 校验: `validated 140 decode kernel rows; decode_cuda_graph_true_logs=4; prefill_cuda_graph_true_logs=0`
  - CSV: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/bench_data/h100_sxm/sglang/v0.5.9/deepseek_v3_mla/decode_csv/20260625_141521_decode_stage1_decode_b4p512_cg_on_b4_f1_p512_layers5_backend_auto_cg_on_pcg_off_tp1_marker_on_profile_nsys__DecodeCudaGraphKernel汇总.csv`
- [3/9] 开始 `decode_b32p512_cg_on`
  - cmd: `python .self/task-localbench/formal-prefill-stage1/run_formal_prefill_stage1.py --gpu-id 5 --num-layers 5 --batch-size 32 --fresh-len 1 --prefix-len 512 --context-length 528 --max-new-tokens 2 --warmup-runs 1 --profile-mode nsys --cuda-graph-mode on --pcg-mode off --tp-size 1 --tag decode_b32p512_cg_on --run-root /home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1 --mem-fraction-static 0.5 --chunked-prefix-cache-threshold 16417 --decode-empty-input-continuation --profile-by-stage --profile-num-steps 1 --profile-decode-only-stage --limit-cuda-graph-bs-to-batch --layerwise-marker`
  - 成功: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1/20260625_141800_decode_stage1_decode_b32p512_cg_on_b32_f1_p512_layers5_backend_auto_cg_on_pcg_off_tp1_marker_on_profile_nsys`
  - 原始runner输出目录: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1/20260625_141800_prefill_stage1_decode_b32p512_cg_on_b32_f1_p512_layers5_backend_auto_cg_on_pcg_off_tp1_marker_on_profile_nsys`
  - 校验: `validated 144 decode kernel rows; decode_cuda_graph_true_logs=4; prefill_cuda_graph_true_logs=0`
  - CSV: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/bench_data/h100_sxm/sglang/v0.5.9/deepseek_v3_mla/decode_csv/20260625_141800_decode_stage1_decode_b32p512_cg_on_b32_f1_p512_layers5_backend_auto_cg_on_pcg_off_tp1_marker_on_profile_nsys__DecodeCudaGraphKernel汇总.csv`
- [4/9] 开始 `decode_b4p2048_cg_on`
  - cmd: `python .self/task-localbench/formal-prefill-stage1/run_formal_prefill_stage1.py --gpu-id 5 --num-layers 5 --batch-size 4 --fresh-len 1 --prefix-len 2048 --context-length 2064 --max-new-tokens 2 --warmup-runs 1 --profile-mode nsys --cuda-graph-mode on --pcg-mode off --tp-size 1 --tag decode_b4p2048_cg_on --run-root /home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1 --mem-fraction-static 0.5 --chunked-prefix-cache-threshold 8197 --decode-empty-input-continuation --profile-by-stage --profile-num-steps 1 --profile-decode-only-stage --limit-cuda-graph-bs-to-batch --layerwise-marker`
  - 成功: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1/20260625_142044_decode_stage1_decode_b4p2048_cg_on_b4_f1_p2048_layers5_backend_auto_cg_on_pcg_off_tp1_marker_on_profile_nsys`
  - 原始runner输出目录: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1/20260625_142044_prefill_stage1_decode_b4p2048_cg_on_b4_f1_p2048_layers5_backend_auto_cg_on_pcg_off_tp1_marker_on_profile_nsys`
  - 校验: `validated 140 decode kernel rows; decode_cuda_graph_true_logs=4; prefill_cuda_graph_true_logs=0`
  - CSV: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/bench_data/h100_sxm/sglang/v0.5.9/deepseek_v3_mla/decode_csv/20260625_142044_decode_stage1_decode_b4p2048_cg_on_b4_f1_p2048_layers5_backend_auto_cg_on_pcg_off_tp1_marker_on_profile_nsys__DecodeCudaGraphKernel汇总.csv`
- [5/9] 开始 `decode_b16p2048_cg_on`
  - cmd: `python .self/task-localbench/formal-prefill-stage1/run_formal_prefill_stage1.py --gpu-id 5 --num-layers 5 --batch-size 16 --fresh-len 1 --prefix-len 2048 --context-length 2064 --max-new-tokens 2 --warmup-runs 1 --profile-mode nsys --cuda-graph-mode on --pcg-mode off --tp-size 1 --tag decode_b16p2048_cg_on --run-root /home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1 --mem-fraction-static 0.5 --chunked-prefix-cache-threshold 32785 --decode-empty-input-continuation --profile-by-stage --profile-num-steps 1 --profile-decode-only-stage --limit-cuda-graph-bs-to-batch --layerwise-marker`
  - 成功: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1/20260625_142332_decode_stage1_decode_b16p2048_cg_on_b16_f1_p2048_layers5_backend_auto_cg_on_pcg_off_tp1_marker_on_profile_nsys`
  - 原始runner输出目录: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1/20260625_142332_prefill_stage1_decode_b16p2048_cg_on_b16_f1_p2048_layers5_backend_auto_cg_on_pcg_off_tp1_marker_on_profile_nsys`
  - 校验: `validated 140 decode kernel rows; decode_cuda_graph_true_logs=4; prefill_cuda_graph_true_logs=0`
  - CSV: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/bench_data/h100_sxm/sglang/v0.5.9/deepseek_v3_mla/decode_csv/20260625_142332_decode_stage1_decode_b16p2048_cg_on_b16_f1_p2048_layers5_backend_auto_cg_on_pcg_off_tp1_marker_on_profile_nsys__DecodeCudaGraphKernel汇总.csv`
- [6/9] 开始 `decode_b4p4096_cg_on`
  - cmd: `python .self/task-localbench/formal-prefill-stage1/run_formal_prefill_stage1.py --gpu-id 5 --num-layers 5 --batch-size 4 --fresh-len 1 --prefix-len 4096 --context-length 4112 --max-new-tokens 2 --warmup-runs 1 --profile-mode nsys --cuda-graph-mode on --pcg-mode off --tp-size 1 --tag decode_b4p4096_cg_on --run-root /home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1 --mem-fraction-static 0.5 --chunked-prefix-cache-threshold 16389 --decode-empty-input-continuation --profile-by-stage --profile-num-steps 1 --profile-decode-only-stage --limit-cuda-graph-bs-to-batch --layerwise-marker`
  - 成功: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1/20260625_142620_decode_stage1_decode_b4p4096_cg_on_b4_f1_p4096_layers5_backend_auto_cg_on_pcg_off_tp1_marker_on_profile_nsys`
  - 原始runner输出目录: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1/20260625_142620_prefill_stage1_decode_b4p4096_cg_on_b4_f1_p4096_layers5_backend_auto_cg_on_pcg_off_tp1_marker_on_profile_nsys`
  - 校验: `validated 140 decode kernel rows; decode_cuda_graph_true_logs=4; prefill_cuda_graph_true_logs=0`
  - CSV: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/bench_data/h100_sxm/sglang/v0.5.9/deepseek_v3_mla/decode_csv/20260625_142620_decode_stage1_decode_b4p4096_cg_on_b4_f1_p4096_layers5_backend_auto_cg_on_pcg_off_tp1_marker_on_profile_nsys__DecodeCudaGraphKernel汇总.csv`
- [7/9] 开始 `decode_b8p4096_cg_on`
  - cmd: `python .self/task-localbench/formal-prefill-stage1/run_formal_prefill_stage1.py --gpu-id 5 --num-layers 5 --batch-size 8 --fresh-len 1 --prefix-len 4096 --context-length 4112 --max-new-tokens 2 --warmup-runs 1 --profile-mode nsys --cuda-graph-mode on --pcg-mode off --tp-size 1 --tag decode_b8p4096_cg_on --run-root /home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1 --mem-fraction-static 0.5 --chunked-prefix-cache-threshold 32777 --decode-empty-input-continuation --profile-by-stage --profile-num-steps 1 --profile-decode-only-stage --limit-cuda-graph-bs-to-batch --layerwise-marker`
  - 成功: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1/20260625_142903_decode_stage1_decode_b8p4096_cg_on_b8_f1_p4096_layers5_backend_auto_cg_on_pcg_off_tp1_marker_on_profile_nsys`
  - 原始runner输出目录: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1/20260625_142903_prefill_stage1_decode_b8p4096_cg_on_b8_f1_p4096_layers5_backend_auto_cg_on_pcg_off_tp1_marker_on_profile_nsys`
  - 校验: `validated 140 decode kernel rows; decode_cuda_graph_true_logs=4; prefill_cuda_graph_true_logs=0`
  - CSV: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/bench_data/h100_sxm/sglang/v0.5.9/deepseek_v3_mla/decode_csv/20260625_142903_decode_stage1_decode_b8p4096_cg_on_b8_f1_p4096_layers5_backend_auto_cg_on_pcg_off_tp1_marker_on_profile_nsys__DecodeCudaGraphKernel汇总.csv`
- [8/9] 开始 `decode_b4p8192_cg_on`
  - cmd: `python .self/task-localbench/formal-prefill-stage1/run_formal_prefill_stage1.py --gpu-id 5 --num-layers 5 --batch-size 4 --fresh-len 1 --prefix-len 8192 --context-length 8208 --max-new-tokens 2 --warmup-runs 1 --profile-mode nsys --cuda-graph-mode on --pcg-mode off --tp-size 1 --tag decode_b4p8192_cg_on --run-root /home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1 --mem-fraction-static 0.5 --chunked-prefix-cache-threshold 32773 --decode-empty-input-continuation --profile-by-stage --profile-num-steps 1 --profile-decode-only-stage --limit-cuda-graph-bs-to-batch --layerwise-marker`
  - 成功: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1/20260625_143152_decode_stage1_decode_b4p8192_cg_on_b4_f1_p8192_layers5_backend_auto_cg_on_pcg_off_tp1_marker_on_profile_nsys`
  - 原始runner输出目录: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1/20260625_143152_prefill_stage1_decode_b4p8192_cg_on_b4_f1_p8192_layers5_backend_auto_cg_on_pcg_off_tp1_marker_on_profile_nsys`
  - 校验: `validated 140 decode kernel rows; decode_cuda_graph_true_logs=4; prefill_cuda_graph_true_logs=0`
  - CSV: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/bench_data/h100_sxm/sglang/v0.5.9/deepseek_v3_mla/decode_csv/20260625_143152_decode_stage1_decode_b4p8192_cg_on_b4_f1_p8192_layers5_backend_auto_cg_on_pcg_off_tp1_marker_on_profile_nsys__DecodeCudaGraphKernel汇总.csv`
- [9/9] 开始 `decode_b2p16384_cg_on`
  - cmd: `python .self/task-localbench/formal-prefill-stage1/run_formal_prefill_stage1.py --gpu-id 5 --num-layers 5 --batch-size 2 --fresh-len 1 --prefix-len 16384 --context-length 16400 --max-new-tokens 2 --warmup-runs 1 --profile-mode nsys --cuda-graph-mode on --pcg-mode off --tp-size 1 --tag decode_b2p16384_cg_on --run-root /home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1 --mem-fraction-static 0.5 --chunked-prefix-cache-threshold 32771 --decode-empty-input-continuation --profile-by-stage --profile-num-steps 1 --profile-decode-only-stage --limit-cuda-graph-bs-to-batch --layerwise-marker`
  - 成功: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1/20260625_143436_decode_stage1_decode_b2p16384_cg_on_b2_f1_p16384_layers5_backend_auto_cg_on_pcg_off_tp1_marker_on_profile_nsys`
  - 原始runner输出目录: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1/20260625_143436_prefill_stage1_decode_b2p16384_cg_on_b2_f1_p16384_layers5_backend_auto_cg_on_pcg_off_tp1_marker_on_profile_nsys`
  - 校验: `validated 140 decode kernel rows; decode_cuda_graph_true_logs=4; prefill_cuda_graph_true_logs=0`
  - CSV: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/bench_data/h100_sxm/sglang/v0.5.9/deepseek_v3_mla/decode_csv/20260625_143436_decode_stage1_decode_b2p16384_cg_on_b2_f1_p16384_layers5_backend_auto_cg_on_pcg_off_tp1_marker_on_profile_nsys__DecodeCudaGraphKernel汇总.csv`

## 本批次总结
- total_cases: `9`
- failures: `0`
### 新批次
- case_group: `smoke_flashmla`
- start_index: `0`
- max_cases: `1`
- skip_existing_ok: `False`
- bench_root: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/bench_data/h100_sxm/sglang/v0.5.9/deepseek_v3_mla`
- [1/1] 开始 `decode_flashmla_smoke_b4p512_cg_on`
  - cmd: `python .self/task-localbench/formal-prefill-stage1/run_formal_prefill_stage1.py --gpu-id 5 --num-layers 3 --batch-size 4 --fresh-len 1 --prefix-len 512 --context-length 528 --max-new-tokens 2 --warmup-runs 1 --profile-mode nsys --cuda-graph-mode on --pcg-mode off --tp-size 1 --tag decode_flashmla_smoke_b4p512_cg_on --run-root /home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1 --mem-fraction-static 0.5 --chunked-prefix-cache-threshold 8192 --decode-empty-input-continuation --profile-by-stage --profile-num-steps 1 --profile-decode-only-stage --limit-cuda-graph-bs-to-batch --decode-attention-backend flashmla --layerwise-marker`
### 新批次
- case_group: `smoke_flashmla`
- start_index: `0`
- max_cases: `1`
- skip_existing_ok: `False`
- bench_root: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/bench_data/h100_sxm/sglang/v0.5.9/deepseek_v3_mla`
- [1/1] 开始 `decode_flashmla_smoke_b4p512_cg_on`
  - cmd: `python .self/task-localbench/formal-prefill-stage1/run_formal_prefill_stage1.py --gpu-id 5 --num-layers 3 --batch-size 4 --fresh-len 1 --prefix-len 512 --context-length 528 --max-new-tokens 2 --warmup-runs 1 --profile-mode nsys --cuda-graph-mode on --pcg-mode off --tp-size 1 --tag decode_flashmla_smoke_b4p512_cg_on --run-root /home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1 --mem-fraction-static 0.5 --chunked-prefix-cache-threshold 8192 --decode-empty-input-continuation --profile-by-stage --profile-num-steps 1 --profile-decode-only-stage --limit-cuda-graph-bs-to-batch --decode-attention-backend flashmla --layerwise-marker`
  - 成功: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1/20260626_151154_decode_stage1_decode_flashmla_smoke_b4p512_cg_on_b4_f1_p512_layers3_backend_auto_decodebackend_flashmla_cg_on_pcg_off_tp1_marker_on_profile_nsys`
  - 原始runner输出目录: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1/20260626_151154_prefill_stage1_decode_flashmla_smoke_b4p512_cg_on_b4_f1_p512_layers3_backend_auto_decodebackend_flashmla_cg_on_pcg_off_tp1_marker_on_profile_nsys`
  - 校验: `validated 83 decode kernel rows; decode_cuda_graph_true_logs=4; prefill_cuda_graph_true_logs=0`
  - CSV: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/bench_data/h100_sxm/sglang/v0.5.9/deepseek_v3_mla/decode_csv/20260626_151154_decode_stage1_decode_flashmla_smoke_b4p512_cg_on_b4_f1_p512_layers3_backend_auto_decodebackend_flashmla_cg_on_pcg_off_tp1_marker_on_profile_nsys__DecodeCudaGraphKernel汇总.csv`

## 本批次总结
- total_cases: `1`
- failures: `0`
### 新批次
- case_group: `flashmla_decode_only`
- start_index: `0`
- max_cases: `None`
- skip_existing_ok: `True`
- bench_root: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/bench_data/h100_sxm/sglang/v0.5.9/deepseek_v3_mla`
- [1/17] 开始 `decode_flashmla_b1p512_cg_on`
  - cmd: `python .self/task-localbench/formal-prefill-stage1/run_formal_prefill_stage1.py --gpu-id 5 --num-layers 5 --batch-size 1 --fresh-len 1 --prefix-len 512 --context-length 528 --max-new-tokens 2 --warmup-runs 1 --profile-mode nsys --cuda-graph-mode on --pcg-mode off --tp-size 1 --tag decode_flashmla_b1p512_cg_on --run-root /home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1 --mem-fraction-static 0.5 --chunked-prefix-cache-threshold 8192 --decode-empty-input-continuation --profile-by-stage --profile-num-steps 1 --profile-decode-only-stage --limit-cuda-graph-bs-to-batch --decode-attention-backend flashmla --layerwise-marker`
  - 成功: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1/20260626_151614_decode_stage1_decode_flashmla_b1p512_cg_on_b1_f1_p512_layers5_backend_auto_decodebackend_flashmla_cg_on_pcg_off_tp1_marker_on_profile_nsys`
  - 原始runner输出目录: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1/20260626_151614_prefill_stage1_decode_flashmla_b1p512_cg_on_b1_f1_p512_layers5_backend_auto_decodebackend_flashmla_cg_on_pcg_off_tp1_marker_on_profile_nsys`
  - 校验: `validated 139 decode kernel rows; decode_cuda_graph_true_logs=4; prefill_cuda_graph_true_logs=0`
  - CSV: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/bench_data/h100_sxm/sglang/v0.5.9/deepseek_v3_mla/decode_csv/20260626_151614_decode_stage1_decode_flashmla_b1p512_cg_on_b1_f1_p512_layers5_backend_auto_decodebackend_flashmla_cg_on_pcg_off_tp1_marker_on_profile_nsys__DecodeCudaGraphKernel汇总.csv`
- [2/17] 开始 `decode_flashmla_b4p512_cg_on`
  - cmd: `python .self/task-localbench/formal-prefill-stage1/run_formal_prefill_stage1.py --gpu-id 5 --num-layers 5 --batch-size 4 --fresh-len 1 --prefix-len 512 --context-length 528 --max-new-tokens 2 --warmup-runs 1 --profile-mode nsys --cuda-graph-mode on --pcg-mode off --tp-size 1 --tag decode_flashmla_b4p512_cg_on --run-root /home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1 --mem-fraction-static 0.5 --chunked-prefix-cache-threshold 8192 --decode-empty-input-continuation --profile-by-stage --profile-num-steps 1 --profile-decode-only-stage --limit-cuda-graph-bs-to-batch --decode-attention-backend flashmla --layerwise-marker`
  - 成功: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1/20260626_151908_decode_stage1_decode_flashmla_b4p512_cg_on_b4_f1_p512_layers5_backend_auto_decodebackend_flashmla_cg_on_pcg_off_tp1_marker_on_profile_nsys`
  - 原始runner输出目录: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1/20260626_151908_prefill_stage1_decode_flashmla_b4p512_cg_on_b4_f1_p512_layers5_backend_auto_decodebackend_flashmla_cg_on_pcg_off_tp1_marker_on_profile_nsys`
  - 校验: `validated 139 decode kernel rows; decode_cuda_graph_true_logs=4; prefill_cuda_graph_true_logs=0`
  - CSV: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/bench_data/h100_sxm/sglang/v0.5.9/deepseek_v3_mla/decode_csv/20260626_151908_decode_stage1_decode_flashmla_b4p512_cg_on_b4_f1_p512_layers5_backend_auto_decodebackend_flashmla_cg_on_pcg_off_tp1_marker_on_profile_nsys__DecodeCudaGraphKernel汇总.csv`
- [3/17] 开始 `decode_flashmla_b16p512_cg_on`
  - cmd: `python .self/task-localbench/formal-prefill-stage1/run_formal_prefill_stage1.py --gpu-id 5 --num-layers 5 --batch-size 16 --fresh-len 1 --prefix-len 512 --context-length 528 --max-new-tokens 2 --warmup-runs 1 --profile-mode nsys --cuda-graph-mode on --pcg-mode off --tp-size 1 --tag decode_flashmla_b16p512_cg_on --run-root /home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1 --mem-fraction-static 0.5 --chunked-prefix-cache-threshold 8209 --decode-empty-input-continuation --profile-by-stage --profile-num-steps 1 --profile-decode-only-stage --limit-cuda-graph-bs-to-batch --decode-attention-backend flashmla --layerwise-marker`
  - 成功: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1/20260626_152200_decode_stage1_decode_flashmla_b16p512_cg_on_b16_f1_p512_layers5_backend_auto_decodebackend_flashmla_cg_on_pcg_off_tp1_marker_on_profile_nsys`
  - 原始runner输出目录: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1/20260626_152200_prefill_stage1_decode_flashmla_b16p512_cg_on_b16_f1_p512_layers5_backend_auto_decodebackend_flashmla_cg_on_pcg_off_tp1_marker_on_profile_nsys`
  - 校验: `validated 139 decode kernel rows; decode_cuda_graph_true_logs=4; prefill_cuda_graph_true_logs=0`
  - CSV: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/bench_data/h100_sxm/sglang/v0.5.9/deepseek_v3_mla/decode_csv/20260626_152200_decode_stage1_decode_flashmla_b16p512_cg_on_b16_f1_p512_layers5_backend_auto_decodebackend_flashmla_cg_on_pcg_off_tp1_marker_on_profile_nsys__DecodeCudaGraphKernel汇总.csv`
- [4/17] 开始 `decode_flashmla_b32p512_cg_on`
  - cmd: `python .self/task-localbench/formal-prefill-stage1/run_formal_prefill_stage1.py --gpu-id 5 --num-layers 5 --batch-size 32 --fresh-len 1 --prefix-len 512 --context-length 528 --max-new-tokens 2 --warmup-runs 1 --profile-mode nsys --cuda-graph-mode on --pcg-mode off --tp-size 1 --tag decode_flashmla_b32p512_cg_on --run-root /home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1 --mem-fraction-static 0.5 --chunked-prefix-cache-threshold 16417 --decode-empty-input-continuation --profile-by-stage --profile-num-steps 1 --profile-decode-only-stage --limit-cuda-graph-bs-to-batch --decode-attention-backend flashmla --layerwise-marker`
  - 成功: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1/20260626_152437_decode_stage1_decode_flashmla_b32p512_cg_on_b32_f1_p512_layers5_backend_auto_decodebackend_flashmla_cg_on_pcg_off_tp1_marker_on_profile_nsys`
  - 原始runner输出目录: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1/20260626_152437_prefill_stage1_decode_flashmla_b32p512_cg_on_b32_f1_p512_layers5_backend_auto_decodebackend_flashmla_cg_on_pcg_off_tp1_marker_on_profile_nsys`
  - 校验: `validated 143 decode kernel rows; decode_cuda_graph_true_logs=4; prefill_cuda_graph_true_logs=0`
  - CSV: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/bench_data/h100_sxm/sglang/v0.5.9/deepseek_v3_mla/decode_csv/20260626_152437_decode_stage1_decode_flashmla_b32p512_cg_on_b32_f1_p512_layers5_backend_auto_decodebackend_flashmla_cg_on_pcg_off_tp1_marker_on_profile_nsys__DecodeCudaGraphKernel汇总.csv`
- [5/17] 开始 `decode_flashmla_b64p512_cg_on`
  - cmd: `python .self/task-localbench/formal-prefill-stage1/run_formal_prefill_stage1.py --gpu-id 5 --num-layers 5 --batch-size 64 --fresh-len 1 --prefix-len 512 --context-length 528 --max-new-tokens 2 --warmup-runs 1 --profile-mode nsys --cuda-graph-mode on --pcg-mode off --tp-size 1 --tag decode_flashmla_b64p512_cg_on --run-root /home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1 --mem-fraction-static 0.6 --chunked-prefix-cache-threshold 32833 --decode-empty-input-continuation --profile-by-stage --profile-num-steps 1 --profile-decode-only-stage --limit-cuda-graph-bs-to-batch --decode-attention-backend flashmla --layerwise-marker`
  - 成功: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1/20260626_152720_decode_stage1_decode_flashmla_b64p512_cg_on_b64_f1_p512_layers5_backend_auto_decodebackend_flashmla_cg_on_pcg_off_tp1_marker_on_profile_nsys`
  - 原始runner输出目录: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1/20260626_152720_prefill_stage1_decode_flashmla_b64p512_cg_on_b64_f1_p512_layers5_backend_auto_decodebackend_flashmla_cg_on_pcg_off_tp1_marker_on_profile_nsys`
  - 校验: `validated 143 decode kernel rows; decode_cuda_graph_true_logs=4; prefill_cuda_graph_true_logs=0`
  - CSV: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/bench_data/h100_sxm/sglang/v0.5.9/deepseek_v3_mla/decode_csv/20260626_152720_decode_stage1_decode_flashmla_b64p512_cg_on_b64_f1_p512_layers5_backend_auto_decodebackend_flashmla_cg_on_pcg_off_tp1_marker_on_profile_nsys__DecodeCudaGraphKernel汇总.csv`
- [6/17] 开始 `decode_flashmla_b4p2048_cg_on`
  - cmd: `python .self/task-localbench/formal-prefill-stage1/run_formal_prefill_stage1.py --gpu-id 5 --num-layers 5 --batch-size 4 --fresh-len 1 --prefix-len 2048 --context-length 2064 --max-new-tokens 2 --warmup-runs 1 --profile-mode nsys --cuda-graph-mode on --pcg-mode off --tp-size 1 --tag decode_flashmla_b4p2048_cg_on --run-root /home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1 --mem-fraction-static 0.5 --chunked-prefix-cache-threshold 8197 --decode-empty-input-continuation --profile-by-stage --profile-num-steps 1 --profile-decode-only-stage --limit-cuda-graph-bs-to-batch --decode-attention-backend flashmla --layerwise-marker`
  - 成功: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1/20260626_153014_decode_stage1_decode_flashmla_b4p2048_cg_on_b4_f1_p2048_layers5_backend_auto_decodebackend_flashmla_cg_on_pcg_off_tp1_marker_on_profile_nsys`
  - 原始runner输出目录: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1/20260626_153014_prefill_stage1_decode_flashmla_b4p2048_cg_on_b4_f1_p2048_layers5_backend_auto_decodebackend_flashmla_cg_on_pcg_off_tp1_marker_on_profile_nsys`
  - 校验: `validated 139 decode kernel rows; decode_cuda_graph_true_logs=4; prefill_cuda_graph_true_logs=0`
  - CSV: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/bench_data/h100_sxm/sglang/v0.5.9/deepseek_v3_mla/decode_csv/20260626_153014_decode_stage1_decode_flashmla_b4p2048_cg_on_b4_f1_p2048_layers5_backend_auto_decodebackend_flashmla_cg_on_pcg_off_tp1_marker_on_profile_nsys__DecodeCudaGraphKernel汇总.csv`
- [7/17] 开始 `decode_flashmla_b16p2048_cg_on`
  - cmd: `python .self/task-localbench/formal-prefill-stage1/run_formal_prefill_stage1.py --gpu-id 5 --num-layers 5 --batch-size 16 --fresh-len 1 --prefix-len 2048 --context-length 2064 --max-new-tokens 2 --warmup-runs 1 --profile-mode nsys --cuda-graph-mode on --pcg-mode off --tp-size 1 --tag decode_flashmla_b16p2048_cg_on --run-root /home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1 --mem-fraction-static 0.5 --chunked-prefix-cache-threshold 32785 --decode-empty-input-continuation --profile-by-stage --profile-num-steps 1 --profile-decode-only-stage --limit-cuda-graph-bs-to-batch --decode-attention-backend flashmla --layerwise-marker`
  - 成功: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1/20260626_153310_decode_stage1_decode_flashmla_b16p2048_cg_on_b16_f1_p2048_layers5_backend_auto_decodebackend_flashmla_cg_on_pcg_off_tp1_marker_on_profile_nsys`
  - 原始runner输出目录: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1/20260626_153310_prefill_stage1_decode_flashmla_b16p2048_cg_on_b16_f1_p2048_layers5_backend_auto_decodebackend_flashmla_cg_on_pcg_off_tp1_marker_on_profile_nsys`
  - 校验: `validated 139 decode kernel rows; decode_cuda_graph_true_logs=4; prefill_cuda_graph_true_logs=0`
  - CSV: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/bench_data/h100_sxm/sglang/v0.5.9/deepseek_v3_mla/decode_csv/20260626_153310_decode_stage1_decode_flashmla_b16p2048_cg_on_b16_f1_p2048_layers5_backend_auto_decodebackend_flashmla_cg_on_pcg_off_tp1_marker_on_profile_nsys__DecodeCudaGraphKernel汇总.csv`
- [8/17] 开始 `decode_flashmla_b32p2048_cg_on`
  - cmd: `python .self/task-localbench/formal-prefill-stage1/run_formal_prefill_stage1.py --gpu-id 5 --num-layers 5 --batch-size 32 --fresh-len 1 --prefix-len 2048 --context-length 2064 --max-new-tokens 2 --warmup-runs 1 --profile-mode nsys --cuda-graph-mode on --pcg-mode off --tp-size 1 --tag decode_flashmla_b32p2048_cg_on --run-root /home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1 --mem-fraction-static 0.6 --chunked-prefix-cache-threshold 65569 --decode-empty-input-continuation --profile-by-stage --profile-num-steps 1 --profile-decode-only-stage --limit-cuda-graph-bs-to-batch --decode-attention-backend flashmla --layerwise-marker`
  - 成功: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1/20260626_153544_decode_stage1_decode_flashmla_b32p2048_cg_on_b32_f1_p2048_layers5_backend_auto_decodebackend_flashmla_cg_on_pcg_off_tp1_marker_on_profile_nsys`
  - 原始runner输出目录: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1/20260626_153544_prefill_stage1_decode_flashmla_b32p2048_cg_on_b32_f1_p2048_layers5_backend_auto_decodebackend_flashmla_cg_on_pcg_off_tp1_marker_on_profile_nsys`
  - 校验: `validated 143 decode kernel rows; decode_cuda_graph_true_logs=4; prefill_cuda_graph_true_logs=0`
  - CSV: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/bench_data/h100_sxm/sglang/v0.5.9/deepseek_v3_mla/decode_csv/20260626_153544_decode_stage1_decode_flashmla_b32p2048_cg_on_b32_f1_p2048_layers5_backend_auto_decodebackend_flashmla_cg_on_pcg_off_tp1_marker_on_profile_nsys__DecodeCudaGraphKernel汇总.csv`
- [9/17] 开始 `decode_flashmla_b4p4096_cg_on`
  - cmd: `python .self/task-localbench/formal-prefill-stage1/run_formal_prefill_stage1.py --gpu-id 5 --num-layers 5 --batch-size 4 --fresh-len 1 --prefix-len 4096 --context-length 4112 --max-new-tokens 2 --warmup-runs 1 --profile-mode nsys --cuda-graph-mode on --pcg-mode off --tp-size 1 --tag decode_flashmla_b4p4096_cg_on --run-root /home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1 --mem-fraction-static 0.5 --chunked-prefix-cache-threshold 16389 --decode-empty-input-continuation --profile-by-stage --profile-num-steps 1 --profile-decode-only-stage --limit-cuda-graph-bs-to-batch --decode-attention-backend flashmla --layerwise-marker`
  - 成功: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1/20260626_153812_decode_stage1_decode_flashmla_b4p4096_cg_on_b4_f1_p4096_layers5_backend_auto_decodebackend_flashmla_cg_on_pcg_off_tp1_marker_on_profile_nsys`
  - 原始runner输出目录: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1/20260626_153812_prefill_stage1_decode_flashmla_b4p4096_cg_on_b4_f1_p4096_layers5_backend_auto_decodebackend_flashmla_cg_on_pcg_off_tp1_marker_on_profile_nsys`
  - 校验: `validated 139 decode kernel rows; decode_cuda_graph_true_logs=4; prefill_cuda_graph_true_logs=0`
  - CSV: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/bench_data/h100_sxm/sglang/v0.5.9/deepseek_v3_mla/decode_csv/20260626_153812_decode_stage1_decode_flashmla_b4p4096_cg_on_b4_f1_p4096_layers5_backend_auto_decodebackend_flashmla_cg_on_pcg_off_tp1_marker_on_profile_nsys__DecodeCudaGraphKernel汇总.csv`
- [10/17] 开始 `decode_flashmla_b8p4096_cg_on`
  - cmd: `python .self/task-localbench/formal-prefill-stage1/run_formal_prefill_stage1.py --gpu-id 5 --num-layers 5 --batch-size 8 --fresh-len 1 --prefix-len 4096 --context-length 4112 --max-new-tokens 2 --warmup-runs 1 --profile-mode nsys --cuda-graph-mode on --pcg-mode off --tp-size 1 --tag decode_flashmla_b8p4096_cg_on --run-root /home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1 --mem-fraction-static 0.5 --chunked-prefix-cache-threshold 32777 --decode-empty-input-continuation --profile-by-stage --profile-num-steps 1 --profile-decode-only-stage --limit-cuda-graph-bs-to-batch --decode-attention-backend flashmla --layerwise-marker`
  - 成功: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1/20260626_154114_decode_stage1_decode_flashmla_b8p4096_cg_on_b8_f1_p4096_layers5_backend_auto_decodebackend_flashmla_cg_on_pcg_off_tp1_marker_on_profile_nsys`
  - 原始runner输出目录: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1/20260626_154114_prefill_stage1_decode_flashmla_b8p4096_cg_on_b8_f1_p4096_layers5_backend_auto_decodebackend_flashmla_cg_on_pcg_off_tp1_marker_on_profile_nsys`
  - 校验: `validated 139 decode kernel rows; decode_cuda_graph_true_logs=4; prefill_cuda_graph_true_logs=0`
  - CSV: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/bench_data/h100_sxm/sglang/v0.5.9/deepseek_v3_mla/decode_csv/20260626_154114_decode_stage1_decode_flashmla_b8p4096_cg_on_b8_f1_p4096_layers5_backend_auto_decodebackend_flashmla_cg_on_pcg_off_tp1_marker_on_profile_nsys__DecodeCudaGraphKernel汇总.csv`
- [11/17] 开始 `decode_flashmla_b16p4096_cg_on`
  - cmd: `python .self/task-localbench/formal-prefill-stage1/run_formal_prefill_stage1.py --gpu-id 5 --num-layers 5 --batch-size 16 --fresh-len 1 --prefix-len 4096 --context-length 4112 --max-new-tokens 2 --warmup-runs 1 --profile-mode nsys --cuda-graph-mode on --pcg-mode off --tp-size 1 --tag decode_flashmla_b16p4096_cg_on --run-root /home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1 --mem-fraction-static 0.6 --chunked-prefix-cache-threshold 65553 --decode-empty-input-continuation --profile-by-stage --profile-num-steps 1 --profile-decode-only-stage --limit-cuda-graph-bs-to-batch --decode-attention-backend flashmla --layerwise-marker`
  - 成功: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1/20260626_154405_decode_stage1_decode_flashmla_b16p4096_cg_on_b16_f1_p4096_layers5_backend_auto_decodebackend_flashmla_cg_on_pcg_off_tp1_marker_on_profile_nsys`
  - 原始runner输出目录: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1/20260626_154405_prefill_stage1_decode_flashmla_b16p4096_cg_on_b16_f1_p4096_layers5_backend_auto_decodebackend_flashmla_cg_on_pcg_off_tp1_marker_on_profile_nsys`
  - 校验: `validated 139 decode kernel rows; decode_cuda_graph_true_logs=4; prefill_cuda_graph_true_logs=0`
  - CSV: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/bench_data/h100_sxm/sglang/v0.5.9/deepseek_v3_mla/decode_csv/20260626_154405_decode_stage1_decode_flashmla_b16p4096_cg_on_b16_f1_p4096_layers5_backend_auto_decodebackend_flashmla_cg_on_pcg_off_tp1_marker_on_profile_nsys__DecodeCudaGraphKernel汇总.csv`
- [12/17] 开始 `decode_flashmla_b4p8192_cg_on`
  - cmd: `python .self/task-localbench/formal-prefill-stage1/run_formal_prefill_stage1.py --gpu-id 5 --num-layers 5 --batch-size 4 --fresh-len 1 --prefix-len 8192 --context-length 8208 --max-new-tokens 2 --warmup-runs 1 --profile-mode nsys --cuda-graph-mode on --pcg-mode off --tp-size 1 --tag decode_flashmla_b4p8192_cg_on --run-root /home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1 --mem-fraction-static 0.5 --chunked-prefix-cache-threshold 32773 --decode-empty-input-continuation --profile-by-stage --profile-num-steps 1 --profile-decode-only-stage --limit-cuda-graph-bs-to-batch --decode-attention-backend flashmla --layerwise-marker`
  - 成功: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1/20260626_154701_decode_stage1_decode_flashmla_b4p8192_cg_on_b4_f1_p8192_layers5_backend_auto_decodebackend_flashmla_cg_on_pcg_off_tp1_marker_on_profile_nsys`
  - 原始runner输出目录: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1/20260626_154701_prefill_stage1_decode_flashmla_b4p8192_cg_on_b4_f1_p8192_layers5_backend_auto_decodebackend_flashmla_cg_on_pcg_off_tp1_marker_on_profile_nsys`
  - 校验: `validated 139 decode kernel rows; decode_cuda_graph_true_logs=4; prefill_cuda_graph_true_logs=0`
  - CSV: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/bench_data/h100_sxm/sglang/v0.5.9/deepseek_v3_mla/decode_csv/20260626_154701_decode_stage1_decode_flashmla_b4p8192_cg_on_b4_f1_p8192_layers5_backend_auto_decodebackend_flashmla_cg_on_pcg_off_tp1_marker_on_profile_nsys__DecodeCudaGraphKernel汇总.csv`
- [13/17] 开始 `decode_flashmla_b8p8192_cg_on`
  - cmd: `python .self/task-localbench/formal-prefill-stage1/run_formal_prefill_stage1.py --gpu-id 5 --num-layers 5 --batch-size 8 --fresh-len 1 --prefix-len 8192 --context-length 8208 --max-new-tokens 2 --warmup-runs 1 --profile-mode nsys --cuda-graph-mode on --pcg-mode off --tp-size 1 --tag decode_flashmla_b8p8192_cg_on --run-root /home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1 --mem-fraction-static 0.6 --chunked-prefix-cache-threshold 65545 --decode-empty-input-continuation --profile-by-stage --profile-num-steps 1 --profile-decode-only-stage --limit-cuda-graph-bs-to-batch --decode-attention-backend flashmla --layerwise-marker`
  - 成功: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1/20260626_155002_decode_stage1_decode_flashmla_b8p8192_cg_on_b8_f1_p8192_layers5_backend_auto_decodebackend_flashmla_cg_on_pcg_off_tp1_marker_on_profile_nsys`
  - 原始runner输出目录: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1/20260626_155002_prefill_stage1_decode_flashmla_b8p8192_cg_on_b8_f1_p8192_layers5_backend_auto_decodebackend_flashmla_cg_on_pcg_off_tp1_marker_on_profile_nsys`
  - 校验: `validated 139 decode kernel rows; decode_cuda_graph_true_logs=4; prefill_cuda_graph_true_logs=0`
  - CSV: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/bench_data/h100_sxm/sglang/v0.5.9/deepseek_v3_mla/decode_csv/20260626_155002_decode_stage1_decode_flashmla_b8p8192_cg_on_b8_f1_p8192_layers5_backend_auto_decodebackend_flashmla_cg_on_pcg_off_tp1_marker_on_profile_nsys__DecodeCudaGraphKernel汇总.csv`
- [14/17] 开始 `decode_flashmla_b2p16384_cg_on`
  - cmd: `python .self/task-localbench/formal-prefill-stage1/run_formal_prefill_stage1.py --gpu-id 5 --num-layers 5 --batch-size 2 --fresh-len 1 --prefix-len 16384 --context-length 16400 --max-new-tokens 2 --warmup-runs 1 --profile-mode nsys --cuda-graph-mode on --pcg-mode off --tp-size 1 --tag decode_flashmla_b2p16384_cg_on --run-root /home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1 --mem-fraction-static 0.5 --chunked-prefix-cache-threshold 32771 --decode-empty-input-continuation --profile-by-stage --profile-num-steps 1 --profile-decode-only-stage --limit-cuda-graph-bs-to-batch --decode-attention-backend flashmla --layerwise-marker`
  - 成功: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1/20260626_155259_decode_stage1_decode_flashmla_b2p16384_cg_on_b2_f1_p16384_layers5_backend_auto_decodebackend_flashmla_cg_on_pcg_off_tp1_marker_on_profile_nsys`
  - 原始runner输出目录: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1/20260626_155259_prefill_stage1_decode_flashmla_b2p16384_cg_on_b2_f1_p16384_layers5_backend_auto_decodebackend_flashmla_cg_on_pcg_off_tp1_marker_on_profile_nsys`
  - 校验: `validated 139 decode kernel rows; decode_cuda_graph_true_logs=4; prefill_cuda_graph_true_logs=0`
  - CSV: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/bench_data/h100_sxm/sglang/v0.5.9/deepseek_v3_mla/decode_csv/20260626_155259_decode_stage1_decode_flashmla_b2p16384_cg_on_b2_f1_p16384_layers5_backend_auto_decodebackend_flashmla_cg_on_pcg_off_tp1_marker_on_profile_nsys__DecodeCudaGraphKernel汇总.csv`
- [15/17] 开始 `decode_flashmla_b4p16384_cg_on`
  - cmd: `python .self/task-localbench/formal-prefill-stage1/run_formal_prefill_stage1.py --gpu-id 5 --num-layers 5 --batch-size 4 --fresh-len 1 --prefix-len 16384 --context-length 16400 --max-new-tokens 2 --warmup-runs 1 --profile-mode nsys --cuda-graph-mode on --pcg-mode off --tp-size 1 --tag decode_flashmla_b4p16384_cg_on --run-root /home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1 --mem-fraction-static 0.6 --chunked-prefix-cache-threshold 65541 --decode-empty-input-continuation --profile-by-stage --profile-num-steps 1 --profile-decode-only-stage --limit-cuda-graph-bs-to-batch --decode-attention-backend flashmla --layerwise-marker`
  - 成功: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1/20260626_155549_decode_stage1_decode_flashmla_b4p16384_cg_on_b4_f1_p16384_layers5_backend_auto_decodebackend_flashmla_cg_on_pcg_off_tp1_marker_on_profile_nsys`
  - 原始runner输出目录: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1/20260626_155549_prefill_stage1_decode_flashmla_b4p16384_cg_on_b4_f1_p16384_layers5_backend_auto_decodebackend_flashmla_cg_on_pcg_off_tp1_marker_on_profile_nsys`
  - 校验: `validated 139 decode kernel rows; decode_cuda_graph_true_logs=4; prefill_cuda_graph_true_logs=0`
  - CSV: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/bench_data/h100_sxm/sglang/v0.5.9/deepseek_v3_mla/decode_csv/20260626_155549_decode_stage1_decode_flashmla_b4p16384_cg_on_b4_f1_p16384_layers5_backend_auto_decodebackend_flashmla_cg_on_pcg_off_tp1_marker_on_profile_nsys__DecodeCudaGraphKernel汇总.csv`
- [16/17] 开始 `decode_flashmla_b1p32768_cg_on`
  - cmd: `python .self/task-localbench/formal-prefill-stage1/run_formal_prefill_stage1.py --gpu-id 5 --num-layers 5 --batch-size 1 --fresh-len 1 --prefix-len 32768 --context-length 32784 --max-new-tokens 2 --warmup-runs 1 --profile-mode nsys --cuda-graph-mode on --pcg-mode off --tp-size 1 --tag decode_flashmla_b1p32768_cg_on --run-root /home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1 --mem-fraction-static 0.5 --chunked-prefix-cache-threshold 32770 --decode-empty-input-continuation --profile-by-stage --profile-num-steps 1 --profile-decode-only-stage --limit-cuda-graph-bs-to-batch --decode-attention-backend flashmla --layerwise-marker`
  - 成功: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1/20260626_155846_decode_stage1_decode_flashmla_b1p32768_cg_on_b1_f1_p32768_layers5_backend_auto_decodebackend_flashmla_cg_on_pcg_off_tp1_marker_on_profile_nsys`
  - 原始runner输出目录: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1/20260626_155846_prefill_stage1_decode_flashmla_b1p32768_cg_on_b1_f1_p32768_layers5_backend_auto_decodebackend_flashmla_cg_on_pcg_off_tp1_marker_on_profile_nsys`
  - 校验: `validated 139 decode kernel rows; decode_cuda_graph_true_logs=4; prefill_cuda_graph_true_logs=0`
  - CSV: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/bench_data/h100_sxm/sglang/v0.5.9/deepseek_v3_mla/decode_csv/20260626_155846_decode_stage1_decode_flashmla_b1p32768_cg_on_b1_f1_p32768_layers5_backend_auto_decodebackend_flashmla_cg_on_pcg_off_tp1_marker_on_profile_nsys__DecodeCudaGraphKernel汇总.csv`
- [17/17] 开始 `decode_flashmla_b2p32768_cg_on`
  - cmd: `python .self/task-localbench/formal-prefill-stage1/run_formal_prefill_stage1.py --gpu-id 5 --num-layers 5 --batch-size 2 --fresh-len 1 --prefix-len 32768 --context-length 32784 --max-new-tokens 2 --warmup-runs 1 --profile-mode nsys --cuda-graph-mode on --pcg-mode off --tp-size 1 --tag decode_flashmla_b2p32768_cg_on --run-root /home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1 --mem-fraction-static 0.6 --chunked-prefix-cache-threshold 65539 --decode-empty-input-continuation --profile-by-stage --profile-num-steps 1 --profile-decode-only-stage --limit-cuda-graph-bs-to-batch --decode-attention-backend flashmla --layerwise-marker`
  - 成功: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1/20260626_160143_decode_stage1_decode_flashmla_b2p32768_cg_on_b2_f1_p32768_layers5_backend_auto_decodebackend_flashmla_cg_on_pcg_off_tp1_marker_on_profile_nsys`
  - 原始runner输出目录: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-decode-stage1/20260626_160143_prefill_stage1_decode_flashmla_b2p32768_cg_on_b2_f1_p32768_layers5_backend_auto_decodebackend_flashmla_cg_on_pcg_off_tp1_marker_on_profile_nsys`
  - 校验: `validated 139 decode kernel rows; decode_cuda_graph_true_logs=4; prefill_cuda_graph_true_logs=0`
  - CSV: `/home/ai_lab/ljc/scale-up-sim/aiconfigurator/bench_data/h100_sxm/sglang/v0.5.9/deepseek_v3_mla/decode_csv/20260626_160143_decode_stage1_decode_flashmla_b2p32768_cg_on_b2_f1_p32768_layers5_backend_auto_decodebackend_flashmla_cg_on_pcg_off_tp1_marker_on_profile_nsys__DecodeCudaGraphKernel汇总.csv`

## 本批次总结
- total_cases: `17`
- failures: `0`
