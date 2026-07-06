# `dsr1-fp8-gb200-dynamo-sglang` 抽取任务

这个目录用于把 InferenceX 中 `dsr1-fp8-gb200-dynamo-sglang` 相关的配置、recipe、以及 benchmark 结果，整理成 AIC 更容易接入的结构化数据。

## 当前已落盘内容

- `extract_inferencex_case.py`
  - 通用抽取脚本
  - 输入：本地 InferenceX repo、config key、可选本地 artifact 目录、可选 `srt-slurm` repo
  - 输出：JSON bundle + CSV 视图

- `out/case_bundle.json`
  - 主结构化结果
  - 包含 case 元信息、scenario summary、逐点展开后的 benchmark points、recipe 解析结果、公开 API / artifact 结果、以及 point/result join

- `out/points.csv`
  - 一行一个 benchmark 点
  - 当前共 20 个点，直接对应 master config 中的 20 个 expected points

- `out/recipes.json`
  - 解析了 `CONFIG_FILE=recipes/...` 引用后，对应的 `srt-slurm` recipe 内容

- `out/recipes_flat.csv`
  - 将 recipe 做了扁平化，便于 grep / 表格查看

- `out/results.json`
  - benchmark 结果层
  - 当前已经通过 InferenceX 公开 API 成功拿到 20 条结果

- `out/joined_points.csv`
  - 以 point 为主表，若后续补充 artifact 结果，会在这里对齐配置点与实测指标
  - 当前 20/20 point 已和公开 benchmark 结果对齐成功

- `out/availability.json`
  - 公开 API 中该 case 的可用日期列表过滤结果

- `out/latest_images.json`
  - 公开 API 中该 case 对应的最新 image 记录

- `out/workflow_info.json`
  - 对应 benchmark 日期的 workflow 元信息
  - 可用于拿到 GitHub run id、run URL、run name

- `out/benchmark_rows_raw.json`
  - 公开 API 原始 benchmark 行
  - 保留原始嵌套 `metrics` 结构

## 当前状态

- 已成功解析：
  - InferenceX master config
  - fixed-seq-len sweep 点位
  - 外部 `srt-slurm` recipe
  - InferenceX 公开 API benchmark 结果
  - 对应 workflow run 元信息

- 当前统计：
  - `point_count = 20`
  - `recipe_count = 7`
  - `result_row_count = 20`
  - `matched_point_count = 20`

## 本次成功抽取到的公开 benchmark

- 选择日期：`2026-02-09`
- GitHub Actions run：`21831958748`
- attempt：`2`
- run URL：
  - `https://github.com/SemiAnalysisAI/InferenceX/actions/runs/21831958748/attempts/2`
- run 标题：
  - `Run Sweep - Update GB200 and GB300 SGLANG FP8 DSR1 Disaggregated inference configs (STP only) (#635)`

这 20 条 benchmark 结果全部是：

- `hardware = gb200`
- `framework = dynamo-sglang`
- `precision = fp8`
- `model = dsr1`
- `spec_method = none`
- `is_multinode = true`
- `disagg = true`

并且与本地从 `nvidia-master.yaml` 展开的 20 个 expected points 完整一一对应。

## 已用本地来源

- InferenceX repo：`/tmp/InferenceX`
- srt-slurm repo：`/tmp/srt-slurm-sa`

## 已用远端来源

- InferenceX dashboard public API：`https://inferencex.semianalysis.com`
- 关键接口：
  - `/api/v1/availability`
  - `/api/v1/latest-images`
  - `/api/v1/workflow-info?date=...`
  - `/api/v1/benchmarks?model=...&date=...&exact=true`

## 复现命令

```bash
python3 .self/task-semibench/dsr1-fp8-gb200-dynamo-sglang/extract_inferencex_case.py \
  --inferencex-repo /tmp/InferenceX \
  --config-key dsr1-fp8-gb200-dynamo-sglang \
  --srt-slurm-repo /tmp/srt-slurm-sa \
  --use-public-api \
  --output-dir .self/task-semibench/dsr1-fp8-gb200-dynamo-sglang/out
```

如果希望固定到某个历史日期而不是自动选最新 image 对应日期，可以额外传：

```bash
--benchmark-date 2026-02-09
```

## 当前脚本支持的三条结果链路

### 1. 公开 API 直连

适合当前这个 case。只要公开 dashboard API 中有对应结果，就可以直接拉取，无需 GitHub shell 登录。

### 2. 提供本地 artifact 目录

如果你先手动下载了某个 InferenceX Actions run 的 `results_bmk` artifact，也可以直接：

```bash
python3 .self/task-semibench/dsr1-fp8-gb200-dynamo-sglang/extract_inferencex_case.py \
  --inferencex-repo /tmp/InferenceX \
  --config-key dsr1-fp8-gb200-dynamo-sglang \
  --srt-slurm-repo /tmp/srt-slurm-sa \
  --artifact-dir /path/to/downloaded/results \
  --output-dir .self/task-semibench/dsr1-fp8-gb200-dynamo-sglang/out
```

### 3. 直接指定 GitHub run id

如果 shell 中 `gh` 已登录，或者提供了可用的 GitHub token，可进一步尝试：

```bash
python3 .self/task-semibench/dsr1-fp8-gb200-dynamo-sglang/extract_inferencex_case.py \
  --inferencex-repo /tmp/InferenceX \
  --config-key dsr1-fp8-gb200-dynamo-sglang \
  --srt-slurm-repo /tmp/srt-slurm-sa \
  --github-run-id <RUN_ID> \
  --output-dir .self/task-semibench/dsr1-fp8-gb200-dynamo-sglang/out
```

脚本会优先尝试下载 `results_bmk` artifact，再自动过滤匹配当前 case 的结果行。

## 对后续 AIC 接入最有用的文件

- `out/points.csv`
  - 表示 InferenceX 期望测试矩阵

- `out/results.csv`
  - 表示该日期下的公开 benchmark 结果

- `out/joined_points.csv`
  - 最适合作为 InferenceX -> AIC 的桥接主表
  - 左边是 config point，右边是已经对齐好的吞吐/时延指标

## `joined_points.csv` 当前已包含的关键结果字段

- 配置字段：
  - `isl`, `osl`, `conc`
  - `prefill_num_workers`, `prefill_tp`, `prefill_ep`, `prefill_dp_attn`
  - `decode_num_workers`, `decode_tp`, `decode_ep`, `decode_dp_attn`
  - `recipe_path`

- 结果字段：
  - `tput_per_gpu`
  - `output_tput_per_gpu`
  - `input_tput_per_gpu`
  - `median_ttft`, `p99_ttft`
  - `median_intvty`, `p99_intvty`
  - `median_e2el`, `p99_e2el`
  - `num_prefill_gpu`, `num_decode_gpu`
  - `result_date`
  - `result_run_url`
