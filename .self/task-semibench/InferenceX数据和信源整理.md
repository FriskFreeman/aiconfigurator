# InferenceX 数据和信源整理

本文面向“对比 InferenceX 实机 bench 与 AIC 仿真”的前置准备，先系统整理 InferenceX 里到底有哪些数据、这些数据分别来自哪里、以及哪些字段最适合后续映射到 AIC。

本文以 `DeepSeek-R1 + GB200 + Dynamo-SGLang` 为主例子。

先说明一个很重要的细节：

- 在当前 InferenceX 官方仓库中，GB200 上并没有看到纯 `sglang` 的配置键。
- 对应条目是 `dynamo-sglang`，即 `dsr1-fp8-gb200-dynamo-sglang` 和 `dsr1-fp4-gb200-dynamo-sglang`。
- 因此后续和 AIC 对齐时，不能简单把它视作“普通单机 SGLang”，而要把它看成“Dynamo + SGLang 的多节点/解耦运行栈”。

---

## 1. 总结结论

对我们这项任务来说，InferenceX 的“数据和信源”大致分成 4 层：

1. **展示层**
   - `https://inferencex.com/`
   - 适合看公开图表、总体趋势、不同硬件/框架对比
   - 不够适合做精确复现，因为图表层通常不暴露完整启动配置

2. **配置定义层**
   - `SemiAnalysisAI/InferenceX` 仓库里的 `.github/configs/*.yaml`
   - 这是“该跑哪些 benchmark、每个 benchmark 的并行配置和 sweep 空间是什么”的官方 source of truth

3. **实际启动层**
   - `benchmarks/`、`runners/`、以及外部 `srt-slurm` recipe
   - 这是“实际如何在机器上启动服务、用哪些 runtime 参数”的真正信源

4. **结果资产层**
   - GitHub Actions run artifacts
   - raw benchmark JSON、processed agg JSON、server logs、GPU telemetry、eval results
   - 这是后续和 AIC 做逐点对比时最关键的一层

对 AIC 对比最重要的不是网页图本身，而是：

- master config
- benchmark script / external recipe
- Actions artifact 中的 raw/agg JSON

---

## 2. 主要信源地图

## 2.1 官方公开说明

- 官方结果仓库：`https://github.com/SemiAnalysisAI/InferenceX`
- 官方 dashboard：`https://inferencex.com/`
- README 明确说明：
  - 只有官方仓库是 official result
  - dashboard 是公开展示层

## 2.2 官方 benchmark 配置定义

最关键的是：

- `.github/configs/nvidia-master.yaml`
- `.github/configs/amd-master.yaml`
- `.github/configs/runners.yaml`
- `.github/configs/CONFIGS.md`

它们定义了：

- model
- model-prefix
- runner type
- image
- precision
- framework
- 是否 multinode
- 是否 disagg
- scenario 类型
- sequence length
- 并行搜索空间

其中 `.github/configs/CONFIGS.md` 明确写了这些配置文件是 benchmark config 的 source of truth。

## 2.3 实际启动脚本

主要来源：

- `benchmarks/single_node/fixed_seq_len/*.sh`
- `benchmarks/single_node/agentic/*.sh`
- `benchmarks/multi_node/**`
- `runners/launch_*.sh`
- `benchmarks/benchmark_lib.sh`

这一层决定：

- 实际启动命令
- TP / EP / DP attention 是否真的启用
- server runtime flags
- benchmark client 如何发请求
- 结果文件如何命名
- GPU telemetry 如何采集

## 2.4 外部 recipe 信源

对于 `dynamo-sglang` / `dynamo-trt` 的多节点配置，还有额外外部信源：

- `https://github.com/NVIDIA/srt-slurm`

InferenceX 仓库中的：

- `benchmarks/multi_node/srt-slurm-recipes/RECIPES.md`

明确说明：

- disaggregated multi-node 配置的 recipes 存在外部 `srt-slurm` 仓库
- `nvidia-master.yaml` 中很多 `additional-settings: CONFIG_FILE=...` 只是引用 recipe 文件
- 更细的 prefill/decode backend 参数、资源拓扑、benchmark 设置，要到外部 recipe YAML 看

这点非常关键，因为：

- InferenceX 顶层 config 里不一定写出所有 runtime 参数
- 真正 launch 细节可能藏在 `CONFIG_FILE=recipes/...yaml`

## 2.5 GitHub Actions 结果资产

结果产物主要来自：

- `run-sweep.yml`
- `benchmark-tmpl.yml`
- `benchmark-multinode-tmpl.yml`
- `collect-results.yml`
- `collect-evals.yml`

可以通过具体 run 页面拿到 artifact，例如用户给出的：

- `https://github.com/SemiAnalysisAI/InferenceX/actions/runs/21793201578`

这一层通常能拿到：

- 单个 benchmark 点的 raw JSON
- 处理后的 agg JSON
- 聚合后的 `results_all` / `results_bmk`
- eval 结果 `eval_results_all`
- server logs
- multinode server logs
- agentic raw outputs

---

## 3. 示例：DeepSeek-R1 + GB200 + Dynamo-SGLang

## 3.1 当前仓库里能定位到的官方配置键

在 `nvidia-master.yaml` 中，这个方向当前能直接找到：

- `dsr1-fp8-gb200-dynamo-sglang`
- `dsr1-fp4-gb200-dynamo-sglang`

对应公共属性：

- `model-prefix: dsr1`
- `runner: gb200`
- `framework: dynamo-sglang`
- `multinode: true`
- `disagg: true`

也就是说，这条数据线天然就是：

- 多节点
- Prefill / Decode 解耦
- 非单机 TP-only 路线

## 3.2 runner / 机器侧标识

`runners.yaml` 中：

- `gb200` 对应 runner nodes：`gb200-nv_0`、`gb200-nv_1`、`gb200-nv_2`

因此从 InferenceX 的命名看：

- 顶层硬件类别叫 `gb200`
- 实际 runner node 名称带 `gb200-nv_*`
- 这和用户口中的 “GB200 NVL72” 是能对应上的，但 InferenceX 内部主键并不是直接写 `nvl72`

## 3.3 FP8 版本 sweep 空间

配置键：`dsr1-fp8-gb200-dynamo-sglang`

基础信息：

- image: `lmsysorg/sglang:v0.5.8.post1-cu130`
- model: `deepseek-ai/DeepSeek-R1-0528`
- runner: `gb200`
- framework: `dynamo-sglang`

### 1k1k

共有 4 个 recipe 组，合计 10 个数据点：

1. low-latency
   - conc: `[4, 8]`
   - prefill: `1 worker, tp=4, ep=1, dp-attn=false`
   - decode: `1 worker, tp=4, ep=1, dp-attn=false`
   - recipe: `recipes/gb200-fp8/1k1k/low-latency.yaml`

2. mid-curve
   - conc: `[1024, 2048, 4096]`
   - prefill: `3 workers, tp=8, ep=8, dp-attn=true`
   - decode: `1 worker, tp=48, ep=48, dp-attn=true`
   - recipe: `recipes/gb200-fp8/1k1k/mid-curve.yaml`

3. max-tpt
   - conc: `[1024, 2048, 4096, 6144]`
   - prefill: `2 workers, tp=8, ep=8, dp-attn=true`
   - decode: `1 worker, tp=32, ep=32, dp-attn=true`
   - recipe: `recipes/gb200-fp8/1k1k/max-tpt.yaml`

4. ultra-tpt
   - conc: `[4096]`
   - prefill: `1 worker, tp=8, ep=8, dp-attn=true`
   - decode: `1 worker, tp=8, ep=8, dp-attn=true`
   - recipe: `recipes/gb200-fp8/1k1k/ultra-tpt.yaml`

### 8k1k

共有 3 个 recipe 组，合计 10 个数据点：

1. low-latency
   - conc: `[4, 8, 16]`
   - prefill: `1 worker, tp=8, ep=1, dp-attn=false`
   - decode: `1 worker, tp=8, ep=1, dp-attn=false`
   - recipe: `recipes/gb200-fp8/8k1k/low-latency.yaml`

2. mid-curve
   - conc: `[512, 1024, 2048, 6144]`
   - prefill: `5 workers, tp=8, ep=8, dp-attn=true`
   - decode: `1 worker, tp=32, ep=32, dp-attn=true`
   - recipe: `recipes/gb200-fp8/8k1k/mid-curve.yaml`

3. max-tpt
   - conc: `[2048, 4096, 6144]`
   - prefill: `6 workers, tp=8, ep=8, dp-attn=true`
   - decode: `1 worker, tp=24, ep=24, dp-attn=true`
   - recipe: `recipes/gb200-fp8/8k1k/max_tpt.yaml`

### 小结

- FP8 / GB200 / Dynamo-SGLang / DSR1
- 固定长度场景至少能得到 `20` 个 throughput benchmark 点

## 3.4 FP4 版本 sweep 空间

配置键：`dsr1-fp4-gb200-dynamo-sglang`

基础信息：

- image: `lmsysorg/sglang:v0.5.8-cu130`
- model: `nvidia/DeepSeek-R1-0528-NVFP4-v2`
- runner: `gb200`
- framework: `dynamo-sglang`

### 1k1k

共有 3 个 recipe 组，合计 9 个数据点：

1. low-latency
   - conc: `[4, 8, 32]`
   - prefill: `1 worker, tp=4, ep=1, dp-attn=false`
   - decode: `2 workers, tp=4, ep=1, dp-attn=false`

2. mid-curve
   - conc: `[512, 2048, 4096, 8192]`
   - prefill: `4 workers, tp=4, ep=4, dp-attn=true`
   - decode: `1 worker, tp=32, ep=32, dp-attn=true`

3. max-tpt
   - conc: `[2048, 4096]`
   - prefill: `4 workers, tp=4, ep=4, dp-attn=true`
   - decode: `1 worker, tp=48, ep=48, dp-attn=true`

### 8k1k

共有 3 个 recipe 组，合计 6 个数据点：

1. low-latency
   - conc: `[4, 8]`
   - prefill: `1 worker, tp=4, ep=1, dp-attn=false`
   - decode: `4 workers, tp=4, ep=1, dp-attn=false`

2. mid-curve
   - conc: `[512, 2048, 4096]`
   - prefill: `6 workers, tp=4, ep=1, dp-attn=false`
   - decode: `1 worker, tp=48, ep=48, dp-attn=true`

3. max-tpt
   - conc: `[2048]`
   - prefill: `10 workers, tp=4, ep=1, dp-attn=false`
   - decode: `1 worker, tp=32, ep=32, dp-attn=true`

### 小结

- FP4 / GB200 / Dynamo-SGLang / DSR1
- 固定长度场景至少能得到 `15` 个 throughput benchmark 点

## 3.5 这个 example 总共能拿到多少点

若把上述 FP8 与 FP4 都算上，仅固定长度 throughput benchmark：

- FP8: 20 点
- FP4: 15 点
- 合计: 35 点

这还不包括：

- eval-only 结果
- run-eval 附加结果
- agentic-coding 场景
- 不同时期 image / recipe 更新带来的新 run

---

## 4. 一条 InferenceX benchmark 配置里能拿到哪些“运行配置”

## 4.1 顶层 master config 可直接看到的字段

从 `nvidia-master.yaml` 可直接读出：

- image
- model
- model-prefix
- runner
- precision
- framework
- multinode
- disagg
- scenario type
- isl
- osl

对于 single-node，还能直接看到：

- tp
- ep
- dp-attn
- conc 或 conc-start / conc-end
- spec-decoding

对于 multi-node/disagg，还能直接看到：

- `prefill.num-worker`
- `prefill.tp`
- `prefill.ep`
- `prefill.dp-attn`
- `decode.num-worker`
- `decode.tp`
- `decode.ep`
- `decode.dp-attn`
- `conc-list`
- `additional-settings`

## 4.2 workflow matrix 展开后可得到的字段

workflow 模板显示实际 matrix 输入还会显式带上：

- exp-name
- max-model-len
- run-eval
- eval-only
- eval-conc
- scenario-type
- duration
- offloading
- total-cpu-dram-gb

因此，从 Actions 的 matrix 视角，可以认为一条 run 至少由这些维度唯一确定。

## 4.3 bench script / recipe 里还能补出的实际启动参数

这一层经常能拿到顶层 config 没写全的内容，例如：

- 具体 server 命令行
- `--tensor-parallel-size`
- `--data-parallel-size`
- `--ep-size`
- `--kv-cache-dtype`
- `--quantization`
- `--mem-fraction-static`
- `--max-running-requests`
- `--cuda-graph-max-bs`
- `--chunked-prefill-size`
- `--max-prefill-tokens`
- `--attention-backend`
- `--moe-runner-backend`
- `--stream-interval`
- `--scheduler-recv-interval`

对于 GB200 `dynamo-sglang`，还会额外牵涉：

- `MODEL_PATH` 的本地重定向
- `SRT_SLURM_MODEL_PREFIX`
- `CONFIG_FILE=recipes/...yaml`
- 通过 `srt-slurm` 提交和收集结果

因此，**顶层 YAML 只能告诉我们“逻辑配置”**，而：

- runner launch script
- external recipe

才更接近“真实部署配置”。

---

## 5. 一次 benchmark run 后能拿到哪些“结果数据”

## 5.1 raw benchmark JSON

`benchmark_serving.py` 直接输出的原始 JSON 至少包含：

- `date`
- `backend`
- `model_id`
- `tokenizer_id`
- `num_prompts`
- `request_rate`
- `burstiness`
- `max_concurrency`
- `duration`
- `benchmark_start_time_unix`
- `benchmark_end_time_unix`
- `completed`
- `total_input_tokens`
- `total_output_tokens`
- `request_throughput`
- `request_goodput`
- `output_throughput`
- `total_token_throughput`

以及 4 类 latency 指标的统计：

- TTFT
- TPOT
- ITL
- E2EL

对每一类，会有：

- `mean_*_ms`
- `median_*_ms`
- `std_*_ms`
- `p90_*_ms`
- `p99_*_ms`
- `p99.9_*_ms`

如果开启详细保存，还能拿到：

- `input_lens`
- `output_lens`
- `ttfts`
- `itls`
- `generated_texts`
- `errors`

这层是后面和 AIC 做“原始指标对齐”时最有价值的结果源。

## 5.2 processed agg JSON

`process_result.py` 会把 raw JSON 转成更适合聚合/入库的 `agg_*.json`。

它会新增或整理出：

- `hw`
- `conc`
- `image`
- `model`
- `infmax_model_prefix`
- `framework`
- `precision`
- `spec_decoding`
- `disagg`
- `isl`
- `osl`

single-node 还会有：

- `tp`
- `ep`
- `dp_attention`
- `tput_per_gpu`
- `output_tput_per_gpu`
- `input_tput_per_gpu`

multinode / disagg 还会有：

- `is_multinode`
- `prefill_tp`
- `prefill_ep`
- `prefill_dp_attention`
- `prefill_num_workers`
- `decode_tp`
- `decode_ep`
- `decode_dp_attention`
- `decode_num_workers`
- `num_prefill_gpu`
- `num_decode_gpu`
- `tput_per_gpu`
- `output_tput_per_gpu`
- `input_tput_per_gpu`

## 5.3 一个非常重要的字段变换

`process_result.py` 会对 raw JSON 做两类转换：

1. 所有 `*_ms` 字段会被改名成不带 `_ms`，并从毫秒转成秒
2. 所有包含 `tpot` 的字段会再进一步转成 `intvty`，即交互性 `tok/s/user`

这意味着：

- raw JSON 里有 `median_tpot_ms`
- agg JSON 里对应的往往不是 `median_tpot`
- 而是 `median_intvty`

所以如果你后面要跟 AIC 的 `TPOT(ms)` 做直接对比：

- 优先使用 raw benchmark JSON
- 不要只看 processed agg JSON

这是后续对比时非常容易踩坑的一点。

## 5.4 GPU telemetry / 功耗数据

`benchmark_lib.sh` 会启动 GPU monitor，采集：

- power.draw
- temperature
- clocks
- utilization

随后 `aggregate_power.py` 会把这些监控结果按 benchmark 时间窗聚合回 `agg_*.json`，补上：

- `avg_power_w`
- `joules_per_output_token`
- `joules_per_total_token`

如果你后面也关心 AIC 的能耗建模，这部分是非常有价值的补充结果。

## 5.5 server logs 和 multinode logs

workflow artifact 还会保留：

- `server.log`
- `multinode_server_logs_*.tar.gz`

这层常用于排查：

- 真正用了什么 runtime 参数
- 启动时 fallback 了什么 kernel/backend
- 是否有失败请求、重试、内存/通信告警

## 5.6 eval 结果

如果该条目被标记为 `run-eval` 或单独跑 `eval-only`，还可以拿到：

- `meta_env.json`
- `results*.json`
- `sample*.jsonl`
- 聚合后的 `agg_eval_*.json`

其指标主要是：

- `score`
- `em_strict`
- `em_flexible`
- `n_eff`
- `task`

这部分不直接用于 throughput 对齐，但可用于判断某些高性能配置是否改变了模型输出质量。

---

## 6. 多节点结果里有哪些信息其实不在 JSON 正文，而在文件名里

这是另一个很值得记住的实现细节。

在 `benchmark-multinode-tmpl.yml` 中，workflow 会从结果文件名里解析：

- 总 GPU 数
- prefill GPU 数
- decode GPU 数

它依赖文件名中的模式，例如：

- `_gpus_`
- `_ctx_`
- `_gen_`

然后再把这些值传给 `process_result.py`。

这说明对于 multi-node/disagg：

- 部分资源拓扑信息不是直接写在 raw JSON body 里
- 而是通过结果文件命名协议补回来的

因此，如果未来下载 artifact 做离线整理：

- 不能只保留 JSON body
- 结果文件名本身也要保留

---

## 7. 对 AIC 对齐最有用的字段清单

若目标是“按 InferenceX 已跑点位复刻 AIC 仿真”，最值得优先抽取的字段是：

- model
- precision
- framework
- image
- runner / hw
- multinode / disagg
- isl
- osl
- conc
- spec-decoding
- prefill worker 数
- decode worker 数
- prefill tp / ep / dp-attn
- decode tp / ep / dp-attn
- num_prefill_gpu
- num_decode_gpu
- raw TTFT / TPOT / E2EL
- total token throughput
- output throughput
- tput_per_gpu

如果能继续下钻，还应补：

- `CONFIG_FILE=...` 指向的 external recipe
- server 实际启动 flags
- GPU power 数据
- server logs

---

## 8. 当前阶段最推荐的“可信信源优先级”

为了后续做 AIC 对齐，我建议按下面优先级取数：

1. **Actions artifact 里的 raw benchmark JSON**
   - 最适合取真实 latency / throughput

2. **master config + workflow matrix**
   - 最适合取逻辑配置维度

3. **runner launch script + external recipe**
   - 最适合补全真实 runtime 参数

4. **processed agg JSON**
   - 适合批量汇总和 quick compare
   - 但要注意 TPOT 被转换成了 interactivity

5. **dashboard 图表**
   - 适合看趋势
   - 不适合做逐字段严谨复现

---

## 9. 对下一步工作的直接建议

对“实机 vs AIC”比对而言，下一步最合适的做法是：

1. 先选一个具体 config key
   - 推荐从 `dsr1-fp8-gb200-dynamo-sglang` 开始

2. 从对应 Actions run artifact 中拿到：
   - raw benchmark JSON
   - agg JSON
   - logs

3. 建立一张“InferenceX -> AIC 参数映射表”
   - 特别注意 `dynamo-sglang`、disagg、多 worker、EP/DP-attn

4. 先逐点复刻，不要先做大规模搜索
   - 先对齐同一组 `isl / osl / conc / prefill / decode` 点位

5. 对比时优先使用 raw JSON 的 latency 指标
   - 避免被 agg JSON 的 `intvty` 变换误导

---

## 10. 一句话总结

InferenceX 的官方“数据”并不只在网页图里，而是分散在配置 YAML、workflow matrix、runner/recipe、Actions artifact、以及最终 dashboard 中。对我们后续做 AIC 对齐最关键的是：先把 `官方配置定义`、`真实启动参数`、`raw benchmark JSON` 这三层对应起来，再做逐点仿真复刻。
