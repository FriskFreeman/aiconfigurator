# PD混合推理实机与AIC仿真对比方案

本文档面向子任务 2：在已经完成单独 Prefill / Decode 实机采集与 AIC 对比的基础上，规划 SGLang 聚合模式下的 PD 混合推理实机采集与仿真对比方案。

这里的“PD 混合”优先指 **SGLang 单引擎 Continuous Batching / mixed chunked prefill**：同一个 scheduler forward step 中同时包含 prefill/extend 请求和已经进入 decode 的 running 请求。它对应 AIC `SGLANGBackend.run_agg()` 的 `agg` 模式，而不是 SGLang server 层 disaggregation `prefill` / `decode` 双实例交互链路。

## 1. 当前任务边界

### 1.1 已完成基础

前序工作已经具备以下能力：

- Prefill 单阶段实机运行：支持 fresh/prefix 长度输入、prefix cache 预热、nsys 导出、MLA/attention 解析、chunked prefill 形状重建。
- Decode 单阶段实机运行：通过 session prefix 预热后进入 decode-only，支持 CUDA Graph、decode-only profile patch、kernel CSV 归档。
- TP 形状语义对齐：通过单卡 rank0 形状覆盖模拟 TP 后本地 head/weight 维度，但不包含真实多卡通信。
- Attention 口径修正：prefill 默认更接近 FA/MHA attention 数据；decode 是否走 FA3/FlashMLA 取决于 SGLang backend 选择，不能只靠 kernel 名称判断。

这些能力可以复用，但 mixed agg 需要新增“并发请求注入”和“只捕获 mixed batch”的能力。

### 1.2 本阶段目标

目标不是构造完整线上 server traffic replay，而是在 `sgl.Engine` 入口下，高保真触发一个或多个真实 SGLang `ForwardMode.MIXED` batch，并从 trace 中提取：

- mixed step 的 prefill 请求组成：请求数、每请求 fresh_len、prefix_len、seq_len_after、chunked_req 状态。
- mixed step 的 decode 请求组成：decode 请求数、每请求当前 KV 长度、当前 output_len / decode step index。
- mixed step 的模块与 kernel 时延：延续前序 `MLA时延拆解.csv` / `DecodeCudaGraphKernel汇总.csv` 的拆解口径。
- 与 AIC `run_agg` 的 `ctx_tokens`、`gen_tokens`、`prefix`、`isl/osl/b` 的映射关系。

## 2. SGLang真实执行机制审查

### 2.1 Scheduler 状态模型

SGLang scheduler 维护两个核心队列：

- `waiting_queue`：等待进入 prefill/extend 的新请求。
- `running_batch`：已经完成 prefill、正在 decode 的请求集合。

源码位置：

- `/home/ai_lab/fjw/miniforge3/envs/ljc01/lib/python3.12/site-packages/sglang/srt/managers/scheduler.py`
- `init_running_status()` 初始化 `waiting_queue` 和 `running_batch`，见约 `scheduler.py:845-859`。
- `event_loop_normal()` 每轮执行 `recv_requests()`、`process_input_requests()`、`get_next_batch_to_run()`、`run_batch()`、`process_batch_result()`，见约 `scheduler.py:1303-1326`。

### 2.2 Mixed batch 的形成条件

SGLang mixed chunked prefill 的关键逻辑在 `get_new_batch_prefill()` / `_get_new_batch_prefill_raw()` 中：

- `server_args.chunked_prefill_size` 不为 `None`。
- `server_args.enable_mixed_chunk=True`。
- scheduler 内部 `is_mixed_chunk = chunked_prefill_size is not None and enable_mixed_chunk`。
- 当前 `running_batch` 非空，即已有 decode 请求正在运行。
- 新 prefill batch 无 logprob、无 input_embeds 等阻断条件。

源码证据：

- `Scheduler.init_chunked_prefill()` 设置 `chunked_prefill_size`、`chunked_req`、`is_mixed_chunk`，见约 `scheduler.py:862-884`。
- `_get_new_batch_prefill_raw()` 创建 `PrefillAdder` 时会传入 `running_bs if self.is_mixed_chunk else 0`，见约 `scheduler.py:2377-2387`。
- mixed 条件满足时执行 `running_batch.prepare_for_decode()`，然后 `new_batch.mix_with_running(self.running_batch)`，见约 `scheduler.py:2525-2540`。
- `ScheduleBatch.mix_with_running()` 将 `forward_mode` 改为 `ForwardMode.MIXED`，把 decode 请求拼到 prefill batch 后面，并把 decode 请求的 `extend_lens` 记为 1，见约 `schedule_batch.py:1873-1902`。
- `ForwardMode.MIXED` 在 SGLang 中明确表示“Contains both EXTEND and DECODE when doing chunked prefill”，且 `is_extend()` / `is_decode()` 都会返回真，见约 `forward_batch_info.py:81-138`。

### 2.3 为什么现有单阶段 runner 不够

现有 `.self/task-localbench/formal-prefill-stage1/run_formal_prefill_stage1.py` 和 container 内 runner 已经支持 session prefix 预热和单次正式 generate，但正式请求是同步调用 `engine.generate()`。同步调用会等待该请求完成，较难让一批 decode 请求持续停留在 `running_batch`，同时再把新的 prefill 请求送入 `waiting_queue`。

因此 agg/mixed 需要新增调度方式：

- 先异步启动一组长输出 decode 请求，让它们完成首轮 prefill 后进入 running decode。
- 在这些 decode 请求尚未完成时，注入 prefill 请求。
- 开启 `enable_mixed_chunk` 和固定 `chunked_prefill_size`，让 scheduler 在下一轮 prefill 中合并 running decode。
- profile patch 只在 `ForwardMode.MIXED` 且满足目标 shape 时启动/停止 profiler。

## 3. AIC `run_agg` 仿真口径审查

### 3.1 入口变量

`SGLANGBackend.run_agg()` 输入来自：

- `runtime_config.isl`：总输入长度。
- `runtime_config.osl`：输出长度。
- `runtime_config.prefix`：已命中 KV cache 的前文长度。
- `runtime_config.batch_size`：聚合并发规模 `b`。
- `kwargs["ctx_tokens"]`：一个 mixed step 中计划处理的 context/prefill token 数。

源码位置：

- `src/aiconfigurator/sdk/backends/sglang_backend.py:35-50`

### 3.2 调度数学

`run_agg()` 先计算：

```text
steps_to_finish_ctx = ceil(isl * b / ctx_tokens)
balance_score = isl * b / ctx_tokens / osl
```

当 `b > 1`：

- 如果 `steps_to_finish_ctx >= osl`：
  - `num_mix_steps = steps_to_finish_ctx`
  - `num_mix_ctx_tokens = ctx_tokens`
  - `num_mix_gen_tokens = max(1, b // (steps_to_finish_ctx / osl))`
  - `num_genonly_steps = 0`
- 如果 `steps_to_finish_ctx < osl`：
  - `num_mix_steps = steps_to_finish_ctx`
  - `num_mix_ctx_tokens = ctx_tokens`
  - `num_mix_gen_tokens = b - ceil(ctx_tokens / isl)`
  - `num_genonly_steps = osl - num_mix_steps`
  - `num_genonly_tokens = b`
  - `num_mix_steps_for_tpot_calc = max(1, num_mix_steps - 3)`

源码位置：

- `src/aiconfigurator/sdk/backends/sglang_backend.py:56-88`

### 3.3 Python 路径的 mixed step 估计

当前 Python `run_agg` 路径不是直接复现 SGLang scheduler，而是三段近似相加：

1. 非 attention 部分：调用 `run_static(..., mode="static_ctx")`，使用 `batch_size=1`、`isl=ctx_tokens + gen_tokens`，再排除 `context_attention`。
2. context attention：调用 `run_static(..., mode="static_ctx")`，使用 `batch_size=ceil(ctx_tokens / isl)`、`isl=isl`、`prefix=prefix`，再除以 `ceil(isl / ctx_tokens)`。
3. generation attention：若 `gen_tokens > 0`，调用 `run_static(..., mode="static_gen")`，使用 `batch_size=gen_tokens`、`isl=isl + osl // 2`、`osl=2`。

源码位置：

- `src/aiconfigurator/sdk/backends/sglang_backend.py:94-220`

这意味着 Python `run_agg` 的 mixed step 口径是“把真实 mixed batch 拆成若干静态查表项后相加”，而不是逐层模拟真实 mixed forward 中 prefill/decode token 的具体拼接顺序。

### 3.4 Rust engine step 路径

如果 `runtime_config.engine_step_backend == "rust"`，`run_agg()` 会调用 `estimate_mixed_step_latency_with_rust()`。这个路径更接近 scheduler-level FPM 输入，因为直接传：

```json
{
  "scheduled_requests": {
    "num_prefill_requests": ...,
    "sum_prefill_tokens": ...,
    "sum_prefill_kv_tokens": ...,
    "num_decode_requests": ...,
    "sum_decode_kv_tokens": ...
  }
}
```

源码位置：

- `src/aiconfigurator/sdk/rust_engine_step.py:126-166`

这为后续对比提供了两个仿真口径：

- Python `run_agg`：保持当前 SDK 默认语义，便于解释已有 AIC 结果。
- Rust engine step：更适合作为 mixed scheduler step 的结构化对照，尤其当实机能准确导出 `scheduled_requests` 时。

## 4. 实机用例构造方案

### 4.1 用例变量

与 prompt 中的任务 2 对齐，实机 case 至少需要四组变量：

- `prefill_req_count`
- `prefill_fresh_len`
- `prefill_prefix_len`
- `decode_req_count`
- `decode_prefix_len`
- `decode_active_output_pos`：触发 mixed 时 decode 请求已生成到第几个 token，默认对齐 AIC 的平均位置 `isl + osl // 2` 附近时另开进阶方案。

第一阶段建议固定各请求等长，后续再扩展为列表：

```text
prefill fresh: [F] * P
prefill prefix: [Pfx] * P
decode prefix: [Dfx] * D
decode max_new_tokens: 足够长，确保注入 prefill 时仍处于 running_batch
```

### 4.2 Engine 入口调度策略

建议新增独立 runner，而不是继续堆叠现有 prefill runner：

- 新建 `.self/task-localbench/formal-agg-stage2/run_formal_agg_stage2.py`
- 新建 `.self/task-localbench/formal-agg-stage2/run_agg_inside_container.py`

container 内执行流程：

1. 初始化 dummy DeepSeek-V3 Engine，沿用已有 local_model、TP shape-only、layerwise NVTX、nsys 逻辑。
2. 设置 `enable_mixed_chunk=True`、`chunked_prefill_size`、`max_prefill_tokens`，并关闭动态 chunking。
3. warmup 阶段执行同类请求，但不 profile。
4. 正式阶段先启动 decode 请求：
   - 使用 `async_generate(..., stream=True)` 或多线程 stream iterator。
   - 每个 decode 请求用 `input_ids=decode_prefix_tokens`，`max_new_tokens` 设为足够大。
   - 消费第一批输出，确保请求已经离开 prefill 并进入 decode running 状态。
5. 注入 prefill 请求：
   - 若需要 prefill prefix 命中，则先用 session 预热 prefix，再以 fresh tokens 继续请求。
   - 若不需要 prefix，则直接提交 fresh tokens。
6. profile patch 监听 `Scheduler.run_batch(batch)`：
   - `batch.forward_mode == ForwardMode.MIXED`
   - `batch.decoding_reqs` 非空
   - `sum(batch.extend_lens for prefill part)` 与目标 `ctx_tokens` 匹配
   - decode 请求数与目标 `gen_tokens` 匹配
7. 捕获一个或多个 mixed batch 后停止 profiler，等待请求清理并 close sessions。

### 4.3 为什么要捕获 `ForwardMode.MIXED`

前序 decode-only 曾通过 `ForwardMode.DECODE` 捕获 decode batch。agg 阶段应该捕获 `ForwardMode.MIXED`，因为它是 SGLang 源码中真实表达“prefill + decode 同 step”的模式。

不要用以下替代口径作为主口径：

- 只采 prefill batch：会丢掉 decode 混入。
- 只采 decode batch：会退化为任务 1 的 decode-only。
- 用 kernel 名反推 mixed：不可靠，前序已经出现 `nvjet_tst` 误分类。

### 4.4 mixed batch 元数据必须落盘

建议在 NVTX payload 和独立 JSON 中同时保存：

```json
{
  "kind": "mixed_batch",
  "forward_ct": 123,
  "forward_mode": "MIXED",
  "prefill": {
    "req_count": 2,
    "prefix_lens": [1024, 1024],
    "fresh_lens": [2048, 2048],
    "seq_lens_after": [3072, 3072],
    "sum_fresh_tokens": 4096,
    "sum_prefill_kv_tokens": 2048
  },
  "decode": {
    "req_count": 8,
    "kv_lens": [4097, 4097, "..."],
    "output_lens": [1, 1, "..."],
    "sum_decode_kv_tokens": 32776
  },
  "scheduler": {
    "chunked_prefill_size": 8192,
    "max_prefill_tokens": 16384,
    "enable_mixed_chunk": true,
    "chunked_req_rid": null
  }
}
```

这里的 `prefill.sum_fresh_tokens` 对齐 AIC `ctx_tokens`，`decode.req_count` 对齐 AIC `gen_tokens`，`sum_prefill_kv_tokens` 和 `sum_decode_kv_tokens` 则对齐 Rust engine step 的 scheduled_requests。

## 5. AIC对比方案

### 5.1 第一主口径：step-level latency

实机侧：

- 取一个 `ForwardMode.MIXED` batch 的模块端到端时间或关键模块/kernels 时间。
- 对全模型 step，可取 profiler range 内 GPU kernel span 或模型 forward range。
- 对 MLA/attention，可沿用已有 module/kernel 拆解 CSV，一行表示 `chunk_instance x layer`。

AIC 侧：

- 使用同一 `isl/osl/prefix/b/ctx_tokens` 调 `SGLANGBackend.run_agg()`。
- 提取 `summary.get_per_ops_data()["mix_step"]` 中：
  - non-attention ops
  - `context_attention (scaled)`
  - `generation_attention`
  - `mix_step_latency_ms`
- 另跑 `engine_step_backend="rust"`，提取 `rust_engine_step_mixed` 作为 scheduler-level 对照。

### 5.2 变量映射建议

对于第一批等长 case：

```text
prefill_req_count = P
prefill_fresh_len = F
prefill_prefix_len = Pfx
decode_req_count = D
decode_prefix_len = Dfx
```

实机 mixed batch：

```text
ctx_tokens_real = P * F
gen_tokens_real = D
sum_prefill_kv_tokens_real = P * Pfx
sum_decode_kv_tokens_real = sum(decode current kv len)
```

AIC 建议配置：

```text
runtime_config.isl = Pfx + F
runtime_config.prefix = Pfx
ctx_tokens = P * F
batch_size b = P + D   # 初始近似，后续可按 run_agg 公式反推 b
osl = decode max_new_tokens 或目标服务输出长度
```

需要注意：`run_agg` 内部 `num_mix_gen_tokens` 不是外部直接传入，而由 `b/isl/osl/ctx_tokens` 推导。因此如果要让 AIC 的 `num_mix_gen_tokens == D`，需要满足：

```text
D = b - ceil(ctx_tokens / isl)           # steps_to_finish_ctx < osl 分支
```

也就是说，如果实机希望一个 mixed step 中有 `P` 个 prefill 请求、`D` 个 decode 请求，并且 `ceil(ctx_tokens / isl) == P`，则可以先取：

```text
b = P + D
```

若 prefill fresh 小于 total `isl` 很多，`ceil(ctx_tokens / isl)` 可能小于实际 prefill 请求数，这会暴露 AIC `run_agg` 的聚合近似误差。文档和结果表必须同时记录“真实 prefill_req_count”和“AIC 推导 num_ctx_reqs”。

### 5.3 第二口径：attention kernel 对齐

由于前序已经确认：

- prefill attention 常与 FA/MHA 路径对齐。
- decode attention 可能是 FA3，也可能是 FlashMLA，取决于 `decode_attention_backend` 和 SGLang 实际选择。
- `nvjet_tst` 等 kernel 不能直接等同于 FlashMLA。

mixed 对比时建议把 attention 拆成三列：

- `prefill_attention_kernel_ms`
- `decode_attention_kernel_ms`
- `mixed_attention_total_ms`

实机解析层面需要依赖 `ForwardMode.MIXED` 的 batch 元数据和 module/kernels 时间区间，而不是只靠 kernel 名。AIC 层面分别取 `context_attention (scaled)` 与 `generation_attention`。

## 6. 推荐实施阶段

### 阶段 A：只验证 mixed batch 可控触发

目标：不做正式归档，只证明 Engine 入口能稳定捕获 `ForwardMode.MIXED`。

建议 case：

```text
num_layers = 3
chunked_prefill_size = 8192
max_prefill_tokens = 16384
enable_mixed_chunk = true
decode_req_count = 2
decode_prefix_len = 512
decode_max_new_tokens = 32
prefill_req_count = 1
prefill_fresh_len = 1024
prefill_prefix_len = 0
profile_num_mixed_steps = 1
```

验收：

- `mixed_batch_meta.jsonl` 至少有一条 `forward_mode=MIXED`。
- `prefill.sum_fresh_tokens=1024`。
- `decode.req_count=2`。
- nsys 可导出，并且已有解析脚本能看到 mixed batch 中的 MLA/attention 模块。

### 阶段 B：与 AIC run_agg 做 step-level 对照

目标：固定一个或两个规整 case，生成：

- `AIC_run_agg_result.json`
- `AIC_run_agg_per_ops.json`
- `mixed_batch_meta.jsonl`
- `MLA时延拆解.csv`
- `AggMixedStep对比.csv`

建议 case：

```text
case_1: P=1, F=1024, Pfx=0, D=2, Dfx=512
case_2: P=2, F=2048, Pfx=1024, D=4, Dfx=2048
case_3: P=1, F=8192, Pfx=4096, D=4, Dfx=4096
```

### 阶段 C：覆盖 prompt 中任务 2 的子情景

沿用任务 1 的 prefill 三类情况，在每类上加 decode 混入：

- 全 fresh 小到中请求 + decode。
- 含 prefix 小到中请求 + decode。
- 单个 chunk 大小 fresh 请求 + 各种 prefix + decode。

每类先取少量代表点，确认 mixed 触发和解析稳定后再扩展。

## 7. 代码改造建议

### 7.1 新增 runner 参数

建议新增而不是直接复用 prefill runner：

```text
--prefill-req-count
--prefill-fresh-len / --prefill-fresh-lens-csv
--prefill-prefix-len / --prefill-prefix-lens-csv
--decode-req-count
--decode-prefix-len / --decode-prefix-lens-csv
--decode-max-new-tokens
--enable-mixed-chunk
--chunked-prefill-size
--max-prefill-tokens
--profile-mixed-stage
--profile-num-mixed-steps
--target-mixed-ctx-tokens
--target-mixed-gen-tokens
```

### 7.2 container 内 patch

新增 `install_mixed_stage_profile_patch()`：

- patch `Scheduler._profile_batch_predicate`，避免默认 profile_by_stage 抢先启动。
- patch `Scheduler.run_batch`：
  - 判断 `batch.forward_mode == ForwardMode.MIXED`。
  - 构造 mixed payload。
  - 写 `mixed_batch_meta.jsonl`。
  - 若 payload 满足目标 shape，则 `start_profile(forward_mode)`。
  - 运行原始 `run_batch()`。
  - 捕获目标步数后 `stop_profile(stage=forward_mode)`。

可以沿用已有 decode-only patch 的结构，但 predicate 从 `is_decode()` 换成严格 `ForwardMode.MIXED`，避免普通 decode 被误捕获。

### 7.3 AIC侧辅助脚本

建议在同目录新增：

```text
run_aic_agg_compare.py
summarize_agg_stage2_runs.py
```

`run_aic_agg_compare.py` 输入一个实机 run_dir，读取 `run_meta.json` 与 `mixed_batch_meta.jsonl`，自动构造两套 AIC 查询：

- `engine_step_backend=None` 或 `"python"`：当前 SDK 默认口径。
- `engine_step_backend="rust"`：scheduled_requests 口径。

输出 `AIC_run_agg_compare.json` 和 CSV。

## 8. 风险与误差来源

### 8.1 实机触发风险

- 如果 `enable_mixed_chunk` 没有显式打开，即使有 running decode，也不会形成 `ForwardMode.MIXED`。
- 如果 decode 请求太短或 prefill 注入太晚，decode 请求可能已经结束，`running_batch` 为空。
- 如果 prefill 请求太大，可能产生多轮 chunk，目标 mixed step 只覆盖其中一轮，需要在 payload 中区分 chunk instance。
- 如果开启 CUDA Graph，mixed batch 的 shape 组合可能没有被 capture，首轮可能走 eager 或触发额外 warmup。

### 8.2 仿真口径风险

- Python `run_agg` 对 non-attention 的 `batch_size=1, isl=ctx_tokens+gen_tokens` 是聚合近似，和真实 mixed batch 中 prefill/decode token 拼接并不完全相同。
- `num_mix_gen_tokens` 由 `b/isl/osl/ctx_tokens` 推导，未必等于实机中人为构造的 `decode_req_count`。
- `generation_attention` 使用 `isl + osl // 2` 作为平均 decode KV 长度，但实机 mixed step 中 decode 请求的当前 KV 长度可能更接近 `decode_prefix_len + consumed_output_tokens`。
- 含 prefix 的 prefill 在 AIC 中通过 `prefix` 和 prefix correction 处理，真实 SGLang 还会受 radix cache 命中、MHA/MLA backend、chunked_kv 等影响。

### 8.3 对比解释原则

因此本阶段结果应分层解释：

- scheduler-level：真实 `scheduled_requests` 与 Rust engine step 是否接近。
- SDK Python run_agg：当前产品口径下 TTFT/TPOT/mix_step 是否合理。
- operator-level：prefill attention、decode attention、GEMM/MoE 等分项误差来源。

不要把单个 mixed step 的实机时延直接等同于 `run_agg` 的最终 `ttft/tpot`，因为 `run_agg` 还包含 mix/genonly 步数、TTFT correction、吞吐换算和内存约束判断。

## 9. 推荐结论

最稳妥的实现路线是：

1. 新建 agg stage2 runner，通过 `async_generate(stream=True)` 保持 decode 请求活跃。
2. 显式开启 `enable_mixed_chunk=True` 和固定 `chunked_prefill_size`，让 SGLang 走真实 `ForwardMode.MIXED`。
3. 用 `ForwardMode.MIXED` patch 捕获目标 mixed step，并落盘完整 scheduler 元数据。
4. 先做 step-level 对照，再扩展到任务 2 的完整场景矩阵。
5. AIC 对比同时保留 Python `run_agg` 和 Rust scheduled_requests 两个口径，避免把当前 SDK 的聚合近似误读为真实 scheduler 的逐 step 复刻。

这个方案能最大程度复用任务 1 的 trace 解析和 CSV 归档能力，同时把新的不确定性控制在 scheduler mixed batch 构造和 AIC `run_agg` 口径映射这两个清晰边界内。
