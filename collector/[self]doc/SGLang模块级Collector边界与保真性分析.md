# SGLang模块级Collector边界与保真性分析

## 结论先行

- `collect_mla_module.py` 这类以 `self_attn(...)` 结束为边界的采集，**边界选择是合理的**，因为真实 SGLang 的 attention module 返回前就已经把 `o_proj` 及其可能的 TP all-reduce 包含进去了。
- 但要注意：当前多数 SGLang collector **并没有把真实多卡通信都测进去**。很多地方是 `tp_size=1`，或用单进程 TP patch 只测 rank-0 计算，所以它们更像“模块计算/调度 makespan”，不是完整多卡端到端。
- 因此，`module` 级 collector 仍然有意义，但应理解为“模块级有效时间”，不是“纯 Python 时间”，也不是“完整 GPU 执行闭包”。

## 源码依据

本结论主要基于以下源码：

- 本仓库 collector：
  - [collector/sglang/collect_mla_module.py](/home/ai_lab/ljc/scale-up-sim/aiconfigurator/collector/sglang/collect_mla_module.py)
  - [collector/sglang/collect_mhc_module.py](/home/ai_lab/ljc/scale-up-sim/aiconfigurator/collector/sglang/collect_mhc_module.py)
  - [collector/sglang/deepseekv4_sparse_modules.py](/home/ai_lab/ljc/scale-up-sim/aiconfigurator/collector/sglang/deepseekv4_sparse_modules.py)
  - [collector/sglang/collect_dsv4_flash_attn.py](/home/ai_lab/ljc/scale-up-sim/aiconfigurator/collector/sglang/collect_dsv4_flash_attn.py)
- SGLang 0.5.12 docker 源码：
  - `sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`
  - `sglang/srt/models/deepseek_v4.py`
  - `sglang/srt/layers/linear.py`

其中最关键的两个事实是：

- `forward_mla.py` 与 `deepseek_v4.py` 都是在 module 内部做完 `o_proj` 后再返回。
- `RowParallelLinear.forward()` 在 `reduce_results=True` 且 `tp_size > 1` 时，会在 `forward()` 里执行 all-reduce，然后才把结果返回给上层 module。

## 1. 为什么 `self_attn` 作为 MLA module 边界是合理的

在 0.5.12 的 SGLang 源码里，DeepSeek MLA 的 attention forward 逻辑是：

- `q/k/v` 准备
- `attn_backend.forward(...)`
- `o_proj(...)`
- return

其中 `o_proj` 对应 `RowParallelLinear`，在 `tp_size > 1` 且 `reduce_results=True` 时，会在 `forward()` 内部直接触发 `all_reduce`，再返回结果。

也就是说，真实语义上：

- 该 module 的“有效结束点”就是 `self_attn.forward()` 返回。
- 如果是多卡 TP，尾部通信本来就属于 module 内部，不应该单独切出去。

所以 `collect_mla_module.py` 把 `layer.self_attn` 整块作为边界，是对的。

## 2. 但当前 collector 的“保真度”并不等于真实多卡推理

### `collect_mla_module.py`

- 走的是 `ServerArgs -> ModelRunner -> ScheduleBatch -> ForwardBatch -> self_attn.forward`
- 计时方式是 `torch.cuda.Event` / `benchmark_with_power()`
- 但脚本里 `tp_size` 实际固定为 `1`

所以它采到的是：

- 真 module forward 路径
- 真实 backend kernel
- 但**没有真实 TP all-reduce 通信**

这意味着它的边界是合理的，但数据更偏向“单卡/单 rank 模块执行时间”。

### `collect_dsv4_flash_attn.py`

这个脚本更明确：

- `_tp_load_model_patch(tp_size)` 只是在**单进程里伪装 TP=N**
- 它测的是“TP=N 部署里单 rank 的 attention 计算成本”
- forward-time 的 collective 在 `world_size == 1` 下会短路

所以它本质上是：

- **TP 形状对齐**
- **但不是 TP 通信保真**

它适合补模块计算，不适合直接当作真实多卡 wall time。

### `collect_mhc_module.py`

- 只测 `hc_pre` / `hc_post`
- `tp_size=1`
- 本身就是子模块微基准

因此它更像“模块内部子步骤采样”，不承担真实多卡通信建模职责。

### `deepseekv4_sparse_modules.py`

- 这是 kernel-level collector
- 采的是 `paged_mqa_logits` / `hca_attn` 等稀疏子核
- 本来就不是 module-level

所以它不适合用“module 边界是否包含通信”来评价。

## 3. 为什么“只采 Python module 事件”不够

Nsight 时间线里可以看到：

- Python/NVTX module 只是 host 侧发起和标记
- 真正的 CUDA runtime launch 与 GPU kernel 可能延后到 module 结束后才完成
- 上一个 module 的 kernel 还在跑时，后一个 module 的 host 逻辑已经开始了

所以：

- 单独看 Python module 事件，不足以反映真实算子执行
- 但“module 边界 + CUDA event + GPU trace”合起来，就能得到更完整的 module 画像

## 4. 当前最合理的理解方式

把模块级 collector 分成三类看：

1. **边界正确但只测单 rank**
   - `collect_mla_module.py`
   - `collect_mhc_module.py`

2. **边界正确且以 rank-0 视角模拟 TP**
   - `collect_dsv4_flash_attn.py`

3. **不是 module-level，只是补底层 kernel**
   - `collect_mla.py`
   - `deepseekv4_sparse_modules.py`

## 5. 结论

- 对 `MLA module` 来说，collector 以最后一个算子结束作为边界，**在语义上是合理的**。
- 但当前 SGLang collector 多数**没有真实覆盖多卡通信尾巴**，因此它们更适合作为：
  - 模块级 compute/makespan 数据
  - 或模块级补充采样
- 若目标是“完整保真地反映真实多卡推理”，还需要把：
  - Python module 边界
  - CUDA runtime 下发
  - GPU kernel
  - TP/DP 通信
 统一采下来再关联。
