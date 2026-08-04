# GEMM Collector：SGLang 与 TensorRT-LLM 差异分析

> 分析日期：2026-07-27
>
> 分析对象：当前工作区的 `collector/sglang/collect_gemm.py`、`collector/trtllm/collect_gemm.py`、公共 case 生成器，以及 `src/aiconfigurator/systems/data/**/{sglang,trtllm}/**/gemm_perf.txt`。
>
> 本文只讨论 GEMM；MoE、MLA BMM 等虽然也执行矩阵乘，但不在本轮范围内。

## 1. 结论先行

两边都把结果写成 `gemm_dtype,m,n,k,latency`，但同一个 `gemm_dtype` **不一定表示相同的量化粒度、scale 布局或后端 kernel**，不能仅凭 dtype 名称把延迟当作同一算子的 A/B 测试。

核心差异如下：

1. SGLang collector 直接拼装具体算子链：BF16 `F.linear`、SGL Kernel FP8、DeepGEMM block-FP8、FlashInfer CUTLASS NVFP4。TRT-LLM collector 则统一构造 `_torch.modules.linear.Linear + QuantConfig`，实际 kernel 由 TRT-LLM 运行时、硬件、版本和 shape 再分派。
2. `fp8` 的语义差异最大。SGLang 明确测量“BF16 激活 per-token 动态量化 + FP8 GEMM”，权重 scale 为 `[N]`；TRT-LLM 使用 `QuantAlgo.FP8`，collector 提供的是单元素 `weight_scale`，量化与 kernel 选择封装在 `Linear.forward` 内。两者不是严格相同的 scale scheme。
3. `fp8_block` 两边都以 128 为主要分组尺度，但 SGLang 明确走 DeepGEMM，激活量化为 per-token/per-128-group；TRT-LLM 在 SM90 记录为 `torch_flow`，源码注释称使用 CUTLASS/FP32 scale，在 SM100+ 才将日志标为 `deepgemm`。相同 dtype 标签跨架构也可能不是同一后端。
4. `nvfp4` 两边都把权重量化放在计时外、把激活量化留在 forward 内，但 SGLang 直接调用 FlashInfer `mm_fp4(..., backend="cutlass")`，TRT-LLM 走 `QuantAlgo.NVFP4` 和版本相关的 scale 反交织/装载路径。
5. 当前公共网格有 74 个 M、22 个 N、22 个 K；完整 dtype 为 35,742 个 shape，要求 `N,K>=128` 的 dtype 为 29,526 个 shape。TRT-LLM 还额外裁剪 32-bit 索引边界、超大 block-FP8 权重和特定 SM120 FP8 崩溃 shape。
6. 仓库已有 31 份 SGLang/TRT-LLM GEMM CSV，但混合了新旧 case 网格和旧 dtype：旧数据可见 `int8_wo`、`int4_wo`、`sq`，当前生成器已经不再产生它们。文件目录中的框架版本不能单独证明它是由当前脚本生成的。
7. 当前 CSV 的 `kernel_source` 不能用于还原实际 CUDA kernel：SGLang 所有 dtype 都写 `sglang`；TRT-LLM 只把 SM100+ 的 `fp8_block` 写成 `deepgemm`，其他都写 `torch_flow`。精确 kernel 名必须以 Nsight Systems/Compute 或 CUPTI trace 为准。
8. 延迟结果呈明显的 dtype/架构相关差异，而不是某框架全面更快。相同 shape 的全网格中位数显示：H100/H200 的 BF16 与 block-FP8 基本接近；SGLang 的普通 FP8 普遍更快；Blackwell 上 TRT-LLM block-FP8 更快，而 SGLang 的普通 FP8/NVFP4 多数更快。由于计时算子链和量化语义不同，这些数字应解释为“collector 路径差异”，不是纯 GEMM kernel 排名。

## 2. 入口、输出与可比边界

| 项目 | SGLang | TensorRT-LLM |
|---|---|---|
| Registry | `collector.sglang.collect_gemm` | `collector.trtllm.collect_gemm` |
| case 函数 | `get_gemm_test_cases()` | `get_gemm_test_cases()` |
| 执行函数 | `run_gemm(gemm_type,batch_size,N,K)` | `run_gemm(gemm_type,m,n,k)` |
| 输出文件 | `gemm_perf.txt` | `gemm_perf.txt` |
| 数据主键 | `gemm_dtype,m,n,k` | `gemm_dtype,m,n,k` |
| 框架兼容声明 | `sglang>=0.5.10rc0` | 当前文件无 `__compat__` |
| kernel 日志粒度 | 固定 `kernel_source=sglang` | `fp8_block && SM>=100` 为 `deepgemm`，其余为 `torch_flow` |
| 调用层次 | 直接调用具体 kernel/wrapper | 通过 TRT-LLM `Linear` 运行时分派 |

共同输出字段只有 shape、dtype 标签和总 latency。CSV 没有记录以下关键变量：

- 实际 CUDA kernel symbol、CUTLASS tile/cluster 配置、cuBLASLt algo ID；
- activation/weight scale 粒度和编码；
- 动态量化是否单独计时、是否使用 CUDA Graph；
- `outside_loop_count`、实际 op 数和缓存策略；
- JIT/autotune cache 状态；
- 采集 commit、容器镜像、CUDA/PyTorch/FlashInfer/sgl-kernel 版本。

因此，`gemm_dtype,m,n,k` 相同只表示查询键相同，不自动表示被测工作相同。

## 3. 不同精度的执行链与后端

### 3.1 总表

| `gemm_dtype` | SGLang collector 实际链路 | TRT-LLM collector 实际链路 | 主要差异与可比性 |
|---|---|---|---|
| `bfloat16` | BF16 A/B -> `torch.nn.functional.linear` | BF16 x/weight -> TRT-LLM `Linear.forward`，无 QuantConfig | 数学 shape 最接近；底层仍可能因框架版本走不同 cuBLAS/cuBLASLt/CUTLASS 路径。 |
| `fp8` | BF16 A -> `sgl_per_token_quant_fp8` -> `fp8_scaled_mm`；B 预转 E4M3，B scale 为 `[N]` | BF16 x -> `Linear(QuantAlgo.FP8).forward`；B 预转 E4M3，collector 提供标量 `weight_scale` | 都包含 forward 侧激活处理，但 scale scheme 不同，不能视为同一 FP8 kernel。 |
| `fp8_block` | BF16 A -> per-token/per-128-group FP8 quant -> `gemm_nt_f8f8bf16` DeepGEMM；B 为 E4M3 + 128x128 scale | `Linear(QuantAlgo.FP8_BLOCK_SCALES, group_size=128)`；源码注释：SM90 CUTLASS/FP32 scale，SM100 TRTLLM/DeepGEMM/UE8M0 | 概念最接近，但 SM90/SM100 后端、scale 编码及权重准备仍有差异。 |
| `nvfp4` | BF16 A 动态 `flashinfer.fp4_quantize` -> `flashinfer.mm_fp4(... backend="cutlass")`；B 预量化并 shuffle | `Linear(QuantAlgo.NVFP4)`；`torch.ops.trtllm.fp4_quantize(..., block=16)`，版本相关反交织后 load | 都含激活动态量化、排除权重预处理；具体 layout、global scale 和 Linear epilogue 不同。 |
| `int8_wo` | assert 白名单和一个未使用的 `per_token_quant_int8()` 仍存在，但 `create_gemm()` 没有执行分支，当前 case 也不生成 | 当前 case 不生成，当前 `_build_weights()` 也无专门分支 | 属于残留接口，不是当前可采能力。历史 CSV 中出现不代表当前脚本可复现。 |
| `int4_wo` / `sq` | 当前脚本无路径 | 当前脚本无路径 | 只存在于部分旧 TRT-LLM 数据；分别曾使用 WeightOnlyQuantMatmul 和 SmoothQuant 插件。 |

### 3.2 BF16

SGLang 使用：

```text
A[M,K] BF16 + B[N,K] BF16
  -> F.linear(A, B, bias=None)
  -> output[M,N] BF16
```

TRT-LLM 使用：

```text
x[M,K] BF16 + Linear(K,N,dtype=BF16,quant_config=None)
  -> load_weights
  -> dry-run
  -> Linear.forward(x)
```

BF16 是两边最适合直接比较的 dtype，但仍要注意：SGLang 的 `F.linear` 是 PyTorch ATen 入口，TRT-LLM `Linear` 可能有自己的选择与封装。源码只能支持“前端调用路径”的判断，不能断言每个 shape 都落到同一个 cuBLASLt kernel。

### 3.3 普通 FP8

SGLang：

- 激活源数据为 BF16；计时闭包中调用 `sgl_per_token_quant_fp8`，输出 E4M3 和 `[M,1]` FP32 scale。
- 权重在计时前转为 E4M3，并转置为 GEMM 所需布局；`scale_b` 为 `[N]` FP32。
- 计时包含 activation dynamic quantization 和 `fp8_scaled_mm`。
- `scale_b` 使用随机值，并未与随机权重严格配对；该 collector 测性能，不验证数值正确性。

TRT-LLM：

- `QuantConfig(QuantAlgo.FP8)`；输入仍以 BF16 传入 `Linear.forward`。
- 权重在计时前从 BF16 转为 E4M3；`weight_scale` 是单元素 FP32 tensor。
- activation quantization、kernel 选择和 epilogue 被封装在 `Linear.forward`，仅凭 collector 源码不能进一步拆分。
- 日志统一写 `torch_flow`，这不是实际 CUDA kernel 名。

所以两份 `fp8` 数据首先比较的是“框架 FP8 Linear 路径”，而不是相同 per-token/per-channel FP8 GEMM 的纯 kernel 对比。

### 3.4 Block FP8

SGLang：

- 仅 SM90 <= SM < SM110 生成；SM100/103 也在此范围内。
- 激活在计时闭包内使用 `sglang_per_token_group_quant_fp8(group_size=128)`，并要求 column-major、TMA aligned scale；scale 是否为 UE8M0 由 `DEEPGEMM_SCALE_UE8M0` 决定。
- 权重为 E4M3 `[N,K]`，scale shape 为 `[ceil(N/128),ceil(K/128)]`。
- GEMM 明确调用 SGLang DeepGEMM wrapper `gemm_nt_f8f8bf16`。
- 通过 `SGLANG_JIT_DEEPGEMM_PRECOMPILE=0` 避免为相同 N/K 预编译所有 M。

TRT-LLM：

- 条件写成 `SM > 86`，因此 SM89 也会生成；这与代码注释“SM90/SM100 support”不完全一致，应在 L40S 上实测确认，而不是只信注释。
- `QuantAlgo.FP8_BLOCK_SCALES, group_size=128`。
- 权重 scale 由每个 128x128 block 的绝对最大值除以 448 得到，类型为 FP32；权重本体则直接 cast 到 E4M3。这里没有数值正确性检查，不能把随机权重/scale 构造当成完整离线量化实现。
- `post_load_weights()` 在 API 存在时执行。
- 源码注释称 SM90 使用 CUTLASS + FP32 scale，SM100 使用 TRTLLM/DeepGEMM + UE8M0；CSV 只有 SM100+ 被标成 `deepgemm`。

这里还存在 SM120 分歧：SGLang 明确不生成 `fp8_block`，理由是没有对应 DeepGEMM recipe；TRT-LLM 当前仍生成并标记 `deepgemm`。RTX PRO 6000 数据确实有 29,268 条 TRT-LLM block-FP8，而 SGLang 完全没有这一 dtype。需要 profiler 证实 TRT-LLM 在 SM120 的真实实现，不能从 `kernel_source=deepgemm` 直接下结论。

### 3.5 NVFP4

SGLang：

- SM100+ 生成；与 block-FP8 不同，SM110+ 分支仍保留 NVFP4。
- 依赖 FlashInfer FP4 API；ImportError 时 case 仍被调度，但 `create_gemm()` 返回 `None`，最终只打印 skip、不写行。
- 权重在计时前量化、shuffle、转置；scale-factor layout 也提前 shuffle。
- 计时闭包包含 activation FP4 动态量化和 `flashinfer_mm_fp4(..., backend="cutlass")`。
- A/B global scale 固定为 1，`alpha=1/(a_global_scale*b_global_scale)`。

TRT-LLM：

- SM100+ 生成。
- 权重通过 `torch.ops.trtllm.fp4_quantize` 以 16 为 block 参数量化，global scale 根据权重 absmax 计算。
- TRT-LLM 1.1/1.2/1.3 使用 `block_scale_interleave_reverse`；其他版本走 `nvfp4_block_scale_interleave_reverse`，说明 layout 与版本强相关。
- 权重及 scale 先搬到 CPU 交给 `load_weights()`，之后 Linear 再迁回目标 GPU；这些加载成本在计时外。
- 每个 M 的 input global scale 会重新计算并作为 `input_scale` 装入，但计时主体只是 `Linear.forward(x)`。

虽然两边都可称为 NVFP4 Linear，SGLang 明确锁定 FlashInfer CUTLASS，TRT-LLM 则由 Linear 内部分派；这解释了同 dtype 仍可有明显性能差异。

## 4. 采集范围

### 4.1 公共 shape 网格

`get_gemm_common_test_cases()` 当前生成 74 个 M：

- `1..15`；
- `16,17,32,33,...,112,113`；
- `128,129,160,161,...,224,225`；
- `256,257,512,513,...,4352,4353`；
- `8192,16384,32768`。

N/K 各取以下 22 个值：

```text
32, 64, 128, 256, 512, 768, 1024, 1536, 2048, 2560, 3072,
3584, 4096, 5120, 6144, 7168, 8192, 10240, 12288, 16384,
51200, 65536
```

只在公共生成器中排除 `(N,K)=(65536,65536)`，因此：

```text
完整 dtype:        74 * (22 * 22 - 1) = 35,742
N,K >= 128 dtype:  74 * (20 * 20 - 1) = 29,526
```

该网格是插值覆盖网格，不是模型 shape 频率分布。`51200/65536` 用于覆盖超大投影；大量 N/K 笛卡尔积并不一定对应真实模型层。

### 4.2 架构到 dtype 的范围

| SM 范围 | 代表设备 | SGLang 生成 dtype | TRT-LLM 生成 dtype |
|---|---|---|---|
| `<89` | A100 / SM80 | BF16 | BF16 |
| `89` | L40S | BF16、FP8 | BF16、FP8、FP8-block（因为条件为 `>86`） |
| `90..<100` | H100/H200 | BF16、FP8、FP8-block | BF16、FP8、FP8-block |
| `100..<110` | B200/B300/GB200/GB300 | BF16、FP8、FP8-block、NVFP4 | BF16、FP8、FP8-block、NVFP4 |
| `>=110`（现有数据主要 SM120） | RTX PRO 6000 | BF16、FP8、NVFP4 | BF16、FP8、FP8-block、NVFP4 |

### 4.3 TRT-LLM 的额外裁剪

当前 TRT-LLM 脚本还有三层过滤：

1. 对全部 dtype，若 `M*N >= 2^31` 或 `M*K >= 2^31` 则跳过。当前网格中即 M=32768 且 N 或 K=65536 的 shape。
2. 对 block-FP8，若 `N*K >= 2^31` 则跳过；当前涉及 `(51200,51200)`、`(51200,65536)`、`(65536,51200)`。
3. TRT-LLM `1.3.0rc5/1.3.0rc10` + SM120 + 普通 FP8 时，跳过 `M=1..8` 且 N/K 都至少 51200 的 shape，以规避会污染 CUDA context 的 illegal memory access。

由此可算出当前脚本的理论条数：

| TRT-LLM dtype/条件 | 当前理论条数 |
|---|---:|
| BF16，或非 SM120 普通 FP8 | 35,700 |
| SM120 普通 FP8 | 35,676 |
| NVFP4 | 29,488 |
| FP8-block | 29,268 |

SGLang 没有这些 32-bit 边界过滤，所以历史上同硬件同 dtype 的 shape 集也可能不相等。

## 5. 计时方法差异

两边都使用 `benchmark_with_power`，默认 3 次 warmup、6 次测量，并优先 CUDA Graph capture。主要差异在 `op_list` 的构造与缓存：

| 方面 | SGLang | TensorRT-LLM |
|---|---|---|
| op 副本上限 | 6 | 5 |
| 副本数依据 | 估算总显存的 70% 可容纳多少份 weight+activation+output | `ceil(L2_bytes/weight_bytes)`，上限 5 |
| 小权重 | 多份独立 A/B/op，顺序执行 | 共用同一个 x，多份独立 weight/Linear |
| 大权重 | 通常缩为 1 份 | 1 份，并按 `(dtype,N,K)` 在 worker 进程内缓存 weight |
| case 顺序 | 固定随机种子 42 后全局 shuffle | 按 N/K 从大到小分组；每个 dtype 内 M 从大到小 |
| 初始化 | 构建 closure；DeepGEMM JIT 在后续 warmup/capture 中触发 | 每个 Linear 先 dry-run，再进入 benchmark |
| latency 归一化 | 除以实际 `len(op_list)` | 除以 `outside_loop_count` |

需特别关注：

- `benchmark_with_power` 默认测的是 CUDA Graph replay。若线上路径是 eager、shape 动态或 graph capture 不同，collector 延迟不等价于线上单次 launch。
- SGLang 的每份 op 有独立输入；TRT-LLM 小权重的多份 Linear 共用 x。二者的 L2/读缓存压力并不完全相同。
- TRT-LLM 的 weight cache 是 per-worker；多 GPU worker 从共享队列取任务，实际命中率取决于任务耗时和调度，不只取决于 case 排序。
- 两边都会把 weight quantization、layout shuffle、`load_weights` 排除在计时外；activation quantization 是否包含则取决于 dtype 路径。
- 两边只测延迟，不检查结果误差。随机 scale、直接 cast 和 dummy global scale 足以驱动 kernel，但不代表真实模型的数值量化过程。

## 6. 仓库现有数据成果审计

### 6.1 总体情况

当前找到 31 份 SGLang/TRT-LLM `gemm_perf.txt`。所有文件都是 10 列 legacy schema，均没有 `power,power_limit`；所有 latency 都可解析且大于 0。

现有成果包含两代网格：

- 新网格：74 个 M、22 个 N/K，完整单 dtype 为 35,742 行。
- 旧网格：21 个 M、21 个 N/K，完整单 dtype 为 9,240 行；低比特过滤后常见 7,560 或 6,048 行。

代表性覆盖如下：

| 设备/框架版本 | dtype 行数 | 判断 |
|---|---|---|
| A100 SGLang 0.5.10 | BF16 35,742 | 新网格完整 |
| A100 TRT-LLM 1.0.0 | BF16 9,240；`int8_wo/int4_wo/sq` 各 6,048 | 旧网格、旧量化路径 |
| L40S SGLang 0.5.10 | BF16 35,735；FP8 35,632 | 新网格，分别缺 7/110 行 |
| L40S TRT-LLM 1.0.0 | BF16/FP8 各 9,240；block-FP8 7,560；三种旧量化各 6,048 | 旧网格 |
| H100 SGLang 0.5.10 | BF16 35,742；FP8 35,741；block-FP8 29,524 | 新网格，共缺 3 行 |
| H100/H200 TRT-LLM 1.3.0rc10 | BF16/FP8 各 35,742；block-FP8 29,526 | 新网格完整，但尚未体现当前新增的 32-bit 过滤 |
| B200/B300/GB200/GB300 SGLang 0.5.9/0.5.10 | BF16/FP8 各 35,742；block-FP8/NVFP4 各 29,526 | 新网格完整 |
| Blackwell TRT-LLM 1.3.0rc10 | BF16/FP8 各 35,742；NVFP4 约 29,526；block-FP8 29,242..29,382 | 基于过滤前网格采集，block-FP8 有失败/缺行，另有少量重复 |
| RTX PRO 6000 SGLang 0.5.10 | BF16/FP8 各 35,742；NVFP4 29,526 | 与 SGLang 当前生成规则完全一致 |
| RTX PRO 6000 TRT-LLM 1.3.0rc10 | BF16 35,700；FP8 35,676；block-FP8 29,268；NVFP4 29,488 | 与 TRT-LLM 当前所有过滤规则完全一致 |

这说明当前源码的理论范围只能直接解释最新一部分数据。旧 CSV 应与当时的 collector commit/镜像绑定使用，不能用今天的脚本反推其全部执行细节。

### 6.2 明确的数据质量问题

1. H100 SGLang 0.5.10 缺 3 行：普通 FP8 缺 `(32768,51200,65536)`；block-FP8 缺 `(32768,51200,65536)` 和 `(32768,65536,51200)`。它们都位于超大 32-bit 边界附近。
2. L40S SGLang 0.5.10 的 BF16 缺 7 行、FP8 缺 110 行，缺失高度集中在 `51200/65536` 超大维度，不能称为完整网格。
3. GB200 TRT-LLM 1.3.0rc10 有 8 个重复主键，均为 `fp8_block,N=2560,K=8192` 的不同 M；GB300 同版本有 4 个重复 NVFP4 主键。
4. `load_gemm_data()` 遇到重复主键时保留第一行、忽略后续行，只打 debug conflict。因此合并或续采产生的后写结果不会生效。
5. H100 TRT-LLM 1.2.0rc5 数据含 483 条 BF16 和 399 条 block-FP8 的 `M=0`，总计 882 个异常 shape；普通 FP8 没有 M=0。这是历史生成器语义，不能与当前 M>=1 网格混用。
6. SGLang 0.5.6.post2 数据和部分 TRT-LLM 1.0/1.2 数据含当前已删除的 `int8_wo/int4_wo/sq`。SDK enum 仍能加载这些名字，但当前 collector 无法复采。
7. SGLang 当前声明只兼容 `>=0.5.10rc0`，仓库却有由不同代码快照生成的 0.5.9/0.5.6 数据。版本目录是数据标签，不是当前源码可复现性的证明。

### 6.3 相同 shape 的延迟结果

下面对每个设备选择较新的 SGLang 0.5.10 与可用 TRT-LLM 版本，以 `(dtype,M,N,K)` 精确内连接；每格为 `TRT-LLM latency / SGLang latency` 的中位数。小于 1 表示 TRT-LLM collector 路径更快。该统计对每个合成 shape 等权，不按真实模型频率加权。

| 设备 | TRT-LLM 版本 | BF16 | FP8 | FP8-block | NVFP4 |
|---|---|---:|---:|---:|---:|
| A100 | 1.0.0 | 0.954 | - | - | - |
| L40S | 1.0.0 | 0.954 | 1.110 | - | - |
| H100 | 1.3.0rc10 | 0.992 | 1.210 | 0.990 | - |
| H200 | 1.3.0rc10 | 0.963 | 1.229 | 0.994 | - |
| B200 | 1.3.0rc10 | 0.967 | 1.393 | 0.856 | 1.258 |
| B300 | 1.3.0rc10 | 0.992 | 1.420 | 0.855 | 1.349 |
| GB200 | 1.3.0rc10 | 0.999 | 1.463 | 0.855 | 1.276 |
| GB300 | 1.3.0rc10 | 0.997 | 1.469 | 0.869 | 1.368 |
| RTX PRO 6000 | 1.3.0rc10 | 0.966 | 0.996 | 无共同 dtype | 0.970 |

这些中位数掩盖了很宽的 shape 分布。例如：

- H100 普通 FP8 的 P10/中位/P90 为 `0.783/1.210/2.075`；block-FP8 为 `0.766/0.990/1.257`。
- B200 普通 FP8 为 `0.874/1.393/2.226`；block-FP8 为 `0.490/0.856/1.100`；NVFP4 为 `0.981/1.258/1.724`。
- RTX PRO 6000 普通 FP8 为 `0.411/0.996/1.737`，整体中位接近并不表示所有 M/N/K 都接近。

M 分段也显示不同趋势：B200 的 TRT/SGL 普通 FP8 中位数在 `M<=16`、`17..256`、`M>256` 分别为 `1.654/1.664/1.194`；block-FP8 则为 `0.667/0.742/0.993`。这更像动态量化、small-M 专用 kernel 和框架分派共同作用，而非单一 Tensor Core 吞吐差异。

## 7. 对 SDK 查询和建模的影响

SDK `load_gemm_data()` 按 `GEMMQuantMode -> M -> N -> K` 装载，不使用 `framework`、`version`、`device`、`kernel_source` 参与查询。结果包括：

- `kernel_source` 粗糙不会直接影响 `query_gemm()`，但会让数据溯源和跨框架 kernel 对齐困难。
- 缺少精确 M 时，优先在固定 N/K 上做 M 方向一维插值；不足两个点再做 3D 插值。超大边界缺行可能被邻近 M 插值掩盖，而不是显式暴露采集失败。
- 重复主键是 first-wins；续采文件若直接 append，后一次更可信的值不会覆盖前值。
- `fp8_static` 查询会归一化到 `fp8` 表。若上层又单独查询 `compute_scale/scale_matrix`，必须确认是否与当前 GEMM collector 已包含的激活动态量化重复计费。
- 历史 `sq/int4_wo/int8_wo` 虽仍在 SDK enum 中，但不是当前两套 GEMM collector 的共同覆盖面。

## 8. 采集与比较时必须关注的细节

### 8.1 采集前

1. 固定框架、collector commit、容器、CUDA、PyTorch、FlashInfer、sgl-kernel/DeepGEMM commit，并把这些元数据随 CSV 保存。
2. 先打印 `get_gemm_test_cases()` 的 dtype/shape 计数，与本文理论值比较；不要等全量结束后才发现版本或 SM 分支错误。
3. 在 L40S 验证 TRT-LLM block-FP8 是否真实支持；在 SM120 用 profiler 验证所谓 `deepgemm` 的实际 kernel。
4. 检查 FlashInfer FP4 import。SGLang 缺依赖时 NVFP4 会静默无数据，而不是在 case 生成阶段失败。
5. 清空或隔离旧输出文件；直接 append 会产生重复主键，数据库仍使用旧的第一行。

### 8.2 采集中

1. 保留 error report/checkpoint，统计每个 dtype 的成功数、失败数和跳过数。
2. 对 `N/K=51200/65536` 和 `M=32768` 单独监控，它们是 OOM、32-bit 索引和 illegal access 的集中区域。
3. 记录 CUDA Graph 是否实际启用；`benchmark_with_power` 返回 `used_cuda_graph`，当前 GEMM CSV 却没有落盘。
4. 记录 `outside_loop_count`。它改变一次 graph replay 中串行执行的 GEMM 数和缓存压力。
5. 功耗采集若启用，应确保所有行 schema 一致；当前仓库 31 份 GEMM 文件均没有功耗列。

### 8.3 比较数据时

1. 只对 `(device,framework_version,dtype,M,N,K)` 精确匹配的行比较，先去重再统计。
2. 将普通 FP8、block-FP8、NVFP4 分开；不要把所有“8 bit/4 bit”合并。
3. 同 dtype 还要核对 activation quant、weight scale 粒度、scale 编码、layout、输出 dtype 和 epilogue。
4. 至少按 small-M/decode、mid-M、large-M/prefill 分段；全网格中位数会掩盖反向趋势。
5. 报告共同 shape 数、缺失数和 P10/P50/P90，不只报告均值或单个代表点。
6. 若目标是比较纯 GEMM kernel，应把 activation quantization 拆出；若目标是模拟线上 Linear，则应确保两边都包含与线上相同的动态量化和 epilogue。

## 9. 建议的后续改进

按优先级建议：

1. 扩充 CSV 元数据：`activation_quant_included`、`weight_scale_granularity`、`backend_requested`、`cuda_graph_used`、`outside_loop_count`、collector commit 和依赖版本。
2. 将 `kernel_source` 改为 dtype 级来源，例如 `aten_linear`、`sgl_fp8_scaled_mm`、`deepgemm`、`flashinfer_cutlass_fp4`、`trtllm_linear_dispatch`；实际 CUDA symbol 另由 profiler sidecar 记录。
3. 在 case 生成后输出 expected-count manifest，在采集后自动审计 missing/duplicate/unexpected dtype/M=0。
4. 明确普通 FP8 的共同语义。如果要做跨框架 kernel 对比，应统一 per-tensor/per-token/per-channel scale；如果保留框架原生路径，文档和查询层应把它定义为 engine-specific Linear。
5. 删除或恢复 SGLang 的 `int8_wo` 半残留路径；TRT-LLM 旧 `int8_wo/int4_wo/sq` 若仍需维护，应使用版本路由而不是让历史 CSV 成为唯一实现证据。
6. 对 TRT-LLM GEMM 增加 `__compat__` 或版本化 collector。当前同一文件无法可靠说明 1.0、1.2、1.3 的全部历史路径。
7. 数据合并前按主键做显式策略：拒绝重复、取最新、或多次采样聚合；不要依赖 loader 隐式 first-wins。

## 10. 最终判断

当前两套 GEMM collector 更适合回答“各自框架原生 Linear 路径在给定 shape 上有多快”，不适合在不加限定的情况下回答“同一个 CUDA GEMM kernel 谁更快”。BF16 的可比性最高；block-FP8 次之，但需按 SM90/SM100/SM120 区分后端；普通 FP8 与 NVFP4 必须把量化和 layout 差异纳入解释。

数据层面，新网格已经覆盖主流 Hopper/Blackwell 设备，SGLang Blackwell 数据尤其完整；但历史网格、旧量化类型、超大 shape 缺行、重复主键和脚本版本漂移仍需在使用前审计。RTX PRO 6000 的 TRT-LLM 数据是当前规则最一致的样本，也清楚展示了 SM120 特殊过滤与两框架 block-FP8 能力边界的差异。
