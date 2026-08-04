# Attention / 基础 MLA dtype 分析工作目录

本目录用于比较 AIC 中三种引擎的普通 Attention 和基础 MLA dtype 行为：

- SGLang
- TensorRT-LLM
- vLLM
- prefill/context 与 decode/generation
- A100、L40S、H100/H200、B200/B300、GB200/GB300、RTX PRO 6000

不包含 MLA module、WideEP MLA 或 DSA module。

## 目录内容

- [`Attention与基础MLA-Dtype全链路分析.md`](Attention与基础MLA-Dtype全链路分析.md)：collector、引擎源码、SDK、数据面的主分析文档
- [`analyze_dtype_data.py`](analyze_dtype_data.py)：从 `src/aiconfigurator/systems/data` 读取选定版本，按 SDK key匹配同 shape dtype组合并作图
- [`results/data_inventory.csv`](results/data_inventory.csv)：所选文件、行数、dtype组合和 kernel_source清单
- [`results/kernel_source_by_dtype.csv`](results/kernel_source_by_dtype.csv)：按硬件/引擎/phase/dtype拆分的 kernel_source计数
- [`results/dtype_pair_summary.csv`](results/dtype_pair_summary.csv)：匹配形状数、缺失形状、ratio分位数、backend变化和相等延迟比例
- [`results/matched_dtype_points.csv.gz`](results/matched_dtype_points.csv.gz)：全部同 shape匹配点；一行对应图中的一个点
- [`results/baseline_comparison_points.csv.gz`](results/baseline_comparison_points.csv.gz)：以BF16/BF16为统一基准的全部作图点，含工作量代理和small/medium/large分桶
- [`results/size_stratified_summary.csv`](results/size_stratified_summary.csv)：overall及small/large的P10/P50/P90、胜率、规模关系、workload相关性和kernel_source变化率
- `results/{attention,mla}_{prefill,decode}_{sglang,trtllm,vllm}_dtype_combinations.{png,pdf}`：12组引擎独立大图；每张图的9个子图分别对应硬件 `.txt` 表

## 复现

在仓库根目录执行：

```bash
python collector/'[self]doc'/engine-diff/attention-mla-dtype-analysis/analyze_dtype_data.py
```

指定其他数据根目录或输出目录：

```bash
python collector/'[self]doc'/engine-diff/attention-mla-dtype-analysis/analyze_dtype_data.py \
  --data-root /path/to/systems/data \
  --output-dir /path/to/output \
  --dpi 150
```

依赖 `pandas`、`numpy` 和 `matplotlib`。脚本使用 `Agg` backend，不需要显示服务。

## 比较定义

所有 ratio 均为 `target_latency / baseline_latency`；小于1表示target更快。图中横轴统一为BF16/BF16，纵轴为同一表中所有其他dtype组合并用颜色区分，对角线下方表示target更快。每张大图只包含一个engine。

同 shape key与当前 SDK loader一致：

- Attention prefill：`B, S, H, Hkv, D, window`
- Attention decode：`B, total_seq_len, H, Hkv, D, window`
- 基础 MLA prefill：`num_heads, B, S`
- 基础 MLA decode：`num_heads, B, total_seq_len`

重复 SDK key采用 first-row-wins。MLA数据中的 `tp_size` 在已有 `num_heads` 时不是 SDK查询轴，因此不参与点匹配。

small/large不是按实测延迟划分，而是按每个 `.txt` 表内的BF16/BF16 shape工作量代理划分：下四分位为small，上四分位为large。图中同时标注overall P10/P50/P90及small/large P50，避免单看中位数掩盖规模反转。

## 版本策略

版本映射显式写在脚本的 `ATTENTION_VERSIONS` 和 `MLA_VERSIONS` 中。vLLM基础 MLA只使用历史 0.14.0数据；当前 vLLM registry已经移除基础 MLA collector，不能用 MLA module数据代替。

在线源码核查使用代理：

```bash
curl -x http://10.250.0.2:7890 https://raw.githubusercontent.com/OWNER/REPO/TAG/PATH
```

主文档中的引擎行为结论以固定 tag源码为准，不以主分支推断历史数据。
