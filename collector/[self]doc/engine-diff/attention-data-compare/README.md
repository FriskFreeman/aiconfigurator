# Attention 数据对比工作目录

本目录用于比较仓库现有 SGLang 与 TensorRT-LLM attention 性能数据。

## 口径

- 横轴为 SGLang latency，纵轴为 TensorRT-LLM latency，单位均为 ms。
- 每个精确匹配 shape 是一个散点，不抽样、不聚合。
- context 精确键：`attn_dtype,kv_cache_dtype,isl,batch_size,num_heads,num_key_value_heads,head_dim,window_size`。
- generation 精确键：`kv_cache_dtype,isl+step,batch_size,num_heads,num_key_value_heads,head_dim,window_size`；这与 SDK 忽略 generation `attn_dtype`、以总序列长度索引的语义一致。
- SGLang 旧格式没有 `window_size`，按 SDK 兼容逻辑补为 0。
- 重复查询键按 SDK loader 的 first-wins 语义保留第一行。

## 运行

```bash
python collector/'[self]doc'/engine-diff/attention-data-compare/plot_attention_engine_comparison.py
```

默认输出到 `results/`：5 类配置各有 PNG/PDF，以及 `matched_shape_summary.csv`。
