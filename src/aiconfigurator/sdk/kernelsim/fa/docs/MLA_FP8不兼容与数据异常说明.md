# MLA FP8 不兼容与数据异常说明

## 1. 生产结论

当前归档的 MLA 模型只支持 BF16。Prefill FP8 与 Decode FP8 均不进入：

- `standard/high/low` 通用 profile；
- H100 precise 生产 profile；
- MLA request JSON schema；
- CLI/API 正常预测路径。

调用端传入 `fp8` 或 `float8` 时会直接报错：

```text
FP8 MLA is not supported by the production model: the available SGLang
0.5.9 measurements show substantial backend performance degradation, so
no transferable FP8 profile is published
```

这里的“不兼容”表示拒绝给出可能误导的预测，不表示底层 FA 核心无法计算 FP8 FLOPs。

## 2. Prefill FP8 异常

H100 SGLang 0.5.9 修复后的 MLA prefill 数据中，BF16/FP8 有 880 对相同物理 shape。实测：

```text
FP8/BF16 latency ratio median = 1.241
P10/P90 = 0.940 / 1.532
只有 14.8% 的 shape 中 FP8 比 BF16 更快
```

这与“FP8 matrix peak 更高，因此 attention 应更快”的简单预期相反。源码和 collector 边界显示：

- Q/V 激活初始为 BF16；
- FP8 prefill 在 FA backend 计时区间内存在 Q/V BF16 到 FP8 的转换；
- K dtype、MHA 路径和 cache dtype 还会影响 kernel 路由；
- 独立 `_concat_and_cast_mha_k` 不属于当前 attention core 主时延。

H100 FP8 precise 拟合曾得到约 `resource_eta=0.319`，显著低于 BF16 的 `0.698`。该 eta 混合吸收了 cast、kernel 路径和资源效率，不能解释为跨硬件 FP8 效率。

曾尝试单列：

```text
T = floor + resource/eta + cast_bytes/(HBM_BW*eta_cast)
```

虽然部分候选误差下降，但 `eta_cast` 在 fold 或 bootstrap 中触及边界，无法与 resource eta 稳定辨识。因此没有把一个表面上可解释、实际不可辨识的 cast 系数写入生产 profile。

## 3. Decode FP8 异常

AIC `generation_mla_perf.txt` 中有 2324 条 Decode FP8 记录，但 SGLang 0.5.9 的该路径存在已知 backend 劣化，且 FA/absorbed MLA 对 FP8 的支持行为不适合作为正常算子性能。实验从数据处理起就将这 2324 行保存为独立排除集：

```text
.self/compass-sim-mla_sglang059/data/processed/excluded_decode_fp8.csv
```

它们没有参与 OOF、准确率、参数选择或三档设计。

## 4. 为什么不能只换 FP8 peak

仅把 `matrix_peak_flops[bf16]` 换成 FP8 peak 会遗漏：

- Q/V dtype 转换流量与 kernel launch；
- BF16 输出和 reduction/normalize 路径；
- backend 对 FP8 cache、scale/descale 和 q/k/v dtype 的特殊处理；
- FP8 kernel recipe、tile、occupancy 与 split-KV 差异；
- 当前 backend 本身的功能或性能退化。

因此底层 `AttentionShape(dtype="fp8")` 的解析能力不能自动等价为“MLA FP8 模型已验证”。

## 5. 恢复支持的验收条件

只有同时满足以下条件才应新增 FP8 profile：

1. 目标 backend 的 prefill/decode FP8 路径功能正确且性能不再异常；
2. collector 明确记录 Q、K、V、KV cache、输出 dtype 和 cast 计时边界；
3. BF16/FP8 同 shape 配对重新采集，确认性能趋势；
4. cast/resource 参数可以在 grouped CV 和 bootstrap 中独立辨识；
5. 至少在两个硬件或两个 backend 上验证，避免把单 H100 劣化固化为通用 eta；
6. FP8 low/standard/high 在完整 shape grid 上通过单调性和误差验收。

在此之前，生产工具保持 fail-closed。
