# FA/MLA 硬件输入来源与估算边界

本验收覆盖 `a100_sxm`、`h100_sxm`、`h200_sxm`、`b200_sxm`、`b300_sxm` 和 `rtx_pro_6000_server`。systems YAML 中的显存带宽、容量和各精度 Tensor Core 峰值是最终覆盖值；独立 JSON 只补充 FA/MLA 所需的调度与 cache 微架构字段。

## 公开来源

- NVIDIA CUDA Blackwell Tuning Guide：<https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html>
  - CC 10.0（B200）每 SM 可用 shared memory 为 228 KiB。
  - CC 12.0 每 SM shared memory 为 128 KiB。
  - GB200/B200 公开 L2 容量为 126 MiB。
- NVIDIA H100 Tensor Core GPU Architecture Whitepaper 与本机 CUDA `deviceQuery`：H100 SXM 为 132 SM、50 MiB L2；H200 沿用 GH100 compute/cache hierarchy。
- NVIDIA B200 与 RTX PRO 6000 Blackwell Server Edition 产品规格：SM 数、产品时钟、显存带宽和计算峰值。峰值/HBM 数值最终以仓库 systems YAML 为准。

## 非官方建模输入

产品页不发布持续 L2 带宽。B200/RTX PRO 6000 的 `l2_bandwidth_bytes_s` 按 H100 已校准值，以 `SM count * clock` 比例缩放，仅用于 roofline 的 L2 端点，不应解释为硬件实测带宽。

RTX PRO 6000 的 96 MiB L2 来自公开设备规格汇总而非 NVIDIA 产品页，属于需在目标机器上用 CUDA device attribute/deviceQuery 复核的输入。由于本机仅有 H100，无法直接读取 SM120 设备属性。

B300 的公开资料没有给出持续 L2 带宽。其 BF16 峰值与仓库 B200 配置一致，本轮采用相同的 148 SM、1.83 GHz、228 KiB/SM 和 126 MiB L2 基线，FP8/FP4 峰值与 HBM 则由 B300 systems YAML 覆盖。这是 Blackwell Ultra 的验收级迁移假设，不是 B300 cache 实测结果。
