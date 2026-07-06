# collect_mla generation_mla 复采小实验

## 目的

临时使用当前 H100 环境和 Docker 镜像 `booleimg.myaddr.io/lmsysorg/sglang:v0.5.9`，复采少量 `collector/sglang/collect_mla.py` 的 `mla_generation` 点，验证已有 0.5.9 `generation_mla_perf.txt` 是否存在明显采集漂移或异常。

## 执行方式

- GPU：空闲 GPU 4。
- 镜像：`booleimg.myaddr.io/lmsysorg/sglang:v0.5.9`。
- 入口：`run_collect_mla_generation_subset.py`，直接调用 `run_mla(..., is_context_phase=False)`。
- 复采点：`(b,s_total) = (1,512), (4,512), (32,512), (4,4096), (2,16384)`。
- dtype/head：`kv_cache_dtype=fp8`, `num_heads=128`, `tp_size=1`, `kernel_source=flash_attention`。

## 结论

复采值与已有数据高度一致，5 个点的差异均小于 1%。因此，本轮 attention 对比中的 AIC 偏高并不是因为当前仓库中的 `generation_mla_perf.txt` 数据文件明显采坏，亲自复采得到的 collector 数据仍然保持同样量级。

| b | s_total | old ms | resample ms | delta % |
|---:|---:|---:|---:|---:|
| 1 | 512 | 0.025619 | 0.025666 | +0.181% |
| 4 | 512 | 0.029062 | 0.029128 | +0.226% |
| 32 | 512 | 0.071939 | 0.072608 | +0.930% |
| 4 | 4096 | 0.065992 | 0.066235 | +0.369% |
| 2 | 16384 | 0.104850 | 0.105067 | +0.208% |

## 产物

- `generation_mla_perf_resample.txt`: docker 内复采原始 collector 输出。
- `generation_mla_resample_compare.csv`: 与 `src/aiconfigurator/systems/data/h100_sxm/sglang/0.5.9/generation_mla_perf.txt` 的逐点对比。
- `docker_resample.log`: docker 执行日志。
