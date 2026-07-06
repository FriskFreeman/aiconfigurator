# task-localbench tmp 目录说明

## 本次整理

- 旧的 `.self/task-localbench/tmp*` 子目录已统一迁到 `.self/task-localbench/tmp/` 下。
- 目录名暂时保持原样，目的是先集中管理，尽量减少语义变化。
- 其中带 `output/` 的目录主要存放原始运行输出；带 `raw_runs/`、`csv/`、`manifest.csv` 的 `*-bench` 目录主要存放归档后的 bench 结果。

## 定位方法

本说明中的“生成代码”主要按以下证据定位：

- 脚本默认 `--run-root` 指向该目录。
- 目录内直接包含对应 runner 脚本或总结文档。
- `formal-prefill-stage1/需求1批量运行记录.md` 中显式记录了 `--run-root` / `bench_root`。

## 目录总览

| 目录 | 生成代码 / 来源 | 含义 | 当前主要产物 | 当前结果 / 结论 |
| --- | --- | --- | --- | --- |
| `tmp/tmp(server)` | `run_sglang_deepseek_v3_dummy5_e2e.sh`、`run_sglang_v059_pcg_matrix.sh` | 用 SGLang 原生 server 路径做 dummy DeepSeek-V3 5 层端到端试跑、PCG/warmup 组合排查 | `output/` 下 `18` 个时间戳运行目录，另有 `sglang_v059_*` 的 TSV/MD 汇总 | [`sglang_v059_pcg_warmup_bug_check.md`](./tmp(server)/sglang_v059_pcg_warmup_bug_check.md) 结论是：`v0.5.9` 某些组合会出现“日志 ready，但 `/get_server_info` / `/server_info` 请求卡死”的假 ready 问题。 |
| `tmp/tmp-collect-mla-gen` | `run_collect_mla_generation_subset.py` | 对 `collect_mla` 的 generation MLA 数据做少量复采，验证数据是否漂移 | `generation_mla_perf_resample.txt`、`generation_mla_resample_compare.csv`、`docker_resample.log` | [`README.md`](./tmp-collect-mla-gen/README.md) 结论是：复采点与已有数据差异都小于 `1%`，现有 `generation_mla_perf.txt` 基本可信。 |
| `tmp/tmp-decode` | `run_sglang_decode_instance.py`、`run_decode_inside_container.py` | 不依赖 router/prefill，仅验证独立 `decode` Engine 实例能否启动 | `output/` 下 `2` 个运行目录，加一份 [`运行结果总结.md`](./tmp-decode/运行结果总结.md) | 结论是：独立 `decode` 实例在 `cuda graph on` 的配置下可以成功初始化并正常退出。 |
| `tmp/tmp-engine` | `run_sglang_engine_deepseek_v3_dummy5.py`、`run_engine_inside_container.py` | 直接从 Engine 入口验证 dummy 5 层模型在不同 CG/PCG/backend 组合下的行为 | `output/` 下 `8` 个运行目录，`engine_pcg_summary.tsv`，以及 [`运行结果总结-20260608.md`](./tmp-engine/运行结果总结-20260608.md) | 结论较明确：`v0.5.9` 下 `PCG on` 会失败；`v0.5.12` 下 `triton + PCG` 可以成功，而 `auto -> fa3 + PCG` 仍可能失败。 |
| `tmp/tmp-pd` | `run_sglang_pd_deepseek_v3_dummy5.py`，外加 [`PD-Engine入口评估.md`](./tmp-pd/PD-Engine入口评估.md) | 用 prefill/decode/router 三段链路与 Engine 方案对比，评估 PD 场景是否能绕开 router 外壳问题 | `output/` 下 `9` 个运行目录，评估文档 1 份 | 文档结论是：Engine 方案可以更干净地暴露问题，但并不能绕开底层 transfer backend / `KVReceiver` 相关问题。 |
| `tmp/tmp-prefill` | `run_sglang_prefill_instance.py`、`run_prefill_inside_container.py` | 不依赖 server/router/decode，仅验证独立 `prefill` Engine 实例和 PCG 行为 | `output/` 下 `10` 个运行目录，加一份 [`运行结果总结.md`](./tmp-prefill/运行结果总结.md) | 结论是：`v0.5.9` 下 prefill 本体可启动，但 PCG 初始化失败；`v0.5.12` 下 `triton + PCG` 已可成功，`auto` 仍失败。 |
| `tmp/tmp-prefill-gap-cpusample` | `run_formal_prefill_stage1.py`，命令记录在 `formal-prefill-stage1/需求1批量运行记录.md` | 为定位 prefill gap / CPU sample 问题做的原始 rerun 目录 | `output/` 下 `1` 个运行目录 | 原始输出目录，仅保存 rerun 现场。 |
| `tmp/tmp-prefill-gap-cpusample-cap` | 同上，使用 `--docker-cap-add SYS_ADMIN --docker-cap-add SYS_PTRACE --docker-security-opt seccomp=unconfined` | 带额外 capability 的 gap cpusample rerun | `output/` 下 `3` 个运行目录 | 属于专题 rerun 原始结果。 |
| `tmp/tmp-prefill-gap-cpusample-cap-bench` | 对应 `run_prefill_stage1_batch.py` 的归档产物 / `bench_root` | 保存上面 `cap` 专题 rerun 的归档结果 | `csv/`、`manifest.csv`、`raw_runs/`，其中 `raw_runs/` 下 `3` 个归档运行目录 | 这是“便于比对/复用”的 bench 视图，不是原始运行目录。 |
| `tmp/tmp-prefill-gap-cpusample-cap-ref` | 同一专题中的 reference case rerun | 作为 `cap` 实验的参考对照运行目录 | `output/` 下 `1` 个运行目录 | 原始 reference 输出。 |
| `tmp/tmp-prefill-gap-cpusample-cap-ref-bench` | 对应 reference case 的归档产物 | 保存 reference case 的 bench 视图 | `csv/`、`manifest.csv`、`raw_runs/`，其中 `raw_runs/` 下 `1` 个归档运行目录 | 供对照分析使用。 |
| `tmp/tmp-prefill-gap-cpusample-dryrun-bench` | `需求1批量运行记录.md` 中的 dry-run 命令 | dry-run 试运行留下的占位 bench 目录 | 仅 `manifest.csv` | 没有真实运行结果。 |
| `tmp/tmp-prefill-gap-cpusample-dryrun2-bench` | 同上 | 第二轮 dry-run 占位目录 | 仅 `manifest.csv` | 没有真实运行结果。 |
| `tmp/tmp-prefill-gap-cpusample-priv` | `run_formal_prefill_stage1.py`，带 `--docker-privileged --docker-pid-host` | privileged 模式下的 gap cpusample rerun | `output/` 下 `1` 个运行目录 | 原始 privileged rerun 输出。 |
| `tmp/tmp-prefill-gap-cpusample-priv-bench` | 上述 privileged rerun 的 bench_root | privileged 方案的占位 bench 目录 | 仅 `manifest.csv` | 当前没有完整归档结果。 |

## 结果目录的共性

### `output/`

- 主要出现在 runner 型目录中。
- 里面通常是按 `YYYYMMDD_HHMMSS_*` 命名的原始运行目录。
- 常见文件包括：`run_meta.json`、`status.json`、`container.stdout.log`、`container.stderr.log`、`docker_command.txt`、`nsys/`、`profile/`。

### `*-bench`

- 主要是“归档/对比友好”的结果视图。
- 常见结构：
  - `manifest.csv`: 批量运行记录。
  - `csv/`: 从原始运行目录抽取出的关键分析表。
  - `raw_runs/`: 归档后的原始运行目录副本。

## 当前建议

- 如果后续继续新增临时实验，优先放到 `.self/task-localbench/tmp/` 下，避免再次在根层扩散。
- runner 型目录继续沿用“目录本体放脚本，运行结果落到 `output/`”的结构即可。
- `tmp-prefill-gap-cpusample*` 这一组更像专题分析缓存；如果后续确认不再使用，可以优先考虑只保留 `*-bench` 或只保留最终分析表。
