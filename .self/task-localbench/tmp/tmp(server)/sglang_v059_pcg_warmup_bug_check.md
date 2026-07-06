# SGLang v0.5.9 PCG / Warmup 试跑排查记录

日期：2026-06-08

镜像：

- `booleimg.myaddr.io/lmsysorg/sglang:v0.5.9`

测试目标：

- 使用最后一张 H100（`GPU 7`）
- 使用 SGLang 原生端到端流程
- dummy `DeepSeek-V3` 前 5 层
- 对比 `PCG on/off` 与 `skip_server_warmup on/off` 的行为

当前阶段结论：

1. `v0.5.9` 已支持 piecewise CUDA graph，但语义是显式开启：
   - 开启：`--enable-piecewise-cuda-graph`
   - 关闭：不传该参数
2. 在 `v0.5.9` 上，`/get_server_info` 比 `/server_info` 更可靠地反映旧版本兼容路径。
3. 已确认至少一组存在明显异常：
   - `PCG off`
   - `skip_server_warmup=1`
4. 该组异常表现：
   - 日志出现 `The server is fired up and ready to roll!`
   - 日志反复出现 `Endpoint '/get_server_info' is deprecated...`
   - 说明 HTTP 请求已进入 handler
   - 但容器内、容器外的 `GET /get_server_info` 和 `GET /server_info` 都会读超时
   - 表现为“日志 ready，但 API 不返回响应体”
5. 因此该问题更像：
   - `v0.5.9` 在此 dummy DeepSeek-V3 5 层场景下出现 API 假 ready / handler 卡死
   - 不是单纯的外部端口映射问题

关键证据目录：

- 基准排查 run：
  - [20260608_145059_sglang_deepseek_v3_dummy5_e2e](./output/20260608_145059_sglang_deepseek_v3_dummy5_e2e)
- 早期 `v0.5.12` 可跑通参考：
  - [20260608_112226_sglang_deepseek_v3_dummy5_e2e](./output/20260608_112226_sglang_deepseek_v3_dummy5_e2e)

`20260608_145059` 这组已确认现象：

- `server.log` 中有：
  - `The server is fired up and ready to roll!`
- `wait_progress.log` 中有多次：
  - `Endpoint '/get_server_info' is deprecated and will be removed in a future version. Please use '/server_info' instead.`
- 但实际 `curl` 结果：
  - host `GET /get_server_info` 超时
  - container 内 `GET /get_server_info` 超时
  - host `GET /server_info` 超时
  - container 内 `GET /server_info` 超时

当前判断：

- `skip_server_warmup=1` 很可能会暴露 `v0.5.9` 这条路径上的服务可用性问题
- 是否与 `PCG on/off` 强相关，仍需继续逐组验证

下一步建议：

1. 优先测 `skip_server_warmup=0` 两组，确认 warmup 是否能避开该假 ready 问题。
2. 再测 `PCG on` 两组，确认问题是否与 PCG 强耦合。
3. 若 warmup 打开后恢复正常，再进一步比较：
   - `start_profile`
   - `/generate`
   - `/stop_profile`
   是否仍存在卡住或超时。
