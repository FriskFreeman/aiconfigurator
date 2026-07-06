# SGLang chunk prefill 调度控制流详解

本文聚焦 SGLang 调度里最核心的一段逻辑: `chunked prefill` 如何把等待队列中的请求切成 chunk, 再把它们组织成一个个 prefill / mixed batch。

重点变量:

- `chunked_prefill_size`: 开启 chunked prefill 后, 单轮 prefill 的 chunk 预算。
- `max_prefill_tokens`: 单轮 prefill 的总 token 预算。
- `rem_chunk_tokens`: 当前这轮 prefill 里还剩多少 chunk 预算。
- `rem_input_tokens`: 当前这轮 prefill 里还剩多少输入 token 预算。

代码主入口:

- `python/sglang/srt/managers/scheduler.py`
- `python/sglang/srt/managers/schedule_policy.py`
- `python/sglang/srt/managers/schedule_batch.py`
- `python/sglang/srt/managers/scheduler_components/batch_result_processor.py`

---

## 1. 先给结论

SGLang 的 chunk prefill 可以概括成 4 句话:

1. `Scheduler._get_new_batch_prefill_raw()` 每轮创建一个 `PrefillAdder`, 用它决定这轮 batch 能塞哪些请求, 以及是否要把某个请求切成 chunk。
2. 一旦某个请求被切成 chunk, 这个请求不会重新回到普通 waiting queue 排队, 而是被单独挂在 `scheduler.chunked_req` 上, 下一轮优先继续跑。
3. 中间 chunk 只是“继续预填充”, 不会产出用户可见的首个生成 token; 只有最后一个 chunk 跑完, 这个请求才真正完成 prefill 并进入 decode 语义。
4. 开启 chunked prefill 后, `chunked_prefill_size` 不只是“大请求的单块大小”, 它还会变成“这一轮整个 prefill batch 的 token 上限”。

最后这一点非常关键, 也是很多人第一次读代码时最容易误解的地方。

---

## 2. 变量分别管什么

### 2.1 `chunked_prefill_size`

初始化入口在 `Scheduler.init_chunked_prefill()`:

- `scheduler.py:872-909`

它来自 `ServerArgs.chunked_prefill_size`:

- `server_args.py:404-411`

语义上:

- `None` 或 `<= 0`: 关闭 chunked prefill。
- `> 0`: 开启 chunked prefill, 并把每轮 prefill 的 chunk 预算设为这个值。

注意:

- 对某些 multimodal + transformers backend 组合, SGLang 会直接禁用 chunked prefill, 防止 partial multimodal chunk 不一致。

### 2.2 `max_prefill_tokens`

`max_prefill_tokens` 不是纯粹的“用户配置值”; scheduler 启动时会从 worker 侧拿到最终值:

- `scheduler.py:782-832`

它是单轮 prefill 的总预算上限, 无论有没有 chunk 模式, 都参与控制。

### 2.3 `rem_input_tokens`

在 `PrefillAdder.__init__()` 中:

- `schedule_policy.py:413-437`

初始化逻辑是:

```text
rem_input_tokens = max_prefill_tokens - num_mixed_decode_tokens
```

随后每接纳一个 prefill 请求, 都会在 `_update_prefill_budget()` 中递减:

- `schedule_policy.py:582-603`

它表达的是:

- “这一轮还能再接纳多少 prefill 输入 token”

如果开启 mixed chunk, 当前轮里还要顺带跑 decode, 那么 decode 的 1-step token 也会先从这里扣掉。

### 2.4 `rem_chunk_tokens`

同样在 `PrefillAdder.__init__()` 中初始化:

- `schedule_policy.py:413-437`

初始化逻辑是:

```text
rem_chunk_tokens = chunked_prefill_size - num_mixed_decode_tokens
```

随后每接纳一个 prefill 请求, 也会在 `_update_prefill_budget()` 中递减:

- `schedule_policy.py:597-600`

它表达的是:

- “当 chunk 模式开启时, 这一轮 prefill batch 还能再放多少 prefill token”

所以它不是“只给一个长请求用的局部变量”, 而是整轮 prefill 的共享预算。

### 2.5 `chunked_req`

这是 scheduler 级状态:

- `scheduler.py:890`
- `scheduler.py:2663-2672`

语义是:

- 当前唯一一个“还没跑完 prefill, 需要下一轮继续切 chunk”的请求。

这也意味着同一时刻 SGLang 只会持续追踪一个正在分块推进的请求。

### 2.6 `inflight_middle_chunks`

定义在 `Req` 上:

- `schedule_batch.py:826-829`

用途:

- 标记这个请求还有多少“已在路上但尚未完成语义收尾”的中间 chunk。

在 `batch_result_processor.py:223-289` 中:

- `inflight_middle_chunks <= 0`: 说明这次已经是最后一个 chunk, 可以追加首个输出 token, 更新 finish state, 决定是否进入 decode。
- `inflight_middle_chunks > 0`: 说明还是中间 chunk, 只记账, 不产生真正的生成输出。

---

## 3. 一轮 prefill batch 是怎么组出来的

主入口在:

- `scheduler.py:2476-2735`

流程可以简化成下面这个顺序。

### 3.1 先决定这一轮是否值得做 prefill

`_get_new_batch_prefill_raw()` 一开始会做几件事:

- grammar ready 的请求先转回 waiting queue
- hierarchical cache 事件先处理
- 如果 `running_batch.batch_is_full` 且没有遗留 `chunked_req`, 直接返回
- 如果本轮可分配 request slot 不够, 也可能直接返回

对应代码:

- `scheduler.py:2499-2530`

### 3.2 排 waiting queue

然后调度策略对 waiting queue 排序:

- `scheduler.py:2532-2533`
- `schedule_policy.py:162-225`

这一步跟 FCFS / LPM / DFS / LOF 等策略有关, 但和 chunk 切分本身无关。

### 3.3 如果上轮已经有没跑完的 chunk, 先续跑它

这是最重要的一步之一:

- `scheduler.py:2568-2575`

逻辑是:

1. `self.chunked_req.init_next_round_input()`
2. `adder.add_chunked_req(self.chunked_req)`
3. 如果还没跑完, 返回值还是这个请求本身, 它继续留在 `self.chunked_req`
4. 如果这次已经是最后一块, 返回 `None`, 说明它跑完后不再是“待续 chunk 请求”

也就是说:

- 一旦某个请求进入 `chunked_req`, 下一轮不是重新跟 waiting queue 一起公平竞争。
- 它会被 scheduler 先拿出来续跑。

这使得长请求一旦开跑, 后续 chunk 有很强的“续跑优先级”。

### 3.4 再从 waiting queue 里一个个尝试塞请求

代码:

- `scheduler.py:2586-2650`

每个请求都会先:

1. `req.init_next_round_input(self.tree_cache)`
2. 进入 `PrefillAdder.add_one_req(...)`

其中 `Req.init_next_round_input()` 会重建:

- `fill_ids = origin_input_ids + output_ids`
- 重新做 prefix match
- 更新 `prefix_indices`
- 更新 `extend_input_len = len(fill_ids) - len(prefix_indices)`

对应代码:

- `schedule_batch.py:1043-1131`

这一步很关键, 因为它把“前面已经 prefill 完并被 cache 住的 chunk”折叠进 prefix, 让下一轮只处理剩余 suffix。

### 3.5 `PrefillAdder.add_one_req()` 决定它是:

- 整个装进当前 batch
- 被截断成一个 chunk
- 还是因为预算 / slot / 其他原因暂时不装

核心代码:

- `schedule_policy.py:813-965`

判断逻辑可以概括成:

1. 先看 `rem_total_tokens` / `rem_swa_tokens` / `rem_input_tokens`
2. 如果 chunk 模式关闭:
   - 只受 `max_prefill_tokens` 约束
3. 如果 chunk 模式开启且 `input_tokens <= rem_chunk_tokens`:
   - 整个请求直接进 batch, 不切
4. 如果 chunk 模式开启且 `input_tokens > rem_chunk_tokens`:
   - 截断成 `trunc_len`
   - `req.fill_ids` 被裁到 `prefix + trunc_len`
   - `req.extend_input_len = trunc_len`
   - 这个请求被记到 `adder.new_chunked_req`

注意两个容易忽略的细节:

- `trunc_len` 要按 `page_size` 对齐:
  - `schedule_policy.py:931-950`
- 如果开了 deterministic inference 对齐约束, 还要再按 `truncation_align_size` 收缩:
  - `schedule_policy.py:937-946`

所以最终 chunk 大小常常不是“刚好等于 `chunked_prefill_size`”, 而是一个对齐后、不超过预算的值。

### 3.6 预算怎么扣

统一在 `_update_prefill_budget()`:

- `schedule_policy.py:582-603`

核心规则:

1. `extend_input_len` 先按 `page_size` 向上对齐。
2. `rem_total_token_offset` 扣:
   - `extend_input_len + max_new_tokens + page_overhead`
3. `cur_rem_token_offset` 扣:
   - `extend_input_len + page_overhead`
4. `rem_input_tokens` 扣:
   - `extend_input_len`
5. `rem_chunk_tokens` 扣:
   - `extend_input_len`

所以:

- `max_prefill_tokens` 控制的是“prefill 输入总量”
- `chunked_prefill_size` 控制的是“chunk 模式下这轮 batch 的 prefill 输入总量”
- `rem_total_tokens` 则把 KV 池可用容量也一起纳入约束

### 3.7 创建 `ScheduleBatch` 并 `prepare_for_extend()`

代码:

- `scheduler.py:2676-2699`
- `schedule_batch.py:1808-1905`

这一步把刚才已经裁好的 `fill_ids / prefix_indices / extend_input_len` 变成真正的 forward batch 输入。

batch 中每个请求的 extend 部分就是:

```text
input_ids = fill_ids[len(prefix_indices):]
```

也就是“当前轮真正要跑的 suffix”。

### 3.8 如果开了 mixed chunk, 再把 decode batch 拼进来

代码:

- `scheduler.py:2715-2733`
- `schedule_batch.py:2217-2246`

含义是:

- 当前轮不是“纯 prefill”
- 而是“一个 extend/chunk batch + 一批 decode 1-step 请求”混在一起跑

这时 `num_mixed_decode_tokens` 会在 `PrefillAdder` 初始化时先扣掉预算:

- `schedule_policy.py:429-439`

所以 mixed chunk 的本质不是白送 decode, 而是:

- 先给 decode 预留掉本轮 token 空间
- 再拿剩下空间做 prefill chunk

---

## 4. 跨轮 chunk 是怎么推进的

这一段是理解大请求行为的关键。

### 4.1 上一轮结束后, 中间 chunk 不算 prefill 完成

在 batch result 处理里:

- `batch_result_processor.py:223-289`

如果 `req.inflight_middle_chunks > 0`, scheduler 会:

- 把它当作“中间 chunk”
- 不 append 首个生成 token
- 不做最终 prefill 完成语义
- 只做 logprob 增量更新
- 然后 `inflight_middle_chunks -= 1`

所以中间 chunk 的 forward 更像是:

- “继续把 prompt 往前推进”

而不是:

- “已经进入正常生成”

### 4.2 下一轮开始时, 已算完的 chunk 会先被 cache/stash

代码:

- `scheduler.py:2309-2310`
- `scheduler.py:2356-2370`

如果 `self.chunked_req` 上轮真的被调度执行了, 下一轮 `get_next_batch_to_run()` 顶部会先:

```text
stash_chunked_request(self.chunked_req)
```

底层是:

```text
maybe_cache_unfinished_req(req, tree_cache, chunked=True)
```

也就是把刚算完的这一段 KV/prefix 纳入 cache 体系。

### 4.3 然后 `init_next_round_input()` 重新匹配 prefix

代码:

- `scheduler.py:2568-2570`
- `schedule_batch.py:1043-1131`

于是这个长请求在下一轮会表现成:

- `prefix_indices` 变长了
- `extend_input_len` 变短了

相当于:

```text
已完成部分 -> 变成 prefix
未完成部分 -> 继续作为 extend suffix
```

### 4.4 为什么说“同一时刻最多一个正在续跑的 chunked 请求”

原因是 scheduler 只维护一个:

```text
self.chunked_req
```

对应代码:

- `scheduler.py:890`
- `scheduler.py:2663-2669`

这意味着:

- 当前有一个大请求在跑中间 chunk 时, 其他大请求不会并行进入“续跑态”
- 其他大请求只能留在 waiting queue, 或者等到当前请求的最后一块 batch 有余量时, 才能顺势开启自己的第一块

---

## 5. `rem_input_tokens` 和 `rem_chunk_tokens` 的真实关系

这是全文最值得记住的一段。

### 5.1 没开 chunked prefill 时

此时:

- `rem_chunk_tokens is None`

batch 能塞多少 prefill, 主要看:

- `rem_input_tokens`
- `rem_total_tokens`
- request slot / radix / SWA / LoRA 等附加约束

此时 `max_prefill_tokens` 是主要的 prefill token 上限。

### 5.2 开了 chunked prefill 时

此时会同时维护:

- `rem_input_tokens`
- `rem_chunk_tokens`

其中:

- `rem_input_tokens` 起点一般比 `rem_chunk_tokens` 大
- 但一轮 batch 是否还能继续接纳 prefill, 往往先被 `rem_chunk_tokens` 卡住

结果就是:

- `max_prefill_tokens` 更像“总上界”
- `chunked_prefill_size` 更像“启用 chunk 模式后的实际单轮 prefill 批量上限”

也就是说, 如果:

```text
max_prefill_tokens = 16384
chunked_prefill_size = 4096
```

那么大多数情况下, 单轮 prefill 实际上先会在 4096 左右停下来, 而不是 16384。

### 5.3 mixed chunk 时

若当前有 `N` 个 decode 请求要混跑:

```text
rem_input_tokens = max_prefill_tokens - N
rem_chunk_tokens = chunked_prefill_size - N
```

对应代码:

- `schedule_policy.py:429-439`
- `scheduler.py:2550-2559`

所以 decode batch 越大, 同轮还能切给 prefill 的 chunk 空间越小。

---

## 6. 多种请求到达场景下, chunk 切分与 batch 生成会怎么演化

下面为了突出核心逻辑, 先做一个教学化简:

- 假设 `page_size = 1`
- 假设 `max_prefill_tokens = 16`
- 假设 `chunked_prefill_size = 8`
- 默认不开 mixed chunk
- 先忽略 prefix cache 命中, 即每个请求的 `prefix_indices` 起初都为 0

实际运行时会有 page 对齐、KV 池容量、prefix 命中、LoRA、SWA 等额外影响, 但下面足够说明主干控制流。

### 场景 A: 很多个小请求连续到达

请求输入长度依次为:

```text
R1=2, R2=2, R3=1, R4=1, R5=2, R6=4, R7=3
```

#### 第 1 轮

- 初始预算: `rem_input=16`, `rem_chunk=8`
- 加入 `R1=2` 后: `rem_input=14`, `rem_chunk=6`
- 加入 `R2=2` 后: `rem_input=12`, `rem_chunk=4`
- 加入 `R3=1` 后: `rem_input=11`, `rem_chunk=3`
- 加入 `R4=1` 后: `rem_input=10`, `rem_chunk=2`
- 加入 `R5=2` 后: `rem_input=8`, `rem_chunk=0`
- 到 `R6=4` 时, 本轮 `rem_chunk=0`, 不能再接

于是 batch1:

```text
[R1(2), R2(2), R3(1), R4(1), R5(2)]
```

观察:

- 虽然 `max_prefill_tokens` 还有富余, 但 batch 已经被 `chunked_prefill_size=8` 卡住了。
- 所以开启 chunked prefill 后, “很多小请求”也会先受 `chunked_prefill_size` 约束。

#### 第 2 轮

- waiting queue 剩 `R6=4, R7=3`
- `R6` 整体接纳
- `R7` 也整体接纳

batch2:

```text
[R6(4), R7(3)]
```

#### 这个场景的关键点

- 小请求并不意味着“不会受到 chunk 逻辑影响”
- 只要 chunk 模式开启, 整个 prefill batch 的累计输入也会被 `rem_chunk_tokens` 限住

### 场景 B: 一个大请求先到, 后面接很多小请求

请求输入长度:

```text
A=20, S1=2, S2=2, S3=2
```

#### 第 1 轮

- `A=20 > rem_chunk=8`
- `A` 被截成第一块 `8`
- `adder.new_chunked_req = A`
- 本轮 `rem_chunk` 被耗尽, 小请求进不来

batch1:

```text
[A: chunk1(8)]
```

batch1 结束后:

- `A` 还是中间 chunk
- 不会产生真正的首个输出 token
- 下一轮由 `scheduler.chunked_req` 优先续跑

#### 第 2 轮

- 先续跑 `A`
- 重新匹配后, `A` 剩余 12
- 再切第二块 `8`
- 仍然没有空间给小请求

batch2:

```text
[A: chunk2(8)]
```

#### 第 3 轮

- 先续跑 `A`
- `A` 剩余 4, 这次已经是最后一块
- 加入 `A: final(4)` 后, 本轮还剩 `rem_chunk=4`
- 继续扫 waiting queue, 可把 `S1=2`、`S2=2` 放进来
- `S3` 留到下一轮

batch3:

```text
[A: final(4), S1(2), S2(2)]
```

batch3 结束后:

- `A` 这次才真正结束 prefill
- 这时才会 append 首个生成 token, 再决定是否 finished / 进入 decode

#### 这个场景的关键点

- 长请求的中间 chunk 往往会“独占”后续若干轮 prefill batch。
- 只有最后一块如果没吃满 `chunked_prefill_size`, 后面的短请求才有机会和它拼到同一轮。

### 场景 C: 很多小请求先到, 后面跟一个大请求

请求输入长度:

```text
S1=2, S2=2, B=10, S3=2, S4=2
```

#### 第 1 轮

- `S1=2` 接纳, 剩 `rem_chunk=6`
- `S2=2` 接纳, 剩 `rem_chunk=4`
- 轮到 `B=10`, 它超过剩余 `rem_chunk=4`
- 于是 `B` 不会整包延后, 而是直接在本轮被切出第一块 `4`

batch1:

```text
[S1(2), S2(2), B: chunk1(4)]
```

这说明一件很重要的事:

- 一个请求即便原始长度不算“超级长”, 只要它超过当前轮剩余的 `rem_chunk_tokens`, 也会在这一轮被切块。

#### 第 2 轮

- `B` 已经进入 `scheduler.chunked_req`
- 所以本轮会先续跑 `B`
- `B` 剩余 6, 可一次跑完
- 本轮剩余 `rem_chunk=2`, 所以 `S3=2` 能一起进

batch2:

```text
[B: final(6), S3(2)]
```

#### 第 3 轮

batch3:

```text
[S4(2)]
```

#### 这个场景的关键点

- 大请求一旦在某轮被“启动 chunk”, 后续 chunk 的优先级就会高于等待队列中的新请求。
- 所以它有明显的“开始后持续推进”特征, 而不是每轮都重新排队。

### 场景 D: 一个大请求后面再来一个大请求

请求输入长度:

```text
A=20, B=20
```

#### 第 1 轮

```text
[A: chunk1(8)]
```

#### 第 2 轮

```text
[A: chunk2(8)]
```

#### 第 3 轮

- `A` 最后一块只剩 `4`
- 本轮还有 `4` 个 chunk 预算
- waiting queue 头部是 `B=20`
- 所以 `B` 会在同一轮启动自己的第一块 `4`

batch3:

```text
[A: final(4), B: chunk1(4)]
```

#### 第 4 轮

```text
[B: chunk2(8)]
```

#### 第 5 轮

```text
[B: final(8)]
```

#### 这个场景的关键点

- SGLang 虽然同一时刻只持续追踪一个 `chunked_req`, 但在“前一个大请求的最后一轮有余量”时, 下一大请求可以在同一 batch 内启动第一块。
- 因此多个大请求不是完全 rigid 串行, 而是可能发生“最后一块 + 下一请求第一块”的接力。

### 场景 E: 多个大请求都很大, 且都超过 chunk 大小

请求输入长度:

```text
A=40, B=36, C=24
```

大致会呈现这种节奏:

```text
轮1: [A:8]
轮2: [A:8]
轮3: [A:8]
轮4: [A:8]
轮5: [A:8 或 A:last + B:first]
轮6+: 继续 B
再之后: 继续 C
```

这种情况下的整体特征是:

- 队列头部长请求一旦启动, 它的 chunk 会长期占据 prefill 机会
- 后续大请求更像在排“启动权”
- 真正的抢占点通常出现在:
  - 当前大请求的最后一块没有吃满预算
  - 或开启 priority preemption

### 场景 F: 只有一个超大请求

请求:

```text
L=100
```

如果 `chunked_prefill_size=8`, 那它会被拆成约:

```text
8 + 8 + 8 + ... + 8 + 4
```

每轮调度的状态机都是:

1. 上轮算完的 chunk 先 stash 进 cache
2. `init_next_round_input()` 让 prefix 变长
3. `add_chunked_req()` 再切下一块
4. 中间 chunk 不产出首 token
5. 最后一块才完成 prefill 语义

这是最纯粹的 chunked prefill 行为。

### 场景 G: mixed chunk, 一边 decode 一边 prefill

现在假设:

- 正在 decode 的 running batch 有 4 个请求
- `enable_mixed_chunk=True`
- `max_prefill_tokens=16`
- `chunked_prefill_size=8`

那么 `PrefillAdder` 初始化时会先扣掉 decode 占用:

```text
rem_input = 16 - 4 = 12
rem_chunk = 8 - 4 = 4
```

这时如果来了一个新请求 `N=10`, 本轮只能先给它 4 个 token 的 prefill 空间。

然后在:

- `scheduler.py:2715-2733`
- `schedule_batch.py:2217-2246`

里, 这个 prefill chunk batch 会和 running decode batch 合并成 `ForwardMode.MIXED`。

这个场景的关键点:

- mixed chunk 的吞吐优化不是“额外塞入 prefill”
- 而是“在 decode 已占预算之后, 用剩余 token 空间再塞一段 prefill”

---

## 7. 几个特别容易看漏的实现细节

### 7.1 `chunked_prefill_size` 开启后, 会限制所有 prefill 请求的累计量

不是只有长请求才受它影响。

很多小请求的 batch 也一样会先被 `rem_chunk_tokens` 卡住。

### 7.2 中间 chunk 不会产生用户可见的首 token

代码在:

- `batch_result_processor.py:223-289`

只有最后一个 chunk 才会:

- `req.output_ids.append(next_token_id)`

所以用户感知上的 TTFT, 本质上仍然要等到最后一块 prefill 跑完。

### 7.3 同一时刻只有一个“持续续跑”的 chunked 请求

因为 scheduler 只有一个:

```text
self.chunked_req
```

这会让长请求表现出明显的“启动后持续推进”行为。

### 7.4 实际 chunk 大小会被 page 对齐缩小

代码:

- `schedule_policy.py:931-950`

所以理论上的 `chunked_prefill_size=4096` 不代表每轮一定精确跑 4096 个新 token。

### 7.5 cache 命中会改变预算感知

`Req.init_next_round_input()` 会先做 prefix match:

- `schedule_batch.py:1079-1131`

而 `add_one_req()` 又会考虑:

- `prefix_indices`
- `host_hit_length`
- `cache_protected_len`

所以实际“新算 token 数”常常小于原始输入长度。

这也是为什么同样长度的两个请求, 如果一个 prefix hit 多, 它更容易被塞进 batch。

---

## 8. 可以把整个控制流记成这个心智模型

```text
waiting_queue
  -> policy 排序
  -> 先续跑 chunked_req
  -> 再扫描新请求
  -> 按 rem_input_tokens / rem_chunk_tokens / rem_total_tokens 决定:
       - 整体接纳
       - 截成 chunk
       - 本轮停止
  -> 形成 ScheduleBatch
  -> prepare_for_extend
  -> forward
  -> 如果是中间 chunk:
       - stash 到 cache
       - 下轮继续
     如果是最后 chunk:
       - 追加首 token
       - 完成 prefill
       - 进入 decode 语义
```

如果只记一个最重要的判断准则, 那就是:

```text
chunked prefill 打开后,
真正控制“这一轮 prefill 能塞多少”的,
往往不是 max_prefill_tokens,
而是 rem_chunk_tokens 对应的 chunked_prefill_size。
```

---

## 9. 关键代码定位索引

- `ServerArgs` 参数定义:
  - `python/sglang/srt/server_args.py:404-411`
- chunked prefill 初始化:
  - `python/sglang/srt/managers/scheduler.py:872-909`
- 调度主入口:
  - `python/sglang/srt/managers/scheduler.py:2476-2735`
- 上轮 chunk stash / merge:
  - `python/sglang/srt/managers/scheduler.py:2309-2408`
- `PrefillAdder` 预算初始化:
  - `python/sglang/srt/managers/schedule_policy.py:413-479`
- 预算扣减:
  - `python/sglang/srt/managers/schedule_policy.py:582-603`
- 续跑已有 chunk:
  - `python/sglang/srt/managers/schedule_policy.py:666-699`
- 新请求接纳 / 截断:
  - `python/sglang/srt/managers/schedule_policy.py:813-965`
- 请求重建 prefix / extend:
  - `python/sglang/srt/managers/schedule_batch.py:1043-1131`
- `extend_input_len` 语义:
  - `python/sglang/srt/managers/schedule_batch.py:1387-1403`
- batch 组织 extend 输入:
  - `python/sglang/srt/managers/schedule_batch.py:1808-1905`
- mixed chunk 合批:
  - `python/sglang/srt/managers/schedule_batch.py:2217-2246`
- 中间 chunk vs 最后 chunk 的结果处理:
  - `python/sglang/srt/managers/scheduler_components/batch_result_processor.py:223-289`

