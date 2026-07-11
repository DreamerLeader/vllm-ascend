## Bug 分析报告

### 一、结论

根因可以高置信度定位为：

> DP 空闲 rank 进入 runtime `_dummy_run(1)` 时没有经过 `_prepare_inputs()`，因此没有生成本轮 `slot_mapping`；PR #10741 又删除了原先针对 DSA dummy run 的残留值防护，导致 dummy forward 使用上一轮真实请求遗留的 slot，向仍被请求占用或已重新分配的 KV block 写入无意义 KV。

这与“DP=2 更容易出现”“KV cache usage 越高越容易复现”完全吻合。

不过更准确地说，最确定被污染的是 DeepSeek V4 的 SWA KV cache；不一定首先是 c4/c128 compressed cache。

触发回归的提交是：

- `cbfe0e14716c92c963af3903bc323c86c08f40e7`
- [PR #10741](https://github.com/vllm-project/vllm-ascend/pull/10741)

最新主线已经出现针对同一根因的修复提交：

- `b9acec8bc4318f04e425800e265a3a904f6bc82f`
- [#11774 Reset slot_mapping to pad id for dummy graph capture](https://github.com/vllm-project/vllm-ascend/commit/b9acec8bc4318f04e425800e265a3a904f6bc82f)

这基本属于上游代码对根因的直接确认。

### 二、完整触发链

1. DP 空 rank 会执行 runtime dummy forward

在 `external_launcher + DP>1` 下，本 rank 没有 token、其他 DP rank 仍有任务时，代码主动调用：

```python
self._dummy_run(1)
```

见 [model_runner_v1.py](/Users/fangjianwei/Documents/Codex/2026-07-11/vllm-project-vllm-ascend-10741-https-2/work/vllm-ascend/vllm_ascend/worker/model_runner_v1.py:2166)。

这是 DP=2 场景特有的高频触发器，TP=16 本身不是原因；但所有 TP rank 会在相同逻辑 slot 上写各自的 KV shard，使该 slot 整体被污染。

2. 真实 forward 会计算 slot，dummy forward 不会

真实请求通过 `_prepare_inputs()` 调用：

```python
block_table.compute_slot_mapping(...)
```

见 [model_runner_v1.py](/Users/fangjianwei/Documents/Codex/2026-07-11/vllm-project-vllm-ascend-10741-https-2/work/vllm-ascend/vllm_ascend/worker/model_runner_v1.py:1339)。

但 `_dummy_run()` 不经过 `_prepare_inputs()`，只直接构建 attention metadata。因此 `blk_table.slot_mapping.gpu` 的有效部分仍是上一轮真实 forward 的结果。

3. PR #10741 删除了原有的 DSA dummy 防护

#10741 之前，代码明确处理过这个问题：

```python
elif self.use_compress:
    # DSA dummy/graph-capture runs do not go through
    # _prepare_inputs(), so no fresh compressed cache
    # metadata is computed for them. Reusing values from
    # the previous real request can feed stale block-table
    # and [block, offset] scatter indices to DSA kernels.
    slot_mapping[:num_tokens_padded].fill_(0)
    blk_table_tensor[:num_reqs_padded].fill_(0)
```

#10741 把 compressed metadata 移到设备计算后，连同这个分支一起删除，变成只清理 padding 尾部：

```python
slot_mapping[num_tokens:num_tokens_padded].fill_(-1)
```

runtime `_dummy_run(1)` 中 `num_tokens=1`，所以索引 0 不会被清理，恰好继续保存上一次真实请求的 slot。

4. DeepSeek V4 SWA builder 会复制这个残留值

DSA metadata builder 在模型 forward 前把公共 slot mapping 复制进私有持久 buffer：

```python
slot_mapping = common_attn_metadata.slot_mapping[:num_input_tokens]
self.slot_mapping[:num_input_tokens] = \
    DeviceOperator.format_dsa_slot_mapping(slot_mapping, self.block_size)
```

见 [dsa_v1.py](/Users/fangjianwei/Documents/Codex/2026-07-11/vllm-project-vllm-ascend-10741-https-2/work/vllm-ascend/vllm_ascend/attention/dsa_v1.py:605)。

随后 SWA cache 无条件按照该 slot scatter：

- Prefill 路径：[dsa_v1.py](/Users/fangjianwei/Documents/Codex/2026-07-11/vllm-project-vllm-ascend-10741-https-2/work/vllm-ascend/vllm_ascend/attention/dsa_v1.py:1932)
- Decode 路径：[dsa_v1.py](/Users/fangjianwei/Documents/Codex/2026-07-11/vllm-project-vllm-ascend-10741-https-2/work/vllm-ascend/vllm_ascend/attention/dsa_v1.py:2249)

因此 dummy hidden state 生成的 KV 会覆盖残留 slot 指向的真实 SWA block。

### 三、为什么高 KV cache usage 更容易出现

高占用会同时增加：

- 残留 slot 仍属于活跃请求的概率；
- 已释放 block 很快被重新分配给另一个请求的概率；
- chunked prefill、调度停顿、preemption 带来的 DP 不均衡；
- 空闲 DP rank 执行 runtime dummy 的次数。

所以它不是数值随机误差，而是“是否撞到正在使用的 physical block”的时序概率问题。

乱码偶发、负载越高越严重，正是这种内存污染的典型表现。

### 四、#10741 的 device compressor op 是否本身算错

目前没有证据表明 `CompressorMetadata` 的压缩位置公式本身错误。

对于 `_dummy_run(1)`：

- `start_pos = 0`
- `seq_len = 1`
- `compress_ratio = 4/128`

kernel 计算有效压缩行数：

```cpp
((startPos + seqLen) / cmpRatio) - (startPos / cmpRatio)
```

结果为 0。因此标准的单 token DP dummy 下，c4/c128 compressed cache 通常不会产生有效写入；padding 行会得到 `slot_mapping=-1`。

所以更直接的污染点是 SWA cache。#10741 的问题在于：迁移 compressed metadata 时，错误地删除了覆盖整个 DSA cache group 的 dummy 安全处理。

### 五、对最新 #11774 修复的审查

最新主线新增了：

```python
if not is_graph_capturing:
    for kv_cache_gid in range(...):
        blk_table.slot_mapping.gpu.fill_(-1)
```

但它目前位于 `_build_attention_metadata()` 之后，见 [model_runner_v1.py](/Users/fangjianwei/Documents/Codex/2026-07-11/vllm-project-vllm-ascend-10741-https-2/work/vllm-ascend/vllm_ascend/worker/model_runner_v1.py:3671)。

这对直接引用公共 slot buffer 的普通 attention 有效，但对 DSV4 DSA 可能仍不完整：DSA builder 已经在构建 metadata 时把旧值复制到了自己的 `self.slot_mapping`，之后再清公共 buffer 不会修改该私有副本。

因此：

- 如果运行的是 `b9acec8b` 之前的主线：根因已明确，就是 #10741 删除 dummy 防护。
- 如果已经包含 #11774 仍可复现：优先怀疑清理时序过晚，DSA 私有 slot mapping 仍然残留。

upstream vLLM 同版本的实现是在 `_build_attention_metadata()` 之前把所有 dummy slot mapping 填成 `-1`，这也是更合理的生命周期顺序。

### 六、建议的最小修正方向

未在工作区应用修改。建议把 #11774 的清理移动到 metadata 构建之前：

```diff
 self.input_batch.block_table.commit_block_table(num_reqs_padded)

+if not is_graph_capturing:
+    for kv_cache_gid in range(len(self.kv_cache_config.kv_cache_groups)):
+        blk_table = self.input_batch.block_table[kv_cache_gid]
+        blk_table.slot_mapping.gpu.fill_(-1)
+
 attn_metadata, _ = self._build_attention_metadata(...)
-
-if not is_graph_capturing:
-    ...
```

这样 DSA builder 复制到私有 buffer 的已经是 PAD slot，而不是上一轮真实请求地址。

### 七、推荐验证方法

最有判别力的验证矩阵：

1. `DP=1`：问题应基本消失，因为不会进入空 DP rank 的 runtime dummy。
2. `DP=2 + external_launcher`，人为制造单 rank 空闲：复现率应明显提高。
3. `--enforce-eager`：若不构建 FULL graph dummy attention metadata，复现率应显著下降。
4. 在 dummy forward 前后 dump：
   - 公共 `blk_table.slot_mapping.gpu[0]`
   - DSA builder 的 `self.slot_mapping[0]`
   - 对应 SWA KV block checksum
5. 将清理移动到 `_build_attention_metadata()` 前：
   - dummy 前后所有已分配 SWA block checksum 应保持不变。
6. 对比 `cbfe0e14^`、`cbfe0e14`、`b9acec8b` 三个版本，可以准确验证回归和修复覆盖情况。

还有一个重要判别条件：如果关闭 prefix cache，且 prompt 能在单个 prefill step 完成，那么“第一生成 token 就乱码”不完全符合单纯的历史 KV 污染；因为本轮 prefill 会重新写入 prompt KV。此时应重点检查是否存在 chunked prefill、prefix-cache 命中，或者 #11774 清理过晚导致同一活跃请求的上一 prefill chunk 被覆盖。
