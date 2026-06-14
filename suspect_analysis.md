# Decode 节点 Stream Hang 可疑代码定位报告

## 0. 背景

- **硬件 / 平台**：昇腾 A3
- **模型**：DeepSeek V3.1
- **部署**：PD 分离（Prefill / Decode 分离）
- **特性开关组合**：MTP=2 + FULL_DECODE_ONLY + lmhead + recompute + 多流(Multi-stream) + KV Cache 池化(Pooling)
- **测试**：GSM8K 精度推理
- **现象**：D（Decode）节点 Stream Hang
- **关键消元线索**：**关闭 AIV（矢量核）后恢复正常**

### 版本基线
| | vLLM | vllm-ascend |
|---|---|---|
| OK Base | `v0.20.2` @ `bc150f50299199599673614f80d12a196f377655` | `main` @ `ff4807eafa60cf588c7e6c3d4baa767a104c27af` |
| NG Target | HEAD `78e7293bb`（+1499 commits） | HEAD `e01af0f0`（+118 commits） |

---

## 1. 根因方向画像（症状反推）

"关 AIV 即恢复"是最强信号，把根因方向收敛到：

> **AIV 侧异步算子下发后，与下游消费方/池化归还/多流捕获之间缺少正确的同步事件 → 异步队列上的悬挂任务永远等不到 record → 整流 Hang。**

具体可拆成三类失败模式：

1. **AIV kernel 写 KV block，与 attention 读 KV、与 pool 回收之间缺 `record_stream/wait_event`** → 数据竞争或非法地址。
2. **`npu_stream_switch` 上下文里下发 AIV 算子，退出时只 record 主流 event 没 record AIV 完成 event** → aclgraph 捕获期固化漏 wait，回放永远 hang。
3. **池化复用 KV block 时 AIV 算子还在异步使用** → 写入触发 HBM 异常 → Stream abort 但上层未感知。

---

## 2. 可疑 commit 总览（按耦合度分级）

### 🚨 Tier S — 头号嫌疑（直接命中所有触发条件）

#### **#1 `caf58a20` — [BugFix] Add AscendKVBlockZeroer to clean used mamba block for full attn (#10087)**

**与症状画像完全契合的 smoking-gun 候选**：

| 证据 | 详情 |
|---|---|
| 是 Triton kernel | `@triton.jit _zero_kv_blocks_kernel` 定义于 `vllm_ascend/worker/utils.py`；Triton 在昇腾后端**就是跑在 AIV（vector cores）上** |
| 直接绑定 AIV | `grid = min(total_work, get_vectorcore_num())` —— grid 直接来自 AIV 核数。关闭 AIV 必然让该 kernel 完全失活 |
| 触发条件 = MTP=2 | `vllm_ascend/worker/worker.py:768` 中：`if ... speculative_config.num_speculative_tokens > 1: self.model_runner._init_kv_zero_meta()` —— 与你的 MTP=2 完美对应 |
| 用绝对地址写入 | `seg_addrs = kv.data_ptr() + off_bytes` 在 `init_meta` 时一次性算好；kernel 用 `tl.cast(seg_addr, tl.pointer_type(tl.int32))` 直接写。**KV pool 若发生重新分配/迁移（PD 切换、recompute）则指针悬空 → 野写 → AIV 硬件异常 → Hang** |
| 异步下发缺同步 | kernel launch 与 attention 读 KV 没有显式 event 同步，依赖默认 stream 序；多流 / aclgraph 捕获回放下次序可能错乱 |
| 与新代码强耦合 | 调用点在 vLLM 上游 `gpu_model_runner.py:1153` 的 `_update_states`，每 step 对 `new_block_ids_to_zero` 调用；写 KV 与下一步 attention 读 KV 之间窗口非常窄 |

**覆盖你所有的开关轴**：
- AIV ✅（Triton kernel）
- MTP=2 ✅（触发条件）
- 池化 ✅（与 block 复用直接耦合）
- FULL_DECODE_ONLY ✅（aclgraph 内 race 概率大）
- recompute ✅（block 复用频次高，悬空指针窗口大）
- 多流 ✅（与 attention 主流共享 KV，无显式同步）

**走查重点**：
- `vllm_ascend/worker/utils.py:53` `AscendKVBlockZeroer.zero_block_ids`
- `vllm_ascend/worker/utils.py:13` `_zero_kv_blocks_kernel`（重点验证 `PAGE_SIZE_EL` 在 MLA / sparse C8 layout 下的正确性）
- `vllm_ascend/worker/worker.py:768` 触发条件分支
- `vllm_ascend/worker/model_runner_v1.py:4852` `_init_kv_zero_meta`

**验证方法**：
- 把 `_init_kv_zero_meta()` 注释掉，或把 `zero_block_ids` 改成 no-op / host-side `torch.zeros_(...)`
- 若 hang 消失，即锁定本 PR

---

### 🔥 Tier A — 高度怀疑

#### **#2 `595c6c77` — [BugFix] Fix bug from cudagraph config mode FULL corner case (#9863)**

- **直接修改 `CUDAGraphMode.FULL` + `num_spec > 1` 的 padding 流程**（`vllm_ascend/worker/model_runner_v1.py` 中新增针对 FULL 模式的 early-exit 分支）
- 你启了 `FULL_DECODE_ONLY` + `MTP=2`，这个分支必走
- **风险**：新的早退分支可能让某个本该捕获到 graph 内的同步算子被跳过 → graph 回放时少一次 event wait → Hang
- 代码：
  ```python
  if (
      num_tokens_padded == num_reqs_padded * self.uniform_decode_query_len
      and self.compilation_config.cudagraph_mode != CUDAGraphMode.FULL  # 新增
  ):
      assert num_reqs <= num_reqs_padded
  ```

**验证方法**：临时去掉 `and self.compilation_config.cudagraph_mode != CUDAGraphMode.FULL` 条件，看是否在 FULL 模式恢复同步行为。

---

#### **#3 `941ba056` — [Feature] Enable weight offload to Ascend NPU via ACL graph execution (#10251)**

- 即使你**没开** `offload_backend=prefetch`，这个 PR 同时**修改了 `vllm_ascend/compilation/acl_graph.py`** —— 给 ACL graph capture 块加了 `sync_prev_onload()` / `join_after_forward()`
- 这个 patch 本身就是 fix `NPU error 107025: capture_end: stream not joined to original stream`
- **意义**：证明近期代码确有 **"侧流捕获完成事件缺失"** 这一类 bug，可能在 KVBlockZeroer / MTP 副流 / lmhead 副流上仍有未补的同类 case
- 即便逻辑上未触发 offloader 路径，对 `acl_graph.py` 的修改本身就可能影响 capture 行为

**走查重点**：`vllm_ascend/compilation/acl_graph.py:195~`（capture 块结构）

---

#### **#4 `1f0f1119` — [BugFix][Performance] Fix MTP copy_valid_sampled_token_count sync (#10205)**

- **显式去掉了 MTP 路径上一次"由 dtype mismatch 触发的隐式 host 阻塞同步"**
- 修复前：`valid_sampled_token_count_cpu (int32)` ↔ `valid_sampled_token_count_gpu (int64)` 的 `copy_` 因隐式 cast 触发 host sync
- 修复后：dtype 统一为 int64，sync 消失
- **风险**：**之前的隐式同步顺带把整条流序列化了；去掉之后，下游算子（包括 AIV kernel）开始真正并发** —— 之前 race 被同步掩盖，现在暴露
- 修改文件 `vllm_ascend/worker/model_runner_v1.py:1593` 也涉及 `valid_sampled_token_count_copy_stream` 这条独立 stream，跟多流深度耦合

**验证方法**：回滚此 PR（恢复 int32/int64 mismatch），让 host sync 重新出现 —— 若 hang 消失，说明根因是 MTP 副流 / AIV 算子的 race，本 PR 仅是 trigger。

---

### ⚠️ Tier B — 间接相关，二线走查

#### **#5 `ee88e356` — [BugFix][CP] Rebuild stale decode state after async token correction (#10032)**
- 重建 slot mapping（即送给 attention 的 KV 地址表）
- 涉及 CP+MTP 路径
- 若新的重建与 KV pool 回收时序不一致，attention 可能读到错块

#### **#6 `9099b7f6` — [Feature] Estimate ACL graph memory before KV cache allocation (#9865)**
- 改了 KV cache 分配与 graph capture 的时序
- 受 `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS` 控制
- 若改变了 KV 地址在 graph capture 时的取值，会影响所有抓进 graph 的 KV 指针

#### **#7 `5f2ef5a0` — [BugFix] Remove legacy capture-size pruning `update_aclgraph_sizes` (#9962)**
- 解除了 capture size 上限
- 会捕获**更多 graph 实例**，可能暴露之前被规避的多流 bug
- 在你的 batch 范围下可能新增了某个 capture size，该 size 下隐藏的同步问题首次显现

#### **#8 `1a44dc5f` — [BugFix][SpecDecode] Sanitize MTP placeholders before model forward (#10307)**
- 在 **PD kv-consumer 首步原地改写 input_ids**（在 D 节点上！）
- 修改了 `self.input_ids.gpu[:num_forward_tokens]` 这个 device 张量
- 若该写入与 graph capture 内的读取存在时序冲突，可能导致首 step Hang

#### **#9 `a53f8a02` — [Feature] add ascendc ops store_kv_block (#10292)**
- 新增自定义 AscendC kernel（运行在 AIV/AIC）
- PR 自称"仅 P 节点使用"（受 `c8_enable_reshape_optim` 控制）
- **必须 grep 验证**：`grep -rn store_kv_block vllm_ascend/` 确认 D 路径不会触发
- 文件：`csrc/attention/store_kv_block/op_kernel/store_kv_block.cpp`

#### **#10 `7ea0447f` / `f88361de` — RecomputeScheduler 相关修复**
- `7ea0447f`: add recompute scheduler to fix mpt+pcp shape error (#10357)
- `f88361de`: Fix AttributeError in RecomputeScheduler due to missing routed experts extraction logic (#10275)
- recompute 路径修改，可能影响重计算时 block 复用时序

#### **#11 `9ef1ce6d` — [Feature] [redo] Cache the code start compilation for npugraph_ex (#9914)**
- 修改了 AscendCompiler 的 cache 协议
- 若 cache 命中时跳过了某些 stream 注册步骤，graph 内同步关系可能不全

#### **#12 `2b4a9daa` / `72797cfa` — Main2Main 大版本号 bump**
- `2b4a9daa`: Main2main Upgrade vLLM to v0.21.0 (#9835)
- `72797cfa`: Main2Main 0605 (#10250)
- 带来 1499 个 vLLM 上游 commit，KV pool / scheduler / lmhead / sampling 行为大幅改动
- 一旦 Tier S/A 都排除，需要在 vLLM 上游做二分

---

## 3. 推荐排查顺序（按 ROI 排序）

| 步骤 | 动作 | 判据 |
|---|---|---|
| **1** | **回滚 `caf58a20`** —— 注释 `_init_kv_zero_meta()`，或把 `AscendKVBlockZeroer.zero_block_ids` 改成 no-op | 不挂死 → 锁定 #10087；仍挂 → 继续 |
| **2** | **回滚 `595c6c77`** —— 恢复 padding 早退判断（去掉 `cudagraph_mode != CUDAGraphMode.FULL` 条件） | 不挂死 → 锁定 #9863 |
| **3** | **回滚 `1f0f1119`** —— 恢复 int32/int64 mismatch 让隐式 host sync 重新出现 | 不挂死 → 锁定 #10205 触发的 MTP 副流 race |
| **4** | **验证 `941ba056`** —— 检查 `acl_graph.py` 的 `sync_prev_onload` / `join_after_forward` 在你的 offloader=Noop 路径下是否还是 no-op | 若非 no-op，回滚 acl_graph.py 部分 |
| **5** | **逐一回滚 Tier B**（#5 → #11） | 每回滚一个测试一次 |
| **6** | **二分 vllm 1499 个 commit** —— 重点过 `gpu_model_runner.py:1153 _zero_block_ids` 的引入 commit、`KVBlockZeroer` 基类引入、async-spec-decode、KV pool / scheduler 相关 PR | — |

---

## 4. 配套的开关二分实验（不改代码）

每行只动一个开关，跑 GSM8K：

| # | MTP | FULL_DECODE_ONLY | lmhead | recompute | 多流 | 池化 | AIV | 预期 |
|---|---|---|---|---|---|---|---|---|
| 0 | 2 | on | on | on | on | on | on | hang（已知）|
| 1 | 2 | on | on | on | on | on | **off** | ok（已知）|
| 2 | **1** | on | on | on | on | on | on | 若 ok → 锁定 MTP 路径 → 强化怀疑 #1/#3/#4/#8 |
| 3 | 2 | on | **off** | on | on | on | on | 若 ok → 锁定 lmhead 路径 |
| 4 | 2 | on | on | **off** | on | on | on | 若 ok → 锁定 recompute / pool 时序 → 强化怀疑 #1/#10 |
| 5 | 2 | on | on | on | **off** | on | on | 若 ok → 锁定 multi-stream sync → 强化怀疑 #3/#4 |
| 6 | 2 | on | on | on | on | **off** | on | 若 ok → 锁定池化归还窗口 → 强化怀疑 #1 |
| 7 | 2 | **off** | on | on | on | on | on | 若 ok → 锁定 FULL graph 路径 → 强化怀疑 #2/#3 |

跑完 #2/#3/#5/#6/#7 这五个，可把根因压到一个 commit 范围内。

---

## 5. 一句话结论

> **最高优先级嫌疑：`caf58a20` (#10087 AscendKVBlockZeroer)** —— 它同时满足"必走 AIV、必在 MTP≥2 时启用、对 KV pool 内 block 做异步写、与 attention 读 KV 之间缺少强同步"四个条件，几乎是按你的现象画像反推出来的代码。**强烈建议第一步先 no-op 这一笔 zeroer 再复现一次。**

如能补充以下信息可继续缩小范围：
1. 回滚 `caf58a20` 后的复现结果
2. D 节点挂死时 `npu-smi info -t cur-runtime-info` 的算子队列状态
3. aclgraph capture 阶段是否有 stderr/warning 输出
4. 是否启用 `additional_config = {"enable_sparse_c8": true, "c8_enable_reshape_optim": true}`（关系到 #9 是否在 D 路径触发）

---

## 附录 A：完整可疑 commit 速查表

| 等级 | Commit | 标题 | 触发轴 |
|---|---|---|---|
| S | `caf58a20` | Add AscendKVBlockZeroer (#10087) | AIV + MTP + 池化 + recompute |
| A | `595c6c77` | Fix cudagraph FULL corner case (#9863) | FULL_DECODE_ONLY + MTP |
| A | `941ba056` | Weight offload via ACL graph (#10251) | aclgraph + 多流同步 |
| A | `1f0f1119` | Fix MTP copy sync (#10205) | MTP + 多流 + sync 移除 |
| B | `ee88e356` | Rebuild stale decode state (#10032) | MTP + CP + slot mapping |
| B | `9099b7f6` | Estimate ACL graph memory (#9865) | KV cache 分配时序 |
| B | `5f2ef5a0` | Remove capture-size pruning (#9962) | aclgraph 数量 |
| B | `1a44dc5f` | Sanitize MTP placeholders (#10307) | PD D 节点首步 |
| B | `a53f8a02` | add ascendc store_kv_block (#10292) | AscendC AIV kernel |
| B | `7ea0447f` | recompute scheduler (#10357) | recompute |
| B | `f88361de` | Fix RecomputeScheduler (#10275) | recompute |
| B | `9ef1ce6d` | Cache npugraph_ex compilation (#9914) | aclgraph cache |
| B | `2b4a9daa` | Main2main vLLM v0.21.0 (#9835) | 全量 vLLM 升级 |
| B | `72797cfa` | Main2Main 0605 (#10250) | 全量 vLLM 升级 |
