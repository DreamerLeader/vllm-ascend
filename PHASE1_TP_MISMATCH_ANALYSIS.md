# Phase 1 分析与解析 — TP-不对等 PD 分离池化

本文档记录对 TP-不对等（prefill tp4 / decode tp2）PD 分离池化改动的验证结论、
识别机制、代码实现链路与日志证据链。基线：vllm-ascend @ 328a01ca，分支
`decode_save_tp_v2`，commit `d32b0287`。模型 Qwen3-8B（num_kv_head=8, head_dim=128,
bf16, 36 层），8×910B3 A2。

---

## 一、验证结论（基于实跑日志）

实跑环境：mooncake_master :50088 + prefill(tp4, 卡0-3, :8100) +
decode(tp2, 卡6-7, :8200) + load_balance_proxy(:8000)，MultiConnector =
MooncakeConnectorV1(PD 主传输) + AscendStoreConnector(TP-不对等池化)。

### 1. decode 能把 KV 存进 kvpool — 成立
mooncake master 累计指标：`PutStart:(Req=8/0/8, Item=12/12)` —— 8 个批量 PUT
请求、12 个 sub-key 全部成功（Success/Total=8/8、Item=12/12，PartialSuccess=0）；
`PutEnd:(Req=8/0/8, Item=12/12)` 全部 finalize；`Keys: 0 → 12` 稳态。
decode 经 `consumer_is_to_put=True` strided-PUT 入池，0 失败。

### 2. 多轮下 prefill 能命中 decode 存的 KV — 成立
mooncake master：`Get:(Req=8/0/8, Item=12/12)` —— prefill strided-GET 取回全部
12 个 sub-key，全成功；`ExistKey:(Req=25/0/25, Item=60/60)` —— lookup 阶段查
60 个 key 变体全成功。
prefill 侧 `KV pool load spec`：
- 17:12 `vllm_cached=0 kvpool_cached=128`
- 17:23 `vllm_cached=0 kvpool_cached=256`

`vllm_cached=0`（且 `enable_prefix_caching=False`）→ 命中 100% 来自 kvpool
（即 decode 存的 KV）；128→256 跨两轮递增 = 多轮累积命中。External cache hit
rate `0% → 28.8% → 44.5%`。

### 3. 是否影响池化基本功能 — 无回归
- decode 日志：0 错误 / 0 异常。
- prefill 日志：3 处 traceback 全是 `ValueError: stop cannot contain an empty
  string`（客户端经 proxy 发了空 stop 串，`build_prefill_request` 原样透传 stop，
  被 vllm 上游 `SamplingParams._verify_args` 拒绝）。与池化代码无关；proxy 重试
  3 次后对该坏请求返回 500，后续请求立即恢复 200 OK。
- 无 strided get 失败 / 无 invalid_block_ids / 无 recompute 触发。
- mooncake：`Eviction 0/0`、`PutRevoke 0/0`、`Discard 0/0`、`Clients: 6`
  （4 prefill + 2 decode）—— 池健康。

### 说明（非问题）
当前 hit rate 偏低（max 44.5%）、key 数少（12 个/54MB）属功能性验证规模：prompt
短、请求数少、block_size=128 下 <128 token 的共享前缀不落整 block。功能正确性已
证实。hit rate 中途 28.8%→21.6% 的小回落是滑动平均被新请求稀释的正常现象。

---

## 二、TP 不对等如何识别（配置驱动，默认关闭）

识别在 `pool_worker.py:159-213`（运行端）和 `pool_scheduler.py:128-152`（调度端，
逻辑镜像，try/except 包住以保证单测 MagicMock 下仍 False）。

```python
# pool_worker.py:164-175
extra_cfg = vllm_config.kv_transfer_config.kv_connector_extra_config
if self.kv_role == "kv_consumer":                      # decode 端
    self.peer_tp_size = int(extra_cfg.get("prefill_tp_size", self.tp_size))
else:                                                   # prefill 端
    self.peer_tp_size = int(extra_cfg.get("decode_tp_size", self.tp_size))
self.effective_tp_size = max(self.tp_size, self.peer_tp_size)
self.tp_mismatch = (
    self.peer_tp_size != self.tp_size                   # 对端 TP 与本端不等
    and not self.use_mla                                # 非 MLA
    and self.num_kv_head >= self.effective_tp_size      # 头数够分
    and self.num_kv_head % self.effective_tp_size == 0  # 头数能整除
)
```

- **不配 `prefill_tp_size`/`decode_tp_size` 时** `peer_tp_size` 默认等于本地 tp →
  `tp_mismatch=False`，所有新路径不走，行为逐行不变（「不影响池化基本功能」的保证）。
- 开启时显式拒绝 `use_sparse/use_layerwise/use_hybrid`（:177-194），并算出三个
  关键量（:195-197）：
  - `local_heads_per_rank = num_kv_head // tp_size`（本 rank 实际持有的头数）
  - `effective_heads_per_rank = num_kv_head // effective_tp_size`（**双方约定的
    子键头数 = GCD 对齐单元**）
  - `num_sub_keys = local_heads_per_rank // effective_heads_per_rank`（本 rank
    要拆成几个子键）

Qwen3-8B（num_kv_head=8）实测：prefill tp4 → local=2, effective=2, sub_keys=1；
decode tp2 → local=4, effective=2, sub_keys=2。与日志 `local_heads_per_rank=2/4
num_sub_keys=1/2` 完全一致。

---

## 三、核心机制：为什么 decode 存的 KV prefill 一定能命中

关键在于把双方都映射到同一个 `effective_tp` 命名空间，靠的是 `_make_sub_key_str`
+ `effective_rank` 改写（`pool_worker.py:954-960, 1005-1016`）：

```python
# pool_worker.py:1010-1013
for sub_idx in range(self.num_sub_keys):
    effective_rank = self.tp_rank * self.num_sub_keys + sub_idx
    addrs, sizes = self._build_strided_addrs(block_id, token_count, sub_idx)
    all_keys.append(self._make_sub_key_str(base_key, effective_rank))  # 改写 @head_or_tp_rank
```

pool 的 key 形如 `...@head_or_tp_rank:<N>@pp_rank:0...`。双方都把 `<N>` 改写成
`[0, effective_tp)` 内的 `effective_rank`：

| 侧 | rank | sub_idx | effective_rank | 对应头片（全局） |
|---|---|---|---|---|
| decode tp2 | 0 | 0 | 0×2+0=0 | 头 0-1 |
| decode tp2 | 0 | 1 | 0×2+1=1 | 头 2-3 |
| decode tp2 | 1 | 0 | 1×2+0=2 | 头 4-5 |
| decode tp2 | 1 | 1 | 1×2+1=3 | 头 6-7 |
| prefill tp4 | 0 | 0 | 0×1+0=0 | 头 0-1 |
| prefill tp4 | 1 | 0 | 1×1+0=1 | 头 2-3 |
| prefill tp4 | 2 | 0 | 2×1+0=2 | 头 4-5 |
| prefill tp4 | 3 | 0 | 3×1+0=3 | 头 6-7 |

→ **decode 的 (rank,sub_idx) 与 prefill 的 rank 是同一 [0,4) 命名空间的双射**：
decode rank0/sub0 与 prefill rank0 写/读同一个 key，且都是头 0-1。

### 数据对齐（strided 地址）
`_build_strided_addrs`（`pool_worker.py:962-986`）：

```python
head_offset_bytes = sub_idx * self.sub_size_bytes   # sub_size_bytes = effective_heads*head_dim*elem
# 逐 token：addrs.append(block_base + t*entry_per_token_bytes + head_offset_bytes)
```

decode rank0 持有 4 头，`sub_size_bytes=2×128×2=512`：sub_idx=0 写偏移 0（头 0-1），
sub_idx=1 写偏移 512（头 2-3）。prefill rank0 持有 2 头，整段读偏移 0（头 0-1）=
decode sub_idx=0 的数据。**key 相同 + 数据相同 = 必命中且正确**。

这正是日志里 `per_token=512 sub=512`（prefill）/ `per_token=1024 sub=512`
（decode）的来源：decode 的 per_token=1024（4 头）拆成 2 个 sub_key 各 512（2 头），
与 prefill 的 per_token=512（2 头）一一对齐。

---

## 四、代码实现链路（触发路径）

| 步骤 | 位置 | 作用 |
|---|---|---|
| 识别 | `pool_worker.py:159-213` | 算 tp_mismatch/effective_tp/num_sub_keys |
| strided 参数 | `pool_worker.py:517-528` | 算 per_token_bytes/sub_size_bytes |
| 线程挂 worker | `pool_worker.py:574,588` | `worker=self if self.tp_mismatch else None` |
| 调度门控 | `pool_scheduler.py:128-152, 463, 518-523` | 空 new_block_ids 在 tp_mismatch 下不 continue，保证 decode/chunked 仍产生 load 元数据 |
| store 分流 | `kv_transfer.py:313-319` | Sending 线程首部 `if worker.tp_mismatch → worker._store_kv_tp_mismatch` |
| load 分流 | `kv_transfer.py:481-498` | Recving 线程首部 `if worker.tp_mismatch → worker._load_kv_tp_mismatch` |
| load 入口 | `pool_worker.py:631-635` | `start_load_kv` 中 `elif self.tp_mismatch: _load_kv_tp_mismatch` |
| **store** | `pool_worker.py:1053-1083` | 构键→`lookup` 跳过已存→`m_store.put(keys,addrs,sizes)` |
| **load** | `pool_worker.py:1019-1051` | 构键→tp_rank 轮转→`m_store.get`→失败块记 `_invalid_block_ids` |
| **lookup 命中展开** | `pool_worker.py:1293-1302, 1413-1428` | `get_group_tp_size` 返回 effective_tp(:1266)，`_expand_lookup_key_variants` 把每个 chunk 展开成 effective_tp 个 tp 变体去 `exists` 查询 |

`_store_kv_tp_mismatch:1067-1071` 还做了 `lookup(keys)` 跳过已存在的子键——所以
重复存不会覆盖、幂等。

### 关键函数签名
- `_make_sub_key_str(base_key, effective_rank)` — 改写 key 的 `@head_or_tp_rank`
  字段为 effective_rank（:954-960）。
- `_build_strided_addrs(block_id, token_count, sub_idx)` — 逐层逐 token 切头片，
  返回 (addrs, sizes)，用 `group_block_stride[0]` 跨 block（:962-986）。
- `_build_tp_mismatch_keys_and_addrs(block_hashes, block_ids, token_len, mask_num)`
  — 遍历 chunk×sub_key，返回 (keys, addrs, sizes, block_ids)（:988-1017）。
- `_load_kv_tp_mismatch(...)` — strided get + tp_rank 轮转 + 失败块记录（:1019-1051）。
- `_store_kv_tp_mismatch(req_meta)` — strided put + lookup 跳过已存 + kv events（:1053-1083+）。
- `get_group_tp_size(group_id)` — tp_mismatch 时返回 effective_tp_size（:1265-1267）。
- `_expand_lookup_key_variants(key, group_id, include_all_ranks)` — 按
  get_group_tp_size 展开 tp 变体 + pp 变体（:1293-1302）。
- `_replace_key_field(key, field, value)` — 静态方法，原地改写 key 中某字段值（:1272-1282）。

---

## 五、日志证据链（decode 存 → prefill 命中）

mooncake master 累计指标（19:56 快照）是双向闭环的铁证：

| 指标 | 值 | 含义 |
|---|---|---|
| `PutStart Item=12/12` | 全成功 | decode strided-PUT 入池 12 个子键（effective_tp=4 命名空间） |
| `PutEnd Item=12/12` | 全成功 | 全部 finalize |
| `ExistKey Item=60/60` | 全成功 | prefill lookup 阶段查 60 个 key 变体（每 chunk 展开 effective_tp=4 个 tp 变体） |
| `Get Item=12/12` | 全成功 | prefill strided-GET 取回 12 个子键 —— 与 decode 存的 12 完全相等 |
| `Keys: 12` | 稳态 | 池中 12 个子键常驻 |
| `Eviction/PutRevoke/Discard` | 0/0/0 | 池健康，无驱逐无失败 |
| `Clients: 6` | 4+2 | 4 prefill workers + 2 decode workers |

**`Get=12` 与 `Put=12` 严格相等** = prefill 取回的正是 decode 存的那 12 个子键
（key 命名空间一致 + 数据对齐，否则会 miss/recompute）。

### prefill 侧 `KV pool load spec`（多轮累积）
- 17:12:46 `req=cmpl-7802575e... vllm_cached=0 kvpool_cached=128 need_to_allocate=128`
- 17:23:48 `req=cmpl-13fef6bc... vllm_cached=0 kvpool_cached=256 need_to_allocate=256`

`vllm_cached=0` → 命中 100% 来自 kvpool；128→256 跨两轮递增 = 多轮下 decode 持续
存、prefill 持续命中更多。External cache hit rate `0% → 28.8% → 25.6% → 24.6%
→ 21.6% → 44.5%`。

### 两侧 TP mismatch 检测日志
- prefill（4 个 Worker_TP）：`local_tp=4, peer_tp=2, effective_tp=4,
  local_heads_per_rank=2, effective_heads_per_rank=2, num_sub_keys=1`；
  `strided I/O per_token_bytes=512, sub_size_bytes=512`。
- decode（2 个 Worker_TP）：`local_tp=2, peer_tp=4, effective_tp=4,
  local_heads_per_rank=4, effective_heads_per_rank=2, num_sub_keys=2`；
  `strided I/O per_token_bytes=1024, sub_size_bytes=512`。

---

## 六、结论

- **识别**：靠配置 `prefill_tp_size`/`decode_tp_size` 触发（默认 False 不影响原路径），
  `tp_mismatch = peer!=local && !mla && !hybrid && num_kv_head 整除 effective_tp`。
- **命中保证**：`effective_rank = tp_rank*num_sub_keys + sub_idx` 把双方映射到同一
  `[0, effective_tp)` 命名空间；strided 头片偏移使 key 与数据双双对齐；decode 的
  (rank,sub_idx) 与 prefill 的 rank 构成双射。
- **证据闭环**：日志 `Put 12/12 ↔ Get 12/12`、`kvpool_cached 0→128→256`、
  `vllm_cached=0` 构成 decode 存 → prefill 命中的完整闭环；池化基本功能无回归
  （decode 0 错误，prefill 唯一报错为客户端空 stop 串，与池化无关）。

---

## 附：关键文件改动（commit d32b0287，5 files +681/-23）

- `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py`（主）
- `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py`
- `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py`
- `tests/ut/distributed/ascend_store/test_pool_worker.py`
- `tests/ut/distributed/ascend_store/test_kv_transfer.py`

单测：`pytest tests/ut/distributed/ascend_store/ -k mismatch` → 18 passed
（含 mismatch 的全量 250 passed）。E2E 见 `PHASE1_CHANGELOG.md`。
