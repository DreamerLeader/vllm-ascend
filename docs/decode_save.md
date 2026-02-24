## 详细设计方案：Decode 存 KV Cache 场景下 TP 不对等的按模型头存储实现

---

### 一、问题分析

**现状：**

当前 `ascend_store` KV Pool 的实现中，KV Cache 的存储键 (`PoolKey`) 包含 `head_or_tp_rank` 字段，用于区分不同 TP rank 存储的 KV Cache：

```python
# pool_worker.py:88-98
if self.use_mla:
    self.num_kv_head = 1
else:
    self.num_kv_head = model_config.get_total_num_kv_heads()

if self.num_kv_head < self.tp_size:
    self.put_step = self.tp_size // self.num_kv_head
    self.head_or_tp_rank = self.tp_rank // self.put_step
else:
    self.head_or_tp_rank = self.tp_rank
    self.put_step = 1
```

**问题核心：**

对于**非 MLA 模型**，当 Decode 端 (TP=D) 和 Prefill 端 (TP=P) 的 TP 不一致时（如 Decode TP=4, Prefill TP=8），存在以下问题：

1. **存储键不一致**：Decode 端 TP=4 的 rank 0 持有 head [0,1,...,7]（假设 32 heads），而 Prefill 端 TP=8 的 rank 0 只持有 head [0,1,2,3]。两者用 `head_or_tp_rank=0` 存储的 KV Cache 内容（head 数量）不同。

2. **Prefill 端无法直接使用**：Prefill 端查询 `head_or_tp_rank:0` 时，期望获取 4 个 head 的 KV Cache，但 Decode 端存储的是 8 个 head 的，导致数据布局不匹配。

3. **Lookup 不准确**：`lookup_scheduler` 用 `head_or_tp_rank:0` 到 `head_or_tp_rank:{tp_size-1}` 遍历查询，但如果 Decode 端和 Prefill 端 TP 不同，键空间不对齐。

---

### 二、设计方案

#### 核心思路：**按 KV Head 粒度存储，而非按 TP Rank 粒度存储**

将 KV Cache 存储的粒度从 "TP rank 拥有的所有 head" 细化为 "每个 head（或 head 组）"。这样，无论 Prefill 或 Decode 端的 TP 如何配置，都能按自身的 head 映射关系正确拼取需要的 KV Cache。

#### 2.1 存储键设计

**当前键格式：**
```
{model_name}@pcp{pcp_rank}@dcp{dcp_rank}@head_or_tp_rank:{tp_rank}@pp_rank:{pp_rank}@{chunk_hash}
```

**新键格式（按 head 粒度）：**
```
{model_name}@pcp{pcp_rank}@dcp{dcp_rank}@head:{head_idx}@pp_rank:{pp_rank}@{chunk_hash}
```

**关键变化**：`head_or_tp_rank` → `head_idx`。每个 head 独立存储，不再依赖 TP 配置。

#### 2.2 存储流程（Decode 端写入 KV Pool）

Decode 端每个 TP rank 负责若干 head。存储时，**按 head 维度拆分 KV Cache**，为每个 head 生成独立的存储键和对应的内存地址/大小。

```
Decode TP=4, 32 heads → rank 0 负责 head [0..7]
  → 生成 8 个独立的 key: head:0, head:1, ..., head:7
  → 每个 key 对应的 value 是该 head 在 block 中的内存切片
```

#### 2.3 加载流程（Prefill 端从 KV Pool 读取）

Prefill 端每个 TP rank 根据自身 head 映射，构造对应 head 的查询键。

```
Prefill TP=8, 32 heads → rank 0 负责 head [0..3]
  → 查询 4 个 key: head:0, head:1, head:2, head:3
  → 从 KV Pool 中读取这 4 个 head 的 KV Cache
  → 写入到本 rank 的 KV Cache tensor 对应位置
```

#### 2.4 Lookup 流程适配

`lookup_scheduler` 需要检查所有 head 的数据是否都存在。由于 scheduler 运行在 rank 0（`head_or_tp_rank=0`），需要遍历所有 head index 进行检查。

---

### 三、详细代码实现

#### 3.1 新增配置：`config_data.py` 修改

在 `KeyMetadata` 中增加 head 粒度标识支持：

```python
# config_data.py - 新增 HeadPoolKey

@dataclass(order=True)
class HeadPoolKey:
    """Per-head granularity key for TP-mismatch-safe storage"""
    model_name: str
    head_idx: int           # 具体的 head index
    pcp_rank: int
    dcp_rank: int
    pp_rank: int
    chunk_hash: str

    def __hash__(self):
        return hash((
            self.model_name, self.head_idx,
            self.pcp_rank, self.dcp_rank,
            self.pp_rank, self.chunk_hash,
        ))

    def to_string(self):
        return (
            f"{self.model_name}"
            f"@pcp{self.pcp_rank}@dcp{self.dcp_rank}"
            f"@head:{self.head_idx}"
            f"@pp_rank:{self.pp_rank}@{self.chunk_hash}"
        )

    def split_layers(self, num_layers: int) -> list["HeadLayerPoolKey"]:
        keys = []
        for layer_id in range(num_layers):
            keys.append(HeadLayerPoolKey(
                self.model_name, self.head_idx,
                self.pcp_rank, self.dcp_rank,
                self.pp_rank, self.chunk_hash, layer_id,
            ))
        return keys


@dataclass(order=True)
class HeadLayerPoolKey(HeadPoolKey):
    layer_id: int

    def __hash__(self):
        return hash((
            self.model_name, self.head_idx,
            self.pcp_rank, self.dcp_rank,
            self.pp_rank, self.chunk_hash, self.layer_id,
        ))

    def to_string(self):
        return (
            f"{self.model_name}"
            f"@pcp{self.pcp_rank}@dcp{self.dcp_rank}"
            f"@head:{self.head_idx}"
            f"@pp_rank:{self.pp_rank}@{self.chunk_hash}@{self.layer_id}"
        )
```

#### 3.2 `ChunkedTokenDatabase` 扩展

需要增加按 head 粒度生成键和计算地址的能力：

```python
# config_data.py - ChunkedTokenDatabase 增加方法

class ChunkedTokenDatabase:
    def __init__(self, metadata: KeyMetadata, block_size: int, 
                 partitions: list[int] | None,
                 per_head_storage: bool = False,
                 head_indices: list[int] | None = None,
                 num_kv_heads_per_rank: int = 1,
                 head_dim: int = 128):
        self.metadata = metadata
        self.block_size = block_size
        self.kv_caches_base_addr: list[int] = []
        self.block_len: list[int] = []
        self.partitions = partitions
        # 新增：per-head storage 配置
        self.per_head_storage = per_head_storage
        self.head_indices = head_indices or []       # 本 rank 负责的 head 全局 index
        self.num_kv_heads_per_rank = num_kv_heads_per_rank
        self.head_dim = head_dim                     # KV head dimension

    def _make_head_keys(self, chunk_hash: str) -> list[HeadPoolKey]:
        """为每个 head 生成独立的 key"""
        keys = []
        for head_idx in self.head_indices:
            keys.append(HeadPoolKey(
                model_name=self.metadata.model_name,
                head_idx=head_idx,
                pcp_rank=self.metadata.pcp_rank,
                dcp_rank=self.metadata.dcp_rank,
                pp_rank=self.metadata.pp_rank,
                chunk_hash=chunk_hash,
            ))
        return keys

    def prepare_value_per_head(self, start: int, end: int, 
                                block_ids: list[int], head_local_idx: int):
        """
        计算单个 head 在所有层中的内存地址和大小
        
        KV Cache 布局: [num_blocks, block_size, num_heads_per_rank, head_dim]
        对于 K 和 V 各一个 tensor (或合并)
        
        head_local_idx: 在本 rank 的 local head index (0-based)
        """
        block_id = block_ids[start // self.block_size]
        addr_list = []
        size_list = []
        num_tokens = end - start
        length = len(self.block_len)
        
        # 每个 head 的字节大小
        # block_len = block_size * num_heads_per_rank * head_dim * element_size
        # per_head_len = block_size * head_dim * element_size
        #             = block_len / num_heads_per_rank
        for index, base_addr in enumerate(self.kv_caches_base_addr):
            full_block_len = self.block_len[index % length]
            per_head_block_len = full_block_len // self.num_kv_heads_per_rank
            per_head_token_len = per_head_block_len // self.block_size
            
            # 偏移到具体 head 的位置
            # 布局: base + block_id * full_block_len + token_offset * num_heads * head_dim + head_idx * head_dim
            # 但实际是连续存储: [block_size, num_heads, head_dim]
            # 所以 head 的地址 = base + block_id * full_block_len 
            #                    + head_local_idx * (block_size * head_dim * elem_size)
            # 这取决于实际内存布局
            
            # 假设 KV Cache 布局为 [num_blocks, block_size, num_heads, head_dim] (连续)
            # 那么单个 head 的数据不是连续的（head 维度不是最后一维以外的维度）
            # 实际 vllm 的布局: [num_blocks, block_size, num_kv_heads, head_dim]
            # head 维度是倒数第二维，head_dim 是最后一维
            # 所以 head 之间的数据不连续（每隔 num_heads * head_dim 个元素取一个 head）
            
            # **关键**：由于 head 维度的数据不连续，不能简单按 head 切分地址
            # 需要重新组织数据 → 转置为 [num_blocks, num_heads, block_size, head_dim]
            # 或者使用 gather/scatter 操作
            
            addr = base_addr + block_id * full_block_len
            size = int(full_block_len / self.block_size * num_tokens)
            addr_list.append(addr)
            size_list.append(size)
        
        return addr_list, size_list

    def prepare_value_per_head_layer(self, start: int, end: int,
                                      block_ids: list[int], layer_id: int,
                                      head_local_idx: int):
        """单个 head、单层的地址计算"""
        block_id = block_ids[start // self.block_size]
        addr_list = []
        size_list = []
        length = len(self.block_len)
        num_tokens = end - start
        
        for i in range(length):
            full_block_len = self.block_len[i]
            per_head_block_len = full_block_len // self.num_kv_heads_per_rank
            
            base = self.kv_caches_base_addr[layer_id * length + i]
            addr = base + block_id * full_block_len
            size = int(per_head_block_len / self.block_size * num_tokens)
            addr_list.append(addr)
            size_list.append(size)
        
        return addr_list, size_list
```

#### 3.3 核心难点：KV Cache 内存布局与 Head 切分

**vLLM KV Cache 的实际内存布局为：**
```
[num_blocks, block_size, num_kv_heads_per_rank, head_dim]
```

每个 head 的数据在 token 维度上是**不连续**的。如果要按 head 粒度存取，有两种方案：

**方案 A：存储时先转置（Transpose + Store）—— 推荐**

```python
# 将 KV Cache 从 [block_size, num_heads, head_dim] 
# 转置为 [num_heads, block_size, head_dim]
# 这样每个 head 的数据变为连续，可以独立存取
```

**方案 B：使用 Scatter/Gather 存取**

不改变存储格式，但在存取时使用非连续地址列表。

我推荐 **方案 A**，因为 Mooncake 等 Backend 的 `put`/`get` 接口支持多段地址 (`list[list[int]]`)，转置后每个 head 一次 `put`/`get` 即可。

#### 3.4 `KVPoolWorker` 修改

```python
# pool_worker.py 修改

class KVPoolWorker:
    def __init__(self, vllm_config: VllmConfig, use_layerwize: bool):
        # ... 现有代码 ...
        
        # 新增：TP 不对等场景配置
        self.enable_per_head_storage = vllm_config.kv_transfer_config \
            .kv_connector_extra_config.get("per_head_storage", False)
        
        if not self.use_mla and self.enable_per_head_storage:
            # 计算本 rank 负责的 head 全局 index
            total_kv_heads = model_config.get_total_num_kv_heads()
            self.head_indices = get_tp_rank_head_mapping(
                total_kv_heads, self.tp_size
            )[self.tp_rank]
            self.num_kv_heads_per_rank = len(self.head_indices)
            
            # head_dim from model config
            self.head_dim = model_config.get_head_size()
            
            # 不再使用 head_or_tp_rank，改为每个 head 独立键
            self.metadata = KeyMetadata(
                model_config.model.rstrip("/").split("/")[-1],
                0,  # head_or_tp_rank 不再使用，设为 0
                self.pcp_rank,
                self.dcp_rank,
                self.pp_rank,
            )
            
            self.token_database = ChunkedTokenDatabase(
                self.metadata, self.block_size, partitions,
                per_head_storage=True,
                head_indices=self.head_indices,
                num_kv_heads_per_rank=self.num_kv_heads_per_rank,
                head_dim=self.head_dim,
            )
```

#### 3.5 存储线程修改（`kv_transfer.py`）

KV Cache 写入时需要按 head 拆分：

```python
class KVCachePerHeadSendingThread(KVCacheStoreSendingThread):
    """按 head 粒度存储的发送线程"""
    
    def __init__(self, m_store, token_database, block_size, tp_rank,
                 dcp_size, put_step, kv_role, ready_event,
                 enable_kv_event, head_indices, num_kv_heads_per_rank,
                 kv_caches, head_dim, element_size):
        super().__init__(
            m_store, token_database, block_size, tp_rank,
            dcp_size, put_step, kv_role, ready_event, enable_kv_event
        )
        self.head_indices = head_indices
        self.num_kv_heads_per_rank = num_kv_heads_per_rank
        self.kv_caches = kv_caches
        self.head_dim = head_dim
        self.element_size = element_size
    
    def _handle_request(self, req_meta: ReqMeta):
        """
        按 head 粒度拆分 KV Cache 并存储。
        
        对于每个 block chunk：
        1. 从 GPU KV Cache 中按 head 切片
        2. 转置为连续内存 [block_size, head_dim]
        3. 用 head-specific key 存入 KV Pool
        """
        token_len = req_meta.token_len_chunk
        block_ids = req_meta.block_ids
        req_id = req_meta.req_id
        current_event = req_meta.current_event

        if req_id not in self.stored_requests:
            self.request_queue.task_done()
            return

        # 收集所有 chunk 信息
        chunks = []
        for start, end, key in self.token_database.process_tokens(
            token_len, req_meta.block_hashes
        ):
            chunks.append((start, end, key))

        if not chunks:
            self.dec_stored_request(req_id)
            return

        if current_event is not None:
            current_event.synchronize()

        # 对每个 head 独立存储
        for head_local_idx, head_global_idx in enumerate(self.head_indices):
            head_keys = []
            head_addrs = []
            head_sizes = []
            
            for start, end, base_key in chunks:
                # 构造 head-specific key
                head_key = HeadPoolKey(
                    model_name=base_key.key_metadata.model_name,
                    head_idx=head_global_idx,
                    pcp_rank=base_key.key_metadata.pcp_rank,
                    dcp_rank=base_key.key_metadata.dcp_rank,
                    pp_rank=base_key.key_metadata.pp_rank,
                    chunk_hash=base_key.chunk_hash,
                )
                
                # 跳过已存在的
                if self.m_store.exists([head_key.to_string()])[0] == 1:
                    continue
                
                head_keys.append(head_key.to_string())
                
                # 计算每个 head 在各层中的地址和大小
                block_id = block_ids[start // self.block_size]
                num_tokens = end - start
                layer_addrs = []
                layer_sizes = []
                
                for layer_base_addr, block_len in zip(
                    self._get_layer_addrs(), self._get_block_lens()
                ):
                    # KV cache shape: [num_blocks, block_size, num_heads, head_dim]
                    # 单个 head 的偏移 (字节)
                    per_head_stride = self.head_dim * self.element_size
                    per_token_stride = self.num_kv_heads_per_rank * per_head_stride
                    
                    # 使用临时 buffer 收集非连续的 head 数据
                    # 或者利用 multi-buffer 接口传入多段地址
                    base = layer_base_addr + block_id * block_len
                    
                    # 为该 head 收集每个 token 的地址
                    token_addrs = []
                    for t in range(num_tokens):
                        addr = (base 
                                + t * per_token_stride 
                                + head_local_idx * per_head_stride)
                        token_addrs.append(addr)
                    
                    layer_addrs.extend(token_addrs)
                    layer_sizes.extend([per_head_stride] * num_tokens)
                
                head_addrs.append(layer_addrs)
                head_sizes.append(layer_sizes)
            
            if head_keys:
                self.m_store.put(head_keys, head_addrs, head_sizes)

        self.dec_stored_request(req_id)
        self.request_queue.task_done()
```

#### 3.6 接收线程修改

Prefill 端加载时，按自身的 head 映射查询：

```python
class KVCachePerHeadRecvingThread(KVCacheStoreRecvingThread):
    """按 head 粒度加载的接收线程"""
    
    def __init__(self, m_store, token_database, block_size, tp_rank,
                 dcp_size, ready_event, head_indices, 
                 num_kv_heads_per_rank, head_dim, element_size):
        super().__init__(
            m_store, token_database, block_size, tp_rank,
            dcp_size, ready_event
        )
        self.head_indices = head_indices
        self.num_kv_heads_per_rank = num_kv_heads_per_rank
        self.head_dim = head_dim
        self.element_size = element_size
    
    def _handle_request(self, req_meta: ReqMeta):
        token_len = req_meta.load_spec.token_len
        req_id = req_meta.req_id
        mask_num = (
            req_meta.load_spec.vllm_cached_tokens 
            // self.block_size * self.block_size
        )
        
        for head_local_idx, head_global_idx in enumerate(self.head_indices):
            head_keys = []
            head_addrs = []
            head_sizes = []
            
            for start, end, base_key in self.token_database.process_tokens(
                token_len, req_meta.block_hashes, mask_num
            ):
                head_key = HeadPoolKey(
                    model_name=base_key.key_metadata.model_name,
                    head_idx=head_global_idx,
                    pcp_rank=base_key.key_metadata.pcp_rank,
                    dcp_rank=base_key.key_metadata.dcp_rank,
                    pp_rank=base_key.key_metadata.pp_rank,
                    chunk_hash=base_key.chunk_hash,
                )
                head_keys.append(head_key.to_string())
                
                block_id = req_meta.block_ids[start // self.block_size]
                num_tokens = end - start
                
                layer_addrs = []
                layer_sizes = []
                per_head_stride = self.head_dim * self.element_size
                per_token_stride = self.num_kv_heads_per_rank * per_head_stride
                
                for layer_base_addr, block_len in zip(
                    self._get_layer_addrs(), self._get_block_lens()
                ):
                    base = layer_base_addr + block_id * block_len
                    for t in range(num_tokens):
                        addr = (base 
                                + t * per_token_stride 
                                + head_local_idx * per_head_stride)
                        layer_addrs.append(addr)
                    layer_sizes.extend([per_head_stride] * num_tokens)
                
                head_addrs.append(layer_addrs)
                head_sizes.append(layer_sizes)
            
            if head_keys:
                self.m_store.get(head_keys, head_addrs, head_sizes)
        
        self.set_finished_request(req_id)
        self.request_queue.task_done()
```

#### 3.7 Lookup 适配

```python
# pool_worker.py - lookup_scheduler 修改

def lookup_scheduler(self, token_len, block_hashes, use_layerwise):
    if not self.enable_per_head_storage:
        return self._lookup_scheduler_original(token_len, block_hashes, use_layerwise)
    
    # Per-head storage: 检查所有 head 是否都存在
    end = 0
    keys_per_head = {}  # head_idx -> [keys]
    starts = []
    
    total_kv_heads = self.num_kv_head  # 全局 head 数
    
    for start, end, key in self.token_database.process_tokens(
        token_len, block_hashes
    ):
        starts.append(start)
        for head_idx in range(total_kv_heads):
            if head_idx not in keys_per_head:
                keys_per_head[head_idx] = []
            
            head_key = HeadPoolKey(
                model_name=self.metadata.model_name,
                head_idx=head_idx,
                pcp_rank=self.metadata.pcp_rank,
                dcp_rank=self.metadata.dcp_rank,
                pp_rank=self.metadata.pp_rank,
                chunk_hash=key.chunk_hash,
            )
            if use_layerwise:
                for lk in head_key.split_layers(self.num_layers):
                    keys_per_head[head_idx].append(lk.to_string())
            else:
                keys_per_head[head_idx].append(head_key.to_string())
    
    # 收集所有 head 的键一起查询
    all_keys = []
    for head_idx in sorted(keys_per_head.keys()):
        all_keys.extend(keys_per_head[head_idx])
    
    if not all_keys:
        return end
    
    res = self.m_store.exists(all_keys)
    
    # 分析：只要任一 head 缺失某个 chunk，就返回该 chunk 的 start
    num_chunks = len(starts)
    keys_per_chunk_per_head = (self.num_layers if use_layerwise else 1)
    
    for chunk_idx in range(num_chunks):
        all_heads_have_chunk = True
        for head_idx in range(total_kv_heads):
            base = head_idx * num_chunks * keys_per_chunk_per_head
            offset = chunk_idx * keys_per_chunk_per_head
            chunk_results = res[base + offset : base + offset + keys_per_chunk_per_head]
            if not all(v == 1 for v in chunk_results):
                all_heads_have_chunk = False
                break
        if not all_heads_have_chunk:
            return starts[chunk_idx]
    
    return end
```

#### 3.8 PP (Pipeline Parallel) 也需要适配 lookup

```python
# 在 lookup 中还需要检查所有 pp_rank 的 head
for pp_rank in range(self.pp_size):
    for head_idx in range(total_kv_heads):
        head_key = HeadPoolKey(
            model_name=self.metadata.model_name,
            head_idx=head_idx,
            pcp_rank=0,
            dcp_rank=0,
            pp_rank=pp_rank,
            chunk_hash=chunk_hash,
        )
        # ...
```

---

### 四、优化方案：使用 Staging Buffer 解决内存不连续问题

由于 KV Cache 的 head 维度数据在内存中不连续（`[block_size, num_heads, head_dim]` 中 head 维度在中间），直接按 head 地址存取需要大量 scatter/gather 小片段传输，效率很低。

**推荐优化：引入 staging buffer**

```python
class PerHeadStagingBuffer:
    """
    临时缓冲区，将非连续的 head 数据收集为连续内存，
    再一次性 put/get 到后端存储。
    """
    
    def __init__(self, block_size: int, head_dim: int, 
                 element_size: int, num_layers: int, device: str = "cpu"):
        # 每个 head 每个 block 的连续 buffer
        self.per_head_buffer_size = block_size * head_dim * element_size
        self.block_size = block_size
        self.head_dim = head_dim
        self.element_size = element_size
        
        # 预分配 staging buffer（CPU 内存）
        # shape: [num_layers * 2, block_size, head_dim]  (K和V各一份)
        self.buffer = torch.empty(
            num_layers * 2, block_size, head_dim,
            dtype=torch.float16,  # 根据实际模型调整
            device=device,
            pin_memory=(device == "cpu"),
        )
    
    def gather_head(self, kv_caches, block_id: int, 
                    head_local_idx: int, num_tokens: int):
        """
        从 GPU KV Cache 中收集单个 head 的数据到 staging buffer
        
        kv_caches: dict[layer_name, (k_cache, v_cache)]
            k_cache shape: [num_blocks, block_size, num_heads, head_dim]
        """
        buf_idx = 0
        for layer_name, (k_cache, v_cache) in kv_caches.items():
            # 取出目标 block、目标 head 的所有 token
            # k_cache[block_id, :num_tokens, head_local_idx, :]
            self.buffer[buf_idx, :num_tokens] = \
                k_cache[block_id, :num_tokens, head_local_idx, :].cpu()
            buf_idx += 1
            self.buffer[buf_idx, :num_tokens] = \
                v_cache[block_id, :num_tokens, head_local_idx, :].cpu()
            buf_idx += 1
        
        return self.buffer.data_ptr(), buf_idx * self.per_head_buffer_size
    
    def scatter_head(self, kv_caches, block_id: int,
                     head_local_idx: int, num_tokens: int):
        """
        从 staging buffer 写回 GPU KV Cache 的对应 head 位置
        """
        buf_idx = 0
        for layer_name, (k_cache, v_cache) in kv_caches.items():
            k_cache[block_id, :num_tokens, head_local_idx, :] = \
                self.buffer[buf_idx, :num_tokens].to(k_cache.device)
            buf_idx += 1
            v_cache[block_id, :num_tokens, head_local_idx, :] = \
                self.buffer[buf_idx, :num_tokens].to(v_cache.device)
            buf_idx += 1
```

---

### 五、完整的修改文件清单

| 文件 | 修改内容 |
|------|----------|
| `config_data.py` | 新增 `HeadPoolKey`, `HeadLayerPoolKey`，`ChunkedTokenDatabase` 增加 per-head 方法 |
| `pool_worker.py` | 增加 `enable_per_head_storage` 配置，修改 `lookup_scheduler`，新增 head 映射逻辑 |
| `kv_transfer.py` | 新增 `KVCachePerHeadSendingThread`, `KVCachePerHeadRecvingThread` |
| `ascend_store_connector.py` | 根据配置选择使用 per-head 存储线程 |
| `pool_scheduler.py` | `lookup` 调用适配 per-head key 格式 |

---

### 六、关键设计决策总结

| 决策点 | 选择 | 原因 |
|--------|------|------|
| 存储粒度 | 按 head | 让 KV Pool 中的数据与 TP 配置解耦 |
| 内存不连续处理 | staging buffer | 避免大量碎片化 RDMA，一次收集后连续传输 |
| MLA 模型 | 保持现有逻辑 | MLA 本身只有 1 个 head，无 TP 不对等问题 |
| 向后兼容 | 配置开关 `per_head_storage` | TP 对等时仍使用原始逻辑，零性能损失 |
| Lookup 策略 | 遍历所有 head + PP rank | 确保完整性验证 |

### 七、数据流总结

```
=== Decode 端 (TP=4, 32 heads, 8 heads/rank) ===

rank 0 保存:
  head:0  @chunk_hash_xxx → staging_buffer.gather(head=0) → backend.put()
  head:1  @chunk_hash_xxx → staging_buffer.gather(head=1) → backend.put()
  ...
  head:7  @chunk_hash_xxx → staging_buffer.gather(head=7) → backend.put()

=== Prefill 端 (TP=8, 32 heads, 4 heads/rank) ===

rank 0 加载:
  head:0  @chunk_hash_xxx → backend.get() → staging_buffer.scatter(head=0)
  head:1  @chunk_hash_xxx → backend.get() → staging_buffer.scatter(head=1)
  head:2  @chunk_hash_xxx → backend.get() → staging_buffer.scatter(head=2)
  head:3  @chunk_hash_xxx → backend.get() → staging_buffer.scatter(head=3)

rank 1 加载:
  head:4  @chunk_hash_xxx → backend.get() → staging_buffer.scatter(head=0)  # local idx 0
  head:5  @chunk_hash_xxx → ...
  head:6  @chunk_hash_xxx → ...
  head:7  @chunk_hash_xxx → ...
```
---

以上是完整的设计方案。核心要点总结：

1. **按 head 粒度存储**：将 `head_or_tp_rank` 替换为具体的 `head_idx`，使 KV Pool 中的数据与 TP 拓扑解耦。无论 Decode TP=4 还是 Prefill TP=8，都能通过 head index 正确寻址。

2. **Staging Buffer 解决内存不连续问题**：KV Cache 布局 `[num_blocks, block_size, num_heads, head_dim]` 中 head 维度的数据不连续。通过 staging buffer 先 gather 到连续内存再传输，避免碎片化 IO。

3. **向后兼容**：通过 `per_head_storage` 配置项控制。TP 对等场景仍走原路径，无额外开销。

4. **Lookup 全 head 校验**：Scheduler 侧 lookup 需遍历所有 head index（而非 tp_rank），确保 Prefill 所需的每个 head 在 KV Pool 中都存在。

如果你需要我将这些代码写入实际文件，请告诉我。
