# vLLM-Ascend AscendStore KV Cache Pool 调用流程分析

> 本文结合 `vllm-ascend` 的 `ascend_store` 池化实现与 `Mooncake` 的 Store/TransferEngine 代码，分析 vLLM-Ascend 是如何调用 KV Cache Pool 的。
>
> 文档中所有图表使用 **Mermaid** 语法绘制，可在 GitHub、VS Code（Mermaid 插件）、Typora、GitLab 等支持 Mermaid 的 Markdown 阅读器中直接渲染为清晰矢量图。

---

## 1. 总体架构

vLLM-Ascend 通过 `AscendStoreConnector` 接入 vLLM v1 的 KV Connector 框架，实现跨实例的 KV Cache 共享（Prefill/Decode 分离、跨节点 Prefix Caching）。底层使用 **Mooncake** 作为分布式 KV Store 后端，借助其 `TransferEngine` 完成 NPU 显存的 RDMA/HCCL 直接传输。

```mermaid
flowchart TB
    subgraph VLLM["vLLM 进程"]
        direction TB
        subgraph SCHED["Scheduler 进程"]
            ASC_S["AscendStoreConnector<br/>(SCHEDULER 角色)"]
            KVS["KVPoolScheduler<br/>· 前缀命中查询<br/>· GVA 地址分配<br/>· 构建 ReqMeta"]
            ASC_S --> KVS
        end
        subgraph WORK["Worker 进程 — 每个 TP rank 一个"]
            ASC_W["AscendStoreConnector<br/>(WORKER 角色)"]
            KTW["KVPoolWorker<br/>· MooncakeBackend (m_store)<br/>· Send/Recv 后台线程<br/>· LookupKeyServer (rank0)"]
            ASC_W --> KTW
        end
    end

    KVS -- "metadata (zmq)" --> KTW

    subgraph MOON["Mooncake 分布式存储集群"]
        direction TB
        MS["Metadata Server"]
        MA["Master Server"]
        MDS["MooncakeDistributedStore<br/>+ TransferEngine (ascend 协议)"]
        MS -.-> MDS
        MA -.-> MDS
    end

    KVS -- "batch_is_exist / batch_alloc" --> MDS
    KTW -- "put / get / register_buffer" --> MDS

    style SCHED fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style WORK fill:#fff3e0,stroke:#e65100,stroke-width:2px
    style MOON fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
```

### 核心组件一览

| 组件 | 文件 | 职责 |
|------|------|------|
| `AscendStoreConnector` | `ascend_store_connector.py` | 入口，实现 vLLM `KVConnectorBase_V1`，按角色分流到 Scheduler/Worker |
| `KVPoolScheduler` | `pool_scheduler.py` | Scheduler 侧：前缀命中查询、GVA 分配、构建 `ReqMeta` 元数据 |
| `KVPoolWorker` | `pool_worker.py` | Worker 侧：初始化后端、注册显存、启动收发线程、执行 load/save |
| `Backend` (ABC) | `backend/backend.py` | 后端抽象接口：`put/get/exists/register_buffer` |
| `MooncakeBackend` | `backend/mooncake_backend.py` | Mooncake 后端实现，封装 `MooncakeDistributedStore` |
| `KVTransferThread` 系列 | `kv_transfer.py` | 后台收发线程，异步执行 put/get/batch_copy |
| `ChunkedTokenDatabase` | `config_data.py` | KV Key 生成、地址/大小计算（`process_tokens`/`prepare_value`） |
| `global_te` | `utils/mooncake_transfer_engine.py` | TransferEngine 单例，`initialize`/`register_memory` |

---

## 2. 初始化流程

```mermaid
flowchart TB
    A["1. vLLM 启动<br/>KVConnectorFactory 选用 'AscendStoreConnector'<br/>(注册在 distributed/kv_transfer/__init__.py)"]
    B["2. AscendStoreConnector.__init__(role)"]
    C1["role == SCHEDULER<br/>→ KVPoolScheduler(...)"]
    C2["create_scheduler_client()<br/>→ MooncakeBackend<br/>(contribute_memory=False, 仅查询客户端)"]
    D1["role == WORKER<br/>→ KVPoolWorker(...)"]
    D2["_init_backend()<br/>→ MooncakeBackend(...)"]
    D3["rank0 且非 layerwise<br/>→ LookupKeyServer"]
    E["3. KVPoolWorker.register_kv_caches(kv_caches)<br/>← vLLM 注入显存张量"]
    F["解析每个 cache 的<br/>base_addr / block_len / block_stride"]
    G["token_database.set_group_buffers(...)<br/>记录地址布局"]
    H["m_store.init_store()<br/>→ MooncakeBackend._setup_store()"]
    H1["MooncakeDistributedStore()"]
    H2["global_te.get_transfer_engine()<br/>→ TransferEngine.initialize(<br/>  host, P2PHANDSHAKE, ascend, dev)"]
    H3["store.setup(local_hostname,<br/>  metadata_server, global_segment_size,<br/>  protocol=ascend, engine=te.get_engine(), ...)"]
    I["m_store.register_buffer(ptrs, lengths)<br/>→ global_te.register_buffer()<br/>→ engine.register_memory()<br/>(把 NPU KV cache 显存注册到传输引擎)"]
    J["_start_kv_transfer_threads()<br/>启动 Send/Recv 线程"]

    A --> B
    B --> C1 --> C2
    B --> D1 --> D2 --> D3
    C2 --> E
    D3 --> E
    E --> F --> G --> H
    H --> H1 --> H2 --> H3
    H3 --> I --> J

    style A fill:#fce4ec,stroke:#c62828,stroke-width:2px
    style H fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    style I fill:#fff3e0,stroke:#e65100,stroke-width:2px
    style J fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
```

**关键点**：Mooncake 的 `store.setup()` 接收一个外部传入的 `TransferEngine`（`engine=` 参数）。该 engine 由 vLLM-Ascend 侧的 `GlobalTE` 单例创建，使用 `ascend` 协议初始化，使得 Mooncake 后续的 `put/get` 能直接对已注册的 NPU 显存做 RDMA 传输，无需 CPU 中转。

---

## 3. 三种工作模式

AscendStore 通过 `use_layerwise` 与 `backend` 两个配置组合出三条数据通路：

```mermaid
flowchart TB
    START["use_layerwise 配置"]

    MODE1["模式 A: 非 layerwise (默认)<br/>整块 put/get, Key 粒度<br/>━━━━━━━━━━━━━━━━━━━━<br/>KVCacheStoreSendingThread<br/>KVCacheStoreRecvingThread<br/>↓ m_store.put / m_store.get<br/>(batch_put_from_multi_buffers)"]

    BRANCH["use_layerwise = True"]

    MODE2["模式 B: layerwise + mooncake<br/>Key 路径 — 每层一个 Key<br/>━━━━━━━━━━━━━━━━━━━━<br/>KVCacheStoreKeyLayer<br/>  Sending/RecvingThread<br/>↓ m_store.put/get (per-layer key)"]

    MODE3["模式 C: layerwise + memcache<br/>GVA 路径 — 按地址 batch_copy<br/>━━━━━━━━━━━━━━━━━━━━<br/>KVCacheStoreLayer<br/>  Sending/RecvingThread<br/>↓ m_store.store.batch_copy<br/>(直接显存拷贝, 走 alloc/gva)"]

    START -- "False" --> MODE1
    START -- "True" --> BRANCH
    BRANCH -- "backend = mooncake" --> MODE2
    BRANCH -- "backend = memcache" --> MODE3

    style MODE1 fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style MODE2 fill:#fff3e0,stroke:#e65100,stroke-width:2px
    style MODE3 fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
```

- **模式 A（默认）**：一次把整块 KV（所有层）作为一个 key 存取。Scheduler 侧通过 ZMQ RPC 到 Worker 的 `LookupKeyServer` 做命中查询。
- **模式 B（layerwise Key）**：每个 layer 单独一个 key，逐层存取，可与计算重叠。
- **模式 C（layerwise GVA）**：用全局虚拟地址（GVA）直接 `batch_copy`，Scheduler 侧通过 `batch_alloc`/`batch_get_key_info` 管理 GVA。

> 下文以 **模式 A（mooncake 后端 + 非 layerwise）** 为主线说明（最常用），并标注其他分支差异。

---

## 4. 关键流程一：前缀命中查询（Scheduler 侧）

发生在请求调度阶段，决定能从 Pool 复用多少 token 的 KV。

```mermaid
sequenceDiagram
    autonumber
    participant VS as vLLM Scheduler
    participant KVS as KVPoolScheduler
    participant LKC as LookupKeyClient<br/>(ZMQ)
    participant LKS as LookupKeyServer<br/>(Worker rank0)
    participant MC as Mooncake Store

    VS->>KVS: get_num_new_matched_tokens(request, num_computed_tokens)

    alt 非 layerwise (模式 A)
        KVS->>LKC: lookup(token_len, block_hashes, group_ids)
        LKC->>LKS: ZMQ RPC (token_len + hashes)
        LKS->>MC: batch_is_exist(keys)
        MC-->>LKS: [1/0/-1, ...]
        LKS-->>LKC: hit_tokens (int)
        LKC-->>KVS: hit_tokens
    else layerwise Key (模式 B)
        KVS->>MC: batch_is_exist (含 per-layer keys)
        MC-->>KVS: [1/0/-1, ...]
        KVS->>KVS: check_all_layers_exists()
    else GVA (模式 C)
        KVS->>MC: batch_get_key_info(keys)
        MC-->>KVS: key_info (gva_list, size)
    end

    KVS->>KVS: 计算 hit_tokens<br/>创建 LoadSpec(vllm_cached, kvpool_cached)
    KVS-->>VS: return (need_allocate, is_async)

    VS->>KVS: update_state_after_alloc(request, blocks, num_external_tokens)
    Note over KVS: 校验 block 数, 标记 can_load

    VS->>KVS: build_connector_meta(scheduler_output)
    Note over KVS: 生成 ReqMeta(含 block_hashes,<br/>block_ids, save/load 区间)
    KVS-->>VS: AscendConnectorMetadata
    Note over VS,KVS: metadata 经 zmq 发往 Worker
```

**Key 生成规则**（`PoolKey.to_string()`）：

```
{model_name}@pcp{pcp_rank}@dcp{dcp_rank}@head_or_tp_rank:{rank}
           @pp_rank:{pp}@group:{gid}@cache_role:kv@cache_family:{fam}@{chunk_hash}
```

即 Key 包含模型名 + 并行维度（TP/PCP/DCP/PP）+ KV group + chunk hash，确保不同 rank/层的数据互不冲突。

---

## 5. 关键流程二：KV Load（Worker 侧，Consumer 加载）

发生在前向计算之前，把 Pool 中命中的 KV 拉回 NPU 显存。

```mermaid
sequenceDiagram
    autonumber
    participant VW as vLLM Worker
    participant KTW as KVPoolWorker
    participant TD as ChunkedTokenDatabase
    participant THR as RecvThread<br/>(异步时)
    participant MB as MooncakeBackend
    participant MC as Mooncake Store

    VW->>KTW: start_load_kv(metadata)

    loop 每个 request
        KTW->>TD: process_tokens_with_block_ids(token_len, hashes, block_ids)
        TD-->>KTW: (start, end, key, block_id) 列表
        KTW->>TD: prepare_value(start, end, block_ids)
        TD-->>KTW: addrs / sizes (指向已注册显存地址)
    end

    alt load_async = False (同步)
        KTW->>MB: get(keys, addrs, sizes)
        MB->>MC: batch_get_into_multi_buffers(keys, addrs, sizes)
        MC-->>MB: result [0/-1, ...]
        MB-->>KTW: result
    else load_async = True (异步)
        KTW->>THR: add_request(request)
        THR->>MB: get(keys, addrs, sizes)
        MB->>MC: batch_get_into_multi_buffers(...)
        MC-->>MB: result
    else layerwise (模式 B/C)
        KTW->>KTW: process_layer_data() 逐层
        Note over KTW: KVCacheStoreLayerRecvingThread<br/>→ m_store.get / batch_copy
    end

    KTW->>KTW: 失败 block_id 记入<br/>_invalid_block_ids (供回退)
    KTW-->>VW: load 完成
```

**地址计算核心**（`prepare_value`）：对每个 cache 子张量，`addr = base_addr + block_id * block_stride`，`size = block_len / block_size * (end-start)`。因为显存已通过 `register_buffer` 注册到 TransferEngine，Mooncake 的 `batch_get_into_multi_buffers` 可直接把远端数据 DMA 写入这些地址。

---

## 6. 关键流程三：KV Save（Worker 侧，Producer 存储）

发生在前向计算之后，把新算出的 KV 推入 Pool 供其他实例复用。

```mermaid
sequenceDiagram
    autonumber
    participant VW as vLLM Worker
    participant KTW as KVPoolWorker
    participant ST as SendingThread
    participant MB as MooncakeBackend
    participant MC as Mooncake Store

    VW->>KTW: wait_for_save(metadata) (非 layerwise)

    loop 每个 can_save 的 request
        KTW->>KTW: add_stored_request(req_id)
        KTW->>ST: add_request(request)
    end

    Note over KTW: request_queue.join() 等待全部存完<br/>(保证可见性)

    rect rgb(245, 245, 245)
        Note over ST,MC: SendingThread._handle_request (后台线程)
        ST->>ST: process_tokens → keys
        ST->>ST: prepare_value → addrs / sizes
        ST->>MB: lookup(keys)
        MB->>MC: batch_is_exist(keys)
        MC-->>MB: [1/0, ...]
        MB-->>ST: exists_states
        ST->>ST: 过滤掉已存在的 key
        ST->>ST: event.synchronize()<br/>(等 NPU 计算落盘)
        ST->>MB: put(missing_keys, addrs, sizes)
        MB->>MC: batch_put_from_multi_buffers(<br/>  keys, addrs, sizes, ReplicateConfig)
        MC-->>MB: result [0/-1, ...]
        Note over ST: (可选) 生成 BlockStored KV event
        ST->>ST: task_done() + dec_stored_request
    end

    ST-->>KTW: 全部完成
    KTW-->>VW: save 完成

    Note over VW,KTW: layerwise 模式: save_kv_layer 逐层<br/>record sync event → KeyLayer/LayerSendingThread<br/>最后一层 wait 全部完成
```

**关键细节**：
- `wait_for_save` 中调用 `request_queue.join()` 阻塞直到所有 put 完成，确保请求被标记 finished 前 KV 已对其他实例可见（避免紧随的相同 prompt lookup miss）。
- `lookup` 先查存在的 key，只 put 缺失的 block，避免重复写入。
- `event.synchronize()`（NPU Event）保证该层计算真正落盘后再传输。

---

## 7. Mooncake 侧被调用的 API 总览

vLLM-Ascend（`MooncakeBackend`）实际调用的 Mooncake Python 接口（定义于 `mooncake-integration/store/store_py.cpp` 与 `transfer_engine_py.cpp`）：

```mermaid
flowchart LR
    subgraph VA["vllm-ascend (MooncakeBackend)"]
        direction TB
        A1["MooncakeBackend._setup_store"]
        A2["global_te.get_transfer_engine"]
        A3["global_te.register_buffer"]
        A4["MooncakeBackend.exists"]
        A5["MooncakeBackend.put"]
        A6["MooncakeBackend.get"]
    end

    subgraph PY["Mooncake Python API"]
        direction TB
        B1["MooncakeDistributedStore.setup(..., engine=te)"]
        B2["TransferEngine.initialize(..., ascend, ...)"]
        B3["engine.register_memory(ptr, size)"]
        B4["store.batch_is_exist(keys)"]
        B5["store.batch_put_from_multi_buffers(<br/>keys, addrs, sizes, ReplicateConfig)"]
        B6["store.batch_get_into_multi_buffers(<br/>keys, addrs, sizes)"]
    end

    subgraph CPP["Mooncake C++ 实现"]
        direction TB
        C1["RealClient::setup_real<br/>→ setup_internal<br/>→ Client::Create"]
        C2["TransferEnginePy::initialize"]
        C3["TransferEnginePy::registerMemory"]
        C4["RealClient::batchIsExist"]
        C5["RealClient::batch_put_from_multi_buffers<br/>→ Client::BatchPut"]
        C6["RealClient::batch_get_into_multi_buffers<br/>→ Client::BatchGet"]
    end

    A1 --> B1 --> C1
    A2 --> B2 --> C2
    A3 --> B3 --> C3
    A4 --> B4 --> C4
    A5 --> B5 --> C5
    A6 --> B6 --> C6

    style VA fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style PY fill:#fff3e0,stroke:#e65100,stroke-width:2px
    style CPP fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
```

| 调用方 (vllm-ascend) | Mooncake API | C++ 实现 | 用途 |
|---|---|---|---|
| `MooncakeBackend._setup_store` | `MooncakeDistributedStore().setup(..., engine=te)` | `RealClient::setup_real` → `setup_internal` → `Client::Create` | 初始化 Store，绑定 TransferEngine |
| `global_te.get_transfer_engine` | `TransferEngine().initialize(host,"P2PHANDSHAKE","ascend",dev)` | `TransferEnginePy::initialize` | 创建 ascend 协议传输引擎 |
| `global_te.register_buffer` | `engine.register_memory(ptr, size)` | `TransferEnginePy::registerMemory` | 注册 NPU 显存段 |
| `MooncakeBackend.exists` | `store.batch_is_exist(keys)` | `RealClient::batchIsExist` | 前缀命中检查 |
| `MooncakeBackend.put` | `store.batch_put_from_multi_buffers(keys, addrs, sizes, ReplicateConfig)` | `RealClient::batch_put_from_multi_buffers` → `Client::BatchPut` | 批量写入 KV |
| `MooncakeBackend.get` | `store.batch_get_into_multi_buffers(keys, addrs, sizes)` | `RealClient::batch_get_into_multi_buffers` → `Client::BatchGet` | 批量读取 KV |

**数据通路**：`batch_put_from_multi_buffers` 把每个 key 对应的多个 `(buffer_ptr, size)` 组装成 `Slice`，调用 `Client::BatchPut`；Mooncake 内部根据 replica 选择 + TransferEngine 完成跨节点 RDMA 写入。`get` 则反向，把远端数据直接拷进已注册的 NPU 显存地址。整个过程释放 GIL（`py::gil_scoped_release`），可与 vLLM 计算异步重叠。

---

## 8. 端到端时序图（非 layerwise + mooncake）

```mermaid
sequenceDiagram
    autonumber
    participant P as Producer 实例<br/>(Prefill / KV Producer)
    participant PW as Producer Worker
    participant MC as Mooncake Store
    participant CW as Consumer Worker
    participant CS as Consumer Scheduler
    participant C as Consumer 实例<br/>(Decode / KV Consumer)

    rect rgb(227, 242, 253)
        Note over P,PW: ① Producer 前向计算产出 KV
        P->>PW: wait_for_save(metadata)
        PW->>MC: lookup → batch_is_exist
        MC-->>PW: exists_states
        PW->>MC: put → batch_put_from_multi_buffers
        Note over PW: request_queue.join() 可见性屏障
        MC-->>PW: put 完成
    end

    rect rgb(255, 243, 224)
        Note over CS,CW: ② Consumer 新请求到达, 前缀命中查询
        C->>CS: 新请求到达
        CS->>CW: get_num_new_matched_tokens<br/>(ZMQ → LookupKeyServer)
        CW->>MC: batch_is_exist
        MC-->>CW: hit_tokens = N
        CW-->>CS: (need_allocate=N, async)
        CS->>CS: vLLM 分配 N 个 block
        CS->>CS: build_connector_meta → ReqMeta
    end

    rect rgb(232, 245, 233)
        Note over CW,C: ③ Consumer 加载 KV
        CW->>CW: prepare_value → addrs(显存)
        CW->>MC: m_store.get<br/>→ batch_get_into_multi_buffers
        Note over MC,CW: RDMA 直接写入 NPU 显存
        MC-->>CW: get 完成
    end

    rect rgb(252, 228, 236)
        Note over C: ④ 前向计算 (复用 KV, 跳过已算 token)
        C->>C: 前向推理
    end
```

---

## 9. 小结

1. **接入方式**：`AscendStoreConnector` 实现 vLLM v1 KVConnector 接口，在 `__init__` 中按 `KVConnectorRole` 分裂为 Scheduler 端（`KVPoolScheduler`，负责命中查询/元数据）和 Worker 端（`KVPoolWorker`，负责实际收发）。

2. **Mooncake 集成**：Worker 通过 `MooncakeBackend` 持有 `MooncakeDistributedStore`，并把自己创建的 `TransferEngine`（ascend 协议）传入 `store.setup(engine=...)`。NPU 显存经 `register_memory` 注册后，Mooncake 的 `batch_put_from_multi_buffers` / `batch_get_into_multi_buffers` 可直接对其做 RDMA 读写，无需 CPU 拷贝。

3. **异步与重叠**：实际 put/get 在独立的 `KVTransferThread` 中执行，通过 `torch.npu.Event` 与计算流同步，配合 `load_async` / `use_layerwise` 实现传输与计算重叠。

4. **Key 设计**：以 `(model, TP/PCP/DCP/PP rank, kv_group, chunk_hash)` 为 key，天然支持多维并行与混合 cache family；layerwise 模式再追加 `layer_id` 维度。

5. **三种模式**：非 layerwise（整块 Key 存取）/ layerwise-Key（逐层 Key）/ layerwise-GVA（memcache 后端，按地址 batch_copy），由 `use_layerwise` + `backend` 配置切换，分别走不同的 Thread 子类与 Mooncake 接口。
