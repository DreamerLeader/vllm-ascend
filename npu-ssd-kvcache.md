# NPU 场景下实现 SSD KVCache 缓存：方案设计与源码解析

## 1. 背景与业务价值

### 为什么 NPU 需要挂载 SSD 缓存？（容量瓶颈与成本账本）

在长上下文推理里，**KVCache 往往比模型权重更快吃掉 HBM**。一个请求进入 Prefill 后，每一层都会生成 Key/Value 张量；Decode 阶段每生成一个 token，又会继续追加 KV。上下文越长、并发越高，NPU HBM 越容易成为瓶颈。

如果只靠 NPU HBM：

- HBM 很快被长 prompt、multi-turn 对话、Agent 工具调用历史占满。
- Prefill 生成过的 KV 在 Decode 或后续相似请求里很有复用价值，但 HBM 放不下就只能丢掉。
- 丢掉后再次命中相同前缀，只能重新 Prefill，TTFT 和算力成本都上升。

Mooncake 的 SSD KVCache Offload 本质是在做一笔成本账：

- **HBM**：最快，但最贵、容量最小。
- **Host DRAM**：较快，容量比 HBM 大，适合热 KV。
- **SSD/NVMe**：慢于 DRAM，但容量大、成本低，适合温/冷 KV。
- **Mooncake Store + Transfer Engine**：把这些层级统一成可寻址、可传输、可复用的 KV 对象池。

所以 SSD 缓存的价值是：**用较低成本保存大量可复用 KVCache，让 NPU 少做重复 Prefill，把 HBM 留给正在计算的热数据。**

## 2. 核心原理通俗解析（小白友好）

### 多级存储模型比喻

可以把一次 LLM 推理想成办公室处理资料：

- **NPU HBM = 办公桌**：离手最近，拿取最快，但桌面很小，只能放当前最急的文件。
- **Host DRAM = 抽屉**：比桌面大，拿取也快，适合放最近还会继续用的文件。
- **SSD = 地下室档案柜**：容量巨大，拿取慢一些，但比重新打印一整套材料便宜得多。
- **请求 = 办公人员**：Prefill 像是第一次阅读并整理长文档；Decode 像是边看笔记边写答案。
- **KVCache = 做好的阅读笔记**：如果下一个人问了同样前缀的问题，就不需要重新读整篇文档，可以直接拿笔记继续工作。

当办公桌满了，不是把资料扔掉，而是：

1. 热资料还在桌面 HBM。
2. 次热资料放进抽屉 DRAM。
3. 更冷但未来可能还会用的资料放进 SSD 档案柜。
4. 再次需要时，先从 SSD 取回抽屉，再搬回桌面。

这就是 Mooncake 的 **HBM -> DRAM -> SSD -> DRAM -> HBM** 层级缓存。

### KVCache 的生命周期（生成 -> 转移 -> 换出 -> 唤醒）

1. **生成：Prefill 阶段**
   - Prefill 节点读取 prompt。
   - 每层 Attention 生成 KVCache block/page。
   - vLLM/SGLang 的 Connector 或 HiCache 把可复用 KV 块登记为可外部传输的数据。

2. **转移：HBM/Host 到 Mooncake Store**
   - vLLM 的 `MooncakeConnector` 负责 PD 分离场景中 Prefill -> Decode 的直接 KV 传输。
   - `MooncakeStoreConnector` 场景下，KV block 以 hash key 的形式写入 `MooncakeDistributedStore`。
   - Mooncake Store 使用 `TransferEngine` 通过 RDMA/TCP/Ascend Direct 把数据写入远端或本地 Store 节点的 DRAM Segment。

3. **换出：DRAM 不够时落 SSD**
   - Master 发现 DRAM Segment 压力过高，选择需要 offload 的对象。
   - Store 侧 `FileStorage` 的 heartbeat 线程向 Master 拉取 offload 任务。
   - `FileStorage::OffloadObjects()` 从本地 DRAM replica 读取对象，写入 SSD 后通知 Master 增加 `LOCAL_DISK` replica。
   - 如果 SSD backend 自身达到容量上限，`BucketStorageBackend` 可用 FIFO/LRU 删除更老的 SSD bucket。

4. **唤醒：命中 SSD 后回到 DRAM/HBM**
   - 新请求根据 block hash 命中 Mooncake Store。
   - 如果 Master 返回的是 `LOCAL_DISK` replica，读取方会先 RPC 到持有 SSD 文件的 Store 节点。
   - 持有方从 SSD 读到 `ClientBuffer`。
   - 读取方再用 `TransferEngine` 从该 `ClientBuffer` 拉回自己的目标 buffer。
   - 对 NPU Decode 来说，最终还要把 KV 放回 NPU HBM，参与后续 Decode。

### Prefill 和 Decode 如何共享 KV 块

Prefill/Decode 分离时：

- Prefill 节点负责重计算或首次计算 prompt KV。
- Decode 节点负责接管请求继续生成 token。
- 如果 Decode 需要 Prefill 生成的 KV：
  - 直接 PD transfer：`MooncakeConnector` 通过 side-channel 获取远端 block 地址，然后用 `TransferEngine` 拉取。
  - Store-backed cache：`MooncakeStoreConnector` 通过 block hash 查询 Mooncake Store。命中 DRAM 就从 DRAM 取；命中 SSD 就走 `LOCAL_DISK` restore。

对于多请求复用：

- 相同 prompt 前缀会产生相同 block hash。
- 后续请求只要 hash 命中，就能复用之前 Prefill 的 KV block。
- 这避免了重复 Prefill，尤其适合长上下文和 Agent 多轮历史。

## 3. 总体架构设计

### 控制面（Control Plane）：元数据服务器（Metadata Server）、路由调度

Mooncake 里需要区分两个控制面概念：

1. **Transfer Engine Metadata Server**
   - 给 `TransferEngine` 管理 segment、endpoint、传输地址。
   - 支持 HTTP/etcd/Redis/P2PHANDSHAKE 等模式。
   - 相关代码：
     - `mooncake-transfer-engine/src/transfer_engine.cpp`
     - `mooncake-transfer-engine/include/transfer_metadata.h`

2. **Mooncake Store Master Service**
   - 管理对象 key、replica、Segment、offload/promotion 任务。
   - 负责决定对象在哪里：`MEMORY`、`LOCAL_DISK`、`DISK`。
   - 相关代码：
     - `mooncake-store/src/master_service.cpp`
     - `mooncake-store/src/master_client.cpp`
     - `mooncake-store/src/rpc_service.cpp`
     - `mooncake-store/src/segment.cpp`

控制面核心职责：

- 注册 DRAM Segment：`MountSegment` / `ReMountSegment`
- 注册本地 SSD 能力：`MountLocalDiskSegment`
- 查询对象位置：`Query` / `BatchQuery`
- 分配 DRAM replica：`PutStart` / allocation strategy
- 下发 offload 任务：`OffloadObjectHeartbeat`
- 提交 SSD replica：`NotifyOffloadSuccess`
- SSD replica 删除：`BatchEvictDiskReplica`
- SSD 命中后回迁 DRAM：`PromotionObjectHeartbeat` / `PromotionAllocStart` / `NotifyPromotionSuccess`

### 数据面（Data Plane）：NPU <-> DRAM <-> SSD 的数据流向图

```text
        Prefill / Decode Worker
        vLLM / SGLang / HiCache
                  |
                  | KV block pointer / hash key
                  v
        Mooncake Connector / Store API
                  |
                  | TransferEngine
                  | RDMA / TCP / Ascend Direct
                  v
        Mooncake Store DRAM Segment
                  |
        DRAM 压力高 / Master 下发 offload
                  v
        FileStorage heartbeat thread
                  |
        StorageBackendInterface
        Bucket / FilePerKey / OffsetAllocator
                  |
                  v
        Local NVMe SSD
```

NPU 相关数据路径：

```text
NPU HBM
  |
  | D2H staging 或 Ascend Direct/Fabric Memory
  v
Host DRAM / registered Segment
  |
  | Mooncake TransferEngine
  v
Remote or local Store DRAM
  |
  | FileStorage async heartbeat
  v
SSD bucket / offset file
```

NPU 侧传输相关代码：

- `mooncake-wheel/mooncake/mooncake_connector_v1.py`
- `mooncake-transfer-engine/include/transport/ascend_transport/ascend_direct_transport/ascend_direct_transport.h`
- `mooncake-transfer-engine/src/transport/ascend_transport/ascend_direct_transport/ascend_direct_transport.cpp`
- `mooncake-integration/allocator_ascend_npu.py`
- `mooncake-store/include/gpu_staging_utils.h`

## 4. 核心代码链路追踪（深度硬核）

### 场景一：写入与落盘（Eviction to SSD）

#### 4.1 NPU 显存满，Connector 捕获 KV 转移需求

vLLM PD 分离路径里，Mooncake 提供 `MooncakeConnector`：

- Scheduler 侧：
  - `MooncakeConnector.get_num_new_matched_tokens()`
  - `MooncakeConnector.update_state_after_alloc()`
  - `MooncakeConnector.build_connector_meta()`
  - `MooncakeConnector.request_finished()`
- Worker 侧：
  - `MooncakeConnectorWorker.register_kv_caches()`
  - `MooncakeConnectorWorker.start_load_kv()`
  - `MooncakeConnectorWorker.send_kv_to_decode()`
  - `_send_blocks()` 内部调用 Transfer Engine batch transfer

对应文件：`mooncake-wheel/mooncake/mooncake_connector_v1.py`。

这条路径主要解决 **Prefill -> Decode 的直接 KV 传输**。而 `MooncakeStoreConnector` 是 vLLM Store-backed KV connector，Mooncake repo 中给出了部署文档；其底层对应 Mooncake 侧的 `MooncakeDistributedStore` / `Client` / `RealClient` 接口。

文档位置：`docs/source/getting_started/examples/vllm-integration/vllm-mooncakestoreconnector.md`。

#### 4.2 Connector/Store 写入 DRAM Segment

Store 写入的 C++ 主链路是：

```text
MooncakeStoreConnector / MooncakeDistributedStore.put_from()
  -> PyClient / RealClient
  -> Client::Put / Client::BatchPut
  -> MasterService allocation
  -> TransferSubmitter
  -> TransferEngine
  -> remote/local DRAM Segment
```

关键模块：

- `mooncake-store/src/real_client.cpp`
- `mooncake-store/src/client_service.cpp`
- `mooncake-store/src/transfer_task.cpp`
- `mooncake-transfer-engine/src/transfer_engine.cpp`

Store 中对象以 key 管理。KVCache 场景下，key 通常来自 block hash / prefix hash；value 是一个 KV block/page 的连续或分片内存。

#### 4.3 DRAM 满，Master 触发 offload-on-evict

Master 构造时读取配置：

- `enable_offload`
- `offload_on_evict`
- `offload_force_evict`
- `promotion_on_hit`

关键代码：

- `mooncake-store/src/master.cpp`
- `mooncake-store/include/master_config.h`
- `mooncake-store/src/master_service.cpp`

当 DRAM Segment 需要腾空间时，Master 会把对象加入本地 SSD holder 的 offloading queue：

```text
MasterService eviction path
  -> PushOffloadingQueue(key, source_memory_replica)
  -> LocalDiskSegment.offloading_objects[key] = size
```

其中 `LocalDiskSegment` 挂在 `SegmentManager` 里，由 Store client 在启动 `FileStorage` 时注册：

```text
FileStorage::Init()
  -> RegisterLocalMemory()
  -> client_->MountLocalDiskSegment(enable_offloading_)
  -> client_->ReportSsdCapacity(total_size_limit)
```

#### 4.4 FileStorage heartbeat 拉取 offload 任务

`FileStorage` 是 SSD offload 的核心协调器。

调用链：

```text
FileStorage::Init()
  -> 启动 heartbeat_thread_
  -> FileStorage::Heartbeat()
       -> client_->OffloadObjectHeartbeat(enable_offloading_, offloading_objects)
       -> FileStorage::OffloadObjects(offloading_objects)
```

Master RPC 链路：

```text
FileStorage::Heartbeat()
  -> Client::OffloadObjectHeartbeat()
  -> MasterClient::OffloadObjectHeartbeat()
  -> WrappedMasterService::OffloadObjectHeartbeat()
  -> MasterService::OffloadObjectHeartbeat()
```

关键文件：

- `mooncake-store/include/file_storage.h`
- `mooncake-store/src/file_storage.cpp`
- `mooncake-store/src/client_service.cpp`
- `mooncake-store/src/master_client.cpp`
- `mooncake-store/src/rpc_service.cpp`
- `mooncake-store/src/master_service.cpp`

#### 4.5 从 DRAM replica 取 Slice，必要时做 NPU D2H staging

`FileStorage::OffloadObjects()` 先查询本地 MEMORY replica：

```text
FileStorage::OffloadObjects()
  -> BatchQuerySegmentSlices(keys, batch_object)
       -> client_->BatchQuery(keys)
       -> 找到 IsReplicaOnLocalMemory(descriptor)
       -> 取 memory_descriptor.buffer_descriptor.buffer_address_
       -> 生成 Slice{ptr, size}
```

如果 Slice 指针是设备地址，代码会走 host staging：

```text
IsDevicePointer(slice.ptr, &device_id)
  -> SetDevice(device_id)
  -> pinned_buffer_pool_->Acquire(slice.size)
  -> CopyDeviceToHost(buf.data, slice.ptr, slice.size)
  -> host_slices.emplace_back(Slice{buf.data, slice.size})
```

这段对 NPU/GPU 都重要：**SSD 文件系统不能直接拿普通 POSIX/io_uring 写设备 HBM 指针**，所以需要把设备 KV 先搬到 host pinned buffer，之后再写 SSD。

相关文件：

- `mooncake-store/src/file_storage.cpp`
- `mooncake-store/include/pinned_buffer_pool.h`
- `mooncake-store/include/gpu_staging_utils.h`

#### 4.6 写入 SSD Backend

`FileStorage` 把 host slices 交给 `StorageBackendInterface`：

```text
FileStorage::OffloadObjects()
  -> storage_backend_->BatchOffload(
         host_batch_object,
         complete_handler,
         eviction_handler)
```

当前有三种 backend：

1. **BucketStorageBackend，默认**
   - 多个 key 聚合成 bucket。
   - 数据文件：`.bucket`
   - 元数据文件：`.meta`
   - 支持 FIFO/LRU bucket eviction。

2. **StorageBackendAdaptor / FilePerKey**
   - 一个 key 一个文件。
   - 简单，但大量小对象时目录/文件开销大。

3. **OffsetAllocatorStorageBackend**
   - 单个 `kv_cache.data` 大文件。
   - 内部用 `OffsetAllocator` 分配 offset。
   - 适合减少小文件数量。

关键定义：

- `mooncake-store/include/storage_backend.h`
- `mooncake-store/src/storage_backend.cpp`
- `mooncake-store/include/offset_allocator/offset_allocator.hpp`

Bucket 写入核心链路：

```text
BucketStorageBackend::BatchOffload()
  -> IsEnableOffloading()
  -> bucket_id_generator_->NextId()
  -> BuildBucket(bucket_id, batch_object, iovs, metadatas)
  -> PrepareEviction(required_size)
  -> eviction_handler(pending.keys)
       -> client_->BatchEvictDiskReplica(evicted_keys, LOCAL_DISK)
  -> FinalizeEviction(pending)
  -> WriteBucket(bucket_id, bucket, iovs)
  -> complete_handler(bucket->keys, metadatas)
       -> client_->NotifyOffloadSuccess(keys, metadatas)
  -> commit object_bucket_map_ / buckets_ / lru_index_
```

#### 4.7 SSD 满时 LRU/FIFO 删除

`BucketStorageBackend` 支持：

```cpp
enum class BucketEvictionPolicy {
    NONE,
    FIFO,
    LRU,
};
```

配置来自环境变量：

- `MOONCAKE_OFFLOAD_BUCKET_MAX_TOTAL_SIZE`
- `MOONCAKE_OFFLOAD_BUCKET_EVICTION_POLICY=fifo|lru|none`
- `MOONCAKE_OFFLOAD_BUCKET_SIZE_LIMIT_BYTES`
- `MOONCAKE_OFFLOAD_BUCKET_KEYS_LIMIT`

LRU 依据：

- `BucketMetadata::last_access_ns_`
- 每次 `BatchLoad()` 命中 bucket 时更新。
- `SelectEvictionCandidate()` 从 `lru_index_` 找最旧 bucket，并懒修复 stale timestamp。

删除采用两阶段协议：

```text
PrepareEviction(required_size)
  -> 从 buckets_ / object_bucket_map_ 移除 bucket 元数据
  -> 收集 evicted keys
  -> 返回 PendingEviction

eviction_handler(evicted_keys)
  -> MasterService::BatchEvictDiskReplica()
  -> Master 删除 LOCAL_DISK replica

FinalizeEviction(pending)
  -> 等 inflight_reads_ 归零
  -> 清 file_cache_
  -> 删除 .bucket / .meta
```

这个顺序很关键：**先让 Master 不再路由到旧 SSD replica，再删除物理文件**，避免读到已经被删的文件。

### 场景二：命中与加载（Restore from SSD）

#### 4.8 新请求通过 Hash 命中 SSD 缓存

Store-backed KVCache 通常按 block hash 查询：

```text
MooncakeStoreConnector / HiCache L3 backend
  -> MooncakeDistributedStore.batch_get_into()
  -> RealClient
  -> Client::BatchQuery(keys)
  -> MasterService::GetReplicaList()
```

Master 如果发现没有 MEMORY replica，但有 `LOCAL_DISK` replica，会返回：

```text
ReplicaType::LOCAL_DISK
  -> transport_endpoint = 持有 SSD 文件的 real client RPC 地址
  -> object_size = value size
```

Master 还可启用 **promotion-on-hit**：

```text
GetReplicaList()
  -> 发现 LOCAL_DISK-only key
  -> TryPushPromotionQueue(key)
       -> CountMinSketch 频率门控
       -> DRAM watermark 门控
       -> PushPromotionQueue()
```

相关代码：

- `mooncake-store/src/master_service.cpp`
- `mooncake-store/include/count_min_sketch.h`

#### 4.9 读取方选择 LOCAL_DISK 路径

`RealClient` 明确按优先级选 replica：

```text
local MEMORY
  -> remote MEMORY
  -> LOCAL_DISK
  -> DISK
```

当选中 `LOCAL_DISK`：

```text
RealClient::get_buffer_internal()
  -> SelectBestReplica()
  -> best_replica->is_local_disk_replica()
  -> batch_get_into_offload_object_internal(endpoint, objects)
```

Batch 路径也会把 LOCAL_DISK op 按 endpoint 分组：

```text
RealClient batch get
  -> group disk_ops by local_disk.transport_endpoint
  -> batch_get_into_offload_object_internal(endpoint, objects)
```

相关代码：`mooncake-store/src/real_client.cpp`。

#### 4.10 RPC 到 SSD holder，holder 从 SSD 读入 ClientBuffer

读取方 RPC：

```text
RealClient::batch_get_into_offload_object_internal()
  -> ClientRequester::batch_get_offload_object(target_rpc_service_addr, keys, sizes)
```

持有方处理：

```text
RealClient::batch_get_offload_object(keys, sizes)
  -> coro_io::post(...)
  -> FileStorage::BatchGet(keys, sizes)
       -> AllocateBatch(keys, sizes)
       -> BatchLoad(allocated_batch->slices)
       -> client_buffer_allocated_batches_[batch_id] = allocated_batch
  -> 返回 BatchGetOffloadObjectResponse{
         batch_id,
         pointers,
         transfer_engine_addr,
         gc_ttl_ms
     }
```

这里 `coro_io::post()` 很重要：**SSD I/O 被丢到专门线程执行，不阻塞 RPC I/O 线程**。

相关代码：

- `mooncake-store/src/real_client.cpp`
- `mooncake-store/src/file_storage.cpp`
- `mooncake-store/include/rpc_types.h`

#### 4.11 Transfer Engine 从 holder ClientBuffer 拉回读取方目标 buffer

读取方拿到 holder 返回的 pointers 后：

```text
RealClient::batch_get_into_offload_object_internal()
  -> client_->BatchGetOffloadObject(
         transfer_engine_addr,
         keys,
         pointers,
         objects)
```

底层：

```text
Client::BatchGetOffloadObject()
  -> transfer_submitter_->submit_batch_get_offload_object()
  -> TransferSubmitter::submit_batch_get_offload_object()
       -> engine_.openSegment(transfer_engine_addr)
       -> build TransferRequest{READ, source=slice.ptr, target_offset=pointer+offset}
       -> submitTransfer(requests)
  -> future->get()
```

注意方向：

- `pointer` 是 holder 侧 `ClientBuffer` 地址。
- `objects[key]` 是读取方目标 Slice，可能是 host buffer，也可能是后续要搬到 NPU 的 buffer。
- `TransferRequest::READ` 表示从远端 segment offset 读到本地 `source` 指针指向的目标 buffer。

相关代码：

- `mooncake-store/src/client_service.cpp`
- `mooncake-store/src/transfer_task.cpp`

传输完成后：

```text
ClientRequester::release_offload_buffer(target_rpc_service_addr, batch_id)
  -> RealClient::release_offload_buffer()
  -> FileStorage::ReleaseBuffer(batch_id)
```

如果读取方异常未释放，`FileStorage::ClientBufferGCThreadFunc()` 会按 `MOONCAKE_OFFLOAD_CLIENT_BUFFER_GC_TTL_MS` 自动回收。

#### 4.12 Promotion from SSD to DRAM

如果开启 `promotion_on_hit`，SSD 命中后 Master 会异步把热 key 搬回 DRAM：

```text
FileStorage::Heartbeat()
  -> ProcessPromotionTasks()
       -> client_->PromotionObjectHeartbeat(promotion_objects)
       -> client_->PromotionAllocStart(key, size, preferred_segments)
       -> AllocateBatch()
       -> BatchLoad() 从 SSD 读入本地 staging buffer
       -> client_->PromotionWrite(memory_descriptor, tx_slices)
            -> Client::TransferWrite()
            -> TransferEngine 写入新 MEMORY replica
       -> client_->NotifyPromotionSuccess(key)
            -> Master 把 PROCESSING MEMORY replica 标记 COMPLETE
```

这相当于：**有人老去地下室拿同一份文件，系统就自动复印一份放回抽屉。**

相关代码：

- `mooncake-store/src/file_storage.cpp`
- `mooncake-store/src/master_service.cpp`

## 5. 关键技术点与 I/O 优化方案

### 异步落盘与零拷贝设计

Mooncake 的 SSD offload 不在请求主路径同步执行，而是通过 `FileStorage` 后台 heartbeat 推进：

```text
请求写 DRAM 成功
  -> Master 记录 MEMORY replica
  -> 后续 eviction 决策进入 offloading queue
  -> FileStorage heartbeat 后台落 SSD
```

好处：

- NPU 的 Prefill/Decode 线程不直接等待 SSD 写。
- Master 仍能先用 MEMORY replica 服务热读。
- SSD 写失败不会立即拖垮主推理路径。

SSD 读也避免阻塞 RPC I/O 线程：

```text
RealClient::batch_get_offload_object()
  -> coro_io::post()
  -> FileStorage::BatchGet()
```

Mooncake 尽量减少无意义 memcpy：

1. **DRAM replica 到远端读取方**
   - Transfer Engine 直接基于 `Slice{ptr, size}` 传输。
   - RDMA/TCP transport 直接读写 registered memory。

2. **SSD restore**
   - SSD holder 读入预注册 `ClientBuffer`。
   - 读取方通过 Transfer Engine 从 holder `ClientBuffer` 拉到自己的目标 Slice。
   - 读取方不需要再经一个中间 Python bytes。

3. **io_uring fixed buffer**
   - `FileStorage` 构造时把 `AlignedClientBufferAllocator` 的 base/size 注册给 `UringFile::register_global_buffer()`。
   - 每个线程的 io_uring ring 懒注册 buffer。
   - 如果读目标落在 registered buffer 内，走 `io_uring_prep_read_fixed()`。

关键代码：

- `mooncake-store/src/uring_file.cpp`
- `mooncake-store/src/aligned_client_buffer.cpp`

### 块大小（Chunk Size）与 NVMe 扇区对齐原理

`UringFile` 支持 `O_DIRECT`。启用后：

- buffer 地址要 4KB 对齐。
- length 要 4KB 对齐。
- file offset 要 4KB 对齐。
- 不满足条件时需要 bounce buffer 或额外 padding。

`FileStorage::AllocateBatch()` 已经为 SSD read 做了超额分配：

```text
alloc_size = align_up(data_size, 4096) + 2 * 4096
aligned_ptr = align(raw_ptr, 4096)
```

这样可以适配：

- 读 buffer 对齐。
- 读尾部 padding。
- bucket 内对象 offset 未必天然 4KB 对齐时的修正空间。

在 KVCache 场景，chunk/page 大小直接影响 I/O 效率：

- 太小：每个 KV block 一个很小 I/O，NVMe queue depth 打不满，元数据开销高。
- 太大：读放大严重，明明只要一个小前缀，却读取一大段无用数据。
- 推荐思路：让 KV page/block 与 Mooncake bucket 里的连续布局对齐，优先批量读写。

Mooncake 当前优化点：

- `BucketStorageBackend` 聚合多个对象成 bucket，减少小文件数量。
- `UringFile::batch_read()` 一次提交最多 `QUEUE_DEPTH=32` 个独立 read。
- `vector_write()` / `vector_read()` 支持 iovec scatter/gather。
- `BucketStorageBackend::BatchLoad()` 会按 bucket 分组，减少重复 open 和随机访问。

### Prefetching 机制

HiCache 文档里把 Mooncake 作为 L3 backend：

```text
Local match L1/L2
  -> query L3 hit prefix
  -> 达到阈值则 prefetch
  -> Mooncake RDMA 并行拉取多个 page
  -> Prefill 前或 Prefill 中使用已完成部分
```

相关文档：`docs/source/design/hicache-design.md`。

Prefetch 的关键不是“等全部 SSD 数据回来”，而是：

- `best_effort`：能算就先算，减少等待。
- `timeout`：给一个按 token 数增长的等待窗口。
- `wait_complete`：适合追求最高命中率的场景。

对 NPU 来说，推荐策略通常是：

```text
SSD -> DRAM prefetch 与 NPU 当前 batch 计算重叠
DRAM -> HBM copy 与 layer compute 重叠
```

### NPU / Ascend 侧优化

Ascend Direct Transport 支持：

- Host-to-Device
- Device-to-Host
- Device-to-Device
- HCCS / RDMA
- 异步传输：`ASCEND_USE_ASYNC_TRANSFER=1`
- Fabric Memory：`ASCEND_ENABLE_USE_FABRIC_MEM=1`

相关代码/文档：

- `docs/source/design/transfer-engine/ascend_direct_transport.md`
- `mooncake-transfer-engine/include/transport/ascend_transport/ascend_direct_transport/ascend_direct_transport.h`
- `mooncake-transfer-engine/src/transport/ascend_transport/ascend_direct_transport/ascend_direct_transport.cpp`

部署时要注意：

- `TransferEngine.initialize()` 前先 `torch.npu.set_device()`。
- HCCS 场景设备内存有 2MB 对齐要求。
- 容器内需要 `/etc/hccn.conf`。
- RDMA 超时需要配合 `HCCL_RDMA_TIMEOUT` / `HCCL_RDMA_RETRY_CNT` / `ASCEND_TRANSFER_TIMEOUT`。

## 6. 总结

NPU 场景下的 SSD KVCache 缓存，本质是把 KVCache 从“只能放在 HBM 的临时数据”，升级成“可跨实例、跨阶段、跨存储层复用的分布式对象”。

Mooncake 中这套机制的核心闭环是：

```text
Prefill 生成 KV
  -> Connector / Store 写入 Mooncake DRAM
  -> Master 在 DRAM 压力下选择 offload
  -> FileStorage 后台 heartbeat 写 SSD
  -> Master 记录 LOCAL_DISK replica
  -> 新请求 hash 命中
  -> SSD holder 读入 ClientBuffer
  -> TransferEngine 拉回请求方
  -> 必要时 promotion 回 DRAM
  -> Decode 继续使用 KV
```

最关键的源码入口是：

- Connector：`mooncake-wheel/mooncake/mooncake_connector_v1.py`
- Store client：`mooncake-store/src/real_client.cpp`
- Store service：`mooncake-store/src/client_service.cpp`
- Master metadata：`mooncake-store/src/master_service.cpp`
- SSD coordinator：`mooncake-store/src/file_storage.cpp`
- SSD backend：`mooncake-store/src/storage_backend.cpp`
- Transfer Engine submit：`mooncake-store/src/transfer_task.cpp`
- NPU transport：`mooncake-transfer-engine/src/transport/ascend_transport/ascend_direct_transport/`

一句话概括：**HBM 当办公桌，DRAM 当抽屉，SSD 当档案柜；Mooncake Master 管目录，Transfer Engine 管搬运，FileStorage 管落盘和唤醒。**
