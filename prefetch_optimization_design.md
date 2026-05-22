# 预取功能技术方案设计文档

## 1. 概述与设计背景 (Overview & Background)

### 1.1 业务背景与痛点

当前分支 `feature/prefetch-optimization` 的代码实现并非原生 Memcached 的 `libevent + worker thread + item/slab` 结构，而是 MemCache Hybrid 的 C++ MetaService/LocalService/Client 架构。其预取能力主要服务于 **UBS_IO SSD 冷数据回读到 DRAM** 的场景。

传统缓存访问路径中，DRAM 元数据或 READABLE blob 不存在时，请求会退化到 UBS_IO/SSD 直接读取。该路径虽然保证可用性，但在批量 key、高并发 miss、热点冷数据重新变热等场景下，会带来以下问题：

- 首次访问延迟被 SSD/UBS_IO 拉长，形成 Tail Latency。
- 多个请求同时 miss 同一个 key 时，可能重复触发后端读取或 DRAM 回填。
- 冷热分层下，如果数据长期停留在 SSD，后续连续读无法复用 DRAM 高速路径。
- 批量查询存在“先探测存在性、再实际读取”的机会，如果不提前搬迁数据，后续 get 仍然走慢路径。

### 1.2 预取优化目标

本分支的目标是：当 MetaService 发现 key 不在 DRAM 元数据中，但 UBS_IO 后端仍存在该 key 时，异步将 SSD 数据搬迁到 DRAM，使后续读取能够命中 DRAM blob。

核心优化目标：

- **提升后续读命中率**：把 UBS_IO 中存在但 DRAM 中缺失的对象提前加载回 DRAM。
- **降低二次访问延迟**：首轮 `BatchExistKey` 触发预取，后续 `Get/BatchGet` 直接返回 READABLE blob。
- **控制回填风暴**：通过 `pendingPrefetchKeys_` 对同一 key 的预取任务做并发去重。
- **保持前台请求轻量**：触发线程只做 UBS_IO 长度探测和任务入队，实际 SSD -> DRAM 复制交给 `prefetch_pool`。

## 2. 总体架构与核心设计 (High-Level Design)

### 2.1 预取触发机制

当前实现属于 **服务端基于存在性探测的显式批量触发**，触发入口是 `src/memcache/csrc/meta_service/mmc_meta_mgr_proxy.cpp` 中的 `MmcMetaMgrProxy::BatchExistKey()`。

触发条件如下：

1. 客户端调用 batch exist。
2. MetaService 先调用 `metaMangerPtr_->ExistKey()` 检查 DRAM 元数据。
3. 对 `resp.results_[i] != MMC_OK` 的 key，认为其未在 DRAM 可读路径中命中。
4. 若 `ubsIoEnable_ == true` 且 `ubsIoPrefetchEnable_ == true`，调用 `ubsIoProxy_->BatchGetLength()` 探测 UBS_IO 是否存在以及对象长度。
5. 对 UBS_IO 返回成功的 key，将结果改为 `MMC_OK`，并加入 `toBePrefetchedKeys/toBePrefetchedLengths`。
6. 调用 `metaMangerPtr_->BatchPrefetch()` 异步发起 SSD -> DRAM 预取。

配置开关来自：

- `ock.mmc.ubs_io.enable`
- `ock.mmc.ubs_io.prefetch.enable`

对应配置项位于 `src/memcache/csrc/config/mmc_config_const.h` 和 `config/mmc-meta.conf`。

当前代码未发现文本协议或 Meta 协议 `mg/ms` flag 扩展，也没有基于历史访问窗口的预测逻辑。当前触发点更接近“batch exist + UBS_IO 存在性探测 + 异步升温”。

### 2.2 线程模型与交互

线程模型由 `MmcMetaManager` 内部两个线程池承担：

- `threadPool_ = MmcThreadPool("metamgr_pool", 16)`：用于异步淘汰。
- `prefetchPool_ = MmcThreadPool("prefetch_pool", 32)`：用于预取任务。

初始化位于 `src/memcache/csrc/meta_service/mmc_meta_manager.h`。线程池实现是标准 mutex + condition_variable + FIFO queue，位于 `src/memcache/csrc/common/mmc_thread_pool.h`。

预取任务生命周期：

1. `BatchExistKey()` 在 MetaService RPC 处理线程中识别待预取 key。
2. `BatchPrefetch()` 对 key 做 pending 去重。
3. `prefetchPool_->Enqueue()` 投递 lambda。
4. 后台线程执行 `CopyBlobUp(key, len)`。
5. `CopyBlobUp()` 分配 DRAM blob，插入 meta container，RPC 到目标 LocalService 执行 `CopyBlob`。
6. LocalService 识别 `src.mediaType_ == MEDIA_SSD`，调用 `ubsIoProxyPtr_->Get()` 把 SSD 数据读入目标 DRAM 地址。
7. MetaService 将 blob 状态更新为 READABLE，并通知等待中的读请求。

### 2.3 资源隔离与生存期

预取数据最终进入正常 DRAM 内存池和正常 LRU 容器，没有独立 slab、独立 LRU 或单独内存配额。

实现细节：

- `CopyBlobUp()` 使用 `AllocOptions` 分配 `MEDIA_DRAM`，`numBlobs_ = 1`，`flags_ = ALLOC_RANDOM`。
- 分配前调用 `CheckAndEvict(MEDIA_DRAM, len)`，触发既有多级淘汰机制。
- 插入 `metaContainer_` 后，key 进入对应媒体类型的 LRU 链表。
- TTL 字段 `defaultTtlMs_` 存在，但当前预取路径没有设置独立 TTL，也没有预取对象专属过期策略。
- 预取可能挤占正常 DRAM 数据，但会先经过 `evictThresholdHigh/Low` 控制。

## 3. 核心模块与关键代码实现 (Detailed Module & Implementation)

### 3.1 核心数据结构

核心结构集中在 `MmcMetaManager`：

- `pendingPrefetchKeys_`：`std::unordered_set<std::string>`，记录正在预取的 key，用于 key 级并发去重。
- `prefetchMutex_`：保护 `pendingPrefetchKeys_`。
- `prefetchPool_`：后台预取线程池。
- `cvs_[META_MAMAGER_MTX_NUM]`：条件变量数组，用于读请求等待 ALLOCATED blob 变为 READABLE。
- `metaContainer_`：元数据容器，LRU map/list 结构。
- `globalAllocator_`：DRAM/HBM 等介质的全局分配器。
- `evictCheck_`：淘汰任务并发保护，避免重复提交淘汰任务。

Blob 状态机使用 `BlobState`：

- `ALLOCATED`：空间已分配，但数据尚未完全写入。
- `READABLE`：数据已完成写入，可读。
- `REMOVING`：删除中。
- `NONE`：无效状态。

状态定义位于 `src/memcache/csrc/entities/mmc_blob_state.h`。

### 3.2 关键函数与伪代码解读

#### `MmcMetaMgrProxy::BatchExistKey()`

位置：`src/memcache/csrc/meta_service/mmc_meta_mgr_proxy.cpp`

核心作用：把 DRAM miss 但 UBS_IO 存在的 key 标记为存在，并异步提交预取。

```cpp
for key in req.keys:
    ret = metaManager.ExistKey(key)
    resp.results.push(ret)

if ubsIoEnable:
    ubsIoKeys = keys where resp != MMC_OK

    if !ubsIoPrefetchEnable:
        BatchExist(ubsIoKeys)
        patch resp
    else:
        BatchGetLength(ubsIoKeys)
        for each success:
            resp = MMC_OK
            collect key and length

        metaManager.BatchPrefetch(keys, lengths)
```

关键点：

- `BatchGetLength()` 同时承担“存在性探测”和“获取 DRAM 分配尺寸”的职责。
- 对 UBS_IO 存在的 key，`BatchExistKey()` 会立即返回 `MMC_OK`，即使预取还未完成。
- 因此 exist 语义被扩展为“DRAM 或 UBS_IO 存在”。

#### `MmcMetaManager::BatchPrefetch()`

位置：`src/memcache/csrc/meta_service/mmc_meta_manager.cpp`

```cpp
for each key,len:
    lock(prefetchMutex)
    if key in pendingPrefetchKeys:
        continue
    insert key
    unlock

    future = prefetchPool.Enqueue(CopyBlobUp(key, len))

    if future invalid:
        lock
        erase key
```

关键点：

- 目前注释明确说明“暂时没有批量接口，这里还得改成循环”。
- 去重粒度是 key，不区分版本、长度、介质。
- 线程池队列无容量上限，`Enqueue()` 只在 stop 时返回 invalid。

#### `MmcMetaManager::CopyBlobUp()`

位置：`src/memcache/csrc/meta_service/mmc_meta_manager.cpp`

```cpp
objMeta = new MmcMemObjMeta()
allocOpt = { blobSize=len, numBlobs=1, mediaType=DRAM, flags=ALLOC_RANDOM }

CheckAndEvict(DRAM, len)
blob = globalAllocator.Alloc(allocOpt)

blob.UpdateState(MMC_ALLOCATED_OK)
objMeta.AddBlob(blob)
metaContainer.Insert(key, objMeta)

srcBlob = { mediaType=SSD }
dstBlob = blob.GetDesc()
SyncCall(dstBlob.rank, BlobCopyRequest(key, srcBlob, dstBlob))

blob.UpdateState(MMC_WRITE_OK)
condition_variable.notify_all()
pendingPrefetchKeys.erase(key)
```

关键点：

- `metaContainer_->Insert()` 发生在数据复制前，因此读请求可能看到该 key 但 blob 仍处于 `ALLOCATED` 状态。
- 成功复制后通过 `MMC_WRITE_OK` 将 blob 转为 READABLE。
- `notify_all()` 唤醒 `Get()` 中等待该 blob 可读的线程。
- 失败时释放已分配 blob、清理 pending set，并尝试 `metaContainer_->Erase(key)`。

#### `MmcMetaManager::Get()`

位置：`src/memcache/csrc/meta_service/mmc_meta_manager.cpp`

读取逻辑：

1. 从 `metaContainer_` 获取 key。
2. `Promote(key)` 更新 LRU。
3. 持有 `memObj->Mutex()`。
4. 优先查找 `READABLE` blob，找到则更新为 `MMC_READ_START` 并返回。
5. 若开启 UBS_IO 预取且没有 READABLE blob，则查找 `ALLOCATED` blob。
6. 如果存在 ALLOCATED blob，最多等待 100ms。
7. 等到 READABLE 后返回；超时返回 `MMC_ERROR`，客户端可走 UBS_IO fallback。

这个设计让“正在预取中的 key”具备短暂等待能力，避免刚刚进入回填阶段的请求立刻穿透到 UBS_IO。

### 3.3 协议扩展（如涉及）

当前代码未发现文本协议、Meta 协议 `mg/ms`、二进制协议或 wire command 的新增 flag。预取能力通过 C/C++ 配置和 MetaService 内部 RPC 触发：

- C 配置结构：`mmc_meta_service_config_t::ubsIoPrefetchEnable`。
- 配置文件项：`ock.mmc.ubs_io.prefetch.enable`。
- 触发 RPC：`BatchExistKey`。
- 后端 RPC：`BlobCopyRequest`，由 MetaService 发给 LocalService。

因此本分支不是“协议级客户端显式 prefetch flag”，而是“MetaService 内部基于 UBS_IO 探测的异步升温”。

## 4. 关键核心流程 (Key Workflows)

### 4.1 全链路预取执行流

1. 客户端发起 `BatchExistKey(keys)`。
2. MetaService 对每个 key 调用 `ExistKey()` 检查 DRAM 元数据和 READABLE blob。
3. DRAM miss 的 key 被收集到 `ubsIoKeys`。
4. 若启用预取，调用 `ubsIoProxy_->BatchGetLength()`。
5. UBS_IO 返回成功时，MetaService 将对应 exist 结果置为 `MMC_OK`。
6. MetaService 调用 `BatchPrefetch(keys, lengths)`。
7. `BatchPrefetch()` 做 pending 去重并投递 `prefetch_pool`。
8. `CopyBlobUp()` 分配 DRAM blob，状态置为 `ALLOCATED`，插入 `metaContainer_`。
9. MetaService 发起 `BlobCopyRequest` 到目标 rank。
10. LocalService 识别 SSD 源，调用 `ubsIoProxyPtr_->Get()` 把数据写入目标 DRAM。
11. MetaService 收到成功响应，调用 `MMC_WRITE_OK`，blob 转为 READABLE。
12. 等待中的 `Get()` 被唤醒；后续读直接走 DRAM blob。

### 4.2 并发去重与防重刷（Anti-Dogpiling）

当前防重刷机制主要是 `pendingPrefetchKeys_`：

- 多个 batch exist 同时发现同一个 key 在 DRAM miss。
- 第一个线程持有 `prefetchMutex_` 插入 key。
- 后续线程发现 key 已存在，直接跳过任务提交。
- 任务成功或失败后都会 erase key。

此外，`metaContainer_->Insert()` 本身也能防止重复插入相同 key。如果两个任务绕过去重并发执行，第二个 insert 会返回 `MMC_DUPLICATED_OBJECT`，进入失败清理路径。

不过当前实现仍有几个边界：

- pending set 不保存 generation/version，无法区分同 key delete 后重新 put 的版本。
- `BatchExistKey()` 对 UBS_IO 存在的 key 立即返回 `MMC_OK`，但此时预取可能尚未完成。
- 线程池队列没有长度上限，高并发不同 key miss 时仍可能形成大量异步任务。

## 5. 性能损耗与边界场景分析 (Performance & Edge Cases)

### 5.1 潜在性能影响

额外开销主要来自：

- **UBS_IO 探测开销**：开启预取后，DRAM miss 的 batch exist 会调用 `BatchGetLength()`，增加一次后端元信息访问。
- **pending 锁竞争**：`prefetchMutex_` 是全局锁，高并发大量 key 触发时会有短临界区竞争。
- **线程池队列堆积**：`MmcThreadPool` 队列无界，突发 miss 可能造成内存膨胀和延迟扩散。
- **DRAM 挤占**：预取对象进入正常 DRAM LRU，可能提前淘汰真实热点。
- **复制前插入 meta**：`CopyBlobUp()` 先插入 ALLOCATED blob，再同步复制数据。读请求会最多等待 100ms，增加局部阻塞。
- **淘汰耦合**：预取前调用 `CheckAndEvict()`，可能触发异步淘汰线程，与正常写入共享 allocator/LRU 资源。

### 5.2 异常边界处理

预取失败：

- `CopyBlobUp()` 中任一步失败都会释放已分配 blob、清理 pending set、擦除 meta。
- 读请求等待 100ms 超时后返回 `MMC_ERROR`，客户端 `Get()` 在 UBS_IO 开启时会走直接 `ubsIoProxy_->Get()` fallback，并随后 `Put()` 回 DRAM。

预取队列满：

- 当前无队列容量限制。
- `Enqueue()` 仅在线程池 stop 时返回 invalid；失败时会清理 pending。
- 建议后续增加队列上限、丢弃策略和指标。

delete/update 并发：

- `Remove()` 会删除 meta，并在 UBS_IO 开启时删除 UBS_IO key。
- 但如果 remove 与 `CopyBlobUp()` 并发，存在预取任务后续重新 insert 同 key 的风险。
- 当前 pending set 不能感知 delete generation，也没有 tombstone 机制。
- update/put 同 key 时，`metaContainer_->Insert()` 会以重复 key 失败，但无法表达“新版本覆盖旧预取”的语义。

数据一致性：

- 预取数据源是 UBS_IO 中的 key。
- 若 UBS_IO 内容在 `BatchGetLength()` 和 `ubsIoProxy_->Get()` 之间变化，当前没有版本校验。
- 长度 `len` 来自探测阶段，若后端对象大小变化，可能出现读取尺寸不一致问题。

## 6. 总结与后续优化方向 (Conclusion & Future Work)

当前方案是一个工程上比较直接的 **UBS_IO 冷数据异步升温机制**：通过 batch exist 发现 DRAM miss，通过 UBS_IO 探测确认对象存在，再由后台 `prefetch_pool` 将 SSD 数据搬回 DRAM。它的优点是侵入面较小、触发路径明确、具备 key 级去重，并且能和现有 allocator/LRU/LocalService 复制链路复用。

主要不足是资源隔离和一致性控制还不够完整：预取共享正常 DRAM LRU，无队列背压，无版本校验，delete/update 并发下存在重新回填旧数据的风险。

后续建议：

- 增加 `prefetch_queue_limit`、drop counter、inflight gauge 等背压与观测指标。
- pending set 升级为 `{key, version/generation, length, timestamp}`。
- 对预取对象引入低优先级 LRU 或独立 quota，避免挤占正常热点。
- `BatchPrefetch()` 改为真正批量复制接口，减少 RPC 和线程调度开销。
- 基于滑动窗口统计 batch exist/get miss，做自适应预取步长。
- 增加 delete tombstone 或 CAS/version 校验，避免预取任务写回过期数据。
- 将 100ms 等待时间配置化，并区分前台 get 和后台批量 get 的等待策略。
