# Ascend Store 池化代码耗时打点位置

本文档整理 `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store` 路径下池化代码中建议打点的位置，重点覆盖：

- 池化请求查询时间
- KVCache 读取时间
- KVCache 存储时间
- 后端真实读写耗时

建议统一使用 `time.perf_counter()` 计时，变量命名采用 `*_start_time` 和 `*_elapsed_ms`。

## 池化请求查询时间

| 文件 | 方法 | 打点位置 | 关键变量 | 建议耗时变量名 |
|---|---|---|---|---|
| `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py` | `KVPoolScheduler.get_num_new_matched_tokens` | `self.client.lookup(...)` 前后 | `request.request_id`, `token_len`, `request.block_hashes`, `self.kv_cache_group_ids`, `num_external_hit_tokens` | `lookup_elapsed_ms` |
| `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py` | `LookupKeyClient.lookup` | `send_multipart` 到 `recv` 前后 | `token_len`, `block_hashes`, `kv_cache_group_ids`, `result` | `lookup_rpc_elapsed_ms` |
| `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/ascend_store_connector.py` | `LookupKeyServer.process_request` 内部函数 | `self.pool_worker.lookup_scheduler(...)` 前后 | `token_len`, `kv_group_ids`, `hashes_str`, `result` | `lookup_server_elapsed_ms` |
| `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py` | `KVPoolWorker.lookup_scheduler` | `self.m_store.exists(multi_tp_keys)` 前后 | `group_id`, `keys`, `multi_tp_keys`, `res`, `first_missing`, `hits` | `backend_exists_elapsed_ms` |
| `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py` | `KVPoolWorker.lookup` | `self.m_store.exists(keys)` 前后 | `group_id`, `keys`, `res`, `hits` | `backend_exists_elapsed_ms` |
| `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py` | `KVTransferThread.lookup` | `self.m_store.exists(keys)` 前后 | `keys`, `res`, `exists_list` | `store_exists_elapsed_ms` |

## KVCache 读取时间

| 文件 | 方法 | 打点位置 | 关键变量 | 建议耗时变量名 |
|---|---|---|---|---|
| `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py` | `KVPoolWorker.start_load_kv` | 同步路径 `self.m_store.get(key_list_c, addr_list_c, size_list_c)` 前后 | `request.req_id`, `token_len`, `load_group_ids`, `key_list_c`, `addr_list_c`, `size_list_c` | `kv_get_elapsed_ms` |
| `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py` | `KVCacheStoreRecvingThread._handle_request` | 异步路径 `self.m_store.get(key_list_c, addr_list_c, size_list_c)` 前后 | `req_id`, `token_len`, `req_meta.kv_cache_group_ids`, `key_list_c`, `addr_list_c`, `size_list_c` | `kv_get_elapsed_ms` |
| `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py` | `KVCacheStoreLayerRecvingThread._handle_request` | layerwise 路径 `self.m_store.get(...)` 前后 | `req_meta.req_id`, `req_meta.layer_id`, `key_list_c`, `addr_list_c`, `size_list_c` | `layer_kv_get_elapsed_ms` |

## KVCache 存储时间

| 文件 | 方法 | 打点位置 | 关键变量 | 建议耗时变量名 |
|---|---|---|---|---|
| `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py` | `KVCacheStoreSendingThread._handle_request` | `current_event.synchronize()` 和 `self.m_store.put(keys, addrs, sizes)` 分别打点 | `req_id`, `group_id`, `keys`, `addrs`, `sizes`, `missing_indices`, `current_event` | `event_sync_elapsed_ms`, `kv_put_elapsed_ms` |
| `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py` | `KVCacheStoreLayerSendingThread._handle_request` | `current_event.synchronize()` 和 `self.m_store.put(key_list, addr_list, size_list)` 分别打点 | `req_meta.req_id`, `layer_id`, `key_list`, `addr_list`, `size_list`, `missing_indices`, `current_event` | `layer_event_sync_elapsed_ms`, `layer_kv_put_elapsed_ms` |
| `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py` | `KVPoolWorker.wait_for_save` | `self.kv_send_thread.request_queue.join()` 前后 | `has_save_request`, `connector_metadata.requests` | `wait_for_save_elapsed_ms` |

## 后端真实读写耗时

如果只统计后端库调用本身，而不包含 key/address 构造耗时，建议在 backend 实现里打点。

| 文件 | 方法 | 实际后端调用 | 关键变量 |
|---|---|---|---|
| `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/memcache_backend.py` | `exists/get/put` | `batch_is_exist`, `batch_get_into_layers`, `batch_put_from_layers` | `keys/key`, `addr`, `size`, `res` |
| `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/mooncake_backend.py` | `exists/get/put` | `batch_is_exist`, `batch_get_into_multi_buffers`, `batch_put_from_multi_buffers` | `keys`, `addrs`, `sizes`, `res/res_list` |
| `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/yuanrong_backend.py` | `exists/get/put` | `exist`, `mget_h2d`, `mset_d2h` | `keys`, `addrs`, `sizes`, `blob_lists`, `failed_keys` |

## 打点层级建议

上层 `kv_transfer.py` 和 `pool_worker.py` 的打点用于观察请求端感知耗时，包含 key 生成、地址准备、队列等待、NPU event 同步等上下文开销。

backend 目录下的打点用于观察具体存储后端调用耗时，适合区分 Mooncake、Memcache、Yuanrong 等后端自身的读写和查询开销。

推荐同时保留两层打点：

- 上层打点：定位业务请求整体耗时。
- backend 打点：定位具体后端库调用耗时。

