
  池化请求查询时间

  ┌────────────────────────────────────────────────────────────────────────────────────────┬────────────────────────────────────────────┬────────────────────┬────────────────────┬───────────────────────────┐
  │ 文件                                                                                   │ 方法                                       │ 打点位置           │ 关键变量           │ 建议耗时变量名            │
  ├────────────────────────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────┼────────────────────┼────────────────────┼───────────────────────────┤
  │ vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py:201         │ KVPoolScheduler.get_num_new_matched_tokens │ self.client.lookup │ request.request_id │ lookup_elapsed_ms         │
  │                                                                                        │                                            │ (...) 前后         │ , token_len,       │                           │
  │                                                                                        │                                            │                    │ request.block_hash │                           │
  │                                                                                        │                                            │                    │ es,                │                           │
  │                                                                                        │                                            │                    │ self.kv_cache_grou │                           │
  │                                                                                        │                                            │                    │ p_ids,             │                           │
  │                                                                                        │                                            │                    │ num_external_hit_t │                           │
  │                                                                                        │                                            │                    │ okens              │                           │
  │ vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py:561         │ LookupKeyClient.lookup                     │ send_multipart 到  │ token_len,         │ lookup_rpc_elapsed_ms     │
  │                                                                                        │                                            │ recv 前后          │ block_hashes,      │                           │
  │                                                                                        │                                            │                    │ kv_cache_group_ids │                           │
  │                                                                                        │                                            │                    │ , result           │                           │
  │ vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/ascend_store_connector.py:276 │ LookupKeyServer.process_request 内部函数   │ self.pool_worker.l │ token_len,         │ lookup_server_elapsed_ms  │
  │                                                                                        │                                            │ ookup_scheduler(.. │ kv_group_ids,      │                           │
  │                                                                                        │                                            │ .) 前后            │ hashes_str, result │                           │
  │ vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py:894            │ KVPoolWorker.lookup_scheduler              │ self.m_store.exist │ group_id, keys,    │ backend_exists_elapsed_ms │
  │                                                                                        │                                            │ s(multi_tp_keys)   │ multi_tp_keys,     │                           │
  │                                                                                        │                                            │ 前后               │ res,               │                           │
  │                                                                                        │                                            │                    │ first_missing,     │                           │
  │                                                                                        │                                            │                    │ hits               │                           │
  │ vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py:785            │ KVPoolWorker.lookup                        │ self.m_store.exist │ group_id, keys,    │ backend_exists_elapsed_ms │
  │                                                                                        │                                            │ s(keys) 前后       │ res, hits          │                           │
  │ vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py:96             │ KVTransferThread.lookup                    │ self.m_store.exist │ keys, res,         │ store_exists_elapsed_ms   │
  │                                                                                        │                                            │ s(keys) 前后       │ exists_list        │                           │
  └────────────────────────────────────────────────────────────────────────────────────────┴────────────────────────────────────────────┴────────────────────┴────────────────────┴───────────────────────────┘

  KVCache 读取时间

  ┌─────────────────────────────────────────────────────────────────────────────┬────────────────────────────────────────────────┬────────────────────────┬─────────────────────────┬─────────────────────────┐
  │ 文件                                                                        │ 方法                                           │ 打点位置               │ 关键变量                │ 建议耗时变量名          │
  ├─────────────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────┼────────────────────────┼─────────────────────────┼─────────────────────────┤
  │ vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py:433 │ KVPoolWorker.start_load_kv                     │ 同步路径               │ request.req_id,         │ kv_get_elapsed_ms       │
  │                                                                             │                                                │ self.m_store.get(key_l │ token_len,              │                         │
  │                                                                             │                                                │ ist_c, addr_list_c,    │ load_group_ids,         │                         │
  │                                                                             │                                                │ size_list_c) 前后      │ key_list_c,             │                         │
  │                                                                             │                                                │                        │ addr_list_c,            │                         │
  │                                                                             │                                                │                        │ size_list_c             │                         │
  │ vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py:384 │ KVCacheStoreRecvingThread._handle_request      │ 异步路径               │ req_id, token_len,      │ kv_get_elapsed_ms       │
  │                                                                             │                                                │ self.m_store.get(key_l │ req_meta.kv_cache_group │                         │
  │                                                                             │                                                │ ist_c, addr_list_c,    │ _ids, key_list_c,       │                         │
  │                                                                             │                                                │ size_list_c) 前后      │ addr_list_c,            │                         │
  │                                                                             │                                                │                        │ size_list_c             │                         │
  │ vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py:550 │ KVCacheStoreLayerRecvingThread._handle_request │ layerwise 路径         │ req_meta.req_id,        │ layer_kv_get_elapsed_ms │
  │                                                                             │                                                │ self.m_store.get(...)  │ req_meta.layer_id,      │                         │
  │                                                                             │                                                │ 前后                   │ key_list_c,             │                         │
  │                                                                             │                                                │                        │ addr_list_c,            │                         │
  │                                                                             │                                                │                        │ size_list_c             │                         │
  └─────────────────────────────────────────────────────────────────────────────┴────────────────────────────────────────────────┴────────────────────────┴─────────────────────────┴─────────────────────────┘

  KVCache 存储时间

  ┌─────────────────────────────────────────────────────────────────────────────┬────────────────────────────────────────────────┬──────────┬──────────┬──────────────────────────────────────────────────────┐
  │ 文件                                                                        │ 方法                                           │ 打点位置 │ 关键变量 │ 建议耗时变量名                                       │
  ├─────────────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────┼──────────┼──────────┼──────────────────────────────────────────────────────┤
  │ vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py:244 │ KVCacheStoreSendingThread._handle_request      │ current_ │ req_id,  │ event_sync_elapsed_ms, kv_put_elapsed_ms             │
  │                                                                             │                                                │ event.sy │ group_id │                                                      │
  │                                                                             │                                                │ nchroniz │ , keys,  │                                                      │
  │                                                                             │                                                │ e() 和   │ addrs,   │                                                      │
  │                                                                             │                                                │ self.m_s │ sizes,   │                                                      │
  │                                                                             │                                                │ tore.put │ missing_ │                                                      │
  │                                                                             │                                                │ (keys,   │ indices, │                                                      │
  │                                                                             │                                                │ addrs,   │ current_ │                                                      │
  │                                                                             │                                                │ sizes)   │ event    │                                                      │
  │                                                                             │                                                │ 分别打点 │          │                                                      │
  │ vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py:467 │ KVCacheStoreLayerSendingThread._handle_request │ current_ │ req_meta │ layer_event_sync_elapsed_ms, layer_kv_put_elapsed_ms │
  │                                                                             │                                                │ event.sy │ .req_id, │                                                      │
  │                                                                             │                                                │ nchroniz │ layer_id │                                                      │
  │                                                                             │                                                │ e() 和   │ ,        │                                                      │
  │                                                                             │                                                │ self.m_s │ key_list │                                                      │
  │                                                                             │                                                │ tore.put │ ,        │                                                      │
  │                                                                             │                                                │ (key_lis │ addr_lis │                                                      │
  │                                                                             │                                                │ t,       │ t,       │                                                      │
  │                                                                             │                                                │ addr_lis │ size_lis │                                                      │
  │                                                                             │                                                │ t,       │ t,       │                                                      │
  │                                                                             │                                                │ size_lis │ missing_ │                                                      │
  │                                                                             │                                                │ t) 分别  │ indices, │                                                      │
  │                                                                             │                                                │ 打点     │ current_ │                                                      │
  │                                                                             │                                                │          │ event    │                                                      │
  │ vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py:563 │ KVPoolWorker.wait_for_save                     │ self.kv_ │ has_save │ wait_for_save_elapsed_ms                             │
  │                                                                             │                                                │ send_thr │ _request │                                                      │
  │                                                                             │                                                │ ead.requ │ ,        │                                                      │
  │                                                                             │                                                │ est_queu │ connecto │                                                      │
  │                                                                             │                                                │ e.join() │ r_metada │                                                      │
  │                                                                             │                                                │ 前后     │ ta.reque │                                                      │
  │                                                                             │                                                │          │ sts      │                                                      │
  └─────────────────────────────────────────────────────────────────────────────┴────────────────────────────────────────────────┴──────────┴──────────┴──────────────────────────────────────────────────────┘

  后端真实读写耗时

  如果希望只统计后端库调用本身，而不包含 key/address 构造耗时，在 backend 里打点：

  ┌──────────────────────────────────────────────────────────────────────────────────────────┬────────────────┬─────────────────────────────────────────────────┬─────────────────────────────────────────────┐
  │ 文件                                                                                     │ 方法           │ 实际后端调用                                    │ 关键变量                                    │
  ├──────────────────────────────────────────────────────────────────────────────────────────┼────────────────┼─────────────────────────────────────────────────┼─────────────────────────────────────────────┤
  │ vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/memcache_backend.py:59  │ exists/get/put │ batch_is_exist, batch_get_into_layers,          │ keys/key, addr, size, res                   │
  │                                                                                          │                │ batch_put_from_layers                           │                                             │
  │ vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/mooncake_backend.py:80  │ exists/get/put │ batch_is_exist, batch_get_into_multi_buffers,   │ keys, addrs, sizes, res/res_list            │
  │                                                                                          │                │ batch_put_from_multi_buffers                    │                                             │
  │ vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/yuanrong_backend.py:133 │ exists/get/put │ exist, mget_h2d, mset_d2h                       │ keys, addrs, sizes, blob_lists, failed_keys │
  └──────────────────────────────────────────────────────────────────────────────────────────┴────────────────┴─────────────────────────────────────────────────┴─────────────────────────────────────────────┘

  建议命名统一用 *_start_time 和 *_elapsed_ms，计时用 time.perf_counter()。上层 kv_transfer.py/pool_worker.py 的打点能反映请求端感知耗时；backend 打点能反映具体存储后端调用耗时。
