import unittest
from unittest.mock import MagicMock, patch, call, ANY
import os
import torch
import numpy as np
from collections import defaultdict

# 模拟依赖项
class MockVllmConfig:
    def __init__(self):
        class MockCacheConfig:
            block_size = 16
        
        class MockParallelConfig:
            tensor_parallel_size = 2
            data_parallel_size = 1
            data_parallel_rank_local = 0
        
        class MockKVTransferConfig:
            kv_port = 5000
            engine_id = "test_engine"
            kv_role = "kv_producer"
            
            def get_from_extra_config(self, key, default):
                if key == "prefill":
                    return {"tp_size": 4, "dp_size": 1}
                elif key == "decode":
                    return {"tp_size": 2, "dp_size": 1}
                return default
        
        self.cache_config = MockCacheConfig()
        self.parallel_config = MockParallelConfig()
        self.kv_transfer_config = MockKVTransferConfig()

class MockRequest:
    def __init__(self, request_id, prompt_token_ids=None, kv_transfer_params=None, status=None):
        self.request_id = request_id
        self.prompt_token_ids = prompt_token_ids or [1, 2, 3, 4]
        self.kv_transfer_params = kv_transfer_params or {}
        self.status = status or "running"
        self.output_token_ids = [101, 102]

class MockKVCacheBlocks:
    def get_unhashed_block_ids(self):
        return [4, 5, 6]

class MockSchedulerOutput:
    pass

class MockForwardContext:
    pass

# 测试 MooncakeConnector 主类
class TestMooncakeConnector(unittest.TestCase):
    def setUp(self):
        self.config = MockVllmConfig()
        # 设置环境变量
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0,1"
    
    def test_scheduler_initialization(self):
        connector = MooncakeConnector(self.config, KVConnectorRole.SCHEDULER)
        self.assertIsNotNone(connector.connector_scheduler)
        self.assertIsNone(connector.connector_worker)
    
    def test_worker_initialization(self):
        connector = MooncakeConnector(self.config, KVConnectorRole.WORKER)
        self.assertIsNone(connector.connector_scheduler)
        self.assertIsNotNone(connector.connector_worker)
    
    @patch.object(MooncakeConnectorScheduler, "get_num_new_matched_tokens")
    def test_get_num_new_matched_tokens(self, mock_method):
        connector = MooncakeConnector(self.config, KVConnectorRole.SCHEDULER)
        request = MockRequest("req1")
        connector.get_num_new_matched_tokens(request, 0)
        mock_method.assert_called_once_with(request, 0)
    
    @patch.object(MooncakeConnectorScheduler, "update_state_after_alloc")
    def test_update_state_after_alloc(self, mock_method):
        connector = MooncakeConnector(self.config, KVConnectorRole.SCHEDULER)
        request = MockRequest("req1")
        blocks = MockKVCacheBlocks()
        connector.update_state_after_alloc(request, blocks, 3)
        mock_method.assert_called_once_with(request, blocks, 3)
    
    @patch.object(MooncakeConnectorScheduler, "build_connector_meta")
    def test_build_connector_meta(self, mock_method):
        connector = MooncakeConnector(self.config, KVConnectorRole.SCHEDULER)
        scheduler_output = MockSchedulerOutput()
        connector.build_connector_meta(scheduler_output)
        mock_method.assert_called_once_with(scheduler_output)
    
    @patch.object(MooncakeConnectorScheduler, "request_finished")
    def test_request_finished(self, mock_method):
        connector = MooncakeConnector(self.config, KVConnectorRole.SCHEDULER)
        request = MockRequest("req1")
        connector.request_finished(request, [1, 2, 3])
        mock_method.assert_called_once_with(request, [1, 2, 3])
    
    @patch.object(MooncakeConnectorWorker, "register_kv_caches")
    def test_register_kv_caches(self, mock_method):
        connector = MooncakeConnector(self.config, KVConnectorRole.WORKER)
        kv_caches = {"layer1": (torch.zeros(10, 16, 8, 64), torch.zeros(10, 16, 8, 64)}
        connector.register_kv_caches(kv_caches)
        mock_method.assert_called_once_with(kv_caches)
    
    @patch.object(MooncakeConnectorWorker, "get_finished")
    def test_get_finished(self, mock_method):
        connector = MooncakeConnector(self.config, KVConnectorRole.WORKER)
        finished_req_ids = {"req1", "req2"}
        connector.get_finished(finished_req_ids)
        mock_method.assert_called_once()
    
    @patch.object(MooncakeConnectorWorker, "start_load_kv")
    def test_start_load_kv(self, mock_method):
        connector = MooncakeConnector(self.config, KVConnectorRole.WORKER)
        # 设置内部状态
        connector._connector_metadata = MooncakeConnectorMetadata()
        forward_context = MockForwardContext()
        connector.start_load_kv(forward_context)
        mock_method.assert_called_once_with(ANY)

# 测试 MooncakeConnectorScheduler
class TestMooncakeConnectorScheduler(unittest.TestCase):
    def setUp(self):
        self.config = MockVllmConfig()
        self.scheduler = MooncakeConnectorScheduler(self.config, "test_engine")
    
    def test_get_num_new_matched_tokens_no_remote_prefill(self):
        request = MockRequest("req1")
        tokens, async_flag = self.scheduler.get_num_new_matched_tokens(request, 0)
        self.assertEqual(tokens, 0)
        self.assertFalse(async_flag)
    
    def test_get_num_new_matched_tokens_with_remote_prefill(self):
        request = MockRequest("req1", kv_transfer_params={"do_remote_prefill": True})
        tokens, async_flag = self.scheduler.get_num_new_matched_tokens(request, 0)
        self.assertEqual(tokens, 3)  # len(prompt_token_ids) - 1
        self.assertTrue(async_flag)
    
    def test_update_state_after_alloc_no_remote_prefill(self):
        request = MockRequest("req1")
        blocks = MagicMock()
        self.scheduler.update_state_after_alloc(request, blocks, 0)
        self.assertEqual(len(self.scheduler._reqs_need_recv), 0)
    
    def test_update_state_after_alloc_with_remote_prefill(self):
        request = MockRequest("req1", kv_transfer_params={
            "do_remote_prefill": True,
            "remote_block_ids": [1, 2, 3],
            "remote_engine_id": "remote",
            "remote_host": "localhost",
            "remote_port": 5000
        })
        blocks = MockKVCacheBlocks()
        self.scheduler.update_state_after_alloc(request, blocks, 3)
        self.assertEqual(len(self.scheduler._reqs_need_recv), 1)
        self.assertEqual(self.scheduler._reqs_need_recv["req1"][0], request)
        self.assertEqual(self.scheduler._reqs_need_recv["req1"][1], [4, 5, 6])
    
    def test_build_connector_meta(self):
        request = MockRequest("req1", kv_transfer_params={
            "remote_block_ids": [1, 2, 3],
            "remote_engine_id": "remote",
            "remote_host": "localhost",
            "remote_port": 5000
        })
        blocks = MockKVCacheBlocks()
        self.scheduler.update_state_after_alloc(request, blocks, 3)
        
        meta = self.scheduler.build_connector_meta(MockSchedulerOutput())
        self.assertIsInstance(meta, MooncakeConnectorMetadata)
        self.assertEqual(len(meta.requests), 1)
        req_meta = meta.requests["req1"]
        self.assertEqual(req_meta.local_block_ids, [4, 5, 6])
        self.assertEqual(req_meta.remote_block_ids, [1, 2, 3])
        self.assertEqual(req_meta.remote_engine_id, "remote")
        self.assertEqual(req_meta.remote_host, "localhost")
        self.assertEqual(req_meta.remote_port, 5000)
        
        # 构建后应该清除请求列表
        self.assertEqual(len(self.scheduler._reqs_need_recv), 0)
    
    def test_request_finished_no_remote_decode(self):
        request = MockRequest("req1")
        delay_free, params = self.scheduler.request_finished(request, [1, 2, 3])
        self.assertFalse(delay_free)
        self.assertIsNone(params)
    
    def test_request_finished_with_remote_decode(self):
        request = MockRequest("req1", 
                             kv_transfer_params={"do_remote_decode": True},
                             status="finished")
        delay_free, params = self.scheduler.request_finished(request, [1, 2, 3])
        self.assertTrue(delay_free)
        self.assertIsNotNone(params)
        self.assertEqual(params["remote_engine_id"], "test_engine")
        self.assertEqual(params["remote_host"], self.scheduler.side_channel_host)

# 测试 MooncakeConnectorWorker
class TestMooncakeConnectorWorker(unittest.TestCase):
    def setUp(self):
        self.config = MockVllmConfig()
        self.worker = MooncakeConnectorWorker(self.config, "test_worker")
        self.worker.engine = MagicMock()  # 模拟TransferEngine
    
    @patch("mooncake_connector.KVCacheSendingThread")
    @patch("mooncake_connector.KVCacheRecvingThread")
    def test_register_kv_caches_producer(self, mock_recv_thread, mock_send_thread):
        # 创建模拟的KV缓存
        kv_caches = {
            "layer1": (
                torch.zeros((10, 16, 8, 64)),  # [num_blocks, block_size, num_heads, head_dim]
                torch.zeros((10, 16, 8, 64))
        }
        
        self.worker.register_kv_caches(kv_caches)
        
        # 验证元数据设置
        self.assertEqual(self.worker.num_blocks, 10)
        self.assertEqual(len(self.worker.block_len), 1)  # 非MLA情况
        
        # 验证注册了内存
        self.worker.engine.register_memory.assert_called()
        
        # 验证发送线程启动
        self.assertIsNotNone(self.worker.kv_send_thread)
        self.assertTrue(self.worker.kv_send_thread.ready_event.is_set())
    
    @patch("mooncake_connector.KVCacheSendingThread")
    @patch("mooncake_connector.KVCacheRecvingThread")
    def test_register_kv_caches_consumer(self, mock_recv_thread, mock_send_thread):
        # 设置为消费者角色
        self.config.kv_transfer_config.kv_role = "kv_consumer"
        worker = MooncakeConnectorWorker(self.config, "test_worker")
        worker.engine = MagicMock()
        
        # 创建模拟的KV缓存
        kv_caches = {
            "layer1": (
                torch.zeros((10, 16, 8, 64)),  # [num_blocks, block_size, num_heads, head_dim]
                torch.zeros((10, 16, 8, 64))
        }
        
        worker.register_kv_caches(kv_caches)
        
        # 验证接收线程启动
        self.assertIsNotNone(worker.kv_recv_thread)
        self.assertTrue(worker.kv_recv_thread.ready_event.is_set())
    
    @patch("mooncake_connector.KVCacheRecvingThread")
    def test_start_load_kv(self, mock_recv_thread):
        # 设置worker为消费者角色
        self.config.kv_transfer_config.kv_role = "kv_consumer"
        worker = MooncakeConnectorWorker(self.config, "test_worker")
        worker.engine = MagicMock()
        worker.kv_recv_thread = MagicMock()
        
        # 创建元数据
        meta = MooncakeConnectorMetadata()
        meta.add_new_req(
            request_id="req1",
            local_block_ids=[1, 2],
            kv_transfer_params={
                "remote_block_ids": [3, 4],
                "remote_engine_id": "remote_engine",
                "remote_host": "localhost",
                "remote_port": 5000
            }
        )
        
        worker.start_load_kv(meta)
        
        # 验证添加了请求到接收线程
        worker.kv_recv_thread.add_request.assert_called_once_with(
            request_id="req1",
            local_block_ids=[1, 2],
            remote_block_ids=[3, 4],
            remote_engine_id="remote_engine",
            remote_host="localhost",
            remote_handshake_port=ANY,  # 5000 + remote_tp_rank
            remote_transfer_port=ANY
        )
    
    def test_get_finished_producer(self):
        # 模拟发送线程
        self.worker.kv_send_thread = MagicMock()
        self.worker.kv_send_thread.get_and_clear_finished_requests.return_value = {"req1", "req2"}
        
        done_sending, done_recving = self.worker.get_finished()
        self.assertEqual(done_sending, {"req1", "req2"})
        self.assertEqual(done_recving, set())
    
    def test_get_finished_consumer(self):
        # 设置为消费者角色
        self.config.kv_transfer_config.kv_role = "kv_consumer"
        worker = MooncakeConnectorWorker(self.config, "test_worker")
        worker.engine = MagicMock()
        
        # 模拟接收线程
        worker.kv_recv_thread = MagicMock()
        worker.kv_recv_thread.get_and_clear_finished_requests.return_value = {"req3"}
        
        done_sending, done_recving = worker.get_finished()
        self.assertEqual(done_sending, set())
        self.assertEqual(done_recving, {"req3"})
    
    def test_remote_tp_ranks_same_size(self):
        # 当预填充和decode的tp大小相同时
        self.worker._prefill_tp_size = 2
        self.worker._decode_tp_size = 2
        ranks = self.worker._get_remote_tp_ranks_for_req("req1")
        self.assertEqual(ranks, [0, 1])
    
    def test_remote_tp_ranks_different_size(self):
        # 当预填充和decode的tp大小不同时
        self.worker._prefill_tp_size = 4
        self.worker._decode_tp_size = 2
        
        # 模拟随机采样
        with patch("random.Random.sample") as mock_sample:
            mock_sample.return_value = [1, 3]
            ranks = self.worker._get_remote_tp_ranks_for_req("req1")
            self.assertEqual(ranks, [1, 3])
            
            # 验证使用正确的种子
            mock_sample.assert_called_once_with(range(4), 2)

# 测试辅助函数
class TestHelperFunctions(unittest.TestCase):
    def test_group_concurrent_contiguous(self):
        src = [1, 2, 3, 5, 6]
        dst = [10, 11, 12, 14, 15]
        
        src_groups, dst_groups = group_concurrent_contiguous(src, dst)
        
        self.assertEqual(len(src_groups), 2)
        self.assertEqual(src_groups[0], [1, 2, 3])
        self.assertEqual(src_groups[1], [5, 6])
        self.assertEqual(dst_groups[0], [10, 11, 12])
        self.assertEqual(dst_groups[1], [14, 15])
    
    def test_group_concurrent_contiguous_empty(self):
        src = []
        dst = []
        src_groups, dst_groups = group_concurrent_contiguous(src, dst)
        self.assertEqual(src_groups, [])
        self.assertEqual(dst_groups, [])
    
    def test_string_to_int64_hash(self):
        # 验证相同字符串产生相同哈希
        hash1 = string_to_int64_hash("test_string")
        hash2 = string_to_int64_hash("test_string")
        self.assertEqual(hash1, hash2)
        
        # 验证不同字符串产生不同哈希
        hash3 = string_to_int64_hash("different_string")
        self.assertNotEqual(hash1, hash3)

# 运行测试
if __name__ == "__main__":
    unittest.main()
