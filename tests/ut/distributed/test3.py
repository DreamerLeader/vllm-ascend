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

# Mock 全局函数
def mock_get_tensor_model_parallel_rank():
    return 1

def mock_get_tp_group():
    return "mock_tp_group"

# 测试 MooncakeConnectorWorker
class TestMooncakeConnectorWorker(unittest.TestCase):
    def setUp(self):
        self.config = MockVllmConfig()
        # 设置环境变量
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0,1"
    
    @patch("mooncake_connector.get_tensor_model_parallel_rank", mock_get_tensor_model_parallel_rank)
    @patch("mooncake_connector.get_tp_group", mock_get_tp_group)
    @patch("mooncake_connector.TransferEngine")
    def test_initialization(self, mock_transfer_engine):
        # 测试初始化
        mock_engine = MagicMock()
        mock_engine.initialize.return_value = 0
        mock_transfer_engine.return_value = mock_engine
        
        worker = MooncakeConnectorWorker(self.config, "test_worker")
        
        # 验证属性设置
        self.assertEqual(worker.engine_id, "test_worker")
        self.assertEqual(worker.tp_rank, 1)  # 来自 mock_get_tensor_model_parallel_rank
        self.assertEqual(worker.tp_size, 2)
        self.assertEqual(worker.tp_group, "mock_tp_group")
        self.assertEqual(worker.dp_rank, 0)
        self.assertEqual(worker.dp_size, 1)
        self.assertEqual(worker.device_id, 1)  # 来自环境变量 ASCEND_RT_VISIBLE_DEVICES="0,1" 的第二个值
        
        # 验证初始化调用
        mock_engine.initialize.assert_called_once()
    
    @patch("mooncake_connector.get_tensor_model_parallel_rank", mock_get_tensor_model_parallel_rank)
    @patch("mooncake_connector.TransferEngine")
    def test_initialization_failure(self, mock_transfer_engine):
        # 测试初始化失败
        mock_engine = MagicMock()
        mock_engine.initialize.return_value = -1
        mock_transfer_engine.return_value = mock_engine
        
        with self.assertRaises(RuntimeError):
            MooncakeConnectorWorker(self.config, "test_worker")
    
    @patch("mooncake_connector.get_tensor_model_parallel_rank", mock_get_tensor_model_parallel_rank)
    @patch("mooncake_connector.KVCacheSendingThread")
    @patch("mooncake_connector.KVCacheRecvingThread")
    def test_register_kv_caches_non_mla(self, mock_recv_thread, mock_send_thread):
        # 测试非MLA情况
        worker = MooncakeConnectorWorker(self.config, "test_worker")
        worker.engine = MagicMock()
        
        kv_caches = {
            "layer1": (
                torch.zeros((10, 16, 8, 64)),  # [num_blocks, block_size, num_heads, head_dim]
                torch.zeros((10, 16, 8, 64))
        }
        
        worker.register_kv_caches(kv_caches)
        
        # 验证元数据设置
        self.assertEqual(worker.num_blocks, 10)
        self.assertEqual(len(worker.block_len), 1)
        self.assertFalse(worker.use_mla)
        
        # 验证内存注册调用次数（2个缓存层）
        self.assertEqual(worker.engine.register_memory.call_count, 2)
        
        # 验证发送线程启动（因为角色是producer）
        mock_send_thread.assert_called_once()
        mock_send_thread.return_value.start.assert_called_once()
        self.assertTrue(worker.kv_send_thread.ready_event.is_set())
    
    @patch("mooncake_connector.get_tensor_model_parallel_rank", mock_get_tensor_model_parallel_rank)
    @patch("mooncake_connector.KVCacheSendingThread")
    @patch("mooncake_connector.KVCacheRecvingThread")
    def test_register_kv_caches_mla(self, mock_recv_thread, mock_send_thread):
        # 测试MLA情况
        worker = MooncakeConnectorWorker(self.config, "test_worker")
        worker.engine = MagicMock()
        
        kv_caches = {
            "layer1": (
                torch.zeros((10, 16, 1, 64)),  # 不同形状的MLA缓存
                torch.zeros((10, 16, 1, 128))
        }
        
        worker.register_kv_caches(kv_caches)
        
        # 验证元数据设置
        self.assertEqual(worker.num_blocks, 10)
        self.assertEqual(len(worker.block_len), 2)
        self.assertTrue(worker.use_mla)
        
        # 验证内存注册调用次数（2个缓存层 * 2个块类型）
        self.assertEqual(worker.engine.register_memory.call_count, 4)
    
    @patch("mooncake_connector.get_tensor_model_parallel_rank", mock_get_tensor_model_parallel_rank)
    @patch("mooncake_connector.KVCacheSendingThread")
    @patch("mooncake_connector.KVCacheRecvingThread")
    def test_register_kv_caches_consumer_role(self, mock_recv_thread, mock_send_thread):
        # 测试消费者角色
        self.config.kv_transfer_config.kv_role = "kv_consumer"
        worker = MooncakeConnectorWorker(self.config, "test_worker")
        worker.engine = MagicMock()
        
        kv_caches = {
            "layer1": (
                torch.zeros((10, 16, 8, 64)),
                torch.zeros((10, 16, 8, 64))
        }
        
        worker.register_kv_caches(kv_caches)
        
        # 验证接收线程启动
        mock_recv_thread.assert_called_once()
        mock_recv_thread.return_value.start.assert_called_once()
        self.assertTrue(worker.kv_recv_thread.ready_event.is_set())
    
    @patch("mooncake_connector.get_tensor_model_parallel_rank", mock_get_tensor_model_parallel_rank)
    def test_register_kv_caches_memory_registration_failure(self):
        # 测试内存注册失败
        worker = MooncakeConnectorWorker(self.config, "test_worker")
        worker.engine = MagicMock()
        worker.engine.register_memory.return_value = -1
        
        kv_caches = {
            "layer1": (
                torch.zeros((10, 16, 8, 64)),
                torch.zeros((10, 16, 8, 64))
        }
        
        with self.assertRaises(RuntimeError):
            worker.register_kv_caches(kv_caches)
    
    @patch("mooncake_connector.get_tensor_model_parallel_rank", mock_get_tensor_model_parallel_rank)
    @patch("mooncake_connector.logger")
    def test_register_kv_caches_logging(self, mock_logger):
        # 测试日志记录
        worker = MooncakeConnectorWorker(self.config, "test_worker")
        worker.engine = MagicMock()
        
        kv_caches = {
            "layer1": (
                torch.zeros((10, 16, 8, 64)),
                torch.zeros((10, 16, 8, 64))
        }
        
        worker.register_kv_caches(kv_caches)
        
        # 验证日志调用
        self.assertTrue(mock_logger.info.called)
        self.assertIn("Registering KV_Caches", mock_logger.info.call_args[0][0])
    
    @patch("mooncake_connector.get_tensor_model_parallel_rank", mock_get_tensor_model_parallel_rank)
    def test_get_finished_producer(self):
        # 模拟发送线程
        worker = MooncakeConnectorWorker(self.config, "test_worker")
        worker.engine = MagicMock()
        worker.kv_send_thread = MagicMock()
        worker.kv_send_thread.get_and_clear_finished_requests.return_value = {"req1", "req2"}
        
        done_sending, done_recving = worker.get_finished()
        self.assertEqual(done_sending, {"req1", "req2"})
        self.assertEqual(done_recving, set())
        
        # 验证TP rank 0的日志记录（当前rank是1，所以不应记录）
        with patch("mooncake_connector.logger") as mock_logger:
            worker.get_finished()
            self.assertFalse(mock_logger.debug.called)
    
    @patch("mooncake_connector.get_tensor_model_parallel_rank", mock_get_tensor_model_parallel_rank)
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
    
    @patch("mooncake_connector.get_tensor_model_parallel_rank", mock_get_tensor_model_parallel_rank)
    @patch("mooncake_connector.logger")
    def test_start_load_kv_logging(self, mock_logger):
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
        
        # 验证日志记录
        self.assertTrue(mock_logger.debug.called)
        self.assertIn("start_load_kv for request", mock_logger.debug.call_args[0][0])
    
    @patch("mooncake_connector.get_tensor_model_parallel_rank", mock_get_tensor_model_parallel_rank)
    def test_start_load_kv_no_requests(self):
        # 设置worker为消费者角色
        self.config.kv_transfer_config.kv_role = "kv_consumer"
        worker = MooncakeConnectorWorker(self.config, "test_worker")
        worker.engine = MagicMock()
        worker.kv_recv_thread = MagicMock()
        
        # 空元数据
        meta = MooncakeConnectorMetadata()
        
        worker.start_load_kv(meta)
        
        # 验证没有添加请求
        worker.kv_recv_thread.add_request.assert_not_called()
    
    @patch("mooncake_connector.get_tensor_model_parallel_rank", mock_get_tensor_model_parallel_rank)
    def test_remote_tp_ranks_same_size(self):
        # 当预填充和decode的tp大小相同时
        worker = MooncakeConnectorWorker(self.config, "test_worker")
        worker.engine = MagicMock()
        worker._prefill_tp_size = 2
        worker._decode_tp_size = 2
        ranks = worker._get_remote_tp_ranks_for_req("req1")
        self.assertEqual(ranks, [0, 1])
    
    @patch("mooncake_connector.get_tensor_model_parallel_rank", mock_get_tensor_model_parallel_rank)
    def test_remote_tp_ranks_different_size(self):
        # 当预填充和decode的tp大小不同时
        worker = MooncakeConnectorWorker(self.config, "test_worker")
        worker.engine = MagicMock()
        worker._prefill_tp_size = 4
        worker._decode_tp_size = 2
        
        # 固定随机种子以确保可重复性
        with patch("mooncake_connector.string_to_int64_hash", return_value=12345):
            with patch("random.Random.sample") as mock_sample:
                mock_sample.return_value = [1, 3]
                ranks = worker._get_remote_tp_ranks_for_req("req1")
                self.assertEqual(ranks, [1, 3])
                
                # 验证使用正确的种子
                mock_sample.assert_called_once_with(range(4), 2)
    
    @patch("mooncake_connector.get_tensor_model_parallel_rank", mock_get_tensor_model_parallel_rank)
    def test_remote_tp_ranks_different_size_single_request(self):
        # 测试多个请求是否使用相同的随机种子
        worker = MooncakeConnectorWorker(self.config, "test_worker")
        worker.engine = MagicMock()
        worker._prefill_tp_size = 4
        worker._decode_tp_size = 2
        
        # 使用相同的请求ID应产生相同的结果
        ranks1 = worker._get_remote_tp_ranks_for_req("same_id")
        ranks2 = worker._get_remote_tp_ranks_for_req("same_id")
        self.assertEqual(ranks1, ranks2)
        
        # 不同的请求ID应产生不同的结果
        ranks3 = worker._get_remote_tp_ranks_for_req("different_id")
        self.assertNotEqual(ranks1, ranks3)
    
    @patch("mooncake_connector.get_tensor_model_parallel_rank", mock_get_tensor_model_parallel_rank)
    def test_get_remote_tp_rank(self):
        worker = MooncakeConnectorWorker(self.config, "test_worker")
        worker.engine = MagicMock()
        worker._prefill_tp_size = 4
        worker._decode_tp_size = 2
        worker.tp_rank = 1
        
        # 模拟返回的rank列表
        with patch.object(worker, '_get_remote_tp_ranks_for_req', return_value=[2, 3]):
            rank = worker._get_remote_tp_rank("req1")
            self.assertEqual(rank, 3)  # tp_rank=1 应返回列表中的第二个元素
    
    @patch("mooncake_connector.get_tensor_model_parallel_rank", mock_get_tensor_model_parallel_rank)
    @patch("mooncake_connector.logger")
    def test_register_memory_logging(self, mock_logger):
        worker = MooncakeConnectorWorker(self.config, "test_worker")
        worker.engine = MagicMock()
        
        kv_caches = {
            "layer1": (
                torch.zeros((10, 16, 8, 64)),
                torch.zeros((10, 16, 8, 64))
        }
        
        worker.register_kv_caches(kv_caches)
        
        # 验证内存注册日志
        self.assertTrue(mock_logger.info.called)
        self.assertIn("Registering KV cache", mock_logger.info.call_args[0][0])
    
    @patch("mooncake_connector.get_tensor_model_parallel_rank", mock_get_tensor_model_parallel_rank)
    def test_register_kv_caches_single_cache(self):
        # 测试只有一个缓存的情况
        worker = MooncakeConnectorWorker(self.config, "test_worker")
        worker.engine = MagicMock()
        
        kv_caches = {
            "layer1": torch.zeros((10, 16, 8, 64))
        }
        
        worker.register_kv_caches(kv_caches)
        
        # 验证元数据设置
        self.assertEqual(worker.num_blocks, 10)
        self.assertEqual(len(worker.block_len), 1)
        self.assertFalse(worker.use_mla)
        
        # 验证内存注册调用次数
        self.assertEqual(worker.engine.register_memory.call_count, 1)
    
    @patch("mooncake_connector.get_tensor_model_parallel_rank", mock_get_tensor_model_parallel_rank)
    @patch("mooncake_connector.threading.Event")
    def test_register_kv_caches_thread_ready(self, mock_event):
        # 测试线程就绪事件
        worker = MooncakeConnectorWorker(self.config, "test_worker")
        worker.engine = MagicMock()
        
        mock_event.return_value = MagicMock()
        kv_caches = {
            "layer1": (
                torch.zeros((10, 16, 8, 64)),
                torch.zeros((10, 16, 8, 64))
        }
        
        worker.register_kv_caches(kv_caches)
        
        # 验证就绪事件被设置
        mock_event.return_value.set.assert_called_once()
        mock_event.return_value.wait.assert_called_once()
    
    @patch("mooncake_connector.get_tensor_model_parallel_rank", mock_get_tensor_model_parallel_rank)
    def test_prefill_tp_size_validation(self):
        # 测试预填充TP大小小于解码TP大小的错误
        with patch("mooncake_connector.logger") as mock_logger:
            with self.assertRaises(ValueError):
                # 模拟配置返回无效值
                self.config.kv_transfer_config.get_from_extra_config = lambda k, d: {
                    "prefill": {"tp_size": 2, "dp_size": 1},
                    "decode": {"tp_size": 4, "dp_size": 1}
                }[k]
                
                worker = MooncakeConnectorWorker(self.config, "test_worker")
    
    @patch("mooncake_connector.get_tensor_model_parallel_rank", mock_get_tensor_model_parallel_rank)
    def test_get_finished_no_threads(self):
        # 测试没有线程的情况
        worker = MooncakeConnectorWorker(self.config, "test_worker")
        worker.engine = MagicMock()
        worker.kv_role = "unknown"  # 无效角色
        
        done_sending, done_recving = worker.get_finished()
        self.assertEqual(done_sending, set())
        self.assertEqual(done_recving, set())
    
    @patch("mooncake_connector.get_tensor_model_parallel_rank", mock_get_tensor_model_parallel_rank)
    def test_device_id_from_env(self):
        # 测试从环境变量获取设备ID
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "3,4,5"
        
        worker = MooncakeConnectorWorker(self.config, "test_worker")
        worker.engine = MagicMock()
        
        # tp_rank=1 应该返回列表中的第二个值
        self.assertEqual(worker.device_id, 4)
    
    @patch("mooncake_connector.get_tensor_model_parallel_rank", mock_get_tensor_model_parallel_rank)
    def test_device_id_no_env(self):
        # 测试没有环境变量时获取设备ID
        if "ASCEND_RT_VISIBLE_DEVICES" in os.environ:
            del os.environ["ASCEND_RT_VISIBLE_DEVICES"]
        
        worker = MooncakeConnectorWorker(self.config, "test_worker")
        worker.engine = MagicMock()
        
        # 应该根据 dp_rank 和 tp_size 计算
        self.assertEqual(worker.device_id, worker.dp_rank * worker.tp_size + worker.tp_rank)

# 运行测试
if __name__ == "__main__":
    unittest.main()
