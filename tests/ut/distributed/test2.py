class TestMooncakeConnectorWorker(unittest.TestCase):
    def setUp(self):
        self.config = MockVllmConfig()
        # 设置环境变量
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0,1"
        self.worker = MooncakeConnectorWorker(self.config, "test_worker")
        self.worker.engine = MagicMock()
    
    @patch("mooncake_connector.KVCacheSendingThread")
    @patch("mooncake_connector.KVCacheRecvingThread")
    def test_register_kv_caches_non_mla(self, mock_recv_thread, mock_send_thread):
        # 测试非MLA情况
        kv_caches = {
            "layer1": (
                torch.zeros((10, 16, 8, 64)),  # [num_blocks, block_size, num_heads, head_dim]
                torch.zeros((10, 16, 8, 64))
        }
        
        self.worker.register_kv_caches(kv_caches)
        
        # 验证元数据设置
        self.assertEqual(self.worker.num_blocks, 10)
        self.assertEqual(len(self.worker.block_len), 1)
        self.assertFalse(self.worker.use_mla)
        
        # 验证内存注册调用次数（2个缓存层）
        self.assertEqual(self.worker.engine.register_memory.call_count, 2)
        
        # 验证发送线程启动
        mock_send_thread.assert_called_once()
        mock_send_thread.return_value.start.assert_called_once()
        self.assertTrue(self.worker.kv_send_thread.ready_event.is_set())
    
    @patch("mooncake_connector.KVCacheSendingThread")
    @patch("mooncake_connector.KVCacheRecvingThread")
    def test_register_kv_caches_mla(self, mock_recv_thread, mock_send_thread):
        # 测试MLA情况
        kv_caches = {
            "layer1": (
                torch.zeros((10, 16, 1, 64)),  # 不同形状的MLA缓存
                torch.zeros((10, 16, 1, 128))
        }
        
        self.worker.register_kv_caches(kv_caches)
        
        # 验证元数据设置
        self.assertEqual(self.worker.num_blocks, 10)
        self.assertEqual(len(self.worker.block_len), 2)
        self.assertTrue(self.worker.use_mla)
        
        # 验证内存注册调用次数（2个缓存层 * 2个块类型）
        self.assertEqual(self.worker.engine.register_memory.call_count, 4)
    
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
    
    def test_register_kv_caches_memory_registration_failure(self):
        # 测试内存注册失败
        self.worker.engine.register_memory.return_value = -1
        
        kv_caches = {
            "layer1": (
                torch.zeros((10, 16, 8, 64)),
                torch.zeros((10, 16, 8, 64))
        }
        
        with self.assertRaises(RuntimeError):
            self.worker.register_kv_caches(kv_caches)
    
    @patch("mooncake_connector.logger")
    def test_register_kv_caches_logging(self, mock_logger):
        # 测试日志记录
        kv_caches = {
            "layer1": (
                torch.zeros((10, 16, 8, 64)),
                torch.zeros((10, 16, 8, 64))
        }
        
        self.worker.register_kv_caches(kv_caches)
        
        # 验证日志调用
        self.assertTrue(mock_logger.info.called)
        self.assertIn("Registering KV_Caches", mock_logger.info.call_args[0][0])
    
    def test_get_finished_producer(self):
        # 模拟发送线程
        self.worker.kv_send_thread = MagicMock()
        self.worker.kv_send_thread.get_and_clear_finished_requests.return_value = {"req1", "req2"}
        
        done_sending, done_recving = self.worker.get_finished()
        self.assertEqual(done_sending, {"req1", "req2"})
        self.assertEqual(done_recving, set())
        
        # 验证TP rank 0的日志记录
        with patch("mooncake_connector.logger") as mock_logger:
            self.worker.get_finished()
            self.assertTrue(mock_logger.debug.called)
    
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
        
        # 固定随机种子以确保可重复性
        with patch("mooncake_connector.string_to_int64_hash", return_value=12345):
            with patch("random.Random.sample") as mock_sample:
                mock_sample.return_value = [1, 3]
                ranks = self.worker._get_remote_tp_ranks_for_req("req1")
                self.assertEqual(ranks, [1, 3])
                
                # 验证使用正确的种子
                mock_sample.assert_called_once_with(range(4), 2)
    
    def test_remote_tp_ranks_different_size_single_request(self):
        # 测试多个请求是否使用相同的随机种子
        self.worker._prefill_tp_size = 4
        self.worker._decode_tp_size = 2
        
        # 使用相同的请求ID应产生相同的结果
        ranks1 = self.worker._get_remote_tp_ranks_for_req("same_id")
        ranks2 = self.worker._get_remote_tp_ranks_for_req("same_id")
        self.assertEqual(ranks1, ranks2)
        
        # 不同的请求ID应产生不同的结果
        ranks3 = self.worker._get_remote_tp_ranks_for_req("different_id")
        self.assertNotEqual(ranks1, ranks3)
    
    def test_get_remote_tp_rank(self):
        self.worker._prefill_tp_size = 4
        self.worker._decode_tp_size = 2
        self.worker.tp_rank = 1
        
        # 模拟返回的rank列表
        with patch.object(self.worker, '_get_remote_tp_ranks_for_req', return_value=[2, 3]):
            rank = self.worker._get_remote_tp_rank("req1")
            self.assertEqual(rank, 3)  # tp_rank=1 应返回列表中的第二个元素
    
    @patch("mooncake_connector.TransferEngine")
    def test_initialize_success(self, mock_transfer_engine):
        worker = MooncakeConnectorWorker(self.config, "test_worker")
        worker.engine.initialize.return_value = 0
        # 初始化应在构造函数中调用，这里只验证调用
        self.assertTrue(worker.engine.initialize.called)
    
    @patch("mooncake_connector.TransferEngine")
    def test_initialize_failure(self, mock_transfer_engine):
        mock_engine = MagicMock()
        mock_engine.initialize.return_value = -1
        mock_transfer_engine.return_value = mock_engine
        
        with self.assertRaises(RuntimeError):
            MooncakeConnectorWorker(self.config, "test_worker")
    
    @patch("mooncake_connector.logger")
    def test_register_memory_logging(self, mock_logger):
        kv_caches = {
            "layer1": (
                torch.zeros((10, 16, 8, 64)),
                torch.zeros((10, 16, 8, 64))
        }
        
        self.worker.register_kv_caches(kv_caches)
        
        # 验证内存注册日志
        self.assertTrue(mock_logger.info.called)
        self.assertIn("Registering KV cache", mock_logger.info.call_args[0][0])
    
    def test_register_kv_caches_single_cache(self):
        # 测试只有一个缓存的情况
        kv_caches = {
            "layer1": torch.zeros((10, 16, 8, 64))
        }
        
        self.worker.register_kv_caches(kv_caches)
        
        # 验证元数据设置
        self.assertEqual(self.worker.num_blocks, 10)
        self.assertEqual(len(self.worker.block_len), 1)
        self.assertFalse(self.worker.use_mla)
        
        # 验证内存注册调用次数
        self.assertEqual(self.worker.engine.register_memory.call_count, 1)
    
    @patch("mooncake_connector.threading.Event")
    def test_register_kv_caches_thread_ready(self, mock_event):
        # 测试线程就绪事件
        mock_event.return_value = MagicMock()
        kv_caches = {
            "layer1": (
                torch.zeros((10, 16, 8, 64)),
                torch.zeros((10, 16, 8, 64))
        }
        
        self.worker.register_kv_caches(kv_caches)
        
        # 验证就绪事件被设置
        mock_event.return_value.set.assert_called_once()
        mock_event.return_value.wait.assert_called_once()
