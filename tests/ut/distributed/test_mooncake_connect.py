
import os
import queue
import socket
import threading
import time
import unittest
from collections import defaultdict, deque
from unittest.mock import ANY, MagicMock, Mock, PropertyMock, call, patch

import msgspec
import zmq
from zmq import Context

GET_META_MSG = b"get_meta_msg"
DONE_RECVING_MSG = b"done_recving_msg"

from vllm.utils import make_zmq_path, make_zmq_socket
from vllm_ascend.distributed.mooncake_connector import (
    KVCacheRecvingThread, KVCacheSendingThread, KVCacheTaskTracker,
    KVConnectorRole, MooncakeAgentMetadata, MooncakeConnector,
    MooncakeConnectorMetadata, MooncakeConnectorScheduler,
    MooncakeConnectorWorker, ReqMeta, ensure_zmq_recv, ensure_zmq_send,
    group_concurrent_contiguous, string_to_int64_hash, zmq_ctx)


class TestKVCacheTaskTracker(unittest.TestCase):

    def test_init_basic_properties(self):
        tracker = KVCacheTaskTracker(tp_rank=1, local_engine_id="engine1", target_count=10)
        
        self.assertEqual(tracker.tp_rank, 1)
        self.assertEqual(tracker.local_engine_id, "engine1")
        self.assertEqual(tracker.target_count, 10)
        self.assertIsInstance(tracker.done_task_lock, type(threading.Lock()))
        self.assertIsInstance(tracker.done_task_counts, defaultdict)
        self.assertIsInstance(tracker.finished_requests, set)

    def test_socket_path_generation(self):
        tracker = KVCacheTaskTracker(tp_rank=1, local_engine_id="engine42", target_count=1)
        self.assertEqual(tracker.socket_path, 
                       "ipc:///tmp/vllm_mooncake_connector_engine42.ipc")

    @patch("vllm_ascend.distributed.mooncake_connector.threading.Thread")
    def test_tp_rank_zero_initialization(self, mock_thread):
        tracker = KVCacheTaskTracker(tp_rank=0, local_engine_id="test", target_count=1)
        
        mock_thread.assert_called_once_with(
            target=tracker._listen_for_completion_signals,
            daemon=True,
            name="KVCacheTaskTrackerListenerThread"
        )
        mock_thread.return_value.start.assert_called_once()
        self.assertIsNone(tracker.socket)
        self.assertTrue(tracker.listener.daemon)

    @patch("vllm_ascend.distributed.mooncake_connector.make_zmq_socket")
    @patch("vllm_ascend.distributed.mooncake_connector.logger")
    def test_tp_rank_non_zero_initialization(self, mock_logger, mock_make_zmq_socket):
        mock_socket = MagicMock()
        mock_make_zmq_socket.return_value = mock_socket
        
        tracker = KVCacheTaskTracker(tp_rank=1, local_engine_id="test", target_count=1)
        
        mock_make_zmq_socket.assert_called_once_with(
            ctx=unittest.mock.ANY,
            path="ipc:///tmp/vllm_mooncake_connector_test.ipc",
            socket_type=zmq.PUSH,
            bind=False
        )
        mock_logger.info.assert_called_once_with(
            "Connecting to transfer socket at %s", 
            "ipc:///tmp/vllm_mooncake_connector_test.ipc"
        )
        self.assertIsNone(tracker.listener)
        self.assertEqual(tracker.socket, mock_socket)

class TestKVCacheTaskTrackerListenMethod(unittest.TestCase):
    """专门测试_listen_for_completion_signals方法的测试类"""
    
    def setUp(self):
        """测试前初始化"""
        self.tp_rank = 0
        self.local_engine_id = "test_engine_ut"
        self.target_count = 3
        # 创建测试实例
        self.tracker = KVCacheTaskTracker(
            self.tp_rank,
            self.local_engine_id,
            self.target_count
        )
        # 保存原始方法引用，用于后续测试
        self.original_listen = self.tracker._listen_for_completion_signals

    def tearDown(self):
        """测试后清理资源"""
        # 恢复可能被修改的方法
        self.tracker._listen_for_completion_signals = self.original_listen
        
        # 清理ZMQ资源
        Context.instance().term()
        # 等待资源释放
        time.sleep(0.1)

    def test_normal_message_processing(self):
        """测试正常接收并处理消息的流程"""
        # 启动监听线程
        listener_thread = threading.Thread(
            target=self.tracker._listen_for_completion_signals,
            daemon=True
        )
        listener_thread.start()
        # 等待监听线程初始化完成
        time.sleep(0.2)

        # 准备测试消息
        test_messages = [
            ("request_001", 1),
            ("request_001", 2),
            ("request_002", 0),
            ("request_003", 1)
        ]

        # 创建发送套接字并发送消息
        ctx = Context()
        sender_socket = ctx.socket(zmq.PUSH)
        sender_socket.connect(self.tracker.socket_path)

        for msg in test_messages:
            sender_socket.send_pyobj(msg)
            time.sleep(0.05)  # 短暂延迟确保消息顺序

        # 等待消息处理完成
        sender_socket.close()
        time.sleep(0.2)

        # 验证结果
        with self.tracker.done_task_lock:
            # 验证request_001接收情况
            self.assertEqual(len(self.tracker.done_task_counts["request_001"]), 2)
            self.assertIn(1, self.tracker.done_task_counts["request_001"])
            self.assertIn(2, self.tracker.done_task_counts["request_001"])
            
            # 验证request_002接收情况
            self.assertEqual(len(self.tracker.done_task_counts["request_002"]), 1)
            self.assertIn(0, self.tracker.done_task_counts["request_002"])
            
            # 验证request_003接收情况
            self.assertEqual(len(self.tracker.done_task_counts["request_003"]), 1)
            self.assertIn(1, self.tracker.done_task_counts["request_003"])


    @patch("vllm_ascend.distributed.mooncake_connector.make_zmq_socket", autospec=True)
    def test_listen_with_timeout(self, mock_make_socket):
        mock_socket = MagicMock()
        def mock_recv():
            start = time.time()
            while time.time() - start < 0.5:
                time.sleep(0.01)
            return ("req1", 0)
        mock_socket.recv_pyobj = mock_recv
        mock_make_socket.return_value = mock_socket
        
        test_thread = threading.Thread(
            target=self.tracker._listen_for_completion_signals,
            daemon=True
        )
        test_thread.start()
        test_thread.join(timeout=1.0)
        mock_make_socket.assert_called_once()

class TestKVCacheTaskTracker(unittest.TestCase):
    def setUp(self):
        # 通用测试参数
        self.local_engine_id = "test_engine"
        self.target_count = 3

    def test_update_done_task_count_tp_rank_0(self):
        """测试tp_rank=0时直接更新任务计数"""
        # 1. 创建tp_rank=0的实例
        tracker = KVCacheTaskTracker(
            tp_rank=0,
            local_engine_id=self.local_engine_id,
            target_count=self.target_count
        )
        
        # 2. 执行测试方法
        test_request_id = "test_req_001"
        test_tp_rank = 1
        tracker.update_done_task_count(test_request_id, test_tp_rank)
        
        # 3. 验证结果
        with tracker.done_task_lock:
            self.assertEqual(len(tracker.done_task_counts[test_request_id]), 1)
            self.assertIn(test_tp_rank, tracker.done_task_counts[test_request_id])

    @patch("vllm_ascend.distributed.mooncake_connector.make_zmq_socket", autospec=True)
    def test_update_done_task_count_non_zero_tp(self, mock_make_socket):
        """测试tp_rank≠0时发送消息到tp_rank=0"""
        # 1. 准备mock socket
        mock_socket = MagicMock()
        mock_make_socket.return_value = mock_socket
        
        # 2. 创建tp_rank=1的实例
        tracker = KVCacheTaskTracker(
            tp_rank=1,
            local_engine_id=self.local_engine_id,
            target_count=self.target_count
        )
        
        # 3. 执行测试方法
        test_request_id = "test_req_002"
        test_tp_rank = 1
        tracker.update_done_task_count(test_request_id, test_tp_rank)
        
        # 4. 验证结果
        # 验证socket发送了正确的消息
        mock_socket.send_pyobj.assert_called_once_with(
            (test_request_id, test_tp_rank)
        )
        # 验证本地计数未更新（应由tp_rank=0处理）
        with tracker.done_task_lock:
            self.assertNotIn(test_request_id, tracker.done_task_counts)

    @patch("vllm_ascend.distributed.mooncake_connector.logger", autospec=True)
    @patch("vllm_ascend.distributed.mooncake_connector.make_zmq_socket", autospec=True)
    def test_update_done_task_count_logging(self, mock_make_socket, mock_logger):
        """测试更新操作的日志输出"""
        # 1. 准备mock socket
        mock_socket = MagicMock()
        mock_make_socket.return_value = mock_socket
        
        # 2. 创建tp_rank=2的实例
        tracker = KVCacheTaskTracker(
            tp_rank=2,
            local_engine_id=self.local_engine_id,
            target_count=self.target_count
        )
        
        # 3. 执行测试方法
        test_request_id = "test_req_003"
        tracker.update_done_task_count(test_request_id, 2)
        
        # 4. 验证日志
        mock_logger.debug.assert_called_once_with(
            "Sent done signal for request %s to tp 0",
            test_request_id
        )

    @patch("vllm_ascend.distributed.mooncake_connector.make_zmq_socket", autospec=True)
    def test_update_multiple_calls(self, mock_make_socket):
        """测试多次调用update_done_task_count的情况"""
        # 1. 准备mock socket
        mock_socket = MagicMock()
        mock_make_socket.return_value = mock_socket
        
        # 2. 创建tp_rank=1的实例
        tracker = KVCacheTaskTracker(
            tp_rank=1,
            local_engine_id=self.local_engine_id,
            target_count=self.target_count
        )
        
        # 3. 多次执行测试方法
        test_data = [
            ("req1", 1),
            ("req1", 1),
            ("req2", 1)
        ]
        for req_id, rank in test_data:
            tracker.update_done_task_count(req_id, rank)
        
        # 4. 验证调用次数和参数
        self.assertEqual(mock_socket.send_pyobj.call_count, 3)
        # 验证最后一次调用参数
        mock_socket.send_pyobj.assert_called_with(("req2", 1))


class TestGetAndClearFinishedRequests(unittest.TestCase):

    def setUp(self):
        """创建测试用的tracker实例"""
        self.tracker = KVCacheTaskTracker(tp_rank=0, local_engine_id="test", target_count=3)
        # 绕过__init__直接设置内部状态
        self.tracker.finished_requests = set()
        self.tracker.done_task_counts = defaultdict(set)
        self.tracker.done_task_lock = threading.Lock()

    def test_empty_requests(self):
        """测试没有完成请求时返回空集合"""
        result = self.tracker.get_and_clear_finished_requests()
        self.assertEqual(result, set())
        self.assertEqual(len(self.tracker.finished_requests), 0)  # 确认清空

    def test_single_request(self):
        """测试单个完成请求的情况"""
        # 准备测试数据
        self.tracker.finished_requests = {"req_123"}
        
        # 执行方法
        result = self.tracker.get_and_clear_finished_requests()
        
        # 验证
        self.assertEqual(result, {"req_123"})
        self.assertEqual(len(self.tracker.finished_requests), 0)  # 确认已清空

    def test_multiple_requests(self):
        """测试多个完成请求的情况"""
        # 准备测试数据
        self.tracker.finished_requests = {"req_1", "req_2", "req_3"}
        
        # 执行方法
        result = self.tracker.get_and_clear_finished_requests()
        
        # 验证
        self.assertSetEqual(result, {"req_1", "req_2", "req_3"})
        self.assertEqual(len(self.tracker.finished_requests), 0)

    @patch("vllm_ascend.distributed.mooncake_connector.logger")
    def test_concurrent_access(self, mock_logger):
        """测试线程安全-并发访问"""
        from concurrent.futures import ThreadPoolExecutor

        # 准备测试数据
        self.tracker.finished_requests = {"req_1", "req_2"}
        
        # 模拟并发访问
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(self.tracker.get_and_clear_finished_requests) 
                      for _ in range(3)]
            results = [f.result() for f in futures]
        
        # 验证只有一个线程获取到数据
        self.assertEqual(sum(1 for r in results if r), 1)
        self.assertEqual(len(self.tracker.finished_requests), 0)

    def test_after_increment(self):
        """测试_increment_task_count后的场景"""
        # 模拟3个rank完成
        self.tracker._increment_task_count("req_123", 0)
        self.tracker._increment_task_count("req_123", 1)
        self.tracker._increment_task_count("req_123", 2)
        
        # 验证请求已进入finished_requests
        result = self.tracker.get_and_clear_finished_requests()
        self.assertEqual(result, {"req_123"})
        
        # 再次获取应为空
        self.assertEqual(self.tracker.get_and_clear_finished_requests(), set())


class TestKVCacheSendingThreadInit(unittest.TestCase):

    def setUp(self):
        """准备测试用的通用参数"""
        self.common_args = {
            'tp_rank': 1,
            'decode_tp_size': 4,
            'local_engine_id': 'engine_1',
            'side_channel_host': 'localhost',
            'side_channel_port': 5555,
            'metadata': MagicMock(),
            'ready_event': threading.Event()
        }
        # 存储创建的线程实例用于清理
        self.threads = []

    def tearDown(self):
        """清理可能创建的线程和资源"""
        for thread in self.threads:
            if hasattr(thread, 'task_tracker') and hasattr(thread.task_tracker, 'socket'):
                thread.task_tracker.socket.close()
            if hasattr(thread, 'is_alive') and thread.is_alive():
                thread.join(timeout=0.1)

    @patch('vllm_ascend.distributed.mooncake_connector.KVCacheTaskTracker')
    def test_initialization_basic(self, mock_tracker):
        """测试基础初始化参数"""
        thread = KVCacheSendingThread(**self.common_args)
        self.threads.append(thread)
        
        # 验证线程属性
        self.assertEqual(thread.tp_rank, 1)
        self.assertEqual(thread.decode_tp_size, 4)
        self.assertEqual(thread.local_engine_id, 'engine_1')
        
        # 修改断言方式，兼容位置参数调用
        mock_tracker.assert_called_once()
        
        # 获取调用参数（可能是位置参数或关键字参数）
        args = mock_tracker.call_args[0]  # 位置参数元组
        kwargs = mock_tracker.call_args[1]  # 关键字参数字典
        
        # 检查参数值而不关心传递方式
        if args:  # 如果是位置参数
            self.assertEqual(args[0], 1)  # tp_rank
            self.assertEqual(args[1], 'engine_1')  # local_engine_id
            self.assertEqual(args[2], 4)  # target_count
        else:  # 如果是关键字参数
            self.assertEqual(kwargs['tp_rank'], 1)
            self.assertEqual(kwargs['local_engine_id'], 'engine_1')
            self.assertEqual(kwargs['target_count'], 4)

    @patch('vllm_ascend.distributed.mooncake_connector.KVCacheTaskTracker')
    def test_task_tracker_initialization(self, mock_tracker):
        """测试task_tracker初始化参数"""
        args = self.common_args.copy()
        args.update({
            'tp_rank': 2,
            'decode_tp_size': 8,
            'local_engine_id': 'engine_2'
        })
        
        thread = KVCacheSendingThread(**args)
        self.threads.append(thread)
        
        # 修改断言方式
        mock_tracker.assert_called_once()
        
        # 获取调用参数
        call_args = mock_tracker.call_args[0]
        call_kwargs = mock_tracker.call_args[1]
        
        # 检查参数值
        if call_args:  # 位置参数调用
            self.assertEqual(call_args[0], 2)  # tp_rank
            self.assertEqual(call_args[1], 'engine_2')  # local_engine_id
            self.assertEqual(call_args[2], 8)  # target_count
        else:  # 关键字参数调用
            self.assertEqual(call_kwargs['tp_rank'], 2)
            self.assertEqual(call_kwargs['local_engine_id'], 'engine_2')
            self.assertEqual(call_kwargs['target_count'], 8)

    # 其他测试方法保持不变...
    def test_thread_daemon_property(self):
        """测试守护线程属性"""
        thread = KVCacheSendingThread(**self.common_args)
        self.threads.append(thread)
        self.assertTrue(thread.daemon)

    def test_thread_name_format(self):
        """测试线程名称格式"""
        thread = KVCacheSendingThread(**self.common_args)
        self.threads.append(thread)
        self.assertEqual(thread.name, "KVCacheSendingThread")

    def test_ready_event_reference(self):
        """测试ready_event引用是否正确"""
        custom_event = threading.Event()
        args = self.common_args.copy()
        args['ready_event'] = custom_event
        
        thread = KVCacheSendingThread(**args)
        self.threads.append(thread)
        self.assertIs(thread.ready_event, custom_event)


class TestGetAndClearFinishedRequests(unittest.TestCase):
    def setUp(self):
        self.common_args = {
            'tp_rank': 1,
            'decode_tp_size': 4,
            'local_engine_id': 'engine_1',
            'side_channel_host': 'localhost',
            'side_channel_port': 5555,
            'metadata': {"test": "metadata"},  # 使用真实可编码的数据
            'ready_event': threading.Event()
        }
        self.thread = KVCacheSendingThread(**self.common_args)
    
    @patch.object(KVCacheTaskTracker, 'get_and_clear_finished_requests')
    def test_get_and_clear_finished_requests(self, mock_get_clear):
        expected_requests = {'req1', 'req2'}
        mock_get_clear.return_value = expected_requests
        
        result = self.thread.get_and_clear_finished_requests()
        
        mock_get_clear.assert_called_once()
        self.assertEqual(result, expected_requests)


class TestKVCacheSendingThread(unittest.TestCase):

    def test_run_handles_get_meta_and_done_recv_msgs(self):
        ready_event = threading.Event()
        metadata = MooncakeAgentMetadata(
            engine_id="engine1",
            kv_caches_base_addr=[12345678],
            num_blocks=2,
        )
        host = "127.0.0.1"

        # 找空闲端口
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            free_port = s.getsockname()[1]

        # 启动服务线程
        thread = KVCacheSendingThread(
            tp_rank=0,
            decode_tp_size=1,
            local_engine_id="engine1",
            side_channel_host=host,
            side_channel_port=free_port,
            metadata=metadata,
            ready_event=ready_event,
        )
        thread.start()

        self.assertTrue(ready_event.wait(timeout=3), "服务线程启动超时")

        # 创建 zmq 客户端 socket
        context = zmq.Context()
        sock = context.socket(zmq.DEALER)
        sock.connect(f"tcp://{host}:{free_port}")

        encoder = msgspec.msgpack.Encoder()
        decoder = msgspec.msgpack.Decoder(type=MooncakeAgentMetadata)

        # 1) 发送 GET_META_MSG，测试返回的 metadata
        sock.send_multipart([b"", encoder.encode((GET_META_MSG,))])
        frames = sock.recv_multipart()
        # DEALER socket第一个帧是空帧
        self.assertEqual(frames[0], b"")
        meta = decoder.decode(frames[1])
        self.assertEqual(meta.engine_id, "engine1")
        self.assertEqual(meta.kv_caches_base_addr, [12345678])
        self.assertEqual(meta.num_blocks, 2)

        # 2) 发送 DONE_RECVING_MSG，测试返回 ACK，并确认任务完成状态
        req_id = "request_42"
        sock.send_multipart([b"", encoder.encode((DONE_RECVING_MSG, req_id, 0))])
        frames = sock.recv_multipart()
        self.assertEqual(frames[0], b"")
        self.assertEqual(frames[1], b"ACK")

        # 检查 KVCacheSendingThread.task_tracker 中已更新
        self.assertIn(req_id, thread.task_tracker.finished_requests)

        # 清理
        sock.close()
        context.term()

class TestKVCacheRecvingThreadBasic(unittest.TestCase):
    def setUp(self):
        self.engine = MagicMock()
        self.ready_event = threading.Event()
        self.thread = KVCacheRecvingThread(
            tp_rank=0,
            tp_size=4,
            engine=self.engine,
            local_engine_id="local_engine",
            local_handshake_port=5555,
            local_kv_caches_base_addr=[0x1000, 0x2000],
            block_len=[1024, 2048],
            ready_event=self.ready_event
        )

    def test_add_request(self):
        """测试请求队列添加功能"""
        test_req = {
            "request_id": "req1",
            "local_block_ids": [1, 2],
            "remote_block_ids": [3, 4],
            "remote_engine_id": "remote_engine",
            "remote_host": "localhost",
            "remote_handshake_port": 6666,
            "remote_transfer_port": 7777
        }
        self.thread.add_request(**test_req)
        
        # 验证队列内容
        queued = self.thread.request_queue.get_nowait()
        self.assertEqual(queued["request_id"], "req1")
        self.assertEqual(queued["remote_host"], "localhost")
        
    @patch.object(KVCacheTaskTracker, 'get_and_clear_finished_requests')
    def test_get_finished_requests(self, mock_tracker):
        """测试完成请求获取"""
        mock_tracker.return_value = {"req1", "req2"}
        result = self.thread.get_and_clear_finished_requests()
        self.assertEqual(result, {"req1", "req2"})


class TestSocketManagement(unittest.TestCase):
    def setUp(self):
        # 准备所有必要的初始化参数
        self.engine = MagicMock()
        self.ready_event = threading.Event()
        self.thread = KVCacheRecvingThread(
            tp_rank=0,
            tp_size=4,
            engine=self.engine,
            local_engine_id="local_engine",
            local_handshake_port=5555,
            local_kv_caches_base_addr=[0x1000, 0x2000],
            block_len=[1024, 2048],
            ready_event=self.ready_event
        )
        
        # Mock远程socket池
        self.thread.remote_sockets = defaultdict(deque)
        self.thread.remote_poller = MagicMock()

    @patch('vllm_ascend.distributed.mooncake_connector.zmq.Context')
    @patch('vllm_ascend.distributed.mooncake_connector.make_zmq_socket')
    def test_get_remote_socket(self, mock_make_socket, mock_context):
        """测试获取新socket"""
        # 准备测试数据
        mock_sock = MagicMock()
        mock_make_socket.return_value = mock_sock
        test_host = "test_host"
        test_port = 12345
        
        # 调用被测方法
        sock = self.thread._get_remote_socket(test_host, test_port)
        
        # 验证结果
        self.assertEqual(sock, mock_sock)
        
        # 验证调用参数（使用更灵活的断言方式）
        mock_make_socket.assert_called_once()
        args, kwargs = mock_make_socket.call_args
        
        # 验证位置参数或关键字参数
        self.assertEqual(kwargs.get('path'), 'tcp://test_host:12345')
        self.assertEqual(kwargs.get('socket_type'), zmq.REQ)
        self.assertFalse(kwargs.get('bind', True))
        
        # 验证poll注册
        self.thread.remote_poller.register.assert_called_with(
            mock_sock, zmq.POLLIN)

    def test_return_socket_to_pool(self):
        """测试socket归还机制"""
        # 准备测试数据
        mock_sock = MagicMock()
        test_host = "test_host"
        test_port = 12345
        test_path = make_zmq_path("tcp", test_host, test_port)
        
        # 调用被测方法
        self.thread._return_remote_socket(mock_sock, test_host, test_port)
        
        # 验证socket池状态
        self.assertEqual(len(self.thread.remote_sockets[test_path]), 1)
        self.assertEqual(self.thread.remote_sockets[test_path][0], mock_sock)
        
        # 验证未错误注册到poller
        self.thread.remote_poller.register.assert_not_called()

class TestCoreFunctionality(unittest.TestCase):
    def setUp(self):
        # 准备所有必要的初始化参数
        self.engine = MagicMock()
        self.ready_event = threading.Event()
        
        # 使用MagicMock替换实际的queue.Queue
        self.mock_queue = MagicMock()
        
        self.thread = KVCacheRecvingThread(
            tp_rank=0,
            tp_size=4,
            engine=self.engine,
            local_engine_id="local_engine",
            local_handshake_port=5555,
            local_kv_caches_base_addr=[0x1000, 0x2000],
            block_len=[1024, 2048],
            ready_event=self.ready_event
        )
        
        # 替换request_queue为Mock对象
        self.thread.request_queue = self.mock_queue
        
        # 准备测试请求数据
        self.test_req = {
            "request_id": "req1",
            "local_block_ids": [1, 2],
            "remote_block_ids": [3, 4],
            "remote_engine_id": "remote_engine",
            "remote_host": "localhost",
            "remote_handshake_port": 6666,
            "remote_transfer_port": 7777
        }

        # Mock任务追踪器
        self.thread.task_tracker = MagicMock()
        
        # 设置engine.batch_transfer_sync_read的返回值
        self.engine.batch_transfer_sync_read.return_value = 0  # 成功状态

    @patch.object(KVCacheRecvingThread, '_transfer_kv_cache')
    @patch.object(KVCacheRecvingThread, '_send_done_recv_signal')
    def test_handle_request(self, mock_send, mock_transfer):
        """测试请求处理全流程"""
        # 调用被测方法
        self.thread._handle_request(self.test_req)
        
        # 验证核心方法调用
        mock_transfer.assert_called_once_with(self.test_req)
        mock_send.assert_called_once_with("req1", "localhost", 6666)
        
        # 验证任务标记完成
        self.thread.task_tracker.update_done_task_count.assert_called_once_with(
            "req1", self.thread.tp_rank)
        
        # 验证队列任务完成标记
        self.mock_queue.task_done.assert_called_once()

    @patch.object(KVCacheRecvingThread, '_get_remote_metadata')
    def test_transfer_kv_cache(self, mock_get_meta):
        """测试KV缓存传输逻辑"""
        # 模拟元数据
        self.thread.kv_caches_base_addr["remote_engine"] = {
            6666: [0x3000, 0x4000]  # 远程KV缓存基地址
        }
        
        # 调用被测方法
        self.thread._transfer_kv_cache(self.test_req)
        
        # 验证引擎调用参数
        self.engine.batch_transfer_sync_read.assert_called_once()
        
        # 获取调用参数
        call_args, call_kwargs = self.engine.batch_transfer_sync_read.call_args
        
        # 验证关键参数
        self.assertEqual(call_args[0], "localhost:7777")  # session_id
        self.assertIsInstance(call_args[1], list)  # src_list
        self.assertIsInstance(call_args[2], list)  # dst_list
        self.assertIsInstance(call_args[3], list)  # length_list
        
        # 验证列表长度一致
        self.assertEqual(len(call_args[1]), len(call_args[2]))
        self.assertEqual(len(call_args[1]), len(call_args[3]))
        
        # 验证没有调用元数据获取（因为已有缓存）
        mock_get_meta.assert_not_called()


    def test_transfer_kv_cache_failure(self):
        """测试KV缓存传输失败场景"""
        # 模拟传输失败
        self.engine.batch_transfer_sync_read.return_value = -1
        self.thread.kv_caches_base_addr["remote_engine"] = {
            6666: [0x3000, 0x4000]
        }
        
        # 验证会抛出异常
        with self.assertRaises(RuntimeError):
            self.thread._transfer_kv_cache(self.test_req)


class TestMetadataHandling(unittest.TestCase):
    def setUp(self):
        """初始化测试环境"""
        self.engine = MagicMock()
        self.ready_event = threading.Event()
        self.thread = KVCacheRecvingThread(
            tp_rank=0,
            tp_size=4,
            engine=self.engine,
            local_engine_id="local_engine",
            local_handshake_port=5555,
            local_kv_caches_base_addr=[0x1000, 0x2000],
            block_len=[1024, 2048],
            ready_event=self.ready_event
        )
        
        # 准备测试元数据
        self.test_metadata = MooncakeAgentMetadata(
            engine_id="remote_engine",
            kv_caches_base_addr=[0x3000, 0x4000],
            num_blocks=2
        )

    @patch('vllm_ascend.distributed.mooncake_connector.ensure_zmq_send')
    @patch('vllm_ascend.distributed.mooncake_connector.ensure_zmq_recv')
    def test_get_remote_metadata_success(self, mock_recv, mock_send):
        """测试成功获取远程元数据"""
        # 模拟返回的元数据
        mock_recv.return_value = msgspec.msgpack.encode(self.test_metadata)
        
        with patch.object(self.thread, '_get_remote_socket') as mock_get_socket, \
             patch.object(self.thread, '_return_remote_socket') as mock_return_socket:
            
            mock_socket = MagicMock()
            mock_get_socket.return_value = mock_socket
            
            # 调用被测方法
            self.thread._get_remote_metadata("host1", 5555)
            
            # 验证socket获取和归还
            mock_get_socket.assert_called_once_with("host1", 5555)
            mock_return_socket.assert_called_once_with(mock_socket, "host1", 5555)
            
            # 验证消息发送和接收
            mock_send.assert_called_once_with(mock_socket, self.thread.encoder.encode((GET_META_MSG, "")))
            mock_recv.assert_called_once_with(mock_socket, self.thread.remote_poller)
            
            # 验证元数据存储
            self.assertEqual(
                self.thread.kv_caches_base_addr["remote_engine"][5555],
                [0x3000, 0x4000]
            )

    @patch('vllm_ascend.distributed.mooncake_connector.ensure_zmq_send')
    @patch('vllm_ascend.distributed.mooncake_connector.ensure_zmq_recv', side_effect=Exception("Network error"))
    def test_get_remote_metadata_failure(self, mock_recv, mock_send):
        """测试获取远程元数据失败场景"""
        with patch.object(self.thread, '_get_remote_socket') as mock_get_socket, \
             patch.object(self.thread, '_return_remote_socket') as mock_return_socket:
            
            mock_socket = MagicMock()
            mock_get_socket.return_value = mock_socket
            
            # 调用被测方法并验证异常
            with self.assertRaises(Exception) as context:
                self.thread._get_remote_metadata("host1", 5555)
            
            self.assertEqual(str(context.exception), "Network error")
            
            # 验证socket仍然被归还
            mock_return_socket.assert_called_once()


class TestMainThreadLoop(unittest.TestCase):
    def setUp(self):
        """初始化测试环境"""
        self.engine = MagicMock()
        self.ready_event = threading.Event()
        self.thread = KVCacheRecvingThread(
            tp_rank=0,
            tp_size=4,
            engine=self.engine,
            local_engine_id="local_engine",
            local_handshake_port=5555,
            local_kv_caches_base_addr=[0x1000, 0x2000],
            block_len=[1024, 2048],
            ready_event=self.ready_event
        )
        
        # 替换为真实的Queue
        self.thread.request_queue = queue.Queue()

    @patch.object(KVCacheRecvingThread, '_handle_request')
    def test_run_loop_normal(self, mock_handle):
        """测试主线程正常处理流程"""
        # 准备测试请求
        test_request = {
            "request_id": "req1",
            "local_block_ids": [1, 2],
            "remote_block_ids": [3, 4],
            "remote_engine_id": "remote_engine",
            "remote_host": "localhost",
            "remote_handshake_port": 6666,
            "remote_transfer_port": 7777
        }
        
        # 添加请求和终止信号
        self.thread.request_queue.put(test_request)
        self.thread.request_queue.put(None)  # 终止信号
        
        # 启动线程
        self.thread.start()
        
        # 等待线程处理
        time.sleep(0.1)
        self.thread.join(timeout=1.0)
        
        # 验证
        self.assertTrue(self.thread.ready_event.is_set())
        mock_handle.assert_called_once_with(test_request)
        self.assertTrue(self.thread.request_queue.empty())


# 模拟 VLLM 配置
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

# 模拟请求对象
class MockRequest:
    def __init__(self, request_id, prompt_token_ids=None, kv_transfer_params=None, status=None):
        self.request_id = request_id
        self.prompt_token_ids = prompt_token_ids or [1, 2, 3, 4]
        self.kv_transfer_params = kv_transfer_params or {}
        self.status = status or "running"
        self.output_token_ids = [101, 102]

# 测试 KVCacheTaskTracker
class TestKVCacheTaskTracker(unittest.TestCase):
    def setUp(self):
        self.tracker = KVCacheTaskTracker(
            tp_rank=0,
            local_engine_id="test_engine",
            target_count=2
        )
    
    def test_update_task_count(self):
        # 初始状态检查
        self.assertEqual(len(self.tracker.done_task_counts), 0)
        self.assertEqual(len(self.tracker.finished_requests), 0)
        
        # 添加任务完成通知
        self.tracker.update_done_task_count("req1", 0)
        self.tracker.update_done_task_count("req1", 1)
        
        # 验证请求完成
        self.assertEqual(len(self.tracker.finished_requests), 1)
        self.assertTrue("req1" in self.tracker.finished_requests)
        
        # 获取并清除完成请求
        finished = self.tracker.get_and_clear_finished_requests()
        self.assertEqual(finished, {"req1"})
        self.assertEqual(len(self.tracker.finished_requests), 0)
    
    def test_duplicate_task_update(self):
        self.tracker.update_done_task_count("req1", 0)
        self.tracker.update_done_task_count("req1", 0)  # 重复更新
        self.tracker.update_done_task_count("req1", 1)
        
        finished = self.tracker.get_and_clear_finished_requests()
        self.assertEqual(finished, {"req1"})

# 测试 MooncakeConnectorMetadata
class TestMooncakeConnectorMetadata(unittest.TestCase):
    def test_add_new_req(self):
        meta = MooncakeConnectorMetadata()
        meta.add_new_req(
            request_id="req1",
            local_block_ids=[1, 2, 3],
            kv_transfer_params={
                "remote_block_ids": [4, 5, 6],
                "remote_engine_id": "remote_engine",
                "remote_host": "localhost",
                "remote_port": 5000
            }
        )
        
        self.assertEqual(len(meta.requests), 1)
        req_meta = meta.requests["req1"]
        self.assertIsInstance(req_meta, ReqMeta)
        self.assertEqual(req_meta.local_block_ids, [1, 2, 3])
        self.assertEqual(req_meta.remote_block_ids, [4, 5, 6])
        self.assertEqual(req_meta.remote_engine_id, "remote_engine")
        self.assertEqual(req_meta.remote_host, "localhost")
        self.assertEqual(req_meta.remote_port, 5000)

# 测试 MooncakeConnectorScheduler
class TestMooncakeConnectorScheduler(unittest.TestCase):
    def setUp(self):
        config = MockVllmConfig()
        self.scheduler = MooncakeConnectorScheduler(config, "test_engine")
    
    def test_get_num_new_matched_tokens(self):
        # 没有远程预填充的情况
        request = MockRequest("req1")
        tokens, async_flag = self.scheduler.get_num_new_matched_tokens(request, 0)
        self.assertEqual(tokens, 0)
        self.assertFalse(async_flag)
        
        # 有远程预填充的情况
        request.kv_transfer_params = {"do_remote_prefill": True}
        tokens, async_flag = self.scheduler.get_num_new_matched_tokens(request, 0)
        self.assertEqual(tokens, 3)  # len(prompt_token_ids) - 1
        self.assertTrue(async_flag)
    
    
    def test_build_connector_meta(self):
        # 添加一个需要接收的请求
        request = MockRequest("req1")
        blocks_mock = MagicMock()
        blocks_mock.get_unhashed_block_ids.return_value = [4, 5, 6]
        self.scheduler._reqs_need_recv["req1"] = (
            request,
            [4, 5, 6]
        )
        request.kv_transfer_params = {
            "remote_block_ids": [1, 2, 3],
            "remote_engine_id": "remote",
            "remote_host": "localhost",
            "remote_port": 5000
        }
        
        meta = self.scheduler.build_connector_meta(MagicMock())
        self.assertIsInstance(meta, MooncakeConnectorMetadata)
        self.assertEqual(len(meta.requests), 1)
        self.assertEqual(meta.requests["req1"].local_block_ids, [4, 5, 6])
        self.assertEqual(meta.requests["req1"].remote_block_ids, [1, 2, 3])
        
        # 构建后应该清除请求列表
        self.assertEqual(len(self.scheduler._reqs_need_recv), 0)

# 测试 MooncakeConnectorWorker
class TestMooncakeConnectorWorker(unittest.TestCase):
    def setUp(self):
        self.config = MockVllmConfig()
        self.worker = MooncakeConnectorWorker(self.config, "test_worker")
        self.worker.engine = MagicMock()  # 模拟TransferEngine
        
        # 设置环境变量
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0,1"
    
    

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
    
    def test_string_to_int64_hash(self):
        # 验证相同字符串产生相同哈希
        hash1 = string_to_int64_hash("test_string")
        hash2 = string_to_int64_hash("test_string")
        self.assertEqual(hash1, hash2)
        
        # 验证不同字符串产生不同哈希
        hash3 = string_to_int64_hash("different_string")
        self.assertNotEqual(hash1, hash3)

# 测试 MooncakeConnector
class TestMooncakeConnector(unittest.TestCase):
    def test_scheduler_role(self):
        config = MockVllmConfig()
        connector = MooncakeConnector(config, KVConnectorRole.SCHEDULER)
        
        self.assertIsNotNone(connector.connector_scheduler)
        self.assertIsNone(connector.connector_worker)
    
    
    @patch.object(MooncakeConnectorScheduler, "get_num_new_matched_tokens")
    def test_scheduler_methods(self, mock_method):
        config = MockVllmConfig()
        connector = MooncakeConnector(config, KVConnectorRole.SCHEDULER)
        
        request = MockRequest("req1")
        connector.get_num_new_matched_tokens(request, 0)
        mock_method.assert_called_once_with(request, 0)


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

    
    def test_request_finished_no_remote_decode(self):
        request = MockRequest("req1")
        delay_free, params = self.scheduler.request_finished(request, [1, 2, 3])
        self.assertFalse(delay_free)
        self.assertIsNone(params)

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

class TestUtils(unittest.TestCase):

    def test_string_to_int64_hash(self):
        h1 = string_to_int64_hash("hello")
        h2 = string_to_int64_hash("hello")
        h3 = string_to_int64_hash("world")
        self.assertEqual(h1, h2)
        self.assertNotEqual(h1, h3)
        self.assertIsInstance(h1, int)

    def test_group_concurrent_contiguous(self):
        src = [1, 2, 3, 5, 6]
        dst = [10, 11, 12, 20, 21]
        src_g, dst_g = group_concurrent_contiguous(src, dst)
        self.assertEqual(src_g, [[1, 2, 3], [5, 6]])
        self.assertEqual(dst_g, [[10, 11, 12], [20, 21]])

    def test_group_empty(self):
        src_g, dst_g = group_concurrent_contiguous([], [])
        self.assertEqual(src_g, [])
        self.assertEqual(dst_g, [])

    def test_zmq_ctx_invalid_type(self):
        with self.assertRaises(ValueError):
            with zmq_ctx("INVALID", "tcp://127.0.0.1:5555"):
                pass

    @patch("vllm_ascend.distributed.mooncake_connector.make_zmq_socket")
    def test_zmq_ctx_ok(self, mock_make_socket):
        mock_socket = MagicMock()
        mock_make_socket.return_value = mock_socket
        with zmq_ctx(zmq.REQ, "tcp://localhost:1234") as s:
            self.assertEqual(s, mock_socket)
        # 确保上下文销毁被调用（非必须）

    @patch("vllm_ascend.distributed.mooncake_connector.logger")
    def test_ensure_zmq_send_success(self, mock_logger):
        mock_socket = MagicMock()
        ensure_zmq_send(mock_socket, b"hello")
        mock_socket.send.assert_called_once_with(b"hello")

    @patch("vllm_ascend.distributed.mooncake_connector.logger")
    def test_ensure_zmq_send_retry_and_fail(self, mock_logger):
        mock_socket = MagicMock()
        mock_socket.send.side_effect = zmq.ZMQError("send failed")
        with self.assertRaises(RuntimeError):
            ensure_zmq_send(mock_socket, b"hello", max_retries=2)
        self.assertEqual(mock_socket.send.call_count, 2)

    @patch("vllm_ascend.distributed.mooncake_connector.logger")
    def test_ensure_zmq_recv_success(self, mock_logger):
        mock_socket = MagicMock()
        mock_socket.recv.return_value = b"response"
        mock_poller = MagicMock()
        mock_poller.poll.return_value = [(mock_socket, zmq.POLLIN)]
        data = ensure_zmq_recv(mock_socket, mock_poller)
        self.assertEqual(data, b"response")

    @patch("vllm_ascend.distributed.mooncake_connector.logger")
    def test_ensure_zmq_recv_timeout_and_fail(self, mock_logger):
        mock_socket = MagicMock()
        mock_poller = MagicMock()
        mock_poller.poll.return_value = []
        with self.assertRaises(RuntimeError):
            ensure_zmq_recv(mock_socket, mock_poller, timeout=0.01, max_retries=2)


# Mock classes for dependencies
class MockVllmConfig:
    def __init__(self):
        self.parallel_config = MagicMock()
        self.cache_config = MagicMock()
        self.kv_transfer_config = MagicMock()
        
        # Default config values
        self.parallel_config.tensor_parallel_size = 2
        self.parallel_config.data_parallel_rank_local = 0
        self.parallel_config.data_parallel_size_local = 1
        self.cache_config.block_size = 16
        self.kv_transfer_config.kv_port = 5000
        self.kv_transfer_config.kv_role = 'kv_producer'
        self.kv_transfer_config.extra_config = {
            "prefill": {"tp_size": 2, "dp_size": 1},
            "decode": {"tp_size": 2, "dp_size": 1}
        }
        
    def get_from_extra_config(self, key, default):
        return self.kv_transfer_config.extra_config.get(key, default)

class MockTransferEngine:
    def __init__(self):
        pass
    
    def initialize(self, *args, **kwargs):
        return 0
    
    def register_memory(self, *args, **kwargs):
        return 0

class MockMooncakeAgentMetadata:
    def __init__(self, **kwargs):
        pass

class MockMooncakeConnectorMetadata:
    def __init__(self):
        self.requests = {}

class MockKVCacheSendingThread(threading.Thread):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.daemon = True
        self._finished_requests = set()
    
    def get_and_clear_finished_requests(self):
        return self._finished_requests
    
    def start(self):
        pass

class MockKVCacheRecvingThread(threading.Thread):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.daemon = True
        self._finished_requests = set()
        self.add_request = MagicMock()
    
    def get_and_clear_finished_requests(self):
        return self._finished_requests
    
    def start(self):
        pass

# Mock Tensor class
class MockTensor:
    def __init__(self, *args, **kwargs):
        self.size = MagicMock(return_value=(10, 16, 8, 16))
        self.element_size = MagicMock(return_value=4)
        self.shape = (10, 16, 8, 16)
        self.data_ptr = MagicMock(return_value=0x1000)

# Module-level mocks
mock_envs_ascend = MagicMock()
mock_envs_ascend.MOONCAKE_CONNECTOR_PROTOCOL = "mock_protocol"

mock_logger = MagicMock()


# Mock classes for dependencies
class MockVllmConfig:
    def __init__(self):
        self.parallel_config = MagicMock()
        self.cache_config = MagicMock()
        self.kv_transfer_config = MagicMock()
        
        self.parallel_config.tensor_parallel_size = 2
        self.parallel_config.data_parallel_rank_local = 0 
        self.parallel_config.data_parallel_size_local = 1
        self.cache_config.block_size = 16
        self.kv_transfer_config.kv_port = 5000
        self.kv_transfer_config.kv_role = 'kv_producer'
        self.kv_transfer_config.get_from_extra_config = MagicMock()
        self.kv_transfer_config.get_from_extra_config.side_effect = lambda k, d: {
            "prefill": {"tp_size": 2, "dp_size": 1},
            "decode": {"tp_size": 2, "dp_size": 1}
        }.get(k, d)

class MockTransferEngine:
    def initialize(self, *args, **kwargs):
        return 0
    
    def register_memory(self, *args, **kwargs):
        return 0

# Mock module for envs_ascend
class MockEnvsAscend:
    MOONCAKE_CONNECTOR_PROTOCOL = "mock_protocol"

# Mock functions
def mock_get_tensor_model_parallel_rank():
    return 0

def mock_get_tp_group():
    return MagicMock()

def mock_get_ip():
    return "127.0.0.1"

def mock_string_to_int64_hash(s):
    return hash(s)

class TestMooncakeConnectorWorker(unittest.TestCase):
    def setUp(self):
        # Create a mock for the envs_ascend module
        self.envs_ascend_mock = MockEnvsAscend()
        
        # Patch all dependencies
        self.patches = [
            patch('os.getenv', return_value="0,1"),
            patch('torch.Tensor.size', return_value=(10, 16, 8, 16)),
            patch('torch.Tensor.element_size', return_value=4),
            patch('torch.Tensor.data_ptr', return_value=0x1000),
            patch('math.prod', return_value=128),
            patch('random.Random'),
            patch('vllm_ascend.distributed.mooncake_connector.get_tensor_model_parallel_rank', mock_get_tensor_model_parallel_rank),
            patch('vllm_ascend.distributed.mooncake_connector.get_tp_group', mock_get_tp_group),
            patch('vllm_ascend.distributed.mooncake_connector.get_ip', mock_get_ip),
            patch('vllm_ascend.distributed.mooncake_connector.string_to_int64_hash', mock_string_to_int64_hash),
            patch('vllm_ascend.distributed.mooncake_connector.TransferEngine', MockTransferEngine),
            patch('vllm_ascend.distributed.mooncake_connector.KVCacheSendingThread', MagicMock()),
            patch('vllm_ascend.distributed.mooncake_connector.KVCacheRecvingThread', MagicMock()),
            patch('vllm_ascend.distributed.mooncake_connector.logger', MagicMock()),
            patch('vllm_ascend.distributed.mooncake_connector.threading.Event', MagicMock()),
            patch.dict('sys.modules', {'vllm_ascend.envs': self.envs_ascend_mock}),
            patch('vllm_ascend.distributed.mooncake_connector.envs_ascend', self.envs_ascend_mock),  # Add this line
        ]
        
        for p in self.patches:
            p.start()
        
        self.vllm_config = MockVllmConfig()
        self.engine_id = "test_engine"
        self.kv_caches = {"layer1": (MagicMock(), MagicMock())}

    def tearDown(self):
        for p in self.patches:
            p.stop()

    def test_register_kv_caches_producer(self):
        """Test KV cache registration in producer mode"""
        worker = MooncakeConnectorWorker(self.vllm_config, self.engine_id)
        worker.register_kv_caches(self.kv_caches)
        
        self.assertEqual(len(worker.kv_caches), 1)
        self.assertIsNotNone(worker.kv_send_thread)
        self.assertIsNone(worker.kv_recv_thread)

    def test_register_kv_caches_consumer(self):
        """Test KV cache registration in consumer mode"""
        self.vllm_config.kv_transfer_config.kv_role = 'kv_consumer'
        worker = MooncakeConnectorWorker(self.vllm_config, self.engine_id)
        worker.register_kv_caches(self.kv_caches)
        
        self.assertIsNone(worker.kv_send_thread)
        self.assertIsNotNone(worker.kv_recv_thread)

    def test_register_kv_caches_mla_case(self):
        """Test MLA-style KV cache registration"""
        mla_cache1 = MagicMock()
        mla_cache1.size.return_value = (10, 16, 1, 16)
        mla_cache2 = MagicMock()
        mla_cache2.size.return_value = (10, 16, 1, 8)
        mla_caches = {"layer1": (mla_cache1, mla_cache2)}
        
        worker = MooncakeConnectorWorker(self.vllm_config, self.engine_id)
        worker.register_kv_caches(mla_caches)
        
        self.assertTrue(worker.use_mla)
        self.assertEqual(len(worker.block_len), 2)

    def test_initialize_failure(self):
        """Test initialization failure"""
        worker = MooncakeConnectorWorker(self.vllm_config, self.engine_id)
        with patch.object(MockTransferEngine, 'initialize', return_value=1):
            with self.assertRaises(RuntimeError):
                worker._initialize("host:port", None)

if __name__ == '__main__':
    unittest.main()
