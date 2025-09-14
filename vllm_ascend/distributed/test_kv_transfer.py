import threading
import queue
import torch
import pytest
from unittest import mock
from vllm_ascend.distributed.mooncake.kv_transfer import KVTransferThread

# vllm_ascend/distributed/mooncake/test_kv_transfer.py



class DummyStore:
    pass

class DummyTokenDB:
    def process_tokens(self, tokens, mask):
        return [(0, 2, "key1"), (2, 4, "key2")]

@pytest.fixture
def kvthread_mla():
    ready_event = threading.Event()
    return KVTransferThread(
        tp_rank=0,
        tp_size=1,
        m_store=DummyStore(),
        local_kv_caches_base_addr=[100, 200],
        token_database=DummyTokenDB(),
        block_len=[16, 32],
        block_size=4,
        ready_event=ready_event,
        name="test"
    )

@pytest.fixture
def kvthread_no_mla():
    ready_event = threading.Event()
    return KVTransferThread(
        tp_rank=0,
        tp_size=1,
        m_store=DummyStore(),
        local_kv_caches_base_addr=[100, 200],
        token_database=DummyTokenDB(),
        block_len=[16],
        block_size=4,
        ready_event=ready_event,
        name="test"
    )

def test_init_use_mla(kvthread_mla, kvthread_no_mla):
    assert kvthread_mla.use_mla is True
    assert kvthread_no_mla.use_mla is False
    assert kvthread_mla.tp_rank == 0
    assert kvthread_mla.tp_size == 1
    assert isinstance(kvthread_mla.request_queue, queue.Queue)
    assert isinstance(kvthread_mla.executor, object)
    assert kvthread_mla.finished_requests == set()

def test_prepare_value_mla(kvthread_mla):
    # block_ids: [0, 1, 2, 3]
    addr, size, block_id = kvthread_mla.prepare_value(4, 8, [0, 1, 2, 3])
    # index 0: base_addr=100, block_len=16, block_id=1
    # index 1: base_addr=200, block_len=32, block_id=1
    assert addr == [100 + 1*16, 200 + 1*32]
    assert size == [int(16/4*4), int(32/4*4)]
    assert block_id == 1

def test_prepare_value_no_mla(kvthread_no_mla):
    addr, size, block_id = kvthread_no_mla.prepare_value(4, 8, [0, 1, 2, 3])
    assert addr == [100 + 1*16, 200 + 1*16]
    assert size == [int(16/4*4), int(16/4*4)]
    assert block_id == 1

def test_prepare_value_layer_mla(kvthread_mla):
    # layer_id=1, block_ids=[0,1,2,3], start=4, end=8
    addr, size = kvthread_mla.prepare_value_layer(4, 8, [0,1,2,3], 1)
    # addr_k = 200 + 1*16, addr_v = 100 + 1*32
    assert addr == [200 + 1*16, 100 + 1*32] or addr == [200 + 1*16, 200 + 1*32]
    assert size == [int(16/4*4), int(32/4*4)]

def test_prepare_value_layer_no_mla(kvthread_no_mla):
    addr, size = kvthread_no_mla.prepare_value_layer(4, 8, [0,1,2,3], 1)
    assert addr == [100 + 2*16, 200 + 2*16]
    assert size == [int(16/4*4), int(16/4*4)]

def test_add_request(kvthread_no_mla):
    kvthread_no_mla.add_request("req1", torch.tensor([1,2]), [0,1], mask=None, is_last_chunk=True)
    req = kvthread_no_mla.request_queue.get_nowait()
    assert req["req_id"] == "req1"
    assert torch.equal(req["tokens"], torch.tensor([1,2]))
    assert req["block_ids"] == [0,1]
    assert req["mask"] is None
    assert req["is_last_chunk"] is True

def test_set_and_get_and_clear_finished_requests(kvthread_no_mla):
    kvthread_no_mla.set_finished_request("req1")
    kvthread_no_mla.set_finished_request("req2")
    finished = kvthread_no_mla.get_and_clear_finished_requests()
    assert finished == {"req1", "req2"}
    assert kvthread_no_mla.finished_requests == set()
    # Should be empty after clear
    finished2 = kvthread_no_mla.get_and_clear_finished_requests()
    assert finished2 == set()

def test_handle_request_noop(kvthread_no_mla):
    # Should do nothing and not raise
    kvthread_no_mla._handle_request({"foo": "bar"})

def test_run_calls_handle(monkeypatch, kvthread_no_mla):
    called = {}
    def fake_handle(req):
        called["yes"] = req
    kvthread_no_mla._handle_request = fake_handle
    # Put a request and a None to exit loop after one iteration
    kvthread_no_mla.request_queue.put({"foo": "bar"})
    kvthread_no_mla.request_queue.put(None)
    # Patch logger to avoid output
    monkeypatch.setattr("vllm.utils.logger", mock.Mock())
    # Run in a thread, stop after two gets
    def run_thread():
        # Only run two iterations
        for _ in range(2):
            req = kvthread_no_mla.request_queue.get()
            if req is None:
                kvthread_no_mla.request_queue.task_done()
                continue
            kvthread_no_mla._handle_request(req)
            kvthread_no_mla.request_queue.task_done()
    run_thread()
    assert called["yes"] == {"foo": "bar"}

def test_run_exception(monkeypatch, kvthread_no_mla):
    def raise_exc(req):
        raise RuntimeError("fail!")
    kvthread_no_mla._handle_request = raise_exc
    kvthread_no_mla.request_queue.put({"foo": "bar"})
    # Patch logger to capture error
    logger_mock = mock.Mock()
    monkeypatch.setattr("vllm.utils.logger", logger_mock)
    # Only run one iteration
    try:
        req = kvthread_no_mla.request_queue.get()
        kvthread_no_mla._handle_request(req)
        kvthread_no_mla.request_queue.task_done()
    except RuntimeError:
        pass
    logger_mock.error.assert_not_called()  # Not called in this direct call
