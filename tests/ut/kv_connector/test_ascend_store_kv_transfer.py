# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Tests for kv_transfer.py - covers KVTransferThread, KVCacheStoreSendingThread,
KVCacheStoreRecvingThread, KVCacheStoreLayerSendingThread,
KVCacheStoreLayerRecvingThread."""

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.config_data import (
    ChunkedTokenDatabase,
    KeyMetadata,
    LasyerMultiBlockReqMeta,
    LayerPoolKey,
    LoadSpec,
    PoolKey,
    ReqMeta,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer import (
    KVCacheStoreLayerRecvingThread,
    KVCacheStoreLayerSendingThread,
    KVCacheStoreRecvingThread,
    KVCacheStoreSendingThread,
    KVTransferThread,
)


def make_mock_backend():
    backend = MagicMock()
    backend.set_device = MagicMock()
    backend.exists = MagicMock(return_value=[1, 1, 1])
    backend.put = MagicMock()
    backend.get = MagicMock()
    return backend


def make_token_database(block_size=4):
    meta = KeyMetadata("model", 0, 0, 0, 0)
    db = ChunkedTokenDatabase(meta, block_size, None)
    db.set_block_len([64])
    db.set_kv_caches_base_addr([1000, 2000])
    return db


# ─── KVTransferThread ────────────────────────────────────────────────────────


class TestKVTransferThread:
    def test_add_request(self):
        backend = make_mock_backend()
        db = make_token_database()
        ready = threading.Event()
        thread = KVTransferThread(backend, db, 4, 0, 1, ready, "test")
        thread.daemon = True
        # Don't start, just test add_request
        assert thread.request_queue.qsize() == 0
        thread.add_request(MagicMock())
        assert thread.request_queue.qsize() == 1

    def test_get_and_clear_finished(self):
        backend = make_mock_backend()
        db = make_token_database()
        ready = threading.Event()
        thread = KVTransferThread(backend, db, 4, 0, 1, ready, "test")

        thread.set_finished_request("r1")
        thread.set_finished_request("r2")
        finished = thread.get_and_clear_finished_requests()
        assert finished == {"r1", "r2"}
        # Should be cleared
        assert thread.get_and_clear_finished_requests() == set()

    def test_lookup_all_exist(self):
        backend = make_mock_backend()
        backend.exists.return_value = [1, 1, 1]
        db = make_token_database()
        ready = threading.Event()
        thread = KVTransferThread(backend, db, 4, 0, 1, ready, "test")

        result = thread.lookup(["k1", "k2", "k3"])
        assert result == 3

    def test_lookup_partial(self):
        backend = make_mock_backend()
        backend.exists.return_value = [1, 0, 1]
        db = make_token_database()
        ready = threading.Event()
        thread = KVTransferThread(backend, db, 4, 0, 1, ready, "test")

        result = thread.lookup(["k1", "k2", "k3"])
        assert result == 1

    def test_lookup_none_exist(self):
        backend = make_mock_backend()
        backend.exists.return_value = [0, 0]
        db = make_token_database()
        ready = threading.Event()
        thread = KVTransferThread(backend, db, 4, 0, 1, ready, "test")

        result = thread.lookup(["k1", "k2"])
        assert result == 0

    def test_lookup_exception(self):
        backend = make_mock_backend()
        backend.exists.side_effect = Exception("connection error")
        db = make_token_database()
        ready = threading.Event()
        thread = KVTransferThread(backend, db, 4, 0, 1, ready, "test")

        result = thread.lookup(["k1"])
        assert result == 0

    def test_update_and_get_kv_events(self):
        backend = make_mock_backend()
        db = make_token_database()
        ready = threading.Event()
        thread = KVTransferThread(backend, db, 4, 0, 1, ready, "test")

        event1 = MagicMock()
        event2 = MagicMock()
        thread.update_kv_event([event1, event2])
        events = thread.get_kv_events()
        assert len(events) == 2
        # Should be cleared
        assert len(thread.get_kv_events()) == 0

    def test_run_sets_ready_event(self):
        backend = make_mock_backend()
        db = make_token_database()
        ready = threading.Event()
        thread = KVTransferThread(backend, db, 4, 0, 1, ready, "test")
        thread.daemon = True

        # Start thread and check ready event is set
        thread.start()
        assert ready.wait(timeout=2)
        # Stop by putting None (which the run loop handles)
        thread.request_queue.put(None)
        time.sleep(0.1)


# ─── KVCacheStoreSendingThread ────────────────────────────────────────────────


class TestKVCacheStoreSendingThread:
    def make_thread(self, put_step=1):
        backend = make_mock_backend()
        db = make_token_database()
        ready = threading.Event()
        thread = KVCacheStoreSendingThread(
            backend, db, 4, 0, 1, put_step, "kv_producer", ready
        )
        return thread, backend

    def test_add_and_delete_stored_request(self):
        thread, _ = self.make_thread()
        thread.add_stored_request("r1")
        assert thread.stored_requests["r1"] == 1
        thread.add_stored_request("r1")
        assert thread.stored_requests["r1"] == 2
        thread.dec_stored_request("r1")
        assert thread.stored_requests["r1"] == 1
        thread.delete_finished_stored_request("r1")
        assert "r1" not in thread.stored_requests

    def test_dec_nonexistent_request(self):
        thread, _ = self.make_thread()
        # Should not raise
        thread.dec_stored_request("nonexistent")

    def test_delete_nonexistent_request(self):
        thread, _ = self.make_thread()
        # Should not raise
        thread.delete_finished_stored_request("nonexistent")

    def test_handle_request_not_in_stored(self):
        thread, _ = self.make_thread()
        req = ReqMeta(
            req_id="r1",
            token_len_chunk=8,
            block_ids=[0, 1],
            block_hashes=[b"\xaa\xbb", b"\xcc\xdd"],
        )
        # Put the request in the queue first so task_done() works
        thread.request_queue.put(req)
        # r1 not in stored_requests, should return early
        thread._handle_request(req)

    def test_handle_request_all_exist(self):
        thread, backend = self.make_thread()
        backend.exists.return_value = [1, 1]
        thread.add_stored_request("r1")

        req = ReqMeta(
            req_id="r1",
            token_len_chunk=8,
            block_ids=[0, 1],
            block_hashes=[b"\xaa\xbb", b"\xcc\xdd"],
        )
        thread.request_queue.put(req)
        thread._handle_request(req)
        # All exist, so no put should be called
        backend.put.assert_not_called()

    def test_handle_request_stores_kv(self):
        thread, backend = self.make_thread()
        backend.exists.return_value = [0, 0]
        thread.add_stored_request("r1")

        req = ReqMeta(
            req_id="r1",
            token_len_chunk=8,
            block_ids=[0, 1],
            block_hashes=[b"\xaa\xbb", b"\xcc\xdd"],
        )
        thread.request_queue.put(req)
        thread._handle_request(req)
        backend.put.assert_called_once()

    def test_handle_request_with_current_event(self):
        thread, backend = self.make_thread()
        backend.exists.return_value = [0]
        thread.add_stored_request("r1")

        mock_event = MagicMock()
        req = ReqMeta(
            req_id="r1",
            token_len_chunk=4,
            block_ids=[0],
            block_hashes=[b"\xaa\xbb"],
            current_event=mock_event,
        )
        thread.request_queue.put(req)
        thread._handle_request(req)
        mock_event.synchronize.assert_called_once()

    def test_handle_request_with_kv_events(self):
        backend = make_mock_backend()
        backend.exists.return_value = [0]
        db = make_token_database()
        ready = threading.Event()
        thread = KVCacheStoreSendingThread(
            backend, db, 4, 0, 1, 1, "kv_producer", ready, enable_kv_event=True
        )
        thread.add_stored_request("r1")

        req = ReqMeta(
            req_id="r1",
            token_len_chunk=4,
            block_ids=[0],
            block_hashes=[b"\xaa\xbb"],
            token_ids=[1, 2, 3, 4],
            original_block_size=4,
        )
        thread.request_queue.put(req)
        thread._handle_request(req)
        events = thread.get_kv_events()
        assert len(events) > 0

    def test_handle_request_consumer_pp(self):
        backend = make_mock_backend()
        backend.exists.return_value = [0]
        meta = KeyMetadata("model", 0, 0, 0, 0)
        db = ChunkedTokenDatabase(meta, 4, [2, 2])
        db.set_block_len([64])
        db.set_kv_caches_base_addr([1000, 2000, 3000, 4000])
        ready = threading.Event()
        thread = KVCacheStoreSendingThread(
            backend, db, 4, 0, 1, 1, "kv_consumer", ready
        )
        thread.add_stored_request("r1")

        req = ReqMeta(
            req_id="r1",
            token_len_chunk=4,
            block_ids=[0],
            block_hashes=[b"\xaa\xbb"],
        )
        thread.request_queue.put(req)
        thread._handle_request(req)
        backend.put.assert_called_once()

    def test_handle_request_empty_keys(self):
        thread, backend = self.make_thread()
        thread.add_stored_request("r1")

        req = ReqMeta(
            req_id="r1",
            token_len_chunk=0,
            block_ids=[],
            block_hashes=[],
        )
        thread.request_queue.put(req)
        thread._handle_request(req)
        backend.put.assert_not_called()


# ─── KVCacheStoreRecvingThread ────────────────────────────────────────────────


class TestKVCacheStoreRecvingThread:
    def test_handle_request(self):
        backend = make_mock_backend()
        db = make_token_database()
        ready = threading.Event()
        thread = KVCacheStoreRecvingThread(backend, db, 4, 0, 1, ready)

        load = LoadSpec(0, 8, can_load=True, token_len=8)
        req = ReqMeta(
            req_id="r1",
            token_len_chunk=8,
            block_ids=[0, 1],
            block_hashes=[b"\xaa\xbb", b"\xcc\xdd"],
            load_spec=load,
        )
        thread.request_queue.put(req)
        thread._handle_request(req)

        backend.get.assert_called_once()
        finished = thread.get_and_clear_finished_requests()
        assert "r1" in finished


# ─── KVCacheStoreLayerSendingThread ──────────────────────────────────────────


class TestKVCacheStoreLayerSendingThread:
    def make_thread(self, num_layers=2):
        backend = make_mock_backend()
        db = make_token_database()
        ready = threading.Event()
        thread = KVCacheStoreLayerSendingThread(
            backend, db, 4, 0, 1, 1, ready, num_layers
        )
        return thread, backend

    def test_handle_request_stores(self):
        thread, backend = self.make_thread()
        backend.exists.return_value = [0]

        meta_k = KeyMetadata("model", 0, 0, 0, 0)
        keys = [LayerPoolKey(meta_k, "h1", 0)]
        req = LasyerMultiBlockReqMeta(
            req_id="r1",
            keys=keys,
            starts=[0],
            ends=[4],
            block_ids=[10],
            layer_id=0,
            is_last_chunk=False,
        )
        thread.request_queue.put(req)
        thread._handle_request(req)
        backend.put.assert_called_once()

    def test_handle_request_all_exist_not_last(self):
        thread, backend = self.make_thread()
        backend.exists.return_value = [1]

        meta_k = KeyMetadata("model", 0, 0, 0, 0)
        keys = [LayerPoolKey(meta_k, "h1", 0)]
        req = LasyerMultiBlockReqMeta(
            req_id="r1",
            keys=keys,
            starts=[0],
            ends=[4],
            block_ids=[10],
            layer_id=0,
            is_last_chunk=False,
        )
        thread._handle_request(req)
        backend.put.assert_not_called()

    def test_handle_request_all_exist_last_layer_last_chunk(self):
        thread, backend = self.make_thread(num_layers=2)
        backend.exists.return_value = [1]

        meta_k = KeyMetadata("model", 0, 0, 0, 0)
        keys = [LayerPoolKey(meta_k, "h1", 1)]
        req = LasyerMultiBlockReqMeta(
            req_id="r1",
            keys=keys,
            starts=[0],
            ends=[4],
            block_ids=[10],
            layer_id=1,  # final layer
            is_last_chunk=True,
        )
        thread._handle_request(req)
        finished = thread.get_and_clear_finished_requests()
        assert "r1" in finished

    def test_handle_request_empty_keys_last_chunk(self):
        thread, backend = self.make_thread()
        req = LasyerMultiBlockReqMeta(
            req_id="r1",
            keys=[],
            starts=[],
            ends=[],
            block_ids=[],
            layer_id=0,
            is_last_chunk=True,
        )
        thread._handle_request(req)
        finished = thread.get_and_clear_finished_requests()
        assert "r1" in finished

    def test_handle_request_with_current_event(self):
        thread, backend = self.make_thread()
        backend.exists.return_value = [0]

        meta_k = KeyMetadata("model", 0, 0, 0, 0)
        keys = [LayerPoolKey(meta_k, "h1", 0)]
        mock_event = MagicMock()
        req = LasyerMultiBlockReqMeta(
            req_id="r1",
            keys=keys,
            starts=[0],
            ends=[4],
            block_ids=[10],
            layer_id=1,
            is_last_chunk=True,
            current_event=mock_event,
        )
        thread.request_queue.put(req)
        thread._handle_request(req)
        mock_event.synchronize.assert_called_once()


# ─── KVCacheStoreLayerRecvingThread ──────────────────────────────────────────


class TestKVCacheStoreLayerRecvingThread:
    def test_handle_request(self):
        backend = make_mock_backend()
        db = make_token_database()
        ready = threading.Event()
        get_event = threading.Event()
        thread = KVCacheStoreLayerRecvingThread(
            backend, db, 4, 0, 1, ready, get_event
        )

        meta_k = KeyMetadata("model", 0, 0, 0, 0)
        keys = [LayerPoolKey(meta_k, "h1", 0)]
        req = LasyerMultiBlockReqMeta(
            req_id="r1",
            keys=keys,
            starts=[0],
            ends=[4],
            block_ids=[10],
            layer_id=0,
        )
        thread.request_queue.put(req)
        thread._handle_request(req)

        backend.get.assert_called_once()
        assert get_event.is_set()
