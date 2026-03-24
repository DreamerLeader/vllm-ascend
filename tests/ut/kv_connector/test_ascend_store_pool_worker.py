# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Tests for pool_worker.py - covers KVPoolWorker helper methods:
check_all_layers_exists, find_min_first_non_one_index, lookup, lookup_scheduler,
get_and_clear_finished_requests, get_kv_events, register_kv_caches."""

import threading
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.config_data import (
    AscendConnectorMetadata,
    ChunkedTokenDatabase,
    KeyMetadata,
    LoadSpec,
    ReqMeta,
)


def make_pool_worker():
    """Create a KVPoolWorker with mocked dependencies."""
    with (
        patch(
            "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_tensor_model_parallel_rank",
            return_value=0,
        ),
        patch(
            "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_tensor_model_parallel_world_size",
            return_value=2,
        ),
        patch(
            "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_pcp_group"
        ) as mock_pcp,
        patch(
            "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_decode_context_model_parallel_world_size",
            return_value=1,
        ),
        patch(
            "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_decode_context_model_parallel_rank",
            return_value=0,
        ),
        patch(
            "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.importlib.import_module"
        ) as mock_import,
    ):
        mock_pcp_group = MagicMock()
        mock_pcp_group.world_size = 1
        mock_pcp_group.rank_in_group = 0
        mock_pcp.return_value = mock_pcp_group

        # Mock backend module
        mock_backend_class = MagicMock()
        mock_backend_instance = MagicMock()
        mock_backend_class.return_value = mock_backend_instance
        mock_module = MagicMock()
        mock_module.MooncakeBackend = mock_backend_class
        setattr(mock_module, "MooncakeBackend", mock_backend_class)
        mock_import.return_value = mock_module

        # Create VllmConfig mock
        vllm_config = MagicMock()
        vllm_config.model_config.model = "/path/to/test-model"
        vllm_config.model_config.use_mla = False
        vllm_config.model_config.hf_text_config = MagicMock(spec=[])
        vllm_config.model_config.get_total_num_kv_heads.return_value = 8
        vllm_config.model_config.get_num_layers.return_value = 4
        vllm_config.parallel_config.data_parallel_rank = 0
        vllm_config.parallel_config.rank = 0
        vllm_config.parallel_config.pipeline_parallel_size = 1
        vllm_config.cache_config.block_size = 4
        vllm_config.kv_transfer_config.kv_role = "kv_producer"
        vllm_config.kv_transfer_config.kv_connector_extra_config = {
            "backend": "mooncake",
        }
        vllm_config.kv_transfer_config.kv_connector_extra_config = {
            "backend": "mooncake",
        }

        vllm_config.kv_events_config = None

        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker

        worker = KVPoolWorker(vllm_config, use_layerwize=False)
        worker.m_store = mock_backend_instance
        return worker


# ─── check_all_layers_exists ─────────────────────────────────────────────────


class TestCheckAllLayersExists:
    def setup_method(self):
        self.worker = make_pool_worker()

    def test_all_exist(self):
        # 2 chunks, 3 layers each: [1,1,1, 1,1,1]
        res = self.worker.check_all_layers_exists([1, 1, 1, 1, 1, 1], 3)
        assert res == [1, 1]

    def test_partial_exist(self):
        # 2 chunks, 3 layers: chunk 0 all exist, chunk 1 missing layer 2
        res = self.worker.check_all_layers_exists([1, 1, 1, 1, 1, 0], 3)
        assert res == [1, 0]

    def test_none_exist(self):
        res = self.worker.check_all_layers_exists([0, 0, 0, 0], 2)
        assert res == [0, 0]

    def test_single_chunk(self):
        res = self.worker.check_all_layers_exists([1, 1], 2)
        assert res == [1]


# ─── find_min_first_non_one_index ────────────────────────────────────────────


class TestFindMinFirstNonOneIndex:
    def setup_method(self):
        self.worker = make_pool_worker()

    def test_all_ones(self):
        result = self.worker.find_min_first_non_one_index([[1, 1, 1], [1, 1, 1]])
        assert result == -1

    def test_first_row_has_zero(self):
        result = self.worker.find_min_first_non_one_index([[1, 0, 1], [1, 1, 1]])
        assert result == 1

    def test_second_row_earlier_zero(self):
        result = self.worker.find_min_first_non_one_index([[1, 1, 0], [0, 1, 1]])
        assert result == 0

    def test_empty_input(self):
        result = self.worker.find_min_first_non_one_index([])
        assert result == -1

    def test_single_element_zero(self):
        result = self.worker.find_min_first_non_one_index([[0]])
        assert result == 0


# ─── lookup ──────────────────────────────────────────────────────────────────


class TestLookup:
    def setup_method(self):
        self.worker = make_pool_worker()
        self.worker.token_database.set_block_len([64])
        self.worker.token_database.set_kv_caches_base_addr([1000, 2000])

    def test_lookup_all_exist(self):
        self.worker.m_store.exists.return_value = [1, 1]
        result = self.worker.lookup(8, ["aabb", "ccdd"], use_layerwise=False)
        assert result == 8  # end value from last iteration

    def test_lookup_partial(self):
        self.worker.m_store.exists.return_value = [1, 0]
        result = self.worker.lookup(8, ["aabb", "ccdd"], use_layerwise=False)
        assert result == 4  # starts[1]

    def test_lookup_none_exist(self):
        self.worker.m_store.exists.return_value = [0, 0]
        result = self.worker.lookup(8, ["aabb", "ccdd"], use_layerwise=False)
        assert result == 0

    def test_lookup_exception(self):
        self.worker.m_store.exists.side_effect = Exception("connection error")
        result = self.worker.lookup(8, ["aabb", "ccdd"], use_layerwise=False)
        assert result == 0

    def test_lookup_layerwise(self):
        # 2 blocks * 4 layers = 8 keys, all exist
        self.worker.m_store.exists.return_value = [1] * 8
        result = self.worker.lookup(8, ["aabb", "ccdd"], use_layerwise=True)
        assert result == 8


# ─── lookup_scheduler ────────────────────────────────────────────────────────


class TestLookupScheduler:
    def setup_method(self):
        self.worker = make_pool_worker()
        self.worker.token_database.set_block_len([64])
        self.worker.token_database.set_kv_caches_base_addr([1000, 2000])

    def test_lookup_scheduler_all_exist(self):
        # tp_size=2, num_kv_head=8, pp_size=1
        # min(2,8) = 2 tp ranks, 1 pp rank => 2 groups
        # 2 blocks per group => 4 total keys
        self.worker.m_store.exists.return_value = [1] * 4
        result = self.worker.lookup_scheduler(8, ["aabb", "ccdd"], use_layerwise=False)
        assert result == 8

    def test_lookup_scheduler_partial(self):
        # First group: [1, 0], second group: [1, 1]
        self.worker.m_store.exists.return_value = [1, 0, 1, 1]
        result = self.worker.lookup_scheduler(8, ["aabb", "ccdd"], use_layerwise=False)
        # min first non-one index across groups => index 1 => starts[1] = 4
        assert result == 4

    def test_lookup_scheduler_exception(self):
        self.worker.m_store.exists.side_effect = Exception("err")
        result = self.worker.lookup_scheduler(8, ["aabb", "ccdd"], use_layerwise=False)
        assert result == 0


# ─── get_and_clear_finished_requests ─────────────────────────────────────────


class TestGetAndClearFinishedRequests:
    def setup_method(self):
        self.worker = make_pool_worker()
        # Create a mock send thread
        self.worker.kv_send_thread = MagicMock()
        self.worker.kv_send_thread.stored_requests = {"r1": 0, "r2": 1}

    def test_finished_with_zero_remaining(self):
        meta = AscendConnectorMetadata(set(), set())
        self.worker.kv_send_thread.stored_requests = {"r1": 0}
        result = self.worker.get_and_clear_finished_requests({"r1"}, meta)
        assert "r1" in result

    def test_finished_with_remaining_jobs(self):
        meta = AscendConnectorMetadata(set(), set())
        self.worker.kv_send_thread.stored_requests = {"r1": 2}
        result = self.worker.get_and_clear_finished_requests({"r1"}, meta)
        assert "r1" not in result
        assert "r1" in self.worker.finished_store_req

    def test_preempted_requests(self):
        meta = AscendConnectorMetadata(set(), {"r1"})
        self.worker.kv_send_thread.stored_requests = {}
        result = self.worker.get_and_clear_finished_requests(set(), meta)
        self.worker.kv_send_thread.delete_finished_stored_request.assert_called_with("r1")

    def test_deferred_finish(self):
        """Test that a request with stored_requests[r1]==0 and r1 in finished_store_req gets finished."""
        meta = AscendConnectorMetadata(set(), set())
        self.worker.kv_send_thread.stored_requests = {"r1": 0}
        self.worker.finished_store_req = {"r1"}
        result = self.worker.get_and_clear_finished_requests(set(), meta)
        assert "r1" in result


# ─── get_kv_events ───────────────────────────────────────────────────────────


class TestGetKVEvents:
    def test_no_events_disabled(self):
        worker = make_pool_worker()
        worker.enable_kv_events = False
        assert worker.get_kv_events() == []

    def test_no_send_thread(self):
        worker = make_pool_worker()
        worker.enable_kv_events = True
        worker.kv_send_thread = None
        assert worker.get_kv_events() == []

    def test_with_events(self):
        worker = make_pool_worker()
        worker.enable_kv_events = True
        worker.kv_send_thread = MagicMock()
        worker.kv_send_thread.get_kv_events.return_value = [MagicMock()]
        events = worker.get_kv_events()
        assert len(events) == 1


# ─── register_kv_caches ─────────────────────────────────────────────────────


class TestRegisterKVCaches:
    def test_register_standard_attention(self):
        worker = make_pool_worker()
        worker.use_mla = False
        worker.use_sparse = False
        worker.kv_role = "kv_producer"

        # Create mock KV caches: 2 layers, each with (K, V) tuple
        k_cache = torch.zeros(10, 4, 8, 16, dtype=torch.float16)
        v_cache = torch.zeros(10, 4, 8, 16, dtype=torch.float16)
        kv_caches = {
            "layer0": (k_cache, v_cache),
            "layer1": (k_cache, v_cache),
        }

        worker.register_kv_caches(kv_caches)
        assert worker.num_blocks == 10
        assert len(worker.block_len) == 1
        # block_len = elem_size * block_size * num_heads * head_dim = 2 * 4 * 8 * 16 = 1024
        assert worker.block_len[0] == 2 * 4 * 8 * 16
        assert len(worker.kv_caches_base_addr) == 4  # 2 layers * 2 (k+v)
        worker.m_store.register_buffer.assert_called_once()

    def test_register_mla(self):
        worker = make_pool_worker()
        worker.use_mla = True
        worker.use_sparse = False
        worker.kv_role = "kv_consumer"
        worker.consumer_is_to_put = False
        worker.load_async = False

        k_cache = torch.zeros(10, 4, 1, 128, dtype=torch.float16)
        v_cache = torch.zeros(10, 4, 1, 64, dtype=torch.float16)
        kv_caches = {
            "layer0": (k_cache, v_cache),
        }

        worker.register_kv_caches(kv_caches)
        assert worker.use_mla is True
        # MLA uses multiple block_len entries
        assert len(worker.block_len) == 2
