# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Tests for pool_scheduler.py - covers get_zmq_rpc_path_lookup and
KVPoolScheduler methods."""

from unittest.mock import MagicMock, patch

import pytest

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler import (
    get_zmq_rpc_path_lookup,
)


# ─── get_zmq_rpc_path_lookup ─────────────────────────────────────────────────


class TestGetZmqRpcPathLookup:
    def test_basic(self):
        vllm_config = MagicMock()
        vllm_config.parallel_config.data_parallel_rank = 0
        vllm_config.kv_transfer_config.kv_connector_extra_config = {
            "lookup_rpc_port": 5000,
        }
        with patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.envs") as mock_envs:
            mock_envs.VLLM_RPC_BASE_PATH = "/tmp/vllm"
            result = get_zmq_rpc_path_lookup(vllm_config)

        assert result == "ipc:///tmp/vllm/lookup_rpc_port_5000_dp_rank0"

    def test_mooncake_rpc_port_fallback(self):
        vllm_config = MagicMock()
        vllm_config.parallel_config.data_parallel_rank = 1
        vllm_config.kv_transfer_config.kv_connector_extra_config = {
            "mooncake_rpc_port": 6000,
        }
        with patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.envs") as mock_envs:
            mock_envs.VLLM_RPC_BASE_PATH = "/tmp/vllm"
            result = get_zmq_rpc_path_lookup(vllm_config)

        assert result == "ipc:///tmp/vllm/lookup_rpc_port_6000_dp_rank1"

    def test_no_port_configured(self):
        vllm_config = MagicMock()
        vllm_config.parallel_config.data_parallel_rank = 0
        vllm_config.kv_transfer_config.kv_connector_extra_config = {}
        with patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.envs") as mock_envs:
            mock_envs.VLLM_RPC_BASE_PATH = "/tmp/vllm"
            result = get_zmq_rpc_path_lookup(vllm_config)

        assert result == "ipc:///tmp/vllm/lookup_rpc_port_0_dp_rank0"


# ─── KVPoolScheduler ─────────────────────────────────────────────────────────


class TestKVPoolScheduler:
    def _make_scheduler(self, kv_role="kv_producer"):
        vllm_config = MagicMock()
        vllm_config.kv_transfer_config.kv_role = kv_role
        vllm_config.kv_transfer_config.kv_connector_extra_config = {
            "consumer_is_to_load": False,
            "consumer_is_to_put": False,
            "load_async": False,
        }
        vllm_config.kv_transfer_config.get_from_extra_config = lambda k, d: d
        vllm_config.parallel_config.data_parallel_rank = 0
        vllm_config.parallel_config.prefill_context_parallel_size = 1
        vllm_config.parallel_config.decode_context_parallel_size = 1
        vllm_config.cache_config.block_size = 4

        with (
            patch(
                "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.LookupKeyClient"
            ) as mock_client_cls,
            patch(
                "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.envs"
            ) as mock_envs,
        ):
            mock_envs.VLLM_RPC_BASE_PATH = "/tmp/vllm"
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client

            from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler import (
                KVPoolScheduler,
            )

            scheduler = KVPoolScheduler(vllm_config, use_layerwise=False)
            scheduler.client = mock_client
            return scheduler, mock_client

    def test_get_num_new_matched_tokens_consumer_no_load(self):
        scheduler, client = self._make_scheduler(kv_role="kv_consumer")
        request = MagicMock()
        result, async_flag = scheduler.get_num_new_matched_tokens(request, 0)
        assert result == 0
        assert async_flag is False

    def test_get_num_new_matched_tokens_too_short(self):
        scheduler, client = self._make_scheduler()
        request = MagicMock()
        request.prompt_token_ids = [1, 2, 3]  # 3 tokens, block_size=4
        result, _ = scheduler.get_num_new_matched_tokens(request, 0)
        assert result == 0

    def test_get_num_new_matched_tokens_hit(self):
        scheduler, client = self._make_scheduler()
        request = MagicMock()
        request.prompt_token_ids = list(range(16))
        request.num_tokens = 16
        request.request_id = "r1"
        request.block_hashes = [b"h1", b"h2", b"h3", b"h4"]
        client.lookup.return_value = 12  # 12 tokens hit

        result, _ = scheduler.get_num_new_matched_tokens(request, 4)
        assert result == 8  # 12 - 4

    def test_get_num_new_matched_tokens_all_hit(self):
        scheduler, client = self._make_scheduler()
        request = MagicMock()
        request.prompt_token_ids = list(range(16))
        request.num_tokens = 16
        request.request_id = "r1"
        request.block_hashes = [b"h1", b"h2", b"h3", b"h4"]
        client.lookup.return_value = 16  # All hit

        result, _ = scheduler.get_num_new_matched_tokens(request, 4)
        # num_external_hit = 16-1 = 15, need_to_allocate = 15-4 = 11
        assert result == 11

    def test_get_num_new_matched_tokens_no_benefit(self):
        scheduler, client = self._make_scheduler()
        request = MagicMock()
        request.prompt_token_ids = list(range(16))
        request.num_tokens = 16
        request.request_id = "r1"
        request.block_hashes = [b"h1", b"h2", b"h3", b"h4"]
        client.lookup.return_value = 4

        result, _ = scheduler.get_num_new_matched_tokens(request, 8)
        # num_external_hit=4 < num_computed=8
        assert result == 0

    def test_update_state_after_alloc_no_load_spec(self):
        scheduler, _ = self._make_scheduler()
        request = MagicMock()
        request.request_id = "r1"
        blocks = MagicMock()

        scheduler.update_state_after_alloc(request, blocks, 0)
        assert "r1" in scheduler._unfinished_request_ids

    def test_update_state_after_alloc_with_load(self):
        scheduler, client = self._make_scheduler()
        request = MagicMock()
        request.request_id = "r1"
        request.prompt_token_ids = list(range(16))
        request.num_tokens = 16
        request.block_hashes = [b"h1", b"h2", b"h3", b"h4"]
        client.lookup.return_value = 12

        scheduler.get_num_new_matched_tokens(request, 4)
        assert "r1" in scheduler.load_specs

        blocks = MagicMock()
        blocks.get_block_ids.return_value = [[0, 1, 2]]
        scheduler.update_state_after_alloc(request, blocks, 8)
        assert scheduler.load_specs["r1"].can_load is True

    def test_request_finished_consumer_no_put(self):
        scheduler, _ = self._make_scheduler(kv_role="kv_consumer")
        request = MagicMock()
        request.request_id = "r1"
        delay, extra = scheduler.request_finished(request, [0, 1])
        assert delay is False

    def test_request_finished_with_tracker(self):
        scheduler, _ = self._make_scheduler()
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.config_data import RequestTracker
        scheduler._request_trackers["r1"] = RequestTracker("r1", 16, [0], num_saved_tokens=8)

        request = MagicMock()
        request.request_id = "r1"
        delay, extra = scheduler.request_finished(request, [0, 1])
        assert delay is True

    def test_request_finished_no_saved_tokens(self):
        scheduler, _ = self._make_scheduler()
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.config_data import RequestTracker
        scheduler._request_trackers["r1"] = RequestTracker("r1", 16, [0], num_saved_tokens=0)

        request = MagicMock()
        request.request_id = "r1"
        delay, extra = scheduler.request_finished(request, [0, 1])
        assert delay is False
