# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Tests for ascend_store_connector.py - covers AscendStoreKVEvents."""

from unittest.mock import MagicMock

from vllm.distributed.kv_events import BlockStored

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector import (
    AscendStoreKVEvents,
)


class TestAscendStoreKVEvents:
    def _make_event(self, hash_val="h1"):
        return BlockStored(
            block_hashes=[hash_val],
            parent_block_hash=None,
            token_ids=[1, 2],
            block_size=4,
            lora_id=None,
            medium="cpu",
            lora_name=None,
        )

    def test_creation(self):
        kv_events = AscendStoreKVEvents(num_workers=2)
        assert kv_events.get_number_of_workers() == 2
        assert len(kv_events.get_all_events()) == 0

    def test_add_events(self):
        kv_events = AscendStoreKVEvents(num_workers=1)
        e1 = self._make_event("h1")
        e2 = self._make_event("h2")
        kv_events.add_events([e1, e2])
        assert len(kv_events.get_all_events()) == 2

    def test_aggregate(self):
        kv_events = AscendStoreKVEvents(num_workers=2)
        e1 = self._make_event("h1")
        kv_events.add_events([e1])
        result = kv_events.aggregate()
        assert result is kv_events

    def test_increment_workers(self):
        kv_events = AscendStoreKVEvents(num_workers=1)
        kv_events.increment_workers(2)
        assert kv_events.get_number_of_workers() == 3

    def test_clear_events(self):
        kv_events = AscendStoreKVEvents(num_workers=1)
        e1 = self._make_event("h1")
        kv_events.add_events([e1])
        kv_events.clear_events()
        assert len(kv_events.get_all_events()) == 0

    def test_repr(self):
        kv_events = AscendStoreKVEvents(num_workers=1)
        r = repr(kv_events)
        assert "AscendStoreKVEvents" in r
