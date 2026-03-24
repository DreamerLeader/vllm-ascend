# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Tests for config_data.py - covers KeyMetadata, PoolKey, LayerPoolKey,
ChunkedTokenDatabase, LoadSpec, RequestTracker, ReqMeta,
AscendConnectorMetadata, LasyerMultiBlockReqMeta."""

from unittest.mock import MagicMock, patch

import pytest

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.config_data import (
    AscendConnectorMetadata,
    ChunkedTokenDatabase,
    KeyMetadata,
    LasyerMultiBlockReqMeta,
    LayerPoolKey,
    LoadSpec,
    PoolKey,
    ReqMeta,
    RequestTracker,
)


# ─── KeyMetadata ──────────────────────────────────────────────────────────────


class TestKeyMetadata:
    def test_creation(self):
        meta = KeyMetadata(
            model_name="test-model",
            head_or_tp_rank=0,
            pcp_rank=0,
            dcp_rank=0,
            pp_rank=0,
        )
        assert meta.model_name == "test-model"
        assert meta.head_or_tp_rank == 0
        assert meta.pcp_rank == 0
        assert meta.dcp_rank == 0
        assert meta.pp_rank == 0


# ─── PoolKey ──────────────────────────────────────────────────────────────────


class TestPoolKey:
    def setup_method(self):
        self.meta = KeyMetadata("model", 1, 0, 0, 0)

    def test_hash(self):
        key1 = PoolKey(self.meta, "abc123")
        key2 = PoolKey(self.meta, "abc123")
        assert hash(key1) == hash(key2)

    def test_hash_different_chunk(self):
        key1 = PoolKey(self.meta, "abc")
        key2 = PoolKey(self.meta, "def")
        assert hash(key1) != hash(key2)

    def test_to_string(self):
        key = PoolKey(self.meta, "hash123")
        s = key.to_string()
        assert "model" in s
        assert "@pcp0" in s
        assert "@dcp0" in s
        assert "@head_or_tp_rank:1" in s
        assert "@pp_rank:0" in s
        assert "hash123" in s

    def test_split_layers(self):
        key = PoolKey(self.meta, "hash123")
        layer_keys = key.split_layers(3)
        assert len(layer_keys) == 3
        for i, lk in enumerate(layer_keys):
            assert isinstance(lk, LayerPoolKey)
            assert lk.layer_id == i
            assert lk.chunk_hash == "hash123"

    def test_used_as_dict_key(self):
        key = PoolKey(self.meta, "abc")
        d = {key: "value"}
        key2 = PoolKey(self.meta, "abc")
        assert d[key2] == "value"


# ─── LayerPoolKey ─────────────────────────────────────────────────────────────


class TestLayerPoolKey:
    def setup_method(self):
        self.meta = KeyMetadata("model", 2, 1, 0, 0)

    def test_hash(self):
        lk1 = LayerPoolKey(self.meta, "hash", 0)
        lk2 = LayerPoolKey(self.meta, "hash", 0)
        assert hash(lk1) == hash(lk2)

    def test_hash_different_layer(self):
        lk1 = LayerPoolKey(self.meta, "hash", 0)
        lk2 = LayerPoolKey(self.meta, "hash", 1)
        assert hash(lk1) != hash(lk2)

    def test_to_string(self):
        lk = LayerPoolKey(self.meta, "hash", 5)
        s = lk.to_string()
        assert "@head_or_tp_rank:2" in s
        assert "@5" in s
        assert "hash" in s


# ─── ChunkedTokenDatabase ────────────────────────────────────────────────────


class TestChunkedTokenDatabase:
    def setup_method(self):
        self.meta = KeyMetadata("model", 0, 0, 0, 0)
        self.block_size = 4
        self.db = ChunkedTokenDatabase(self.meta, self.block_size, None)
        # Simulate 2 layers, each with K and V tensors
        # block_len per buffer entry = block_size * num_heads * head_dim * elem_size
        # For simplicity: block_len = 64 bytes
        self.db.set_block_len([64])
        # 2 layers * 2 (k+v) = 4 base addresses
        self.db.set_kv_caches_base_addr([1000, 2000, 3000, 4000])

    def test_set_kv_caches_base_addr(self):
        assert self.db.kv_caches_base_addr == [1000, 2000, 3000, 4000]

    def test_set_block_len(self):
        assert self.db.block_len == [64]

    def test_prepare_value(self):
        # block_ids: block 0 for tokens 0-3, block 1 for tokens 4-7
        block_ids = [10, 11]
        addr_list, size_list, block_id = self.db.prepare_value(0, 4, block_ids)
        assert block_id == 10
        assert len(addr_list) == 4  # 4 base addrs
        assert len(size_list) == 4
        # addr = base + block_id * block_len
        assert addr_list[0] == 1000 + 10 * 64
        assert addr_list[1] == 2000 + 10 * 64
        # size = block_len / block_size * (end - start) = 64/4*4 = 64
        assert size_list[0] == 64

    def test_prepare_value_partial_block(self):
        block_ids = [5, 6]
        addr_list, size_list, block_id = self.db.prepare_value(0, 2, block_ids)
        assert block_id == 5
        # size = 64/4*2 = 32
        assert size_list[0] == 32

    def test_prepare_value_layer(self):
        # block_len has 1 entry, so length=1
        # layer_id=1 means kv_caches_base_addr[1*1] = 2000
        block_ids = [10, 11]
        addr_list, size_list = self.db.prepare_value_layer(0, 4, block_ids, 1)
        assert len(addr_list) == 1
        assert addr_list[0] == 2000 + 10 * 64
        assert size_list[0] == 64

    def test_process_tokens_basic(self):
        # 8 tokens, block_size=4, so 2 blocks
        hashes = ["aabb", "ccdd"]
        results = list(self.db.process_tokens(8, hashes))
        assert len(results) == 2
        assert results[0][0] == 0  # start
        assert results[0][1] == 4  # end
        assert results[1][0] == 4
        assert results[1][1] == 8

    def test_process_tokens_with_mask(self):
        hashes = ["aabb", "ccdd"]
        # mask_num=4 means skip first block
        results = list(self.db.process_tokens(8, hashes, mask_num=4))
        assert len(results) == 1
        assert results[0][0] == 4

    def test_process_tokens_empty_hashes(self):
        results = list(self.db.process_tokens(8, []))
        assert len(results) == 0

    def test_process_tokens_with_block_hash_objects(self):
        # BlockHash objects have .hex() method
        mock_hash = MagicMock()
        mock_hash.hex.return_value = "aabb"
        results = list(self.db.process_tokens(4, [mock_hash]))
        assert len(results) == 1

    def test_process_tokens_token_len_less_than_block(self):
        hashes = ["aabb", "ccdd"]
        # token_len=3, block_size=4, so first block: start=0, end=3
        results = list(self.db.process_tokens(3, hashes))
        assert len(results) == 1
        assert results[0][0] == 0
        assert results[0][1] == 3

    def test_decode_adaptor_prefill_pp_no_partitions(self):
        key, addr, size = self.db.decode_adaptor_prefill_pp(
            ["k1"], [[100, 200]], [[10, 20]]
        )
        assert key == ["k1"]
        assert addr == [[100, 200]]

    def test_decode_adaptor_prefill_pp_with_partitions(self):
        db = ChunkedTokenDatabase(self.meta, self.block_size, [2, 2])
        db.set_block_len([64])
        db.set_kv_caches_base_addr([1000, 2000, 3000, 4000])

        # 1 block, 4 addr entries (2 layers * 2 kv)
        keys = ["key@pp_rank:0"]
        addrs = [[100, 200, 300, 400]]
        sizes = [[10, 20, 30, 40]]

        new_keys, new_addrs, new_sizes = db.decode_adaptor_prefill_pp(
            keys, addrs, sizes
        )
        # Should split into 2 PP ranks
        assert len(new_keys) == 2
        assert "@pp_rank:0" in new_keys[0]
        assert "@pp_rank:1" in new_keys[1]

    def test_decode_adaptor_prefill_pp_single_partition(self):
        db = ChunkedTokenDatabase(self.meta, self.block_size, [4])
        db.set_block_len([64])
        key, addr, size = db.decode_adaptor_prefill_pp(
            ["k1"], [[100, 200]], [[10, 20]]
        )
        assert key == ["k1"]

    def test_make_key_by_hash(self):
        key = self.db._make_key_by_hash("test_hash")
        assert isinstance(key, PoolKey)
        assert key.chunk_hash == "test_hash"


# ─── LoadSpec ─────────────────────────────────────────────────────────────────


class TestLoadSpec:
    def test_creation(self):
        spec = LoadSpec(
            vllm_cached_tokens=10,
            kvpool_cached_tokens=20,
            can_load=True,
        )
        assert spec.vllm_cached_tokens == 10
        assert spec.kvpool_cached_tokens == 20
        assert spec.can_load is True
        assert spec.token_len == 0

    def test_token_len_default(self):
        spec = LoadSpec(0, 0, False)
        assert spec.token_len == 0

    def test_token_len_set(self):
        spec = LoadSpec(0, 0, False, token_len=100)
        assert spec.token_len == 100


# ─── RequestTracker ───────────────────────────────────────────────────────────


class TestRequestTracker:
    def test_creation(self):
        tracker = RequestTracker(
            req_id="req-1",
            token_len=100,
            allocated_block_ids=[0, 1, 2],
        )
        assert tracker.req_id == "req-1"
        assert tracker.token_len == 100
        assert tracker.num_saved_tokens == 0

    def test_from_new_request_flat_block_ids(self):
        new_req = MagicMock()
        new_req.req_id = "req-1"
        new_req.block_ids = [0, 1, 2]
        new_req.prompt_token_ids = list(range(50))

        tracker = RequestTracker.from_new_request(new_req, 30)
        assert tracker.req_id == "req-1"
        assert tracker.token_len == 30
        assert tracker.allocated_block_ids == [0, 1, 2]
        assert len(tracker.token_ids) == 30

    def test_from_new_request_nested_block_ids(self):
        new_req = MagicMock()
        new_req.req_id = "req-2"
        new_req.block_ids = [[10, 11]]
        new_req.prompt_token_ids = list(range(20))

        tracker = RequestTracker.from_new_request(new_req, 15)
        assert tracker.allocated_block_ids == [10, 11]

    def test_update_with_list(self):
        tracker = RequestTracker("r1", 10, [0, 1])
        tracker.update([2, 3])
        assert tracker.allocated_block_ids == [0, 1, 2, 3]

    def test_update_with_tuple(self):
        tracker = RequestTracker("r1", 10, [0, 1])
        tracker.update(([2, 3],))
        assert tracker.allocated_block_ids == [0, 1, 2, 3]

    def test_update_with_empty(self):
        tracker = RequestTracker("r1", 10, [0, 1])
        tracker.update([])
        assert tracker.allocated_block_ids == [0, 1]

    def test_update_with_invalid_type(self):
        tracker = RequestTracker("r1", 10, [0])
        with pytest.raises(TypeError):
            tracker.update(42)


# ─── ReqMeta ─────────────────────────────────────────────────────────────────


class TestReqMeta:
    def test_from_request_tracker_basic_save(self):
        tracker = RequestTracker("r1", 16, [0, 1], num_saved_tokens=0)
        meta = ReqMeta.from_request_tracker(
            tracker,
            block_size=8,
            skip_save=False,
            block_hashes=["h1", "h2"],
        )
        assert meta is not None
        assert meta.req_id == "r1"
        assert meta.can_save is True
        assert meta.token_len_chunk == 16
        assert tracker.num_saved_tokens == 16

    def test_from_request_tracker_skip_save_and_no_load(self):
        tracker = RequestTracker("r1", 16, [0])
        meta = ReqMeta.from_request_tracker(
            tracker, block_size=8, skip_save=True, load_spec=None
        )
        assert meta is None

    def test_from_request_tracker_with_load_spec(self):
        tracker = RequestTracker("r1", 16, [0])
        load = LoadSpec(4, 12, can_load=True)
        meta = ReqMeta.from_request_tracker(
            tracker, block_size=8, skip_save=True, load_spec=load
        )
        assert meta is not None
        assert meta.load_spec is not None
        assert meta.load_spec.can_load is True

    def test_from_request_tracker_load_not_can_load(self):
        tracker = RequestTracker("r1", 16, [0])
        load = LoadSpec(4, 12, can_load=False)
        meta = ReqMeta.from_request_tracker(
            tracker, block_size=8, skip_save=True, load_spec=load
        )
        # load_spec.can_load is False, so load_spec is set to None in meta
        # But skip_save=True means can_save=False.
        # The method still returns a ReqMeta if skip_save was not initially True
        # (the skip_save logic interacts with chunk_boundary).
        # In this case: num_tokens_to_save=16, chunk_boundary=cdiv(1,8)*8=8, 16>=8 so skip_save stays True
        # But load_spec is provided so it returns a meta with load_spec=None
        assert meta is not None
        assert meta.load_spec is None
        assert meta.can_save is False

    def test_from_request_tracker_partial_chunk_discard(self):
        # 10 tokens, block_size=8 => num_tokens_to_save = 8
        tracker = RequestTracker("r1", 10, [0, 1], num_saved_tokens=0)
        meta = ReqMeta.from_request_tracker(
            tracker,
            block_size=8,
            skip_save=False,
            discard_partial_chunks=True,
        )
        assert meta is not None
        assert meta.token_len_chunk == 8
        assert tracker.num_saved_tokens == 8

    def test_from_request_tracker_no_discard_partial(self):
        tracker = RequestTracker("r1", 10, [0, 1], num_saved_tokens=0)
        meta = ReqMeta.from_request_tracker(
            tracker,
            block_size=8,
            skip_save=False,
            discard_partial_chunks=False,
        )
        assert meta is not None
        assert meta.token_len_chunk == 10

    def test_from_request_tracker_already_saved(self):
        # Already saved 16 tokens, trying to save again with same token_len
        tracker = RequestTracker("r1", 16, [0, 1], num_saved_tokens=16)
        meta = ReqMeta.from_request_tracker(
            tracker, block_size=8, skip_save=False
        )
        # chunk_boundary = cdiv(16+1, 8)*8 = 24
        # num_tokens_to_save = 16
        # 16 < 24, so skip_save = True
        assert meta is None

    def test_from_request_tracker_with_token_ids(self):
        tracker = RequestTracker("r1", 8, [0], token_ids=[1, 2, 3, 4, 5, 6, 7, 8])
        meta = ReqMeta.from_request_tracker(
            tracker, block_size=8, skip_save=False
        )
        assert meta is not None
        assert meta.token_ids == [1, 2, 3, 4, 5, 6, 7, 8]

    def test_from_request_tracker_with_original_block_size(self):
        tracker = RequestTracker("r1", 8, [0])
        meta = ReqMeta.from_request_tracker(
            tracker, block_size=8, skip_save=False, original_block_size=4
        )
        assert meta is not None
        assert meta.original_block_size == 4


# ─── AscendConnectorMetadata ─────────────────────────────────────────────────


class TestAscendConnectorMetadata:
    def test_creation_and_add_request(self):
        meta = AscendConnectorMetadata({"r1", "r2"}, {"r3"})
        assert meta.unfinished_request_ids == {"r1", "r2"}
        assert meta.preempted_req_ids == {"r3"}
        assert len(meta.requests) == 0

        req_meta = ReqMeta(
            req_id="r1",
            token_len_chunk=16,
            block_ids=[0],
            block_hashes=[],
        )
        meta.add_request(req_meta)
        assert len(meta.requests) == 1


# ─── LasyerMultiBlockReqMeta ─────────────────────────────────────────────────


class TestLasyerMultiBlockReqMeta:
    def test_creation(self):
        meta_k = KeyMetadata("m", 0, 0, 0, 0)
        keys = [LayerPoolKey(meta_k, "h1", 0), LayerPoolKey(meta_k, "h2", 0)]
        req = LasyerMultiBlockReqMeta(
            req_id="r1",
            keys=keys,
            starts=[0, 4],
            ends=[4, 8],
            block_ids=[10, 11],
            layer_id=0,
        )
        assert req.req_id == "r1"
        assert req.is_last_chunk is True
        assert req.current_event is None
        assert len(req.keys) == 2
