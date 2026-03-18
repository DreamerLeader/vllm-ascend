import queue
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import torch
from vllm.distributed.kv_events import BlockStored
from vllm.logger import logger
from vllm.v1.core.kv_cache_utils import maybe_convert_block_hash

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.backend import Backend

# isort: off
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.config_data import (
    ChunkedTokenDatabase,
    KeyMetadata,
    LasyerMultiBlockReqMeta,
    PoolKey,
    ReqMeta,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.tp_head_mapper import (
    TPHeadMapper,
)
# isort: on


class KVTransferThread(threading.Thread):
    def __init__(
        self,
        m_store: Backend,
        token_database: ChunkedTokenDatabase,
        block_size: int,
        tp_rank: int,
        dcp_size: int,
        ready_event: threading.Event,
        name: str,
    ):
        super().__init__(daemon=True, name=name)
        self.m_store = m_store
        self.ready_event = ready_event
        self.block_size = block_size
        self.tp_rank = tp_rank
        self.dcp_size = dcp_size
        self.token_database = token_database
        self.done_task_lock = threading.Lock()
        self.request_queue: queue.Queue[Any] = queue.Queue()
        # TODO(jianzs): make this configurable
        self.executor = ThreadPoolExecutor(max_workers=32)
        self.finished_requests: set[str] = set()
        self.kv_event_lock = threading.Lock()
        self.kv_events: list[BlockStored] = []

    def add_request(
        self,
        request: ReqMeta | LasyerMultiBlockReqMeta,
    ) -> torch.Tensor:
        self.request_queue.put(request)

    def get_and_clear_finished_requests(self) -> set[str]:
        """
        Get and clear the requests that have been completed.
        Returns:
            A set of request IDs that have been completed.
        """
        with self.done_task_lock:
            finished_requests = self.finished_requests.copy()
            self.finished_requests.clear()
        return finished_requests

    def set_finished_request(self, req_id):
        with self.done_task_lock:
            self.finished_requests.add(req_id)

    def run(self):
        """Run the thread to handle KV cache transfer requests."""
        self.m_store.set_device()
        self.ready_event.set()
        while True:
            try:
                request_data = self.request_queue.get()
                if request_data is None:
                    logger.warning("Received a None request!")
                    self.request_queue.task_done()
                    continue
                self._handle_request(request_data)
            except Exception as e:
                logger.error(f"Error in KVCacheTransferThread: {e}")

    def _handle_request(self, req_meta: Any):
        pass

    def lookup(
        self,
        keys: list[str],
    ) -> int:
        """
        Checks the existence of KV cache of the tokens from the cache engine.
        :param tokens: the input tokens, with shape [seq_len]
        :return: An int indicating how many prefix tokens are cached.
        """
        try:
            res = self.m_store.exists(keys)  # type: ignore[assignment]
            for index, value in enumerate(res):  # type: ignore[arg-type]
                if value != 1:
                    return index
            # all tokens where found, return the maximal end
        except Exception as e:
            logger.error(f"Remote connection failed in contains: {e}")
            return 0
        return len(keys)

    def update_kv_event(self, event: list[BlockStored]):
        with self.kv_event_lock:
            self.kv_events.extend(event)

    def get_kv_events(self) -> list[BlockStored]:
        with self.kv_event_lock:
            events = self.kv_events.copy()
            self.kv_events.clear()
        return events


class KVCacheStoreSendingThread(KVTransferThread):
    def __init__(
        self,
        m_store: Backend,
        token_database: ChunkedTokenDatabase,
        block_size: int,
        tp_rank: int,
        dcp_size: int,
        put_step: int,
        kv_role: str,
        ready_event: threading.Event,
        enable_kv_event: bool = False,
    ):
        super().__init__(
            m_store, token_database, block_size, tp_rank, dcp_size, ready_event, name="KVCacheSendingThread"
        )
        self.put_step = put_step
        self.kv_role = kv_role
        self.stored_requests = defaultdict[str, int](int)
        self.enable_kv_event = enable_kv_event

    def add_stored_request(self, req_id: str):
        with self.done_task_lock:
            self.stored_requests[req_id] += 1

    def dec_stored_request(self, req_id: str):
        with self.done_task_lock:
            if req_id in self.stored_requests:
                self.stored_requests[req_id] -= 1

    def delete_finished_stored_request(self, req_id: str):
        with self.done_task_lock:
            if req_id in self.stored_requests:
                del self.stored_requests[req_id]

    def _handle_request(self, req_meta: ReqMeta):
        token_len = req_meta.token_len_chunk
        block_ids = req_meta.block_ids
        req_id = req_meta.req_id
        current_event = req_meta.current_event
        starts = []
        ends = []
        keys = []
        if req_id not in self.stored_requests:
            self.request_queue.task_done()
            return

        for start, end, key in self.token_database.process_tokens(token_len, req_meta.block_hashes):
            starts.append(start)
            ends.append(end)
            keys.append(key.to_string())

        if not self.dcp_size > 1:
            starts = starts[self.tp_rank % self.put_step :: self.put_step]
            ends = ends[self.tp_rank % self.put_step :: self.put_step]
            keys = keys[self.tp_rank % self.put_step :: self.put_step]

        if not keys:
            self.dec_stored_request(req_id)
            return

        skip_block_num = self.lookup(keys)

        if skip_block_num == len(keys):
            self.dec_stored_request(req_id)
            return

        starts = starts[skip_block_num:]
        ends = ends[skip_block_num:]
        keys = keys[skip_block_num:]

        logger.debug(
            "Storing KV cache for %d out of %d blocks (skip_block_num=%d) for request %s",
            len(keys),
            token_len // self.block_size,
            skip_block_num,
            req_id,
        )

        if keys:
            """
            Note: Due to a bug in ADXL, calling current_event.synchronize() may occasionally hang.
            This issue will be fixed in CANN version 8.5.rc1.
            You can manually build the master branch of the project at https://gitcode.com/cann/hixl
            to resolve this issue before the 8.5.RC1 release.
            """
            addrs = []
            sizes = []
            stored_events: list[BlockStored] = []
            prev_key = None
            new_block_hashes = [maybe_convert_block_hash(bh) for bh in req_meta.block_hashes[skip_block_num:]]
            for index, start in enumerate(starts):
                addr, size, _ = self.token_database.prepare_value(start, ends[index], block_ids)
                addrs.append(addr)
                sizes.append(size)

                # Create KV event
                if self.enable_kv_event:
                    token_ids = req_meta.token_ids[start : ends[index]] if req_meta.token_ids is not None else None
                    stored_event = BlockStored(
                        block_hashes=[new_block_hashes[index]],
                        parent_block_hash=prev_key,
                        token_ids=token_ids,
                        block_size=req_meta.original_block_size,
                        lora_id=None,
                        medium="cpu",
                        lora_name=None,
                    )
                    stored_events.append(stored_event)
                    prev_key = new_block_hashes[index]
                    logger.debug(f"Added kv cache event '{stored_event}' to kv cache events queue")

            if self.kv_role == "kv_consumer":
                keys, addrs, sizes = self.token_database.decode_adaptor_prefill_pp(keys, addrs, sizes)

            if current_event is not None:
                current_event.synchronize()
            self.m_store.put(keys, addrs, sizes)

            # TODO Query specific replica info to update the event
            if self.enable_kv_event and stored_events is not None:
                self.update_kv_event(stored_events)

        self.dec_stored_request(req_id)
        self.request_queue.task_done()


class KVCacheStoreRecvingThread(KVTransferThread):
    def __init__(
        self,
        m_store: Backend,
        token_database: ChunkedTokenDatabase,
        block_size: int,
        tp_rank: int,
        dcp_size: int,
        ready_event: threading.Event,
    ):
        super().__init__(
            m_store, token_database, block_size, tp_rank, dcp_size, ready_event, name="KVCacheStoreRecvingThread"
        )

    def _handle_request(self, req_meta: ReqMeta):
        token_len = req_meta.load_spec.token_len  # type: ignore[union-attr]
        req_id = req_meta.req_id
        mask_num = (
            req_meta.load_spec.vllm_cached_tokens  # type: ignore[union-attr]
            // self.block_size
            * self.block_size
        )
        addr_list = []
        size_list = []
        key_list = []
        for start, end, key in self.token_database.process_tokens(token_len, req_meta.block_hashes, mask_num):
            addr, size, _ = self.token_database.prepare_value(start, end, req_meta.block_ids)
            key_list.append(key.to_string())
            addr_list.append(addr)
            size_list.append(size)
        key_list_c = key_list[self.tp_rank % len(key_list) :] + key_list[: self.tp_rank % len(key_list)]
        addr_list_c = addr_list[self.tp_rank % len(addr_list) :] + addr_list[: self.tp_rank % len(addr_list)]
        size_list_c = size_list[self.tp_rank % len(size_list) :] + size_list[: self.tp_rank % len(size_list)]
        self.m_store.get(key_list_c, addr_list_c, size_list_c)
        self.set_finished_request(req_id)
        self.request_queue.task_done()


class KVCacheStoreLayerSendingThread(KVTransferThread):
    def __init__(
        self,
        m_store: Backend,
        token_database: ChunkedTokenDatabase,
        block_size: int,
        tp_rank: int,
        dcp_size: int,
        put_step: int,
        ready_event: threading.Event,
        num_layers: int,
        enable_kv_event: bool = False,
    ):
        super().__init__(
            m_store, token_database, block_size, tp_rank, dcp_size, ready_event, name="KVCacheStoreLayerSendingThread"
        )
        self.final_layer_id = num_layers - 1
        self.put_step = put_step
        self.enable_kv_event = enable_kv_event

    def add_request(  # type: ignore[override]
        self, req_meta: ReqMeta
    ) -> torch.Tensor:
        self.request_queue.put(req_meta)

    def _handle_request(  # type: ignore[override]
        self, req_meta: LasyerMultiBlockReqMeta
    ):
        starts = req_meta.starts
        ends = req_meta.ends
        keys = req_meta.keys
        layer_id = req_meta.layer_id
        current_event = req_meta.current_event
        total_block = len(keys)
        is_last_chunk = req_meta.is_last_chunk
        if not self.dcp_size > 1:
            starts = starts[self.tp_rank % self.put_step :: self.put_step]
            ends = ends[self.tp_rank % self.put_step :: self.put_step]
            keys = keys[self.tp_rank % self.put_step :: self.put_step]

        if not keys:
            if is_last_chunk:
                self.set_finished_request(req_meta.req_id)
            return

        key_list = []
        for key in keys:
            key_list.append(key.to_string())

        skip_block_num = self.lookup(key_list)

        if skip_block_num == len(key_list):
            if is_last_chunk and layer_id == self.final_layer_id:
                self.set_finished_request(req_meta.req_id)
            return

        starts = starts[skip_block_num:]
        ends = ends[skip_block_num:]
        key_list = key_list[skip_block_num:]

        addr_list = []
        size_list = []
        for index, key in enumerate(key_list):
            addr, size = self.token_database.prepare_value_layer(
                starts[index], ends[index], req_meta.block_ids, layer_id
            )
            addr_list.append(addr)
            size_list.append(size)

        if current_event is not None:
            current_event.synchronize()
        self.m_store.put(key_list, addr_list, size_list)

        if layer_id == self.final_layer_id and is_last_chunk:
            self.set_finished_request(req_meta.req_id)
        self.request_queue.task_done()

        logger.info(
            "Storing KV cache for %d out of %d blocks (skip_block_num=%d) for request %s",
            len(keys),
            total_block,
            skip_block_num,
            req_meta.req_id,
        )


class KVCacheStoreLayerRecvingThread(KVTransferThread):
    def __init__(
        self,
        m_store: Backend,
        token_database: ChunkedTokenDatabase,
        block_size: int,
        tp_rank: int,
        dcp_size: int,
        ready_event: threading.Event,
        get_event: threading.Event,
    ):
        super().__init__(
            m_store, token_database, block_size, tp_rank, dcp_size, ready_event, name="KVCacheStoreLayerRecvingThread"
        )
        self.get_event = get_event

    def add_request(  # type: ignore[override]
        self, req_meta: LasyerMultiBlockReqMeta
    ) -> torch.Tensor:
        self.request_queue.put(req_meta)

    def _handle_request(  # type: ignore[override]
        self, req_meta: LasyerMultiBlockReqMeta
    ):
        addr_list = []
        size_list = []
        key_list = []
        for index, key in enumerate(req_meta.keys):
            addr, size = self.token_database.prepare_value_layer(
                req_meta.starts[index], req_meta.ends[index], req_meta.block_ids, req_meta.layer_id
            )
            key_list.append(key.to_string())
            addr_list.append(addr)
            size_list.append(size)
        key_list_c = key_list[self.tp_rank % len(key_list) :] + key_list[: self.tp_rank % len(key_list)]
        addr_list_c = addr_list[self.tp_rank % len(addr_list) :] + addr_list[: self.tp_rank % len(addr_list)]
        size_list_c = size_list[self.tp_rank % len(size_list) :] + size_list[: self.tp_rank % len(size_list)]
        self.m_store.get(key_list_c, addr_list_c, size_list_c)

        self.request_queue.task_done()
        self.get_event.set()


# ======================================================================
# Cross-TP KV Cache Transfer Threads (GQA + TP Asymmetry)
# ======================================================================


class CrossTPKVCacheSendingThread(KVTransferThread):
    """
    Send thread for cross-TP scenario (Decode side saves KV Cache).

    When prefill_tp != decode_tp, this thread saves KV cache blocks
    using staging buffers so that the data is stored in a canonical
    per-producer-rank format. The consumer side uses TPHeadMapper to
    determine which producer ranks to fetch from and how to remap heads.

    Key Design:
        - The producer (Decode) stores blocks as-is, keyed by its own
          head_or_tp_rank. The stored block contains all local KV heads
          belonging to this decode rank.
        - The consumer (Prefill) is responsible for head remapping on load.
        - This thread adds detailed logging for cross-TP debugging.
    """

    def __init__(
        self,
        m_store: Backend,
        token_database: ChunkedTokenDatabase,
        block_size: int,
        tp_rank: int,
        dcp_size: int,
        put_step: int,
        kv_role: str,
        ready_event: threading.Event,
        mapper: TPHeadMapper,
        kv_caches: dict[str, torch.Tensor] | None = None,
        enable_kv_event: bool = False,
    ):
        super().__init__(
            m_store,
            token_database,
            block_size,
            tp_rank,
            dcp_size,
            ready_event,
            name="CrossTPKVCacheSendingThread",
        )
        self.put_step = put_step
        self.kv_role = kv_role
        self.stored_requests = defaultdict[str, int](int)
        self.enable_kv_event = enable_kv_event
        self.mapper = mapper
        self.kv_caches = kv_caches

        logger.info(
            "[CrossTPSend] Initialized: tp_rank=%d, put_step=%d, "
            "decode_tp=%d, prefill_tp=%d, total_kv_heads=%d",
            tp_rank,
            put_step,
            mapper.decode_tp_size,
            mapper.prefill_tp_size,
            mapper.total_kv_heads,
        )

    def add_stored_request(self, req_id: str):
        with self.done_task_lock:
            self.stored_requests[req_id] += 1

    def dec_stored_request(self, req_id: str):
        with self.done_task_lock:
            if req_id in self.stored_requests:
                self.stored_requests[req_id] -= 1

    def delete_finished_stored_request(self, req_id: str):
        with self.done_task_lock:
            if req_id in self.stored_requests:
                del self.stored_requests[req_id]

    def _handle_request(self, req_meta: ReqMeta):
        token_len = req_meta.token_len_chunk
        block_ids = req_meta.block_ids
        req_id = req_meta.req_id
        current_event = req_meta.current_event

        if req_id not in self.stored_requests:
            self.request_queue.task_done()
            return

        starts = []
        ends = []
        keys = []
        for start, end, key in self.token_database.process_tokens(
            token_len, req_meta.block_hashes
        ):
            starts.append(start)
            ends.append(end)
            keys.append(key.to_string())

        # Apply put_step striding for GQA deduplication (same as base class)
        if not self.dcp_size > 1:
            starts = starts[self.tp_rank % self.put_step :: self.put_step]
            ends = ends[self.tp_rank % self.put_step :: self.put_step]
            keys = keys[self.tp_rank % self.put_step :: self.put_step]

        if not keys:
            self.dec_stored_request(req_id)
            return

        skip_block_num = self.lookup(keys)
        if skip_block_num == len(keys):
            self.dec_stored_request(req_id)
            return

        starts = starts[skip_block_num:]
        ends = ends[skip_block_num:]
        keys = keys[skip_block_num:]

        logger.info(
            "[CrossTPSend] Storing KV for req=%s: %d blocks "
            "(skip=%d), tp_rank=%d, decode_tp=%d, prefill_tp=%d, "
            "kv_heads=%d",
            req_id,
            len(keys),
            skip_block_num,
            self.tp_rank,
            self.mapper.decode_tp_size,
            self.mapper.prefill_tp_size,
            self.mapper.total_kv_heads,
        )

        if keys:
            addrs = []
            sizes = []
            for index, start in enumerate(starts):
                addr, size, _ = self.token_database.prepare_value(
                    start, ends[index], block_ids
                )
                addrs.append(addr)
                sizes.append(size)

            if current_event is not None:
                current_event.synchronize()

            self.m_store.put(keys, addrs, sizes)

            logger.debug(
                "[CrossTPSend] PUT completed for req=%s: "
                "num_blocks=%d, first_key=%s",
                req_id,
                len(keys),
                keys[0] if keys else "N/A",
            )

        self.dec_stored_request(req_id)
        self.request_queue.task_done()


class CrossTPKVCacheRecvingThread(KVTransferThread):
    """
    Receive thread for cross-TP scenario (Prefill side loads KV Cache).

    When prefill_tp != decode_tp, this thread:
    1. Uses TPHeadMapper to determine which producer (Decode) head_or_tp_ranks
       contain the needed KV heads for this consumer (Prefill) rank.
    2. Generates keys using the producer's head_or_tp_rank.
    3. Fetches the producer's blocks into staging buffers.
    4. Performs head-level slicing/remapping from staging to the consumer's
       actual KV cache blocks.

    This handles the core challenge of GQA + TP asymmetry where:
    - Producer block shape: [block_size, decode_local_kv_heads, head_dim]
    - Consumer block shape: [block_size, prefill_local_kv_heads, head_dim]
    - decode_local_kv_heads != prefill_local_kv_heads
    """

    def __init__(
        self,
        m_store: Backend,
        token_database: ChunkedTokenDatabase,
        block_size: int,
        tp_rank: int,
        dcp_size: int,
        ready_event: threading.Event,
        mapper: TPHeadMapper,
        kv_caches: dict[str, torch.Tensor] | None = None,
        staging_buffers: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
    ):
        super().__init__(
            m_store,
            token_database,
            block_size,
            tp_rank,
            dcp_size,
            ready_event,
            name="CrossTPKVCacheRecvingThread",
        )
        self.mapper = mapper
        self.kv_caches = kv_caches
        self.staging_buffers = staging_buffers

        logger.info(
            "[CrossTPRecv] Initialized: tp_rank=%d, prefill_tp=%d, "
            "decode_tp=%d, total_kv_heads=%d, "
            "decode_heads_per_rank=%d, prefill_heads_per_rank=%d, "
            "has_staging=%s",
            tp_rank,
            mapper.prefill_tp_size,
            mapper.decode_tp_size,
            mapper.total_kv_heads,
            mapper.decode_heads_per_rank,
            mapper.prefill_heads_per_rank,
            staging_buffers is not None,
        )

    def _handle_request(self, req_meta: ReqMeta):
        """
        Handle a cross-TP KV cache load request.

        Strategy:
        ---------
        Case A: Producer has MORE heads per rank than consumer
            (decode_heads_per_rank > prefill_heads_per_rank)
            e.g., D=2, P=8, H=4: decode has 2 heads/rank, prefill has 1
            => Fetch producer block into staging, slice out needed head(s),
               copy to consumer's KV cache.

        Case B: Producer has FEWER heads per rank than consumer
            (decode_heads_per_rank < prefill_heads_per_rank)
            e.g., D=8, P=2, H=4: decode has 1 head/rank, prefill has 2
            => Fetch multiple producer blocks (one per needed head),
               gather into consumer's KV cache.

        Case C: Same heads per rank (but TP sizes still differ due to replication)
            => Adjust key mapping, direct fetch.
        """
        token_len = req_meta.load_spec.token_len  # type: ignore[union-attr]
        req_id = req_meta.req_id
        mask_num = (
            req_meta.load_spec.vllm_cached_tokens  # type: ignore[union-attr]
            // self.block_size
            * self.block_size
        )

        mapping = self.mapper.get_mapping(self.tp_rank)
        unique_producer_ranks = self.mapper.get_unique_producer_ranks(self.tp_rank)

        logger.info(
            "[CrossTPRecv] Loading KV for req=%s: token_len=%d, mask_num=%d, "
            "tp_rank=%d, needed_producer_ranks=%s, global_heads=%s",
            req_id,
            token_len,
            mask_num,
            self.tp_rank,
            unique_producer_ranks,
            mapping.global_head_indices,
        )

        decode_heads_per_rank = self.mapper.decode_heads_per_rank
        prefill_heads_per_rank = self.mapper.prefill_heads_per_rank

        if decode_heads_per_rank == prefill_heads_per_rank:
            # Same number of heads per rank, just remap keys
            self._handle_same_heads_per_rank(
                req_meta, token_len, mask_num, mapping
            )
        elif decode_heads_per_rank > prefill_heads_per_rank:
            # Producer has more heads -> need to slice
            self._handle_producer_has_more_heads(
                req_meta, token_len, mask_num, mapping, unique_producer_ranks
            )
        else:
            # Producer has fewer heads -> need to gather
            self._handle_producer_has_fewer_heads(
                req_meta, token_len, mask_num, mapping, unique_producer_ranks
            )

        self.set_finished_request(req_id)
        self.request_queue.task_done()

    def _handle_same_heads_per_rank(
        self, req_meta, token_len, mask_num, mapping
    ):
        """
        When heads_per_rank is the same but head_or_tp_rank differs.
        Just remap the key to use the producer's head_or_tp_rank.
        """
        producer_head_or_tp = mapping.producer_head_or_tp_ranks[0]

        addr_list = []
        size_list = []
        key_list = []

        for start, end, key in self.token_database.process_tokens(
            token_len, req_meta.block_hashes, mask_num
        ):
            remote_key = self.token_database.make_key_with_remote_head(
                key.chunk_hash, producer_head_or_tp
            )
            addr, size, _ = self.token_database.prepare_value(
                start, end, req_meta.block_ids
            )
            key_list.append(remote_key.to_string())
            addr_list.append(addr)
            size_list.append(size)

        if not key_list:
            logger.debug(
                "[CrossTPRecv] No blocks to fetch for req=%s (same heads)",
                req_meta.req_id,
            )
            return

        # Rotate for load balancing
        offset = self.tp_rank % len(key_list)
        key_list = key_list[offset:] + key_list[:offset]
        addr_list = addr_list[offset:] + addr_list[:offset]
        size_list = size_list[offset:] + size_list[:offset]

        logger.debug(
            "[CrossTPRecv] Same-heads fetch: req=%s, blocks=%d, "
            "producer_head_or_tp=%d, first_key=%s",
            req_meta.req_id,
            len(key_list),
            producer_head_or_tp,
            key_list[0],
        )

        self.m_store.get(key_list, addr_list, size_list)

    def _handle_producer_has_more_heads(
        self, req_meta, token_len, mask_num, mapping, unique_producer_ranks
    ):
        """
        Producer (Decode) has more local heads per rank than consumer (Prefill).
        e.g., Decode has 2 heads/rank, Prefill has 1.

        Strategy:
        1. Fetch producer's full block into staging buffer.
        2. Slice out the needed head(s).
        3. Copy sliced data to consumer's KV cache.
        """
        if self.staging_buffers is None or self.kv_caches is None:
            logger.error(
                "[CrossTPRecv] Staging buffers or kv_caches not set for "
                "cross-TP recv with producer_heads > consumer_heads!"
            )
            raise RuntimeError(
                "Staging buffers required for cross-TP with "
                "decode_heads_per_rank > prefill_heads_per_rank"
            )

        # For each chunk block
        for start, end, key in self.token_database.process_tokens(
            token_len, req_meta.block_hashes, mask_num
        ):
            num_tokens_in_block = end - start
            consumer_block_id = req_meta.block_ids[start // self.block_size]

            # For each unique producer rank we need data from
            for prod_rank in unique_producer_ranks:
                # Get extraction info for this producer rank
                head_extractions = self.mapper.get_heads_from_producer(
                    self.tp_rank, prod_rank
                )

                if not head_extractions:
                    continue

                # Generate the producer's key
                remote_key = self.token_database.make_key_with_remote_head(
                    key.chunk_hash, prod_rank
                )

                logger.debug(
                    "[CrossTPRecv] Fetching from producer_rank=%d for block "
                    "[%d:%d], key=%s, head_extractions=%s",
                    prod_rank,
                    start,
                    end,
                    remote_key.to_string(),
                    head_extractions,
                )

                # Fetch into staging buffer
                staging_addrs = []
                staging_sizes = []
                for layer_name, (staging_k, staging_v) in self.staging_buffers.items():
                    # Staging buffer shape: [block_size, decode_heads_per_rank, head_dim]
                    staging_addrs.append(staging_k.data_ptr())
                    staging_sizes.append(
                        num_tokens_in_block
                        * self.mapper.decode_heads_per_rank
                        * self.token_database.head_dim
                        * self.token_database.element_size
                    )
                    staging_addrs.append(staging_v.data_ptr())
                    staging_sizes.append(
                        num_tokens_in_block
                        * self.mapper.decode_heads_per_rank
                        * self.token_database.head_dim
                        * self.token_database.element_size
                    )

                self.m_store.get(
                    [remote_key.to_string()],
                    [staging_addrs],
                    [staging_sizes],
                )

                # Now slice and copy heads from staging to consumer's KV cache
                for (
                    global_head,
                    prod_local_idx,
                    consumer_local_idx,
                ) in head_extractions:
                    for layer_name, (staging_k, staging_v) in self.staging_buffers.items():
                        # Extract per-layer KV caches
                        kv_cache_tuple = self.kv_caches[layer_name]
                        consumer_k = kv_cache_tuple[0]  # [num_blocks, block_size, num_local_heads, head_dim]
                        consumer_v = kv_cache_tuple[1]

                        # Slice from staging: [block_size, decode_heads, head_dim]
                        # -> [block_size, 1, head_dim] for the target head
                        src_k_head = staging_k[
                            :num_tokens_in_block, prod_local_idx : prod_local_idx + 1, :
                        ]
                        src_v_head = staging_v[
                            :num_tokens_in_block, prod_local_idx : prod_local_idx + 1, :
                        ]

                        # Copy to consumer KV cache
                        consumer_k[
                            consumer_block_id,
                            :num_tokens_in_block,
                            consumer_local_idx : consumer_local_idx + 1,
                            :,
                        ] = src_k_head
                        consumer_v[
                            consumer_block_id,
                            :num_tokens_in_block,
                            consumer_local_idx : consumer_local_idx + 1,
                            :,
                        ] = src_v_head

                        logger.debug(
                            "[CrossTPRecv] Copied head: global_head=%d, "
                            "prod_local=%d -> consumer_local=%d, "
                            "block_id=%d, layer=%s, tokens=%d",
                            global_head,
                            prod_local_idx,
                            consumer_local_idx,
                            consumer_block_id,
                            layer_name,
                            num_tokens_in_block,
                        )

    def _handle_producer_has_fewer_heads(
        self, req_meta, token_len, mask_num, mapping, unique_producer_ranks
    ):
        """
        Producer (Decode) has fewer local heads per rank than consumer (Prefill).
        e.g., Decode has 1 head/rank, Prefill has 2.

        Strategy:
        Fetch from multiple producer ranks and assemble into consumer's block.
        Since producer blocks have fewer heads, they might directly fit into
        the consumer's per-head slot.
        """
        if self.kv_caches is None:
            logger.error(
                "[CrossTPRecv] kv_caches not set for cross-TP recv!"
            )
            raise RuntimeError("kv_caches required for cross-TP recv")

        for start, end, key in self.token_database.process_tokens(
            token_len, req_meta.block_hashes, mask_num
        ):
            num_tokens_in_block = end - start
            consumer_block_id = req_meta.block_ids[start // self.block_size]

            # Fetch from each producer rank that has heads we need
            for prod_rank in unique_producer_ranks:
                head_extractions = self.mapper.get_heads_from_producer(
                    self.tp_rank, prod_rank
                )

                if not head_extractions:
                    continue

                remote_key = self.token_database.make_key_with_remote_head(
                    key.chunk_hash, prod_rank
                )

                logger.debug(
                    "[CrossTPRecv] Gathering from producer_rank=%d for "
                    "block [%d:%d], heads=%s",
                    prod_rank,
                    start,
                    end,
                    head_extractions,
                )

                if self.staging_buffers is not None:
                    # Use staging buffer for fetch + slice
                    staging_addrs = []
                    staging_sizes = []
                    for layer_name, (staging_k, staging_v) in self.staging_buffers.items():
                        staging_addrs.append(staging_k.data_ptr())
                        per_block_bytes = (
                            num_tokens_in_block
                            * self.mapper.decode_heads_per_rank
                            * self.token_database.head_dim
                            * self.token_database.element_size
                        )
                        staging_sizes.append(per_block_bytes)
                        staging_addrs.append(staging_v.data_ptr())
                        staging_sizes.append(per_block_bytes)

                    self.m_store.get(
                        [remote_key.to_string()],
                        [staging_addrs],
                        [staging_sizes],
                    )

                    # Copy each head from staging to consumer KV cache
                    for (
                        global_head,
                        prod_local_idx,
                        consumer_local_idx,
                    ) in head_extractions:
                        for layer_name, (staging_k, staging_v) in self.staging_buffers.items():
                            kv_cache_tuple = self.kv_caches[layer_name]
                            consumer_k = kv_cache_tuple[0]
                            consumer_v = kv_cache_tuple[1]

                            src_k = staging_k[
                                :num_tokens_in_block,
                                prod_local_idx : prod_local_idx + 1,
                                :,
                            ]
                            src_v = staging_v[
                                :num_tokens_in_block,
                                prod_local_idx : prod_local_idx + 1,
                                :,
                            ]

                            consumer_k[
                                consumer_block_id,
                                :num_tokens_in_block,
                                consumer_local_idx : consumer_local_idx + 1,
                                :,
                            ] = src_k
                            consumer_v[
                                consumer_block_id,
                                :num_tokens_in_block,
                                consumer_local_idx : consumer_local_idx + 1,
                                :,
                            ] = src_v

                            logger.debug(
                                "[CrossTPRecv] Gathered head: global=%d, "
                                "prod_local=%d -> consumer_local=%d, "
                                "block=%d, layer=%s",
                                global_head,
                                prod_local_idx,
                                consumer_local_idx,
                                consumer_block_id,
                                layer_name,
                            )
                else:
                    # No staging: producer and consumer have same per-head size.
                    # Direct fetch is possible only if heads_per_rank matches.
                    # This path is a fallback and may not produce correct results
                    # if actual shapes differ.
                    logger.warning(
                        "[CrossTPRecv] No staging buffers available. "
                        "Attempting direct fetch (may be incorrect if "
                        "block shapes differ). req=%s, prod_rank=%d",
                        req_meta.req_id,
                        prod_rank,
                    )
                    addr, size, _ = self.token_database.prepare_value(
                        start, end, req_meta.block_ids
                    )
                    self.m_store.get(
                        [remote_key.to_string()], [addr], [size]
                    )
