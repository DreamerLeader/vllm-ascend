from collections.abc import Iterable
from dataclasses import dataclass
from typing import Optional

import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.logger import logger
from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.core.sched.output import NewRequestData


# Parameters related to the key
@dataclass
class KeyMetadata:
    """name of the LLM model"""

    model_name: str
    """ worker id when running under a distributed setting """
    head_or_tp_rank: int
    """ Initialize the current prefill context model parallel rank """
    pcp_rank: int
    """ Initialize the current decode context model parallel rank """
    dcp_rank: int
    """ Initialize the current pipeline parallel rank """
    pp_rank: int


@dataclass(order=True)
class PoolKey:
    key_metadata: KeyMetadata
    chunk_hash: str

    def __hash__(self):
        return hash(
            (
                self.key_metadata.model_name,
                self.key_metadata.head_or_tp_rank,
                self.key_metadata.pcp_rank,
                self.key_metadata.dcp_rank,
                self.key_metadata.pp_rank,
                self.chunk_hash,
            )
        )

    def to_string(self):
        return (
            f"{self.key_metadata.model_name}"
            f"@pcp{self.key_metadata.pcp_rank}@dcp{self.key_metadata.dcp_rank}"
            f"@head_or_tp_rank:{self.key_metadata.head_or_tp_rank}"
            f"@pp_rank:{self.key_metadata.pp_rank}@{self.chunk_hash}"
        )

    def split_layers(self, num_layers: int) -> list["LayerPoolKey"]:
        """Split the key into multiple keys for each layer"""
        keys = []
        for layer_id in range(num_layers):
            keys.append(
                LayerPoolKey(
                    self.key_metadata,
                    self.chunk_hash,
                    layer_id,
                )
            )
        return keys


@dataclass(order=True)
class LayerPoolKey(PoolKey):
    """A key for the layer cache engine"""

    layer_id: int

    def __hash__(self):
        return hash(
            (
                self.key_metadata.model_name,
                self.key_metadata.head_or_tp_rank,
                self.key_metadata.pcp_rank,
                self.key_metadata.dcp_rank,
                self.chunk_hash,
                self.layer_id,
            )
        )

    def to_string(self):
        return (
            f"{self.key_metadata.model_name}"
            f"@pcp{self.key_metadata.pcp_rank}@dcp{self.key_metadata.dcp_rank}"
            f"@head_or_tp_rank:{self.key_metadata.head_or_tp_rank}@{self.chunk_hash}@{self.layer_id}"
        )


class ChunkedTokenDatabase:
    def __init__(self, metadata: KeyMetadata, block_size: int, partitions: list[int] | None):
        self.metadata = metadata
        self.block_size = block_size
        self.kv_caches_base_addr: list[int] = []
        self.block_len: list[int] = []
        self.partitions = partitions

    def _make_key_by_hash(self, chunk_hash: str, layer_id: int | None = None):
        assert self.metadata is not None
        return PoolKey(
            self.metadata,
            chunk_hash,
        )

    def set_kv_caches_base_addr(self, kv_caches_base_addr: list[int]):
        self.kv_caches_base_addr = kv_caches_base_addr

    def set_block_len(self, block_len: list[int]):
        self.block_len = block_len

    def set_per_head_params(self, num_kv_heads_per_tp: int, head_dim: int, elem_size: int):
        """Set parameters needed for per-head KV cache storage.

        Computes byte-level strides and per-head base offsets based on the
        row-major KV cache layout: [num_blocks, block_size, num_local_heads, head_dim].

        Strides (in bytes):
          - head_stride:  head_dim * elem_size             (offset to next head within same token)
          - token_stride: num_local_heads * head_stride     (offset to next token within same block)
          - block_stride: block_size * token_stride         (offset to next block, equals block_len)

        Args:
            num_kv_heads_per_tp: Number of KV heads per tensor parallel rank (num_local_heads).
            head_dim: Dimension of each attention head.
            elem_size: Element size in bytes (e.g. 2 for float16).
        """
        self.num_kv_heads_per_tp = num_kv_heads_per_tp
        self.head_dim = head_dim
        self.elem_size = elem_size

        # Byte strides for [num_blocks, block_size, num_local_heads, head_dim]
        self.head_stride = head_dim * elem_size
        self.token_stride = num_kv_heads_per_tp * self.head_stride
        self.block_stride = self.block_size * self.token_stride

        # Pre-compute base byte offset for each local head
        self.head_offsets = [h * self.head_stride for h in range(num_kv_heads_per_tp)]

    def prepare_value(self, start: int, end: int, block_ids: list[int]):
        addr_list = []
        size_list = []
        block_id = block_ids[start // self.block_size]
        length = len(self.block_len)
        for index, base_addr in enumerate(self.kv_caches_base_addr):
            addr = base_addr + block_id * self.block_len[index % length]
            size = int(self.block_len[index % length] / self.block_size * (end - start))
            addr_list.append(addr)
            size_list.append(size)
        return addr_list, size_list, block_id

    def _get_token_head_addr(self, base_addr: int, block_id: int, token_idx: int, head_idx: int) -> int:
        """Compute the physical address of a specific (block, token, head) position.

        For KV cache layout [num_blocks, block_size, num_local_heads, head_dim]
        stored in row-major order:
            addr = base + block_id * block_stride + token_idx * token_stride + head_offsets[head_idx]

        Args:
            base_addr: Base address of the KV cache buffer.
            block_id: Physical block index.
            token_idx: Token offset within the block (0 .. block_size-1).
            head_idx: Local head index within this TP rank.

        Returns:
            Physical byte address of the start of head_dim contiguous elements.
        """
        return (
            base_addr
            + block_id * self.block_stride
            + token_idx * self.token_stride
            + self.head_offsets[head_idx]
        )

    def prepare_value_per_head(
        self, start: int, end: int, block_ids: list[int]
    ) -> tuple[list[list[int]], list[list[int]], int]:
        """Prepare per-head addresses for ALL local heads in a block range.

        For non-MLA models with TP-unequal scenarios, computes the starting
        address of each head at each token position within a block.
        KV cache layout: [num_blocks, block_size, num_local_heads, head_dim]

        For each local head h, and each token position t in [start, end),
        the address is computed as:
            base + block_id * block_stride + t * token_stride + head_offsets[h]
        where each entry covers head_stride (= head_dim * elem_size) contiguous bytes.

        Args:
            start: Start token index (block-aligned).
            end: End token index.
            block_ids: Block ID mapping from logical block index to physical block ID.

        Returns:
            (all_head_addrs, all_head_sizes, block_id) where:
              - all_head_addrs[h]: list of addresses for local head h,
                ordered as [buf0_t0, buf0_t1, ..., buf1_t0, buf1_t1, ...]
                across all KV cache buffers (K and V per layer).
              - all_head_sizes[h]: list of sizes (each = head_stride) matching addrs.
              - block_id: the physical block ID used.
        """
        block_id = block_ids[start // self.block_size]
        num_tokens = end - start

        all_head_addrs: list[list[int]] = [[] for _ in range(self.num_kv_heads_per_tp)]
        all_head_sizes: list[list[int]] = [[] for _ in range(self.num_kv_heads_per_tp)]

        for base_addr in self.kv_caches_base_addr:
            for t in range(num_tokens):
                for h in range(self.num_kv_heads_per_tp):
                    addr = self._get_token_head_addr(base_addr, block_id, t, h)
                    all_head_addrs[h].append(addr)
                    all_head_sizes[h].append(self.head_stride)

        return all_head_addrs, all_head_sizes, block_id

    def prepare_value_layer(self, start: int, end: int, block_ids: list[int], layer_id: int):
        block_id = block_ids[start // self.block_size]
        addr_list = []
        size_list = []
        length = len(self.block_len)
        for i in range(length):
            addr = self.kv_caches_base_addr[layer_id * length] + block_id * self.block_len[i]
            size = int(self.block_len[i] / self.block_size * (end - start))
            addr_list.append(addr)
            size_list.append(size)
        return addr_list, size_list

    def process_tokens(
        self,
        token_len: int,
        block_hashes: list[BlockHash] | list[str],
        mask_num: int = 0,
    ) -> Iterable[tuple[int, int, PoolKey]]:
        """Process the tokens and return the corresponding cache engine keys.

        :param Union[torch.Tensor, List[int]] tokens: The tokens to process.

        :param Optional[torch.Tensor] mask: The mask for the tokens. Should
            have the same length as tokens. And the mask should ALWAYS be like
            FFFFFTTTTTTT, where True means the tokens needs to be matched,
            and the Falses will ALWAYS be at the PREFIX of the tensor.

        :param bool make_key: Whether to make the cache engine key or not.
            If False, the hash value will be returned instead.

        :returns: A iterable of tuples with three elements. The first element
            is the start index of the tokens for the key. The second element
            is the end index of the tokens for the key. The third element is
            the cache engine key (or hash) for the tokens.

        :raises: ValueError if the number of Falses in the mask is not a
            multiple of the chunk size.
        """
        if not block_hashes:
            return
        if not isinstance(block_hashes[0], str):
            block_hashes = [
                h.hex()  # type: ignore[union-attr]
                for h in block_hashes
            ]
        start_idx = 0
        for chunk_id, hash_val in enumerate(block_hashes):
            start_idx = chunk_id * self.block_size
            if start_idx >= token_len:
                break
            end_idx = min(start_idx + self.block_size, token_len)
            if start_idx < mask_num:
                continue
            else:
                yield start_idx, end_idx, self._make_key_by_hash(hash_val)

    def decode_adaptor_prefill_pp(self, key, addr, size):
        if self.partitions is None or len(self.partitions) == 1:
            return key, addr, size

        new_key = []
        new_addr = []
        new_size = []

        for i, (addr_list, size_list) in enumerate(zip(addr, size)):
            start = 0
            for j, part in enumerate(self.partitions):
                # part * 2 because addr and size contain both k and v
                end = len(addr_list) if j == len(self.partitions) - 1 else start + part * 2
                new_str = key[i].replace(  # type: ignore[attr-defined]
                    "@pp_rank:0", f"@pp_rank:{j}", 1
                )
                new_key.append(new_str)
                new_addr.append(addr_list[start:end])
                new_size.append(size_list[start:end])
                start = end
        return new_key, new_addr, new_size


# Parameters related to the connector metadata
@dataclass
class LoadSpec:
    # Number of tokens cached in vLLM
    vllm_cached_tokens: int
    # Number of tokens that are cached in kvpool
    kvpool_cached_tokens: int
    # Whether the scheduler allow us to load the tokens
    can_load: bool

    token_len: int = 0


@dataclass
class RequestTracker:
    # Request id
    req_id: str

    token_len: int

    # The block ids that has been allocated so far
    # NOTE: allocated blocks could be more than the number of tokens
    # FIXME: need to check whether the block ids will be changed after
    #        preemption
    allocated_block_ids: list[int]

    # The number of tokens that has been savd
    num_saved_tokens: int = 0

    # The token ids that has been scheduled so far
    # NOTE: This field will only be used when you enable kv-event
    token_ids: list[int] | None = None

    @staticmethod
    def from_new_request(
        new_request: "NewRequestData",
        num_tokens_to_compute: int,
    ) -> "RequestTracker":
        """Create the request tracker from a new request.

        Args:
            new_request (NewRequestData): the new request data.
            num_tokens_to_compute (int): the number of tokens that will
                be 'computed', including the `num_computed_tokens` (vLLM's
                local cache hit) and new tokens that will be scheduled.

        """
        unfolded_block_ids = []

        if not isinstance(new_request.block_ids[0], list):
            unfolded_block_ids = new_request.block_ids.copy()
        else:
            unfolded_block_ids = new_request.block_ids[0].copy()

        return RequestTracker(
            req_id=new_request.req_id,
            token_ids=new_request.prompt_token_ids[:num_tokens_to_compute].copy(),
            token_len=num_tokens_to_compute,
            allocated_block_ids=unfolded_block_ids,
            num_saved_tokens=0,
        )

    def update(
        self,
        new_block_ids: tuple[list[int], ...] | list[int],
    ) -> None:
        """Update the request tracker when a running request is
        scheduled again
        """
        if len(new_block_ids) == 0:
            new_block_ids = []
        elif isinstance(new_block_ids, tuple):
            new_block_ids = new_block_ids[0]
        elif isinstance(new_block_ids, list):
            pass
        else:
            raise ValueError(f"Unsupported new_block_ids type {type(new_block_ids)}")
        self.allocated_block_ids.extend(new_block_ids)


@dataclass
class ReqMeta:
    # Request id
    req_id: str
    # Number of tokens in this chunk
    token_len_chunk: int

    block_ids: list[int]

    block_hashes: list[BlockHash]

    can_save: bool | None = None
    # load_spec
    load_spec: LoadSpec | None = None

    is_last_chunk: bool | None = None

    current_event: torch.npu.Event | None = None

    # The following parameters are only used for kv event generation
    # TODO: add lora_request which used for gen lora_id/lora_name in kv event
    token_ids: list[int] | None = None
    original_block_size: int | None = None

    @staticmethod
    def from_request_tracker(
        tracker: RequestTracker,
        block_size: int,
        load_spec: LoadSpec | None = None,
        skip_save: bool | None = False,
        block_hashes: list[BlockHash] | None = None,
        is_last_chunk: bool | None = None,
        discard_partial_chunks: bool = True,
        original_block_size: int | None = None,
    ) -> Optional["ReqMeta"]:
        """Create the request metadata from a request tracker.

        Args:
            tracker (RequestTracker): the request tracker.
            block_size (int): the block size in vLLM scheduler and AscendConnector.
                If context parallelism is enabled, block_size = block_size * pcp_size * dcp_size.
            load_spec (Optional[LoadSpec]): the load spec for KV cache loading.
            skip_save (bool): whether to skip the save operation.
            discard_partial_chunks (bool): whether to discard partial chunks.
            original_block_size (int | None): the block size in vLLM worker. This is only used for kv event generation.

        Returns:
            the request metadata if we need to perform load/save
            operations, None otherwise.
        """
        if block_hashes is None:
            block_hashes = []
        input_token_len = tracker.token_len

        # For save operation: do not save if the following condition is met
        # 1. has already been saved before (num_saved_tokens > 0)
        # 2. number of unsaved tokens is not reached the chunk boundary
        chunk_boundary = cdiv(tracker.num_saved_tokens + 1, block_size) * block_size if discard_partial_chunks else 0
        # Calculate number of tokens to save based on discard_partial_chunks
        # setting
        num_tokens_to_save = (input_token_len // block_size * block_size) if discard_partial_chunks else input_token_len

        skip_save = skip_save or num_tokens_to_save < chunk_boundary
        if skip_save and load_spec is None:
            return None

        # If we need to save, update the number of saved tokens
        if not skip_save:
            tracker.num_saved_tokens = num_tokens_to_save

        # Get the token ids for kv event generation in kv_transfer
        token_ids = None
        if tracker.token_ids:
            token_ids = tracker.token_ids

        # # For load operation: check whether the request is scheduled to load
        if load_spec is not None and load_spec.can_load:
            logger.debug(
                "Scheduled to load %d tokens for request %s",
                load_spec.kvpool_cached_tokens,
                tracker.req_id,
            )
        else:
            # Do not load if not in `can_load` state
            load_spec = None
        logger.debug(f"request:{tracker.req_id}, meta save spec:{not skip_save}, meta load spec:{load_spec}")
        return ReqMeta(
            req_id=tracker.req_id,
            token_len_chunk=num_tokens_to_save,
            block_ids=tracker.allocated_block_ids,
            can_save=not skip_save,
            load_spec=load_spec,
            block_hashes=block_hashes,
            is_last_chunk=is_last_chunk,
            token_ids=token_ids,
            original_block_size=original_block_size,
        )


class AscendConnectorMetadata(KVConnectorMetadata):
    def __init__(self, unfinished_request_ids, preempted_req_ids):
        self.requests = []
        self.unfinished_request_ids = unfinished_request_ids
        self.preempted_req_ids = preempted_req_ids

    def add_request(self, req_meta: ReqMeta) -> None:
        """Add a request to the metadata.

        Args:
            req_meta (ReqMeta): the request metadata.
        """
        self.requests.append(req_meta)


@dataclass
class LasyerMultiBlockReqMeta:
    req_id: str
    keys: list[LayerPoolKey]
    starts: list[int]
    ends: list[int]
    block_ids: list[int]
    layer_id: int
    is_last_chunk: bool | None = True
    current_event: torch.npu.Event | None = None
