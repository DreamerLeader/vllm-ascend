"""
GQA + Asymmetric TP Head Mapper for Cross-TP KV Cache Transfer.

When Prefill_TP != Decode_TP in GQA models, the distribution of KV heads
across TP ranks differs. This module computes the exact mapping between
producer (Decode) and consumer (Prefill) head layouts.

Mathematical Model:
-------------------
Given:
    H = total_num_kv_heads (global)
    P = prefill_tp_size (consumer)
    D = decode_tp_size (producer)

For any TP size T and rank r:
    If T <= H:
        local_heads_per_rank = H // T
        rank r owns global heads: [r * (H//T), ..., (r+1) * (H//T) - 1]
        head_or_tp_rank = r
        put_step = 1

    If T > H:
        ranks_per_head = T // H
        rank r owns global head: r // (T // H)
        head_or_tp_rank = r // (T // H)
        put_step = T // H

Cross-TP Mapping (consumer=Prefill loads from producer=Decode):
    For each consumer rank:
    1. Determine which global KV heads it needs.
    2. Find which producer head_or_tp_rank stored each head.
    3. Determine the local head index within the producer's stored block.
    4. Determine the local head index in the consumer's KV cache block.

Example: H=4, D=2, P=8
    Decode: rank 0 -> heads [0,1], rank 1 -> heads [2,3]
    Prefill: ranks 0,1 -> head 0, ranks 2,3 -> head 1,
             ranks 4,5 -> head 2, ranks 6,7 -> head 3

    Prefill rank 0 needs head 0 -> from decode rank 0 (head_or_tp=0), local_idx=0
    Prefill rank 2 needs head 1 -> from decode rank 0 (head_or_tp=0), local_idx=1
    Prefill rank 4 needs head 2 -> from decode rank 1 (head_or_tp=1), local_idx=0
    Prefill rank 6 needs head 3 -> from decode rank 1 (head_or_tp=1), local_idx=1
"""

from dataclasses import dataclass, field

from vllm.logger import logger


@dataclass
class HeadMapping:
    """Mapping info for one consumer (Prefill) TP rank."""

    # Global KV head indices this consumer rank needs
    global_head_indices: list[int] = field(default_factory=list)
    # For each global head: which producer head_or_tp_rank stores it
    producer_head_or_tp_ranks: list[int] = field(default_factory=list)
    # For each global head: local head index within the producer's block
    producer_local_head_indices: list[int] = field(default_factory=list)
    # For each global head: local head index within the consumer's block
    consumer_local_head_indices: list[int] = field(default_factory=list)


class TPHeadMapper:
    """
    Computes KV head mappings between asymmetric Prefill/Decode TP configs.

    Usage:
        mapper = TPHeadMapper(total_kv_heads=4, prefill_tp=8, decode_tp=2)
        mapping = mapper.get_mapping(consumer_tp_rank=2)
        # mapping.global_head_indices = [1]
        # mapping.producer_head_or_tp_ranks = [0]
        # mapping.producer_local_head_indices = [1]
        # mapping.consumer_local_head_indices = [0]
    """

    def __init__(
        self,
        total_num_kv_heads: int,
        prefill_tp_size: int,
        decode_tp_size: int,
    ):
        self.total_kv_heads = total_num_kv_heads
        self.prefill_tp_size = prefill_tp_size
        self.decode_tp_size = decode_tp_size

        # Producer (Decode) layout
        if decode_tp_size <= total_num_kv_heads:
            assert total_num_kv_heads % decode_tp_size == 0, (
                f"num_kv_heads ({total_num_kv_heads}) must be divisible by "
                f"decode_tp_size ({decode_tp_size})"
            )
            self.decode_heads_per_rank = total_num_kv_heads // decode_tp_size
            self.decode_put_step = 1
        else:
            assert decode_tp_size % total_num_kv_heads == 0, (
                f"decode_tp_size ({decode_tp_size}) must be divisible by "
                f"num_kv_heads ({total_num_kv_heads})"
            )
            self.decode_heads_per_rank = 1
            self.decode_put_step = decode_tp_size // total_num_kv_heads

        # Consumer (Prefill) layout
        if prefill_tp_size <= total_num_kv_heads:
            assert total_num_kv_heads % prefill_tp_size == 0, (
                f"num_kv_heads ({total_num_kv_heads}) must be divisible by "
                f"prefill_tp_size ({prefill_tp_size})"
            )
            self.prefill_heads_per_rank = total_num_kv_heads // prefill_tp_size
            self.prefill_put_step = 1
        else:
            assert prefill_tp_size % total_num_kv_heads == 0, (
                f"prefill_tp_size ({prefill_tp_size}) must be divisible by "
                f"num_kv_heads ({total_num_kv_heads})"
            )
            self.prefill_heads_per_rank = 1
            self.prefill_put_step = prefill_tp_size // total_num_kv_heads

        self.is_asymmetric = prefill_tp_size != decode_tp_size

        # Build mappings for all prefill ranks
        self._mappings: dict[int, HeadMapping] = {}
        self._build_mappings()

        logger.info(
            "[TPHeadMapper] total_kv_heads=%d, prefill_tp=%d (heads_per_rank=%d, put_step=%d), "
            "decode_tp=%d (heads_per_rank=%d, put_step=%d), is_asymmetric=%s",
            total_num_kv_heads,
            prefill_tp_size,
            self.prefill_heads_per_rank,
            self.prefill_put_step,
            decode_tp_size,
            self.decode_heads_per_rank,
            self.decode_put_step,
            self.is_asymmetric,
        )

    def _global_heads_for_rank(self, tp_rank: int, tp_size: int) -> list[int]:
        """Get global KV head indices owned by a TP rank."""
        H = self.total_kv_heads
        if tp_size <= H:
            heads_per_rank = H // tp_size
            start = tp_rank * heads_per_rank
            return list(range(start, start + heads_per_rank))
        else:
            ranks_per_head = tp_size // H
            head_idx = tp_rank // ranks_per_head
            return [head_idx]

    @staticmethod
    def compute_head_or_tp_rank(tp_rank: int, tp_size: int, num_kv_heads: int) -> int:
        """Compute head_or_tp_rank, consistent with pool_worker.py logic."""
        if num_kv_heads >= tp_size:
            return tp_rank
        else:
            put_step = tp_size // num_kv_heads
            return tp_rank // put_step

    def _decode_rank_for_global_head(self, global_head: int) -> int:
        """Find which decode TP rank owns a given global head."""
        D = self.decode_tp_size
        H = self.total_kv_heads
        if D <= H:
            return global_head // self.decode_heads_per_rank
        else:
            # Multiple decode ranks share the same head; pick the first
            return global_head * self.decode_put_step

    def _build_mappings(self):
        """Build the complete prefill_rank -> head mapping table."""
        for prefill_rank in range(self.prefill_tp_size):
            consumer_heads = self._global_heads_for_rank(
                prefill_rank, self.prefill_tp_size
            )

            producer_ranks = []
            producer_local_indices = []
            consumer_local_indices = []

            for consumer_local_idx, global_head in enumerate(consumer_heads):
                decode_rank = self._decode_rank_for_global_head(global_head)
                prod_head_or_tp = self.compute_head_or_tp_rank(
                    decode_rank, self.decode_tp_size, self.total_kv_heads
                )

                # Local index of global_head within the decode rank's block
                if self.decode_tp_size <= self.total_kv_heads:
                    prod_local_idx = (
                        global_head - decode_rank * self.decode_heads_per_rank
                    )
                else:
                    prod_local_idx = 0

                producer_ranks.append(prod_head_or_tp)
                producer_local_indices.append(prod_local_idx)
                consumer_local_indices.append(consumer_local_idx)

            mapping = HeadMapping(
                global_head_indices=consumer_heads,
                producer_head_or_tp_ranks=producer_ranks,
                producer_local_head_indices=producer_local_indices,
                consumer_local_head_indices=consumer_local_indices,
            )
            self._mappings[prefill_rank] = mapping

            logger.debug(
                "[TPHeadMapper] prefill_rank=%d -> global_heads=%s, "
                "producer_head_or_tp_ranks=%s, producer_local_head_idx=%s, "
                "consumer_local_head_idx=%s",
                prefill_rank,
                consumer_heads,
                producer_ranks,
                producer_local_indices,
                consumer_local_indices,
            )

    def get_mapping(self, consumer_tp_rank: int) -> HeadMapping:
        """Get the head mapping for a consumer (Prefill) TP rank."""
        if consumer_tp_rank not in self._mappings:
            raise ValueError(
                f"[TPHeadMapper] No mapping for consumer_tp_rank={consumer_tp_rank}, "
                f"prefill_tp_size={self.prefill_tp_size}"
            )
        return self._mappings[consumer_tp_rank]

    def get_unique_producer_ranks(self, consumer_tp_rank: int) -> list[int]:
        """Get unique producer head_or_tp_ranks needed by a consumer rank."""
        mapping = self._mappings[consumer_tp_rank]
        return sorted(set(mapping.producer_head_or_tp_ranks))

    def get_heads_from_producer(
        self, consumer_tp_rank: int, producer_head_or_tp_rank: int
    ) -> list[tuple[int, int, int]]:
        """
        For a given consumer rank and producer rank, get the head extraction info.

        Returns:
            list of (global_head, producer_local_idx, consumer_local_idx) tuples
        """
        mapping = self._mappings[consumer_tp_rank]
        result = []
        for i, prod_rank in enumerate(mapping.producer_head_or_tp_ranks):
            if prod_rank == producer_head_or_tp_rank:
                result.append(
                    (
                        mapping.global_head_indices[i],
                        mapping.producer_local_head_indices[i],
                        mapping.consumer_local_head_indices[i],
                    )
                )
        return result

    def build_producer_key_suffix(self, producer_head_or_tp_rank: int) -> str:
        """Build the key suffix used by the producer for lookup."""
        return f"@head_or_tp_rank:{producer_head_or_tp_rank}"

    def __repr__(self) -> str:
        return (
            f"TPHeadMapper(kv_heads={self.total_kv_heads}, "
            f"prefill_tp={self.prefill_tp_size}, decode_tp={self.decode_tp_size})"
        )
