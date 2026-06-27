# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from types import SimpleNamespace

import pytest
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec

from vllm_ascend import utils as ascend_utils
from vllm_ascend.utils import (
    DSV4_USER_BLOCK_SIZE_ATTR,
    AscendDeviceType,
    get_dsv4_configured_block_size,
    refresh_block_size,
)

pytestmark = pytest.mark.cpu_test


@pytest.fixture()
def dsv4_modules():
    original_device_type = ascend_utils._ascend_device_type
    ascend_utils._ascend_device_type = AscendDeviceType.A3

    from vllm_ascend.models.layer.attention import layer as dsv4_layer
    from vllm_ascend.patch.platform import patch_kv_cache_coordinator as coordinator_patch
    from vllm_ascend.patch.worker import patch_deepseek_compressor as compressor_patch

    yield SimpleNamespace(
        layer=dsv4_layer,
        coordinator_patch=coordinator_patch,
        compressor_patch=compressor_patch,
    )
    ascend_utils._ascend_device_type = original_device_type


def _make_vllm_config(block_size=None, model_type="deepseek_v4"):
    return SimpleNamespace(
        cache_config=SimpleNamespace(
            block_size=block_size,
            cache_dtype="float16",
            enable_prefix_caching=False,
        ),
        scheduler_config=SimpleNamespace(enable_chunked_prefill=False),
        model_config=SimpleNamespace(
            dtype=torch.float16,
            hf_config=SimpleNamespace(model_type=model_type),
            is_hybrid=False,
        ),
    )


@pytest.mark.parametrize(
    ("block_size", "expected"),
    [
        (None, 32),
        (32, 32),
        (64, 64),
        (128, 128),
        (16, 32),
    ],
)
def test_refresh_block_size_for_deepseek_v4(block_size, expected):
    vllm_config = _make_vllm_config(block_size=block_size)

    refresh_block_size(vllm_config)

    assert vllm_config.cache_config.block_size == expected
    assert getattr(vllm_config.cache_config, DSV4_USER_BLOCK_SIZE_ATTR) == expected
    assert get_dsv4_configured_block_size(vllm_config.cache_config) == expected


def test_refresh_block_size_keeps_runtime_block_size_after_user_value_is_saved():
    vllm_config = _make_vllm_config(block_size=64)

    refresh_block_size(vllm_config)
    vllm_config.cache_config.block_size = 4
    refresh_block_size(vllm_config)

    assert vllm_config.cache_config.block_size == 4
    assert getattr(vllm_config.cache_config, DSV4_USER_BLOCK_SIZE_ATTR) == 64
    assert get_dsv4_configured_block_size(vllm_config.cache_config) == 64


def test_refresh_block_size_keeps_generic_default():
    vllm_config = _make_vllm_config(block_size=None, model_type="deepseek_v3")

    refresh_block_size(vllm_config)

    assert vllm_config.cache_config.block_size == 128


def test_dsv4_block_size_table_for_non_a5(monkeypatch, dsv4_modules):
    dsv4_layer = dsv4_modules.layer
    monkeypatch.setattr(dsv4_layer, "get_ascend_device_type", lambda: AscendDeviceType.A3)

    assert dsv4_layer.get_dsv4_block_sizes() == {
        128: [[128, 128, 8, 32], [16640, 131072]],
        64: [[64, 64, 4, 16], [8320, 65536]],
        32: [[32, 32, 2, 8], [4160, 32768]],
    }


def test_dsv4_block_size_table_for_a5(monkeypatch, dsv4_modules):
    dsv4_layer = dsv4_modules.layer
    monkeypatch.setattr(dsv4_layer, "get_ascend_device_type", lambda: AscendDeviceType.A5)

    assert dsv4_layer.get_dsv4_block_sizes() == {
        128: [[128, 128, 8, 16], [16896, 81920]],
        64: [[64, 64, 4, 8], [8448, 40960]],
        32: [[32, 32, 2, 4], [4224, 20480]],
    }


@pytest.mark.parametrize(
    ("block_size", "expected"),
    [
        (None, [[32, 32, 2, 8], [4160, 32768]]),
        (32, [[32, 32, 2, 8], [4160, 32768]]),
        (64, [[64, 64, 4, 16], [8320, 65536]]),
        (128, [[128, 128, 8, 32], [16640, 131072]]),
        (16, [[32, 32, 2, 8], [4160, 32768]]),
    ],
)
def test_get_dsv4_cache_sizes_falls_back_to_32(monkeypatch, dsv4_modules, block_size, expected):
    dsv4_layer = dsv4_modules.layer
    monkeypatch.setattr(dsv4_layer, "get_ascend_device_type", lambda: AscendDeviceType.A3)
    monkeypatch.setattr(dsv4_layer, "DSV4_BLOCK_SIZES", dsv4_layer.get_dsv4_block_sizes())

    assert dsv4_layer.get_dsv4_cache_sizes(block_size) == expected


def test_get_dsv4_cache_sizes_for_config_uses_saved_user_block_size(monkeypatch, dsv4_modules):
    dsv4_layer = dsv4_modules.layer
    monkeypatch.setattr(dsv4_layer, "get_ascend_device_type", lambda: AscendDeviceType.A3)
    monkeypatch.setattr(dsv4_layer, "DSV4_BLOCK_SIZES", dsv4_layer.get_dsv4_block_sizes())
    vllm_config = _make_vllm_config(block_size=64)

    refresh_block_size(vllm_config)
    vllm_config.cache_config.block_size = 4

    assert dsv4_layer.get_dsv4_cache_sizes_for_config(vllm_config.cache_config) == [
        [64, 64, 4, 16],
        [8320, 65536],
    ]


def test_dsa_attention_cache_spec_uses_a5_head_padding(monkeypatch, dsv4_modules):
    dsv4_layer = dsv4_modules.layer
    monkeypatch.setattr(dsv4_layer, "get_ascend_device_type", lambda: AscendDeviceType.A5)
    monkeypatch.setattr(dsv4_layer, "DSV4_BLOCK_SIZES", dsv4_layer.get_dsv4_block_sizes())

    attention = dsv4_layer.DSAAttention.__new__(dsv4_layer.DSAAttention)
    attention.compress_ratio = 4
    attention.head_size = 192
    attention.kv_cache_dtype = "float16"
    vllm_config = _make_vllm_config(block_size=64)

    spec = attention.get_kv_cache_spec(vllm_config)

    assert spec.block_size == 64
    assert spec.head_size == 320
    assert spec.dtype is torch.float8_e4m3fn
    assert spec.cache_dtype_str == "float8_e4m3fn"
    assert vllm_config.cache_config.cache_dtype == "float8_e4m3fn"


def test_dsa_attention_cache_spec_uses_saved_block_size_after_runtime_min(monkeypatch, dsv4_modules):
    dsv4_layer = dsv4_modules.layer
    monkeypatch.setattr(dsv4_layer, "get_ascend_device_type", lambda: AscendDeviceType.A3)
    monkeypatch.setattr(dsv4_layer, "DSV4_BLOCK_SIZES", dsv4_layer.get_dsv4_block_sizes())

    attention = dsv4_layer.DSAAttention.__new__(dsv4_layer.DSAAttention)
    attention.compress_ratio = 4
    attention.head_size = 192
    attention.kv_cache_dtype = "float16"
    vllm_config = _make_vllm_config(block_size=64)
    refresh_block_size(vllm_config)
    vllm_config.cache_config.block_size = 4

    spec = attention.get_kv_cache_spec(vllm_config)

    assert spec.block_size == 64


def test_swa_cache_spec_uses_saved_block_size_after_runtime_min(monkeypatch, dsv4_modules):
    dsv4_layer = dsv4_modules.layer
    compressor_patch = dsv4_modules.compressor_patch
    monkeypatch.setattr(dsv4_layer, "get_ascend_device_type", lambda: AscendDeviceType.A3)
    monkeypatch.setattr(dsv4_layer, "DSV4_BLOCK_SIZES", dsv4_layer.get_dsv4_block_sizes())

    def fake_swa_init(self, head_dim, window_size, dtype, prefix, cache_config):
        self.head_dim = head_dim
        self.window_size = window_size
        self.dtype = dtype
        self.prefix = prefix
        self.cache_config = cache_config

    monkeypatch.setattr(compressor_patch.DeepseekV4SWACache, "__init__", fake_swa_init)
    vllm_config = _make_vllm_config(block_size=64)
    refresh_block_size(vllm_config)
    vllm_config.cache_config.block_size = 4

    cache = compressor_patch.AscendDeepseekV4SWACache(
        head_dim=192,
        window_size=4096,
        dtype=torch.float16,
        prefix="model.layers.0.swa",
        cache_config=vllm_config.cache_config,
    )
    spec = cache.get_kv_cache_spec(vllm_config)

    assert cache.block_size == 64
    assert spec.block_size == 64


@pytest.mark.parametrize(
    ("block_size", "state_dim", "compress_ratio", "expected_block", "expected_page"),
    [
        (128, 512, 4, 8, 16640),
        (128, 128, 128, 32, 131072),
        (64, 512, 4, 4, 8320),
        (64, 128, 128, 16, 65536),
        (32, 512, 4, 2, 4160),
        (32, 128, 128, 8, 32768),
    ],
)
def test_compressor_state_cache_uses_non_a5_cache_table(
    monkeypatch,
    dsv4_modules,
    block_size,
    state_dim,
    compress_ratio,
    expected_block,
    expected_page,
):
    dsv4_layer = dsv4_modules.layer
    compressor_patch = dsv4_modules.compressor_patch
    monkeypatch.setattr(dsv4_layer, "get_ascend_device_type", lambda: AscendDeviceType.A3)
    monkeypatch.setattr(dsv4_layer, "DSV4_BLOCK_SIZES", dsv4_layer.get_dsv4_block_sizes())
    cache = compressor_patch.AscendCompressorStateCache.__new__(compressor_patch.AscendCompressorStateCache)
    cache.state_dim = state_dim
    cache.dtype = torch.float32
    cache.compress_ratio = compress_ratio
    cache.sliding_window = compress_ratio * (1 + (compress_ratio == 4))
    cache.block_size = dsv4_layer.get_dsv4_cache_sizes(block_size)[0][2 if compress_ratio == 4 else 3]

    spec = cache.get_kv_cache_spec(_make_vllm_config(block_size=block_size))

    assert spec.block_size == expected_block
    assert spec.page_size_padded == expected_page


def test_compressor_state_cache_uses_saved_block_size_after_runtime_min(monkeypatch, dsv4_modules):
    dsv4_layer = dsv4_modules.layer
    compressor_patch = dsv4_modules.compressor_patch
    monkeypatch.setattr(dsv4_layer, "get_ascend_device_type", lambda: AscendDeviceType.A3)
    monkeypatch.setattr(dsv4_layer, "DSV4_BLOCK_SIZES", dsv4_layer.get_dsv4_block_sizes())
    cache = compressor_patch.AscendCompressorStateCache.__new__(compressor_patch.AscendCompressorStateCache)
    cache.state_dim = 512
    cache.dtype = torch.float32
    cache.compress_ratio = 4
    cache.sliding_window = 8
    cache.block_size = 4
    vllm_config = _make_vllm_config(block_size=64)
    refresh_block_size(vllm_config)
    vllm_config.cache_config.block_size = 4

    spec = cache.get_kv_cache_spec(vllm_config)

    assert spec.block_size == 4
    assert spec.page_size_padded == 8320


def test_kv_cache_coordinator_falls_back_for_non_deepseek_v4(monkeypatch, dsv4_modules):
    coordinator_patch = dsv4_modules.coordinator_patch
    sentinel = object()
    captured = {}

    def fake_original_coordinator(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(coordinator_patch, "_orig_get_kv_cache_coordinator", fake_original_coordinator)
    kv_cache_config = KVCacheConfig(
        num_blocks=8,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["full"],
                FullAttentionSpec(
                    block_size=128,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float16,
                ),
            ),
        ],
    )

    coordinator = coordinator_patch.get_kv_cache_coordinator(
        kv_cache_config=kv_cache_config,
        max_model_len=128,
        max_num_batched_tokens=128,
        use_eagle=False,
        enable_caching=False,
        enable_kv_cache_events=False,
        dcp_world_size=1,
        pcp_world_size=1,
        hash_block_size=128,
    )

    assert coordinator is sentinel
    assert captured["kv_cache_config"] is kv_cache_config
