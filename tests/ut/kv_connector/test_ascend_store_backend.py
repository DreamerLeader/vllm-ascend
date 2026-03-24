# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Tests for backend modules - covers mooncake_backend.py utility functions
and memcache_backend.py MmcDirect enum."""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_backend import (
    DEFAULT_GLOBAL_SEGMENT_SIZE,
    DEFAULT_LOCAL_BUFFER_SIZE,
    MooncakeStoreConfig,
    _convert_to_bytes,
    _parse_global_segment_size,
)


# ─── _parse_global_segment_size ──────────────────────────────────────────────


class TestParseGlobalSegmentSize:
    def test_int_input(self):
        assert _parse_global_segment_size(1024) == 1024

    def test_string_gb(self):
        assert _parse_global_segment_size("1GB") == 1024**3

    def test_string_mb(self):
        assert _parse_global_segment_size("1MB") == 1024**2

    def test_string_kb(self):
        assert _parse_global_segment_size("1KB") == 1024

    def test_string_b(self):
        assert _parse_global_segment_size("100B") == 100

    def test_string_no_unit(self):
        assert _parse_global_segment_size("1024") == 1024

    def test_string_with_spaces(self):
        assert _parse_global_segment_size("  2 GB  ") == 2 * 1024**3

    def test_float_gb(self):
        assert _parse_global_segment_size("1.5GB") == int(1.5 * 1024**3)

    def test_case_insensitive(self):
        assert _parse_global_segment_size("1gb") == 1024**3
        assert _parse_global_segment_size("1Gb") == 1024**3

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            _parse_global_segment_size("")

    def test_invalid_format_raises(self):
        with pytest.raises(ValueError, match="Invalid format"):
            _parse_global_segment_size("abc")

    def test_float_input(self):
        assert _parse_global_segment_size(1024.0) == 1024

    def test_unsupported_type_raises(self):
        with pytest.raises(TypeError, match="Unsupported type"):
            _parse_global_segment_size([1024])


# ─── _convert_to_bytes ───────────────────────────────────────────────────────


class TestConvertToBytes:
    def test_basic(self):
        assert _convert_to_bytes("100", 1, "100") == 100

    def test_with_multiplier(self):
        assert _convert_to_bytes("2", 1024, "2KB") == 2048

    def test_float_value(self):
        assert _convert_to_bytes("1.5", 1024, "1.5KB") == 1536

    def test_invalid_number(self):
        with pytest.raises(ValueError, match="Invalid numeric value"):
            _convert_to_bytes("abc", 1, "abc")


# ─── MooncakeStoreConfig ─────────────────────────────────────────────────────


class TestMooncakeStoreConfig:
    def test_from_file(self):
        config_data = {
            "metadata_server": "127.0.0.1:2379",
            "global_segment_size": "1GB",
            "local_buffer_size": "512MB",
            "protocol": "ascend",
            "device_name": "npu0",
            "master_server_address": "127.0.0.1:50051",
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            f.flush()
            config = MooncakeStoreConfig.from_file(f.name)

        os.unlink(f.name)
        assert config.metadata_server == "127.0.0.1:2379"
        assert config.global_segment_size == 1024**3
        assert config.local_buffer_size == 512 * 1024**2
        assert config.protocol == "ascend"
        assert config.device_name == "npu0"

    def test_from_file_defaults(self):
        config_data = {
            "metadata_server": "127.0.0.1:2379",
            "master_server_address": "127.0.0.1:50051",
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            f.flush()
            config = MooncakeStoreConfig.from_file(f.name)

        os.unlink(f.name)
        assert config.global_segment_size == DEFAULT_GLOBAL_SEGMENT_SIZE
        assert config.local_buffer_size == DEFAULT_LOCAL_BUFFER_SIZE
        assert config.protocol == "ascend"
        assert config.device_name == ""

    def test_load_from_env_missing(self):
        with patch.dict(os.environ, {}, clear=True):
            # Remove MOONCAKE_CONFIG_PATH if it exists
            os.environ.pop("MOONCAKE_CONFIG_PATH", None)
            with pytest.raises(ValueError, match="MOONCAKE_CONFIG_PATH"):
                MooncakeStoreConfig.load_from_env()

    def test_load_from_env(self):
        config_data = {
            "metadata_server": "127.0.0.1:2379",
            "master_server_address": "127.0.0.1:50051",
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            f.flush()
            with patch.dict(os.environ, {"MOONCAKE_CONFIG_PATH": f.name}):
                config = MooncakeStoreConfig.load_from_env()

        os.unlink(f.name)
        assert config.metadata_server == "127.0.0.1:2379"


# ─── MmcDirect Enum ──────────────────────────────────────────────────────────


class TestMmcDirect:
    def test_values(self):
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.memcache_backend import (
            MmcDirect,
        )

        assert MmcDirect.COPY_L2G.value == 0
        assert MmcDirect.COPY_G2L.value == 1
        assert MmcDirect.COPY_G2H.value == 2
        assert MmcDirect.COPY_H2G.value == 3
