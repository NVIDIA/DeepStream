# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for AssetManager.create_from_path().

Test tier: Tier 2 (no GPU required).

To run:
    pytest tests/test_asset_manager_create_from_path.py -v
"""

import json
import os
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Fixture: isolated AssetManager with a temporary asset directory
# ---------------------------------------------------------------------------


@pytest.fixture
def asset_manager(tmp_path, monkeypatch):
    """Return a fresh AssetManager instance backed by a temp directory.

    The singleton is reset between tests so each test gets a clean state.
    """
    # Reset singleton between tests
    import lib.asset_manager as am_module

    am_module.AssetManager._instance = None

    # Override DEFAULT_ASSET_DIR to use tmp_path
    monkeypatch.setattr(am_module, "DEFAULT_ASSET_DIR", str(tmp_path / "assets"))

    manager = am_module.AssetManager()
    return manager


@pytest.fixture
def temp_mp4(tmp_path):
    """Create a minimal placeholder MP4-like file for testing."""
    mp4_path = str(tmp_path / "source_video.mp4")
    with open(mp4_path, "wb") as f:
        # Write enough bytes to look like a non-empty file
        f.write(b"\x00\x00\x00\x1cftypisom" + b"\x00" * 100)
    return mp4_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCreateFromPath:
    """Tests for AssetManager.create_from_path()."""

    def test_creates_asset_with_correct_metadata(self, asset_manager, temp_mp4):
        """create_from_path returns an Asset with correct id, file_name, mime_type."""
        with patch("lib.asset_manager.MediaInfo") as MockMediaInfo:
            mock_info = MagicMock()
            mock_info.duration = 333  # ms
            MockMediaInfo.discover.return_value = mock_info

            asset = asset_manager.create_from_path(
                file_path=temp_mp4,
                file_name="output.mp4",
                mime_type="video/mp4",
            )

        assert asset is not None
        assert asset.file_name == "output.mp4"
        assert asset.mime_type == "video/mp4"
        assert asset.duration == 333
        assert len(asset.id) > 0

    def test_asset_is_registered_in_asset_map(self, asset_manager, temp_mp4):
        """create_from_path registers the new asset in the manager's map."""
        with patch("lib.asset_manager.MediaInfo") as MockMediaInfo:
            mock_info = MagicMock()
            mock_info.duration = 0
            MockMediaInfo.discover.return_value = mock_info

            asset = asset_manager.create_from_path(
                file_path=temp_mp4,
                file_name="output.mp4",
                mime_type="video/mp4",
            )

        assert asset is not None
        retrieved = asset_manager.get_asset(asset.id)
        assert retrieved is not None
        assert retrieved.id == asset.id

    def test_source_file_is_moved_not_copied(self, asset_manager, temp_mp4):
        """The source file is moved into the asset dir; original path disappears."""
        with patch("lib.asset_manager.MediaInfo") as MockMediaInfo:
            mock_info = MagicMock()
            mock_info.duration = 0
            MockMediaInfo.discover.return_value = mock_info

            assert os.path.exists(temp_mp4)
            asset = asset_manager.create_from_path(
                file_path=temp_mp4,
                file_name="output.mp4",
                mime_type="video/mp4",
            )

        # Source should no longer exist after move
        assert not os.path.exists(temp_mp4)
        # Destination file should exist
        assert os.path.exists(asset.path)

    def test_info_json_written_correctly(self, asset_manager, temp_mp4):
        """info.json in the asset directory contains all required fields."""
        with patch("lib.asset_manager.MediaInfo") as MockMediaInfo:
            mock_info = MagicMock()
            mock_info.duration = 500
            MockMediaInfo.discover.return_value = mock_info

            asset = asset_manager.create_from_path(
                file_path=temp_mp4,
                file_name="output.mp4",
                mime_type="video/mp4",
            )

        info_path = os.path.join(asset.asset_dir, "info.json")
        assert os.path.exists(info_path)
        with open(info_path) as f:
            info = json.load(f)

        assert info["assetId"] == asset.id
        assert info["fileName"] == "output.mp4"
        assert info["mimeType"] == "video/mp4"
        assert info["duration"] == 500
        assert info["live"] is False

    def test_returns_none_when_source_missing(self, asset_manager):
        """Returns None when source file does not exist."""
        with patch("lib.asset_manager.MediaInfo"):
            result = asset_manager.create_from_path(
                file_path="/nonexistent/path/file.mp4",
                file_name="output.mp4",
                mime_type="video/mp4",
            )
        assert result is None

    def test_concurrent_calls_produce_unique_asset_ids(self, asset_manager, tmp_path):
        """Concurrent create_from_path calls each get a unique asset ID."""
        import threading

        n_threads = 8
        assets = []
        errors = []
        lock = threading.Lock()

        def _create(i):
            mp4_path = str(tmp_path / f"video_{i}.mp4")
            with open(mp4_path, "wb") as f:
                f.write(b"\x00\x00\x00\x1cftypisom" + b"\x00" * 50)

            with patch("lib.asset_manager.MediaInfo") as MockMediaInfo:
                mock_info = MagicMock()
                mock_info.duration = 0
                MockMediaInfo.discover.return_value = mock_info

                try:
                    asset = asset_manager.create_from_path(
                        file_path=mp4_path,
                        file_name="output.mp4",
                        mime_type="video/mp4",
                    )
                    with lock:
                        assets.append(asset)
                except Exception as exc:
                    with lock:
                        errors.append(exc)

        threads = [threading.Thread(target=_create, args=(i,)) for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Unexpected errors: {errors}"
        asset_ids = [a.id for a in assets if a is not None]
        assert len(asset_ids) == len(set(asset_ids)), "Duplicate asset IDs detected"

    def test_size_is_recorded_correctly(self, asset_manager, temp_mp4):
        """The asset size matches the file size on disk."""
        expected_size = os.path.getsize(temp_mp4)

        with patch("lib.asset_manager.MediaInfo") as MockMediaInfo:
            mock_info = MagicMock()
            mock_info.duration = 0
            MockMediaInfo.discover.return_value = mock_info

            asset = asset_manager.create_from_path(
                file_path=temp_mp4,
                file_name="output.mp4",
                mime_type="video/mp4",
            )

        assert asset.size == expected_size
