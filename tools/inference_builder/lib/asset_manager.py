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


from .utils import get_logger
import os
from dataclasses import dataclass, field
from typing import Dict, TYPE_CHECKING
import json
import uuid
import shutil

if TYPE_CHECKING:
    from fastapi import UploadFile
from pyservicemaker.utils import MediaInfo
import tempfile
from urllib.parse import urlparse
import threading

logger = get_logger(__name__)

DEFAULT_ASSET_DIR = "/tmp/assets"
@dataclass
class Asset:
    id: str
    file_name: str
    mime_type: str
    size: int
    duration: int
    path: str
    use_count: int
    asset_dir: str
    description: str
    username: str
    password: str
    live: bool
    managed: bool
    _lock: threading.Lock = field(init=False, repr=False, compare=False, default=None)
    _condition: threading.Condition = field(init=False, repr=False, compare=False, default=None)
    _marked_for_deletion: bool = field(init=False, repr=False, compare=False, default=False)

    def __post_init__(self):
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._marked_for_deletion = False

    def to_dict(self):
        """Convert Asset to dictionary, excluding non-serializable fields."""
        return {
            "id": self.id,
            "file_name": self.file_name,
            "mime_type": self.mime_type,
            "size": self.size,
            "duration": self.duration,
            "path": self.path,
            "use_count": self.use_count,
            "asset_dir": self.asset_dir,
            "description": self.description,
            "username": self.username,
            "password": self.password,
            "live": self.live,
            "managed": self.managed
        }

    def lock(self):
        with self._condition:
            self.use_count += 1

    def unlock(self):
        with self._condition:
            if self.use_count <= 0:
                logger.warning("Attempted to unlock asset %s with use_count %d",
                               self.id, self.use_count)
                return
            self.use_count -= 1
            if self.use_count == 0:
                # Notify waiting threads (e.g., delete_asset)
                self._condition.notify_all()

                # Safety net: cleanup unmanaged assets when object is destroyed
                if not self.managed and os.path.exists(self.path):
                    try:
                        os.remove(self.path)
                        logger.info("Removed unmanaged asset %s from %s",
                                    self.id, self.path)
                    except OSError as e:
                        logger.exception(
                            "Failed to cleanup asset %s: %s", self.id, e)

    def is_marked_for_deletion(self):
        """Check if this asset is marked for deletion."""
        with self._condition:
            return self._marked_for_deletion

    def mark_for_deletion(self):
        """Mark this asset for deletion, signaling processing loops to stop."""
        with self._condition:
            self._marked_for_deletion = True
            logger.info("Marked asset %s for deletion", self.id)

    @classmethod
    def fromdir(cls, asset_dir):
        with open(os.path.join(asset_dir, "info.json")) as f:
            info = json.load(f)

            return Asset(id=info["assetId"],
                         path=info["path"],
                         file_name=info["fileName"],
                         mime_type=info["mimeType"],
                         duration=info["duration"],
                         username=info["username"],
                         password=info["password"],
                         description=info["description"],
                         asset_dir=asset_dir,
                         use_count=0,
                         size=info["size"],
                         live=info["live"],
                         managed=True)


class AssetManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AssetManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._asset_dir = DEFAULT_ASSET_DIR
        self._asset_map:Dict[str, Asset] = dict()
        os.makedirs(self._asset_dir, exist_ok=True)
        asset_ids = self._get_existing_asset_ids()
        self._asset_map: dict[str, Asset] = {
            asset_id: Asset.fromdir(os.path.join(self._asset_dir, asset_id))
            for asset_id in asset_ids
        }
        self._initialized = True

    def save_file(self, file: "UploadFile", file_name: str, mime_type: str) -> Asset | None:
        asset_id = str(uuid.uuid4())
        while asset_id in self._asset_map:
            asset_id = str(uuid.uuid4())
        asset_dir = os.path.join(self._asset_dir, asset_id)
        try:
            os.makedirs(asset_dir)
        except:
            logger.error(f"Failed to create asset directory: {asset_dir}")
            return None

        with open(os.path.join(asset_dir, file_name), "wb") as f:
            shutil.copyfileobj(file, f)
        try:
            size = os.path.getsize(os.path.join(asset_dir, file_name))
        except:
            size = 0
        mediainfo = MediaInfo.discover(os.path.join(asset_dir, file_name))

        with open(os.path.join(asset_dir, "info.json"), "w") as f:
            json.dump(
                {
                    "assetId": asset_id,
                    "path": os.path.join(asset_dir, file_name),
                    "fileName": file_name,
                    "mimeType": mime_type,
                    "duration": mediainfo.duration,
                    "username": "",
                    "password": "",
                    "description": "",
                    "size": size,
                    "live": False,
                }, f)

        self._asset_map[asset_id] = Asset.fromdir(asset_dir)

        logger.info(f"Saved file - asset-id: {asset_id} name: {file_name}")

        return self._asset_map[asset_id]

    def add_live_stream(self, url: str, description="", username="", password="") -> Asset | None:
        asset_id = str(uuid.uuid4())
        while asset_id in self._asset_map:
            asset_id = str(uuid.uuid4())
        asset_dir = os.path.join(self._asset_dir, asset_id)
        try:
            os.makedirs(asset_dir)
        except Exception as e:
            logger.error(f"Failed to create asset directory: {asset_dir}")
            return None

        with open(os.path.join(asset_dir, "info.json"), "w") as f:
            json.dump(
                {
                    "assetId": asset_id,
                    "path": url,
                    "fileName": "",
                    "mimeType": "",
                    "duration": 0,
                    "username": username,
                    "password": password,
                    "description": description,
                    "size": 0,
                    "live": True
                }, f)

        self._asset_map[asset_id] = Asset.fromdir(asset_dir)

        logger.info(f"Added live stream - asset-id: {asset_id} url: {url}")

        return self._asset_map[asset_id]


    def list_assets(self):
        return list(self._asset_map.values())

    def get_asset(self, asset_id):
        if asset_id in self._asset_map:
            return self._asset_map[asset_id]
        else:
            # check if the asset is an id or a path
            # Handle file:// URIs by converting them to paths
            file_path = asset_id
            if asset_id.startswith("file://"):
                parsed = urlparse(asset_id)
                file_path = parsed.path

            if file_path.startswith("/") and os.path.exists(file_path):
                # Check if this path already exists in the asset map
                for existing_asset in self._asset_map.values():
                    if existing_asset.path == file_path:
                        return existing_asset

                mediainfo = MediaInfo.discover(file_path)
                asset = Asset(id=str(uuid.uuid4()),
                              path=file_path,
                              file_name=os.path.basename(file_path),
                              mime_type="",
                              duration=mediainfo.duration,
                              username="",
                              password="",
                              description="",
                              asset_dir=file_path,
                              use_count=0,
                              size=os.path.getsize(file_path),
                              live=False,
                              managed=True)
                self._asset_map[asset.id] = asset
                return asset
            elif (asset_id.startswith("http://") or
                  asset_id.startswith("https://")):
                # Download the video from the web
                try:
                    # Extract filename from URL
                    parsed_url = urlparse(asset_id)
                    file_name = os.path.basename(parsed_url.path)
                    if not file_name:
                        file_name = "downloaded_video"

                    # Create a temporary directory
                    temp_dir = tempfile.mkdtemp(prefix="web_asset_")
                    temp_file_path = os.path.join(temp_dir, file_name)

                    # Download the file
                    logger.info(f"Downloading video from {asset_id}")
                    import requests # lazy import to avoid GIL error
                    response = requests.get(
                        asset_id, stream=True, timeout=30)
                    response.raise_for_status()

                    with open(temp_file_path, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)

                    # Get media info
                    mediainfo = MediaInfo.discover(temp_file_path)
                    file_size = os.path.getsize(temp_file_path)

                    logger.info(f"Downloaded video to {temp_file_path}")

                    return Asset(id=str(uuid.uuid4()),
                                 path=temp_file_path,
                                 file_name=file_name,
                                 mime_type="",
                                 duration=mediainfo.duration,
                                 username="",
                                 password="",
                                 description=f"Downloaded from {asset_id}",
                                 asset_dir=temp_dir,
                                 use_count=0,
                                 size=file_size,
                                 live=False,
                                 managed=False)
                except Exception as e:
                    logger.exception(
                        "Failed to download video from %s: %s", asset_id, e)
                    return None
            elif asset_id.startswith("rtsp://"):
                return self.add_live_stream(asset_id)
            else:
                logger.error(f"Invalid asset ID: {asset_id}")
                return None

    def delete_asset(self, asset_id, timeout=None):
        """
        Delete an asset, waiting for it to be released if currently in use.

        Args:
            asset_id: The ID of the asset to delete
            timeout: Maximum time to wait in seconds (None for indefinite wait)

        Returns:
            bool: True if asset was deleted, False if asset not found or timeout occurred
        """
        if asset_id not in self._asset_map:
            return False

        asset = self._asset_map[asset_id]

        # Mark asset for deletion first (signals processing loops to stop)
        logger.info(f"Marking asset {asset_id} for deletion")
        asset.mark_for_deletion()
        logger.info(f"Marked asset {asset_id} for deletion")

        # Wait for asset to be released
        with asset._condition:
            if asset.use_count > 0:
                logger.info(f"Asset {asset_id} is in use (use_count={asset.use_count}), waiting for release...")
                # Wait until use_count becomes 0
                if not asset._condition.wait_for(lambda: asset.use_count == 0, timeout=timeout):
                    logger.error(f"Timeout waiting for asset {asset_id} to be released")
                    return False
                logger.info(f"Asset {asset_id} released, proceeding with deletion")

        # Now safe to delete
        asset_dir = os.path.join(self._asset_dir, asset_id)
        if os.path.exists(asset_dir):
            shutil.rmtree(asset_dir)
        self._asset_map.pop(asset_id)
        logger.info(f"Removed asset {asset_id} and cleaned up associated resources")
        return True

    def _get_existing_asset_ids(self):
        entries = os.listdir(self._asset_dir)
        return [
            entry for entry in entries if os.path.isdir(os.path.join(self._asset_dir, entry))
            and os.path.isfile(os.path.join(self._asset_dir, entry, "info.json"))
        ]
