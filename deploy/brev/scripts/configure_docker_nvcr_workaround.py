#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Disable Docker 29.5.x containerd image store for NVCR pulls."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path


def main() -> int:
    try:
        docker_version = subprocess.check_output(
            ["docker", "version", "--format", "{{.Server.Version}}"],
            text=True,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        print(f"Docker is not available yet; skipping NVCR workaround: {exc}")
        return 0

    if not docker_version.startswith("29.5."):
        print(f"Docker {docker_version}: NVCR workaround not needed.")
        return 0

    daemon_path = Path("/etc/docker/daemon.json")
    try:
        current = subprocess.check_output(["sudo", "cat", str(daemon_path)], text=True)
        daemon = json.loads(current or "{}")
    except subprocess.CalledProcessError:
        daemon = {}

    features = daemon.setdefault("features", {})
    if features.get("containerd-snapshotter") is False:
        print(f"Docker {docker_version}: NVCR workaround already configured.")
        return 0

    features["containerd-snapshotter"] = False
    # Write to a uniquely-named, 0600 temp file to avoid symlink/race attacks on a
    # predictable /tmp path before the privileged copy below.
    fd, tmp_path = tempfile.mkstemp(prefix="docker-daemon-nvcr-", suffix=".json", dir="/tmp")
    try:
        with os.fdopen(fd, "w") as fp:
            fp.write(json.dumps(daemon, indent=2) + "\n")
            fp.flush()
            os.fsync(fp.fileno())
        os.chmod(tmp_path, 0o600)
        subprocess.run(["sudo", "cp", tmp_path, str(daemon_path)], check=True)
    finally:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass
    subprocess.run(["sudo", "systemctl", "restart", "docker"], check=True)
    print(f"Docker {docker_version}: disabled containerd-snapshotter and restarted Docker.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
