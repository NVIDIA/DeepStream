#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stage the complete, digest-pinned Triton 26.10 C++ SDK; never install it."""

import argparse
import hashlib
import json
import pathlib
import posixpath
import shutil
import tarfile
import tempfile
import urllib.parse
import urllib.request


REGISTRY = "https://nvcr.io/v2/nvidia/tritonserver/"
# /workspace/install COPY layers, not complete images or runnable containers.
# Source: nvcr.io/nvidia/tritonserver:26.10-py3-sdk, per-platform manifests.
PROVIDERS = {
    "x86": (
        "1114e9c6e916ba0fd5ca8bcd0c6173f7226a603057979c15ca67f7906d22c969",
        "01c2ce22fb5455aa92b4766f972dee969ad2b47a3c352802276cf6eab8578e23",
        231350404,
    ),
    "aarch64": (
        "f627e4010cdcdc23e84ad29721b6c2f703d1b6a83a73a68170645226eb6c143d",
        "069ac3bbab47f333168e6040772970fc78ce58403e24d3419b0ff6c4cff68f08",
        223522296,
    ),
}


class SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, url):
        target = urllib.parse.urlsplit(url)
        if target.scheme != "https":
            raise ValueError("SDK download redirected outside HTTPS")
        redirected = super().redirect_request(request, response, code, message, headers, url)
        if urllib.parse.urlsplit(request.full_url).netloc != target.netloc:
            redirected.remove_header("Authorization")
        return redirected


def verified_payload(platform, download):
    """Fetch anonymous metadata and one SDK layer; verify bytes before extraction."""
    manifest_hash, layer_hash, size = PROVIDERS[platform]
    opener = urllib.request.build_opener(SafeRedirect())
    with opener.open("https://nvcr.io/token?service=registry&scope=repository:nvidia/tritonserver:pull", timeout=60) as response:
        token = json.loads(response.read(1024 * 1024))["token"]
    headers = {"Authorization": f"Bearer {token}",
               "Accept": "application/vnd.docker.distribution.manifest.v2+json"}
    request = urllib.request.Request(REGISTRY + "manifests/sha256:" + manifest_hash, headers=headers)
    with opener.open(request, timeout=60) as response:
        metadata = response.read(1024 * 1024)
    if hashlib.sha256(metadata).hexdigest() != manifest_hash:
        raise ValueError("Triton SDK manifest checksum mismatch")
    layers = json.loads(metadata)["layers"]
    if not any(layer["digest"] == "sha256:" + layer_hash and layer["size"] == size for layer in layers):
        raise ValueError("Pinned SDK payload is absent from its platform manifest")
    request = urllib.request.Request(REGISTRY + "blobs/sha256:" + layer_hash, headers=headers)
    digest, received = hashlib.sha256(), 0
    with opener.open(request, timeout=60) as response, download.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            received += len(chunk)
            if received > size:
                raise ValueError("SDK payload exceeds its pinned size")
            digest.update(chunk)
            output.write(chunk)
    if received != size or digest.hexdigest() != layer_hash:
        raise ValueError("Triton SDK payload checksum or size mismatch")


def extract_sdk(archive, destination):
    """Extract only SDK headers/libraries, without owners, hardlinks or devices."""
    selected, names, symlinks = [], {}, []
    with tarfile.open(archive, "r:gz") as payload:
        for member in payload:
            path = pathlib.PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("Unsafe SDK archive path")
            parts = path.parts
            if len(parts) < 3 or parts[:2] != ("workspace", "install") or parts[2] not in ("include", "lib", "lib64"):
                continue
            relative = pathlib.PurePosixPath(*parts[2:])
            if relative in names or any(part.startswith(".wh.") for part in relative.parts):
                raise ValueError("Duplicate or whiteout SDK archive entry")
            names[relative] = member
            if not (member.isdir() or member.isfile() or member.issym()):
                raise ValueError("Unsupported SDK archive entry type")
            if member.issym():
                target = pathlib.PurePosixPath(member.linkname)
                resolved = pathlib.PurePosixPath(posixpath.normpath(str(relative.parent / target)))
                if target.is_absolute() or ".." in target.parts or not resolved.parts or resolved.parts[0] != relative.parts[0]:
                    raise ValueError("SDK symlink escapes its component")
                symlinks.append((member, relative))
            else:
                selected.append((member, relative))
        # No regular-file extraction can traverse a symlink: links are last.
        link_paths = {relative for _, relative in symlinks}
        for member, relative in selected + symlinks:
            if any(parent in link_paths for parent in relative.parents):
                raise ValueError("SDK archive entry traverses a symlink")
            if any(parent in names and not names[parent].isdir()
                   for parent in relative.parents if str(parent) != "."):
                raise ValueError("SDK archive entry traverses a file")
        for member, relative in selected:
            target = destination / relative
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with payload.extractfile(member) as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output)
                target.chmod(member.mode & 0o777)
        for member, relative in symlinks:
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(member.linkname)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=("x86", "aarch64", "sbsa"), required=True)
    parser.add_argument("--destination", type=pathlib.Path, required=True)
    args = parser.parse_args()
    destination = args.destination.absolute()
    if destination.is_symlink() or not destination.is_dir() or any(destination.iterdir()):
        parser.error("destination must be an existing empty, non-symlink staging directory")
    platform = "aarch64" if args.platform == "sbsa" else args.platform
    try:
        with tempfile.TemporaryDirectory(prefix="ds-triton-sdk-") as temporary:
            archive = pathlib.Path(temporary) / "sdk-layer.tar.gz"
            verified_payload(platform, archive)
            extract_sdk(archive, destination)
    except (OSError, ValueError, KeyError, tarfile.TarError) as error:
        parser.exit(1, f"error: could not stage Triton client SDK: {error}\n")
    print(destination)


if __name__ == "__main__":
    main()
