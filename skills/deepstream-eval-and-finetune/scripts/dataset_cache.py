# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Persistent Linux-backed staging for datasets mounted from Windows/WSL.

The source is read-only. A complete copy is assembled under a temporary sibling and
atomically promoted, so interrupted staging is never mistaken for a usable cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _cache_key(source: Path) -> str:
    return hashlib.sha256(str(source.resolve()).encode("utf-8")).hexdigest()[:20]


def _remove_cache_path(path: Path, cache_root: Path) -> None:
    sources_root = (cache_root / ".sources").resolve()
    resolved = path.resolve()
    if resolved.parent != sources_root:
        raise ValueError(f"refusing to remove path outside cache source root: {resolved}")
    if resolved.is_dir():
        shutil.rmtree(resolved)
    elif resolved.exists():
        resolved.unlink()


def _copy_directory(source: Path, payload: Path, workers: int) -> tuple[int, int]:
    files = []
    total_bytes = 0
    for root, dirs, names in os.walk(source, followlinks=False):
        root_path = Path(root)
        dirs[:] = [d for d in dirs if not (root_path / d).is_symlink()]
        for name in names:
            src = root_path / name
            if not src.is_file():
                continue
            rel = src.relative_to(source)
            files.append((src, payload / rel))
            total_bytes += src.stat().st_size

    free = shutil.disk_usage(payload.parent).free
    if free < total_bytes:
        raise OSError(
            f"insufficient cache space: need {total_bytes} bytes, have {free} bytes"
        )

    def copy_one(pair):
        src, dst = pair
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        list(pool.map(copy_one, files))
    return len(files), total_bytes


def stage_local_source(
    source: str | os.PathLike,
    cache_root: str | os.PathLike,
    *,
    refresh: bool = False,
    workers: int = 8,
) -> Path:
    """Return a Linux-cache path for an existing local file or directory."""
    source = Path(source).expanduser().resolve()
    cache_root = Path(cache_root).resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    if _inside(source, cache_root):
        return source

    sources_root = cache_root / ".sources"
    sources_root.mkdir(parents=True, exist_ok=True)
    target = sources_root / _cache_key(source)
    marker = target / "cache_manifest.json"
    if target.exists() and not refresh:
        if marker.is_file():
            payload = target / "payload"
            return payload / source.name if source.is_file() else payload
        raise RuntimeError(f"incomplete dataset cache exists: {target}; refresh it")
    if target.exists():
        _remove_cache_path(target, cache_root)

    for stale in sources_root.glob(f"{target.name}.partial-*"):
        _remove_cache_path(stale, cache_root)
    partial = sources_root / f"{target.name}.partial-{os.getpid()}"
    if partial.exists():
        _remove_cache_path(partial, cache_root)
    payload = partial / "payload"
    payload.mkdir(parents=True)
    started = time.time()
    try:
        if source.is_dir():
            n_files, total_bytes = _copy_directory(source, payload, workers)
            result = payload
        else:
            size = source.stat().st_size
            if shutil.disk_usage(payload).free < size:
                raise OSError(
                    f"insufficient cache space: need {size} bytes"
                )
            shutil.copy2(source, payload / source.name)
            n_files, total_bytes = 1, size
            result = payload / source.name
        manifest = {
            "source": str(source),
            "source_kind": "directory" if source.is_dir() else "file",
            "files": n_files,
            "bytes": total_bytes,
            "staged_seconds": round(time.time() - started, 3),
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        (partial / "cache_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        os.replace(partial, target)
        return target / "payload" / source.name if source.is_file() else target / "payload"
    except Exception:
        if partial.exists():
            _remove_cache_path(partial, cache_root)
        raise


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("--cache-root", required=True)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    print(stage_local_source(
        args.source, args.cache_root, refresh=args.refresh, workers=args.workers
    ))


if __name__ == "__main__":
    main()
