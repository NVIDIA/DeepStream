# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SKILL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL_ROOT / "scripts"))
from dataset_cache import stage_local_source  # noqa: E402


class TestDatasetCache(unittest.TestCase):
    def test_directory_is_staged_and_reused_until_refresh(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "windows_source"
            cache = root / "linux_cache"
            source.mkdir()
            (source / "images").mkdir()
            (source / "images" / "one.jpg").write_bytes(b"first")

            staged = stage_local_source(source, cache, workers=2)
            self.assertEqual((staged / "images" / "one.jpg").read_bytes(), b"first")
            marker = json.loads((staged.parent / "cache_manifest.json").read_text())
            self.assertEqual(marker["source"], str(source.resolve()))
            self.assertEqual(marker["files"], 1)

            (source / "images" / "one.jpg").write_bytes(b"changed")
            reused = stage_local_source(source, cache)
            self.assertEqual(reused, staged)
            self.assertEqual((reused / "images" / "one.jpg").read_bytes(), b"first")

            refreshed = stage_local_source(source, cache, refresh=True)
            self.assertEqual((refreshed / "images" / "one.jpg").read_bytes(), b"changed")

    def test_file_source_preserves_filename(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "dataset.zip"
            source.write_bytes(b"zip payload")
            staged = stage_local_source(source, root / "cache")
            self.assertEqual(staged.name, "dataset.zip")
            self.assertEqual(staged.read_bytes(), b"zip payload")

    def test_source_already_in_cache_is_not_copied(self):
        with tempfile.TemporaryDirectory() as td:
            cache = Path(td) / "cache"
            source = cache / "already-local"
            source.mkdir(parents=True)
            self.assertEqual(stage_local_source(source, cache), source.resolve())

    def test_missing_source_fails_without_cache_artifact(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cache = root / "cache"
            with self.assertRaises(FileNotFoundError):
                stage_local_source(root / "missing", cache)
            self.assertFalse((cache / ".sources").exists())

    def test_ingest_manifest_points_to_cached_images(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "windows_source"
            split = source / "train"
            split.mkdir(parents=True)
            (split / "frame.jpg").write_bytes(b"jpeg")
            (split / "_annotations.coco.json").write_text(json.dumps({
                "images": [{"id": 1, "file_name": "frame.jpg", "width": 10, "height": 10}],
                "annotations": [],
                "categories": [{"id": 0, "name": "part"}],
            }))
            cache = root / "linux_cache"
            dest = cache / "processed"
            # Explicit allowlist rather than a copy of os.environ: the child only
            # needs an interpreter environment plus the cache root under test.
            env = {
                k: os.environ[k]
                for k in ("PATH", "HOME", "LANG", "TMPDIR", "PYTHONPATH", "HF_HOME")
                if k in os.environ
            }
            env["DS_DATA_CACHE_ROOT"] = str(cache)
            subprocess.run([
                sys.executable, str(SKILL_ROOT / "scripts" / "ingest_dataset.py"),
                "--repo-id", str(source), "--dest", str(dest), "--splits", "train",
            ], check=True, env=env, capture_output=True, text=True)
            manifest = json.loads((dest / "ingest_manifest.json").read_text())
            image_root = Path(manifest["splits"]["train"]["images_dir"]).resolve()
            self.assertTrue(str(image_root).startswith(str((cache / ".sources").resolve())))
            self.assertFalse(str(image_root).startswith(str(source.resolve())))


if __name__ == "__main__":
    unittest.main()
