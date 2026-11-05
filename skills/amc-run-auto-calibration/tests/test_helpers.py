#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused unit tests for AMC skill helper validation."""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import io
import os
import re
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from amc_common import create_project, maybe_run_vggt, print_log_tail, redact_value, timeout_from_env
from run_rtsp_calibration import capture_ssl_verify, default_ui_url, validate_capture_duration
from run_sample_calibration import SAMPLE_ZIP, find_sample_archive, verify_sample_archive
from run_video_calibration import discover_video_files


class FakeResponse:
    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict[str, str]:
        return {"project_id": "project-123"}


class FakeSession:
    def post(self, url: str, data: dict[str, str], timeout: object) -> FakeResponse:
        return FakeResponse()


class TestProjectNameValidation(unittest.TestCase):
    def test_accepts_valid_project_name(self) -> None:
        self.assertEqual(create_project(FakeSession(), "http://localhost/v1", "my_project-01"), "project-123")

    def test_rejects_invalid_project_name(self) -> None:
        with self.assertRaises(SystemExit):
            create_project(FakeSession(), "http://localhost/v1", "invalid project name")


class TestUIURL(unittest.TestCase):
    def test_defaults_to_release_ui_port(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(default_ui_url("http://amc.example:8000/v1"), "http://amc.example:5000")

    def test_uses_configured_ui_port(self) -> None:
        with patch.dict(os.environ, {"AUTO_MAGIC_CALIB_UI_PORT": "5500"}, clear=True):
            self.assertEqual(default_ui_url("https://amc.example:8000/v1"), "https://amc.example:5500")

    def test_explicit_urls_keep_precedence(self) -> None:
        env = {"AMC_UI_URL": "https://ui.example/calibration/", "UI_URL": "http://other.example/",
               "AUTO_MAGIC_CALIB_UI_PORT": "invalid"}
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(default_ui_url("http://amc.example:8000/v1"), "https://ui.example/calibration")
        with patch.dict(os.environ, {"UI_URL": "http://ui.example:5500/"}, clear=True):
            self.assertEqual(default_ui_url("http://amc.example:8000/v1"), "http://ui.example:5500")

    def test_empty_port_uses_compose_default(self) -> None:
        with patch.dict(os.environ, {"AUTO_MAGIC_CALIB_UI_PORT": ""}, clear=True):
            self.assertEqual(default_ui_url("http://amc.example/v1"), "http://amc.example:5000")

    def test_ipv6_host_keeps_url_brackets(self) -> None:
        with patch.dict(os.environ, {"AUTO_MAGIC_CALIB_UI_PORT": "5500"}, clear=True):
            self.assertEqual(default_ui_url("http://[2001:db8::1]:8000/v1"), "http://[2001:db8::1]:5500")

    def test_rejects_invalid_ports(self) -> None:
        for port in ("0", "65536", "-1", "not-a-port", "55/00"):
            with self.subTest(port=port), patch.dict(os.environ, {"AUTO_MAGIC_CALIB_UI_PORT": port}, clear=True):
                with self.assertRaises(SystemExit):
                    default_ui_url("http://amc.example:8000/v1")


class TestSampleArchiveLocations(unittest.TestCase):
    def test_supports_documented_fallback_and_prefers_tracked_archive(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            amc_root = repo / "tools/auto-magic-calib"
            fallback = repo / "data/amc" / SAMPLE_ZIP
            fallback.parent.mkdir(parents=True)
            fallback.touch()
            self.assertEqual(find_sample_archive(repo, amc_root), fallback)
            tracked = amc_root / "assets" / SAMPLE_ZIP
            tracked.parent.mkdir(parents=True)
            tracked.touch()
            self.assertEqual(find_sample_archive(repo, amc_root), tracked)


class TestCaptureTLS(unittest.TestCase):
    def test_verifies_by_default(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertTrue(capture_ssl_verify())

    def test_explicit_opt_out_warns_without_secrets(self) -> None:
        output = io.StringIO()
        test_token = uuid4().hex
        with patch.dict(os.environ, {"SSL_VERIFY": "false", "AMC_ALLOW_INSECURE": "1", "VIOS_TOKEN": test_token}, clear=True):
            with contextlib.redirect_stderr(output):
                self.assertFalse(capture_ssl_verify())
        self.assertIn("WARNING", output.getvalue())
        self.assertNotIn(test_token, output.getvalue())

    def test_opt_out_requires_explicit_lab_permission(self) -> None:
        with patch.dict(os.environ, {"SSL_VERIFY": "false"}, clear=True):
            with self.assertRaises(SystemExit):
                capture_ssl_verify()


class TestDiagnosticRedaction(unittest.TestCase):
    def test_embedded_urls_and_query_tokens(self) -> None:
        line = 'failed rtsp://operator:fixture-pass@camera.example/live?token=fixture-query and rtsps://user:fixture-other@[::1]:554/live'
        result = redact_value(line)
        for secret in ('fixture-pass', 'fixture-query', 'fixture-other'):
            self.assertNotIn(secret, result)
        self.assertIn('camera.example', result)
        self.assertIn('[::1]:554', result)

    def test_failure_log_output_redacts_urls_and_env_tokens(self) -> None:
        test_token = uuid4().hex
        response = type('LogResponse', (), {
            'text': f'failed rtsp://operator:fixture-pass@camera.example/live?token=fixture-query\nvios={test_token}',
            'raise_for_status': lambda self: None,
        })()
        session = type('LogSession', (), {'get': lambda self, *args, **kwargs: response})()
        output = io.StringIO()
        with patch.dict(os.environ, {'VIOS_TOKEN': test_token}, clear=True):
            with contextlib.redirect_stdout(output):
                print_log_tail(session, 'http://localhost/v1', 'project-123')
        for secret in ('fixture-pass', 'fixture-query', test_token):
            self.assertNotIn(secret, output.getvalue())
        self.assertIn('camera.example', output.getvalue())


class TestStageTimeouts(unittest.TestCase):
    def test_defaults_and_positive_override(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(timeout_from_env('VGGT_TIMEOUT_SECONDS', 900), 900)
        with patch.dict(os.environ, {'VGGT_TIMEOUT_SECONDS': '3600'}, clear=True):
            self.assertEqual(timeout_from_env('VGGT_TIMEOUT_SECONDS', 900), 3600)

    def test_rejects_invalid_or_unbounded_values(self) -> None:
        for value in ('0', '-1', 'invalid'):
            with self.subTest(value=value), patch.dict(os.environ, {'VGGT_TIMEOUT_SECONDS': value}, clear=True):
                with self.assertRaises(SystemExit):
                    timeout_from_env('VGGT_TIMEOUT_SECONDS', 900)

    def test_disabled_vggt_does_not_parse_unused_timeout(self) -> None:
        with patch.dict(os.environ, {'VGGT_TIMEOUT_SECONDS': 'invalid'}, clear=True):
            with patch('amc_common.project_info', return_value={'video_files_count': 2, 'vggt_state': 'READY'}):
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertFalse(maybe_run_vggt(None, 'http://localhost/v1', 'project-123', False, False))


class TestVGGTDownloadPreparation(unittest.TestCase):
    def test_fresh_writable_checkout_downloads_in_place(self) -> None:
        reference = Path(__file__).resolve().parents[1] / 'references/setup.md'
        block = next(block for block in re.findall(r'^```bash\n(.*?)^```', reference.read_text(), re.M | re.S)
                     if 'VGGT_DOWNLOAD_DIR=' in block)
        block = block.replace('export DEEPSTREAM_REPO="/path/to/DeepStream"', 'export DEEPSTREAM_REPO="$FIXTURE_REPO"')
        stub = '''hf() {
  local destination="${@: -1}"
  printf 'fixture model' > "$destination/vggt_1B_commercial.pt"
}
'''
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / 'tools/auto-magic-calib').mkdir(parents=True)
            result = subprocess.run(['bash', '-c', stub + block], input=uuid4().hex + '\n',
                                    text=True, capture_output=True,
                                    env={'PATH': os.defpath, 'FIXTURE_REPO': temp_dir})
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((root / 'tools/auto-magic-calib/models/vggt/vggt_1B_commercial.pt').is_file())
            self.assertFalse((root / 'data/amc/vggt-download').exists())


class TestTuningLock(unittest.TestCase):
    def test_holder_survives_separate_calls_and_releases_on_termination(self) -> None:
        reference = Path(__file__).resolve().parents[1] / 'references/tuning.md'
        script = next(block for block in re.findall(r'^```bash\n(.*?)^```', reference.read_text(), re.M | re.S)
                      if 'exec flock' in block)
        with tempfile.TemporaryDirectory() as temp_dir:
            lock_path = Path(temp_dir) / 'tuning.lock'
            env = {'PATH': os.defpath, 'AMC_TUNING_LOCK': str(lock_path)}
            process = subprocess.Popen(['bash'], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, text=True, env=env)
            try:
                process.stdin.write(script)
                process.stdin.close()
                with lock_path.open('a') as probe:
                    deadline = time.monotonic() + 2
                    while time.monotonic() < deadline:
                        self.assertIsNone(process.poll())
                        try:
                            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
                            fcntl.flock(probe, fcntl.LOCK_UN)
                        except BlockingIOError:
                            break
                        time.sleep(0.01)
                    else:
                        self.fail('Holder failed to acquire the lock')
                    competing = subprocess.run(['bash'], input=script, capture_output=True,
                                               text=True, env=env, timeout=2)
                    self.assertNotEqual(competing.returncode, 0)
                    self.assertIsNone(process.poll())
                    process.terminate()
                    process.wait(timeout=2)
                    fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    fcntl.flock(probe, fcntl.LOCK_UN)
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=2)
                process.stdout.close()
                process.stderr.close()


class TestSampleIntegrity(unittest.TestCase):
    def test_accepts_matching_release_digest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            archive = Path(temp_dir) / 'sample.zip'
            archive.write_bytes(b'fixture-archive')
            verify_sample_archive(archive, hashlib.sha256(b'fixture-archive').hexdigest())

    def test_rejects_modified_archive(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            archive = Path(temp_dir) / 'sample.zip'
            archive.write_bytes(b'not-the-release-archive')
            with self.assertRaises(RuntimeError):
                verify_sample_archive(archive)


class TestCaptureDurationValidation(unittest.TestCase):
    def test_accepts_duration_bounds(self) -> None:
        self.assertEqual(validate_capture_duration(60), 60)
        self.assertEqual(validate_capture_duration(1200), 1200)

    def test_rejects_duration_outside_bounds(self) -> None:
        for duration in (59, 1201):
            with self.subTest(duration=duration), self.assertRaises(SystemExit):
                validate_capture_duration(duration)


class TestVideoDiscovery(unittest.TestCase):
    def test_accepts_arbitrary_mp4_names_in_sorted_order(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            video_dir = Path(temp_dir)
            expected = [video_dir / "camera-a.mp4", video_dir / "west-view.mp4"]
            for path in reversed(expected):
                path.touch()
            (video_dir / "ignore.txt").touch()

            self.assertEqual(discover_video_files(video_dir), expected)

    def test_rejects_directory_without_mp4_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir, self.assertRaises(SystemExit):
            discover_video_files(Path(temp_dir))


if __name__ == "__main__":
    unittest.main()
