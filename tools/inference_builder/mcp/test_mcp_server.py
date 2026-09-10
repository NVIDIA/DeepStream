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

import unittest
import tempfile
import os
import uuid
from pathlib import Path
from types import SimpleNamespace

import httpx

from mcp_server import ApiKeyMiddleware, InferenceBuilderMCPServer, create_sse_app
from mcp.types import CallToolRequest, CallToolRequestParams


class MCPServerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.server = InferenceBuilderMCPServer()

    async def test_list_tools_contains_expected(self):
        # Our server ignores the request payload; pass None
        result = await self.server.list_tools(None)
        tool_names = {t.name for t in result.tools}
        self.assertIn("generate_inference_pipeline", tool_names)
        self.assertIn("build_docker_image", tool_names)
        self.assertIn("generate_nvinfer_config", tool_names)

    async def test_call_unknown_tool(self):
        params = CallToolRequestParams(name="does_not_exist", arguments={})
        req = CallToolRequest(params=params)
        res = await self.server.call_tool(req)
        self.assertTrue(res.isError)
        self.assertIn("Unknown tool", res.content[0].text)

    async def test_generate_deepstream_nvinfer_config(self):
        """generate_nvinfer_config returns YAML text (no output_path needed)."""
        params = CallToolRequestParams(
            name="generate_nvinfer_config",
            arguments={
                "onnx_file": "test_model.onnx",
                "network_type": 0,  # detection
                "input_dims": "3;640;640",
                "label_file": "labels.txt",
                "precision_mode": 2,  # FP16
                "num_classes": 80,
                "custom_lib_path": "/opt/nvidia/deepstream/deepstream/lib/libnvds_infercustomparser_tao.so",
            }
        )
        req = CallToolRequest(params=params)
        res = await self.server.call_tool(req)

        self.assertFalse(res.isError, f"Tool call failed: {res.content[0].text if res.content else 'No content'}")
        content = res.content[0].text
        self.assertIn("onnx-file: test_model.onnx", content)
        self.assertIn("network-type: 0", content)
        self.assertIn("network-mode: 2", content)
        self.assertIn("infer-dims: 3;640;640", content)
        self.assertIn("num-detected-classes: 80", content)

    async def test_generate_deepstream_nvinfer_config_invalid_network_type(self):
        """Invalid network type is rejected."""
        params = CallToolRequestParams(
            name="generate_nvinfer_config",
            arguments={
                "onnx_file": "test_model.onnx",
                "network_type": 99,  # invalid
                "input_dims": "3;640;640",
                "label_file": "labels.txt",
            }
        )
        req = CallToolRequest(params=params)
        res = await self.server.call_tool(req)

        self.assertTrue(res.isError)
        self.assertIn("Invalid network_type", res.content[0].text)

    async def test_generate_deepstream_nvinfer_config_invalid_dims(self):
        """Invalid input dimensions are rejected."""
        params = CallToolRequestParams(
            name="generate_nvinfer_config",
            arguments={
                "onnx_file": "test_model.onnx",
                "network_type": 0,
                "input_dims": "3;640",  # invalid - missing dimension
                "label_file": "labels.txt",
            }
        )
        req = CallToolRequest(params=params)
        res = await self.server.call_tool(req)

        self.assertTrue(res.isError)
        self.assertIn("Invalid input_dims format", res.content[0].text)

    async def test_generate_deepstream_nvinfer_config_with_output_tensor_meta(self):
        """output_tensor_meta flag is reflected in the returned YAML text."""
        params = CallToolRequestParams(
            name="generate_nvinfer_config",
            arguments={
                "onnx_file": "custom_model.onnx",
                "network_type": 0,  # detection
                "input_dims": "3;640;640",
                "label_file": "labels.txt",
                "precision_mode": 2,
                "num_classes": 80,
                "output_tensor_meta": 1,
            }
        )
        req = CallToolRequest(params=params)
        res = await self.server.call_tool(req)

        self.assertFalse(res.isError, f"Tool call failed: {res.content[0].text if res.content else 'No content'}")
        content = res.content[0].text
        self.assertIn("output-tensor-meta: 1", content)
        self.assertIn("onnx-file: custom_model.onnx", content)

class SSETransportTests(unittest.IsolatedAsyncioTestCase):
    """Tests for the SSE/HTTP remote transport layer.

    All requests are dispatched in-process via ``httpx.ASGITransport`` — no
    real network socket is opened.  The tests cover:

    * ``ApiKeyMiddleware`` authentication logic (missing / wrong / correct key).
    * MCP SSE-transport endpoint plumbing (missing / invalid / unknown session).
    """

    async def asyncSetUp(self):
        self.server = InferenceBuilderMCPServer()

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    async def _post_messages(
        self,
        app,
        *,
        extra_headers: dict | None = None,
        params: dict | None = None,
        body: bytes = b"{}",
    ) -> httpx.Response:
        """POST to /messages/ through the ASGI app in-process.

        Content-Type is set to application/json by default because the MCP
        transport security layer rejects POST requests that lack it.
        """
        headers = {"Content-Type": "application/json"}
        if extra_headers:
            headers.update(extra_headers)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            return await client.post(
                "/messages/", headers=headers, params=params or {}, content=body
            )

    # ------------------------------------------------------------------
    # ApiKeyMiddleware — authentication
    # ------------------------------------------------------------------

    async def test_no_api_key_configured_allows_requests(self):
        """When no api_key is configured the middleware is absent; requests
        flow through to MCP handling (which returns 400, not 401)."""
        app = create_sse_app(self.server, api_key=None)
        response = await self._post_messages(app)
        self.assertNotEqual(response.status_code, 401)

    async def test_valid_api_key_allows_requests(self):
        """A correct Bearer token passes the middleware."""
        app = create_sse_app(self.server, api_key="secret-token")
        response = await self._post_messages(
            app, extra_headers={"Authorization": "Bearer secret-token"}
        )
        self.assertNotEqual(response.status_code, 401)

    async def test_missing_authorization_header_rejected(self):
        """Requests with no Authorization header are rejected with 401."""
        app = create_sse_app(self.server, api_key="secret-token")
        response = await self._post_messages(app)
        self.assertEqual(response.status_code, 401)

    async def test_wrong_api_key_rejected(self):
        """Requests with an incorrect Bearer token are rejected with 401."""
        app = create_sse_app(self.server, api_key="secret-token")
        response = await self._post_messages(
            app, extra_headers={"Authorization": "Bearer wrong-token"}
        )
        self.assertEqual(response.status_code, 401)

    async def test_non_bearer_scheme_rejected(self):
        """Non-Bearer auth schemes (e.g. Basic) are rejected with 401."""
        app = create_sse_app(self.server, api_key="secret-token")
        response = await self._post_messages(
            app, extra_headers={"Authorization": "Basic c2VjcmV0LXRva2Vu"}
        )
        self.assertEqual(response.status_code, 401)

    # ------------------------------------------------------------------
    # SSE transport /messages/ endpoint plumbing
    # ------------------------------------------------------------------

    async def test_messages_missing_session_id_returns_400(self):
        """POST /messages/ without a session_id query param → 400."""
        app = create_sse_app(self.server, api_key=None)
        response = await self._post_messages(app)
        self.assertEqual(response.status_code, 400)

    async def test_messages_invalid_session_id_returns_400(self):
        """POST /messages/ with a malformed session_id UUID → 400."""
        app = create_sse_app(self.server, api_key=None)
        response = await self._post_messages(app, params={"session_id": "not-a-uuid"})
        self.assertEqual(response.status_code, 400)

    async def test_messages_unknown_session_id_returns_404(self):
        """POST /messages/ with a valid UUID that has no live SSE session → 404."""
        app = create_sse_app(self.server, api_key=None)
        valid_uuid = uuid.uuid4().hex
        response = await self._post_messages(app, params={"session_id": valid_uuid})
        self.assertEqual(response.status_code, 404)


class ArtifactEndpointTests(unittest.IsolatedAsyncioTestCase):
    """Tests for the GET /artifact/{session_id}/{filename} download endpoint.

    All requests are dispatched in-process via httpx.ASGITransport — no real
    network socket is opened.  Tests cover:

    * Serving an existing file from a registered session workspace.
    * 404 for unknown session IDs and missing files.
    * 400 for path traversal attempts in the filename.
    * 401 when an API key is configured but the request lacks a valid token.
    """

    async def asyncSetUp(self):
        self.server = InferenceBuilderMCPServer()

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    async def _get_artifact(
        self,
        app,
        session_id: str,
        filename: str,
        *,
        extra_headers: dict | None = None,
    ) -> httpx.Response:
        headers = {"mcp-session-id": session_id, **(extra_headers or {})}
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            return await client.get(f"/{filename}", headers=headers)

    def _register_workspace(self, session_id: str, tmp_path: Path) -> None:
        """Inject a session workspace directly into the server's registry."""
        self.server._session_workspaces[session_id] = tmp_path

    # ------------------------------------------------------------------
    # Happy path
    # ------------------------------------------------------------------

    async def test_download_existing_file(self):
        """A registered session file is returned with 200 and correct content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = Path(tmpdir)
            content = b"PK\x03\x04fake-tar-gz-content"
            (ws / "my-pipeline.tgz").write_bytes(content)

            sid = uuid.uuid4().hex
            self._register_workspace(sid, ws)

            app = create_sse_app(self.server, api_key=None)
            response = await self._get_artifact(app, sid, "my-pipeline.tgz")

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.content, content)
            self.assertIn("application/", response.headers["content-type"])

    async def test_download_text_file(self):
        """Text files (e.g. config.yaml) are also served correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = Path(tmpdir)
            (ws / "config.yaml").write_text("name: test-pipeline\n")

            sid = uuid.uuid4().hex
            self._register_workspace(sid, ws)

            app = create_sse_app(self.server, api_key=None)
            response = await self._get_artifact(app, sid, "config.yaml")

            self.assertEqual(response.status_code, 200)
            self.assertIn(b"test-pipeline", response.content)

    # ------------------------------------------------------------------
    # Error paths
    # ------------------------------------------------------------------

    async def test_unknown_session_returns_404(self):
        """A session_id that has no registered workspace returns 404."""
        app = create_sse_app(self.server, api_key=None)
        response = await self._get_artifact(app, "nonexistent-session-id", "file.tgz")
        self.assertEqual(response.status_code, 404)

    async def test_missing_file_returns_404(self):
        """A valid session but a filename that does not exist returns 404."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sid = uuid.uuid4().hex
            self._register_workspace(sid, Path(tmpdir))

            app = create_sse_app(self.server, api_key=None)
            response = await self._get_artifact(app, sid, "ghost.tgz")
            self.assertEqual(response.status_code, 404)

    async def test_path_traversal_dotdot_rejected(self):
        """Filenames containing '..' are rejected (httpx normalises '..' before send, so router returns 4xx)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sid = uuid.uuid4().hex
            self._register_workspace(sid, Path(tmpdir))

            app = create_sse_app(self.server, api_key=None)
            response = await self._get_artifact(app, sid, "../secret.txt")
            self.assertGreaterEqual(response.status_code, 400)

    async def test_path_traversal_slash_rejected(self):
        """Filenames containing '/' are rejected (Starlette {filename} param doesn't capture slashes, returns 4xx)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sid = uuid.uuid4().hex
            self._register_workspace(sid, Path(tmpdir))

            app = create_sse_app(self.server, api_key=None)
            response = await self._get_artifact(app, sid, "sub/secret.txt")
            self.assertGreaterEqual(response.status_code, 400)

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------

    async def test_valid_api_key_allows_download(self):
        """A correct Bearer token is accepted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = Path(tmpdir)
            (ws / "artifact.tgz").write_bytes(b"data")

            sid = uuid.uuid4().hex
            self._register_workspace(sid, ws)

            app = create_sse_app(self.server, api_key="tok")
            response = await self._get_artifact(
                app, sid, "artifact.tgz",
                extra_headers={"Authorization": "Bearer tok"},
            )
            self.assertEqual(response.status_code, 200)

    async def test_missing_api_key_returns_401(self):
        """Requests without a token are rejected when an API key is configured."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = Path(tmpdir)
            (ws / "artifact.tgz").write_bytes(b"data")

            sid = uuid.uuid4().hex
            self._register_workspace(sid, ws)

            app = create_sse_app(self.server, api_key="tok")
            response = await self._get_artifact(app, sid, "artifact.tgz")
            self.assertEqual(response.status_code, 401)

    async def test_wrong_api_key_returns_401(self):
        """Requests with an incorrect token are rejected with 401."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = Path(tmpdir)
            (ws / "artifact.tgz").write_bytes(b"data")

            sid = uuid.uuid4().hex
            self._register_workspace(sid, ws)

            app = create_sse_app(self.server, api_key="tok")
            response = await self._get_artifact(
                app, sid, "artifact.tgz",
                extra_headers={"Authorization": "Bearer wrong"},
            )
            self.assertEqual(response.status_code, 401)


class DockerToolTests(unittest.IsolatedAsyncioTestCase):
    """Integration tests for docker_run_image, docker_stop_container, and docker_fetch_log.

    Requires Docker to be installed and accessible on the host.  Tests are
    automatically skipped when Docker is unavailable.

    alpine is used as the test image — it is tiny (~8 MB) and needs no GPU.
    Tests that require a running container start one directly via subprocess
    (without --rm) so they can be inspected and stopped independently of the
    MCP tool under test.
    """

    TEST_IMAGE = "alpine"

    # ------------------------------------------------------------------
    # Class-level Docker availability check
    # ------------------------------------------------------------------

    @classmethod
    def setUpClass(cls):
        import subprocess
        try:
            r = subprocess.run(
                ["docker", "info"], capture_output=True, timeout=10, check=False
            )
            cls._docker_available = r.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            cls._docker_available = False

    def setUp(self):
        if not self._docker_available:
            self.skipTest("Docker is not available on this host")

    async def asyncSetUp(self):
        self.server = InferenceBuilderMCPServer()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _call(self, name: str, arguments: dict):
        params = CallToolRequestParams(name=name, arguments=arguments)
        req = CallToolRequest(params=params)
        return await self.server.call_tool(req)

    @staticmethod
    def _structured(res) -> dict:
        """Parse the structured JSON block (second content item)."""
        import json
        return json.loads(res.content[-1].text)

    @staticmethod
    async def _run_detached(image: str, name: str, cmd: list | None = None) -> int:
        """Start a detached container WITHOUT --rm so it can be inspected/stopped."""
        import asyncio
        docker_cmd = ["docker", "run", "-d", "--name", name, image]
        if cmd:
            docker_cmd.extend(cmd)
        proc = await asyncio.create_subprocess_exec(
            *docker_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, _ = await asyncio.wait_for(proc.communicate(), timeout=30)
        return proc.returncode

    @staticmethod
    async def _force_remove(name: str) -> None:
        """Force-remove a container, ignoring any error (best-effort cleanup)."""
        import asyncio
        try:
            proc = await asyncio.create_subprocess_exec(
                "docker", "rm", "-f", name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await asyncio.wait_for(proc.communicate(), timeout=15)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # docker_run_image
    # ------------------------------------------------------------------

    async def test_run_image_clean_exit(self):
        """alpine echo exits 0 and structured data has container_name + exit_code."""
        res = await self._call("docker_run_image", {
            "image_name": self.TEST_IMAGE,
            "cmd": ["echo", "hello-mcp"],
            "gpus": None,
            "server_type": "serverless",
            "timeout": 30,
        })
        self.assertFalse(res.isError, res.content[0].text)
        data = self._structured(res)
        self.assertIn("container_name", data)
        self.assertEqual(data["image_name"], self.TEST_IMAGE)
        self.assertEqual(data["status"], "exited")
        self.assertEqual(data["exit_code"], 0)

    async def test_run_image_nonzero_exit(self):
        """alpine sh -c 'exit 42' is captured correctly in exit_code."""
        res = await self._call("docker_run_image", {
            "image_name": self.TEST_IMAGE,
            "cmd": ["sh", "-c", "exit 42"],
            "gpus": None,
            "server_type": "serverless",
            "timeout": 30,
        })
        data = self._structured(res)
        self.assertEqual(data["status"], "exited")
        self.assertEqual(data["exit_code"], 42)

    async def test_run_image_stdout_captured(self):
        """Output written by the container appears in the text content block."""
        res = await self._call("docker_run_image", {
            "image_name": self.TEST_IMAGE,
            "cmd": ["echo", "mcp-marker-12345"],
            "gpus": None,
            "server_type": "serverless",
            "timeout": 30,
        })
        self.assertFalse(res.isError)
        self.assertIn("mcp-marker-12345", res.content[0].text)

    async def test_run_image_timeout_leaves_container_running(self):
        """Non-serverless container that outlives timeout returns status=running."""
        name = f"ib-test-{uuid.uuid4().hex[:10]}"
        # Override deterministic name not possible via MCP args, so we rely on
        # the structured response to tell us the generated name.
        res = await self._call("docker_run_image", {
            "image_name": self.TEST_IMAGE,
            "cmd": ["sleep", "300"],
            "gpus": None,
            "server_type": "fastapi",   # non-serverless → timeout leaves it running
            "timeout": 3,
        })
        data = self._structured(res)
        self.assertFalse(res.isError, res.content[0].text)
        self.assertEqual(data["status"], "running")
        self.assertIn("container_name", data)
        # Cleanup so the container doesn't linger
        await self._force_remove(data["container_name"])

    # ------------------------------------------------------------------
    # docker_stop_container
    # ------------------------------------------------------------------

    async def test_stop_container_removes_it(self):
        """docker_stop_container removes a running container."""
        name = f"ib-test-{uuid.uuid4().hex[:10]}"
        rc = await self._run_detached(self.TEST_IMAGE, name, ["sleep", "300"])
        self.assertEqual(rc, 0, f"Failed to start test container '{name}'")
        try:
            res = await self._call("docker_stop_container", {"container_name": name})
            self.assertFalse(res.isError, res.content[0].text)
            data = self._structured(res)
            self.assertEqual(data["container_name"], name)
            self.assertEqual(data["status"], "removed")
        except Exception:
            await self._force_remove(name)
            raise

    async def test_stop_nonexistent_container_is_idempotent(self):
        """docker rm -f on a non-existent container exits 0 (idempotent on modern Docker).
        The tool should therefore succeed rather than return an error."""
        res = await self._call("docker_stop_container", {
            "container_name": "ib-test-nonexistent-zzz",
        })
        # Modern Docker (20.10+) exits 0 for 'docker rm -f <missing>' — treat as OK.
        self.assertFalse(res.isError, res.content[0].text)
        data = self._structured(res)
        self.assertIn(data["status"], {"removed", "error"})

    async def test_stop_container_with_remove_image(self):
        """remove_image=True attempts image removal; response includes image_status."""
        name = f"ib-test-{uuid.uuid4().hex[:10]}"
        rc = await self._run_detached(self.TEST_IMAGE, name, ["sleep", "300"])
        self.assertEqual(rc, 0)
        try:
            res = await self._call("docker_stop_container", {
                "container_name": name,
                "remove_image": True,
            })
            self.assertFalse(res.isError, res.content[0].text)
            data = self._structured(res)
            self.assertEqual(data["status"], "removed")
            self.assertIn("image_name", data)
            # image_status may be "removed" or "error" (e.g. alpine has other tags /
            # is referenced elsewhere) — either is acceptable; what matters is the
            # field is present and the container itself was removed.
            self.assertIn(data.get("image_status"), {"removed", "error"})
        except Exception:
            await self._force_remove(name)
            raise

    # ------------------------------------------------------------------
    # docker_fetch_log
    # ------------------------------------------------------------------

    async def test_fetch_log_running_container(self):
        """Logs from a running container are returned with timestamps."""
        name = f"ib-test-{uuid.uuid4().hex[:10]}"
        rc = await self._run_detached(
            self.TEST_IMAGE, name,
            ["sh", "-c", "echo log-line-one; echo log-line-two; sleep 300"],
        )
        self.assertEqual(rc, 0)
        try:
            import asyncio
            await asyncio.sleep(1)   # give the container a moment to emit output

            res = await self._call("docker_fetch_log", {"container_name": name})
            self.assertFalse(res.isError, res.content[0].text)
            self.assertIn("log-line-one", res.content[0].text)
            self.assertIn("log-line-two", res.content[0].text)
            data = self._structured(res)
            self.assertEqual(data["container_name"], name)
            self.assertGreaterEqual(data["line_count"], 2)
        finally:
            await self._force_remove(name)

    async def test_fetch_log_with_tail(self):
        """tail=1 returns only the last line."""
        name = f"ib-test-{uuid.uuid4().hex[:10]}"
        rc = await self._run_detached(
            self.TEST_IMAGE, name,
            ["sh", "-c", "echo first-line; echo second-line; sleep 300"],
        )
        self.assertEqual(rc, 0)
        try:
            import asyncio
            await asyncio.sleep(1)

            res = await self._call("docker_fetch_log", {
                "container_name": name,
                "tail": 1,
            })
            self.assertFalse(res.isError, res.content[0].text)
            self.assertIn("second-line", res.content[0].text)
            data = self._structured(res)
            self.assertEqual(data["line_count"], 1)
        finally:
            await self._force_remove(name)

    async def test_fetch_log_last_timestamp_for_incremental_polling(self):
        """last_timestamp is returned and can be passed back as since."""
        name = f"ib-test-{uuid.uuid4().hex[:10]}"
        rc = await self._run_detached(
            self.TEST_IMAGE, name,
            ["sh", "-c", "echo poll-line-a; sleep 300"],
        )
        self.assertEqual(rc, 0)
        try:
            import asyncio
            await asyncio.sleep(1)

            res = await self._call("docker_fetch_log", {"container_name": name})
            data = self._structured(res)
            self.assertIn("last_timestamp", data)

            # Using last_timestamp as since should yield no MORE lines than the
            # original fetch. Docker's --since is inclusive of the boundary
            # timestamp so the last line may appear once more, but nothing new.
            res2 = await self._call("docker_fetch_log", {
                "container_name": name,
                "since": data["last_timestamp"],
            })
            self.assertFalse(res2.isError)
            data2 = self._structured(res2)
            self.assertLessEqual(data2["line_count"], data["line_count"])
        finally:
            await self._force_remove(name)

    async def test_fetch_log_nonexistent_container_returns_error(self):
        """Fetching logs from a non-existent container returns isError=True."""
        res = await self._call("docker_fetch_log", {
            "container_name": "ib-test-nonexistent-log-zzz",
        })
        self.assertTrue(res.isError)
        data = self._structured(res)
        self.assertEqual(data["status"], "error")


class GetSystemInfoTests(unittest.IsolatedAsyncioTestCase):
    """Tests for the get_system_info tool.

    System info is collected once at startup via _collect_system_info and cached.
    The tool itself just returns the cached dict.
    """

    async def asyncSetUp(self):
        self.server = InferenceBuilderMCPServer()

    async def _collect(self, fake_run_subprocess):
        from unittest.mock import patch
        with patch.object(self.server, "_run_subprocess", side_effect=fake_run_subprocess):
            await self.server._collect_system_info()

    async def _call(self):
        params = CallToolRequestParams(name="get_system_info", arguments={})
        req = CallToolRequest(params=params)
        return await self.server.call_tool(req)

    async def test_returns_expected_keys(self):
        """Result is valid JSON containing arch, os, cuda_version, gpus, docker."""
        import json
        import subprocess

        smi_csv = "GPU 0, 8192\nGPU 1, 4096\n"
        smi_header = "+-------------+\n| CUDA Version: 12.4 |\n"
        docker_ver = "Docker version 24.0.5, build ced0996"

        async def fake_run(cmd, **kwargs):
            cmd0 = cmd[0] if cmd else ""
            if cmd0 == "nvidia-smi" and any(a.startswith("--query-gpu") for a in cmd):
                return subprocess.CompletedProcess(cmd, 0, smi_csv, "")
            if cmd0 == "nvidia-smi":
                return subprocess.CompletedProcess(cmd, 0, smi_header, "")
            if cmd0 == "docker":
                return subprocess.CompletedProcess(cmd, 0, docker_ver, "")
            return subprocess.CompletedProcess(cmd, 1, "", "not found")

        await self._collect(fake_run)
        res = await self._call()

        self.assertFalse(res.isError)
        data = json.loads(res.content[0].text)
        for key in ("arch", "os", "cuda_version", "gpus", "docker"):
            self.assertIn(key, data)

    async def test_gpu_list_parsed(self):
        """Each GPU entry contains name, driver_version, memory_mib, compute_capability."""
        import json
        import subprocess

        smi_csv = "NVIDIA A100-SXM4-80GB, 525.85.12, 81251, 8.0\n"

        async def fake_run(cmd, **kwargs):
            if cmd[0] == "nvidia-smi" and any(a.startswith("--query-gpu") for a in cmd):
                return subprocess.CompletedProcess(cmd, 0, smi_csv, "")
            return subprocess.CompletedProcess(cmd, 1, "", "")

        await self._collect(fake_run)
        res = await self._call()

        data = json.loads(res.content[0].text)
        self.assertEqual(len(data["gpus"]), 1)
        gpu = data["gpus"][0]
        self.assertEqual(gpu["name"], "NVIDIA A100-SXM4-80GB")
        self.assertEqual(gpu["driver_version"], "525.85.12")
        self.assertEqual(gpu["memory_mib"], "81251")
        self.assertEqual(gpu["compute_capability"], "8.0")

    async def test_no_nvidia_smi_returns_empty_gpus(self):
        """When nvidia-smi is absent, gpus list is empty and tool still succeeds."""
        import json
        import subprocess

        async def fake_run(cmd, **kwargs):
            return subprocess.CompletedProcess(cmd, 1, "", "command not found")

        await self._collect(fake_run)
        res = await self._call()

        self.assertFalse(res.isError)
        data = json.loads(res.content[0].text)
        self.assertEqual(data["gpus"], [])
        self.assertIsNone(data["cuda_version"])


class DockerRunGpuSelectionTests(unittest.IsolatedAsyncioTestCase):
    """Tests for the automatic GPU selection logic in docker_run_image."""

    async def asyncSetUp(self):
        self.server = InferenceBuilderMCPServer()

    async def _call(self, arguments: dict):
        params = CallToolRequestParams(name="docker_run_image", arguments=arguments)
        req = CallToolRequest(params=params)
        return await self.server.call_tool(req)

    async def test_selects_gpu_with_most_free_memory(self):
        """docker run is invoked with the device index that has the most free memory."""
        from unittest.mock import patch
        import subprocess

        captured_cmds = []

        # GPU 0: 4096 MiB free, GPU 1: 12000 MiB free — expect device=1
        smi_out = "0, 4096\n1, 12000\n"

        async def fake_run_subprocess(cmd, **kwargs):
            captured_cmds.append(list(cmd))
            if cmd[0] == "nvidia-smi":
                return subprocess.CompletedProcess(cmd, 0, smi_out, "")
            # Simulate docker run completing immediately
            return subprocess.CompletedProcess(cmd, 0, "ok", "")

        with patch.object(self.server, "_run_subprocess", side_effect=fake_run_subprocess):
            await self._call({"image_name": "alpine", "cmd": ["echo", "hi"]})

        docker_cmd = next(c for c in captured_cmds if c[0] == "docker")
        self.assertIn("--gpus", docker_cmd)
        gpus_idx = docker_cmd.index("--gpus")
        self.assertEqual(docker_cmd[gpus_idx + 1], "device=1")



class DockerRunLogUrlTests(unittest.IsolatedAsyncioTestCase):
    """Tests that docker_run_image saves logs and returns log_url in the response JSON."""

    async def asyncSetUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.ws = Path(self.tmpdir.name)
        self.session_id = uuid.uuid4().hex
        self.server = InferenceBuilderMCPServer(workspace_root=self.ws)
        self.server._session_workspaces[self.session_id] = self.ws
        self.server._base_url = "http://testserver"

    async def asyncTearDown(self):
        self.tmpdir.cleanup()

    async def _call(self, arguments: dict):
        import subprocess
        from unittest.mock import patch

        async def fake_run_subprocess(cmd, **kwargs):
            if cmd[0] == "nvidia-smi":
                return subprocess.CompletedProcess(cmd, 1, "", "")
            return subprocess.CompletedProcess(cmd, 0, "inference done\n", "")

        with patch.object(self.server, "_get_session_id", return_value=self.session_id), \
             patch.object(self.server, "_get_session_workspace", return_value=self.ws), \
             patch.object(self.server, "_run_subprocess", side_effect=fake_run_subprocess):
            params = CallToolRequestParams(name="docker_run_image", arguments=arguments)
            req = CallToolRequest(params=params)
            return await self.server.call_tool(req)

    async def test_log_url_in_response(self):
        """url is present in the JSON response after a successful serverless run."""
        import json

        res = await self._call({"image_name": "alpine", "cmd": ["echo", "hi"]})
        self.assertFalse(res.isError)

        data = json.loads(res.content[1].text)
        self.assertIn("url", data)
        self.assertIn("/logs/", data["url"])

    async def test_log_file_saved_to_workspace(self):
        """A .log file named after the container is written to the session workspace."""
        import json

        res = await self._call({"image_name": "alpine", "cmd": ["echo", "hi"]})
        data = json.loads(res.content[1].text)
        container_name = data["container_name"]

        log_file = self.ws / "logs" / f"{container_name}.log"
        self.assertTrue(log_file.is_file())
        self.assertIn("inference done", log_file.read_text())

    async def test_no_log_url_without_base_url(self):
        """log_url is absent when _base_url is not set (stdio mode)."""
        import json

        self.server._base_url = None
        res = await self._call({"image_name": "alpine"})
        data = json.loads(res.content[1].text)
        self.assertNotIn("log_url", data)


class LogEndpointTests(unittest.IsolatedAsyncioTestCase):
    """Tests for the GET /logs/{session_id}/{container_name} endpoint."""

    async def asyncSetUp(self):
        self.server = InferenceBuilderMCPServer()

    def _register_workspace(self, session_id: str, path: Path) -> None:
        self.server._session_workspaces[session_id] = path

    async def _get_logs(self, app, session_id: str, container_name: str) -> httpx.Response:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            return await client.get(
                f"/logs/{container_name}.log",
                headers={"mcp-session-id": session_id},
            )

    async def test_returns_log_content(self):
        """Saved log file is returned as plain text with 200."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = Path(tmpdir)
            container = "ib-run-abc123"
            log_dir = ws / "logs"
            log_dir.mkdir()
            (log_dir / f"{container}.log").write_text("stdout: all good\nstderr: \n")

            sid = uuid.uuid4().hex
            self._register_workspace(sid, ws)

            app = create_sse_app(self.server, api_key=None)
            response = await self._get_logs(app, sid, container)

            self.assertEqual(response.status_code, 200)
            self.assertIn("text/plain", response.headers["content-type"])
            self.assertIn("all good", response.text)

    async def test_unknown_session_returns_404(self):
        app = create_sse_app(self.server, api_key=None)
        response = await self._get_logs(app, "no-such-session", "ib-run-abc")
        self.assertEqual(response.status_code, 404)

    async def test_missing_log_returns_404(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sid = uuid.uuid4().hex
            self._register_workspace(sid, Path(tmpdir))

            app = create_sse_app(self.server, api_key=None)
            response = await self._get_logs(app, sid, "ib-run-notexist")
            self.assertEqual(response.status_code, 404)

    async def test_path_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sid = uuid.uuid4().hex
            self._register_workspace(sid, Path(tmpdir))

            app = create_sse_app(self.server, api_key=None)
            response = await self._get_logs(app, sid, "../etc/passwd")
            self.assertGreaterEqual(response.status_code, 400)


class SessionCleanupTests(unittest.IsolatedAsyncioTestCase):
    """Tests for _cleanup_session: workspace is removed on explicit client disconnect."""

    async def asyncSetUp(self):
        self.server = InferenceBuilderMCPServer()

    async def test_cleanup_removes_workspace_dir_and_registry(self):
        """Workspace directory and registry entry are both removed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = Path(tmpdir) / "session-ws"
            ws.mkdir()
            (ws / "pipeline.tgz").write_bytes(b"data")

            sid = uuid.uuid4().hex
            self.server._session_workspaces[sid] = ws

            await self.server._cleanup_session(sid)

            self.assertNotIn(sid, self.server._session_workspaces)
            self.assertFalse(ws.exists())

    async def test_cleanup_unknown_session_is_harmless(self):
        """Calling cleanup with an unknown session ID does not raise."""
        await self.server._cleanup_session("no-such-session-id")

    async def test_cleanup_already_deleted_dir_is_harmless(self):
        """Cleanup succeeds even if the directory was already removed from disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = Path(tmpdir) / "gone"
            # Do NOT create it on disk — simulates already-deleted workspace
            sid = uuid.uuid4().hex
            self.server._session_workspaces[sid] = ws

            await self.server._cleanup_session(sid)

            self.assertNotIn(sid, self.server._session_workspaces)


if __name__ == "__main__":
    unittest.main()
