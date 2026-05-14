import unittest
import tempfile
import os
from pathlib import Path
from types import SimpleNamespace

from mcp_server import InferenceBuilderMCPServer
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
        """Test generating a DeepStream nvinfer config file"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "nvdsinfer_config.yaml")

            params = CallToolRequestParams(
                name="generate_nvinfer_config",
                arguments={
                    "output_path": output_path,
                    "onnx_file": "test_model.onnx",
                    "network_type": 0,  # detection
                    "input_dims": "3;640;640",
                    "label_file": "labels.txt",
                    "precision_mode": 2,  # FP16
                    "num_classes": 80,
                    "custom_lib_path": "/opt/nvidia/deepstream/deepstream/lib/libnvds_infercustomparser_tao.so"
                }
            )
            req = CallToolRequest(params=params)
            res = await self.server.call_tool(req)

            # Check that the call was successful
            self.assertFalse(res.isError, f"Tool call failed: {res.content[0].text if res.content else 'No content'}")

            # Check that the file was created
            self.assertTrue(os.path.exists(output_path), f"Config file was not created at {output_path}")

            # Check file contents
            with open(output_path, 'r') as f:
                content = f.read()
                self.assertIn("onnx-file: test_model.onnx", content)
                self.assertIn("network-type: 0", content)
                self.assertIn("network-mode: 2", content)
                self.assertIn("infer-dims: 3;640;640", content)
                self.assertIn("num-detected-classes: 80", content)

    async def test_generate_deepstream_nvinfer_config_invalid_network_type(self):
        """Test that invalid network type is rejected"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "nvdsinfer_config.yaml")

            params = CallToolRequestParams(
                name="generate_nvinfer_config",
                arguments={
                    "output_path": output_path,
                    "onnx_file": "test_model.onnx",
                    "network_type": 99,  # invalid
                    "input_dims": "3;640;640",
                    "label_file": "labels.txt"
                }
            )
            req = CallToolRequest(params=params)
            res = await self.server.call_tool(req)

            # Check that the call failed
            self.assertTrue(res.isError)
            self.assertIn("Invalid network_type", res.content[0].text)

    async def test_generate_deepstream_nvinfer_config_invalid_dims(self):
        """Test that invalid input dimensions are rejected"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "nvdsinfer_config.yaml")

            params = CallToolRequestParams(
                name="generate_nvinfer_config",
                arguments={
                    "output_path": output_path,
                    "onnx_file": "test_model.onnx",
                    "network_type": 0,
                    "input_dims": "3;640",  # invalid - missing dimension
                    "label_file": "labels.txt"
                }
            )
            req = CallToolRequest(params=params)
            res = await self.server.call_tool(req)

            # Check that the call failed
            self.assertTrue(res.isError)
            self.assertIn("Invalid input_dims format", res.content[0].text)

    async def test_generate_deepstream_nvinfer_config_with_output_tensor_meta(self):
        """Test generating config with output-tensor-meta parameter"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "nvdsinfer_config.yaml")

            params = CallToolRequestParams(
                name="generate_nvinfer_config",
                arguments={
                    "output_path": output_path,
                    "onnx_file": "custom_model.onnx",
                    "network_type": 0,  # detection
                    "input_dims": "3;640;640",
                    "label_file": "labels.txt",
                    "precision_mode": 2,
                    "num_classes": 80,
                    "output_tensor_meta": 1  # Output raw tensors in DS META format
                }
            )
            req = CallToolRequest(params=params)
            res = await self.server.call_tool(req)

            # Check that the call was successful
            self.assertFalse(res.isError, f"Tool call failed: {res.content[0].text if res.content else 'No content'}")

            # Check that the file was created
            self.assertTrue(os.path.exists(output_path), f"Config file was not created at {output_path}")

            # Check file contents
            with open(output_path, 'r') as f:
                content = f.read()
                self.assertIn("output-tensor-meta: 1", content)
                self.assertIn("onnx-file: custom_model.onnx", content)

if __name__ == "__main__":
    unittest.main()
