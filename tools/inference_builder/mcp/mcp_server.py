# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
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

"""
MCP (Model Context Protocol) Server for Inference Builder

This module provides an MCP server that exposes Inference Builder functionality
to MCP-compatible clients like Cursor.
"""

import argparse
import asyncio
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.server.stdio import stdio_server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.types import (
    CallToolRequest,
    CallToolResult,
    ListToolsRequest,
    ListToolsResult,
    Tool,
    TextContent,
    Resource,
)
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response as StarletteResponse
from starlette.routing import Mount, Route
import subprocess
import logging
import time
import shutil
import uuid

# Add the builder directory to Python path
sys.path.append(str(Path(__file__).parent.parent / "builder"))


def load_allowed_servers() -> list:
    """Load allowed server types from the builder's allowed_servers.txt file.

    Returns:
        List of allowed server type strings.

    Raises:
        FileNotFoundError: If the configuration file is not found.
        ValueError: If the file is empty or contains invalid entries.
    """
    file_path = Path(__file__).parent.parent / "builder" / "allowed_servers.txt"

    if not file_path.exists():
        raise FileNotFoundError(
            f"Allowed servers configuration file not found: {file_path}"
        )

    allowed_servers = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                allowed_servers.append(line)

    if not allowed_servers:
        raise ValueError(f"No valid server types found in {file_path}")

    return allowed_servers


# Load allowed server types from configuration file
ALLOWED_SERVERS = load_allowed_servers()


class InferenceBuilderMCPServer:
    """MCP Server that exposes Inference Builder functionality"""

    def __init__(self, workspace_root: Path | None = None, model_root: Path | None = None):
        self.server = Server("deepstream-inference-builder")
        self.logger = logging.getLogger("deepstream-inference-builder")
        # Per-client workspace support (SSE transport only)
        self._workspace_root = workspace_root
        self._session_workspaces: dict[str, Path] = {}
        # Shared model repository root (across all sessions)
        self._model_root = model_root
        # Base URL set by run_sse_server so tools can construct download links
        self._base_url: str | None = None
        # System info collected once at startup
        self._system_info: dict | None = None

        # Register handlers using decorators
        self._setup_handlers()

    def _setup_handlers(self):
        """Setup MCP server handlers using decorators"""

        @self.server.list_tools()
        async def handle_list_tools():
            # The low-level server expects a list[Tool], not ListToolsResult
            result = await self.list_tools(None)
            return result.tools

        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: Dict[str, Any]):
            # Bridge to existing implementation which returns CallToolResult
            from mcp.types import CallToolRequestParams
            params = CallToolRequestParams(name=name, arguments=arguments)
            request = CallToolRequest(params=params)
            result = await self.call_tool(request)
            if getattr(result, "isError", False):
                # Let the low-level server wrap this as an error result
                message = ""
                for block in getattr(result, "content", []) or []:
                    if getattr(block, "type", "") == "text":
                        message = getattr(block, "text", "")
                        break
                raise RuntimeError(message or "Tool execution error")
            # Return unstructured content so low-level server wraps it properly
            return getattr(result, "content", [])

        @self.server.list_resources()
        async def handle_list_resources():
            """List available schema and sample resources"""
            self.logger.info("list_resources handler called")
            resources = []

            # Add base schema resources
            try:
                self.logger.info("Creating base schema resources...")
                base_resources = [
                    # Schema resources
                    Resource(
                        uri="schema://config.schema.json",
                        name="Configuration Schema",
                        description=(
                            "JSON Schema for inference builder configuration files. "
                            "Defines required fields (name, model_repo, models), "
                            "tensor specifications, backend types, and server configuration."
                        ),
                        mimeType="application/json"
                    ),
                    Resource(
                        uri="schema://readme",
                        name="Schema Documentation",
                        description=(
                            "Comprehensive documentation for configuration schemas including "
                            "examples, backend-specific parameters, data types, "
                            "preprocessors/postprocessors, and server responder configuration."
                        ),
                        mimeType="text/markdown"
                    ),
                    Resource(
                        uri="schema://index.json",
                        name="Schema Index",
                        description=(
                            "Index mapping backend types to their parameter schemas. "
                            "Use this to find the correct parameter schema for a given backend "
                            "(e.g., 'vllm' -> 'backends/parameters/vllm-parameters.schema.json'). "
                            "Read this first to understand schema navigation."
                        ),
                        mimeType="application/json"
                    ),
                    # Documentation resources
                    Resource(
                        uri="docs://README.md",
                        name="Project README",
                        description=(
                            "Main project documentation for Inference Builder. "
                            "Includes overview, getting started guide, installation instructions, "
                            "and links to examples and detailed documentation."
                        ),
                        mimeType="text/markdown"
                    ),
                    Resource(
                        uri="docs://mcp/README-MCP.md",
                        name="MCP Integration Documentation",
                        description=(
                            "Agent reference for the MCP server. "
                            "Covers the typical development workflow (project setup, sample exploration, "
                            "schema navigation, pipeline generation, container build, finalisation), "
                            "version control integration, and troubleshooting common issues."
                        ),
                        mimeType="text/markdown"
                    ),
                    Resource(
                        uri="docs://usage.md",
                        name="Usage Documentation",
                        description=(
                            "Comprehensive usage guide for Inference Builder. "
                            "Covers command line arguments, configuration file format, "
                            "model definitions, preprocessors/postprocessors, server configuration, "
                            "routing, and runtime environment variables."
                        ),
                        mimeType="text/markdown"
                    ),
                    Resource(
                        uri="docs://platform-guide.md",
                        name="Platform Guide",
                        description=(
                            "Hardware platform selection guide. "
                            "Maps target hardware (x86_64 datacenter, Jetson/Tegra, arm-sbsa servers "
                            "such as GB10/GB300/DGX Spark) to the correct Dockerfile template, "
                            "DeepStream base image, GPU architecture flags, PyTorch install method, "
                            "and CUDA version. **Read this first** when building a Docker image to "
                            "avoid runtime failures caused by mismatched base images or missing "
                            "platform libraries (e.g. libnvbufsurface on Tegra)."
                        ),
                        mimeType="text/markdown"
                    ),
                ]
                self.logger.info("Created %d base_resources objects", len(base_resources))
                for i, res in enumerate(base_resources):
                    self.logger.debug("  base_resource[%d]: uri=%s, name=%s", i, res.uri, res.name)
                resources.extend(base_resources)
                self.logger.info("After extend, resources has %d items", len(resources))
            except Exception as exc:
                self.logger.exception("Error creating base resources: %s", exc)

            # Dynamically discover backend schema resources
            try:
                schema_dir = Path(__file__).parent.parent / "schemas"
                backends_dir = schema_dir / "backends"
                self.logger.debug("Schema dir: %s (exists: %s)", schema_dir, schema_dir.exists())
                self.logger.debug("Backends dir: %s (exists: %s)", backends_dir, backends_dir.exists())
                if backends_dir.exists():
                    # Backend schemas (e.g., deepstream.schema.json, triton.schema.json)
                    for schema_file in sorted(backends_dir.glob("*.schema.json")):
                        backend_name = schema_file.stem.replace(".schema", "")
                        resources.append(Resource(
                            uri=f"schema://backends/{schema_file.name}",
                            name=f"Backend Schema: {backend_name}",
                            description=(
                                f"JSON Schema for {backend_name} backend configuration. "
                                f"Defines backend-specific parameters and settings."
                            ),
                            mimeType="application/json"
                        ))

                    # Backend parameter schemas (e.g., deepstream-parameters.schema.json)
                    params_dir = backends_dir / "parameters"
                    if params_dir.exists():
                        for param_file in sorted(params_dir.glob("*-parameters.schema.json")):
                            backend_name = param_file.stem.replace("-parameters.schema", "")
                            resources.append(Resource(
                                uri=f"schema://backends/parameters/{param_file.name}",
                                name=f"Backend Parameters: {backend_name}",
                                description=(
                                    f"JSON Schema for {backend_name} backend parameters. "
                                    f"Defines detailed parameter options for model configuration."
                                ),
                                mimeType="application/json"
                            ))
                self.logger.info("After backend discovery, resources has %d items", len(resources))
            except Exception as exc:
                self.logger.exception("Error discovering backend schemas: %s", exc)

            # Dynamically discover sample resources categorized by type
            try:
                samples_dir = Path(__file__).parent.parent / "builder" / "samples"
                self.logger.debug("Samples dir: %s (exists: %s)", samples_dir, samples_dir.exists())
                if not samples_dir.exists():
                    self.logger.warning("Samples directory not found: %s", samples_dir)
                else:
                    for sample_dir in sorted(samples_dir.iterdir()):
                        if sample_dir.is_dir():
                            sample_name = sample_dir.name
                            description = self._get_sample_description(sample_name)

                            # Collect all files recursively from the sample directory
                            # This ensures deeply nested configs like nvdsinfer_config.yaml
                            # under ds_app/*/* are exposed as resources.
                            all_files = [
                                f for f in sample_dir.rglob("*") if f.is_file()
                            ]

                            # Category 1a: DeepStream nvinfer runtime configs (e.g., nvdsinfer_config.yaml)
                            # These are NOT pipeline definitions; they are runtime nvinfer config
                            # files that belong in the model repository when DeepStream backend is used.
                            runtime_yaml_files = [
                                f
                                for f in all_files
                                if f.is_file()
                                and f.suffix == ".yaml"
                                and f.name == "nvdsinfer_config.yaml"
                            ]
                            for runtime_yaml in sorted(runtime_yaml_files):
                                rel_path = runtime_yaml.relative_to(samples_dir)
                                resources.append(Resource(
                                    uri=f"samples://runtime_config/{rel_path}",
                                    name=f"DeepStream Runtime Config for Inference: {rel_path}",
                                    description=(
                                        "DeepStream nvinfer runtime configuration file "
                                        "(nvdsinfer_config.yaml). "
                                        "This file should live in the model repository when Deepstream backend is used "
                                        "and be referenced via 'infer_config_path' in the "
                                        "DeepStream backend parameters, not used as a pipeline "
                                        "configuration YAML."
                                    ),
                                    mimeType="application/x-yaml"
                                ))

                            # Category 1b: DeepStream preprocess runtime configs (e.g., nvdspreprocess_config*.yaml)
                            # These are runtime configuration files for the nvdspreprocess plugin and
                            # should also live in the model repository when DeepStream backend is used.
                            preprocess_runtime_yaml_files = [
                                f
                                for f in all_files
                                if f.is_file()
                                and f.suffix == ".yaml"
                                and f.name.startswith("nvdspreprocess_config")
                            ]
                            for preprocess_yaml in sorted(preprocess_runtime_yaml_files):
                                rel_path = preprocess_yaml.relative_to(samples_dir)
                                resources.append(Resource(
                                    uri=f"samples://runtime_preprocess/{rel_path}",
                                    name=f"DeepStream Runtime Config for Preprocessing: {rel_path}",
                                    description=(
                                        "DeepStream nvdspreprocess runtime configuration file "
                                        "(nvdspreprocess_config*.yaml). "
                                        "This file should live in the model repository when DeepStream "
                                        "backend is used and be referenced via 'preprocess_config_path' "
                                        "in the DeepStream backend parameters, not used as a pipeline "
                                        "configuration YAML."
                                    ),
                                    mimeType="application/x-yaml"
                                ))

                            # Category 1c: OpenAPI / server specification YAML files
                            # These define the HTTP API contract (e.g., for FastAPI, Triton, NIM)
                            # and should be treated as server configuration, not pipeline configs.
                            openapi_yaml_files = [
                                f
                                for f in all_files
                                if f.is_file()
                                and f.suffix == ".yaml"
                                and (
                                    f.name in ("openapi.yaml", "openapi.yml")
                                    or f.name.endswith("_openapi.yaml")
                                    or f.name.endswith("_openapi.yml")
                                )
                            ]
                            for openapi_yaml in sorted(openapi_yaml_files):
                                rel_path = openapi_yaml.relative_to(samples_dir)
                                resources.append(Resource(
                                    uri=f"samples://openapi/{rel_path}",
                                    name=f"OpenAPI Server Spec: {rel_path}",
                                    description=(
                                        "OpenAPI specification YAML describing the HTTP API for an "
                                        "inference server (e.g., FastAPI, Triton, NIM). "
                                        "Treat this as server configuration: it defines request/response "
                                        "schemas and endpoints and is referenced via 'api_spec' when "
                                        "generating a FastAPI/NIM/TRITON server, not as a model or "
                                        "pipeline configuration."
                                    ),
                                    mimeType="application/x-yaml"
                                ))

                            # Category 1d: Pipeline / application configuration YAML files
                            yaml_files = [
                                f
                                for f in all_files
                                if f.is_file()
                                and f.suffix == ".yaml"
                                and f.name != "nvdsinfer_config.yaml"
                                and not f.name.startswith("nvdspreprocess_config")
                                and not (
                                    f.name in ("openapi.yaml", "openapi.yml")
                                    or f.name.endswith("_openapi.yaml")
                                    or f.name.endswith("_openapi.yml")
                                )
                            ]
                            for yaml_file in sorted(yaml_files):
                                rel_path = yaml_file.relative_to(samples_dir)
                                resources.append(Resource(
                                    uri=f"samples://config/{rel_path}",
                                    name=f"Sample Config: {rel_path}",
                                    description=(
                                        f"Sample pipeline/application configuration YAML for "
                                        f"{sample_name}. {description}"
                                    ) if description else f"Sample configuration: {rel_path}",
                                    mimeType="application/x-yaml"
                                ))

                            # Category 2: Dockerfiles
                            dockerfiles = [f for f in all_files if f.is_file() and f.name.startswith("Dockerfile")]
                            for dockerfile in sorted(dockerfiles):
                                rel_path = dockerfile.relative_to(samples_dir)
                                resources.append(Resource(
                                    uri=f"samples://dockerfile/{rel_path}",
                                    name=f"Sample Dockerfile: {rel_path}",
                                    description=(
                                        f"Dockerfile for {sample_name}. "
                                        f"Use as reference for container image builds."
                                    ),
                                    mimeType="text/plain"
                                ))

                            # Category 3: Processor Python files (files with "processor" in name)
                            processor_files = [
                                f for f in all_files
                                if f.is_file() and f.suffix == ".py" and "processor" in f.stem.lower()
                            ]
                            for processor in sorted(processor_files):
                                rel_path = processor.relative_to(samples_dir)
                                resources.append(Resource(
                                    uri=f"samples://processor/{rel_path}",
                                    name=f"Sample Processor: {rel_path}",
                                    description=(
                                        f"Sample preprocessor/postprocessor for {sample_name}. "
                                        f"Python module with callable classes for pipeline processing."
                                    ),
                                    mimeType="text/x-python"
                                ))

                self.logger.info("After sample discovery, resources has %d items", len(resources))
            except Exception as exc:
                self.logger.exception("Error discovering sample resources: %s", exc)

            self.logger.info("Listed %d MCP resources", len(resources))
            if len(resources) > 0:
                self.logger.info("First resource URI: %s", resources[0].uri)
                self.logger.info("Last resource URI: %s", resources[-1].uri)
            else:
                self.logger.warning("Resources list is EMPTY - this is the problem!")

            return resources

        @self.server.read_resource()
        async def handle_read_resource(uri):
            """Read a schema or sample resource by URI"""
            # Convert AnyUrl to string for string operations
            uri = str(uri)
            schema_dir = Path(__file__).parent.parent / "schemas"
            samples_dir = Path(__file__).parent.parent / "builder" / "samples"

            # Handle docs:// URIs for documentation files
            if uri.startswith("docs://"):
                project_root = Path(__file__).parent.parent
                docs_mappings = {
                    "docs://README.md": project_root / "README.md",
                    "docs://mcp/README-MCP.md": project_root / "mcp" / "README-MCP.md",
                    "docs://usage.md": project_root / "doc" / "usage.md",
                    "docs://platform-guide.md": project_root / "doc" / "platform-guide.md",
                }
                if uri in docs_mappings:
                    file_path = docs_mappings[uri]
                else:
                    raise ValueError(
                        f"Unknown docs resource URI: {uri}. "
                        f"Available: docs://README.md, docs://mcp/README-MCP.md, docs://usage.md"
                    )

            # Handle schema:// URIs
            elif uri.startswith("schema://"):
                # Static schema mappings
                uri_to_file = {
                    "schema://config.schema.json": schema_dir / "config.schema.json",
                    "schema://readme": schema_dir / "README.md",
                    "schema://index.json": schema_dir / "index.json",
                }

                if uri in uri_to_file:
                    file_path = uri_to_file[uri]
                elif uri.startswith("schema://backends/"):
                    # Handle dynamic backend schema URIs
                    rel_path = uri.replace("schema://backends/", "")
                    file_path = schema_dir / "backends" / rel_path

                    # Security check: ensure path doesn't escape backends directory
                    backends_dir = schema_dir / "backends"
                    try:
                        file_path.resolve().relative_to(backends_dir.resolve())
                    except ValueError as exc:
                        raise ValueError(f"Invalid backend schema path: {rel_path}") from exc
                else:
                    raise ValueError(
                        f"Unknown schema resource URI: {uri}. "
                        f"Use schema://config.schema.json, schema://readme, "
                        f"schema://index.json, or schema://backends/*"
                    )

            # Handle samples:// URIs (with category prefix: config/, dockerfile/, processor/, "
            # runtime_config/, runtime_preprocess/, openapi/)
            elif uri.startswith("samples://"):
                rel_path = uri.replace("samples://", "")

                # If the URI is only a category prefix (e.g., samples://runtime_config),
                # return a helpful error instead of trying to map it to a file.
                category_labels = {
                    "config": "pipeline/application configuration YAML files",
                    "dockerfile": "sample Dockerfiles",
                    "processor": "sample pre/postprocessor Python modules",
                    "runtime_config": "DeepStream nvinfer runtime configuration files (nvdsinfer_config.yaml)",
                    "runtime_preprocess": "DeepStream nvdspreprocess runtime configuration files (nvdspreprocess_config*.yaml)",
                    "openapi": "OpenAPI server specification YAML files",
                }
                rel_path_stripped = rel_path.rstrip("/")
                if rel_path_stripped in category_labels:
                    label = category_labels[rel_path_stripped]
                    raise ValueError(
                        f"'{uri}' is a category prefix for {label}, not a specific resource. "
                        f"Call list_resources to see the available 'samples://{rel_path_stripped}/...'"
                        " URIs, then use read_resource on one of those full URIs."
                    )

                # Strip category prefix if present
                for category in (
                    "config/",
                    "dockerfile/",
                    "processor/",
                    "runtime_config/",
                    "runtime_preprocess/",
                    "openapi/",
                ):
                    if rel_path.startswith(category):
                        rel_path = rel_path[len(category):]
                        break
                file_path = samples_dir / rel_path

                # Security check: ensure path doesn't escape samples directory
                try:
                    file_path.resolve().relative_to(samples_dir.resolve())
                except ValueError as exc:
                    raise ValueError(f"Invalid sample path: {rel_path}") from exc

            else:
                raise ValueError(
                    f"Unknown resource URI scheme: {uri}. "
                    f"Supported schemes: docs://, schema://, samples://"
                )

            if not file_path.exists():
                raise FileNotFoundError(f"Resource file not found: {file_path}")

            # Check if the path is a directory
            if file_path.is_dir():
                # List files in the directory to help the user
                try:
                    files = [f.name for f in file_path.iterdir() if f.is_file()]
                    files_list = "\n  - ".join(sorted(files)) if files else "(no files found)"
                    raise ValueError(
                        f"'{uri}' points to a directory, not a file. "
                        f"The MCP resource reader can only read individual files. "
                        f"Files in this directory:\n  - {files_list}\n\n"
                        f"To read a specific file, use a URI like:\n"
                        f"  {uri}/{{filename}}"
                    )
                except Exception as exc:
                    raise ValueError(
                        f"'{uri}' points to a directory, not a file. "
                        f"The MCP resource reader can only read individual files."
                    ) from exc

            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            return content

    def _sanitize_arguments(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Redact sensitive values in tool arguments for logging."""
        if not isinstance(arguments, dict):
            return {}
        sensitive_keys = {
            "api_key", "apikey", "token", "password", "secret",
            "authorization", "auth", "bearer", "access_token",
        }
        redacted: Dict[str, Any] = {}
        for key, value in arguments.items():
            if isinstance(key, str) and key.lower() in sensitive_keys:
                redacted[key] = "<redacted>"
            else:
                redacted[key] = value
        return redacted

    async def list_tools(self, _request: ListToolsRequest) -> ListToolsResult:
        """List available tools"""
        return ListToolsResult(
            tools=[
                Tool(
                    name="generate_inference_pipeline",
                    description=(
                        "Generate a deployable inference application or microservice from a YAML configuration file. "
                        "Outputs a tar.gz archive named after the 'name' field in the pipeline config "
                        "(e.g. '{name}.tgz'), written to the session workspace. "
                        "The structured response includes url in the form "
                        "'http://{mcp_server}/{filename}' — replace {mcp_server} with the MCP server "
                        "address and fetch via HTTP GET with the mcp-session-id header. "
                        "Read 'samples://*' resources for example configurations, "
                        "or 'schema://config.schema.json' resource for the full configuration schema."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "config": {
                                "type": "string",
                                "description": (
                                    "Pipeline YAML configuration as inline text. "
                                    "If the user has not provided a config, generate one based on their "
                                    "requirements: read 'schema://config.schema.json' for the full schema, "
                                    "'schema://index.json' to find backend-specific parameter schemas "
                                    "(maps backend type to 'schema://backends/parameters/*'), "
                                    "'schema://readme' for documentation, and 'samples://*' resources for "
                                    "reference examples."
                                )
                            },
                            "server_type": {
                                "type": "string",
                                "enum": ALLOWED_SERVERS,
                                "default": "serverless",
                                "description": (
                                    "Type of server to generate. Use 'serverless' for standalone applications "
                                    "designed for batch inference "
                                    "(no API spec needed and no server section should appear in the config_file). "
                                    "Use 'fastapi' for microservices that serve inference requests via HTTP endpoints "
                                    " (requires api_spec and 'server' section in config_file)."
                                    " If the user's intent is unclear, ask them to explicitly choose "
                                    "between 'serverless' and 'fastapi' before proceeding."
                                )
                            },
                            "api_spec": {
                                "type": "string",
                                "description": (
                                    "OpenAPI specification as inline YAML/JSON text. "
                                    "Required for 'fastapi', 'triton', and 'nim' server types. "
                                    "Not needed for 'serverless' type. If the user has not provided an "
                                    "OpenAPI spec, attempt to generate one based on their requirements. "
                                    "If the given information is insufficient to generate a valid spec "
                                    "(e.g., missing endpoint definitions, request/response schemas), "
                                    "ask the user to provide an OpenAPI specification."
                                )
                            },
                            "custom_modules": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": (
                                    "List of custom Python processor modules as inline code snippets. "
                                    "Each module must define classes with a 'name' attribute and '__call__' "
                                    "method to be registered in the pipeline. Determine if processors are "
                                    "needed based on the user's requirements and pipeline structure "
                                    "(e.g., input transformations, output formatting). If processors are "
                                    "required, read 'samples://processor/*' resources for reference examples "
                                    "and generate the Python code accordingly."
                                )
                            },
                        },
                        "required": ["config"]
                    }
                ),
                Tool(
                    name="build_docker_image",
                    description=(
                        "Build a Docker image from a generated inference "
                        "pipeline. The Dockerfile is expected to consume the "
                        "tar.gz archive produced by 'generate_inference_pipeline' "
                        "(for example, using 'ADD <pipeline>.tar.gz /app')."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "image_name": {
                                "type": "string",
                                "description": "Name for the Docker image"
                            },
                            "dockerfile": {
                                "type": "string",
                                "description": (
                                    "Dockerfile as inline text. Written to the session workspace, "
                                    "which also contains the pipeline tar.gz from 'generate_inference_pipeline' "
                                    "and serves as the Docker build context. "
                                    "Always generate the Dockerfile based on user requirements and "
                                    "'samples://dockerfile/*' resources. "
                                    "**Before choosing a Dockerfile template, read 'docs://platform-guide.md' "
                                    "to identify the correct base image and template for the target hardware** "
                                    "(x86_64 datacenter, Jetson/Tegra, or arm-sbsa servers like GB10/GB300/DGX Spark). "
                                    "Consider model backend (triton, vllm, tensorrt-llm, deepstream), server type "
                                    "(serverless, fastapi), and hardware platform."
                                )
                            }
                        },
                        "required": ["image_name", "dockerfile"]
                    }
                ),
                Tool(
                    name="docker_run_image",
                    description=(
                        "Optionally run a built Docker image to help with testing and "
                        "troubleshooting. This is a lightweight helper around "
                        "'docker run' that can mount a prepared model repository "
                        "from the host into the container, set environment "
                        "variables, and pass command-line arguments. "
                        "The container is started with --ipc=host so that "
                        "PyTorch / TensorRT-LLM multiprocessing can share "
                        "tensors via /dev/shm without hitting the default "
                        "64 MB Docker limit. "
                        "Use this after 'prepare_model_repository' and "
                        "'build_docker_image' have completed. "
                        "The structured response includes url in the form "
                        "'http://{mcp_server}/logs/{container}.log' — replace {mcp_server} with the "
                        "MCP server address and fetch via HTTP GET with the mcp-session-id header."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "image_name": {
                                "type": "string",
                                "description": (
                                    "Name of the Docker image to run "
                                    "(for example, 'my-inference-service:latest')."
                                )
                            },
                            "model_repo_container": {
                                "type": "string",
                                "description": (
                                    "Container path where the server's model repository is mounted. "
                                    "Must exactly match the 'model_repo' field in the pipeline config — "
                                    "that value is baked into the generated application when generate_inference_pipeline is invoked "
                                    "cannot be overridden at runtime. "
                                    "Omit if the pipeline requires no model repository."
                                ),
                            },
                            "server_type": {
                                "type": "string",
                                "description": (
                                    "High-level server type hint to choose sensible "
                                    "defaults (for example, 'serverless' or 'fastapi'). "
                                    "For non-serverless servers, the container will be "
                                    "run on the host network to make HTTP endpoints "
                                    "reachable from the host."
                                ),
                                "default": "serverless"
                            },
                            "env": {
                                "type": "object",
                                "description": (
                                    "Optional environment variables to set inside the "
                                    "container. Keys and values are serialized as "
                                    "`-e KEY=VALUE`."
                                ),
                                "additionalProperties": {
                                    "type": "string"
                                }
                            },
                            "cmd": {
                                "type": "array",
                                "description": (
                                    "Optional list of command-line arguments or an "
                                    "alternative command to pass to 'docker run' after "
                                    "the image name. For serverless flows this is "
                                    "typically the inference entrypoint arguments. "
                                    "IMPORTANT: when server_type is 'serverless', the "
                                    "user's input must be supplied via CLI arguments in "
                                    "this list. Argument names use hyphens, not "
                                    "underscores (e.g., the input named 'media_url' "
                                    "becomes '--media-url <value>'). Each flag and its "
                                    "value should be separate items, for example: "
                                    "[\"--media-url\", \"/path/to/video.mp4\"]."
                                ),
                                "items": {
                                    "type": "string"
                                }
                            },
                            "timeout": {
                                "type": "integer",
                                "description": (
                                    "Maximum time in seconds to wait for the container "
                                    "process to complete before timing out. For "
                                    "non-serverless servers this is the time to wait "
                                    "before collecting logs and stopping the container."
                                ),
                                "default": 300
                            }
                        },
                        "required": ["image_name"]
                    }
                ),
                Tool(
                    name="docker_stop_container",
                    description=(
                        "Stop and remove a running Docker container by name. "
                        "Use this to stop a container started by docker_run_image — "
                        "pass the container_name from that tool's structured response."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "container_name": {
                                "type": "string",
                                "description": "Name of the container to stop and remove.",
                            },
                            "remove_image": {
                                "type": "boolean",
                                "description": "If true, also remove the Docker image after the container is stopped. Defaults to false.",
                                "default": False,
                            },
                        },
                        "required": ["container_name"],
                    },
                ),
                Tool(
                    name="docker_fetch_log",
                    description=(
                        "Fetch logs from a running or stopped Docker container. "
                        "Use tail to limit the number of lines returned, since to fetch only "
                        "new output since a previous call (pass the last returned timestamp), "
                        "and follow_seconds to stream live output for a fixed window before "
                        "returning. Timestamps are always included in the output so the caller "
                        "can pass them back via since for incremental polling."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "container_name": {
                                "type": "string",
                                "description": "Name of the container to fetch logs from.",
                            },
                            "tail": {
                                "type": "integer",
                                "description": "Number of lines to return from the end of the log. Omit to return all available lines (or all since the since timestamp).",
                                "minimum": 1,
                            },
                            "since": {
                                "type": "string",
                                "description": (
                                    "Return only log lines after this point. Accepts a Docker "
                                    "timestamp (e.g. '2024-01-15T10:30:00Z') or a relative "
                                    "duration (e.g. '5s', '2m', '1h'). Pass the last "
                                    "timestamp from a previous call to poll incrementally."
                                ),
                            },
                            "follow_seconds": {
                                "type": "integer",
                                "description": "Follow the log stream for this many seconds before returning. Useful for capturing live output from a running container. Maximum 60.",
                                "minimum": 1,
                                "maximum": 60,
                            },
                        },
                        "required": ["container_name"],
                    },
                ),
                Tool(
                    name="prepare_model_repository",
                    description=(
                        "Optionally prepare a model repository directory for deployment. "
                        "Given a model configuration list, this tool can download model artifacts "
                        "(for example from NGC or Hugging Face) and copy associated runtime "
                        "configuration files (such as DeepStream nvdsinfer_config.yaml and "
                        "nvdspreprocess_config*.yaml) into a model repository layout that "
                        "matches the generated pipeline and Dockerfile. Use this when the "
                        "user does not already have a model repository prepared. "
                        "It is recommended to mount the model repository as a volume when running the Docker image. "
                        "**IMPORTANT: If the NGC model path or version is ambiguous or unknown, search NGC before "
                        "calling this tool. Use the NGC CLI to find the correct values: "
                        "'ngc registry model info <org>/<team>/<model_name>' to list available versions. "
                        "Always confirm the exact registry path and version tag before downloading.**"
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "model_configs": {
                                "type": "array",
                                "description": (
                                    "List of model configurations to prepare, following the same "
                                    "structure as the 'models' section in builder/tests/"
                                    "test_docker_builds.py. Each item is an object with fields: "
                                    "'name' (required, model identifier matching the name used in the "
                                    "pipeline config; the generated pipeline looks for model files in a "
                                    "folder with this name inside the model repository directory), "
                                    "'source' (optional, 'NGC' or "
                                    "'HF', default 'NGC'), 'path' (for NGC: "
                                    "registry model path; for HF: repo id like "
                                    "'Qwen/Qwen2.5-VL-3B-Instruct'), 'version' (for NGC), "
                                    "'configs' (optional, dict of {filename: inline_content} for "
                                    "runtime config files to write into the model repository), and "
                                    "'post_script' (optional, inline shell script to execute after model download)."
                                ),
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "name": {
                                            "type": "string",
                                            "description": (
                                                "The model identifier used in the pipeline configuration file "
                                                "to reference this model. The generated pipeline expects model "
                                                "files to be located in a folder with this exact name inside "
                                                "the model repository directory. Must match the name used in "
                                                "the pipeline configuration."
                                            ),
                                        },
                                        "source": {
                                            "type": "string",
                                            "enum": ["NGC", "HF"],
                                            "description": (
                                                "Model source: 'NGC' for NVIDIA NGC registry models, "
                                                "or 'HF' for Hugging Face models."
                                            ),
                                        },
                                        "path": {
                                            "type": "string",
                                            "description": (
                                                "For NGC: registry model path (for example, "
                                                "'nvidia/tao/grounding_dino'). For HF: model repo id "
                                                "(for example, 'Qwen/Qwen2.5-VL-3B-Instruct')."
                                            ),
                                        },
                                        "version": {
                                            "type": "string",
                                            "description": (
                                                "For NGC models, the version/tag to download (for example, "
                                                "'grounding_dino_swin_tiny_commercial_deployable_v1.0')."
                                            ),
                                        },
                                        "configs": {
                                            "type": "object",
                                            "additionalProperties": {"type": "string"},
                                            "description": (
                                                "Optional dict of runtime configuration files to write into the "
                                                "model repository. Keys are destination filenames (for example, "
                                                "'nvdsinfer_config.yaml', 'labels.txt'). Values are inline file "
                                                "content written directly to the model directory."
                                            ),
                                        },
                                        "post_script": {
                                            "type": "string",
                                            "description": (
                                                "Optional shell script as inline content to execute after the "
                                                "model is downloaded. Written to post_script.sh in the model "
                                                "directory and executed from there."
                                            ),
                                        },
                                    },
                                    "required": ["name"],
                                },
                            },
                        },
                        "required": ["model_configs"]
                    }
                ),
                Tool(
                    name="generate_nvinfer_config",
                    description=(
                        "Generate a DeepStream nvinfer runtime configuration file (nvdsinfer_config.yaml). "
                        "This configuration file is required by the DeepStream backend and must be placed "
                        "in the model repository. It defines inference parameters such as model file paths, "
                        "precision mode, network type, input dimensions, and custom parsers. "
                        "The generated file should be referenced via 'infer_config_path' in the DeepStream "
                        "backend parameters. "
                        "**IMPORTANT: Before calling this tool, you MUST read at least one relevant example "
                        "from 'samples://runtime_config/*' resources to understand the expected structure and "
                        "parameter conventions. Also search NGC for model information and check whether an "
                        "nvinfer configuration file (YAML or TXT) is already provided with the model. "
                        "If one exists, use the information it contains for that model.** "),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "onnx_file": {
                                "type": "string",
                                "pattern": "\\.onnx$",
                                "description": (
                                    "Name of the ONNX model file. **Must end with `.onnx`** — "
                                    "other formats (.etlt, .caffemodel, .trt, .engine, etc.) are "
                                    "not accepted. Use read_model_file to list the model directory "
                                    "and identify the correct .onnx filename if not known."
                                )
                            },
                            "precision_mode": {
                                "type": "integer",
                                "enum": [0, 1, 2],
                                "default": 2,
                                "description": (
                                    "Precision mode for inference: 0=FP32, 1=INT8, 2=FP16. "
                                    "Default is FP16 (2) for optimal performance on modern GPUs."
                                )
                            },
                            "network_type": {
                                "type": "integer",
                                "enum": [0, 1, 2, 3, 100],
                                "description": (
                                    "Network type: 0=detection, 1=classification, 2=segmentation, "
                                    "3=instance_segmentation. This determines how the model output is interpreted."
                                    "100=custom is used for models that need to output raw tensors for custom downstream processing along with output-tensor-meta: 1."
                                )
                            },
                            "input_dims": {
                                "type": "string",
                                "description": (
                                    "Model input dimensions in format 'channel;height;width' "
                                    "(e.g., '3;224;224' for a 224x224 RGB image). "
                                    "Note: the format is C;H;W, not C;W;H."
                                )
                            },
                            "label_file": {
                                "type": "string",
                                "description": (
                                    "Name of the label file (e.g., 'labels.txt'). "
                                    "This file should contain class names, one per line, and be placed "
                                    "in the same directory as the config file."
                                )
                            },
                            "custom_lib_path": {
                                "type": "string",
                                "description": (
                                    "Path to a C++ shared library (.so) that provides custom output parsing functions. "
                                    "Required for all models except classic ResNet when network_type is 0, 1, 2, or 3 (not required for network_type 100 which outputs raw tensors). "
                                    "For TAO-trained models, build the library from https://github.com/NVIDIA-AI-IOT/deepstream_tao_apps/tree/master/post_processor. "
                                    "For other models, provide the path to your own custom parser library."
                                )
                            },
                            "custom_parse_func": {
                                "type": "string",
                                "description": (
                                    "Symbol name of the custom parsing function exported by custom_lib_path. "
                                    "Behavior varies by network_type: "
                                    "(0) Detection - if omitted, the built-in parser assumes a ResNet model with 'bbox' and 'cov' output layers. "
                                    "(1) Classification - if omitted, the built-in parser treats model output as a softmax layer. "
                                    "(2) Segmentation, (3) Instance segmentation - no built-in parser available, custom_parse_func is required. "
                                    "(100) Custom - not required, raw tensors are output directly for downstream processing."
                                )
                            },
                            "num_classes": {
                                "type": "integer",
                                "description": (
                                    "Number of detected classes for the model. Required for detection and classification networks and can be deducted from the label file."
                                )
                            },
                            "gie_unique_id": {
                                "type": "integer",
                                "default": 1,
                                "description": (
                                    "Unique ID for this GIE (inference engine) in the pipeline. "
                                    "Use different IDs if you have multiple inference engines in the same pipeline."
                                )
                            },
                            "net_scale_factor": {
                                "type": "number",
                                "default": 0.00392156862745098,
                                "description": (
                                    "Scale factor for input normalization. Default is 1/255 = 0.00392156862745098 "
                                    "for models expecting [0,1] normalized inputs. IMPORTANT: This must match the "
                                    "scaling factor used during model training. Common patterns: "
                                    "(1) Simple scaling: 1/255; "
                                    "(2) With uniform std: 1/(255*std) - dividing by std; "
                                    "(3) With scale factor d: d/255 - multiplying by d; "
                                    "(4) Custom: any value matching training. "
                                    "For per-channel std normalization, consider baking into the ONNX model or using "
                                    "a custom preprocessing plugin, as DeepStream only supports a single scale factor."
                                )
                            },
                            "offsets": {
                                "type": "string",
                                "description": (
                                    "Per-channel mean subtraction values in format 'R;G;B' "
                                    "(e.g., '127.5;127.5;127.5' for [-1,1] normalization, "
                                    "'123.675;116.28;103.53' for ImageNet models). "
                                    "IMPORTANT: This must match the per-channel mean subtraction used during "
                                    "model training. If training used mean subtraction, you MUST specify this "
                                    "parameter. If no mean subtraction was used during training, omit this parameter "
                                    "or use '0;0;0'. Incorrect offsets will result in poor inference accuracy."
                                )
                            },
                            "classifier_threshold": {
                                "type": "number",
                                "default": 0.0,
                                "description": (
                                    "Confidence threshold for classification results. "
                                    "Only classifications above this threshold will be reported."
                                )
                            },
                            "input_tensor_from_meta": {
                                "type": "integer",
                                "enum": [0, 1],
                                "default": 0,
                                "description": (
                                    "Whether to read input tensor from metadata (1) or from frame buffer (0). "
                                    "Set to 1 when using nvdspreprocess for custom preprocessing."
                                )
                            },
                            "output_tensor_meta": {
                                "type": "integer",
                                "enum": [0, 1],
                                "default": 0,
                                "description": (
                                    "Whether to output in DeepStream metadata format (0) or raw tensor format (1). "
                                    "Set to 1 if raw tensors are expected as inference output, which allows downstream "
                                    "components to access raw tensor data as a Dictionary."
                                    "Set to 0 for TYPE_CUSTOM_DS_METADATA output type, where the custom parser library processes the raw tensors into detected objects."
                                )
                            }
                        },
                        "required": ["onnx_file", "network_type", "input_dims", "label_file"]
                    }
                ),
                Tool(
                    name="get_system_info",
                    description=(
                        "Query hardware and software information from the deployment machine where "
                        "the MCP server is running — which may be a remote machine, NOT necessarily "
                        "the same machine as the agent or user. Use this before building a Docker "
                        "image to identify the correct base image and platform flags for the target "
                        "hardware. Returns GPU model, driver and CUDA version, CPU architecture "
                        "(x86_64 / aarch64), OS distribution, and Docker version."
                    ),
                    inputSchema={"type": "object", "properties": {}},
                ),
                Tool(
                    name="read_model_file",
                    description=(
                        "Read the content of a file from the shared model root "
                        "(MCP_MODEL_ROOT). Use this to inspect files downloaded by "
                        "prepare_model_repository, such as nvinfer configuration files, "
                        "label files, or any other model artifacts. Paths must be relative "
                        "to the model root — absolute paths and path traversal (e.g. '../') "
                        "are not allowed. For example, after prepare_model_repository "
                        "downloads a model named 'trafficcamnet', its files are accessible "
                        "at 'trafficcamnet/{filename}'."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": (
                                    "Relative path to the file within the model root "
                                    "(for example, 'trafficcamnet/nvdsinfer_config.yaml')."
                                ),
                            }
                        },
                        "required": ["path"],
                    },
                ),
            ]
        )

    async def call_tool(self, request: CallToolRequest) -> CallToolResult:
        """Execute a tool"""
        name = request.params.name
        arguments = request.params.arguments

        start_time = time.perf_counter()
        self.logger.info(
            "tool_call_started name=%s args=%s",
            name,
            self._sanitize_arguments(arguments),
        )

        try:
            if name == "generate_inference_pipeline":
                result = await self._generate_pipeline(arguments)
            elif name == "build_docker_image":
                result = await self._build_docker_image(arguments)
            elif name == "docker_run_image":
                result = await self._docker_run_image(arguments)
            elif name == "docker_stop_container":
                result = await self._docker_stop_container(arguments)
            elif name == "docker_fetch_log":
                result = await self._docker_fetch_log(arguments)
            elif name == "prepare_model_repository":
                result = await self._prepare_model_repository(arguments)
            elif name == "generate_nvinfer_config":
                result = await self._generate_deepstream_nvinfer_config(arguments)
            elif name == "get_system_info":
                result = await self._get_system_info()
            elif name == "read_model_file":
                result = await self._read_file(arguments)
            else:
                raise ValueError(f"Unknown tool: {name}")

            duration_ms = (time.perf_counter() - start_time) * 1000.0
            self.logger.info(
                "tool_call_completed name=%s error=%s duration_ms=%.2f",
                name,
                bool(getattr(result, "isError", False)),
                duration_ms,
            )
            return self._inject_session_context(result)
        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            self.logger.exception(
                "tool_call_failed name=%s duration_ms=%.2f error=%s",
                name,
                duration_ms,
                str(e),
            )
            return CallToolResult(
                content=[TextContent(type="text", text=f"Error: {str(e)}")],
                isError=True
            )

    async def _log(self, message: str, level: str = "info") -> None:
        """Send a log notification to the connected MCP client.

        Always writes to the local logger first.  When running under the SSE/HTTP
        transport the message is also forwarded to the client as an MCP
        ``notifications/message`` event so the user sees real-time progress.
        Silently ignored when there is no active request context (stdio transport).
        """
        self.logger.info("[%s] %s", level, message)
        try:
            ctx = self.server.request_context
            await ctx.session.send_log_message(level=level, data=message)
        except Exception:
            pass  # no request context in stdio mode — local log is sufficient

    def _get_session_id(self) -> str | None:
        """Return the ``mcp-session-id`` for the current HTTP request, or None.

        Returns None when running over stdio or when the transport is stateless.
        """
        try:
            ctx = self.server.request_context
            if ctx.request is not None:
                return ctx.request.headers.get("mcp-session-id")
        except Exception:
            pass
        return None

    def _inject_session_context(self, result: "CallToolResult") -> "CallToolResult":
        """Append session_id metadata to every tool response when in HTTP mode."""
        import json
        sid = self._get_session_id()
        if not sid:
            return result
        meta = TextContent(type="text", text=json.dumps({"session_id": sid}))
        return CallToolResult(
            content=[*result.content, meta],
            isError=getattr(result, "isError", False),
        )

    def _get_session_workspace(self) -> Path | None:
        """Return (and lazily create) the workspace directory for the current session.

        The workspace is ``{workspace_root}/{session_id}/``.  Returns None when
        workspace_root is not configured or when there is no active HTTP session
        (e.g. stdio transport or stateless mode).
        """
        if self._workspace_root is None:
            return None
        session_id = self._get_session_id()
        if session_id is None:
            return None
        if session_id not in self._session_workspaces:
            ws = self._workspace_root / session_id
            ws.mkdir(parents=True, exist_ok=True)
            self.logger.info("workspace_created session=%s path=%s", session_id, ws)
            self._session_workspaces[session_id] = ws
        return self._session_workspaces[session_id]

    def _sweep_stale_workspaces(self) -> None:
        """Delete any workspace directories left over from a previous server run."""
        import shutil
        if self._workspace_root is None or not self._workspace_root.exists():
            return
        for child in self._workspace_root.iterdir():
            if child.is_dir():
                try:
                    shutil.rmtree(child)
                    self.logger.info("stale_workspace_removed path=%s", child)
                except Exception as exc:
                    self.logger.warning("stale_workspace_remove_failed path=%s: %s", child, exc)

    async def _cleanup_session(self, session_id: str) -> None:
        """Remove the workspace for a session that has explicitly disconnected."""
        import shutil
        ws = self._session_workspaces.pop(session_id, None)
        if ws is not None and ws.exists():
            try:
                await asyncio.get_running_loop().run_in_executor(None, shutil.rmtree, ws)
                self.logger.info("workspace_removed session=%s path=%s", session_id, ws)
            except Exception as exc:
                self.logger.warning("workspace_remove_failed session=%s: %s", session_id, exc)

    async def _run_subprocess(self, *args, **kwargs) -> subprocess.CompletedProcess:
        """Run subprocess.run() in a thread-pool executor.

        Identical call signature and return/exception behaviour to
        subprocess.run() — the only difference is that the asyncio event loop
        is not blocked while the child process executes, so concurrent HTTP
        clients are served normally during long-running tool calls.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, lambda: subprocess.run(*args, **kwargs)
        )

    async def _generate_pipeline(
        self, arguments: Dict[str, Any]
    ) -> CallToolResult:
        """Generate inference pipeline"""
        server_type = arguments.get("server_type", "serverless")

        workspace = self._get_session_workspace()
        output_dir = str(workspace) if workspace else "."

        ws = Path(output_dir)
        ws.mkdir(parents=True, exist_ok=True)

        config_file = str(ws / "pipeline_config.yaml")
        Path(config_file).write_text(arguments["config"])

        api_spec_file: str | None = None
        if "api_spec" in arguments:
            api_spec_file = str(ws / "api_spec.yaml")
            Path(api_spec_file).write_text(arguments["api_spec"])

        await self._log(
            f"generate_inference_pipeline: server_type={server_type} output_dir={output_dir}"
        )

        # Build command - use the same Python executable as the parent process
        cmd = [
            sys.executable, "builder/main.py",
            "--server-type", server_type,
            "-o", output_dir,
            config_file
        ]

        # Add optional arguments
        if api_spec_file is not None:
            cmd.extend(["-a", api_spec_file])

        for idx, module in enumerate(arguments.get("custom_modules", [])):
            snippet_file = ws / f"custom_module_{idx}.py"
            snippet_file.write_text(module)
            cmd.extend(["-c", str(snippet_file)])

        cmd.append("-t")

        # Execute the command - inherit the current process environment
        await self._log(f"Running: {' '.join(cmd)}")
        result = await self._run_subprocess(
            cmd,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
            env=os.environ.copy(),  # Inherit the current environment
            check=False,
        )

        import json as _json
        import yaml as _yaml

        if result.returncode == 0:
            await self._log(f"generate_inference_pipeline succeeded: output_dir={output_dir}")

            # Derive artifact path from pipeline name in the config.
            # builder/main.py writes: output_dir/{name}.tgz
            try:
                with open(config_file) as _f:
                    _cfg = _yaml.safe_load(_f)
                pipeline_name = _cfg.get("name") if isinstance(_cfg, dict) else None
            except Exception:
                pipeline_name = None

            if pipeline_name:
                artifact_path = str(Path(output_dir).resolve() / f"{pipeline_name}.tgz")
            else:
                artifact_path = output_dir

            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=f"Successfully generated inference pipeline!\n\n"
                             f"Output directory: {output_dir}\n\n"
                             f"Command executed: {' '.join(cmd)}\n\n"
                             f"Output:\n{result.stdout}"
                    ),
                    TextContent(
                        type="text",
                        text=_json.dumps({
                            "status": "success",
                            "url": f"http://{{mcp_server}}/{Path(artifact_path).name}" if self._get_session_id() else artifact_path,
                        }, indent=2),
                    ),
                ]
            )
        else:
            await self._log(
                f"generate_inference_pipeline failed (exit {result.returncode})", level="error"
            )
            try:
                with open(config_file) as _f:
                    _cfg = _yaml.safe_load(_f)
                pipeline_name = _cfg.get("name") if isinstance(_cfg, dict) else None
            except Exception:
                pipeline_name = None
            log_name = f"{pipeline_name}-generate.log" if pipeline_name else "generate_pipeline.log"
            url = None
            workspace = self._get_session_workspace()
            if workspace is not None:
                log_dir = workspace / "logs"
                log_dir.mkdir(exist_ok=True)
                (log_dir / log_name).write_text(result.stderr)
                if self._get_session_id():
                    url = f"http://{{mcp_server}}/logs/{log_name}"
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=(
                            "Failed to generate pipeline:\n\n"
                            f"Error:\n{result.stderr}\n\n"
                            f"Command: {' '.join(cmd)}"
                        )
                    ),
                    TextContent(
                        type="text",
                        text=_json.dumps({
                            "status": "error",
                            **({"url": url} if url else {}),
                        }, indent=2),
                    ),
                ],
                isError=True
            )

    def _get_sample_description(self, sample_name: str) -> str:
        """Get description for a sample"""
        descriptions = {
            "ds_app": (
                "DeepStream applications for object detection and "
                "segmentation"
            ),
            "qwen": "Qwen language models with various backends",
            "changenet": "Change detection models",
            "nvclip": "NVIDIA CLIP models for vision-language tasks",
            "tao": "NVIDIA TAO toolkit models",
            "dummy": "Dummy models for testing and development"
        }
        return descriptions.get(sample_name, "")

    async def _docker_run_image(
        self, arguments: Dict[str, Any]
    ) -> CallToolResult:
        """Run a Docker image for testing/troubleshooting."""
        image_name = arguments["image_name"]
        model_repo_container = arguments.get("model_repo_container")
        server_type = arguments.get("server_type", "serverless")
        env_vars = arguments.get("env") or {}
        cmd_args = arguments.get("cmd") or []
        timeout = int(arguments.get("timeout", 300))

        # Select the GPU with the most free memory
        gpu_device = None
        try:
            smi = await self._run_subprocess(
                ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, check=False,
            )
            # override only when nvidia-smi cooperates
            if smi.returncode == 0 and smi.stdout.strip():
                best = max(
                    (line.split(",") for line in smi.stdout.strip().splitlines()),
                    key=lambda p: int(p[1].strip()),
                )
                gpu_device = best[0].strip()
        except Exception:
            pass

        # Base docker run command (no shell, argument list only)
        # Use --ipc=host so the container shares the host's /dev/shm.
        # Docker's default 64 MB /dev/shm is insufficient for
        # PyTorch / TensorRT-LLM multiprocessing which shares tensors
        # via shared memory, causing SIGBUS when /dev/shm is exhausted.
        # Assign a deterministic name so we can force-remove the container
        # on timeout — subprocess.TimeoutExpired kills the `docker run`
        # client process but leaves the container running with GPUs held.
        container_name = f"ib-run-{uuid.uuid4().hex[:12]}"
        cmd: list[str] = [
            "docker", "run", "--rm", "--ipc=host",
            "--name", container_name,
        ]

        if gpu_device is None:
            # fall back to --gpus all, FIXME: in some cases nvidia-smi returns empty memory usage
            cmd.extend(["--gpus", "all"])
        else:
            # use the selected GPU
            cmd.extend(["--gpus", f"device={gpu_device}"])

        # Network: for serverless we still use host network so local clients can connect
        cmd.append("--network=host")

        # Mount model repository if configured, non-empty, and container path is specified
        if (model_repo_container and self._model_root
                and self._model_root.is_dir() and any(self._model_root.iterdir())):
            cmd.extend(["-v", f"{self._model_root}:{model_repo_container}"])

        # Environment variables
        if isinstance(env_vars, dict):
            for key, value in env_vars.items():
                cmd.extend(["-e", f"{key}={value}"])

        # Image name and optional command/args
        cmd.append(image_name)
        if isinstance(cmd_args, list):
            cmd.extend([str(a) for a in cmd_args])

        self.logger.info(
            "docker_run_invoked image=%s cmd=%s",
            image_name,
            " ".join(cmd),
        )
        await self._log(
            f"Starting Docker container '{container_name}' from image '{image_name}' "
            f"(timeout={timeout}s)"
        )

        try:
            result = await self._run_subprocess(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except FileNotFoundError:
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=(
                            "Docker not found. Please ensure Docker is installed "
                            "and available in your PATH before using docker_run_image."
                        ),
                    )
                ],
                isError=True,
            )
        except subprocess.TimeoutExpired:
            await self._log(
                f"Docker run timed out after {timeout}s for '{image_name}'", level="warning"
            )
            is_serverless = server_type == "serverless"
            if is_serverless:
                # Serverless containers should finish within the timeout.
                # The `docker run` client is dead but the container keeps
                # running (and holding GPU memory).  Force-remove it.
                import json as _json
                # Fetch logs before removing so the client can diagnose the timeout
                logs_result = await self._run_subprocess(
                    ["docker", "logs", "--timestamps", container_name],
                    capture_output=True, text=True, check=False,
                )
                logs = (logs_result.stdout + logs_result.stderr).strip() or None

                container_removed = False
                try:
                    await self._run_subprocess(
                        ["docker", "rm", "-f", container_name],
                        capture_output=True, timeout=30, check=False,
                    )
                    container_removed = True
                except Exception as cleanup_exc:
                    self.logger.warning(
                        "Failed to remove timed-out container %s: %s",
                        container_name, cleanup_exc,
                    )
                # Save logs to workspace
                log_path = None
                workspace = self._get_session_workspace()
                if workspace is not None and logs:
                    log_dir = workspace / "logs"
                    log_dir.mkdir(exist_ok=True)
                    log_file = log_dir / f"{container_name}.log"
                    log_file.write_text(logs)
                    log_path = f"http://{{mcp_server}}/logs/{container_name}.log"

                return CallToolResult(
                    content=[
                        TextContent(
                            type="text",
                            text=_json.dumps({
                                "container_name": container_name,
                                "image_name": image_name,
                                "status": "timeout",
                                "container_removed": container_removed,
                                "logs": logs,
                                **({"url": log_path} if log_path else {}),
                            }, indent=2),
                        )
                    ],
                    isError=True,
                )
            else:
                # Non-serverless (persistent) servers are expected to keep
                # running past the timeout — that means the server started
                # successfully.  Leave the container running.
                import json as _json
                return CallToolResult(
                    content=[
                        TextContent(
                            type="text",
                            text=(
                                f"Container '{container_name}' is running "
                                f"(image '{image_name}'). The server appears to "
                                f"have started successfully (still alive after "
                                f"{timeout}s). To stop it later: "
                                f"docker rm -f {container_name}"
                            ),
                        ),
                        TextContent(
                            type="text",
                            text=_json.dumps({
                                "container_name": container_name,
                                "image_name": image_name,
                                "gpu_device": gpu_device,
                                "status": "running",
                            }, indent=2),
                        ),
                    ],
                    isError=False,
                )
        except Exception as exc:
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=f"Exception while running Docker image '{image_name}': {exc}",
                    )
                ],
                isError=True,
            )

        await self._log(
            f"Docker run finished for '{image_name}' (exit code {result.returncode})",
            level="info" if result.returncode == 0 else "error",
        )
        import json as _json
        text = (
            f"docker run command: {' '.join(cmd)}\n\n"
            f"exit code: {result.returncode}\n\n"
            f"stdout:\n{result.stdout}\n\n"
            f"stderr:\n{result.stderr}"
        )

        # Save logs to session workspace so they can be fetched via the /logs endpoint
        log_path = None
        workspace = self._get_session_workspace()
        if workspace is not None:
            log_dir = workspace / "logs"
            log_dir.mkdir(exist_ok=True)
            log_file = log_dir / f"{container_name}.log"
            log_file.write_text(text)
            log_path = f"http://{{mcp_server}}/logs/{container_name}.log"

        return CallToolResult(
            content=[
                TextContent(type="text", text=text),
                TextContent(type="text", text=_json.dumps({
                    "container_name": container_name,
                    "image_name": image_name,
                    "gpu_device": gpu_device,
                    "status": "exited",
                    "exit_code": result.returncode,
                    **({"url": log_path} if log_path else {}),
                }, indent=2)),
            ],
            isError=result.returncode != 0,
        )

    async def _docker_stop_container(self, arguments: Dict[str, Any]) -> CallToolResult:
        """Stop and remove a running Docker container, optionally removing the image."""
        import json as _json
        container_name = arguments["container_name"]
        remove_image = arguments.get("remove_image", False)
        await self._log(f"Stopping container '{container_name}'")

        # First, get the image name before removing the container (needed if remove_image=True)
        image_name = None
        if remove_image:
            inspect = await self._run_subprocess(
                ["docker", "inspect", "--format", "{{.Config.Image}}", container_name],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            if inspect.returncode == 0:
                image_name = inspect.stdout.strip()

        try:
            result = await self._run_subprocess(
                ["docker", "rm", "-f", container_name],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            if result.returncode != 0:
                return CallToolResult(
                    content=[
                        TextContent(type="text", text=f"Failed to stop container '{container_name}': {result.stderr.strip()}"),
                        TextContent(type="text", text=_json.dumps({
                            "container_name": container_name,
                            "status": "error",
                            "error": result.stderr.strip(),
                        }, indent=2)),
                    ],
                    isError=True,
                )

            structured: Dict[str, Any] = {"container_name": container_name, "status": "removed"}
            messages = [f"Container '{container_name}' stopped and removed."]

            if remove_image and image_name:
                await self._log(f"Removing image '{image_name}'")
                rmi = await self._run_subprocess(
                    ["docker", "rmi", image_name],
                    capture_output=True,
                    text=True,
                    timeout=60,
                    check=False,
                )
                if rmi.returncode == 0:
                    structured["image_name"] = image_name
                    structured["image_status"] = "removed"
                    messages.append(f"Image '{image_name}' removed.")
                else:
                    structured["image_name"] = image_name
                    structured["image_status"] = "error"
                    structured["image_error"] = rmi.stderr.strip()
                    messages.append(f"Failed to remove image '{image_name}': {rmi.stderr.strip()}")

            return CallToolResult(
                content=[
                    TextContent(type="text", text="\n".join(messages)),
                    TextContent(type="text", text=_json.dumps(structured, indent=2)),
                ]
            )
        except FileNotFoundError:
            return CallToolResult(
                content=[TextContent(type="text", text="Docker not found. Please ensure Docker is installed.")],
                isError=True,
            )
        except Exception as exc:
            return CallToolResult(
                content=[TextContent(type="text", text=f"Exception stopping container '{container_name}': {exc}")],
                isError=True,
            )

    async def _docker_fetch_log(self, arguments: Dict[str, Any]) -> CallToolResult:
        """Fetch logs from a Docker container, with optional tail/since/follow."""
        import json as _json
        container_name = arguments["container_name"]
        tail = arguments.get("tail")
        since = arguments.get("since")
        follow_seconds = arguments.get("follow_seconds")

        cmd = ["docker", "logs", "--timestamps", container_name]
        if tail is not None:
            cmd.extend(["--tail", str(tail)])
        if since is not None:
            cmd.extend(["--since", since])
        if follow_seconds is not None:
            cmd.append("--follow")

        timeout = (follow_seconds or 0) + 10

        await self._log(f"Fetching logs for container '{container_name}'")
        try:
            result = await self._run_subprocess(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except FileNotFoundError:
            return CallToolResult(
                content=[TextContent(type="text", text="Docker not found. Please ensure Docker is installed.")],
                isError=True,
            )
        except Exception as exc:
            return CallToolResult(
                content=[TextContent(type="text", text=f"Exception fetching logs for '{container_name}': {exc}")],
                isError=True,
            )

        # docker logs writes log lines to stderr by default; stdout is only used
        # when the container was started with a TTY.  Merge both streams so callers
        # always see all output regardless of how the container was started.
        log_text = (result.stdout or "") + (result.stderr or "")

        if result.returncode != 0:
            error_msg = result.stderr.strip() or result.stdout.strip() or "Unknown error"
            return CallToolResult(
                content=[
                    TextContent(type="text", text=f"Failed to fetch logs for '{container_name}': {error_msg}"),
                    TextContent(type="text", text=_json.dumps({
                        "container_name": container_name,
                        "status": "error",
                        "error": error_msg,
                    }, indent=2)),
                ],
                isError=True,
            )

        lines = log_text.splitlines()
        # Extract the timestamp of the last line so the caller can use it for incremental polling
        last_timestamp = None
        for line in reversed(lines):
            parts = line.split(" ", 1)
            if parts and parts[0].endswith("Z"):
                last_timestamp = parts[0]
                break

        structured: Dict[str, Any] = {
            "container_name": container_name,
            "line_count": len(lines),
        }
        if last_timestamp:
            structured["last_timestamp"] = last_timestamp

        return CallToolResult(
            content=[
                TextContent(type="text", text=log_text if log_text.strip() else "(no log output)"),
                TextContent(type="text", text=_json.dumps(structured, indent=2)),
            ]
        )

    async def _prepare_model_repository(
        self, arguments: Dict[str, Any]
    ) -> CallToolResult:
        """Prepare model repositories by downloading models and copying configs.

        This mirrors the 'models' handling in builder/tests/test_docker_builds.py,
        but is exposed as an optional MCP tool so agents can help set up
        model repositories before Docker image builds.
        """
        # New interface: model_configs is a list of dicts (each includes 'name')
        # Backward-compat: accept legacy 'models_config' dict keyed by model name.
        model_configs = arguments.get("model_configs")
        legacy_models_config = arguments.get("models_config")

        if model_configs is None and legacy_models_config is not None:
            # Convert dict[name -> config] into list[{name, ...config}]
            if isinstance(legacy_models_config, dict):
                converted: list[dict[str, Any]] = []
                for legacy_name, legacy_info in legacy_models_config.items():
                    if isinstance(legacy_info, dict):
                        merged = dict(legacy_info)
                        merged["name"] = legacy_name
                        converted.append(merged)
                model_configs = converted
            else:
                model_configs = []

        if not isinstance(model_configs, list) or not model_configs:
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=(
                            "No models specified in model_configs. "
                            "Nothing to prepare for model repositories."
                        ),
                    )
                ]
            )

        if self._model_root is None:
            return CallToolResult(
                content=[TextContent(type="text", text=(
                    "Model root is not configured. "
                    "Start the server with --model-root or set MCP_MODEL_ROOT."
                ))],
                isError=True,
            )

        model_root = self._model_root
        model_root.mkdir(parents=True, exist_ok=True)

        messages: list[str] = []
        model_results: list[dict] = []
        total = len(model_configs)
        await self._log(f"prepare_model_repository: processing {total} model(s)")

        for idx, model_info in enumerate(model_configs, start=1):
            try:
                if not isinstance(model_info, dict):
                    messages.append(
                        "❌ Skipping model entry: config must be an object"
                    )
                    continue

                model_name = model_info.get("name")
                if not model_name or not isinstance(model_name, str):
                    messages.append(
                        "❌ Skipping model entry: missing required 'name'"
                    )
                    continue

                await self._log(f"[{idx}/{total}] Preparing model '{model_name}'")

                source = str(model_info.get("source", "NGC")).upper()
                final_model_dir = model_root / model_name

                # Skip download only if the directory exists AND contains files.
                if final_model_dir.is_dir() and any(final_model_dir.rglob("*")):
                    messages.append(
                        f"✅ Model '{model_name}' already exists at {final_model_dir}, "
                        f"skipping download"
                    )
                else:
                    final_model_dir.mkdir(parents=True, exist_ok=True)
                    if source == "HF":
                        # Hugging Face model download, based on DockerBuildTester.download_models
                        hf_repo = model_info.get("path")
                        if not hf_repo:
                            messages.append(
                                f"❌ Skipping HF model '{model_name}': missing 'path'"
                            )
                            continue

                        messages.append(
                            f"📥 Downloading Hugging Face model '{model_name}' "
                            f"({hf_repo}) to {final_model_dir}"
                        )

                        # Use huggingface_hub.snapshot_download instead of git clone
                        # so that actual binary weights are fetched via the HF CDN
                        # (git clone only retrieves LFS pointer stubs unless a full
                        # LFS pull is performed, which fails for gated repos without
                        # proper credential configuration).
                        try:
                            from huggingface_hub import snapshot_download
                            snapshot_download(
                                repo_id=hf_repo,
                                local_dir=str(final_model_dir),
                                ignore_patterns=["*.pt", "*.bin", "original/*"],
                            )
                        except Exception as e:
                            messages.append(
                                f"❌ Failed to download HF model '{model_name}': {e}"
                            )
                            continue
                    else:
                        # Default to NGC
                        ngc_path = model_info.get("path")
                        version = model_info.get("version")
                        if not ngc_path or not version:
                            messages.append(
                                f"❌ Skipping NGC model '{model_name}': "
                                f"both 'path' and 'version' are required"
                            )
                            continue

                        full_ngc_path = f"{ngc_path}:{version}"
                        messages.append(
                            f"📥 Downloading NGC model '{model_name}' "
                            f"({full_ngc_path}) to {final_model_dir}"
                        )

                        # Run NGC CLI to download the model version
                        download_cmd = [
                            "ngc",
                            "registry",
                            "model",
                            "download-version",
                            full_ngc_path,
                        ]
                        result = await self._run_subprocess(
                            download_cmd,
                            capture_output=True,
                            text=True,
                            timeout=900,
                            check=False,
                        )
                        if result.returncode != 0:
                            messages.append(
                                f"❌ Failed to download NGC model '{model_name}': "
                                f"{result.stderr.strip()}"
                            )
                            continue

                        # Infer downloaded folder name as in test_docker_builds.py
                        model_base_name = ngc_path.split("/")[-1]
                        downloaded_folder = Path(
                            f"{model_base_name}_v{version}"
                        )
                        if not downloaded_folder.exists():
                            messages.append(
                                f"❌ Downloaded folder not found for model "
                                f"'{model_name}': {downloaded_folder}"
                            )
                            continue

                        try:
                            shutil.move(str(downloaded_folder), str(final_model_dir))
                        except Exception as exc:
                            messages.append(
                                f"❌ Failed to move downloaded model '{model_name}' "
                                f"into {final_model_dir}: {exc}"
                            )
                            continue

                configs_dict = model_info.get("configs")
                if configs_dict and isinstance(configs_dict, dict):
                    for dest_name, cfg_value in configs_dict.items():
                        dest = final_model_dir / dest_name
                        try:
                            final_model_dir.mkdir(parents=True, exist_ok=True)
                            dest.write_text(cfg_value)
                            messages.append(f"📄 Wrote config → {dest_name}")
                        except Exception as exc:
                            messages.append(
                                f"⚠️ Failed to write config '{dest_name}' for '{model_name}': {exc}"
                            )

                post_script = model_info.get("post_script")
                if post_script:
                    script_path = final_model_dir / "post_script.sh"
                    final_model_dir.mkdir(parents=True, exist_ok=True)
                    script_path.write_text(post_script)
                    script_path.chmod(0o755)
                    script_cmd = str(script_path)
                    messages.append(
                        f"🔧 Executing post-script for '{model_name}': {script_cmd}"
                    )
                    try:
                        script_result = await self._run_subprocess(
                            script_cmd,
                            shell=True,
                            capture_output=True,
                            text=True,
                            cwd=str(final_model_dir),
                            timeout=600,
                            check=False,
                        )
                        if script_result.returncode == 0:
                            messages.append(
                                f"✅ Post-script executed successfully for '{model_name}'"
                            )
                            if script_result.stdout.strip():
                                messages.append(
                                    f"   Output: {script_result.stdout.strip()}"
                                )
                        else:
                            messages.append(
                                f"❌ Post-script failed for '{model_name}' "
                                f"(exit code {script_result.returncode}): "
                                f"{script_result.stderr.strip()}"
                            )
                    except subprocess.TimeoutExpired:
                        messages.append(
                            f"❌ Post-script timed out for '{model_name}' after 600 seconds"
                        )
                    except Exception as exc:
                        messages.append(
                            f"❌ Failed to execute post-script for '{model_name}': {exc}"
                        )

                messages.append(
                    f"✅ Prepared model repository for '{model_name}' at "
                    f"{final_model_dir}"
                )
                await self._log(f"[{idx}/{total}] Model '{model_name}' ready at {final_model_dir}")
                files = sorted(
                    str(f.relative_to(final_model_dir))
                    for f in final_model_dir.rglob("*") if f.is_file()
                )
                model_results.append({
                    "name": model_name,
                    "host_path": str(final_model_dir),
                    "files": files,
                    "status": "success",
                })

            except Exception as exc:  # Catch-all per model
                messages.append(
                    f"❌ Exception while preparing model '{model_name}': {exc}"
                )
                try:
                    _failed_path = str(final_model_dir)
                except NameError:
                    _failed_path = None
                model_results.append({
                    "name": model_name,
                    "host_path": _failed_path,
                    "files": [],
                    "status": "failed",
                    "error": str(exc),
                })

        import json as _json
        summary = "\n".join(messages)
        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text=(
                        "Model repository preparation completed.\n\n"
                        f"{summary}"
                    ),
                ),
                TextContent(
                    type="text",
                    text=_json.dumps({"models": model_results}, indent=2),
                ),
            ]
        )

    async def _build_docker_image(
        self, arguments: Dict[str, Any]
    ) -> CallToolResult:
        """Build Docker image from generated pipeline"""
        image_name = arguments["image_name"]

        ws = self._get_session_workspace()
        if ws is None:
            import tempfile
            ws = Path(tempfile.mkdtemp(prefix="ib-docker-"))
        ws.mkdir(parents=True, exist_ok=True)
        dockerfile_path = ws / "Dockerfile"
        dockerfile_path.write_text(arguments["dockerfile"])

        build_context = str(ws)

        # Build Docker image
        cmd = [
            "docker", "build",
            "-f", str(dockerfile_path),
            "-t", image_name,
            build_context
        ]
        self.logger.info(
            "docker_build_invoked image=%s dockerfile=%s context=%s",
            image_name,
            dockerfile_path,
            build_context,
        )
        await self._log(f"Building Docker image '{image_name}' (context: {build_context})")

        try:
            result = await self._run_subprocess(
                cmd,
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent,
                check=False,
            )

            if result.returncode == 0:
                await self._log(f"Docker image '{image_name}' built successfully")
                return CallToolResult(
                    content=[
                        TextContent(
                            type="text",
                            text=(
                                f"Successfully built Docker image "
                                f"'{image_name}'!\n\n"
                                f"Command executed: {' '.join(cmd)}\n\n"
                                f"Output:\n{result.stdout}"
                            )
                        )
                    ]
                )
            else:
                await self._log(
                    f"Docker build failed for '{image_name}' (exit {result.returncode})",
                    level="error",
                )
                return CallToolResult(
                    content=[
                        TextContent(
                            type="text",
                            text=f"Failed to build Docker image:\n\n"
                                 f"Error:\n{result.stderr}\n\n"
                                 f"Command: {' '.join(cmd)}"
                        )
                    ],
                    isError=True
                )
        except FileNotFoundError:
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=(
                            "Docker not found. Please ensure Docker is "
                            "installed and available in your PATH."
                        )
                    )
                ],
                isError=True
            )

    async def _collect_system_info(self) -> None:
        """Collect system info once at startup and cache in self._system_info."""
        import platform

        async def _run(cmd):
            try:
                r = await self._run_subprocess(
                    cmd, capture_output=True, text=True, check=False, timeout=10
                )
                return r.stdout.strip() if r.returncode == 0 else None
            except Exception:
                return None

        # GPU info: one row per GPU
        smi_out = await _run([
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,compute_cap",
            "--format=csv,noheader,nounits",
        ])
        gpus = []
        if smi_out:
            for line in smi_out.splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) == 4:
                    gpus.append({
                        "name": parts[0],
                        "driver_version": parts[1],
                        "memory_mib": parts[2],
                        "compute_capability": parts[3],
                    })

        # CUDA version from the plain nvidia-smi output header
        smi_header = await _run(["nvidia-smi"])
        cuda_version = None
        if smi_header:
            for line in smi_header.splitlines():
                if "CUDA Version" in line:
                    cuda_version = line.split(":")[-1].strip()
                    break

        # Docker version
        docker_out = await _run(["docker", "--version"])

        # OS info from /etc/os-release
        os_info = {}
        try:
            with open("/etc/os-release") as f:
                for line in f:
                    k, _, v = line.strip().partition("=")
                    os_info[k] = v.strip('"')
        except Exception:
            pass

        self._system_info = {
            "arch": platform.machine(),
            "os": {
                "id": os_info.get("ID"),
                "version": os_info.get("VERSION_ID"),
                "pretty_name": os_info.get("PRETTY_NAME"),
            },
            "cuda_version": cuda_version,
            "gpus": gpus,
            "docker": docker_out,
        }
        self.logger.info("system_info_collected arch=%s gpus=%d", self._system_info["arch"], len(gpus))

    async def _get_system_info(self) -> CallToolResult:
        import json as _json
        if self._system_info is None:
            # Still collecting — wait up to 30s for the background task to finish
            for _ in range(30):
                await asyncio.sleep(1)
                if self._system_info is not None:
                    break
        return CallToolResult(
            content=[TextContent(type="text", text=_json.dumps(self._system_info, indent=2))]
        )

    async def _read_file(self, arguments: Dict[str, Any]) -> CallToolResult:
        """Read a file from the shared model root."""
        import json as _json

        def _result(data: dict, is_error: bool = False) -> CallToolResult:
            return CallToolResult(
                content=[TextContent(type="text", text=_json.dumps(data, indent=2))],
                isError=is_error,
            )

        if self._model_root is None:
            return _result({"path": arguments.get("path"), "error": (
                "Model root is not configured. "
                "Start the server with --model-root or set MCP_MODEL_ROOT."
            )}, is_error=True)

        rel_path = arguments["path"]
        model_root = self._model_root.resolve()

        try:
            target = (model_root / rel_path).resolve()
            target.relative_to(model_root)  # path-traversal guard
        except ValueError:
            return _result({"path": rel_path, "error": f"Access denied: '{rel_path}' is outside the model root."}, is_error=True)

        if not target.is_file():
            return _result({"path": rel_path, "error": f"File not found: '{rel_path}'"}, is_error=True)

        # Detect binary files by checking for null bytes in the first chunk
        try:
            raw = target.read_bytes()
            if b"\x00" in raw[:8192]:
                return _result({"path": rel_path, "error": f"'{rel_path}' is a binary file and cannot be read as text."}, is_error=True)
            text = raw.decode("utf-8", errors="replace")
            return _result({"path": rel_path, "content": text})
        except Exception as exc:
            return _result({"path": rel_path, "error": f"Failed to read '{rel_path}': {exc}"}, is_error=True)

    async def _generate_deepstream_nvinfer_config(
        self, arguments: Dict[str, Any]
    ) -> CallToolResult:
        """Generate DeepStream nvinfer runtime configuration file"""
        import yaml

        # Extract required parameters
        onnx_file = arguments["onnx_file"]
        if not onnx_file.endswith(".onnx"):
            return CallToolResult(
                content=[TextContent(type="text", text=(
                    f"Invalid onnx_file '{onnx_file}': must end with '.onnx'. "
                    f"Use read_model_file to inspect the model directory and locate the correct .onnx file."
                ))],
                isError=True,
            )
        network_type = arguments["network_type"]
        input_dims = arguments["input_dims"]
        label_file = arguments["label_file"]

        # Extract optional parameters with defaults
        precision_mode = arguments.get("precision_mode", 2)  # Default FP16
        custom_lib_path = arguments.get("custom_lib_path", "")
        custom_parse_func = arguments.get("custom_parse_func", "")
        num_classes = arguments.get("num_classes")
        gie_unique_id = arguments.get("gie_unique_id", 1)
        net_scale_factor = arguments.get("net_scale_factor", 0.00392156862745098)
        offsets = arguments.get("offsets")
        classifier_threshold = arguments.get("classifier_threshold", 0.0)
        input_tensor_from_meta = arguments.get("input_tensor_from_meta", 0)
        output_tensor_meta = arguments.get("output_tensor_meta", 0)

        # Validate network_type
        network_type_names = {
            0: "detection",
            1: "classification",
            2: "segmentation",
            3: "instance_segmentation",
            100: "custom"
        }
        if network_type not in network_type_names:
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=f"Invalid network_type: {network_type}. Must be 0 (detection), "
                             f"1 (classification), 2 (segmentation), 3 (instance_segmentation), "
                             f"or 100 (custom)."
                    )
                ],
                isError=True
            )

        # Validate precision_mode
        precision_names = {0: "FP32", 1: "INT8", 2: "FP16"}
        if precision_mode not in precision_names:
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=f"Invalid precision_mode: {precision_mode}. Must be 0 (FP32), 1 (INT8), or 2 (FP16)."
                    )
                ],
                isError=True
            )

        # Validate input_dims format (should be channel;height;width)
        dims_parts = input_dims.split(';')
        if len(dims_parts) != 3:
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=f"Invalid input_dims format: '{input_dims}'. "
                             f"Expected format: 'channel;height;width' (e.g., '3;224;224')"
                    )
                ],
                isError=True
            )

        try:
            # Validate dimensions are integers
            for dim in dims_parts:
                int(dim)
        except ValueError:
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=f"Invalid input_dims: '{input_dims}'. All dimensions must be integers."
                    )
                ],
                isError=True
            )

        # Build the configuration
        config = {
            "property": {
                "gie-unique-id": gie_unique_id,
                "net-scale-factor": net_scale_factor,
                "onnx-file": onnx_file,
                "network-mode": precision_mode,
                "network-type": network_type,
                "infer-dims": input_dims,
                "labelfile-path": label_file,
            }
        }

        # Add optional fields
        if offsets:
            config["property"]["offsets"] = offsets

        if num_classes is not None:
            config["property"]["num-detected-classes"] = num_classes

        if input_tensor_from_meta:
            config["property"]["input-tensor-from-meta"] = input_tensor_from_meta

        if output_tensor_meta:
            config["property"]["output-tensor-meta"] = output_tensor_meta

        if network_type == 1:  # classification
            config["property"]["classifier-threshold"] = classifier_threshold

        if custom_lib_path:
            config["property"]["custom-lib-path"] = custom_lib_path

            # Add custom parse function name based on network type
            if custom_parse_func:
                if network_type == 0:  # detection
                    config["property"]["parse-bbox-func-name"] = custom_parse_func
                elif network_type == 1:  # classification
                    config["property"]["parse-classifier-func-name"] = custom_parse_func
                elif network_type == 2:  # segmentation
                    config["property"]["parse-segmentation-func-name"] = custom_parse_func
                elif network_type == 3:  # instance_segmentation
                    config["property"]["parse-bbox-instance-mask-func-name"] = custom_parse_func
                # network_type 100 (custom) doesn't need a parse function name

        # Generate YAML content with header
        header = """# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# IMPORTANT NORMALIZATION PARAMETERS:
# - net-scale-factor: Must match the scaling factor used during training
# - offsets: Must match any per-channel mean subtraction used during training
# Incorrect normalization will result in poor inference accuracy!

"""

        yaml_content = yaml.dump(config, default_flow_style=False, sort_keys=False)
        full_content = header + yaml_content

        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text=(
                        f"Successfully generated DeepStream nvinfer configuration!\n\n"
                        f"Configuration summary:\n"
                        f"  - Network type: {network_type_names[network_type]} ({network_type})\n"
                        f"  - Precision mode: {precision_names[precision_mode]} ({precision_mode})\n"
                        f"  - ONNX file: {onnx_file}\n"
                        f"  - Input dimensions: {input_dims}\n"
                        f"  - Label file: {label_file}\n"
                        f"  - Scale factor: {net_scale_factor}\n"
                        + (f"  - Per-channel offsets: {offsets}\n" if offsets else "  - Per-channel offsets: NOT SET (no mean subtraction)\n")
                        + (f"  - Number of classes: {num_classes}\n" if num_classes else "")
                        + (f"  - Input from metadata: {bool(input_tensor_from_meta)}\n" if input_tensor_from_meta else "")
                        + (f"  - Output as metadata: {bool(output_tensor_meta)} (raw tensors in DS META)\n" if output_tensor_meta else "")
                        + (f"  - Custom library: {custom_lib_path}\n" if custom_lib_path else "")
                        + (f"  - Custom parse function: {custom_parse_func}\n" if custom_parse_func else "")
                        + "\n⚠️  IMPORTANT: Verify that net-scale-factor and per-channel offsets match your model's training normalization!\n"
                        + "   - net-scale-factor must match the scaling applied during training\n"
                        + "   - offsets must match any per-channel mean subtraction used during training\n"
                        + "   - If training used no mean subtraction, offsets should not be set (or set to 0;0;0)\n"
                        + f"\nGenerated configuration (nvdsinfer_config.yaml):\n\n{full_content}"
                    )
                )
            ]
        )


class ApiKeyMiddleware(BaseHTTPMiddleware):
    """ASGI middleware that enforces Bearer-token authentication.

    When an *api_key* is configured on the server, every incoming request must
    carry an ``Authorization: Bearer <api_key>`` header.  Requests that are
    missing the header or carry a different token are rejected immediately with
    HTTP 401 Unauthorized.
    """

    def __init__(self, app, api_key: str) -> None:
        super().__init__(app)
        self._api_key = api_key

    async def dispatch(self, request: Request, call_next):
        auth_header = request.headers.get("Authorization", "")
        prefix = "Bearer "
        if not auth_header.startswith(prefix) or auth_header[len(prefix):] != self._api_key:
            return StarletteResponse("Unauthorized", status_code=401)
        return await call_next(request)


def create_sse_app(
    server: "InferenceBuilderMCPServer",
    api_key: str | None = None,
) -> Starlette:
    """Build a Starlette ASGI application that exposes the MCP server over HTTP.

    Two transports are available simultaneously:

    **Streamable HTTP** (preferred — Claude Code and MCP SDK ≥ 1.0)

    * ``POST /mcp``  – Client sends JSON-RPC requests; server responds with
                       JSON or an SSE stream depending on the ``Accept`` header.
    * ``GET  /mcp``  – Client opens a long-lived SSE channel for server-push
                       notifications (optional, used alongside POST).

    **Legacy SSE transport** (older clients)

    * ``GET  /sse``       – Client opens a persistent SSE stream.
    * ``POST /messages/`` – Client POSTs JSON-RPC requests tagged with the
                            ``session_id`` received from the SSE stream.

    Args:
        server:  The :class:`InferenceBuilderMCPServer` instance to expose.
        api_key: Optional Bearer token.  When provided, every request must
                 carry ``Authorization: Bearer <api_key>`` or receive 401.
    """
    import contextlib

    # --- Streamable HTTP transport (Claude Code / newer MCP clients) -----------
    session_manager = StreamableHTTPSessionManager(
        app=server.server,
        stateless=False,  # enables mcp-session-id tracking for per-client workspace
    )

    @contextlib.asynccontextmanager
    async def lifespan(app):
        asyncio.ensure_future(server._collect_system_info())
        server._sweep_stale_workspaces()
        async with session_manager.run():
            yield

    # --- Legacy SSE transport --------------------------------------------------
    sse_transport = SseServerTransport("/messages/")

    async def handle_sse(request: Request) -> StarletteResponse:
        async with sse_transport.connect_sse(
            request.scope, request.receive, request._send
        ) as streams:
            init_options = server.server.create_initialization_options()
            await server.server.run(streams[0], streams[1], init_options)
        return StarletteResponse()

    # Sentinel response used by endpoints that send the ASGI response themselves.
    # Starlette requires endpoints to return a Response object; returning this
    # no-op subclass prevents a double-send when session_manager.handle_request
    # (or connect_sse) has already written headers+body to the send callable.
    class _AlreadyHandled(StarletteResponse):
        async def __call__(self, scope, receive, send) -> None:
            pass

    async def handle_mcp(request: Request) -> _AlreadyHandled:
        if request.method == "DELETE":
            session_id = request.headers.get("mcp-session-id")
            if session_id:
                await server._cleanup_session(session_id)
        await session_manager.handle_request(request.scope, request.receive, request._send)
        return _AlreadyHandled()

    # --- Artifact download endpoint -------------------------------------------
    # GET /{path:path}
    # Unified session workspace browser — lists directories as JSON, serves files
    # with auto-detected MIME type.  Session is identified via the mcp-session-id
    # header so no session ID appears in the URL.
    import mimetypes
    from starlette.responses import FileResponse, JSONResponse

    async def handle_workspace(request: Request) -> StarletteResponse:
        session_id = request.headers.get("mcp-session-id")
        if not session_id:
            return JSONResponse({"error": "Missing mcp-session-id header"}, status_code=400)

        workspace = server._session_workspaces.get(session_id)
        if workspace is None:
            return JSONResponse({"error": "Session not found"}, status_code=404)

        rel = request.path_params.get("path", "")
        target = (workspace / rel).resolve() if rel else workspace.resolve()

        # Reject path traversal
        try:
            target.relative_to(workspace.resolve())
        except ValueError:
            return JSONResponse({"error": "Invalid path"}, status_code=400)

        if not target.exists():
            return JSONResponse({"error": "Not found"}, status_code=404)

        if target.is_dir():
            entries = []
            for entry in sorted(target.iterdir()):
                info: Dict[str, Any] = {"name": entry.name, "type": "directory" if entry.is_dir() else "file"}
                if entry.is_file():
                    info["size"] = entry.stat().st_size
                entries.append(info)
            return JSONResponse({"path": f"/{rel}", "entries": entries})

        mime_type, _ = mimetypes.guess_type(target.name)
        if mime_type is None:
            mime_type = "text/plain" if target.suffix in (".log", ".txt", ".yaml", ".yml", ".json") else "application/octet-stream"

        return FileResponse(path=str(target), media_type=mime_type)

    # ---------------------------------------------------------------------------
    routes = [
        # Streamable HTTP — single Route (no Starlette 307 redirect), all methods
        Route("/mcp", endpoint=handle_mcp, methods=["GET", "POST", "DELETE"]),
        # Legacy SSE — separate GET (stream) and POST (messages) endpoints
        Route("/sse", endpoint=handle_sse, methods=["GET"]),
        Mount("/messages/", app=sse_transport.handle_post_message),
        # Session workspace browser — list directories, download files
        Route("/{path:path}", endpoint=handle_workspace, methods=["GET"]),
    ]

    middleware: list[Middleware] = []
    if api_key:
        middleware.append(Middleware(ApiKeyMiddleware, api_key=api_key))

    return Starlette(routes=routes, middleware=middleware, lifespan=lifespan)


async def run_sse_server(
    server: "InferenceBuilderMCPServer",
    host: str = "0.0.0.0",
    port: int = 8000,
    api_key: str | None = None,
) -> None:
    """Run the MCP server with SSE/HTTP transport using uvicorn.

    Args:
        server:  The :class:`InferenceBuilderMCPServer` instance to serve.
        host:    Network interface to bind (default ``0.0.0.0``).
        port:    TCP port to listen on (default ``8000``).
        api_key: Optional Bearer token for authentication (see
                 :func:`create_sse_app`).
    """
    import socket
    import uvicorn

    logger = logging.getLogger("deepstream-inference-builder")

    # Resolve a human-readable address for the config banner.
    display_host = host if host not in ("0.0.0.0", "::") else socket.gethostbyname(socket.gethostname())

    server._base_url = f"http://{display_host}:{port}"
    app = create_sse_app(server, api_key=api_key)
    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    uv_server = uvicorn.Server(config)

    # Print the startup banner after uvicorn has finished its own setup but
    # before it blocks in the serve loop.  We hook into the lifespan startup
    # by monkey-patching startup() — simplest approach without subclassing.
    _original_startup = uv_server.startup

    async def _startup_with_banner(sockets=None):
        await _original_startup(sockets=sockets)
        banner_lines = [
            "",
            "=" * 62,
            "  Inference Builder MCP Server — HTTP transport",
            "=" * 62,
            f"  Streamable HTTP : http://{display_host}:{port}/mcp   (Claude Code)",
            f"  Legacy SSE      : http://{display_host}:{port}/sse   (older clients)",
            f"  Bind address    : {host}:{port}",
            f"  Auth            : {'Bearer token (MCP_API_KEY / --api-key)' if api_key else 'none (open access)'}",
            f"  Workspaces      : {server._workspace_root or 'disabled (--workspace-root not set)'}",
            "=" * 62,
            "",
            "  MCP client configuration snippet:",
            "  {",
            '    "mcpServers": {',
            '      "deepstream-inference-builder": {',
            f'        "url": "http://{display_host}:{port}/mcp"' + ("," if api_key else ""),
        ]
        if api_key:
            banner_lines += [
                '        "headers": {',
                '          "Authorization": "Bearer <your-api-key>"',
                "        }",
            ]
        banner_lines += [
            "      }",
            "    }",
            "  }",
            "=" * 62,
            "",
        ]
        print("\n".join(banner_lines), flush=True)

    uv_server.startup = _startup_with_banner
    await uv_server.serve()


async def main():
    """Main entry point for the MCP server."""
    parser = argparse.ArgumentParser(
        description="Inference Builder MCP Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Transport modes\n"
            "---------------\n"
            "  stdio  Local client only. The MCP client (e.g. Claude Code) launches\n"
            "         this process directly and communicates over stdin/stdout.\n"
            "         No network port is opened. Workspace management is disabled.\n\n"
            "  sse    HTTP server for one or more remote clients. Two endpoints are\n"
            "         exposed on the same port:\n"
            "           POST/GET /mcp  — Streamable HTTP (Claude Code, MCP SDK ≥ 1.0)\n"
            "           GET  /sse      — Legacy SSE stream  (older clients)\n"
            "           POST /messages/ — Legacy SSE messages\n"
            "         Each client is assigned a unique mcp-session-id and gets an\n"
            "         isolated workspace directory under --workspace-root.\n\n"
            "Environment variables (SSE mode)\n"
            "--------------------------------\n"
            "  MCP_API_KEY         Bearer token clients must send in the\n"
            "                      Authorization header (same as --api-key).\n"
            "  MCP_WORKSPACE_ROOT  Base directory for per-client workspaces\n"
            f"                      (default: {Path(tempfile.gettempdir()) / 'inference-builder-workspaces'}).\n\n"
            "Examples\n"
            "--------\n"
            "  # Local client via stdio (default)\n"
            "  python mcp_server.py\n\n"
            "  # Remote clients on port 8888, open access\n"
            "  python mcp_server.py --transport sse --port 8888\n\n"
            "  # Remote clients with Bearer-token auth\n"
            "  python mcp_server.py --transport sse --port 8888 --api-key MY_SECRET\n"
            "  MCP_API_KEY=MY_SECRET python mcp_server.py --transport sse --port 8888\n\n"
            "  # Custom workspace root\n"
            "  python mcp_server.py --transport sse --port 8888 \\\n"
            "      --workspace-root /var/tmp/ib-workspaces\n"
            "  MCP_WORKSPACE_ROOT=/var/tmp/ib-workspaces \\\n"
            "      python mcp_server.py --transport sse --port 8888\n\n"
            "MCP client configuration snippet\n"
            "--------------------------------\n"
            '  { "mcpServers": { "deepstream-inference-builder": {\n'
            '      "url": "http://<host>:<port>/mcp",\n'
            '      "headers": { "Authorization": "Bearer <token>" } } } }\n'
        ),
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="stdio",
        help=(
            "Transport type: 'stdio' for a single local client (default), "
            "'sse' for one or more remote clients over HTTP."
        ),
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Network interface to bind in SSE mode (default: 0.0.0.0 — all interfaces).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="TCP port to listen on in SSE mode (default: 8000).",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("MCP_API_KEY"),
        metavar="TOKEN",
        help=(
            "Bearer token remote clients must supply via 'Authorization: Bearer TOKEN'. "
            "Defaults to MCP_API_KEY env var. Omit to allow unauthenticated access. "
            "SSE mode only."
        ),
    )
    parser.add_argument(
        "--workspace-root",
        default=os.environ.get(
            "MCP_WORKSPACE_ROOT",
            str(Path(tempfile.gettempdir()) / "inference-builder-workspaces"),
        ),
        metavar="DIR",
        help=(
            "Base directory under which a private workspace folder is created "
            "for each connected client (named by mcp-session-id). "
            "Tools that accept output_dir will default to the session workspace "
            "when the caller does not specify one. "
            "Defaults to MCP_WORKSPACE_ROOT env var, or "
            f"{Path(tempfile.gettempdir()) / 'inference-builder-workspaces'}. "
            "Only applies to SSE transport."
        ),
    )
    parser.add_argument(
        "--model-root",
        default=os.environ.get(
            "MCP_MODEL_ROOT",
            str(Path(tempfile.gettempdir()) / "inference-builder-models"),
        ),
        metavar="DIR",
        help=(
            "Shared directory where prepare_model_repository downloads models. "
            "Shared across all client sessions; files are accessible via read_model_file. "
            "Defaults to MCP_MODEL_ROOT env var, or "
            f"{Path(tempfile.gettempdir()) / 'inference-builder-models'}."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG,
        format=("%(asctime)s %(levelname)s %(name)s - %(message)s"),
    )
    logger = logging.getLogger("deepstream-inference-builder")
    logger.info("starting_mcp_server")

    # Workspace is only meaningful for SSE transport (requires mcp-session-id).
    # In stdio mode there is a single local client and no session identity,
    # so workspace management is left entirely to the caller.
    workspace_root = Path(args.workspace_root) if args.transport == "sse" else None
    model_root = Path(args.model_root)
    server = InferenceBuilderMCPServer(workspace_root=workspace_root, model_root=model_root)
    logger.info("server_created")

    if args.transport == "sse":
        await run_sse_server(server, host=args.host, port=args.port, api_key=args.api_key)
    else:
        async with stdio_server() as (read_stream, write_stream):
            logger.info("stdio_server_started")
            init_options = server.server.create_initialization_options()
            logger.info("initialization_options_created")
            await server.server.run(
                read_stream,
                write_stream,
                init_options,
            )


if __name__ == "__main__":
    asyncio.run(main())
