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

import asyncio
import os
import sys
from pathlib import Path
from typing import Any, Dict
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolRequest,
    CallToolResult,
    ListToolsRequest,
    ListToolsResult,
    Tool,
    TextContent,
    Resource,
)
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

    def __init__(self):
        self.server = Server("deepstream-inference-builder")
        self.logger = logging.getLogger("deepstream-inference-builder")

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
                            "Detailed documentation for MCP server integration. "
                            "Includes tool reference with parameters, resource navigation guide, "
                            "usage examples, workflow integration, and troubleshooting."
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
                        "Generate a deployable inference service from a YAML configuration file. "
                        "Outputs a complete project with model serving code, API endpoints, and deployment files. "
                        "Read 'samples://*' resources for example configurations, "
                        "or 'schema://config.schema.json' resource for the full configuration schema."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "config_file": {
                                "type": "string",
                                "description": (
                                    "Path to the YAML configuration file defining the pipeline. "
                                    "If the user has not provided a config file, generate one based on "
                                    "their requirements: read 'schema://config.schema.json' for the full schema, "
                                    "'schema://index.json' to find backend-specific parameter schemas "
                                    "(maps backend type to 'schema://backends/parameters/*'), "
                                    "'schema://readme' for documentation, and 'samples://*' resources for "
                                    "reference examples. Create the YAML file at a suitable path, then provide "
                                    "that path here."
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
                            "output_dir": {
                                "type": "string",
                                "description": (
                                    "The user's project directory where generated artifacts will be placed. "
                                    "This directory will contain configurations, custom processors, Dockerfiles, "
                                    "and other files for the inference project. A subdirectory named after the "
                                    "pipeline (from config YAML) will be created here, or a tar.gz archive if "
                                    "tar_output is enabled. If the project location is unclear, ask the user to "
                                    "specify it, or prompt them to confirm after creating a new project directory."
                                ),
                                "default": "."
                            },
                            "api_spec": {
                                "type": "string",
                                "description": (
                                    "Path to OpenAPI specification file (YAML/JSON). "
                                    "Required for 'fastapi', 'triton', and 'nim' server types. "
                                    "Not needed for 'serverless' type. If the user has not provided an "
                                    "OpenAPI spec, attempt to generate one based on their requirements. "
                                    "If the given information is insufficient to generate a valid spec "
                                    "(e.g., missing endpoint definitions, request/response schemas), "
                                    "ask the user to provide an OpenAPI specification file."
                                )
                            },
                            "custom_modules": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": (
                                    "List of paths to custom Python module files for preprocessors and "
                                    "postprocessors. Each module must define classes with a 'name' attribute "
                                    "and '__call__' method to be registered in the pipeline. Determine if "
                                    "processors are needed based on the user's requirements and pipeline "
                                    "structure (e.g., input transformations, output formatting). If processors "
                                    "are required, read 'samples://processor/*' resources for reference examples, "
                                    "then generate the Python files accordingly, save them in the project "
                                    "directory, and include their paths here."
                                )
                            },
                            "exclude_lib": {
                                "type": "boolean",
                                "default": False,
                                "description": (
                                    "If true, exclude the common utility library from generated code. "
                                    "Use when deploying to an environment where the library is already installed."
                                )
                            },
                            "tar_output": {
                                "type": "boolean",
                                "default": False,
                                "description": (
                                    "If true, output a tar.gz archive instead of a directory. "
                                    "This affects how the generated pipeline is copied into the container "
                                    "image in the Dockerfile: use COPY with directory when false, or ADD "
                                    "with tar.gz extraction when true. Set to true when transferring the "
                                    "pipeline to a remote build system."
                                )
                            },
                            "validation_dir": {
                                "type": "string",
                                "description": (
                                    "Path to output directory for API validation/test artifacts. "
                                    "When provided, generates OpenAPI client code and test cases "
                                    "for validating the inference endpoints."
                                )
                            },
                            "no_docker": {
                                "type": "boolean",
                                "default": False,
                                "description": (
                                    "If true, use locally installed OpenAPI Generator CLI instead of Docker. "
                                    "Requires 'openapi-generator-cli' to be installed and in PATH."
                                )
                            },
                            "test_cases_abs_path": {
                                "type": "boolean",
                                "default": False,
                                "description": (
                                    "If true, use absolute paths in generated test_cases.yaml. "
                                    "Use when test files are located outside the validation directory."
                                )
                            }
                        },
                        "required": ["config_file"]
                    }
                ),
                Tool(
                    name="build_docker_image",
                    description=(
                        "Build a Docker image from a generated inference "
                        "pipeline. The Dockerfile is expected to consume the "
                        "tar.gz archive produced by 'generate_inference_pipeline' "
                        "when 'tar_output' is true (for example, using "
                        "'ADD <pipeline>.tar.gz /app') rather than copying a "
                        "loose directory of files. Agents should prefer "
                        "tar_output=true for Docker-based deployments."
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
                                    "Path to Dockerfile. Always generate the Dockerfile based on user "
                                    "requirements and 'samples://dockerfile/*' resources. "
                                    "**Before choosing a Dockerfile template, read 'docs://platform-guide.md' "
                                    "to identify the correct base image and template for the target hardware** "
                                    "(x86_64 datacenter, Jetson/Tegra, or arm-sbsa servers like GB10/GB300/DGX Spark). "
                                    "Consider model backend (triton, vllm, tensorrt-llm, deepstream), server type "
                                    "(serverless, fastapi), and hardware platform. "
                                    "The Dockerfile MUST be placed in the output_dir where the generated "
                                    "pipeline resides, as its parent directory becomes the Docker build "
                                    "context for properly transferring the pipeline code into the container."
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
                        "'build_docker_image' have completed."
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
                            "model_repo_host": {
                                "type": "string",
                                "description": (
                                    "Optional host path to a prepared model repository "
                                    "directory (for example, produced by "
                                    "'prepare_model_repository'). If provided, it will "
                                    "be mounted into the container at "
                                    "model_repo_container via a Docker volume."
                                )
                            },
                            "model_repo_container": {
                                "type": "string",
                                "description": (
                                    "Container path where the model repository should "
                                    "be mounted (for example, '/models'). Used only "
                                    "when model_repo_host is provided."
                                ),
                                "default": "/models"
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
                            "gpus": {
                                "type": "string",
                                "description": (
                                    "GPU devices to use for the container, passed as "
                                    "`--gpus` (for example, 'all', 'device=0', "
                                    "'device=0,1')."
                                ),
                                "default": "all"
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
                    name="prepare_model_repository",
                    description=(
                        "Optionally prepare a model repository directory for deployment. "
                        "Given a model configuration list, this tool can download model artifacts "
                        "(for example from NGC or Hugging Face) and copy associated runtime "
                        "configuration files (such as DeepStream nvdsinfer_config.yaml and "
                        "nvdspreprocess_config*.yaml) into a model repository layout that "
                        "matches the generated pipeline and Dockerfile. Use this when the "
                        "user does not already have a model repository prepared."
                        "It is recommended to mount the model repository as a volume when running the Docker image."
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
                                    "'name' (required, model name), 'source' (optional, 'NGC' or "
                                    "'HF', default 'NGC'), 'target' (required, directory where the "
                                    "prepared model repository should be created), 'path' (for NGC: "
                                    "registry model path; for HF: repo id like "
                                    "'Qwen/Qwen2.5-VL-3B-Instruct'), 'version' (for NGC), "
                                    "'configs' (optional, path relative to config_dir of a directory "
                                    "of runtime config files to copy into the final model repository), and "
                                    "'post_script' (optional, shell command to execute after model download "
                                    "for post-processing steps)."
                                ),
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "name": {
                                            "type": "string",
                                            "description": "Model name (used as the final directory name).",
                                        },
                                        "source": {
                                            "type": "string",
                                            "enum": ["NGC", "HF"],
                                            "description": (
                                                "Model source: 'NGC' for NVIDIA NGC registry models, "
                                                "or 'HF' for Hugging Face models."
                                            ),
                                        },
                                        "target": {
                                            "type": "string",
                                            "description": (
                                                "Target directory on the local filesystem where the prepared "
                                                "model repository should be created."
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
                                            "type": "string",
                                            "description": (
                                                "Optional path (relative to config_dir) to a directory containing "
                                                "runtime configuration files (for example, DeepStream "
                                                "nvdsinfer_config.yaml, nvdspreprocess_config*.yaml, label files) "
                                                "that should be copied into the final model repository."
                                            ),
                                        },
                                        "post_script": {
                                            "type": "string",
                                            "description": (
                                                "Optional shell command or script to execute after the model is downloaded. "
                                                "This can be used for post-processing steps like installing additional "
                                                "dependencies, converting model formats, or applying patches. "
                                                "The script is executed with the model's target directory as the current "
                                                "working directory."
                                            ),
                                        },
                                    },
                                    "required": ["name", "target"],
                                },
                            },
                            "config_dir": {
                                "type": "string",
                                "description": (
                                    "Base directory used to resolve relative 'configs' "
                                    "paths in model_configs. Typically this is the "
                                    "directory containing your pipeline/test "
                                    "configuration files."
                                ),
                                "default": "."
                            }
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
                        "backend parameters. See 'samples://runtime_config/*' resources for examples."
                        "**IMPORTANT: Before generating a new config file, first call 'prepare_model_repository' to download the model and check for existing nvinfer configuration files.** "
                        "Models from NGC typically include a configuration file (YAML or TXT) with all required inference parameters - always prefer using this provided configuration when available. "),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "output_path": {
                                "type": "string",
                                "description": (
                                    "Path where the generated nvdsinfer_config.yaml file should be saved. "
                                    "This should typically be in your model repository directory."
                                )
                            },
                            "onnx_file": {
                                "type": "string",
                                "description": (
                                    "Name of the ONNX model file (e.g., 'model.onnx'). "
                                    "This file should exist in the same directory as the config file."
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
                        "required": ["output_path", "onnx_file", "network_type", "input_dims", "label_file"]
                    }
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
            elif name == "prepare_model_repository":
                result = await self._prepare_model_repository(arguments)
            elif name == "generate_nvinfer_config":
                result = await self._generate_deepstream_nvinfer_config(arguments)
            else:
                raise ValueError(f"Unknown tool: {name}")

            duration_ms = (time.perf_counter() - start_time) * 1000.0
            self.logger.info(
                "tool_call_completed name=%s error=%s duration_ms=%.2f",
                name,
                bool(getattr(result, "isError", False)),
                duration_ms,
            )
            return result
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

    async def _generate_pipeline(
        self, arguments: Dict[str, Any]
    ) -> CallToolResult:
        """Generate inference pipeline"""
        config_file = arguments["config_file"]
        server_type = arguments.get("server_type", "serverless")
        output_dir = arguments.get("output_dir", ".")

        # Build command - use the same Python executable as the parent process
        cmd = [
            sys.executable, "builder/main.py",
            "--server-type", server_type,
            "-o", output_dir,
            config_file
        ]

        # Add optional arguments
        if "api_spec" in arguments:
            cmd.extend(["-a", arguments["api_spec"]])

        if "custom_modules" in arguments:
            for module in arguments["custom_modules"]:
                cmd.extend(["-c", module])

        if arguments.get("exclude_lib", False):
            cmd.append("-x")

        if arguments.get("tar_output", False):
            cmd.append("-t")

        if "validation_dir" in arguments:
            cmd.extend(["--validation-dir", arguments["validation_dir"]])

        if arguments.get("no_docker", False):
            cmd.append("--no-docker")

        if arguments.get("test_cases_abs_path", False):
            cmd.append("--test-cases-abs-path")

        # Execute the command - inherit the current process environment

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
            env=os.environ.copy(),  # Inherit the current environment
            check=False,
        )

        if result.returncode == 0:
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=f"Successfully generated inference pipeline!\n\n"
                             f"Output directory: {output_dir}\n\n"
                             f"Command executed: {' '.join(cmd)}\n\n"
                             f"Output:\n{result.stdout}"
                    )
                ]
            )
        else:
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=(
                            "Failed to generate pipeline:\n\n"
                            f"Error:\n{result.stderr}\n\n"
                            f"Command: {' '.join(cmd)}"
                        )
                    )
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
        model_repo_host = arguments.get("model_repo_host")
        model_repo_container = arguments.get("model_repo_container", "/models")
        server_type = arguments.get("server_type", "serverless")
        env_vars = arguments.get("env") or {}
        cmd_args = arguments.get("cmd") or []
        gpus = arguments.get("gpus", "all")
        timeout = int(arguments.get("timeout", 300))

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

        # GPU configuration
        if gpus:
            cmd.extend(["--gpus", str(gpus)])

        # Network: for serverless we still use host network so local clients can connect
        cmd.append("--network=host")

        # Volume for model repository if provided
        if model_repo_host:
            host_path = str(Path(model_repo_host).expanduser().resolve())
            volume_arg = f"{host_path}:{model_repo_container}"
            cmd.extend(["-v", volume_arg])

        # Environment variables
        if isinstance(env_vars, dict):
            for key, value in env_vars.items():
                cmd.extend(["-e", f"{key}={value}"])

        # Image name and optional command/args
        cmd.append(image_name)
        if isinstance(cmd_args, list):
            cmd.extend([str(a) for a in cmd_args])

        self.logger.info(
            "docker_run_invoked image=%s model_repo_host=%s cmd=%s",
            image_name,
            model_repo_host,
            " ".join(cmd),
        )

        try:
            result = subprocess.run(
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
            is_serverless = server_type == "serverless"
            if is_serverless:
                # Serverless containers should finish within the timeout.
                # The `docker run` client is dead but the container keeps
                # running (and holding GPU memory).  Force-remove it.
                cleanup_msg = ""
                try:
                    subprocess.run(
                        ["docker", "rm", "-f", container_name],
                        capture_output=True,
                        timeout=30,
                        check=False,
                    )
                    cleanup_msg = (
                        f" The orphaned container '{container_name}' has "
                        "been force-removed."
                    )
                except Exception as cleanup_exc:
                    cleanup_msg = (
                        f" WARNING: failed to remove orphaned container "
                        f"'{container_name}': {cleanup_exc}. "
                        "Please remove it manually with: "
                        f"docker rm -f {container_name}"
                    )
                    self.logger.warning(
                        "Failed to remove timed-out container %s: %s",
                        container_name,
                        cleanup_exc,
                    )
                return CallToolResult(
                    content=[
                        TextContent(
                            type="text",
                            text=(
                                f"Docker run timed out after {timeout} seconds "
                                f"for image '{image_name}'.{cleanup_msg} You may "
                                "want to increase the timeout or inspect the "
                                "image logs manually."
                            ),
                        )
                    ],
                    isError=True,
                )
            else:
                # Non-serverless (persistent) servers are expected to keep
                # running past the timeout — that means the server started
                # successfully.  Leave the container running.
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
                        )
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

        text = (
            f"docker run command: {' '.join(cmd)}\n\n"
            f"exit code: {result.returncode}\n\n"
            f"stdout:\n{result.stdout}\n\n"
            f"stderr:\n{result.stderr}"
        )

        return CallToolResult(
            content=[TextContent(type="text", text=text)],
            isError=result.returncode != 0,
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
        config_dir_arg = arguments.get("config_dir", ".")

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

        # Resolve config_dir relative to project root (repo root)
        project_root = Path(__file__).resolve().parents[1]
        config_dir = Path(config_dir_arg)
        if not config_dir.is_absolute():
            config_dir = (project_root / config_dir).resolve()

        messages: list[str] = []

        for model_info in model_configs:
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

                source = str(model_info.get("source", "NGC")).upper()
                target = model_info.get("target")
                if not target:
                    messages.append(
                        f"❌ Skipping model '{model_name}': missing required 'target'"
                    )
                    continue

                target_dir = Path(target).expanduser()
                final_model_dir = target_dir / model_name

                # Create target parent directory
                target_dir.mkdir(parents=True, exist_ok=True)

                if final_model_dir.exists():
                    messages.append(
                        f"✅ Model '{model_name}' already exists at {final_model_dir}, "
                        f"skipping download"
                    )
                else:
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
                        result = subprocess.run(
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

                # Copy configs into final_model_dir if specified
                configs_rel = model_info.get("configs")
                if configs_rel:
                    source_configs = (config_dir / configs_rel).resolve()
                    if source_configs.exists():
                        messages.append(
                            f"📄 Copying runtime configs from {source_configs} "
                            f"to {final_model_dir}"
                        )
                        try:
                            for item in source_configs.iterdir():
                                dest = final_model_dir / item.name
                                if item.is_file():
                                    shutil.copy2(str(item), str(dest))
                                elif item.is_dir():
                                    shutil.copytree(
                                        str(item), str(dest), dirs_exist_ok=True
                                    )
                        except Exception as exc:
                            messages.append(
                                f"⚠️ Failed to copy configs for '{model_name}': {exc}"
                            )
                    else:
                        messages.append(
                            f"⚠️ Config path not found for model '{model_name}': "
                            f"{source_configs}"
                        )

                # Execute post-script if specified
                post_script = model_info.get("post_script")
                if post_script:
                    messages.append(
                        f"🔧 Executing post-script for '{model_name}': {post_script}"
                    )
                    try:
                        script_result = subprocess.run(
                            post_script,
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

            except Exception as exc:  # Catch-all per model
                messages.append(
                    f"❌ Exception while preparing model '{model_name}': {exc}"
                )

        summary = "\n".join(messages)
        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text=(
                        "Model repository preparation completed.\n\n"
                        f"{summary}"
                    ),
                )
            ]
        )

    async def _build_docker_image(
        self, arguments: Dict[str, Any]
    ) -> CallToolResult:
        """Build Docker image from generated pipeline"""
        image_name = arguments["image_name"]
        dockerfile = arguments["dockerfile"]

        # Validate Dockerfile exists
        dockerfile_path = Path(dockerfile)
        if not dockerfile_path.exists():
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=f"Dockerfile not found: {dockerfile}. Please "
                             f"provide a valid Dockerfile path."
                    )
                ],
                isError=True
            )

        # Use Dockerfile's parent directory as build context
        build_context = str(dockerfile_path.parent)

        # Build Docker image
        cmd = [
            "docker", "build",
            "-f", dockerfile,
            "-t", image_name,
            build_context
        ]
        self.logger.info(
            "docker_build_invoked image=%s dockerfile=%s context=%s",
            image_name,
            dockerfile,
            build_context,
        )

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent,
                check=False,
            )

            if result.returncode == 0:
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

    async def _generate_deepstream_nvinfer_config(
        self, arguments: Dict[str, Any]
    ) -> CallToolResult:
        """Generate DeepStream nvinfer runtime configuration file"""
        import yaml

        # Extract required parameters
        output_path = arguments["output_path"]
        onnx_file = arguments["onnx_file"]
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

        # Write to file
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(full_content)

            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=(
                            f"Successfully generated DeepStream nvinfer configuration!\n\n"
                            f"Output file: {output_path}\n\n"
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
                            + f"\nGenerated content:\n\n{full_content}"
                        )
                    )
                ]
            )
        except Exception as e:
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=f"Failed to write configuration file:\n\n{str(e)}"
                    )
                ],
                isError=True
            )


async def main():
    """Main entry point for the MCP server"""
    logging.basicConfig(
        level=logging.DEBUG,
        format=("%(asctime)s %(levelname)s %(name)s - %(message)s"),
    )
    logger = logging.getLogger("deepstream-inference-builder")
    logger.info("starting_mcp_server")
    server = InferenceBuilderMCPServer()
    logger.info("server_created")

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
