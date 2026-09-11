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

"""
Test client for the inference-builder MCP server (HTTP/Streamable-HTTP transport).

Usage:
    python mcp/test_mcp_client.py [--url URL] [--api-key TOKEN]

The server must already be running, e.g.:
    MCP_API_KEY=MY_SECRET ./mcp/server_manager.sh start --port 8000

Arguments:
    --url     MCP server URL  (default: http://localhost:8000/mcp)
    --api-key Bearer token    (default: none)
"""

import argparse
import asyncio
import json
import sys

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client


async def run_tests(url: str, api_key: str | None) -> None:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    print(f"Connecting to MCP server at {url}")
    if api_key:
        print("Using Bearer token authentication")

    async with streamablehttp_client(url, headers=headers) as (read, write, _):
        async with ClientSession(read, write) as session:
            # Step 1: Initialize
            print("\nStep 1: Initialize MCP connection")
            await session.initialize()
            print("Connected")

            # Step 2: List tools
            print("\nStep 2: List available tools")
            tools_result = await session.list_tools()
            tools = tools_result.tools
            print(f"Found {len(tools)} tools:")
            for tool in tools:
                print(f"  {tool.name}: {tool.description}")

            # Step 3: List resources
            print("\nStep 3: List available resources")
            resources_result = await session.list_resources()
            resources = resources_result.resources
            print(f"Found {len(resources)} resources:")
            for res in resources:
                print(f"  {res.uri}")

            # Step 4: Error-handling smoke test
            print("\nStep 4: generate_inference_pipeline error-handling smoke test")
            result = await session.call_tool(
                "generate_inference_pipeline",
                {"config": "invalid: yaml: ["},
            )
            if not result.isError:
                for block in result.content:
                    print(f"  {block.text}")
                raise AssertionError(
                    "generate_inference_pipeline succeeded with invalid config — "
                    "error handling may have regressed"
                )
            print("Tool correctly returned an error for invalid config")

            print("\nAll tests passed")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test client for the inference-builder MCP server (HTTP mode)"
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8000/mcp",
        help="MCP server URL (default: http://localhost:8000/mcp)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        metavar="TOKEN",
        help="Bearer token for authentication (omit if server has no API key)",
    )
    args = parser.parse_args()

    try:
        asyncio.run(run_tests(args.url, args.api_key))
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
