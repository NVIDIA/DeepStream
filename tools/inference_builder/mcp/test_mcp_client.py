#!/usr/bin/env python3
"""
Simple MCP client to test direct communication with the inference-builder MCP server
"""

import asyncio
import json
import subprocess
import sys
from pathlib import Path


class SimpleMCPClient:
    def __init__(self, server_command, server_args, cwd=None):
        self.server_command = server_command
        self.server_args = server_args
        self.cwd = cwd or Path.cwd()
        self.process = None

    async def start_server(self):
        """Start the MCP server process"""
        cmd = [self.server_command] + self.server_args
        print(f"🚀 Starting MCP server: {' '.join(cmd)}")
        print(f"📁 Working directory: {self.cwd}")

        self.process = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=self.cwd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        print("✅ MCP server started")

    async def send_request(self, request):
        """Send a JSON-RPC request to the server"""
        if not self.process:
            raise RuntimeError("Server not started")

        request_str = json.dumps(request) + '\n'
        print(f"📤 Sending request: {request}")

        self.process.stdin.write(request_str.encode())
        await self.process.stdin.drain()

        # Read response
        response_line = await self.process.stdout.readline()
        if not response_line:
            stderr_output = await self.process.stderr.read()
            raise RuntimeError(f"No response from server. stderr: {stderr_output.decode()}")

        response = json.loads(response_line.decode().strip())
        print(f"📥 Received response: {json.dumps(response, indent=2)}")
        return response

    async def initialize(self):
        """Initialize the MCP connection"""
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {}
                },
                "clientInfo": {
                    "name": "test-client",
                    "version": "1.0.0"
                }
            }
        }
        response = await self.send_request(init_request)

        # Send initialized notification
        initialized_notification = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
            "params": {}
        }
        print(f"📤 Sending initialized notification: {initialized_notification}")
        request_str = json.dumps(initialized_notification) + '\n'
        self.process.stdin.write(request_str.encode())
        await self.process.stdin.drain()

        return response

    async def list_tools(self):
        """List available tools"""
        request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {}
        }
        return await self.send_request(request)

    async def call_tool(self, tool_name, arguments=None):
        """Call a specific tool"""
        request = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments or {}
            }
        }
        return await self.send_request(request)

    async def cleanup(self):
        """Clean up the server process"""
        if self.process:
            self.process.terminate()
            await self.process.wait()
            print("🛑 MCP server stopped")


async def main():
    """Test the MCP server with a direct client"""
    print("🧪 Testing MCP Server with Direct Client")
    print("=" * 50)

    # Configuration matching the Cursor MCP setup
    script_dir = Path(__file__).parent
    server_command = str(Path.cwd() / ".venv" / "bin" / "python")
    server_args = [str(script_dir / "mcp_server.py")]
    cwd = Path.cwd()

    client = SimpleMCPClient(server_command, server_args, cwd)

    try:
        # Start server
        await client.start_server()

        # Initialize connection
        print("\n📋 Step 1: Initialize MCP connection")
        init_response = await client.initialize()

        # List tools
        print("\n📋 Step 2: List available tools")
        tools_response = await client.list_tools()

        if 'result' in tools_response:
            tools = tools_response['result']['tools']
            print(f"✅ Found {len(tools)} tools:")
            for tool in tools:
                print(f"  • {tool['name']}: {tool['description']}")

        # Test generate_inference_pipeline with missing config to verify error handling
        print("\n📋 Step 3: Test generate_inference_pipeline error handling")
        gen_response = await client.call_tool("generate_inference_pipeline", {
            "config_file": "/nonexistent/path/config.yaml"
        })

        if 'error' in gen_response or ('result' in gen_response and gen_response['result'].get('isError')):
            print("✅ generate_inference_pipeline correctly returned error for missing config")
        else:
            print(f"📋 generate_inference_pipeline response: {gen_response}")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
