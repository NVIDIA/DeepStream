#!/usr/bin/env python3
"""
Setup script for Inference Builder MCP Integration

This script helps install dependencies and configure the MCP integration
for use with MCP-compatible clients such as Cursor and Claude Code.

Usage:
    python setup_mcp.py [CONFIG_PATH]

Arguments:
    CONFIG_PATH  Path to the MCP config file (default: ~/.cursor/mcp.json)

Examples:
    Cursor (global):    python setup_mcp.py ~/.cursor/mcp.json
    Cursor (project):   python setup_mcp.py /path/to/project/.cursor/mcp.json
    Claude Code:        python setup_mcp.py ~/.claude/.mcp.json
"""

import argparse
import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✓ {description} completed successfully")
        if result.stdout:
            print(f"Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed")
        print(f"Error: {e.stderr.strip()}")
        return False
    except FileNotFoundError:
        print(f"✗ Command not found: {cmd[0]}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"✗ Python 3.8+ required, found {version.major}.{version.minor}")
        return False
    print(f"✓ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def test_integration():
    """Test the MCP integration"""
    print("\nTesting MCP integration...")

    if not run_command([sys.executable, "mcp/test_mcp_server.py"], "Running integration tests"):
        print("Integration tests failed. Please check the errors above.")
        return False

    return True

def create_mcp_config(config_path=None):
    """Create or update MCP configuration file

    Args:
        config_path: Optional path to the MCP config file.
                     Defaults to ~/.cursor/mcp.json if not specified.
    """
    import json

    print("\nSetting up MCP configuration...")

    if config_path:
        config_file = Path(config_path).expanduser().resolve()
        config_dir = config_file.parent
        if not config_dir.exists():
            config_dir.mkdir(parents=True, exist_ok=True)
            print(f"✓ Created directory: {config_dir}")
    else:
        default_dir = Path.home() / ".cursor"
        if not default_dir.exists():
            default_dir.mkdir(parents=True, exist_ok=True)
            print(f"✓ Created directory: {default_dir}")
        config_file = default_dir / "mcp.json"

    # Define the server configuration for this project
    server_name = "deepstream-inference-builder"
    server_config = {
        "command": str(Path(sys.executable)),
        "args": [str(Path.cwd() / "mcp" / "mcp_server.py")],
        "cwd": str(Path.cwd()),
        "env": {
            "PYTHONPATH": str(Path.cwd() / "mcp")
        }
    }

    # Load existing config or start with empty structure
    existing_config = {"mcpServers": {}}
    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                existing_config = json.load(f)
            # Ensure mcpServers key exists
            if "mcpServers" not in existing_config:
                existing_config["mcpServers"] = {}
            print(f"✓ Found existing config with {len(existing_config['mcpServers'])} server(s)")
        except json.JSONDecodeError as e:
            print(f"⚠️  Existing config is not valid JSON: {e}")
            print("Creating new configuration (existing file will be overwritten)...")
            existing_config = {"mcpServers": {}}
        except Exception as e:
            print(f"⚠️  Could not read existing config: {e}")
            print("Creating new configuration...")
            existing_config = {"mcpServers": {}}

    # Check if the server is already correctly configured
    existing_server = existing_config["mcpServers"].get(server_name, {})
    if (existing_server.get("cwd") == server_config["cwd"] and
        existing_server.get("command") == server_config["command"]):
        print(f"✓ MCP server '{server_name}' already exists and is correct: {config_file}")
        return True

    # Update or add the server configuration (preserving other servers)
    if server_name in existing_config["mcpServers"]:
        print(f"⚠️  Updating existing '{server_name}' server configuration...")
    else:
        print(f"Adding '{server_name}' server to configuration...")

    existing_config["mcpServers"][server_name] = server_config

    # Write the merged config back
    try:
        with open(config_file, 'w') as f:
            json.dump(existing_config, f, indent=2)
        print(f"✓ Updated MCP config: {config_file}")
        print(f"  Total servers configured: {len(existing_config['mcpServers'])}")
        print("Note: You may need to restart your MCP client for the configuration to take effect.")
        return True
    except Exception as e:
        print(f"✗ Failed to update MCP config: {e}")
        print("Please manually update the MCP configuration file.")
        return False

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Setup script for Inference Builder MCP Integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python setup_mcp.py                                  # Default: ~/.cursor/mcp.json
    python setup_mcp.py ~/.cursor/mcp.json               # Cursor (global)
    python setup_mcp.py .cursor/mcp.json                 # Cursor (project-specific)
    python setup_mcp.py ~/.claude/.mcp.json              # Claude Code (global)
    python setup_mcp.py .mcp.json                        # Claude Code (project-specific)
        """
    )
    parser.add_argument(
        "config_path",
        nargs="?",
        default=None,
        help="Path to the MCP config file (default: ~/.cursor/mcp.json)"
    )
    return parser.parse_args()


def main(config_path=None):
    """Main setup function

    Args:
        config_path: Optional path to the MCP config file.
    """
    print("Inference Builder MCP Integration Setup")
    print("=" * 50)

    # Check Python version
    if not check_python_version():
        print("\n❌ Setup failed: Python version incompatible")
        return False

    # Test integration
    if not test_integration():
        print("\n❌ Setup failed: Integration tests failed")
        return False

    # Setup MCP config
    create_mcp_config(config_path)

    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Restart your MCP client (Cursor, Claude Code, etc.)")
    print("2. Verify the 'deepstream-inference-builder' MCP server is connected")
    print("3. Try using the tools:")
    print("   - 'Show me what sample configurations are available from the inference builder?'")
    print("   - 'Generate a DeepStream object detection pipeline using the inference builder with PeopleNet transformer model from NGC.'")
    print("\nFor more information, see mcp/README-MCP.md")

    return True


if __name__ == "__main__":
    args = parse_args()
    success = main(args.config_path)
    sys.exit(0 if success else 1)

