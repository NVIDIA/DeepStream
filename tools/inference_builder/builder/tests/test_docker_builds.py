#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
Test script for Docker container builds with different arguments.
This script tests the Dockerfile in the tests directory with various configurations.

Security Features:
- Input validation for app names and script commands to prevent command injection
- File path validation to prevent directory traversal attacks
- Safe subprocess calls using argument lists instead of shell string interpolation
- Logging of all executed commands for audit trails
"""

import subprocess
import sys
import os
import json
import time
import argparse
from pathlib import Path
import shutil
import socket
from typing import Dict, List, Optional, Tuple
import logging
import re
import urllib.request
import urllib.error
from huggingface_hub import snapshot_download

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def validate_app_name(app_name: str) -> bool:
    """Validate app name to prevent command injection."""
    # Allow only alphanumeric characters, underscores, and hyphens
    return bool(re.match(r'^[a-zA-Z0-9_-]+$', app_name))


def validate_script_command(script_command: str) -> bool:
    """Validate script command using a flexible approach that allows legitimate arguments."""
    if not isinstance(script_command, str) or not script_command.strip():
        return False

    script_command = script_command.strip()

    # Split command into parts for analysis
    try:
        import shlex
        parts = shlex.split(script_command)
    except ValueError:
        # Invalid shell syntax
        return False

    if not parts:
        return False

    # Check the main command/script
    main_command = parts[0]

    # Allow specific known scripts
    allowed_script_names = [
        "setup_rtsp_server.sh",
        "./setup_rtsp_server.sh",
        "prepare_engine.sh",
        "./prepare_engine.sh",
    ]

    allowed_shell_commands = ["bash", "sh"]

    # Validate main command
    if main_command not in allowed_script_names and main_command not in allowed_shell_commands:
        return False

    # If it's a shell command, check the script being executed
    if main_command in allowed_shell_commands and len(parts) > 1:
        script_name = parts[1]
        if script_name not in allowed_script_names:
            return False

    # Validate all arguments
    for arg in parts[1:]:
        if not validate_script_arg(arg):
            return False

    # Check for dangerous patterns in the full command
    dangerous_patterns = [
        r'[;&|`]',        # Shell metacharacters (but allow some like spaces)
        r'\$\(',          # Command substitution
        r'`',             # Backticks
        r'>>?',           # Redirections
        r'\|\|',          # OR operator
        r'&&',            # AND operator
        r'<',             # Input redirection
        r'\x00',          # Null bytes
    ]

    for pattern in dangerous_patterns:
        if re.search(pattern, script_command):
            return False

    return True


def validate_script_arg(arg: str) -> bool:
    """Validate individual script argument."""
    if not isinstance(arg, str):
        return False

    # Check for null bytes and control characters
    if '\x00' in arg or any(ord(c) < 32 for c in arg if c not in ['\t']):
        return False

    # Allow common script arguments
    dangerous_chars = ['`', '$', ';', '|', '>', '<', '(', ')']
    if any(char in arg for char in dangerous_chars):
        return False

    # Allow legitimate flags and file arguments
    if arg.startswith('-'):
        # Allow common flag patterns
        if not re.match(r'^-{1,2}[a-zA-Z0-9][a-zA-Z0-9_-]*$', arg):
            return False

    # Prevent excessively long arguments
    if len(arg) > 1024:
        return False

    return True


def validate_safe_path(path: str) -> bool:
    """Validate file path to prevent directory traversal and command injection."""
    if not path or not isinstance(path, str):
        return False

    # Check for double slashes
    if '//' in path:
        return False

    # Check for absolute paths that could access system directories
    if os.path.isabs(path):
        system_dirs = ['/etc', '/sys', '/proc', '/dev', '/boot', '/usr/bin', '/usr/sbin', '/root']
        for sys_dir in system_dirs:
            if path.startswith(sys_dir):
                return False

    # Check for invalid characters
    invalid_chars = ['<', '>', ':', '"', '|', '?', '*', '\\']
    if any(char in path for char in invalid_chars):
        return False

    return True

def validate_config_file_path(config_file: str) -> bool:
    """Validate configuration file path for security."""
    if not validate_safe_path(config_file):
        return False

    # Ensure the filename ends with .json
    filename = os.path.basename(config_file)
    if not filename.lower().endswith('.json'):
        return False

    return True

def validate_dockerfile_path(dockerfile: str) -> bool:
    """Validate Dockerfile path for security."""
    if not validate_safe_path(dockerfile):
        return False

    # Ensure the filename is Dockerfile or has .dockerfile extension
    filename = os.path.basename(dockerfile)
    if not (filename.lower() == 'dockerfile' or filename.lower().endswith('.dockerfile')):
        return False

    return True

def validate_log_directory(log_dir: str) -> bool:
    """Validate log directory path for security."""
    if not validate_safe_path(log_dir):
        return False

    # Prevent access to system directories
    if log_dir.startswith('/') and not log_dir.startswith('./'):
        return False

    return True

def validate_gitlab_token(token: str) -> bool:
    """Validate GitLab token format."""
    if not token:
        return True  # Empty token is allowed

    # Basic validation for GitLab token format
    if not isinstance(token, str) or len(token) < 10:
        return False

    # Check for suspicious patterns
    suspicious_patterns = ['<script', 'javascript:', 'data:', 'vbscript:']
    if any(pattern in token.lower() for pattern in suspicious_patterns):
        return False

    return True


def validate_docker_arg(arg: str) -> bool:
    """Validate docker argument to prevent command injection while allowing legitimate flags."""
    if not isinstance(arg, str):
        return False

    # Check for null bytes and control characters
    if '\x00' in arg or any(ord(c) < 32 for c in arg if c not in ['\t', '\n', '\r']):
        return False

    # Check for dangerous characters and patterns (but allow legitimate usage)
    dangerous_patterns = [
        r'[;|`$()]',      # Shell metacharacters (removed & for now)
        r'\$\(',          # Command substitution
        r'`',             # Backticks
        r'>>?',           # Redirections
        r'\|\|',          # OR operator
        r'&&',            # AND operator for command chaining
        r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]',  # Control characters
    ]

    for pattern in dangerous_patterns:
        if re.search(pattern, arg):
            return False

    # Check for dangerous & usage (but allow in URL query parameters)
    if '&' in arg:
        # Allow & in URL-like contexts (after ?)
        if '?' not in arg:
            # & without ? suggests command chaining, not URL parameter
            return False
        # Check for command chaining patterns even with ?
        if ' &' in arg or '& ' in arg or arg.endswith('&'):
            return False

    # Additional dangerous characters for docker contexts (be selective)
    # Note: Removed '?' to allow URL query parameters, '*' and '!' for legitimate use
    dangerous_chars = ['[', ']', '{', '}', '\\']
    if any(char in arg for char in dangerous_chars):
        return False

    # Check for wildcard patterns that could be dangerous in specific contexts
    if '*' in arg and any(pattern in arg for pattern in ['*.*', '*.sh', '*.py', '*/', '/*']):
        return False

    # Allow legitimate command-line flags but prevent injection attempts
    if arg.strip().startswith('-'):
        # Allow common legitimate patterns
        legitimate_flag_patterns = [
            r'^--[a-zA-Z0-9][a-zA-Z0-9_-]*$',           # --flag-name
            r'^--[a-zA-Z0-9][a-zA-Z0-9_-]*=.*$',        # --flag=value
            r'^-[a-zA-Z0-9]$',                          # -f
            r'^-[a-zA-Z0-9][a-zA-Z0-9]*$',              # -abc
        ]

        # Check if it matches any legitimate pattern
        is_legitimate = any(re.match(pattern, arg) for pattern in legitimate_flag_patterns)

        if not is_legitimate:
            return False

        # Additional checks for flag values (after =)
        if '=' in arg:
            flag_value = arg.split('=', 1)[1]
            # Check flag value for dangerous patterns
            if any(char in flag_value for char in ['`', '$', ';', '&', '|', '(', ')']):
                return False

    # Prevent excessively long arguments (potential DoS)
    if len(arg) > 8192:  # Reasonable limit for docker arguments
        return False

    # Check for potential escape sequences in non-flag arguments
    if not arg.startswith('-') and '\\' in arg:
        if any(seq in arg for seq in ['\\n', '\\r', '\\t', '\\x', '\\u']):
            return False

    return True


def validate_env_var_name(name: str) -> bool:
    """Validate environment variable name."""
    if not isinstance(name, str) or not name:
        return False

    # Environment variable names should be alphanumeric + underscore, starting with letter or underscore
    return bool(re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', name))


def validate_env_var_value(value: str) -> bool:
    """Validate environment variable value."""
    if not isinstance(value, str):
        return False

    # Check for null bytes
    if '\x00' in value:
        return False

    # Check for dangerous shell characters
    dangerous_chars = ['`', '$', ';', '&', '|', '(', ')', '>', '<']
    if any(char in value for char in dangerous_chars):
        return False

    return True


def validate_http_method(method: str) -> bool:
    """Validate HTTP method."""
    if not isinstance(method, str):
        return False

    # Allow common HTTP methods
    allowed_methods = ['GET', 'POST', 'PUT', 'PATCH', 'DELETE', 'HEAD', 'OPTIONS']
    return method.upper() in allowed_methods


def validate_http_header_name(name: str) -> bool:
    """Validate HTTP header name."""
    if not isinstance(name, str) or not name:
        return False

    # Header names should be alphanumeric with hyphens
    # Follow RFC 7230 field-name = token
    if not re.match(r'^[a-zA-Z0-9!#$%&\'*+\-.^_`|~]+$', name):
        return False

    # Prevent excessively long header names
    if len(name) > 256:
        return False

    return True


def validate_http_header_value(value: str) -> bool:
    """Validate HTTP header value."""
    if not isinstance(value, str):
        return False

    # Check for null bytes and control characters (except tab and space)
    if '\x00' in value or any(ord(c) < 32 for c in value if c not in ['\t', ' ']):
        return False

    # Check for newlines that could cause header injection
    if '\n' in value or '\r' in value:
        return False

    # Prevent excessively long header values
    if len(value) > 8192:
        return False

    return True


def validate_volume_path(path: str) -> bool:
    """Validate volume mount path to prevent injection attacks.

    Note: Allows legitimate relative paths with .. for test configs.
    """
    if not isinstance(path, str) or not path:
        return False

    # Check for null bytes and control characters
    if '\x00' in path or any(ord(c) < 32 for c in path if c not in ['\t']):
        return False

    # Check for dangerous path patterns (but allow relative paths)
    # Allow: ../tao/, ./models, ../../shared
    # Reject: /../../../etc/passwd (absolute path traversal attacks)

    # Check for double slashes (suspicious)
    if '//' in path:
        return False

    # Check for Windows-style backslashes (not needed in Docker contexts)
    if '\\' in path:
        return False

    # Check for absolute path traversal attacks (starting with / and going up)
    # This would be trying to escape from an absolute path to system directories
    if path.startswith('/') and '/../' in path:
        # Check if trying to access system directories
        system_dirs = ['/../etc', '/../root', '/../sys', '/../proc', '/../dev']
        if any(path.startswith(sys_dir) for sys_dir in system_dirs):
            return False

    # Check for dangerous shell characters and command injection attempts
    # Allow ~ for home directory expansion at the start of path
    dangerous_chars = ['`', '$', ';', '&', '|', '(', ')', '>', '<', '*', '?', '!', '[', ']', '{', '}']
    if any(char in path for char in dangerous_chars):
        return False

    # Allow ~ only at the start of path (for home directory expansion)
    if '~' in path and not path.startswith('~'):
        return False

    # Check for spaces at beginning/end (could be injection attempts)
    if path.startswith(' ') or path.endswith(' '):
        return False

    # Check for argument injection (starting with dash)
    if path.startswith('-'):
        return False

    # Ensure path doesn't contain colon (except for Windows drive letters or container paths)
    colon_count = path.count(':')
    if colon_count > 1:  # Allow one colon for Windows drive letters or container paths
        return False
    if colon_count == 1:
        # Allow Windows drive letters (C:) or absolute container paths (/app:)
        if not (re.match(r'^[A-Za-z]:', path) or ':' in path[1:]):
            return False

    # Additional check: ensure reasonable path length to prevent buffer overflow attacks
    if len(path) > 4096:  # Most systems limit paths to 4096 characters
        return False

    return True


def validate_build_arg_name(name: str) -> bool:
    """Validate Docker build argument name to prevent injection."""
    if not isinstance(name, str) or not name:
        return False

    # Build arg names should be alphanumeric + underscore, no dashes at start
    if name.startswith('-'):
        return False

    # Allow standard environment variable naming convention
    return bool(re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', name))


def validate_build_arg_value(value: str) -> bool:
    """Validate Docker build argument value to prevent injection."""
    if not isinstance(value, str):
        return False

    # Check for null bytes
    if '\x00' in value:
        return False

    # Check for dangerous shell characters that could cause issues
    dangerous_chars = ['`', '$', ';', '&', '|', '(', ')', '>', '<', '\n', '\r']
    if any(char in value for char in dangerous_chars):
        return False

    # Check for argument injection attempts
    if value.strip().startswith('-'):
        return False

    return True


def validate_image_name(image_name: str) -> bool:
    """Validate Docker image name to prevent injection."""
    if not isinstance(image_name, str) or not image_name:
        return False

    # Check for null bytes
    if '\x00' in image_name:
        return False

    # Check for dangerous characters
    dangerous_chars = ['`', '$', ';', '&', '|', '(', ')', '>', '<', ' ', '\n', '\r']
    if any(char in image_name for char in dangerous_chars):
        return False

    # Basic docker image name validation (simplified)
    # Allow alphanumeric, hyphens, underscores, slashes, colons, dots
    if not re.match(r'^[a-zA-Z0-9._/-]+(?::[a-zA-Z0-9._-]+)?$', image_name):
        return False

    return True


def validate_test_config(test_config: dict) -> Tuple[bool, str]:
    """Validate test configuration to prevent command injection."""
    if not isinstance(test_config, dict):
        return False, "Test config must be a dictionary"

    # Validate environment variables
    if "env" in test_config:
        if not isinstance(test_config["env"], dict):
            return False, "env must be a dictionary"

        for key, value in test_config["env"].items():
            if not validate_env_var_name(key):
                return False, f"Invalid environment variable name: {key}"
            if not validate_env_var_value(str(value)):
                return False, f"Invalid environment variable value for {key}: {value}"

    # Validate volume mounts
    if "volumes" in test_config:
        if not isinstance(test_config["volumes"], dict):
            return False, "volumes must be a dictionary"

        for host_path, container_path in test_config["volumes"].items():
            if not validate_volume_path(host_path):
                return False, f"Invalid host path in volume mount: {host_path}"
            if not validate_volume_path(container_path):
                return False, f"Invalid container path in volume mount: {container_path}"

    # Validate command arguments
    if "cmd" in test_config:
        if not isinstance(test_config["cmd"], list):
            return False, "cmd must be a list"

        for arg in test_config["cmd"]:
            if not validate_docker_arg(str(arg)):
                return False, f"Invalid command argument: {arg}"

    # Validate timeout
    if "timeout" in test_config:
        if not isinstance(test_config["timeout"], (int, float)) or test_config["timeout"] <= 0:
            return False, "timeout must be a positive number"
        if test_config["timeout"] > 3600:  # Max 1 hour
            return False, "timeout cannot exceed 3600 seconds"

    # Validate prerequisite script
    if "prerequisite_script" in test_config:
        script = test_config["prerequisite_script"]
        if script and not validate_script_command(script):
            return False, f"Invalid prerequisite script: {script}"

    # Validate auto_validation configuration
    if "auto_validation" in test_config:
        if not isinstance(test_config["auto_validation"], str):
            return False, "auto_validation must be a string path"
        if not validate_safe_path(test_config["auto_validation"]):
            return False, f"Invalid auto_validation path: {test_config['auto_validation']}"
    # Validate test_requests (must be a list of objects with optional payload_path, method, headers)
    if "test_requests" in test_config:
        test_requests = test_config["test_requests"]
        if not isinstance(test_requests, list):
            return False, "test_requests must be a list"

        for item in test_requests:
            # Must be dict format with payload_path (optional), method (optional), and headers (optional)
            if not isinstance(item, dict):
                return False, "test_requests items must be dictionaries"

            # Validate payload_path if present (optional for GET requests without body)
            if "payload_path" in item:
                payload_path = item["payload_path"]
                if not isinstance(payload_path, str):
                    return False, "test_requests 'payload_path' must be a string"
                if not validate_safe_path(payload_path):
                    return False, f"Invalid test_requests payload_path: {payload_path}"

            # Validate method if specified (optional, defaults to POST)
            if "method" in item:
                method = item["method"]
                if not validate_http_method(method):
                    return False, f"Invalid HTTP method in payload_config: {method}"

            # Validate headers if specified (optional)
            if "headers" in item:
                headers = item["headers"]
                if not isinstance(headers, dict):
                    return False, "payload_config headers must be a dictionary"
                for header_name, header_value in headers.items():
                    if not validate_http_header_name(header_name):
                        return False, f"Invalid HTTP header name in payload_config: {header_name}"
                    if not validate_http_header_value(str(header_value)):
                        return False, f"Invalid HTTP header value in payload_config for {header_name}: {header_value}"

            # Validate scheduled_time if specified (optional, for sequential execution)
            if "scheduled_time" in item:
                if not isinstance(item["scheduled_time"], (int, float)) or item["scheduled_time"] < 0:
                    return False, "test_requests 'scheduled_time' must be a non-negative number"

            # Validate async if specified (optional, for non-blocking requests)
            if "async" in item:
                if not isinstance(item["async"], bool):
                    return False, "test_requests 'async' must be a boolean"

    # Validate endpoint
    if "endpoint" in test_config:
        endpoint = test_config["endpoint"]
        if not isinstance(endpoint, str):
            return False, "endpoint must be a string"
        # Basic validation for endpoint path format
        if not endpoint.startswith("/"):
            return False, "endpoint must start with /"
        # Check for dangerous characters
        if any(char in endpoint for char in ['<', '>', '"', '{', '}', '|', '\\', '^', '`', ' ']):
            return False, f"Invalid characters in endpoint: {endpoint}"

    # Validate expected_error for negative tests
    if "expected_error" in test_config:
        expected_error = test_config["expected_error"]
        if not isinstance(expected_error, str):
            return False, "expected_error must be a string"
        if not expected_error:
            return False, "expected_error cannot be empty"
        # Basic validation - should be an error code pattern (e.g., ERR_ROUTE_001)
        if len(expected_error) > 100:
            return False, "expected_error is too long (max 100 characters)"

    return True, ""


class DockerBuildTester:
    def __init__(self, dockerfile_path: str, base_dir: str, log_dir: Optional[str] = None):
        self.dockerfile_path = Path(dockerfile_path)
        self.base_dir = Path(base_dir)
        self.log_dir = Path(log_dir) if log_dir else Path("logs")
        self.test_results = []

        # Create log directory if it doesn't exist
        self.log_dir.mkdir(exist_ok=True)

    def download_models(self, models_config: Dict, config_dir: Path) -> Tuple[bool, str]:
        """Download models from NGC based on the models configuration.

        Args:
            models_config: Dictionary with model configurations
            config_dir: Directory containing the test config (for resolving relative paths)

        Returns:
            Tuple of (success, message)
        """
        if not models_config:
            return True, "No models to download"

        for model_name, model_info in models_config.items():
            try:
                source = model_info.get("source", "NGC")

                if source == "HF":
                    # Handle Hugging Face model download
                    target_dir = Path(model_info["target"]).expanduser()
                    model_path = model_info["path"]  # e.g., "Qwen/Qwen2.5-VL-3B-Instruct"

                    # Final destination
                    final_model_dir = target_dir / model_name

                    # Check if model already exists
                    if final_model_dir.exists():
                        logger.info(f"✅ Model '{model_name}' already exists at {final_model_dir}, skipping download")
                        continue

                    logger.info(f"📥 Downloading model '{model_name}' from Hugging Face...")
                    logger.info(f"   HF path: {model_path}")
                    logger.info(f"   Target: {final_model_dir}")

                    # Create target directory
                    target_dir.mkdir(parents=True, exist_ok=True)

                    # Download using huggingface_hub
                    hf_token = os.environ.get("HF_TOKEN")
                    try:
                        snapshot_download(
                            model_path,
                            local_dir=str(final_model_dir),
                            token=hf_token,
                        )
                    except Exception as e:
                        error_msg = f"Failed to download model '{model_name}' from HF: {e}"
                        logger.error(f"❌ {error_msg}")
                        return False, error_msg

                    # Execute post-script if specified (for HF models)
                    post_script = model_info.get("post_script")
                    if post_script:
                        logger.info(f"🔧 Executing post-script for '{model_name}': {post_script}")
                        try:
                            script_result = subprocess.run(
                                post_script,
                                shell=True,
                                capture_output=True,
                                text=True,
                                cwd=str(final_model_dir),
                                timeout=600
                            )
                            if script_result.returncode == 0:
                                logger.info(f"✅ Post-script executed successfully for '{model_name}'")
                                if script_result.stdout.strip():
                                    logger.info(f"   Output: {script_result.stdout.strip()}")
                            else:
                                error_msg = (
                                    f"Post-script failed for '{model_name}' "
                                    f"(exit code {script_result.returncode}): "
                                    f"{script_result.stderr.strip()}"
                                )
                                logger.error(f"❌ {error_msg}")
                                return False, error_msg
                        except subprocess.TimeoutExpired:
                            error_msg = f"Post-script timed out for '{model_name}' after 600 seconds"
                            logger.error(f"❌ {error_msg}")
                            return False, error_msg
                        except Exception as e:
                            error_msg = f"Failed to execute post-script for '{model_name}': {str(e)}"
                            logger.error(f"❌ {error_msg}")
                            return False, error_msg

                    logger.info(f"✅ Successfully downloaded model '{model_name}' from Hugging Face")
                    continue

                if source != "NGC":
                    logger.warning(f"⚠️  Unsupported model source '{source}' for {model_name}, skipping")
                    continue

                # NGC model download logic
                target_dir = Path(model_info["target"]).expanduser()
                model_path = model_info["path"]
                version = model_info["version"]
                configs_path = model_info.get("configs", "")

                # Construct the full NGC path
                ngc_path = f"{model_path}:{version}"

                # Determine the downloaded folder name (NGC naming convention)
                # e.g., "nvidia/tao/grounding_dino:grounding_dino_swin_tiny_commercial_deployable_v1.0"
                # becomes "grounding_dino_vgrounding_dino_swin_tiny_commercial_deployable_v1.0"
                model_base_name = model_path.split('/')[-1]  # e.g., "grounding_dino"
                downloaded_folder = f"{model_base_name}_v{version}"

                # Final destination
                final_model_dir = target_dir / model_name

                # Check if model already exists
                if final_model_dir.exists():
                    logger.info(f"✅ Model '{model_name}' already exists at {final_model_dir}, skipping download")
                    continue

                logger.info(f"📥 Downloading model '{model_name}' from NGC...")
                logger.info(f"   NGC path: {ngc_path}")
                logger.info(f"   Target: {final_model_dir}")

                # Create target directory
                target_dir.mkdir(parents=True, exist_ok=True)

                # Download from NGC
                download_cmd = ["ngc", "registry", "model", "download-version", ngc_path]
                logger.info(f"   Running: {' '.join(download_cmd)}")

                result = subprocess.run(
                    download_cmd,
                    capture_output=True,
                    text=True,
                    timeout=600  # 10 minute timeout for download
                )

                if result.returncode != 0:
                    error_msg = f"Failed to download model '{model_name}': {result.stderr}"
                    logger.error(f"❌ {error_msg}")
                    return False, error_msg

                # Move the downloaded folder to the target location
                downloaded_path = Path(downloaded_folder)
                if not downloaded_path.exists():
                    error_msg = f"Downloaded folder not found: {downloaded_path}"
                    logger.error(f"❌ {error_msg}")
                    return False, error_msg

                logger.info(f"   Moving {downloaded_path} to {final_model_dir}")
                shutil.move(str(downloaded_path), str(final_model_dir))

                # Set permissions
                try:
                    os.chmod(final_model_dir, 0o777)
                    logger.info(f"   Set permissions: chmod 777 {final_model_dir}")
                except Exception as e:
                    logger.warning(f"⚠️  Failed to set permissions on {final_model_dir}: {e}")

                # Copy config files if specified
                if configs_path:
                    # Resolve configs_path relative to test config directory
                    source_configs = (config_dir / configs_path).resolve()
                    if source_configs.exists():
                        logger.info(f"   Copying configs from {source_configs} to {final_model_dir}")
                        # Copy all files from source_configs to final_model_dir
                        for item in source_configs.iterdir():
                            dest = final_model_dir / item.name
                            if item.is_file():
                                shutil.copy2(str(item), str(dest))
                            elif item.is_dir():
                                shutil.copytree(str(item), str(dest), dirs_exist_ok=True)
                        logger.info(f"   ✅ Copied config files")
                    else:
                        logger.warning(f"⚠️  Config path not found: {source_configs}")

                # Execute post-script if specified
                post_script = model_info.get("post_script")
                if post_script:
                    logger.info(f"🔧 Executing post-script for '{model_name}': {post_script}")
                    try:
                        script_result = subprocess.run(
                            post_script,
                            shell=True,
                            capture_output=True,
                            text=True,
                            cwd=str(final_model_dir),
                            timeout=600
                        )
                        if script_result.returncode == 0:
                            logger.info(f"✅ Post-script executed successfully for '{model_name}'")
                            if script_result.stdout.strip():
                                logger.info(f"   Output: {script_result.stdout.strip()}")
                        else:
                            error_msg = (
                                f"Post-script failed for '{model_name}' "
                                f"(exit code {script_result.returncode}): "
                                f"{script_result.stderr.strip()}"
                            )
                            logger.error(f"❌ {error_msg}")
                            return False, error_msg
                    except subprocess.TimeoutExpired:
                        error_msg = f"Post-script timed out for '{model_name}' after 600 seconds"
                        logger.error(f"❌ {error_msg}")
                        return False, error_msg
                    except Exception as e:
                        error_msg = f"Failed to execute post-script for '{model_name}': {str(e)}"
                        logger.error(f"❌ {error_msg}")
                        return False, error_msg

                logger.info(f"✅ Successfully downloaded and setup model '{model_name}'")

            except KeyError as e:
                error_msg = f"Missing required field in model config for '{model_name}': {e}"
                logger.error(f"❌ {error_msg}")
                return False, error_msg
            except subprocess.TimeoutExpired:
                error_msg = f"Model download timed out for '{model_name}'"
                logger.error(f"❌ {error_msg}")
                return False, error_msg
            except Exception as e:
                error_msg = f"Exception during model download for '{model_name}': {str(e)}"
                logger.error(f"❌ {error_msg}")
                return False, error_msg

        return True, "All models downloaded successfully"

    def get_service_host(self) -> str:
        """
        Determine the appropriate host to use for connecting to Docker
        containers. In CI environments (like GitLab CI with
        Docker-in-Docker), 127.0.0.1 won't work because each container has
        its own localhost. Use Docker gateway IP instead.
        """
        # Check if running in CI environment
        if os.environ.get('CI') or os.environ.get('GITLAB_CI'):
            logger.info(
                "🔍 CI environment detected, using Docker gateway IP "
                "for service connectivity"
            )
            # Try to get docker gateway IP from bridge network
            try:
                result = subprocess.run(
                    [
                        "docker", "network", "inspect", "bridge", "-f",
                        "{{range .IPAM.Config}}{{.Gateway}}{{end}}"
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0 and result.stdout.strip():
                    gateway_ip = result.stdout.strip()
                    logger.info(f"📡 Using Docker gateway IP: {gateway_ip}")
                    return gateway_ip
            except Exception as e:
                logger.warning(f"⚠️  Failed to get Docker gateway IP: {e}")

            # Fallback to default Docker bridge gateway
            logger.info(
                "📡 Using default Docker bridge gateway: 172.17.0.1"
            )
            return "172.17.0.1"

        # Local development environment - use localhost
        logger.info("📡 Using localhost for service connectivity")
        return "127.0.0.1"

    def generate_inference_code(self, build_args: Dict[str, str], test_config: Dict = None) -> Tuple[bool, str]:
        """Generate inference code (codegen) without building or testing Docker images.

        This runs the same pre-build code generation step used by build_image(),
        including optional OPENAPI_SPEC staging, but skips Docker build.

        Supports flexible path configuration via build_args:
        - APP_YAML_PATH: Custom path to app.yaml (overrides {TEST_APP_NAME}/app.yaml)
        - OUTPUT_DIR: Custom output directory (overrides {TEST_APP_NAME})
        - PROCESSORS_PATH: Custom processors.py path (overrides auto-detection)

        Args:
            build_args: Build arguments for code generation
            test_config: Optional test configuration. If present and contains 'auto_validation',
                        the validation directory will be passed to code generation via -v flag.
        """
        try:
            # Validate all build arguments to prevent command injection
            for key, value in build_args.items():
                if not validate_build_arg_name(key):
                    error_msg = f"Invalid build arg name: {key}. Only alphanumeric characters and underscores allowed."
                    logger.error(f"❌ {error_msg}")
                    return False, error_msg
                if not validate_build_arg_value(str(value)):
                    error_msg = f"Invalid build arg value for {key}: {value}. Value contains dangerous characters."
                    logger.error(f"❌ {error_msg}")
                    return False, error_msg

            # Determine parameters for code generation
            test_app_name = build_args.get("TEST_APP_NAME", "frame_sampling")
            if not validate_app_name(test_app_name):
                error_msg = (
                    f"Invalid app name: {test_app_name}. Only alphanumeric characters, underscores, and hyphens are allowed."
                )
                logger.error(f"❌ {error_msg}")
                return False, error_msg

            server_type = build_args.get("SERVER_TYPE", "serverless")
            openapi_spec = build_args.get("OPENAPI_SPEC")

            # Support flexible paths via build_args
            app_yaml_path = build_args.get("APP_YAML_PATH", f"{test_app_name}/app.yaml")
            output_dir = build_args.get("OUTPUT_DIR", test_app_name)
            processors_path_arg = build_args.get("PROCESSORS_PATH", "")

            # Find main.py relative to this script (in builder/main.py)
            main_py_path = Path(__file__).parent.parent / "main.py"
            if not main_py_path.exists():
                error_msg = f"Cannot find main.py at {main_py_path}"
                logger.error(f"❌ {error_msg}")
                return False, error_msg

            pre_build_command = [
                "python", str(main_py_path), app_yaml_path,
                "-o", output_dir
            ]

            # Add processors.py if specified or auto-detect
            if processors_path_arg:
                processors_path = Path(processors_path_arg)
                if processors_path.exists():
                    pre_build_command.extend(["-c", processors_path_arg])
                else:
                    logger.warning(f"⚠️  Specified PROCESSORS_PATH not found: {processors_path_arg}")
            else:
                # Auto-detect in output directory
                processors_path = Path(f"{test_app_name}/processors.py")
                if processors_path.exists():
                    pre_build_command.extend(["-c", f"{test_app_name}/processors.py"])

            # Add validation directory if specified in test_config
            if test_config and "auto_validation" in test_config:
                validation_folder = test_config["auto_validation"]
                # Resolve path relative to config directory if it's relative
                config_dir = test_config.get("_config_dir")
                if config_dir:
                    validation_path = (Path(config_dir) / validation_folder).resolve()
                else:
                    validation_path = Path(validation_folder).resolve()

                if validation_path.exists():
                    pre_build_command.extend(["--validation-dir", str(validation_path), "--no-docker"])
                    logger.info(f"📁 Adding validation directory for build: {validation_path}")
                else:
                    logger.warning(f"⚠️  Auto validation folder not found: {validation_path}")

            pre_build_command.extend(["--server-type", server_type, "-t"])

            if openapi_spec:
                # Resolve provided spec relative to project root, copy into local app folder to avoid unsafe paths
                project_root = Path(__file__).resolve().parents[2]
                resolved_spec = None
                candidates = []
                spec_path = Path(openapi_spec)
                if spec_path.is_absolute():
                    candidates.append(spec_path)
                else:
                    candidates.append((project_root / openapi_spec).resolve())
                    candidates.append((project_root / "builder" / openapi_spec).resolve())

                for cand in candidates:
                    try:
                        if cand.exists() and str(cand).startswith(str(project_root)):
                            resolved_spec = cand
                            break
                    except Exception:
                        continue

                if not resolved_spec:
                    error_msg = f"Invalid OPENAPI_SPEC path: {openapi_spec}"
                    logger.error(f"❌ {error_msg}")
                    return False, error_msg

                local_spec_path = Path(output_dir) / "openapi.yaml"

                # Only copy if source and destination are different
                if resolved_spec.resolve() != local_spec_path.resolve():
                    try:
                        shutil.copyfile(str(resolved_spec), str(local_spec_path))
                        logger.info(f"📄 Staged OPENAPI_SPEC: {resolved_spec} -> {local_spec_path}")
                    except Exception as e:
                        error_msg = f"Failed to stage OPENAPI_SPEC: {e}"
                        logger.error(f"❌ {error_msg}")
                        return False, error_msg
                else:
                    logger.info(f"📄 OPENAPI_SPEC already in output directory: {local_spec_path}")

                pre_build_command[3:3] = ["-a", str(local_spec_path)]

            logger.info(f"🔧 Executing codegen command: {' '.join(pre_build_command)}")
            pre_build_result = subprocess.run(
                pre_build_command,
                capture_output=True,
                text=True,
                timeout=600
            )

            if pre_build_result.returncode != 0:
                error_msg = f"Code generation failed: {pre_build_result.stderr}"
                logger.error(f"❌ {error_msg}")
                return False, error_msg

            logger.info("✅ Code generation completed successfully")
            if pre_build_result.stdout:
                logger.info(f"Codegen output: {pre_build_result.stdout}")
            return True, pre_build_result.stdout

        except subprocess.TimeoutExpired:
            error_msg = "Code generation timed out"
            logger.error(f"❌ {error_msg}")
            return False, error_msg
        except Exception as e:
            error_msg = f"Exception during code generation: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return False, error_msg

    def build_image(self, build_args: Dict[str, str], image_name: str, dockerfile: Optional[str] = None, base_dir: Optional[str] = None, test_config: Dict = None) -> Tuple[bool, str]:
        """Build Docker image with given arguments.

        Supports flexible path configuration via build_args:
        - APP_YAML_PATH: Custom path to app.yaml (overrides {TEST_APP_NAME}/app.yaml)
        - OUTPUT_DIR: Custom output directory (overrides {TEST_APP_NAME})
        - PROCESSORS_PATH: Custom processors.py path (overrides auto-detection)

        Args:
            build_args: Build arguments for the Docker image
            image_name: Name for the Docker image
            dockerfile: Optional path to Dockerfile (overrides self.dockerfile_path)
            base_dir: Optional Docker build context directory (overrides self.base_dir)
            test_config: Optional test configuration. If present and contains 'auto_validation',
                        the validation directory will be passed to code generation via -v flag.
        """
        try:
            # Use provided paths or fall back to instance defaults
            dockerfile_path = Path(dockerfile) if dockerfile else self.dockerfile_path
            build_context = Path(base_dir) if base_dir else self.base_dir
            # Validate image name to prevent command injection
            if not validate_image_name(image_name):
                error_msg = f"Invalid image name: {image_name}. Image name contains invalid characters."
                logger.error(f"❌ {error_msg}")
                return False, error_msg

            # Validate all build arguments to prevent command injection
            for key, value in build_args.items():
                if not validate_build_arg_name(key):
                    error_msg = f"Invalid build arg name: {key}. Only alphanumeric characters and underscores allowed."
                    logger.error(f"❌ {error_msg}")
                    return False, error_msg
                if not validate_build_arg_value(str(value)):
                    error_msg = f"Invalid build arg value for {key}: {value}. Value contains dangerous characters."
                    logger.error(f"❌ {error_msg}")
                    return False, error_msg

            # Execute pre-build command using TEST_APP_NAME
            test_app_name = build_args.get("TEST_APP_NAME", "frame_sampling")

            # Validate app name to prevent command injection
            if not validate_app_name(test_app_name):
                error_msg = f"Invalid app name: {test_app_name}. Only alphanumeric characters, underscores, and hyphens are allowed."
                logger.error(f"❌ {error_msg}")
                return False, error_msg

            # Determine server type from build args (fallback to 'serverless')
            server_type = build_args.get("SERVER_TYPE", "serverless")

            # Optional: OpenAPI spec path (to pass -a)
            openapi_spec = build_args.get("OPENAPI_SPEC")

            # Require OpenAPI spec for non-serverless builds
            if server_type != "serverless" and not openapi_spec:
                error_msg = (
                    "OPENAPI_SPEC is required in build_args when SERVER_TYPE is not 'serverless'"
                )
                logger.error(f"❌ {error_msg}")
                return False, error_msg

            # Support flexible paths via build_args
            app_yaml_path = build_args.get("APP_YAML_PATH", f"{test_app_name}/app.yaml")
            output_dir = build_args.get("OUTPUT_DIR", test_app_name)
            processors_path_arg = build_args.get("PROCESSORS_PATH", "")

            # Find main.py relative to this script (in builder/main.py)
            main_py_path = Path(__file__).parent.parent / "main.py"
            if not main_py_path.exists():
                error_msg = f"Cannot find main.py at {main_py_path}"
                logger.error(f"❌ {error_msg}")
                return False, error_msg

            # Use safer subprocess call without shell=True
            pre_build_command = [
                "python", str(main_py_path), app_yaml_path,
                "-o", output_dir
            ]

            # Add processors.py if specified or auto-detect
            if processors_path_arg:
                processors_path = Path(processors_path_arg)
                if processors_path.exists():
                    pre_build_command.extend(["-c", processors_path_arg])
                else:
                    logger.warning(f"⚠️  Specified PROCESSORS_PATH not found: {processors_path_arg}")
            else:
                # Auto-detect in test_app_name directory
                processors_path = Path(f"{test_app_name}/processors.py")
                if processors_path.exists():
                    pre_build_command.extend(["-c", f"{test_app_name}/processors.py"])

            # Add validation directory if specified in test_config
            if test_config and "auto_validation" in test_config:
                validation_folder = test_config["auto_validation"]
                # Resolve path relative to config directory if it's relative
                config_dir = test_config.get("_config_dir")
                if config_dir:
                    validation_path = (Path(config_dir) / validation_folder).resolve()
                else:
                    validation_path = Path(validation_folder).resolve()

                if validation_path.exists():
                    pre_build_command.extend(["--validation-dir", str(validation_path), "--no-docker"])
                    logger.info(f"📁 Adding validation directory for build: {validation_path}")
                else:
                    logger.warning(f"⚠️  Auto validation folder not found: {validation_path}")

            pre_build_command.extend(["--server-type", server_type, "-t"])

            if openapi_spec:
                # Resolve provided spec relative to project root, copy into local app folder to avoid unsafe paths
                project_root = Path(__file__).resolve().parents[2]
                resolved_spec = None
                candidates = []
                spec_path = Path(openapi_spec)
                if spec_path.is_absolute():
                    candidates.append(spec_path)
                else:
                    candidates.append((project_root / openapi_spec).resolve())
                    candidates.append((project_root / "builder" / openapi_spec).resolve())

                for cand in candidates:
                    try:
                        if cand.exists() and str(cand).startswith(str(project_root)):
                            resolved_spec = cand
                            break
                    except Exception:
                        continue

                if not resolved_spec:
                    error_msg = f"Invalid OPENAPI_SPEC path: {openapi_spec}"
                    logger.error(f"❌ {error_msg}")
                    return False, error_msg

                local_spec_path = Path(output_dir) / "openapi.yaml"

                # Only copy if source and destination are different
                if resolved_spec.resolve() != local_spec_path.resolve():
                    try:
                        shutil.copyfile(str(resolved_spec), str(local_spec_path))
                        logger.info(f"📄 Staged OPENAPI_SPEC: {resolved_spec} -> {local_spec_path}")
                    except Exception as e:
                        error_msg = f"Failed to stage OPENAPI_SPEC: {e}"
                        logger.error(f"❌ {error_msg}")
                        return False, error_msg
                else:
                    logger.info(f"📄 OPENAPI_SPEC already in output directory: {local_spec_path}")

                pre_build_command[3:3] = ["-a", str(local_spec_path)]

            logger.info(f"🔧 Executing pre-build command: {' '.join(pre_build_command)}")
            pre_build_result = subprocess.run(
                pre_build_command,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout for pre-build
            )

            if pre_build_result.returncode != 0:
                error_msg = f"Pre-build command failed: {pre_build_result.stderr}"
                logger.error(f"❌ {error_msg}")
                return False, error_msg

            logger.info("✅ Pre-build command completed successfully")
            if pre_build_result.stdout:
                logger.info(f"Pre-build output: {pre_build_result.stdout}")

            cmd = [
                "docker", "build",
                "-f", str(dockerfile_path),
                "-t", image_name
            ]

            # Add build arguments
            for key, value in build_args.items():
                cmd.extend(["--build-arg", f"{key}={value}"])

            # Add context directory
            cmd.append(str(build_context))

            logger.info(f"Building image: {image_name}")
            logger.info(f"Command: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )

            if result.returncode == 0:
                logger.info(f"✅ Successfully built image: {image_name}")
                return True, result.stdout
            else:
                logger.error(f"❌ Failed to build image: {image_name}")
                logger.error(f"Error: {result.stderr}")
                return False, result.stderr

        except subprocess.TimeoutExpired:
            error_msg = f"Build timed out for image: {image_name}"
            logger.error(f"❌ {error_msg}")
            return False, error_msg
        except Exception as e:
            error_msg = f"Exception during build: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return False, error_msg

    def run_prerequisite_script(self, script_command: str, test_id: int, image_name: str = "", cwd: str = None) -> Tuple[bool, str]:
        """Run a prerequisite script before testing the docker container.

        Supports {image_name} placeholder in script_command, which is replaced
        with the built Docker image name before execution.
        """
        if not script_command:
            return True, "No prerequisite script specified"

        # Substitute {image_name} placeholder
        if image_name:
            script_command = script_command.replace("{image_name}", image_name)

        # Validate script command to prevent command injection
        if not validate_script_command(script_command):
            error_msg = f"Invalid script command: {script_command}. Command contains potentially dangerous characters."
            logger.error(f"❌ {error_msg}")
            return False, error_msg

        log_file = self.log_dir / f"prerequisite_{test_id}.log"

        try:
            logger.info(f"🔧 Running prerequisite script: {script_command}")
            logger.info(f"📄 Prerequisite logs will be saved to: {log_file}")

            # Use safer command execution - split command into arguments when possible
            import shlex
            try:
                # Attempt to parse command safely first
                cmd_args = shlex.split(script_command)
                if len(cmd_args) == 1 and cmd_args[0].endswith('.sh'):
                    # Single script file - use direct execution without shell
                    result = subprocess.run(
                        cmd_args,
                        capture_output=True,
                        text=True,
                        timeout=1800,
                        cwd=cwd,
                        shell=False  # Safer execution without shell
                    )
                else:
                    # Complex command - use shell with additional validation
                    # Double-check validation before shell execution
                    if not validate_script_command(script_command):
                        raise ValueError("Command failed security validation")

                    result = subprocess.run(
                        script_command,
                        shell=True,  # Only when necessary and after validation
                        capture_output=True,
                        text=True,
                        timeout=1800,
                        cwd=cwd
                    )
            except ValueError as e:
                if "failed security validation" in str(e):
                    raise e
                # Fall back to shell execution for complex commands (with validation)
                result = subprocess.run(
                    script_command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=1800,
                    cwd=cwd
                )

            # Save prerequisite script logs
            with open(log_file, 'w') as f:
                f.write("=== Prerequisite Script Execution ===\n")
                f.write(f"Command: {script_command}\n")
                f.write(f"Return Code: {result.returncode}\n")
                f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("\n=== STDOUT ===\n")
                f.write(result.stdout)
                f.write("\n=== STDERR ===\n")
                f.write(result.stderr)
                f.write("\n=== END LOG ===\n")

            if result.returncode == 0:
                logger.info(f"✅ Prerequisite script completed successfully")
                if result.stdout:
                    logger.info(f"Prerequisite output: {result.stdout}")
                return True, result.stdout
            else:
                error_msg = f"Prerequisite script failed with return code {result.returncode}"
                logger.error(f"❌ {error_msg}")
                logger.error(f"Prerequisite stderr: {result.stderr}")
                return False, error_msg

        except subprocess.TimeoutExpired:
            error_msg = f"Prerequisite script timed out after 1800 seconds"
            logger.error(f"❌ {error_msg}")

            # Save timeout log
            with open(log_file, 'w') as f:
                f.write("=== Prerequisite Script Execution ===\n")
                f.write(f"Command: {script_command}\n")
                f.write(f"Status: TIMEOUT\n")
                f.write(f"Timeout: 1800 seconds\n")
                f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("\n=== ERROR ===\n")
                f.write(error_msg)
                f.write("\n=== END LOG ===\n")

            return False, error_msg
        except Exception as e:
            error_msg = f"Exception during prerequisite script execution: {str(e)}"
            logger.error(f"❌ {error_msg}")

            # Save exception log
            with open(log_file, 'w') as f:
                f.write("=== Prerequisite Script Execution ===\n")
                f.write(f"Command: {script_command}\n")
                f.write(f"Status: EXCEPTION\n")
                f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("\n=== ERROR ===\n")
                f.write(error_msg)
                f.write("\n=== END LOG ===\n")

            return False, error_msg

    def test_image(self, image_name: str, test_config: Dict, test_id: int, gpus: str = "all") -> Tuple[bool, str, str]:
        """Test the built image by running it and capture logs.

        Args:
            image_name: Name of the Docker image to test
            test_config: Test configuration dictionary
            test_id: Unique test identifier
            gpus: GPU devices to use (default: 'all')
        """
        log_file = self.log_dir / f"test_{test_id}_{image_name.replace(':', '_')}.log"

        # Validate image name to prevent command injection
        if not validate_image_name(image_name):
            error_msg = f"Invalid image name: {image_name}. Image name contains invalid characters."
            logger.error(f"❌ {error_msg}")
            return False, error_msg, ""

        # Validate test configuration to prevent command injection
        config_valid, config_error = validate_test_config(test_config)
        if not config_valid:
            error_msg = f"Invalid test configuration: {config_error}"
            logger.error(f"❌ {error_msg}")
            return False, error_msg, ""

        # Get timeout from test config, default to 10 seconds
        timeout = test_config.get("timeout", 10)

        try:
            # Check if image exists
            result = subprocess.run(
                ["docker", "image", "inspect", image_name],
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                return False, f"Image {image_name} not found", ""

            # Run prerequisite script if specified
            prerequisite_script = test_config.get("prerequisite_script")
            if prerequisite_script:
                config_dir = str(Path(test_config.get("_config_dir", ".")).resolve())
                prerequisite_success, prerequisite_output = self.run_prerequisite_script(
                    prerequisite_script, test_id, image_name=image_name, cwd=config_dir
                )
                if not prerequisite_success:
                    return False, f"Prerequisite script failed: {prerequisite_output}", ""

            # Detect server type to choose test strategy
            server_type = test_config.get("SERVER_TYPE", "serverless")
            is_serverless = server_type == "serverless"

            # Prepare assets if specified
            asset_ids = []
            asset_temp_dir = None
            if "assets" in test_config:
                import tempfile

                assets_config = test_config["assets"]
                logger.info(f"📦 Preparing {len(assets_config)} asset(s)...")

                # Create temp directory for asset info.json files
                asset_temp_dir = Path(tempfile.mkdtemp(prefix=f"test_assets_{test_id}_"))

                for idx, asset_data in enumerate(assets_config):
                    # Use the provided assetId
                    asset_id = asset_data["assetId"]
                    asset_ids.append(asset_id)

                    # Create asset directory structure
                    asset_dir = asset_temp_dir / asset_id
                    asset_dir.mkdir(parents=True)

                    # Write asset data to info.json inside the directory
                    info_file = asset_dir / "info.json"
                    with info_file.open("w") as f:
                        json.dump(asset_data, f, indent=2)

                    logger.info(f"✅ Created asset {idx}: {asset_id} ({asset_data['path']})")

            # Run the container with test configuration
            # Use host network for serverless; use port mapping for non-serverless to avoid port conflicts
            # Use --ipc=host to share host's /dev/shm with the container;
            # Docker's default 64MB /dev/shm is insufficient for PyTorch/TensorRT-LLM
            # multiprocessing which shares tensors via shared memory, causing SIGBUS
            # (Bus error: nonexistent physical address) when /dev/shm is exhausted.
            cmd = ["docker", "run", "--ipc=host", "--gpus", gpus]

            # Add environment variables if specified
            if "env" in test_config:
                for key, value in test_config["env"].items():
                    # Double-check validation to prevent injection
                    if not validate_env_var_name(key) or not validate_env_var_value(str(value)):
                        error_msg = f"Invalid environment variable: {key}={value}"
                        logger.error(f"❌ {error_msg}")
                        return False, error_msg, ""
                    cmd.extend(["-e", f"{key}={value}"])

            # Get the test config directory for resolving relative paths
            config_dir = Path(test_config.get("_config_dir", ".")).resolve()

            # Automatically add ERROR_EXPORT_PATH for error file collection
            # Use unique error file names for different test cases
            error_export_dir = Path("/tmp/error")
            error_export_dir.mkdir(parents=True, exist_ok=True)
            error_filename = f"inference_errors_{test_id}_{int(time.time())}.json"
            error_export_file = error_export_dir / error_filename
            container_error_path = f"/tmp/error/{error_filename}"
            cmd.extend(["-e", f"ERROR_EXPORT_PATH={container_error_path}"])
            cmd.extend(["-v", f"{error_export_dir}:{error_export_dir}"])
            logger.info(f"📄 Error export: {error_export_file} -> {container_error_path}")

            # Add volume mounts if specified
            if "volumes" in test_config:
                logger.info(f"📁 Processing volumes: {test_config['volumes']}")

                for host_path, container_path in test_config["volumes"].items():
                    # Double-check validation to prevent injection
                    if not validate_volume_path(host_path) or not validate_volume_path(container_path):
                        error_msg = f"Invalid volume path: {host_path}:{container_path}"
                        logger.error(f"❌ {error_msg}")
                        return False, error_msg, ""

                    # Resolve host path to absolute path
                    # Priority: 1) Expand ~ for home directory, 2) Resolve relative to config dir, 3) Use as-is if absolute
                    expanded_path = os.path.expanduser(host_path)

                    if not os.path.isabs(expanded_path):
                        # Relative path - resolve relative to test config directory
                        abs_host_path = str((config_dir / expanded_path).resolve())
                        logger.info(f"📁 Resolved relative path: {host_path} -> {abs_host_path} (relative to config dir)")
                    else:
                        # Already absolute path
                        abs_host_path = expanded_path
                        logger.info(f"📁 Using absolute path: {host_path} -> {abs_host_path}")

                    # Verify the resolved path exists (optional warning, not error)
                    if not os.path.exists(abs_host_path):
                        logger.warning(f"⚠️  Volume path does not exist yet: {abs_host_path}")
                        logger.warning(f"⚠️  This is OK if the path will be created by model download or other setup")

                    volume_arg = f"{abs_host_path}:{container_path}"
                    cmd.extend(["-v", volume_arg])
                    logger.info(f"📁 Adding volume mount: {volume_arg}")
            else:
                logger.info("📁 No additional volumes specified in test config")

            # Give the container a name for logging/cleanup purposes
            expected_error = test_config.get("expected_error")
            container_name = f"test-{test_id}-{int(time.time())}"
            cmd.extend(["--name", container_name])
            logger.info(f"🏷️  Container name: {container_name}")

            # For FastAPI (or any non-serverless), run detached on host network (DinD friendly)
            if not is_serverless:
                cmd.insert(2, "--network=host")
                cmd.insert(2, "-d")  # run detached
            else:
                cmd.insert(2, "--network=host")

            # For serverless non-negative tests, save inference results for validation
            result_export_file = None
            if is_serverless and not expected_error:
                result_export_dir = Path("/tmp/result")
                result_export_dir.mkdir(parents=True, exist_ok=True)
                result_filename = f"result_{test_id}_{int(time.time())}.json"
                result_export_file = result_export_dir / result_filename
                container_result_path = f"/tmp/result/{result_filename}"
                cmd.extend(["-v", f"{result_export_dir}:{result_export_dir}"])
                logger.info(f"📄 Result export: {result_export_file} -> {container_result_path}")

            cmd.append(image_name)

            # Add command arguments if specified (serverless only)
            if server_type == "serverless" and "cmd" in test_config:
                cmd_args = test_config["cmd"]

                # Double-check validation for each command argument
                for arg in cmd_args:
                    if not validate_docker_arg(str(arg)):
                        error_msg = f"Invalid command argument: {arg}"
                        logger.error(f"❌ {error_msg}")
                        return False, error_msg, ""
                cmd.extend(cmd_args)

            # Append -s flag for serverless non-negative tests
            if result_export_file:
                cmd.extend(["-s", container_result_path])

            logger.info(f"Testing image: {image_name}")
            logger.info(f"Command: {' '.join(cmd)}")
            logger.info(f"Timeout: {timeout} seconds")
            logger.info(f"Logs will be saved to: {log_file}")

            # Log the complete command for debugging
            logger.info("🔍 Complete docker run command:")
            logger.info(f"   {' '.join(cmd)}")

            if is_serverless:
                # For serverless with assets, we need to start detached, copy assets, then wait
                if asset_temp_dir:
                    # Insert -d flag to run detached
                    cmd.insert(2, "-d")
                    logger.info("🔍 Starting container in detached mode to copy assets...")
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=30
                    )
                    if result.returncode != 0:
                        logger.error(f"❌ Failed to start container: {result.stderr}")
                        return False, f"Container start failed: {result.stderr}", ""

                    # Container ID is in stdout
                    container_id = result.stdout.strip()
                    logger.info(f"✅ Container started: {container_id[:12]}")

                    # Create /tmp/assets directory in container
                    logger.info("📁 Creating /tmp/assets directory in container...")
                    mkdir_result = subprocess.run(
                        ["docker", "exec", container_name, "mkdir", "-p", "/tmp/assets"],
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                    if mkdir_result.returncode != 0:
                        logger.error(f"❌ Failed to create /tmp/assets: {mkdir_result.stderr}")
                        subprocess.run(["docker", "stop", container_name], capture_output=True, timeout=10)
                        subprocess.run(["docker", "rm", container_name], capture_output=True, timeout=10)
                        return False, f"Failed to create /tmp/assets: {mkdir_result.stderr}", ""

                    # Copy assets into container
                    for idx, asset_id in enumerate(asset_ids):
                        src_path = asset_temp_dir / asset_id
                        dest_path = f"{container_name}:/tmp/assets/{asset_id}"

                        logger.info(f"📦 Copying asset {idx} to container: {asset_id}")
                        copy_result = subprocess.run(
                            ["docker", "cp", str(src_path), dest_path],
                            capture_output=True,
                            text=True,
                            timeout=30
                        )
                        if copy_result.returncode != 0:
                            logger.error(f"❌ Failed to copy asset: {copy_result.stderr}")
                            subprocess.run(["docker", "stop", container_name], capture_output=True, timeout=10)
                            subprocess.run(["docker", "rm", container_name], capture_output=True, timeout=10)
                            return False, f"Asset copy failed: {copy_result.stderr}", ""
                        logger.info(f"✅ Asset {idx} copied successfully")

                    # Now wait for container to complete
                    logger.info("⏳ Waiting for container to complete...")
                    wait_result = subprocess.run(
                        ["docker", "wait", container_name],
                        capture_output=True,
                        text=True,
                        timeout=timeout
                    )

                    # Get logs
                    logs_result = subprocess.run(
                        ["docker", "logs", container_name],
                        capture_output=True,
                        text=True,
                        timeout=30
                    )
                    stdout_output = logs_result.stdout
                    stderr_output = logs_result.stderr

                    # Clean up the temp asset directory
                    import shutil as shutil_module
                    shutil_module.rmtree(asset_temp_dir, ignore_errors=True)
                    logger.info("🧹 Cleaned up temporary asset directory")
                else:
                    # Run container normally and capture output
                    logger.info("🔍 Running container and capturing output...")
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=timeout
                    )
                    stdout_output = result.stdout
                    stderr_output = result.stderr

                # Clean up the container
                subprocess.run(["docker", "rm", container_name], capture_output=True, text=True, timeout=10)

                # Check if error export file was created
                error_copy_failed = False
                error_copy_msg = ""
                if error_export_file.exists():
                    file_size = error_export_file.stat().st_size
                    logger.info(f"📄 Found error export file: {error_export_file} (size: {file_size} bytes)")
                else:
                    error_copy_failed = True
                    error_copy_msg = f"Error export file not found: {error_export_file}"
                    logger.warning(f"⚠️  {error_copy_msg}")
            else:
                # FastAPI-like server flow: start container detached, poll readiness, run client, then stop and collect logs
                logger.info("🚀 Starting server container in detached mode")
                start_proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True
                )
                if start_proc.returncode != 0:
                    error_msg = f"Failed to start server container: {start_proc.stderr}"
                    logger.error(f"❌ {error_msg}")
                    return False, error_msg, ""
                container_id = start_proc.stdout.strip()
                logger.info(f"🆔 Container ID: {container_id}")

                # Copy assets into container if specified
                if asset_temp_dir:
                    # Create /tmp/assets directory in container
                    logger.info("📁 Creating /tmp/assets directory in container...")
                    mkdir_result = subprocess.run(
                        ["docker", "exec", container_name, "mkdir", "-p", "/tmp/assets"],
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                    if mkdir_result.returncode != 0:
                        logger.error(f"❌ Failed to create /tmp/assets: {mkdir_result.stderr}")
                        subprocess.run(["docker", "stop", container_name], capture_output=True, timeout=10)
                        subprocess.run(["docker", "rm", container_name], capture_output=True, timeout=10)
                        return False, f"Failed to create /tmp/assets: {mkdir_result.stderr}", ""

                    for idx, asset_id in enumerate(asset_ids):
                        src_path = asset_temp_dir / asset_id
                        dest_path = f"{container_name}:/tmp/assets/{asset_id}"

                        logger.info(f"📦 Copying asset {idx} to container: {asset_id}")
                        copy_result = subprocess.run(
                            ["docker", "cp", str(src_path), dest_path],
                            capture_output=True,
                            text=True,
                            timeout=30
                        )
                        if copy_result.returncode != 0:
                            logger.error(f"❌ Failed to copy asset: {copy_result.stderr}")
                            subprocess.run(["docker", "stop", container_name], capture_output=True, timeout=10)
                            subprocess.run(["docker", "rm", container_name], capture_output=True, timeout=10)
                            return False, f"Asset copy failed: {copy_result.stderr}", ""
                        logger.info(f"✅ Asset {idx} copied successfully")

                    # Clean up the temp asset directory
                    import shutil as shutil_module
                    shutil_module.rmtree(asset_temp_dir, ignore_errors=True)
                    logger.info("🧹 Cleaned up temporary asset directory")

                # Readiness probe
                ready = False
                http_error_status = None  # Track if we got an HTTP error
                ready_deadline = time.time() + max(1, timeout)
                service_host = self.get_service_host()
                # Get HTTP port from test config env, default to 8000
                http_port = test_config.get("env", {}).get("HTTP_PORT", "8000")
                health_url = f"http://{service_host}:{http_port}/v1/health/ready"
                logger.info(f"🔎 Probing readiness: {health_url}")
                while time.time() < ready_deadline:
                    try:
                        with urllib.request.urlopen(health_url, timeout=2) as resp:
                            if resp.status == 200:
                                ready = True
                                break
                            else:
                                logger.info(f"Health probe returned non-200 status: {resp.status}")
                    except urllib.error.HTTPError as e:
                        http_error_status = e.code
                        logger.error(f"❌ Health probe HTTPError: status={e.code} - server returned error, stopping retry")
                        break  # No retry on HTTP errors - server is responding but not healthy
                    except urllib.error.URLError as e:
                        if isinstance(e.reason, socket.timeout):
                            logger.warning("Health probe request timed out")
                        else:
                            logger.info(f"Health probe URLError: {e.reason}, retrying...")
                    except (TimeoutError, socket.timeout):
                        logger.warning("Health probe request timed out, retrying...")
                    time.sleep(2)

                client_stdout = ""
                client_stderr = ""
                client_rc = 1

                if not ready:
                    if http_error_status:
                        logger.error(f"❌ Server health check failed with HTTP error {http_error_status}")
                    else:
                        logger.error("❌ Server did not become ready within timeout")
                    # Collect logs to see what went wrong during initialization
                    logger.info("📋 Collecting container logs for failed readiness...")
                    logs_proc = subprocess.run(["docker", "logs", container_id], capture_output=True, text=True)
                    if logs_proc.stdout:
                        logger.info("🔍 Container initialization logs:")
                        for line in logs_proc.stdout.split('\n')[-20:]:  # Show last 20 lines
                            if line.strip():
                                logger.info(f"   {line}")
                    if logs_proc.stderr:
                        logger.error("🔍 Container error logs:")
                        for line in logs_proc.stderr.split('\n')[-10:]:  # Show last 10 error lines
                            if line.strip():
                                logger.error(f"   {line}")
                else:
                    # Check if auto_validation is specified - run directly on host
                    auto_validation_path = test_config.get("auto_validation")
                    if auto_validation_path:
                        logger.info("✅ Server is ready. Running validation script on host...")

                        try:
                            # Resolve validation folder path relative to config directory
                            config_dir = Path(test_config.get("_config_dir", ".")).resolve()
                            validation_folder = (config_dir / auto_validation_path).resolve()

                            # Fixed script name is test_runner.py (generated during build via -v flag)
                            # It's located in the validation folder subdirectories (e.g., gdino/.tmp/test_runner.py)
                            # Determine which subdirectory based on TAO_MODEL_NAME
                            tao_model_name = test_config.get("env", {}).get("TAO_MODEL_NAME", "")
                            if not tao_model_name:
                                logger.error("❌ TAO_MODEL_NAME not specified in test_config.env")
                                client_rc = 1
                                client_stdout = ""
                                client_stderr = "TAO_MODEL_NAME not specified"
                            else:
                                validation_script_path = validation_folder / ".tmp" / "test_runner.py"

                                if not validation_script_path.exists():
                                    logger.error(f"❌ Validation script not found: {validation_script_path}")
                                    client_rc = 1
                                    client_stdout = ""
                                    client_stderr = f"Validation script not found: {validation_script_path}"
                                else:
                                    logger.info(f"🔧 Running validation script: {validation_script_path}")

                                    # Set up environment variables for validation script
                                    validation_env = os.environ.copy()
                                    if "env" in test_config:
                                        validation_env.update({k: str(v) for k, v in test_config["env"].items()})

                                    # Add service host for validation script to connect to server
                                    # The validation script expects TEST_HOST environment variable
                                    # Use HTTP_PORT from test config env, default to 8000
                                    validation_env["TEST_HOST"] = f"http://{service_host}:{http_port}"

                                    # Run validation script with Python
                                    validation_proc = subprocess.run(
                                        ["python", str(validation_script_path)],
                                        capture_output=True,
                                        text=True,
                                        timeout=max(60, timeout),
                                        cwd=str(validation_script_path.parent),
                                        env=validation_env
                                    )
                                    client_rc = validation_proc.returncode
                                    client_stdout = validation_proc.stdout
                                    client_stderr = validation_proc.stderr

                                    if client_rc == 0:
                                        logger.info("✅ Validation script completed successfully")
                                        if client_stdout:
                                            logger.info(f"Validation output:\n{client_stdout}")
                                    else:
                                        logger.error(f"❌ Validation script failed with return code {client_rc}")
                                        if client_stderr:
                                            logger.error(f"Validation stderr:\n{client_stderr}")
                                        if client_stdout:
                                            logger.error(f"Validation stdout:\n{client_stdout}")
                        except subprocess.TimeoutExpired:
                            logger.error("❌ Validation script timed out")
                            client_rc = 1
                            client_stdout = ""
                            client_stderr = "Validation script timed out"
                        except Exception as e:
                            logger.error(f"❌ Exception running validation script: {e}")
                            client_rc = 1
                            client_stdout = ""
                            client_stderr = str(e)
                    else:
                        logger.info("✅ Server is ready. Launching concurrent curl requests...")
                        # Read NDJSON files and launch curl for each line
                        # Get test_requests from test_config, with a default fallback based on app name
                        test_app_name = test_config.get("TEST_APP_NAME") or None
                        default_test_requests = [{"payload_path": f"{test_app_name}/payloads/payloads.jsonl"}]
                        test_requests_list = test_config.get("test_requests", default_test_requests)

                        # Ensure test_requests is a list
                        if not isinstance(test_requests_list, list):
                            logger.error(f"❌ test_requests must be a list, got: {type(test_requests_list)}")
                            client_rc = 1
                            client_stdout = ""
                            client_stderr = "test_requests must be a list"
                        else:
                            # Check if any request has scheduled_time (for sequential execution)
                            has_scheduled = any("scheduled_time" in item for item in test_requests_list)

                            # Resolve payloads_path relative to config directory
                            config_dir = Path(test_config.get("_config_dir", ".")).resolve()

                            # Get endpoint from test_config, default to /v1/inference
                            endpoint = test_config.get("endpoint", "/v1/inference")
                            logger.info(f"🎯 Using endpoint: {endpoint}")

                            if has_scheduled:
                                # Sequential execution with timing
                                logger.info("⏰ Using sequential execution with scheduled times")

                                # Sort requests by scheduled_time
                                sorted_requests = sorted(test_requests_list, key=lambda x: x.get("scheduled_time", 0))

                                start_time = time.time()
                                all_succeeded = True
                                async_processes = []  # Track async processes for later status check

                                for request_item in sorted_requests:
                                    scheduled_time = request_item.get("scheduled_time", 0)
                                    elapsed = time.time() - start_time

                                    # Wait until scheduled time
                                    if scheduled_time > elapsed:
                                        wait_time = scheduled_time - elapsed
                                        logger.info(f"⏳ Waiting {wait_time:.1f}s until scheduled time {scheduled_time}s...")
                                        time.sleep(wait_time)

                                    # Extract request config
                                    http_method = request_item.get("method", "POST").upper()
                                    http_headers = request_item.get("headers", {"Content-Type": "application/json"})
                                    request_endpoint = request_item.get("endpoint", endpoint)

                                    # Get payload
                                    payload_data = None
                                    if "payload_path" in request_item:
                                        payload_path = (config_dir / request_item["payload_path"]).resolve()
                                        if payload_path.exists():
                                            with payload_path.open("r") as f:
                                                first_line = f.readline().strip()
                                                if first_line:
                                                    payload_data = first_line
                                        else:
                                            logger.error(f"❌ Payload file not found: {payload_path}")
                                            all_succeeded = False
                                            continue

                                    # Build curl command
                                    url = f"http://{service_host}:{http_port}{request_endpoint}"
                                    curl_cmd = ["curl", "-sS", "-X", http_method, "-w", "\\n%{http_code}"]

                                    for header_name, header_value in http_headers.items():
                                        curl_cmd.extend(["-H", f"{header_name}: {header_value}"])

                                    if payload_data and http_method in ["POST", "PUT", "PATCH", "DELETE"]:
                                        curl_cmd.extend(["-d", payload_data])

                                    curl_cmd.append(url)

                                    logger.info(f"🌐 [{scheduled_time}s] {http_method} {request_endpoint}")
                                    if payload_data:
                                        logger.info(f"📦 Payload: {payload_data[:200]}{'...' if len(payload_data) > 200 else ''}")

                                    # Check if this is an async request
                                    is_async = request_item.get("async", False)

                                    if is_async:
                                        # Launch async request without waiting for response
                                        logger.info(f"🚀 Launching async request")
                                        try:
                                            proc = subprocess.Popen(
                                                curl_cmd,
                                                stdout=subprocess.PIPE,
                                                stderr=subprocess.PIPE,
                                                text=True
                                            )
                                            async_processes.append((proc, http_method, request_endpoint))
                                            logger.info(f"✅ Async request launched (PID: {proc.pid})")
                                        except Exception as e:
                                            logger.error(f"❌ Failed to launch async request: {e}")
                                            all_succeeded = False
                                    else:
                                        # Execute synchronous request
                                        try:
                                            result = subprocess.run(curl_cmd, capture_output=True, text=True, timeout=30)
                                            response_output = result.stdout.strip()

                                            # Split response body and status code
                                            lines = response_output.split('\n')
                                            status_code = int(lines[-1]) if lines and lines[-1].isdigit() else 0
                                            response_body = '\n'.join(lines[:-1]) if len(lines) > 1 else ""

                                            logger.info(f"📡 Status: {status_code}")
                                            if response_body:
                                                logger.info(f"✅ Response: {response_body[:300]}{'...' if len(response_body) > 300 else ''}")

                                            if status_code < 200 or status_code >= 300:
                                                logger.error(f"❌ Request failed with status {status_code}")
                                                all_succeeded = False
                                        except subprocess.TimeoutExpired:
                                            logger.error(f"❌ Request timed out")
                                            all_succeeded = False
                                        except Exception as e:
                                            logger.error(f"❌ Request failed: {e}")
                                            all_succeeded = False

                                # Wait for async processes and check their results
                                if async_processes:
                                    logger.info(f"\n⏳ Waiting for {len(async_processes)} async request(s) to complete...")
                                    for proc, method, endpoint in async_processes:
                                        try:
                                            stdout, stderr = proc.communicate(timeout=30)
                                            response_output = stdout.strip()

                                            # Split response body and status code
                                            lines = response_output.split('\n')
                                            status_code = int(lines[-1]) if lines and lines[-1].isdigit() else 0
                                            response_body = '\n'.join(lines[:-1]) if len(lines) > 1 else ""

                                            logger.info(f"📡 Async {method} {endpoint} - Status: {status_code}")
                                            if response_body:
                                                logger.info(f"   Response: {response_body[:200]}{'...' if len(response_body) > 200 else ''}")

                                            if status_code < 200 or status_code >= 300:
                                                logger.error(f"❌ Async request failed with status {status_code}")
                                                all_succeeded = False
                                        except subprocess.TimeoutExpired:
                                            logger.error(f"❌ Async {method} {endpoint} timed out")
                                            proc.kill()
                                            all_succeeded = False
                                        except Exception as e:
                                            logger.error(f"❌ Async {method} {endpoint} failed: {e}")
                                            all_succeeded = False

                                # Set final result
                                if all_succeeded:
                                    logger.info("\n✅ All scheduled requests completed successfully")
                                    client_rc = 0
                                    client_stdout = "All requests succeeded"
                                    client_stderr = ""
                                else:
                                    logger.error("\n❌ Some requests failed")
                                    client_rc = 1
                                    client_stdout = ""
                                    client_stderr = "One or more requests failed"
                            else:
                                # Parallel execution (existing logic)
                                logger.info("🚀 Using parallel execution")

                                # Collect all payload lines from all files with their configs
                                # Each entry: (payload_line, method, headers)
                                all_payloads = []
                                for request_item in test_requests_list:
                                    # Parse test request (must be dict)
                                    if not isinstance(request_item, dict):
                                        logger.error(f"❌ Invalid test_requests item (must be dict): {request_item}")
                                        continue

                                    # Extract config with defaults
                                    http_method = request_item.get("method", "POST").upper()
                                    http_headers = request_item.get("headers", {"Content-Type": "application/json"})

                                    # Check if payload_path is provided
                                    if "payload_path" not in request_item:
                                        # No payload_path - single request with no payload (useful for GET)
                                        logger.info(f"📄 No payload file specified")
                                        logger.info(f"   Method: {http_method}, Headers: {http_headers}")
                                        all_payloads.append(("", http_method, http_headers))
                                    else:
                                        # Path provided - load payloads from file
                                        payload_path_str = request_item["payload_path"]
                                        payload_path = (config_dir / payload_path_str).resolve()
                                        logger.info(f"📄 Resolved payload path: {payload_path_str} -> {payload_path}")
                                        logger.info(f"   Method: {http_method}, Headers: {http_headers}")

                                        if not payload_path.exists():
                                            logger.error(f"❌ Payload file not found: {payload_path}")
                                        else:
                                            with payload_path.open("r") as f:
                                                payload_lines = [line.strip() for line in f if line.strip()]
                                                # Associate each payload with its method and headers
                                                for line in payload_lines:
                                                    all_payloads.append((line, http_method, http_headers))
                                                logger.info(f"📋 Loaded {len(payload_lines)} payloads from {payload_path.name}")

                                if not all_payloads:
                                    logger.error(f"❌ No payloads found in any file")
                                    client_rc = 1
                                    client_stdout = ""
                                    client_stderr = "No payloads found"
                                else:
                                    logger.info(f"📋 Total payloads to process: {len(all_payloads)}")

                                    procs: List[subprocess.Popen] = []
                                    for payload_line, method, headers in all_payloads:
                                        # Check if this is a multipart/form-data upload
                                        content_type = headers.get("Content-Type", "").lower()
                                        is_multipart = "multipart/form-data" in content_type

                                        # Build curl command with method and headers from payload config
                                        curl_cmd = ["curl", "-sS", "-X", method, "-w", "%{http_code}"]

                                        if is_multipart and method in ["POST", "PUT", "PATCH"] and payload_line:
                                            # Multipart upload: parse JSON to get file field and use -F
                                            try:
                                                payload_data = json.loads(payload_line)

                                                # Add non-Content-Type headers (curl -F sets Content-Type automatically)
                                                for header_name, header_value in headers.items():
                                                    if header_name.lower() != "content-type":
                                                        curl_cmd.extend(["-H", f"{header_name}: {header_value}"])

                                                # Add form fields with -F
                                                for field_name, field_value in payload_data.items():
                                                    if isinstance(field_value, str):
                                                        # Try to resolve as a file path (relative to config dir)
                                                        file_path = Path(field_value)
                                                        if not file_path.is_absolute():
                                                            # Try relative to config dir
                                                            resolved_path = (config_dir / field_value).resolve()
                                                        else:
                                                            # Already absolute
                                                            resolved_path = file_path.resolve()

                                                        # Check if the resolved path exists and is a file
                                                        if resolved_path.exists() and resolved_path.is_file():
                                                            # Use @ prefix for file upload
                                                            curl_cmd.extend(["-F", f"{field_name}=@{resolved_path}"])
                                                            logger.info(f"   📎 Uploading file: {field_name}={resolved_path}")
                                                        else:
                                                            # Not a valid file path, send as string value
                                                            curl_cmd.extend(["-F", f"{field_name}={field_value}"])
                                                            if field_value:
                                                                logger.debug(f"   📝 Form field: {field_name}={field_value}")
                                                    else:
                                                        # Non-string value (number, bool, etc.)
                                                        curl_cmd.extend(["-F", f"{field_name}={field_value}"])
                                            except json.JSONDecodeError as e:
                                                logger.error(f"❌ Failed to parse payload as JSON for multipart upload: {e}")
                                                logger.error(f"   Payload: {payload_line}")
                                                continue
                                        else:
                                            # Regular JSON request
                                            # Add headers
                                            for header_name, header_value in headers.items():
                                                curl_cmd.extend(["-H", f"{header_name}: {header_value}"])

                                            # Add data for methods that support body (only if payload is not empty)
                                            if method in ["POST", "PUT", "PATCH"] and payload_line:
                                                curl_cmd.extend(["--data", payload_line])

                                        # Add URL
                                        curl_cmd.append(f"http://{service_host}:{http_port}{endpoint}")

                                        procs.append(subprocess.Popen(
                                            curl_cmd,
                                            stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE,
                                            text=True
                                        ))

                                    outs: List[str] = []
                                    errs: List[str] = []
                                    client_rc = 0
                                    http_status_errors = []
                                    for i, p in enumerate(procs):
                                        out, err = p.communicate(timeout=max(5, timeout))
                                        outs.append(out or "")
                                        errs.append(err or "")
                                        if p.returncode != 0:
                                            client_rc = p.returncode
                                        else:
                                            # For non-serverless, check HTTP status is 200
                                            # curl -w "%{http_code}" appends status to stdout
                                            if out and len(out) >= 3:
                                                # Extract last 3 chars as HTTP status code
                                                http_status = out[-3:]
                                                if http_status != "200":
                                                    http_status_errors.append(
                                                        f"Request {i+1}: HTTP {http_status}"
                                                    )
                                                    client_rc = 1  # Mark as failed

                                    client_stdout = "\n".join(outs)
                                    client_stderr = "\n".join(errs)

                                    # Log HTTP status errors for non-serverless
                                    if http_status_errors:
                                        error_msg = (
                                            f"Non-serverless server returned non-200 status "
                                            f"codes: {'; '.join(http_status_errors)}"
                                        )
                                        logger.error("❌ %s", error_msg)
                                        client_stderr += f"\n{error_msg}"

                # Always attempt to stop and collect logs
                logger.info("🛑 Stopping server container")

                # Collect logs BEFORE stopping container (while it's still running)
                logger.info("📋 Collecting container logs...")
                logs_proc = subprocess.run(["docker", "logs", container_id], capture_output=True, text=True)
                server_logs = logs_proc.stdout
                if logs_proc.stderr:
                    server_logs += "\n=== STDERR ===\n" + logs_proc.stderr

                # Check if error export file was created
                error_copy_failed = False
                error_copy_msg = ""
                if error_export_file.exists():
                    file_size = error_export_file.stat().st_size
                    logger.info(f"📄 Found error export file: {error_export_file} (size: {file_size} bytes)")
                else:
                    error_copy_failed = True
                    error_copy_msg = f"Error export file not found: {error_export_file}"
                    logger.warning(f"⚠️  {error_copy_msg}")

                # Now stop and remove container
                subprocess.run(["docker", "stop", container_id], capture_output=True, text=True)
                subprocess.run(["docker", "rm", container_id], capture_output=True, text=True)

                # Prepare a Result-like object
                class Result:
                    def __init__(self, returncode, stdout, stderr):
                        self.returncode = returncode
                        self.stdout = stdout
                        self.stderr = stderr

                combined_stdout = "".join([
                    "=== SERVER LOGS ===\n", server_logs or "",
                    "\n=== CLIENT STDOUT ===\n", client_stdout or "",
                ])
                combined_stderr = client_stderr or ""
                # For non-serverless flows, we care about HTTP responses and validation status,
                # not the container's own exit code. Track client failure separately and
                # keep Result.returncode neutral (0) so final status can branch by server type.
                client_failed = not (ready and client_rc == 0)
                result = Result(0, combined_stdout, combined_stderr)

            # Save logs to file
            with open(log_file, 'w') as f:
                f.write("=== Test Configuration ===\n")
                f.write(f"Image: {image_name}\n")
                f.write(f"Command: {' '.join(cmd)}\n")
                f.write(f"Return Code: {result.returncode}\n")
                f.write(f"Timeout: {timeout} seconds\n")
                f.write("Error Detection: Enabled (via JSON export)\n")
                f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("\n=== STDOUT ===\n")
                f.write(result.stdout)
                f.write("\n=== STDERR ===\n")
                f.write(result.stderr)
                f.write("\n=== END LOG ===\n")

            # Check for expected_error (negative test) using exported error JSON
            expected_error = test_config.get("expected_error")
            if expected_error:
                # This is a negative test - fail immediately if error file not found
                if error_copy_failed or error_export_file is None:
                    error_msg = f"Negative test FAILED: Could not find error export file: {error_copy_msg}"
                    logger.error(f"❌ {error_msg}")
                    logger.info(f"📄 Logs saved to: {log_file}")
                    return False, error_msg, str(log_file)

                # Try to read the exported error file
                matched_error = None  # Will hold the error that matches expected_error exactly

                if error_export_file.exists():
                    try:
                        with open(error_export_file, 'r') as f:
                            error_data = json.load(f)

                        stats = error_data.get("stats", {})
                        total_errors = stats.get("total_errors", 0)
                        errors = error_data.get("errors", [])
                        collected_codes = [err.get("error_code") for err in errors]

                        # Search for EXACT match of expected error code
                        for err in errors:
                            if err.get("error_code") == expected_error:
                                matched_error = err
                                break

                        if matched_error is not None:
                            # EXACT match found - test passes
                            logger.info(f"✅ Negative test PASSED: Error code '{expected_error}' matches exactly")
                            logger.info(f"   Error message: {matched_error.get('message', 'N/A')[:200]}")
                            logger.info(f"   Component: {matched_error.get('component', 'N/A')}")
                            logger.info(f"   Operation: {matched_error.get('operation', 'N/A')}")
                            logger.info(f"   Total errors collected: {total_errors}")
                            logger.info(f"📄 Logs saved to: {log_file}")
                            return True, result.stdout, str(log_file)
                        else:
                            # No exact match - test fails
                            error_msg = f"Negative test FAILED: Expected error '{expected_error}' not found"
                            logger.error(f"❌ {error_msg}")
                            logger.error(f"   Collected error codes: {collected_codes}")
                            logger.error(f"   Total errors: {total_errors}")
                            logger.info(f"📄 Logs saved to: {log_file}")
                            return False, error_msg, str(log_file)
                    except json.JSONDecodeError as e:
                        error_msg = f"Negative test FAILED: Failed to parse error export file: {e}"
                        logger.error(f"❌ {error_msg}")
                        logger.info(f"📄 Logs saved to: {log_file}")
                        return False, error_msg, str(log_file)
                    except Exception as e:
                        error_msg = f"Negative test FAILED: Failed to read error export file: {e}"
                        logger.error(f"❌ {error_msg}")
                        logger.info(f"📄 Logs saved to: {log_file}")
                        return False, error_msg, str(log_file)
                else:
                    # Error export file doesn't exist
                    error_msg = f"Negative test FAILED: Error export file not found: {error_export_file}"
                    logger.error(f"❌ {error_msg}")
                    logger.info(f"📄 Logs saved to: {log_file}")
                    return False, error_msg, str(log_file)

            # For serverless non-negative tests, check that the result file exists and is non-empty
            if result_export_file:
                if not result_export_file.exists():
                    error_msg = f"Result file not found: {result_export_file}"
                    logger.error(f"❌ {error_msg}")
                    logger.info(f"📄 Logs saved to: {log_file}")
                    return False, error_msg, str(log_file)
                file_size = result_export_file.stat().st_size
                if file_size == 0:
                    error_msg = f"Result file is empty: {result_export_file}"
                    logger.error(f"❌ {error_msg}")
                    logger.info(f"📄 Logs saved to: {log_file}")
                    return False, error_msg, str(log_file)
                logger.info(f"📄 Result file OK: {result_export_file} ({file_size} bytes)")

            # Check for errors using the exported error JSON file (if ERROR_EXPORT_PATH was configured)
            has_errors = False
            error_details = []

            if error_export_file and error_export_file.exists():
                try:
                    with open(error_export_file, 'r') as f:
                        error_data = json.load(f)

                    stats = error_data.get("stats", {})
                    total_errors = stats.get("total_errors", 0)
                    errors = error_data.get("errors", [])

                    if total_errors > 0:
                        has_errors = True
                        for err in errors:
                            error_details.append({
                                "code": err.get("error_code", "UNKNOWN"),
                                "message": err.get("message", "N/A"),
                                "severity": err.get("severity", "unknown"),
                                "component": err.get("component", "N/A")
                            })
                except json.JSONDecodeError as e:
                    logger.warning(f"⚠️  Failed to parse error export file: {e}")
                except Exception as e:
                    logger.warning(f"⚠️  Failed to read error export file: {e}")

            # Determine overall success:
            # - For serverless flows, rely on container return code and error exports.
            # - For non-serverless HTTP server flows, rely on HTTP/client status (client_failed)
            #   and error exports; ignore the container's own exit code.
            if is_serverless:
                rc_failed = result.returncode < 0
            else:
                # client_failed is only set in non-serverless branch; default to False otherwise
                rc_failed = 'client_failed' in locals() and client_failed

            if not rc_failed and not has_errors:
                logger.info(f"✅ Successfully tested image: {image_name}")
                logger.info(f"📄 Logs saved to: {log_file}")
                return True, result.stdout, str(log_file)
            elif has_errors:
                error_msg = f"Test failed: {len(error_details)} error(s) detected in image: {image_name}"
                logger.error(f"❌ {error_msg}")
                logger.error("Errors found:")
                for err in error_details[:10]:  # Show first 10 errors
                    logger.error(f"  [{err['severity'].upper()}] {err['code']}: {err['message'][:100]}")
                if len(error_details) > 10:
                    logger.error(f"  ... and {len(error_details) - 10} more errors")
                logger.info(f"📄 Logs saved to: {log_file}")
                logger.info(f"📄 Error export: {error_export_file}")
                return False, error_msg, str(log_file)
            else:
                if is_serverless:
                    logger.error(
                        f"❌ Test failed with negative return code for image: {image_name}"
                    )
                    logger.error(f"Return code: {result.returncode}")
                    failure_reason = f"Test failed with negative return code {result.returncode}"
                else:
                    logger.error(
                        f"❌ Test failed due to client/HTTP errors for image: {image_name}"
                    )
                    failure_reason = "Test failed due to client/HTTP errors"
                logger.info(f"📄 Logs saved to: {log_file}")
                return False, failure_reason, str(log_file)

        except subprocess.TimeoutExpired as timeout_ex:
            error_msg = f"Test timed out after {timeout} seconds for image: {image_name}"
            logger.error(f"❌ {error_msg}")

            # Try to collect container logs if this was a detached container
            container_logs = ""
            if server_type != "serverless":
                # For non-serverless, we have a detached container that needs cleanup
                try:
                    # Check if container_id was defined (container was started)
                    if 'container_id' in locals():
                        logger.info(f"📋 Collecting logs from timed-out container: {container_id}")
                        logs_proc = subprocess.run(
                            ["docker", "logs", container_id],
                            capture_output=True,
                            text=True,
                            timeout=10
                        )
                        container_logs = logs_proc.stdout
                        if logs_proc.stderr:
                            container_logs += "\n=== STDERR ===\n" + logs_proc.stderr

                        # Stop and remove the container
                        logger.info(f"🛑 Stopping timed-out container: {container_id}")
                        subprocess.run(["docker", "stop", container_id], capture_output=True, text=True, timeout=30)
                        subprocess.run(["docker", "rm", container_id], capture_output=True, text=True, timeout=10)
                except Exception as cleanup_error:
                    logger.warning(f"⚠️  Failed to collect logs or cleanup container: {cleanup_error}")
                    container_logs += f"\n\n[Error during log collection: {cleanup_error}]\n"

            # Save timeout log with container logs if available
            with open(log_file, 'w') as f:
                f.write("=== Test Configuration ===\n")
                f.write(f"Image: {image_name}\n")
                f.write(f"Command: {' '.join(cmd)}\n")
                f.write(f"Status: TIMEOUT\n")
                f.write(f"Timeout: {timeout} seconds\n")
                f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("\n=== ERROR ===\n")
                f.write(error_msg)
                if container_logs:
                    f.write("\n\n=== CONTAINER LOGS ===\n")
                    f.write(container_logs)
                f.write("\n=== END LOG ===\n")

            return False, error_msg, str(log_file)
        except Exception as e:
            error_msg = f"Exception during test: {str(e)}"
            logger.error(f"❌ {error_msg}")

            # Add full traceback for debugging
            import traceback
            full_traceback = traceback.format_exc()
            logger.error(f"Traceback:\n{full_traceback}")

            # Try to collect container logs if this was a detached container
            container_logs = ""
            if server_type != "serverless":
                # For non-serverless, we have a detached container that needs cleanup
                try:
                    # Check if container_id was defined (container was started)
                    if 'container_id' in locals():
                        logger.info(f"📋 Collecting logs from failed container: {container_id}")
                        logs_proc = subprocess.run(
                            ["docker", "logs", container_id],
                            capture_output=True,
                            text=True,
                            timeout=10
                        )
                        container_logs = logs_proc.stdout
                        if logs_proc.stderr:
                            container_logs += "\n=== STDERR ===\n" + logs_proc.stderr

                        # Stop and remove the container
                        logger.info(f"🛑 Stopping failed container: {container_id}")
                        subprocess.run(["docker", "stop", container_id], capture_output=True, text=True, timeout=30)
                        subprocess.run(["docker", "rm", container_id], capture_output=True, text=True, timeout=10)
                except Exception as cleanup_error:
                    logger.warning(f"⚠️  Failed to collect logs or cleanup container: {cleanup_error}")
                    container_logs += f"\n\n[Error during log collection: {cleanup_error}]\n"

            # Save exception log
            with open(log_file, 'w') as f:
                f.write("=== Test Configuration ===\n")
                f.write(f"Image: {image_name}\n")
                f.write(f"Status: EXCEPTION\n")
                f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("\n=== ERROR ===\n")
                f.write(error_msg)
                f.write("\n\n=== TRACEBACK ===\n")
                f.write(full_traceback)
                if container_logs:
                    f.write("\n\n=== CONTAINER LOGS ===\n")
                    f.write(container_logs)
                f.write("\n=== END LOG ===\n")

            return False, error_msg, str(log_file)

    def cleanup_prerequisite_script(self, test_config: Dict, test_id: int) -> bool:
        """Clean up resources created by prerequisite scripts."""
        prerequisite_script = test_config.get("prerequisite_script")
        if not prerequisite_script:
            return True

        try:
            # Check if the script is the RTSP server setup script
            if "setup_rtsp_server.sh" in prerequisite_script:
                logger.info("🧹 Cleaning up RTSP server...")
                # Use secure command execution for cleanup
                cleanup_command = ["./setup_rtsp_server.sh", "--kill"]
                cleanup_result = subprocess.run(
                    cleanup_command,
                    shell=False,  # Safer execution without shell
                    capture_output=True,
                    text=True,
                    timeout=30
                )

                if cleanup_result.returncode == 0:
                    logger.info("✅ RTSP server cleanup completed")
                    return True
                else:
                    logger.warning(f"⚠️  RTSP server cleanup failed: {cleanup_result.stderr}")
                    return False

            # Add more cleanup logic for other prerequisite scripts here
            logger.info("🧹 No specific cleanup needed for prerequisite script")
            return True

        except Exception as e:
            logger.warning(f"⚠️  Exception during prerequisite cleanup: {str(e)}")
            return False

    def cleanup_image(self, image_name: str) -> bool:
        """Remove the test image."""
        try:
            result = subprocess.run(
                ["docker", "rmi", image_name],
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                logger.info(f"🧹 Cleaned up image: {image_name}")
                return True
            else:
                logger.warning(f"⚠️  Failed to cleanup image: {image_name}")
                return False

        except Exception as e:
            logger.warning(f"⚠️  Exception during cleanup: {str(e)}")
            return False

    def run_test_suite(self, test_configs: List[Dict], cleanup: bool = True, gitlab_token: Optional[str] = None, force_full_flow: bool = False, gpus: str = "all") -> Dict:
        """Run a suite of tests with different configurations.

        Args:
            test_configs: List of test configurations
            cleanup: Whether to cleanup Docker images after testing
            gitlab_token: GitLab token for authentication
            force_full_flow: Force full build+test flow even for disabled tests
            gpus: GPU devices to use (default: 'all')
        """
        results = {
            "total_tests": len(test_configs),
            "passed": 0,
            "failed": 0,
            "results": []
        }

        for i, config in enumerate(test_configs, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Running test {i}/{len(test_configs)}")
            logger.info(f"{'='*60}")

            # Generate unique image name
            image_name = f"test-inference_builder-{i}-{int(time.time())}"

            build_args = config.get("build_args", {}).copy()
            test_cfg = config.get("test_config", {}).copy()

            # Add _config_dir to test_cfg so paths can be resolved correctly
            if "_config_dir" in config:
                test_cfg["_config_dir"] = config["_config_dir"]

            # Note: auto_validation is passed directly to build_image via test_config
            # No need to add it to build_args

            # Debug logging for flow decision
            default_enable = test_cfg.get("default_enable", True)
            logger.info(f"🔍 Test flow decision: default_enable={default_enable}, force_full_flow={force_full_flow}")

            # If default_enable is False, skip Docker build/test but run codegen,
            # unless full flow is forced by selection (e.g., --test-case provided)
            if not default_enable and not force_full_flow:
                logger.info("⚙️  Test disabled via default_enable=false. Running code generation only (skipping model download)...")
                codegen_success, codegen_output = self.generate_inference_code(build_args, test_cfg)

                status = "SKIPPED"
                test_result = {
                    "test_id": i,
                    "config": config,
                    "status": status,
                    "build_success": False,
                    "test_success": False,
                    "build_output": codegen_output,
                    "test_output": "",
                    "log_file": "",
                    "image_name": image_name
                }
                results["results"].append(test_result)
                logger.info(f"Test {i} result: {status} (codegen_only)")
                # Do not increment passed/failed counters for skipped tests
                continue

            # Download models if specified in test config (only for full build+test flow)
            models_config = test_cfg.get("models", {})
            if models_config:
                logger.info(f"📦 Checking and downloading models (full build+test flow)...")
                config_dir = Path(config.get("_config_dir", "."))
                download_success, download_msg = self.download_models(models_config, config_dir)
                if not download_success:
                    logger.error(f"❌ Model download failed: {download_msg}")
                    results["failed"] += 1
                    test_result = {
                        "test_id": i,
                        "config": config,
                        "status": "FAILED",
                        "build_success": False,
                        "test_success": False,
                        "build_output": "",
                        "test_output": download_msg,
                        "log_file": "",
                        "image_name": image_name
                    }
                    results["results"].append(test_result)
                    logger.info(f"Test {i} result: FAILED (model download)")
                    continue

            # Build image with resolved dockerfile and base_dir
            dockerfile_to_use = config.get("_resolved_dockerfile")
            base_dir_to_use = config.get("_resolved_base_dir")
            build_success, build_output = self.build_image(build_args, image_name, dockerfile_to_use, base_dir_to_use, test_cfg)

            if build_success:
                # Test image - pass server type and config dir to test_config
                test_config = config.get("test_config", {}).copy()
                if "SERVER_TYPE" in build_args:
                    test_config["SERVER_TYPE"] = build_args["SERVER_TYPE"]
                # Pass config directory for relative path resolution
                if "_config_dir" in config:
                    test_config["_config_dir"] = config["_config_dir"]
                if "TEST_APP_NAME" in build_args:
                    test_config["TEST_APP_NAME"] = build_args["TEST_APP_NAME"]
                test_success, test_output, log_file = self.test_image(
                    image_name,
                    test_config,
                    i,
                    gpus=gpus
                )

                if test_success:
                    results["passed"] += 1
                    status = "PASSED"
                else:
                    results["failed"] += 1
                    status = "FAILED"
            else:
                results["failed"] += 1
                test_success = False
                test_output = ""
                log_file = ""
                status = "FAILED"

            # Cleanup (run cleanup for full-flow tests; skip if codegen-only path was taken)
            if cleanup:
                self.cleanup_image(image_name)
                # Clean up prerequisite scripts
                self.cleanup_prerequisite_script(config.get("test_config", {}), i)

            # Store result
            test_result = {
                "test_id": i,
                "config": config,
                "status": status,
                "build_success": build_success,
                "test_success": test_success,
                "build_output": build_output,
                "test_output": test_output,
                "log_file": log_file,
                "image_name": image_name
            }

            results["results"].append(test_result)

            logger.info(f"Test {i} result: {status}")

        return results

    def generate_report(self, results: Dict, output_file: Optional[str] = None):
        """Generate a test report."""
        report = {
            "summary": {
                "total_tests": results["total_tests"],
                "passed": results["passed"],
                "failed": results["failed"],
                "success_rate": f"{(results['passed'] / results['total_tests'] * 100):.1f}%" if results["total_tests"] > 0 else "0%"
            },
            "results": results["results"]
        }

        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"📄 Report saved to: {output_file}")

        # Print summary
        logger.info(f"\n{'='*60}")
        logger.info("TEST SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"Total tests: {results['total_tests']}")
        logger.info(f"Passed: {results['passed']}")
        logger.info(f"Failed: {results['failed']}")
        logger.info(f"Success rate: {report['summary']['success_rate']}")

        # Print test results with log file locations
        logger.info(f"\n{'='*60}")
        logger.info("TEST RESULTS")
        logger.info(f"{'='*60}")
        for result in results["results"]:
            test_name = result["config"].get("name", f"Test {result['test_id']}")
            status = result.get("status", "UNKNOWN")
            log_file = result.get("log_file", "")

            if status == "PASSED":
                status_icon = "✅"
            elif status == "FAILED":
                status_icon = "❌"
            elif status == "SKIPPED":
                status_icon = "⏭️"
            else:
                status_icon = "❓"

            if log_file:
                logger.info(f"{status_icon} [{result['test_id']}] {test_name}: {log_file}")
            else:
                logger.info(f"{status_icon} [{result['test_id']}] {test_name}")

        return report


def main():
    parser = argparse.ArgumentParser(description="Test Docker builds with different arguments")
    parser.add_argument("--dockerfile", default="Dockerfile", help="Path to Dockerfile")
    parser.add_argument("--base-dir", default=".", help="Base directory for Docker build context")
    parser.add_argument("--config-file", required=True, help="JSON file with test configurations")
    parser.add_argument("--output", help="Output file for test report")
    parser.add_argument("--log-dir", default="logs", help="Directory to save container logs")
    parser.add_argument("--no-cleanup", action="store_true", help="Don't cleanup images after testing")
    parser.add_argument("--gitlab-token", help="GitLab token for authentication")
    parser.add_argument("--test-case", help="Run only the test case with this name (partial match supported). Supplying this forces full flow (build+test) even if disabled.")
    parser.add_argument("--gpus", default="all", help="GPU devices to use for Docker containers (default: 'all'). Examples: 'all', 'device=0', 'device=0,1', '\"device=0,1\"'")

    # Parse arguments with security validation
    try:
        args = parser.parse_args()
    except Exception as e:
        logger.error(f"❌ Argument parsing failed: {str(e)}")
        sys.exit(1)

    # Comprehensive security validation
    validation_errors = []

    # Validate dockerfile path
    if not validate_dockerfile_path(args.dockerfile):
        validation_errors.append(f"Invalid Dockerfile path: {args.dockerfile}")

    # Validate base directory
    if not validate_safe_path(args.base_dir):
        validation_errors.append(f"Invalid base directory path: {args.base_dir}")

    # Validate config file
    if not validate_config_file_path(args.config_file):
        validation_errors.append(f"Invalid config file path: {args.config_file}")

    # Validate output file if provided
    if args.output and not validate_safe_path(args.output):
        validation_errors.append(f"Invalid output file path: {args.output}")

    # Validate log directory
    if not validate_log_directory(args.log_dir):
        validation_errors.append(f"Invalid log directory path: {args.log_dir}")

    # Validate GitLab token if provided
    if args.gitlab_token and not validate_gitlab_token(args.gitlab_token):
        validation_errors.append("Invalid GitLab token format")

    # Exit if any validation errors
    if validation_errors:
        logger.error("❌ Security validation failed:")
        for error in validation_errors:
            logger.error(f"  - {error}")
        sys.exit(1)

    # Additional file existence checks with security
    try:
        # Validate config file first (required)
        if not os.path.isfile(args.config_file):
            logger.error(f"❌ Config file not found: {args.config_file}")
            sys.exit(1)

        # Dockerfile and base-dir are optional - they'll be resolved from config file if not found
        # Only warn if they're explicitly specified but don't exist
        dockerfile_exists = os.path.isfile(args.dockerfile)
        base_dir_exists = os.path.isdir(args.base_dir)

        if not dockerfile_exists:
            logger.info(f"ℹ️  Dockerfile not found at command-line path: {args.dockerfile}")
            logger.info(f"ℹ️  Will use Dockerfile from test config or config directory")

        if not base_dir_exists:
            logger.info(f"ℹ️  Base directory not found at command-line path: {args.base_dir}")
            logger.info(f"ℹ️  Will use base directory from test config or config directory")

        # Validate log directory path (prevent directory traversal)
        log_dir_path = Path(args.log_dir).resolve()
        current_dir = Path.cwd().resolve()
        try:
            log_dir_path.relative_to(current_dir)
        except ValueError:
            logger.error(f"❌ Log directory path is outside current directory: {args.log_dir}")
            sys.exit(1)

    except Exception as e:
        logger.error(f"❌ File validation failed: {str(e)}")
        sys.exit(1)

    # Initialize tester with validated arguments
    try:
        tester = DockerBuildTester(args.dockerfile, args.base_dir, args.log_dir)
    except Exception as e:
        logger.error(f"❌ Failed to initialize DockerBuildTester: {str(e)}")
        sys.exit(1)

    # Load test configurations from file with security validation
    try:
        with open(args.config_file, 'r') as f:
            test_configs = json.load(f)

        # Validate JSON structure
        if not isinstance(test_configs, list):
            logger.error("❌ Config file must contain a list of test configurations")
            sys.exit(1)

        # Get the directory containing the config file for resolving relative paths
        config_dir = Path(args.config_file).parent.resolve()
        logger.info(f"📁 Config file directory: {config_dir}")

        # Validate each test configuration for security
        for i, config in enumerate(test_configs):
            if not isinstance(config, dict):
                logger.error(f"❌ Test configuration {i+1} must be a dictionary")
                sys.exit(1)

            # Handle dockerfile specification
            # Priority: 1) dockerfile in test_config, 2) Dockerfile in config dir, 3) command-line arg
            if "dockerfile" in config:
                # User explicitly specified dockerfile in config
                dockerfile_path = (config_dir / config["dockerfile"]).resolve()
                if not dockerfile_path.exists():
                    logger.error(f"❌ Dockerfile specified in test config not found: {dockerfile_path}")
                    sys.exit(1)
                config["_resolved_dockerfile"] = str(dockerfile_path)
                logger.info(f"🔗 Using Dockerfile from config: {config['dockerfile']} -> {dockerfile_path}")
            else:
                # Check for Dockerfile in same directory as config
                default_dockerfile = config_dir / "Dockerfile"
                if default_dockerfile.exists():
                    config["_resolved_dockerfile"] = str(default_dockerfile)
                    logger.info(f"🔗 Using default Dockerfile from config directory: {default_dockerfile}")
                else:
                    # Fall back to command-line specified dockerfile
                    config["_resolved_dockerfile"] = args.dockerfile
                    logger.info(f"🔗 Using Dockerfile from command-line: {args.dockerfile}")

            # Base directory is ALWAYS the directory containing the test config JSON
            # This makes test configs self-contained and portable
            config["_resolved_base_dir"] = str(config_dir)
            config["_config_dir"] = str(config_dir)  # Store for model downloads
            logger.info(f"🔗 Using base_dir (config directory): {config_dir}")

            # Validate build_args if present
            if "build_args" in config:
                if not isinstance(config["build_args"], dict):
                    logger.error(f"❌ build_args in test configuration {i+1} must be a dictionary")
                    sys.exit(1)

                # Resolve paths relative to config file directory
                path_args = ["APP_YAML_PATH", "OUTPUT_DIR", "PROCESSORS_PATH", "OPENAPI_SPEC"]
                for path_arg in path_args:
                    if path_arg in config["build_args"]:
                        original_path = config["build_args"][path_arg]
                        # Resolve relative to config file directory
                        resolved_path = (config_dir / original_path).resolve()
                        # Convert back to relative path from current working directory
                        try:
                            rel_path = resolved_path.relative_to(Path.cwd().resolve())
                            config["build_args"][path_arg] = str(rel_path)
                            logger.info(f"🔗 Resolved {path_arg}: {original_path} -> {rel_path}")
                        except ValueError:
                            # If can't make relative, use absolute path
                            config["build_args"][path_arg] = str(resolved_path)
                            logger.info(f"🔗 Resolved {path_arg}: {original_path} -> {resolved_path}")

                for key, value in config["build_args"].items():
                    if not validate_build_arg_name(key):
                        logger.error(f"❌ Invalid build arg name in test configuration {i+1}: {key}")
                        sys.exit(1)
                    if not isinstance(value, str) or not validate_build_arg_value(value):
                        logger.error(f"❌ Invalid build arg value in test configuration {i+1}: {value}")
                        sys.exit(1)

            # Validate test_config if present
            if "test_config" in config:
                test_config = config["test_config"]
                config_valid, config_error = validate_test_config(test_config)
                if not config_valid:
                    logger.error(f"❌ Test configuration {i+1} validation failed: {config_error}")
                    sys.exit(1)

        # Filter test configurations by test case name if specified
        if args.test_case:
            original_count = len(test_configs)
            if args.test_case == "*":
                logger.info("--test-case '*' specified: selecting all tests (including disabled).")
            else:
                test_configs = [
                    config for config in test_configs
                    if args.test_case.lower() in config.get("name", "").lower()
                ]

                if not test_configs:
                    logger.error(f"❌ No test cases found matching '{args.test_case}'")
                    logger.info("Available test cases:")
                    # Reload original configs to show available names
                    with open(args.config_file, 'r') as f:
                        original_configs = json.load(f)
                    for i, config in enumerate(original_configs, 1):
                        logger.info(f"  {i}. {config.get('name', f'Unnamed test {i}')}")
                    sys.exit(1)

                logger.info(f"Filtered {original_count} test configurations to {len(test_configs)} matching '{args.test_case}'")
                for i, config in enumerate(test_configs, 1):
                    logger.info(f"  {i}. {config.get('name', f'Test {i}')}")
        else:
            # Include disabled tests; they will run codegen-only path unless -c/--test-case forces full flow.
            logger.info("Including disabled tests (default_enable=false). They will run codegen-only unless --test-case is provided.")

    except json.JSONDecodeError as e:
        logger.error(f"❌ Invalid JSON in config file: {str(e)}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Failed to load config file: {str(e)}")
        sys.exit(1)

    # Run tests with error handling
    try:
        logger.info(f"Starting test suite with {len(test_configs)} configurations")
        logger.info(f"Logs will be saved to: {args.log_dir}")
        logger.info(f"Using GPU devices: {args.gpus}")
        # Force full flow if --test-case provided (including "*")
        force_full_flow = args.test_case is not None and len(args.test_case) > 0
        logger.info(f"🔍 Test suite settings: test_case={args.test_case}, force_full_flow={force_full_flow}")
        results = tester.run_test_suite(test_configs, cleanup=not args.no_cleanup, gitlab_token=args.gitlab_token, force_full_flow=force_full_flow, gpus=args.gpus)
    except Exception as e:
        logger.error(f"❌ Test suite execution failed: {str(e)}")
        sys.exit(1)

    # Generate report with error handling
    try:
        tester.generate_report(results, args.output)
    except Exception as e:
        logger.error(f"❌ Report generation failed: {str(e)}")
        sys.exit(1)

    # Exit with appropriate code
    if results["failed"] > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()