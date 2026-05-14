# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
Security validation utilities for TAO evaluation scripts.

This module provides various validation functions to ensure that user-provided
input is safe and cannot be used for path traversal or other security attacks.
"""

import os


def validate_safe_path(path_component: str) -> bool:
    """Validate that a path component is safe and doesn't contain traversal sequences.

    Args:
        path_component: The path component to validate

    Returns:
        bool: True if safe, False if potentially malicious
    """
    # Check for path traversal sequences
    if '..' in path_component:
        return False

    # Check for absolute paths
    if os.path.isabs(path_component):
        return False

    # Check for hidden files starting with dot (optional security measure)
    if path_component.startswith('.'):
        return False

    # Check for empty or whitespace-only names
    if not path_component.strip():
        return False

    # Check for invalid characters that might be used in attacks
    invalid_chars = ['\\', '/', ':', '*', '?', '"', '<', '>', '|']
    if any(char in path_component for char in invalid_chars):
        return False

    return True


def validate_config_path(config_path: str) -> bool:
    """Validate configuration file path for security."""
    if not config_path or not isinstance(config_path, str):
        return False

    # Check for path traversal attempts
    if '..' in config_path or '//' in config_path:
        return False

    # Check for absolute paths that could access system directories
    if os.path.isabs(config_path):
        system_dirs = ['/etc', '/sys', '/proc', '/dev', '/boot',
                       '/usr/bin', '/usr/sbin', '/root']
        for sys_dir in system_dirs:
            if config_path.startswith(sys_dir):
                return False

    # Ensure it's a YAML file
    if not config_path.lower().endswith(('.yaml', '.yml')):
        return False

    # Check for invalid characters
    invalid_chars = ['<', '>', ':', '"', '|', '?', '*', '\\']
    if any(char in config_path for char in invalid_chars):
        return False

    return True


def validate_csv_path(csv_path: str) -> bool:
    """Validate CSV file path for security."""
    if not csv_path:
        return True  # Empty path is allowed

    if not isinstance(csv_path, str):
        return False

    # Check for path traversal attempts
    if '..' in csv_path or '//' in csv_path:
        return False

    # Check for absolute paths that could access system directories
    if os.path.isabs(csv_path):
        system_dirs = ['/etc', '/sys', '/proc', '/dev', '/boot',
                       '/usr/bin', '/usr/sbin', '/root']
        for sys_dir in system_dirs:
            if csv_path.startswith(sys_dir):
                return False

    # Ensure it's a CSV file
    if not csv_path.lower().endswith('.csv'):
        return False

    # Check for invalid characters
    invalid_chars = ['<', '>', ':', '"', '|', '?', '*', '\\']
    if any(char in csv_path for char in invalid_chars):
        return False

    return True


def validate_dump_vis_path(dump_vis_path: str) -> bool:
    """Validate dump visualization path for security."""
    if not dump_vis_path:
        return True  # Empty path is allowed

    if not isinstance(dump_vis_path, str):
        return False

    # Check for path traversal attempts
    if '..' in dump_vis_path or '//' in dump_vis_path:
        return False

    # Check for absolute paths that could access system directories
    if os.path.isabs(dump_vis_path):
        system_dirs = ['/etc', '/sys', '/proc', '/dev', '/boot',
                       '/usr/bin', '/usr/sbin', '/root']
        for sys_dir in system_dirs:
            if dump_vis_path.startswith(sys_dir):
                return False

    # Check for invalid characters
    invalid_chars = ['<', '>', ':', '"', '|', '?', '*', '\\']
    if any(char in dump_vis_path for char in invalid_chars):
        return False

    return True


def validate_directory_path(dir_path: str) -> bool:
    """Validate directory path from config for security."""
    if not dir_path or not isinstance(dir_path, str):
        return False

    # Check for path traversal attempts
    if '..' in dir_path or '//' in dir_path:
        return False

    # Check for absolute paths that could access system directories
    if os.path.isabs(dir_path):
        system_dirs = ['/etc', '/sys', '/proc', '/dev', '/boot',
                       '/usr/bin', '/usr/sbin', '/root']
        for sys_dir in system_dirs:
            if dir_path.startswith(sys_dir):
                return False

    # Check for invalid characters
    invalid_chars = ['<', '>', ':', '"', '|', '?', '*', '\\']
    if any(char in dir_path for char in invalid_chars):
        return False

    return True


def validate_split_name(split_name: str) -> bool:
    """Validate dataset split name from config for security."""
    if not split_name or not isinstance(split_name, str):
        return False

    # Split names should be simple strings without path separators
    if '/' in split_name or '\\' in split_name:
        return False

    # Check for path traversal attempts
    if '..' in split_name:
        return False

    # Check for invalid characters
    invalid_chars = ['<', '>', ':', '"', '|', '?', '*']
    if any(char in split_name for char in invalid_chars):
        return False

    # Ensure it's a reasonable split name (alphanumeric, underscore, dash)
    if not split_name.replace('_', '').replace('-', '').isalnum():
        return False

    return True


def validate_test_prefix(test_prefix: str) -> bool:
    """Validate test prefix path from config for security."""
    if not test_prefix or not isinstance(test_prefix, str):
        return False

    # Check for path traversal attempts
    if '..' in test_prefix or '//' in test_prefix:
        return False

    # Check for absolute paths (test prefix should be relative)
    if os.path.isabs(test_prefix):
        return False

    # Check for invalid characters
    invalid_chars = ['<', '>', ':', '"', '|', '?', '*', '\\']
    if any(char in test_prefix for char in invalid_chars):
        return False

    return True


def validate_integer_parameter(value: int, min_val: int = 1,
                               max_val: int = 1000000) -> bool:
    """Validate integer parameters for security."""
    if value is None:
        return True  # None is allowed

    if not isinstance(value, int):
        return False

    if value < min_val or value > max_val:
        return False

    return True
