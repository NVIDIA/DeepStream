#!/usr/bin/env python3
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
Validate Inference Builder configuration files against JSON schemas.

Usage:
    python validate_config.py config.yaml
    python validate_config.py --schema custom_schema.json config.yaml
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any

try:
    import yaml
except ImportError:
    print("Error: PyYAML is required. Install with: pip install pyyaml")
    sys.exit(1)

try:
    from jsonschema import validate, ValidationError, Draft7Validator
except ImportError:
    print("Error: jsonschema is required. Install with: pip install jsonschema")
    sys.exit(1)


def load_yaml(file_path: Path) -> Dict[Any, Any]:
    """Load YAML configuration file.

    Args:
        file_path: Path to YAML file

    Returns:
        Parsed YAML as dictionary
    """
    try:
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        sys.exit(1)


def load_schema(schema_path: Path) -> Dict[Any, Any]:
    """Load JSON schema file.

    Args:
        schema_path: Path to schema file

    Returns:
        Parsed schema as dictionary
    """
    try:
        with open(schema_path, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error parsing schema file: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print(f"Error: Schema file not found: {schema_path}")
        sys.exit(1)


def validate_config(config: Dict[Any, Any], schema: Dict[Any, Any], verbose: bool = False) -> bool:
    """Validate configuration against schema.

    Args:
        config: Configuration dictionary
        schema: JSON schema dictionary
        verbose: Enable verbose output

    Returns:
        True if valid, False otherwise
    """
    try:
        # Create validator
        validator = Draft7Validator(schema)

        # Check for errors
        errors = list(validator.iter_errors(config))

        if not errors:
            print("✓ Configuration is valid!")
            return True

        print(f"✗ Configuration validation failed with {len(errors)} error(s):\n")

        for i, error in enumerate(errors, 1):
            print(f"Error {i}:")
            print(f"  Path: {'.'.join(str(p) for p in error.path)}")
            print(f"  Message: {error.message}")

            if verbose and error.context:
                print(f"  Context:")
                for ctx_error in error.context:
                    print(f"    - {ctx_error.message}")

            print()

        return False

    except Exception as e:
        print(f"Error during validation: {e}")
        return False


def get_backend_schema(config: Dict[Any, Any], schema_dir: Path) -> Path:
    """Determine the appropriate backend schema based on config.

    Args:
        config: Configuration dictionary
        schema_dir: Directory containing schemas

    Returns:
        Path to backend-specific schema or main schema
    """
    if 'models' not in config or not config['models']:
        return schema_dir / 'config.schema.json'

    # Get the backend from the first model
    backend = config['models'][0].get('backend', '')

    backend_map = {
        'deepstream/nvinfer': 'deepstream.schema.json',
        'triton': 'triton.schema.json',
        'vllm': 'vllm.schema.json',
        'tensorrtllm': 'tensorrtllm.schema.json',
        'polygraphy': 'polygraphy.schema.json',
        'dummy': 'dummy.schema.json',
        'pytorch': 'pytorch.schema.json'
    }

    for prefix, schema_file in backend_map.items():
        if backend.startswith(prefix):
            backend_schema = schema_dir / 'backends' / schema_file
            if backend_schema.exists():
                return backend_schema

    return schema_dir / 'config.schema.json'


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Validate Inference Builder configuration files'
    )
    parser.add_argument(
        'config',
        type=Path,
        help='Path to configuration YAML file'
    )
    parser.add_argument(
        '-s', '--schema',
        type=Path,
        help='Path to JSON schema file (default: auto-detect from config)'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    parser.add_argument(
        '--schema-dir',
        type=Path,
        default=Path(__file__).parent,
        help='Directory containing schema files'
    )

    args = parser.parse_args()

    # Load configuration
    if args.verbose:
        print(f"Loading configuration from: {args.config}")
    config = load_yaml(args.config)

    # Determine schema to use
    if args.schema:
        schema_path = args.schema
    else:
        schema_path = get_backend_schema(config, args.schema_dir)
        if args.verbose:
            print(f"Auto-detected schema: {schema_path}")

    # Load schema
    if args.verbose:
        print(f"Loading schema from: {schema_path}")
    schema = load_schema(schema_path)

    # Validate
    if args.verbose:
        print("\nValidating configuration...\n")

    is_valid = validate_config(config, schema, args.verbose)

    sys.exit(0 if is_valid else 1)


if __name__ == '__main__':
    main()

