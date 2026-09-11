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

import argparse
import base64
import tempfile
import shutil
from omegaconf import OmegaConf
import cookiecutter.main
import cookiecutter
import logging
from typing import Dict, List
from pathlib import Path
from utils import get_resource_path, copy_files, create_tar_gz
from triton.utils import generate_pbtxt
from omegaconf.errors import ConfigKeyError
from jinja2 import Environment, FileSystemLoader
import ast
import os
import sys
import subprocess
import validate
import re
import json
import sys

# Import schema validation functions
try:
    from jsonschema import Draft7Validator
except ImportError:
    Draft7Validator = None
    logging.warning("jsonschema not installed. Schema validation will be skipped. Install with: pip install jsonschema")

"""
Inference Builder Main Module

Security Features:
- Input validation for file paths and directory paths to prevent directory traversal attacks
- Server type validation against allowed values to prevent injection
- Safe subprocess calls using argument lists instead of shell string interpolation
- Comprehensive argument validation in main function to prevent command injection
- Logging of all executed commands for audit trails
"""

LICENSE_HEADER = """
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


def load_allowed_servers(file_path: str = None) -> List[str]:
    """Load allowed server types from a text file.

    Args:
        file_path: Path to the text file containing allowed server types.
                  If None, uses the default 'allowed_servers.txt'
                  in the builder directory.

    Returns:
        List of allowed server type strings.

    Raises:
        FileNotFoundError: If the configuration file is not found.
        ValueError: If the file is empty or contains invalid entries.
    """
    if file_path is None:
        # Use default file in the same directory as this script
        script_dir = Path(__file__).parent
        file_path = script_dir / "allowed_servers.txt"
    else:
        file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Allowed servers configuration file not found: {file_path}"
        )

    allowed_servers = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Strip whitespace and ignore empty lines and comments
            line = line.strip()
            if line and not line.startswith('#'):
                allowed_servers.append(line)

    if not allowed_servers:
        raise ValueError(f"No valid server types found in {file_path}")

    return allowed_servers


# Load allowed server types from configuration file
ALLOWED_SERVER = load_allowed_servers()


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Main")
OmegaConf.register_new_resolver("multiline", lambda x: x, replace=False)


def validate_file_path(file_path: str) -> str:
    """Validate and sanitize a file path to prevent directory traversal and command injection.

    Returns:
        str: Validated and resolved absolute file path.
    Raises:
        ValueError: If path is invalid or potentially dangerous.
    """
    if not file_path:
        raise ValueError("File path cannot be empty")

    try:
        abs_path = os.path.abspath(file_path)
        resolved_path = os.path.realpath(abs_path)
    except (OSError, ValueError) as e:
        raise ValueError(f"Invalid file path: {e}")

    if ".." in os.path.normpath(file_path):
        raise ValueError("Path traversal detected in file path")

    invalid_chars = ['<', '>', ':', '"', '|', '?', '*']
    if any(char in file_path for char in invalid_chars):
        raise ValueError("Invalid characters in file path")

    system_dirs = ['/etc', '/sys', '/proc', '/dev', '/boot', '/usr/bin', '/usr/sbin']
    for sys_dir in system_dirs:
        if resolved_path.startswith(sys_dir):
            raise ValueError(f"Access to system directory {sys_dir} is not allowed")

    path = Path(resolved_path)
    if not path.exists() or not path.is_file():
        raise ValueError(f"File not found or not a regular file: {resolved_path}")

    return resolved_path


def validate_directory_path(dir_path: str) -> str:
    """Validate and sanitize a directory path to prevent directory traversal attacks.

    Returns:
        str: Validated and resolved absolute directory path.
    Raises:
        ValueError: If path is invalid or potentially dangerous.
    """
    if not dir_path:
        raise ValueError("Directory path cannot be empty")

    try:
        abs_path = os.path.abspath(dir_path)
        resolved_path = os.path.realpath(abs_path)
    except (OSError, ValueError) as e:
        raise ValueError(f"Invalid directory path: {e}")

    if ".." in os.path.normpath(dir_path):
        raise ValueError("Path traversal detected in directory path")

    invalid_chars = ['<', '>', ':', '"', '|', '?', '*']
    if any(char in dir_path for char in invalid_chars):
        raise ValueError("Invalid characters in directory path")

    system_dirs = ['/etc', '/sys', '/proc', '/dev', '/boot', '/usr/bin', '/usr/sbin']
    for sys_dir in system_dirs:
        if resolved_path.startswith(sys_dir):
            raise ValueError(f"Access to system directory {sys_dir} is not allowed")

    return resolved_path


def validate_server_type(server_type: str) -> bool:
    """Validate server type against allowed values."""
    return server_type in ALLOWED_SERVER


def validate_config_against_schema(config_dict: Dict, schema_dir: Path = None) -> bool:
    """Validate configuration against JSON schema.

    Args:
        config_dict: Configuration dictionary to validate
        schema_dir: Directory containing schema files (default: ../schemas from this file)

    Returns:
        True if valid, False otherwise
    """
    if Draft7Validator is None:
        logger.warning("Skipping schema validation (jsonschema not installed)")
        return True

    # Determine schema directory
    if schema_dir is None:
        schema_dir = Path(__file__).parent.parent / "schemas"

    if not schema_dir.exists():
        logger.warning(f"Schema directory not found: {schema_dir}. Skipping schema validation.")
        return True

    # Load main schema
    schema_path = schema_dir / "config.schema.json"
    try:
        with open(schema_path, 'r') as f:
            schema = json.load(f)
    except FileNotFoundError:
        logger.warning(f"Schema file not found: {schema_path}. Skipping schema validation.")
        return True
    except json.JSONDecodeError as e:
        logger.error(f"Invalid schema file: {e}")
        return False

    # Validate configuration
    try:
        validator = Draft7Validator(schema)
        errors = list(validator.iter_errors(config_dict))

        if not errors:
            logger.info("✓ Configuration schema validation passed")
            return True

        logger.error(f"✗ Configuration validation failed with {len(errors)} error(s):")
        for i, error in enumerate(errors, 1):
            path = '.'.join(str(p) for p in error.path) if error.path else 'root'
            logger.error(f"  Error {i} at '{path}': {error.message}")

            if error.context:
                for ctx_error in error.context:
                    logger.error(f"    - {ctx_error.message}")

        return False

    except Exception as e:
        logger.error(f"Error during schema validation: {e}")
        return False


def get_version():
    version_file = Path(__file__).parent.parent / "VERSION"
    with open(version_file, 'r') as f:
        return f.read().strip()


def build_args(parser):
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"%(prog)s {get_version()}"
    )
    parser.add_argument(
        "--server-type",
        type=str,
        nargs='?',
        default='fastapi',
        choices=ALLOWED_SERVER,
        help="Choose the server type"
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        nargs='?',
        default='.',
        help="Output directory"
    )
    parser.add_argument(
        "-a",
        "--api-spec",
        type=argparse.FileType('r'),
        nargs='?',
        help="File for OpenAPI specification"
    )
    parser.add_argument(
        "-c",
        "--custom-module",
        type=argparse.FileType('r'),
        nargs='*',
        help="Custom python modules"
    )
    parser.add_argument(
        "-x",
        "--exclude-lib",
        action='store_true',
        help="Don't include common lib to the generated code."
    )
    parser.add_argument(
        "-t",
        "--tar-output",
        action='store_true',
        help="Zip the output to a single file"
    )
    parser.add_argument("config", type=str, help="Path the the configuration")
    parser.add_argument(
        "--validation-dir",
        type=str,
        help="valid validation directory path to build validator"
    )
    parser.add_argument(
        "--no-docker",
        action="store_true",
        help="Use local OpenAPI Generator instead of Docker for OpenAPI client generation")
    parser.add_argument(
        "--test-cases-abs-path",
        action="store_true",
        help="Use absolute paths in generated test_cases.yaml"
    )

def build_tree(server_type, config, temp_dir):
    cookiecutter.main.cookiecutter(
        get_resource_path(f"builder/boilerplates/{server_type}"),
        no_input=True,
        extra_context={"service_name": config.name},
        output_dir=temp_dir)
    return Path(temp_dir) / Path(config.name)

def build_custom_modules(custom_modules: List, tree):
    cls_list = []
    tpl_dir = get_resource_path("templates")
    jinja_env = Environment(loader=FileSystemLoader(tpl_dir))
    tree = tree / "custom"
    for m in custom_modules:
        filename = os.path.basename(m.name)
        module_id = os.path.splitext(filename)[0]
        with m as f:
            valid = False
            source = f.read()
            py = ast.parse(source)
            for node in ast.walk(py):
                if isinstance(node, ast.ClassDef):
                    name = None
                    has_call = False
                    for cls_node in node.body:
                        if isinstance(cls_node, ast.Assign):
                            for target in cls_node.targets:
                                if isinstance(target, ast.Name) and target.id == "name":
                                    name = ast.literal_eval(cls_node.value)
                        if isinstance(cls_node, ast.FunctionDef):
                            if cls_node.name == "__call__":
                                has_call = True
                    if name and has_call:
                        if next((i for i in cls_list if i["name"] == name), None):
                            logger.warning(f"Custom class {name} defined more than once")
                            continue
                        cls_list.append({
                            "name": name,
                            "module": module_id,
                            "class_name": node.name
                        })
                        valid = True
            if valid:
                # write the python module
                target_path = tree / f"{module_id}.py"
                with open(target_path, "w") as t:
                    t.write(source)
    custom_tpl = jinja_env.get_template('common/custom.__init__.jinja.py')
    output = custom_tpl.render(classes=cls_list)
    with open(tree/"__init__.py", 'w') as f:
        f.write(output)

def build_inference(server_type, config, output_dir: Path):
    tpl_dir = get_resource_path("templates")
    jinja_env = Environment(loader=FileSystemLoader(tpl_dir))
    triton_model_repo_dir = output_dir/'model_repo'
    triton_tpl = jinja_env.get_template("triton/model.jinja.py")
    generic_tpl = jinja_env.get_template("generic/model.jinja.py")

    # collect the backend: build model.py and pbtxt if the backend is triton
    t_backends = []
    for model in config.models:
        backend_spec = model.backend.split('/')
        if backend_spec[0] == "triton":
            os.makedirs(triton_model_repo_dir/f"{model.name}/1", exist_ok=True)
            if len(backend_spec) < 2:
                raise Exception("Triton backend needs a triton backend type")
            if backend_spec[1] == "python":
                if len(backend_spec) < 3:
                    raise Exception("Triton python backend needs an implementation type")
                # generating triton model for triton backend
                target_dir = triton_model_repo_dir/f"{model.name}/1"
                    # Triton python backend needs a model.py
                backend_tpl = jinja_env.get_template(f"backend/{backend_spec[2]}.jinja.py")
                backend = backend_tpl.render(server_type=server_type)
                output = triton_tpl.render(backends=[backend], top_level=False, license=LICENSE_HEADER)
                with open (target_dir/"model.py", 'w') as o:
                    o.write(output)
            # triton python backend to communicate with triton fastapi server
            if "triton" not in t_backends:
                t_backends.append("triton")
            # write the pbtxt
            pbtxt_str = generate_pbtxt(OmegaConf.to_container(model), backend_spec[1] )
            pbtxt_path = triton_model_repo_dir/model.name/"config.pbtxt"
            with open(pbtxt_path, 'w') as f:
                f.write(pbtxt_str)
        else:
            bare_backend = backend_spec[0]
            if bare_backend not in t_backends:
                t_backends.append(bare_backend)

    # create backends and model.py
    backends = []
    if server_type == "serverless":
        target_dir = output_dir / "app"
    elif server_type == "triton":
        target_dir = triton_model_repo_dir/f"{config.name}"/"1/"
    else:
        target_dir = output_dir / "server"
    for backend in t_backends:
        backend_tpl = jinja_env.get_template(f"backend/{backend}.jinja.py")
        backends.append(backend_tpl.render(server_type=server_type))
        if server_type == "triton":
            # render top level triton backend
            output = triton_tpl.render(backends=backends, top_level=True)
        else:
            # render generic model.y
            output = generic_tpl.render(backends=backends, license=LICENSE_HEADER)
        with open (target_dir/"model.py", 'w') as o:
            o.write(output)

def build_serverless(name: str, output_dir: Path):
    output_dir = output_dir / "app"
    tpl_dir = get_resource_path("templates")
    jinja_env = Environment(loader=FileSystemLoader(tpl_dir))
    app_tpl = jinja_env.get_template("serverless/inference.jinja.py")
    output = app_tpl.render(service_name=name, license=LICENSE_HEADER)
    with open(output_dir/"inference.py", 'w') as f:
        f.write(output)

def build_server(server_type, model_name, api_spec, config: Dict, output_dir):
    output_dir = output_dir / "server"
    # generate pydantic data models and inference base class from swagger spec
    tpl_dir = get_resource_path("templates")
    jinja_env = Environment(loader=FileSystemLoader(tpl_dir))
    api_tpl_dir = get_resource_path(f"templates/api_server/{server_type}/route")

    # Use safer subprocess call with argument list instead of shell string interpolation
    # Find fastapi-codegen in the same directory as the current Python executable
    python_dir = Path(sys.executable).parent
    fastapi_codegen = python_dir / "fastapi-codegen"
    
    command = [
        str(fastapi_codegen),
        "--input", api_spec.name,
        "--output", str(output_dir),
        "--output-model-type", "pydantic_v2.BaseModel",
        "--template-dir", str(api_tpl_dir),
        "-m", "data_model.py",
        "--disable-timestamp"
    ]

    logger.info(f"Executing command: {' '.join(command)}")
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"Failed to generate fastapi data models: {result.stderr}")

    responders = []
    for name, r in config["responders"].items():
        responder = {
            "name": name,
            "operation": r["operation"],
        }
        tpl = jinja_env.get_template(f"responder/{name}.jinja.py")
        responder["implementation"] = tpl.render(**responder)
        responders.append(responder)
    svr_tpl = jinja_env.get_template(f"api_server/{server_type}/responder.jinja.py")
    # render the responder.py
    if server_type == "triton":
        req_cls = [k for k in config["responders"]["infer"]["requests"].keys()]
        res_cls = [k for k in config["responders"]["infer"]["responses"].keys()]
        triton_config = {
            "request_class": req_cls[0],
            "response_class": res_cls[0],
            "streaming_response_class": res_cls[1] if len(res_cls) > 1 else res_cls[0]
        }
        output = svr_tpl.render(
            service_name=model_name,
            license=LICENSE_HEADER,
            responders=responders,
            triton=triton_config
        )
    elif server_type == "fastapi":
        output = svr_tpl.render(
            service_name=model_name,
            license=LICENSE_HEADER,
            responders=responders
        )
    else:
        raise ValueError(f"Unsupported server type: {server_type}")
    with open(output_dir/"responder.py", 'w') as f:
        f.write(output)


def generate_configuration(config, tree):
    def encode_templates(templates):
        encoded_templates = dict()
        for key, value in templates.items():
            # these are json templates and need be encoded before being
            # embeded as yaml strings
            if isinstance(value, str):
                encoded_templates[key] = base64.b64encode(value.encode())
            else:
                encoded_templates[key] = value
        return encoded_templates
    # base64 encode the templates if found in the config
    config_map = OmegaConf.to_container(config)
    if "input" not in config_map:
        config_map["input"] = []
        for m in config_map["models"]:
            for i in m["input"]:
                config_map["input"].append(i)
    if "output" not in config_map:
        config_map["output"] = []
        for m in config_map["models"]:
            for o in m["output"]:
                config_map["output"].append(o)
    if "server" in config_map:
        try:
            for responder in config_map["server"]["responders"].values():
                input_templates = responder.get("requests", None)
                output_templates = responder.get("responses", None)
                if input_templates:
                    responder["requests"] = encode_templates(input_templates)
                if output_templates:
                    responder["responses"] = encode_templates(output_templates)
        except ConfigKeyError:
            raise ValueError("Server config error: responders not found")
    # write the config to a python file
    tpl_dir = get_resource_path("templates")
    jinja_env = Environment(loader=FileSystemLoader(tpl_dir))
    config_tpl = jinja_env.get_template('common/config.jinja.py')
    config = OmegaConf.create(config_map)
    output = config_tpl.render(config=OmegaConf.to_yaml(config), license=LICENSE_HEADER)
    with open(tree/"config/__init__.py", 'w') as f:
        f.write(output)


def main(args):
    # Defense-in-depth: validate and sanitize all path arguments
    try:
        args.config = validate_file_path(args.config)
    except ValueError as e:
        raise ValueError(f"Invalid config file path: {e}")

    if not validate_server_type(args.server_type):
        raise ValueError(f"Invalid server type: {args.server_type}")

    try:
        args.output_dir = validate_directory_path(args.output_dir)
    except ValueError as e:
        raise ValueError(f"Invalid output directory: {e}")

    if args.validation_dir:
        try:
            args.validation_dir = validate_directory_path(args.validation_dir)
        except ValueError as e:
            raise ValueError(f"Invalid validation directory: {e}")

    if args.custom_module:
        for module in args.custom_module:
            try:
                validate_file_path(module.name)
            except ValueError as e:
                raise ValueError(f"Invalid custom module file path: {e}")

    # Load configuration
    logger.info(f"Loading configuration from: {args.config}")
    config = OmegaConf.load(args.config)

    # Validate configuration against JSON schema
    logger.info("Validating configuration against JSON schema...")
    config_dict = OmegaConf.to_container(config, resolve=True)

    if not validate_config_against_schema(config_dict):
        logger.error("Configuration validation failed. Fix the errors above and try again.")
        sys.exit(1)

    with tempfile.TemporaryDirectory() as temp_dir:
        tree = build_tree(args.server_type, config, temp_dir)
        if args.server_type == "serverless":
            build_serverless(config.name, tree)
        else:
            build_server(args.server_type, config.name, args.api_spec, OmegaConf.to_container(config.server), tree)
        build_inference(args.server_type, config, tree)
        generate_configuration(config, tree)
        if not args.exclude_lib :
            copy_files(get_resource_path("lib"), tree/"lib")
        if args.custom_module:
            build_custom_modules(args.custom_module, tree)
        if args.tar_output:
            target = Path(args.output_dir).resolve() / f"{config.name}.tgz"
            create_tar_gz(target, tree)
        else:
            try:
                target = Path(args.output_dir).resolve() / config.name
                shutil.copytree(tree, target, dirs_exist_ok=True)
            except FileExistsError:
                logging.error(f"{target} already exists in the output directory")
        # Build validator if validation dir provided
        if args.validation_dir:
            try:
                validation_dir = Path(args.validation_dir).resolve()
                api_spec_path = Path(args.api_spec.name).resolve()
                if validate.build_validation(api_spec_path, validation_dir, not args.no_docker, args.test_cases_abs_path):
                    logger.info("✓ Successfully built validation")
                else:
                    logger.error("✗ Failed to build validation")
            except Exception as e:
                logger.error(f"validation build failed: {e}")
                # Continue with other builds
                # Continue to finish the build without validation
    print("Build completed successfully")



if __name__ == "__main__":
    parser = argparse.ArgumentParser("Inference Builder")
    build_args(parser)

    try:
        args = parser.parse_args()
    except Exception as e:
        print(f"Error: Argument parsing failed: {str(e)}")
        sys.exit(1)

    # Comprehensive security validation
    validation_errors = []

    # Validate and sanitize config file path
    try:
        args.config = validate_file_path(args.config)
    except ValueError as e:
        validation_errors.append(f"Invalid config file path: {e}")

    # Validate server type (already constrained by argparse choices)
    if not validate_server_type(args.server_type):
        validation_errors.append(f"Invalid server type: {args.server_type}")

    # Validate and sanitize output directory
    try:
        args.output_dir = validate_directory_path(args.output_dir)
    except ValueError as e:
        validation_errors.append(f"Invalid output directory: {e}")

    # Validate and sanitize validation directory
    if args.validation_dir:
        try:
            args.validation_dir = validate_directory_path(args.validation_dir)
        except ValueError as e:
            validation_errors.append(f"Invalid validation directory: {e}")

    # Exit if any validation errors
    if validation_errors:
        print("Security validation failed:")
        for error in validation_errors:
            print(f"  - {error}")
        sys.exit(1)

    main(args)