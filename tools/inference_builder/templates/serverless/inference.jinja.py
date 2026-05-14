{#
 SPDX-FileCopyrightText: Copyright (c) <year> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 SPDX-License-Identifier: Apache-2.0

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
#}
{{ license }}

import argparse
from config import global_config
from typing import List, Optional
from lib.inference import py_datatype_mapping
from .model import GenericInference
import json
import sys
from lib.utils import NumpyFlatEncoder


def create_parser(inputs: List) -> argparse.ArgumentParser:
    """Create and configure the argument parser for the application."""
    parser = argparse.ArgumentParser(
        description='Command line interface for{{ service_name }}'
    )
    for input_item in inputs:
        input_name = input_item.name.replace("_", "-")
        nargs = 1
        optional = input_item.optional if hasattr(input_item, "optional") else False
        if len(input_item.dims) == 1:
            if input_item.dims[0] == -1:
                nargs = "+"
            elif input_item.dims[0] == 1 and optional:
                nargs = "?"
            else:
                nargs = input_item.dims[0]
        parser.add_argument(
            f'--{input_name}',
            type=py_datatype_mapping[input_item.data_type],
            nargs=nargs,
            required=not optional
        )
    parser.add_argument(
        '-s',
        '--save-to',
        type=str,
        nargs='?',
        const='stdout',
        help='Save results to a file, or use -s alone for stdout'
    )
    return parser


def run_inference(args) -> Optional[int]:
    """Run the inference service synchronously.

    Args:
        args: Parsed command line arguments

    Returns:
        Optional[int]: Exit code, 0 for success
    """
    # Read and remove save_to argument before converting to dict
    save_to = getattr(args, 'save_to', None)
    if hasattr(args, 'save_to'):
        delattr(args, 'save_to')

    # Convert remaining args to dictionary
    inputs = vars(args)

    # Convert input names back to original format (replace '-' with '_')
    inputs = {k.replace('-', '_'): v for k, v in inputs.items()}

    service = GenericInference()
    service.initialize()
    status = 0

    # Determine output target: file, stdout, or None (no output)
    output_file = None
    close_file = False
    if save_to:
        if save_to in ('-', 'stdout'):
            output_file = sys.stdout
        else:
            output_file = open(save_to, 'w', encoding='utf-8')
            close_file = True

    try:
        for result in service.exec_sync(inputs):
            if not result:
                status = -1
                break
            # Unified logic: write JSON to output target if specified
            if output_file:
                json_str = json.dumps(result, indent=4, cls=NumpyFlatEncoder)
                output_file.write(json_str)
                output_file.write("\n")
                if close_file:
                    output_file.flush()  # Ensure data is written to disk
    finally:
        if close_file and output_file:
            output_file.close()

    print("Inference completed.")
    service.finalize()
    return status


def main() -> int:
    """Main entry point for the inference service.

    Returns:
        Optional[int]: Exit code, None for success
    """
    parser = create_parser(global_config.input)
    args = parser.parse_args()

    try:
        return run_inference(args)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 130
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
