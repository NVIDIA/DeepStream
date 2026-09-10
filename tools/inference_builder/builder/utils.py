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

from pathlib import Path
import os
import shutil
from typing import Callable
import tarfile
import mimetypes
import base64

class PayloadBuilder:
    """
    Builder class for constructing the payload with required and optional fields

    Example:
        >>> builder = PayloadBuilder(files=["image1.jpg"], model="nvdino-v2")
        >>> payload = builder.add_text(["cat,dog"]).build()
        {
            "input": ["data:image/jpeg;base64,/9j/4AAQSkZJRg..."],
            "model": "nvidia/nvdino-v2",
            "text": [["cat", "dog"]]
        }
    """
    @staticmethod
    def prepare_image_inputs(files):
        """
        Prepare base64 encoded image inputs for the payload
        Args:
            files: List of file paths, e.g. ["path/to/image1.jpg", "path/to/image2.png"]
        Returns:
            list: List of encoded image strings in format "data:mime_type;base64,..."

        Example:
            >>> prepare_image_inputs(["image1.jpg", "image2.png"])
            [
                "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABgAAD/4QBmRX...",
                "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAA7..."
            ]
        """
        encoded_inputs = []
        for file in files:
            mime_type = mimetypes.guess_type(file)[0]
            with open(file, "rb") as f:
                b64_image = base64.b64encode(f.read()).decode()
                encoded_inputs.append(f"data:{mime_type};base64,{b64_image}")
        return encoded_inputs

    @staticmethod
    def prepare_text_input(text):
        """
        Prepare text input for the payload
        Args:
            text: List of comma-separated text strings, e.g. ["cat,dog", "person,car"]
        Returns:
            list: List of text arrays split by comma, or None if text is None

        Example:
            >>> prepare_text_input(["cat,dog", "person,car"])
            [["cat", "dog"], ["person", "car"]]

            >>> prepare_text_input(None)
            None
        """
        return [t.split(",") for t in text] if text else None

    @staticmethod
    def prepare_text_input_from_file(txt_file_path):
        """
        Prepare text input from a .txt file
        Args:
            txt_file_path: Path to .txt file containing comma-separated text
        Returns:
            list: List of text arrays split by comma, or None if file not found/empty

        Example:
            >>> prepare_text_input_from_file("labels.txt")
            # For a file containing "cat,car,truck"
            [["cat", "car", "truck"]]

            >>> prepare_text_input_from_file("nonexistent.txt")
            None
        """
        try:
            with open(txt_file_path, 'r') as f:
                content = f.read().strip()
                return PayloadBuilder.prepare_text_input([content]) if content else None
        except FileNotFoundError:
            print(f"Warning: Text file not found: {txt_file_path}")
            return None
        except Exception as e:
            print(f"Warning: Error reading text file: {e}")
            return None

    @staticmethod
    def prepare_model_name(model):
        """
        Prepare model name for the payload
        Args:
            model: Base model name, e.g. "nvdino-v2"
        Returns:
            str: Full model name with nvidia prefix

        Example:
            >>> prepare_model_name("nvdino-v2")
            "nvidia/nvdino-v2"
        """
        return f"nvidia/{model}"

    def __init__(self, files, model):
        """Initialize with required fields"""
        self.payload = {
            "input": self.prepare_image_inputs(files),
            "model": self.prepare_model_name(model)
        }

    def add_text(self, text):
        """Add optional text field if provided"""
        if text:
            text_input = self.prepare_text_input(text)
            if text_input:
                self.payload["text"] = text_input
        return self

    def add_more_field(self, key, value):
        """Add more field to the payload"""
        if key is not None and value is not None:
            self.payload[key] = value
        return self

    def build(self):
        """Return the constructed payload"""
        return self.payload

def get_resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    base_path = Path(__file__).absolute().parent.parent
    return str(Path(base_path, relative_path))

def copy_files(source_dir, destination_dir, filter: Callable[[str], bool]=None):
    # Ensure destination directory exists
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    # List all files in the source directory
    files = os.listdir(source_dir)

    for file in files:
        if filter and not filter(file):
            continue
        # Full path to the source file
        full_file_name = os.path.join(source_dir, file)

        # Check if it's a file (not a directory)
        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, destination_dir)

def create_tar_gz(output_filename, source_dir):
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=".")