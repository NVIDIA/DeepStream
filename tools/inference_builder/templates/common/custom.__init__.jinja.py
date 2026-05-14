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

import importlib
from pathlib import Path
from lib.utils import get_logger

logger = get_logger(__name__)

custom_module_table = {
    {% for item in classes %}
    "{{ item.name }}" : {
        "module": "{{ item.module }}",
        "class": "{{ item.class_name }}"
    }{% if not loop.last %},{% endif %}
    {% endfor %}
}

def create_instance(name:str, config):
    if not name in custom_module_table:
        logger.error(f"{name} not found")
        return None
    module_name = custom_module_table[name]["module"]
    class_name = custom_module_table[name]["class"]
    try:
        # Dynamically import the module
        module = importlib.import_module(f".{module_name}", __package__)

        # Get the class from the module
        cls = getattr(module, class_name)

        # Instantiate and return the class object
        return cls(config)
    except Exception as e:
        logger.error(f"Error creating class '{class_name}' from module '{module_name}': {e}")
        return None