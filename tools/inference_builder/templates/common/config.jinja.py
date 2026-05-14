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

import os
import re
from omegaconf import OmegaConf


def _resolve_env_vars(config_dict):
    """
    Recursively resolve environment variable references in the config.
    
    Supports the format: $ENV_VAR|default_value
    If ENV_VAR exists, uses its value. Otherwise, uses default_value.
    
    Type conversion is performed based on the default value:
    - "true"/"false" (case-insensitive) -> bool
    - numeric strings -> int or float
    - everything else -> string
    """
    env_pattern = re.compile(r'^\$([A-Za-z_][A-Za-z0-9_]*)\|(.*)$')
    
    def convert_value(value_str):
        """Convert string value to appropriate type."""
        # Handle boolean strings
        if value_str.lower() in ('true', 'false'):
            return value_str.lower() == 'true'
        
        # Try to convert to int
        try:
            if '.' not in value_str:
                return int(value_str)
        except ValueError:
            pass
        
        # Try to convert to float
        try:
            return float(value_str)
        except ValueError:
            pass
        
        # Return as string
        return value_str
    
    def process_value(value):
        """Process a single value for environment variable substitution."""
        if isinstance(value, str):
            match = env_pattern.match(value)
            if match:
                env_var = match.group(1)
                default_value = match.group(2)
                
                # Check if environment variable exists
                if env_var in os.environ:
                    return convert_value(os.environ[env_var])
                else:
                    return convert_value(default_value)
        elif isinstance(value, dict):
            return {k: process_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [process_value(item) for item in value]
        
        return value
    
    return process_value(config_dict)


# Load the base configuration
_base_config = OmegaConf.create("""{{ config }}""")

# Convert to dict, resolve environment variables, and convert back to OmegaConf
_config_dict = OmegaConf.to_container(_base_config)
_resolved_config = _resolve_env_vars(_config_dict)
global_config = OmegaConf.create(_resolved_config)