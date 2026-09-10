# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from . import _pydeepstream


def _build_param_spec(params):
    """Convert a params dict to the YAML param spec string expected by CustomObject.

    params format:
        {"font-size": ("integer", 12)}
        {"threshold": ("float", 0.5, "confidence threshold")}
    Supported types: "integer", "string", "boolean".
    """
    if not params:
        return ""
    entries = []
    for name, spec in params.items():
        type_str, default = spec[0], spec[1]
        brief = spec[2] if len(spec) > 2 else name
        if type_str in ("string", "path"):
            default_str = f'"{default}"'
        elif isinstance(default, bool):
            default_str = str(default).lower()
        else:
            default_str = str(default)
        entries.append(
            f"{{name: {name}, type: {type_str}, brief: {brief}, "
            f"description: {brief}, default_value: {default_str}}}"
        )
    return "[" + ", ".join(entries) + "]"


def register_probe(name, cls, params=None):
    """Register a Python BatchMetadataOperator subclass as a named probe plugin.

    Once registered, the probe can be used by name string exactly like a C++ plugin:
        pipeline.attach("infer", "my_probe", "instance1")
        pipeline["instance1"].set({"font-size": 24})

        flow.attach(what="my_probe", name="instance1")

    Args:
        name:   Plugin name used to look up the probe by string.
        cls:    Python class subclassing BatchMetadataOperator.
        params: Optional dict of parameter specs:
                {"param-name": ("type", default_value)} or
                {"param-name": ("type", default_value, "brief description")}
    """
    _pydeepstream._register_probe(name, cls, _build_param_spec(params))
    return cls


def probe(name, params=None):
    """Decorator to register a Python class as a named probe plugin.

    Equivalent to calling register_probe() immediately after the class definition.

    Usage:
        @probe(name="count_vehicles", params={"font-size": ("integer", 12)})
        class CountVehicles(BatchMetadataOperator):
            def handle_metadata(self, batch_meta):
                ...

        # Attach by name string, identical to a C++ plugin:
        pipeline.attach("infer", "count_vehicles", "cv1")
        pipeline["cv1"].set({"font-size": 24})
    """
    def decorator(cls):
        return register_probe(name, cls, params)
    return decorator
