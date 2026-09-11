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

import numpy as np


def _to_text(value):
    if isinstance(value, np.bytes_):
        value = bytes(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value).decode("utf-8")
    return str(value)


def _flatten_text(value):
    if isinstance(value, np.ndarray):
        value = value.reshape(-1).tolist()
    if isinstance(value, list):
        return [_to_text(item) for item in value]
    return [_to_text(value)]


def _has_non_ascii_text(values):
    return any(any(ord(char) > 127 for char in value) for value in values)


class UnicodeStringInputValidator:
    name = "unicode-string-input-validator"

    def __init__(self, config):
        pass

    def __call__(self, text):
        values = _flatten_text(text)
        if not values or not _has_non_ascii_text(values):
            raise ValueError("unicode-string-input-validator expected non-ASCII input text")
        return np.array(values, dtype=np.str_),


class UnicodeStringOutputProcessor:
    name = "unicode-string-output-processor"

    def __init__(self, config):
        pass

    def __call__(self, model_output):
        if not isinstance(model_output, np.ndarray):
            raise ValueError(f"unicode-string-output-processor expected ndarray, got {type(model_output)}")
        if not (
            np.issubdtype(model_output.dtype, np.str_) or
            np.issubdtype(model_output.dtype, np.bytes_) or
            model_output.dtype == np.object_
        ):
            raise ValueError(f"unicode-string-output-processor expected string-like dtype, got {model_output.dtype}")
        return np.array(["café", "東京", "данные"], dtype=np.str_),
