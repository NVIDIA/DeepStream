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

class RequestCounter:
    name = "request-counter"

    def __init__(self, config):
        self.config = config
        self.num_requests = 0

    def __call__(self, *args, **kwargs):
        if len(args) != 1:
            raise ValueError(
                "request-counter expects exactly one argument"
            )
        self.num_requests += 1
        return args[0], self.num_requests


class ResponseChecker:
    name = "response-checker"

    def __init__(self, config):
        self.config = config
        self.last_id = -1

    def __call__(self, *args, **kwargs):
        if self.last_id < 0:
            self.last_id = args[1]
        elif args[1] != self.last_id + 1:
            raise ValueError(
                f"response-checker expects request id to be {self.last_id + 1} vs {args[1]}"
            )
        self.last_id = args[1]
        return args[0]
