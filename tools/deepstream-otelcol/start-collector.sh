#!/bin/bash
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

OTELCOL="${HOME}/.local/bin/otelcol"
CONFIG="$(dirname "$0")/otel-collector-config.yaml"

if [[ ! -f "$OTELCOL" ]]; then
    echo "Error: OpenTelemetry Collector not found at $OTELCOL"
    echo "Please run ./setup-collector.sh first"
    exit 1
fi

if [[ ! -f "$CONFIG" ]]; then
    echo "Error: Config file not found at $CONFIG"
    exit 1
fi

echo "Starting OpenTelemetry Collector..."
echo "Config: $CONFIG"
echo "Press Ctrl+C to stop"
echo ""

exec "$OTELCOL" --config="$CONFIG"

