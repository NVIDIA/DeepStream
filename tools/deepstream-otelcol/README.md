<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
-->

# OpenTelemetry Collector Setup

This directory contains scripts and configuration to set up and run the OpenTelemetry Collector for collecting metrics from streaming applications.

## Prerequisites

- Linux system (x86_64 or aarch64/arm64)
- `wget` installed
- Network access to GitHub releases

## Quick Start

### 1. Install the Collector

Run the setup script to download and install the OpenTelemetry Collector binary:

```bash
chmod +x setup-collector.sh
./setup-collector.sh
```

This script will:
- Automatically detect your system architecture (x86_64 or aarch64)
- Download the appropriate OpenTelemetry Collector binary (v0.91.0)
- Install it to `~/.local/bin/otelcol`

> **Note:** `~/.local/bin` may not be on your `PATH`. The `start-collector.sh` script uses the absolute path so it works regardless, but if you want to run `otelcol` directly, add `~/.local/bin` to your `PATH`.

### 2. Start the Collector

```bash
chmod +x start-collector.sh
./start-collector.sh
```

The collector will start and listen on:
- **gRPC**: `0.0.0.0:4317` - For OTLP gRPC connections
- **HTTP**: `0.0.0.0:4318` - For OTLP HTTP connections
- **Prometheus**: `0.0.0.0:8889` - Prometheus metrics endpoint

Press `Ctrl+C` to stop the collector.

---

## Configuration Explained

The collector is configured via `otel-collector-config.yaml`. Here's a breakdown of each component:

### Receivers

```yaml
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317
      http:
        endpoint: 0.0.0.0:4318
```

The OTLP receiver accepts telemetry data via both gRPC and HTTP protocols. This is the standard OpenTelemetry protocol, allowing any OTLP-compatible application to send metrics, traces, or logs.

### Processors

#### Batch Processor

```yaml
processors:
  batch:
```

The batch processor groups telemetry data before sending to exporters. This improves performance by reducing the number of outgoing connections and allowing better compression.

#### Filter Processor (Drop Inactive Streams)

```yaml
filter/drop_inactive_streams:
  error_mode: ignore
  metrics:
    datapoint:
      - 'metric.name == "stream_fps" and value_double == -1.0'
      - 'metric.name == "stream_latency" and value_double == -1.0'
      - 'metric.name == "stream_frame_number" and value_int == -1'
```

**Why this filter?**

When streams are inactive or disconnected, the application reports `-1` as a sentinel value for metrics like FPS, latency, and frame number. These values are not meaningful for monitoring and would pollute dashboards and alerting systems.

This filter:
- **Drops `stream_fps` datapoints** when the value is `-1.0` (stream not producing frames)
- **Drops `stream_latency` datapoints** when the value is `-1.0` (no latency measurement available)
- **Drops `stream_frame_number` datapoints** when the value is `-1` (no frames being processed)

The `error_mode: ignore` setting ensures that if the filter encounters any evaluation errors, it silently continues rather than failing the pipeline.

### Exporters

```yaml
exporters:
  debug:
    verbosity: detailed

  prometheus:
    endpoint: "0.0.0.0:8889"
    metric_expiration: 10s
```

- **debug**: Outputs detailed telemetry data to stdout (useful for troubleshooting)
- **prometheus**: Exposes metrics in Prometheus format at port 8889
  - `metric_expiration: 10s` - Metrics are removed from the endpoint if not updated within 10 seconds (keeps the endpoint clean when streams stop)

### Pipeline

```yaml
service:
  pipelines:
    metrics:
      receivers: [otlp]
      processors: [batch, filter/drop_inactive_streams]
      exporters: [debug, prometheus]
```

The metrics pipeline:
1. **Receives** data via OTLP protocol
2. **Processes** it through batching and filtering
3. **Exports** to the debug console and Prometheus

---

## Viewing Metrics

### Prometheus Endpoint

Access the Prometheus metrics at:
```text
http://localhost:8889/metrics
```

### Scraping with Prometheus

Add this to your Prometheus config:

```yaml
scrape_configs:
  - job_name: 'otel-collector'
    static_configs:
      - targets: ['localhost:8889']
    scrape_interval: 5s
```

---

## Troubleshooting

### Collector not found
If you see "OpenTelemetry Collector not found", run the setup script first:
```bash
./setup-collector.sh
```

### Port already in use
If ports 4317, 4318, or 8889 are in use, modify the endpoints in `otel-collector-config.yaml`.

### Check collector logs
The collector outputs to stdout by default. Check the terminal output for any error messages.

---

## File Structure

```text
deepstream-otelcol/
├── README.md                    # This file
├── setup-collector.sh           # Downloads and installs the collector
├── start-collector.sh           # Starts the collector with config
└── otel-collector-config.yaml   # Collector configuration
```
