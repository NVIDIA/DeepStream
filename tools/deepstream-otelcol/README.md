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
- **gRPC**: `127.0.0.1:4317` - For OTLP gRPC connections
- **HTTP**: `127.0.0.1:4318` - For OTLP HTTP connections
- **Prometheus**: `127.0.0.1:8889` - Prometheus metrics endpoint

Press `Ctrl+C` to stop the collector.

---

## Using with DeepStream

DeepStream OpenTelemetry support is built around three components:

1. **Gst-nvmultiurisrcbin** — hosts the REST server and wires up metric export when telemetry is enabled.
2. **Gst-nvdslogger** — collects per-stream FPS, latency, and frame-number data from the pipeline.
3. **nvds_rest_metrics** — shares that data with the OpenTelemetry client, which sends OTLP/HTTP metrics to this collector.

### Step 1: Start the collector

From this directory, install (once) and start the collector as shown in [Quick Start](#quick-start). The collector listens on `http://127.0.0.1:4318` for OTLP/HTTP, which matches DeepStream's default export endpoint.

Keep the collector running in a separate terminal while the DeepStream application is active.

### Step 2: Configure the DeepStream application

OpenTelemetry is available only when **nvmultiurisrcbin** is in the pipeline. The shipped **deepstream-test5** sample demonstrates this.

In your DeepStream config file (for example `src/apps/sample_apps/deepstream-test5/configs/test5_config_file_nvmultiurisrcbin_src_list_attr_all.txt`):

```ini
[tiled-display]
enable=3

[source-list]
use-nvmultiurisrcbin=1

[sink0]
nvdslogger=1
```

> **Note:** The sample config ships with `[tiled-display] enable=1` and `[sink0] nvdslogger=0`. Set `enable=3` and `nvdslogger=1` before running with OpenTelemetry.

### Step 3: Set OpenTelemetry environment variables

Export these variables in the same shell where you launch DeepStream:

```bash
export OTEL_SDK_DISABLED=false
export OTEL_SERVICE_NAME=deepstream-test5
export OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:4318
export OTEL_METRICS_EXPORTER=otlp
export OTEL_METRIC_EXPORT_INTERVAL=5000
```

| Environment variable | Description |
|---------------------|-------------|
| `OTEL_SDK_DISABLED` | Set to `false` to enable telemetry export |
| `OTEL_SERVICE_NAME` | Service identifier shown in observability backends (e.g. `deepstream-test5`) |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | Collector base URL — must match this collector's HTTP endpoint (`http://127.0.0.1:4318`) |
| `OTEL_METRICS_EXPORTER` | `otlp` (default), `console`, or `none` |
| `OTEL_METRIC_EXPORT_INTERVAL` | Export interval in milliseconds (default: `60000`) |

If the collector runs on another host, set `OTEL_EXPORTER_OTLP_ENDPOINT` to that host (for example `http://otel-collector:4318`).

### Step 4: Run a DeepStream sample application

Run **deepstream-test5** from its source directory (config paths are relative to the config file location):

```bash
cd src/apps/sample_apps/deepstream-test5

deepstream-test5-app \
  -c configs/test5_config_file_nvmultiurisrcbin_src_list_attr_all.txt
```

After startup, look for OpenTelemetry log lines such as `[OpenTelemetry] OTLP URL: http://127.0.0.1:4318/v1/metrics` in the application output.

### Step 5: Verify metrics

With the pipeline running and the collector started:

1. **Collector debug output** — the `debug` exporter prints received metrics to the collector terminal.
2. **Prometheus endpoint** — scrape `http://127.0.0.1:8889/metrics` for exported DeepStream metrics.

### Supported metrics

DeepStream exports the following metrics (see the [official metrics table](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_OpenTelemetry.html#supported-prometheus-metrics) for typical values):

| Metric | Description |
|--------|-------------|
| `stream_fps` | Frames per second per stream |
| `stream_latency` | End-to-end pipeline latency per stream (milliseconds) |
| `stream_frame_number` | Current frame number per stream |
| `stream_count` | Number of active streams |
| `cpu_utilization` | CPU utilization (%) — via `metric_type` attribute on system gauge |
| `gpu_utilization` | GPU compute utilization (%) |
| `ram_memory_gb` | System RAM usage (GB) |
| `gpu_memory_gb` | GPU memory usage (GB); returns `-1` on aarch64 unified-memory devices |

Inactive streams report `-1` for FPS, latency, and frame number for one export cycle. The `filter/drop_inactive_streams` processor in `otel-collector-config.yaml` removes those sentinel values before they reach Prometheus.

When tuning `OTEL_METRIC_EXPORT_INTERVAL`, set the Prometheus exporter's `metric_expiration` in `otel-collector-config.yaml` to at least the export interval so stale stream metrics are dropped cleanly.

---

## Configuration Explained

The collector is configured via `otel-collector-config.yaml`. Here's a breakdown of each component:

### Receivers

```yaml
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 127.0.0.1:4317
      http:
        endpoint: 127.0.0.1:4318
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
    endpoint: "127.0.0.1:8889"
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
http://127.0.0.1:8889/metrics
```

### Scraping with Prometheus

Add this to your Prometheus config:

```yaml
scrape_configs:
  - job_name: 'otel-collector'
    static_configs:
      - targets: ['127.0.0.1:8889']
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
If ports 4317, 4318, or 8889 are in use, modify the endpoints in `otel-collector-config.yaml` and update `OTEL_EXPORTER_OTLP_ENDPOINT` in DeepStream accordingly.

### No metrics from DeepStream
- Confirm the collector is running before starting the DeepStream app.
- Verify `[tiled-display] enable=3`, `use-nvmultiurisrcbin=1`, and `nvdslogger=1` in the config file.
- Confirm `OTEL_SDK_DISABLED=false` and `OTEL_EXPORTER_OTLP_ENDPOINT` points to the collector HTTP port (`4318`).
- Check that the pipeline is processing frames (FPS/latency are only meaningful when streams are active).

### Check collector logs
The collector outputs to stdout by default. Check the terminal output for any error messages.

---

## References

- [DeepStream OpenTelemetry Support](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_OpenTelemetry.html) — configuration, metrics, and architecture
- [OpenTelemetry Collector documentation](https://opentelemetry.io/docs/collector/)
- [OTLP/HTTP specification](https://opentelemetry.io/docs/specs/otlp/#otlphttp)

---

## File Structure

```text
deepstream-otelcol/
├── README.md                    # This file
├── setup-collector.sh           # Downloads and installs the collector
├── start-collector.sh           # Starts the collector with config
└── otel-collector-config.yaml   # Collector configuration
```
