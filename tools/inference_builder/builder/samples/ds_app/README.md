# DeepStream Application Samples

This directory contains sample configurations demonstrating how to build DeepStream applications using Inference Builder.

While the samples support Ampere, Hopper, and Blackwell architectures, the model and the backend set the real hardware requirements.

**Platform Support:** All ds_app samples are compatible with both x86-64 systems and NVIDIA Jetson Thor device.

## Models used in the samples

All the models can be downloaded from NGC and certain models need active subscription.

If you don't have NGC CLI installed, please download and install it from [this page](https://org.ngc.nvidia.com/setup/installers/cli).

### Image Classification
- **PCB Classification**: [PCB Classification Model](https://catalog.ngc.nvidia.com/orgs/nvaie/models/pcbclassification)

### Visual Change Detection
- **Visual Changenet Classification**: [Visual Changenet Classification Model](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/visual_changenet_classification)

### Semantic Segmentation
- **CitySemSegFormer**: [CitySemSegFormer Model](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/citysemsegformer)

### Instance Segmentation
- **Mask2Fomer**: [Masked-attention Mask Transformer](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/mask2former)

### Grounding Dino
- **Grounding DINO**: [Grounding DINO Model](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/grounding_dino)
- **Mask Grounding DINO**: [Mask Grounding DINO Model](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/mask_grounding_dino)

### RT-DETR Detector
- **RT-DETR**: [TrafficCamNet Lite](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/trafficcamnet_transformer_lite)
- **RT-DETR**: [PeopleNet Transformer](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/peoplenet_transformer)

## Getting Started

Each sample includes its own README with specific instructions for building and running the DeepStream application. Please refer to the individual sample directories for detailed setup and usage instructions.

The provided samples can serve as references for creating inference pipelines across a wide range of TAO computer vision models. Each model falls into one of the following categories, with corresponding instructions available:

- classification
- detection
- segmentation
- GDINO (Open vocabulary detection and segmentation)

Required model files including the onnx, nvinfer configuration, preprocess configuration can be found from NGC model repository or exported through TAO Finetune Microservice.

## Configurable Parameters for the DeepStream Backend

You can enable these features under the [`parameters section of the configuration file`](#complete-configuration-example).

### Inference Configuration

The core inference configuration that specifies the model and inference parameters.

```yaml
parameters:
  infer_config_path:
    - nvdsinfer_config.yaml  # Path to the nvinfer configuration file
```

### Preprocess Configuration

Used to define preprocessing operations that run before inference. This is required when non-image tensors must be injected into the pipeline (e.g., for models that require additional metadata or custom tensor inputs). Refer to [nvdspreprocess plugin documentation](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvdspreprocess.html) for more details.

```yaml
parameters:
  preprocess_config_path:
    - nvdspreprocess_config.yaml  # Path to the nvdspreprocess configuration file
```

### Tracker Configuration

Used for object tracking along with inference using nvtracker. This enables multi-object tracking across frames.

```yaml
tracker_config:
  ll_lib_file: /opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so  # Path to the tracker library
  ll_config_file: /opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_IOU.yml  # Path to tracker configuration file
  width: 1920 # Tracker width
  height: 1088 # Tracker height
  display_tracking_id: true # Display tracking id in object text
```

### Analytics Configuration

Used for object activity analysis such as line-crossing, ROI enetering/exiting, crowding and direction detection.

```yaml
analytics_config:
  config_file_path: /opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-nvdsanalytics-test/config_nvdsanalytics.txt # Path to the nvdsanalytics configuration file
  gpu_id: 0 # GPU core id for analytics running
```


### Message Broker Configuration

Used to send inference data over the cloud using nvmsgbroker and nvmsgconverter. Supports various message brokers like Kafka, MQTT, etc.

> **Note:** To use message broker functionality, you need to set up a broker service first and then update the parameters below accordingly.

```yaml
msgbroker_config:
  msgbroker_proto_lib_path: /opt/nvidia/deepstream/deepstream/lib/libnvds_kafka_proto.so  # Path to the message broker protocol library
  msgbroker_conn_str: localhost:9092  # Message broker connection string (host:port)
  msgbroker_topic: ds_app  # Topic name for publishing inference results
  msgconv_config_path: /opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-test5/configs/dstest5_msgconv_sample_config.txt  # Path to message converter configuration
  msgconv_payload_type: 2  # Payload type for message conversion (0=DeepStream JSON, 1=DeepStream Minimal JSON, 2=DeepStream Protobuf)
  msgconv_msg2p_new_api: false  # Use new API which supports publishing multiple payloads using NvDsFrameMeta
  msgconv_frame_interval: 1  # Frame interval at which payload is generated (every Nth frame)
  msgconv_msg2p_lib: /opt/nvidia/deepstream/deepstream/lib/libnvds_msgconv.so  # Path to the message converter library
```

### Render Configuration

To visually see the inference output, you have two options:

#### 1. Display on Host Machine Terminal

```yaml
render_config:
  enable_display: true    # Enable display output on host machine
  enable_osd: true        # Enable on-screen display with bounding boxes and metadata using nvdsosd
```

#### 2. Stream to RTSP Port

```yaml
render_config:
  enable_stream: true     # Enable RTSP streaming output
  enable_osd: true        # Enable on-screen display with bounding boxes and metadata using nvdsosd
  rtsp_mount_point: /ds_app  # RTSP mount point for the stream
  rtsp_port: 8554         # RTSP server port number
```

This will start RTSP streaming at `rtsp://localhost:8554/ds_app`

### Performance Measurement Configuration

Used to monitor the performance of the inference pipeline and individual components.

```yaml
perf_config:
  enable_fps_logs: true      # Enable FPS logging for each source on the host machine terminal
  enable_latency_logs: true  # Enable element-specific latency logging on host machine terminal
```

>**Note:** For latency measurements to work properly, make sure to export these environment variables:

```bash
export NVDS_ENABLE_LATENCY_MEASUREMENT=1
export NVDS_ENABLE_COMPONENT_LATENCY_MEASUREMENT=1
```

### KITTI Data Configuration

When enabled, this configuration dumps inference and tracking data in KITTI format to the specified directories.

```yaml
kitti_output_path:
  infer: /workspace/models/infer-kitti-dump/    # Directory for inference KITTI data
  tracker: /workspace/models/tracker-kitti-dump/  # Directory for tracking KITTI data
```

> **Note:** Make sure to create the directories mentioned above before running the application to ensure proper KITTI data dumping.

### Resolution Configuration

When specified in the configuration, the input stream will be resized to the specified dimensions before processing.

```yaml
resize_video: [height, width]  # Format: [height, width]
resize_video: [1080, 1920]     # Example: Resize to 1080p (1920x1080)
```

### Timeout Configuration

Inference timeout ensures the inference process completes within a specified time limit. The timeout value is specified in seconds.

```yaml
parameters:
  inference_timeout: 5  # Timeout in seconds for inference operations
```

Batch timeout specifies the time in microseconds to wait for batched buffers to be sent out after the first buffer is available.

- **Default value:** `1000 * max_batch_size` microseconds
- **Set to -1:** Wait infinitely for batch formation

> **Warning:** If your `max_batch_size` in the models section is greater than the number of streams you're running, do not set the timeout to -1. This will cause the pipeline to wait infinitely for batch formation and become unresponsive. In such scenarios, set an appropriate timeout value.

```yaml
parameters:
  batch_timeout: 33000 # Timeout in microseconds for batching
```

### [Complete Configuration Example](#complete-configuration-example)

Here's a complete example showing all the features enabled:

```yaml
parameters:
  # Core inference configuration
  infer_config_path:
    - nvdsinfer_config.yaml  # Path to nvinfer configuration file

  # Resolution configuration
  resize_video: [1080, 1920]  # Resize input stream to 1080p (1920x1080)

  # Object tracking configuration
  tracker_config:
    ll_lib_file: /opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so  # Tracker library path
    ll_config_file: /opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_IOU.yml  # Tracker config path

  # Message broker configuration for cloud integration
  # Note: To use message broker functionality, you need to set up a broker service first and then update the parameters below accordingly.
  msgbroker_config:
    msgbroker_proto_lib_path: /opt/nvidia/deepstream/deepstream/lib/libnvds_kafka_proto.so  # Protocol library
    msgbroker_conn_str: localhost:9092  # Broker connection string
    msgbroker_topic: ds_app  # Topic for publishing results
    msgconv_config_path: /opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-test5/configs/dstest5_msgconv_sample_config.txt  # Message converter config
    msgconv_payload_type: 2  # Payload type for message conversion (0=DeepStream JSON, 1=DeepStream Minimal JSON, 2=DeepStream Protobuf)
    msgconv_msg2p_new_api: false  # Use new API which supports publishing multiple payloads using NvDsFrameMeta
    msgconv_frame_interval: 1  # Frame interval at which payload is generated (every Nth frame)
    msgconv_msg2p_lib: /opt/nvidia/deepstream/deepstream/lib/libnvds_msgconv.so  # Path to the message converter library

  # Rendering and display configuration
  # Choose your output method: enable_display (local display), enable_stream (RTSP streaming), or both
  render_config:
    enable_display: true     # Enable display output
    enable_osd: true         # Enable on-screen display with metadata
    enable_stream: true      # Enable RTSP streaming
    rtsp_mount_point: /ds_app  # RTSP mount point
    rtsp_port: 8554          # RTSP server port

  # Performance monitoring configuration
  perf_config:
    enable_fps_logs: true      # Enable FPS logging
    enable_latency_logs: true  # Enable latency logging

  # KITTI dump configuration
  kitti_output_path:
    infer: /workspace/models/infer-kitti-dump/    # Directory for inference KITTI data
    tracker: /workspace/models/tracker-kitti-dump/  # Directory for tracking KITTI data

  # Inference timeout configuration
  inference_timeout: 5  # Timeout in seconds for inference operations

  # Batch timeout configuration
  batch_timeout: 33000 # Timeout in microseconds to wait for batching operation
```
