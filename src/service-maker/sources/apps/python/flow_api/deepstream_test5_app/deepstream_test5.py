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

from pyservicemaker import Pipeline, Flow, BatchMetadataOperator, Probe, RecordConfig, RenderMode
from multiprocessing import Process
import sys
import os

# ===== Configuration Constants =====

# Primary GIE (GPU Inference Engine) Configuration
# Used for initial object detection in the video stream
PGIE_CONFIG_FILE_PATH = "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_infer_primary.yml"
PGIE_MODEL_ENGINE_FILE_PATH = "/opt/nvidia/deepstream/deepstream/samples/models/Primary_Detector/resnet18_trafficcamnet_pruned.onnx_b16_gpu0_fp16.engine"
PGIE_BATCH_SIZE = 16  # Number of frames processed in parallel for primary inference
PGIE_UNIQUE_ID = 1    # Unique identifier for the primary inference engine

# Tracker Configuration
# Used for maintaining object identity across frames
TRACKER_LL_CONFIG_FILE = "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_NvDCF_perf.yml"
TRACKER_LL_LIB_FILE = "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so"

# Secondary GIE Configurations
# SGIE1: Classifies detected vehicles into different types (car, truck, bus, etc.)
SGIE1_CONFIG_FILE_PATH = "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_infer_secondary_vehicletypes.yml"
SGIE1_MODEL_ENGINE_FILE_PATH = "/opt/nvidia/deepstream/deepstream/samples/models/Secondary_VehicleTypes/resnet18_vehicletypenet_pruned.onnx_b40_gpu0_fp16.engine"
SGIE1_BATCH_SIZE = 40  # Batch size for vehicle type classification
SGIE1_UNIQUE_ID = 4    # Unique identifier for vehicle type classification

# SGIE2: Classifies vehicles by their make and model
SGIE2_CONFIG_FILE_PATH = "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_infer_secondary_vehiclemake.yml"
SGIE2_MODEL_ENGINE_FILE_PATH = "/opt/nvidia/deepstream/deepstream/samples/models/Secondary_VehicleMake/resnet18_vehiclemakenet_pruned.onnx_b40_gpu0_fp16.engine"
SGIE2_BATCH_SIZE = 40  # Batch size for make/model classification
SGIE2_UNIQUE_ID = 6    # Unique identifier for make/model classification

# Kafka Protocol Configuration
# Used for streaming processed data to external systems
PROTO_LIB_PATH = "/opt/nvidia/deepstream/deepstream/lib/libnvds_kafka_proto.so"
PROTO_CONFIG_FILE = "/opt/nvidia/deepstream/deepstream/sources/libs/kafka_protocol_adaptor/cfg_kafka.txt"

# Message Broker Configuration
# Settings for Kafka message broker connection and topic
MSGCONV_CONFIG_FILE = "/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-test5/configs/dstest5_msgconv_sample_config.txt"
MSGBROKER_CONN_STR = "localhost;9092"  # Kafka broker connection string
MGSBROKER_TOPIC = "test5app"           # Kafka topic for message publishing

# Smart Record Configuration
# Used for intelligent video recording based on events
SMART_RECORD_TOPIC_LIST = "sr-test"

# Display Configuration
# Resolution settings for the output video stream
TILER_WIDTH = 1920
TILER_HEIGHT = 1080

# enable this to stream the output video to an RTSP server
ENABLE_RTSP_STREAMING = False
# RTSP server configuration
RTSP_MOUNT_POINT = "/ds-test5"
RTSP_PORT = 8554



def deepstream_test5_app(source_config_path):
    """
    Main function to create and run the DeepStream pipeline.
    This function sets up a complete video analytics pipeline with:
    - Object detection
    - Object tracking
    - Vehicle classification
    - Smart recording
    - Message publishing
    - Video rendering
    
    Args:
        source_config_path: Path to the source configuration file containing input video settings
    """
    # Initialize the DeepStream pipeline
    pipeline = Pipeline("deepstream-test5")

    # Configure smart recording with event-based video capture (cloud recording)
    record_config = RecordConfig(
        recording_type="cloud",
        proto_lib=PROTO_LIB_PATH,
        conn_str=MSGBROKER_CONN_STR,
        msgconv_config_file=MSGCONV_CONFIG_FILE,
        proto_config_file=PROTO_CONFIG_FILE,
        topic_list=SMART_RECORD_TOPIC_LIST,
        rec_cache=20,          # Number of frames to cache for smart recording
        rec_container=0,       # Container format (0 = MP4)
        rec_dir_path=".",      # Output directory for recordings
        rec_mode=0             # Recording mode (0 = event-based)
    )

    # Create pipeline flow and configure input with record config
    flow = Flow(pipeline).batch_capture(source_config_path, record_config=record_config)

    # Add primary inference for object detection
    # This is the first stage that detects objects in the video stream
    flow = flow.infer(PGIE_CONFIG_FILE_PATH, unique_id=PGIE_UNIQUE_ID, batch_size=PGIE_BATCH_SIZE, model_engine_file=PGIE_MODEL_ENGINE_FILE_PATH)

    # Add OSD (On-Screen Display) for object counting
    # Displays count of detected objects on the video
    flow = flow.attach(
            what="sample_video_probe",
            name="osd_counter",
            properties={
                "font-size": 15
            }
        )

    # Add object tracking to maintain object identity across frames
    flow = flow.track(ll_config_file=TRACKER_LL_CONFIG_FILE, ll_lib_file=TRACKER_LL_LIB_FILE)

    # Add secondary inference for vehicle type classification
    # Classifies detected vehicles into categories (car, truck, bus, etc.)
    flow = flow.infer(config=SGIE1_CONFIG_FILE_PATH, unique_id=SGIE1_UNIQUE_ID, batch_size=SGIE1_BATCH_SIZE, model_engine_file=SGIE1_MODEL_ENGINE_FILE_PATH)
    
    # Add secondary inference for vehicle make/model classification
    # Identifies the specific make and model of detected vehicles
    flow = flow.infer(config=SGIE2_CONFIG_FILE_PATH, unique_id=SGIE2_UNIQUE_ID, batch_size=SGIE2_BATCH_SIZE, model_engine_file=SGIE2_MODEL_ENGINE_FILE_PATH)

    # Add various monitoring and data collection probes
    # FPS measurement probe - monitors pipeline performance
    flow = flow.attach(
            what="measure_fps_probe",
            name="fps_probe"
        )

    # Message metadata generation probe - prepares data for Kafka publishing
    flow = flow.attach(
            what="add_message_meta_probe",
            name="message_generator"
        )

    # KITTI format data dump probe - saves detection data in KITTI format
    flow = flow.attach(
            what="kitti_dump_probe",
            name="kitti_dump"
        )

    # Latency measurement probe - monitors processing delays
    flow = flow.attach(
            what="measure_latency_probe",
            name="latency_probe"
        )
        
    # Fork the pipeline for parallel processing
    # This allows simultaneous video rendering and message publishing
    flow = flow.fork()

    # Configure message publishing to Kafka
    # Sends processed data to external systems
    flow.publish(
            msg_broker_proto_lib=PROTO_LIB_PATH,
            msg_broker_conn_str=MSGBROKER_CONN_STR,
            topic=MGSBROKER_TOPIC,
            msg_conv_config=MSGCONV_CONFIG_FILE,
            sync=False,  # Asynchronous publishing for better performance
        )

    # Add video rendering component
    # Displays the processed video stream with annotations
    if ENABLE_RTSP_STREAMING:
        flow.render(mode=RenderMode.STREAM, rtsp_mount_point=RTSP_MOUNT_POINT, rtsp_port=RTSP_PORT, width=TILER_WIDTH, height=TILER_HEIGHT, sync=False)
    else:
        flow.render(mode=RenderMode.DISPLAY, width=TILER_WIDTH, height=TILER_HEIGHT, sync=False)

    # Execute the pipeline
    flow()

if __name__ == '__main__':
    # Validate command line arguments
    if len(sys.argv) != 2:
        sys.stderr.write("usage: %s <source_config.yaml>\n" % sys.argv[0])
        sys.exit(1)
    
    # Validate input file existence and format
    source_config_path = sys.argv[1]
    if not os.path.exists(source_config_path):
        sys.stderr.write(f"Error: File '{source_config_path}' does not exist\n")
        sys.exit(1)
    
    if not source_config_path.lower().endswith(('.yaml', '.yml')):
        sys.stderr.write(f"Error: Input file '{source_config_path}' must be a YAML file (.yaml or .yml)\n")
        sys.exit(1)

    # Use multiprocessing for proper handling of pipeline execution
    # This ensures clean termination on keyboard interrupt
    process = Process(target=deepstream_test5_app, args=(sys.argv[1],))
    try:
        process.start()
        process.join()
    except KeyboardInterrupt:
        print("\nCtrl+C detected. Terminating process...")
        process.terminate()