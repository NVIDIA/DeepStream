#
################################################################################
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
################################################################################
#

from pyservicemaker import Pipeline, Probe, BatchMetadataOperator, CommonFactory, signal
from multiprocessing import Process
import os
import sys
import platform
import argparse

PIPELINE_NAME = "deepstream-sr-test"

KAFKA_PROTO_LIB_PATH = "/opt/nvidia/deepstream/deepstream/lib/libnvds_kafka_proto.so"
KAFKA_CONFIG_FILE = "/opt/nvidia/deepstream/deepstream/sources/libs/kafka_protocol_adaptor/cfg_kafka.txt"
MSGCONV_CONFIG_FILE = "/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-test5/configs/dstest5_msgconv_sample_config.txt"
KAFKA_CONN_STR = "localhost;9092"
KAFKA_TOPIC_LIST = "sr-test"

CONFIG_FILE_PATH = "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_infer_primary.yml"

BATCHED_PUSH_TIMEOUT = 33000
MUXER_WIDTH = 1920
MUXER_HEIGHT = 1080
TILER_WIDTH = 1280
TILER_HEIGHT = 720

class LocalRecorder(BatchMetadataOperator):
    def __init__(self, pipeline):
        super().__init__()
        self.pipeline = pipeline
        self.frame_count = 0
        self.session_id = None

    def handle_metadata(self, batch_meta):
        for frame_meta in batch_meta.frame_items:
            self.frame_count += 1
            if self.frame_count == 30:
                self.session_id = self.pipeline.start_recording("src_0", 0, 20, lambda info: print(f"Source recorded at: {info.file_directory}/{info.file_name}. Duration {info.duration / 1000.0} sec"))
            elif self.frame_count == 1000:
                self.pipeline.stop_recording_by_session_id(self.session_id)

def main(file_path, recording_type="local"):
    file_list = file_path if isinstance(file_path, list) else [file_path]
    num_sources = len(file_list)

    pipeline = Pipeline(PIPELINE_NAME)
    pipeline.add("nvstreammux", "mux", {"batch-size": num_sources, "batched-push-timeout": BATCHED_PUSH_TIMEOUT, "width": MUXER_WIDTH, "height": MUXER_HEIGHT})

    if recording_type not in ("local", "kafka"):
        raise ValueError("Invalid recording type. Expected 'local' or 'kafka'.")

    elif recording_type == "kafka":
        sr_controller = CommonFactory.create("smart_recording_action", "sr_controller")
        if isinstance(sr_controller, signal.Emitter):
            sr_controller_properties = {
                    "proto-lib": KAFKA_PROTO_LIB_PATH,
                    "conn-str": KAFKA_CONN_STR,
                    "msgconv-config-file": MSGCONV_CONFIG_FILE,
                    "proto-config-file": KAFKA_CONFIG_FILE,
                    "topic-list": KAFKA_TOPIC_LIST
            }
            sr_controller.set(sr_controller_properties)

        for i, file in enumerate(file_list):
            source_name = f"src_{i}"
            pipeline.add("nvurisrcbin", source_name, {"uri": file, "smart-record": 1, "smart-rec-cache": 20, "smart-rec-container": 0, "smart-rec-dir-path": ".", "smart-rec-mode": 0})
            pipeline.link((source_name, "mux"), ("", "sink_%u"))
            sr_controller.attach("start-sr", pipeline[source_name])
            sr_controller.attach("stop-sr", pipeline[source_name])
            pipeline.attach(source_name, "smart_recording_signal", "sr", "sr-done")
    
    else:
        for i, file in enumerate(file_list):
            source_name = f"src_{i}"
            pipeline.add("nvurisrcbin", source_name, {"uri": file, "smart-record": 2, "smart-rec-cache": 20, "smart-rec-container": 0, "smart-rec-dir-path": ".", "smart-rec-mode": 0})
            pipeline.link((source_name, "mux"), ("", "sink_%u"))

    pipeline.add("nvinfer", "infer", {"config-file-path": CONFIG_FILE_PATH, "batch-size": len(file_list)})
    pipeline.add("nvmultistreamtiler", "tiler", {"width": TILER_WIDTH, "height": TILER_HEIGHT})
    pipeline.add("nvosdbin", "osd").add("nv3dsink" if platform.processor() == "aarch64" else "nveglglessink", "sink", {"sync": False})
    pipeline.link("mux", "infer", "tiler", "osd", "sink")

    if recording_type == "local":
        pipeline.attach("infer", Probe("local_recorder", LocalRecorder(pipeline)))

    pipeline.start().wait()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]))
    parser.add_argument("uris", nargs="+", help="Input URI(s)")
    parser.add_argument(
      "-r",
      "--recording",
      choices=["local", "kafka"],
      default="local",
      help="Recording type: 'local' or 'kafka' (default: local)"
    )

    args = parser.parse_args()

    # pipeline.wait() in the main function is a blocking call due to which the KeyboardInterrupt may not be processed immediately.
    # we use Process from multiprocessing which runs the main function in a different process and processes KeyboardInterrupt immediately.
    process = Process(target=main, args=(args.uris, args.recording,))
    try:
        process.start()
        process.join()
    except KeyboardInterrupt:
        print("\nCtrl+C detected. Terminating process...")
        process.terminate()