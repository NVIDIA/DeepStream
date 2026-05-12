#
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#

from pyservicemaker import Pipeline, Flow, BatchMetadataOperator, Probe, RecordConfig
from multiprocessing import Process
import sys
import os
import argparse

CONFIG_FILE_PATH = "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_infer_primary.yml"

KAFKA_PROTO_LIB_PATH = "/opt/nvidia/deepstream/deepstream/lib/libnvds_kafka_proto.so"
KAFKA_CONFIG_FILE = "/opt/nvidia/deepstream/deepstream/sources/libs/kafka_protocol_adaptor/cfg_kafka.txt"
MSGCONV_CONFIG_FILE = "/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-test5/configs/dstest5_msgconv_sample_config.txt"
KAFKA_CONN_STR = "localhost;9092"
KAFKA_TOPIC_LIST = "sr-test"

MUXER_WIDTH = 1920
MUXER_HEIGHT = 1080

class LocalRecorder(BatchMetadataOperator):
    def __init__(self, flow):
        super().__init__()
        self.flow = flow
        self.frame_count = 0

    def handle_metadata(self, batch_meta):
        for frame_meta in batch_meta.frame_items:
            self.frame_count += 1
            if self.frame_count == 30:
                self.flow.pipeline.start_recording("UniqueSensorId1", 0, 20, lambda info: print(f"Source recorded at: {info.file_directory}/{info.file_name}. Duration {info.duration / 1000.0} sec"))
            elif self.frame_count == 1000:
                self.flow.pipeline.stop_recording("UniqueSensorId1")


def main(input_arg, recording_type="local"):
    pipeline = Pipeline("deepstream-sr-test")
    flow = Flow(pipeline)

    if recording_type not in ("local", "cloud"):
        raise ValueError("Invalid recording type. Expected 'local' or 'cloud'.")

    elif recording_type == "cloud":
        record_config = RecordConfig(
            recording_type="cloud",
            proto_lib=KAFKA_PROTO_LIB_PATH,
            conn_str=KAFKA_CONN_STR,
            msgconv_config_file=MSGCONV_CONFIG_FILE,
            proto_config_file=KAFKA_CONFIG_FILE,
            topic_list=KAFKA_TOPIC_LIST,
            rec_cache=20,
            rec_container=0,
            rec_dir_path=".",
            rec_mode=0
        )
        uri_list = input_arg if isinstance(input_arg, list) else [input_arg]
        flow = flow.batch_capture(
            uri_list,
            record_config=record_config,
            width=MUXER_WIDTH,
            height=MUXER_HEIGHT,
        )
    
    else:
        record_config = RecordConfig(
            recording_type="local",
            rec_cache=20,
            rec_container=0,
            rec_dir_path=".",
            rec_mode=0
        )
        # local mode expects a source config YAML path
        source_config_path = input_arg if isinstance(input_arg, str) else input_arg[0]
        flow = flow.batch_capture(source_config_path, record_config=record_config, width=MUXER_WIDTH, height=MUXER_HEIGHT)

    flow = flow.infer(CONFIG_FILE_PATH)

    if recording_type == "local":
        flow = flow.attach(what=Probe("local_recorder", LocalRecorder(flow)))
    
    flow.render(sync=False)()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]))
    parser.add_argument(
      "-r",
      "--recording",
      choices=["local", "cloud"],
      default="local",
      help="Recording type: 'local' (uses --source-config) or 'cloud' (uses URIs)"
    )
    parser.add_argument(
      "--source-config",
      help="Path to source config YAML (required for local recording mode)"
    )
    parser.add_argument(
      "uris",
      nargs="*",
      help="Input URI(s) (required for cloud recording mode)"
    )

    args = parser.parse_args()

    # Validate mode-specific inputs
    if args.recording == "local":
        if not args.source_config:
            parser.error("Only --source-config is required when --recording=local")
        elif args.uris:
            parser.error("URI(s) are not allowed when --recording=local")
        input_arg = args.source_config
    else:  # cloud
        if not args.uris:
            parser.error("At least one URI is required when --recording=cloud")
        elif args.source_config:
            parser.error("Source config is not allowed when --recording=cloud")
        input_arg = args.uris

    # pipeline.wait() in the main function is a blocking call due to which the KeyboardInterrupt may not be processed immediately.
    # we use Process from multiprocessing which runs the main function in a different process and processes KeyboardInterrupt immediately.
    process = Process(target=main, args=(input_arg, args.recording,))
    try:
        process.start()
        process.join()
    except KeyboardInterrupt:
        print("\nCtrl+C detected. Terminating process...")
        process.terminate()