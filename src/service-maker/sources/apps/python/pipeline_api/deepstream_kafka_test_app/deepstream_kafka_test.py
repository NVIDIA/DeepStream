#
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#

from pyservicemaker import Pipeline, Probe, BatchMetadataOperator
from kafka import KafkaProducer
import sys
import platform
import os
import json

PIPELINE_NAME = "deepstream_test_kafka"
PGIE_CONFIG_FILE_PATH = "/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-test1/dstest1_pgie_config.yml"
CONN_STR = "localhost:9092"
TOPIC = "test-kafka"

producer = KafkaProducer(bootstrap_servers=CONN_STR, value_serializer=lambda v: json.dumps(v).encode('utf-8'))

class SendCustomData(BatchMetadataOperator):
    def handle_metadata(self, batch_meta):
        for frame_meta in batch_meta.frame_items:
            objects = [
                {
                    "class_id": object.class_id,
                    "confidence": object.confidence,
                    "bbox": {
                        "left": object.rect_params.left,
                        "top": object.rect_params.top,
                        "width": object.rect_params.width,
                        "height": object.rect_params.height
                    }
                }
                for object in frame_meta.object_items
            ]
            producer.send(topic=TOPIC, value= {"frame_num": frame_meta.frame_number, "objects": objects})

                
def main(file_path):
    file_ext = os.path.splitext(file_path)[1]

    if file_ext in [".yaml", ".yml"]:
        Pipeline(PIPELINE_NAME, file_path).attach("pgie", Probe("custom-data-probe", SendCustomData())).start().wait()
    else:
        (Pipeline(PIPELINE_NAME).add("filesrc", "src", {"location": file_path}).add("h264parse", "parser").add("nvv4l2decoder", "decoder")
            .add("nvstreammux", "mux", {"batch-size": 1, "width": 1280, "height": 720})
            .add("nvinfer", "pgie", {"config-file-path": PGIE_CONFIG_FILE_PATH})
            .add("nvosdbin", "osd").add("nv3dsink" if platform.processor() == "aarch64" else "nveglglessink", "sink")
            .link("src", "parser", "decoder").link(("decoder", "mux"), ("", "sink_%u")).link("mux","pgie","osd", "sink")
            .attach("pgie", Probe("custom-data-probe", SendCustomData()))
            .start().wait())

if __name__ == '__main__':
    # Check input arguments
    if len(sys.argv) != 2:
        sys.stderr.write("usage: %s <H264 filename> OR <YAML config file>\n" % sys.argv[0])
        sys.exit(1)

    main(sys.argv[1])

    producer.flush()
    producer.close()