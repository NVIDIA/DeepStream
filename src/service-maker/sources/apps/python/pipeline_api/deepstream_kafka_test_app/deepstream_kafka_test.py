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