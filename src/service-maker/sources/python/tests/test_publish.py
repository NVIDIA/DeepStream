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

from pyservicemaker import Pipeline, Flow
import threading, time

def stop_pipeline(pipeline, timeout=1):
    time.sleep(timeout)
    print("Stop")
    pipeline.stop()

source_list = [
    "/opt/nvidia/deepstream/deepstream/service-maker/sources/apps/cpp/deepstream_test5_app/source_list_dynamic.yaml",
    "/opt/nvidia/deepstream/deepstream/service-maker/sources/apps/cpp/deepstream_test5_app/source_list_static.yaml"
]

pgie_config = "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_infer_primary.yml"

msg_topic = "test4app"
conn_str = "localhost;9092"
msg_conv_config_file = "/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-test4/dstest4_msgconv_config.yml"
proto_lib = "/opt/nvidia/deepstream/deepstream/lib/libnvds_kafka_proto.so"

def test_publish():
    for source in source_list:
        pipeline = Pipeline("test")
        thread = threading.Thread(target=stop_pipeline, args=(pipeline, 10))
        thread.start()
        Flow(pipeline).batch_capture(source).infer(pgie_config).attach(
            what="add_message_meta_probe",
            name="message_generator"
        ).publish(
            msg_broker_proto_lib=proto_lib,
            msg_broker_conn_str=conn_str,
            topic=msg_topic,
            msg_conv_config=msg_conv_config_file
        )()
        thread.join()

if __name__ == '__main__':
    test_publish()