/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <iostream>
#include <string>

#include "pipeline.hpp"

#define MUXER_WIDTH 1280
#define MUXER_HEIGHT 720
#define CONFIG_FILE_PATH "/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-test4/dstest4_pgie_config.yml"
#define CONFIG_MSGCONV_PATH "/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-test4/dstest4_msgconv_config.yml"
#define KAFKA_PROTO_LIB "/opt/nvidia/deepstream/deepstream/lib/libnvds_kafka_proto.so"
#define MSGBROKER_CONN_STR "localhost;9092"
#define MGSBROKER_TOPIC "test4app"
#define PGIE_CLASS_ID_VEHICLE 0
#define PGIE_CLASS_ID_PERSON 2

using namespace deepstream;

class ObjectCounter : public BufferProbe::IBatchMetadataObserver
{
public:
  ObjectCounter() {}
  virtual ~ObjectCounter() {}

  virtual probeReturn handleData(BufferProbe &probe, const BatchMetadata &data) {
    data.iterate([](const FrameMetadata &frame_data)
                 {
      auto vehicle_count = 0;
      auto person_count = 0;
      frame_data.iterate([&](const ObjectMetadata& object_data) {
        auto class_id = object_data.classId();
        if (class_id == PGIE_CLASS_ID_VEHICLE) {
          vehicle_count++;
        } else if (class_id == PGIE_CLASS_ID_PERSON ) {
          person_count++;
        }
      });
      std::cout << "Object Counter: " <<
        " Pad Idx = " << frame_data.padIndex() <<
        " Frame Number = " << frame_data.frameNum() <<
        " Vehicle Count = " << vehicle_count <<
        " Person Count = " << person_count << std::endl; });

    return probeReturn::Probe_Ok;
  }
};

int main (int argc, char *argv[])
{
  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << " <YAML config file>" << std::endl;
    std::cout << "OR: " << argv[0] << " <H264 filename>" << std::endl;
    return 0;
  }

  std::string sink = "nveglglessink";

#if defined(__aarch64__)
  sink = "nv3dsink";
#endif

  try {
    std::string file = argv[1];
    std::string suffix = "yaml";
    if (std::equal(suffix.rbegin(), suffix.rend(), file.rbegin())) {
      Pipeline pipeline("deepstream-test4", file);
      auto* counter = new ObjectCounter;
      pipeline.attach("infer", new BufferProbe("counter", counter));
      pipeline.start().wait();
    } else {
      Pipeline pipeline("deepstream-test4");
      pipeline.add("filesrc", "src", "location", argv[1])
          .add("h264parse", "parser")
          .add("nvv4l2decoder", "decoder")
          .add("nvstreammux", "mux", "batch-size", 1, "width", MUXER_WIDTH, "height", MUXER_HEIGHT)
          .add("nvinfer", "infer", "config-file-path", CONFIG_FILE_PATH)
          .add("nvvideoconvert", "converter")
          .add("nvdsosd", "osd")
          .add("tee", "tee")
          .add("queue", "queue1")
          .add("queue", "queue2")
          .add("nvmsgconv", "msgconv", "config", CONFIG_MSGCONV_PATH)
          .add("nvmsgbroker", "msgbroker", "conn-str", MSGBROKER_CONN_STR, "proto-lib", KAFKA_PROTO_LIB, "sync", false, "topic", MGSBROKER_TOPIC)
          .add(sink, "sink")
          .link("src", "parser", "decoder")
          .link({"decoder", "mux"}, {"", "sink_%u"})
          .link("mux", "infer", "converter", "osd", "tee")
          .link("tee", "queue1", "sink")
          .link("tee", "queue2", "msgconv", "msgbroker");
      auto* counter = new ObjectCounter;
      pipeline.attach("infer", new BufferProbe("counter", counter))
          .attach("osd", "add_message_meta_probe", "metadata generator")
          .start()
          .wait();
    }
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return -1;
  }

  return 0;
}
