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

#define BATCHED_PUSH_TIMEOUT 33000
#define MUXER_WIDTH 1920
#define MUXER_HEIGHT 1080
#define TILER_WIDTH 1280
#define TILER_HEIGHT 720
#define CONFIG_FILE_PATH "/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-test3/dstest3_pgie_config.yml"
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
    std::cout << "OR: " << argv[0] << " <uri1> [uri2] ... [uriN]" << std::endl;
    return 0;
  }

  uint i, num_sources = argc - 1;
  std::string sink = "nveglglessink";

#if defined(__aarch64__)
  sink = "nv3dsink";
#endif

  try {
    std::string file = argv[1];
    std::string suffix = "yaml";
    if (std::equal(suffix.rbegin(), suffix.rend(), file.rbegin())) {
      Pipeline pipeline("deepstream-test3", file);
      auto* counter = new ObjectCounter;
      auto* probe = new BufferProbe("counter", counter);
      pipeline.attach("infer", probe).start().wait();
    } else {
      Pipeline pipeline("deepstream-test3");

      for (i = 0; i<num_sources; i++) {
          std::string name = "src_";
          name += std::to_string (i);
          pipeline.add("uridecodebin", name, "uri", argv[i + 1]);
      }

      auto* counter = new ObjectCounter;
      auto* probe = new BufferProbe("counter", counter);

      pipeline.add("nvstreammux", "mux", "batch-size", num_sources, "batched-push-timeout", BATCHED_PUSH_TIMEOUT,"width", MUXER_WIDTH, "height", MUXER_HEIGHT)
          .add("nvinfer", "infer", "config-file-path", CONFIG_FILE_PATH, "batch-size", num_sources)
          .add("nvmultistreamtiler", "tiler", "width", TILER_WIDTH, "height", TILER_HEIGHT)
          .add("nvvideoconvert", "converter")
          .add("nvdsosd", "osd")
          .add(sink, "sink")
          .attach("infer", probe)
          .link("mux", "infer", "tiler", "converter", "osd", "sink");

      for (i=0;i<num_sources;i++) {
        std::string src="src_" + std::to_string(i);
        pipeline.link({src, "mux"}, {"", "sink_%u"});
      }

      pipeline.start().wait();
    }
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return -1;
  }

  return 0;
}
