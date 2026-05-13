/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
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
      pipeline.attach("infer", new BufferProbe("counter", new ObjectCounter)).start().wait();
    } else {
      Pipeline pipeline("deepstream-test3");

      for (i = 0; i<num_sources; i++) {
          std::string name = "src_";
          name += std::to_string (i);
          pipeline.add("uridecodebin", name, "uri", argv[i + 1]);
      }

      pipeline.add("nvstreammux", "mux", "batch-size", num_sources, "batched-push-timeout", BATCHED_PUSH_TIMEOUT,"width", MUXER_WIDTH, "height", MUXER_HEIGHT)
          .add("nvinfer", "infer", "config-file-path", CONFIG_FILE_PATH, "batch-size", num_sources)
          .add("nvmultistreamtiler", "tiler", "width", TILER_WIDTH, "height", TILER_HEIGHT)
          .add("nvvideoconvert", "converter")
          .add("nvdsosd", "osd")
          .add(sink, "sink")
          .attach("infer", new BufferProbe("counter", new ObjectCounter))
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
