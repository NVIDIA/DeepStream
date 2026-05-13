/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#define CONFIG_FILE_PATH "/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-test1/dstest1_pgie_config.yml"
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

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << " <YAML config file>" << std::endl;
    std::cout << "OR: " << argv[0] << " <H264 filename>" << std::endl;
    return 0;
  }

  std::string sink = "nveglglessink";

#if defined(__aarch64__)
  sink = "nv3dsink";
#endif

  try  {
    std::string file = argv[1];
    std::string suffix = "yaml";
    if (std::equal(suffix.rbegin(), suffix.rend(), file.rbegin())) {
      Pipeline pipeline("deepstream-test1", file);
      bool output_tensor_meta = false;
      pipeline["infer"].getProperty("output-tensor-meta", output_tensor_meta);
      if (output_tensor_meta) {
        unsigned int stream_width = 0;
        unsigned int stream_height = 0;
        pipeline["nvstreammux"].getProperty("width", stream_width).getProperty("height", stream_height);
        // disable metadata generation from the built-in postprocessing
        pipeline["infer"].set("filter-out-class-ids", "0;1;2;3");
        pipeline.attach("infer", "resnet_tensor_parser", "tensor parser", "", "stream-width", (int)stream_width, "stream-height", (int)stream_height);
      } else {
        pipeline.attach("infer", new BufferProbe("counter", new ObjectCounter));
      }
      pipeline.start().wait();
    } else {
      Pipeline pipeline("deepstream-test1");
      pipeline.add("filesrc", "src", "location", file)
          .add("h264parse", "parser")
          .add("nvv4l2decoder", "decoder")
          .add("nvstreammux", "mux", "batch-size", 1, "width", 1280, "height", 720)
          .add("nvinfer", "infer", "config-file-path", CONFIG_FILE_PATH)
          .add("nvvideoconvert", "converter")
          .add("nvdsosd", "osd")
          .add(sink, "sink")
          .link("src", "parser", "decoder")
          .link({"decoder", "mux"}, {"", "sink_%u"})
          .link("mux", "infer", "converter", "osd", "sink")
          .attach("infer", new BufferProbe("counter", new ObjectCounter))
          .attach("infer", "sample_video_probe", "my probe", "src", "font-size", 20)
          .start()
          .wait();
    }
  }  catch (const std::exception &e)  {
    std::cerr << e.what() << std::endl;
    return -1;
  }

  return 0;
}
