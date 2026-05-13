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

#define MUXER_BATCH_SIZE 1
#define MUXER_WIDTH 1280
#define MUXER_HEIGHT 720
#define PGIE_CONFIG_FILE_PATH "/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-test2/dstest2_pgie_config.yml"
#define TRACKER_LL_CONFIG_FILE "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_NvDCF_perf.yml"
#define TRACKER_LL_LIB_FILE "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so"
#define SGIE1_CONFIG_FILE_PATH "/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-test2/dstest2_sgie1_config.yml"
#define SGIE2_CONFIG_FILE_PATH "/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-test2/dstest2_sgie2_config.yml"
#define SINK_SYNC_VALUE false
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
        }
        else if (class_id == PGIE_CLASS_ID_PERSON)
        {
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

  try {
    std::string file = argv[1];
    std::string suffix = "yaml";
    if (std::equal(suffix.rbegin(), suffix.rend(), file.rbegin())) {
      Pipeline pipeline("deepstream-test2", file);
      pipeline.attach("converter", new BufferProbe("counter", new ObjectCounter)).start().wait();
    } else {
      Pipeline pipeline("deepstream-test2");
      pipeline.add("filesrc", "src", "location", file)
          .add("h264parse", "parser")
          .add("nvv4l2decoder", "decoder")
          .add("nvstreammux", "mux", "batch-size", MUXER_BATCH_SIZE, "width", MUXER_WIDTH, "height", MUXER_HEIGHT)
          .add("nvinfer", "infer", "config-file-path", PGIE_CONFIG_FILE_PATH)
          .add("nvtracker", "tracker", "ll-config-file", TRACKER_LL_CONFIG_FILE, "ll-lib-file", TRACKER_LL_LIB_FILE)
          .add("nvinfer", "vehicle_make_classifier", "config-file-path", SGIE1_CONFIG_FILE_PATH)
          .add("nvinfer", "vehicle_type_classifier", "config-file-path", SGIE2_CONFIG_FILE_PATH)
          .add("nvvideoconvert", "converter")
          .add("nvdsosd", "osd")
          .add(sink, "sink", "sync", SINK_SYNC_VALUE)
          .link("src", "parser", "decoder")
          .link({"decoder", "mux"}, {"", "sink_%u"})
          .link("mux", "infer", "tracker", "vehicle_make_classifier", "vehicle_type_classifier", "converter", "osd", "sink")
          .attach("converter", new BufferProbe("counter", new ObjectCounter))
          .attach("converter", "sample_video_probe", "my probe")
          .start()
          .wait();
    }
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return -1;
  }

  return 0;
}
