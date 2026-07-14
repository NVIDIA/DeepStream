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
#include <vector>
#include "pipeline.hpp"
#include "common_factory.hpp"

#define KAFKA_PROTO_LIB_PATH "/opt/nvidia/deepstream/deepstream/lib/libnvds_kafka_proto.so"
#define KAFKA_CONFIG_FILE "/opt/nvidia/deepstream/deepstream/sources/libs/kafka_protocol_adaptor/cfg_kafka.txt"
#define MSGCONV_CONFIG_FILE "/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-test5/configs/dstest5_msgconv_sample_config.txt"
#define KAFKA_CONN_STR "localhost;9092"
#define KAFKA_TOPIC_LIST "sr-test"

#define CONFIG_FILE_PATH "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_infer_primary.yml"

#define BATCHED_PUSH_TIMEOUT 33000
#define MUXER_WIDTH 1920
#define MUXER_HEIGHT 1080
#define TILER_WIDTH 1280
#define TILER_HEIGHT 720

using namespace deepstream;

class LocalRecorder : public BufferProbe::IBatchMetadataObserver
{
private:
  Pipeline &pipeline_;
  int frame_count_;
  uint32_t session_id_;

public:
  LocalRecorder(Pipeline &pipeline) : pipeline_(pipeline), frame_count_(0) {}
  virtual ~LocalRecorder() {}

  virtual probeReturn handleData(BufferProbe &probe, const BatchMetadata &data) {
    data.iterate([this](const FrameMetadata &frame_data)
                 {
      this->frame_count_++;
      if(this->frame_count_ == 30) {
        this->session_id_ = this->pipeline_.startRecording("src_0", 0, 20, [](const RecordingInfo &info) {
          printf("\nSource recorded at: %s/%s. Duration %.2f sec\n",
            info.getFileDirectory().c_str(), info.getFileName().c_str(),
            info.getDuration() / 1000.0);
        });
      } else if(this->frame_count_ == 1000) {
        this->pipeline_.stopRecording(this->session_id_);
      }
    });

    return probeReturn::Probe_Ok;
  }
};


int main (int argc, char *argv[])
{
  std::string recording_type = "local"; // default recording option
  std::vector<std::string> uris;

  for (int arg_index = 1; arg_index < argc; ++arg_index) {
    std::string arg = argv[arg_index];
    if (arg == "--help" || arg == "-h") {
      std::cout << "Usage: " << argv[0] << " [--recording {local|kafka}] <uri1> [uri2] ... [uriN]" << std::endl;
      return 0;
    } else if (arg.rfind("--recording=", 0) == 0) {
      std::string value = arg.substr(std::string("--recording=").size());
      if (value == "local" || value == "kafka") {
        recording_type = value;
      } else {
        std::cerr << "Invalid value for --recording: " << value << ". Expected 'local' or 'kafka'." << std::endl;
        return 1;
      }
    } else if (arg == "--recording" || arg == "-r") {
      if (arg_index + 1 < argc) {
        std::string value = argv[++arg_index];
        if (value == "local" || value == "kafka") {
          recording_type = value;
        } else {
          std::cerr << "Invalid value for --recording: " << value << ". Expected 'local' or 'kafka'." << std::endl;
          return 1;
        }
      } else {
        std::cerr << "Missing value after " << arg << std::endl;
        return 1;
      }
    } else {
      uris.push_back(arg);
    }
  }

  if (uris.empty()) {
    std::cout << "Usage: " << argv[0] << " [--recording {local|kafka}] <uri1> [uri2] ... [uriN]" << std::endl;
    return 0;
  }

  uint i, num_sources = static_cast<uint>(uris.size());
  std::string sink = "nveglglessink";

#if defined(__aarch64__)
  sink = "nv3dsink";
#endif

  try {
      Pipeline pipeline("deepstream-sr-test");

      std::unique_ptr<CustomObject> sr_object;

      if (recording_type == "kafka") {

        sr_object = CommonFactory::getInstance().createObject("smart_recording_action", "sr_action");
        auto* sr_action = dynamic_cast<SignalEmitter*>(sr_object.get());
        if (!sr_action) {
          std::cerr << "Failed to create signal emitter" << std::endl;
          return -1;
        }

        sr_action->set(
            "proto-lib", KAFKA_PROTO_LIB_PATH,
            "conn-str", KAFKA_CONN_STR,
            "msgconv-config-file", MSGCONV_CONFIG_FILE,
            "proto-config-file", KAFKA_CONFIG_FILE,
            "topic-list", KAFKA_TOPIC_LIST);

        for (i = 0; i < num_sources; i++)
        {
          std::string src_name = "src_";
          src_name += std::to_string(i);
          pipeline.add("nvurisrcbin", src_name, "uri", uris[i], "smart-record", 1, "smart-rec-cache", 20, "smart-rec-container", 0, "smart-rec-dir-path", ".", "smart-rec-mode", 0);
          sr_action->attach("start-sr", pipeline[src_name]);
          sr_action->attach("stop-sr", pipeline[src_name]);
          pipeline[src_name].connectSignal(
              "smart_recording_signal", "sr", "sr-done");
        }

      }

      else {
        for (i = 0; i < num_sources; i++)
        {
          std::string src_name = "src_";
          src_name += std::to_string(i);
          pipeline.add("nvurisrcbin", src_name, "uri", uris[i], "smart-record", 2, "smart-rec-cache", 20, "smart-rec-container", 0, "smart-rec-dir-path", ".", "smart-rec-mode", 0);
        }
      }

      pipeline.add("nvstreammux", "mux", "batch-size", num_sources, "batched-push-timeout", BATCHED_PUSH_TIMEOUT,"width", MUXER_WIDTH, "height", MUXER_HEIGHT)
          .add("nvinfer", "infer", "config-file-path", CONFIG_FILE_PATH, "batch-size", num_sources)
          .add("nvmultistreamtiler", "tiler", "width", TILER_WIDTH, "height", TILER_HEIGHT)
          .add("nvvideoconvert", "converter")
          .add("nvdsosd", "osd")
          .add(sink, "sink", "sync", false)
          .link("mux", "infer", "tiler", "converter", "osd", "sink");

      for (i=0;i<num_sources;i++) {
        std::string src="src_" + std::to_string(i);
        pipeline.link({src, "mux"}, {"", "sink_%u"});
      }

      if (recording_type == "local") {
        LocalRecorder* recorder = new LocalRecorder(pipeline);
        pipeline.attach("infer", new BufferProbe("local_recorder", recorder));
      }

      pipeline.start().wait();

  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return -1;
  }

  return 0;
}
