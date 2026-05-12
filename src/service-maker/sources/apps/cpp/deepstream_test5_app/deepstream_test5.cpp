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
#include <memory>
#include <string.h>
#include <fstream>

#include "pipeline.hpp"
#include "common_factory.hpp"
#include "lib/model_engine_watch_otf_trigger.hpp"
#include "tiler_event_handler.hpp"
#include "lib/perf_monitor.hpp"
#include "source_config.hpp"

using namespace deepstream;

class MsgMetaGenerator : public BufferProbe::IBatchMetadataOperator {
public:
  MsgMetaGenerator(const std::map<uint32_t, SensorInfo> &sensor_map, const std::vector<std::string> labels, int frame_interval=1)
  : sensor_map_(sensor_map), labels_(labels), frame_interval_(frame_interval) {}
  virtual ~MsgMetaGenerator() {}

  virtual probeReturn handleData(BufferProbe& probe, BatchMetadata& data) {
    FrameMetadata::Iterator frame_itr;
    for (data.initiateIterator(frame_itr); !frame_itr->done(); frame_itr->next()) {
      ObjectMetadata::Iterator obj_itr;
      for ((*frame_itr)->initiateIterator(obj_itr); !obj_itr->done(); obj_itr->next()) {
        if (0 == (frames_ % frame_interval_)) {
          EventMessageUserMetadata event_user_meta;
          if (data.acquire(event_user_meta)) {
            auto source_id = (*frame_itr)->sourceId();
            auto itr = sensor_map_.find(source_id);
            if (itr != sensor_map_.end()) {
              const std::string sensor = itr->second.sensor_id;
              const std::string uri = itr->second.uri;
              event_user_meta.generate(**obj_itr, **frame_itr, sensor, uri, labels_);
              (*frame_itr)->append(event_user_meta);
            }
          }
        }
      }
      frames_++;
    }

    return probeReturn::Probe_Ok;
  }

 protected:
   int frames_ = 0;
   const std::map<uint32_t, SensorInfo> &sensor_map_;
   std::vector<std::string> labels_;
   int frame_interval_;
};

static void usage()
{
  std::cout << "Usage: deepstream-test5-app -s source-lists -c config-files [-l label-file] [--perf-measurement-interval-sec] [--hide-stream-name]" << std::endl;
}

static std::string parse_config(std::vector<std::string>& arguments, const std::string flag) {
  std::string str;
  auto itr = std::find(arguments.begin(), arguments.end(), flag);
  if (itr != arguments.end()) {
    auto next_itr = itr + 1;
    if (next_itr != arguments.end()) {
      str = *next_itr;
    }
  }
  return str;
}

static std::vector<std::string> split(const std::string& input, char delimiter) {
    std::vector<std::string> result;
    std::istringstream stream(input);
    std::string token;

    while (std::getline(stream, token, delimiter)) {
        result.push_back(token);
    }

    return result;
}

static std::vector<std::string> parse_labels(const std::string& filename) {
  std::ifstream file(filename); // Open the file
  std::vector<std::string> lines; // Vector to store lines

  if (file.is_open()) { // Check if the file is open
    std::string line;
    while (std::getline(file, line)) { // Read each line
      std::string result=line;
      std::transform(line.begin(), line.end(), result.begin(), [](unsigned char c){ return std::tolower(c); });
      lines.push_back(result); // Store the line in the vector
    }
    file.close(); // Close the file
  } else {
      std::cerr << "Unable to open file: " << filename << std::endl;
  }

  return lines;
}

int main(int argc, char *argv[])
{
  if (argc < 5) {
    usage();
    return 0;
  }

  std::string model_engine_file;
  uint32_t perf_interval = 1;
  std::vector<std::string> arguments;
  for (int n = 1; n < argc; n++)
  {
    arguments.push_back(argv[n]);
  }
  // find the config files
  std::vector<std::string> config_files = split(parse_config(arguments, "-c"), ',');
  std::vector<std::string> source_config_files = split(parse_config(arguments, "-s"), ',');
  std::string label_file = parse_config(arguments, "-l");
  std::string perf_interval_sec = parse_config(arguments, "--perf-measurement-interval-sec");
  bool hide_stream_name = std::find(arguments.begin(), arguments.end(), "--hide-stream-name") != arguments.end();
  if (config_files.empty() || source_config_files.empty()) {
    usage();
    return 0;
  }
  if (config_files.size() != source_config_files.size()) {
    std::cout << "The number of pipeline configuration file must be equal to that of the source list file" << std::endl;
    return 0;
  }

  std::vector<std::map<uint32_t, SensorInfo>> sensor_maps(config_files.size());
  std::vector<std::string> labels;
  if (!label_file.empty()) {
    labels = parse_labels(label_file);
  }

  try
  {
    std::vector<std::unique_ptr<Pipeline>> pipelines;
    std::vector<std::unique_ptr<NvDsModelEngineWatchOTFTrigger>> otf_triggers;
    std::vector<std::unique_ptr<NvDsTilerEventHandler>> tiler_event_handlers;
    std::vector<std::unique_ptr<PerfMonitor>> perf_monitors;
    std::vector<std::unique_ptr<deepstream::CustomObject>> smart_record_objects;
    for (unsigned int idx = 0; idx < config_files.size(); idx++) {
      std::map<uint32_t, SensorInfo> &sensor_map = sensor_maps[idx];
      SourceConfig source_config(source_config_files[idx]);
      std::string pipeline_name = "deepstream-test5-";
      pipeline_name += std::to_string(idx);
      pipelines.push_back(std::make_unique<Pipeline>(pipeline_name.c_str(), config_files[idx]));
      Pipeline& pipeline = *pipelines.back().get();
      std::string kafka_proto_lib_path;
      std::string msgbroker_conn_str;
      std::string msgconv_config_path;
      std::string msgbroker_config_path;
      uint32_t tiler_width = 0;
      uint32_t tiler_height = 0;
      std::string pgie = "pgie";

      pipeline["msgbroker"].getProperty(
        "config", msgbroker_config_path,
        "proto-lib", kafka_proto_lib_path,
        "conn-str", msgbroker_conn_str
      );
      pipeline["msgconv"].getProperty("config", msgconv_config_path);
      pipeline["tiler"].getProperty("width", tiler_width, "height", tiler_height);
      pipeline[pgie].getProperty("model-engine-file", model_engine_file);

      // create the smart recording action
      auto object = CommonFactory::getInstance().createObject("smart_recording_action", "sr_action");
      auto* sr_action = dynamic_cast<SignalEmitter*>(object.get());
      if (!sr_action) {
        std::cerr << "Failed to create signal emitter" << std::endl;
        return -1;
      }

      smart_record_objects.push_back(std::move(object));

      sr_action->set(
          "proto-lib", kafka_proto_lib_path,
          "conn-str", msgbroker_conn_str,
          "msgconv-config-file", msgconv_config_path,
          "proto-config-file", msgbroker_config_path,
          "topic-list", "test5-sr");

      std::string checkpoint = "queue_tee";
      if (source_config.useMultiUriSrcBin()) {
        // nvmultiurisrcbin used
        pipeline.add("nvmultiurisrcbin", "multiurisrcbin");
        pipeline["multiurisrcbin"].set(source_config.getProperties())
                                  .set("uri-list", source_config.listUris())
                                  .set("sensor-id-list", source_config.listSensorIds())
                                  .set("sensor-name-list", source_config.listSensorNames());

        pipeline.link("multiurisrcbin", pgie);
      }

      else if (source_config.useCameraBin())
      {
        // camerabin used
        auto &properties = source_config.getProperties();
        pipeline.add("nvstreammux", "mux", "batch-size", source_config.nCameraSources(),
                     "width", tiler_width, "height", tiler_height, "live-source", 1, "batched-push-timeout", 40000);
        if (properties["gpu-id"])
        {
          pipeline["mux"].set("gpu-id", properties["gpu-id"].as<int>());
        }

        for (size_t i = 0; i < source_config.nCameraSources(); i++)
        {
          // multiple sources added
          std::string src_name = "src_";
          src_name += std::to_string(i);
          std::string invalid_value_check = "";
          std::string camera_type = source_config.getCameraInfo(i).camera_type;
          if (camera_type.empty())
            invalid_value_check = "camera-type";
          std::string camera_video_format = source_config.getCameraInfo(i).camera_video_format;
          if (camera_video_format.empty())
            invalid_value_check = "camera-video-format";
          std::string camera_width = source_config.getCameraInfo(i).camera_width;
          if (camera_width.empty())
            invalid_value_check = "camera-width";
          std::string camera_height = source_config.getCameraInfo(i).camera_height;
          if (camera_height.empty())
            invalid_value_check = "camera-height";
          std::string camera_fps_n = source_config.getCameraInfo(i).camera_fps_n;
          if (camera_fps_n.empty())
            invalid_value_check = "camera-fps-n";
          std::string camera_fps_d = source_config.getCameraInfo(i).camera_fps_d;
          if (camera_fps_d.empty())
            invalid_value_check = "camera-fps-d";

          if (!invalid_value_check.empty())
          {
            printf("Unexpected Value for: %s\n", invalid_value_check.c_str());
            return -1;
          }

          // common caps_filter for both camera types
          std::string src_cap_filter = src_name + "_cap_filter";
          char capfilterCaps[100];
          snprintf(capfilterCaps, 100, "video/x-raw(memory:NVMM), format=%s, width=%s, height=%s, framerate=%s/%s",
                   camera_video_format.c_str(), camera_width.c_str(), camera_height.c_str(), camera_fps_n.c_str(), camera_fps_d.c_str());
          pipeline.add("capsfilter", src_cap_filter, "caps", capfilterCaps);

          // queue to be added before muxer
          std::string src_queue_mux = src_name + "_queue_mux";
          pipeline.add("queue", src_queue_mux);

          if (camera_type == "CSI")
          {
            int camera_csi_sensor_id = source_config.getCameraInfo(i).camera_csi_sensor_id;
            if (camera_csi_sensor_id == -1)
            {
              printf("camera-csi-sensor-id not set\n");
              return -1;
            }
#ifdef __aarch64__
            pipeline.add("nvarguscamerasrc", src_name, "sensor-id", camera_csi_sensor_id);
#else
            pipeline.add("videotestsrc", src_name, "sensor-id", camera_csi_sensor_id);
#endif
            // nvarguscamerasrc/videtestsrc -> src_caps_filter -> mux
            pipeline.link(src_name, src_cap_filter, src_queue_mux, "mux");

            std::string uri = std::to_string(camera_csi_sensor_id);
            sensor_map[i] = SensorInfo{uri, std::to_string(i), src_name};
          }
          else if (camera_type == "V4L2")
          {
            std::string camera_v4l2_dev_node = source_config.getCameraInfo(i).camera_v4l2_dev_node;
            if (camera_v4l2_dev_node.empty())
            {
              printf("camera-v4l2-dev-node not set \n");
              return -1;
            }
            // Check from where to get these prop
            int gpu_id = source_config.getCameraInfo(i).gpu_id;
            int nvbuf_mem_type = source_config.getCameraInfo(i).nvbuf_mem_type;

            char device[100];
            snprintf(device, 100, "/dev/video%s", camera_v4l2_dev_node.c_str());

            // v4l2src
            pipeline.add("v4l2src", src_name, "device", device);

            // caps_filter1
            std::string src_cap_filter1 = src_name + "_cap_filter1";
            char capfilterCaps1[100];
            snprintf(capfilterCaps1, 100, "video/x-raw, width=%s, height=%s, framerate=%s/%s",
                     camera_width.c_str(), camera_height.c_str(), camera_fps_n.c_str(), camera_fps_d.c_str());
            pipeline.add("capsfilter", src_cap_filter1, "caps", capfilterCaps1);

            // videoconvert2
            std::string nvvidconv2 = src_name + "_nvvidconv2";
            pipeline.add("nvvideoconvert", nvvidconv2, "gpu-id", gpu_id, "nvbuf-memory-type", nvbuf_mem_type);

#if defined(__aarch64__) && !defined(AARCH64_IS_SBSA)

            int nvvideoconvert_copy_hw = source_config.getCameraInfo(i).nvvideoconvert_copy_hw;
            if (nvvideoconvert_copy_hw != 0)
            {
              pipeline[nvvidconv2].set("copy-hw", nvvideoconvert_copy_hw);
            }
#endif

            // v4l2src -> caps_filter1
            pipeline.link(src_name, src_cap_filter1);

#if !defined(__aarch64__)
            // videoconvert1 (only for x86)
            std::string nvvidconv1 = src_name + "_nvvidconv1";
            pipeline.add("videoconvert", nvvidconv1);
            // caps_filter1 -> nvvidconv1 -> nvvidconv2
            pipeline.link(src_cap_filter1, nvvidconv1, nvvidconv2);

#else
            // for aarch64 caps_filter1 -> nvvidconv2
            pipeline.link(src_cap_filter1, nvvidconv2);
#endif
            // nvvidconv2 -> caps_filter -> mux
            pipeline.link(nvvidconv2, src_cap_filter, src_queue_mux, "mux");

            std::string uri = device;
            sensor_map[i] = SensorInfo{uri, std::to_string(i), src_name};
          }
          else
          {
            printf("Invalid/NO Camera type\n");
            return -1;
          }
        }
        pipeline.add("queue", "queue_pgie");
        pipeline.link("mux", "queue_pgie", pgie);
      }

      else
      {
        // nvurisrcbin used
        auto &properties = source_config.getProperties();
        pipeline.add("nvstreammux", "mux", "batch-size", source_config.nSources(), "width", tiler_width, "height", tiler_height);
        if (properties["gpu-id"]) {
          pipeline["mux"].set("gpu-id", properties["gpu-id"].as<int>());
        }
        // create and add all the required elements
        for (size_t i = 0; i < source_config.nSources(); i++) {
          // multiple sources added
          std::string src_name = "src_";
          src_name += std::to_string(i);
          std::string uri = source_config.getSensorInfo(i).uri;
          pipeline.add("nvurisrcbin", src_name, "uri", uri.c_str()).link({src_name, "mux"}, {"vsrc_%u", ""});
          pipeline[src_name].set(properties);
          sensor_map[i] = SensorInfo{uri, std::to_string(i), src_name};
        }
        pipeline.link("mux", pgie);
      }
      pipeline.attach(checkpoint, new BufferProbe("message_generator", new MsgMetaGenerator(sensor_map, labels)));

      // create model update watcher
      otf_triggers.push_back(std::make_unique<NvDsModelEngineWatchOTFTrigger>(&pipeline[pgie], model_engine_file));
      NvDsModelEngineWatchOTFTrigger& otf_trigger = *otf_triggers.back().get();

      // create the event monitor for the tiler
      tiler_event_handlers.push_back(std::make_unique<NvDsTilerEventHandler>(&pipeline["tiler"], &pipeline["osd"], &pipeline["sink"]));
      NvDsTilerEventHandler& tiler_event_handler = *tiler_event_handlers.back().get();
#if !defined(__aarch64__)
      // only for x86
      tiler_event_handler.start();
#endif
      if (!source_config.useMultiUriSrcBin() && !source_config.useCameraBin())
      {
        // add the smart recording controll to all the sources
        for (size_t i = 0; i < source_config.nSources(); i++)
        {
          std::string src_name = "src_";
          src_name += std::to_string(i);
          sr_action->attach("start-sr", pipeline[src_name]);
          sr_action->attach("stop-sr", pipeline[src_name]);
          pipeline[src_name].connectSignal(
              "smart_recording_signal", "sr", "sr-done");
        }
      }

      // performance monitor
      if (!perf_interval_sec.empty()) {
        perf_interval = std::stoi(perf_interval_sec);
      }

      perf_monitors.push_back(std::make_unique<PerfMonitor>
      (
        source_config.useCameraBin()?source_config.nCameraSources():source_config.nSources(),
        perf_interval,
        source_config.useMultiUriSrcBin()?"nvmultiurisrcbin":"nvurisrcbin",
        !hide_stream_name
      ));
      PerfMonitor& perf_monitor = *perf_monitors.back().get();
      perf_monitor.apply(pipeline["tiler"], "sink");

      // install keyboard listener
      pipeline.install([&perf_monitor](Pipeline& pipeline, int key) {
        switch (key) {
          case 'p':
            pipeline.pause();
            perf_monitor.pause();
            break;
          case 'r':
            pipeline.resume();
            perf_monitor.resume();
            break;
          case 'q':
            pipeline.stop();
            break;
          default:
            break;
        }
      });

      // intialize pipeline with message monitor
      pipeline.prepare([&otf_trigger, &tiler_event_handler, &sensor_map, &perf_monitor](Pipeline &p, const Pipeline::Message &msg)
                       {
        const Pipeline::StateTransitionMessage *state_change_msg = dynamic_cast<const Pipeline::StateTransitionMessage*>(&msg);
        const Pipeline::DynamicSourceMessage *source_change_msg = dynamic_cast<const Pipeline::DynamicSourceMessage*>(&msg);
        // handle state change message
        if (state_change_msg) {
          if (!tiler_event_handler.started)
          {
            Pipeline::State old_state = Pipeline::State::EMPTY;
            Pipeline::State new_state = Pipeline::State::EMPTY;
            state_change_msg->getState(old_state, new_state);
            if (new_state == Pipeline::State::PLAYING && state_change_msg->getName() == "sink")
            {
              tiler_event_handler.start();
            }
          }
          if (!otf_trigger.started)
          {
            otf_trigger.start();
          }
        }

        // handle source add/remove message
        if (source_change_msg) {
          auto source_id   = source_change_msg->getSourceId();
          auto uri         = source_change_msg->getUri();
          auto sensor_id   = source_change_msg->getSensorId();
          auto sensor_name = source_change_msg->getSensorName();
          bool is_ipc      = (!uri.empty() && uri.rfind("ipc://", 0) == 0) ? true : false;
          uint32_t batch_size = 0;

          if (p.find("multiurisrcbin")) {
            p["multiurisrcbin"].getProperty("max-batch-size", batch_size);
          }

          if (source_change_msg->isSourceAdded()) {
            if (sensor_map.find(source_id) == sensor_map.end()) {
              sensor_map[source_id] = SensorInfo{uri, sensor_id, sensor_name};
            } else {
              std::cout << "Warning, sensor with source id " << source_id << " already added" << std::endl;
            }
            perf_monitor.addStream(source_id, uri.c_str(), sensor_id.c_str(), sensor_name.c_str());
            if (is_ipc) {
              std::string get_socket_path;
              std::string set_socket_path;
              const std::string ipc_prefix  = "ipc://";
              const std::string path_prefix = "/tmp/ds";
              std::string sink_name_prefix  = "nvunixfdsink";

              set_socket_path = uri.erase(0, ipc_prefix.length());
              set_socket_path = set_socket_path + "_ds";
              std::string nvunixfd_sink_name = sink_name_prefix + std::to_string(source_id);
              // find if any element named nvunixfdsink0, nvunixfdsink1, ... nvunixfdsinkN exists
              if (p.find(nvunixfd_sink_name.c_str())) {
                // get the socket-path property on the element nvunixfdsink
                p[nvunixfd_sink_name.c_str()].getProperty("socket-path", get_socket_path);
                // check if any of the nvunixfdsink element has default property set i.e. /tmp/ds
                if (get_socket_path.find(path_prefix) != std::string::npos) {
                  // set that particular element to NULL and set the socket-path property and then PLAYING
                  if (p[nvunixfd_sink_name.c_str()].setState(Element::State::EMPTY) == true) {
                    p[nvunixfd_sink_name.c_str()].set("socket-path", set_socket_path.c_str());
                    if (p[nvunixfd_sink_name.c_str()].setState(Element::State::PLAYING) == true) {
                      std::cout << "Using socket_path = " << set_socket_path << " with sink = " << nvunixfd_sink_name.c_str() << std::endl;
                    } else {
                      std::cout << "Failed to set socket_path = " << set_socket_path << " with sink = " << nvunixfd_sink_name.c_str() << std::endl;
                    }
                  } else {
                    std::cout << "Failed to set sink element = " << nvunixfd_sink_name.c_str() << " to NULL State" << std::endl;
                  }
                }
              }
            }
          } else {
            sensor_map.erase(source_id);
            perf_monitor.removeStream(source_id);
            if (is_ipc) {
              std::string socket_path;
              std::string set_socket_path;
              std::string get_socket_path;
              const std::string ipc_prefix  = "ipc://";
              const std::string path_prefix = "/tmp/ds";
              std::string sink_name_prefix  = "nvunixfdsink";
              std::string pipeline_name     = p.getName();
              uint ipc_node                = 0;

              // create the socket-path that needs to be searched on element property
              socket_path = uri.erase(0, ipc_prefix.length());
              socket_path = socket_path + "_ds";

              get_socket_path = "";
              std::string nvunixfd_sink_name = sink_name_prefix + std::to_string(source_id);
              // find if any element named nvunixfdsink0, nvunixfdsink1, ... nvunixfdsinkN exists
              if (p.find(nvunixfd_sink_name.c_str())) {
                // get the socket-path property on the element nvunixfdsink
                p[nvunixfd_sink_name.c_str()].getProperty("socket-path", get_socket_path);
                // check if any of the nvunixfdsink element has property set to socket_path
                if (socket_path.find(get_socket_path) != std::string::npos) {
                  ipc_node = source_id;
                  // if its pipeline 0 then default paths are /tmp/ds0 to /tmp/dsN-1, N is batch size
                  // and for pipeline 1 the default paths are /tmp/dsN to /tmp/dsN+N-1, N is batch size
                  if (pipeline_name == "deepstream-test5-1") {
                    ipc_node = ipc_node + batch_size;
                  }
                  // set that particular element to NULL and set the socket-path property and then PLAYING
                  set_socket_path = path_prefix + std::to_string(ipc_node);
                  if (p[nvunixfd_sink_name.c_str()].setState(Element::State::EMPTY) == true) {
                    p[nvunixfd_sink_name.c_str()].set("socket-path", set_socket_path.c_str());
                    if (p[nvunixfd_sink_name.c_str()].setState(Element::State::PLAYING) == true) {
                      std::cout << "Restored to default socket_path = " << set_socket_path << " with sink = " << nvunixfd_sink_name.c_str() << std::endl;
                    } else {
                      std::cout << "Failed to restore default socket_path = " << set_socket_path << " with sink = " << nvunixfd_sink_name.c_str() << std::endl;
                    }
                  } else {
                    std::cout << "Failed to set sink element = " << nvunixfd_sink_name.c_str() << " to NULL State" << std::endl;
                  }
                }
              }
            }
          }
        } });
#if defined(__aarch64__)
      // only for Jetson
      tiler_event_handler.start();
#endif
    }
    // start all the pipelines
    for (auto &pipeline : pipelines)
    {
      pipeline->activate();
    }
    // wait for all the pipelines to end
    for (auto& pipeline : pipelines) {
      pipeline->wait();
    }
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return -1;
  }

  return 0;
}
