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

#ifndef DEEPSTREAM_MSGBROKER_C2D_RECEIVER_HPP
#define DEEPSTREAM_MSGBROKER_C2D_RECEIVER_HPP

#include <string>
#include <functional>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cstdint>

typedef struct NvDsC2DContext NvDsC2DContext;

namespace deepstream {
  class Cloud2DeviceReceiver {
   public:
   typedef struct {
    std::string proto_lib;
    std::string conn_str;
    std::string config_file_path;
    std::string topicList;
    std::string sensor_list_file;
   } Config;
    class IHandler {
     public:
      virtual ~IHandler() {}
    };

    // smart recording controller
    class ISmartRecordingController : public IHandler {
     public:
      virtual void startSmartRecord(int64_t camera_id, uint32_t *sessionId,
                          unsigned int startTime, unsigned int duration,
                          void *userData) = 0;
      virtual void stopSmartRecord(int64_t camera_id, uint32_t sessionId) = 0;
    };

    Cloud2DeviceReceiver();
    virtual ~Cloud2DeviceReceiver();
    
    void connect(Config& config);
    void disconnect();
    bool isConnected() { return context_ != nullptr; }
    bool hasHandler(IHandler* handler) {
      return std::find(handlers_.begin(), handlers_.end(), handler) != handlers_.end(); 
    }
    void addHandler(IHandler* handler) { handlers_.push_back(handler); }

    bool handleMessage(const char* topic, const char* payload, unsigned int size);

    static Cloud2DeviceReceiver& getInstance();

    // static Cloud2DeviceReceiver& getInstance() {
    //     static Cloud2DeviceReceiver instance; // create a single instance on first use
    //     return instance;
    // }

   private:
  
    bool parse_msgconv_config(const std::string& file_path);
  
    std::string config_path_;
    NvDsC2DContext* context_ = nullptr;
    std::vector<IHandler*> handlers_;
    std::unordered_map<std::string, int64_t> sensor_name_id_map_;
    std::unordered_map<std::string, uint32_t> sensor_sr_session_id_map_;
};
}

#endif