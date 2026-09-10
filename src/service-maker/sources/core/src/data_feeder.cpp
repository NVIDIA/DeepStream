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

#include "data_feeder.hpp"

#include <gst/gst.h>
#include <gst/app/gstappsrc.h>

using namespace deepstream;

static void need_data_cb(GstElement* object, guint arg0, gpointer user_data) {
  DataFeeder* feeder = static_cast<DataFeeder*>(user_data);
  feeder->startFeed(object, arg0);
}

static void enough_data_cb(GstElement* object, gpointer user_data) {
  DataFeeder* feeder = static_cast<DataFeeder*>(user_data);
  feeder->stopFeed();
}

static const SignalHandler::Callback callbacks[] = {
  {"need-data", (void*)need_data_cb},
  {"enough-data",(void*)enough_data_cb },
  {"", (void*) nullptr}
};

class DataFeedHandler : public SignalHandler::IActionProvider {
 public:
  virtual const SignalHandler::Callback* getCallbacks() {
    return &callbacks[0];
  }
};

DataFeeder::DataFeeder(const std::string &name, IDataProvider* provider)
: DataFeeder(name, nullptr, provider) {
}

DataFeeder::DataFeeder(const std::string &name, const char* factory, IDataProvider* provider)
: SignalHandler(name, factory, new DataFeedHandler),
  data_provider_(provider), data_size_(0), data_ready_(false), appsrc_(nullptr), eos_(false) {
  worker_ = std::thread([this](){
    while (true) {
      void* appsrc = NULL;
      unsigned int size = 0;
      {
        std::unique_lock lock(mutex_);
        if (data_size_ == 0 || !data_ready_)
          cv_data_.wait_for(lock,  std::chrono::milliseconds(200));
        if (eos_) break;
        if (data_size_ == 0) continue;
        size = data_size_;
        appsrc = appsrc_;
      }
      // reset the data_ready flag before trying read
      data_ready_ = true;
      this->doFeed(appsrc, size);
    }
  });
}

DataFeeder::~DataFeeder() {
  eos_ = true;
  cv_data_.notify_all();
  worker_.join();
}

void DataFeeder::startFeed(void* appsrc, unsigned int size) {
  std::unique_lock lock(mutex_);
  data_size_ = size;
  appsrc_ = appsrc;
  cv_data_.notify_all();
}
void DataFeeder::stopFeed() {
  std::unique_lock lock(mutex_);
  data_size_ = 0;
  appsrc_ = nullptr;
  cv_data_.notify_all();
}

bool DataFeeder::doFeed(void* appsrc, unsigned int size) {
  Buffer buffer = data_provider_->read(*this, size, eos_);
  GstFlowReturn gstret = GST_FLOW_OK;
  if (eos_) {
    gstret = gst_app_src_end_of_stream((GstAppSrc *)appsrc);
  } else if (buffer.size() > 0) {
    OpaqueBuffer* gst_buffer = buffer.give();
    gstret = gst_app_src_push_buffer((GstAppSrc *)appsrc, gst_buffer);
  } else {
    // sleep the feeding thread if data is not ready
    data_ready_ = false;
  }
  if (gstret != GST_FLOW_OK) {
    g_printerr("gst_app_src_push_buffer returned %d \n", gstret);
    return false;
  }

  return true;
}