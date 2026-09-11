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

#include "data_receiver.hpp"

#include <gst/gst.h>
#include <gst/app/gstappsink.h>

#include <thread>

using namespace deepstream;

static GstFlowReturn new_sample_cb(GstElement* sink, gpointer user_data) {
  DataReceiver* receiver = (DataReceiver*)user_data;
  GstSample *sample = gst_app_sink_pull_sample (GST_APP_SINK (sink));
  if (!sample) {
    return GST_FLOW_ERROR;
  }

  OpaqueBuffer* gst_buffer = gst_sample_get_buffer (sample);
  if (!gst_buffer) {
    return GST_FLOW_ERROR;
  }

  if (receiver->consume(Buffer(gst_buffer)) < 0) {
    return GST_FLOW_ERROR;
  }

  gst_sample_unref (sample);
  return GST_FLOW_OK;
}

static const SignalHandler::Callback callbacks[] = {
  {"new-sample", (void*)new_sample_cb},
  {"", (void*) nullptr}
};

class DataSinkHandler : public SignalHandler::IActionProvider {
 public:
  virtual const SignalHandler::Callback* getCallbacks() {
    return &callbacks[0];
  }
};

DataReceiver::DataReceiver(const std::string &name, IDataConsumer* consumer, unsigned int retries)
: SignalHandler(name, new DataSinkHandler), data_consumer_(consumer), retries_(retries) {
}

DataReceiver::DataReceiver(const std::string &name, const char* factory, IDataConsumer* consumer, unsigned int retries)
: SignalHandler(name, factory, new DataSinkHandler), data_consumer_(consumer), retries_(retries) {
}

DataReceiver::~DataReceiver() {}

int DataReceiver::consume(Buffer buffer) {
  unsigned int retries = 0;
  int ret = 0;
  do {
    ret = data_consumer_->consume(*this, buffer);
    if (ret == 0) {
      retries++;
      // put the current thread to sleep before another try
      std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }
  } while (!ret && retries <= retries_);

  return ret;
}