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

#include "plugin.h"
#include "custom_factory.hpp"
#include "common_factory.hpp"
#include "signal_handler.hpp"
#include "gst-nvdssr.h"

using namespace std;


namespace deepstream {

/**
 * Callback function to notify the status of smart recording
 */
static void
sr_done_cb (GstElement * src, NvDsSRRecordingInfo* recordingInfo, void* data, void* user_data) {
  printf("Source recorded at %s/%s. Duration %.2f sec",
               recordingInfo->dirpath, recordingInfo->filename,
               recordingInfo->duration / 1000.0);

}

static const SignalHandler::Callback callbacks[] = {
  {"sr-done", (void*)sr_done_cb},
  {"", (void*) nullptr}
};

class SmartRecordingdHandler : public SignalHandler::IActionProvider {
 public:
  virtual const SignalHandler::Callback* getCallbacks() {
    return &callbacks[0];
  }
};

#define FACTORY_NAME "smart_recording_signal"

DS_CUSTOM_PLUGIN_DEFINE(
    smart_recording_signal,
    "this is a smart recording signal handler plugin",
    "0.1",
    "Proprietary")

DS_CUSTOM_FACTORY_DEFINE_WITH_SIGNALS(
  FACTORY_NAME,
  "smart recording signal handler factory",
  "signal",
  "this is a smart recording handler factory",
  "NVIDIA",
  "sr-done",
  SignalHandler,
  SmartRecordingdHandler
)

}