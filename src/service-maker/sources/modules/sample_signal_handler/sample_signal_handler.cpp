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
#include <memory>
#include <vector>

#include "plugin.h"
#include "custom_factory.hpp"
#include "common_factory.hpp"
#include "signal_handler.hpp"

using namespace std;

class GstElement;

namespace deepstream {

/**
 * Callback function to notify the status of the model update
 */
static void
infer_model_updated_cb (GstElement * gie, int err, const char * config_file, void* user_data)
{
  const char *err_str = (err == 0 ? "ok" : "failed");
  std::cout << "\nModel Update Status:" << err_str << std::endl;
}

static const SignalHandler::Callback callbacks[] = {
  {"model-updated", (void*)infer_model_updated_cb},
  {"", (void*) nullptr}
};

class ModelUpdatedHandler : public SignalHandler::IActionProvider {
 public:
  virtual const SignalHandler::Callback* getCallbacks() {
    return &callbacks[0];
  }
};

#define FACTORY_NAME "sample_signal_handler"

DS_CUSTOM_PLUGIN_DEFINE(
    sample_signal_handler,
    "this is a sample signal handler plugin",
    "0.1",
    "Proprietary")

DS_CUSTOM_FACTORY_DEFINE_WITH_SIGNALS(
  FACTORY_NAME,
  "sample signal handler factory",
  "signal",
  "this is a signal handler factory",
  "NVIDIA",
  "model-updated",
  SignalHandler,
  ModelUpdatedHandler
)

}