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