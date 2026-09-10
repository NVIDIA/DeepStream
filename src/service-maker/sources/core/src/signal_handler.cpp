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

#include "signal_handler.hpp"
#include <gst/gst.h>
#include "gst/signal_handler.h"

namespace deepstream {

SignalHandler::SignalHandler(
    const std::string& name, SignalHandler::IActionProvider* provider
):  CustomObject(SignalHandler::type(), nullptr, name), provider_(provider) {
}

SignalHandler::SignalHandler(
    const std::string& name, const char* factory, SignalHandler::IActionProvider* provider
):  CustomObject(SignalHandler::type(), factory, name), provider_(provider) {
}

SignalHandler::~SignalHandler() {

}

void* SignalHandler::getCallbackFn(const std::string& name) const {
    auto callbacks = provider_->getCallbacks();
    if (!callbacks) {
        return nullptr;
    }

    const Callback* p = callbacks;
    while (p->fn && p->name != name) {
       p++;
    }

    return p->fn;
}

unsigned long SignalHandler::type() {
    return GST_TYPE_SIGNAL_HANDLER;
}

}