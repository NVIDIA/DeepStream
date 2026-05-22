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

#ifndef MODEL_ENGINE_WATCH_OTF_TRIGGER_HPP
#define MODEL_ENGINE_WATCH_OTF_TRIGGER_HPP

#include <string>
#include <thread>
#include "element.hpp"

namespace deepstream {

class NvDsModelEngineWatchOTFTrigger {
    public:
    NvDsModelEngineWatchOTFTrigger (Element *infer, const std::string watch_file)
    :infer_(infer), watch_file_path_(watch_file) {}

    ~NvDsModelEngineWatchOTFTrigger () {
        stop();
    }
    bool start();
    bool stop();
    void file_watch_thread_func();
    bool stop_watch = false;

    int ota_inotify_fd_;
    std::thread file_watch_thread_;
    Element *infer_;
    std::string watch_file_path_;
    bool started = false;
};

}

#endif