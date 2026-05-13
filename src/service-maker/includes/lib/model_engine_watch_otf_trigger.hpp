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