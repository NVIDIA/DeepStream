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

#include "model_engine_watch_otf_trigger.hpp"
#include <gst/gst.h>
#include <sys/inotify.h>
#include <sys/stat.h>
#include <unistd.h>
#include <string.h>

namespace deepstream {

    static constexpr size_t INOTIFY_EVENT_SIZE = (sizeof(struct inotify_event));
    static constexpr size_t INOTIFY_EVENT_BUF_LEN =
        (1024 * (INOTIFY_EVENT_SIZE + 16));

    void NvDsModelEngineWatchOTFTrigger::file_watch_thread_func() {
        char buffer[INOTIFY_EVENT_BUF_LEN];
        struct timeval timeout;
        gchar *filename = g_path_get_basename(watch_file_path_.c_str());
        std::string file_basename = filename;
        g_free(filename);

        while (!stop_watch) {
            fd_set rfds, wfds;
            FD_ZERO(&rfds);
            FD_ZERO(&wfds);

            FD_SET(ota_inotify_fd_, &rfds);
            FD_SET(ota_inotify_fd_, &wfds);

            timeout.tv_sec = 0;
            timeout.tv_usec = 100000;
            int ret = select(ota_inotify_fd_ + 1, &rfds, &wfds, NULL, &timeout);
            if (ret < 0) {
            g_printerr("%s: Error while watching file '%s' : %s", "NvDsModelEngineWatchOTFTrigger",
                            watch_file_path_.c_str(), strerror(errno));
            break;
            }
            if (ret == 0) {
            continue;
            }

            ssize_t length = read(ota_inotify_fd_, buffer, INOTIFY_EVENT_BUF_LEN);

            ssize_t i = 0;
            while (i < length - (ssize_t) INOTIFY_EVENT_SIZE) {
                struct inotify_event *event = (struct inotify_event *)&buffer[i];

                if ((ssize_t) INOTIFY_EVENT_SIZE + event->len >= length) {
                    break;
                }

                if (event->len) {
                    if (event->mask & (IN_CLOSE_WRITE | IN_MOVED_TO) &&
                        file_basename == event->name) {
                    g_print("%s: '%s' updated. Triggering model update on-the-fly\n",
                                "NvDsModelEngineWatchOTFTrigger", watch_file_path_.c_str());
                    infer_->set("model-engine-file", watch_file_path_.c_str());
                    // nvdsinfer_prop_controller_.try_get().value()->set_model_engine_file(
                    //     filename.c_str());
                    }
                }

                i += INOTIFY_EVENT_SIZE + event->len;
            }
        }
    }

    bool NvDsModelEngineWatchOTFTrigger::start() {
        started = true;
        struct stat file_stat = {0};

        // if (!nvdsinfer_prop_controller_.try_get()) return true;

        if (watch_file_path_.empty()) {
            g_printerr("%s: watch-file not set", "NvDsModelEngineWatchOTFTrigger");
            return false;
        }
        std::string watch_file = watch_file_path_;

        ota_inotify_fd_ = inotify_init();
        if (ota_inotify_fd_ < 0) {
            g_printerr(
                "%s: Could not initialize a INotify instance to watch file: %s", "NvDsModelEngineWatchOTFTrigger",
                strerror(errno));
            return false;
        }

        int stat_ret = stat(watch_file.c_str(), &file_stat);
        if (stat_ret == 0) {
            if (S_ISDIR(file_stat.st_mode)) {
            g_printerr(
                "%s: '%s' is a directory. Only an existing/non-existent file can be "
                "watched.",
                "NvDsModelEngineWatchOTFTrigger", watch_file.c_str());
            return false;
            }
        }

        gchar *watch_file_dir = g_path_get_dirname(watch_file.c_str());
        char watch_file_dir_real[PATH_MAX + 1];
        if (!realpath(watch_file_dir, watch_file_dir_real)) {
            g_free(watch_file_dir);
            g_printerr("%s: Could not access directory '%s'", "NvDsModelEngineWatchOTFTrigger",
                        watch_file_dir_real);
            return false;
        }
        g_free(watch_file_dir);

        char *watch_file_basename = g_path_get_basename(watch_file.c_str());
        std::string file_basename = watch_file_basename;
        g_free(watch_file_basename);

        inotify_add_watch(ota_inotify_fd_, watch_file_dir_real, IN_ALL_EVENTS);

        file_watch_thread_ = std::thread([=] { this->file_watch_thread_func(); });

        return true;
    }

    bool NvDsModelEngineWatchOTFTrigger::stop() {
        stop_watch = true;
        if (file_watch_thread_.joinable()) file_watch_thread_.join();
        stop_watch = false;

        return true;
    }

}