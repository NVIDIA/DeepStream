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

#include "deepstream_3d_sensor_fusion.hpp"

using namespace ds3d;

/** Set global gAppCtx */
static std::weak_ptr<SensorFusionApp> gAppCtx;
/** Set flag to check pipeline stopped */
std::atomic_bool gStopped(false);
std::mutex gStopMutex;
std::condition_variable gStopCond;

static void
help(const char* bin)
{
  printf("Usage: %s -c <ds3d_lidar_video_alignment_render.yaml or another "
         "config file>\n",
         bin);
}

void static stopSignaled()
{
    auto appCtx = gAppCtx.lock();
    if (!appCtx) {
        return;
    }
    std::unique_lock<std::mutex> locker(gStopMutex);
    gStopped = true;
    gStopCond.notify_all();
}

void static WindowClosed()
{
    LOG_INFO("Window closed.");
    stopSignaled();
}

/**
 * Function to handle program interrupt signal.
 * It installs default handler after handling the interrupt.
 */
static void
_intr_handler(int signum)
{
    LOG_INFO("User Interrupted..");

    ShrdPtr<SensorFusionApp> appCtx = gAppCtx.lock();
    if (appCtx) {
        stopSignaled();
    } else {
        LOG_ERROR("program terminated.");
        std::terminate();
    }
}

/**
 * Function to install custom handler for program interrupt signal.
 */
static void
_intr_setup(void)
{
    struct sigaction action;
    memset(&action, 0, sizeof(action));
    action.sa_handler = _intr_handler;
    sigaction(SIGINT, &action, NULL);
}

int
main(int argc, char* argv[])
{
    std::string configPath;

    /* Standard GStreamer initialization */
    gst_init(&argc, &argv);

    /* setup signal handler */
    _intr_setup();

    /* Parse program arguments */
    opterr = 0;
    int c = -1;
    while ((c = getopt(argc, argv, "hc:")) != -1) {
        switch (c) {
        case 'c':  // get config file path
            configPath = optarg;
            break;
        case 'h':
            help(argv[0]);
            return 0;
        case '?':
        default:
            help(argv[0]);
            return -1;
        }
    }
    if (configPath.empty()) {
        LOG_ERROR("config file is not set!");
        help(argv[0]);
        return -1;
    }

    ShrdPtr<SensorFusionApp> appCtx = std::make_shared<SensorFusionApp>();
    gAppCtx = appCtx;

    CHECK_ERROR(
        isGood(appCtx->setup(configPath, WindowClosed)),
        "Failed to setup sensor fusion application.");
    CHECK_ERROR(isGood(appCtx->start(stopSignaled)), "Failed to start playing sensor fusion app.");
    LOG_INFO("Play...");

    std::chrono::seconds oneSec(1);
    while (!gStopped && appCtx->isRunning(3000)) {
        std::unique_lock<std::mutex> locker(gStopMutex);
        if (gStopCond.wait_for(locker, oneSec) != std::cv_status::timeout) {
            break;
        }
    }

    /* Wait till pipeline encounters an error or EOS */
    appCtx->stop();
    appCtx->deinit();

    return 0;
}
