/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "pipeline.hpp"
#include <stdio.h>
#include <string.h>

#define MUXER_BATCH_TIMEOUT_USEC 33000
#define CONFIG_FILE_PATH "/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-appsrc-test/dstest_appsrc_config.txt"
#define USE_GPU_MEMORY false

using namespace deepstream;

int main(int argc, char *argv[])
{
    bool use_gpu_memory = USE_GPU_MEMORY;
    if (argc < 6)
    {
        printf("Usage: %s <Raw filename> <width> <height> <fps> <format(I420, NV12, RGBA)> [gpu]\n", argv[0]);
        return -1;
    }
    else if (argc == 7 && strcmp(argv[6], "gpu") == 0)
    {
        use_gpu_memory = true;
    }

    int width, height;
    char *format, *p1, *p2;

    width = strtol(argv[2], &p1, 10);
    height = strtol(argv[3], &p2, 10);
    format = argv[5];

    char appsrcCaps[100];
    char capsfilterCaps[100];

    std::string capsfilterCapsFormat = "NV12";
    int blockSize = width * height * 1.5;

    if (!strcmp(format, "RGBA"))
    {
        capsfilterCapsFormat = "RGBA";
        blockSize = width * height * 4;
    }
    if (use_gpu_memory)
    {
        snprintf(appsrcCaps, 100, "video/x-raw(memory:NVMM), framerate=%s/1, format=%s, width=%s, height=%s", argv[4], argv[5], argv[2], argv[3]);
    }
    else
    {
        snprintf(appsrcCaps, 100, "video/x-raw, framerate=%s/1, format=%s, width=%s, height=%s", argv[4], argv[5], argv[2], argv[3]);
    }
    snprintf(capsfilterCaps, 100, "video/x-raw(memory:NVMM), format=%s", capsfilterCapsFormat.c_str());

    std::string sink = "nveglglessink";

#if defined(__aarch64__)
    sink = "nv3dsink";
#endif

    try
    {
        Pipeline pipeline("deepstream-appsrc-test-app");

        pipeline.add("appsrc", "appsrc", "do-timestamp", true, "caps", appsrcCaps, "blocksize", blockSize)
            .add("nvvideoconvert", "converter1")
            .add("capsfilter", "capsfilter1", "caps", capsfilterCaps)
            .add("nvstreammux", "mux", "width", width, "height", height, "batch-size", 1, "live-source", true, "batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC)
            .add("nvinfer", "infer", "config-file-path", CONFIG_FILE_PATH)
            .add("nvvideoconvert", "converter2")
            .add("nvdsosd", "osd")
            .add("tee", "tee")
            .add(sink, "sink", "sync", false)
            .add("appsink", "appsink", "emit-signals", true, "async", false)
            .link("appsrc", "converter1", "capsfilter1")
            .link({"capsfilter1", "mux"}, {"", "sink_%u"})
            .link("mux", "infer", "converter2", "tee")
            .link("tee", "osd", "sink")
            .link("tee", "appsink")
            .attach("appsrc", "sample_video_feeder", "feeder", "need-data/enough-data",
                    "location", argv[1], "frame-width", width, "frame-height", height, "format", format,
                    "use-gpu-memory", use_gpu_memory)
            .attach("appsink", "sample_video_receiver", "receiver", "new-sample")
            .start()
            .wait();
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << std::endl;
        return -1;
    }

    return 0;
}