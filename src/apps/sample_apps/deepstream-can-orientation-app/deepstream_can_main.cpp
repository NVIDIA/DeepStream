/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "deepstream_can_context.hpp"
#include <future>
#include <chrono>
#include <atomic>

std::weak_ptr<DsCanContext> gAppCtx;
std::atomic_bool gStopped(false);

static void
help(const char* bin)
{
    printf("Usage: %s -c <ds_can_orientation.yaml>\n", bin);
}

/**
 * Function to handle program interrupt signal.
 * It installs default handler after handling the interrupt.
 */
static void
_intr_handler(int signum)
{
    LOG_INFO("User Interrupted..");

    ShrdPtr<DsCanContext> canCtx = gAppCtx.lock();
    if (canCtx) {
        gStopped = true;
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

#undef CHECK_ERROR
#define CHECK_ERROR(statement, fmt, ...) DS3D_FAILED_RETURN(statement, -1, fmt, ##__VA_ARGS__)

#undef RETURN_ERROR
#define RETURN_ERROR(statement, fmt, ...) DS3D_ERROR_RETURN(statement, fmt, ##__VA_ARGS__)

#undef PITCH_ALIGNED
#define PITCH_ALIGNED(v, align) (((v) + (align) - 1) & (~((align) - 1)))

static profiling::Timing gTiming;
static uint32_t gFrameNum = 0;

static ErrCode
ProcessCtxRawFrames(DsCanContext* ctx, const NvDsCanFrameInfo& info, const std::string& path)
{
    DS3D_FAILED_RETURN(
        info.fourcc == NvDsCanFormatRGBA || info.fourcc == NvDsCanFormatGREY, ErrCode::kConfig,
        "source format is not supported, check config file");
    uint32_t pixelBytes = (info.fourcc == NvDsCanFormatGREY ? 1 : 4);
    uint32_t pitch = PITCH_ALIGNED(pixelBytes * info.width, 4);
    uint32_t frameBytes = pitch * info.height;

    DS_ASSERT(frameBytes);
    std::ifstream fileIn(path, std::ios::in | std::ios::binary);
    DS3D_FAILED_RETURN(fileIn, ErrCode::kNotFound, "open file %s failed", path.c_str());

    while (NvDs_CanContextIsRunning(ctx) && !fileIn.eof()) {
        std::vector<char> data(frameBytes);
        fileIn.read(&data[0], frameBytes);
        if (!fileIn) {
            LOG_DEBUG("read file failed");
            return ErrCode::kGood;
        }
        DS_ASSERT(fileIn.gcount() == frameBytes);

        NvDsCanContextFrame frame{&data[0], frameBytes, info.fourcc, info.width, info.height};
        NvDsCanContextResultMeta res{0.0f};
        auto t0 = std::chrono::high_resolution_clock::now();
        if (isGood(NvDs_CanContextProcessFrame(ctx, &frame, &res))) {
            LOG_DEBUG(
                "process frame: %4d;, result, rotation value: %.03f", gFrameNum, res.rotation);
        } else {
            LOG_ERROR("process frame: %4d failed", gFrameNum);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        auto ms_int0 = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);
        gTiming.push((double)ms_int0.count());
        if (gFrameNum && (gFrameNum % 30) == 0) {
            LOG_INFO("Current average processing frame time: %.03fms", gTiming.avg());
        }
        ++gFrameNum;
    }
    return ErrCode::kGood;
}

static void
CallFunc(void* d)
{
    std::function<void()>* f = (std::function<void()>*)(d);
    DS_ASSERT(f);
    (*f)();
}

int
main(int argc, char* argv[])
{
    std::string configPath;
    std::string rawPath;
    bool rawFileLoop = false;

    char muxenv[256] = "USE_NEW_NVSTREAMMUX=yes";
    putenv(muxenv);
    char pylonenv[256] = "PYLON_CAMEMU=1";
    putenv(pylonenv);

    /* Standard GStreamer initialization */
    gst_init(&argc, &argv);

    /* setup signal handler */
    _intr_setup();

    /* Parse program arguments */
    opterr = 0;
    int c = -1;
    while ((c = getopt(argc, argv, "hc:r:l")) != -1) {
        switch (c) {
        case 'c':  // get config file path
            configPath = optarg;
            break;
        case 'r':  // get raw input files
            rawPath = optarg;
            break;
        case 'l':  // raw file loop
            rawFileLoop = true;
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

    DsCanContext* ctx = nullptr;
    NvDsCanContextConfig config;
    CHECK_ERROR(
        isGood(NvDs_CanContextParseConfig(
            &config, configPath.c_str(), 0, NvDsCanContextConfigType::kConfigPath)),
        "parse can context config failed");
    if (config.srcType == NvDsCanSrcType::kFrame) {
        CHECK_ERROR(
            !rawPath.empty(), "raw file path is required for frame input, append [-r rawfilepath]");
    }

    CHECK_ERROR(isGood(NvDs_CanContextCreate(&ctx, &config)), "create context failed");
    CHECK_ERROR(ctx, "can context is null");
    std::shared_ptr<DsCanContext> canCtx(ctx, [](DsCanContext* c) {
        if (c) {
            NvDs_CanContextStop(c);
            NvDs_CanContextDestroy(c);
        }
    });
    gAppCtx = canCtx;

    std::promise<void> p;
    std::future<void> future = p.get_future();
    std::function<void()> waitQuitF = [&p]() { p.set_value(); };
    CHECK_ERROR(
        isGood(NvDs_CanContextStart(ctx, CallFunc, &waitQuitF)), "start can context failed");
    LOG_INFO("Context started...");

    if (config.srcType == NvDsCanSrcType::kFrame) {
        do {
            CHECK_ERROR(
                isGood(ProcessCtxRawFrames(ctx, config.srcFrameInfo, rawPath)),
                "process raw frames failed");
        } while (rawFileLoop && NvDs_CanContextIsRunning(ctx) && !gStopped);
    } else {
        std::chrono::seconds oneSec(1);
        while (NvDs_CanContextIsRunning(ctx) && !gStopped) {
            if (future.wait_for(oneSec) != std::future_status::timeout) {
                break;
            }
        }
    }

    LOG_INFO("Context is stopping...");
    CHECK_ERROR(isGood(NvDs_CanContextStop(ctx)), "start can context failed");

    canCtx.reset();
    return 0;
}
