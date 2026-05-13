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

// inlcude all ds3d hpp header files
#include <ds3d/common/hpp/dataloader.hpp>
#include <ds3d/common/hpp/datamap.hpp>
#include <ds3d/common/hpp/frame.hpp>
#include <ds3d/common/hpp/yaml_config.hpp>
#include <ds3d/common/hpp/profiling.hpp>

// inlucde nvds3d Gst header files
#include <ds3d/gst/nvds3d_gst_plugin.h>
#include <ds3d/gst/nvds3d_gst_ptr.h>
#include <ds3d/gst/nvds3d_meta.h>
#include <unistd.h>

#include <functional>

#include "deepstream_lidar_infer_context.hpp"
#include <math.h>
#include "ds3d/common/impl/impl_frames.h"

using namespace ds3d;

constexpr const char* kDs3dFilterPluginName = "nvds3dfilter";

/* Print FPS per every several frames*/
constexpr const int kFpsPrintInterval = 30;

static uint64_t sFrameCount = 0;

static profiling::FpsCalculation gFpsCal(50);

struct AppProfiler {
    config::ComponentConfig config;
    profiling::FileWriter pointWriter;
    bool enableDebug = false;

    AppProfiler() = default;
    AppProfiler(const AppProfiler&) = delete;
    void operator=(const AppProfiler&) = delete;
    ~AppProfiler()
    {
        if (pointWriter.isOpen()) {
            pointWriter.close();
        }
    }

    ErrCode initProfiling(const config::ComponentConfig& compConf)
    {
        DS_ASSERT(compConf.type == config::ComponentType::kUserApp);
        config = compConf;
        YAML::Node node = YAML::Load(compConf.rawContent);
        std::string dumpPointFile;
        if (node["dump_points"]) {
            dumpPointFile = node["dump_points"].as<std::string>();
        }
        if (node["enable_debug"]) {
            enableDebug = node["enable_debug"].as<bool>();
            if (enableDebug) {
                setenv("DS3D_ENABLE_DEBUG", "1", 1);
            }
        }

        if (!dumpPointFile.empty()) {
            DS3D_FAILED_RETURN(
                pointWriter.open(dumpPointFile), ErrCode::kConfig, "create point file: %s failed",
                dumpPointFile.c_str());
        }
        return ErrCode::kGood;
    }
};

static GstPadProbeReturn
appsinkBufferProbe(GstPad* pad, GstPadProbeInfo* info, gpointer udata)
{
  float fps = gFpsCal.updateFps(0);
  if(sFrameCount == kFpsPrintInterval) {
    LOG_INFO("fps %f\n", fps);
    sFrameCount = 0;
  }

  sFrameCount++;

  return GST_PAD_PROBE_OK;
}

class LidarInferAppContext : public lidarinfer::DsLidarInferAppContext {
public:
    LidarInferAppContext() = default;
    ~LidarInferAppContext() { deinit(); }
    ErrCode initUserAppProfiling(const config::ComponentConfig& config)
    {
        auto initP = [this, &config]() { return _appProfiler.initProfiling(config); };
        DS3D_ERROR_RETURN(config::CatchYamlCall(initP), "parse ds3d::userapp failed");
        return ErrCode::kGood;
    }
    void setDataloaderSrc(gst::DataLoaderSrc src)
    {
        DS_ASSERT(src.gstElement);
        add(src.gstElement);
        _dataloaderSrc = std::move(src);
    }


    void setDataRenderSink(gst::DataRenderSink sink)
    {
        DS_ASSERT(sink.gstElement);
        add(sink.gstElement);
        _datarenderSink = std::move(sink);
    }

    AppProfiler& profiler() { return _appProfiler; }

    ErrCode stop()
    {
        if (_datarenderSink.customProcessor) {
            _datarenderSink.customProcessor.stop();
            _datarenderSink.gstElement.reset();
            _datarenderSink.customProcessor.reset();
        }

        ErrCode c = lidarinfer::DsLidarInferAppContext::stop();

        if (_dataloaderSrc.customProcessor) {
            _dataloaderSrc.customProcessor.stop();
            _dataloaderSrc.gstElement.reset();
            _dataloaderSrc.customProcessor.reset();
        }

        return c;
    }

    void deinit() override
    {
        lidarinfer::DsLidarInferAppContext::deinit();
        _datarenderSink.customlib.reset();
        _dataloaderSrc.customlib.reset();
    }

private:
    bool busCall(GstMessage* msg) final
    {
        DS_ASSERT(mainLoop());
        switch (GST_MESSAGE_TYPE(msg)) {
        case GST_MESSAGE_EOS:
            LOG_INFO("End of stream\n");
            quitMainLoop();
            break;
        case GST_MESSAGE_ERROR: {
            gchar* debug = nullptr;
            GError* error = nullptr;
            gst_message_parse_error(msg, &error, &debug);
            g_printerr("ERROR from element %s: %s\n", GST_OBJECT_NAME(msg->src), error->message);
            if (debug)
                g_printerr("Error details: %s\n", debug);
            g_free(debug);
            g_error_free(error);

            quitMainLoop();
            break;
        }
        default:
            break;
        }
        return TRUE;
    }

private:
    gst::DataLoaderSrc _dataloaderSrc;
    gst::DataRenderSink _datarenderSink;

    AppProfiler _appProfiler;
};

//Nothing to do now
static GstPadProbeReturn
appfilterBufferProbe(GstPad* pad, GstPadProbeInfo* info, gpointer udata)
{
    return GST_PAD_PROBE_OK;
}
/** Set global gAppCtx */
static std::weak_ptr<LidarInferAppContext> gAppCtx;

static void
help(const char* bin)
{
    printf("Usage: %s -c <../configs/config_lidar_triton_infer.yaml>\n", bin);
}

/**
 * Function to handle program interrupt signal.
 * It installs default handler after handling the interrupt.
 */
static void
_intr_handler(int signum)
{
    LOG_INFO("User Interrupted..");

    ShrdPtr<LidarInferAppContext> appCtx = gAppCtx.lock();
    if (appCtx) {
        if (appCtx->isRunning(2*GST_SECOND)) {
            appCtx->sendEOS();
        } else {
            appCtx->quitMainLoop();
        }
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

using ConfigList = std::vector<config::ComponentConfig>;

/**
 * Function to create dataloader source.
 */
static ErrCode
CreateLoaderSource(
    std::map<config::ComponentType, ConfigList>& configTable, gst::DataLoaderSrc& loaderSrc,
    bool startLoader)
{
    // Check whether dataloader is configured
    DS3D_FAILED_RETURN(
        configTable.count(config::ComponentType::kDataLoader), ErrCode::kConfig,
        "config file doesn't have dataloader types");
    DS_ASSERT(configTable[config::ComponentType::kDataLoader].size() == 1);
    config::ComponentConfig& srcConfig = configTable[config::ComponentType::kDataLoader][0];

    // creat appsrc and dataloader
    DS3D_ERROR_RETURN(
        NvDs3D_CreateDataLoaderSrc(srcConfig, loaderSrc, startLoader),
        "Create appsrc and dataloader failed");
    DS_ASSERT(loaderSrc.gstElement);
    DS_ASSERT(loaderSrc.customProcessor);

    return ErrCode::kGood;
}

/**
 * Function to create datarender sink.
 */
static ErrCode
CreateRenderSink(
    std::map<config::ComponentType, ConfigList>& configTable, gst::DataRenderSink& renderSink,
    bool startRender)
{
    // Check whether datarender is configured
    if (configTable.find(config::ComponentType::kDataRender) == configTable.end()) {
        LOG_INFO("config file does not have datarender component, using fakesink instead");
        renderSink.gstElement = gst::elementMake("fakesink", "fakesink");
        DS_ASSERT(renderSink.gstElement);
        return ErrCode::kGood;
    }

    DS3D_FAILED_RETURN(
        configTable[config::ComponentType::kDataRender].size() == 1, ErrCode::kConfig,
        "multiple datarender component found, please update and keep 1 render only");

    config::ComponentConfig& sinkConfig = configTable[config::ComponentType::kDataRender][0];

    // creat appsink and datarender
    DS3D_ERROR_RETURN(
        NvDs3D_CreateDataRenderSink(sinkConfig, renderSink, startRender),
        "Create appsink and datarender failed");
    DS_ASSERT(renderSink.gstElement);
    DS_ASSERT(renderSink.customProcessor);

    return ErrCode::kGood;
}

int
main(int argc, char* argv[])
{
    gst::DataLoaderSrc loaderSrc;
    gst::ElePtr filterSrc;
    gst::DataRenderSink renderSink;
    std::string configPath;
    std::string configContent;

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
    CHECK_ERROR(readFile(configPath, configContent), "read file: %s failed", configPath.c_str());

    // parse all components in config file
    ConfigList componentConfigs;
    ErrCode code =
        CatchConfigCall(config::parseFullConfig, configContent, configPath, componentConfigs);
    CHECK_ERROR(isGood(code), "parse config failed");

    // Order all parsed component configs into config table
    std::map<config::ComponentType, ConfigList> configTable;
    for (const auto& c : componentConfigs) {
        configTable[c.type].emplace_back(c);
    }

    ShrdPtr<LidarInferAppContext> appCtx = std::make_shared<LidarInferAppContext>();
    gAppCtx = appCtx;

    // update userapp configuration
    if (configTable.count(config::ComponentType::kUserApp)) {
        CHECK_ERROR(
            isGood(appCtx->initUserAppProfiling(configTable[config::ComponentType::kUserApp][0])),
            "parse userapp data failed");
    }

    // Initialize app context with main loop and pipelines
    appCtx->setMainloop(g_main_loop_new(NULL, FALSE));
    CHECK_ERROR(appCtx->mainLoop(), "set main loop failed");
    CHECK_ERROR(isGood(appCtx->init("deepstream-lidar-infer-pipeline")), "init pipeline failed");

    bool startLoaderDirectly = true;
    bool startRenderDirectly = true;
    CHECK_ERROR(
        isGood(CreateLoaderSource(configTable, loaderSrc, startLoaderDirectly)),
        "create dataloader source failed");

    CHECK_ERROR(
        isGood(CreateRenderSink(configTable, renderSink, startRenderDirectly)),
        "create datarender sink failed");

    appCtx->setDataloaderSrc(loaderSrc);
    appCtx->setDataRenderSink(renderSink);

    DS_ASSERT(loaderSrc.gstElement);
    DS_ASSERT(renderSink.gstElement);

    /* create and add all filters */
    bool hasFilters = configTable.count(config::ComponentType::kDataFilter);
    /* link all pad/elements together */
    code = CatchVoidCall([&loaderSrc, &filterSrc, &renderSink, hasFilters, &configTable, appCtx]() {
        gst::ElePtr lastEle = loaderSrc.gstElement;
        if (hasFilters) {
            auto& filterConfigs = configTable[config::ComponentType::kDataFilter];
            DS_ASSERT(filterConfigs.size());
            for (size_t i = 0; i < filterConfigs.size(); ++i) {
                auto queue = gst::elementMake("queue", ("filterQueue" + std::to_string(i)).c_str());
                DS_ASSERT(queue);
                auto filter =
                    gst::elementMake(kDs3dFilterPluginName, ("filter" + std::to_string(i)).c_str());
                DS3D_THROW_ERROR_FMT(
                    filter, ErrCode::kGst, "gst-plugin: %s is not found", kDs3dFilterPluginName);
                //append cfg_path becuase app only pass filter content to low level.
                std::string rawContent = filterConfigs[i].rawContent + "\n  config_path: "+ filterConfigs[i].filePath;
                g_object_set(
                    G_OBJECT(filter.get()), "config-content", rawContent.c_str(),
                    nullptr);
                filterSrc = filter;
                appCtx->add(queue).add(filter);
                lastEle.link(queue).link(filter);
                lastEle = filter;
            }
        }
        lastEle.link(renderSink.gstElement);
    });
    CHECK_ERROR(isGood(code), "Link pipeline elements failed");

    /*add probe to filter's src*/
    if (filterSrc) {
        gst::PadPtr srcPad = filterSrc.staticPad("src");
        CHECK_ERROR(srcPad, "lidarInferenceFilter src pad is not detected.");
        srcPad.addProbe(GST_PAD_PROBE_TYPE_BUFFER, appfilterBufferProbe, appCtx.get(), NULL);
        srcPad.reset();
    }

    /* Add probe to datarender to print fps */
    if (renderSink.gstElement) {
        gst::PadPtr sinkPad = renderSink.gstElement.staticPad("sink");
        CHECK_ERROR(sinkPad, "appsink sink pad is not detected.");
        sinkPad.addProbe(GST_PAD_PROBE_TYPE_BUFFER, appsinkBufferProbe, NULL, NULL);
        sinkPad.reset();
    }

    CHECK_ERROR(isGood(appCtx->play()), "app context play failed");
    LOG_INFO("Play...");

    /* Wait till pipeline encounters an error or EOS */
    appCtx->runMainLoop();

    loaderSrc.reset();
    renderSink.reset();
    appCtx->stop();
    appCtx->deinit();

    return 0;
}
