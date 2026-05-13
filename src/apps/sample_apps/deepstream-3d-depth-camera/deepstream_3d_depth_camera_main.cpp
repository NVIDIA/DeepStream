/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <cuda_runtime_api.h>

#include "gstnvdsmeta.h"

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

#include "deepstream_3d_context.hpp"
#include "deepstream_3d_context.hpp"

using namespace ds3d;

constexpr const char* kDs3dFilterPluginName = "nvds3dfilter";

struct AppProfiler {
    config::ComponentConfig config;
    profiling::FileWriter depthWriter;
    profiling::FileWriter colorWriter;
    profiling::FileWriter pointWriter;
    bool enableDebug = false;

    AppProfiler() = default;
    AppProfiler(const AppProfiler&) = delete;
    void operator=(const AppProfiler&) = delete;
    ~AppProfiler()
    {
        if (depthWriter.isOpen()) {
            depthWriter.close();
        }

        if (colorWriter.isOpen()) {
            colorWriter.close();
        }

        if (pointWriter.isOpen()) {
            pointWriter.close();
        }
    }

    ErrCode initProfiling(const config::ComponentConfig& compConf)
    {
        DS_ASSERT(compConf.type == config::ComponentType::kUserApp);
        config = compConf;
        YAML::Node node = YAML::Load(compConf.rawContent);
        std::string dumpDepthFile;
        std::string dumpColorFile;
        std::string dumpPointFile;
        if (node["dump_depth"]) {
            dumpDepthFile = node["dump_depth"].as<std::string>();
        }
        if (node["dump_color"]) {
            dumpColorFile = node["dump_color"].as<std::string>();
        }
        if (node["dump_points"]) {
            dumpPointFile = node["dump_points"].as<std::string>();
        }
        if (node["enable_debug"]) {
            enableDebug = node["enable_debug"].as<bool>();
            if (enableDebug) {
                setenv("DS3D_ENABLE_DEBUG", "1", 1);
            }
        }

        if (!dumpDepthFile.empty()) {
            DS3D_FAILED_RETURN(
                depthWriter.open(dumpDepthFile), ErrCode::kConfig, "create depth file: %s failed",
                dumpDepthFile.c_str());
        }

        if (!dumpColorFile.empty()) {
            DS3D_FAILED_RETURN(
                colorWriter.open(dumpColorFile), ErrCode::kConfig, "create color file: %s failed",
                dumpColorFile.c_str());
        }
        if (!dumpPointFile.empty()) {
            DS3D_FAILED_RETURN(
                pointWriter.open(dumpPointFile), ErrCode::kConfig, "create point file: %s failed",
                dumpPointFile.c_str());
        }
        return ErrCode::kGood;
    }
};

class DepthCameraApp : public app::Ds3dAppContext {
public:
    DepthCameraApp() = default;
    ~DepthCameraApp() { deinit(); }
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
        if (_dataloaderSrc.customProcessor) {
            _dataloaderSrc.customProcessor.stop();
            _dataloaderSrc.gstElement.reset();
            _dataloaderSrc.customProcessor.reset();
        }

        if (_datarenderSink.customProcessor) {
            _datarenderSink.customProcessor.stop();
            _datarenderSink.gstElement.reset();
            _datarenderSink.customProcessor.reset();
        }
        ErrCode c = app::Ds3dAppContext::stop();
        return c;
    }

    void deinit() override
    {
        app::Ds3dAppContext::deinit();
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

static GstPadProbeReturn
appsrcBufferProbe(GstPad* pad, GstPadProbeInfo* info, gpointer udata)
{
    DepthCameraApp* appCtx = (DepthCameraApp*)udata;
    GstBuffer* buf = (GstBuffer*)info->data;
    ErrCode c = ErrCode::kGood;

    DS3D_UNUSED(c);
    DS_ASSERT(appCtx);
    AppProfiler& profiler = appCtx->profiler();

    if (!NvDs3D_IsDs3DBuf(buf)) {
        LOG_WARNING("appsrc buffer is not DS3D buffer");
    }
    const abiRefDataMap* refDataMap = nullptr;
    if (!isGood(NvDs3D_Find1stDataMap(buf, refDataMap))) {
        LOG_ERROR("didn't find datamap from GstBuffer, need to stop");
        if (appCtx->isRunning(5000)) {
            appCtx->sendEOS();
        }
        return GST_PAD_PROBE_DROP;
    }

    DS_ASSERT(refDataMap);
    GuardDataMap dataMap(*refDataMap);
    DS_ASSERT(dataMap);
    Frame2DGuard depthFrame;
    if (dataMap.hasData(kDepthFrame)) {
        DS3D_FAILED_RETURN(
            isGood(dataMap.getGuardData(kDepthFrame, depthFrame)), GST_PAD_PROBE_DROP,
            "get depthFrame failed");
        DS_ASSERT(depthFrame);
        DS_ASSERT(depthFrame->dataType() == DataType::kUint16);
        Frame2DPlane p = depthFrame->getPlane(0);
        DepthScale scale;
        c = dataMap.getData(kDepthScaleUnit, scale);
        DS_ASSERT(isGood(c));
        LOG_DEBUG(
            "depth frame is found, depth-scale: %.04f, w: %u, h: %u", scale.scaleUnit, p.width,
            p.height);
    }

    Frame2DGuard colorFrame;
    if (dataMap.hasData(kColorFrame)) {
        DS3D_FAILED_RETURN(
            isGood(dataMap.getGuardData(kColorFrame, colorFrame)), GST_PAD_PROBE_DROP,
            "get color Frame failed");
        DS_ASSERT(colorFrame);
        DS_ASSERT(colorFrame->dataType() == DataType::kUint8);
        Frame2DPlane p = colorFrame->getPlane(0);
        LOG_DEBUG("RGBA frame is found,  w: %d, h: %d", p.width, p.height);
    }

    // dump depth data for debug
    if (depthFrame && profiler.depthWriter.isOpen()) {
        DS_ASSERT(depthFrame->memType() != MemType::kGpuCuda);
        DS3D_FAILED_RETURN(
            profiler.depthWriter.write((const uint8_t*)depthFrame->base(), depthFrame->bytes()),
            GST_PAD_PROBE_DROP, "Dump depth data failed");
    }

    // dump color data for debug
    if (colorFrame && profiler.colorWriter.isOpen()) {
        DS_ASSERT(colorFrame->memType() != MemType::kGpuCuda);
        DS3D_FAILED_RETURN(
            profiler.colorWriter.write((const uint8_t*)colorFrame->base(), colorFrame->bytes()),
            GST_PAD_PROBE_DROP, "Dump color data failed");
    }

    return GST_PAD_PROBE_OK;
}

static GstPadProbeReturn
appsinkBufferProbe(GstPad* pad, GstPadProbeInfo* info, gpointer udata)
{
    DepthCameraApp* appCtx = (DepthCameraApp*)udata;
    GstBuffer* buf = (GstBuffer*)info->data;

    DS_ASSERT(appCtx);
    AppProfiler& profiler = appCtx->profiler();

    if (!NvDs3D_IsDs3DBuf(buf)) {
        LOG_WARNING("appsink buffer is not DS3D buffer");
    }
    const abiRefDataMap* refDataMap = nullptr;
    if (!isGood(NvDs3D_Find1stDataMap(buf, refDataMap))) {
        LOG_ERROR("didn't find datamap from GstBuffer, need to stop");
        if (appCtx->isRunning(5000)) {
            appCtx->sendEOS();
        }
        return GST_PAD_PROBE_DROP;
    }

    uint32_t numPoints = 0;
    DS_ASSERT(refDataMap);
    GuardDataMap dataMap(*refDataMap);
    DS_ASSERT(dataMap);

    FrameGuard pointFrame;
    if (dataMap.hasData(kPointXYZ)) {
        DS3D_FAILED_RETURN(
            isGood(dataMap.getGuardData(kPointXYZ, pointFrame)), GST_PAD_PROBE_DROP,
            "get pointXYZ frame failed.");
        DS_ASSERT(pointFrame);
        DS_ASSERT(pointFrame->dataType() == DataType::kFp32);
        DS_ASSERT(pointFrame->frameType() == FrameType::kPointXYZ);
        Shape pShape = pointFrame->shape();  // N x 3
        DS_ASSERT(pShape.numDims == 2 && pShape.d[1] == 3);  // PointXYZ
        numPoints = (size_t)pShape.d[0];
        LOG_DEBUG("pointcloudXYZ frame is found, points num: %u", numPoints);
    }

    FrameGuard colorCoord;
    if (dataMap.hasData(kPointCoordUV)) {
        DS3D_FAILED_RETURN(
            isGood(dataMap.getGuardData(kPointCoordUV, colorCoord)), GST_PAD_PROBE_DROP,
            "get PointCoordUV frame failed.");
        DS_ASSERT(colorCoord);
        DS_ASSERT(colorCoord->dataType() == DataType::kFp32);
        Shape cShape = colorCoord->shape();  // N x 2
        DS_ASSERT(cShape.numDims == 2 && cShape.d[1] == 2);  // PointColorCoord
        numPoints = (size_t)cShape.d[0];
        LOG_DEBUG("PointColorCoord frame is found,  points num: %u", numPoints);
    }

    // get depth & color intrinsic parameters, and also get depth-to-color extrinsic parameters
    IntrinsicsParam depthIntrinsics;
    IntrinsicsParam colorIntrinsics;
    ExtrinsicsParam d2cExtrinsics;  // rotation matrix is in the column-major order
    if (dataMap.hasData(kDepthIntrinsics)) {
        DS3D_FAILED_RETURN(
            isGood(dataMap.getData(kDepthIntrinsics, depthIntrinsics)), GST_PAD_PROBE_DROP,
            "get depth intrinsic parameters failed.");
        LOG_DEBUG(
            "DepthIntrinsics parameters is found, fx: %.4f, fy: %.4f", depthIntrinsics.fx,
            depthIntrinsics.fy);
    }
    if (dataMap.hasData(kColorIntrinsics)) {
        DS3D_FAILED_RETURN(
            isGood(dataMap.getData(kColorIntrinsics, colorIntrinsics)), GST_PAD_PROBE_DROP,
            "get color intrinsic parameters failed.");
        LOG_DEBUG(
            "ColorIntrinsics parameters is found, fx: %.4f, fy: %.4f", colorIntrinsics.fx,
            colorIntrinsics.fy);
    }
    if (dataMap.hasData(kDepth2ColorExtrinsics)) {
        DS3D_FAILED_RETURN(
            isGood(dataMap.getData(kDepth2ColorExtrinsics, d2cExtrinsics)), GST_PAD_PROBE_DROP,
            "get depth2color extrinsic parameters failed.");
        LOG_DEBUG(
            "depth2color extrinsic parameters is found, t:[%.3f, %.3f, %3.f]",
            d2cExtrinsics.translation.x, d2cExtrinsics.translation.y, d2cExtrinsics.translation.z);
    }

    // dump depth data for debug
    if (pointFrame && profiler.pointWriter.isOpen()) {
        DS_ASSERT(pointFrame->memType() != MemType::kGpuCuda);
        DS3D_FAILED_RETURN(
            profiler.pointWriter.write((const uint8_t*)pointFrame->base(), pointFrame->bytes()),
            GST_PAD_PROBE_DROP, "Dump points data failed");
    }

    return GST_PAD_PROBE_OK;
}

/** Set global gAppCtx */
static std::weak_ptr<DepthCameraApp> gAppCtx;

void static WindowClosed()
{
    LOG_DEBUG("Window closed.");
    auto appCtx = gAppCtx.lock();
    if (!appCtx) {
        return;
    }
    if (appCtx->isRunning(5000)) {
        appCtx->sendEOS();
    } else {
        appCtx->quitMainLoop();
    }
}

static void
help(const char* bin)
{
    printf("Usage: %s -c <deepstream-3d-depth-camera-config.txt>\n", bin);
}

/**
 * Function to handle program interrupt signal.
 * It installs default handler after handling the interrupt.
 */
static void
_intr_handler(int signum)
{
    LOG_INFO("User Interrupted..");

    ShrdPtr<DepthCameraApp> appCtx = gAppCtx.lock();
    if (appCtx) {
        if (appCtx->isRunning()) {
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

    ShrdPtr<DepthCameraApp> appCtx = std::make_shared<DepthCameraApp>();
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
    CHECK_ERROR(isGood(appCtx->init("deepstream-depth-camera-pipeline")), "init pipeline failed");

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
    code = CatchVoidCall([&loaderSrc, &renderSink, hasFilters, &configTable, appCtx]() {
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
                g_object_set(
                    G_OBJECT(filter.get()), "config-content", filterConfigs[i].rawContent.c_str(),
                    nullptr);
                appCtx->add(queue).add(filter);
                lastEle.link(queue).link(filter);
                lastEle = filter;
            }
        }
        lastEle.link(renderSink.gstElement);
    });
    CHECK_ERROR(isGood(code), "Link pipeline elements failed");

    /* Add probe to get informed of the meta data generated, we add probe to
     * gstappsrc src pad of the dataloader */
    gst::PadPtr srcPad = loaderSrc.gstElement.staticPad("src");
    CHECK_ERROR(srcPad, "appsrc src pad is not detected.");
    srcPad.addProbe(GST_PAD_PROBE_TYPE_BUFFER, appsrcBufferProbe, appCtx.get(), NULL);
    srcPad.reset();

    /* Add probe to get informed of the meta data generated, we add probe to
     * gstappsink sink pad of the datrender */
    if (renderSink.gstElement) {
        gst::PadPtr sinkPad = renderSink.gstElement.staticPad("sink");
        CHECK_ERROR(sinkPad, "appsink sink pad is not detected.");
        sinkPad.addProbe(GST_PAD_PROBE_TYPE_BUFFER, appsinkBufferProbe, appCtx.get(), NULL);
        sinkPad.reset();
    }

    CHECK_ERROR(isGood(appCtx->play()), "app context play failed");
    LOG_INFO("Play...");

    // get window system and set close event callback
    if (renderSink.customProcessor) {
        GuardWindow win = renderSink.customProcessor.getWindow();
        if (win) {
            GuardCB<abiWindow::CloseCB> windowClosedCb;
            windowClosedCb.setFn<>(WindowClosed);
            win->setCloseCallback(windowClosedCb.abiRef());
        }
    }

    /* Wait till pipeline encounters an error or EOS */
    appCtx->runMainLoop();

    loaderSrc.reset();
    renderSink.reset();
    appCtx->stop();
    appCtx->deinit();

    return 0;
}
