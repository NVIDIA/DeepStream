/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "deepstream_3d_sensor_fusion.hpp"

#include <unordered_set>

struct ComponentLoader : public Component {
    gst::DataLoaderSrc loaderSrc;
    gst::BinPtr bin;
};

struct ComponentRender : public Component {
    gst::DataRenderSink renderSink;
    gst::BinPtr bin;
};

struct ComponentFilter : public Component {
    gst::BinPtr bin;
};

struct ComponentParseBin : public Component {
    gst::BinPtr parseBin;
};

bool
Component::isGstPlugin() const
{
    using namespace config;
    if (config.type != ComponentType::kUserApp && config.type != ComponentType::kNone) {
        return true;
    }
    return false;
}


static GstPadProbeReturn
srcBufferProbe(GstPad* pad, GstPadProbeInfo* info, gpointer udata)
{
    SensorFusionApp* appCtx = (SensorFusionApp*)udata;
    GstBuffer* buf = (GstBuffer*)info->data;
    ErrCode c = ErrCode::kGood;

    DS3D_UNUSED(c);
    DS_ASSERT(appCtx);
    const auto& profiler = appCtx->profiler();
    DS3D_UNUSED(profiler);

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

    return GST_PAD_PROBE_OK;
}

static GstPadProbeReturn
sinkBufferProbe(GstPad* pad, GstPadProbeInfo* info, gpointer udata)
{
    SensorFusionApp* appCtx = (SensorFusionApp*)udata;
    GstBuffer* buf = (GstBuffer*)info->data;

    DS_ASSERT(appCtx);
    const auto& profiler = appCtx->profiler();
    DS3D_UNUSED(profiler);

    const abiRefDataMap* refDataMap = nullptr;
    if (!isGood(NvDs3D_Find1stDataMap(buf, refDataMap))) {
        LOG_DEBUG("didn't find datamap from GstBuffer, need to skip");
        return GST_PAD_PROBE_OK;
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


    return GST_PAD_PROBE_OK;
}

ErrCode
SensorFusionApp::setup(const std::string& configPath, std::function<void()> windowClosed)
{
    _configPath = configPath;
    std::string configContent;
    DS3D_FAILED_RETURN(
        readFile(configPath, configContent), ErrCode::kNotFound, "read file: %s failed",
        configPath.c_str());

    // parse all components in config file
    ConfigList componentConfigs;
    ErrCode code =
        CatchConfigCall(config::parseFullConfig, configContent, configPath, componentConfigs);
    DS3D_ERROR_RETURN(code, "parse config fille: %s failed", configPath.c_str());

    setMainloop(g_main_loop_new(NULL, FALSE));
    _windowClosedCb = std::move(windowClosed);
    // initilize pipeline
    DS3D_ERROR_RETURN(init("DeepStreamSensorFusionApp"), "init componets in pipeline failed");

    auto iteAppComp = std::find_if(
        componentConfigs.begin(), componentConfigs.end(), [this](const config::ComponentConfig& c) {
            return c.type == config::ComponentType::kUserApp;
        });
    if (iteAppComp != componentConfigs.end()) {
        // setup profiling
        auto& appComp = *iteAppComp;
        DS3D_ERROR_RETURN(_appProfiler.parse(appComp), "parsing profiling component failed");
    }
    DS3D_ERROR_RETURN(setupProfiling(), "setup profiling with config failed");

    // build all components for pipeline and application
    DS3D_ERROR_RETURN(
        buildComponents(componentConfigs), "build componets with config: %s failed",
        configPath.c_str());

    // link components in pipeline
    DS3D_ERROR_RETURN(LinkComponents(), "link componets in pipeline failed");

    return ErrCode::kGood;
}

ErrCode
SensorFusionApp::init(const std::string& name)
{
    return gst::PipelineContext::init(name);
}

ErrCode
SensorFusionApp::stop()
{
    ErrCode c = gst::PipelineContext::stop();
    return c;
}

void
SensorFusionApp::deinit()
{
    _components.clear();
    gst::PipelineContext::deinit();
}

ErrCode
SensorFusionApp::LinkComponents()
{
    std::set<std::tuple<GstElement*, GstElement*>> paired;
    for (auto i : _components) {
        auto comp = i.second;
        if (!comp->isGstPlugin()) {
            continue;
        }
        DS_ASSERT(comp->gstElement);
        if (!comp->config.linkTo.empty()) {
            auto isMultiInput = (comp->config.linkTo.find(".") != std::string::npos);
            auto linkTo =
                isMultiInput ? comp->config.linkTo.substr(0, comp->config.linkTo.find(".")) : comp->config.linkTo;
            DS3D_FAILED_RETURN(
                _components.count(linkTo), ErrCode::kConfig,
                "component: %s links to %s failed since it is not found in config", comp->config.name.c_str(),
                linkTo.c_str());
            auto to = _components[linkTo];
            DS3D_FAILED_RETURN(
                to->isGstPlugin(), ErrCode::kConfig,
                "component: %s links to %s failed since it is not a gst plugin compoment",
                comp->config.name.c_str(), comp->config.linkTo.c_str());
            DS_ASSERT(to->gstElement);
            auto hashkey = std::make_tuple(comp->gstElement.get(), to->gstElement.get());
            if (!paired.count(hashkey)) {
                if (isMultiInput) {
                    auto sinkPadName = comp->config.linkTo.substr(comp->config.linkTo.find(".") + 1);
                    LOG_DEBUG("multi-input downstream plugin [%s].[%s]", linkTo.c_str(), sinkPadName.c_str());
                    comp->gstElement.link(to->gstElement, sinkPadName);
                } else {
                    comp->gstElement.link(to->gstElement);
                }
                paired.insert(hashkey);
            }
        }

        if (!comp->config.linkFrom.empty()) {
            DS3D_FAILED_RETURN(
                _components.count(comp->config.linkFrom), ErrCode::kConfig,
                "component: %s links from %s failed since it is not found in config",
                comp->config.name.c_str(), comp->config.linkFrom.c_str());
            auto from = _components[comp->config.linkFrom];
            DS3D_FAILED_RETURN(
                from->isGstPlugin(), ErrCode::kConfig,
                "component: %s links from %s failed since it is not a gst plugin compoment",
                comp->config.name.c_str(), comp->config.linkFrom.c_str());
            DS_ASSERT(from->gstElement);
            auto hashkey = std::make_tuple(from->gstElement.get(), comp->gstElement.get());
            if (!paired.count(hashkey)) {
                from->gstElement.link(comp->gstElement);
                paired.insert(hashkey);
            }
        }
    }
    return ErrCode::kGood;
}

ErrCode
SensorFusionApp::buildComponents(const ConfigList& componentConfigs)
{
    using namespace ds3d::config;
    for (const auto& config : componentConfigs) {
        DS3D_FAILED_RETURN(
            !_components.count(config.name), ErrCode::kConfig,
            "config component: %s duplicate definition. Please check config.", config.name.c_str());

        Ptr<Component> comp;
        bool skip = false;
        switch (config.type) {
        case ComponentType::kDataLoader:
            comp = createLoader(config);
            break;
        case ComponentType::kDataRender:
            comp = createRender(config);
            break;
        case ComponentType::kDataFilter:
        case ComponentType::kDataBridge:
            comp = createFilterBridge(config);
            break;
        case ComponentType::kGstParseBin:
            comp = createGstParseBin(config);
            break;
        case ComponentType::kUserApp:
        case ComponentType::kNone:
            skip = true;
            break;
        case ComponentType::kDataMixer:
            comp = createMixer(config);
            break;
        default:
            skip = true;
            LOG_ERROR("skipped unsupported config type, content: %s", config.rawContent.c_str());
            break;
        }

        if (skip) {
            continue;
        }

        DS3D_FAILED_RETURN(comp, ErrCode::kNullPtr, "create component failed");
        _components[config.name] = comp;

        if (comp->gstElement) {
            add(comp->gstElement);
        }
    }

    return ErrCode::kGood;
}

ErrCode
SensorFusionApp::setupProfiling()
{
    const auto& p = profiler();
    if (p.enableDebug) {
        setenv("DS3D_ENABLE_DEBUG", "1", 1);
    } else {
        unsetenv("DS3D_ENABLE_DEBUG");
    }

    setEosAutoQuit(p.eosAutoStop);

    return ErrCode::kGood;
}

void static parsebinElementIndicate(gst::BinPtr bin, gst::ElePtr ele)
{
    GstElementFactory* factory = gst_element_get_factory(GST_ELEMENT(ele.get()));
    if (!factory) {
        return;
    }
    const char* factoryName = GST_OBJECT_NAME(factory);
    if (strcasecmp(factoryName, "uridecodebin") == 0 ||
        strcasecmp(factoryName, "nvurisrcbin") == 0) {
        // g_signal_connect(G_OBJECT(uri_decode_bin), "pad-added",G_CALLBACK(cb_newpad), bin);
    }
}

std::unique_ptr<Component>
SensorFusionApp::createGstParseBin(const config::ComponentConfig& c)
{
    YAML::Node node = YAML::Load(c.configBody);
    auto parseNode = node["parse_bin"];
    DS3D_FAILED_RETURN(
        parseNode, nullptr, "parse_bin is missing in config:\n%s.", c.rawContent.c_str());
    auto parseStr = parseNode.as<std::string>();
    GError* err = nullptr;
    GstElement* binRaw = gst_parse_bin_from_description(parseStr.c_str(), TRUE, &err);
    DS3D_FAILED_RETURN(
        binRaw && !err, nullptr, "creating parse bin failed with gst error: %s, with config:\n%s",
        ((err && err->message) ? err->message : ""), c.rawContent.c_str());
    gst_element_set_name(binRaw, (c.name + "_parsebin").c_str());
    gst::BinPtr bin(binRaw, true);
    DS_ASSERT(bin);

    GstIterator* itr = nullptr;
    GValue data = {
        0,
    };
    for (itr = gst_bin_iterate_elements(GST_BIN(binRaw));
         gst_iterator_next(itr, &data) == GST_ITERATOR_OK;) {
        gst::ElePtr iteEle(GST_ELEMENT_CAST(g_value_get_object(&data)), false);
        DS_ASSERT(iteEle);
        parsebinElementIndicate(bin, iteEle);
        g_value_reset(&data);
    }
    if(itr) {
        gst_iterator_free(itr);
    }

    gst::ElePtr decodebin = gst_bin_get_by_name(GST_BIN(binRaw), "video_source");

    auto parseBin = std::make_unique<ComponentParseBin>();
    parseBin->config = c;
    parseBin->gstElement = bin;
    parseBin->parseBin = bin;
    return parseBin;
}

std::unique_ptr<Component>
SensorFusionApp::createFilterBridge(const config::ComponentConfig& c)
{
    auto filter = std::make_unique<ComponentFilter>();
    DS_ASSERT(filter);
    gst::ElePtr plugin;
    if (c.type == config::ComponentType::kDataFilter) {
        plugin = gst::elementMake(gst::kDs3dFilterPluginName, (c.name + "_filter").c_str());
    } else if (c.type == config::ComponentType::kDataBridge) {
        plugin = gst::elementMake(gst::kDs3dBridgePluginName, (c.name + "_bridge").c_str());
    } else {
        assert(false);
    }
    DS3D_THROW_ERROR_FMT(
        plugin, ErrCode::kGst, "gst-plugin: %s is not found", gst::kDs3dFilterPluginName);
    g_object_set(G_OBJECT(plugin.get()), "config-content", c.rawContent.c_str(), nullptr);

    if (profiler().probeBuffer) {
        gst::PadPtr sinkPad = plugin.staticPad("sink");
        DS3D_FAILED_RETURN(sinkPad, nullptr, "sink pad is not detected from %s", c.name.c_str());
        sinkPad.addProbe(GST_PAD_PROBE_TYPE_BUFFER, sinkBufferProbe, this, NULL);

        gst::PadPtr srcPad = plugin.staticPad("src");
        DS3D_FAILED_RETURN(srcPad, nullptr, "src pad is not detected from %s", c.name.c_str());
        srcPad.addProbe(GST_PAD_PROBE_TYPE_BUFFER, srcBufferProbe, this, NULL);
    }

    filter->config = c;
    filter->gstElement = plugin;

    if (!c.withQueue.empty()) {
        gst::BinPtr bin(gst_bin_new((c.name + "_filter_bin").c_str()));
        bin.pushBack(plugin);
        gst::ElePtr q;
        if (c.withQueue == "src") {
            q = bin.addSrcQueue();
        } else if (c.withQueue == "sink") {
            q = bin.addSinkQueue();
        } else {
            LOG_ERROR("with_queue value is unsupported in: \n%s", c.rawContent.c_str());
            return nullptr;
        }
        DS3D_FAILED_RETURN(q, nullptr, "Adding src/sink queue into filter failed.");
        DS3D_FAILED_RETURN(
            isGood(bin.addGhostSinkPad()), nullptr, "Adding sink ghost pad into filter bin failed");
        DS3D_FAILED_RETURN(
            isGood(bin.addGhostSrcPad()), nullptr, "Adding src ghost pad into filter bin failed");
        filter->gstElement = bin;
        filter->bin = bin;
    }

    return filter;
}

std::unique_ptr<Component>
SensorFusionApp::createMixer(const config::ComponentConfig& c)
{
    auto mixer = std::make_unique<ComponentFilter>();
    DS_ASSERT(mixer);
    gst::ElePtr plugin = gst::elementMake(gst::kDs3dMixerPluginName, (c.name + "_mixer").c_str());
    DS3D_THROW_ERROR_FMT(plugin, ErrCode::kGst, "gst-plugin: %s is not found", gst::kDs3dMixerPluginName);
    g_object_set(G_OBJECT(plugin.get()), "config-content", c.rawContent.c_str(), nullptr);

#if 0
    if (profiler().probeBuffer) {
        gst::PadPtr sinkPad = plugin.staticPad("sink");
        DS3D_FAILED_RETURN(sinkPad, nullptr, "sink pad is not detected from %s", c.name.c_str());
        sinkPad.addProbe(GST_PAD_PROBE_TYPE_BUFFER, sinkBufferProbe, this, NULL);

        gst::PadPtr srcPad = plugin.staticPad("src");
        DS3D_FAILED_RETURN(srcPad, nullptr, "src pad is not detected from %s", c.name.c_str());
        srcPad.addProbe(GST_PAD_PROBE_TYPE_BUFFER, srcBufferProbe, this, NULL);
    }
#endif

    mixer->config = c;
    mixer->gstElement = plugin;

    if (!c.withQueue.empty()) {
        gst::BinPtr bin(gst_bin_new((c.name + "_mixer_bin").c_str()));
        bin.pushBack(plugin);
        gst::ElePtr q;
        if (c.withQueue == "src") {
            q = bin.addSrcQueue();
        } else {
            LOG_ERROR("with_queue value is unsupported in: \n%s", c.rawContent.c_str());
            return nullptr;
        }
        DS3D_FAILED_RETURN(q, nullptr, "Adding src/sink queue into mixer failed.");
        DS3D_FAILED_RETURN(isGood(bin.addGhostSrcPad()), nullptr, "Adding src ghost pad into mixer bin failed");
        mixer->gstElement = bin;
        mixer->bin = bin;
    }

    return mixer;
}

std::unique_ptr<Component>
SensorFusionApp::createLoader(const config::ComponentConfig& c)
{
    // creat appsrc and dataloader
    auto loader = std::make_unique<ComponentLoader>();
    DS_ASSERT(loader);

    DS3D_FAILED_RETURN(
        isGood(NvDs3D_CreateDataLoaderSrc(c, loader->loaderSrc, true)), nullptr,
        "Create dataloader src failed");
    DS_ASSERT(loader->loaderSrc.gstElement);
    DS_ASSERT(loader->loaderSrc.customProcessor);

    if (profiler().probeBuffer) {
        gst::PadPtr srcPad = loader->loaderSrc.gstElement.staticPad("src");
        DS3D_FAILED_RETURN(srcPad, nullptr, "src pad is not detected from %s", c.name.c_str());
        srcPad.addProbe(GST_PAD_PROBE_TYPE_BUFFER, srcBufferProbe, this, NULL);
    }

    gst::BinPtr bin(gst_bin_new(c.name.c_str()));
    bin.pushBack(loader->loaderSrc.gstElement);

    if (c.withQueue == "src") {
        DS3D_FAILED_RETURN(
            bin.addSrcQueue(true), nullptr, "Failed to add src queue into loader bin");
    }

    DS3D_FAILED_RETURN(
        isGood(bin.addGhostSrcPad()), nullptr, "Failed to add ghost src pad into loader bin");

    loader->config = c;
    loader->gstElement = bin;
    loader->bin = bin;
    return loader;
}

std::unique_ptr<Component>
SensorFusionApp::createRender(const config::ComponentConfig& c)
{
    // creat appsink and datarender
    auto render = std::make_unique<ComponentRender>();
    DS_ASSERT(render);

    DS3D_FAILED_RETURN(
        isGood(NvDs3D_CreateDataRenderSink(c, render->renderSink, true)), nullptr,
        "Create datarender sink failed");
    DS_ASSERT(render->renderSink.gstElement);
    DS_ASSERT(render->renderSink.customProcessor);

    if (profiler().probeBuffer) {
        gst::PadPtr sinkPad = render->renderSink.gstElement.staticPad("sink");
        DS3D_FAILED_RETURN(sinkPad, nullptr, "sink pad is not detected from %s", c.name.c_str());
        sinkPad.addProbe(GST_PAD_PROBE_TYPE_BUFFER, sinkBufferProbe, this, NULL);
    }

    if (render->renderSink.customProcessor) {
        GuardWindow win = render->renderSink.customProcessor.getWindow();
        if (win && _windowClosedCb) {
            GuardCB<abiWindow::CloseCB> windowClosedCb;
            windowClosedCb.setFn<>(_windowClosedCb);
            win->setCloseCallback(windowClosedCb.abiRef());
        }
    }

    gst::BinPtr bin(gst_bin_new(c.name.c_str()));
    bin.pushBack(render->renderSink.gstElement);

    if (c.withQueue == "sink") {
        DS3D_FAILED_RETURN(
            bin.addSinkQueue(true), nullptr, "Failed to add sink queue into render bin");
    }

    DS3D_FAILED_RETURN(
        isGood(bin.addGhostSinkPad()), nullptr, "Failed to add ghost sink pad into render bin");

    render->config = c;
    render->gstElement = bin;
    render->bin = bin;
    return render;
}
