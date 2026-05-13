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
#include "deepstream_can_context_priv.hpp"


#define MAX_STR_LEN 2048

/* By default, OSD process-mode is set to GPU_MODE. To change mode, set as:
 * 0: CPU mode
 * 1: GPU mode
 */
#define OSD_PROCESS_MODE 1

/* By default, OSD will not display text. To display text, change this to 1 */
#define OSD_DISPLAY_TEXT 1


using namespace ds3d;

static void
cb_newpad(GstElement* decodebin, GstPad* decoder_src_pad, gpointer data)
{
    g_print("In cb_newpad\n");
    GstCaps* caps = gst_pad_get_current_caps(decoder_src_pad);
    if (!caps) {
        return;
    }
    const GstStructure* str = gst_caps_get_structure(caps, 0);
    if (!str) {
        return;
    }
    const gchar* name = gst_structure_get_name(str);
    if (!name) {
        return;
    }
    GstElement* source_bin = (GstElement*)data;
    GstCapsFeatures* features = gst_caps_get_features(caps, 0);

    /* Need to check if the pad created by the decodebin is for video and not
     * audio. */
    if (!strncmp(name, "video", 5)) {
        /* Link the decodebin pad only if decodebin has picked nvidia
         * decoder plugin nvdec_*. We do this by checking if the pad caps contain
         * NVMM memory features. */
        if (gst_caps_features_contains(features, GST_CAPS_FEATURES_NVMM)) {
            /* Get the source bin ghost pad */
            GstPad* bin_ghost_pad = gst_element_get_static_pad(source_bin, "src");
            if (!gst_ghost_pad_set_target(GST_GHOST_PAD(bin_ghost_pad), decoder_src_pad)) {
                g_printerr("Failed to link decoder src pad to source bin ghost pad\n");
            }
            gst_object_unref(bin_ghost_pad);
        } else {
            g_printerr("Error: Decodebin did not pick nvidia decoder plugin.\n");
        }
    }
}

static void
decodebin_child_added(GstChildProxy* child_proxy, GObject* object, gchar* name, gpointer user_data)
{
    g_print("Decodebin child added: %s\n", name);
    CameraCanApp* app = (CameraCanApp*)user_data;
    const NvDsCanContextConfig& config = app->config();
    DS_ASSERT(app);
    if (g_strrstr(name, "decodebin") == name) {
        g_signal_connect(
            G_OBJECT(object), "child-added", G_CALLBACK(decodebin_child_added), user_data);
        return;
    }

    GstElementFactory* factory = gst_element_get_factory(GST_ELEMENT(object));
    if (!factory) {
        return;
    }
    const char* factoryName = GST_OBJECT_NAME(factory);


    if (strcasecmp(factoryName, "multifilesrc") == 0 && !config.srcFrameRate.empty()) {
        std::string uri = config.srcUri;
        auto endsWith = [](const std::string& str, const char* ends) {
            int pos = (int)str.length() - (int)strlen(ends);
            if (pos < 0) {
                return false;
            }
            return strncasecmp(&str[pos], ends, strlen(ends)) == 0;
        };
        std::string format;
        if (endsWith(uri, ".jpeg") || endsWith(uri, ".jpg")) {
            format = "image/jpeg";
        } else if (endsWith(uri, ".png")) {
            format = "image/png";
        }
        if (format.empty()) {
            return;
        }
        format += ",framerate=(fraction)" + config.srcFrameRate;
        gst::CapsPtr caps(gst_caps_from_string(format.c_str()));
        DS_ASSERT(caps);
        g_object_set(G_OBJECT(object), "caps", caps.get(), NULL);
    }
}

gst::ElePtr
CameraCanApp::createSourceBin(const std::string& uri)
{
    gst::ElePtr bin, decodeBin;
    std::stringstream ss;
    static uint32_t srcId = 0;

    ss << "source-bin-" << srcId++;
    auto binName = ss.str();
    bin.reset(gst_bin_new(binName.c_str()));
    decodeBin.reset(gst_element_factory_make("uridecodebin", "uri-decode-bin"));

    DS3D_FAILED_RETURN(bin && decodeBin, nullptr, "failed to create decode bin");

    /* We set the input uri to the source element */
    g_object_set(G_OBJECT(decodeBin.get()), "uri", uri.c_str(), NULL);

    /* Connect to the "pad-added" signal of the decodebin which generates a
     * callback once a new pad for raw data has beed created by the decodebin */
    g_signal_connect(G_OBJECT(decodeBin.get()), "pad-added", G_CALLBACK(cb_newpad), bin.get());
    g_signal_connect(
        G_OBJECT(decodeBin.get()), "child-added", G_CALLBACK(decodebin_child_added), this);

    gst_bin_add(GST_BIN(bin.get()), decodeBin.copy());

    DS3D_FAILED_RETURN(
        gst_element_add_pad(bin.get(), gst_ghost_pad_new_no_target("src", GST_PAD_SRC)), nullptr,
        "Failed to add ghost pad in source bin");

    return bin;
}


static void
AppsrcFillData(GstAppSrc* src, guint length, gpointer userData)
{
    return;
}

static void
AppsrcEnoughData(GstAppSrc* src, gpointer user_data)
{
    return;
}

static gboolean
AppsrcSeekData(GstAppSrc* src, guint64 offset, gpointer user_data)
{
    return FALSE;
}


gst::ElePtr
CameraCanApp::createAppSrc(const NvDsCanFrameInfo& info)
{
    gst::ElePtr src;
    src.reset(gst_element_factory_make("appsrc", "app-source"));
    DS3D_FAILED_RETURN(src, nullptr, "failed to create appsrc");

    const char* format = nullptr;
    if (info.fourcc == NvDsCanFormatRGBA) {
        format = "RGBA";
    } else if (info.fourcc == NvDsCanFormatGREY) {
        format = "GRAY8";
    }

    std::string strFramerate = "60/1";
    if (!config().srcFrameRate.empty()) {
        strFramerate = config().srcFrameRate;
    }

    char strCaps[MAX_STR_LEN] = {0};
    snprintf(
        strCaps, MAX_STR_LEN - 1, "video/x-raw,format=%s,width=%u,height=%u,framerate=%s", format,
        info.width, info.height, strFramerate.c_str());
    gst::CapsPtr caps(gst_caps_from_string(strCaps));
    DS_ASSERT(caps);
    g_object_set(G_OBJECT(src.get()), "caps", caps.get(), NULL);

    GstAppSrcCallbacks callbacks = {
        AppsrcFillData,
        AppsrcEnoughData,
        AppsrcSeekData,
    };
    gst_app_src_set_callbacks((GstAppSrc*)src.get(), &callbacks, this, NULL);

    return src;
}

gst::ElePtr
CameraCanApp::createCameraSrc(const NvDsCanCameraInfo& conf)
{
    gst::ElePtr bin, src, pylonCapsFilter, q0;
    bin.reset(gst_bin_new("basler_source_bin"));
    src.reset(gst_element_factory_make("pylonsrc", "pylonsrc"));
    pylonCapsFilter.reset(gst_element_factory_make("capsfilter", "pylonCaps"));
    q0.reset(gst_element_factory_make("queue", "pylonqueue"));
    DS3D_FAILED_RETURN(bin && src && pylonCapsFilter, nullptr, "failed to create pylonsrc bin");

    g_object_set(
        G_OBJECT(src.get()), "device-serial-number", conf.devSN.c_str(), "capture-error", 1, NULL);
    if (!conf.pfsPath.empty()) {
        g_object_set(G_OBJECT(src.get()), "pfs-location", conf.pfsPath.c_str(), NULL);
    }

    std::stringstream ss;
    ss << "video/x-raw, format=" << conf.format;
    if (conf.width) {
        ss << ", width=" << conf.width;
    }
    if (conf.height) {
        ss << ", height=" << conf.height;
    }
    if (!config().srcFrameRate.empty()) {
        ss << ", framerate=" << config().srcFrameRate;
    }
    std::string capStr = ss.str();
    gst::CapsPtr pylonCaps(gst_caps_from_string(capStr.c_str()));
    g_object_set(G_OBJECT(pylonCapsFilter.get()), "caps", pylonCaps.get(), NULL);

    gst_bin_add_many(GST_BIN(bin.get()), src.copy(), pylonCapsFilter.copy(), q0.copy(), NULL);
    ErrCode c = CatchVoidCall(
        [this, &src, &pylonCapsFilter, &q0]() mutable { src.link(pylonCapsFilter).link(q0); });
    DS3D_FAILED_RETURN(isGood(c), nullptr, "link camera src bin failed");

    gst::PadPtr srcPad = q0.staticPad("src");
    DS_ASSERT(srcPad);
    DS3D_FAILED_RETURN(
        gst_element_add_pad(bin.get(), gst_ghost_pad_new("src", srcPad.get())), nullptr,
        "Failed to add ghost pad in camera source bin");

    return bin;
}

GstPadProbeReturn
CameraCanApp::lastSinkBufferProbe(GstPad* pad, GstPadProbeInfo* info, gpointer udata)
{
    CameraCanApp* appCtx = (CameraCanApp*)udata;
    GstBuffer* buf = (GstBuffer*)info->data;
    static uint32_t idx = 0;
    LOG_DEBUG("sink received buf: %u", ++idx);

    DS_ASSERT(appCtx);
    DS3D_UNUSED(buf);
    appCtx->outputBuffer(buf);

    return GST_PAD_PROBE_OK;
}

GstPadProbeReturn
CameraCanApp::processedBufferProbe(GstPad* pad, GstPadProbeInfo* info, gpointer udata)
{
    CameraCanApp* appCtx = (CameraCanApp*)udata;
    GstBuffer* buf = (GstBuffer*)info->data;
    static uint32_t idx = 0;
    LOG_DEBUG("received buf: %u", ++idx);

    DS_ASSERT(appCtx);
    DS3D_UNUSED(buf);
    appCtx->visualMetaUpdate(buf);

    return GST_PAD_PROBE_OK;
}

void
CameraCanApp::visualMetaUpdate(GstBuffer* buf)
{
    DS_ASSERT(buf);
    NvDsBatchMeta* batchMeta = gst_buffer_get_nvds_batch_meta(buf);
    /* Iterate frame metadata in batches */
    for (NvDsMetaList* l_frame = batchMeta->frame_meta_list; l_frame != nullptr;
         l_frame = l_frame->next) {
        NvDsFrameMeta* frame = (NvDsFrameMeta*)l_frame->data;
        NvDsCanContextResultMeta* res = nullptr;

        for (NvDsMetaList* l_user = frame->frame_user_meta_list; l_user != nullptr;
             l_user = l_user->next) {
            NvDsUserMeta* userMeta = (NvDsUserMeta*)l_user->data;
            if (userMeta->base_meta.meta_type != kNvDsCanResultMeta)
                continue;
            res = (NvDsCanContextResultMeta*)userMeta->user_meta_data;
            break;
        }
        if (res) {
            NvDsDisplayMeta* display = nvds_acquire_display_meta_from_pool(batchMeta);
            display->num_labels = 1;
            NvOSD_TextParams* txt_params = &display->text_params[0];
            txt_params->display_text = (char*)g_malloc0(MAX_STR_LEN);

            snprintf(txt_params->display_text, MAX_STR_LEN - 1, "yaw: %.03f", res->rotation);
            /* Now set the offsets where the string should appear */
            txt_params->x_offset = 0;
            txt_params->y_offset = 40;

            /* Font , font-color and font-size */
            txt_params->font_params.font_name = (char*)"Serif";
            txt_params->font_params.font_size = 10;
            txt_params->font_params.font_color.red = 1.0;
            txt_params->font_params.font_color.green = 1.0;
            txt_params->font_params.font_color.blue = 1.0;
            txt_params->font_params.font_color.alpha = 1.0;

            /* Text background color */
            txt_params->set_bg_clr = 1;
            txt_params->text_bg_clr.red = 0.0;
            txt_params->text_bg_clr.green = 0.0;
            txt_params->text_bg_clr.blue = 0.0;
            txt_params->text_bg_clr.alpha = 1.0;

            nvds_add_display_meta_to_frame(frame, display);
        }
    }
}

static bool
getResultFromBuf(GstBuffer* buf, NvDsCanContextResultMeta& res)
{
    DS_ASSERT(buf);
    NvDsBatchMeta* batchMeta = gst_buffer_get_nvds_batch_meta(buf);
    /* Iterate frame metadata in batches */
    for (NvDsMetaList* l_frame = batchMeta->frame_meta_list; l_frame != nullptr;
         l_frame = l_frame->next) {
        NvDsFrameMeta* frame_meta = (NvDsFrameMeta*)l_frame->data;

        /* Iterate user metadata in frames to search result metadata */
        for (NvDsMetaList* l_user = frame_meta->frame_user_meta_list; l_user != nullptr;
             l_user = l_user->next) {
            NvDsUserMeta* userMeta = (NvDsUserMeta*)l_user->data;
            if (userMeta->base_meta.meta_type != kNvDsCanResultMeta)
                continue;
            /* convert to tensor metadata */
            NvDsCanContextResultMeta* meta = (NvDsCanContextResultMeta*)userMeta->user_meta_data;
            res = *meta;
            return true;
        }
    }
    return false;
}

void
CameraCanApp::printOutputResult(GstBuffer* buf)
{
    NvDsCanContextResultMeta res{0.0f};
    if (!getResultFromBuf(buf, res)) {
        LOG_ERROR("get can context result from output buffer failed");
    }

    LOG_INFO("output res: buf(%" PRIu64 "), yaw: %.04f", _bufId, res.rotation);
}

void
CameraCanApp::outputBuffer(GstBuffer* buf)
{
    std::unique_lock<std::mutex> locker(_streamMutex);
    if (_outputCallback) {
        _outputCallback(buf);
    }
    locker.unlock();
    float fps = _fpsCalculator.updateFps(0);
    if (_bufId && (_bufId % DS_CAN_FPS_INTERVAL) == 0) {
        LOG_INFO("Current FPS: %.02f", fps);
    }
    if (_config.printRes) {
        printOutputResult(buf);
    }
    ++_bufId;
}

static ErrCode
parseConfig(const std::string& content, NvDsCanContextConfig& c)
{
    auto node = YAML::Load(content);

    auto src = node["source"];
    auto mux = node["streammux"];
    auto videotemplate = node["video_template"];
    auto sink = node["sink"];
    auto debug = node["debug"];
    if (src["frame_settings"]) {
        c.srcType = NvDsCanSrcType::kFrame;
        const auto& frameNode = src["frame_settings"];
        std::string format = frameNode["format"].as<std::string>();
        if (format == "GREY") {
            c.srcFrameInfo.fourcc = NvDsCanFormatGREY;
        } else if (format == "RGBA") {
            c.srcFrameInfo.fourcc = NvDsCanFormatRGBA;
        } else {
            LOG_ERROR("config file src_frame_info has unsupported format: %s", format.c_str());
            return ErrCode::kConfig;
        }
        c.srcFrameInfo.width = frameNode["width"].as<uint32_t>();
        c.srcFrameInfo.height = frameNode["height"].as<uint32_t>();
        DS_ASSERT(c.srcFrameInfo.width);
        DS_ASSERT(c.srcFrameInfo.height);
    } else if (src["uri"]) {
        c.srcType = NvDsCanSrcType::kUri;
        c.srcUri = src["uri"].as<std::string>();
    } else if (src["camera"]) {
        c.srcType = NvDsCanSrcType::kBaslerCamera;
        const auto& camNode = src["camera"];
        c.cameraInfo.devSN = camNode["device_serial_number"].as<std::string>();
        if (camNode["pfs_path"]) {
            c.cameraInfo.pfsPath = camNode["pfs_path"].as<std::string>();
        }
        if (camNode["format"]) {
            c.cameraInfo.format = camNode["format"].as<std::string>();
        }
        if (camNode["height"]) {
            c.cameraInfo.height = camNode["height"].as<uint32_t>();
        }
        if (camNode["width"]) {
            c.cameraInfo.width = camNode["width"].as<uint32_t>();
        }
    }
    if (src["framerate"]) {
        c.srcFrameRate = src["framerate"].as<std::string>();
    }

    if (node["infer_config"]) {
        c.inferConfig = node["infer_config"].as<std::string>();
    }

    if (mux["width"]) {
        c.muxWidth = mux["width"].as<uint32_t>();
    }
    if (mux["height"]) {
        c.muxHeight = mux["height"].as<uint32_t>();
    }
    if (mux["batched_push_timeout"]) {
        c.muxBatchTimeout = mux["batched_push_timeout"].as<int32_t>();
    }

    if (videotemplate["customlib_name"]) {
        c.templateCustomLibName = videotemplate["customlib_name"].as<std::string>();
    }

    if (videotemplate["customlib_props"]) {
        c.templateCustomLibProps = videotemplate["customlib_props"].as<std::string>();
    }

    if (sink["egl_display"]) {
        c.enableEglSink = sink["egl_display"].as<bool>();
    }
    if (sink["sync"]) {
        c.syncDisplay = sink["sync"].as<bool>();
    }

    if (debug["debug_level"]) {
        c.debugLevel = debug["debug_level"].as<uint32_t>();
    }
    if (debug["print_result"]) {
        c.printRes = debug["print_result"].as<bool>();
    }

    return ErrCode::kGood;
}

gst::ElePtr
CameraCanApp::createSink()
{
    gst::ElePtr sink;
    gboolean sync = _config.syncDisplay;
    if (_config.enableEglSink) {
        gst::ElePtr bin, q0, q1, tiler, conv, osd, osdCapFilter, eglsink;
        bin.reset(gst_bin_new("eglsink_bin"));

        q0.reset(gst_element_factory_make("queue", "queue_tile"));
        q1.reset(gst_element_factory_make("queue", "queue_osd"));
        tiler.reset(gst_element_factory_make("nvmultistreamtiler", "nvtiler"));
        conv.reset(gst_element_factory_make("nvvideoconvert", "conv_tiler"));
        osd.reset(gst_element_factory_make("nvdsosd", "nv_osd"));
        osdCapFilter.reset(gst_element_factory_make("capsfilter", "osdCaps"));
#ifdef __aarch64__
        eglsink.reset(gst_element_factory_make("nv3dsink", "nv3d-sink"));
#else
        eglsink.reset(gst_element_factory_make("nveglglessink", "nvvideo-renderer"));
#endif

        DS_ASSERT(q0 && q1);
        DS_ASSERT(tiler);
        DS_ASSERT(conv);
        DS_ASSERT(osd);
        DS_ASSERT(osdCapFilter);
        DS_ASSERT(eglsink);

        const char* osdFmt = "RGBA";
        gst::CapsPtr osdCaps(
            gst_caps_new_simple("video/x-raw", "format", G_TYPE_STRING, osdFmt, NULL));
        gst_caps_set_features(osdCaps.get(), 0, gst_caps_features_new("memory:NVMM", NULL));
        g_object_set(G_OBJECT(osdCapFilter.get()), "caps", osdCaps.get(), NULL);

#ifdef PLATFORM_TEGRA
        g_object_set(G_OBJECT(conv.get()), "nvbuf-memory-type", 0, "compute-hw", 1, NULL);
        g_object_set(G_OBJECT(tiler.get()), "nvbuf-memory-type", 2, "compute-hw", 1, NULL);
#endif

        g_object_set(
            G_OBJECT(tiler.get()), "rows", 1, "columns", 1, "width", 1920, "height", 1080, NULL);
        g_object_set(
            G_OBJECT(osd.get()), "process-mode", OSD_PROCESS_MODE, "display-text", OSD_DISPLAY_TEXT,
            NULL);
        g_object_set(G_OBJECT(eglsink.get()), "qos", 0, "sync", sync, NULL);


        gst_bin_add_many(
            GST_BIN(bin.get()), q0.copy(), q1.copy(), tiler.copy(), conv.copy(),
            osdCapFilter.copy(), osd.copy(), eglsink.copy(), NULL);

        ErrCode c =
            CatchVoidCall([this, &q0, &q1, &tiler, &conv, &osdCapFilter, &osd, &eglsink]() mutable {
                tiler.link(q0).link(conv).link(osdCapFilter).link(q1).link(osd).link(eglsink);
            });
        DS3D_FAILED_RETURN(isGood(c), nullptr, "link egl sink failed");
        gst::PadPtr sinkPad = tiler.staticPad("sink");
        DS_ASSERT(sinkPad);
        gst_element_add_pad(bin.get(), gst_ghost_pad_new("sink", sinkPad.get()));
        sink = bin;
    } else {
        sink.reset(gst_element_factory_make("fakesink", "fakesink"));
        g_object_set(G_OBJECT(sink.get()), "qos", 0, "sync", sync, NULL);
    }
    return sink;
}

ErrCode
CameraCanApp::buildPipeline()
{
    gst::ElePtr src, mx, match, sink, q1, q2, vidconv0, conv0Filter;

    switch (_config.srcType) {
    case NvDsCanSrcType::kUri:
        src = createSourceBin(_config.srcUri);
        DS3D_FAILED_RETURN(src, ErrCode::kNotFound, "create source bin failed");
        break;
    case NvDsCanSrcType::kFrame:
        src = createAppSrc(_config.srcFrameInfo);
        DS3D_FAILED_RETURN(src, ErrCode::kNotFound, "create app src failed");
        break;
    case NvDsCanSrcType::kBaslerCamera:
        src = createCameraSrc(_config.cameraInfo);
        DS3D_FAILED_RETURN(src, ErrCode::kNotFound, "create camera source failed");
        break;
    default:
        DS3D_FAILED_RETURN(false, ErrCode::kConfig, "unknow source types");
        break;
    }
    _src = src;

    vidconv0.reset(gst_element_factory_make("nvvideoconvert", "nvvideoconvert0"));
    conv0Filter.reset(gst_element_factory_make("capsfilter", "convert0Caps"));
    mx.reset(gst_element_factory_make("nvstreammux", "stream-muxer"));
    match.reset(gst_element_factory_make("nvdsvideotemplate", "template_match"));
    q1.reset(gst_element_factory_make("queue", "queue1"));
    q2.reset(gst_element_factory_make("queue", "queue2"));
    sink = createSink();
    DS3D_FAILED_RETURN(sink, ErrCode::kGst, "create sink failed");

    DS_ASSERT(vidconv0);
    DS_ASSERT(conv0Filter);
    DS_ASSERT(mx);
    DS_ASSERT(match);
    DS_ASSERT(q1);
    DS_ASSERT(sink);
#ifdef PLATFORM_TEGRA
    g_object_set(G_OBJECT(vidconv0.get()), "nvbuf-memory-type", 2, "compute-hw", 1, NULL);
#else
    g_object_set(G_OBJECT(vidconv0.get()), "nvbuf-memory-type", 2, NULL);
#endif

    // convert to NV12 since videotemplate does not have GRAY8 format
    const char* vidconvFmt = "NV12";
    gst::CapsPtr vidconv0Caps(
        gst_caps_new_simple("video/x-raw", "format", G_TYPE_STRING, vidconvFmt, NULL));
    gst_caps_set_features(vidconv0Caps.get(), 0, gst_caps_features_new("memory:NVMM", NULL));
    g_object_set(G_OBJECT(conv0Filter.get()), "caps", vidconv0Caps.get(), NULL);

    const gchar* envNewStreamMux = g_getenv("USE_NEW_NVSTREAMMUX");
    bool useNewMux = !g_strcmp0(envNewStreamMux, "yes");
    if (useNewMux) {
        g_object_set(G_OBJECT(mx.get()), "batch-size", 1, NULL);
    } else {
        g_object_set(
            G_OBJECT(mx.get()), "batch-size", 1, "width", _config.muxWidth, "height",
            _config.muxHeight, "batched-push-timeout", _config.muxBatchTimeout, NULL);
    }

    g_object_set(
        G_OBJECT(match.get()), "customlib-name", _config.templateCustomLibName.c_str(),
        "customlib-props", _config.templateCustomLibProps.c_str(), NULL);

    ErrCode c = CatchVoidCall([this, &src, &mx, &match, &sink, &q1, &q2, &vidconv0,
                               &conv0Filter]() mutable {
        this->add(src).add(vidconv0).add(conv0Filter).add(mx).add(q1).add(match).add(q2).add(sink);
        gst::PadPtr srcPad(gst_element_get_static_pad(conv0Filter, "src"));
        std::string padName("sink_0");
        gst::PadPtr sinkPad(gst_element_request_pad_simple(mx, padName.c_str()));
        DS_ASSERT(srcPad && sinkPad);
        DS3D_THROW_ERROR(
            gst_pad_link(srcPad.get(), sinkPad.get()) == GST_PAD_LINK_OK, ErrCode::kGst,
            "link sourc and mux failed.");
        src.link(vidconv0).link(conv0Filter);
        mx.link(q1).link(match).link(q2).link(sink);
    });
    DS3D_ERROR_RETURN(c, "build pipeline failed");

    gst::PadPtr matchSrcPad = match.staticPad("src");
    DS3D_FAILED_RETURN(matchSrcPad.get(), ErrCode::kGst, "match's src pad is not loacted.");
    matchSrcPad.addProbe(GST_PAD_PROBE_TYPE_BUFFER, CameraCanApp::processedBufferProbe, this, NULL);

    gst::PadPtr lastSinkPad = sink.staticPad("sink");
    DS3D_FAILED_RETURN(lastSinkPad.get(), ErrCode::kGst, "sink's sink pad is not detected.");
    lastSinkPad.addProbe(GST_PAD_PROBE_TYPE_BUFFER, CameraCanApp::lastSinkBufferProbe, this, NULL);

    return ErrCode::kGood;
}

bool
CameraCanApp::busCall(GstMessage* msg)
{
    DS_ASSERT(mainLoop());
    switch (GST_MESSAGE_TYPE(msg)) {
    case GST_MESSAGE_EOS:
        LOG_INFO("End of stream\n");
        {
            std::unique_lock<std::mutex> locker(mutex());
            _eosReceived = true;
            _stoppedCond.notify_all();
        }
        if (_eosAutoQuit) {
            quitMainLoop();
        }
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
    case GST_MESSAGE_STATE_CHANGED: {
        GstState oldState, newState, pendingState;

        gst_message_parse_state_changed(msg, &oldState, &newState, &pendingState);
        LOG_DEBUG(
            "Element %s changed state from %s to %s, pending: %s.", GST_OBJECT_NAME(msg->src),
            gst_element_state_get_name(oldState), gst_element_state_get_name(newState),
            gst_element_state_get_name(pendingState));
        break;
    }
    default:
        break;
    }
    return TRUE;
}

ErrCode
CameraCanApp::stop()
{
    if (mainLoop() && isRunning(1000)) {
        std::unique_lock<std::mutex> lock(mutex());
        sendEOS();
        _stoppedCond.wait(lock, [this]() { return _mainStopped || _eosReceived; });
    }

    quitMainLoop();
    waitLoopQuit();
    ErrCode c = Ds3dAppContext::stop();
    _bufId = 0;
    return c;
}

ErrCode
CameraCanApp::processFrame(
    const NvDsCanContextFrame* frame, std::function<void(GstBuffer*)> callback)
{
    DS_ASSERT(frame);
    DS3D_FAILED_RETURN(
        _config.srcType == NvDsCanSrcType::kFrame, ErrCode::kParam,
        "frame format or size does NOT match pipeline source settings.");
    DS3D_FAILED_RETURN(
        frame->fourcc == _config.srcFrameInfo.fourcc &&
            frame->height == _config.srcFrameInfo.height &&
            frame->width == _config.srcFrameInfo.width,
        ErrCode::kParam, "frame format or size does NOT match pipeline source settings.");

    uint32_t frameSize = 0;
    if (frame->fourcc == NvDsCanFormatRGBA) {
        frameSize = frame->width * frame->height * 4;
    }
    if (frame->fourcc == NvDsCanFormatGREY) {
        frameSize = frame->width * frame->height;
    }

    GstAppSrc* appsrc = GST_APP_SRC(_src.get());
    DS3D_FAILED_RETURN(
        appsrc, ErrCode::kGst, "source element is not appsrc, failed to process frame");
    gst::CapsPtr caps(gst_app_src_get_caps(appsrc));
    GstStructure* s0 = nullptr;
    int frNum = 0, frDenom = 0;
    bool hasFrameRate = true;
    if (caps) {
        s0 = gst_caps_get_structure(caps, 0);
        if (s0 && gst_structure_get_fraction(s0, "framerate", &frNum, &frDenom)) {
            hasFrameRate = true;
        }
    }

    DS3D_FAILED_RETURN(
        frameSize <= frame->bufLen, ErrCode::kParam,
        "frame format or size does NOT match pipeline source settings.");

    GstBuffer* buffer = gst_buffer_new_wrapped_full(
        GST_MEMORY_FLAG_READONLY, frame->buf, frame->bufLen, 0, frameSize, nullptr, nullptr);
    if (hasFrameRate && !config().srcFrameRate.empty()) {
        GST_BUFFER_PTS(buffer) =
            gst_util_uint64_scale(_appsrcFrameIdx, GST_SECOND * (uint64_t)frDenom, (uint64_t)frNum);
    }

    std::promise<void> p;
    std::future<void> f = p.get_future();
    std::unique_lock<std::mutex> locker(_streamMutex);
    _outputCallback = [&p, cb = std::move(callback)](GstBuffer* b) mutable {
        cb(b);
        p.set_value();
    };
    GstFlowReturn ret = gst_app_src_push_buffer((GstAppSrc*)_src.get(), buffer);
    DS3D_FAILED_RETURN(ret == GST_FLOW_OK, ErrCode::kGst, "");
    _appsrcFrameIdx++;

    std::chrono::seconds fiveSec(5);
    locker.unlock();
    if (f.wait_for(fiveSec) == std::future_status::timeout) {
        locker.lock();
        _outputCallback = nullptr;
        LOG_ERROR("Waiting for process frame timeout after 5 seconds.");
        return ErrCode::kTimeOut;
    }
    locker.lock();
    _outputCallback = nullptr;
    return ErrCode::kGood;
}

DS3D_EXTERN_C_BEGIN

DS3D_EXPORT_API ErrCode
NvDs_CanContextParseConfig(
    NvDsCanContextConfig* ctxConfig, const char* configStr, uint32_t len,
    NvDsCanContextConfigType type)
{
    DS_ASSERT(ctxConfig);
    DS3D_FAILED_RETURN(configStr, ErrCode::kConfig, "config string is null");
    DS3D_FAILED_RETURN(
        NvDsCanContextConfigType::kConfigPath == type ||
            NvDsCanContextConfigType::kConfigContent == type,
        ErrCode::kConfig, "Unknown config type");

    std::string content;
    if (NvDsCanContextConfigType::kConfigPath == type) {
        std::string configPath(configStr, (len ? len : strlen(configStr)));
        DS3D_FAILED_RETURN(
            readFile(configPath, content), ErrCode::kConfig, "read config file: %s failed",
            configPath.c_str());
    } else if (NvDsCanContextConfigType::kConfigContent == type) {
        content.assign(configStr, (len ? len : strlen(configStr)));
    }

    NvDsCanContextConfig ccc;
    // parse all components in config file
    ErrCode code = config::CatchConfigCall(parseConfig, content, ccc);
    DS3D_ERROR_RETURN(code, "parse config failed");
    DS_ASSERT(ccc.srcType != NvDsCanSrcType::kNone);
    *ctxConfig = ccc;
    return ErrCode::kGood;
}

DS3D_EXPORT_API ErrCode
NvDs_CanContextCreate(DsCanContext** ctx, const NvDsCanContextConfig* ctxConfig)
{
    DS_ASSERT(ctx);
    DS_ASSERT(ctxConfig);

    ShrdPtr<CameraCanApp> appCtx = std::make_shared<CameraCanApp>();
    DS_ASSERT(appCtx);
    appCtx->setConfig(*ctxConfig);
    appCtx->setMainloop(g_main_loop_new(NULL, FALSE));
    DS3D_FAILED_RETURN(appCtx->mainLoop(), ErrCode::kUnknown, "pipeline set main loop failed");
    DS3D_ERROR_RETURN(appCtx->init("deepstream-can-pipeline"), "init ds can pipeline failed");

    ShrdPtr<CameraCanApp>* ret = new ShrdPtr<CameraCanApp>(appCtx);
    *ctx = reinterpret_cast<DsCanContext*>(ret);
    return ErrCode::kGood;
}

DS3D_EXPORT_API ErrCode
NvDs_CanContextStart(DsCanContext* ctx, NvDsCanContextTaskQuitCb cb, void* usrData)
{
    DS_ASSERT(ctx);
    ShrdPtr<CameraCanApp>* appCtx = reinterpret_cast<ShrdPtr<CameraCanApp>*>(ctx);
    DS_ASSERT(*appCtx);
    DS3D_ERROR_RETURN((*appCtx)->buildPipeline(), "build pipeline failed");
    DS3D_ERROR_RETURN((*appCtx)->play(), "app context play failed");
    (*appCtx)->runMainLoop([cb, usrData]() { cb(usrData); });
    return ErrCode::kGood;
}

DS3D_EXPORT_API bool
NvDs_CanContextIsRunning(DsCanContext* ctx)
{
    DS_ASSERT(ctx);
    ShrdPtr<CameraCanApp>* appCtx = reinterpret_cast<ShrdPtr<CameraCanApp>*>(ctx);
    DS_ASSERT(*appCtx);
    return (*appCtx)->isRunning(3000);
}

DS3D_EXPORT_API ErrCode
NvDs_CanContextProcessFrame(
    DsCanContext* ctx, const NvDsCanContextFrame* frame, NvDsCanContextResultMeta* res)
{
    DS_ASSERT(ctx);
    DS_ASSERT(frame);
    DS_ASSERT(res);
    ShrdPtr<CameraCanApp>* appCtx = reinterpret_cast<ShrdPtr<CameraCanApp>*>(ctx);
    DS_ASSERT(*appCtx);

    bool resFound = false;
    auto bufCb = [app = *appCtx, frame, res, &resFound](GstBuffer* buf) mutable {
        DS_ASSERT(buf);
        resFound = getResultFromBuf(buf, *res);
    };

    DS3D_FAILED_RETURN(
        (*appCtx)->srcType() == NvDsCanSrcType::kFrame, ErrCode::kParam,
        "Process frame requires config setting as frame only");

    DS3D_ERROR_RETURN((*appCtx)->processFrame(frame, bufCb), "Can Context process Frame failed");
    DS3D_FAILED_RETURN(resFound, ErrCode::kGst, "frame is processed but result is not found");
    return ErrCode::kGood;
}

DS3D_EXPORT_API ErrCode
NvDs_CanContextStop(DsCanContext* ctx)
{
    DS_ASSERT(ctx);
    ShrdPtr<CameraCanApp>* appCtx = reinterpret_cast<ShrdPtr<CameraCanApp>*>(ctx);
    DS_ASSERT(*appCtx);

    (*appCtx)->stop();
    return ErrCode::kGood;
}

DS3D_EXPORT_API ErrCode
NvDs_CanContextDestroy(DsCanContext* ctx)
{
    if (!ctx) {
        return ErrCode::kGood;
    }
    ShrdPtr<CameraCanApp>* appCtx = reinterpret_cast<ShrdPtr<CameraCanApp>*>(ctx);
    (*appCtx)->deinit();
    delete appCtx;
    return ErrCode::kGood;
}

DS3D_EXTERN_C_END