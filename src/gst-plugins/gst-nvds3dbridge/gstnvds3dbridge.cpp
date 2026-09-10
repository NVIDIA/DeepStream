/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


#include <string.h>
#include <sys/time.h>

#include <cassert>
#include <fstream>
#include <iostream>
#include <memory>
#include <ostream>
#include <sstream>
#include <string>

#include "gstnvds3dbridge.h"

GST_DEBUG_CATEGORY_STATIC(gst_nvds3d_bridge_debug);
#define GST_CAT_DEFAULT gst_nvds3d_bridge_debug

#define UPDATE_FILTER_ERROR(ele, impl, code, msg) \
    impl->setError(code);                         \
    GST_ELEMENT_ERROR(nvds3dbridge, RESOURCE, FAILED, (msg), (msg))

/* Enum to identify properties */
enum {
    PROP_0,
    PROP_CONFIG_CONTENT,
    PROP_CONFIG_FILE_NAME,
};

/* Define our element type. Standard GObject/GStreamer boilerplate stuff */
#define gst_nvds3d_bridge_parent_class parent_class
G_DEFINE_TYPE(GstNvDs3dBridge, gst_nvds3d_bridge, GST_TYPE_BASE_TRANSFORM);

static GstStaticPadTemplate gst_nvds3d_bridge_sink_template =
    GST_STATIC_PAD_TEMPLATE("sink", GST_PAD_SINK, GST_PAD_ALWAYS, GST_STATIC_CAPS_ANY);

static GstStaticPadTemplate gst_nvds3d_bridge_src_template =
    GST_STATIC_PAD_TEMPLATE("src", GST_PAD_SRC, GST_PAD_ALWAYS, GST_STATIC_CAPS_ANY);

static void gst_nvds3d_bridge_finalize(GObject* object);
static void gst_nvds3d_bridge_set_property(
    GObject* object, guint prop_id, const GValue* value, GParamSpec* pspec);
static void gst_nvds3d_bridge_get_property(
    GObject* object, guint prop_id, GValue* value, GParamSpec* pspec);

static gboolean gst_nvds3d_bridge_set_caps(
    GstBaseTransform* btrans, GstCaps* incaps, GstCaps* outcaps);
static gboolean gst_nvds3d_bridge_start(GstBaseTransform* btrans);
static gboolean gst_nvds3d_bridge_stop(GstBaseTransform* btrans);

static GstFlowReturn gst_nvds3d_bridge_submit_input_buffer(
    GstBaseTransform* btrans, gboolean discont, GstBuffer* inbuf);
static GstFlowReturn gst_nvds3d_bridge_generate_output(
    GstBaseTransform* btrans, GstBuffer** outbuf);
static gboolean gst_nvds3d_bridge_sink_event(GstBaseTransform* btrans, GstEvent* event);

#define FILTER_IMPL(obj) GST_NVDS3DFILTER_CAST(obj)->impl


using namespace ds3d;
using namespace ds3d::gst;


struct GstNvDs3dBridgeImpl {
    GstNvDs3dBridge* gstTransform = nullptr;

    /** Custom Library Factory and Interface */
    GuardDataBridge bridge;
    Ptr<CustomLibFactory> customlib;

    /* Store custom lib property values */
    std::string configContent;
    std::string configFile;
    config::ComponentConfig bridgeConfig;

    CapsPtr sinkCaps;
    CapsPtr srcCaps;

    std::mutex mutex;
    ErrCode runtimeError = ErrCode::kGood;

    GstNvDs3dBridgeImpl(GstNvDs3dBridge* plugin) : gstTransform(plugin) {}
    ~GstNvDs3dBridgeImpl()
    {
        bridge.reset();
        customlib.reset();
    }
    void setError(ErrCode code)
    {
        std::unique_lock<std::mutex> locker(mutex);
        runtimeError = code;
    }
    ErrCode getError()
    {
        std::unique_lock<std::mutex> locker(mutex);
        return runtimeError;
    }
};

namespace {
inline const char*
CStr(const char* s)
{
    return s ? s : "";
}
}  // namespace

static GstCaps*
gst_nvds3d_bridge_transform_caps(
    GstBaseTransform* trans, GstPadDirection direction, GstCaps* caps, GstCaps* bridgeCaps)
{
    CapsPtr ret;
    GstNvDs3dBridgeImpl* impl = FILTER_IMPL(trans);
    if (direction == GST_PAD_SINK) {
        ret = impl->srcCaps;
    } else {
        ret = impl->sinkCaps;
    }

    if (ret && bridgeCaps) {
        ret.reset(gst_caps_intersect(ret.get(), bridgeCaps));
    }
    if (!ret || gst_caps_is_empty(ret.get())) {
        GST_DEBUG_OBJECT(
            trans,
            "transform from direction: %s, caps: %" GST_PTR_FORMAT
            " failed, no caps found or intersect failed, bridge-caps: %" GST_PTR_FORMAT,
            direction == GST_PAD_SINK ? "sink" : "src", caps, bridgeCaps);
        return gst_caps_new_empty();
    }
    GST_DEBUG_OBJECT(
        trans,
        "transformed from direction:%s, caps: %" GST_PTR_FORMAT ", to caps: %" GST_PTR_FORMAT,
        direction == GST_PAD_SINK ? "sink" : "src", caps, ret.get());
    return ret.copy();
}

static gboolean
gst_nvds3d_bridge_accept_caps(GstBaseTransform* btrans, GstPadDirection direction, GstCaps* caps)
{
    GstNvDs3dBridge* bridge = GST_NVDS3DFILTER(btrans);
    CapsPtr allowed;
    GstNvDs3dBridgeImpl* impl = FILTER_IMPL(bridge);
    const char* directStr = (direction == GST_PAD_SINK ? "SINK" : "SRC");

    GST_DEBUG_OBJECT(btrans, "accept caps %" GST_PTR_FORMAT, caps);

    /* get all the formats we can handle on this pad */
    if (direction == GST_PAD_SINK)
        allowed = impl->sinkCaps;
    else
        allowed = impl->srcCaps;

    if (!allowed) {
        GST_DEBUG_OBJECT(
            btrans, "accept caps failed due to no allowed caps in direction:%s", directStr);
        return FALSE;
    }

    GST_DEBUG_OBJECT(btrans, "allowed caps %" GST_PTR_FORMAT, allowed.get());

    /* intersect with the requested format */
    CapsPtr intersect(gst_caps_intersect(allowed.get(), caps));
    if (!intersect || gst_caps_is_empty(intersect.get())) {
        GST_DEBUG_OBJECT(
            btrans,
            "accept caps failed due to requested caps:%" GST_PTR_FORMAT
            " in direction:%s is not compatible with allowed caps:%" GST_PTR_FORMAT,
            caps, directStr, allowed.get());
        return FALSE;
    }

    return TRUE;
}

/* Install properties, set sink and src pad capabilities, override the required
 * functions of the base class, These are common to all instances of the
 * element.
 */
static void
gst_nvds3d_bridge_class_init(GstNvDs3dBridgeClass* klass)
{
    GObjectClass* gobject_class;
    GstBaseTransformClass* gstbasetransform_class;
    GstElementClass* gstelement_class = (GstElementClass*)klass;

    // Indicates we want to use DS buf api
    g_setenv("DS_NEW_BUFAPI", "1", TRUE);

    GST_DEBUG_CATEGORY_INIT(
        gst_nvds3d_bridge_debug, "nvds3dbridge", 0, "nvds3d custom bridge libs");

    gobject_class = (GObjectClass*)klass;
    gstbasetransform_class = (GstBaseTransformClass*)klass;

    /* Overide base class functions */
    gobject_class->finalize = GST_DEBUG_FUNCPTR(gst_nvds3d_bridge_finalize);
    gobject_class->set_property = GST_DEBUG_FUNCPTR(gst_nvds3d_bridge_set_property);
    gobject_class->get_property = GST_DEBUG_FUNCPTR(gst_nvds3d_bridge_get_property);

    gstbasetransform_class->transform_caps = GST_DEBUG_FUNCPTR(gst_nvds3d_bridge_transform_caps);
    gstbasetransform_class->accept_caps = GST_DEBUG_FUNCPTR(gst_nvds3d_bridge_accept_caps);

    gstbasetransform_class->set_caps = GST_DEBUG_FUNCPTR(gst_nvds3d_bridge_set_caps);
    gstbasetransform_class->start = GST_DEBUG_FUNCPTR(gst_nvds3d_bridge_start);
    gstbasetransform_class->stop = GST_DEBUG_FUNCPTR(gst_nvds3d_bridge_stop);

    gstbasetransform_class->submit_input_buffer =
        GST_DEBUG_FUNCPTR(gst_nvds3d_bridge_submit_input_buffer);
    gstbasetransform_class->generate_output = GST_DEBUG_FUNCPTR(gst_nvds3d_bridge_generate_output);
    gstbasetransform_class->sink_event = GST_DEBUG_FUNCPTR(gst_nvds3d_bridge_sink_event);

    /* Install properties */
    g_object_class_install_property(
        gobject_class, PROP_CONFIG_CONTENT,
        g_param_spec_string(
            "config-content", "Config Content for Custom DS3D bridge",
            "Set config content to load custom bridges", "",
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property(
        gobject_class, PROP_CONFIG_FILE_NAME,
        g_param_spec_string(
            "config-file", "Config File Path for Custom Speech Ctx",
            "Set config content to load custom bridges", "",
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    /* Set sink and src pad capabilities */
    gst_element_class_add_pad_template(
        gstelement_class, gst_static_pad_template_get(&gst_nvds3d_bridge_sink_template));
    gst_element_class_add_pad_template(
        gstelement_class, gst_static_pad_template_get(&gst_nvds3d_bridge_src_template));

    /* Set metadata describing the element */
    gst_element_class_set_static_metadata(
        gstelement_class, "DS 3D bridge custom plugin",
        "nvds3dbridge Plugin for custom bridge libs",
        "Load custom bridge libs with input/output caps settings",
        "NVIDIA Corporation. Post on Deepstream for Tesla forum for any "
        "queries @ https://devtalk.nvidia.com/default/board/209/");
}

static void
gst_nvds3d_bridge_init(GstNvDs3dBridge* nvds3dbridge)
{
    GstNvDs3dBridgeImpl* impl = new GstNvDs3dBridgeImpl(nvds3dbridge);
    FILTER_IMPL(nvds3dbridge) = impl;
    impl->sinkCaps.reset(gst_pad_get_pad_template_caps(GST_BASE_TRANSFORM_SINK_PAD(nvds3dbridge)));
    impl->srcCaps.reset(gst_pad_get_pad_template_caps(GST_BASE_TRANSFORM_SRC_PAD(nvds3dbridge)));
}

static void
gst_nvds3d_bridge_finalize(GObject* object)
{
    GstNvDs3dBridge* bridge = GST_NVDS3DFILTER(object);
    assert(bridge);
    GstNvDs3dBridgeImpl* impl = FILTER_IMPL(bridge);
    impl->bridge.reset();
    delete impl;
}

static bool
checkBridgeConfig(const config::ComponentConfig& conf)
{
    DS3D_FAILED_RETURN(
        conf.type == config::ComponentType::kDataBridge, false,
        "ds3dbridge type must be ds3d::databridge");
    DS3D_FAILED_RETURN(!conf.gstInCaps.empty(), false, "ds3dbridge config must have in_caps");
    CapsPtr sinkCaps(gst_caps_from_string(conf.gstInCaps.c_str()));
    DS3D_FAILED_RETURN(
        sinkCaps && !gst_caps_is_empty(sinkCaps), false,
        "ds3dbridge in_caps must be convertible to gstCaps");
    DS3D_FAILED_RETURN(!conf.gstOutCaps.empty(), false, "ds3dbridge config must have out_caps");
    CapsPtr srcCaps(gst_caps_from_string(conf.gstOutCaps.c_str()));
    DS3D_FAILED_RETURN(
        srcCaps && !gst_caps_is_empty(srcCaps), false,
        "ds3dbridge out_caps must be convertible to gstCaps");
    DS3D_FAILED_RETURN(
        !conf.customLibPath.empty(), false, "ds3dbridge config must have custom_lib_path");
    DS3D_FAILED_RETURN(
        !conf.customCreateFunction.empty(), false,
        "ds3dbridge config must have custom_create_function");
    return true;
}

/* Function called when a property of the element is set. Standard boilerplate.
 */
static void
gst_nvds3d_bridge_set_property(
    GObject* object, guint prop_id, const GValue* value, GParamSpec* pspec)
{
    GstNvDs3dBridge* nvds3dbridge = GST_NVDS3DFILTER(object);
    GstNvDs3dBridgeImpl* impl = FILTER_IMPL(nvds3dbridge);
    switch (prop_id) {
    case PROP_CONFIG_CONTENT: {
        if (!impl->configFile.empty()) {
            GST_ERROR_OBJECT(
                object,
                "Only one property from [config-content, config-file] could be set but detect 2 "
                "properties, result would be unexpected");
        }
        impl->configContent = CStr(g_value_get_string(value));
        ErrCode code = CatchConfigCall(
            config::parseComponentConfig, impl->configContent, "", impl->bridgeConfig);
        if (!isGood(code) || !checkBridgeConfig(impl->bridgeConfig)) {
            GST_ERROR_OBJECT(
                object, "parsing property config-content failed, please check property settings");
        }
        impl->sinkCaps.reset(gst_caps_from_string(impl->bridgeConfig.gstInCaps.c_str()), true);
        impl->srcCaps.reset(gst_caps_from_string(impl->bridgeConfig.gstOutCaps.c_str()), true);
        break;
    }
    case PROP_CONFIG_FILE_NAME: {
        if (!impl->configContent.empty()) {
            GST_ERROR_OBJECT(
                object,
                "Only one property from [config-content, config-file] could be set but detect 2 "
                "properties, result would be unexpected");
        }
        impl->configFile = CStr(g_value_get_string(value));
        impl->bridgeConfig.filePath = impl->configFile;
        std::string content;
        if (!readFile(impl->configFile, content)) {
            GST_ERROR_OBJECT(object, "open config-file: %s failed", impl->configFile.c_str());
            break;
        }
        ErrCode code =
            CatchConfigCall(config::parseComponentConfig, content, "", impl->bridgeConfig);
        if (!isGood(code) || !checkBridgeConfig(impl->bridgeConfig)) {
            GST_ERROR_OBJECT(
                object,
                "parsing content of config-file failed, please check config-file: %s settings",
                impl->configFile.c_str());
        }
        impl->sinkCaps.reset(gst_caps_from_string(impl->bridgeConfig.gstInCaps.c_str()), true);
        impl->srcCaps.reset(gst_caps_from_string(impl->bridgeConfig.gstOutCaps.c_str()), true);
        break;
    }
    default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
        break;
    }
}

/* Function called when a property of the element is requested. Standard
 * boilerplate.
 */
static void
gst_nvds3d_bridge_get_property(GObject* object, guint prop_id, GValue* value, GParamSpec* pspec)
{
    GstNvDs3dBridge* nvds3dbridge = GST_NVDS3DFILTER(object);
    GstNvDs3dBridgeImpl* impl = FILTER_IMPL(nvds3dbridge);

    switch (prop_id) {
    case PROP_CONFIG_CONTENT:
        g_value_set_string(value, impl->configContent.c_str());
        break;
    case PROP_CONFIG_FILE_NAME:
        g_value_set_string(value, impl->configFile.c_str());
        break;
    default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
        break;
    }
}

/**
 * Initialize all resources and start the bridge
 */
static gboolean
gst_nvds3d_bridge_start(GstBaseTransform* btrans)
{
    GstNvDs3dBridge* nvds3dbridge = GST_NVDS3DFILTER(btrans);
    GstNvDs3dBridgeImpl* impl = FILTER_IMPL(nvds3dbridge);

    if (!checkBridgeConfig(impl->bridgeConfig)) {
        GST_ERROR_OBJECT(
            btrans,
            "failed to check bridge config settings, make sure config-content/config-file settings "
            "correct");
        return FALSE;
    }

    /* load custom ds3d bridge from custom lib */
    ErrCode code = loadCustomProcessor(impl->bridgeConfig, impl->bridge, impl->customlib);
    if (!isGood(code)) {
        GST_ERROR_OBJECT(
            btrans, "load custom bridge from custom lib: %s failed, function name: %s",
            impl->bridgeConfig.customLibPath.c_str(),
            impl->bridgeConfig.customCreateFunction.c_str());
        return FALSE;
    }

    /* start the bridge */
    DS_ASSERT(impl->bridge);
    code = impl->bridge.start(impl->bridgeConfig.rawContent, impl->bridgeConfig.filePath);
    if (!isGood(code)) {
        GST_ERROR_OBJECT(btrans, "ds3d custom bridge start failed");
        return FALSE;
    }

    return TRUE;
}

/**
 * Stop the process thread and free up all the resources
 */
static gboolean
gst_nvds3d_bridge_stop(GstBaseTransform* btrans)
{
    GstNvDs3dBridge* nvds3dbridge = GST_NVDS3DFILTER(btrans);
    GstNvDs3dBridgeImpl* impl = FILTER_IMPL(nvds3dbridge);

    if (impl->bridge) {
        impl->bridge.stop();
    }

    GST_DEBUG_OBJECT(nvds3dbridge, "custom bridge stopped and released \n");
    return TRUE;
}

/**
 * Called when source / sink pad capabilities have been negotiated.
 */
static gboolean
gst_nvds3d_bridge_set_caps(GstBaseTransform* btrans, GstCaps* incaps, GstCaps* outcaps)
{
    GstNvDs3dBridge* nvds3dbridge = GST_NVDS3DFILTER(btrans);
    GstNvDs3dBridgeImpl* impl = FILTER_IMPL(nvds3dbridge);

    GST_DEBUG_OBJECT(
        btrans, "set caps, in_caps: %" GST_PTR_FORMAT ",  out_caps: %" GST_PTR_FORMAT, incaps,
        outcaps);

    impl->sinkCaps.reset(incaps, false);
    impl->srcCaps.reset(outcaps, false);
    return TRUE;
}

static gboolean
gst_nvds3d_bridge_sink_event(GstBaseTransform* btrans, GstEvent* event)
{
    GstNvDs3dBridge* nvds3dbridge = GST_NVDS3DFILTER(btrans);
    GstNvDs3dBridgeImpl* impl = FILTER_IMPL(nvds3dbridge);
    ErrCode ret = ErrCode::kGood;

    switch (GST_EVENT_TYPE(event)) {
    case GST_EVENT_EOS:
        g_print("ds3dbridge EOS received\n");
        ret = impl->bridge.flush();
        if (!isNotBad(ret)) {
            UPDATE_FILTER_ERROR(btrans, impl, ret, "bridge flush data on EOS failed");
            return FALSE;
        }
        break;
    default:
        break;
    }
    return GST_BASE_TRANSFORM_CLASS(parent_class)->sink_event(btrans, event);
}

/**
 * Called when element recieves an input buffer from upstream element.
 */
static GstFlowReturn
gst_nvds3d_bridge_submit_input_buffer(GstBaseTransform* btrans, gboolean discont, GstBuffer* inbuf)
{
    GstNvDs3dBridge* nvds3dbridge = GST_NVDS3DFILTER(btrans);
    GstNvDs3dBridgeImpl* impl = FILTER_IMPL(nvds3dbridge);

    if (!impl->bridge) {
        GST_ERROR_OBJECT(btrans, "ds3d custom bridge is not loaded from custom lib. \n");
        return GST_FLOW_CUSTOM_ERROR;
    }
    if (!isNotBad(impl->getError())) {
        GST_ERROR_OBJECT(btrans, "ds3d custom bridge reports error in previous callbacks. \n");
        return GST_FLOW_CUSTOM_ERROR;
    }

    ShrdPtr<VideoBridge2dInput> inputData(new VideoBridge2dInput, [](VideoBridge2dInput* data) {
        if (!data) {
            return;
        }
        if (data->buffer) {
            gst_buffer_unref(data->buffer);
        }
        delete data;
    });
    DS_ASSERT(inputData);
    inputData->buffer = inbuf;
    /** extract and save pointers for NvBufSurface and NvDsBatchMeta */
    /** set NvDsBatchMeta */
    NvDsBatchMeta* batch_meta = gst_buffer_get_nvds_batch_meta (inbuf);
    inputData->batchMeta = batch_meta;

    GstMapInfo mapInfo;
    memset(&mapInfo, 0, sizeof(mapInfo));
    if(!gst_buffer_map(inbuf, &mapInfo, GST_MAP_READ)) {
        GST_ERROR_OBJECT(btrans, "ds3d custom bridge; buffer map error\n");
        return GST_FLOW_ERROR;
    }

    /** set NvBufSurface */
    NvBufSurface* surf = (NvBufSurface*)mapInfo.data;
    inputData->surfaceBatchBuffer =  surf;
    
    if(mapInfo.data)
    {
      gst_buffer_unmap(inbuf, &mapInfo);
    }

    auto consumedData = [impl, nvds3dbridge, inputData](
                            ErrCode c, const struct VideoBridge2dInput* in) {
        if (!isNotBad(c)) {
            UPDATE_FILTER_ERROR(
                nvds3dbridge, impl, c, "failed in consumed callbak of bridge process");
        }
    };

    auto bridgeDone = [impl, nvds3dbridge](ErrCode c, const abiRefDataMap* out) {
        if (!isNotBad(c)) {
            UPDATE_FILTER_ERROR(
                nvds3dbridge, impl, c, "failed in bridge done callbak of bridge process");
            return;
        }
        if (!out) {
            return;
        }
        GstBuffer* bufferOut = nullptr;
        NvDs3D_CreateGstBuf(bufferOut, out->refCopy(), true);
        GstFlowReturn flowRet =
            gst_pad_push(GST_BASE_TRANSFORM_SRC_PAD(nvds3dbridge), bufferOut);
        if (flowRet < GST_FLOW_OK) {
            UPDATE_FILTER_ERROR(
                nvds3dbridge, impl, ErrCode::kGst, "ds3d bridge failed to push data downstream");
            return;
        }
    };

    ErrCode ret = impl->bridge.process(inputData.get(), bridgeDone, consumedData);
    if (!isNotBad(ret) || !isNotBad(impl->getError())) {
        UPDATE_FILTER_ERROR(nvds3dbridge, impl, ret, "failed in bridge.process");
        return GST_FLOW_ERROR;
    }

    return GST_FLOW_OK;
}

/**
 * If submit_input_buffer is implemented, it is mandatory to implement
 * generate_output. Buffers are not pushed to the downstream element from here.
 * Return the GstFlowReturn value of the latest pad push so that any error might
 * be caught by the application.
 */
static GstFlowReturn
gst_nvds3d_bridge_generate_output(GstBaseTransform* btrans, GstBuffer** outbuf)
{
    return GST_FLOW_OK;
}


static gboolean
ds3dbridge_plugin_init(GstPlugin* plugin)
{
    return gst_element_register(plugin, "nvds3dbridge", GST_RANK_PRIMARY, GST_TYPE_NVDS3DFILTER);
}

GST_PLUGIN_DEFINE(
    GST_VERSION_MAJOR, GST_VERSION_MINOR, nvdsgst_3dbridge, DESCRIPTION, ds3dbridge_plugin_init,
    DS_VERSION, LICENSE, BINARY_PACKAGE, URL)
