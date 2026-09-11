/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <gst/gst.h>

#include "gstnvds3dmixer.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include "gst-nvquery.h"
#include "gst-nvquery-internal.h"
#include "gst-nvmessage.h"
#include "gst-nvevent.h"
#include "gst-nvcommon.h"

#include <ds3d/common/func_utils.h>
#include <ds3d/common/config.h>
#include <ds3d/common/hpp/datamixer.hpp>
#include <ds3d/common/hpp/yaml_config.hpp>

#include <ds3d/gst/custom_lib_factory.h>
#include <ds3d/gst/nvds3d_gst_plugin.h>
#include <ds3d/gst/nvds3d_gst_ptr.h>
#include <ds3d/gst/nvds3d_meta.h>
#include <unordered_map>

#define PAD_DATA_KEY "pad-data"
#define UPDATE_MIXER_ERROR(ele, impl, code, msg) \
    impl->setError(code);                        \
    GST_ELEMENT_ERROR(ele, RESOURCE, FAILED, (msg), (msg))

GST_DEBUG_CATEGORY(gst_nvds3dmixer_debug);
#define GST_CAT_DEFAULT gst_nvds3dmixer_debug

/* Define our element type. Standard GObject/GStreamer boilerplate stuff */
#define gst_nvds3dmixer_parent_class parent_class
G_DEFINE_TYPE(GstNvDs3dMixer, gst_nvds3dmixer, GST_TYPE_ELEMENT);

using namespace ds3d;
using namespace ds3d::gst;

namespace {
inline const char*
CStr(const char* s)
{
    return s ? s : "";
}
}  // namespace

typedef struct {
    uint32_t padId;
} GstNvDs3dMixerPadData;

struct GstNvDs3dMixerImpl {
    GstNvDs3dMixer* gstElement;

    /** Custom Library Factory and Interface */
    GuardDataMixer mixer;
    Ptr<CustomLibFactory> customlib;


    /* Store custom lib property values */
    std::string configContent;
    std::string configFile;
    config::ComponentConfig mixerConfig;

    CapsPtr sinkCaps;
    CapsPtr srcCaps;

    std::mutex mutex;
    ErrCode runtimeError = ErrCode::kGood;

    std::unordered_map<uint32_t, uint32_t> padIndexes;
    std::unordered_map<uint32_t, uint32_t> padIndexesWithoutEOS;

    GstNvDs3dMixerImpl(GstNvDs3dMixer* plugin) : gstElement(plugin), mutex() {}
    ~GstNvDs3dMixerImpl()
    {
        mixer.reset();
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

static GstStaticPadTemplate nvds3dmixer_sinkpad_template =
    GST_STATIC_PAD_TEMPLATE("sink_%u", GST_PAD_SINK, GST_PAD_REQUEST, GST_STATIC_CAPS_ANY);

static GstStaticPadTemplate nvds3dmixer_srcpad_template =
    GST_STATIC_PAD_TEMPLATE("src", GST_PAD_SRC, GST_PAD_ALWAYS, GST_STATIC_CAPS_ANY);

enum {
    PROP_0,
    PROP_CONFIG_CONTENT,
    PROP_CONFIG_FILE_NAME,
};

static void loadMixerLowLevel(GstNvDs3dMixerImpl* impl);
static void unloadMixerLowLevel(GstNvDs3dMixerImpl* impl);

static GstFlowReturn
gst_nvds3dmixer_chain(GstPad* pad, GstObject* parent, GstBuffer* buffer)
{
    GstNvDs3dMixer* mixer = GST_NVDS3DMIXER(parent);
    GstNvDs3dMixerImpl* impl = mixer->impl;
    int padId = 0;
    gchar* name = gst_pad_get_name(pad);
    if (name == NULL) {
        return GST_FLOW_ERROR;
    } else if (sscanf(name, "sink_%u", &padId) < 1) {
        g_free(name);
        return GST_FLOW_ERROR;
    }

    if (!impl->mixer) {
        GST_ERROR_OBJECT(mixer, "ds3d custom mixer is not loaded from custom lib. \n");
        return GST_FLOW_CUSTOM_ERROR;
    }
    if (!isNotBad(impl->getError())) {
        GST_ERROR_OBJECT(mixer, "ds3d custom mixer reports error in previous callbacks. \n");
        return GST_FLOW_CUSTOM_ERROR;
    }

    const abiRefDataMap* refDataMap = nullptr;
    if (!isGood(NvDs3D_Find1stDataMap(buffer, refDataMap))) {
        GST_ERROR_OBJECT(mixer, "gst buffer doesn have ds3d datamap, will support later");
        return GST_FLOW_ERROR;
    }

    BufferPtr gstInBuf(buffer);  // take owner of buffer
    DS_ASSERT(refDataMap);
    GuardDataMap inputData(*refDataMap);

    auto consumedData = [impl, gstInBuf, mixer](ErrCode c, const abiRefDataMap* in) {
        if (!isNotBad(c)) {
            UPDATE_MIXER_ERROR(mixer, impl, c, "failed in consumed callbak of mixer process");
        }
    };

    ErrCode ret = impl->mixer.process(padId, inputData, consumedData);
    if (!isNotBad(ret) || !isNotBad(impl->getError())) {
        UPDATE_MIXER_ERROR(mixer, impl, ret, "failed in mixer.process");
        return GST_FLOW_ERROR;
    }

    return GST_FLOW_OK;
}

static gboolean
gst_nvds3dmixer_src_query(GstPad* pad, GstObject* parent, GstQuery* query)
{
    GstNvDs3dMixer* mixer = GST_NVDS3DMIXER(parent);
    (void)mixer;

    return gst_pad_query_default(pad, parent, query);
}

static gboolean
gst_nvds3dmixer_sink_query(GstPad* pad, GstObject* parent, GstQuery* query)
{
    GstNvDs3dMixer* mixer = GST_NVDS3DMIXER(parent);
    GstNvDs3dMixerPadData* pad_data = (GstNvDs3dMixerPadData*)g_object_get_data(G_OBJECT(pad), PAD_DATA_KEY);
    (void)pad_data;
    (void)mixer;
    return gst_pad_query_default(pad, parent, query);
}

static gboolean
gst_nvds3dmixer_src_event(GstPad* pad, GstObject* parent, GstEvent* event)
{
    GstNvDs3dMixer* mixer = GST_NVDS3DMIXER(parent);
    GstElement* element = GST_ELEMENT(parent);
    gboolean ret = TRUE;
    (void)mixer;
    (void)element;

    gst_event_unref(event);
    return ret;
}

static gboolean
gst_nvds3dmixer_sink_event(GstPad* pad, GstObject* parent, GstEvent* event)
{
    GstNvDs3dMixer* mixer = GST_NVDS3DMIXER(parent);
    GstElement* element = GST_ELEMENT(parent);
    GstNvDs3dMixerPadData* pad_data = (GstNvDs3dMixerPadData*)g_object_get_data(G_OBJECT(pad), PAD_DATA_KEY);
    gboolean ret = TRUE;
    uint32_t padId = 0;
    gboolean sendEOSDownstream = FALSE;
    GstEvent* eventEOS = NULL;
    (void)element;

    gchar* name = gst_pad_get_name(pad);
    if (name) {
        if (sscanf(name, "sink_%u", &padId) < 1) {
            GST_DEBUG_OBJECT(mixer, "Streammixer sink event name invalid\n");
            g_free(name);
        }
    }

    switch (GST_EVENT_TYPE(event)) {
    case GST_EVENT_EOS: {
        GST_DEBUG_OBJECT(mixer, "EOS received from pad %d\n", padId);
        eventEOS = gst_nvevent_new_stream_eos(pad_data->padId);
        gst_pad_push_event(mixer->srcpad, eventEOS);
        std::unique_lock<std::mutex> locker(mixer->impl->mutex);
        mixer->impl->padIndexesWithoutEOS.erase(pad_data->padId);
        if (mixer->impl->padIndexesWithoutEOS.size() == 0) {
            mixer->impl->mixer.flush();
            GST_DEBUG_OBJECT(
                mixer,
                "EOS received from last pad %d; "
                "flushed mixer; sending EOS downstream\n",
                padId);
            sendEOSDownstream = TRUE;
        }
        // clear all pending buffer to avoid force_sync hung on other plugin's buffer pool
        // Todo: find better way to fix hung rather than remove port
        mixer->impl->mixer.updateInput(padId, MixerUpdate::kPortRemoved);
        locker.unlock();
        if (sendEOSDownstream) {
            gst_pad_push_event(mixer->srcpad, gst_event_new_eos());
        }
        return TRUE;
    } break;
    default:
        break;
    }

    ret = gst_pad_push_event(mixer->srcpad, event);

    return ret;
}

static void
gst_nvds3dmixer_free_pad_data(GstNvDs3dMixerPadData* data)
{
    if (data) {
        g_free(data);
        data = NULL;
    }
}

static GstPad*
gst_nvds3dmixer_request_new_pad(GstElement* element, GstPadTemplate* templ, const gchar* name, const GstCaps* caps)
{
    GstNvDs3dMixer* mixer = GST_NVDS3DMIXER(element);
    GstNvDs3dMixerPadData* pad_data;
    GstPad* sinkpad = NULL;
    uint32_t padId = 0;

    GST_DEBUG_OBJECT(element, "Requesting new sink pad");

    if (!name || sscanf(name, "sink_%u", &padId) < 1) {
        GST_ERROR_OBJECT(element, "Pad should be named 'sink_%%u' when requesting a pad");
        return NULL;
    }
    std::unique_lock<std::mutex> locker(mixer->impl->mutex);

    if (mixer->impl->padIndexes.find(padId) != mixer->impl->padIndexes.end()) {
        GST_ERROR_OBJECT(element, "Pad named '%s' already requested", name);
        return NULL;
    }

    sinkpad = GST_PAD_CAST(
        g_object_new(GST_TYPE_NVDS3DMIXER_PAD, "name", name, "direction", templ->direction, "template", templ, NULL));

    mixer->impl->padIndexes[padId] = padId;
    mixer->impl->padIndexesWithoutEOS[padId] = padId;
    locker.unlock();

    pad_data = (GstNvDs3dMixerPadData*)g_malloc0(sizeof(GstNvDs3dMixerPadData));
    pad_data->padId = padId;
    g_object_set_data_full(G_OBJECT(sinkpad), PAD_DATA_KEY, pad_data, (GDestroyNotify)gst_nvds3dmixer_free_pad_data);

    gst_pad_set_chain_function(sinkpad, GST_DEBUG_FUNCPTR(gst_nvds3dmixer_chain));

    gst_pad_set_event_function(sinkpad, GST_DEBUG_FUNCPTR(gst_nvds3dmixer_sink_event));

    gst_pad_set_query_function(sinkpad, GST_DEBUG_FUNCPTR(gst_nvds3dmixer_sink_query));

    gst_pad_set_active(sinkpad, TRUE);

    locker.lock();
    gst_element_add_pad(element, sinkpad);
    locker.unlock();

    if (!mixer->impl->mixer) {
        loadMixerLowLevel(mixer->impl);
    }
    mixer->impl->mixer.updateInput(padId, MixerUpdate::kPortAdded);

    return sinkpad;
}

static void
gst_nvds3dmixer_release_pad(GstElement* element, GstPad* pad)
{
    GstNvDs3dMixer* mixer = GST_NVDS3DMIXER(element);
    GstNvDs3dMixerPadData* data = (GstNvDs3dMixerPadData*)g_object_get_data(G_OBJECT(pad), PAD_DATA_KEY);
    uint32_t padId = data->padId;
    GstNvDs3dMixerImpl* impl = mixer->impl;
    assert(impl);

    if (GST_STATE(element) >= GST_STATE_PAUSED) {
        gst_pad_push_event (mixer->srcpad, gst_nvevent_new_pad_deleted(padId));
    }

    gst_pad_set_active(pad, FALSE);

    std::unique_lock<std::mutex> locker(impl->mutex);
    impl->padIndexes.erase(padId);

    GST_DEBUG_OBJECT(mixer, "Pad deleted %d\n", padId);

    gst_element_remove_pad(GST_ELEMENT_CAST(mixer), pad);

    impl->mixer.updateInput(padId, MixerUpdate::kPortRemoved);
}

static GstStateChangeReturn
gst_nvds3dmixer_change_state(GstElement* element, GstStateChange transition)
{
    GstNvDs3dMixer* mixer = GST_NVDS3DMIXER(element);
    GstStateChangeReturn ret;
    (void)mixer;

    switch (transition) {
    case GST_STATE_CHANGE_NULL_TO_READY:
        break;
    case GST_STATE_CHANGE_READY_TO_NULL:
        unloadMixerLowLevel(mixer->impl);
        break;
    case GST_STATE_CHANGE_READY_TO_PAUSED:
        break;
    case GST_STATE_CHANGE_PAUSED_TO_READY:
        break;
    default:
        break;
    }
    ret = GST_ELEMENT_CLASS(parent_class)->change_state(element, transition);
    return ret;
}

static bool
checkMixerConfig(const config::ComponentConfig& conf)
{
    DS3D_FAILED_RETURN(conf.type == config::ComponentType::kDataMixer, false, "ds3dmixer type must be ds3d::datamixer");
    DS3D_FAILED_RETURN(!conf.gstInCaps.empty(), false, "ds3dmixer config must have in_caps");
    CapsPtr sinkCaps(gst_caps_from_string(conf.gstInCaps.c_str()));
    DS3D_FAILED_RETURN(
        sinkCaps && !gst_caps_is_empty(sinkCaps), false, "ds3dmixer in_caps must be convertible to gstCaps");
    DS3D_FAILED_RETURN(!conf.gstOutCaps.empty(), false, "ds3dmixer config must have out_caps");
    CapsPtr srcCaps(gst_caps_from_string(conf.gstOutCaps.c_str()));
    DS3D_FAILED_RETURN(
        srcCaps && !gst_caps_is_empty(srcCaps), false, "ds3dmixer out_caps must be convertible to gstCaps");
    DS3D_FAILED_RETURN(!conf.customLibPath.empty(), false, "ds3dmixer config must have custom_lib_path");
    DS3D_FAILED_RETURN(!conf.customCreateFunction.empty(), false, "ds3dmixer config must have custom_create_function");
    return true;
}

static void
loadMixerLowLevel(GstNvDs3dMixerImpl* impl)
{
    GstNvDs3dMixer* mixer = impl->gstElement;
    if (!checkMixerConfig(impl->mixerConfig)) {
        GST_ERROR_OBJECT(
            mixer,
            "failed to check mixer config settings, make sure config-content/config-file settings "
            "correct");
        return;
    }

    /* load custom ds3d mixer from custom lib */
    ErrCode code = loadCustomProcessor(impl->mixerConfig, impl->mixer, impl->customlib);
    if (!isGood(code)) {
        GST_ERROR_OBJECT(
            mixer, "load custom mixer from custom lib: %s failed, function name: %s",
            impl->mixerConfig.customLibPath.c_str(), impl->mixerConfig.customCreateFunction.c_str());
        return;
    }

    /* start the mixer */
    DS_ASSERT(impl->mixer);
    code = impl->mixer.start(impl->mixerConfig.rawContent, impl->mixerConfig.filePath);
    if (!isGood(code)) {
        GST_ERROR_OBJECT(mixer, "ds3d custom mixer start failed");
        return;
    }

    /* set output callback */
    auto outputCb = [impl, mixer](ErrCode error, const abiRefDataMap* data) {
        if (!data) {
            GST_ERROR_OBJECT(mixer, "ds3d custom mixer LL gave NULL output");
            return;
        }
        GuardDataMap datamap(*data);
        GstBuffer* buf = nullptr;
        ErrCode c = NvDs3D_CreateGstBuf(buf, datamap.abiRef(), false);
        if (!isGood(c)) {
            return;
        }
        TimeStamp ts{0};
        c = datamap.getData(kTimeStamp, ts);
        if (isGood(c)) {
            GST_BUFFER_PTS(buf) = ts.t0;
        }
        GstFlowReturn flowRet = gst_pad_push(mixer->srcpad, buf);
        if (flowRet != GST_FLOW_OK) {
            GST_ERROR_OBJECT(
                mixer,
                "ds3d custom mixer output buffer push error"
                " (%d)",
                flowRet);
            return;
        }
    };

    impl->mixer.setOutputCb(outputCb);
}

static void
unloadMixerLowLevel(GstNvDs3dMixerImpl* impl)
{
    if (!impl->mixer) {
        return;
    }
    impl->mixer.flush();
    impl->mixer.stop();
    // keep impl pointer since other funtions(e.g. gst_nvds3dmixer_release_pad)
    // or GstBuffer unref have hooks for this plugin or custom-lib
}

static void
gst_nvds3dmixer_set_property(GObject* object, guint prop_id, const GValue* value, GParamSpec* pspec)
{
    GstNvDs3dMixer* mixer = GST_NVDS3DMIXER(object);
    GstNvDs3dMixerImpl* impl = mixer->impl;
    (void)mixer;
    switch (prop_id) {
    case PROP_CONFIG_CONTENT: {
        if (!impl->configFile.empty()) {
            GST_ERROR_OBJECT(
                object,
                "Only one property from [config-content, config-file] could be set but detect 2 "
                "properties, result would be unexpected");
        }
        impl->configContent = CStr(g_value_get_string(value));
        ErrCode code = CatchConfigCall(config::parseComponentConfig, impl->configContent, "", impl->mixerConfig);
        if (!isGood(code) || !checkMixerConfig(impl->mixerConfig)) {
            GST_ERROR_OBJECT(object, "parsing property config-content failed, please check property settings");
        }
        impl->sinkCaps.reset(gst_caps_from_string(impl->mixerConfig.gstInCaps.c_str()), true);
        impl->srcCaps.reset(gst_caps_from_string(impl->mixerConfig.gstOutCaps.c_str()), true);
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
        impl->mixerConfig.filePath = impl->configFile;
        std::string content;
        if (!readFile(impl->configFile, content)) {
            GST_ERROR_OBJECT(object, "open config-file: %s failed", impl->configFile.c_str());
            break;
        }
        ErrCode code = CatchConfigCall(config::parseComponentConfig, content, "", impl->mixerConfig);
        if (!isGood(code) || !checkMixerConfig(impl->mixerConfig)) {
            GST_ERROR_OBJECT(
                object, "parsing content of config-file failed, please check config-file: %s settings",
                impl->configFile.c_str());
        }
        impl->sinkCaps.reset(gst_caps_from_string(impl->mixerConfig.gstInCaps.c_str()), true);
        impl->srcCaps.reset(gst_caps_from_string(impl->mixerConfig.gstOutCaps.c_str()), true);
        break;
    }
    default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
        break;
    }
}

static void
gst_nvds3dmixer_get_property(GObject* object, guint prop_id, GValue* value, GParamSpec* pspec)
{
    GstNvDs3dMixer* mixer = GST_NVDS3DMIXER(object);
    GstNvDs3dMixerImpl* impl = mixer->impl;
    (void)mixer;
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

static void
gst_nvds3dmixer_finalize(GObject* object)
{
    GstNvDs3dMixer* mixer = GST_NVDS3DMIXER(object);
    GstNvDs3dMixerImpl* impl = mixer->impl;
    impl->mixer.reset();
    delete impl;
    mixer->impl = nullptr;
}

static void
gst_nvds3dmixer_class_init(GstNvDs3dMixerClass* klass)
{
    GObjectClass* gobject_class = G_OBJECT_CLASS(klass);
    GstElementClass* gstelement_class = GST_ELEMENT_CLASS(klass);

    // Indicate the use of DS buf api version
    g_setenv("DS_NEW_BUFAPI", "1", TRUE);

    gst_element_class_set_static_metadata(
        gstelement_class, "Stream multiplexer", "Generic", "N-to-1 pipe stream multiplexing",
        "NVIDIA Corporation. Post on Deepstream for Tesla forum for any queries "
        "@ https://devtalk.nvidia.com/default/board/209/");

    gobject_class->set_property = GST_DEBUG_FUNCPTR(gst_nvds3dmixer_set_property);
    gobject_class->get_property = GST_DEBUG_FUNCPTR(gst_nvds3dmixer_get_property);
    gobject_class->finalize = GST_DEBUG_FUNCPTR(gst_nvds3dmixer_finalize);

    /* Install properties */
    g_object_class_install_property(
        gobject_class, PROP_CONFIG_CONTENT,
        g_param_spec_string(
            "config-content", "Config Content for Custom DS3D mixer", "Set config content to load custom mixers", "",
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property(
        gobject_class, PROP_CONFIG_FILE_NAME,
        g_param_spec_string(
            "config-file", "Config File Path for Custom DS3D mixer Lib Ctx", "Set config content to load custom mixers",
            "", (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    gst_element_class_add_static_pad_template(gstelement_class, &nvds3dmixer_sinkpad_template);
    gst_element_class_add_static_pad_template(gstelement_class, &nvds3dmixer_srcpad_template);

    gstelement_class->request_new_pad = GST_DEBUG_FUNCPTR(gst_nvds3dmixer_request_new_pad);
    gstelement_class->release_pad = GST_DEBUG_FUNCPTR(gst_nvds3dmixer_release_pad);
    gstelement_class->change_state = GST_DEBUG_FUNCPTR(gst_nvds3dmixer_change_state);
}

static void
gst_nvds3dmixer_init(GstNvDs3dMixer* mixer)
{
    mixer->impl = new GstNvDs3dMixerImpl(mixer);

    mixer->srcpad = gst_pad_new_from_static_template(&nvds3dmixer_srcpad_template, "src");
    gst_pad_set_query_function(mixer->srcpad, GST_DEBUG_FUNCPTR(gst_nvds3dmixer_src_query));
    gst_pad_use_fixed_caps(mixer->srcpad);

    gst_pad_set_event_function(mixer->srcpad, GST_DEBUG_FUNCPTR(gst_nvds3dmixer_src_event));

    gst_element_add_pad(GST_ELEMENT(mixer), mixer->srcpad);

}

static gboolean
nvds3dmixer_plugin_init(GstPlugin* plugin)
{
    GST_DEBUG_CATEGORY_INIT(gst_nvds3dmixer_debug, "nvds3dmixer", 0, "nvds3dmixer plugin");

    if (!gst_element_register(plugin, "nvds3dmixer", GST_RANK_PRIMARY, GST_TYPE_NVDS3DMIXER))
        return FALSE;

    return TRUE;
}

GST_PLUGIN_DEFINE(
    GST_VERSION_MAJOR, GST_VERSION_MINOR, nvdsgst_3dmixer, DESCRIPTION, nvds3dmixer_plugin_init, DS_VERSION, LICENSE,
    BINARY_PACKAGE, URL)

G_DEFINE_TYPE(GstNvDs3dMixerPad, gst_nvds3dmixer_pad, GST_TYPE_PAD);

static void
gst_nvds3dmixer_pad_class_init(GstNvDs3dMixerPadClass* klass)
{
}

static void
gst_nvds3dmixer_pad_init(GstNvDs3dMixerPad* pad)
{
    pad->got_eos = FALSE;
}
