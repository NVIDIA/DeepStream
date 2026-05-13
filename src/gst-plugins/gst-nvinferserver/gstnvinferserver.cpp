/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

/**
 * @file gstnvinferserver.cpp
 *
 * @brief nvdsgst_inferserver plugin source file.
 *
 * This file contains the definitions of the standard GStreamer functions
 * for the nvinferserver element/plugin and the required virtual methods of
 * the GstBaseTransform class.
 */

#include <string.h>
#include <sys/time.h>
#include <cassert>
#include <condition_variable>
#include <list>
#include <memory>
#include <mutex>
#include <sstream>
#include <thread>

#include "gst-nvevent.h"
#include "gstnvdsmeta.h"

#include "gst-nvdscustomevent.h"
#include "gstnvinferserver.h"
#include "gstnvinferserver_impl.h"

using namespace gstnvinferserver;
using namespace nvdsinferserver;

GST_DEBUG_CATEGORY(gst_nvinfer_server_debug);
#define GST_CAT_DEFAULT gst_nvinfer_server_debug

#define GST_NVINFER_SERVER_IMPL(gst_nvinfer_server) \
    reinterpret_cast<GstNvInferServerImpl*>((gst_nvinfer_server)->impl)

static GQuark _dsmeta_quark = 0;

#define MAX_BATCH_SIZE 1024

/* Default values for properties */
#define DEFAULT_CONFIG_FILE_PATH ""
#define DEFAULT_BATCH_SIZE 0
#define DEFAULT_INTERVAL 0
#define DEFAULT_OUTPUT_TENSOR_META FALSE

#define PROCESS_MODEL_NONE 0
#define PROCESS_MODEL_FULL_FRAME 1
#define PROCESS_MODEL_OBJECTS 2
#define DEFAULT_PROCESS_MODEL PROCESS_MODEL_NONE
#define PROCESS_MODEL_MIN PROCESS_MODEL_NONE
#define PROCESS_MODEL_MAX PROCESS_MODEL_OBJECTS

#define DEFAULT_OPERATE_ON_GIE_ID -1
#define DEFAULT_INPUT_TENSOR_META FALSE

#define NVTX_DEEPBLUE_COLOR 0xFF667EBE

/**
 * By default NVIDIA Hardware allocated memory flows through the pipeline. We
 * will be processing on this type of memory only. */
#define GST_CAPS_FEATURE_MEMORY_NVMM "memory:NVMM"

static GstStaticPadTemplate gst_nvinfer_server_sink_template =
    GST_STATIC_PAD_TEMPLATE("sink", GST_PAD_SINK, GST_PAD_ALWAYS,
        GST_STATIC_CAPS(GST_VIDEO_CAPS_MAKE_WITH_FEATURES(
            GST_CAPS_FEATURE_MEMORY_NVMM, "{ NV12, RGBA }")));

static GstStaticPadTemplate gst_nvinfer_server_src_template =
    GST_STATIC_PAD_TEMPLATE("src", GST_PAD_SRC, GST_PAD_ALWAYS,
        GST_STATIC_CAPS(GST_VIDEO_CAPS_MAKE_WITH_FEATURES(
            GST_CAPS_FEATURE_MEMORY_NVMM, "{ NV12, RGBA }")));

/* Define our element type. Standard GObject/GStreamer boilerplate stuff */
#define gst_nvinfer_server_parent_class parent_class

G_DEFINE_TYPE(GstNvInferServer, gst_nvinfer_server, GST_TYPE_BASE_TRANSFORM);

/* Implementation of the GObject/GstBaseTransform interfaces. */
static void gst_nvinfer_server_finalize(GObject* object);
static void gst_nvinfer_server_set_property(
    GObject* object, guint prop_id, const GValue* value, GParamSpec* pspec);
static void gst_nvinfer_server_get_property(
    GObject* object, guint prop_id, GValue* value, GParamSpec* pspec);

static gboolean gst_nvinfer_server_start(GstBaseTransform* btrans);
static gboolean gst_nvinfer_server_stop(GstBaseTransform* btrans);
static gboolean gst_nvinfer_server_sink_event(
    GstBaseTransform* trans, GstEvent* event);

static GstFlowReturn gst_nvinfer_server_submit_input_buffer(
    GstBaseTransform* btrans, gboolean discont, GstBuffer* inbuf);
static GstFlowReturn gst_nvinfer_server_generate_output(
    GstBaseTransform* btrans, GstBuffer** outbuf);

/**
 * @brief The class initialization function for the nvinferserver element.
 *
 * Install properties, set sink and src pad capabilities, override the required
 * functions of the base class. These are common to all instances of the
 * element.
 */
static void gst_nvinfer_server_class_init(GstNvInferServerClass* klass) {
    GObjectClass* gobject_class;
    GstElementClass* gstelement_class;
    GstBaseTransformClass* gstbasetransform_class;

    gobject_class = (GObjectClass*)klass;
    gstelement_class = (GstElementClass*)klass;
    gstbasetransform_class = (GstBaseTransformClass*)klass;

    /* Override base class functions */
    gobject_class->finalize = GST_DEBUG_FUNCPTR(gst_nvinfer_server_finalize);
    gobject_class->set_property =
        GST_DEBUG_FUNCPTR(gst_nvinfer_server_set_property);
    gobject_class->get_property =
        GST_DEBUG_FUNCPTR(gst_nvinfer_server_get_property);

    gstbasetransform_class->start = GST_DEBUG_FUNCPTR(gst_nvinfer_server_start);
    gstbasetransform_class->stop = GST_DEBUG_FUNCPTR(gst_nvinfer_server_stop);
    gstbasetransform_class->sink_event =
        GST_DEBUG_FUNCPTR(gst_nvinfer_server_sink_event);

    gstbasetransform_class->submit_input_buffer =
        GST_DEBUG_FUNCPTR(gst_nvinfer_server_submit_input_buffer);
    gstbasetransform_class->generate_output =
        GST_DEBUG_FUNCPTR(gst_nvinfer_server_generate_output);

    /* Install properties. Values set through these properties override the ones
     * in the config file. */
    g_object_class_install_property(gobject_class, PROP_UNIQUE_ID,
        g_param_spec_uint("unique-id", "Unique ID",
            "Unique ID for the element. Can be used to "
            "identify output of the element",
            0, G_MAXUINT, 0,
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
                          GST_PARAM_MUTABLE_READY)));

    g_object_class_install_property(
        gobject_class, PROP_PROCESS_MODE,
        g_param_spec_uint(
            "process-mode", "Process Mode",
            "Inferserver processing mode, (0):None, (1)FullFrame, "
            "(2)ClipObject",
            PROCESS_MODEL_MIN, PROCESS_MODEL_MAX, DEFAULT_PROCESS_MODEL,
            (GParamFlags)(
                G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
                GST_PARAM_MUTABLE_READY)));

    g_object_class_install_property(gobject_class, PROP_CONFIG_FILE_PATH,
        g_param_spec_string("config-file-path", "Config File Path",
            "Path to the configuration file for this instance of nvinferserver",
            DEFAULT_CONFIG_FILE_PATH,
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
                          GST_PARAM_MUTABLE_PLAYING)));

    g_object_class_install_property(gobject_class, PROP_BATCH_SIZE,
        g_param_spec_uint("batch-size", "Batch Size",
            "Maximum batch size for inference", 0, MAX_BATCH_SIZE,
            DEFAULT_BATCH_SIZE,
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
                          GST_PARAM_MUTABLE_READY)));

    g_object_class_install_property(
        gobject_class, PROP_INFER_ON_GIE_ID,
        g_param_spec_int(
            "infer-on-gie-id", "Infer on Gie ID",
            "Infer on metadata generated by GIE with this unique ID.\n"
            "\t\t\tSet to -1 to infer on all metadata.",
            -1, G_MAXINT, DEFAULT_OPERATE_ON_GIE_ID,
            (GParamFlags)(
                G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
                GST_PARAM_MUTABLE_READY)));
    g_object_class_install_property(
        gobject_class, PROP_INPUT_TENSOR_META,
        g_param_spec_boolean(
            "input-tensor-meta", "Input Tensor Meta",
            "Use preprocessed input tensors attached as metadata instead of preprocessing inside "
            "the plugin",
            DEFAULT_INPUT_TENSOR_META,
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | GST_PARAM_MUTABLE_READY)));
    g_object_class_install_property(
        gobject_class, PROP_OPERATE_ON_CLASS_IDS,
        g_param_spec_string(
            "infer-on-class-ids", "Infer on Class ids",
            "Operate on objects with specified class ids\n"
            "\t\t\tUse string with values of class ids in ClassID (int) to set "
            "the property.\n"
            "\t\t\t e.g. 0:2:3",
            "",
            (GParamFlags)(
                G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
                GST_PARAM_MUTABLE_READY)));

    g_object_class_install_property (gobject_class, PROP_INTERVAL,
      g_param_spec_uint ("interval", "Interval",
          "Specifies number of consecutive batches to be skipped for inference",
          0, G_MAXINT, DEFAULT_INTERVAL,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

    g_object_class_install_property(gobject_class, PROP_OUTPUT_CALLBACK,
        g_param_spec_pointer("raw-output-generated-callback",
            "Raw Output Generated Callback",
            "Pointer to the raw output generated callback function\n"
            "\t\t\t(type: gst_nvinfer_server_raw_output_generated_callback in "
            "'gstnvdsinfer.h')",
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
                          GST_PARAM_MUTABLE_READY)));

    g_object_class_install_property(gobject_class,
        PROP_OUTPUT_CALLBACK_USERDATA,
        g_param_spec_pointer("raw-output-generated-userdata",
            "Raw Output Generated UserData",
            "Pointer to the userdata to be supplied with raw output generated "
            "callback",
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
                          GST_PARAM_MUTABLE_READY)));

    /* Set sink and src pad capabilities */
    gst_element_class_add_pad_template(gstelement_class,
        gst_static_pad_template_get(&gst_nvinfer_server_src_template));
    gst_element_class_add_pad_template(gstelement_class,
        gst_static_pad_template_get(&gst_nvinfer_server_sink_template));

    /* Set metadata describing the element */
    gst_element_class_set_details_simple(gstelement_class,
        "NvInferServer plugin", "NvInferServer Plugin",
        "Nvidia DeepStream SDK Triton Inference Server plugin",
        "NVIDIA Corporation. Deepstream for Tesla forum: "
        "https://devtalk.nvidia.com/default/board/209");
}

/**
 * @brief Initialization of the nvinferserver element instance.
 *
 * Allocate the implementation object.
 * Configure the base transform for in-place operation and pass through
 * of the input buffer.
 */
static void gst_nvinfer_server_init(GstNvInferServer* nvinferserver) {
    GstBaseTransform* btrans = GST_BASE_TRANSFORM(nvinferserver);

    /* We will not be generating a new buffer. Just adding / updating
     * metadata. */
    gst_base_transform_set_in_place(GST_BASE_TRANSFORM(btrans), TRUE);
    /* We do not want to change the input caps. Set to passthrough. transform_ip
     * is still called. */
    gst_base_transform_set_passthrough(GST_BASE_TRANSFORM(btrans), TRUE);

    nvinferserver->impl = new GstNvInferServerImpl(nvinferserver);
    assert(nvinferserver->impl);
    nvinferserver->last_flow_ret = GST_FLOW_OK;
    nvinferserver->file_write_batch_num = UINT64_C(0);
    nvinferserver->current_batch_num = UINT64_C(0);

    /* This quark is required to identify NvDsMeta when iterating through
     * the buffer metadatas */
    if (!_dsmeta_quark)
        _dsmeta_quark = g_quark_from_static_string(NVDS_META_STRING);
}

/**
 * @brief Free resources allocated during initialization.
 *
 * Delete the implementation object.
 */
static void gst_nvinfer_server_finalize(GObject* object) {
    GstNvInferServer* nvinferserver = GST_NVINFER_SERVER(object);

    delete GST_NVINFER_SERVER_IMPL(nvinferserver);

    G_OBJECT_CLASS(parent_class)->finalize(object);
}

/**
 * @brief Function called when a property of the element is set. Standard boilerplate.
 */
static void gst_nvinfer_server_set_property(
    GObject* object, guint prop_id, const GValue* value, GParamSpec* pspec) {
    GstNvInferServer* nvinferserver = GST_NVINFER_SERVER(object);
    GstNvInferServerImpl* impl = GST_NVINFER_SERVER_IMPL(nvinferserver);

    switch (prop_id) {
    case PROP_UNIQUE_ID:
        impl->m_GstProperties.uniqueId = g_value_get_uint(value);
        break;
    case PROP_PROCESS_MODE:
        impl->m_GstProperties.processMode = g_value_get_uint(value);
        break;
    case PROP_CONFIG_FILE_PATH:
        impl->m_GstProperties.configPath = g_value_get_string(value);
        break;
    case PROP_BATCH_SIZE:
        impl->m_GstProperties.maxBatchSize = g_value_get_uint(value);
        break;
    case PROP_INFER_ON_GIE_ID:
        impl->m_GstProperties.inferOnGieId = g_value_get_int(value);
        break;
    case PROP_INPUT_TENSOR_META:
        impl->m_GstProperties.inputTensorFromMeta = g_value_get_boolean(value);
        break;
    case PROP_OPERATE_ON_CLASS_IDS: {
        std::stringstream str(g_value_get_string(value));
        std::set<gint> class_ids;

        while (str.peek() != EOF) {
            gint class_id;
            str >> class_id;
            class_ids.insert(class_id);
            str.get();
        }
        impl->m_GstProperties.operateOnClassIds.assign(
            class_ids.begin(), class_ids.end());
    } break;
    case PROP_INTERVAL:
        impl->m_GstProperties.interval = (int32_t)g_value_get_uint(value);
        break;
    case PROP_OUTPUT_CALLBACK:
        nvinferserver->output_generated_callback =
            (gst_nvinfer_raw_output_generated_callback)g_value_get_pointer(
                value);
        break;
    case PROP_OUTPUT_CALLBACK_USERDATA:
        nvinferserver->output_generated_userdata = g_value_get_pointer(value);
        break;
    default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
        break;
    }
}

/**
 * @brief Function called when a property of the element is requested. Standard
 * boilerplate.
 */
static void gst_nvinfer_server_get_property(
    GObject* object, guint prop_id, GValue* value, GParamSpec* pspec) {
    GstNvInferServer* nvinferserver = GST_NVINFER_SERVER(object);
    GstNvInferServerImpl* impl = GST_NVINFER_SERVER_IMPL(nvinferserver);

    switch (prop_id) {
    case PROP_UNIQUE_ID:
        g_value_set_uint(value, impl->m_GstProperties.uniqueId);
        break;
    case PROP_PROCESS_MODE:
        g_value_set_uint(value, impl->m_GstProperties.processMode);
        break;
    case PROP_CONFIG_FILE_PATH:
        g_value_set_string(value, impl->m_GstProperties.configPath.c_str());
        break;
    case PROP_BATCH_SIZE:
        g_value_set_uint(value, impl->m_GstProperties.maxBatchSize);
        break;
    case PROP_INFER_ON_GIE_ID:
        g_value_set_int(value, impl->m_GstProperties.inferOnGieId);
        break;
    case PROP_INPUT_TENSOR_META:
        g_value_set_boolean(value, impl->m_GstProperties.inputTensorFromMeta);
        break;
    case PROP_OPERATE_ON_CLASS_IDS: {
        std::stringstream str;
        for (int32_t i : impl->m_GstProperties.operateOnClassIds) {
            str << i << ":";
        }
        g_value_set_string(value, str.str().c_str());
    } break;
    case PROP_INTERVAL: {
      uint32_t interval = impl->m_GstProperties.interval < 0 ?
        0 : impl->m_GstProperties.interval;
      g_value_set_uint (value, interval);
    }
      break;
    case PROP_OUTPUT_CALLBACK:
        g_value_set_pointer(
            value, (gpointer)nvinferserver->output_generated_callback);
        break;
    case PROP_OUTPUT_CALLBACK_USERDATA:
        g_value_set_pointer(value, nvinferserver->output_generated_userdata);
        break;
    default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
        break;
    }
}

void gst_nvinfer_server_logger(uint32_t unique_id, NvDsInferLogLevel log_level,
    const char* log_message, void* user_ctx) {
    GstNvInferServer* nvinferserver = GST_NVINFER_SERVER(user_ctx);

    switch (log_level) {
    case NVDSINFER_LOG_ERROR:
        GST_ERROR_OBJECT(
            nvinferserver, "nvinferserver[UID %d]: %s", unique_id, log_message);
        return;
    case NVDSINFER_LOG_WARNING:
        GST_WARNING_OBJECT(
            nvinferserver, "nvinferserver[UID %d]: %s", unique_id, log_message);
        return;
    case NVDSINFER_LOG_INFO:
        GST_INFO_OBJECT(
            nvinferserver, "nvinferserver[UID %d]: %s", unique_id, log_message);
        return;
    case NVDSINFER_LOG_DEBUG:
        GST_DEBUG_OBJECT(
            nvinferserver, "nvinferserver[UID %d]: %s", unique_id, log_message);
        return;
    }
}

/**
 * @brief Sink pad event handler.
 *
 * Called when an event is received on the sink pad. We need to make sure
 * serialized events and buffers are pushed downstream while maintaining the
 * order. To ensure this, we push all the buffers in the internal queue to the
 * downstream element before forwarding the serialized event to the downstream
 * element.
 */
static gboolean gst_nvinfer_server_sink_event(
    GstBaseTransform* trans, GstEvent* event) {
    GstNvInferServer* nvinferserver = GST_NVINFER_SERVER(trans);
    GstNvInferServerImpl* impl = GST_NVINFER_SERVER_IMPL(nvinferserver);
    gboolean ignore_serialized_event = FALSE;

    /** The TAG event is sent many times leading to drop in performance because
     * of buffer/event serialization. We can ignore such events which won't
     * cause issues if we don't serialize the events. */
    switch (GST_EVENT_TYPE(event)) {
    case GST_EVENT_TAG:
        ignore_serialized_event = TRUE;
        break;
    default:
        break;
    }

    /* Serialize events. Wait for pending buffers to be processed and pushed
     * downstream. No need to wait in case of classifier async mode since all
     * the buffers are already pushed downstream. */
    if (GST_EVENT_IS_SERIALIZED(event) && !ignore_serialized_event &&
        !impl->isAsyncMode()) {
        impl->sync();
    }

    if ((GstNvEventType)GST_EVENT_TYPE(event) == GST_NVEVENT_PAD_ADDED) {
        /* New source added in the pipeline. Create a source info instance for
         * it. */
        guint source_id = 0;
        gst_nvevent_parse_pad_added(event, &source_id);
        impl->addTrackingSource(source_id);
    }

    if ((GstNvEventType)GST_EVENT_TYPE(event) == GST_NVEVENT_PAD_DELETED) {
        /* Source removed from the pipeline. Remove the related structure. */
        guint source_id = 0;
        gst_nvevent_parse_pad_deleted(event, &source_id);
        impl->eraseTrackingSource(source_id);
    }

    if ((GstNvEventType)GST_EVENT_TYPE(event) == GST_NVEVENT_STREAM_EOS) {
        /* Got EOS from a source. Clean up the object history map. */
        guint source_id = 0;
        gst_nvevent_parse_stream_eos(event, &source_id);
    }

    if (GST_EVENT_TYPE(event) == GST_EVENT_EOS) {
        impl->resetIntervalCounter();
    }
    if ((GstNvDsCustomEventType)GST_EVENT_TYPE (event) == GST_NVEVENT_INFER_INTERVAL_UPDATE) {
      gchar* stream_id = NULL;
      guint interval = 0;

      gst_nvevent_parse_infer_interval_update (event, &stream_id, &interval);
      impl->updateInterval(interval);
    }


    /* Call the sink event handler of the base class. */
    return GST_BASE_TRANSFORM_CLASS(parent_class)->sink_event(trans, event);
}

/**
 * @brief Start function.
 *
 * Initialize all resources and start the output thread.
 */
static gboolean gst_nvinfer_server_start(GstBaseTransform* btrans) {
    GstNvInferServer* nvinferserver = GST_NVINFER_SERVER(btrans);
    GstNvInferServerImpl* impl = GST_NVINFER_SERVER_IMPL(nvinferserver);

    GST_DEBUG_OBJECT(nvinferserver, "unique_id:%u start", impl->uniqueId());
    if (nvinferserver->output_generated_callback) {
        impl->setRawoutputCb(
            [cb = nvinferserver->output_generated_callback,
            usr_data = nvinferserver->output_generated_userdata] (
            void *gstBuf, NvDsInferNetworkInfo &network_info,
            NvDsInferLayerInfo *layers_info, uint32_t num_layers, uint32_t batch_size){
              cb((GstBuffer*)(gstBuf), &network_info, layers_info, num_layers,
                batch_size, usr_data);
            });
    }
    if (impl->start() != NVDSINFER_SUCCESS) {
        GST_ELEMENT_ERROR(nvinferserver, LIBRARY, FAILED,
            ("gstnvinferserver_impl start failed"), (NULL));
        return FALSE;
    }
    return TRUE;
}

/**
 * @brief Stop function.
 *
 * Stop the output thread and free up all the resources.
 */
static gboolean gst_nvinfer_server_stop(GstBaseTransform* btrans) {
    GstNvInferServer* nvinferserver = GST_NVINFER_SERVER(btrans);
    GstNvInferServerImpl* impl = GST_NVINFER_SERVER_IMPL(nvinferserver);

    GST_DEBUG_OBJECT(nvinferserver, "unique_id:%u stop", impl->uniqueId());

    if (impl->stop() != NVDSINFER_SUCCESS) {
        GST_ELEMENT_ERROR(nvinferserver, LIBRARY, FAILED,
            ("gstnvinferserver_impl stop failed"), (NULL));
    }

    return TRUE;
}

/**
 * @brief Function to push the input buffer downstream.
 */
static GstFlowReturn
gst_nvinfer_server_push_buffer(
    GstNvInferServer* nvinferserver, guint64 seqId, GstBuffer* inbuf, nvtxRangeId_t buf_process_range)
{
    GstNvInferServerImpl* impl = GST_NVINFER_SERVER_IMPL(nvinferserver);
    nvtxDomainRangeEnd(impl->nvtxDomain(), buf_process_range);
    nvds_set_output_system_timestamp(
        inbuf, GST_ELEMENT_NAME(nvinferserver));

    GST_DEBUG_OBJECT(nvinferserver, "unique_id:%u push buffer:%" PRIu64,
        impl->uniqueId(), seqId);
    GstFlowReturn flow_ret =
        gst_pad_push(GST_BASE_TRANSFORM_SRC_PAD(nvinferserver), inbuf);
    if (nvinferserver->last_flow_ret != flow_ret) {
        switch (flow_ret) {
        /* Signal the application for pad push errors by posting a error
         * message on the pipeline bus. */
        case GST_FLOW_ERROR:
        case GST_FLOW_NOT_LINKED:
        case GST_FLOW_NOT_NEGOTIATED:
            GST_ELEMENT_ERROR(nvinferserver, STREAM, FAILED,
                ("Internal data stream error."),
                ("streaming stopped, reason %s (%d)",
                    gst_flow_get_name(flow_ret), flow_ret));
            break;
        default:
            break;
        }
    }
    nvinferserver->last_flow_ret = flow_ret;

#define GST_FLOW_INFERSERVER_ERROR (GstFlowReturn)(GST_FLOW_CUSTOM_ERROR_2 - 10)
    if (flow_ret == GST_FLOW_OK && impl->lastError() != NVDSINFER_SUCCESS) {
        nvinferserver->last_flow_ret = GST_FLOW_INFERSERVER_ERROR;
        GST_ELEMENT_ERROR(nvinferserver, LIBRARY, FAILED,
            ("inference failed with unique-id:%d", impl->uniqueId()),
            (nullptr));
    }
    return nvinferserver->last_flow_ret;
}

/**
 * @brief submit_input_buffer function for the nvinferserver element.
 *
 * Called when element receives an input buffer from upstream element.
 */
static GstFlowReturn gst_nvinfer_server_submit_input_buffer(
    GstBaseTransform* btrans, gboolean discont, GstBuffer* inbuf) {
    GstNvInferServer* nvinferserver = GST_NVINFER_SERVER(btrans);
    GstNvInferServerImpl* impl = GST_NVINFER_SERVER_IMPL(nvinferserver);
    GstMapInfo in_map_info = {0};
    NvBufSurface* in_surf = nullptr;
    std::string nvtx_str;

    memset(&in_map_info, 0, sizeof(in_map_info));

    nvinferserver->current_batch_num++;
    GST_DEBUG_OBJECT(nvinferserver, "unique_id:%u submit buffer:%" PRIu64,
        impl->uniqueId(), nvinferserver->current_batch_num);

    NvDsBatchMeta* batch_meta = gst_buffer_get_nvds_batch_meta(inbuf);
    if (batch_meta == nullptr) {
        GST_ELEMENT_ERROR(nvinferserver, STREAM, FAILED,
            ("NvDsBatchMeta not found for input buffer."), (NULL));
        return GST_FLOW_ERROR;
    }

    nvtxEventAttributes_t eventAttrib = {0};
    eventAttrib.version = NVTX_VERSION;
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    eventAttrib.colorType = NVTX_COLOR_ARGB;
    eventAttrib.color = NVTX_DEEPBLUE_COLOR;
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
    nvtx_str = "buffer_process batch_num=" +
               std::to_string(nvinferserver->current_batch_num);
    eventAttrib.message.ascii = nvtx_str.c_str();
    nvtxRangeId_t buf_process_range =
        nvtxDomainRangeStartEx(impl->nvtxDomain(), &eventAttrib);

    nvds_set_input_system_timestamp(inbuf, GST_ELEMENT_NAME(nvinferserver));

    /* Map the buffer contents and get the pointer to NvBufSurface. */
    if (!gst_buffer_map(inbuf, &in_map_info, GST_MAP_READ)) {
        return GST_FLOW_ERROR;
    }
    in_surf = (NvBufSurface*)in_map_info.data;

    if (((in_surf->memType == NVBUF_MEM_DEFAULT || in_surf->memType == NVBUF_MEM_CUDA_DEVICE) &&
         (!impl->canSupportGpu((int)in_surf->gpuId))) ||
        ((impl->canSupportGpu((int)in_surf->gpuId)) && (in_surf->memType == NVBUF_MEM_SYSTEM))) {
          GST_ELEMENT_ERROR(nvinferserver, RESOURCE, FAILED,
              ("Memory Compatibility Error:Input surface gpu-id doesn't match "
               "with configured gpu-id for element,"
               " please allocate input using unified memory, or use same gpu-ids "
               "OR,"
               " if same gpu-ids are used ensure appropriate Cuda memories are "
               "used"),
              ("surface-gpu-id=%d,%s-", in_surf->gpuId,
                  GST_ELEMENT_NAME(nvinferserver)));
          return GST_FLOW_ERROR;
    }

  NvDsInferStatus status = impl->processBatchMeta(
      batch_meta, in_surf, nvinferserver->current_batch_num, inbuf);

  /* Unmap the input buffer contents. */
  if (in_map_info.data)
    gst_buffer_unmap (inbuf, &in_map_info);

  if (status != NVDSINFER_SUCCESS)
    return GST_FLOW_ERROR;

  guint64 seqId = nvinferserver->current_batch_num;
  if (impl->isAsyncMode()) {
        return gst_nvinfer_server_push_buffer(nvinferserver, seqId, inbuf, buf_process_range);
  } else {
        /* Queue a push buffer batch. This batch is not inferred. This batch is
         * to signal the input-queue and output thread that there are no more
         * batches belonging to this input buffer and this GstBuffer can be
         * pushed to downstream element once all the previous processing is
         * done. */
        impl->queueOperation([nvinferserver, seqId, inbuf, buf_process_range]() ->void {
            gst_nvinfer_server_push_buffer(nvinferserver, seqId, inbuf, buf_process_range);
        });
  }

  return GST_FLOW_OK;
}

/**
 * @brief Mandatory generate_output function for nvinferserver element.
 *
 * If submit_input_buffer is implemented, it is mandatory to implement
 * generate_output. Buffers are not pushed to the downstream element from here.
 * Return the GstFlowReturn value of the latest pad push so that any error might
 * be caught by the application.
 */
static GstFlowReturn gst_nvinfer_server_generate_output(
    GstBaseTransform* btrans, GstBuffer** outbuf) {
    GstNvInferServer* nvinferserver = GST_NVINFER_SERVER(btrans);
    return nvinferserver->last_flow_ret;
}

/**
 * @brief Boilerplate for registering a plugin and an element.
 */
static gboolean nvinfer_server_plugin_init(GstPlugin* plugin) {
    GST_DEBUG_CATEGORY_INIT(
        gst_nvinfer_server_debug, "nvinferserver", 0, "nvinferserver plugin");
    gst_debug_category_set_threshold(gst_nvinfer_server_debug, GST_LEVEL_INFO);

    return gst_element_register(
        plugin, "nvinferserver", GST_RANK_PRIMARY, GST_TYPE_NVINFER_SERVER);
}

GST_PLUGIN_DEFINE(GST_VERSION_MAJOR, GST_VERSION_MINOR, nvdsgst_inferserver,
    DESCRIPTION, nvinfer_server_plugin_init, "9.0", LICENSE,
    BINARY_PACKAGE, URL)
