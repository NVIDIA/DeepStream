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

#include <gst/gst.h>
#include "gstnvdsmeta.h"

/* Custom metadata type for source identification */
#define NVDS_GST_META_SOURCE (nvds_get_user_meta_type("NVIDIA.DECODER.GST_SOURCE"))

GST_DEBUG_CATEGORY_STATIC(dynamic_src_bin_debug);
#define GST_CAT_DEFAULT dynamic_src_bin_debug

/* Source metadata structure to track chunk/source information */
typedef struct _NvDsSourceMeta
{
  gint chunk_id;  /* Unique identifier for each source/chunk */
} NvDsSourceMeta;

/* gst meta copy function set by user */
static gpointer source_meta_copy_func(gpointer data, gpointer user_data)
{
  NvDsSourceMeta *src_source_meta = (NvDsSourceMeta *)data;
  NvDsSourceMeta *dst_source_meta = (NvDsSourceMeta*)g_malloc0(
      sizeof(NvDsSourceMeta));
  memcpy(dst_source_meta, src_source_meta, sizeof(NvDsSourceMeta));
  return (gpointer)dst_source_meta;
}

/* gst meta release function set by user */
static void source_meta_release_func(gpointer data, gpointer user_data)
{
  NvDsSourceMeta *source_meta = (NvDsSourceMeta *)data;
  if(source_meta) {
    g_free(source_meta);
    source_meta = NULL;
  }
}

/* gst to nvds transform function set by user. "data" holds a pointer to NvDsUserMeta */
static gpointer source_gst_to_nvds_meta_transform_func(gpointer data, gpointer user_data)
{
  NvDsUserMeta *user_meta = (NvDsUserMeta *)data;
  NvDsSourceMeta *src_source_meta =
    (NvDsSourceMeta*)user_meta->user_meta_data;
  NvDsSourceMeta *dst_source_meta =
    (NvDsSourceMeta *)source_meta_copy_func(src_source_meta, NULL);
  return (gpointer)dst_source_meta;
}

/* release function set by user to release gst to nvds transformed metadata.
 * "data" holds a pointer to NvDsUserMeta */
static void source_gst_nvds_meta_release_func(gpointer data, gpointer user_data)
{
  NvDsUserMeta *user_meta = (NvDsUserMeta *) data;
  NvDsSourceMeta *source_meta = (NvDsSourceMeta *)user_meta->user_meta_data;
  source_meta_release_func(source_meta, NULL);
}

#define DYNAMIC_SRC_BIN_TYPE (dynamic_src_bin_get_type())
G_DECLARE_FINAL_TYPE(DynamicSrcBin, dynamic_src_bin, DYNAMIC, SRC_BIN, GstBin)

/* DynamicSrcBin structure definition
 * Contains all the necessary elements and state information for the source bin
 */
struct _DynamicSrcBin {
    GstBin parent;
    
    /* Core GStreamer pipeline elements */
    GstElement *filesrc;         /* Source element for reading files */
    GstElement *queue_filesrc;   /* Queue element for buffering source data */
    GstElement *parsebin;        /* Parser element for media format detection */
    GstElement *queue_parsebin;  /* Queue element for buffering parsed data */
    GstElement *decoder;         /* Decoder element for media decoding */

    /* Source management data structures */
    GQueue *srcbin_source_queue; /* Queue to track active sources */
    GHashTable *source_map;      /* Maps source IDs to file paths */

    /* State variables */
    guint gpu_id;               /* GPU ID for hardware decoding */
    GMutex source_id_list_lock; /* Thread synchronization for source operations */
    gboolean first_source_added;/* Flag to track initial source addition */
    gint current_source_id;     /* Currently active source ID */
    gint frame_id;              /* Current frame counter */
};

G_DEFINE_TYPE(DynamicSrcBin, dynamic_src_bin, GST_TYPE_BIN)

// Property IDs
enum {
    PROP_0,
    PROP_CURRENT_FILE,
    PROP_CURRENT_ID,
    PROP_GPU_ID
};

static void dynamic_src_bin_set_property(GObject *object, guint prop_id,
                                       const GValue *value, GParamSpec *pspec) {
    DynamicSrcBin *srcbin = DYNAMIC_SRC_BIN(object);
    switch (prop_id) {
        case PROP_GPU_ID:
            srcbin->gpu_id = g_value_get_uint(value);
            g_object_set(srcbin->decoder, "gpu-id", srcbin->gpu_id, NULL);
            break;
        default:
            G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
            break;
    }
}

static void dynamic_src_bin_get_property(GObject *object, guint prop_id,
                                       GValue *value, GParamSpec *pspec) {
    DynamicSrcBin *srcbin = DYNAMIC_SRC_BIN(object);

    switch (prop_id) {
        case PROP_CURRENT_FILE: {
            // Lock scope reduced to minimize critical section
            gchar *file_path = NULL;
            g_mutex_lock(&srcbin->source_id_list_lock);
            if (!g_queue_is_empty(srcbin->srcbin_source_queue)) {
                file_path = g_strdup(g_hash_table_lookup(srcbin->source_map,
                                                g_queue_peek_head(srcbin->srcbin_source_queue)));
            }
            g_mutex_unlock(&srcbin->source_id_list_lock);

            g_value_set_string(value, file_path);
            g_free(file_path);  // Free the duplicated string
            break;
        }
        case PROP_CURRENT_ID:
            g_mutex_lock(&srcbin->source_id_list_lock);
            if (!g_queue_is_empty(srcbin->srcbin_source_queue)) {
                g_value_set_int(value, GPOINTER_TO_INT(g_queue_peek_head(srcbin->srcbin_source_queue)));
            } else {
                g_value_set_int(value, -1);
            }
            g_mutex_unlock(&srcbin->source_id_list_lock);
            break;
        case PROP_GPU_ID:
            g_value_set_uint(value, srcbin->gpu_id);
            break;
        default:
            G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
            break;
    }
}

static void dynamic_src_bin_dispose(GObject *object) {
    DynamicSrcBin *srcbin = DYNAMIC_SRC_BIN(object);
    g_mutex_lock(&srcbin->source_id_list_lock);
    if (srcbin->source_map) {
        g_hash_table_destroy(srcbin->source_map);
        srcbin->source_map = NULL;
    }

    if (srcbin->srcbin_source_queue) {
        g_queue_free(srcbin->srcbin_source_queue);
        srcbin->srcbin_source_queue = NULL;
    }
    g_mutex_unlock(&srcbin->source_id_list_lock);
    g_mutex_clear(&srcbin->source_id_list_lock);

    G_OBJECT_CLASS(dynamic_src_bin_parent_class)->dispose(object);
}

static void post_file_change_message(DynamicSrcBin *srcbin) {
    GST_INFO_OBJECT(srcbin, "Posting file change message");
    GstStructure *structure = gst_structure_new_empty("dynamic-src-bin-file-change");
    GstMessage *message = gst_message_new_application(GST_OBJECT(srcbin), structure);
    GstBus *bus = gst_element_get_bus(GST_ELEMENT(srcbin));
    gst_bus_post(bus, message);
    gst_object_unref(bus);
}

/* Pad probe callback for source bin's src pad
 * Handles custom events and attaches source metadata to buffers
 */
static GstPadProbeReturn srcbin_src_pad_probe_callback(GstPad *pad, GstPadProbeInfo *info, gpointer user_data) {
    DynamicSrcBin *srcbin = DYNAMIC_SRC_BIN(user_data);

    if (srcbin == NULL) {
        GST_DEBUG_OBJECT(srcbin, "Source bin is null");
        return GST_PAD_PROBE_OK;
    }

    // Handle downstream events
    if (GST_PAD_PROBE_INFO_TYPE(info) & GST_PAD_PROBE_TYPE_EVENT_DOWNSTREAM) {
        GstEvent *event = GST_PAD_PROBE_INFO_EVENT(info);
        if (GST_EVENT_TYPE(event) == GST_EVENT_CUSTOM_DOWNSTREAM) {
            const GstStructure *structure = gst_event_get_structure(event);
            if(structure && gst_structure_has_name(structure, "custom-stream-start-event")) {
                // Get source_id from the custom event
                gint received_source_id = -1;
                if (gst_structure_get_int(structure, "source-id", &received_source_id)) {
                    srcbin->current_source_id = received_source_id;
                    GST_INFO_OBJECT(srcbin, "Current Source ID received from custom-stream-start-event: %d", srcbin->current_source_id);
                    srcbin->frame_id = 0;
                } else {
                    GST_WARNING_OBJECT(srcbin, "Failed to get source-id from custom-stream-start-event");
                    srcbin->current_source_id = -1;
                }
            }
            else if (structure && gst_structure_has_name(structure, "custom-eos-event")) {
                gint received_source_id = -1;

                if (gst_structure_get_int(structure, "source-id", &received_source_id)) {
                    GST_INFO_OBJECT(srcbin, "Current Source ID received from custom-eos-event: %d", received_source_id);
                } else {
                    GST_WARNING_OBJECT(srcbin, "Failed to get source-id from custom-eos-event");
                }

                if (srcbin->current_source_id == received_source_id) {
                    GST_INFO_OBJECT(srcbin, "Custom EOS received on current source ID, stopping current file");
                    GST_INFO_OBJECT(srcbin, "Removing source ID: %d from source map and srcbin queue", received_source_id);
                    g_mutex_lock(&srcbin->source_id_list_lock);
                    g_hash_table_remove(srcbin->source_map, GINT_TO_POINTER(received_source_id));
                    g_queue_pop_head(srcbin->srcbin_source_queue);
                    post_file_change_message(srcbin);
                    g_mutex_unlock(&srcbin->source_id_list_lock);
                }
                else {
                    GST_WARNING_OBJECT(srcbin, "Custom EOS received on different source ID, ignoring");
                }

            }
        }
    }
    // Handle buffers
    else if (GST_PAD_PROBE_INFO_TYPE(info) & GST_PAD_PROBE_TYPE_BUFFER) {
            // Move mutex lock to smaller scope
            gint current_source_id;
            g_mutex_lock(&srcbin->source_id_list_lock);
            current_source_id = srcbin->current_source_id;
            g_mutex_unlock(&srcbin->source_id_list_lock);

            srcbin->frame_id++;
            GstBuffer *buffer = GST_PAD_PROBE_INFO_BUFFER(info);
            if (buffer) {
                NvDsMeta *meta = NULL;

                NvDsSourceMeta *source_meta = (NvDsSourceMeta *)g_malloc0(sizeof(NvDsSourceMeta));
                if (source_meta == NULL) {
                    return GST_PAD_PROBE_OK;
                }

                source_meta->chunk_id = current_source_id;

                /* Attach decoder metadata to gst buffer using gst_buffer_add_nvds_meta() */
                meta = gst_buffer_add_nvds_meta(buffer, source_meta, NULL,
                    source_meta_copy_func, source_meta_release_func);

                /* Set metadata type */
                meta->meta_type = (GstNvDsMetaType)NVDS_GST_META_SOURCE;

                /* Set transform function to transform decoder metadata from Gst meta to
                * nvds meta */
                meta->gst_to_nvds_meta_transform_func = source_gst_to_nvds_meta_transform_func;

                /* Set release function to release the transformed nvds metadata */
                meta->gst_to_nvds_meta_release_func = source_gst_nvds_meta_release_func;

                GST_DEBUG_OBJECT(srcbin, "Frame ID: %d, Current Source ID: %d, pts %" GST_TIME_FORMAT,
                              srcbin->frame_id, current_source_id,
                              GST_TIME_ARGS(GST_BUFFER_PTS(buffer)));
            }

    }

    return GST_PAD_PROBE_OK;
}

/* Handler for new pad creation from parsebin
 * Links newly created video pads to the downstream elements
 */
static void parsebin_pad_added_handler(GstElement *src, GstPad *new_pad, gpointer data) {
    DynamicSrcBin *srcbin = DYNAMIC_SRC_BIN(data);
    GstCaps *new_pad_caps = NULL;
    GstPad *queue_parsebin_sink_pad = NULL;

    if (!new_pad || !srcbin) {
        GST_ERROR_OBJECT(srcbin, "Received null pad or data");
        return;
    }

    new_pad_caps = gst_pad_get_current_caps(new_pad);
    if (!new_pad_caps) {
        new_pad_caps = gst_pad_query_caps(new_pad, NULL);
        if (!new_pad_caps) {
            GST_ERROR_OBJECT(srcbin, "Failed to get pad caps");
            return;
        }
    }

    GstStructure *new_pad_struct = gst_caps_get_structure(new_pad_caps, 0);
    const gchar *name = gst_structure_get_name(new_pad_struct);

    if (g_str_has_prefix(name, "video/")) {
        queue_parsebin_sink_pad = gst_element_get_static_pad(srcbin->queue_parsebin, "sink");
        if (queue_parsebin_sink_pad) {
            // Check if pads are already linked
            if (gst_pad_is_linked(queue_parsebin_sink_pad)) {
                GST_DEBUG_OBJECT(srcbin, "queue_parsebin's sink pad is already linked");
                gst_object_unref(queue_parsebin_sink_pad);
            }
            else {
                // Attempt to link the pads
                GstPadLinkReturn ret = gst_pad_link(new_pad, queue_parsebin_sink_pad);
                if (GST_PAD_LINK_FAILED(ret)) {
                    GST_ERROR_OBJECT(srcbin, "Failed to link parsebin's src pad to queue_parsebin's sink pad (error: %d)", ret);
                }
            }
            gst_object_unref(queue_parsebin_sink_pad);
        }
        else {
            GST_ERROR_OBJECT(srcbin, "Failed to get queue_parsebin's sink pad");
        }
    }
    if (new_pad_caps)
        gst_caps_unref(new_pad_caps);
}

gint stream_start_count = 0;

/* Custom event generator probe
 * Generates and injects custom events for stream start and EOS handling
 */
static GstPadProbeReturn custom_event_generator_probe(GstPad *pad, GstPadProbeInfo *info, gpointer user_data) {
    DynamicSrcBin *srcbin = DYNAMIC_SRC_BIN(user_data);
    if (GST_PAD_PROBE_INFO_TYPE(info) & GST_PAD_PROBE_TYPE_EVENT_DOWNSTREAM) {
        GstEvent *event = gst_pad_probe_info_get_event(info);
        if (GST_EVENT_TYPE(event) == GST_EVENT_STREAM_START) {
            GST_INFO_OBJECT(srcbin, "Stream-start event received on parsebin's src pad. Sending custom-stream-start event downstream");
            // Minimize critical section
            gint current_source_id = -1;
            g_mutex_lock(&srcbin->source_id_list_lock);
            if (!g_queue_is_empty(srcbin->srcbin_source_queue)) {
                current_source_id = GPOINTER_TO_INT(g_queue_peek_head(srcbin->srcbin_source_queue));
            }
            g_mutex_unlock(&srcbin->source_id_list_lock);

            // Create and push events outside the lock
            if (current_source_id != -1) {
                GST_DEBUG_OBJECT(srcbin, "Source ID to be sent with custom-stream-start-event: %d", current_source_id);
                GstEvent *custom_event = gst_event_new_custom(
                    GST_EVENT_CUSTOM_DOWNSTREAM,
                    gst_structure_new("custom-stream-start-event",
                                    "source-id", G_TYPE_INT, current_source_id,
                                    NULL));

                if (!gst_pad_push_event(pad, custom_event)) {
                    GST_ERROR_OBJECT(srcbin, "Failed to push custom-stream-start-event");
                }
                GST_DEBUG_OBJECT(srcbin, "Custom-stream-start-event pushed downstream");
            }
            else {
                GST_WARNING_OBJECT(srcbin, "No current source ID found");
            }
            return GST_PAD_PROBE_OK;
        }
        else if (GST_EVENT_TYPE(event) == GST_EVENT_EOS) {
            gint current_source_id = -1;

            g_mutex_lock(&srcbin->source_id_list_lock);
            if (!g_queue_is_empty(srcbin->srcbin_source_queue)) {
                gpointer source_id_ptr = g_queue_peek_head(srcbin->srcbin_source_queue);
                current_source_id = GPOINTER_TO_INT(source_id_ptr);
                GST_INFO_OBJECT(srcbin, "Source ID to be sent with custom-eos-event: %d", current_source_id);
            }
            else {
                GST_WARNING_OBJECT(srcbin, "No source ID found");
            }
            g_mutex_unlock(&srcbin->source_id_list_lock);

            // Create the custom event
            GstEvent *custom_event = gst_event_new_custom(
                GST_EVENT_CUSTOM_DOWNSTREAM,
                gst_structure_new("custom-eos-event",
                                "source-id", G_TYPE_INT, current_source_id,
                                NULL));

            // Push the custom event downstream
            if (!gst_pad_push_event(pad, custom_event)) {
                GST_ERROR_OBJECT(srcbin, "Failed to push custom-eos-event");
            }
            GST_DEBUG_OBJECT(srcbin, "EOS event pushed downstream");

            // Create gap event with valid timestamp (0) and duration (1)
            GstEvent *gap = gst_event_new_gap(0, 1);
            if (!gst_pad_push_event(pad, gap)) {
                GST_ERROR_OBJECT(srcbin, "Failed to push gap event");
            }
            GST_DEBUG_OBJECT(srcbin, "Gap event pushed downstream");

            return GST_PAD_PROBE_DROP;
        }
    }
    return GST_PAD_PROBE_OK;
}

/* Signal handler for adding new sources to the bin
 * Manages source addition and initial pipeline setup
 */
static void dynamic_src_bin_add_source(DynamicSrcBin *srcbin, const gchar *file_path, gint source_id) {
    GST_INFO_OBJECT(srcbin, "SOURCE ADD SIGNAL SOURCE ID: %d", source_id);

    // Validate file path
    if (!file_path || !g_file_test(file_path, G_FILE_TEST_EXISTS)) {
        GST_ERROR_OBJECT(srcbin, "Invalid file path: %s", file_path);
        return;
    }

    g_mutex_lock(&srcbin->source_id_list_lock);

    // Prevent duplicate source IDs
    if (g_hash_table_lookup(srcbin->source_map, GINT_TO_POINTER(source_id))) {
        GST_WARNING_OBJECT(srcbin, "Source ID %d already exists", source_id);
        g_mutex_unlock(&srcbin->source_id_list_lock);
        return;
    }

    // Store source_id -> file_path mapping
    g_hash_table_insert(srcbin->source_map, GINT_TO_POINTER(source_id),
                       g_strdup(file_path));

    // Add source_id to both queues for tracking
    // g_queue_push_tail(srcbin->eos_source_queue, GINT_TO_POINTER(source_id));
    g_queue_push_tail(srcbin->srcbin_source_queue, GINT_TO_POINTER(source_id));

    GST_DEBUG_OBJECT(srcbin, "Added file to queue: %s (%d)", file_path, source_id);

    // Handle first source differently - needs to be linked to pipeline
    if (!srcbin->first_source_added) {
        g_object_set(srcbin->filesrc, "location", file_path, NULL);
        gst_bin_add(GST_BIN(srcbin), srcbin->queue_filesrc);
        gst_bin_add(GST_BIN(srcbin), srcbin->filesrc);
        gst_element_link(srcbin->queue_filesrc, srcbin->parsebin);
        gst_element_link(srcbin->filesrc, srcbin->queue_filesrc);
        gst_element_sync_state_with_parent(srcbin->queue_filesrc);
        gst_element_sync_state_with_parent(srcbin->filesrc);
        srcbin->first_source_added = TRUE;
    }

    g_mutex_unlock(&srcbin->source_id_list_lock);
}

/* Signal handler for removing sources from the bin
 * Handles source cleanup and EOS generation
 */
static void dynamic_src_bin_remove_source(DynamicSrcBin *srcbin, gint source_id) {
    GST_INFO_OBJECT(srcbin, "SOURCE REMOVE SIGNAL SOURCE ID: %d", source_id);
    gboolean send_eos = TRUE;
    g_mutex_lock(&srcbin->source_id_list_lock);
    if (source_id != GPOINTER_TO_INT(g_queue_peek_head(srcbin->srcbin_source_queue))) {
        if (g_hash_table_lookup(srcbin->source_map, GINT_TO_POINTER(source_id)) &&
            g_queue_find(srcbin->srcbin_source_queue, GINT_TO_POINTER(source_id))) {
                g_hash_table_remove(srcbin->source_map, GINT_TO_POINTER(source_id));
                g_queue_remove(srcbin->srcbin_source_queue, GINT_TO_POINTER(source_id));
                GST_INFO_OBJECT(srcbin, "Source ID %d removed from source map and source queue", source_id);
        } else {
            GST_WARNING_OBJECT(srcbin, "Source ID %d not found in source map or source queue", source_id);
        }
        send_eos = FALSE;
    }
    g_mutex_unlock(&srcbin->source_id_list_lock);

    if (send_eos) {
        GST_DEBUG_OBJECT(srcbin, "Sending EOS event to srcbin");
        if (gst_element_send_event(GST_ELEMENT(srcbin), gst_event_new_eos())) {
            GST_DEBUG_OBJECT(srcbin, "EOS event sent successfully");
        }
        else {
            GST_ERROR_OBJECT(srcbin, "Failed to send EOS event");
        }
    }
}

/* Signal handler for terminating the pipeline
 * Sends EOS event and posts termination message to exit the pipeline
 */
static void dynamic_src_bin_terminate(DynamicSrcBin *srcbin) {
    GST_INFO_OBJECT(srcbin, "TERMINATE SIGNAL received - exiting pipeline");
    // Send EOS event from the decoder element using push_event
    if (srcbin->decoder && GST_IS_ELEMENT(srcbin->decoder)) {
        GstPad *decoder_src_pad = gst_element_get_static_pad(srcbin->decoder, "src");
        if (decoder_src_pad) {
            GstEvent *eos_event = gst_event_new_eos();
            if (gst_pad_push_event(decoder_src_pad, eos_event)) {
                GST_DEBUG_OBJECT(srcbin, "EOS event pushed successfully from decoder");
            } else {
                GST_ERROR_OBJECT(srcbin, "Failed to push EOS event from decoder");
            }
            gst_object_unref(decoder_src_pad);
        } else {
            GST_ERROR_OBJECT(srcbin, "Failed to get decoder src pad for EOS");
        }
    } else {
        GST_ERROR_OBJECT(srcbin, "Decoder element is not available for EOS");
    }
}

static void srcbin_child_added(GstChildProxy *child_proxy, GObject *object,
                                gchar *name, gpointer user_data) {
    DynamicSrcBin *srcbin = DYNAMIC_SRC_BIN(user_data);

    GST_INFO_OBJECT(srcbin, "NvDynamicSrcBin Child Added: %s", name);

    if (g_strrstr(name, "parsebin") == name) {
        srcbin->parsebin = GST_ELEMENT(object);
        // Update the callback name here
        g_signal_connect(srcbin->parsebin, "pad-added",
                        G_CALLBACK(parsebin_pad_added_handler), srcbin);
    }
}

/* Initialization function for the DynamicSrcBin class
 * Sets up properties, signals, and pad templates
 */
static void dynamic_src_bin_class_init(DynamicSrcBinClass *klass) {
    GObjectClass *gobject_class = G_OBJECT_CLASS(klass);
    GstElementClass *gstelement_class = GST_ELEMENT_CLASS(klass);

    // Signal to append source to source_id_queue
    g_signal_new(
        "add-source",
        G_TYPE_FROM_CLASS(klass),
        G_SIGNAL_RUN_LAST | G_SIGNAL_ACTION,
        0,
        NULL, NULL,
        g_cclosure_marshal_generic,
        G_TYPE_NONE,  // return type
        2,            // Number of parameters
        G_TYPE_STRING, // First parameter type (file path)
        G_TYPE_INT    // Second parameter type (source_id)
    );

    // Signal to remove source from source_id_queue
    g_signal_new(
        "remove-source",
        G_TYPE_FROM_CLASS(klass),
        G_SIGNAL_RUN_LAST | G_SIGNAL_ACTION,
        0,
        NULL, NULL,
        g_cclosure_marshal_generic,
        G_TYPE_NONE,  // return type
        1,            // Number of parameters
        G_TYPE_INT    // Parameter type (source_id)
    );

    // Signal to terminate the pipeline
    g_signal_new(
        "terminate",
        G_TYPE_FROM_CLASS(klass),
        G_SIGNAL_RUN_LAST | G_SIGNAL_ACTION,
        0,
        NULL, NULL,
        g_cclosure_marshal_generic,
        G_TYPE_NONE,  // return type
        0,            // Number of parameters (no parameters needed)
        G_TYPE_NONE   // No parameter types
    );



    static GstStaticPadTemplate src_template = GST_STATIC_PAD_TEMPLATE(
        "src",
        GST_PAD_SRC,
        GST_PAD_ALWAYS,
        GST_STATIC_CAPS_ANY
    );

    gst_element_class_add_pad_template(gstelement_class,
    gst_static_pad_template_get(&src_template));

    gobject_class->set_property = dynamic_src_bin_set_property;
    gobject_class->get_property = dynamic_src_bin_get_property;
    gobject_class->dispose = dynamic_src_bin_dispose;

    g_object_class_install_property(gobject_class, PROP_CURRENT_FILE,
        g_param_spec_string("current-file", "Current File",
                          "Current file being processed",
                          NULL,
                          G_PARAM_READABLE | G_PARAM_STATIC_STRINGS));

    g_object_class_install_property(gobject_class, PROP_CURRENT_ID,
        g_param_spec_int("current-id", "Current Id",
                          "Current id of the chunk being processed",
                          0, G_MAXINT, 0,
                          G_PARAM_READABLE | G_PARAM_STATIC_STRINGS));

    // In class_init function
    g_object_class_install_property(gobject_class, PROP_GPU_ID,
        g_param_spec_uint("gpu-id", "GPU ID",
                        "GPU Device ID to use for decoding",
                        0, G_MAXUINT, 0,
                        G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
}

/* Instance initialization function for DynamicSrcBin
 * Creates and configures all necessary GStreamer elements
 */
static void dynamic_src_bin_init(DynamicSrcBin *srcbin) {
    // Initialize mutex for thread-safe operations
    g_mutex_init(&srcbin->source_id_list_lock);

    // Initialize data structures for source management
    srcbin->source_map = g_hash_table_new_full(g_direct_hash, g_direct_equal,
                                              NULL, g_free);
    // srcbin->eos_source_queue = g_queue_new();
    srcbin->srcbin_source_queue = g_queue_new();
    srcbin->first_source_added = FALSE;
    srcbin->current_source_id = -1;

    // Create GStreamer elements
    srcbin->filesrc = gst_element_factory_make("filesrc", "source"); // For reading files
    srcbin->queue_filesrc = gst_element_factory_make("queue", "queue_filesrc"); // For buffering
    srcbin->parsebin = gst_element_factory_make("parsebin", "parsebin"); // For parsing media
    srcbin->queue_parsebin = gst_element_factory_make("queue", "queue_parsebin"); // For buffering after parsebin
    srcbin->decoder = gst_element_factory_make("nvv4l2decoder", "decoder"); // For decoding media

    g_object_set(srcbin->queue_parsebin, "max-size-buffers", 1, NULL);
    // Add child-added signal connection to srcbin
    g_signal_connect(G_OBJECT(srcbin), "child-added",
                    G_CALLBACK(srcbin_child_added), srcbin);
    // Connect signal handlers to add and remove source
    g_signal_connect(GST_ELEMENT(srcbin), "add-source",
                    G_CALLBACK(dynamic_src_bin_add_source), NULL);
    g_signal_connect(GST_ELEMENT(srcbin), "remove-source",
                    G_CALLBACK(dynamic_src_bin_remove_source), NULL);
    g_signal_connect(GST_ELEMENT(srcbin), "terminate",
                    G_CALLBACK(dynamic_src_bin_terminate), NULL);

    // Validate element creation
    if (!srcbin->filesrc || !srcbin->queue_filesrc || !srcbin->parsebin || !srcbin->queue_parsebin || !srcbin->decoder) {
        GST_ERROR_OBJECT(srcbin, "Failed to create elements within dynamic src bin");
        return;
    }

    // Add elements to bin (except filesrc which is added dynamically)
    gst_bin_add(GST_BIN(srcbin), srcbin->parsebin);
    gst_bin_add(GST_BIN(srcbin), srcbin->queue_parsebin);
    gst_bin_add(GST_BIN(srcbin), srcbin->decoder);

    // Link queue_parsebin -> decoder
    gst_element_link(srcbin->queue_parsebin, srcbin->decoder);

    // Create ghost pad for srcbin's output
    if (!gst_element_add_pad(GST_ELEMENT(srcbin),
        gst_ghost_pad_new_no_target("src", GST_PAD_SRC))) {
        GST_ERROR_OBJECT(srcbin, "Failed to add ghost pad in dynamic source bin");
    }

    // Link decoder -> srcbin
    GstPad *ghost_pad = gst_element_get_static_pad(GST_ELEMENT(srcbin), "src");
    GstPad *decoder_src_pad = gst_element_get_static_pad(srcbin->decoder, "src");

    if (!gst_ghost_pad_set_target(GST_GHOST_PAD(ghost_pad), decoder_src_pad)) {
        GST_ERROR_OBJECT(srcbin, "Failed to link decoder's src pad to srcbin's ghost pad");
    }
    else {
        // Add probe to srcbin's ghost pad
        gst_pad_add_probe(ghost_pad,
                    GST_PAD_PROBE_TYPE_BUFFER | GST_PAD_PROBE_TYPE_EVENT_DOWNSTREAM,
                    srcbin_src_pad_probe_callback,
                    srcbin,
                    NULL);
    }
    gst_object_unref(ghost_pad);
    gst_object_unref(decoder_src_pad);

    // Add probe to queue_parsebin's src pad
    GstPad *queue_parsebin_src_pad = gst_element_get_static_pad(srcbin->queue_parsebin, "src");
    if (queue_parsebin_src_pad) {
        gst_pad_add_probe(queue_parsebin_src_pad,
                        GST_PAD_PROBE_TYPE_EVENT_DOWNSTREAM,
                        (GstPadProbeCallback)custom_event_generator_probe,
                        srcbin,
                        NULL);
        gst_object_unref(queue_parsebin_src_pad);
    }
    else {
        GST_ERROR_OBJECT(srcbin, "Failed to get queue_parsebin's src pad");
    }
}

/* Plugin initialization function
 * Registers the plugin with GStreamer
 */
static gboolean plugin_init(GstPlugin *plugin) {
    GST_DEBUG_CATEGORY_INIT(dynamic_src_bin_debug, "nvdsdynamicsrcbin", 0, "NVIDIA Dynamic Source Bin");
    return gst_element_register(plugin, "nvdsdynamicsrcbin",
                              GST_RANK_PRIMARY,
                              DYNAMIC_SRC_BIN_TYPE);
}

#define PACKAGE "nvdsdynamicsrcbin"

#define VERSION "1.0.0"
#define PACKAGE_DESCRIPTION "Gstreamer plugin to read multiple files"
/* Define under which licence the package has been released */
#define PACKAGE_LICENSE "Proprietary"
#define PACKAGE_NAME "GStreamer nvdynamicsrcbin Plugin"
/* Define to the home page for this package. */
#define PACKAGE_URL "http://nvidia.com/"


GST_PLUGIN_DEFINE (GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    nvdsgst_dynamicsrcbin,
    PACKAGE_DESCRIPTION,
    plugin_init, VERSION, PACKAGE_LICENSE, PACKAGE_NAME, PACKAGE_URL)