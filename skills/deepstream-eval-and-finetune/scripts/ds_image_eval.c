// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * ds_image_eval — deployed-engine accuracy harness for deepstream-eval-and-finetune.
 *
 * Decodes a numbered folder of JPEGs DIRECTLY (no MJPEG container, no re-encode),
 * runs them through the SAME nvinfer config the import skill deployed (TRT engine
 * + compiled custom bbox parser + nvinfer preprocessing), and writes one detection
 * line per object. The frame number (== image index, because multifilesrc reads
 * NNNNNN.jpg in order through a single source) gives deterministic image_id
 * alignment for mAP scoring.
 *
 * Pipeline:
 *   multifilesrc location=DIR/%06d.jpg index=0 stop-index=N-1 caps=image/jpeg
 *     ! jpegparse ! nvv4l2decoder ! nvvideoconvert
 *     ! "video/x-raw(memory:NVMM),format=RGBA"
 *     ! nvstreammux (W x H, batch-size=1)
 *     ! nvinfer config-file-path=<custom config>
 *     ! fakesink
 *
 * A buffer probe on nvinfer's src pad reads NvDsObjectMeta and appends:
 *     <frame_num> <class_id> <confidence> <left> <top> <width> <height>
 * to the output file (rect_params are in nvstreammux pixel space == W x H, which
 * matches the letterboxed ground truth from build_eval_set.py).
 *
 * Usage:
 *   ds_image_eval <nvinfer_config> <image_dir> <num_images> <out_txt> [W] [H]
 */

#include <gst/gst.h>
#include <glib.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gstnvdsmeta.h"

static FILE *g_out = NULL;
static guint g_obj_count = 0;
static guint g_frame_count = 0;

/* Probe on nvinfer src pad: dump every detected object for this frame. */
static GstPadProbeReturn
osd_sink_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info, gpointer u_data)
{
    GstBuffer *buf = (GstBuffer *) info->data;
    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);
    if (!batch_meta)
        return GST_PAD_PROBE_OK;

    for (NvDsMetaList *l_frame = batch_meta->frame_meta_list; l_frame != NULL;
         l_frame = l_frame->next) {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) l_frame->data;
        /* frame_num is the per-source running index == image index. */
        guint frame_num = frame_meta->frame_num;
        g_frame_count++;

        for (NvDsMetaList *l_obj = frame_meta->obj_meta_list; l_obj != NULL;
             l_obj = l_obj->next) {
            NvDsObjectMeta *obj = (NvDsObjectMeta *) l_obj->data;
            NvOSD_RectParams *r = &obj->rect_params;  /* nvstreammux pixel space */
            fprintf(g_out, "%u %d %.6f %.4f %.4f %.4f %.4f\n",
                    frame_num, obj->class_id, obj->confidence,
                    r->left, r->top, r->width, r->height);
            g_obj_count++;
        }
    }
    return GST_PAD_PROBE_OK;
}

static gboolean
bus_call(GstBus *bus, GstMessage *msg, gpointer data)
{
    GMainLoop *loop = (GMainLoop *) data;
    switch (GST_MESSAGE_TYPE(msg)) {
    case GST_MESSAGE_EOS:
        g_print("[ds_image_eval] EOS: %u frames, %u objects\n",
                g_frame_count, g_obj_count);
        g_main_loop_quit(loop);
        break;
    case GST_MESSAGE_ERROR: {
        gchar *dbg = NULL;
        GError *err = NULL;
        gst_message_parse_error(msg, &err, &dbg);
        g_printerr("[ds_image_eval] ERROR from %s: %s\n",
                   GST_OBJECT_NAME(msg->src), err->message);
        if (dbg) g_printerr("[ds_image_eval] debug: %s\n", dbg);
        g_clear_error(&err);
        g_free(dbg);
        g_main_loop_quit(loop);
        break;
    }
    default:
        break;
    }
    return TRUE;
}

static GstElement *make(const char *factory, const char *name)
{
    GstElement *e = gst_element_factory_make(factory, name);
    if (!e) {
        g_printerr("[ds_image_eval] failed to create element: %s (%s)\n", name, factory);
        exit(1);
    }
    return e;
}

int main(int argc, char *argv[])
{
    if (argc < 5) {
        g_printerr("Usage: %s <nvinfer_config> <image_dir> <num_images> <out_txt> [W] [H]\n",
                   argv[0]);
        return 1;
    }
    const char *nvinfer_config = argv[1];
    const char *image_dir = argv[2];
    int num_images = atoi(argv[3]);
    const char *out_txt = argv[4];
    int width = (argc > 5) ? atoi(argv[5]) : 1280;
    int height = (argc > 6) ? atoi(argv[6]) : 720;

    g_out = fopen(out_txt, "w");
    if (!g_out) {
        g_printerr("[ds_image_eval] cannot open output: %s\n", out_txt);
        return 1;
    }

    gst_init(&argc, &argv);
    GMainLoop *loop = g_main_loop_new(NULL, FALSE);

    GstElement *pipeline = gst_pipeline_new("ds-image-eval");
    GstElement *source   = make("multifilesrc", "file-source");
    GstElement *parser   = make("jpegparse", "jpeg-parser");
    GstElement *decoder  = make("nvv4l2decoder", "nvv4l2-decoder");
    GstElement *conv     = make("nvvideoconvert", "nvvidconv");
    GstElement *capsf    = make("capsfilter", "nvmm-caps");
    GstElement *streammux = make("nvstreammux", "stream-muxer");
    GstElement *pgie     = make("nvinfer", "primary-nvinference");
    GstElement *sink     = make("fakesink", "fake-sink");

    /* multifilesrc: read DIR/%06d.jpg, indices 0..N-1, as a JPEG stream. */
    gchar *location = g_strdup_printf("%s/%%06d.jpg", image_dir);
    GstCaps *jpeg_caps = gst_caps_from_string("image/jpeg,framerate=30/1");
    g_object_set(G_OBJECT(source),
                 "location", location,
                 "index", 0,
                 "stop-index", num_images - 1,
                 "loop", FALSE,
                 "caps", jpeg_caps, NULL);
    gst_caps_unref(jpeg_caps);
    g_free(location);

    GstCaps *nvmm = gst_caps_from_string("video/x-raw(memory:NVMM),format=RGBA");
    g_object_set(G_OBJECT(capsf), "caps", nvmm, NULL);
    gst_caps_unref(nvmm);

    g_object_set(G_OBJECT(streammux),
                 "batch-size", 1,
                 "width", width,
                 "height", height,
                 "batched-push-timeout", 40000, NULL);

    g_object_set(G_OBJECT(pgie), "config-file-path", nvinfer_config, NULL);
    g_object_set(G_OBJECT(sink), "sync", FALSE, "async", FALSE, NULL);

    gst_bin_add_many(GST_BIN(pipeline), source, parser, decoder, conv, capsf,
                     streammux, pgie, sink, NULL);

    /* Link source chain up to the converter/caps, then into streammux sink_0. */
    if (!gst_element_link_many(source, parser, decoder, conv, capsf, NULL)) {
        g_printerr("[ds_image_eval] failed to link source -> capsfilter\n");
        return 1;
    }
    GstPad *mux_sink = gst_element_request_pad_simple(streammux, "sink_0");
    GstPad *caps_src = gst_element_get_static_pad(capsf, "src");
    if (gst_pad_link(caps_src, mux_sink) != GST_PAD_LINK_OK) {
        g_printerr("[ds_image_eval] failed to link capsfilter -> streammux\n");
        return 1;
    }
    gst_object_unref(caps_src);
    gst_object_unref(mux_sink);

    if (!gst_element_link_many(streammux, pgie, sink, NULL)) {
        g_printerr("[ds_image_eval] failed to link streammux -> nvinfer -> sink\n");
        return 1;
    }

    /* Probe nvinfer src pad: metadata is populated after inference. */
    GstPad *pgie_src = gst_element_get_static_pad(pgie, "src");
    gst_pad_add_probe(pgie_src, GST_PAD_PROBE_TYPE_BUFFER,
                      osd_sink_pad_buffer_probe, NULL, NULL);
    gst_object_unref(pgie_src);

    GstBus *bus = gst_pipeline_get_bus(GST_PIPELINE(pipeline));
    guint bus_watch_id = gst_bus_add_watch(bus, bus_call, loop);
    gst_object_unref(bus);

    g_print("[ds_image_eval] config=%s dir=%s n=%d canvas=%dx%d -> %s\n",
            nvinfer_config, image_dir, num_images, width, height, out_txt);
    gst_element_set_state(pipeline, GST_STATE_PLAYING);
    g_main_loop_run(loop);

    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(GST_OBJECT(pipeline));
    g_source_remove(bus_watch_id);
    g_main_loop_unref(loop);
    fclose(g_out);
    return 0;
}
