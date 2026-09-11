/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <glib.h>


#define NVGSTDS_ELEM_ADD_PROBE(probe_id, elem, pad, probe_func, probe_type, probe_data) \
      do { \
        GstPad *gstpad = gst_element_get_static_pad (elem, pad); \
        if (!gstpad) { \
          g_print ("Could not find '%s' in '%s'", pad, \
              GST_ELEMENT_NAME(elem)); \
        } \
        probe_id = gst_pad_add_probe(gstpad, (probe_type), probe_func, probe_data, NULL); \
        gst_object_unref (gstpad); \
      } while (0)

static gboolean
seek_decode (gpointer data)
{
  GstElement *bin = (GstElement *) data;
  gboolean ret = TRUE;

  gst_element_set_state (bin, GST_STATE_PAUSED);

  g_print ("seeking to start %s\n", GST_ELEMENT_NAME(bin));
  ret = gst_element_seek (bin, 1.0, GST_FORMAT_TIME,
      (GstSeekFlags) (GST_SEEK_FLAG_KEY_UNIT | GST_SEEK_FLAG_FLUSH),
      GST_SEEK_TYPE_SET, 0, GST_SEEK_TYPE_NONE, GST_CLOCK_TIME_NONE);

  if (!ret)
    GST_WARNING ("Error in seeking pipeline");

  gst_element_set_state (bin, GST_STATE_PLAYING);

  return FALSE;
}

//static guint64 prev_accumulated_base = 0;
//static guint64 accumulated_base = 0;

typedef struct _decoder_data
{
    guint64 prev_accumulated_base;
    guint64 accumulated_base;
    GstElement *decoder;
}decoder_data;

static GstPadProbeReturn
restart_stream_buf_prob (GstPad * pad, GstPadProbeInfo * info,
    gpointer u_data)
{
  GstEvent *event = GST_EVENT (info->data);
  //GstElement *bin = (GstElement *) u_data;
  decoder_data *data = (decoder_data *) u_data;

  //g_print ("inside buffer probe function\n");
  if ((info->type & GST_PAD_PROBE_TYPE_BUFFER)) {
      GST_BUFFER_PTS(GST_BUFFER(info->data)) += data->prev_accumulated_base;
  }
  if ((info->type & GST_PAD_PROBE_TYPE_EVENT_BOTH)) {
    if (GST_EVENT_TYPE (event) == GST_EVENT_EOS) {
      g_timeout_add (1, seek_decode, data->decoder);
    }
    
    if (GST_EVENT_TYPE (event) == GST_EVENT_SEGMENT) {
        GstSegment *segment;

        gst_event_parse_segment (event, (const GstSegment **) &segment);
        segment->base = data->accumulated_base;
        data->prev_accumulated_base = data->accumulated_base;
        data->accumulated_base += segment->stop;
    }

    switch (GST_EVENT_TYPE (event)) {
      case GST_EVENT_EOS:
        /* QOS events from downstream sink elements cause decoder to drop
         * frames after looping the file since the timestamps reset to 0.
         * We should drop the QOS events since we have custom logic for
         * looping individual sources. */
         //g_print ("GST_EVENT_EOS received\n");
      case GST_EVENT_QOS:
      case GST_EVENT_SEGMENT:
      case GST_EVENT_FLUSH_START:
      case GST_EVENT_FLUSH_STOP:
        return GST_PAD_PROBE_DROP;
      default:
        break;
    }
  }
  return GST_PAD_PROBE_OK;
}

gboolean bus_call(GstBus *bus, GstMessage *msg, gpointer data)
{
    GstElement *pipeline = GST_ELEMENT(data);
    switch (GST_MESSAGE_TYPE(msg))
    {
        case GST_MESSAGE_STATE_CHANGED: 
            if (GST_ELEMENT (GST_MESSAGE_SRC (msg)) == pipeline)
            {
                GstState oldstate, newstate; 
                gst_message_parse_state_changed (msg, &oldstate, &newstate, NULL);
                switch (newstate) {
                    case GST_STATE_PLAYING:
                        //g_print ("Pipeline running\n");
                        GST_DEBUG_BIN_TO_DOT_FILE_WITH_TS (GST_BIN (pipeline),
                                GST_DEBUG_GRAPH_SHOW_ALL, "app-playing");
                        break;
                    default:
                        break;
                }
                break;
            }
            break;
        case GST_MESSAGE_EOS:
#if 0
            g_print ("buss_callback with message EOS received\n");
            /* restart playback if at end */
            gst_element_set_state (pipeline, GST_STATE_PAUSED);

            if (!gst_element_seek(pipeline, 
                        1.0, GST_FORMAT_TIME, GST_SEEK_FLAG_FLUSH | GST_SEEK_FLAG_KEY_UNIT,
                        GST_SEEK_TYPE_SET,  0, //2000000000, //2 seconds (in nanoseconds)
                        GST_SEEK_TYPE_NONE, GST_CLOCK_TIME_NONE)) {
                g_print("Seek failed!\n");
            }

            gst_element_set_state (pipeline, GST_STATE_PLAYING);
#endif

            break;
        case GST_MESSAGE_WARNING:
            {
                GError *error = NULL;
                gchar *debuginfo = NULL;
                gst_message_parse_warning (msg, &error, &debuginfo);
                g_printerr ("WARNING from %s: %s\n",
                        GST_OBJECT_NAME (msg->src), error->message);
                if (debuginfo) {
                    g_printerr ("Debug info: %s\n", debuginfo);
                }
                g_error_free (error);
                g_free (debuginfo);
                break;
            }
        case GST_MESSAGE_ERROR:
            {
                GError *error = NULL;
                gchar *debuginfo = NULL;
                gst_message_parse_error (msg, &error, &debuginfo);
                g_printerr ("ERROR from %s: %s\n",
                        GST_OBJECT_NAME (msg->src), error->message);
                if (debuginfo) {
                    g_printerr ("Debug info: %s\n", debuginfo);
                }
                g_error_free (error);
                g_free (debuginfo);
                break;
            }
        default:
            break;
    }
    return TRUE;
}

static void
on_pad_added (GstElement *element,
              GstPad     *pad,
              gpointer    data)
{
  GstPad *sinkpad;
  GstElement *parser = (GstElement *) data;

  /* We can now link this pad with the vorbis-decoder sink pad */
  g_print ("Dynamic pad created, linking demuxer/parser\n");

  sinkpad = gst_element_get_static_pad (parser, "sink");

  gst_pad_link (pad, sinkpad);

  gst_object_unref (sinkpad);
}

void init_decoder_data (decoder_data *dec_data, GstElement *decoder)
{
    dec_data->prev_accumulated_base = 0;
    dec_data->accumulated_base = 0;
    dec_data->decoder = decoder;
}

gint
main (gint   argc,
      gchar *argv[])
{
    GMainLoop *loop;

    GstElement *pipeline, *sink, *nvblender;

    GstElement *bin;
    GstElement *source, *demuxer, *parser, *decoder, *conv, *caps_filter;
    decoder_data *dec_data;
    GstCaps *caps;
    GstCapsFeatures *feature = NULL;

    GstElement *bin1;
    GstElement *source1, *demuxer1, *parser1, *decoder1, *conv1, *caps_filter1;
    decoder_data *dec_data1;
    GstCaps *caps1;
    GstCapsFeatures *feature1 = NULL;

    GstPad *blender_sinkpad;
    GstPad *blender_sinkpad1;
    GstBus *bus;
    guint bus_watch_id;

    /* Initialisation */
    gst_init (&argc, &argv);

    loop = g_main_loop_new (NULL, FALSE);

    /* Check input arguments */
    if (argc != 3) {
        g_printerr ("Usage: %s <nvblender_unittest filename>\n", argv[0]);
        return -1;
    }

    dec_data = (decoder_data *) malloc (sizeof (struct _decoder_data));
    if (dec_data == NULL)
    {
        g_print ("Failed to allocate decoder data\n");
        return -1;
    }

    dec_data1 = (decoder_data *) malloc (sizeof (struct _decoder_data));
    if (dec_data1 == NULL)
    {
        g_print ("Failed to allocate decoder data\n");
        free (dec_data);
        return -1;
    }

    /* Create gstreamer elements */
    pipeline = gst_pipeline_new ("nvblender_unittest");
    bin = gst_bin_new ("decode_bin");
    bin1 = gst_bin_new ("decode_bin1");
    
    source          = gst_element_factory_make ("filesrc",       "file-source");
    demuxer         = gst_element_factory_make ("qtdemux",       "mp4-demuxer");
    parser          = gst_element_factory_make ("h264parse",     "h264-parser");
    decoder         = gst_element_factory_make ("nvv4l2decoder", "video-decoder");
    conv            = gst_element_factory_make ("nvvideoconvert","converter");
    caps_filter     = gst_element_factory_make ("capsfilter",    "caps-filter");
    init_decoder_data (dec_data, decoder);

    source1          = gst_element_factory_make ("filesrc",       "file-source1");
    demuxer1         = gst_element_factory_make ("qtdemux",       "mp4-demuxer1");
    parser1          = gst_element_factory_make ("h264parse",     "h264-parser1");
    decoder1         = gst_element_factory_make ("nvv4l2decoder", "video-decoder1");
    conv1            = gst_element_factory_make ("nvvideoconvert","converter1");
    caps_filter1     = gst_element_factory_make ("capsfilter",    "caps-filter1");
    init_decoder_data (dec_data1, decoder1);
    
    nvblender       = gst_element_factory_make ("nvblender", "nv-blender");
    sink            = gst_element_factory_make ("nveglglessink", "video-sink");
    //g_object_set (G_OBJECT(sink), "window-width", 640, NULL);
    //g_object_set (G_OBJECT(sink), "window-height", 480, NULL);

    caps = gst_caps_new_simple ("video/x-raw",
            "format", G_TYPE_STRING, "NV12",
            "width", G_TYPE_INT, 640,
            "height", G_TYPE_INT, 480,
             NULL);
    feature = gst_caps_features_new ("memory:NVMM", NULL);
    gst_caps_set_features (caps, 0, feature);

    g_object_set (G_OBJECT(caps_filter), "caps", caps, NULL);

    caps1 = gst_caps_new_simple ("video/x-raw",
            "format", G_TYPE_STRING, "NV12",
            "width", G_TYPE_INT, 640,
            "height", G_TYPE_INT, 480,
             NULL);
    feature1 = gst_caps_features_new ("memory:NVMM", NULL);
    gst_caps_set_features (caps1, 0, feature1);

    g_object_set (G_OBJECT(caps_filter1), "caps", caps1, NULL);

    if (!pipeline ||
         !source || !demuxer || !parser || !decoder || !conv || !caps_filter ||
         !source1 || !demuxer1 || !parser1 || !decoder1 || !conv1 || !caps_filter1 ||
        !nvblender  || !sink ) {
        g_printerr ("One element could not be created. Exiting.\n");
        free (dec_data);
        free (dec_data1);
        return -1;
    }

    /* Set up the pipeline */

    /* we set the input filename to the source element */
    g_object_set (G_OBJECT (source), "location", argv[1], NULL);
    g_object_set (G_OBJECT (source1), "location", argv[2], NULL);
    
    g_object_set (G_OBJECT (conv), "nvbuf-memory-type", 3, NULL);
    g_object_set (G_OBJECT (conv1), "nvbuf-memory-type", 3, NULL);

    g_object_set (G_OBJECT (sink), "sync", 1, NULL);
    g_object_set (G_OBJECT (sink), "qos", 0, NULL);

#if 1
    gulong src_buffer_probe;
    NVGSTDS_ELEM_ADD_PROBE (src_buffer_probe, decoder,
            "sink", restart_stream_buf_prob,
            (GstPadProbeType) (GST_PAD_PROBE_TYPE_EVENT_BOTH |
                GST_PAD_PROBE_TYPE_EVENT_FLUSH | GST_PAD_PROBE_TYPE_BUFFER),
            dec_data);

    gulong src_buffer_probe1;
    NVGSTDS_ELEM_ADD_PROBE (src_buffer_probe1, decoder1,
            "sink", restart_stream_buf_prob,
            (GstPadProbeType) (GST_PAD_PROBE_TYPE_EVENT_BOTH |
                GST_PAD_PROBE_TYPE_EVENT_FLUSH | GST_PAD_PROBE_TYPE_BUFFER),
            dec_data1);
#endif

    /* we add a message handler */
    bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline));
    bus_watch_id = gst_bus_add_watch (bus, bus_call, pipeline);
    gst_object_unref (bus);

    /* we add all elements into the pipeline */
    /* file-source | ogg-demuxer | vorbis-decoder | converter | alsa-output */
    gst_bin_add_many (GST_BIN (bin),
            source, demuxer, parser, decoder, conv, caps_filter, NULL);
    gst_bin_add_many (GST_BIN (bin1),
            source1, demuxer1, parser1, decoder1, conv1, caps_filter1, NULL);
    gst_bin_add_many (GST_BIN (pipeline), bin, bin1, nvblender, sink, NULL);
    //gst_bin_add_many (GST_BIN (pipeline), bin1, nvblender, sink, NULL);

    blender_sinkpad = gst_element_request_pad_simple (nvblender, "sink_%u");
    GstPad *srcpad = gst_element_get_static_pad (caps_filter, "src");
    gst_pad_link (srcpad, blender_sinkpad);

    blender_sinkpad1 = gst_element_request_pad_simple (nvblender, "sink_%u");
    GstPad *srcpad1 = gst_element_get_static_pad (caps_filter1, "src");
    gst_pad_link (srcpad1, blender_sinkpad1);

    /* we link the elements together */
    /* file-source -> ogg-demuxer ~> vorbis-decoder -> converter -> alsa-output */
#if 1
    gst_element_link_many (source, demuxer, NULL);
    gst_element_link_many (parser, decoder, conv, caps_filter, NULL);
    g_signal_connect (demuxer, "pad-added", G_CALLBACK (on_pad_added), parser);
    gst_element_link_many (caps_filter, nvblender, sink, NULL);

    gst_element_link_many (source1, demuxer1, NULL);
    gst_element_link_many (parser1, decoder1, conv1, caps_filter1, NULL);
    g_signal_connect (demuxer1, "pad-added", G_CALLBACK (on_pad_added), parser1);
    gst_element_link_many (caps_filter1, nvblender, sink, NULL);
#else // for h264 stream input
    gst_element_link_many (source, parser, decoder, conv, sink, NULL);
#endif

    /* Set the pipeline to "playing" state*/
    g_print ("Now playing: %s\n", argv[1]);
    gst_element_set_state (pipeline, GST_STATE_PLAYING);


    /* Iterate */
    g_print ("Running...\n");
    g_main_loop_run (loop);


    /* Out of the main loop, clean up nicely */
    g_print ("Returned, stopping playback\n");
    gst_element_set_state (pipeline, GST_STATE_NULL);

    g_print ("Deleting pipeline\n");
    gst_object_unref (GST_OBJECT (pipeline));
    g_source_remove (bus_watch_id);
    g_main_loop_unref (loop);
    free (dec_data);
    free (dec_data1);

    return 0;
}
