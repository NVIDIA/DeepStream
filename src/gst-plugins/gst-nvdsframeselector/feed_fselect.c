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

/* feed_fselect.c — feed nvdsframeselector the SAME frames RTVI feeds.
 *
 * RTVI decimates the decoded stream with an *equidistant, absolute-timestamp*
 * picker (chunk_start + i/fps), not `videorate`. `videorate` phase-locks to the
 * first buffer (e.g. 0.0667s when the first frame isn't at 0), landing off the
 * 0.5s grid. This app replaces `videorate` with a pad probe that keeps the first
 * frame whose PTS crosses each absolute i/fps mark — exactly like RTVI's
 * choose_frame — so on a 30fps clip it emits 60.0, 60.5, 61.0 ... matching RTVI.
 *
 * Pipeline:
 *   filesrc ! parsebin ! nvv4l2decoder ! nvvideoconvert ! caps(WxH,NVMM)
 *           ! [equidistant probe] ! nvdsframeselector(<props>)
 *           ! ( fakesink | nvvideoconvert ! nvv4l2h264enc ! h264parse ! qtmux ! filesink )
 *
 * Build:
 *   gcc -O2 -Wall feed_fselect.c -o feed_fselect $(pkg-config --cflags --libs gstreamer-1.0)
 *
 * Run:
 *   ./feed_fselect \
 *       --input ~/dedup/small_streams/admin.mp4 --fps 2 \
 *       --cache-size 20 --selection-count 20 --optical-flow-interval 0 \
 *       --enable-motion-detection 1 --source-fps 30 \
 *       --width 1920 --height 1080 --output test_dump.mp4
 *
 * Frame-selection algorithm is a 3-way enum on nvdsframeselector
 * (--frame-selection-algorithm): 0=BASIC (equidistant), 1=RANGE_BASED (SAD),
 * 2=OF (optical-flow, the default). Append e.g. `--frame-selection-algorithm 1`
 * to the run above to compare RANGE_BASED against the default OF.
 */
#include <gst/gst.h>
#include <glib.h>

/* ---- CLI options (glib GOptionEntry) ---- */
static gchar   *opt_input   = NULL;
static gdouble  opt_fps     = 2.0;
static gint     opt_width   = 1920;
static gint     opt_height  = 1080;
static gdouble  opt_start   = 0.0;
static gint     opt_cache   = 20;
static gint     opt_sel     = 20;
static gint     opt_ofi     = 0;      /* 0 = AUTO */
static gint     opt_motion  = 1;
static gdouble  opt_srcfps  = 30.0;
static gint     opt_excl    = 2;
static gint     opt_algo    = -1;     /* -1 = leave plugin default */
static gint     opt_equidist = -1;    /* -1 = leave plugin default (FALSE) */
static gdouble  opt_tighten  = -1.0;  /* -1 = leave plugin default (0.5) */
static gchar   *opt_output  = NULL;   /* NULL/"" = fakesink */
static gboolean opt_verbose = FALSE;

static GOptionEntry entries[] = {
  {"input",  'i', 0, G_OPTION_ARG_STRING, &opt_input,  "input video file", "FILE"},
  {"fps",     0,  0, G_OPTION_ARG_DOUBLE, &opt_fps,    "decimation rate fed to plugin (frames/s on the absolute grid; RTVI = num_frames/chunk_dur = 2). default 2", "N"},
  {"width",   0,  0, G_OPTION_ARG_INT,    &opt_width,  "feed width  (default 1920)", "W"},
  {"height",  0,  0, G_OPTION_ARG_INT,    &opt_height, "feed height (default 1080)", "H"},
  {"start",   0,  0, G_OPTION_ARG_DOUBLE, &opt_start,  "abs time (s) of first grid mark (default 0)", "S"},
  {"cache-size",             0, 0, G_OPTION_ARG_INT,    &opt_cache,  "nvdsframeselector cache-size (default 20)", "N"},
  {"selection-count",        0, 0, G_OPTION_ARG_INT,    &opt_sel,    "selection-count (default 20)", "N"},
  {"optical-flow-interval",  0, 0, G_OPTION_ARG_INT,    &opt_ofi,    "0 = AUTO (plugin derives it). default 0", "N"},
  {"enable-motion-detection",0, 0, G_OPTION_ARG_INT,    &opt_motion, "1/0 (default 1)", "N"},
  {"source-fps",             0, 0, G_OPTION_ARG_DOUBLE, &opt_srcfps, "TRUE source fps for motion normalization (default 30)", "N"},
  {"exclusion-range",        0, 0, G_OPTION_ARG_INT,    &opt_excl,   "exclusion-range (default 2)", "N"},
  {"frame-selection-algorithm",0,0,G_OPTION_ARG_INT,    &opt_algo,   "optional: override plugin algorithm", "N"},
  {"equidistant-output",      0,0,G_OPTION_ARG_INT,    &opt_equidist,"1=output equidistant frames (count from OF); 0=motion-ranked (default)", "N"},
  {"of-tighten",             0,0,G_OPTION_ARG_DOUBLE, &opt_tighten, "tighten factor for OF frame count (0.0,1.0]; default 0.5", "F"},
  {"output", 'o', 0, G_OPTION_ARG_STRING, &opt_output, "encode selection to this .mp4 (else fakesink)", "FILE"},
  {"verbose",'v', 0, G_OPTION_ARG_NONE,   &opt_verbose,"print each frame the probe KEEPS", NULL},
  {NULL}
};

/* ---- equidistant absolute-timestamp probe state ---- */
typedef struct {
  gdouble  next;     /* next grid target (s)   */
  gdouble  step;     /* 1/fps                  */
  guint    kept;
  gboolean verbose;
} ProbeState;

/* Keep the first frame whose PTS has reached each absolute grid mark
 * start + k/fps, then advance the target. This is RTVI's choose_frame logic
 * and lands exactly on the grid when a frame exists there (30fps -> 0.5s). */
static GstPadProbeReturn
grid_probe (GstPad *pad, GstPadProbeInfo *info, gpointer user_data)
{
  ProbeState *st = (ProbeState *) user_data;
  GstBuffer  *buf = GST_PAD_PROBE_INFO_BUFFER (info);
  const gdouble EPS = 1e-6;

  if (!buf || !GST_BUFFER_PTS_IS_VALID (buf))
    return GST_PAD_PROBE_DROP;

  gdouble pts = (gdouble) GST_BUFFER_PTS (buf) / (gdouble) GST_SECOND;
  if (pts + EPS >= st->next) {
    while (st->next <= pts + EPS)
      st->next += st->step;
    st->kept++;
    if (st->verbose)
      g_print ("[keep] %.4fs\n", pts);
    return GST_PAD_PROBE_OK;
  }
  return GST_PAD_PROBE_DROP;
}

static gboolean
bus_cb (GstBus *bus, GstMessage *msg, gpointer data)
{
  GMainLoop *loop = (GMainLoop *) data;
  switch (GST_MESSAGE_TYPE (msg)) {
    case GST_MESSAGE_EOS:
      g_print ("[feed_fselect] EOS\n");
      g_main_loop_quit (loop);
      break;
    case GST_MESSAGE_ERROR: {
      GError *err = NULL; gchar *dbg = NULL;
      gst_message_parse_error (msg, &err, &dbg);
      g_printerr ("[feed_fselect] ERROR: %s | %s\n",
                  err ? err->message : "?", dbg ? dbg : "");
      if (err) g_error_free (err);
      g_free (dbg);
      g_main_loop_quit (loop);
      break;
    }
    default: break;
  }
  return TRUE;
}

int
main (int argc, char *argv[])
{
  GError *err = NULL;
  GOptionContext *ctx = g_option_context_new ("- feed nvdsframeselector RTVI-identical frames");
  g_option_context_add_main_entries (ctx, entries, NULL);
  g_option_context_add_group (ctx, gst_init_get_option_group ());
  if (!g_option_context_parse (ctx, &argc, &argv, &err)) {
    g_printerr ("option parsing failed: %s\n", err ? err->message : "?");
    return 1;
  }
  g_option_context_free (ctx);

  if (!opt_input) { g_printerr ("--input FILE is required\n"); return 1; }
  if (opt_fps <= 0.0) { g_printerr ("--fps must be > 0\n"); return 1; }

  /* build nvdsframeselector property string (only what the user set) */
  GString *props = g_string_new (NULL);
  g_string_append_printf (props,
      "cache-size=%d selection-count=%d optical-flow-interval=%d "
      "enable-motion-detection=%d source-fps=%f exclusion-range=%d",
      opt_cache, opt_sel, opt_ofi, opt_motion, opt_srcfps, opt_excl);
  if (opt_algo >= 0)
    g_string_append_printf (props, " frame-selection-algorithm=%d", opt_algo);
  if (opt_equidist >= 0)
    g_string_append_printf (props, " equidistant-output=%s", opt_equidist ? "true" : "false");
  if (opt_tighten >= 0.0)
    g_string_append_printf (props, " of-tighten=%f", opt_tighten);

  /* sink vs encoder tail */
  gchar *tail;
  if (opt_output && *opt_output)
    tail = g_strdup_printf (
        "nvvideoconvert ! nvv4l2h264enc ! h264parse ! qtmux ! filesink location=%s",
        opt_output);
  else
    tail = g_strdup ("fakesink sync=false");

  gchar *desc = g_strdup_printf (
      "filesrc location=%s ! parsebin ! nvv4l2decoder num-extra-surfaces=5 ! queue ! "
      "nvvideoconvert ! video/x-raw(memory:NVMM),width=%d,height=%d ! "
      "queue name=gridq ! nvdsframeselector name=fsel %s ! %s",
      opt_input, opt_width, opt_height, props->str, tail);

  g_print ("[feed_fselect] pipeline:\n  %s\n\n", desc);

  GstElement *pipeline = gst_parse_launch (desc, &err);
  if (!pipeline) {
    g_printerr ("failed to build pipeline: %s\n", err ? err->message : "?");
    return 1;
  }

  /* attach the equidistant probe on the queue feeding the plugin */
  ProbeState st = { .next = opt_start, .step = 1.0 / opt_fps,
                    .kept = 0, .verbose = opt_verbose };
  GstElement *gridq = gst_bin_get_by_name (GST_BIN (pipeline), "gridq");
  GstPad     *srcpad = gst_element_get_static_pad (gridq, "src");
  gst_pad_add_probe (srcpad, GST_PAD_PROBE_TYPE_BUFFER, grid_probe, &st, NULL);
  gst_object_unref (srcpad);
  gst_object_unref (gridq);

  GMainLoop *loop = g_main_loop_new (NULL, FALSE);
  GstBus *bus = gst_element_get_bus (pipeline);
  gst_bus_add_watch (bus, bus_cb, loop);
  gst_object_unref (bus);

  gst_element_set_state (pipeline, GST_STATE_PLAYING);
  g_main_loop_run (loop);

  g_print ("[feed_fselect] frames fed to plugin: %u\n", st.kept);

  gst_element_set_state (pipeline, GST_STATE_NULL);
  gst_object_unref (pipeline);
  g_main_loop_unref (loop);
  g_string_free (props, TRUE);
  g_free (tail);
  g_free (desc);
  return 0;
}
