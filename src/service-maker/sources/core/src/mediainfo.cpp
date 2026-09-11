/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "mediainfo.hpp"

#include <gst/gst.h>
#include <gst/pbutils/pbutils.h>

namespace deepstream {

extern void init_gst();
extern bool _gst_initialized;

static std::string get_codec_info(GstDiscovererStreamInfo *sinfo);
std::string check_uri(std::string uri);


std::unique_ptr<struct MediaInfo> MediaInfo::discover(std::string uri) {
  GError *err = NULL;
  GstDiscoverer *discoverer = NULL;
  GstDiscovererInfo *discoverer_info = NULL;
  auto media_info = std::make_unique<struct MediaInfo>();

  uri = check_uri(uri);
  if (uri.empty()) {
    return media_info;
  }

  if (!_gst_initialized) {
    init_gst();
    _gst_initialized = true;
  }

  discoverer = gst_discoverer_new(5 * GST_SECOND, &err);
  if (!discoverer) {
    g_printerr("Error creating discoverer instance: %s\n", err->message);
    g_clear_error(&err);
    return media_info;
  }

  discoverer_info = gst_discoverer_discover_uri(discoverer, uri.c_str(), &err);
  if (!discoverer_info) {
    g_printerr("Failed to discover URI: %s with error %s\n", uri.c_str(), err->message);
    g_clear_error(&err);
    g_object_unref (discoverer);
    return media_info;
  }

  media_info->duration = gst_discoverer_info_get_duration(discoverer_info);
//  media_info->live = gst_discoverer_info_get_live(discoverer_info);

  GList * audio_l = gst_discoverer_info_get_audio_streams(discoverer_info);
  for (GList *l = audio_l; l; l = l->next) {
    GstDiscovererAudioInfo *audio = (GstDiscovererAudioInfo *) l->data;
    AudioStreamInfo* info = new AudioStreamInfo;
    info->codec = get_codec_info(GST_DISCOVERER_STREAM_INFO(audio));
    info->channels = gst_discoverer_audio_info_get_channels(audio);
    info->sampling_rate = gst_discoverer_audio_info_get_sample_rate(audio);
    media_info->streams.push_back(std::unique_ptr<StreamInfo>(info));
  }
  gst_discoverer_stream_info_list_free(audio_l);

  GList * video_l = gst_discoverer_info_get_video_streams(discoverer_info);
  for (GList *l = video_l; l; l = l->next) {
    GstDiscovererVideoInfo *video = (GstDiscovererVideoInfo *) l->data;
    VideoStreamInfo* info = new VideoStreamInfo;
    info->codec = get_codec_info(GST_DISCOVERER_STREAM_INFO(video));
    info->framerate.num = gst_discoverer_video_info_get_framerate_num(video);
    info->framerate.denom = gst_discoverer_video_info_get_framerate_denom(video);
    info->height = gst_discoverer_video_info_get_width(video);
    info->width = gst_discoverer_video_info_get_height(video);
    media_info->streams.push_back(std::unique_ptr<StreamInfo>(info));
  }
  gst_discoverer_stream_info_list_free(video_l);

  /* Free resources */
  if (discoverer_info) {
    g_object_unref(discoverer_info);
  }
  g_object_unref (discoverer);

  return media_info;
}

std::string get_codec_info(GstDiscovererStreamInfo *sinfo) {
  gchar *desc = NULL;
  GstCaps *caps;
  std::string codec_string;

  caps = gst_discoverer_stream_info_get_caps(sinfo);
  if (caps) {
    if (gst_caps_is_fixed (caps))
      desc = gst_pb_utils_get_codec_description(caps);
    else
      desc = gst_caps_to_string (caps);
    gst_caps_unref (caps);
  }

  if (desc) {
    codec_string = std::string(desc);
    g_free (desc);
    desc = NULL;
  }

  return codec_string;
}

std::string check_uri(std::string uri) {
  std::string r;
   GError *err = NULL;
  const gchar* filename = uri.c_str();

  if (!gst_uri_is_valid(filename)) {
    gchar *path;
    if (!g_path_is_absolute(filename)) {
      gchar *cur_dir;

      cur_dir = g_get_current_dir();
      path = g_build_filename(cur_dir, filename, NULL);
      g_free(cur_dir);
    } else {
      path = g_strdup(filename);
    }

    gchar* c_uri = g_filename_to_uri(path, NULL, &err);
    g_free(path);
    path = NULL;
    if (err) {
      g_warning("Couldn't convert filename to URI: %s\n", err->message);
      g_clear_error (&err);
    } else if (c_uri) {
      r = c_uri;
      g_free(c_uri);
    }
  } else {
    r = uri;
  }
  return r;
}
}
