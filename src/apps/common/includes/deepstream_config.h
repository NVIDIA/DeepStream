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

#ifndef __NVGSTDS_CONFIG_H__
#define __NVGSTDS_CONFIG_H__

#ifdef __aarch64__
#define IS_TEGRA
#endif

#define MEMORY_FEATURES "memory:NVMM"

#ifdef IS_TEGRA
#define NVDS_ELEM_SRC_CAMERA_CSI "nvarguscamerasrc"
#else
#define NVDS_ELEM_SRC_CAMERA_CSI "videotestsrc"
#endif
#define NVDS_ELEM_SRC_CAMERA_V4L2 "v4l2src"
#define NVDS_ELEM_SRC_URI "uridecodebin"
#define NVDS_ELEM_SRC_MULTIFILE "multifilesrc"
#define NVDS_ELEM_SRC_ALSA "alsasrc"

#define NVDS_ELEM_DECODEBIN "decodebin"
#define NVDS_ELEM_WAVPARSE "wavparse"

#define NVDS_ELEM_QUEUE "queue"
#define NVDS_ELEM_CAPS_FILTER "capsfilter"
#define NVDS_ELEM_TEE "tee"
#define NVDS_ELEM_IDENTITY "identity"

#define NVDS_ELEM_PREPROCESS "nvdspreprocess"
#define NVDS_ELEM_SECONDARY_PREPROCESS "nvdspreprocess"
#define NVDS_ELEM_PGIE "nvinfer"
#define NVDS_ELEM_SGIE "nvinfer"
#define NVDS_ELEM_NVINFER "nvinfer"
#define NVDS_ELEM_INFER_SERVER "nvinferserver"
#define NVDS_ELEM_INFER_AUDIO "nvinferaudio"
#define NVDS_ELEM_TRACKER "nvtracker"
#define NVDS_ELEM_TEXT_EMBEDDER "nvdstextembedder"
#define NVDS_ELEM_REPLAY "nvreplay"

#define NVDS_ELEM_VIDEO_CONV "nvvideoconvert"
#define NVDS_ELEM_AUDIO_CONV "audioconvert"
#define NVDS_ELEM_AUDIO_RESAMPLER "audioresample"
#define NVDS_ELEM_STREAM_MUX "nvstreammux"
#define NVDS_ELEM_STREAM_DEMUX "nvstreamdemux"
#define NVDS_ELEM_TILER "nvmultistreamtiler"
#define NVDS_ELEM_OSD "nvdsosd"
#define NVDS_ELEM_SEGVISUAL "nvsegvisual"
#define NVDS_ELEM_DSANALYTICS_ELEMENT "nvdsanalytics"
#define NVDS_ELEM_DSEXAMPLE_ELEMENT "dsexample"

#define NVDS_ELEM_DEWARPER "nvdewarper"
#define NVDS_ELEM_SPOTANALYSIS "nvspot"
#define NVDS_ELEM_NVAISLE "nvaisle"
#define NVDS_ELEM_BBOXFILTER "nvbboxfilter"
#define NVDS_ELEM_MSG_CONV "nvmsgconv"
#define NVDS_ELEM_MSG_BROKER "nvmsgbroker"

#define NVDS_ELEM_SINK_FAKESINK "fakesink"
#define NVDS_ELEM_SINK_FILE "filesink"
#define NVDS_ELEM_SINK_EGL "nveglglessink"
#define NVDS_ELEM_SINK_3D "nv3dsink"
#define NVDS_ELEM_SINK_DRM "nvdrmvideosink"
#define NVDS_ELEM_EGLTRANSFORM "nvegltransform"

#define NVDS_ELEM_MUX_MP4 "qtmux"
#define NVDS_ELEM_MKV "matroskamux"

#define NVDS_ELEM_ENC_H264_HW "nvv4l2h264enc"
#define NVDS_ELEM_ENC_H265_HW "nvv4l2h265enc"
#define NVDS_ELEM_ENC_MPEG4 "avenc_mpeg4"

#define NVDS_ELEM_ENC_H264_SW "x264enc"
#define NVDS_ELEM_ENC_H265_SW "x265enc"

#define MAX_SOURCE_BINS 1024
#define MAX_SINK_BINS (1024)
#define MAX_SECONDARY_GIE_BINS (16)
#define MAX_SECONDARY_PREPROCESS_BINS (16)
#define MAX_MESSAGE_CONSUMERS (16)

#define NVDS_ELEM_NVMULTIURISRCBIN "nvmultiurisrcbin"

#endif
