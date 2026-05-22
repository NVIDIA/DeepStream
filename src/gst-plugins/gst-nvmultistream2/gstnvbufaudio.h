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

/**
 * @file gstnvbufaudio.h
 * <b>Gst Helper APIs for NvBufAudio Interface </b>
 *
 * This file specifies the API to fetch audio specific
 * information from GstCaps.
 *
 */

#ifndef _GST_NVSTREAMMUX_AUDIO_H_
#define _GST_NVSTREAMMUX_AUDIO_H_


#include <gst/gst.h>
#include <gst/audio/audio.h>
#include "gstnvstreammuxdebug.h"
#include "nvbufaudio.h"

class GstNvBufAudioCaps
{
    public:
    GstNvBufAudioCaps(GstCaps* aCaps);
    NvBufAudioFormat GetAudioFormat();
    NvBufAudioLayout GetAudioLayout();
    uint32_t         GetAudioRate();
    uint32_t         GetAudioChannels();
    bool             GetAudioParams(NvBufAudioParams& aAudioParams);

    private:
    gchar const *    GetFieldStringValue(gchar const * fieldName);
    uint32_t const   GetFieldIntValue(gchar const * fieldName);

    GstCaps*         caps;
    GstStructure*    capsStruct;
    GstAudioInfo     audioInfo;
    NvBufAudioParams audioParams;

    /** audio/x-raw
     *   format: { S8, U8, S16LE, S16BE, U16LE, U16BE, S24_32LE, S24_32BE,
     *             U24_32LE, U24_32BE, S32LE, S32BE, U32LE, U32BE, S24LE,
     *             S24BE, U24LE, U24BE, S20LE, S20BE, U20LE, U20BE, S18LE,
     *             S18BE, U18LE, U18BE, F32LE, F32BE, F64LE, F64BE }
     *   rate: [ 1, 2147483647 ]
     *   channels: [ 1, 2147483647 ]
     *   layout: interleaved
     */
    gchar const * format;
    gchar const * rate;
    gchar const * channels;
    gchar const * layout;
};

#endif /**< _GST_NVSTREAMMUX_AUDIO_H_ */
