/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
