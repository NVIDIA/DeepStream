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
 * @file gstnvbufaudio.cpp
 * <b>Gst Helper APIs for NvBufAudio Interface </b>
 *
 * This file specifies the API to fetch audio specific
 * information from GstCaps and populate NvBufAudioParams.
 *
 */

#include "gstnvbufaudio.h"
#include <string.h>
#include <stdlib.h>

GstNvBufAudioCaps::GstNvBufAudioCaps(GstCaps* aCaps) : caps(aCaps), capsStruct(nullptr), audioInfo(), audioParams()
{
}

bool GstNvBufAudioCaps::GetAudioParams(NvBufAudioParams& aAudioParams)
{

    if (!gst_audio_info_from_caps (&audioInfo, caps))
      return false;
    capsStruct = gst_caps_get_structure (caps, 0);
    format     = GetFieldStringValue("format");
    layout     = GetFieldStringValue("layout");
    if(!format || !layout)
    {
        return false;
    }
    audioParams.format = GetAudioFormat();
    audioParams.bpf    = audioInfo.bpf;
    audioParams.channels = GetFieldIntValue("channels");
    audioParams.rate = GetFieldIntValue("rate");
    audioParams.layout = GetAudioLayout();
    if((audioParams.format == NVBUF_AUDIO_INVALID_FORMAT)
       || (audioParams.layout == NVBUF_AUDIO_INVALID_LAYOUT)
       || !audioParams.rate
       || !audioParams.channels
      )
    {
         return false;
    }
    aAudioParams = audioParams;
    return true;
}

gchar const * GstNvBufAudioCaps::GetFieldStringValue(gchar const * fieldName)
{
    if (gst_structure_has_field (capsStruct, fieldName))
    {
        GValue const * val =  gst_structure_get_value (capsStruct, fieldName);
        if(val)
        {
            gchar const * stringVal = g_value_get_string(val);
            LOGD("%s=%s\n", fieldName, stringVal);
            return stringVal;
        }
    }
    return nullptr;
}

uint32_t const GstNvBufAudioCaps::GetFieldIntValue(gchar const * fieldName)
{
    if (gst_structure_has_field (capsStruct, fieldName))
    {
        GValue const * val =  gst_structure_get_value (capsStruct, fieldName);
        if(val)
        {
            uint32_t intVal = g_value_get_int(val);
            LOGD("%s=%u\n", fieldName, intVal);
            return intVal;
        }
    }
    return 0;
}

NvBufAudioFormat GstNvBufAudioCaps::GetAudioFormat()
{
    /** audio/x-raw
     *   format: { S8, U8, S16LE, S16BE, U16LE, U16BE, S24_32LE, S24_32BE,
     *             U24_32LE, U24_32BE, S32LE, S32BE, U32LE, U32BE, S24LE,
     *             S24BE, U24LE, U24BE, S20LE, S20BE, U20LE, U20BE, S18LE,
     *             S18BE, U18LE, U18BE, F32LE, F32BE, F64LE, F64BE }
     *   rate: [ 1, 2147483647 ]
     *   channels: [ 1, 2147483647 ]
     *   layout: interleaved
     */
    if(strcmp(format, "S8") == 0)
    {
        return NVBUF_AUDIO_S8;
    }
    else if(strcmp(format, "U8") == 0)
    {
        return NVBUF_AUDIO_U8;
    }
    else if(strcmp(format, "S16LE") == 0)
    {
        return NVBUF_AUDIO_S16LE;
    }
    else if(strcmp(format, "S16BE") == 0)
    {
        return NVBUF_AUDIO_S16BE;
    }
    else if(strcmp(format, "U16LE") == 0)
    {
        return NVBUF_AUDIO_U16LE;
    }
    else if(strcmp(format, "U16BE") == 0)
    {
        return NVBUF_AUDIO_U16BE;
    }
    else if(strcmp(format, "S24_32LE") == 0)
    {
        return NVBUF_AUDIO_S24_32LE;
    }
    else if(strcmp(format, "S24_32BE") == 0)
    {
        return NVBUF_AUDIO_S24_32BE;
    }
    else if(strcmp(format, "U24_32LE") == 0)
    {
        return NVBUF_AUDIO_U24_32LE;
    }
    else if(strcmp(format, "U24_32BE") == 0)
    {
        return NVBUF_AUDIO_U24_32BE;
    }
    else if(strcmp(format, "S32LE") == 0)
    {
        return NVBUF_AUDIO_S32LE;
    }
    else if(strcmp(format, "S32BE") == 0)
    {
        return NVBUF_AUDIO_S32BE;
    }
    else if(strcmp(format, "U32LE") == 0)
    {
        return NVBUF_AUDIO_U32LE;
    }
    else if(strcmp(format, "U32BE") == 0)
    {
        return NVBUF_AUDIO_U32BE;
    }
    else if(strcmp(format, "S24LE") == 0)
    {
        return NVBUF_AUDIO_S24LE;
    }
    else if(strcmp(format, "S24BE") == 0)
    {
        return NVBUF_AUDIO_S24BE;
    }
    else if(strcmp(format, "U24LE") == 0)
    {
        return NVBUF_AUDIO_U24LE;
    }
    else if(strcmp(format, "U24BE") == 0)
    {
        return NVBUF_AUDIO_U24BE;
    }
    else if(strcmp(format, "S20LE") == 0)
    {
        return NVBUF_AUDIO_S20LE;
    }
    else if(strcmp(format, "S20BE") == 0)
    {
        return NVBUF_AUDIO_S20BE;
    }
    else if(strcmp(format, "U20LE") == 0)
    {
        return NVBUF_AUDIO_U20LE;
    }
    else if(strcmp(format, "U20BE") == 0)
    {
        return NVBUF_AUDIO_U20BE;
    }
    else if(strcmp(format, "S18LE") == 0)
    {
        return NVBUF_AUDIO_S18LE;
    }
    else if(strcmp(format, "S18BE") == 0)
    {
        return NVBUF_AUDIO_S18BE;
    }
    else if(strcmp(format, "U18LE") == 0)
    {
        return NVBUF_AUDIO_U18LE;
    }
    else if(strcmp(format, "U18BE") == 0)
    {
        return NVBUF_AUDIO_U18BE;
    }
    else if(strcmp(format, "F32LE") == 0)
    {
        return NVBUF_AUDIO_F32LE;
    }
    else if(strcmp(format, "F32BE") == 0)
    {
        return NVBUF_AUDIO_F32BE;
    }
    else if(strcmp(format, "F64LE") == 0)
    {
        return NVBUF_AUDIO_F64LE;
    }
    else if(strcmp(format, "F64BE") == 0)
    {
        return NVBUF_AUDIO_F64BE;
    }
    return NVBUF_AUDIO_INVALID_FORMAT;
}

NvBufAudioLayout GstNvBufAudioCaps::GetAudioLayout()
{
    if(strcmp(layout, "interleaved") == 0)
    {
        return NVBUF_AUDIO_INTERLEAVED;
    }
    else if(strcmp(layout, "non-interleaved") == 0)
    {
        return NVBUF_AUDIO_NON_INTERLEAVED;
    }
    return NVBUF_AUDIO_INVALID_LAYOUT;
}

uint32_t GstNvBufAudioCaps::GetAudioRate()
{
    return static_cast<uint32_t>(atoi(rate));
}

uint32_t GstNvBufAudioCaps::GetAudioChannels()
{
    return static_cast<uint32_t>(atoi(channels));
}
