/*
 * SPDX-FileCopyrightText: Copyright (c) 2020 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * @file nvbufaudio.h
 * <b>NvBufAudio Interface </b>
 *
 * This file specifies the NvBufAudio management API.
 *
 * The NvBufAudio API provides data structure definition
 * for batched audio buffers.
 * NOTE: Currently the audio data buffers are raw (on system memory).
 * GPU memory support is unavailable.
 */

#ifndef _NVBUFAUDIO_H_
#define _NVBUFAUDIO_H_

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C"
{
#endif

#define RESERVED_BYTES 16

/** Specifies audio formats */
typedef enum
{
    NVBUF_AUDIO_INVALID_FORMAT,
    NVBUF_AUDIO_S8,
    NVBUF_AUDIO_U8,
    NVBUF_AUDIO_S16LE,
    NVBUF_AUDIO_S16BE,
    NVBUF_AUDIO_U16LE,
    NVBUF_AUDIO_U16BE,
    NVBUF_AUDIO_S24_32LE,
    NVBUF_AUDIO_S24_32BE,
    NVBUF_AUDIO_U24_32LE,
    NVBUF_AUDIO_U24_32BE,
    NVBUF_AUDIO_S32LE,
    NVBUF_AUDIO_S32BE,
    NVBUF_AUDIO_U32LE,
    NVBUF_AUDIO_U32BE,
    NVBUF_AUDIO_S24LE,
    NVBUF_AUDIO_S24BE,
    NVBUF_AUDIO_U24LE,
    NVBUF_AUDIO_U24BE,
    NVBUF_AUDIO_S20LE,
    NVBUF_AUDIO_S20BE,
    NVBUF_AUDIO_U20LE,
    NVBUF_AUDIO_U20BE,
    NVBUF_AUDIO_S18LE,
    NVBUF_AUDIO_S18BE,
    NVBUF_AUDIO_U18LE,
    NVBUF_AUDIO_U18BE,
    NVBUF_AUDIO_F32LE,
    NVBUF_AUDIO_F32BE,
    NVBUF_AUDIO_F64LE,
    NVBUF_AUDIO_F64BE
} NvBufAudioFormat;

/** Specifies audio data layout in memory */
typedef enum
{
    NVBUF_AUDIO_INVALID_LAYOUT,
    NVBUF_AUDIO_INTERLEAVED, /**< audio sample from each channel shall be interleaved LRLRLRLR */
    NVBUF_AUDIO_NON_INTERLEAVED, /**< audio sample from each channel shall be interleaved ; LLLLLLLLRRRRRRRR */
} NvBufAudioLayout;

typedef struct
{
    NvBufAudioLayout layout;
    NvBufAudioFormat format;
    uint32_t         bpf;      /**< Bytes per frame; the size of a frame;
                                * size of one sample * @channels */
    uint32_t         channels; /**< Number of audio channels */
    uint32_t         rate;     /**< audio sample rate in samples per second */
    uint32_t         dataSize;
    void*            dataPtr;
    /** Holds the pad or port index of the Gst-streammux plugin for the frame
     * in the batch. */
    uint32_t         padId;
    /** source ID of this buffer;
     * This is w.r.t the multisrc DeepStream usecases
     */
    uint32_t         sourceId;
    /** NTP Timestamp of this audio buffer */
    uint64_t         ntpTimestamp;
    /** Buffer PTS of this audio buffer */
    uint64_t         bufPts;
    /** duration of this audio buffer */
    uint64_t         duration;

    uint8_t reserved[RESERVED_BYTES];
} NvBufAudioParams;

typedef struct
{
    /** The size of this NvBufAudio batch */
    uint32_t           numFilled;
    /** The size of this NvBufAudio batch */
    uint32_t           batchSize;
    /** isContiguous is true when
     * the dataPtr in audioBuffers[] array is
     * contiguous with the previous and following entry
     * in the array
     */
    bool               isContiguous;
    /** Array of #batchSize audio bufffers */
    NvBufAudioParams*  audioBuffers;
} NvBufAudio;
#ifdef __cplusplus
}
#endif

#endif /**< _NVBUFAUDIO_H_ */
