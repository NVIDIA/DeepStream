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
 * @file nvdspreprocess_conversion.h
 * <b>NVIDIA DeepStream Preprocess lib implementation </b>
 *
 * @b Description: This file contains cuda kernels used for custom
 * tensor preparation after normalization and mean subtraction for 2d conv
 * NCHW/NHWC models.
 */

/** @defgroup   gstreamer_nvdspreprocess_api NvDsPreProcess Plugin
 * Defines an API for the GStreamer NvDsPreProcess custom lib implementation.
 * @ingroup custom_gstreamer
 * @{
 */

#ifndef __NVDSPREPROCESS_CONVERSION_H__
#define __NVDSPREPROCESS_CONVERSION_H__

#include <cuda_fp16.h>

/**
 * Converts an input packed 3 channel buffer of width x height resolution into an
 * planar 3-channel float buffer of width x height resolution. The input buffer can
 * have a pitch > (width * 3). The cuda kernel supports normalization and mean
 * image subtraction.
 *
 * This kernel can be used for RGB -> RGB and BGR -> BGR conversions.
 *
 * @param outBuffer      Cuda device buffer for planar float output. Should
 *                       be at least (width * height * 3 * sizeof(float)) bytes.
 * @param inBuffer       Cuda device buffer for packed input. Should be
 *                       at least (pitch * height) bytes.
 * @param width          Width of the buffers in pixels.
 * @param height         Height of the buffers in pixels.
 * @param pitch          Pitch of the input buffer in bytes.
 * @param scaleFactor    Normalization factor.
 * @param meanDataBuffer Mean Image Data buffer. Should be at least
 *                       (width * height * 3 * sizeof(float)) bytes.
 * @param stream         Cuda stream identifier.
 */
void
NvDsPreProcessConvert_C3ToP3Float(
    float *outBuffer,
    unsigned char *inBuffer,
    unsigned int width,
    unsigned int height,
    unsigned int pitch,
    float scaleFactor,
    float *meanDataBuffer,
    cudaStream_t stream);

/**
 * Converts an input packed 3 channel buffer of width x height resolution into an
 * linear 3-channel float buffer of width x height resolution. The input buffer can
 * have a pitch > (width * 3). The cuda kernel supports normalization and mean
 * image subtraction.
 *
 * This kernel can be used for RGB -> RGB and BGR -> BGR conversions.
 *
 * @param outBuffer      Cuda device buffer for planar float output. Should
 *                       be at least (width * height * 3 * sizeof(float)) bytes.
 * @param inBuffer       Cuda device buffer for packed input. Should be
 *                       at least (pitch * height) bytes.
 * @param width          Width of the buffers in pixels.
 * @param height         Height of the buffers in pixels.
 * @param pitch          Pitch of the input buffer in bytes.
 * @param scaleFactor    Normalization factor.
 * @param meanDataBuffer Mean Image Data buffer. Should be at least
 *                       (width * height * 3 * sizeof(float)) bytes.
 * @param stream         Cuda stream identifier.
 */
void
NvDsPreProcessConvert_C3ToL3Float(
    float *outBuffer,
    unsigned char *inBuffer,
    unsigned int width,
    unsigned int height,
    unsigned int pitch,
    float scaleFactor,
    float *meanDataBuffer,
    cudaStream_t stream);

/**
 * Converts an input packed 3 channel buffer of width x height resolution into an
 * planar 3-channel float buffer of width x height resolution. The input buffer can
 * have a pitch > (width * 3). The cuda kernel supports normalization and mean
 * image subtraction.
 *
 * This kernel can be used for RGBA -> RGB and BGRx -> BGR conversions.
 *
 * @param outBuffer      Cuda device buffer for planar float output. Should
 *                       be at least (width * height * 3 * sizeof(float)) bytes.
 * @param inBuffer       Cuda device buffer for packed input. Should be
 *                       at least (pitch * height) bytes.
 * @param width          Width of the buffers in pixels.
 * @param height         Height of the buffers in pixels.
 * @param pitch          Pitch of the input buffer in bytes.
 * @param scaleFactor    Normalization factor.
 * @param meanDataBuffer Mean Image Data buffer. Should be at least
 *                       (width * height * 3 * sizeof(float)) bytes.
 * @param stream         Cuda stream identifier.
 */
void
NvDsPreProcessConvert_C4ToP3Float(
    float *outBuffer,
    unsigned char *inBuffer,
    unsigned int width,
    unsigned int height,
    unsigned int pitch,
    float scaleFactor,
    float *meanDataBuffer,
    cudaStream_t stream);

/**
 * Converts an input packed 3 channel buffer of width x height resolution into an
 * planar 3-channel half buffer of width x height resolution. The input buffer can
 * have a pitch > (width * 3). The cuda kernel supports normalization and mean
 * image subtraction.
 *
 * This kernel can be used for RGBA -> RGB and BGRx -> BGR conversions.
 *
 * @param outBuffer      Cuda device buffer for planar half output. Should
 *                       be at least (width * height * 3 * sizeof(half)) bytes.
 * @param inBuffer       Cuda device buffer for packed input. Should be
 *                       at least (pitch * height) bytes.
 * @param width          Width of the buffers in pixels.
 * @param height         Height of the buffers in pixels.
 * @param pitch          Pitch of the input buffer in bytes.
 * @param scaleFactor    Normalization factor.
 * @param meanDataBuffer Mean Image Data buffer. Should be at least
 *                       (width * height * 3 * sizeof(float)) bytes.
 * @param stream         Cuda stream identifier.
 */
void
NvDsPreProcessConvert_C4ToP3Half(
    half *outBuffer,
    unsigned char *inBuffer,
    unsigned int width,
    unsigned int height,
    unsigned int pitch,
    float scaleFactor,
    float *meanDataBuffer,
    cudaStream_t stream);

/**
 * Converts an input packed 3 channel buffer of width x height resolution into an
 * linear 3-channel float buffer of width x height resolution. The input buffer can
 * have a pitch > (width * 3). The cuda kernel supports normalization and mean
 * image subtraction.
 *
 * This kernel can be used for RGBA -> RGB and BGRx -> BGR conversions.
 *
 * @param outBuffer      Cuda device buffer for linear float output. Should
 *                       be at least (width * height * 3 * sizeof(float)) bytes.
 * @param inBuffer       Cuda device buffer for packed input. Should be
 *                       at least (pitch * height) bytes.
 * @param width          Width of the buffers in pixels.
 * @param height         Height of the buffers in pixels.
 * @param pitch          Pitch of the input buffer in bytes.
 * @param scaleFactor    Normalization factor.
 * @param meanDataBuffer Mean Image Data buffer. Should be at least
 *                       (width * height * 3 * sizeof(float)) bytes.
 * @param stream         Cuda stream identifier.
 */
void
NvDsPreProcessConvert_C4ToL3Float(
    float *outBuffer,
    unsigned char *inBuffer,
    unsigned int width,
    unsigned int height,
    unsigned int pitch,
    float scaleFactor,
    float *meanDataBuffer,
    cudaStream_t stream);

/**
 * Converts an input packed 3 channel buffer of width x height resolution into an
 * planar 3-channel float buffer of width x height resolution with plane order
 * reversed. The input buffer can have a pitch > (width * 3). The cuda kernel
 * supports normalization and mean image subtraction.
 *
 * This kernel can be used for BGR -> RGB and RGB -> BGR conversions.
 *
 * @param outBuffer      Cuda device buffer for planar float output. Should
 *                       be at least (width * height * 3 * sizeof(float)) bytes.
 * @param inBuffer       Cuda device buffer for packed input. Should be
 *                       at least (pitch * height) bytes.
 * @param width          Width of the buffers in pixels.
 * @param height         Height of the buffers in pixels.
 * @param pitch          Pitch of the input buffer in bytes.
 * @param scaleFactor    Normalization factor.
 * @param meanDataBuffer Mean Image Data buffer. Should be at least
 *                       (width * height * 3 * sizeof(float)) bytes.
 * @param stream         Cuda stream identifier.
 */
void
NvDsPreProcessConvert_C3ToP3RFloat(
    float *outBuffer,
    unsigned char *inBuffer,
    unsigned int width,
    unsigned int height,
    unsigned int pitch,
    float scaleFactor,
    float *meanDataBuffer,
    cudaStream_t stream);

/**
 * Converts an input packed 3 channel buffer of width x height resolution into an
 * linear 3-channel float buffer of width x height resolution with plane order
 * reversed. The input buffer can have a pitch > (width * 3). The cuda kernel
 * supports normalization and mean image subtraction.
 *
 * This kernel can be used for BGR -> RGB and RGB -> BGR conversions.
 *
 * @param outBuffer      Cuda device buffer for planar float output. Should
 *                       be at least (width * height * 3 * sizeof(float)) bytes.
 * @param inBuffer       Cuda device buffer for packed input. Should be
 *                       at least (pitch * height) bytes.
 * @param width          Width of the buffers in pixels.
 * @param height         Height of the buffers in pixels.
 * @param pitch          Pitch of the input buffer in bytes.
 * @param scaleFactor    Normalization factor.
 * @param meanDataBuffer Mean Image Data buffer. Should be at least
 *                       (width * height * 3 * sizeof(float)) bytes.
 * @param stream         Cuda stream identifier.
 */
void
NvDsPreProcessConvert_C3ToL3RFloat(
    float *outBuffer,
    unsigned char *inBuffer,
    unsigned int width,
    unsigned int height,
    unsigned int pitch,
    float scaleFactor,
    float *meanDataBuffer,
    cudaStream_t stream);

/**
 * Converts an input packed 4 channel buffer of width x height resolution into an
 * planar 3-channel float buffer of width x height resolution with plane order
 * reversed. The input buffer can have a pitch > (width * 3). The cuda kernel
 * supports normalization and mean image subtraction.
 *
 * This kernel can be used for BGRx -> RGB and RGBA -> BGR conversions.
 *
 * @param outBuffer      Cuda device buffer for planar float output. Should
 *                       be at least (width * height * 3 * sizeof(half)) bytes.
 * @param inBuffer       Cuda device buffer for packed input. Should be
 *                       at least (pitch * height) bytes.
 * @param width          Width of the buffers in pixels.
 * @param height         Height of the buffers in pixels.
 * @param pitch          Pitch of the input buffer in bytes.
 * @param scaleFactor    Normalization factor.
 * @param meanDataBuffer Mean Image Data buffer. Should be at least
 *                       (width * height * 3 * sizeof(float)) bytes.
 * @param stream         Cuda stream identifier.
 */
void
NvDsPreProcessConvert_C4ToP3RHalf(
    half *outBuffer,
    unsigned char *inBuffer,
    unsigned int width,
    unsigned int height,
    unsigned int pitch,
    float scaleFactor,
    float *meanDataBuffer,
    cudaStream_t stream);

/**
 * Converts an input packed 4 channel buffer of width x height resolution into an
 * planar 3-channel float buffer of width x height resolution with plane order
 * reversed. The input buffer can have a pitch > (width * 3). The cuda kernel
 * supports normalization and mean image subtraction.
 *
 * This kernel can be used for BGRx -> RGB and RGBA -> BGR conversions.
 *
 * @param outBuffer      Cuda device buffer for planar float output. Should
 *                       be at least (width * height * 3 * sizeof(float)) bytes.
 * @param inBuffer       Cuda device buffer for packed input. Should be
 *                       at least (pitch * height) bytes.
 * @param width          Width of the buffers in pixels.
 * @param height         Height of the buffers in pixels.
 * @param pitch          Pitch of the input buffer in bytes.
 * @param scaleFactor    Normalization factor.
 * @param meanDataBuffer Mean Image Data buffer. Should be at least
 *                       (width * height * 3 * sizeof(float)) bytes.
 * @param stream         Cuda stream identifier.
 */
void
NvDsPreProcessConvert_C4ToP3RFloat(
    float *outBuffer,
    unsigned char *inBuffer,
    unsigned int width,
    unsigned int height,
    unsigned int pitch,
    float scaleFactor,
    float *meanDataBuffer,
    cudaStream_t stream);

/**
 * Converts an input packed 4 channel buffer of width x height resolution into an
 * linear 3-channel float buffer of width x height resolution with plane order
 * reversed. The input buffer can have a pitch > (width * 3). The cuda kernel
 * supports normalization and mean image subtraction.
 *
 * This kernel can be used for BGRx -> RGB and RGBA -> BGR conversions.
 *
 * @param outBuffer      Cuda device buffer for planar float output. Should
 *                       be at least (width * height * 3 * sizeof(float)) bytes.
 * @param inBuffer       Cuda device buffer for packed input. Should be
 *                       at least (pitch * height) bytes.
 * @param width          Width of the buffers in pixels.
 * @param height         Height of the buffers in pixels.
 * @param pitch          Pitch of the input buffer in bytes.
 * @param scaleFactor    Normalization factor.
 * @param meanDataBuffer Mean Image Data buffer. Should be at least
 *                       (width * height * 3 * sizeof(float)) bytes.
 * @param stream         Cuda stream identifier.
 */
void
NvDsPreProcessConvert_C4ToL3RFloat(
    float *outBuffer,
    unsigned char *inBuffer,
    unsigned int width,
    unsigned int height,
    unsigned int pitch,
    float scaleFactor,
    float *meanDataBuffer,
    cudaStream_t stream);

/**
 * Converts an 1 channel UINT8 input of width x height resolution into an
 * 1 channel float buffer of width x height resolution. The input buffer can
 * have a pitch > width . The cuda kernel supports normalization and mean
 * image subtraction.
 *
 * @param outBuffer  Cuda device buffer for float output. Should
 *                       be at least (width * height * sizeof(float)) bytes.
 * @param inBuffer   Cuda device buffer for UINT8 input. Should be
 *                       at least (pitch * height) bytes.
 * @param width          Width of the buffers in pixels.
 * @param height         Height of the buffers in pixels.
 * @param pitch          Pitch of the input  buffer in bytes.
 * @param scaleFactor    Normalization factor.
 * @param meanDataBuffer Mean Image Data buffer. Should be at least
 *                       (width * height * sizeof(float)) bytes.
 * @param stream         Cuda stream identifier.
 */
void
NvDsPreProcessConvert_C1ToP1Float(
        float *outBuffer,
        unsigned char *inBuffer,
        unsigned int width,
        unsigned int height,
        unsigned int pitch,
        float scaleFactor,
        float *meanDataBuffer,
        cudaStream_t stream);

void
NvDsPreProcessConvert_FtFTensor(
        float *outBuffer,
        float *inBuffer,
        unsigned int width,
        unsigned int height,
        unsigned int pitch,
        float scaleFactor,
        float *meanDataBuffer,
        cudaStream_t stream);

/**
 * Function pointer type to which any of the NvDsPreProcessConvert functions can be
 * assigned.
 */
typedef void (* NvDsPreProcessConvertFcnHalf)(
        half *outBuffer,
        unsigned char *inBuffer,
        unsigned int width,
        unsigned int height,
        unsigned int pitch,
        float scaleFactor,
        float *meanDataBuffer,
        cudaStream_t stream);

/**
 * Function pointer type to which any of the NvDsPreProcessConvert functions can be
 * assigned.
 */
typedef void (* NvDsPreProcessConvertFcn)(
        float *outBuffer,
        unsigned char *inBuffer,
        unsigned int width,
        unsigned int height,
        unsigned int pitch,
        float scaleFactor,
        float *meanDataBuffer,
        cudaStream_t stream);

#endif /* __NVDSPREPROCESS_CONVERSION_H__ */
