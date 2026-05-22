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
 * This is a header file for pre-processing cuda kernels with normalization and
 * mean subtraction required by nvdsinfer.
 */
#ifndef __NVDSINFER_CONVERSION_H__
#define __NVDSINFER_CONVERSION_H__

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
NvDsInferConvert_C3ToP3Float(
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
NvDsInferConvert_C3ToL3Float(
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
NvDsInferConvert_C4ToP3Float(
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
NvDsInferConvert_C4ToL3Float(
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
NvDsInferConvert_C3ToP3RFloat(
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
NvDsInferConvert_C3ToL3RFloat(
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
NvDsInferConvert_C4ToP3RFloat(
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
NvDsInferConvert_C4ToL3RFloat(
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
NvDsInferConvert_C1ToP1Float(
        float *outBuffer,
        unsigned char *inBuffer,
        unsigned int width,
        unsigned int height,
        unsigned int pitch,
        float scaleFactor,
        float *meanDataBuffer,
        cudaStream_t stream);

void
NvDsInferConvert_FtFTensor(
        float *outBuffer,
        float *inBuffer,
        unsigned int width,
        unsigned int height,
        unsigned int pitch,
        float scaleFactor,
        float *meanDataBuffer,
        cudaStream_t stream);


/**
 * Function pointer type to which any of the NvDsInferConvert functions can be
 * assigned.
 */
typedef void (* NvDsInferConvertFcn)(
        float *outBuffer,
        unsigned char *inBuffer,
        unsigned int width,
        unsigned int height,
        unsigned int pitch,
        float scaleFactor,
        float *meanDataBuffer,
        cudaStream_t stream);

typedef void (* NvDsInferConvertFcnFloat)(
        float *outBuffer,
        float *inBuffer,
        unsigned int width,
        unsigned int height,
        unsigned int pitch,
        float scaleFactor,
        float *meanDataBuffer,
        cudaStream_t stream);

#endif /* __NVDSINFER_CONVERSION_H__ */
