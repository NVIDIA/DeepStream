/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * @file
 * <b>NVIDIA DeepStream: Object Encoder</b>
 */
/**
 * @defgroup  ee_object_encoder Object Encoder API
 *
 * Defines APIs for Object Encoder.
 *
 * @ingroup NvllBatchJpegEncodeinterfaceApi
 * @{
 */

#ifndef __NVDS_ENCODE_OBJECT_H__
#define __NVDS_ENCODE_OBJECT_H__

#include <sys/time.h>

#ifdef __cplusplus
extern "C"
{
#endif

#define FILE_NAME_SIZE (1024)
struct _NvDsObjEncCtx;

/** Opaque Handle to the Object Encoder Context */
typedef struct _NvDsObjEncCtx * NvDsObjEncCtxHandle;

/**
 * Holds output parameters. This structure is available to the user.
 */
typedef struct _NvDsObjEncOutParams
{
  /** Pointer to the JPEG Encoded Object */
  uint8_t *outBuffer;
  /** Length of the JPEG Encoded Object */
  uint64_t outLen;
} NvDsObjEncOutParams;

/**
 * Holds user parameters for a nvds_obj_enc_process call.
 */
typedef struct _NvDsObjEncUsrArgs
{
  /** Boolean variable to save image */
  bool saveImg;
  /** Variable to attach user metadata.
   * Metadata Type is "NVDS_CROP_IMAGE_META".
   */
  bool attachUsrMeta;
  /** Boolean variable to scale image. */
  bool scaleImg;
  /** Image dimensions shouldn't exceed frame dimensions.
   * Width Parameter for scaling.
   */
  int scaledWidth;
  /** Height Paramater for scaling. */
  int scaledHeight;
  /** If user specifies the filename then it'll be used otherwise the
   * following naming convention is used to create filename of the encoded
   * objects -
   * "frame-number_stream-number_object-number_object-type_widthxheight.jpg".
   * For example - 0_2_3_PERSON_126x148.jpg
   */
  char fileNameImg[FILE_NAME_SIZE];
  /** Object number in the frame */
  int objNum;
  /** Image quality value */
  int quality;
  /** Variable for frame/object encoding.
   * 1: Encodes the entire frame.
   * 0: Encodes object of specified resolution. */
  bool isFrame;
  /** Boolean variable to calculate time of encode */
  bool calcEncodeTime;
} NvDsObjEncUsrArgs;

/** Create context and return a handle to NvObjEncCtx */
NvDsObjEncCtxHandle nvds_obj_enc_create_context (int);

/** Enqueue an object crop for JPEG encode.
 * This is a non-blocking call and user should call nvds_obj_enc_finish()
 * to make sure all enqueued object crops have been processed.
 */
bool nvds_obj_enc_process (NvDsObjEncCtxHandle, NvDsObjEncUsrArgs *,
    NvBufSurface *, NvDsObjectMeta *, NvDsFrameMeta *);

/** Wait for all enqueued crops to be encoded */
void nvds_obj_enc_finish (NvDsObjEncCtxHandle);

/** Destroy context */
void nvds_obj_enc_destroy_context (NvDsObjEncCtxHandle);

#ifdef __cplusplus
}
#endif

#endif

/** @} */
