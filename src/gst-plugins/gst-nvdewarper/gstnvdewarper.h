/**
 * SPDX-FileCopyrightText: Copyright (c) 2019-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

/**
 * @file gstnvdewarper.h
 * <b>NVIDIA DeepStream GStreamer nvdewarper API Specification </b>
 *
 * @b Description: This file specifies the data structures for
 * the DeepStream GStreamer nvdewarper Plugin.
 */

#ifndef __GST_NVDEWARPER_H__
#define __GST_NVDEWARPER_H__

#include <gst/gst.h>
#include <gst/video/gstvideometa.h>
#include <gst/video/video.h>
#include <gst/base/gstbasetransform.h>

#include <cuda.h>
#include <npp.h>

#include "nv_aisle_csvparser.hpp"
#include "nv_spot_csvparser.hpp"
#include "gstnvdsmeta.h"
#include "nvds_dewarper_meta.h"
#include "nvbufsurface.h"
#include "nvbufsurftransform.h"

using namespace nvaisle_csv;
using namespace nvspot_csv;

#define DISTORTION_SIZE 5 /**< Maximum number of distortion coefficients */
#define FOCAL_LENGTH_SIZE 2 /**< Focal length array size : two values for X & Y direction */
#define ROTATION_MATRIX_SIZE 9  /**< Standard rotation matrix size */

G_BEGIN_DECLS

/**
 * @addtogroup three Standard GStreamer boilerplate
 * @{
 */
#define GST_TYPE_NVDEWARPER \
  (gst_nvdewarper_get_type())
#define GST_NVDEWARPER(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_NVDEWARPER,Gstnvdewarper))
#define GST_NVDEWARPER_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_NVDEWARPER,GstnvdewarperClass))
#define GST_IS_NVDEWARPER(obj) \
  (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_NVDEWARPER))
#define GST_IS_NVDEWARPER_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_NVDEWARPER))

typedef struct _Gstnvdewarper      Gstnvdewarper;
typedef struct _GstnvdewarperClass GstnvdewarperClass;
/** @} */

/**
 * Holds all the configuration parameters required for dewarping a surface.
 * All these configurations can be set by the user under the "surface"
 * category in config file
 */
typedef struct _NvDewarperParams
{
  guint projection_type;                        /**< Projection type of type NvDsSurfaceType */

  gfloat top_angle;                             /**< The top view angle, in degrees */
  gfloat bottom_angle;                          /**< The bottom view angle, in degrees */
  gfloat pitch;                                 /**< The pitch angle, in degrees */
  gfloat roll;                                  /**< The roll angle, in degrees */
  gfloat yaw;                                   /**< The yaw angle, in degrees */
  gfloat dewarpFocalLength[FOCAL_LENGTH_SIZE];  /**< The X & Y focal length of the source, in pixels */
  char rot_axes[4];                             /**< A sequence of 3 rotation axes:  upper case 'X', 'Y', and 'Z'. */
                                                /**< 4th character is set to '\0'. */
                                                /**< X rotation rotates the view upward, Y rightward, and Z clockwise.  */
                                                /**< The default is "YXZ", a.k.a. yaw, pitch, roll */
  gfloat control;                               /**< Projection-specific controls for Panini, Stereographic and Pushbroom projections */

  guint dewarpWidth;                            /**< Dewarped Surface width */
  guint dewarpHeight;                           /**< Dewarped Surface height */
  guint dewarpPitch;                            /**< Dewarped Surface pitch */

  guint addressMode;                            /**< Cuda Texture Address Mode */
  guchar *surface;                              /**< Pointer to Cuda Surface used for Projection */
  guint isValid;                                /**< Boolean indicating if the surface parameters structure is valid */


  guint id;                                     /**< Surface id. This is to distinguish between views of same type */
  guint surface_index;                          /**< Surface index */

  gdouble distortion[DISTORTION_SIZE];          /**< Distortion polynomial coefficients */
  gfloat src_x0;                                /**< source principal point in X direction */
  gfloat src_y0;                                /**< source principal point in Y direction */
  gfloat srcFov;                                /**< Source field of view */
  gfloat rot_matrix[ROTATION_MATRIX_SIZE];      /**< Rotation matrix */
  guint rot_matrix_valid;                       /**< Boolean indicating if the values in "rot_matrix" are valid */

  gfloat dstFocalLength[FOCAL_LENGTH_SIZE];     /**< destination surface focal length */
  gfloat dstPrincipalPoint[2];                  /**< destination surface principal point */
} NvDewarperParams;

/** Data structure contaning dewarping parameters for all the output surfaces */
typedef struct _NvDewarperPriv
{
  std::vector<NvDewarperParams> vecDewarpSurface; /**< Array of surface parameters of type "NvDewarperParams". Maximum 4. */
} NvDewarperPriv;

/**
 * Gstnvdewarper element structure.
 */
struct _Gstnvdewarper
{
  GstBaseTransform element; /**< Should be the first member when extending from GstBaseTransform. */

  GstCaps *sinkcaps;        /**< Sink pad caps */
  GstCaps *srccaps;         /**< Source pad caps */

  guint input_width;        /**<Input frame width */
  guint input_height;       /**<Input frame height */
  guint output_width;       /**<Output frame width */
  guint output_height;      /**<Output frame height */

  guint num_batch_buffers;  /**< Number of batch buffers */
  guint gpu_id;             /**< ID of the GPU this element uses for dewarping/scaling. */

  gchar* config_file;             /**< String contaning path and name of configuration file */
  gchar* spot_calibration_file;   /**< String contaning path and name of spot calibration file */
  gchar* aisle_calibration_file;  /**< String contaning path and name of aisle calibration file */

  GstBufferPool *pool;            /**< Internal buffer pool for output buffers  */

  /** Input memory feature can take values MEM_FEATURE_NVMM/MEM_FEATURE_RAW
   * based on input  memory type caps*/
  gint input_feature;
  /** Output memory feature can take values MEM_FEATURE_NVMM/MEM_FEATURE_RAW
   * based on output  memory type caps*/
  gint output_feature;

  NvBufSurfaceMemType cuda_mem_type;              /**< Cuda surface memory type set by "nvbuf-memory-type" */
  NvBufSurfTransform_Inter interpolation_method;  /**< Interpolation method for scaling. Set by config param "interpolation-method" */
  GstVideoFormat input_fmt;                       /**< Input stream format derived from sink caps */
  GstVideoFormat output_fmt;                      /**< Output stream format derived from src caps */

  cudaStream_t stream;                            /**< Cuda Stream to launch operations on. */

  guint frame_num;                                /**< Number of the frame in the stream that was last processed. */

  guint dump_frames;  /**< Number of dewarped output frames to be dumped in a *.rgba file. Useful for debugging */
  void *aisle_output; /**< Placeholder for aisle output host memory pointer. Currently unused. */
  void *spot_output;  /**< Placeholder for spot output host memory pointer. Currently unused. */
  void *output;       /**< Host memory  pointer for output buffer. Used for frame dumps. */

  gboolean silent;                    /**< Boolean indicating swtiching on/off of verbose output */
  gboolean spot_calibrationfile_set;  /**< Boolean indicating whether the spot calibration file is specified */
  gboolean aisle_calibrationfile_set; /**< Boolean indicating whether the aisle calibration file is specified */
  AisleCSVParser *aisleCSVParser;     /**< CSV parsed structure for aisle calibration */
  SpotCSVParser *spotCSVParser;       /**< CSV parsed structure for spot calibration */

  guint source_id;                            /**< Source ID of the input source */
  guint num_output_buffers;                   /**< Number of Output Buffers to be allocated by buffer pool */
  guint aisleCSVInit;                         /**< Boolean indicating whether the aisle surface is initialized from CSV data */
  guint spotCSVInit;                          /**< Boolean indicating whether the spot surface is initialized from CSV data */
  guint num_spot_views;                       /**< Number of spot views */
  guint num_aisle_views;                      /**< Number of aisle views */
  guint spot_surf_index[MAX_DEWARPED_VIEWS];  /**< Array containing surface indices of spot surfaces */
  guint aisle_surf_index[MAX_DEWARPED_VIEWS]; /**< Array containing surface indices of aisle surfaces */
  guint surface_index[MAX_DEWARPED_VIEWS];    /**< Array of all surface indices */
  guint surface_type[MAX_DEWARPED_VIEWS];     /**< Array of type of projection for each surface. Values from enum NvDsSurfaceType */

  GstBuffer * out_gst_buf;                    /**< Pointer to the output buffer */

  NvDewarperPriv *priv;                       /**< Pointer to private data structure contaning dewarping parameters for all the output surfaces */
};

/** GStreamer boilerplate. */
struct _GstnvdewarperClass
{
  GstBaseTransformClass parent_class;
};

GType gst_nvdewarper_get_type (void);

G_END_DECLS

#endif /* __GST_NVDEWARPER_H__ */
