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
 * @file nvdewarper_property_parser.h
 * <b>NVIDIA DeepStream GStreamer nvdewarper properties and API Specification </b>
 *
 * @b Description: This file specifies the properties and APIs for
 * the DeepStream GStreamer nvdewarper Plugin.
 */

#ifndef NVDEWARPER_PROPERTY_FILE_PARSER_H_
#define NVDEWARPER_PROPERTY_FILE_PARSER_H_

#include <gst/gst.h>
#include "gstnvdewarper.h"

/**
 * @addtogroup one Global properties
 *
 * @brief Global properties that apply to all surfaces and are specified under [property] group
 * @{
 */
#define CONFIG_GROUP_DEWARPER_PROPERTY "property"                                   /**< Identifier for [property] group */
#define CONFIG_GROUP_DEWARPER_PROPERTY_OUTPUT_WIDTH "output-width"                  /**< Scale dewarped surfaces to specified output width */
#define CONFIG_GROUP_DEWARPER_PROPERTY_OUTPUT_HEIGHT "output-height"                /**< Scale dewarped surfaces to specified output height */
#define CONFIG_GROUP_DEWARPER_PROPERTY_CUDA_MEMORY_TYPE "cuda-memory-type"          /**< NVDS CUDA memory type */
#define CONFIG_GROUP_DEWARPER_PROPERTY_NUM_BATCH_BUFFERS "num-batch-buffers"        /**< Number of dewarped output surfaces per buffer */
#define CONFIG_GROUP_DEWARPER_PROPERTY_DUMP_FRAMES "dewarp-dump-frames"             /**< Number of dewarped frames to dump */
#define CONFIG_GROUP_DEWARPER_PROPERTY_AISLE_CALIB_FILE "aisle-calibration-file"    /**< Pathname of the configuration file for aisle view. */
#define CONFIG_GROUP_DEWARPER_PROPERTY_SPOT_CALIB_FILE "spot-calibration-file"      /**< Pathname of the configuration file for spot view. */
/** @} */

/**
 * @addtogroup two Surface properties
 *
 * @brief Surface properties which can be different for every surface.
 * These are specified under [surface<n>] group
 * @{
 */
#define CONFIG_GROUP_DEWARPER_SURFACE_ATTRS_PREFIX "surface"                /**< Identifier for [surface<n>] group */
#define CONFIG_GROUP_DEWARPER_SURFACE_INDEX "surface-index"                 /**< An index that distinguishes surfaces of the same projection type. */
#define CONFIG_GROUP_DEWARPER_SURFACE_PROJECTION_TYPE "projection-type"     /**< Surface projection of type "NvDsSurfaceType" */
#define CONFIG_GROUP_DEWARPER_SURFACE_WIDTH "width"                         /**< Dewarped surface width. */
#define CONFIG_GROUP_DEWARPER_SURFACE_HEIGHT "height"                       /**< Dewarped surface height. */
#define CONFIG_GROUP_DEWARPER_SURFACE_TOP_ANGLE "top-angle"                 /**< Desired Top field of view angle, in degrees. */
#define CONFIG_GROUP_DEWARPER_SURFACE_BOTTOM_ANGLE "bottom-angle"           /**< Desired Bottom field of view angle, in degrees. */
#define CONFIG_GROUP_DEWARPER_SURFACE_PITCH "pitch"                         /**< View selection parameter pitch in degrees. Corresponds to X axis */
#define CONFIG_GROUP_DEWARPER_SURFACE_YAW "yaw"                             /**< View selection parameter yaw in degrees. Corresponds to Y axis */
#define CONFIG_GROUP_DEWARPER_SURFACE_ROLL "roll"                           /**< View selection parameter roll in degrees. Corresponds to Z axis */
#define CONFIG_GROUP_DEWARPER_SURFACE_FOCAL_LENGTH "focal-length"           /**< Array of 2 numbers. X & Y focal length of camera lens */
#define CONFIG_GROUP_DEWARPER_SURFACE_ADDRESS_MODE "cuda-address-mode"      /**< Cuda Texture addressing mode. */
/** A four-element array of three-character C-strings terminated by \0 that
 * represent the dequence of rotation axes about which the view gets rotated based on rotation angles specified */
#define CONFIG_GROUP_DEWARPER_SURFACE_ROTATION_AXES "rot-axes"
#define CONFIG_GROUP_DEWARPER_SURFACE_CONTROL "control"                     /**< Projection-specific controls for Panini, Stereographic and Pushbroom projections */
#define CONFIG_GROUP_DEWARPER_SURFACE_ROTATION_MATRIX "rot-matrix"          /**< Conventional 9 element rotation matrix */

#define CONFIG_GROUP_DEWARPER_SURFACE_FIELD_OF_VIEW "src-fov"               /**< used to compute focal-length if the latter is not provided */
#define CONFIG_GROUP_DEWARPER_SURFACE_DISTORTION "distortion"               /**< distortion: up to 5 distortion parameters */
#define CONFIG_GROUP_DEWARPER_SURFACE_SRC_X0 "src-x0"                       /**< source principal point X */
#define CONFIG_GROUP_DEWARPER_SURFACE_SRC_Y0 "src-y0"                       /**< source principal point Y */

/** Destination focal length : Useful to introduce zoom-in and zoom-out effect */
#define CONFIG_GROUP_DEWARPER_SURFACE_DST_FOCAL_LENGTH "dst-focal-length"
/** Destination principal point corresponding to ""dst-focal-length" */
#define CONFIG_GROUP_DEWARPER_SURFACE_DST_PRINCIPAL_POINT "dst-principal-point"
/** @} */

gboolean
nvdewarper_parse_config_file (Gstnvdewarper *nvdewarper, gchar *cfg_file_path);

gboolean
nvdewarper_parse_dewarper_props (Gstnvdewarper *nvdewarper, GKeyFile *key_file,
    gchar *group, gchar *cfg_file_path);


#endif /* NVDEWARPER_PROPERTY_FILE_PARSER_H_ */
