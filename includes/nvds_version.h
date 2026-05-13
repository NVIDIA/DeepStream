/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * <b>NVIDIA DeepStream version API</b>
 *
 * @b Description: This file specifies the APIs used to view the version of
 * NVIDIA DEEPSTREAM and its dependencies, such as TensorRT, CUDA and cuDNN.
 */

/**
 * @defgroup  ee_version  Version Number API
 *
 * Defines the API used to get the current version number of DeepStream and
 * its dependencies.
 *
 * @ingroup NvDsUtilsApi
 * @{
 */

#ifndef _NVDS_VERSION_H_
#define _NVDS_VERSION_H_

#define NVDS_VERSION_MAJOR 9
#define NVDS_VERSION_MINOR 0
#define NVDS_VERSION_MICRO 0

#ifdef __cplusplus
extern "C"
{
#endif

/**
 * Get the DEEPSTREAM_SDK major and minor version
 * numbers and return them in major and minor variable pointers.
 *
 * @param[in] major holds the major part of DEEPSTREAM_SDK version.
 * @param[in] minor holds the minor part of DEEPSTREAM_SDK version.
 */
void nvds_version (unsigned int * major, unsigned int * minor);

/**
 * Print the version as major.minor.
 * To obtain major and minor, this function calls @ref nvds_version.
 */
void nvds_version_print (void);

/**
 * Print the versions of dependencies such as Cuda, cuDNN and TensorRT.
 */
void nvds_dependencies_version_print (void);

#ifdef __cplusplus
}
#endif

#endif

/** @} */
