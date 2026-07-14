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
#define NVDS_VERSION_MINOR 1
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
