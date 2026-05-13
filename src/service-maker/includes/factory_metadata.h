/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * <b>Header used for creating custom factories </b>
 *
 */
#ifndef _DEEPSTREAM_FACTORY_METADATA_H_
#define _DEEPSTREAM_FACTORY_METADATA_H_

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Information required by a custom factory */
typedef struct _FactoryMetadata {
  /** A brief name */
  const char* name;
  /** Long name */
  const char* long_name;
  /** Category name */
  const char* klass;
  /** Detailed introduction */
  const char* description;
  /** Author */
  const char* author;
  /** Unique number to identify the type of the object created by the factory */
  unsigned long object_type;
  /** Supported signal names separated by '/', used only by signal handlers */
  const char* signals;
} FactoryMetadata;


unsigned long gst_custom_factory_get_type (const char* name);
const FactoryMetadata get_custom_factory_info(void);
const char* get_custom_factory_product_param_spec(void);
#ifdef __cplusplus
}
#endif

#endif