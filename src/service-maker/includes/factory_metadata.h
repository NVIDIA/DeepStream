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