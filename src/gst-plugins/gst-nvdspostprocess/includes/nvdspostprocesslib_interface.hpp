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

#ifndef __NVDSPOSTPROCESSLIB_INTERFACE_HPP__
#define __NVDSPOSTPROCESSLIB_INTERFACE_HPP__

#include <string>
#include <gst/gstbuffer.h>

enum class BufferResult {
    Buffer_Ok,      // Push the buffer from submit_input function
    Buffer_Drop,    // Drop the buffer inside submit_input function
    Buffer_Async,   // Return from submit_input function, postprocess lib to push the buffer
    Buffer_Error    // Error occured
};

struct DSPostProcess_CreateParams {
    GstBaseTransform *m_element;
    guint m_gpuId;
    cudaStream_t m_cudaStream;
    bool m_preprocessor_support;
};

struct Property
{
  Property(std::string arg_key, std::string arg_value) : key(arg_key), value(arg_value)
  {
  }

  std::string key;
  std::string value;
};

class IDSPostProcessLibrary
{
public:
    virtual bool HandleEvent (GstEvent *event) = 0;
    virtual bool SetConfigFile (const gchar *config_file) = 0;
    virtual BufferResult ProcessBuffer (GstBuffer *inbuf) = 0;
    virtual ~IDSPostProcessLibrary() {};
};

#endif
