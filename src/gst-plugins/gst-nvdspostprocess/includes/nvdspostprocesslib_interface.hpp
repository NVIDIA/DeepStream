/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
