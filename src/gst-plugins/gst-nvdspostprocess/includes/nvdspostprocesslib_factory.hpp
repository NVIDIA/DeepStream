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

#ifndef __NVDSPOSTPROCESSLIB_FACTORY_HPP__
#define __NVDSPOSTPROCESSLIB_FACTORY_HPP__

#include <dlfcn.h>
#include <errno.h>

#include <iostream>
#include <functional>

#include "nvdspostprocesslib_interface.hpp"

template<class T>
T* dlsym_ptr(void* handle, char const* name) {
  return reinterpret_cast<T*>(dlsym(handle, name));
}

class DSPostProcessLibrary_Factory
{
public:
    DSPostProcessLibrary_Factory()
    {
    }

    ~DSPostProcessLibrary_Factory()
    {
        if (m_libHandle)
        {
            dlclose(m_libHandle);
            m_libHandle = NULL;
            m_libName.clear();
        }
    }

    IDSPostProcessLibrary *CreateCustomAlgoCtx(std::string libName, DSPostProcess_CreateParams  *params)
    {
        m_libName.assign(libName);

        m_libHandle = dlopen(m_libName.c_str(), RTLD_NOW);
        if (m_libHandle)
        {
            m_CreateAlgoCtx = dlsym_ptr<IDSPostProcessLibrary*(DSPostProcess_CreateParams*)>(m_libHandle, "CreateCustomAlgoCtx");
            if (!m_CreateAlgoCtx)
            {
              std::cout << "CreateCustomAlgoCtx function not found in library\n";
              return nullptr;
            }
        }
        else
        {
           // throw std::runtime_error(dlerror());
            std::cout << dlerror() << std::endl;
            return nullptr;
        }
        return m_CreateAlgoCtx((DSPostProcess_CreateParams*)params);
    }

public:
    void *m_libHandle;
    std::string m_libName;
    std::function<IDSPostProcessLibrary*(DSPostProcess_CreateParams *)> m_CreateAlgoCtx;
};

#endif
