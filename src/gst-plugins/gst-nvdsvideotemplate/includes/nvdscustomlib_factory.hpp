/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef __NVDSCUSTOMLIB_FACTORY_HPP__
#define __NVDSCUSTOMLIB_FACTORY_HPP__

#include <dlfcn.h>
#include <errno.h>

#include <iostream>
#include <functional>

#include "nvdscustomlib_interface.hpp"

template<class T>
T* dlsym_ptr(void* handle, char const* name) {
  return reinterpret_cast<T*>(dlsym(handle, name));
}

class DSCustomLibrary_Factory
{
public:
    DSCustomLibrary_Factory()
    {
    }

    ~DSCustomLibrary_Factory()
    {
        if (m_libHandle)
        {
            dlclose(m_libHandle);
            m_libHandle = NULL;
            m_libName.clear();
        }
    }

    IDSCustomLibrary *CreateCustomAlgoCtx(std::string libName, GObject* object)
    {
        m_libName.assign(libName);

        m_libHandle = dlopen(m_libName.c_str(), RTLD_NOW);
        std::function<IDSCustomLibrary*(GObject*)> createAlgoCtx = nullptr;
        if (m_libHandle)
        {
            //std::cout << "Library Opened Successfully" << std::endl;

            createAlgoCtx = dlsym_ptr<IDSCustomLibrary*(GObject*)>(m_libHandle, "CreateCustomAlgoCtx");
            if (!createAlgoCtx)
            {
                throw std::runtime_error("createCustomAlgoCtx function not found in library");
            }
        }
        else
        {
            throw std::runtime_error(dlerror());
        }

        return createAlgoCtx ? createAlgoCtx(object) : nullptr;
    }

    IDSCustomLibrary *Initialize()
    {
        m_libHandle = nullptr;
        m_libName.clear();
        return nullptr;
    }

public:
    void *m_libHandle;
    std::string m_libName;
};

#endif
