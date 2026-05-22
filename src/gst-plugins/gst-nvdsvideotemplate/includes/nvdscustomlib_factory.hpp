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
