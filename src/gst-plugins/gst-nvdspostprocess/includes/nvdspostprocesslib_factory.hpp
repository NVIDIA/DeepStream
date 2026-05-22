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
