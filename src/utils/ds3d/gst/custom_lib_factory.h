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

#ifndef NVDS3D_CUSTOMLIB_FACTORY_HPP
#define NVDS3D_CUSTOMLIB_FACTORY_HPP

#include <dlfcn.h>
#include <ds3d/common/common.h>
#include <errno.h>

#include <functional>
#include <iostream>

namespace ds3d {

template <class T>
T*
dlsym_ptr(void* handle, char const* name)
{
    return reinterpret_cast<T*>(dlsym(handle, name));
}

class CustomLibFactory {
public:
    CustomLibFactory() = default;

    ~CustomLibFactory()
    {
        if (_libHandle && !_keepOpen) {
            LOG_DEBUG("dlclose %s", _libName.c_str());
            dlclose(_libHandle);
        }
    }
    void keepOpen(bool b) { _keepOpen = b; }

    template <class CustomRefCtx>
    CustomRefCtx* CreateCtx(const std::string& libName, const std::string& symName)
    {
        if (!_libHandle) {
            _libName = libName;
            _libHandle = dlopen(libName.c_str(), RTLD_NOW | RTLD_LOCAL);
        } else {
            DS3D_FAILED_RETURN(
                _libName == libName || libName.empty(), nullptr,
                "CustomLibFactory existing libname: %s is different from new lib: %s",
                _libName.c_str(), libName.c_str());
        }
        DS3D_FAILED_RETURN(
            _libHandle, nullptr, "open custome-lib: %s failed. dlerr: %s", _libName.c_str(),
            dlerror());
        LOG_INFO("Library Opened Successfully");

        std::function<CustomRefCtx*()> createCtxFunc =
            dlsym_ptr<CustomRefCtx*()>(_libHandle, symName.c_str());
        DS3D_FAILED_RETURN(
            createCtxFunc, nullptr, "dlsym: %s not found, error: %s", symName.c_str(), dlerror());

        CustomRefCtx* customCtx = createCtxFunc();
        DS3D_FAILED_RETURN(
            customCtx, nullptr, "create custom context failed during call: %s", symName.c_str());
        LOG_INFO("Custom Context created from %s", symName.c_str());
        return customCtx;
    }

public:
    void* _libHandle = nullptr;
    std::string _libName;
    bool _keepOpen = false;
};

}  // namespace ds3d

#endif
