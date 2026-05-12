/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
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
