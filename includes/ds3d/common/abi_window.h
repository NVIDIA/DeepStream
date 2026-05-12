/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */


#ifndef _DS3D_COMMON_ABI_WINDOW_H
#define _DS3D_COMMON_ABI_WINDOW_H

#include <ds3d/common/common.h>
#include <ds3d/common/abi_obj.h>
#include <ds3d/common/abi_dataprocess.h>

namespace ds3d {

// ABI Interface for window system
class abiWindow {
public:
    // window closed callback
    typedef abiCallBackT<> CloseCB;
    // key pressed callback function(int key, int scancode, int action, int mods)
    typedef abiCallBackT<int, int, int, int> KeyPressCB;
    // framebuffer size changed callback function(int width, int height)
    typedef abiCallBackT<int, int> FbSizeChangedCB;

    // mouse cursor changed callback function(double xpos, double ypos)
    typedef abiCallBackT<double, double> MouseChangedCB;

    virtual ~abiWindow() = default;
    // get GLFWwindow raw pointer
    virtual void* getNativeWindow() = 0;
    // set close event callback
    virtual void setCloseCallback(const CloseCB* closeCb) = 0;
    // set key pressed event callback
    virtual void setKeyPressCallback(const KeyPressCB* keyCb) = 0;
    // set frame buffer size changed event callback
    virtual void setFbSizeChangedCallback(const FbSizeChangedCB* fbSizeChangedCb) = 0;
    // set mouse changed event callback, replace all other callbacks
    virtual void setMouseChangedCallback(const MouseChangedCB* mouseChangedCb) = 0;
    // append mouse changed event callback
    virtual void appendMouseChangedCallback(const MouseChangedCB* mouseChangedCb) = 0;
};

// raw reference pointer for abiWindow
typedef abiRefT<abiWindow> abiRefWindow;

}  // namespace ds3d

#endif  // _DS3D_COMMON_ABI_WINDOW_H