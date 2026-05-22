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