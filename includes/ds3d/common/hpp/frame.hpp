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


#ifndef _DS3D_COMMON_HPP_FRAME__HPP
#define _DS3D_COMMON_HPP_FRAME__HPP

#include <ds3d/common/abi_frame.h>
#include <ds3d/common/func_utils.h>

#include <ds3d/common/hpp/obj.hpp>

namespace ds3d {


using FrameGuard = GuardDataT<abiFrame>;

#if 0
// abiFrame Guard
class FrameGuard : public GuardDataT<abiFrame> {
    using _Base = GuardDataT<abiFrame>;

public:
    FrameGuard() = default;
    template <typename... Args>
    FrameGuard(Args&&... args) : _Base(std::forward<Args>(args)...)
    {
    }

    template <class EleT>
    EleT& at(size_t idx)
    {
        abiFrame* f = this->ptr();
        DS_ASSERT(this->ptr());
        DS3D_THROW_ERROR(idx < ShapeSize(f->shape()), ErrCode::kOutOfRange, "idx out of range");
        uint32_t eleSize = dataTypeBytes(f->dataType());
        DS_ASSERT(sizeof(EleT) < eleSize);
        return *static_cast<EleT*>((uint8_t*)f->base() + eleSize * idx);
    }
};
#endif

// abi2DFrame Guard
using Frame2DGuard = GuardDataT<abi2DFrame>;

}  // namespace ds3d

#endif  // _DS3D_COMMON_HPP_FRAME__HPP