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