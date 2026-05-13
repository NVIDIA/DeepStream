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


#ifndef _DS3D_COMMON_ABI_FRAME__H
#define _DS3D_COMMON_ABI_FRAME__H

#include <ds3d/common/abi_obj.h>
#include <ds3d/common/common.h>

namespace ds3d {

struct abiFrame {
    abiFrame() = default;
    virtual ~abiFrame() = default;

    // define all ABI interface
    virtual DataType dataType() const = 0;
    virtual FrameType frameType() const = 0;
    virtual MemType memType() const = 0;
    virtual int64_t devId() const = 0;
    virtual size_t bytes() const = 0;
    // plane is for some planner 2D only
    virtual const Shape& shape() const = 0;
    virtual void* base() const = 0;
    REGISTER_TYPE_ID(0x20001)

private:
    DS3D_DISABLE_CLASS_COPY(abiFrame);
};

typedef abiRefT<abiFrame> abiRefFrame;

struct Frame2DPlane {
    uint32_t width;
    uint32_t height;
    uint32_t pitchInBytes;
    uint32_t bytesPerPixel;
    size_t offset;
    REGISTER_TYPE_ID(0x20003)
};

struct abi2DFrame : public abiFrame {
    abi2DFrame() = default;
    virtual uint32_t planes() const = 0;
    virtual const Frame2DPlane& getPlane(uint32_t idx) const = 0;
};

typedef abiRefT<abi2DFrame> abiRef2DFrame;

}  // namespace ds3d

#endif  // _DS3D_COMMON_ABI_FRAME__H