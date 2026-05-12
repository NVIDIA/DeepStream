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

#ifndef _DS3D_COMMON_IMPL_FRAMES__H
#define _DS3D_COMMON_IMPL_FRAMES__H

#include <ds3d/common/abi_frame.h>
#include <ds3d/common/common.h>
#include <ds3d/common/func_utils.h>
#include <ds3d/common/hpp/frame.hpp>

namespace ds3d { namespace impl {

template <typename DataTypeTP, FrameType ft, class abiBase>
class BaseFrame : public abiBase {
public:
    DataType dataType() const final { return _dataType(); }
    FrameType frameType() const final { return _frameType(); }
    MemType memType() const final { return _memType; }

    size_t bytes() const override { return _bytes; }
    void* base() const final { return (void*)(_data); }
    const Shape& shape() const final { return _shape; }
    int64_t devId() const final { return _devId; }
    bool isValid() const { return base() && bytes() > 0; }

    // Implementation public functions
    using Deleter = std::function<void(void*)>;
    void resetShape(const Shape& s) { _shape = s; }
    // function reset is not virtual
    void reset()
    {
        if (_data && _del) {
            _del(_data);
        }
        _data = nullptr;
        _bytes = 0;
        _shape = {0, {0}};
        _memType = MemType::kNone;
        _devId = 0;
        _del = nullptr;
    }

    void resetData(void* data, size_t bytes, Deleter del = nullptr)
    {
        if (_data && _del) {
            _del(_data);
        }
        _data = data;
        _bytes = bytes;
        _del = std::move(del);
    }

    BaseFrame(
        void* data, size_t bytes, const Shape& shape, MemType memType, uint64_t devId,
        Deleter&& deleter = nullptr)
        : _data(data), _bytes(bytes), _memType(memType), _devId(devId), _shape(shape),
          _del(std::move(deleter))
    {
    }
    ~BaseFrame() override { reset(); }

    template <class EleT>
    EleT& at(size_t idx)
    {
        DS_ASSERT(sizeof(EleT) * idx < _bytes);
        return static_cast<EleT*>(_data)[idx];
    }

protected:
    template <typename F>
    void setDeleter(F&& f)
    {
        _del = std::move(f);
    }

private:
    // internal
    static constexpr DataType _dataType() { return __DataTypeTrait<DataTypeTP>::_data_type(); }
    static constexpr FrameType _frameType() { return ft; }

private:
    void* _data = nullptr;
    size_t _bytes = 0;
    MemType _memType = MemType::kNone;
    int64_t _devId = 0;
    Shape _shape = {0, {0}};
    Deleter _del = nullptr;
};

template <typename DataTypeTP, FrameType ft>
using FrameBaseImpl = BaseFrame<DataTypeTP, ft, abiFrame>;

template <typename DataTypeTP, FrameType ft>
class Frame2DBaseImpl : public BaseFrame<DataTypeTP, ft, abi2DFrame> {
public:
    using typename BaseFrame<DataTypeTP, ft, abi2DFrame>::Deleter;
    Frame2DBaseImpl(
        void* data, size_t bytes, const Shape& shape, MemType memType, uint64_t devId,
        Deleter&& deleter)
        : BaseFrame<DataTypeTP, ft, abi2DFrame>(
              data, bytes, shape, memType, devId, std::move(deleter))
    {
    }

    uint32_t planes() const final { return _planes.size(); }
    const Frame2DPlane& getPlane(uint32_t idx) const final
    {
        if (idx > _planes.size()) {
            DS_ASSERT(false);
            throwError(ErrCode::kParam, "idx is larger than planes number");
        }
        return _planes[idx];
    }
    void setPlanes(const std::vector<Frame2DPlane>& p)
    {
        _planes = p;
        if (p.empty()) {
            this->resetShape(Shape{0, {0}});
        } else {
            this->resetShape(Shape{2, {(int)p[0].height, (int)p[0].width}});
        }
    }

    Frame2DBaseImpl(
        void* data, const std::vector<Frame2DPlane>& planes, size_t bytes, MemType memType,
        uint64_t devId, Deleter&& deleter)
        : BaseFrame<DataTypeTP, ft, abi2DFrame>(
              data, bytes, {0, {0}}, memType, devId, std::move(deleter))
    {
        setPlanes(planes);
    }
    ~Frame2DBaseImpl() { reset(); }

    void reset()
    {
        _planes.clear();
        BaseFrame<DataTypeTP, ft, abi2DFrame>::reset();
    }

    template <class EleT>
    EleT& at(size_t row, size_t column, uint32_t plane = 0)
    {
        DS_ASSERT(plane < _planes.size());
        DS_ASSERT(row < _planes[plane].height && column < _planes[plane].width);
        uint8_t* buf =
            (uint8_t*)this->base() + _planes[plane].offset + row * _planes[plane].pitchInBytes;
        return reinterpret_cast<EleT*>(buf + _planes[plane].bytesPerPixel * column);
    }

private:
    std::vector<Frame2DPlane> _planes;
};

// template<typename TP>
// using DepthFrameImpl = FrameImpl<vec1<TP>, TP, FrameType::kDepth>;

// template<typename TP>
// using RGBFrameImpl = FrameImpl<vec3<TP>, TP, FrameType::kColorRGB>;

// template<typename TP>
// using RGBAFrameImpl = FrameImpl<vec4<TP>, TP, FrameType::kColorRGBA>;

template <typename DataTypeTP, FrameType ft>
FrameGuard
WrapFrame(
    void* data, size_t bytes, const Shape& shape, MemType memType, uint64_t devId,
    std::function<void(void*)>&& deleter)
{
    abiRefFrame* f = NewAbiRef<abiFrame>(
        new FrameBaseImpl<DataTypeTP, ft>(data, bytes, shape, memType, devId, std::move(deleter)));
    FrameGuard frame(f, true);
    return frame;
}

template <typename DataTypeTP, FrameType ft>
Frame2DGuard
Wrap2DFrame(
    void* data, const std::vector<Frame2DPlane>& planes, size_t bytes, MemType memType,
    uint64_t devId, std::function<void(void*)> deleter)
{
    abiRef2DFrame* f = NewAbiRef<abi2DFrame>(new Frame2DBaseImpl<DataTypeTP, ft>(
        data, planes, bytes, memType, devId, std::move(deleter)));
    Frame2DGuard frame(f, true);
    return frame;
}

template <typename DataTypeTP>
FrameGuard
wrapPointXYZFrame(
    void* data, uint32_t points, MemType memType, uint64_t devId,
    std::function<void(void*)>&& deleter)
{
    Shape shape{2, {(int)points, 3}};
    return WrapFrame<DataTypeTP, FrameType::kPointXYZ>(
        data, sizeof(DataTypeTP) * ShapeSize(shape), shape, memType, devId, std::move(deleter));
}

template <typename DataTypeTP>
FrameGuard
wrapLidarXYZIFrame(
    void* data, uint32_t points, MemType memType, uint64_t devId,
    std::function<void(void*)>&& deleter)
{
    Shape shape{2, {(int)points, 4}};
    return WrapFrame<DataTypeTP, FrameType::kLidarXYZI>(
        data, sizeof(DataTypeTP) * ShapeSize(shape), shape, memType, devId, std::move(deleter));
}

template <typename DataTypeTP>
FrameGuard
wrapPointCoordUVFrame(
    void* data, uint32_t points, MemType memType, uint64_t devId,
    std::function<void(void*)>&& deleter)
{
    Shape shape{2U, {(int)points, 2}};
    return WrapFrame<DataTypeTP, FrameType::kPointCoordUV>(
        data, sizeof(DataTypeTP) * ShapeSize(shape), shape, memType, devId, std::move(deleter));
}

}}  // namespace ds3d::impl

#endif  // _DS3D_COMMON_IMPL_FRAMES__H