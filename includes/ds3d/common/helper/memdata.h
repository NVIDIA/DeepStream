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


#ifndef DS3D_COMMON_HELPER_MEMDATA_H
#define DS3D_COMMON_HELPER_MEMDATA_H

#include "ds3d/common/helper/safe_queue.h"

namespace ds3d {

struct MemData {
    void* data = nullptr;
    size_t byteSize = 0;
    int devId = 0;
    MemType type = MemType::kCpu;
    MemData() = default;
    virtual ~MemData() = default;

protected:
    DS3D_DISABLE_CLASS_COPY(MemData);
};

class CpuMemBuf : public MemData {
public:
    CpuMemBuf() {}
    static std::unique_ptr<CpuMemBuf> CreateBuf(size_t size)
    {
        auto mem = std::make_unique<CpuMemBuf>();
        void* data = malloc(size);
        if (!data) {
            LOG_ERROR("DS3D, malloc out of memory");
            return nullptr;
        }
        mem->data = data;
        mem->byteSize = size;
        mem->devId = 0;
        mem->type = MemType::kCpu;
        return mem;
    }
    ~CpuMemBuf()
    {
        if (data) {
            free(data);
        }
    }
};

};
#endif