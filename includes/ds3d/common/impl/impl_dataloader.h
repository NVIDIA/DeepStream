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


#ifndef _DS3D_COMMON_IMPL_BASE_IDATALOADER__H
#define _DS3D_COMMON_IMPL_BASE_IDATALOADER__H

#include "ds3d/common/hpp/datamap.hpp"
#include "ds3d/common/impl/impl_dataprocess.h"

namespace ds3d { namespace impl {

/**
 * @brief Any custom dataloader must derive from BaseImplDataLoader,
 *   For convenience, If dataloader is in sync mode, user can also derive
 *   from SyncImplDataLoader
 *
 *   For custom lib implementation, user need to implement the following
 *   virtual functions:
 *     startImpl(), // user also need setCaps(port) in startImpl
 *     stopImpl(), // stop all resources for dataloader
 *     readDataImpl(), // read data in sync mode, return kEndOfStream if end of stream.
 *     readDataAsyncImpl(dataReadyCb), // read data in async mode. Once data is ready,
 *         need call dataReadyCb(datamap) to notify application
 *     flushImpl(), // flush data in process
 */
class BaseImplDataLoader : public BaseImplDataProcessor<abiDataLoader> {
public:
    BaseImplDataLoader() {}
    ~BaseImplDataLoader() override = default;

    ErrCode readData_i(abiRefDataMap*& datamap) final
    {
        DS_ASSERT(!datamap);
        DS3D_FAILED_RETURN(
            getStateSafe() == State::kRunning, ErrCode::kState, "dataloader is not started");
        GuardDataMap data;
        ErrCode code = readDataImpl(data);
        datamap = data.release();
        return code;
    }
    ErrCode readDataAsync_i(const abiOnDataCB* dataReadyCb) final
    {
        DS3D_FAILED_RETURN(
            getStateSafe() == State::kRunning, ErrCode::kState, "dataloader is not started");

        GuardCB<abiOnDataCB> guardCb(dataReadyCb ? dataReadyCb->refCopy() : nullptr);
        OnGuardDataCBImpl readyCB = [guardCb = std::move(guardCb), this](
                                        ErrCode code, GuardDataMap data) {
            guardCb(code, data.abiRef());
        };
        return readDataAsyncImpl(std::move(readyCB));
    }

protected:
    // cplusplus virtual interface, user need to derive from
    virtual ErrCode readDataImpl(GuardDataMap& datamap) = 0;
    virtual ErrCode readDataAsyncImpl(OnGuardDataCBImpl dataReadCB) = 0;
};

class SyncImplDataLoader : public BaseImplDataLoader {
protected:
    SyncImplDataLoader() = default;
    ErrCode readDataAsyncImpl(OnGuardDataCBImpl dataReadCB) final { return ErrCode::kUnsupported; }
    ErrCode flushImpl() override { return ErrCode::kGood; }
};

}}  // namespace ds3d::impl

#endif  // _DS3D_COMMON_IMPL_BASE_IDATALOADER__H
