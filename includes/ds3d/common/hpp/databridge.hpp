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


#ifndef DS3D_COMMON_HPP_DATA_BRIDGE_HPP
#define DS3D_COMMON_HPP_DATA_BRIDGE_HPP

#include <ds3d/common/common.h>
#include <ds3d/common/func_utils.h>
#include <ds3d/common/hpp/dataprocess.hpp>

namespace ds3d {

/**
 * @brief GuardDataBridge is the safe access entry for abiDataBridge.
 *   Applications can use it to make C-based APIs safer. it would manage
 *   abiRefDataBridge automatically. with that, App user do not need to
 *   refCopy_i or destroy abiRefDataBridge manually.
 *
 *   For example:
 *     abiRefDataBridge* rawRef = creator();
 *     GuardDataBridge guardFilter(rawRef, true); // take the ownership of rawRef
 *     guardFilter.setUserData(userdata, [](void*){ ...free... });
 *     guardFilter.setErrorCallback([](ErrCode c, const char* msg){ stderr << msg; });
 *     ErrCode c = guardFilter.start(config, path);
 *     DS_ASSERT(isGood(c));
 *     c = guardFilter.start(config, path);  // invoke abiDataBridge::start_i(...)
 *     GuardDataMap inputData = ...; // prepare input data
 *     // invoke abiDataBridge::process_i(...)
 *     c = guardFilter.process(inputData,
 *         [](ErrCode c, const abiRefDataMap* d){
 *            GuardDataMap outputData(*d); // output data processing
 *            std::cout << "output data processing starts" << std::endl;
 *         },
 *         [](ErrCode c, const abiRefDataMap* d){
 *            GuardDataMap doneData(*d);
 *            std::cout << "input data consumed" << std::endl;
 *         });
 *     DS_ASSERT(isGood(c));
 *     //... wait for all data processed before stop
 *     c = guardFilter.flush();
 *     c = guardFilter.stop(); // invoke abiDataBridge::stop_i(...)
 *     guardFilter.reset(); // destroy abiRefDataBridge, when all reference
 *                          // destroyed, abiDataBridge would be freed.
 */
class GuardDataBridge : public GuardDataProcess<abiDataBridge> {
    using _Base = GuardDataProcess<abiDataBridge>;

public:
    template <typename... Args>
    GuardDataBridge(Args&&... args) : _Base(std::forward<Args>(args)...)
    {
    }
    ~GuardDataBridge() = default;

    ErrCode process(
        const struct VideoBridge2dInput* inputData, abiOnDataCB::CppFunc outputDataCB,
        abiOnBridgeDataCB::CppFunc inputConsumedCB)
    {
        GuardCB<abiOnDataCB> guardOutputCb;
        GuardCB<abiOnBridgeDataCB> guardConsumedCb;
        guardOutputCb.setFn<ErrCode, const abiRefDataMap*>(std::move(outputDataCB));
        guardConsumedCb.setFn<ErrCode, const struct VideoBridge2dInput*>(std::move(inputConsumedCB));

        DS_ASSERT(ptr());
        ErrCode code =
            ptr()->process_i(inputData, guardOutputCb.abiRef(), guardConsumedCb.abiRef());
        return code;
    }
};

}  // namespace ds3d

#endif  // DS3D_COMMON_HPP_DATA_BRIDGE_HPP
