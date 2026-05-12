/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */


#ifndef DS3D_COMMON_HPP_DATA_MIXER_HPP
#define DS3D_COMMON_HPP_DATA_MIXER_HPP

#include <ds3d/common/common.h>
#include <ds3d/common/func_utils.h>
#include <ds3d/common/hpp/dataprocess.hpp>

namespace ds3d {

/**
 * @brief GuardDataMixer is the safe access entry for abiDataMixer.
 *   Applications can use it to make C-based APIs safer. it would manage
 *   abiRefDataMixer automatically. with that, App user do not need to
 *   refCopy_i or destroy abiRefDataMixer manually.
 *
 *   For example:
 *     abiRefDataMixer* rawRef = creator();
 *     GuardDataMixer guardMixer(rawRef, true); // take the ownership of rawRef
 *     guardMixer.setUserData(userdata, [](void*){ ...free... });
 *     guardMixer.setErrorCallback([](ErrCode c, const char* msg){ stderr << msg; });
 *     ErrCode c = guardMixer.start(config, path);
 *     DS_ASSERT(isGood(c));
 *     c = guardMixer.start(config, path);  // invoke abiDataMixer::start_i(...)
 *     GuardDataMap inputData = ...; // prepare input data
 *     // invoke abiDataMixer::process_i(...)
 *     c = guardMixer.process(inputData,
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
 *     c = guardMixer.flush();
 *     c = guardMixer.stop(); // invoke abiDataMixer::stop_i(...)
 *     guardMixer.reset(); // destroy abiRefDataMixer, when all reference
 *                          // destroyed, abiDataMixer would be freed.
 */
class GuardDataMixer : public GuardDataProcess<abiDataMixer> {
    using _Base = GuardDataProcess<abiDataMixer>;

public:
    template <typename... Args>
    GuardDataMixer(Args&&... args) : _Base(std::forward<Args>(args)...)
    {
    }
    ~GuardDataMixer() = default;

    ErrCode process(
        const int portId,
        GuardDataMap datamap,
        abiOnDataCB::CppFunc inputConsumedCB)
    {
        GuardCB<abiOnDataCB> guardConsumedCb;
        guardConsumedCb.setFn<ErrCode, const abiRefDataMap*>(std::move(inputConsumedCB));

        DS_ASSERT(ptr());
        ErrCode code =
            ptr()->process_i(portId, datamap.abiRef(), guardConsumedCb.abiRef());
        return code;
    }

    ErrCode setOutputCb(
        abiOnDataCB::CppFunc outputDataCB)
    {
        GuardCB<abiOnDataCB> guardOutputCb;
        guardOutputCb.setFn<ErrCode, const abiRefDataMap*>(std::move(outputDataCB));

        DS_ASSERT(ptr());
        ErrCode code =
            ptr()->setOutputCb_i(guardOutputCb.abiRef());
        return code;
    }

    ErrCode updateInput(int portId, MixerUpdate updateType) { return ptr()->updateInput_i(portId, updateType); }
};

}  // namespace ds3d

#endif  // DS3D_COMMON_HPP_DATA_MIXER_HPP
