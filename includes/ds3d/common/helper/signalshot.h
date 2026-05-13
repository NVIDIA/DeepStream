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


#ifndef DS3D_COMMON_HELPER_SIGNALSHOT_H
#define DS3D_COMMON_HELPER_SIGNALSHOT_H

#include <ds3d/common/common.h>

namespace ds3d {

class SignalShot {
    std::mutex _mutex;
    std::condition_variable _cond;

public:
    void wait(uint64_t msec)
    {
        std::chrono::milliseconds t(msec);
        std::unique_lock<std::mutex> locker(_mutex);
        _cond.wait_for(locker, t);
    }
    void signal() { _cond.notify_all(); }
    std::mutex& mutex() { return _mutex; }
};

}  // namespace ds3d

#endif  //
