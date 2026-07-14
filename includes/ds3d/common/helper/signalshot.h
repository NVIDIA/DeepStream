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
