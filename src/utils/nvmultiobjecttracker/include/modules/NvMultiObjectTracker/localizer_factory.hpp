/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef _NVTLOC_FACTORY_H
#define _NVTLOC_FACTORY_H

#include "modules/NvMultiObjectTracker/localizer.hpp"
#include "modules/NvMultiObjectTracker/NvMultiObjectTracker.hpp"

/** @brief Creates batch localizers through a singleton factory. */
class LocalizerFactory
{
public:
    /** @brief Return the static instance reference to the factory. */
    static LocalizerFactory& getInstance()
    {
        static LocalizerFactory instance;
        return instance;
    }

    /**
     * @brief Create a concrete localizer from a tracker configuration.
     * @param[in] customConfigFilePath Path to the custom configuration file.
     * @return Shared pointer to the created localizer.
     */
    std::shared_ptr<IBatchLocalizer> makeLocalizer(const std::string customConfigFilePath)
    {
		/// TODO: Allow other types of trackers to be made.
		return std::make_shared<NvMultiObjectTrackerBase>(customConfigFilePath);
    }
};

#endif // _NVTLOC_FACTORY_H
