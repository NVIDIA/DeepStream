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

#ifndef REIDLIB_REID_FACTORY_HPP_
#define REIDLIB_REID_FACTORY_HPP_

#include <memory>
#include "modules/ReID/IReID.hpp"
// Included so the visual tracker type can be compared.
#include "modules/VisualTracker/IVisualTracker.hpp"
#include "modules/ReID/DummyReID.hpp"
#ifndef _WIN32
#include "modules/ReID/ReID.hpp"
#include "modules/ReID/EnsembleReID.hpp"
#endif

/** @brief Stores the implementation type selected by the factory. */
class ReIDFactoryParams : public NvTrackerParams
{
public:
    /** @brief Construct factory parameters with the dummy ReID implementation selected. */
    ReIDFactoryParams()
    {
        m_reidType = REID_TYPES::DUMMY_REID;
    }

    /**
     * @brief Construct a ReIDFactoryParams instance.
     * @param[in] configFilePath Path to the tracker configuration file.
     * @param[in] moduleName Configuration module name.
     */
    ReIDFactoryParams(const std::string &configFilePath, const std::string moduleName)
    {
        YAML::Node configRoot = getConfigRoot(configFilePath);
        this->read(configRoot[moduleName]);
    }

    /**
     * @brief Construct a ReIDFactoryParams instance.
     * @param[in] fn Parsed YAML node.
     */
    ReIDFactoryParams(const YAML::Node &fn)
    {
        read(fn);
    }

    /**
     * @brief Load configuration values from a YAML node.
     * @param[in] fn Parsed YAML node.
     */
    void read(const YAML::Node &fn) override
    {
        /// Set the default values.
        *this = ReIDFactoryParams();
        if(!fn.size()) { return;}

        for (YAML::const_iterator fit = fn.begin(); fit != fn.end(); ++fit)
        {
            std::string paramKey = fit->first.as<std::string>();

            if (paramKey == "reidType")
            {
                int reidTypeInt = (int)fit->second.as<float>();
                /// Validate the ReID type in the range 0 through 3.
                if (reidTypeInt < 0 || reidTypeInt > 3)
                {
                    throw std::runtime_error(
                        "Invalid reidType value: " + std::to_string(reidTypeInt) + ". Valid range is 0-3 (0=DUMMY_REID, 1=NvDEEPSORT, 2=REASSOC, 3=BOTH)."
                    );
                }
                m_reidType = (REID_TYPES)reidTypeInt;
            }
        }
    }

    REID_TYPES m_reidType; //!< The type of reid module.
};

/** @brief Constructs the configured ReID implementation. */
class ReIDFactory
{
public:
    /** @brief Get the singleton factory instance. */
    static ReIDFactory& getInstance()
    {
        static ReIDFactory instance;
        return instance;
    }

    /**
     * @brief Construct the configured ReID implementation.
     * @param[in] configFilePath Path to the tracker configuration file.
     * @param[in] visualTrackerType Visual-tracker implementation paired with ReID.
     * @return The ReID implementation selected by the configured reidType.
     */
    std::shared_ptr<IReID> makeReID(
        const std::string configFilePath, const VISUAL_TRACKER_TYPES visualTrackerType = VISUAL_TRACKER_TYPES::DUMMY_VOT
    )
    {
#ifdef _WIN32
        m_params = ReIDFactoryParams(configFilePath, "ReID");
        if (m_params.m_reidType != REID_TYPES::DUMMY_REID) {
            throw std::runtime_error("ReID is not available in the Windows IOU-only tracker build");
        }
        return std::make_shared<DummyReID>(configFilePath);
#else
        /// Parse the ReID config.
        m_params = ReIDFactoryParams(configFilePath, "ReID");

        switch (m_params.m_reidType)
        {
            case REID_TYPES::NvDEEPSORT:
            case REID_TYPES::REASSOC:
            case REID_TYPES::BOTH:
            {
                if ((visualTrackerType == VISUAL_TRACKER_TYPES::NvDCF_VOT ||
                     visualTrackerType == VISUAL_TRACKER_TYPES::NvDCF_VPI_VOT)
                && (m_params.m_reidType & REID_TYPES::NvDEEPSORT) )
                {
                    /// Only one of NvDCF or NvDeepSORT can be used for feature extraction.
                    throw std::runtime_error(
                        "Only one of NvDCF or NvDeepSORT can be used for feature extraction. To use NvDCF, please set reidType as 0; To use NvDeepSORT, please set visualTrackerType: 0"
                    );
                }
                else
                {
                    return std::make_shared<EnsembleReID>(configFilePath);
                }
            }
            case REID_TYPES::DUMMY_REID:
            default:
            {
                return std::make_shared<DummyReID>(configFilePath);
            }
        }
        return nullptr;
#endif
    }

protected:
    ReIDFactoryParams m_params; //!< Config params.
};

#endif // REIDLIB_REID_FACTORY_HPP_
