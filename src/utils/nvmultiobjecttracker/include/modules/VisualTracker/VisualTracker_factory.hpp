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

#ifndef VISUALTRACKERLIB_VISUALTRACKER_FACTORY_HPP_
#define VISUALTRACKERLIB_VISUALTRACKER_FACTORY_HPP_

#include <memory>
#include "modules/NvMultiObjectTracker/NvTrackerParams.hpp"
#include "modules/VisualTracker/IVisualTracker.hpp"

// Visual trackers.
#include "modules/VisualTracker/DummyVisualTracker.hpp"
#ifndef _WIN32
#include "modules/VisualTracker/VisualTracker.hpp"
#include "modules/VisualTracker/DCFTracker.hpp"
#endif

/** @brief Configuration parameters for the visual tracker factory. */
class VisualTrackerFactoryParams : public NvTrackerParams
{
public:
	/** @brief Construct default factory parameters selecting the dummy tracker. */
	VisualTrackerFactoryParams()
	{
		m_visualTrackerType = VISUAL_TRACKER_TYPES::DUMMY_VOT;
	}

	/**
	 * @brief Construct factory parameters from a named module in a file.
	 * @param[in] configFilePath The path to the configuration file.
	 * @param[in] moduleName The configuration module name.
	 */
	VisualTrackerFactoryParams(const std::string &configFilePath, const std::string moduleName)
	{
		YAML::Node configRoot = getConfigRoot(configFilePath);
		this->read(configRoot[moduleName]);
	}

	/**
	 * @brief Construct factory parameters from a YAML node.
	 * @param[in] fn Parsed YAML structure containing the configuration.
	 */
	VisualTrackerFactoryParams(const YAML::Node &fn)
	{
		read(fn);
	}

    /**
     * @brief Read visual tracker factory parameters from a YAML node.
     * @param[in] fn Parsed YAML structure of configuration parameters.
     */
    void read(const YAML::Node &fn) override
    {
		/// Set the default values.
		*this = VisualTrackerFactoryParams();
        if(!fn.size()) { return;}

        for (YAML::const_iterator fit = fn.begin(); fit != fn.end(); ++fit)
        {
			std::string paramKey = fit->first.as<std::string>();

			if (paramKey == "visualTrackerType")
			{
				int trackerTypeInt = (int)fit->second.as<float>();
				/// Validate the visual tracker type (valid range: 0-2).
				if (trackerTypeInt < 0 || trackerTypeInt > 2)
				{
					throw std::runtime_error(
						"Invalid visualTrackerType value: " + std::to_string(trackerTypeInt) + ". Valid range is 0-2 (0=DUMMY_VOT, 1=NvDCF_VOT, 2=NvDCF_VPI_VOT)."
					);
				}
				m_visualTrackerType = (VISUAL_TRACKER_TYPES)trackerTypeInt;
			}
        }
    }

	// [Configuration parameters for the factory]
    VISUAL_TRACKER_TYPES m_visualTrackerType; //!< The type of visual tracker.
};

/** @brief Create a concrete visual tracker from the configuration. */
class VisualTrackerFactory
{
public:
    /** @brief Return the process-wide visual tracker factory instance. */
    static VisualTrackerFactory& getInstance()
    {
        static VisualTrackerFactory instance;
        return instance;
    }

    /**
     * @brief Create the visual tracker selected by a configuration file.
     * @param[in] configFilePath The path to the configuration file.
     * @return The configured visual tracker instance.
     */
    std::shared_ptr<IVisualTracker> makeVisualTracker(const std::string configFilePath)
    {
#ifdef _WIN32
		m_params = VisualTrackerFactoryParams(configFilePath, "VisualTracker");
		if (m_params.m_visualTrackerType != VISUAL_TRACKER_TYPES::DUMMY_VOT) {
			throw std::runtime_error("Only IOU tracking is available in the Windows tracker build");
		}
		return std::make_shared<DummyVisualTracker>(configFilePath);
#else
		/// Parse the visual tracker configuration.
		m_params = VisualTrackerFactoryParams(configFilePath, "VisualTracker");

		switch (m_params.m_visualTrackerType)
		{
			case VISUAL_TRACKER_TYPES::NvDCF_VOT:
			{
				return std::make_shared<VisualTracker>(configFilePath);
			}
			case VISUAL_TRACKER_TYPES::NvDCF_VPI_VOT:
			{
				return std::make_shared<DCFTracker>(configFilePath);
			}
			case VISUAL_TRACKER_TYPES::DUMMY_VOT:
			default:
			{
				return std::make_shared<DummyVisualTracker>(configFilePath);
			}
		}
		return nullptr;
#endif
	}

protected:

    VisualTrackerFactoryParams m_params; //!< Config params for visual tracker module.
};

#endif // VISUALTRACKERLIB_VISUALTRACKER_FACTORY_HPP_
