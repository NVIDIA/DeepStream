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

#ifndef ESTIMATORLIB_ESTIMATOR_FACTORY2_HPP_
#define ESTIMATORLIB_ESTIMATOR_FACTORY2_HPP_

#include <memory>

#include "modules/NvMultiObjectTracker/NvTrackerParams.hpp"
#include "modules/Estimatorv2/IEstimator2.hpp"

// State estimators.
#include "modules/Estimatorv2/DummyEstimator2.hpp"
#include "modules/Estimatorv2/SimpleEstimator2.hpp"

/** @brief Config parser for factory. */
class EstimatorFactoryParams2 : public NvTrackerParams
{
public:
	/** @brief Initialize factory parameters with the default estimator type. */
	EstimatorFactoryParams2()
	{
		m_stateEstimatorType = STATE_ESTIMATOR_TYPES::DUMMY_ESTIMATOR;
	}

	/**
	 * @brief Initialize factory parameters from a configuration module.
	 * @param[in] configFilePath The path to a configuration file.
	 * @param[in] moduleName The configuration module name.
	 */
	EstimatorFactoryParams2(const std::string &configFilePath, const std::string moduleName)
	{
		YAML::Node configRoot = getConfigRoot(configFilePath);
		this->read(configRoot[moduleName]);
	}

	/**
	 * @brief Initialize factory parameters from a YAML node.
	 * @param[in] fn Parsed YAML structure of config to update.
	 */
	EstimatorFactoryParams2(const YAML::Node &fn)
	{
		read(fn);
	}

    /**
     * @brief Update factory parameters from a YAML node.
     * @param[in] fn Parsed YAML structure of config to update.
     */
    void read(const YAML::Node &fn) override
    {
		*this = EstimatorFactoryParams2();
        if(!fn.size()) { return;}

        for (YAML::const_iterator fit = fn.begin(); fit != fn.end(); ++fit)
        {
			std::string paramKey = fit->first.as<std::string>();

			if (paramKey == "stateEstimatorType")
			{
				int stateEstimatorTypeInt = (int)fit->second.as<float>();
				/// Reject unsupported state-estimator types.
				if (stateEstimatorTypeInt < 0 || stateEstimatorTypeInt > 4)
				{
					throw std::runtime_error(
						"Invalid stateEstimatorType value: " + std::to_string(stateEstimatorTypeInt) + ". Valid range is 0-4 (0=DUMMY_ESTIMATOR, 1=BBOX_ESTIMATOR_SIMPLE, 2=BBOX_ESTIMATOR, 3=LOCATION_ESTIMATOR_SIMPLE, 4=LOCATION_ESTIMATOR_PROJECTIVE)."
					);
				}
				m_stateEstimatorType = (STATE_ESTIMATOR_TYPES)stateEstimatorTypeInt;
			}
        }
    }

    STATE_ESTIMATOR_TYPES m_stateEstimatorType; //!< Selects the state-estimator type.
};

/** @brief Singleton factory for state estimator objects. */
class EstimatorFactory2
{
public:
    /** @brief Return the static instance reference to the factory. */
    static EstimatorFactory2& getInstance()
    {
        static EstimatorFactory2 instance;
        return instance;
    }

    /**
     * @brief Create the configured state-estimator implementation.
     * @param[in] configFilePath The path to a configuration file.
     */
    std::shared_ptr<IEstimator2> makeEstimator(const std::string configFilePath)
    {
		/// Parse config for state estimator module.
		m_params = EstimatorFactoryParams2(configFilePath, "StateEstimator");

		switch (m_params.m_stateEstimatorType)
		{
			case STATE_ESTIMATOR_TYPES::BBOX_ESTIMATOR:
			{
				return std::make_shared<SimpleEstimator2>(configFilePath, 8, 4);
			}
			case STATE_ESTIMATOR_TYPES::BBOX_ESTIMATOR_SIMPLE:
			{
				return std::make_shared<SimpleEstimator2>(configFilePath, 6, 4);
			}
			case STATE_ESTIMATOR_TYPES::LOCATION_ESTIMATOR_SIMPLE:
			{
				return std::make_shared<SimpleEstimator2>(configFilePath, 4, 2);
			}
			case STATE_ESTIMATOR_TYPES::LOCATION_ESTIMATOR_PROJECTIVE:
			{
				/// Same as LOCATION_ESTIMATOR_SIMPLE but uses covariance-based measurement noise.
				return std::make_shared<SimpleEstimator2>(configFilePath, 4, 2);
			}
			case STATE_ESTIMATOR_TYPES::DUMMY_ESTIMATOR:
			default:
			{
				return std::make_shared<DummyEstimator2>(configFilePath);
			}
		}
		return nullptr;
	}

protected:

    EstimatorFactoryParams2 m_params; //!< Stores configuration parameters for the state-estimation module.
};

#endif // ESTIMATORLIB_ESTIMATOR_FACTORY2_HPP_
