/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef POSEESTIMATORLIB_POSEESTIMATOR_FACTORY_HPP_
#define POSEESTIMATORLIB_POSEESTIMATOR_FACTORY_HPP_

#include <memory>
#include "modules/NvMultiObjectTracker/NvTrackerParams.hpp"
#include "modules/PoseEstimator/IPoseEstimator.hpp"
#ifndef _WIN32
#include "modules/PoseEstimator/PoseInference.hpp"
#endif

/** @brief Stores the implementation type selected by the factory. */
class PoseEstimatorFactoryParams : public NvTrackerParams
{
public:
	/** @brief Construct pose-estimator factory parameters. */
	PoseEstimatorFactoryParams()
	{
		m_poseEstimatorType = POSEESTIMATOR_TYPES::DUMMY_POSE;
	}

	/**
	 * @brief Construct a PoseEstimatorFactoryParams instance.
	 * @param[in] configFilePath Path to the tracker configuration file.
	 * @param[in] moduleName Configuration module name.
	 */
	PoseEstimatorFactoryParams(const std::string &configFilePath, const std::string moduleName)
	{
		YAML::Node configRoot = getConfigRoot(configFilePath);
		this->read(configRoot[moduleName]);
	}

	/**
	 * @brief Construct a PoseEstimatorFactoryParams instance.
	 * @param[in] fn Parsed YAML node.
	 */
	PoseEstimatorFactoryParams(const YAML::Node &fn)
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
		*this = PoseEstimatorFactoryParams();
		if(!fn.size())
		{
			return;
		}

		for (YAML::const_iterator fit = fn.begin(); fit != fn.end(); ++fit)
		{
			std::string paramKey = fit->first.as<std::string>();
			if (paramKey == "poseEstimatorType")
			{
				int poseEstimatorTypeInt = (int)fit->second.as<float>();
				/// Validate the pose estimator type in the range 0 through 2.
				if (poseEstimatorTypeInt < 0 || poseEstimatorTypeInt > 2)
				{
					throw std::runtime_error(
						"Invalid poseEstimatorType value: " + std::to_string(poseEstimatorTypeInt) + ". Valid range is 0-2 (0=DUMMY_POSE, 1=POSE_INFERENCE, 2=POSE_TRACKER)."
					);
				}
				m_poseEstimatorType = (POSEESTIMATOR_TYPES)poseEstimatorTypeInt;
			}
		}
	}

	POSEESTIMATOR_TYPES m_poseEstimatorType; //!< The implementation to construct.
};

/** @brief Singleton factory that constructs the configured implementation. */
class PoseEstimatorFactory
{
public:
	/** @brief Get the singleton factory instance. */
	static PoseEstimatorFactory& getInstance()
	{
		static PoseEstimatorFactory instance;
		return instance;
	}

	/**
	 * @brief Construct the configured pose-estimator implementation.
	 * @param[in] configFilePath Path to the tracker configuration file.
	 * @return Shared pointer to the pose estimator, or null when pose estimation is disabled.
	 */
	std::shared_ptr<IPoseEstimator> makePoseEstimator(const std::string &configFilePath)
	{
#ifdef _WIN32
		m_params = PoseEstimatorFactoryParams(configFilePath, "PoseEstimator");
		if (m_params.m_poseEstimatorType != POSEESTIMATOR_TYPES::DUMMY_POSE) {
			throw std::runtime_error("Pose estimation is not available in the Windows IOU-only tracker build");
		}
		return nullptr;
#else
		/// Parse the module config.
		m_params = PoseEstimatorFactoryParams(configFilePath, "PoseEstimator");

		switch (m_params.m_poseEstimatorType)
		{
			case POSEESTIMATOR_TYPES::POSE_INFERENCE:
			case POSEESTIMATOR_TYPES::POSE_TRACKER:
			{
				return std::make_shared<PoseInference>(configFilePath);
			}
			case POSEESTIMATOR_TYPES::DUMMY_POSE:
			default:
			{
				return nullptr;
			}
		}
		return nullptr;
#endif
	}

protected:

	PoseEstimatorFactoryParams m_params; //!< Config params.
};

#endif // POSEESTIMATORLIB_POSEESTIMATOR_FACTORY_HPP_
