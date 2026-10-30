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

#ifndef SEGMENTERLIB_SEGMENTER_FACTORY_HPP_
#define SEGMENTERLIB_SEGMENTER_FACTORY_HPP_

#include <memory>
#include "modules/NvMultiObjectTracker/NvTrackerParams.hpp"
#include "modules/Segmenter/ISegmenter.hpp"
#include "modules/Segmenter/DummySegmenter.hpp"
#ifndef _WIN32
#include "modules/Segmenter/Segmenter.hpp"
#endif

/** @brief Stores the implementation type selected by the factory. */
class SegmenterFactoryParams : public NvTrackerParams
{
public:
	/** @brief Construct segmenter factory parameters. */
	SegmenterFactoryParams()
	{
		m_segmenterType = SEGMENTER_TYPES::DUMMY_SEGMENTER;
	}

	/**
	 * @brief Construct a SegmenterFactoryParams instance.
	 * @param[in] configFilePath Path to the tracker configuration file.
	 * @param[in] moduleName Configuration module name.
	 */
	SegmenterFactoryParams(const std::string &configFilePath, const std::string moduleName)
	{
		YAML::Node configRoot = getConfigRoot(configFilePath);
		this->read(configRoot[moduleName]);
	}

	/**
	 * @brief Construct a SegmenterFactoryParams instance.
	 * @param[in] fn Parsed YAML node.
	 */
	SegmenterFactoryParams(const YAML::Node &fn)
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
		*this = SegmenterFactoryParams();
		if(!fn.size())
		{
			return;
		}

		for (YAML::const_iterator fit = fn.begin(); fit != fn.end(); ++fit)
		{
			std::string paramKey = fit->first.as<std::string>();
			if (paramKey == "segmenterType")
			{
				int segmenterTypeInt = (int)fit->second.as<float>();
				/// Validate the segmenter type (valid range: 0-3) SEGMENTER_TYPES: DUMMY_SEGMENTER=0, SAM2=1, NanoSAM=2, EfficientVitSAM=3.
				if (segmenterTypeInt < 0 || segmenterTypeInt > 3)
				{
					throw std::runtime_error(
						"Invalid segmenterType value: " + std::to_string(segmenterTypeInt) + ". Valid range is 0-3 (0=DUMMY_SEGMENTER, 1=SAM2, 2=NanoSAM, 3=EfficientVitSAM)."
					);
				}
				m_segmenterType = (SEGMENTER_TYPES)segmenterTypeInt;
			}
		}
	}

	SEGMENTER_TYPES m_segmenterType; //!< The implementation to construct.
};

/** @brief Singleton factory that constructs the configured implementation. */
class SegmenterFactory
{
public:
	/** @brief Get the singleton factory instance. */
	static SegmenterFactory& getInstance()
	{
		static SegmenterFactory instance;
		return instance;
	}

	/**
	 * @brief Construct the configured segmenter implementation.
	 * @param[in] configFilePath Path to the tracker configuration file.
	 * @return The segmenter implementation selected by the configuration.
	 */
	std::shared_ptr<ISegmenter> makeSegmenter(const std::string &configFilePath)
	{
#ifdef _WIN32
		m_params = SegmenterFactoryParams(configFilePath, "Segmenter");
		if (m_params.m_segmenterType != SEGMENTER_TYPES::DUMMY_SEGMENTER) {
			throw std::runtime_error("Segmentation is not available in the Windows IOU-only tracker build");
		}
		return std::make_shared<DummySegmenter>(configFilePath);
#else
		/// Parse the module config.
		m_params = SegmenterFactoryParams(configFilePath, "Segmenter");

		switch (m_params.m_segmenterType)
		{
			case SEGMENTER_TYPES::NanoSAM:
			case SEGMENTER_TYPES::EfficientVitSAM:
			case SEGMENTER_TYPES::SAM2:
			{
				return std::make_shared<Segmenter>(configFilePath);
			}
			case SEGMENTER_TYPES::DUMMY_SEGMENTER:
			default:
			{
				return std::make_shared<DummySegmenter>(configFilePath);
			}
		}
		return nullptr;
#endif
	}

protected:
	SegmenterFactoryParams m_params; //!< Config params.
};

#endif // SEGMENTERLIB_SEGMENTER_FACTORY_HPP_
