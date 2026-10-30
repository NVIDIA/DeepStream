/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef REIDSERVICE_FACTORY_HPP_
#define REIDSERVICE_FACTORY_HPP_

#include <memory>
#include "modules/NvMultiObjectTracker/NvTrackerParams.hpp"
#include "modules/ReIDService/IReIDService.hpp"
#include "modules/ReIDService/DummyReIDService.hpp"
#ifndef _WIN32
#include "modules/ReIDService/ReIDService.hpp"
#endif

/** @brief Stores the implementation type selected by the factory. */
class ReIDServiceFactoryParams : public NvTrackerParams
{
public:
	/** @brief Construct ReID-service factory parameters. */
	ReIDServiceFactoryParams()
	{
		m_reidServiceType = REID_SERVICE_TYPES::DUMMY_REID_SERVICE;
	}

	/**
	 * @brief Construct a ReIDServiceFactoryParams instance.
	 * @param[in] configFilePath Path to the tracker configuration file.
	 * @param[in] moduleName Configuration module name.
	 */
	ReIDServiceFactoryParams(const std::string &configFilePath, const std::string moduleName)
	{
		YAML::Node configRoot = getConfigRoot(configFilePath);
		this->read(configRoot[moduleName]);
	}

	/**
	 * @brief Construct a ReIDServiceFactoryParams instance.
	 * @param[in] fn Parsed YAML node.
	 */
	ReIDServiceFactoryParams(const YAML::Node &fn)
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
		*this = ReIDServiceFactoryParams();
		if(!fn.size())
		{
			return;
		}

		/// TODO: Validate the type.
		for (YAML::const_iterator fit = fn.begin(); fit != fn.end(); ++fit)
		{
			std::string paramKey = fit->first.as<std::string>();
			if (paramKey == "reidServiceType")
			{
				m_reidServiceType = (REID_SERVICE_TYPES)((int)fit->second.as<float>());
			}
		}
	}

	REID_SERVICE_TYPES m_reidServiceType; //!< The type of reid service.
};

/** @brief Singleton factory that constructs the configured implementation. */
class ReIDServiceFactory
{
public:
	/** @brief Get the singleton factory instance. */
	static ReIDServiceFactory& getInstance()
	{
		static ReIDServiceFactory instance;
		return instance;
	}

	/**
	 * @brief Construct the configured ReID service implementation.
	 * @param[in] configFilePath Path to the tracker configuration file.
	 */
	std::shared_ptr<IReIDService> makeReIDService(const std::string &configFilePath)
	{
#ifdef _WIN32
		m_params = ReIDServiceFactoryParams(configFilePath, "ReIDService");
		if (m_params.m_reidServiceType != REID_SERVICE_TYPES::DUMMY_REID_SERVICE) {
			throw std::runtime_error("ReID service is not available in the Windows IOU-only tracker build");
		}
		return std::make_shared<DummyReIDService>(configFilePath);
#else
		/// Parse the module config.
		m_params = ReIDServiceFactoryParams(configFilePath, "ReIDService");

		switch (m_params.m_reidServiceType)
		{
			case REID_SERVICE_TYPES::REID_SERVICE:
			{
				return std::make_shared<ReIDService>(configFilePath);
			}
			case REID_SERVICE_TYPES::DUMMY_REID_SERVICE:
			{
				return std::make_shared<DummyReIDService>(configFilePath);
			}
			default:
			{
				std::cerr << "WARNING: Invalid ReIDService type: " << m_params.m_reidServiceType << std::endl;
				return std::make_shared<DummyReIDService>(configFilePath);
			}
		}
		return nullptr;
#endif
	}

protected:
	ReIDServiceFactoryParams m_params; //!< Config params.
};

#endif // REIDSERVICE_FACTORY_HPP_
