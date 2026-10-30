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

#ifndef MULTIVIEWASSOCIATORLIB_COMMUNICATOR_FACTORY_HPP_
#define MULTIVIEWASSOCIATORLIB_COMMUNICATOR_FACTORY_HPP_

// Communicator implementations.
#include "modules/MultiViewAssociator/DummyCommunicator.hpp"
#include "modules/MultiViewAssociator/MemoryAccessCommunicator.hpp"
#include "modules/MultiViewAssociator/MqttCommunicator.hpp"

/** @brief Store communicator-factory configuration parameters. */
class CommunicatorFactoryParams : public NvTrackerParams
{
  public:
	/** @brief Initialize factory parameters with the dummy communicator type. */
	CommunicatorFactoryParams()
	{
		m_communicatorType = DUMMY_COMMUNICATOR;
	}

	/**
	 * @brief Initialize factory parameters from a configuration module.
	 * @param[in] configFilePath The path to a configuration file.
	 * @param[in] moduleName The configuration module name.
	 */
	CommunicatorFactoryParams(const std::string &configFilePath, const std::string moduleName)
	{
		YAML::Node configRoot = getConfigRoot(configFilePath);
		this->read(configRoot[moduleName]);
	}

	/**
	 * @brief Initialize factory parameters from a YAML node.
	 * @param[in] fn Parsed YAML structure of config to update.
	 */
	CommunicatorFactoryParams(const YAML::Node &fn)
	{
		read(fn);
	}

    /**
     * @brief Update factory parameters from a YAML node.
     * @param[in] fn Parsed YAML structure of config to update.
     */
    void read(const YAML::Node &fn) override
    {
		*this = CommunicatorFactoryParams();
		if(!fn.size()) { return;}
		for (YAML::const_iterator fit = fn.begin(); fit != fn.end(); ++fit)
		{
			std::string paramKey = fit->first.as<std::string>();
			if (paramKey == "communicatorType")
			{
				m_communicatorType = (COMMUNICATOR_TYPES)((int)fit->second.as<float>());
			}
		}
	}

	COMMUNICATOR_TYPES m_communicatorType; //!< Selects the communicator implementation.
};

/** @brief Singleton factory for communicator objects. */
class CommunicatorFactory
{
public:
    /** @brief Return the static instance reference to the factory. */
    static CommunicatorFactory& getInstance()
	{
		static CommunicatorFactory instance;
		return instance;
	}

    /**
     * @brief Create the configured communicator implementation.
     * @param[in] configFilePath The path to a configuration file.
     */
    std::shared_ptr<ICommunicator> makeCommunicator(const std::string configFilePath)
	{
		m_params = CommunicatorFactoryParams(configFilePath, "Communicator");
		switch (m_params.m_communicatorType)
		{
			case DUMMY_COMMUNICATOR:
			{
				return std::make_shared<DummyCommunicator>(configFilePath);
			}
			case MEMORY_ACCESS_COMMUNICATOR:
			{
				return std::make_shared<MemoryAccessCommunicator>(configFilePath);
			}
			case MQTT_COMMUNICATOR:
				return std::make_shared<MqttCommunicator>(configFilePath);
			default:
			{
				std::cerr << "WARNING: Invalid communicator type. Using DummyCommunicator\n";
				return std::make_shared<DummyCommunicator>(configFilePath);
			}
		}
	}

private:
	CommunicatorFactoryParams m_params; //!< Stores communicator-factory parameters.
};

#endif // MULTIVIEWASSOCIATORLIB_COMMUNICATOR_FACTORY_HPP_
