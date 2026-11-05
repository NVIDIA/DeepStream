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

#ifndef DATAASSOCIATORLIB_DATAASSOCIATOR_FACTORY_HPP_
#define DATAASSOCIATORLIB_DATAASSOCIATOR_FACTORY_HPP_

#include <memory>
#include "modules/NvMultiObjectTracker/NvTrackerParams.hpp"
#include "modules/DataAssociator/IDataAssociator.hpp"
// Data associators.
#include "modules/DataAssociator/DataAssociator.hpp"

/** @brief Config parser for factory. */
class DataAssociatorFactoryParams : public NvTrackerParams
{
public:
	/** @brief Initialize factory parameters with the default associator type. */
	DataAssociatorFactoryParams()
	{
		m_dataAssociatorType = DATA_ASSOCIATOR_TYPES::DEFAULT_ASSOCIATOR;
	}

	/**
	 * @brief Initialize factory parameters from a configuration module.
	 * @param[in] configFilePath The path to a configuration file.
	 * @param[in] moduleName The configuration module name.
	 */
	DataAssociatorFactoryParams(const std::string &configFilePath, const std::string moduleName)
	{
		YAML::Node configRoot = getConfigRoot(configFilePath);
		this->read(configRoot[moduleName]);
	}

	/**
	 * @brief Initialize factory parameters from a YAML node.
	 * @param[in] fn Parsed YAML structure of config to update.
	 */
	DataAssociatorFactoryParams(const YAML::Node &fn)
	{
		read(fn);
	}

	/**
	 * @brief Update factory parameters from a YAML node.
	 * @param[in] fn Parsed YAML structure of config to update.
	 */
	void read(const YAML::Node &fn) override
	{
		*this = DataAssociatorFactoryParams();
		if(!fn.size())
		{
			return;
		}

		/// TODO: Validate the configured data-associator type.
		for (YAML::const_iterator fit = fn.begin(); fit != fn.end(); ++fit)
		{
			std::string paramKey = fit->first.as<std::string>();
			if (paramKey == "dataAssociatorType")
			{
				m_dataAssociatorType = (DATA_ASSOCIATOR_TYPES)((int)fit->second.as<float>());
			}
		}
	}

	DATA_ASSOCIATOR_TYPES m_dataAssociatorType; //!< The type of data associator.
};

/** @brief Singleton factory for data associator objects. */
class DataAssociatorFactory
{
public:
	/** @brief Return the static instance reference to the factory. */
	static DataAssociatorFactory& getInstance()
	{
		static DataAssociatorFactory instance;
		return instance;
	}

	/**
	 * @brief Create the configured data-associator implementation.
	 * @param[in] configFilePath The path to a configuration file.
	 */
	std::shared_ptr<IDataAssociator> makeDataAssociator(const std::string &configFilePath)
	{
		/// Parse the data-associator configuration module.
		m_params = DataAssociatorFactoryParams(configFilePath, "DataAssociator");

		switch (m_params.m_dataAssociatorType)
		{
			case DATA_ASSOCIATOR_TYPES::DEFAULT_ASSOCIATOR:
			default:
			{
				return std::make_shared<DataAssociator>(configFilePath);
			}
		}
		return nullptr;
	}

protected:

	DataAssociatorFactoryParams m_params; //!< Config params.
};

#endif // DATAASSOCIATORLIB_DATAASSOCIATOR_FACTORY_HPP_
