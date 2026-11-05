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
#ifndef NVTRACKERPARAMS_HPP_
#define NVTRACKERPARAMS_HPP_

#include <cerrno>
#include <cstdlib>
#include <limits.h>
#include <string>

#ifdef _WIN32
#include <filesystem>
namespace NvTrackerFilesystem = std::filesystem;
#else
#include <experimental/filesystem>
#endif
#include <yaml-cpp/yaml.h>
#include "modules/utility/debug_common.hpp"

/** @brief Provides common loading helpers for tracker configuration parameters. */
class NvTrackerParams
{
public:
	/** @brief Initialize the tracker parameter interface. */
	NvTrackerParams() {}

	/**
	 * @brief Read tracker parameters from a YAML node.
	 * @param[in] fn YAML node containing tracker configuration.
	 */
	virtual void read(const YAML::Node& fn) = 0;

	/**
	 * @brief Resolve a configured path relative to the tracker configuration file.
	 *
	 * Existing paths are canonicalized, while nonexistent relative paths remain joined to the configuration directory.
	 * @param[in] configFilePath Path to the tracker configuration file.
	 * @param[in] filePath Absolute or configuration-relative path to resolve.
	 * @return Resolved path, or the original path when the configuration path cannot be resolved.
	 */
	static std::string resolvePathRelativeToConfig(const std::string &configFilePath, const std::string &filePath)
	{
		if (filePath.empty() || configFilePath.empty())
		{
			return filePath;
		}

#ifdef _WIN32
		std::filesystem::path requestedPath(filePath);
		if (requestedPath.is_absolute()) {
			return requestedPath.string();
		}
		return (std::filesystem::absolute(std::filesystem::path(configFilePath)).parent_path() /
			requestedPath).string();
#else
		char resolved[PATH_MAX + 1];

		if (filePath[0] == '/')
		{
			if (realpath(filePath.c_str(), resolved) != nullptr)
			{
				return std::string(resolved);
			}
			if (errno == ENOENT)
			{
				return filePath;
			}
			return filePath;
		}

		char absCfg[PATH_MAX + 1];
		if (realpath(configFilePath.c_str(), absCfg) == nullptr)
		{
			return filePath;
		}

		std::string cfgDir(absCfg);
		const size_t last = cfgDir.find_last_of('/');
		if (last != std::string::npos)
		{
			cfgDir.resize(last + 1);
		}
		else
		{
			cfgDir.clear();
		}

		const std::string combined = cfgDir + filePath;
		if (realpath(combined.c_str(), resolved) != nullptr)
		{
			return std::string(resolved);
		}
		if (errno == ENOENT)
		{
			return combined;
		}
		return combined;
#endif
	}

	/**
	 * @brief Load the root YAML node from a tracker configuration file.
	 * @param[in] configFilePath Path to the tracker configuration file.
	 * @param[in] printMessage Enables warning output when the configuration cannot be loaded.
	 * @return Loaded YAML root node, or an empty node when loading fails.
	 */
	static const YAML::Node getConfigRoot(const std::string &configFilePath, const bool printMessage = false)
	{

		if(configFilePath.size() == 0)
		{
			if (printMessage)
			{
				std::cout << "[NvTrackerParams::getConfigRoot()] !!![WARNING] Empty config file path is provided. Will go ahead with default values" << std::endl;
			}
		}
		else if(!
#ifdef _WIN32
			std::filesystem::exists(configFilePath)
#else
			std::experimental::filesystem::exists(configFilePath)
#endif
		)
		{
			if (printMessage)
			{
				std::cout << "[NvTrackerParams::getConfigRoot()] !!![WARNING] File doesn't exist. Will go ahead with default values" << std::endl;
			}
		}
		else
		{
			try
			{
				YAML::Node configFn = YAML::LoadFile(configFilePath);

				if(configFn.size() > 0)
				{
					return configFn;
				}
				else
				{
					if (printMessage)
					{
						std::cout << "[NvTrackerParams::getConfigRoot()] !!![WARNING] Can't open config file (" << configFilePath << "). Will go ahead with default values" << std::endl;
					}
				}
			}
			catch (std::exception& e)
			{
				if (printMessage)
				{
					CLOG(
						std::string( "[NvTrackerParams::getConfigRoot()] !!![WARNING] Invalid low-level config file caused an exception, but will go ahead with the default config values" ).c_str()
					);
					CLOG(e.what());
				}
			}
		}

		return YAML::Node();
	}
};

/** @brief Reid model name format needed by NvTrackerParamsQuery and ReIDConfigParams classes. */
class ReIDModelNameParams {
public:
	/**
	 * @brief Check whether a model name has the form "model" followed by an integer.
	 * @param[in] name Model name to validate.
	 */
	static bool isValidName(const std::string& name)
	{
		if (name.size() <= 5)
		{
			return false;
		}
		else if (name.substr(0, 5) != "model")
		{
			return false;
		}
		else
		{
			for (uint ind = 5; ind < name.size(); ind++)
			{
				if (!std::isdigit(name.at(ind)) )
				{
					return false;
				}
			}
		}
		return true;
	}

	/**
	 * @brief Generate a model name from a numeric index.
	 * @param[in] index Numeric suffix for the generated model name.
	 */
	static std::string generateName(const int index)
	{
		return "model" + std::to_string(index);
	}
};

#endif // NVTRACKERPARAMS_HPP_
