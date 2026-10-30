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

#include "modules/Estimatorv2/DummyEstimator2.hpp"

using std::cout;
using std::endl;

/**
 * @brief Construct a dummy state estimator from configuration.
 * @param[in] configFilePath The path to a config file.
 * @param[in] numTotalStates The # of total states to be estimated.
 * @param[in] numObservStates The # of observation (i.e., measurement) states.
 */
DummyEstimator2::DummyEstimator2(const std::string &configFilePath, const int numTotalStates, const int numObservStates)
    : m_numTotalStates(numTotalStates),
      m_numObservStates(numObservStates),
      m_X(m_numTotalStates),
      m_P(m_numTotalStates, m_numTotalStates)
{
	m_params = DummyEstimatorParams2(configFilePath, "StateEstimator");

	m_age = 0;
}

/** @brief Destroy the dummy state estimator. */
DummyEstimator2::~DummyEstimator2()
{
}

/** @brief Reset the states and make the age to be zero. */
void DummyEstimator2::reset()
{
	m_age = 0;
}

/** @brief Initialize the estimator covariance. */
void DummyEstimator2::initStates()
{
	/// Priori Error Estimate Covariance Matrix P'(k).
	m_P.setIdentity();
}

/**
 * @brief Update without any measurement, using the prediction only.
 * @param[in] dT The time step.
 * @return The estimated states.
 */
Eigen::VectorXd DummyEstimator2::update(const float dT)
{
	if(m_age == 0)
	{
#ifdef _WIN32
		throw std::runtime_error("[DummyEstimator2::update()] Cannot do the very first state updates without measurement !!");
#else
		std::runtime_error("[DummyEstimator2::update()] Cannot do the very first state updates without measurement !!");
#endif
	}

	m_age++;

	return m_X;
}

/**
 * @brief Update the estimator from measurements and their covariances.
 * @param[in] measList A set of (measurement, covariance, isFromDetector) tuples. Pass a negative identity covariance to select the default measurement noise. A positive covariance(0,0) uses the provided noise.
 * @param[in] dT The time step.
 * @return The estimated states.
 */
Eigen::VectorXd DummyEstimator2::update(
	const std::vector<std::tuple<Eigen::VectorXd, Eigen::Matrix2d, bool> > &measList, const float dT
)
{
	if(measList.size() == 0)
	{
		/// If there's no measurement, then make prediction only without correction step.
		return update(dT);
	}
	else
	{
		for(uint i=0; i<measList.size(); i++)
		{
			const Eigen::VectorXd &meas = std::get<0>(measList[i]);

			if(m_age == 0 && i == 0)
			{
				initStates();
			}

			int j=0;
			for(; j<m_numObservStates; j++)
			{
				m_X(j) = meas(j);
			}
			for(; j<m_numTotalStates; j++)
			{
				m_X(j) = 0;
			}
		}

		m_age++;

		return m_X;
	}
}

/** @brief Initialize dummy-estimator parameters with default values. */
DummyEstimatorParams2::DummyEstimatorParams2()
{
	m_debugVerboseLevel = 0;
}

/**
 * @brief Initialize dummy-estimator parameters from a configuration module.
 * @param[in] configFilePath The path to a configuration file.
 * @param[in] moduleName The configuration module name.
 */
DummyEstimatorParams2::DummyEstimatorParams2(const std::string &configFilePath, const std::string moduleName)
{
	YAML::Node configRoot = getConfigRoot(configFilePath);
	this->read(configRoot[moduleName]);
}

/**
 * @brief Initialize dummy-estimator parameters from a YAML node.
 * @param[in] fn Parsed YAML structure of config to update.
 */
DummyEstimatorParams2::DummyEstimatorParams2(const YAML::Node &fn)
{
	read(fn);
}

/**
 * @brief Update dummy-estimator parameters from a YAML node.
 * @param[in] fn Parsed YAML structure of config to update.
 */
void DummyEstimatorParams2::read(const YAML::Node &fn)
{
	*this = DummyEstimatorParams2();

    if(!fn.size())
    {
		return;
    }

    for (YAML::const_iterator fit = fn.begin(); fit != fn.end(); ++fit)
    {
		std::string paramKey = fit->first.as<std::string>();

		if (paramKey == "debugVerboseLevel")
		{
			m_debugVerboseLevel = (int)fit->second.as<float>();
		}
		else if (paramKey == "stateEstimatorType")
		{
			m_stateEstimatorType = (STATE_ESTIMATOR_TYPES)((int)fit->second.as<float>());
		}
    }
}
