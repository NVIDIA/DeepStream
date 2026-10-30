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

#ifndef ESTIMATORLIB_DUMMYESTIMATOR2_HPP_
#define ESTIMATORLIB_DUMMYESTIMATOR2_HPP_

#include <memory>

#include "modules/Estimatorv2/IEstimator2.hpp"
#include "modules/NvMultiObjectTracker/NvTrackerParams.hpp"

/** @brief Config params for the dummy state estimator. */
class DummyEstimatorParams2 : public NvTrackerParams
{
public:
	/** @brief Initialize dummy-estimator parameters with default values. */
	DummyEstimatorParams2();

	/**
	 * @brief Initialize dummy-estimator parameters from a configuration module.
	 * @param[in] configFilePath The path to a configuration file.
	 * @param[in] moduleName The configuration module name.
	 */
	DummyEstimatorParams2(const std::string &configFilePath, const std::string moduleName);

	/**
	 * @brief Initialize dummy-estimator parameters from a YAML node.
	 * @param[in] fn Parsed YAML structure of config to update.
	 */
	DummyEstimatorParams2(const YAML::Node &fn);

    /**
     * @brief Update dummy-estimator parameters from a YAML node.
     * @param[in] fn Parsed YAML structure of config to update.
     */
    void read(const YAML::Node &fn) override;

	int m_debugVerboseLevel; //!< Controls debug message verbosity.
    STATE_ESTIMATOR_TYPES m_stateEstimatorType; //!< Selects the state-estimator type.
};

/** @brief Dummy state estimator whose prediction is identical to the last known states. */
class DummyEstimator2 : public IEstimator2
{
public:
    /**
     * @brief Construct a dummy state estimator from configuration.
     * @param[in] configFilePath The path to a config file.
     * @param[in] numTotalStates The # of total states to be estimated.
     * @param[in] numObservStates The # of observation (i.e., measurement) states.
     */
    DummyEstimator2(const std::string &configFilePath, const int numTotalStates = 4, const int numObservStates = 4);

    /** @brief Destroy the dummy state estimator. */
    ~DummyEstimator2();

    /** @brief Reset the states and make the age to be zero. */
    void reset() override;

    /**
     * @brief Update without any measurement, using the prediction only.
     * @param[in] dT The time step.
     * @return The estimated states.
     */
    Eigen::VectorXd update(const float dT = 1) override;

    /**
     * @brief Given a set of measurements and their covariances, update the estimator and returns the estimated states.
     * @param[in] measurements A set of (measurement, covariance, isFromDetector) tuples. Pass a negative identity covariance to select the default measurement noise. A positive covariance(0,0) uses the provided noise.
     * @param[in] dT The time step.
     * @return The estimated states.
     */
    Eigen::VectorXd update(
        const std::vector<std::tuple<Eigen::VectorXd, Eigen::Matrix2d, bool> > &measurements, const float dT = 1
    ) override;

    /** @brief Return the current estimated states. */
    Eigen::VectorXd getStates() const override { return m_X; }

    /**
     * @brief Predict states without modifying the current estimate.
     * @param[in] dT The time step.
     */
    Eigen::VectorXd getPredictedStates(const float dT = 1) override { return m_X; }

    /** @brief Return the measurement-noise covariance matrix. */
    Eigen::MatrixXd getMeasurementNoiseCovariance() const override { return m_P.block<2,2>(2,2); }

    /** @brief Return the current states and their covariance. */
    std::tuple<Eigen::VectorXd, Eigen::MatrixXd> getStatesWithCov() const override { return {m_X, m_P}; }

    /**
     * @brief Predict states and covariance without modifying the current estimate.
     * @param[in] dT The time step.
     */
    std::tuple<Eigen::VectorXd, Eigen::MatrixXd> getPredictedStatesWithCov(const float dT = 1) override
    {
        return {m_X, m_P};
    }

    /**
     * @brief Predict measurements and covariance without modifying the current estimate.
     * @param[in] dT The time step.
     */
    std::tuple<Eigen::VectorXd, Eigen::MatrixXd> getPredictedMeasurementsWithCov(const float dT = 1) override
    {
        return {m_X, m_P};
    }

    /** @brief Return the number of estimator updates. */
    uint32_t getAge() const override { return m_age; }

    /** @brief Return the configured state-estimator type. */
    const STATE_ESTIMATOR_TYPES getStateEstimatorType() override { return m_params.m_stateEstimatorType; }

    /** @brief Return whether the estimator represents size using aspect ratio. */
    const bool useAspectRatio() const override { return false; }

    /**
     * @brief Overwrite the position and velocity components.
     * @param[in] position The replacement position.
     * @param[in] velocity The replacement velocity.
     */
    void overrideState(const Eigen::Vector2d& position, const Eigen::Vector2d& velocity) override
    {
        m_X.head<2>() = position; m_X.segment<2>(2) = velocity;
    }
protected:

    DummyEstimatorParams2 m_params; //!< Stores configuration parameters for the state-estimation module.
    const int m_numTotalStates; //!< Specifies the total number of estimated states.
    const int m_numObservStates; //!< Specifies the number of observed states.

    // [Estimator state]
    Eigen::VectorXd m_X; //!< Estimated states.
    Eigen::MatrixXd m_P; //!< Estimate error covariance.
    uint32_t m_age; //!< Counts estimator updates.

    /** @brief Initialize all the target states. */
    void initStates();
};

#endif // ESTIMATORLIB_DUMMYESTIMATOR2_HPP_
