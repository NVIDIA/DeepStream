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

#ifndef ESTIMATORLIB_IESTIMATOR2_HPP_
#define ESTIMATORLIB_IESTIMATOR2_HPP_

#include <vector>
#include <tuple>

#include <Eigen/Dense>

#include "modules/Estimatorv2/Estimator_common.hpp"

/** @brief Define the interface for state estimators. */
class IEstimator2
{
public:
    /** @brief Destroy the state-estimator interface. */
    virtual ~IEstimator2() {}

    /**
     * @brief Update without any measurement, using the prediction only.
     * @param[in] dT The time step.
     * @return The estimated states.
     */
    virtual Eigen::VectorXd update(const float dT = 1) = 0;

    /**
     * @brief Given a set of measurements and their covariances, update the estimator and returns the estimated states.
     * @param[in] measurements A set of (measurement, covariance, isFromDetector) tuples. Pass a negative identity covariance to select the default measurement noise. A positive covariance(0,0) uses the provided noise.
     * @param[in] dT The time step.
     * @return The estimated states.
     */
    virtual Eigen::VectorXd update(
        const std::vector<std::tuple<Eigen::VectorXd, Eigen::Matrix2d, bool> > &measurements, const float dT = 1
    ) = 0;

    /** @brief Reset the states and make the age to be zero. */
    virtual void reset() = 0;

    /** @brief Return the current estimated states. */
    virtual Eigen::VectorXd getStates() const = 0;

    /**
     * @brief Predict states without modifying the current estimate.
     * @param[in] dT The time step.
     */
    virtual Eigen::VectorXd getPredictedStates(const float dT = 1) = 0;

    /** @brief Return the measurement-noise covariance matrix. */
    virtual Eigen::MatrixXd getMeasurementNoiseCovariance() const = 0;

    /** @brief Return the current states and their covariance. */
    virtual std::tuple<Eigen::VectorXd, Eigen::MatrixXd> getStatesWithCov() const = 0;

    /**
     * @brief Predict states and covariance without modifying the current estimate.
     * @param[in] dT The time step.
     */
    virtual std::tuple<Eigen::VectorXd, Eigen::MatrixXd> getPredictedStatesWithCov(const float dT = 1) = 0;

    /**
     * @brief Predict measurements and covariance without modifying the current estimate.
     * @param[in] dT The time step.
     */
    virtual std::tuple<Eigen::VectorXd, Eigen::MatrixXd> getPredictedMeasurementsWithCov(const float dT = 1) = 0;

    /** @brief Return the number of estimator updates. */
    virtual uint32_t getAge() const = 0;

    /** @brief Return the configured state-estimator type. */
    virtual const STATE_ESTIMATOR_TYPES getStateEstimatorType() = 0;

    /** @brief Return whether the estimator represents size using aspect ratio. */
    virtual const bool useAspectRatio() const = 0;

    /**
     * @brief Overwrite the position and velocity components.
     * @param[in] position The replacement position.
     * @param[in] velocity The replacement velocity.
     */
    virtual void overrideState(const Eigen::Vector2d& position, const Eigen::Vector2d& velocity) = 0;
};

#endif // ESTIMATORLIB_IESTIMATOR2_HPP_
