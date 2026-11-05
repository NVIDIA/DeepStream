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

#ifndef ESTIMATORLIB_SIMPLEESTIMATOR2_HPP_
#define ESTIMATORLIB_SIMPLEESTIMATOR2_HPP_

#include <cassert>
#include <memory>

#include "modules/Estimatorv2/IEstimator2.hpp"
#include "modules/NvMultiObjectTracker/NvTrackerParams.hpp"

/** @brief Config params for the simple state estimator. */
class SimpleEstimatorParams2 : public NvTrackerParams
{
public:
	/** @brief Initialize simple-estimator parameters with default values. */
	SimpleEstimatorParams2();

	/**
	 * @brief Initialize simple-estimator parameters from a configuration module.
	 * @param[in] configFilePath The path to a configuration file.
	 * @param[in] moduleName The configuration module name.
	 */
	SimpleEstimatorParams2(const std::string &configFilePath, const std::string moduleName);

	/**
	 * @brief Initialize simple-estimator parameters from a YAML node.
	 * @param[in] fn Parsed YAML structure of config to update.
	 */
	SimpleEstimatorParams2(const YAML::Node &fn);

    /**
     * @brief Update simple-estimator parameters from a YAML node.
     * @param[in] fn Parsed YAML structure of config to update.
     */
    void read(const YAML::Node &fn) override;

	int m_debugVerboseLevel; //!< Controls debug message verbosity.

    // [Estimator configuration]
    STATE_ESTIMATOR_TYPES m_stateEstimatorType; //!< Selects the state-estimator type.
    float m_measurementNoiseVar4Detector; //!< Specifies measurement-noise variance for detector observations.
    float m_measurementNoiseVar4Tracker; //!< Specifies measurement-noise variance for tracker localization.
    float m_measurementNoiseVar4Aspc; //!< Specifies measurement-noise variance for aspect ratio.
    float m_processNoiseVar4Loc; //!< Specifies process-noise variance for the bounding-box center.
    float m_processNoiseVar4Size; //!< Specifies process-noise variance for bounding-box size.
    float m_processNoiseVar4Vel; //!< Specifies process-noise variance for velocity.
    float m_processNoiseVar4Aspc; //!< Specifies process-noise variance for aspect ratio.
    float m_processNoiseVar4AspcVel; //!< Specifies process-noise variance for the aspect-ratio change rate.
    float m_noiseWeightVar4Loc; //!< Specifies the noise-variance weight for the bounding-box center.
    float m_noiseWeightVar4Vel; //!< Noise variance weight for velocity.
    float m_highUncertainty; //!< Specifies initial uncertainty for faster adaptation.
    bool m_useAspectRatio; //!< Indicates whether the estimator uses aspect ratio instead of width.
};

/** @brief Simple state estimator using a Kalman filter with a constant velocity model. */
class SimpleEstimator2 : public IEstimator2
{
public:
    /**
     * @brief Construct a Kalman-filter state estimator from configuration.
     * @param[in] configFilePath The path to a config file.
     * @param[in] numTotalStates The # of total states to be estimated.
     * @param[in] numObservStates The # of observation (i.e., measurement) states.
     * @param[in] processNoiseScale Scale applied to the location, size, and velocity process-noise variances.
     * @param[in] measurementNoiseScale Scale applied to the detector and tracker measurement-noise variances.
     */
    SimpleEstimator2(
        const std::string &configFilePath, const int numTotalStates = 6, const int numObservStates = 4,
        const float processNoiseScale=1.0, const float measurementNoiseScale=1.0
    );

    /** @brief Destroy the simple state estimator. */
    ~SimpleEstimator2();

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
    Eigen::VectorXd getPredictedStates(const float dT = 1) override;

    /** @brief Return the measurement-noise covariance matrix. */
    Eigen::MatrixXd getMeasurementNoiseCovariance() const override { return m_R; }

    /** @brief Return the current states and their covariance. */
    std::tuple<Eigen::VectorXd, Eigen::MatrixXd> getStatesWithCov() const override { return {m_X, m_P}; }

    /**
     * @brief Predict states and covariance without modifying the current estimate.
     * @param[in] dT The time step.
     */
    std::tuple<Eigen::VectorXd, Eigen::MatrixXd> getPredictedStatesWithCov(const float dT = 1) override;

    /**
     * @brief Predict measurements and covariance without modifying the current estimate.
     * @param[in] dT The time step.
     */
    std::tuple<Eigen::VectorXd, Eigen::MatrixXd> getPredictedMeasurementsWithCov(const float dT = 1) override;

    /**
     * @brief Overwrite the position and velocity components for a location estimator.
     * @param[in] position The replacement position.
     * @param[in] velocity The replacement velocity.
     */
    void overrideState(const Eigen::Vector2d& position, const Eigen::Vector2d& velocity) override
    {
        assert(
            m_params.m_stateEstimatorType == STATE_ESTIMATOR_TYPES::LOCATION_ESTIMATOR_SIMPLE ||
            m_params.m_stateEstimatorType == STATE_ESTIMATOR_TYPES::LOCATION_ESTIMATOR_PROJECTIVE
        );
        m_X.head<2>() = position; m_X.segment<2>(2) = velocity;
        m_X_pred.head<2>() = position; m_X_pred.segment<2>(2) = velocity;
    }

    /** @brief Return the number of estimator updates. */
    uint32_t getAge() const override { return m_age; }

    /** @brief Return whether the estimator represents size using aspect ratio. */
    const bool useAspectRatio() const override { return m_params.m_useAspectRatio; };

    /** @brief Return the configured state-estimator type. */
    const STATE_ESTIMATOR_TYPES getStateEstimatorType() override { return m_params.m_stateEstimatorType; }

protected:

    SimpleEstimatorParams2 m_params; //!< Stores configuration parameters for the state-estimation module.
    const int m_numTotalStates; //!< Specifies the total number of estimated states.
    const int m_numObservStates; //!< Specifies the number of observed states.
    Eigen::VectorXd m_X, m_X_pred; //!< Estimated and predicted states.
    Eigen::MatrixXd m_A; //!< Transition matrix.
    Eigen::MatrixXd m_Q; //!< Process noise covariance matrix.
    Eigen::MatrixXd m_P, m_P_pred; //!< Estimate and prediction error covariance.
    Eigen::MatrixXd m_H; //!< Measurement matrix.
    Eigen::MatrixXd m_R; //!< Measurement noise covariance matrix.
    Eigen::MatrixXd m_I; //!< Identity matrix.
    Eigen::MatrixXd m_K; //!< Kalman gain.
    uint32_t m_age; //!< Counts estimator updates.
    bool m_useDynamicCov; //!< Indicates whether measurements provide dynamic covariance.
    Eigen::IOFormat m_CommaInitFmt; //!< Eigen output format used for debug.

    /** @brief Options to set constant measurement noise. */
    enum MEASUREMENT_NOISE_OPTION
    {
        UNCHANGED, //!< Keeps the current measurement noise.
        DETECTOR, //!< Uses detector measurement noise.
        TRACKER //!< Uses tracker measurement noise.
    };

    /**
     * @brief Initialize the target state from a measurement.
     * @param[in] meas The initial measurement.
     */
    void initStates(const Eigen::VectorXd &meas);

    /**
     * @brief Set the transition matrix based on the input time interval.
     * @param[in] dT The prediction time interval.
     */
    void setTransitionMatrix(const float dT = 1);

    /**
     * @brief Set measurement noise matrix.
     * @param[in] meas Input measurement used when noise is proportional to box size.
     * @param[in] option Measurement-noise selection.
     */
    void setMeasurementNoiseMatrix(
        const Eigen::VectorXd &meas, const MEASUREMENT_NOISE_OPTION option=MEASUREMENT_NOISE_OPTION::UNCHANGED
    );

    /**
     * @brief Set process noise matrix.
     * @param[in] meas Input measurement used when noise is proportional to box size.
     * @param[in] isInit Whether the process-noise matrix is being initialized.
     */
    void setProcessNoiseMatrix(const Eigen::VectorXd &meas, const bool isInit=false);

    /**
     * @brief Predict the states for a time interval.
     * @param[in] dT The time step.
     */
    void predict(const float dT = 1);
};

#endif // ESTIMATORLIB_SIMPLEESTIMATOR2_HPP_
