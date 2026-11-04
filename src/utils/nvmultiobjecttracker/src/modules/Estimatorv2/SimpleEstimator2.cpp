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

#include "modules/Estimatorv2/SimpleEstimator2.hpp"

using std::cout;
using std::endl;

/**
 * @brief Construct a Kalman-filter state estimator from configuration.
 * @param[in] configFilePath The path to a config file.
 * @param[in] numTotalStates The # of total states to be estimated.
 * @param[in] numObservStates The # of observation (i.e., measurement) states.
 * @param[in] processNoiseScale Scale applied to the location, size, and velocity process-noise variances.
 * @param[in] measurementNoiseScale Scale applied to the detector and tracker measurement-noise variances.
 */
SimpleEstimator2::SimpleEstimator2(
    const std::string &configFilePath, const int numTotalStates, const int numObservStates,
    const float processNoiseScale, const float measurementNoiseScale
)
    : m_numTotalStates(numTotalStates),
      m_numObservStates(numObservStates),
      m_X(m_numTotalStates),
      m_A(m_numTotalStates, m_numTotalStates),
      m_Q(m_numTotalStates, m_numTotalStates),
      m_P(m_numTotalStates, m_numTotalStates),
      m_H(m_numObservStates, m_numTotalStates),
      m_R(m_numObservStates, m_numObservStates),
      m_I(m_numTotalStates, m_numTotalStates)
{
	m_params = SimpleEstimatorParams2(configFilePath, "StateEstimator");

	m_params.m_processNoiseVar4Loc *= processNoiseScale;
	m_params.m_processNoiseVar4Size *= processNoiseScale;
	m_params.m_processNoiseVar4Vel *= processNoiseScale;

	m_params.m_measurementNoiseVar4Detector *= measurementNoiseScale;
	m_params.m_measurementNoiseVar4Tracker *= measurementNoiseScale;

	m_I.setIdentity();
	m_age = 0;

	/// Check if only one of noiseWeightVar4Loc and noiseWeightVar4Vel is set.
	if (m_params.m_noiseWeightVar4Loc * m_params.m_noiseWeightVar4Vel <= 0)
	{
		printf(
			"!! [WARNING][SimpleEstimatorParams] Both noiseWeightVar4Loc and noiseWeightVar4Vel should be positive to enable dynamic covariance for Kalman filter. Fall back to const covariance\n"
		);
	}
	m_useDynamicCov = (m_params.m_noiseWeightVar4Loc > 0 && m_params.m_noiseWeightVar4Vel > 0);
	m_CommaInitFmt = Eigen::IOFormat(5, Eigen::DontAlignCols, ", ", ", ", "", "", " ", ";");
}

/** @brief Destroy the simple state estimator. */
SimpleEstimator2::~SimpleEstimator2()
{
}

/** @brief Reset the states and make the age to be zero. */
void SimpleEstimator2::reset()
{
	m_age = 0;
}

/**
 * @brief Set the transition matrix based on the input time interval.
 * @param[in] dT The prediction time interval.
 */
void SimpleEstimator2::setTransitionMatrix(const float dT)
{
	/// States for 6: [x, y, w, h, x', y'].
	m_A.setIdentity();
	for(int i=m_numObservStates; i<m_numTotalStates; i++)
	{
		m_A(i-m_numObservStates, i) = dT;
	}
}

/**
 * @brief Set the measurement-noise matrix.
 * @param[in] meas Input measurement used when noise is proportional to box size.
 * @param[in] option Measurement-noise selection.
 */
void SimpleEstimator2::setMeasurementNoiseMatrix(const Eigen::VectorXd &meas, const MEASUREMENT_NOISE_OPTION option)
{
	if (m_useDynamicCov)
	{
		/// Only when both noiseWeightVar4Loc and noiseWeightVar4Vel>0, adjust measurement noise every step based on measurement.
		m_R.setIdentity();
		for(int i=0; i<m_numObservStates; i++)
		{
			if (i == 2 && m_params.m_useAspectRatio)
			{
				/// Aspect ratio state, a very small constant variance invariant to box size.
				m_R(i,i) = std::pow(m_params.m_measurementNoiseVar4Aspc, 2);
			}
			else
			{
				m_R(i,i) = std::pow(m_params.m_noiseWeightVar4Loc * meas[3], 2);
			}
		}
	}
	else
	{
		/// Const measurement noise, set based on option.
		switch (option)
		{
			case DETECTOR:
				m_R.setIdentity();
				m_R *=  m_params.m_measurementNoiseVar4Detector;
				break;
			case TRACKER:
				m_R.setIdentity();
				m_R *=  m_params.m_measurementNoiseVar4Tracker;
				break;
			default:
				break;
		}
	}
}

/**
 * @brief Set the process-noise matrix.
 * @param[in] meas Input measurement used when noise is proportional to box size.
 * @param[in] isInit Whether the process-noise matrix is being initialized.
 */
void SimpleEstimator2::setProcessNoiseMatrix(const Eigen::VectorXd &meas, const bool isInit)
{
	/// Use dynamic noise based on measurement.
	if (m_useDynamicCov)
	{
		m_Q.setZero();
		/// Initialize observed states.
		for(int i=0; i<m_numObservStates; i++)
		{
			if (i == 2 && m_params.m_useAspectRatio)
			{
				/// Aspect ratio state, a small constant variance invariant to box size and other parameters.
				m_Q(i,i) = std::pow(m_params.m_processNoiseVar4Aspc, 2);
			}
			else
			{
				/// Motion and observation uncertainty are chosen relative to the current box size.
				m_Q(i,i) = std::pow(m_params.m_noiseWeightVar4Loc * meas[3], 2);
			}
		}

		/// Initialize velocity.
		for(int i=m_numObservStates; i<m_numTotalStates; i++)
		{
			if (i == m_numObservStates + 2 && m_params.m_useAspectRatio)
			{
				/// Aspect ratio velocity, a small constant variance invariant to box size and other parameters.
				m_Q(i,i) = std::pow(m_params.m_processNoiseVar4AspcVel, 2);
			}
			else
			{
				/// Motion and observation uncertainty are chosen relative to the current box size.
				m_Q(i,i) = std::pow(m_params.m_noiseWeightVar4Vel * meas[3], 2);
			}
		}
	}
	else if (!isInit)
	{
		/// Use constant process noise: only need to set during init.
		m_Q.setZero();
		/// Initialize observed states.
		for(int i=0; i<m_numObservStates; i++)
		{
			if (i == 2 && m_params.m_useAspectRatio)
			{
				/// Aspect ratio state, a small constant variance invariant to box size and other parameters.
				m_Q(i,i) = std::pow(m_params.m_processNoiseVar4Aspc, 2);
			}
			else
			{
				/// Default constant variance.
				m_Q(i,i) = i < 2 ? m_params.m_processNoiseVar4Loc : m_params.m_processNoiseVar4Size;
			}
		}

		/// Initialize velocity.
		for(int i=m_numObservStates; i<m_numTotalStates; i++)
		{
			if (i == m_numObservStates + 2 && m_params.m_useAspectRatio)
			{
				/// Aspect ratio velocity, a small constant variance invariant to box size and other parameters.
				m_Q(i,i) = std::pow(m_params.m_processNoiseVar4AspcVel, 2);;
			}
			else
			{
				/// Default constant variance.
				m_Q(i,i) = m_params.m_processNoiseVar4Vel;
			}
		}
	}
}

/**
 * @brief Initialize the target state from a measurement.
 * @param[in] meas The initial measurement.
 */
void SimpleEstimator2::initStates(const Eigen::VectorXd &meas)
{
	if(m_numTotalStates >= m_numObservStates && m_numObservStates == 4)
	{
		/// Initialize Kalman Filter with random acceleration dynamics model.
		/// States: [x, y, w, h, x', y']'.
		setTransitionMatrix();

		/// Measurement matrix H: [1 0 0 0 0 0; 0 1 0 0 0 0; 0 0 1 0 0 0; 0 0 0 1 0 0].
		m_H.setIdentity();

		/// Process Noise Covariance Matrix Q: [Ex 0 0 0 0 0; 0 Ey 0 0 0 0; 0 0 Ew 0 0 0; 0 0 0 Eh 0 0; 0 0 0 0 Evx 0; 0 0 0 0 0 Evy].
		setProcessNoiseMatrix(meas);

		/// Measurement Noise Covariance Matrix R: [1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1].
		setMeasurementNoiseMatrix(meas, MEASUREMENT_NOISE_OPTION::DETECTOR);

		/// Priori Error Estimate Covariance Matrix P'(k).
		if (m_useDynamicCov)
		{
			/// Scale initial position and height variance by 4 and velocity variance by 100.
			m_P = m_Q;
			m_P(0,0) *= 4.0;
			m_P(1,1) *= 4.0;
			m_P(3,3) *= 4.0;
			m_P(4,4) *= 100.0;
			m_P(5,5) *= 100.0;
			m_P(7,7) *= 100.0;
		}
		else
		{
			/// Set fixed initial values.
			m_P.setIdentity();
			m_P *= m_params.m_highUncertainty;
		}
	}
	else if(m_numTotalStates == 4 && m_numObservStates == 2)
	{
		/// Initialize Kalman Filter with random acceleration dynamics model.
		/// States: [x, y, x', y']': transition matrix A:= [1 0 dT 0; 0 1 0 dT; 0 0 1 0; 0 0 0 1].
		setTransitionMatrix();

		/// Measurement matrix H: [1 0 0 0; 0 1 0 0].
		m_H.setIdentity();

		/// Process Noise Covariance Matrix Q: [Ex 0 0 0; 0 Ey 0 0; 0 0 Evx 0; 0 0 0 Evy].
		setProcessNoiseMatrix(meas);

		/// Measurement Noise Covariance Matrix R: [1 0; 0 1].
		setMeasurementNoiseMatrix(meas, MEASUREMENT_NOISE_OPTION::DETECTOR);

		/// Set fixed initial values.
		m_P.setIdentity();
		m_P *= m_params.m_highUncertainty;
	}
	else
	{
#ifdef _WIN32
		throw std::runtime_error("[SimpleEstimator2::initStates()] Not supported state config");
#else
		std::runtime_error("[SimpleEstimator2::initStates()] Not supported state config");
#endif
	}

	if (m_params.m_debugVerboseLevel > 2)
	{
		std::cout << "Init: covariance:" << m_P.diagonal().format(m_CommaInitFmt) << std::endl;
	}
}

/**
 * @brief Predict the states for a time interval.
 * @param[in] dT The time step.
 */
void SimpleEstimator2::predict(const float dT)
{
	setTransitionMatrix(dT);
	setProcessNoiseMatrix(m_X, true);

	m_X_pred = m_A * m_X;
	m_P_pred = m_A*m_P*m_A.transpose() + m_Q;

	if (m_params.m_debugVerboseLevel > 2)
	{
		std::cout << "Predict: [Mean]:" << m_X_pred.format(m_CommaInitFmt) << " [Covariance]: " <<
			m_P.diagonal().format(m_CommaInitFmt) << " [Motion_cov]: " << m_Q.diagonal().format(m_CommaInitFmt) << std::endl;
	}
}

/**
 * @brief Update without any measurement, using the prediction only.
 * @param[in] dT The time step.
 * @return The estimated states.
 */
Eigen::VectorXd SimpleEstimator2::update(const float dT)
{
	if(m_age == 0)
	{
#ifdef _WIN32
		throw std::runtime_error(
			"[SimpleEstimator2::update()] Cannot do the very first state updates without measurement !!"
		);
#else
		std::runtime_error(
			"[SimpleEstimator2::update()] Cannot do the very first state updates without measurement !!"
		);
#endif
	}

	predict(dT);

	/// Given no measurement, make prediction only, without any correction step.
	m_X = m_X_pred;

	if(m_params.m_debugVerboseLevel > 1 && m_X.allFinite() == false)
	{
		printf("[SimpleEstimator2::update()][Warning] Filter updates include nan !!");
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
Eigen::VectorXd SimpleEstimator2::update(
	const std::vector<std::tuple<Eigen::VectorXd, Eigen::Matrix2d, bool>> &measList, const float dT
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
			Eigen::VectorXd meas = std::get<0>(measList[i]);
			const Eigen::Matrix2d &measCov = std::get<1>(measList[i]);
			const bool isDetectorInput = std::get<2>(measList[i]);

			if(i == 0)
			{
				predict(dT);
			}
			else
			{
				m_P_pred = m_P;
			}
			/// Use provided covariance for LOCATION_ESTIMATOR_PROJECTIVE, otherwise use default noise.
			if(m_params.m_stateEstimatorType == STATE_ESTIMATOR_TYPES::LOCATION_ESTIMATOR_PROJECTIVE)
			{
				/// Use the provided covariance as measurement noise.
				m_R = measCov;
			}
			else
			{
				/// Use default noise based on detector/tracker.
				if(isDetectorInput)
				{
					setMeasurementNoiseMatrix(meas, MEASUREMENT_NOISE_OPTION::DETECTOR);
				}
				else
				{
					setMeasurementNoiseMatrix(meas, MEASUREMENT_NOISE_OPTION::TRACKER);
				}
			}

			if(m_age == 0 && i == 0)
			{
				initStates(meas);

				int j=0;
				for(; j<m_numObservStates; j++)
				{
					m_X(j) = meas(j);
				}
				for(; j<m_numTotalStates; j++)
				{
					m_X(j) = 0;
				}

				m_X_pred = m_X;
			}
			else
			{
				m_K = m_P_pred*m_H.transpose()*(m_H*m_P_pred*m_H.transpose() + m_R).inverse();
				m_X_pred += m_K * (meas - m_H*m_X_pred);
				m_P = (m_I - m_K*m_H)*m_P_pred*(m_I - m_K*m_H).transpose() + m_K*m_R*m_K.transpose();
				m_X = m_X_pred;
			}
		}

		if(m_params.m_debugVerboseLevel > 1 && m_X.allFinite() == false)
		{
			cout << "[SimpleEstimator2::update()][Warning] Estimated states (m_X) include nan or inf  !!" << endl;
		}

		m_age++;

		return m_X;
	}
}

/**
 * @brief Predict states without modifying the current estimate.
 * @param[in] dT The time step.
 */
Eigen::VectorXd SimpleEstimator2::getPredictedStates(const float dT)
{
	setTransitionMatrix(dT);

	return m_A * m_X;
}

/**
 * @brief Predict states and covariance without modifying the current estimate.
 * @param[in] dT The time step.
 */
std::tuple<Eigen::VectorXd, Eigen::MatrixXd> SimpleEstimator2::getPredictedStatesWithCov(const float dT)
{
	predict(dT);

	return {m_X_pred, m_P_pred};
}

/**
 * @brief Predict measurements and covariance without modifying the current estimate.
 * @param[in] dT The time step.
 */
std::tuple<Eigen::VectorXd, Eigen::MatrixXd> SimpleEstimator2::getPredictedMeasurementsWithCov(const float dT)
{
	predict(dT);

	if(m_params.m_debugVerboseLevel > 2)
	{
		std::cout << "Before Mahala distance (state): mean:" << m_X_pred.format(m_CommaInitFmt) << " covariance:" <<
				m_P_pred.diagonal().format(m_CommaInitFmt) << std::endl;
	}

	setMeasurementNoiseMatrix(m_X_pred);

	Eigen::VectorXd Z = m_H*m_X_pred;
	Eigen::MatrixXd Zcov = m_H*m_P_pred*m_H.transpose() + m_R;

	return {Z, Zcov};
}

/** @brief Initialize simple-estimator parameters with default values. */
SimpleEstimatorParams2::SimpleEstimatorParams2()
{
	m_debugVerboseLevel = 0;

    m_processNoiseVar4Loc = 2.f;
    m_processNoiseVar4Size = 1.f;
    m_processNoiseVar4Vel = 0.1f;
	m_processNoiseVar4Aspc = 1e-2;
	m_processNoiseVar4AspcVel = 1e-5;
    m_measurementNoiseVar4Detector = 4.f;
    m_measurementNoiseVar4Tracker = 16.f;
	m_measurementNoiseVar4Aspc = 1e-1;

	m_noiseWeightVar4Loc = -0.1f;
    m_noiseWeightVar4Vel = -0.1f;
    m_highUncertainty = 1e+4;

	m_useAspectRatio = false;
}

/**
 * @brief Initialize simple-estimator parameters from a configuration module.
 * @param[in] configFilePath The path to a configuration file.
 * @param[in] moduleName The configuration module name.
 */
SimpleEstimatorParams2::SimpleEstimatorParams2(const std::string &configFilePath, const std::string moduleName)
{
	YAML::Node configRoot = getConfigRoot(configFilePath);
	this->read(configRoot[moduleName]);
}

/**
 * @brief Initialize simple-estimator parameters from a YAML node.
 * @param[in] fn Parsed YAML structure of config to update.
 */
SimpleEstimatorParams2::SimpleEstimatorParams2(const YAML::Node &fn)
{
	this->read(fn);
}

/**
 * @brief Update simple-estimator parameters from a YAML node.
 * @param[in] fn Parsed YAML structure of config to update.
 */
void SimpleEstimatorParams2::read(const YAML::Node &fn)
{
	*this = SimpleEstimatorParams2();

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
		else if (paramKey == "measurementNoiseVar4Detector")
		{
			m_measurementNoiseVar4Detector = fit->second.as<float>();
		}
		else if (paramKey == "measurementNoiseVar4Tracker")
		{
			m_measurementNoiseVar4Tracker = fit->second.as<float>();
		}
		else if (paramKey == "measurementNoiseVar4Aspc")
		{
			m_measurementNoiseVar4Aspc = fit->second.as<float>();
		}
		else if (paramKey == "processNoiseVar4Loc")
		{
			m_processNoiseVar4Loc = fit->second.as<float>();
		}
		else if (paramKey == "processNoiseVar4Size")
		{
			m_processNoiseVar4Size = fit->second.as<float>();
		}
		else if (paramKey == "processNoiseVar4Vel")
		{
			m_processNoiseVar4Vel = fit->second.as<float>();
		}
		else if (paramKey == "processNoiseVar4Aspc")
		{
			m_processNoiseVar4Aspc = fit->second.as<float>();
		}
		else if (paramKey == "processNoiseVar4AspcVel")
		{
			m_processNoiseVar4AspcVel = fit->second.as<float>();
		}
		else if (paramKey == "highUncertainty")
		{
			m_highUncertainty = fit->second.as<float>();
		}
		else if (paramKey == "noiseWeightVar4Loc")
		{
			m_noiseWeightVar4Loc = fit->second.as<float>();
		}
		else if (paramKey == "noiseWeightVar4Vel")
		{
			m_noiseWeightVar4Vel = fit->second.as<float>();
		}
		else if (paramKey == "useAspectRatio")
		{
			m_useAspectRatio = (int)fit->second.as<float>();
		}
		else
		{
			cout << "!! [WARNING][SimpleEstimatorParams] Unknown param found: " << paramKey << endl;
		}
    }
}
