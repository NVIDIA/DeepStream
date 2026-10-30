/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef NVMULTIOBJECTTRACKERLIB_OBJECTMODELPROJECTOR_HPP_
#define NVMULTIOBJECTTRACKERLIB_OBJECTMODELPROJECTOR_HPP_

#include <atomic>

#ifdef VIZ_DEBUG
#include <opencv2/opencv.hpp>
#endif

#include "modules/utility/dataTypes.hpp"
#include "modules/utility/debug_common.hpp"
#include "modules/NvMultiObjectTracker/NvMultiObjectTracker_common.hpp"
#include "modules/NvMultiObjectTracker/NvTrackerParams.hpp"

/// Fixed-size Eigen matrix alias.
template <typename K, int N, int M>
using Matrix = Eigen::Matrix<K, N, M>;

using IntrinsicMatrix = Matrix<double, 3, 3>; //!< 3x3 Intrinsic matrix: K.
using RotationMatrix = Matrix<double, 3, 3>; //!< 3x3 Rotation matrix: R.
using TranslationMatrix = Matrix<double, 3, 1>; //!< 3x1 Translation matrix T.
using CameraMatrix = Matrix<double, 3, 4>; //!< 3x4 Camera matrix: [ R | T ].
using HomographyMatrix = Matrix<double, 3, 3>; //!< 3x3 Homography matrix: H.

/** @brief Store camera calibration and image-mapping parameters. */
struct CameraInfo
{
	/** @brief Initialize camera rotation order to YZX. */
	CameraInfo() : order("YZX") {};

	double focalLength; //!< Specifies the camera focal length.
	double yaw; //!< Specifies yaw in degrees.
	double roll; //!< Specifies roll in degrees.
	double pitch; //!< Specifies pitch in degrees.
	std::string order; //!< Specifies the rotation-matrix axis order.
	double Tx; //!< Specifies camera translation along the X axis.
	double Ty; //!< Specifies camera translation along the Y axis.
	double Tz; //!< Specifies camera translation along the Z axis.
	Eigen::Vector2d H_Camera2MappedImage_posOffset; //!< Stores the mapped-image position offset.
	double H_Camera2MappedImage_Scale; //!< Stores the mapped-image scale.
};

/** @brief Describe a class-specific three-dimensional target model. */
struct Model3DInfo
{
	float height; //!< Specifies model height.
	float radius; //!< Specifies model radius.
	float heightConfidence = 0.f; //!< Confidence of height estimation (0 = default/no pose data).
};

/** @brief Store a projected three-dimensional bounding box and its corners. */
struct Proj3DBboxInfo
{
	/** @brief Construct an empty projected bounding-box record. */
	Proj3DBboxInfo() = default;

	/** @brief Move-construct a projected bounding-box record. */
	Proj3DBboxInfo(Proj3DBboxInfo&&) noexcept = default;

	/** @brief Move-assign a projected bounding-box record. */
	Proj3DBboxInfo& operator=(Proj3DBboxInfo&&) noexcept = default;

	/** @brief Copy-construct a projected bounding-box record. */
	Proj3DBboxInfo(const Proj3DBboxInfo&) = default;

	/** @brief Copy-assign a projected bounding-box record. */
	Proj3DBboxInfo& operator=(const Proj3DBboxInfo&) = default;

	NvMotDataTypes::Rect2f bbox; //!< Stores the projected bounding rectangle.
	std::vector<Eigen::Vector2f> projCorners; //!< Bottom four points, and top four points.
};

/** @brief Store geometry recovered from an object model. */
struct RecoveredObjModel
{
	/** @brief Construct an empty recovered object model. */
	RecoveredObjModel() = default;

	/** @brief Move-construct a recovered object model. */
	RecoveredObjModel(RecoveredObjModel&&) noexcept = default;

	/** @brief Move-assign a recovered object model. */
	RecoveredObjModel& operator=(RecoveredObjModel&&) noexcept = default;

	/** @brief Copy-construct a recovered object model. */
	RecoveredObjModel(const RecoveredObjModel&) = default;

	/** @brief Copy-assign a recovered object model. */
	RecoveredObjModel& operator=(const RecoveredObjModel&) = default;

	Eigen::Vector2d pt_img_center; //!< The refined image coordinate of the human waist (i.e., center of bbox).
	NvMotDataTypes::Rect2f bbox; //!< The refined bbox using the projected cylindrical model.
	Eigen::Vector2d pt_img_feet; //!< The feet position in camera image frame.
	Eigen::Vector2d pt_world_feet; //!< The feet position on the floor in world coordinate system.
	std::vector<Eigen::Vector2f> convexHull; //!< The convex hull of the human cylinder model.
	float visibility; //!< Stores the visible fraction after occlusion and image-border cropping.
	float distFromCam; //!< Stores the distance from the camera.
};

/** @brief Enumerate the supported object-model types. */
enum OBJ_MODEL_TYPE : int
{
	CYLINDER=0, //!< Models the target as a cylinder.
	BBOX3D=1 //!< Models the target as a three-dimensional bounding box.
};

/** @brief Store plane and inter-plane homographies for a target model. */
struct TargetHomography
{
	HomographyMatrix WorldFeetPlane2CamOpticalCenter; //!< Maps the world feet plane to the camera optical-center plane.
	HomographyMatrix WorldHeadPlane2CamOpticalCenter; //!< Maps the world head plane to the camera optical-center plane.
	HomographyMatrix WorldWaistPlane2CamOpticalCenter; //!< Maps the world waist plane to the camera optical-center plane.
	HomographyMatrix CamOpticalCenter2WorldFeetPlane; //!< Maps the camera optical-center plane to the world feet plane.
	HomographyMatrix CamOpticalCenter2WorldHeadPlane; //!< Maps the camera optical-center plane to the world head plane.
	HomographyMatrix CamOpticalCenter2WorldWaistPlane; //!< Maps the camera optical-center plane to the world waist plane.

	// [Composite homographies based on the camera optical center]
	HomographyMatrix Feet2Waist; //!< Maps feet image points to waist image points.
	HomographyMatrix Waist2Feet; //!< Maps waist image points to feet image points.
	HomographyMatrix Waist2Head; //!< Maps waist image points to head image points.
	HomographyMatrix Head2Waist; //!< Maps head image points to waist image points.
};

/** @brief Store object-model projector configuration parameters. */
class NvTrackerParamsObjectModelProjector : public NvTrackerParams
{
public:
	/** @brief Initialize object-model projector parameters with default values. */
	NvTrackerParamsObjectModelProjector();

	/**
	 * @brief Initialize object-model projector parameters from a configuration module.
	 * @param[in] configFilePath The path to a configuration file.
	 * @param[in] moduleName The configuration module name.
	 */
	NvTrackerParamsObjectModelProjector(const std::string &configFilePath, const std::string moduleName);

	/**
	 * @brief Initialize object-model projector parameters from a YAML node.
	 * @param[in] fn Parsed YAML structure of config to update.
	 */
	NvTrackerParamsObjectModelProjector(const YAML::Node &fn);

	/**
	 * @brief Update object-model projector parameters from a YAML node.
	 * @param[in] fn Parsed YAML structure of config to update.
	 */
	void read(const YAML::Node &fn) override;

	int m_debugVerboseLevel; //!< Controls debug message verbosity.
	std::vector<std::string> m_cameraModelFilepath; //!< Stores camera-model paths indexed by stream.
	std::map<std::string, std::string> m_cameraModelFilepathMap; //!< Stores camera-model paths keyed by sensor name.
	std::vector<std::string> m_proj3DBboxFilePath; //!< Stores external three-dimensional bounding-box paths.
	bool m_outputFootLocation; //!< Enables foot-location output.
	bool m_outputConvexHull; //!< Enables convex-hull output.
	OBJ_MODEL_TYPE m_objectModelType; //!< Selects the object-model type.
	int m_maxConvexHullSize; //!< Specifies the maximum number of convex-hull points.
	float m_waistHeightRatio; //!< Specifies the ratio of hip-keypoint height to total height.
	float m_eyeHeightRatio; //!< Specifies the ratio of eye-keypoint height to total height.
	bool m_keepDetBboxForInitEstimation; //!< Keeps the detector box for initial association instead of the recovered-model box.
	float m_minPoseConfidence; //!< Minimum pose keypoint confidence for adaptive height estimation.
	float m_useDefaultModelHeight; //!< Overrides class-specific model height when positive.
	float m_useDefaultModelRadius; //!< Overrides class-specific model radius when positive.
};

/** @brief Project and recover target geometry using camera and object models. */
class ObjectModelProjector
{
public:
	/** @brief Construct an uninitialized object-model projector. */
	ObjectModelProjector();

	/**
	 * @brief Construct an object-model projector from configuration.
	 * @param[in] configFilePath Path to the configuration file.
	 */
	ObjectModelProjector(const std::string &configFilePath);

	/**
	 * @brief Construct and initialize a projector for a source stream.
	 * @param[in] configFilePath Path to the configuration file.
	 * @param[in] imgSize Source image size.
	 * @param[in] streamId Source stream identifier.
	 */
	ObjectModelProjector(
		const std::string &configFilePath, const NvMotDataTypes::Size imgSize, const NvMOTStreamId streamId
	);

	/**
	 * @brief Construct and initialize a projector for a named source sensor.
	 * @param[in] configFilePath Path to the configuration file.
	 * @param[in] imgSize Source image size.
	 * @param[in] streamId Source stream identifier.
	 * @param[in] sensorName Source sensor name.
	 */
	ObjectModelProjector(
		const std::string &configFilePath, const NvMotDataTypes::Size imgSize,
		const NvMOTStreamId streamId, const std::string &sensorName
	);

	/** @brief Destroy the object-model projector. */
	virtual ~ObjectModelProjector();

	/**
	 * @brief Populate tracker query capabilities from projector configuration.
	 * @param[out] paramsQuery Tracker query structure to update.
	 */
	void queryParams(NvMOTQuery &paramsQuery);

	/** @brief Return whether the projector has been initialized. */
	bool isInitialized() { return m_isInit; }

	/** @brief Return the source sensor name. */
	const std::string &getSensorName() const { return m_sensorName; }

	/**
	 * @brief Recover an object model from a detector bounding box.
	 * @param[in] detObj Detected object to project.
	 * @param[in] frameNum Current frame number.
	 * @param[in] targetInd Target index, or a negative value when unavailable.
	 * @param[in] isInitEstimation Whether this is the target's initial estimate.
	 */
	RecoveredObjModel getRecoveredObjectModelFromBbox(
		const NvMOTObjToTrack *detObj, const NvMOTFrameNum frameNum, const int targetInd, const bool isInitEstimation
	);

	/**
	 * @brief Recover an object model from externally projected three-dimensional bounding-box data.
	 * @param[in] detObj Detected object to match.
	 * @param[in] frameNum Current frame number.
	 */
	RecoveredObjModel getRecoveredObjectModelFromProj3DBbox(
		const NvMOTObjToTrack *detObj, const NvMOTFrameNum frameNum
	);

	/**
	 * @brief Project a cylinder bounding box from an image-space feet point.
	 * @param[in] pt_img_head Image-space feet point.
	 * @param[in] classId Target class ID.
	 * @param[in] targetInd Target index.
	 */
	const NvMotDataTypes::Rect2f getBboxOfProjectedCylinderFromImgFeet(
		const Eigen::Vector2d pt_img_head, const NvMOTClassId classId, const int targetInd
	);

	/**
	 * @brief Project a cylinder bounding box from an image-space waist point.
	 * @param[in] pt_img_waist Image-space waist point.
	 * @param[in] classId Target class ID.
	 * @param[in] targetInd Target index.
	 */
	const NvMotDataTypes::Rect2f getBboxOfProjectedCylinderFromImgWaist(
		const Eigen::Vector2d pt_img_waist, const NvMOTClassId classId, const int targetInd
	);

	/**
	 * @brief Project a cylinder bounding box from an image-space head point.
	 * @param[in] pt_img_head Image-space head point.
	 * @param[in] classId Target class ID.
	 * @param[in] targetInd Target index.
	 */
	const NvMotDataTypes::Rect2f getBboxOfProjectedCylinderFromImgHead(
		const Eigen::Vector2d pt_img_head, const NvMOTClassId classId, const int targetInd
	);

	/**
	 * @brief Get recovered 3D cylinder model with dynamic height from pose keypoints.
	 * @param[in] pt_img_head The head location on the image.
	 * @param[in] pt_img_waist The waist location on the image.
	 * @param[in] pt_img_foot The foot location on the image.
	 * @param[in] bbox The bounding box.
	 * @param[in] targetInd The target index.
	 * @param[in] classId The class ID.
	 * @param[in] conf_head Confidence of the head keypoint.
	 * @param[in] conf_waist Confidence of the waist keypoint.
	 * @param[in] conf_foot Confidence of the foot keypoint.
	 * @return Recovered 3D cylinder model with dynamic height.
	 */
	RecoveredObjModel getRecoveredObjectModelWithAdaptiveHeight(
		const Eigen::Vector2d pt_img_head, const Eigen::Vector2d pt_img_waist,
		const Eigen::Vector2d pt_img_foot, const NvMotDataTypes::Rect2f& bbox, const TrackerIndex &targetInd,
		const NvMOTClassId classId, float conf_head, float conf_waist, float conf_foot
	);

	/**
	 * @brief Estimate target height and world feet position from image head and waist points.
	 * @param[in] pt_img_head Image-space head point.
	 * @param[in] pt_img_waist Image-space waist point.
	 * @param[in] targetInd Target index.
	 * @param[in] classId Target class ID.
	 */
	Eigen::Vector2d computeHeightAndWorldFeetFromHeadAndWaist(
		const Eigen::Vector2d pt_img_head, const Eigen::Vector2d pt_img_waist,
		const TrackerIndex &targetInd, const NvMOTClassId classId
	);

	/**
	 * @brief Estimate target height and world feet position from image head and foot points.
	 * @param[in] pt_img_head Image-space head point.
	 * @param[in] pt_img_foot Image-space foot point.
	 * @param[in] targetInd Target index.
	 * @param[in] classId Target class ID.
	 */
	Eigen::Vector2d computeHeightAndWorldFeetFromHeadAndFoot(
		const Eigen::Vector2d pt_img_head, const Eigen::Vector2d pt_img_foot,
		const TrackerIndex &targetInd, const NvMOTClassId classId
	);

	/**
	 * @brief Project a cylinder bounding box from world-space feet coordinates.
	 * @param[in] pt_world_feet World-space feet position.
	 * @param[in] classId Target class ID.
	 * @param[in] targetInd Target index.
	 */
	const NvMotDataTypes::Rect2f getBboxOfProjectedCylinderFromWorldFeet(
		const Eigen::Vector2d pt_world_feet, const NvMOTClassId classId, const int targetInd
	);

	/**
	 * @brief Project world-ground points into image coordinates.
	 * @param[in] pt_world_feet_list World-ground points to project.
	 */
	const std::vector<Eigen::Vector2d> getProjectedWorldGroundPts(
		const std::vector<Eigen::Vector2d>& pt_world_feet_list
	);

	/**
	 * @brief Map an image waist point to the corresponding image head point.
	 * @param[in] imgWaistPt Image-space waist point.
	 * @param[in] classId Target class ID.
	 * @param[in] targetInd Target index.
	 */
	const Eigen::Vector2d getCamHeadPtFromCamWaistPt(
		const Eigen::Vector2d imgWaistPt, const NvMOTClassId classId, const int targetInd
	);

	/**
	 * @brief Map an image head point to the corresponding image waist point.
	 * @param[in] imgHeadPt Image-space head point.
	 * @param[in] classId Target class ID.
	 * @param[in] targetInd Target index.
	 */
	const Eigen::Vector2d getCamWaistPtFromCamHeadPt(
		const Eigen::Vector2d imgHeadPt, const NvMOTClassId classId, const int targetInd
	);

	/**
	 * @brief Map an image feet point to the corresponding image waist point.
	 * @param[in] imgWaistPt Image-space feet point.
	 * @param[in] classId Target class ID.
	 * @param[in] targetInd Target index.
	 */
	const Eigen::Vector2d getCamWaistPtFromCamFeetPt(
		const Eigen::Vector2d imgWaistPt, const NvMOTClassId classId, const int targetInd
	);

	/**
	 * @brief Map an image waist point to the corresponding image feet point.
	 * @param[in] imgWaistPt Image-space waist point.
	 * @param[in] classId Target class ID.
	 * @param[in] targetInd Target index.
	 */
	const Eigen::Vector2d getCamFeetPtFromCamWaistPt(
		const Eigen::Vector2d imgWaistPt, const NvMOTClassId classId, const int targetInd
	);

	/**
	 * @brief Recover world-ground feet coordinates from an image waist point.
	 * @param[in] imgWaistPt Image-space waist point.
	 * @param[in] classId Target class ID.
	 * @param[in] targetInd Target index.
	 */
	const Eigen::Vector2d getWorldFeetPtOnGroundFromCamWaistPt(
		const Eigen::Vector2d imgWaistPt, const NvMOTClassId classId, const int targetInd
	);

	/**
	 * @brief Recover world-ground feet coordinates from an image feet point.
	 * @param[in] imgFeetPt Image-space feet point.
	 * @param[in] classId Target class ID.
	 * @param[in] targetInd Target index.
	 */
	const Eigen::Vector2d getWorldFeetPtOnGroundFromCamFeetPt(
		const Eigen::Vector2d imgFeetPt, const NvMOTClassId classId, const int targetInd
	);

	/**
	 * @brief Build a world-space three-dimensional bounding box from feet coordinates.
	 * @param[in] pt_world_feet World-space feet position.
	 * @param[in] classId Target class ID.
	 * @param[in] targetInd Target index.
	 */
	const NvDsObj3DBbox getWorld3DBboxFromWorldFeet(
		const Eigen::Vector2d pt_world_feet, const NvMOTClassId classId, const int targetInd
	);

	/**
	 * @brief Project the eight corners of a world-space three-dimensional bounding box.
	 * @param[in] pt_world_feet World-space feet position.
	 * @param[in] classId Target class ID.
	 * @param[in] targetInd Target index.
	 */
	std::vector<Eigen::Vector2d> calculateProjCornersOf3DBboxFromWorldFeet(
		const Eigen::Vector2d pt_world_feet, const NvMOTClassId classId, const int targetInd
	);

	/**
	 * @brief Project a world-space point into image coordinates.
	 * @param[in] pt_world World-space point.
	 */
	const Eigen::Vector2d projectWorldPt2ImgPt(const Eigen::Vector3d pt_world);

	/**
	 * @brief Check if any point of a vertical target (from feet to head) projects into the camera FOV.
	 * @param[in] pt_world_feet World coordinates of feet (X, Y on ground plane).
	 * @param[in] targetHeight Actual height of the target (uses adaptive height if available).
	 * @return True if any part of the target's vertical extent is visible in the image.
	 */
	bool isTargetVisibleInImage(const Eigen::Vector2d& pt_world_feet, float targetHeight);

	/**
	 * @brief Project a horizontal circle centered at world-space feet coordinates.
	 * @param[in] pt_world_feet World-space feet position.
	 * @param[in] radius Circle radius.
	 */
	std::vector<Eigen::Vector2d> calculateProjectedCircleFromWorldFeet(
		const Eigen::Vector2d pt_world_feet, const float radius
	);

	/** @brief Return whether foot-location output is enabled. */
	const bool getOutputFootLocation() { return m_params.m_outputFootLocation; }

	/** @brief Return whether convex-hull output is enabled. */
	const bool getOutputConvexHull() { return m_params.m_outputConvexHull; }

	/** @brief Return the minimum pose-keypoint confidence. */
	float getMinPoseConfidence() const { return m_params.m_minPoseConfidence; }

	/**
	 * @brief Compute ground plane position covariance from pixel uncertainty using Jacobian.
	 * @param[in] pixel_pt The pixel coordinates (u, v).
	 * @param[in] pixel_sigma Standard deviation of pixel noise.
	 */
	Eigen::Matrix2d computeGroundCovariance(const Eigen::Vector2d& pixel_pt, float pixel_sigma);

	/** @brief Return the current target-model height. */
	float getTargetHeight() const { return m_targetModel.height; }

	/** @brief Return the current height-estimation confidence. */
	float getHeightConfidence() const { return m_targetModel.heightConfidence; }

	/**
	 * @brief Update target-model height when confidence permits.
	 * @param[in] height New target height.
	 * @param[in] confidence Confidence of the new height.
	 * @param[in] force Whether to update regardless of confidence.
	 */
	void setTargetHeight(float height, float confidence, bool force = false)
	{
		/// Only update if new confidence >= current (or force is true).
		if (force || confidence >= m_targetModel.heightConfidence) {
			m_targetModel.height = height;
			m_targetModel.heightConfidence = confidence;
			deriveDataFromCameraProjMatrix(m_targetHomographyMat, m_targetModel);
		}
	}

	/**
	 * @brief Select the target model for a class ID.
	 * @param[in] classId Target class ID.
	 */
	void setTargetModelForClass(NvMOTClassId classId)
	{
		/// Early return if current class matches the new class.
		if (classId == m_targetClassId) return;

		m_targetModel = getDefaultModelByClass(classId);
		m_targetHomographyMat = getDefaultHomographyMatByClass(classId);
		m_targetClassId = classId;

		if (m_params.m_debugVerboseLevel > 2) {
			std::cout << "[ObjectModelProjector] Set target model for classId " << classId
					  << ", height=" << m_targetModel.height << ", radius=" << m_targetModel.radius << std::endl;
		}
	}

	/** @brief Return the current target-model class ID. */
	NvMOTClassId getTargetClassId() const { return m_targetClassId; }

protected:

	bool m_isInit; //!< Indicates whether the projector has been initialized.
	uint m_streamInd; //!< Index of the corresponding source stream.
	std::string m_sensorName; //!< Sensor name of the corresponding source stream.

	// [Configuration]
	const std::string m_configFilePath; //!< Path of the module config file.
	std::string m_resolvedCameraModelFilepath; //!< Resolved camera model filepath for this stream.
	NvTrackerParamsObjectModelProjector m_params; //!< Config params for target management.
	NvMotDataTypes::Size m_srcFrameSize; //!< Source video frame size where the bbox info is based on.
	CameraInfo m_cameraInfo; //!< Set camera parameters.
	std::vector<std::vector<Proj3DBboxInfo>> m_externalProj3DBboxInfo; //!< Store projected 3D bounding box info for all frames in the stream.
	Model3DInfo m_targetModel; //!< Each target's object model can be different in size.
	NvMOTClassId m_targetClassId; //!< Projector's current target model class ID.
	std::map<NvMOTClassId, Model3DInfo> m_defaultModelByClass; //!< Each class's object model.
	float m_groundPlaneHeight; //!< Specifies the world-space ground-plane height.
	CameraMatrix m_ProjMat_World2Cam; //!< Stores the world-to-camera projection matrix.

	// [Pre-computed matrices]
	IntrinsicMatrix m_K; //!< Intrinsic matrix.
	IntrinsicMatrix m_K_inv; //!< Inverse of intrinsic matrix.
	RotationMatrix m_R; //!< Rotation matrix (world to camera).
	RotationMatrix m_R_T; //!< Transpose of rotation matrix.
	TranslationMatrix m_t; //!< Translation vector.
	Eigen::Vector2d m_pix2cam_offset; //!< Shifts corner-origin pixels to the optical center for projectionMatrix_3x4.
	//!< Zero when projectionMatrix_3x4_w2p already maps into pixel coordinates.

	// [Sherman-Morrison homography-inverse optimization]
	HomographyMatrix m_H_ground; //!< Ground-plane homography, independent of target height.
	HomographyMatrix m_H_ground_inv; //!< Inverse of ground plane homography.
	Eigen::Vector3d m_r3; //!< Third row of R^T (for ground plane intersection).
	Eigen::Vector3d m_P_z_col; //!< P[:,2] - projection matrix Z column (height update direction).
	Eigen::Vector3d m_H_inv_Pz; //!< H_ground_inv * P_z_col (precomputed for Sherman-Morrison).
	TranslationMatrix m_camLocation_World; //!< Camera position in world coordinate system.
	TargetHomography m_targetHomographyMat; //!< Homography matrices for head, waist, and feet planes composed from the projection matrix.
	std::map<NvMOTClassId, TargetHomography> m_defaultHomographyMatByClass; //!< Stores default target homographies by class.

	/**
	 * @brief Convert a nested YAML sequence to an Eigen matrix.
	 * @param[in] yamlVec YAML matrix values.
	 * @param[in] rows Matrix row count.
	 * @param[in] cols Matrix column count.
	 */
	const Eigen::MatrixXd convertYamlVec2EigenMatrixXd(YAML::Node yamlVec, const int rows, const int cols);

	/**
	 * @brief Import camera-model calibration from a YAML file.
	 * @param[in] filename Camera-model file path.
	 * @return True when the camera model is imported successfully.
	 */
	bool importCameraModelInfo(const std::string filename);

	/**
	 * @brief Import externally projected three-dimensional bounding boxes.
	 * @param[in] filename Projected bounding-box file path.
	 * @return True when the bounding boxes are imported successfully.
	 */
	bool importExternalProj3DBboxInfo(const std::string filename);

	/**
	 * @brief Return the default three-dimensional model for a class.
	 * @param[in] classId Target class ID.
	 */
	const Model3DInfo& getDefaultModelByClass(const NvMOTClassId classId)
	{
		if (m_defaultModelByClass.size() == 1)
		{
			return m_defaultModelByClass.begin()->second;
		}
		/// Fallback to first available model if no model is defined for the given class ID and print a warning.
		if (m_defaultModelByClass.find(classId) == m_defaultModelByClass.end())
		{
			if (m_params.m_debugVerboseLevel > 0) {
				std::cout << "[ObjectModelProjector] Warning: No model defined for classId " << classId
						  << ", using first available model" << std::endl;
			}
			return m_defaultModelByClass.begin()->second;
		}
		return m_defaultModelByClass.at(classId);
	}

	/**
	 * @brief Return the default homographies for a class.
	 * @param[in] classId Target class ID.
	 */
	const TargetHomography& getDefaultHomographyMatByClass(const NvMOTClassId classId)
	{
		if (m_defaultHomographyMatByClass.size() == 1)
		{
			return m_defaultHomographyMatByClass.begin()->second;
		}
		/// Fallback to first available homography matrix if no homography matrix is defined for the given class ID and print a warning.
		if (m_defaultHomographyMatByClass.find(classId) == m_defaultHomographyMatByClass.end())
		{
			if (m_params.m_debugVerboseLevel > 0) {
				std::cout << "[ObjectModelProjector] Warning: No homography for classId " << classId
						  << ", using first available" << std::endl;
			}
			return m_defaultHomographyMatByClass.begin()->second;
		}
		return m_defaultHomographyMatByClass.at(classId);
	}

	/**
	 * @brief Derive target-plane homographies from the camera projection matrix.
	 * @param[out] homographyMat Target homographies to populate.
	 * @param[in] humanModel Target model dimensions.
	 */
	void deriveDataFromCameraProjMatrix(TargetHomography& homographyMat, const Model3DInfo& humanModel);

	/**
	 * @brief Project a cylinder from an image-space waist point.
	 * @param[in] pt_img_waist Image-space waist point.
	 * @param[in] classId Target class ID.
	 * @param[in] targetInd Target index.
	 */
	std::vector<Eigen::Vector2d> calculateProjectedCylinderFromImgWaist(
		const Eigen::Vector2d pt_img_waist, const NvMOTClassId classId, const int targetInd
	);

	/**
	 * @brief Project a cylinder from an image-space head point.
	 * @param[in] pt_img_head Image-space head point.
	 * @param[in] classId Target class ID.
	 * @param[in] targetInd Target index.
	 */
	std::vector<Eigen::Vector2d> calculateProjectedCylinderFromImgHead(
		const Eigen::Vector2d pt_img_head, const NvMOTClassId classId, const int targetInd
	);

	/**
	 * @brief Project a cylinder from an image-space feet point.
	 * @param[in] pt_img_feet Image-space feet point.
	 * @param[in] classId Target class ID.
	 * @param[in] targetInd Target index.
	 */
	std::vector<Eigen::Vector2d> calculateProjectedCylinderFromImgFeet(
		const Eigen::Vector2d pt_img_feet, const NvMOTClassId classId, const int targetInd
	);

	/**
	 * @brief Project a cylinder from world-space feet coordinates.
	 * @param[in] pt_world_feet World-space feet position.
	 * @param[in] classId Target class ID.
	 * @param[in] targetInd Target index.
	 */
	std::vector<Eigen::Vector2d> calculateProjectedCylinderFromWorldFeet(
		const Eigen::Vector2d pt_world_feet, const NvMOTClassId classId, const int targetInd
	);

	/**
	 * @brief Return whether the current target model is valid for a target.
	 * @param[in] targetInd Target index.
	 * @param[in] classId Target class ID.
	 */
	bool isTargetModelValid(TrackerIndex targetInd, NvMOTClassId classId)
	{
		return m_targetModel.height > 0 && m_targetModel.radius > 0 && targetInd >= 0 && classId == m_targetClassId;
	}
	#ifdef VIZ_DEBUG
		friend class TrackerVizOutput;
	#endif
};

#endif // NVMULTIOBJECTTRACKERLIB_OBJECTMODELPROJECTOR_HPP_
