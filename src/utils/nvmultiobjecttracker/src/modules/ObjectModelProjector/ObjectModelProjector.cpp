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

#include <Eigen/Core>

#include <fstream>
#ifdef VIZ_DEBUG
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#endif

#include "modules/ObjectModelProjector/ObjectModelProjector.hpp"

using std::cout;
using std::endl;

/// Class validation is handled by getDefaultModelByClass.
#define IS_VALID_CLASS_ID(classId) {if(m_defaultModelByClass.find(classId) == m_defaultModelByClass.end())\
    {std::string msg = "Incorrect detected object class ID = " + std::to_string(classId) + ". Check class ids specified in " + m_resolvedCameraModelFilepath ;\
    CLOG(msg.c_str());throw std::runtime_error(msg);}}

/** @brief Construct an uninitialized object-model projector. */
ObjectModelProjector::ObjectModelProjector()
{
    m_isInit = false;
}

/**
 * @brief Construct an object-model projector from configuration.
 * @param[in] configFilePath Path to the configuration file.
 */
ObjectModelProjector::ObjectModelProjector(const std::string &configFilePath)
    : m_configFilePath(configFilePath)
{
    m_params = NvTrackerParamsObjectModelProjector(configFilePath, "ObjectModelProjection");
}

/**
 * @brief Construct and initialize a projector for a source stream.
 * @param[in] configFilePath Path to the configuration file.
 * @param[in] imgSize Source image size.
 * @param[in] streamId Source stream identifier.
 */
ObjectModelProjector::ObjectModelProjector(
    const std::string &configFilePath, const NvMotDataTypes::Size imgSize, const NvMOTStreamId streamId
)
    : ObjectModelProjector(configFilePath, imgSize, streamId, "")
{
}

/**
 * @brief Construct and initialize a projector for a named source sensor.
 * @param[in] configFilePath Path to the configuration file.
 * @param[in] imgSize Source image size.
 * @param[in] streamId Source stream identifier.
 * @param[in] sensorName Source sensor name.
 */
ObjectModelProjector::ObjectModelProjector(
    const std::string &configFilePath, const NvMotDataTypes::Size imgSize,
    const NvMOTStreamId streamId, const std::string &sensorName
)
    : m_configFilePath(configFilePath)
{
    m_srcFrameSize = imgSize;
    m_params = NvTrackerParamsObjectModelProjector(configFilePath, "ObjectModelProjection");
    m_streamInd = streamId >> 32;
    m_sensorName = sensorName;

    if (m_params.m_objectModelType == OBJ_MODEL_TYPE::BBOX3D)
    {
        if (m_params.m_proj3DBboxFilePath.size() > m_streamInd)
            importExternalProj3DBboxInfo(m_params.m_proj3DBboxFilePath[m_streamInd]);
    }

    if (!m_sensorName.empty() && !m_params.m_cameraModelFilepathMap.empty())
    {
        auto it = m_params.m_cameraModelFilepathMap.find(m_sensorName);
        if (it != m_params.m_cameraModelFilepathMap.end())
        {
            m_resolvedCameraModelFilepath = it->second;
        }
    }

    if (m_resolvedCameraModelFilepath.empty() && m_params.m_cameraModelFilepath.size() > m_streamInd)
    {
        m_resolvedCameraModelFilepath = m_params.m_cameraModelFilepath[m_streamInd];
    }

    if (!m_resolvedCameraModelFilepath.empty() && importCameraModelInfo(m_resolvedCameraModelFilepath))
    {
        m_isInit = true;

        if (m_params.m_debugVerboseLevel > 1)
        {
            cout << "Stream[" << m_streamInd << "]";
            if (!m_sensorName.empty())
                cout << " sensorName=" << m_sensorName;
            cout << " importCameraModelInfo(" << m_resolvedCameraModelFilepath << ") SUCCESS!" << endl;
        }
    }
    else
    {
        m_isInit = false;

        if (m_params.m_debugVerboseLevel > 1)
        {
            cout << "Stream[" << m_streamInd << "]";
            if (!m_sensorName.empty())
                cout << " sensorName=" << m_sensorName;
            cout << " importCameraModelInfo() FAIL !!" << endl;
        }
    }
}

/** @brief Destroy the object-model projector. */
ObjectModelProjector::~ObjectModelProjector()
{
}

/**
 * @brief Populate tracker query capabilities from projector configuration.
 * @param[out] paramsQuery Tracker query structure to update.
 */
void ObjectModelProjector::queryParams(NvMOTQuery &paramsQuery)
{
    paramsQuery.outputFootLocation = m_params.m_outputFootLocation;
    paramsQuery.outputConvexHull = m_params.m_outputConvexHull;
    paramsQuery.maxConvexHullSize = m_params.m_maxConvexHullSize;
}

/**
 * @brief Convert a nested YAML sequence to an Eigen matrix.
 * @param[in] yamlVec YAML matrix values.
 * @param[in] rows Matrix row count.
 * @param[in] cols Matrix column count.
 */
const Eigen::MatrixXd ObjectModelProjector::convertYamlVec2EigenMatrixXd(
    YAML::Node yamlVec, const int rows, const int cols
)
{
    if(yamlVec.size() != uint(rows*cols))
    {
        throw std::runtime_error(
            "[ObjectModelProjector::convertYamlVec2EigenMatrixXd] Invalid yaml input! Expected " + std::to_string(rows*cols) + " items, but got only " + std::to_string(yamlVec.size())
        );
    }

    Eigen::MatrixXd eMat(rows, cols);

    for(uint row = 0; row < uint(rows); row++)
    {
        for(uint col = 0; col < uint(cols); col++)
        {
            /// Input is row-major.
            eMat(row, col) = yamlVec[cols*row + col].as<double>();
        }
    }

    return eMat;
}

/**
 * @brief Import externally projected three-dimensional bounding boxes.
 * @param[in] filename Projected bounding-box file path.
 * @return True when the bounding boxes are imported successfully.
 */
bool ObjectModelProjector::importExternalProj3DBboxInfo(const std::string filename)
{
    m_externalProj3DBboxInfo.clear();

    if(filename.size() == 0 || !
#ifdef _WIN32
        std::filesystem::exists(filename)
#else
        std::experimental::filesystem::exists(filename)
#endif
    )
    {
        std::cout << "Stream[" << m_streamInd <<"][ObjectModelProjector::importExternalProj3DBboxInfo()] !!![ERROR] External 3D bbox info file doesn't exist: " << filename << std::endl;

        throw std::runtime_error("[ObjectModelProjector::importExternalProj3DBboxInfo] FAILED !!!");
    }

    /// Load labels from file.
    std::ifstream file(filename);
    int frame = 1;

    /// Read labels from a single file.
    while(file)
    {
        int num_token=0;
        std::string line, token, corner_token;
        std::vector<float> bbox_coord, corners;
        getline(file, line);
        std::stringstream ss_line(line);
        /// Read bbox from a line.
        while(getline(ss_line, token, ','))
        {
            if(num_token == 0)
            {
                /// In ground truth label files staring from 1, but in DS starting from 0.
                frame = stoi(token);
                if(frame < 1)
                {
                    std::cout << "Stream[" << m_streamInd <<"][ObjectModelProjector::importExternalProj3DBboxInfo()] !!![ERROR] frame number should be greater or equal than 1 " << std::endl;
                    throw std::runtime_error("[ObjectModelProjector::importExternalProj3DBboxInfo] FAILED !!!");
                }

                /// Add new frames to stream_bboxes.
                for(int i = m_externalProj3DBboxInfo.size(); i < frame; i++)
                {
                    m_externalProj3DBboxInfo.push_back(std::vector<Proj3DBboxInfo>());
                    if(m_params.m_debugVerboseLevel > 1)
                    {
                        cout << "[ObjectModelProjector::importExternalProj3DBboxInfo()] frame: " << frame << ", m_externalProj3DBboxInfo size: " << m_externalProj3DBboxInfo.size() << endl;
                    }
                }
            }
            else if(num_token >= 2 && num_token <= 6)
            {
                /// Token [2, 5] bbox, [6] confidence.
                bbox_coord.push_back(stof(token));
            }
            else if(num_token == 14)
            {
                /// Token [14] corners.
                std::stringstream ss_token(token);
                while(getline(ss_token, corner_token, '|'))
                {
                    corners.push_back(stof(corner_token));
                }

            }
            num_token++;
        }

        if (bbox_coord.empty())
        {
            continue;
        }

        NvMotDataTypes::Rect2f bbox = CV_FUNCS::constructRect2f(
            bbox_coord.at(0), bbox_coord.at(1), bbox_coord.at(2), bbox_coord.at(3)
        );
        Eigen::Vector2f bbox_center = (bbox.min() + bbox.max()) * 0.5;
        std::vector<Eigen::Vector2f> projCorners;
        for (auto i = 0; i < 8; i++)
        {
            projCorners.push_back(
                Eigen::Vector2f( bbox_center(0) + corners.at(2*i), bbox_center(1) + corners.at(2*i+1) )
            );

        }
        if(m_params.m_debugVerboseLevel > 1)
        {
            cout << "bbox: [" << bbox.min()(0) << ", " << bbox.min()(1) << ", " << bbox.max()(0) << ", " << bbox.max()(1) << "], ";
            cout << "corners: [" << projCorners.at(0)(0) << ", " << projCorners.at(0)(1)  << "...] " << endl;
        }
#ifdef _WIN32
        Proj3DBboxInfo projectedBbox;
        projectedBbox.bbox = bbox;
        projectedBbox.projCorners = projCorners;
        m_externalProj3DBboxInfo[frame-1].push_back(std::move(projectedBbox));
#else
        m_externalProj3DBboxInfo[frame-1].push_back({bbox, projCorners});
#endif

    }
    if(m_params.m_debugVerboseLevel > 1)
    {
        cout << "[ObjectModelProjector::importExternalProj3DBboxInfo()] returned true" << endl;
    }
    return true;
}

/**
 * @brief Import camera-model calibration from a YAML file.
 * @param[in] filename Camera-model file path.
 * @return True when the camera model is imported successfully.
 */
bool ObjectModelProjector::importCameraModelInfo(const std::string filename)
{
    if(filename.size() == 0 || !
#ifdef _WIN32
        std::filesystem::exists(filename)
#else
        std::experimental::filesystem::exists(filename)
#endif
    )
    {
        std::cout << "Stream[" << m_streamInd <<"][ObjectModelProjector::importCameraModelInfo()] !!![ERROR] Camera model file doesn't exist: " << filename << std::endl;

        throw std::runtime_error("[ObjectModelProjector::importCameraModelInfo] FAILED !!!");
    }

    YAML::Node cameraModelYaml = YAML::LoadFile(filename);

    /// DEBUG.
    if(m_params.m_debugVerboseLevel > 1)
    {
        cout << "Stream[" << m_streamInd <<"][ObjectModelProjector::importCameraModelInfo] cameraModelYaml.size() = " << cameraModelYaml.size() << endl;
    }

    if (!cameraModelYaml["modelInfo"] ||
        ( !cameraModelYaml["modelInfo"].IsSequence() &&
        ( !cameraModelYaml["modelInfo"]["height"] || !cameraModelYaml["modelInfo"]["radius"] ))
    )
    {
        throw std::runtime_error("[ObjectModelProjector::importCameraModelInfo] No valid modelInfo is provided!");
    }
    else if (cameraModelYaml["modelInfo"].IsSequence())
    {
        /// Per class model info.
        const YAML::Node& modelsYaml = cameraModelYaml["modelInfo"];
        NvMOTClassId classId = 0;
        for (const auto& modelYaml : modelsYaml) {
            if(!modelYaml["classID"] || !modelYaml["height"] || !modelYaml["radius"])
            {
                throw std::runtime_error(
                    "[ObjectModelProjector::importCameraModelInfo] No valid modelInfo is provided!"
                );
            }
            else
            {
                classId = (NvMOTClassId)modelYaml["classID"].as<int>();
                m_defaultModelByClass[classId].height = (float)modelYaml["height"].as<double>();
                m_defaultModelByClass[classId].radius = (float)modelYaml["radius"].as<double>();
            }
            if(m_params.m_debugVerboseLevel > 1)
            {
                cout << "m_defaultModelByClass: classId, height, radius = " << classId << "," << m_defaultModelByClass[classId].height << ", " << m_defaultModelByClass[classId].radius << endl;
            }
        }
    }
    else
    {
        /// Default model info.
        NvMOTClassId classId = 0;
        m_defaultModelByClass[classId].height = (float)cameraModelYaml["modelInfo"]["height"].as<double>();
        m_defaultModelByClass[classId].radius = (float)cameraModelYaml["modelInfo"]["radius"].as<double>();
    }

    /// Override model height/radius if user specified custom values via params.
    for (auto& it : m_defaultModelByClass)
    {
        if (m_params.m_useDefaultModelHeight > 0) it.second.height = m_params.m_useDefaultModelHeight;
        if (m_params.m_useDefaultModelRadius > 0) it.second.radius = m_params.m_useDefaultModelRadius;
        if (m_params.m_debugVerboseLevel > 1 && (m_params.m_useDefaultModelHeight > 0 || m_params.m_useDefaultModelRadius > 0))
        {
            cout << "m_defaultModelByClass (after override): classId, height, radius = " << it.first << "," << it.second.height << ", " << it.second.radius << endl;
        }
    }

    if (cameraModelYaml["groundPlaneHeight"])
    {
        m_groundPlaneHeight = (float)cameraModelYaml["groundPlaneHeight"].as<double>();
    }
    else
    {
        m_groundPlaneHeight = 0.0;
    }

    if(cameraModelYaml["projectionMatrix_3x4_w2p"])
    {
        /// Import world-to-camera pixel projection matrix.
        m_ProjMat_World2Cam = convertYamlVec2EigenMatrixXd(cameraModelYaml["projectionMatrix_3x4_w2p"], 3, 4);

        m_pix2cam_offset = Eigen::Vector2d(0, 0);
    }
    else if(cameraModelYaml["projectionMatrix_3x4"])
    {
        /// Import world-to-camera optical center projection matrix.
        m_ProjMat_World2Cam = convertYamlVec2EigenMatrixXd(cameraModelYaml["projectionMatrix_3x4"], 3, 4);

        m_pix2cam_offset = Eigen::Vector2d(m_srcFrameSize.width/2., m_srcFrameSize.height/2.);
    }
    else
    {
        throw std::runtime_error(
            "[ObjectModelProjector::importCameraModelInfo] No valid projectionMatrix_3x4 or projectionMatrix_3x4_w2p is provided!"
        );
    }

    if(m_params.m_debugVerboseLevel > 1)
    {
        cout << "m_ProjMat_World2Cam = " << endl << m_ProjMat_World2Cam << endl;
    }

    /// Parse intrinsic and extrinsic matrices for uncertainty propagation.
    if (cameraModelYaml["intrinsicMatrix_3x3"] && cameraModelYaml["rotationMatrix_3x3"] && cameraModelYaml["translationVector_3x1"])
    {
        m_K = convertYamlVec2EigenMatrixXd(cameraModelYaml["intrinsicMatrix_3x3"], 3, 3);
        m_R = convertYamlVec2EigenMatrixXd(cameraModelYaml["rotationMatrix_3x3"], 3, 3);
        m_t = convertYamlVec2EigenMatrixXd(cameraModelYaml["translationVector_3x1"], 3, 1);

        if(m_params.m_debugVerboseLevel > 1)
        {
            cout << "Stream[" << m_streamInd << "] Loaded explicit camera matrices (K, R, t)" << endl;
        }
        m_K_inv = m_K.inverse();
    }
    else
    {
        /// Decompose P = K[R|t] using RQ decomposition of left 3x3 block.
        Eigen::Matrix3d M = m_ProjMat_World2Cam.topLeftCorner(3, 3);
        Eigen::Matrix3d M_T = M.transpose();
        Eigen::HouseholderQR<Eigen::Matrix3d> qr(M_T);
        Eigen::Matrix3d Q = qr.householderQ();
        Eigen::Matrix3d R_qr = qr.matrixQR().triangularView<Eigen::Upper>();

        m_K = R_qr.transpose();
        m_R = Q.transpose();

        /// Ensure K has positive diagonal (adjust signs if needed).
        for (int i = 0; i < 3; i++) {
            if (m_K(i, i) < 0) {
                m_K.col(i) *= -1;
                m_R.row(i) *= -1;
            }
        }

        if(m_params.m_debugVerboseLevel > 1)
        {
            cout << "Stream[" << m_streamInd << "] Decomposed camera matrices from projection matrix" << endl;
        }

        m_K_inv = m_K.inverse();
        m_t = m_K_inv * m_ProjMat_World2Cam.col(3);
    }

    m_R_T = m_R.transpose();
    m_r3 = m_R_T.row(2);
    m_camLocation_World = -m_R_T * m_t;

    /// Precompute ground plane homography and Sherman-Morrison components.
    m_H_ground.col(0) = m_ProjMat_World2Cam.col(0);
    m_H_ground.col(1) = m_ProjMat_World2Cam.col(1);
    m_H_ground.col(2) = m_ProjMat_World2Cam.col(3);
    m_H_ground_inv = m_H_ground.inverse();

    /// Precompute Sherman-Morrison components for efficient height-plane homography inverses.
    m_P_z_col = m_ProjMat_World2Cam.col(2);
    m_H_inv_Pz = m_H_ground_inv * m_P_z_col;

    if(m_params.m_debugVerboseLevel > 1)
    {
        cout << "Stream[" << m_streamInd << "] K = " << endl << m_K << endl;
        cout << "Stream[" << m_streamInd << "] R = " << endl << m_R << endl;
        cout << "Stream[" << m_streamInd << "] t = " << m_t.transpose() << endl;
        cout << "Stream[" << m_streamInd << "] Camera location in world = " << m_camLocation_World.transpose() << endl;
    }

    /// Given the camera projection matrix and the offsets and scale for camera-to-mapped view homography, calculate all the necessary homography matrices.
    for (auto it = m_defaultModelByClass.begin(); it != m_defaultModelByClass.end(); it++)
    {
        const NvMOTClassId& classId = it->first;
        const Model3DInfo& defaultModel = it->second;
        deriveDataFromCameraProjMatrix(m_defaultHomographyMatByClass[classId], defaultModel);
    }

    m_targetModel = m_defaultModelByClass.begin()->second;
    m_targetHomographyMat = m_defaultHomographyMatByClass.begin()->second;
    m_targetClassId = m_defaultModelByClass.begin()->first;
    return true;
}

/**
 * @brief Derive target-plane homographies from the camera projection matrix.
 * @param[out] homographyMat Target homographies to populate.
 * @param[in] humanModel Target model dimensions.
 */
void ObjectModelProjector::deriveDataFromCameraProjMatrix(
    TargetHomography& homographyMat, const Model3DInfo& humanModel
)
{
    const float waistRatio = m_params.m_waistHeightRatio;
    const float z_waist = humanModel.height * waistRatio;
    const float z_head = humanModel.height;

    /// The world ground plane-to-camera's optical center homography (use precomputed).
    homographyMat.WorldFeetPlane2CamOpticalCenter = m_H_ground;
    homographyMat.CamOpticalCenter2WorldFeetPlane = m_H_ground_inv;

    /// The world waist plane-to-camera's optical center homography.
    homographyMat.WorldWaistPlane2CamOpticalCenter = m_H_ground;
    homographyMat.WorldWaistPlane2CamOpticalCenter.col(2) += z_waist * m_P_z_col;

    /// Inverse using Sherman-Morrison: (A + z*u*v^T)^-1 = A^-1 - z*(A^-1*u)*(v^T*A^-1)/(1 + z*v^T*A^-1*u) where v = e3, so v^T*A^-1 = A^-1.row(2), and v^T*A^-1*u = w(2).
    double denom_waist = 1.0 + z_waist * m_H_inv_Pz(2);
    homographyMat.CamOpticalCenter2WorldWaistPlane = m_H_ground_inv - (z_waist / denom_waist) * m_H_inv_Pz * m_H_ground_inv.row(2);

    /// The world head plane-to-camera's optical center homography.
    homographyMat.WorldHeadPlane2CamOpticalCenter = m_H_ground;
    homographyMat.WorldHeadPlane2CamOpticalCenter.col(2) += z_head * m_P_z_col;

    double denom_head = 1.0 + z_head * m_H_inv_Pz(2);
    homographyMat.CamOpticalCenter2WorldHeadPlane = m_H_ground_inv - (z_head / denom_head) * m_H_inv_Pz * m_H_ground_inv.row(2);

    if(m_params.m_debugVerboseLevel > 1)
    {
        cout << "Stream[" << m_streamInd <<"] homographyMat.WorldHeadPlane2CamOpticalCenter = " << endl << homographyMat.WorldHeadPlane2CamOpticalCenter << endl;
    }

    /// Composite Homographies (use precomputed inverses).
    homographyMat.Feet2Waist = homographyMat.WorldWaistPlane2CamOpticalCenter * homographyMat.CamOpticalCenter2WorldFeetPlane;
    homographyMat.Waist2Feet = homographyMat.WorldFeetPlane2CamOpticalCenter * homographyMat.CamOpticalCenter2WorldWaistPlane;
    homographyMat.Waist2Head = homographyMat.WorldHeadPlane2CamOpticalCenter * homographyMat.CamOpticalCenter2WorldWaistPlane;
    homographyMat.Head2Waist = homographyMat.WorldWaistPlane2CamOpticalCenter * homographyMat.CamOpticalCenter2WorldHeadPlane;

    /// Check the re-projection error: World ground --> camera image --> world ground.
    if(m_params.m_debugVerboseLevel > 1)
    {
        Eigen::Vector2d pt_world{0., 0.};
        Eigen::Vector3d pt_img_homogeneous = homographyMat.WorldFeetPlane2CamOpticalCenter * pt_world.homogeneous();
        Eigen::Vector2d pt_img{pt_img_homogeneous(0)/pt_img_homogeneous(2), pt_img_homogeneous(1)/pt_img_homogeneous(2)};

        Eigen::Vector3d pt_world_homogeneous = homographyMat.CamOpticalCenter2WorldFeetPlane * pt_img.homogeneous();
        Eigen::Vector2d pt_worldg_reproj; pt_worldg_reproj << pt_world_homogeneous(0)/pt_world_homogeneous(2), pt_world_homogeneous(1)/pt_world_homogeneous(2);

        cout << "Stream[" << m_streamInd <<"] [projMat_WorldGround2Cam] world origin on ground (0 0)\n";
        cout << "                                    --> image pt (" << pt_img.transpose() << ")\n";
        cout << "                                    --> world pt on ground (" << pt_worldg_reproj.transpose() << ")\n";
    }

    /// Check the re-projection error: camera optical center on image --> world ground --> camera image.
    if(m_params.m_debugVerboseLevel > 1)
    {
        /// Camera's optical center.
        Eigen::Vector2d pt_img{0., 0.};
        Eigen::Vector3d pt_world_homogeneous = homographyMat.CamOpticalCenter2WorldFeetPlane * pt_img.homogeneous();
        Eigen::Vector2d pt_world{pt_world_homogeneous(0)/pt_world_homogeneous(2), pt_world_homogeneous(1)/pt_world_homogeneous(2)};

        Eigen::Vector3d pt_img_homogeneous = homographyMat.WorldFeetPlane2CamOpticalCenter * pt_world.homogeneous();
        Eigen::Vector2d pt_img_reproj; pt_img_reproj << pt_img_homogeneous(0)/pt_img_homogeneous(2), pt_img_homogeneous(1)/pt_img_homogeneous(2);

        cout << "Stream[" << m_streamInd <<"] [projMat_Cam2WorldGround] camera optical center (" << pt_img.transpose() << ")\n";
        cout << "                                    --> world ground pt (" << pt_world.transpose() << ")\n";
        cout << "                                    --> camera optical center (" << pt_img_reproj.transpose() << ")\n";

        /// Camera's location in world.
        cout << "Stream[" << m_streamInd <<"] [projMat_World2Cam] camera location in world (" << m_camLocation_World.transpose() << ")\n";
    }
}

/**
 * @brief Project a world-space point into image coordinates.
 * @param[in] pt_world World-space point.
 */
const Eigen::Vector2d ObjectModelProjector::projectWorldPt2ImgPt(const Eigen::Vector3d pt_world)
{
    Eigen::Vector3d pt_img = m_ProjMat_World2Cam * pt_world.homogeneous();

    /// Perform world-to-camera transformation: note: This conversion is with respect to the camera's optical center.
    Eigen::Vector2d pt_img_cv = Eigen::Vector2d(pt_img(0)/pt_img(2), pt_img(1)/pt_img(2));

    /// Move to the camera center to align the camera center to the world center.
    pt_img_cv(0) += m_pix2cam_offset(0);
    pt_img_cv(1) += m_pix2cam_offset(1);

    return pt_img_cv;
}

/**
 * @brief Estimate target height and world feet position from image head and waist points.
 * @param[in] pt_img_head Image-space head point.
 * @param[in] pt_img_waist Image-space waist point.
 * @param[in] targetInd Target index.
 * @param[in] classId Target class ID.
 */
Eigen::Vector2d ObjectModelProjector::computeHeightAndWorldFeetFromHeadAndWaist(
    const Eigen::Vector2d pt_img_head, const Eigen::Vector2d pt_img_waist,
    const TrackerIndex &targetInd, const NvMOTClassId classId
)
{
    /// Using Direct linear transformation to estimate body height H and world (X, Y) from eyes and hip keypoints.
    /// Eyes projection: m_ProjMat * (X, Y, reh*H, 1) -> (x1, y1, 1), with reh = eyeHeightRatio (eye height / total height).
    /// Hip projection:  m_ProjMat * (X, Y, rwh*H, 1) -> (x2, y2, 1), with rwh = waistHeightRatio (hip height / total height).
    /// Solve the linear system for X, Y, H:
    /// m00-x1m20, m01-x1m21, reh(m02-x1m22)  | X   -m03+x1m23
    /// m10-y1m20, m11-y1m21, reh(m12-y1m22)  | Y = -m13+y1m23
    /// m00-x2m20, m01-x2m21, rwh(m02-x2m22)  | H   -m03+x2m23
    /// m10-y2m20, m11-y2m21, rwh(m12-y2m22)  |     -m13+y2m23
    const float rwh = m_params.m_waistHeightRatio;
    const float reh = m_params.m_eyeHeightRatio;
    const float x1 = pt_img_head(0), y1 = pt_img_head(1), x2 = pt_img_waist(0), y2 = pt_img_waist(1);
    const CameraMatrix &m = m_ProjMat_World2Cam;
    Eigen::MatrixXd A(4, 3);
    A << m(0,0) - x1 * m(2,0), m(0,1) - x1 * m(2,1), reh * (m(0,2) - x1 * m(2,2)),
         m(1,0) - y1 * m(2,0), m(1,1) - y1 * m(2,1), reh * (m(1,2) - y1 * m(2,2)),
         m(0,0) - x2 * m(2,0), m(0,1) - x2 * m(2,1), rwh * (m(0,2) - x2 * m(2,2)),
         m(1,0) - y2 * m(2,0), m(1,1) - y2 * m(2,1), rwh * (m(1,2) - y2 * m(2,2));

    Eigen::Vector4d b(4);
    b << -m(0,3) + x1 * m(2,3), -m(1,3) + y1 * m(2,3), -m(0,3) + x2 * m(2,3), -m(1,3) + y2 * m(2,3);

    Eigen::Vector3d sol = A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);
    float height_new = static_cast<float>(sol(2));
    Model3DInfo defaultModel = getDefaultModelByClass(classId);
    m_targetModel.height = std::max(std::min(height_new, defaultModel.height * 1.3f), defaultModel.height * 0.7f);
    if(m_params.m_debugVerboseLevel > 1)
    {
        cout << "Target[" << targetInd << "] Original model height = " << defaultModel.height
                                              << ", updated height = " << m_targetModel.height <<endl;
    }

    Eigen::Vector2d pt_world; pt_world << sol(0), sol(1);
    return pt_world;
}

/**
 * @brief Estimate target height and world feet position from image head and foot points.
 * @param[in] pt_img_head Image-space head point.
 * @param[in] pt_img_foot Image-space foot point.
 * @param[in] targetInd Target index.
 * @param[in] classId Target class ID.
 */
Eigen::Vector2d ObjectModelProjector::computeHeightAndWorldFeetFromHeadAndFoot(
    const Eigen::Vector2d pt_img_head, const Eigen::Vector2d pt_img_foot,
    const TrackerIndex &targetInd, const NvMOTClassId classId
)
{
    /// World feet (X, Y) from existing back-projection; ground-plane homography is height-independent.
    Eigen::Vector2d pt_world = getWorldFeetPtOnGroundFromCamFeetPt(pt_img_foot, classId, targetInd);
    const double X = pt_world(0), Y = pt_world(1);

    /// With (X,Y) known, solve for H only: head at (X, Y, reh*H) projects to pt_img_head.
    const float reh = m_params.m_eyeHeightRatio;
    const double x1 = pt_img_head(0), y1 = pt_img_head(1);
    const CameraMatrix &m = m_ProjMat_World2Cam;

    const double denom_x = m(2,0)*X + m(2,1)*Y + m(2,3);
    const double num_x   = x1 * denom_x - (m(0,0)*X + m(0,1)*Y + m(0,3));
    const double coeff_x = reh * (m(0,2) - x1 * m(2,2));
    const double denom_y = denom_x;
    const double num_y   = y1 * denom_x - (m(1,0)*X + m(1,1)*Y + m(1,3));
    const double coeff_y = reh * (m(1,2) - y1 * m(2,2));

    double height_new = 0;
    const double denom_H = coeff_x*coeff_x + coeff_y*coeff_y;
    if (denom_H > 1e-12)
        height_new = (coeff_x*num_x + coeff_y*num_y) / denom_H;
    else
        height_new = (std::abs(coeff_x) > 1e-12) ? (num_x / coeff_x) : ((std::abs(coeff_y) > 1e-12) ? (num_y / coeff_y) : 0);

    Model3DInfo defaultModel = getDefaultModelByClass(classId);
    m_targetModel.height = std::max(
        std::min(static_cast<float>(height_new), defaultModel.height * 1.3f), defaultModel.height * 0.7f
    );
    if (m_params.m_debugVerboseLevel > 1)
    {
        cout << "Target[" << targetInd << "] Head+foot: world feet = (" << pt_world.transpose() << "), height = " << m_targetModel.height << endl;
    }
    return pt_world;
}

/**
 * @brief Check whether a vertical target projects into the camera field of view.
 * @param[in] pt_world_feet World coordinates of feet (X, Y on ground plane).
 * @param[in] targetHeight Actual height of the target (uses adaptive height if available).
 * @return True if any part of the target's vertical extent is visible in the image.
 */
bool ObjectModelProjector::isTargetVisibleInImage(const Eigen::Vector2d& pt_world_feet, float targetHeight)
{
    const float w = static_cast<float>(m_srcFrameSize.width);
    const float h = static_cast<float>(m_srcFrameSize.height);

    /// Project feet (z=0) and head (z=targetHeight).
    Eigen::Vector3d world_feet(pt_world_feet(0), pt_world_feet(1), 0);
    Eigen::Vector3d world_head(pt_world_feet(0), pt_world_feet(1), targetHeight);
    Eigen::Vector2d img_feet = projectWorldPt2ImgPt(world_feet);
    Eigen::Vector2d img_head = projectWorldPt2ImgPt(world_head);

    /// Check if either endpoint is inside image bounds.
    auto isInside = [w, h](const Eigen::Vector2d& p) {
        return p(0) >= 0 && p(0) < w && p(1) >= 0 && p(1) < h;
    };

    return isInside(img_feet) || isInside(img_head);
}

/**
 * @brief Project a cylinder from an image-space head point.
 * @param[in] imgHeadPt Image-space head point.
 * @param[in] classId Target class ID.
 * @param[in] targetInd Target index.
 */
std::vector<Eigen::Vector2d> ObjectModelProjector::calculateProjectedCylinderFromImgHead(
    const Eigen::Vector2d imgHeadPt, const NvMOTClassId classId, const int targetInd
)
{
    /// Convert camera image coordinates to be based on the camera's optical center, instead of the left-top pt.
    Eigen::Vector3d pt_img_head_opticalCenter; pt_img_head_opticalCenter << imgHeadPt(0) - m_pix2cam_offset(0), imgHeadPt(1) - m_pix2cam_offset(1), 1;

    HomographyMatrix homographyMat_CamOpticalCenter2WorldHeadPlane;
    if (isTargetModelValid(targetInd, classId))
    {
        homographyMat_CamOpticalCenter2WorldHeadPlane = m_targetHomographyMat.CamOpticalCenter2WorldHeadPlane;
    }
    else
    {
        homographyMat_CamOpticalCenter2WorldHeadPlane = getDefaultHomographyMatByClass(classId).CamOpticalCenter2WorldHeadPlane;
    }

    Eigen::Vector3d pt_world_homogeneous = homographyMat_CamOpticalCenter2WorldHeadPlane * pt_img_head_opticalCenter;
    Eigen::Vector2d pt_world = Eigen::Vector2d(
        pt_world_homogeneous(0)/pt_world_homogeneous(2), pt_world_homogeneous(1)/pt_world_homogeneous(2)
    );

    return calculateProjectedCylinderFromWorldFeet(pt_world, classId, targetInd);
}

/**
 * @brief Project a cylinder from an image-space waist point.
 * @param[in] imgWaistPt Image-space waist point.
 * @param[in] classId Target class ID.
 * @param[in] targetInd Target index.
 */
std::vector<Eigen::Vector2d> ObjectModelProjector::calculateProjectedCylinderFromImgWaist(
    const Eigen::Vector2d imgWaistPt, const NvMOTClassId classId, const int targetInd
)
{
    /// Convert camera image coordinates to be based on the camera's optical center, instead of the left-top pt.
    Eigen::Vector3d pt_img_waist_opticalCenter; pt_img_waist_opticalCenter << imgWaistPt(0) - m_pix2cam_offset(0), imgWaistPt(1) - m_pix2cam_offset(1), 1;

    HomographyMatrix homographyMat_CamOpticalCenter2WorldWaistPlane;
    if (isTargetModelValid(targetInd, classId))
    {
        homographyMat_CamOpticalCenter2WorldWaistPlane = m_targetHomographyMat.CamOpticalCenter2WorldWaistPlane;
    }
    else
    {
        homographyMat_CamOpticalCenter2WorldWaistPlane = getDefaultHomographyMatByClass(classId).CamOpticalCenter2WorldWaistPlane;
    }

    /// On world waist plane.
    Eigen::Vector3d pt_world_homogeneous = homographyMat_CamOpticalCenter2WorldWaistPlane * pt_img_waist_opticalCenter;
    Eigen::Vector2d pt_world(
        pt_world_homogeneous(0)/pt_world_homogeneous(2), pt_world_homogeneous(1)/pt_world_homogeneous(2)
    );

    return calculateProjectedCylinderFromWorldFeet(pt_world, classId, targetInd);
}

/**
 * @brief Project a cylinder from an image-space feet point.
 * @param[in] imgFeetPt Image-space feet point.
 * @param[in] classId Target class ID.
 * @param[in] targetInd Target index.
 */
std::vector<Eigen::Vector2d> ObjectModelProjector::calculateProjectedCylinderFromImgFeet(
    const Eigen::Vector2d imgFeetPt, const NvMOTClassId classId, const int targetInd
)
{
    /// Convert camera image coordinates to be based on the camera's optical center, instead of the left-top pt.
    Eigen::Vector3d pt_img_feet_opticalCenter; pt_img_feet_opticalCenter << imgFeetPt(0) - m_pix2cam_offset(0), imgFeetPt(1) - m_pix2cam_offset(1), 1;

    HomographyMatrix homographyMat_CamOpticalCenter2WorldFeetPlane;
    if (isTargetModelValid(targetInd, classId))
    {
        homographyMat_CamOpticalCenter2WorldFeetPlane = m_targetHomographyMat.CamOpticalCenter2WorldFeetPlane;
    }
    else
    {
        homographyMat_CamOpticalCenter2WorldFeetPlane = getDefaultHomographyMatByClass(classId).CamOpticalCenter2WorldFeetPlane;
    }

    /// On world feet plane.
    Eigen::Vector3d pt_world_homogeneous = homographyMat_CamOpticalCenter2WorldFeetPlane * pt_img_feet_opticalCenter;
    Eigen::Vector2d pt_world(
        pt_world_homogeneous(0)/pt_world_homogeneous(2), pt_world_homogeneous(1)/pt_world_homogeneous(2)
    );

    return calculateProjectedCylinderFromWorldFeet(pt_world, classId, targetInd);
}

/**
 * @brief Project a horizontal circle centered at world-space feet coordinates.
 * @param[in] pt_world_feet World-space feet position.
 * @param[in] radius Circle radius.
 */
std::vector<Eigen::Vector2d> ObjectModelProjector::calculateProjectedCircleFromWorldFeet(
    const Eigen::Vector2d pt_world_feet, const float radius
)
{
    const int angleNum = m_params.m_maxConvexHullSize;

    std::vector<Eigen::Vector2d> projected_circle_pt;

    /// On world feet plane.
    Eigen::Vector2d pt_world = pt_world_feet;

    Eigen::Vector3d pt_world_circle, pt_img_feet_circle_opticCenter_homogeneous;

    for(float piDeg = 0; piDeg < 360; piDeg += 360.f/angleNum)
    {
        Eigen::Vector4d pt_world_feet_circle; pt_world_feet_circle << pt_world(0) + radius*cos(piDeg/180*M_PI), pt_world(1) + radius*sin(piDeg/180*M_PI), 0, 1;

        pt_img_feet_circle_opticCenter_homogeneous = m_ProjMat_World2Cam * pt_world_feet_circle;

        Eigen::Vector2d pt_img_feet_circle_opticCenter(
            pt_img_feet_circle_opticCenter_homogeneous(0)/pt_img_feet_circle_opticCenter_homogeneous(2),
            pt_img_feet_circle_opticCenter_homogeneous(1)/pt_img_feet_circle_opticCenter_homogeneous(2)
        );

        /// Convert the camera optical center coordinate into the camera image coordinate by translation.
        Eigen::Vector2d pt_img_feet_circle(
            pt_img_feet_circle_opticCenter(0) + m_pix2cam_offset(0),
            pt_img_feet_circle_opticCenter(1) + m_pix2cam_offset(1)
        );

        projected_circle_pt.push_back(pt_img_feet_circle);
    }

    return projected_circle_pt;
}

/**
 * @brief Project a cylinder from world-space feet coordinates.
 * @param[in] pt_world_feet World-space feet position.
 * @param[in] classId Target class ID.
 * @param[in] targetInd Target index.
 */
std::vector<Eigen::Vector2d> ObjectModelProjector::calculateProjectedCylinderFromWorldFeet(
    const Eigen::Vector2d pt_world_feet, const NvMOTClassId classId, const int targetInd
)
{
    const int angleNum = m_params.m_maxConvexHullSize;

    std::vector<Eigen::Vector2d> projected_cylinder_pt_unsorted;

    /// On world feet plane.
    Eigen::Vector2d pt_world = pt_world_feet;
    Eigen::Vector3d pt_world_circle, pt_img_head_circle_opticCenter_homogeneous, pt_img_feet_circle_opticCenter_homogeneous;

    Model3DInfo humanModel;
    if (isTargetModelValid(targetInd, classId))
    {
        humanModel = m_targetModel;
    }
    else
    {
        humanModel = getDefaultModelByClass(classId);
    }

    for(float piDeg = 0; piDeg < 360; piDeg += 360.f/angleNum)
    {
        Eigen::Vector4d pt_world_feet_circle; pt_world_feet_circle << pt_world(0) + humanModel.radius*cos(piDeg/180*M_PI), pt_world(1) + humanModel.radius*sin(piDeg/180*M_PI), 0, 1;
        Eigen::Vector4d pt_world_head_circle; pt_world_head_circle << pt_world_feet_circle(0), pt_world_feet_circle(1), humanModel.height, 1;

        pt_img_head_circle_opticCenter_homogeneous = m_ProjMat_World2Cam * pt_world_head_circle;
        pt_img_feet_circle_opticCenter_homogeneous = m_ProjMat_World2Cam * pt_world_feet_circle;

        Eigen::Vector2d pt_img_head_circle_opticCenter(
            pt_img_head_circle_opticCenter_homogeneous(0)/pt_img_head_circle_opticCenter_homogeneous(2),
            pt_img_head_circle_opticCenter_homogeneous(1)/pt_img_head_circle_opticCenter_homogeneous(2)
        );
        Eigen::Vector2d pt_img_feet_circle_opticCenter(
            pt_img_feet_circle_opticCenter_homogeneous(0)/pt_img_feet_circle_opticCenter_homogeneous(2),
            pt_img_feet_circle_opticCenter_homogeneous(1)/pt_img_feet_circle_opticCenter_homogeneous(2)
        );

        /// Convert the camera optical center coordinate into the camera image coordinate by translation.
        Eigen::Vector2d pt_img_feet_circle(
            pt_img_feet_circle_opticCenter(0) + m_pix2cam_offset(0),
            pt_img_feet_circle_opticCenter(1) + m_pix2cam_offset(1)
        );
        Eigen::Vector2d pt_img_head_circle(
            pt_img_head_circle_opticCenter(0) + m_pix2cam_offset(0),
            pt_img_head_circle_opticCenter(1) + m_pix2cam_offset(1)
        );

        projected_cylinder_pt_unsorted.push_back(pt_img_feet_circle);
        projected_cylinder_pt_unsorted.push_back(pt_img_head_circle);
    }

    #if !defined(VIZ_DEBUG)
        return projected_cylinder_pt_unsorted;
    #else

        std::vector<Eigen::Vector2d> projected_cylinder_convexHull;

        CV_FUNCS::convexHull(projected_cylinder_pt_unsorted, projected_cylinder_convexHull);

        return projected_cylinder_convexHull;
    #endif
}

/**
 * @brief Recover an object model from externally projected three-dimensional bounding-box data.
 * @param[in] detObj Detected object to match.
 * @param[in] frameNum Current frame number.
 */
RecoveredObjModel ObjectModelProjector::getRecoveredObjectModelFromProj3DBbox(
    const NvMOTObjToTrack *detObj, const NvMOTFrameNum frameNum
)
{

    /// Tracking 3D by accepting 3D detections.
    RecoveredObjModel refinedMot = {};
    if (frameNum >= m_externalProj3DBboxInfo.size() || m_externalProj3DBboxInfo.at(frameNum).empty())
    {
        return refinedMot;
    }

    const NvMotDataTypes::Rect2f bbox = CV_FUNCS::constructRect2f(
        detObj->bbox.x, detObj->bbox.y, detObj->bbox.width, detObj->bbox.height
    );
    Eigen::Vector2f pt_img_center = (bbox.min() + bbox.max()) * 0.5;

    auto itBest = m_externalProj3DBboxInfo.at(frameNum).begin();
    float iouBest = -0.1;
    for (auto it = m_externalProj3DBboxInfo.at(frameNum).begin(); it != m_externalProj3DBboxInfo.at(frameNum).end(); it++)
    {
        float iou = CV_FUNCS::computeIOU(bbox, it->bbox);
        if (iou > iouBest)
        {
            itBest = it;
            iouBest = iou;
        }
    }
    if (m_params.m_debugVerboseLevel > 1 && frameNum < m_externalProj3DBboxInfo.size())
    {
        cout << "Number of points: " << itBest->projCorners.size() << ", number of objs in frame: " << m_externalProj3DBboxInfo.at(frameNum).size() <<
            ", frame num: " << frameNum << ", streamid: " << m_streamInd << ", iou: " << iouBest << endl;
    }
    if (itBest == m_externalProj3DBboxInfo.at(frameNum).end())
    {
        cout << "bbox not found: " << bbox.min() << " " << bbox.max() << ", frame num: " << frameNum <<endl;
        throw std::runtime_error(
            "[ObjectModelProjector::getRecoveredObjectModelFromProj3DBbox] Unable to found 3D bbox for detection input"
        );

    }
    Eigen::Vector2d pt_world_feet(0,0);
    int numCorners = 4;
    for (int i = 0; i < numCorners; i++)
    {
        auto &imgPt = itBest->projCorners.at(i);
        Eigen::Vector3d pt_img_opticalCenter; pt_img_opticalCenter << imgPt(0) - m_pix2cam_offset(0), imgPt(1) - m_pix2cam_offset(1), 1;
        Eigen::Vector3d pt_world_corner_homogeneous = getDefaultHomographyMatByClass(detObj->classId).CamOpticalCenter2WorldFeetPlane * pt_img_opticalCenter;

        pt_world_feet += Eigen::Vector2d (
            pt_world_corner_homogeneous(0)/pt_world_corner_homogeneous(2),
            pt_world_corner_homogeneous(1)/pt_world_corner_homogeneous(2)
        );
    }
    pt_world_feet = pt_world_feet / numCorners;

    /// Use bbox bottom point.
    Eigen::Vector3d pt_img_opticalCenter; pt_img_opticalCenter << (bbox.min()(0) + bbox.max()(0))*0.5 - m_pix2cam_offset(0), bbox.max()(1) - m_pix2cam_offset(1), 1;
    Eigen::Vector3d pt_world_corner_homogeneous = getDefaultHomographyMatByClass(detObj->classId).CamOpticalCenter2WorldFeetPlane * pt_img_opticalCenter;

    pt_world_feet = Eigen::Vector2d (
        pt_world_corner_homogeneous(0)/pt_world_corner_homogeneous(2),
        pt_world_corner_homogeneous(1)/pt_world_corner_homogeneous(2)
    );

    Eigen::Vector3d pt_world_homogeneous; pt_world_homogeneous << pt_world_feet(0), pt_world_feet(1), 1;
    Eigen::Vector3d pt_img_feet_homogeneous = getDefaultHomographyMatByClass(detObj->classId).WorldFeetPlane2CamOpticalCenter * pt_world_homogeneous;
    Eigen::Vector2d pt_img_feet = Eigen::Vector2d (
        pt_img_feet_homogeneous(0)/pt_img_feet_homogeneous(2), pt_img_feet_homogeneous(1)/pt_img_feet_homogeneous(2)
    );

    refinedMot.convexHull = itBest->projCorners;
    refinedMot.bbox = itBest->bbox;
    refinedMot.visibility = 1.0;

    refinedMot.pt_world_feet = pt_world_feet;
    refinedMot.pt_img_feet = pt_img_feet;
    refinedMot.pt_img_center = Eigen::Vector2d(pt_img_center(0), pt_img_center(1));

    /// 3D locations.
    Eigen::Vector3d pt_world_cam(m_camLocation_World(0), m_camLocation_World(1), m_camLocation_World(2));
    /// Distance from the camera to the ground foot point.
    refinedMot.distFromCam = (Eigen::Vector3d(pt_world_feet(0), pt_world_feet(1), 0.0) - pt_world_cam).norm();
    return refinedMot;
}

/**
 * @brief Recover an object model from a detector bounding box.
 * @param[in] detObj Detected object to project.
 * @param[in] frameNum Current frame number.
 * @param[in] targetInd Target index, or a negative value when unavailable.
 * @param[in] isInitEstimation Whether this is the target's initial estimate.
 */
RecoveredObjModel ObjectModelProjector::getRecoveredObjectModelFromBbox(
    const NvMOTObjToTrack *detObj, const NvMOTFrameNum frameNum, const int targetInd, const bool isInitEstimation
)
{
    if (m_params.m_objectModelType == OBJ_MODEL_TYPE::BBOX3D)
    {
        return getRecoveredObjectModelFromProj3DBbox(detObj, frameNum);
    }

    std::vector<Eigen::Vector2d> projected_cylinder, convexHull;
    /// Unit: pixel.
    int margin = 20;

    /// Input bounding box.
    const NvMotDataTypes::Rect2f bbox = CV_FUNCS::constructRect2f(
        detObj->bbox.x, detObj->bbox.y, detObj->bbox.width, detObj->bbox.height
    );

    bool topClipped = bbox.min()(1) < margin;
    bool leftClipped = bbox.min()(0) < margin;
    bool bottomClipped = bbox.max()(1) >= m_srcFrameSize.height - margin;
    bool rightClipped = bbox.max()(0) >= m_srcFrameSize.width - margin;

    /// The image coordinate of the human waist (i.e., center of bbox).
    Eigen::Vector2d pt_img_waist = (bbox.max() + bbox.min()).cast<double>() / 2;

    /// Bounding box of the projected cylindrical human model and the convex hull.
    projected_cylinder = calculateProjectedCylinderFromImgWaist(pt_img_waist, detObj->classId, targetInd);
    CV_FUNCS::convexHull(projected_cylinder, convexHull);
    NvMotDataTypes::Rect2f modelBbox = CV_FUNCS::boundingRect(convexHull);

    /// Check if the model is clipped --- Align.
    float xShift = 0, yShift = 0;
    /// The cylinder fits inside the bbox.
    if (bbox.sizes()(1) > modelBbox.sizes()(1) ||
       ((topClipped || leftClipped || rightClipped) && !bottomClipped))
    {
        yShift = bbox.max()(1) - modelBbox.max()(1);
    }
    else
    {
        yShift = bbox.min()(1) - modelBbox.min()(1);
    }

    /// Calculate xShift based on leaning direction.
    Eigen::Vector2d pt_img_head = getCamHeadPtFromCamWaistPt(pt_img_waist, detObj->classId, targetInd);
    xShift = yShift * (pt_img_waist(0) - pt_img_head(0)) / (pt_img_waist(1) - pt_img_head(1));

    /// Shift the model.
    pt_img_waist += Eigen::Vector2d(xShift, yShift);
    modelBbox = CV_FUNCS::constructRect2f(
        modelBbox.min()(0) + xShift, modelBbox.min()(1) + yShift, modelBbox.sizes()(0), modelBbox.sizes()(1)
    );

    RecoveredObjModel refinedMot;

    for(auto &pt : convexHull)
    {
        /// The convex hull is not zero-centered.
        refinedMot.convexHull.push_back(Eigen::Vector2f(pt(0) + xShift, pt(1) + yShift));
    }

    /// The initial 3D estimation always uses the default object model.
    if (isInitEstimation && m_params.m_keepDetBboxForInitEstimation)
    {
        refinedMot.bbox = bbox;
    }
    else
    {
        refinedMot.bbox = modelBbox;
    }
    refinedMot.visibility = CV_FUNCS::computeVisibility(modelBbox, bbox);

    refinedMot.pt_world_feet = getWorldFeetPtOnGroundFromCamWaistPt(pt_img_waist, detObj->classId, targetInd);
    refinedMot.pt_img_feet = getCamFeetPtFromCamWaistPt(pt_img_waist, detObj->classId, targetInd);
    refinedMot.pt_img_center = pt_img_waist;

    /// 3D locations.
    float waistHeight = 0.0;
    if (isTargetModelValid(targetInd, detObj->classId))
    {
        waistHeight = m_targetModel.height / 2;
    }
    else
    {
        waistHeight = getDefaultModelByClass(detObj->classId).height / 2;
    }
    Eigen::Vector3d pt_world_waist(refinedMot.pt_world_feet(0), refinedMot.pt_world_feet(1), waistHeight);
    Eigen::Vector3d pt_world_cam(m_camLocation_World(0), m_camLocation_World(1), m_camLocation_World(2));
    /// Calculate the distance from the camera.
    refinedMot.distFromCam = (pt_world_waist - pt_world_cam).norm();

    return refinedMot;
}

/**
 * @brief Recover a cylinder model with height estimated from pose keypoints.
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
RecoveredObjModel ObjectModelProjector::getRecoveredObjectModelWithAdaptiveHeight(
    const Eigen::Vector2d pt_img_head, const Eigen::Vector2d pt_img_waist,
    const Eigen::Vector2d pt_img_foot, const NvMotDataTypes::Rect2f& bbox, const TrackerIndex &targetInd,
    const NvMOTClassId classId, float conf_head, float conf_waist, float conf_foot
)
{
    RecoveredObjModel refinedMot = {};

    /// Decide if using foot for height+position (foot confident) or waist (waist confident).
    bool use_pose_foot = (conf_foot > m_params.m_minPoseConfidence);
    Eigen::Vector2d pt_world_feet;

    if (use_pose_foot)
    {
        pt_world_feet = computeHeightAndWorldFeetFromHeadAndFoot(pt_img_head, pt_img_foot, targetInd, classId);
        m_targetModel.heightConfidence = (conf_head + conf_foot) / 2.0f;
    }
    else
    {
        pt_world_feet = computeHeightAndWorldFeetFromHeadAndWaist(pt_img_head, pt_img_waist, targetInd, classId);
        m_targetModel.heightConfidence = (conf_head + conf_waist) / 2.0f;
    }

    /// Derive data from adaptive height and build cylinder from world feet.
    deriveDataFromCameraProjMatrix(m_targetHomographyMat, m_targetModel);
    std::vector<Eigen::Vector2d> projected_cylinder = calculateProjectedCylinderFromWorldFeet(
        pt_world_feet, classId, targetInd
    );
    NvMotDataTypes::Rect2f modelBbox = CV_FUNCS::boundingRect(projected_cylinder);

    refinedMot.pt_img_center = (modelBbox.max() + modelBbox.min()).cast<double>() / 2;
    refinedMot.bbox = modelBbox;
    if (use_pose_foot)
    {
        refinedMot.pt_img_feet = pt_img_foot;
        refinedMot.pt_world_feet = pt_world_feet;
    }
    else
    {
        refinedMot.pt_img_feet = getCamFeetPtFromCamWaistPt(pt_img_waist, classId, targetInd);
        refinedMot.pt_world_feet = getWorldFeetPtOnGroundFromCamWaistPt(pt_img_waist, classId, targetInd);
    }
    refinedMot.visibility = CV_FUNCS::computeVisibility(refinedMot.bbox, bbox);

    /// Convex hull from projected cylinder (unsorted in non-VIZ_DEBUG).
    std::vector<Eigen::Vector2d> hull;
    CV_FUNCS::convexHull(projected_cylinder, hull);
    for (const auto& pt : hull)
        refinedMot.convexHull.push_back(Eigen::Vector2f(static_cast<float>(pt(0)), static_cast<float>(pt(1))));

    return refinedMot;
}

/**
 * @brief Project a cylinder bounding box from an image-space feet point.
 * @param[in] pt_img_feet Image-space feet point.
 * @param[in] classId Target class ID.
 * @param[in] targetInd Target index.
 */
const NvMotDataTypes::Rect2f ObjectModelProjector::getBboxOfProjectedCylinderFromImgFeet(
    const Eigen::Vector2d pt_img_feet, const NvMOTClassId classId, const int targetInd
)
{
    std::vector<Eigen::Vector2d> projected_cylinder = calculateProjectedCylinderFromImgFeet(
        pt_img_feet, classId, targetInd
    );

    return CV_FUNCS::boundingRect(projected_cylinder);
}

/**
 * @brief Project a cylinder bounding box from an image-space waist point.
 * @param[in] pt_img_waist Image-space waist point.
 * @param[in] classId Target class ID.
 * @param[in] targetInd Target index.
 */
const NvMotDataTypes::Rect2f ObjectModelProjector::getBboxOfProjectedCylinderFromImgWaist(
    const Eigen::Vector2d pt_img_waist, const NvMOTClassId classId, const int targetInd
)
{
    std::vector<Eigen::Vector2d> projected_cylinder = calculateProjectedCylinderFromImgWaist(
        pt_img_waist, classId, targetInd
    );

    return CV_FUNCS::boundingRect(projected_cylinder);
}

/**
 * @brief Project a cylinder bounding box from an image-space head point.
 * @param[in] pt_img_head Image-space head point.
 * @param[in] classId Target class ID.
 * @param[in] targetInd Target index.
 */
const NvMotDataTypes::Rect2f ObjectModelProjector::getBboxOfProjectedCylinderFromImgHead(
    const Eigen::Vector2d pt_img_head, const NvMOTClassId classId, const int targetInd
)
{
    std::vector<Eigen::Vector2d> projected_cylinder = calculateProjectedCylinderFromImgHead(
        pt_img_head, classId, targetInd
    );

    return CV_FUNCS::boundingRect(projected_cylinder);
}

/**
 * @brief Project a cylinder bounding box from world-space feet coordinates.
 * @param[in] pt_world_feet World-space feet position.
 * @param[in] classId Target class ID.
 * @param[in] targetInd Target index.
 */
const NvMotDataTypes::Rect2f ObjectModelProjector::getBboxOfProjectedCylinderFromWorldFeet(
    const Eigen::Vector2d pt_world_feet, const NvMOTClassId classId, const int targetInd
)
{
    std::vector<Eigen::Vector2d> projected_cylinder = calculateProjectedCylinderFromWorldFeet(
        pt_world_feet, classId, targetInd
    );

    return CV_FUNCS::boundingRect(projected_cylinder);
}

/**
 * @brief Project world-ground points into image coordinates.
 * @param[in] pt_world_feet_list World-ground points to project.
 */
const std::vector<Eigen::Vector2d> ObjectModelProjector::getProjectedWorldGroundPts(
    const std::vector<Eigen::Vector2d>& pt_world_feet_list
)
{
    std::vector<Eigen::Vector2d> pt_img_feet_list;
    for (auto pt_world_feet : pt_world_feet_list)
    {
        Eigen::Vector3d pt_world_homogeneous; pt_world_homogeneous << pt_world_feet(0), pt_world_feet(1), 1;
        Eigen::Vector3d pt_img_feet_homogeneous = getDefaultHomographyMatByClass(0).WorldFeetPlane2CamOpticalCenter * pt_world_homogeneous;
        Eigen::Vector2d pt_img_feet = Eigen::Vector2d (
            pt_img_feet_homogeneous(0)/pt_img_feet_homogeneous(2), pt_img_feet_homogeneous(1)/pt_img_feet_homogeneous(2)
        );
        pt_img_feet_list.push_back(pt_img_feet);
    }

    return pt_img_feet_list;
}

/**
 * @brief Map an image waist point to the corresponding image head point.
 * @param[in] imgWaistPt Image-space waist point.
 * @param[in] classId Target class ID.
 * @param[in] targetInd Target index.
 */
const Eigen::Vector2d ObjectModelProjector::getCamHeadPtFromCamWaistPt(
    const Eigen::Vector2d imgWaistPt, const NvMOTClassId classId, const int targetInd
)
{
    /// Convert image coordinates from the top-left origin to the camera optical center.
    Eigen::Vector3d pt_img_waist; pt_img_waist << imgWaistPt(0) - m_pix2cam_offset(0), imgWaistPt(1) - m_pix2cam_offset(1), 1;

    HomographyMatrix homographyMat_Waist2Head;
    if (isTargetModelValid(targetInd, classId))
    {
        homographyMat_Waist2Head = m_targetHomographyMat.Waist2Head;
    }
    else
    {
        homographyMat_Waist2Head = getDefaultHomographyMatByClass(classId).Waist2Head;
    }

    Eigen::Vector3d pt_head_homogeneous = homographyMat_Waist2Head * pt_img_waist;

    Eigen::Vector2d pt_head_opticCenter(
        pt_head_homogeneous(0)/pt_head_homogeneous(2), pt_head_homogeneous(1)/pt_head_homogeneous(2)
    );

    /// Convert the camera optical center coordinate into the camera image coordinate by translation.
    Eigen::Vector2d pt_head(pt_head_opticCenter(0) + m_pix2cam_offset(0), pt_head_opticCenter(1) + m_pix2cam_offset(1));

    return pt_head;
}

/**
 * @brief Map an image head point to the corresponding image waist point.
 * @param[in] imgHeadPt Image-space head point.
 * @param[in] classId Target class ID.
 * @param[in] targetInd Target index.
 */
const Eigen::Vector2d ObjectModelProjector::getCamWaistPtFromCamHeadPt(
    const Eigen::Vector2d imgHeadPt, const NvMOTClassId classId, const int targetInd
)
{
    /// Convert image coordinates from the top-left origin to the camera optical center.
    Eigen::Vector3d pt_img_head_opticalCenter; pt_img_head_opticalCenter << imgHeadPt(0) - m_pix2cam_offset(0), imgHeadPt(1) - m_pix2cam_offset(1), 1;

    HomographyMatrix homographyMat_Head2Waist;
    if (isTargetModelValid(targetInd, classId))
    {
        homographyMat_Head2Waist = m_targetHomographyMat.Head2Waist;
    }
    else
    {
        homographyMat_Head2Waist = getDefaultHomographyMatByClass(classId).Head2Waist;
    }

    Eigen::Vector3d pt_waist_homogeneous = homographyMat_Head2Waist * pt_img_head_opticalCenter;

    Eigen::Vector2d pt_waist_opticCenter(
        pt_waist_homogeneous(0)/pt_waist_homogeneous(2), pt_waist_homogeneous(1)/pt_waist_homogeneous(2)
    );

    /// Convert the camera optical center coordinate into the camera image coordinate by translation.
    Eigen::Vector2d pt_waist(
        pt_waist_opticCenter(0) + m_pix2cam_offset(0), pt_waist_opticCenter(1) + m_pix2cam_offset(1)
    );

    return pt_waist;
}

/**
 * @brief Map an image feet point to the corresponding image waist point.
 * @param[in] imgFeetPt Image-space feet point.
 * @param[in] classId Target class ID.
 * @param[in] targetInd Target index.
 */
const Eigen::Vector2d ObjectModelProjector::getCamWaistPtFromCamFeetPt(
    const Eigen::Vector2d imgFeetPt, const NvMOTClassId classId, const int targetInd
)
{
    /// Convert image coordinates from the top-left origin to the camera optical center.
    Eigen::Vector3d pt_img_feet_opticalCenter; pt_img_feet_opticalCenter << imgFeetPt(0) - m_pix2cam_offset(0), imgFeetPt(1) - m_pix2cam_offset(1), 1;

    HomographyMatrix homographyMat_Feet2Waist;
    if (isTargetModelValid(targetInd, classId))
    {
        homographyMat_Feet2Waist = m_targetHomographyMat.Feet2Waist;
    }
    else
    {
        homographyMat_Feet2Waist = getDefaultHomographyMatByClass(classId).Feet2Waist;
    }

    Eigen::Vector3d pt_waist_homogeneous = homographyMat_Feet2Waist * pt_img_feet_opticalCenter;

    Eigen::Vector2d pt_waist_opticCenter(
        pt_waist_homogeneous(0)/pt_waist_homogeneous(2), pt_waist_homogeneous(1)/pt_waist_homogeneous(2)
    );

    /// Convert the camera optical center coordinate into the camera image coordinate by translation.
    Eigen::Vector2d pt_waist(
        pt_waist_opticCenter(0) + m_pix2cam_offset(0), pt_waist_opticCenter(1) + m_pix2cam_offset(1)
    );

    return pt_waist;
}

/**
 * @brief Map an image waist point to the corresponding image feet point.
 * @param[in] imgWaistPt Image-space waist point.
 * @param[in] classId Target class ID.
 * @param[in] targetInd Target index.
 */
const Eigen::Vector2d ObjectModelProjector::getCamFeetPtFromCamWaistPt(
    const Eigen::Vector2d imgWaistPt, const NvMOTClassId classId, const int targetInd
)
{
    /// Convert image coordinates from the top-left origin to the camera optical center.
    Eigen::Vector3d pt_img_waist_opticalCenter; pt_img_waist_opticalCenter << imgWaistPt(0) - m_pix2cam_offset(0), imgWaistPt(1) - m_pix2cam_offset(1), 1;

    HomographyMatrix homographyMat_Waist2Feet;
    if (isTargetModelValid(targetInd, classId))
    {
        homographyMat_Waist2Feet = m_targetHomographyMat.Waist2Feet;
    }
    else
    {
        homographyMat_Waist2Feet = getDefaultHomographyMatByClass(classId).Waist2Feet;
    }

    Eigen::Vector3d pt_feet_homogeneous = homographyMat_Waist2Feet * pt_img_waist_opticalCenter;

    Eigen::Vector2d pt_feet_opticCenter(
        pt_feet_homogeneous(0)/pt_feet_homogeneous(2), pt_feet_homogeneous(1)/pt_feet_homogeneous(2)
    );

    /// Convert the camera optical center coordinate into the camera image coordinate by translation.
    Eigen::Vector2d pt_feet(pt_feet_opticCenter(0) + m_pix2cam_offset(0), pt_feet_opticCenter(1) + m_pix2cam_offset(1));

    return pt_feet;
}

/**
 * @brief Recover world-ground feet coordinates from an image waist point.
 * @param[in] imgWaistPt Image-space waist point.
 * @param[in] classId Target class ID.
 * @param[in] targetInd Target index.
 */
const Eigen::Vector2d ObjectModelProjector::getWorldFeetPtOnGroundFromCamWaistPt(
    const Eigen::Vector2d imgWaistPt, const NvMOTClassId classId, const int targetInd
)
{
    /// Convert image coordinates from the top-left origin to the camera optical center.
    Eigen::Vector3d pt_img_waist_opticalCenter; pt_img_waist_opticalCenter << imgWaistPt(0) - m_pix2cam_offset(0), imgWaistPt(1) - m_pix2cam_offset(1), 1;

    HomographyMatrix homographyMat_CamOpticalCenter2WorldWaistPlane;
    if (isTargetModelValid(targetInd, classId))
    {
        homographyMat_CamOpticalCenter2WorldWaistPlane = m_targetHomographyMat.CamOpticalCenter2WorldWaistPlane;
    }
    else
    {
        homographyMat_CamOpticalCenter2WorldWaistPlane = getDefaultHomographyMatByClass(classId).CamOpticalCenter2WorldWaistPlane;
    }

    Eigen::Vector3d pt_world_feet_homogeneous = homographyMat_CamOpticalCenter2WorldWaistPlane * pt_img_waist_opticalCenter;

    const Eigen::Vector2d pt_feet(
        pt_world_feet_homogeneous(0)/pt_world_feet_homogeneous(2),
        pt_world_feet_homogeneous(1)/pt_world_feet_homogeneous(2)
    );

    return pt_feet;
}

/**
 * @brief Recover world-ground feet coordinates from an image feet point.
 * @param[in] imgFeetPt Image-space feet point.
 * @param[in] classId Target class ID.
 * @param[in] targetInd Target index.
 */
const Eigen::Vector2d ObjectModelProjector::getWorldFeetPtOnGroundFromCamFeetPt(
    const Eigen::Vector2d imgFeetPt, const NvMOTClassId classId, const int targetInd
)
{
    /// Convert image coordinates from the top-left origin to the camera optical center.
    Eigen::Vector3d pt_img_feet_opticalCenter; pt_img_feet_opticalCenter << imgFeetPt(0) - m_pix2cam_offset(0), imgFeetPt(1) - m_pix2cam_offset(1), 1;

    Eigen::Vector3d pt_world_feet_homogeneous;
    if (isTargetModelValid(targetInd, classId))
    {
        pt_world_feet_homogeneous = m_targetHomographyMat.CamOpticalCenter2WorldFeetPlane * pt_img_feet_opticalCenter;
    }
    else
    {
        pt_world_feet_homogeneous = getDefaultHomographyMatByClass(classId).CamOpticalCenter2WorldFeetPlane * pt_img_feet_opticalCenter;
    }

    const Eigen::Vector2d pt_feet(
        pt_world_feet_homogeneous(0)/pt_world_feet_homogeneous(2),
        pt_world_feet_homogeneous(1)/pt_world_feet_homogeneous(2)
    );

    return pt_feet;
}

/**
 * @brief Build a world-space three-dimensional bounding box from feet coordinates.
 * @param[in] pt_world_feet World-space feet position.
 * @param[in] classId Target class ID.
 * @param[in] targetInd Target index.
 */
const NvDsObj3DBbox ObjectModelProjector::getWorld3DBboxFromWorldFeet(
    const Eigen::Vector2d pt_world_feet, const NvMOTClassId classId, const int targetInd
)
{
    NvDsObj3DBbox bbox = {};
    Model3DInfo modelSize = {};
    bbox.xCentre = pt_world_feet(0);
    bbox.yCentre = pt_world_feet(1);
    if(isTargetModelValid(targetInd, classId))
    {
        modelSize = m_targetModel;
    }
    else
    {
        modelSize = getDefaultModelByClass(classId);
    }
    bbox.zCentre = modelSize.height / 2;
    bbox.xLen = 2 * modelSize.radius;
    bbox.yLen = 2 * modelSize.radius;
    bbox.zLen = modelSize.height ;

    if (m_params.m_debugVerboseLevel > 4)
    {
        std::cout << "[]Target index: " << targetInd << ", is target model valid: " <<
            isTargetModelValid(targetInd, classId) << ", target model classId: " << m_targetClassId << ", classId: " << classId <<
            ", model size: " << modelSize.height << ", " << modelSize.radius << ", target model: " << m_targetModel.height <<
            ", " << m_targetModel.radius << std::endl;
    }
    return bbox;
}

/**
 * @brief Project the eight corners of a world-space three-dimensional bounding box.
 * @param[in] pt_world_feet World-space feet position.
 * @param[in] classId Target class ID.
 * @param[in] targetInd Target index.
 */
std::vector<Eigen::Vector2d> ObjectModelProjector::calculateProjCornersOf3DBboxFromWorldFeet(
    const Eigen::Vector2d pt_world_feet, const NvMOTClassId classId, const int targetInd
)
{
    std::vector<Eigen::Vector2d> projected_corners;
    std::vector<Eigen::Vector3d> world_corners;
    NvDsObj3DBbox bbox = getWorld3DBboxFromWorldFeet(pt_world_feet, classId, targetInd);

    /// Get 3d 8 corner points from bbox Bottom 4 corners.
    world_corners.push_back(
        Eigen::Vector3d(bbox.xCentre-bbox.xLen/2,bbox.yCentre-bbox.yLen/2,bbox.zCentre-bbox.zLen/2)
    );
    world_corners.push_back(
        Eigen::Vector3d(bbox.xCentre+bbox.xLen/2,bbox.yCentre-bbox.yLen/2,bbox.zCentre-bbox.zLen/2)
    );
    world_corners.push_back(
        Eigen::Vector3d(bbox.xCentre+bbox.xLen/2,bbox.yCentre+bbox.yLen/2,bbox.zCentre-bbox.zLen/2)
    );
    world_corners.push_back(
        Eigen::Vector3d(bbox.xCentre-bbox.xLen/2,bbox.yCentre+bbox.yLen/2,bbox.zCentre-bbox.zLen/2)
    );

    /// Top 4 corners.
    world_corners.push_back(
        Eigen::Vector3d(bbox.xCentre-bbox.xLen/2,bbox.yCentre-bbox.yLen/2,bbox.zCentre+bbox.zLen/2)
    );
    world_corners.push_back(
        Eigen::Vector3d(bbox.xCentre+bbox.xLen/2,bbox.yCentre-bbox.yLen/2,bbox.zCentre+bbox.zLen/2)
    );
    world_corners.push_back(
        Eigen::Vector3d(bbox.xCentre+bbox.xLen/2,bbox.yCentre+bbox.yLen/2,bbox.zCentre+bbox.zLen/2)
    );
    world_corners.push_back(
        Eigen::Vector3d(bbox.xCentre-bbox.xLen/2,bbox.yCentre+bbox.yLen/2,bbox.zCentre+bbox.zLen/2)
    );

    /// Project the 3d 8 corner points to 2d points.
    for(size_t i=0; i < world_corners.size(); i++)
    {
        projected_corners.push_back(projectWorldPt2ImgPt(world_corners[i]));
    }
    return projected_corners;
}

/**
 * @brief Compute ground-plane covariance from pixel uncertainty.
 * @param[in] pixel_pt The pixel coordinates (u, v).
 * @param[in] pixel_sigma Standard deviation of pixel noise.
 */
Eigen::Matrix2d ObjectModelProjector::computeGroundCovariance(const Eigen::Vector2d& pixel_pt, float pixel_sigma)
{
    /// Convert pixel to optical center coordinates.
    Eigen::Vector2d pixel_centered = pixel_pt - m_pix2cam_offset;

    /// Compute normalized direction: p = K^(-1) * [u, v, 1]^T.
    Eigen::Vector3d u_tilde(pixel_centered(0), pixel_centered(1), 1.0);
    Eigen::Vector3d p = m_K_inv * u_tilde;

    /// Compute ray-plane intersection parameters.
    double beta = m_r3.dot(p);

    /// Check for near-parallel ray (would give infinite uncertainty).
    if (std::abs(beta) < 1e-10) {
        /// Return high uncertainty for degenerate case.
        return Eigen::Matrix2d::Identity() * 1e6;
    }

    double r3_dot_t = m_r3.dot(m_t);
    double lambda = r3_dot_t / beta;

    /// Eq(3): dX/dp = R^T * (-lambda/beta * p * r3^T + lambda * I_3).
    Eigen::Matrix3d dX_dp = m_R_T * (-lambda / beta * p * m_r3.transpose() + lambda * Eigen::Matrix3d::Identity());

    /// Jacobian from pixels to ground: J = dX/dp * K^(-1)[:, 0:2] Only need first 2 columns of K_inv (du, dv derivatives).
    Eigen::Matrix<double, 3, 2> J_full = dX_dp * m_K_inv.leftCols<2>();

    /// Take only X,Y rows (first 2 rows) for ground plane covariance.
    Eigen::Matrix2d J_xy = J_full.topRows<2>();

    /// Σ_XY = J * σ² * I_2 * J^T.
    double sigma_sq = static_cast<double>(pixel_sigma * pixel_sigma);
    Eigen::Matrix2d cov = J_xy * sigma_sq * J_xy.transpose();

    /// Ensure positive definiteness (numerical stability).
    cov = (cov + cov.transpose()) / 2.0;

    return cov;
}

/** @brief Initialize object-model projector parameters with default values. */
NvTrackerParamsObjectModelProjector::NvTrackerParamsObjectModelProjector()
{

    m_debugVerboseLevel = 0;
    m_outputFootLocation = false;
    m_outputConvexHull = false;
    m_maxConvexHullSize = 30;
    m_objectModelType = OBJ_MODEL_TYPE::CYLINDER;
    m_waistHeightRatio = 0.5f;
    m_eyeHeightRatio = 0.93f;
    m_keepDetBboxForInitEstimation = false;
    m_useDefaultModelHeight = -1.f;
    m_useDefaultModelRadius = -1.f;
    m_minPoseConfidence = 0.5f;
}

/**
 * @brief Initialize object-model projector parameters from a configuration module.
 * @param[in] configFilePath The path to a configuration file.
 * @param[in] moduleName The configuration module name.
 */
NvTrackerParamsObjectModelProjector::NvTrackerParamsObjectModelProjector(
    const std::string &configFilePath, const std::string moduleName
)
{
    YAML::Node configRoot = getConfigRoot(configFilePath);
    this->read(configRoot[moduleName]);
    for (std::string &p : m_cameraModelFilepath)
    {
        p = NvTrackerParams::resolvePathRelativeToConfig(configFilePath, p);
    }
    for (auto &kv : m_cameraModelFilepathMap)
    {
        kv.second = NvTrackerParams::resolvePathRelativeToConfig(configFilePath, kv.second);
    }
    for (std::string &p : m_proj3DBboxFilePath)
    {
        p = NvTrackerParams::resolvePathRelativeToConfig(configFilePath, p);
    }
}

/**
 * @brief Initialize object-model projector parameters from a YAML node.
 * @param[in] fn Parsed YAML structure of config to update.
 */
NvTrackerParamsObjectModelProjector::NvTrackerParamsObjectModelProjector(const YAML::Node &fn)
{
    this->read(fn);
}

/**
 * @brief Update object-model projector parameters from a YAML node.
 * @param[in] fn Parsed YAML structure of config to update.
 */
void NvTrackerParamsObjectModelProjector::read(const YAML::Node &fn)
{
    /// Set the default values.
    *this = NvTrackerParamsObjectModelProjector();

    if(!fn.size())
    {
        return;
    }

    for (YAML::const_iterator fit = fn.begin(); fit != fn.end(); ++fit)
    {
        std::string paramKey = fit->first.as<std::string>();

        if (paramKey == "debugVerboseLevel")
        {
            m_debugVerboseLevel = fit->second.as<float>();
        }
        else if (paramKey == "cameraModelFilepath")
        {

            const YAML::Node filePaths = fit->second;

            m_cameraModelFilepath.clear();
            m_cameraModelFilepathMap.clear();

            if (filePaths.IsMap())
            {
                for (auto it = filePaths.begin(); it != filePaths.end(); ++it)
                {
                    m_cameraModelFilepathMap[it->first.as<std::string>()] = it->second.as<std::string>();
                }
            }
            else
            {
                for (std::size_t streamInd = 0; streamInd < filePaths.size(); streamInd++)
                {
                    const YAML::Node filePath = filePaths[streamInd];
                    m_cameraModelFilepath.push_back(filePath.as<std::string>());
                }
            }
        }
        else if (paramKey == "proj3DBboxFilePath")
        {

            const YAML::Node filePaths = fit->second;

            m_proj3DBboxFilePath.clear();
            for (std::size_t streamInd = 0; streamInd < filePaths.size(); streamInd++)
            {
                const YAML::Node filePath = filePaths[streamInd];
                m_proj3DBboxFilePath.push_back(filePath.as<std::string>());
            }
        }
        else if (paramKey == "outputFootLocation")
        {
            m_outputFootLocation = (int)fit->second.as<float>();
        }
        else if (paramKey == "outputConvexHull")
        {
            m_outputConvexHull = (int)fit->second.as<float>();
        }
        else if (paramKey == "maxConvexHullSize")
        {
            m_maxConvexHullSize = fit->second.as<float>();
        }
        else if (paramKey == "objectModelType")
        {
            m_objectModelType = (OBJ_MODEL_TYPE)((int)fit->second.as<float>());
        }
        else if (paramKey == "waistHeightRatio")
        {
            m_waistHeightRatio = fit->second.as<float>();
        }
        else if (paramKey == "eyeHeightRatio")
        {
            m_eyeHeightRatio = fit->second.as<float>();
        }
        else if (paramKey == "keepDetBboxForInitEstimation")
        {
            m_keepDetBboxForInitEstimation = (int)fit->second.as<float>();
        }
        else if (paramKey == "useDefaultModelHeight")
        {
            m_useDefaultModelHeight = fit->second.as<float>();
        }
        else if (paramKey == "useDefaultModelRadius")
        {
            m_useDefaultModelRadius = fit->second.as<float>();
        }
        else if (paramKey == "minPoseConfidence")
        {
            m_minPoseConfidence = fit->second.as<float>();
        }
        else
        {
        }
    }
}
