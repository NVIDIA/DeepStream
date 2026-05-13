/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef _DS3D_COMMON_INFERENCE_DATATYPE__H
#define _DS3D_COMMON_INFERENCE_DATATYPE__H

#include <ds3d/common/common.h>
#include <ds3d/common/idatatype.h>
#include <ds3d/common/func_utils.h>
#include <ds3d/common/defines.h>

#undef DS3D_MAX_LABEL_SIZE
#define DS3D_MAX_LABEL_SIZE 128

/*
 * Pointcloud Coordinate System
 * pointcloud & box coordinate system, x -> front, y -> left, z -> up
 *
 *
 *                                up z    x front (yaw=0)
 *                                   ^   ^
 *                                   |  /
 *                                   | /
 *       (yaw=0.5*pi) left y <------ 0
 */
namespace ds3d {

/**
 * @brief Object 3D bounding box description. vector of data stored into
 * FrameGuard, dims as [batch, N, sizeof(Lidar3DBbox)];
 *
 */
struct Lidar3DBbox {
    float centerX = 0.0f;
    float centerY = 0.0f;
    float centerZ = 0.0f;
    float dx = 0.0f;  // length, distance of x
    float dy = 0.0f;  // width, distance of y
    float dz = 0.0f;  // // height, distance of z
    float yaw = 0.0f;
    int cid = 0;  // class id
    float score = 0.0f;
    vec4f bboxColor = {{1.0f, 1.0f, 0, 1.0f}};  // RGBA, default value: yellow
    char labels[DS3D_MAX_LABEL_SIZE] = {0};

    Lidar3DBbox() = default;
    Lidar3DBbox(
        float centerX_, float centerY_, float centerZ_, float length_, float width_, float height_,
        float yaw_, int cid_, float score_)
        : centerX(centerX_), centerY(centerY_), centerZ(centerZ_), dx(length_), dy(width_),
          dz(height_), yaw(yaw_), cid(cid_), score(score_)
    {
    }

    void dump() const {
        LOG_DEBUG("pos (%.3f %.3f %.3f) size (%.3f %.3f %.3f) score %.3f labels %s",
                        centerX, centerY, centerZ, dx, dy, dz, score, labels);
    }
};

/**
 * @brief Object 2D bounding box description. vector of data stored into
 * FrameGuard, dims as [N, sizeof(Object2DBbox)];
 *
 */
struct Object2DBbox {
    float centerX = 0.0f;
    float centerY = 0.0f;
    float dx = 0.0f;  // width, distance of x
    float dy = 0.0f;  // height, distance of y
    int cid = 0;  // class id
    float score = 0.0f;
    int trackerId = -1;
    float trackerScore = 0.0f;
    int sourceId = 0;  // objects in which source stream
    vec4f bboxColor = {{0, 0, 0, 1.0f}};  // RGBA, black
    char labels[DS3D_MAX_LABEL_SIZE] = {0};
};

/**
 * @brief Object fused bounding box description. vector of data stored into
 * FrameGuard, dims as [N, sizeof(FusedDetection)];
 *
 */
struct FusedDetection {
    Lidar3DBbox obj3D;
    Object2DBbox obj2D;
    float score = 0.0f;
    vec4f bboxColor = {{1.0f, 0, 0, 1.0f}};  // RGBA, red
    char labels[DS3D_MAX_LABEL_SIZE] = {0};
};

}  // namespace ds3d

#endif