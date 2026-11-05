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

#ifndef UTILITY_DATATYPES_HPP
#define UTILITY_DATATYPES_HPP

#include <sys/stat.h>
#include <stdexcept>
#include <iostream>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <unsupported/Eigen/MatrixFunctions>

typedef unsigned char uchar;

namespace NvMotDataTypes {

// [Matrix]
template <typename K, int N, int M>
using RowMajorMatrix = Eigen::Matrix<K, N, M, Eigen::RowMajor>; //!< Row-major.
template <typename K>
using RowMajorMatrixX = RowMajorMatrix<K, Eigen::Dynamic, Eigen::Dynamic>;
using RowMajorMatrixXd = RowMajorMatrixX<double>;
using RowMajorMatrixXf = RowMajorMatrixX<float>;
using RowMajorMatrixXi = RowMajorMatrixX<int>;

// [Bounding box]
template <typename K, int N>
using Rect_ = Eigen::AlignedBox<K, N>;
template <typename K>
using Rect2d = Rect_<double, 2>;
using Rect2f = Rect_<float, 2>;
using Rect2i = Rect_<int, 2>;
using Rect = Rect_<int, 2>;

/** @brief Represents a two-dimensional integer size. */
struct Size
{
	/** @brief Initialize an unspecified size. */
	Size() {}

	/**
	 * @brief Initialize a size from width and height.
	 * @param[in] w Width in pixels.
	 * @param[in] h Height in pixels.
	 */
	Size(const int w, const int h) : width(w), height(h) {}

	int width; //!< Width in pixels.
	int height; //!< Height in pixels.
};

}

#endif // UTILITY_DATATYPES_HPP
