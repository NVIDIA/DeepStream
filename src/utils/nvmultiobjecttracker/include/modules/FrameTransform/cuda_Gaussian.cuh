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

#include "modules/utility/cuda_common.hpp"

/// Gaussian smoothing weights for sigma 0.5.
__device__ __constant__ float GaussianWeights3x3[9] =
{
	0.0113f, 0.0837f, 0.0113f,
	0.0837f, 0.6187f, 0.0837f,
	0.0113f, 0.0837f, 0.0113f
};

/**
 * @brief Apply a three-by-three Gaussian filter to a texture sample.
 * @param[in] tex CUDA texture object to sample.
 * @param[in] pt Texture coordinate to filter.
 */
template <typename T>
__device__ __inline__ T cuda_GaussianSmoothing_3x3(cudaTextureObject_t tex, const float2 pt)
{
	T val;

	/// TODO: Make this configurable.
	float step = 0.25;

	uint cnt = 0;

	val = GaussianWeights3x3[cnt++] * tex2D<T>(tex, pt.x-step, pt.y-step);
	val += GaussianWeights3x3[cnt++] * tex2D<T>(tex, pt.x+0, pt.y-step);
	val += GaussianWeights3x3[cnt++] * tex2D<T>(tex, pt.x+step, pt.y-step);

	val += GaussianWeights3x3[cnt++] * tex2D<T>(tex, pt.x-step, pt.y);
	val += GaussianWeights3x3[cnt++] * tex2D<T>(tex, pt.x+0, pt.y);
	val += GaussianWeights3x3[cnt++] * tex2D<T>(tex, pt.x+step, pt.y);

	val += GaussianWeights3x3[cnt++] * tex2D<T>(tex, pt.x-step, pt.y+step);
	val += GaussianWeights3x3[cnt++] * tex2D<T>(tex, pt.x+0, pt.y+step);
	val += GaussianWeights3x3[cnt++] * tex2D<T>(tex, pt.x+step, pt.y+step);

	return val;
}
