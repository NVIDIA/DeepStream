/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef __GST_NVOF_VISUAL_DRAW_H__
#define __GST_NVOF_VISUAL_DRAW_H__

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

void DrawOpticalFlow (void* vectors,  void* dst, int cols, int rows, float maxmotion);

void DrawOpticalFlow_Cuda (void* vectors,  void* dst, int cols, int rows, int pitch, float maxmotion, cudaStream_t stream);

#endif /* __GST_NVOF_VISUAL_DRAW_H__ */
