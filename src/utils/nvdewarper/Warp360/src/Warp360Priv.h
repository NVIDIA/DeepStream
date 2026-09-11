/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
/** @file Warp360Priv.h
 *  Internal definitions for 360 Image and Coordinate Warp SDK.
 */

#ifndef WARP360_PRIV_H
#define WARP360_PRIV_H

/********************************************************************************
 * Input odd order coefficients, get odd/even Pade rational polynomial.
 * @param[in]   dist  r + d0 * r^3 + d1 * r^5 + d2 * r^7 + d3 * r^9
 * @param[out]  pade  (r + p0 * r^3 + p2 * r^5 + p4 * r^7) / (1 + p1 * r^2 + p3 * r^4 + p5 * r^6)
 ********************************************************************************/
void   Compute6InverseRadialPadeCoefficients(const float *dist, float *pade);

#endif // WARP360_PRIV_H