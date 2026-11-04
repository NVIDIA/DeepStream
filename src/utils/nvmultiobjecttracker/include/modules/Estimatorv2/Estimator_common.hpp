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

#ifndef ESTIMATORLIB_ESTIMATOR_COMMON_HPP_
#define ESTIMATORLIB_ESTIMATOR_COMMON_HPP_

/** @brief State-estimator implementations that can be selected by configuration. */
enum STATE_ESTIMATOR_TYPES : int
{
    DUMMY_ESTIMATOR = 0, //!< Selects the pass-through dummy estimator.
    BBOX_ESTIMATOR_SIMPLE = 1, //!< Selects the six-state bounding-box estimator.
    BBOX_ESTIMATOR = 2, //!< Selects the eight-state bounding-box estimator.
    LOCATION_ESTIMATOR_SIMPLE = 3, //!< Selects the simple location estimator.
    LOCATION_ESTIMATOR_PROJECTIVE = 4 //!< Selects the projective location estimator.
};

#endif // ESTIMATORLIB_ESTIMATOR_COMMON_HPP_
