/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

/**
 * @file
 * <b>Pad definition </b>
 */

#ifndef NVIDIA_DEEPSTREAM_PAD
#define NVIDIA_DEEPSTREAM_PAD

#include "object.hpp"

namespace deepstream {

/**
 * @brief Pad is an abstraction of the I/O with an Element, @see Element
 *
 * Pad class derives from the base Object class, so it is reference based,
 * supports copying and moving.
 * A Pad instance must be either for input or for output
 *
 */
class Pad : public Object {
public:
  /** empty constructor */
  Pad();
  /** substantial constructor */
  Pad(bool is_input, const std::string& name = std::string());
  /** copy constructor */
  Pad(const Object&);
  /** move constructor */
  Pad(Object&&);

};
}

#endif