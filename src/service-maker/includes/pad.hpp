/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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