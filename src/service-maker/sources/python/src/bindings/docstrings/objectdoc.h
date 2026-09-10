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

#pragma once

namespace pydeepstreamdoc
{
    namespace object
    {
        namespace ObjectDoc
        {
            constexpr const char* descr = R"pydeepstream(Base class of a custom object.)pydeepstream";
            constexpr const char* set = R"pydeepstream(Set the properties of an object with a dictionary.)pydeepstream";
        }

        namespace CommonFactoryDoc
        {
            constexpr const char* descr = R"pydeepstream(Factory class for creating instances from shared libraries.)pydeepstream";
            constexpr const char* create = R"pydeepstream(Create an object from a plugin and name.)pydeepstream";
        }
    }
}