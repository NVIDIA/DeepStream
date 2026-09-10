/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    namespace node
    {
        namespace NodeDoc
        {
            constexpr const char* descr = R"pydeepstream(
                Abstraction of a Deepstream element instance with the pipeline

                :ivar name: *str*, Name of the node instance.)pydeepstream";
            constexpr const char* link = R"pydeepstream(Link the node to a target node.)pydeepstream";
            constexpr const char* link_info = R"pydeepstream(Link the node to a target node with extra information.)pydeepstream";
            constexpr const char* set = R"pydeepstream(Set node properties with a dictionary.)pydeepstream";
            constexpr const char* set2 = R"pydeepstream(Set node properties with a list of dictionaries.)pydeepstream";
            constexpr const char* get = R"pydeepstream(Get the current value of a property with the given name.)pydeepstream";
            constexpr const char* attach_probe = R"pydeepstream(Attach a probe.)pydeepstream";
            constexpr const char* attach_receiver = R"pydeepstream(Attach a receiver.)pydeepstream";
            constexpr const char* attach_feeder = R"pydeepstream(Attach a feeder.)pydeepstream";
            constexpr const char* find = R"pydeepstream(Find a node by name with the pipeline.)pydeepstream";
        }
    }
}