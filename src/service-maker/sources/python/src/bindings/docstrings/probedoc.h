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
    namespace probe
    {
        namespace ProbeDoc
        {
            constexpr const char* descr = R"pydeepstream(Probe class for peeking into the buffers.)pydeepstream";
            constexpr const char* init_buffer_operator = R"pydeepstream(Initialize the probe with a name and a :class:`BufferOperator` handler.)pydeepstream";
            constexpr const char* init_batch_metadata_operator = R"pydeepstream(Initialize the probe with a name and a :class:`BatchMetadataOperator` handler.)pydeepstream";
            constexpr const char* create = R"pydeepstream(Create a probe from a plugin and a name.)pydeepstream";
        }

        namespace BufferObserverDoc
        {
            constexpr const char* descr = R"pydeepstream(Read-only interface for observing buffers without modifying them.)pydeepstream";
            constexpr const char* handle_buffer = R"pydeepstream(Observe a buffer. Called for every buffer; return value is ignored.)pydeepstream";
        }

        namespace BufferOperatorDoc
        {
            constexpr const char* descr = R"pydeepstream(Interface for accessing the buffers.)pydeepstream";
            constexpr const char* handle_buffer = R"pydeepstream(Handle a buffer. Must return True after successfully handling the buffer, otherwise the buffer will be dropped.)pydeepstream";
        }

        namespace BatchMetadataOperatorDoc
        {
            constexpr const char* descr = R"pydeepstream(Interface for accessing the metadata.)pydeepstream";
            constexpr const char* handle_metadata = R"pydeepstream(Handle batch metadata.)pydeepstream";
        }
    }
}