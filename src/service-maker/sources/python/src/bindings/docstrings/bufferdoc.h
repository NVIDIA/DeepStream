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
    namespace buffer
    {
        namespace BufferDoc
        {
            constexpr const char* descr = R"pydeepstream(
                Wrapper of a deepstream buffer

                :ivar batch_size: *int*, Batch size of the buffer
                :ivar batch_meta: :class:`BatchMetadata`, Batch metadata of the buffer
                :ivar timestamp: *int*, Timestamp of the buffer)pydeepstream";
            constexpr const char* init_empty = R"pydeepstream(Construct an empty buffer.)pydeepstream";
            constexpr const char* init_bytes = R"pydeepstream(
                Construct a buffer from a list of bytes.

                :arg data: *list of int* Byte list to construct the buffer)pydeepstream";

            constexpr const char* init_gst_buffer = R"pydeepstream(
                Typecast a buffer from a GstBuffer object.

                :arg gst_buffer: :class:`Gst.Buffer` GstBuffer object)pydeepstream
                :returns: :class:`Buffer` Buffer object)pydeepstream";
            constexpr const char* get_chunk_id = R"pydeepstream(
                Retrieve chunk ID of the buffer from the batch ID, in case of dynamic src decoder.
                Always returns 0 in case of uridecodebin.

                :arg batch_id: Batch id of the buffer
                :returns: Chunk id of the buffer)pydeepstream";
            constexpr const char* extract = R"pydeepstream(
                Extract data from the buffer in the batch with given id to a tensor

                :arg batch_id: *int* Batch id of the buffer
                :returns: :class:`Tensor` Buffer data as tensor)pydeepstream";
        }

        namespace FeederDoc
        {
            constexpr const char* descr = R"pydeepstream(Class for injecting buffers to an appsrc instance, requiring a :class:`BufferProvider` implementation to generate buffers)pydeepstream";
            constexpr const char* init = R"pydeepstream(
                Initialize the feeder with the given name and buffer provider

                :arg name: *str* Name of the feeder
                :arg provider: :class:`BufferProvider` Buffer provider)pydeepstream";
        }

        namespace BufferProviderDoc
        {
            constexpr const char* descr = R"pydeepstream(Interface for generating buffers)pydeepstream";
            constexpr const char* init = R"pydeepstream(Construct an empty buffer provider.)pydeepstream";
            constexpr const char* generate = R"pydeepstream(
                Generate a buffer of size *size*. Empty buffer indicates the EOS

                :arg size: *int* Size of the buffer
                :returns: :class:`Buffer` Generated buffer object)pydeepstream";
        }

        namespace ReceiverDoc
        {
            constexpr const char* descr = R"pydeepstream(Class for receiving buffers from an appsink instance, requiring a :class:`BufferRetriever` implementation to consume the buffers.)pydeepstream";
            constexpr const char* init = R"pydeepstream(
                Initialize the receiver with the given name and buffer retriever.

                :arg name: *str* Name of the receiver
                :arg retriever: :class:`BufferRetriever` Buffer retriever)pydeepstream";
        }

        namespace BufferRetrieverDoc
        {
            constexpr const char* descr = R"pydeepstream(Interface for retrieving buffers)pydeepstream";
            constexpr const char* init = R"pydeepstream(Construct an empty buffer retriever.)pydeepstream";
            constexpr const char* consume = R"pydeepstream(
                Consume a buffer.

                :arg receiver: :class:`Receiver` Receiver to consume the buffer
                :arg buffer: :class:`Receiver` Buffer to consume
                :returns: Consumed bytes, a negative value indicates error.)pydeepstream";
        }
    }
}