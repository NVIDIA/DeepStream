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
    namespace tensor
    {
        namespace ColorFormatDoc
        {
            constexpr const char* descr = R"pydeepstream(Color format enum.)pydeepstream";
        }

        namespace TensorDoc
        {
            constexpr const char* descr = R"pydeepstream(
                Tensor abstraction class.

                :ivar shape: *tuple*, Shape of the tensor.
                :ivar strides: *tuple*, Strides of the tensor.
                :ivar dtype: *dtype*, Data type of the tensor.)pydeepstream";
            constexpr const char* dlpack = R"pydeepstream(Get the tensor as DLPack capsule.)pydeepstream";
            constexpr const char* dlpack_device = R"pydeepstream(Get the device type and device ID in DLPack format.)pydeepstream";
            constexpr const char* wrap = R"pydeepstream(
                Wrap a tensor as a buffer.

                :arg format: *:class:`ColorFormat`*, Color format of the buffer.
                :return: *:class:`Buffer`*, Buffer of wrapped tensor.)pydeepstream";
            constexpr const char* clone = R"pydeepstream(Clone a tensor.)pydeepstream";
            constexpr const char* to_gpu = R"pydeepstream(Copy the tensor to GPU given ID of target device.)pydeepstream";
            constexpr const char* size = R"pydeepstream(Get the size of the tensor in bytes.)pydeepstream";
            constexpr const char* as_tensor = R"pydeepstream(
                Wrap tensor data from other frameworks.

                :arg object: *object*, Object to wrap. If has __dlpack_device__ attribute, it will fill the :class:`Tensor` from the given DLPack information.
                    Otherwise, it will create an empty :class:`Tensor`
                :arg format: *str*, Format of the tensor.
                :return: *:class:`Tensor`*, Wrapped :class:`Tensor`.)pydeepstream";
        }
    }
}