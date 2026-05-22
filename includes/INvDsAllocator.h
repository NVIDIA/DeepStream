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

#ifndef _NVDS_ALLOCATOR_H_
#define _NVDS_ALLOCATOR_H_

#include <stdint.h>

class INvDsAllocator
{
    public:
    /**
     * @brief  Allocate memory of @size Bytes
     * @param  size [IN] The # of bytes to allocate
     * @return The newly allocated Memory buffer handle
     */
    virtual void* Allocate (uint32_t size) = 0;
    /**
     * @brief Deallocate the memory allocated using Allocate()
     * @param data [IN] The Memory buffer handle
     */
    virtual void Deallocate (void* data) = 0;
    virtual ~INvDsAllocator() {}
};

#endif
