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

#include <queue>
#include <mutex>

/// Simple concurrent Queue class using an stl queue.
/// Nothing is special here compare to stl queue except
/// it has only simple operations and it is thread safe.
template <typename T>
class ConcurrentQueue
{
public:
    void push(const T &elm);
    T pop();
    bool is_empty();

private:
    std::queue<T> queue_;
    std::mutex mutex_;
};

template <typename T>
void ConcurrentQueue<T>::push(const T &elm)
{
    mutex_.lock();
    queue_.push(elm);
    mutex_.unlock();
}

template <typename T>
T ConcurrentQueue<T>::pop()
{
    mutex_.lock();
    T elm = queue_.front();
    queue_.pop();
    mutex_.unlock();
    return elm;
}

template <typename T>
bool ConcurrentQueue<T>::is_empty()
{
    mutex_.lock();
    bool res = queue_.empty();
    mutex_.unlock();
    return res;
}