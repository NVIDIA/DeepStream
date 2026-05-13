/*
 * SPDX-FileCopyrightText: Copyright (c) 2020 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
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