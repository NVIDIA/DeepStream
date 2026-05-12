/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */


#ifndef DS3D_COMMON_HELPER_SAFE_QUEUE_H
#define DS3D_COMMON_HELPER_SAFE_QUEUE_H

#include <ds3d/common/common.h>
#include <ds3d/common/func_utils.h>

#include <chrono>
#include <deque>

namespace ds3d {

template <typename T, typename Container = std::deque<T>>
class SafeQueue {
    // using namespace std::chrono_literals;
public:
    void push(T data)
    {
        std::unique_lock<std::mutex> lock(_mutex);
        _queue.emplace_back(std::move(data));
        _cond.notify_one();
    }
    T pop(uint64_t timeoutMs = 0)
    {
        std::unique_lock<std::mutex> lock(_mutex);
        auto stopWait = [this]() { return _wakeupOnce || !_queue.empty(); };
        if (!timeoutMs) {
            _cond.wait(lock, stopWait);
        } else {
            using namespace std::chrono_literals;
            if (!_cond.wait_for(lock, timeoutMs * 1ms, stopWait)) {
                throw Exception(ErrCode::kTimeOut, "queue pop timeout");
            }
        }
        if (_wakeupOnce) {
            _wakeupOnce = false;
            LOG_DEBUG("SafeQueue pop end on wakeup signal");
            throw Exception(ErrCode::kLockWakeup, "queue wakedup");
        }
        assert(!_queue.empty());
        T ret = std::move(*_queue.begin());
        _queue.erase(_queue.begin());
        return ret;
    }
    void wakeupOnce()
    {
        LOG_DEBUG("SafeQueue trigger wakeup once");
        std::unique_lock<std::mutex> lock(_mutex);
        _wakeupOnce = true;
        _cond.notify_all();
    }
    void clear()
    {
        LOG_DEBUG("SafeQueue clear");
        std::unique_lock<std::mutex> lock(_mutex);
        _queue.clear();
        _wakeupOnce = false;
    }
    size_t size()
    {
        std::unique_lock<std::mutex> lock(_mutex);
        return _queue.size();
    }

private:
    std::mutex _mutex;
    std::condition_variable _cond;
    Container _queue;
    bool _wakeupOnce = false;
};

template <class UniPtr>
class BufferPool : public std::enable_shared_from_this<BufferPool<UniPtr>> {
public:
    using ItemType = typename UniPtr::element_type;
    using RecylePtr = std::unique_ptr<ItemType, std::function<void(ItemType*)>>;
    BufferPool(const std::string& name) : m_Name(name) {}
    virtual ~BufferPool()
    {
        LOG_DEBUG(
            "BufferPool: %s deleted with free buffer size:%d", m_Name.c_str(),
            (int)m_FreeBuffers.size());
    }
    bool setBuffer(UniPtr buf)
    {
        assert(buf);
        m_FreeBuffers.push(std::move(buf));
        LOG_DEBUG(
            "BufferPool: %s set buf to free, available size:%d", m_Name.c_str(),
            (int)m_FreeBuffers.size());
        return true;
    }
    uint32_t size() { return m_FreeBuffers.size(); }

    RecylePtr acquireBuffer()
    {
        try {
            UniPtr p = m_FreeBuffers.pop();
            auto deleter = p.get_deleter();
            std::weak_ptr<BufferPool<UniPtr>> poolPtr = this->shared_from_this();
            RecylePtr recBuf(p.release(), [poolPtr, d = deleter](ItemType* buf) {
                assert(buf);
                UniPtr data(buf, d);
                auto pool = poolPtr.lock();
                if (pool) {
                    LOG_DEBUG("BufferPool: %s release a buffer", pool->m_Name.c_str());
                    pool->setBuffer(std::move(data));
                } else {
                    LOG_DEBUG("BufferPool was deleted before buffer release, maybe application is closing.");
                    //assert(false);
                }
            });
            LOG_DEBUG(
                "BufferPool: %s acquired buffer, available free buffer left:%d", m_Name.c_str(),
                (int)m_FreeBuffers.size());
            return recBuf;
        }
        catch (...) {
            LOG_DEBUG(
                "BufferPool: %s acquired buffer failed, queue may be waked up", m_Name.c_str());
            assert(false);
            return nullptr;
        }
    }

private:
    SafeQueue<UniPtr> m_FreeBuffers;
    const std::string m_Name;
};

}  // namespace ds3d

#endif  //