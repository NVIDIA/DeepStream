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

#ifndef DS3D_COMMON_HPP_PROFILING_HPP
#define DS3D_COMMON_HPP_PROFILING_HPP

// inlcude all ds3d hpp header files
#include <ds3d/common/common.h>
#include <sys/time.h>
#include <chrono>

namespace ds3d { namespace profiling {

class FpsCalculation {
public:
    FpsCalculation(uint32_t interval) : _max_frame_nums(interval) {}
    float updateFps(uint32_t source_id)
    {
        struct timeval time_now;
        gettimeofday(&time_now, nullptr);
        double now = (double)time_now.tv_sec + time_now.tv_usec / (double)1000000;  // second
        float fps = -1.0f;
        auto iSrc = _timestamps.find(source_id);
        if (iSrc != _timestamps.end()) {
            auto& tms = iSrc->second;
            fps = tms.size() / (now - tms.front());
            while (tms.size() >= _max_frame_nums) {
                tms.pop();
            }
        } else {
            iSrc = _timestamps.emplace(source_id, std::queue<double>()).first;
        }
        iSrc->second.push(now);

        return fps;
    }

private:
    std::unordered_map<uint32_t, std::queue<double>> _timestamps;
    uint32_t _max_frame_nums = 50;
};

class Timing {
public:
    Timing(uint32_t maxSlots = 50) : _maxTimeLotNum(maxSlots) {}
    ~Timing() = default;
    void push(double t)
    {
        _timelots.push(t);
        _total += t;
        DS_ASSERT(_maxTimeLotNum > 0);
        while (_timelots.size() > _maxTimeLotNum) {
            _total -= _timelots.front();
            _timelots.pop();
        }
    }
    double avg() const
    {
        if (_timelots.size()) {
            return _total / _timelots.size();
        }
        return 0;
    }

private:
    std::queue<double> _timelots;
    double _total = 0;
    uint32_t _maxTimeLotNum = 0;
};

class FileWriter {
    std::ofstream _file;
    std::string _path;

public:
    FileWriter() = default;
    ~FileWriter() { close(); }

    bool open(const std::string& path, std::ios::openmode mode = std::ios::out | std::ios::binary)
    {
        _path = path;
        _file.open(path.c_str(), mode);
        return _file.is_open();
    }
    bool isOpen() const { return _file.is_open(); }

    void close()
    {
        if (_file.is_open()) {
            _file.close();
        }
    }

    bool write(const void* buf, size_t size)
    {
        DS_ASSERT(_file.is_open());
        return _file.write((const char*)buf, size).good();
    }
};

class FileReader {
    std::ifstream _file;
    std::string _path;

public:
    FileReader() = default;
    ~FileReader() { close(); }

    bool open(const std::string& path, std::ios::openmode mode = std::ios::in | std::ios::binary)
    {
        _path = path;
        _file.open(path.c_str(), mode);
        return _file.is_open();
    }
    bool isOpen() const { return _file.is_open(); }
    bool eof() const { return _file.eof(); }

    void close()
    {
        if (_file.is_open()) {
            _file.close();
        }
    }

    int32_t read(void* buf, size_t size)
    {
        DS_ASSERT(_file.is_open());
        if (_file) {
            return (int32_t)_file.readsome((char*)buf, size);
        }
        return -1;
    }
};

}}  // namespace ds3d::profiling

#endif  // DS3D_COMMON_HPP_PROFILING_HPP