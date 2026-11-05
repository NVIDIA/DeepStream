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

#include "modules/NvMultiObjectTracker/nvdstracker.h"

#ifndef UTILITY_TIMESTAMP_HPP
#define UTILITY_TIMESTAMP_HPP

/** @brief Formats and compares system-clock timestamps. */
class Time {
    public:
        /** @brief Initialize the timestamp to the current system time. */
#ifdef _WIN32
        Time() : m_timePoint(std::chrono::system_clock::time_point(
            std::chrono::duration_cast<std::chrono::system_clock::duration>(
                std::chrono::nanoseconds(NvDsCurrentTimestampNs())))) {}
#else
        Time() : m_timePoint(std::chrono::nanoseconds(NvDsCurrentTimestampNs())) {}
#endif

        /**
         * @brief Initialize from a system-clock time point.
         * @param[in] time System-clock time point to store.
         */
        Time(std::chrono::system_clock::time_point time) : m_timePoint(time) {}

        /** @brief Release the timestamp wrapper. */
        ~Time() {}

        /** @brief Return nanoseconds elapsed since the Unix epoch. */
        TimeStamp getUnixTimestampNanoSeconds()
        {
            return std::chrono::duration_cast<std::chrono::nanoseconds>(m_timePoint.time_since_epoch()).count();
        }

        /** @brief Return the stored system-clock time point. */
        std::chrono::system_clock::time_point getTimePoint()
        {
            return m_timePoint;
        }

        /**
         * @brief Format the stored timestamp relative to another time point.
         * @param[in] time Reference time point subtracted from the stored timestamp.
         */
        std::string getStrTimeRelativeTo(std::chrono::system_clock::time_point time)
        {
            TimeStamp nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(m_timePoint - time).count();
            return getStrTimeFromNanoseconds(nanoseconds, "%H:%M:%S");
        }

        /** @brief Format the stored timestamp as a system date and time. */
        std::string getStrSystemTime()
        {
            TimeStamp nanoseconds = getUnixTimestampNanoSeconds();
            return getStrTimeFromNanoseconds(nanoseconds, "%Y-%m-%d %H:%M:%S");
        }

    private:
        std::chrono::system_clock::time_point m_timePoint; //!< Stored system-clock time point.

        /**
         * @brief Format a nanosecond timestamp with microsecond precision.
         * @param[in] nanoseconds Nanoseconds since the Unix epoch.
         * @param[in] format Strftime-compatible base time format.
         */
        std::string getStrTimeFromNanoseconds(TimeStamp nanoseconds, std::string format)
        {
            std::tm tm;
            std::stringstream ss;
            std::time_t seconds = static_cast<std::time_t>(nanoseconds / 1000000000ULL);
            TimeStamp frac_ns = nanoseconds % 1000000000ULL;
            TimeStamp frac_us = frac_ns / 1000ULL;
#ifdef _WIN32
            localtime_s(&tm, &seconds);
#else
            localtime_r(&seconds, &tm);
#endif
            ss << std::put_time(&tm, format.c_str()) << "." << std::setfill('0') << std::setw(6) << frac_us;
            return ss.str();
        }
};

#endif // UTILITY_TIMESTAMP_HPP
