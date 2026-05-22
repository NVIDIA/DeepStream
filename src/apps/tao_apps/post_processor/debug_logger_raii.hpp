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

#ifndef DEBUG_LOGGER_RAII_H
#define DEBUG_LOGGER_RAII_H

#include <fstream>
#include <string>
#include <cstdarg>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <cstring>
#include <iostream>

#define ENABLE_ENV_NAME "DEBUG"

/**
 * @brief Debug logger class using RAII (Resource Acquisition Is Initialization)
 * to automatically manage the log file.
 * 
 * This class is used to log debug messages to a file.
 * The log file is automatically closed when the object is destroyed.
 * With the help of the DEBUG_DUMP_SECTION macro, we can start and end a debug section.
 * Advantages of using this class:
 * - Automatically closes the log file even if there's an early return or exception.
 * - clearly defines the scope of debug logging using {} block
 * - No need to remember to close anything
 * - Thread-safe as each instance has its own file handle
 * - Object construction and destruction has minimal overhead when it is disabled
 * 
 * @param func The function name
 * @param line The line number
 * @param is_enabled Whether the logger is enabled
 * 
 * @note The log file is created in the /tmp directory.
 */
class DebugLoggerRAII {
public:
    DebugLoggerRAII(const char* func, int line, bool is_enabled);
    ~DebugLoggerRAII();
    void log(const char* format, ...);

private:
    std::string filename;
    std::ofstream log_file;
    const char* func_name;
    int start_line;
    bool enabled;
    
    // Helper function to get formatted timestamp
    static std::string GetTimestamp();
    // Helper function to get log header with timestamp and file info
    std::string GetLogHeader();
};


class DebugConfig {
private:
    static const bool enabled;
public:
    static bool IsEnabled() { return enabled; }
};

// Debug macros
#define DEBUG_DUMP_SECTION() \
    DebugLoggerRAII debug_logger(__func__, __LINE__, DebugConfig::IsEnabled())

#define DEBUG_DUMP(...) \
    debug_logger.log(__VA_ARGS__)

#endif // DEBUG_LOGGER_RAII_H