/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <stdarg.h>

#include "nvdsinfer_logger.h"

namespace nvdsinfer {

class NvDsInferLogger : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char* msg) noexcept override
    {
        switch (severity)
        {
            case Severity::kINTERNAL_ERROR:
            case Severity::kERROR:
                dsInferError("[TRT]: %s", msg);
                break;
            case Severity::kWARNING:
                dsInferWarning("[TRT]: %s", msg);
                break;
            case Severity::kINFO:
                dsInferDebug("[TRT]: %s", msg);
                break;
            case Severity::kVERBOSE:
                dsInferDebug("[TRT]: %s", msg);
                break;
            default:
                dsInferLogPrint__((NvDsInferLogLevel)100, "[TRT][severity:%d]: %s",
                        (int)severity, msg);
                return;
        }
    }
};

static const char*
strLogLevel(NvDsInferLogLevel l)
{
    switch (l)
    {
        case NVDSINFER_LOG_ERROR:
            return "ERROR";
        case NVDSINFER_LOG_WARNING:
            return "WARNING";
        case NVDSINFER_LOG_INFO:
            return "INFO";
        case NVDSINFER_LOG_DEBUG:
            return "DEBUG";
        default:
            return "UNKNOWN";
    }
}

struct LogEnv
{
    NvDsInferLogLevel levelLimit = NVDSINFER_LOG_INFO;
    std::mutex printMutex;
    LogEnv()
    {
        const char* cEnv = std::getenv("NVDSINFER_LOG_LEVEL");
        if (cEnv)
        {
            levelLimit = (NvDsInferLogLevel)std::stoi(cEnv);
        }
    }
};

static LogEnv gLogEnv;


void dsInferLogPrint__(NvDsInferLogLevel level, const char* fmt, ...)
{
    if (level > nvdsinfer::gLogEnv.levelLimit)
    {
        return;
    }
    constexpr int kMaxBufLen = 4096;

    va_list args;
    va_start(args, fmt);
    std::array<char, kMaxBufLen> logMsgBuffer{{'\0'}};
    vsnprintf(logMsgBuffer.data(), kMaxBufLen - 1, fmt, args);
    va_end(args);

    FILE* f = (level <= NVDSINFER_LOG_ERROR) ? stderr : stdout;

    std::unique_lock<std::mutex> locker(nvdsinfer::gLogEnv.printMutex);
    fprintf(f, "%s: %s\n", nvdsinfer::strLogLevel(level), logMsgBuffer.data());
}
} // namespace nvdsinfer

std::unique_ptr<nvinfer1::ILogger> gTrtLogger(new nvdsinfer::NvDsInferLogger);
