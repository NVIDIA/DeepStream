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

#ifndef DEBUG_COMMON_HPP_
#define DEBUG_COMMON_HPP_

#include <cstdlib>
#include <iostream>
#include <string>
#include <cassert>

#define CLOG(message) \
	do { \
		fprintf(stdout, "~~ CLOG[%s, %s() @line %d]: %s\n", __FILE__, __func__, __LINE__,  message); \
	} while (0)

#define MEAS_TIME(func, doPrint, name) \
  do { \
    if (doPrint) { \
    auto timeStart = std::chrono::high_resolution_clock::now(); \
    func; \
    double timeUsed = std::chrono::duration<double>(std::chrono::high_resolution_clock::now()-timeStart).count() * 1e+3; \
    fprintf(stdout, "~~ MEAS_TIME[%s]: %.2f ms\n", name, timeUsed); } \
    else { func; } \
  } while (0)

namespace DEBUG_UTIL
{

/**
 * @brief Format a string using printf-style arguments.
 * @param[in] format Printf-style format string.
 * @param[in] args Values substituted into the format string.
 * @return Formatted string.
 */
template< typename... Args >
std::string to_string_fmt( const char* format, Args... args )
{
  int length = std::snprintf( nullptr, 0, format, args... );
  assert( length >= 0 );

  char* buf = new char[length + 1];
  std::snprintf( buf, length + 1, format, args... );

  std::string str( buf );
  delete[] buf;
  return str;
}

}

#endif // DEBUG_COMMON_HPP_
