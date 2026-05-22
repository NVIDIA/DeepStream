
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


#ifndef __COMMAND_LINE_H_
#define __COMMAND_LINE_H_

#include <stdint.h>
#include <stdlib.h>
#include <vector>
#include <string>

#include <string.h>
#include <strings.h>

/**
 * args line parser 
 */
class argsParser {
public:
    argsParser(const int argc, char** argv);

    /**
     * Parse Flag
     */
    bool ParseFlag(const std::string argName) const;

    /**
     * Parse String
     */
    const char* ParseString(const std::string  argName) const;
    // const char* ParseString2(const std::string argName, const char* defaultValue = NULL, bool allowOtherDelimiters = true) const;

    /**
     * Parse String list delimited by ","
     */
    std::vector<std::string> ParseStringList(std::string argName, const char delimiter = ',') const;

    /**
     * The argument count that the object was created with from main()
     */
    int argc;
    char** argv;
};

#endif
