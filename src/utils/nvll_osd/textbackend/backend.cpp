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

#include "backend.hpp"

#ifdef ENABLE_TEXT_BACKEND_PANGO
#include "pango-cairo.hpp"
#endif

#ifdef ENABLE_TEXT_BACKEND_STB
#include "stb.hpp"
#endif

#include <sstream>
#include <stdio.h>

const char* text_backend_type_name(TextBackendType backend){

    switch(backend){
    case TextBackendType::PangoCairo:  return "PangoCairo";
    case TextBackendType::StbTrueType: return "StbTrueType";
    default: return "Unknow";
    }
}

std::shared_ptr<TextBackend> create_text_backend(TextBackendType backend){

    switch(backend){

    #ifdef ENABLE_TEXT_BACKEND_PANGO
    case TextBackendType::PangoCairo:  return create_pango_cairo_backend();
    #endif

    #ifdef ENABLE_TEXT_BACKEND_STB
    case TextBackendType::StbTrueType: return create_stb_backend();
    #endif

    default:
        printf("Unsupport text backend: %s\n", text_backend_type_name(backend));
        return nullptr;
    }
}

std::string concat_font_name_size(const char* name, int size){
    std::stringstream ss;
    ss << name;
    ss << " ";
    ss << size;
    return ss.str();
}
