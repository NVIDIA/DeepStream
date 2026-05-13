/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
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
