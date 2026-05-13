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
 
#ifndef BACKEND_PANGO_CAIRO_HPP
#define BACKEND_PANGO_CAIRO_HPP

#include "backend.hpp"

#ifdef ENABLE_TEXT_BACKEND_PANGO
std::shared_ptr<TextBackend> create_pango_cairo_backend();
#endif // ENABLE_TEXT_BACKEND_PANGO

#endif // BACKEND_PANGO_CAIRO_HPP
