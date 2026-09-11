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

#ifndef PROMETHEUS_CPP_CORE_EXPORT_H
#define PROMETHEUS_CPP_CORE_EXPORT_H

#ifdef PROMETHEUS_CPP_CORE_STATIC_DEFINE
#  define PROMETHEUS_CPP_CORE_EXPORT
#  define PROMETHEUS_CPP_CORE_NO_EXPORT
#else
#  ifndef PROMETHEUS_CPP_CORE_EXPORT
#    ifdef PROMETHEUS_CPP_CORE_EXPORTS
        /* We are building this library */
#      define PROMETHEUS_CPP_CORE_EXPORT __attribute__((visibility("default")))
#    else
        /* We are using this library */
#      define PROMETHEUS_CPP_CORE_EXPORT __attribute__((visibility("default")))
#    endif
#  endif

#  ifndef PROMETHEUS_CPP_CORE_NO_EXPORT
#    define PROMETHEUS_CPP_CORE_NO_EXPORT __attribute__((visibility("hidden")))
#  endif
#endif

#ifndef PROMETHEUS_CPP_CORE_DEPRECATED
#  define PROMETHEUS_CPP_CORE_DEPRECATED __attribute__ ((__deprecated__))
#endif

#ifndef PROMETHEUS_CPP_CORE_DEPRECATED_EXPORT
#  define PROMETHEUS_CPP_CORE_DEPRECATED_EXPORT PROMETHEUS_CPP_CORE_EXPORT PROMETHEUS_CPP_CORE_DEPRECATED
#endif

#ifndef PROMETHEUS_CPP_CORE_DEPRECATED_NO_EXPORT
#  define PROMETHEUS_CPP_CORE_DEPRECATED_NO_EXPORT PROMETHEUS_CPP_CORE_NO_EXPORT PROMETHEUS_CPP_CORE_DEPRECATED
#endif

#if 0 /* DEFINE_NO_DEPRECATED */
#  ifndef PROMETHEUS_CPP_CORE_NO_DEPRECATED
#    define PROMETHEUS_CPP_CORE_NO_DEPRECATED
#  endif
#endif

#endif /* PROMETHEUS_CPP_CORE_EXPORT_H */
