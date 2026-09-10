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

// #define PYBIND11_DETAILED_ERROR_MESSAGES
#include <nvdsmeta.h>
#include <nvds_analytics_meta.h>
#include <nvds_tracker_meta.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pipeline.hpp>
#include <tensor.hpp>
#include <data_receiver.hpp>
#include <common_factory.hpp>
#include <lib/perf_monitor.hpp>
#include <lib/model_engine_watch_otf_trigger.hpp>
#include <mediainfo.hpp>
#include <source_manager.hpp>

namespace py = pybind11;
using namespace deepstream;
using namespace std;

#include "probe.h"
#include "tensor.h"
#include "buffer.h"
#include "node.h"
#include "osd.h"
#include "metadata.h"
#include "value.h"
#include "pipeline.h"
#include "object.h"
#include "signal_emitter.h"
#include "utils.h"


PYBIND11_MODULE(_pydeepstream, m) {
    m.doc() = "Python binding of Deepstream SDK";

    module_osd_bind(m);
    module_tensor_bind(m);
    module_buffer_bind(m);
    module_metadata_bind(m);
    module_probe_bind(m);
    module_node_bind(m);
    module_pipeline_bind(m);
    module_object_bind(m);
    module_signal_bind(m);
    module_utils_bind(m);
}
