/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "signalemitterdoc.h"

void module_signal_bind(py::module &m);

void module_signal_bind(py::module &m) {
    py::module signal_module = m.def_submodule("signal");
    py::class_<SignalEmitter, CustomObject>(signal_module, "Emitter", pydeepstreamdoc::signalemitter::EmitterDoc::descr)
        .def("attach",  [](SignalEmitter& self, string action, Node& node) {
          self.attach(action, node);
        }, pydeepstreamdoc::signalemitter::EmitterDoc::attach);
    py::class_<SourceManager, SignalEmitter>(signal_module, "SourceManager", pydeepstreamdoc::signalemitter::SourceManagerDoc::descr)
        .def(py::init<string>())
        .def("add_source",  &SourceManager::addSource, py::call_guard<py::gil_scoped_release>(), pydeepstreamdoc::signalemitter::SourceManagerDoc::add_source)
        .def("remove_source",  &SourceManager::removeSource, py::call_guard<py::gil_scoped_release>(), pydeepstreamdoc::signalemitter::SourceManagerDoc::remove_source)
        .def("terminate",  &SourceManager::terminate, py::call_guard<py::gil_scoped_release>(), pydeepstreamdoc::signalemitter::SourceManagerDoc::terminate);
}