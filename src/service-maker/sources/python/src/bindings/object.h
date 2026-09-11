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

#include "objectdoc.h"

class CommonFactoryWrapper {
    public:
      static unique_ptr<CustomObject> create(const string& plugin, const string& name) {
        return CommonFactory::getInstance().createObject(plugin, name);
      }
};

void module_object_bind(py::module &m);

void module_object_bind(py::module &m) {
    py::class_<CustomObject>(m, "Object", pydeepstreamdoc::object::ObjectDoc::descr)
        .def("set",  [](CustomObject& self, py::dict args){
            for (auto item : args) {
                string key = item.first.cast<string>();
                Object::Value value = item.second.cast<Object::Value>();
                self.set(key, value);
            }
        }, pydeepstreamdoc::object::ObjectDoc::set);
    py::class_<CommonFactoryWrapper>(m, "CommonFactory", pydeepstreamdoc::object::CommonFactoryDoc::descr)
        .def("create", &CommonFactoryWrapper::create, pydeepstreamdoc::object::CommonFactoryDoc::create);
}