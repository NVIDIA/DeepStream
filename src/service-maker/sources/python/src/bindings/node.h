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

#include "nodedoc.h"

class Node : public Element {
public:
    Node& link(Node& target) {
        Element::link(static_cast<Element&>(target));
        return *this;
    }

    Node& link(Node& target, pair<string, string> tips) {
        Element::link(static_cast<Element&>(target), tips);
        return *this;
    }

    Node& attach(Probe* probe, const string tips) {
        Element::addProbe(probe, tips);
        return *this;
    }

    Node& attach(Receiver_* receiver, const string tips) {
        size_t begin = 0;
        size_t end = 0;
        while (end != std::string::npos) {
            end = tips.find("/", begin);
            std::string signal = tips.substr(begin, end);
            Element::connectSignal(signal, receiver);
            begin = end+1;
        }
        return *this;
    }

    Node& attach(Feeder_* feeder, const string tips) {
        size_t begin = 0;
        size_t end = 0;
        while (end != std::string::npos) {
            end = tips.find("/", begin);
            std::string signal = tips.substr(begin, end);
            Element::connectSignal(signal, feeder);
            begin = end+1;
        }
        return *this;
    }

    CustomObject* find(const string& name) {
        auto itr = objects_->find(name);
        if (itr != objects_->end()) {
            return itr->second.get();
        }
        return nullptr;
    }

    virtual ~Node() {
        if (objects_.use_count() == 1) {
            /** relinguish the ownership, never try deleting the python objects */
            for (auto& pair: *objects_) {
                pair.second.release();
            }
        }
    }
};

void module_node_bind(py::module &m);

void module_node_bind(py::module &m) {
    py::class_<Node>(m, "Node", pydeepstreamdoc::node::NodeDoc::descr)
        .def_property_readonly("name", &Object::getName)
        .def("link", (Node& (Node::*)(Node&)) &Node::link, pydeepstreamdoc::node::NodeDoc::link)
        .def("link", (Node& (Node::*)(Node&, pair<string, string>)) &Node::link, pydeepstreamdoc::node::NodeDoc::link_info)
        .def("set",  [](Node& self, py::dict args){
            for (auto item : args) {
                string key = item.first.cast<string>();
                Object::Value value = item.second.cast<Object::Value>();
                self.set(key, value);
            }
        }, pydeepstreamdoc::node::NodeDoc::set)
        .def("set",  [](Node& self, py::list dicts){
            for (auto dict : dicts) {
                py::dict d = dict.cast<py::dict>();
                for (auto item : d) {
                    string key = item.first.cast<string>();
                    Object::Value value = item.second.cast<Object::Value>();
                    self.set(key, value);
                }
            }
        }, pydeepstreamdoc::node::NodeDoc::set2)
        .def("get", [](Node& self, string& key) {
            Object::Value value;
            self.getProperty(key, value);
            return value;
        }, pydeepstreamdoc::node::NodeDoc::get)
        .def("attach", (Node& (Node::*)(Probe*, const string)) &Node::attach, py::keep_alive<1, 2>(), pydeepstreamdoc::node::NodeDoc::attach_probe, py::arg("probe"), py::arg("tips")="")
        .def("attach", (Node& (Node::*)(Receiver_*, const string)) &Node::attach, py::keep_alive<1, 2>(), pydeepstreamdoc::node::NodeDoc::attach_receiver, py::arg("receiver"), py::arg("tips")="")
        .def("attach", (Node& (Node::*)(Feeder_*, const string)) &Node::attach, py::keep_alive<1, 2>(), pydeepstreamdoc::node::NodeDoc::attach_feeder, py::arg("feeder"), py::arg("tips")="")
        .def("find", &Node::find, py::return_value_policy::reference, pydeepstreamdoc::node::NodeDoc::find)
        .def("__repr__", [](Node& self) {
            return "Node(name=" + self.getName() + ")";
        });
}