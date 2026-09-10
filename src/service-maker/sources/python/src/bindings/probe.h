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

#include "probedoc.h"

class BufferObserver_ : public BufferProbe::IBufferObserver {
public:
    probeReturn handleBuffer(BufferProbe& probe, const Buffer& buffer) override {
        this->handle_buffer(buffer);
        return probeReturn::Probe_Ok;
    }

    virtual void handle_buffer(const Buffer& buffer) = 0;
};

class PyBufferObserver : public BufferObserver_ {
    virtual void handle_buffer(const Buffer& buffer) {
        PYBIND11_OVERRIDE_PURE_NAME(
            void,
            BufferObserver_,
            "handle_buffer",
            handle_buffer,
            buffer
        );
    }
};

class BufferOperator_ : public BufferProbe::IBufferOperator {
public:
    probeReturn handleBuffer(BufferProbe& probe, Buffer& buffer) override {
        auto ret = this->handle_buffer(buffer);
        return ret ? probeReturn::Probe_Ok:probeReturn::Probe_Drop;
    }

    virtual bool handle_buffer(Buffer buffer) = 0;
};

class PyBufferOperator : public BufferOperator_ {
    virtual bool handle_buffer(Buffer buffer) {
        PYBIND11_OVERRIDE_PURE_NAME(
            bool,
            BufferOperator_,
            "handle_buffer",
            handle_buffer,
            buffer
        );
    }
};

class BatchMetadataOperator_ : public BufferProbe::IBatchMetadataOperator {
public:
    probeReturn handleData(BufferProbe& probe, BatchMetadata& metadata) override {
        probe_ = &probe;
        this->handle_metadata(metadata);
        probe_ = nullptr;
        return probeReturn::Probe_Ok;
    }

    virtual void handle_metadata(BatchMetadata& metadata) = 0;

    Object::Value get_property(const std::string& name) {
        Object::Value v;
        if (probe_) probe_->getProperty(name, v);
        return v;
    }

protected:
    BufferProbe* probe_ = nullptr;
};

class PyBatchMetadataOperator : public BatchMetadataOperator_ {
    virtual void handle_metadata(BatchMetadata& metadata) {
        PYBIND11_OVERRIDE_PURE_NAME(
            void,
            BatchMetadataOperator_,
            "handle_metadata",
            handle_metadata,
            metadata
        );
    }

};

class Probe : public BufferProbe {
public:
    Probe(const string& name, BufferObserver_* handler)
    : BufferProbe(name, handler)
    {}

    Probe(const string& name, const char* factory, BufferObserver_* handler)
    : BufferProbe(name, factory, handler)
    {}

    Probe(const string& name, BufferOperator_* handler)
    : BufferProbe(name, handler)
    {}

    Probe(const string& name, const char* factory, BufferOperator_* handler)
    : BufferProbe(name, factory, handler)
    {}

    Probe(const string& name, BatchMetadataOperator_* handler)
    : BufferProbe(name, handler)
    {}

    Probe(const string& name, const char* factory, BatchMetadataOperator_* handler)
    : BufferProbe(name, factory, handler)
    {}

    virtual ~Probe() {
        /** relinguish the ownership, never try deleting the python object */
        metadata_handler_.release();
    }

    static unique_ptr<Probe> create(const string& plugin, const string& name) {
        auto obj = CommonFactory::getInstance().createObject(plugin, name).release();
        Probe* probe = dynamic_cast<Probe*>(obj);
        return unique_ptr<Probe>(probe);
    }
};

/**
 * CustomFactory backed by a Python class.
 *
 * Registered in CommonFactory by name so that
 *   pipeline.attach(node, "my_probe", "instance")
 * works identically to a compiled C++ plugin.
 */
class __attribute__((visibility("hidden"))) PythonProbeFactory : public CustomFactory {
    py::object py_class_;
    string param_spec_;
    string factory_name_;
public:
    PythonProbeFactory(const string& name, py::object py_class, string param_spec)
        : CustomFactory(name, 0)   // 0 → Object() skips g_object_new, no gst_custom_factory_init
        , py_class_(py_class), param_spec_(param_spec), factory_name_(name) {}

    CustomObject* createObject(const string& instance_name) override {
        py::gil_scoped_acquire gil;
        py::object py_instance = py_class_();
        Probe* probe = nullptr;
        const char* factory_name = factory_name_.c_str();
        if (py::isinstance<BufferObserver_>(py_instance)) {
            auto* obs = py_instance.cast<BufferObserver_*>();
            py_instance.release();
            probe = new Probe(instance_name, factory_name, obs);
        } else {
            auto* handler = py_instance.cast<BatchMetadataOperator_*>();
            py_instance.release();
            probe = new Probe(instance_name, factory_name, handler);
        }
        return probe;
    }

protected:
    // Serve our stored param spec so CustomObject::CustomObject populates
    // properties_ with the right defaults, without touching the null object_.
    Value get_(const std::string& property_name) override {
        if (property_name == "param-spec") {
            return Value(param_spec_);
        }
        return Value();
    }
};

void module_probe_bind(py::module &m);

void module_probe_bind(py::module &m) {
    py::class_<Probe>(m, "Probe", pydeepstreamdoc::probe::ProbeDoc::descr)
        .def(py::init<const string&, BufferOperator_*>(), py::keep_alive<1, 3>(), pydeepstreamdoc::probe::ProbeDoc::init_buffer_operator)
        .def(py::init<const string&, BatchMetadataOperator_*>(), py::keep_alive<1, 3>(), pydeepstreamdoc::probe::ProbeDoc::init_batch_metadata_operator)
        .def("create", &Probe::create, pydeepstreamdoc::probe::ProbeDoc::create)
        .def("set", [](Probe& self, py::dict args) {
            for (auto item : args) {
                string key = item.first.cast<string>();
                Object::Value value = item.second.cast<Object::Value>();
                self.set(key, value);
            }
        });
    py::class_<BufferObserver_, PyBufferObserver>(m, "BufferObserver", pydeepstreamdoc::probe::BufferObserverDoc::descr)
        .def(py::init<>())
        .def("handle_buffer", &BufferObserver_::handle_buffer, pydeepstreamdoc::probe::BufferObserverDoc::handle_buffer);
    py::class_<BufferOperator_, PyBufferOperator>(m, "BufferOperator", pydeepstreamdoc::probe::BufferOperatorDoc::descr)
        .def(py::init<>())
        .def("handle_buffer", &BufferOperator_::handle_buffer, pydeepstreamdoc::probe::BufferOperatorDoc::handle_buffer);
    py::class_<BatchMetadataOperator_, PyBatchMetadataOperator>(m, "BatchMetadataOperator", pydeepstreamdoc::probe::BatchMetadataOperatorDoc::descr)
        .def(py::init<>())
        .def("handle_metadata", &BatchMetadataOperator_::handle_metadata, pydeepstreamdoc::probe::BatchMetadataOperatorDoc::handle_metadata)
        .def("get_property", &BatchMetadataOperator_::get_property, py::arg("name"));

    m.def("_register_probe",
        [](const string& name, py::object cls, const string& param_spec) {
            CommonFactory::getInstance().addCustomFactory(
                new PythonProbeFactory(name, cls, param_spec), name.c_str());
        },
        py::arg("name"), py::arg("cls"), py::arg("param_spec") = string(""));
}