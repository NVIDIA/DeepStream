/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <data_feeder.hpp>
#include <data_receiver.hpp>
#include <buffer.hpp>
#include "bufferdoc.h"
#include <gst/gst.h>

class BufferProvider_ : public DataFeeder::IDataProvider {
public:
    Buffer read(DataFeeder& feeder, unsigned int size, bool& eos) override {
        Buffer buffer = this->generate(size);
        eos = !buffer;
        return buffer;
    }

    virtual Buffer generate(unsigned int size) = 0;
};

class PyBufferProvider : public BufferProvider_ {
    virtual Buffer generate(unsigned int size) {
        PYBIND11_OVERRIDE_PURE_NAME(
            Buffer,
            BufferProvider_,
            "generate",
            generate,
            size
        );
    }
};

class Feeder_ : public DataFeeder {
public:
    Feeder_(const string&name, BufferProvider_* provider)
    : DataFeeder(name, provider)
    {}

    virtual ~Feeder_() {
        /** relinguish the ownership, never try deleting the python object */
        data_provider_.release();
    }
};

class BufferRetriever_ : public DataReceiver::IDataConsumer {
public:
    int consume(DataReceiver& receiver, Buffer buffer) override {
        return this->retrieve(buffer);
    }

    virtual int retrieve(Buffer buffer) = 0;
};

class PyBufferRetriever : public BufferRetriever_ {
    virtual int retrieve(Buffer buffer) {
        PYBIND11_OVERRIDE_PURE_NAME(
            int,
            BufferRetriever_,
            "consume",
            retrieve,
            buffer
        );
    }
};

class Receiver_ : public DataReceiver {
public:
    Receiver_(const string& name, BufferRetriever_* consumer)
    : DataReceiver(name, consumer)
    {}

    virtual ~Receiver_() {
        /** relinguish the ownership, never try deleting the python object */
        data_consumer_.release();
    }
};

void module_buffer_bind(py::module &m);

void module_buffer_bind(py::module &m) {
    py::class_<Buffer>(m, "Buffer", pydeepstreamdoc::buffer::BufferDoc::descr)
        .def(py::init<>(), pydeepstreamdoc::buffer::BufferDoc::init_empty)
        .def(py::init<vector<uint8_t>>(), pydeepstreamdoc::buffer::BufferDoc::init_bytes, py::arg("data"))
        .def(py::init([](py::object gst_buffer_obj) {
            PyObject* py_obj = gst_buffer_obj.ptr();
            if (!py_obj || !Py_TYPE(py_obj) || !Py_TYPE(py_obj)->tp_name ||
                (strcmp(Py_TYPE(py_obj)->tp_name, "Buffer") != 0 &&
                 strcmp(Py_TYPE(py_obj)->tp_name, "Gst.Buffer") != 0 &&
                !strstr(Py_TYPE(py_obj)->tp_name, "gi.repository.Gst.Buffer"))) {
                return std::make_unique<Buffer>();
            }
            py::int_ hash_obj = py::reinterpret_steal<py::int_>(PyObject_CallMethod(py_obj, "__hash__", NULL));
            GstBuffer* gst_buffer = reinterpret_cast<GstBuffer*>(hash_obj.cast<size_t>());
            OpaqueBuffer* opaque_buffer = reinterpret_cast<OpaqueBuffer*>(gst_buffer);
            return std::make_unique<Buffer>(opaque_buffer);
        }), pydeepstreamdoc::buffer::BufferDoc::init_gst_buffer, py::arg("gst_buffer"))
        .def_property_readonly("batch_size", &Buffer::batchSize)
        .def_property_readonly("batch_meta", [](Buffer& self) {
            return self.getBatchMetadata();
        })
        .def_property_readonly("timestamp", &Buffer::timestamp)
        .def("measure_latency", [](Buffer& self) {
            py::list result;
            for (auto& l : self.measureLatency()) {
                py::dict entry;
                entry["source_id"] = l.source_id;
                entry["frame_num"] = l.frame_num;
                entry["latency"]   = l.latency;
                result.append(entry);
            }
            return result;
        })
        .def("get_chunk_id", [](Buffer& self, unsigned int id) {
            return self.chunkId(id);
        }, pydeepstreamdoc::buffer::BufferDoc::get_chunk_id)
        .def("extract", [](Buffer& self, unsigned int id) {
            return make_unique<TensorWrapper>(self.extract(id));
        },
        pydeepstreamdoc::buffer::BufferDoc::extract, py::arg("batch_id")
        )
        .def("__repr__", [](Buffer& self) {
            return "Buffer(batch_size=" + std::to_string(self.batchSize()) + 
                ", batch_meta=" + py::repr(py::cast(self.getBatchMetadata())).cast<std::string>() +
                ", timestamp=" + std::to_string(self.timestamp()) + ")";
        });
    py::class_<Feeder_>(m, "Feeder", pydeepstreamdoc::buffer::FeederDoc::descr)
        .def(py::init<const string&, BufferProvider_*>(), py::keep_alive<1, 3>(), pydeepstreamdoc::buffer::FeederDoc::init, py::arg("name"), py::arg("provider"));
    py::class_<BufferProvider_, PyBufferProvider>(m, "BufferProvider", pydeepstreamdoc::buffer::BufferProviderDoc::descr)
        .def(py::init<>(), pydeepstreamdoc::buffer::BufferProviderDoc::init)
        .def("generate", &BufferProvider_::generate, pydeepstreamdoc::buffer::BufferProviderDoc::generate);
    py::class_<Receiver_>(m, "Receiver", pydeepstreamdoc::buffer::ReceiverDoc::descr)
        .def(py::init<const string&, BufferRetriever_*>(), py::keep_alive<1, 3>(),  pydeepstreamdoc::buffer::ReceiverDoc::init, py::arg("name"), py::arg("retriever"));
    py::class_<BufferRetriever_, PyBufferRetriever>(m, "BufferRetriever", pydeepstreamdoc::buffer::BufferRetrieverDoc::descr)
        .def(py::init<>(), pydeepstreamdoc::buffer::BufferRetrieverDoc::init)
        .def("consume", &BufferRetriever_::retrieve, pydeepstreamdoc::buffer::BufferRetrieverDoc::consume);
}