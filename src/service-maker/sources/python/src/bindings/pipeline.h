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

#include "pipelinedoc.h"

class __attribute__ ((visibility("hidden"))) PipelineWrapper : public Pipeline {
  public:
    PipelineWrapper(const string& name)
    :Pipeline(name.c_str()) {}

    PipelineWrapper(const string& name, const string& config_path)
    :Pipeline(name.c_str(), config_path) {}

    function<void(py::object)> on_message;
};

static void on_pipeline_message(Pipeline& p, const Pipeline::Message& m) {
    PipelineWrapper &pipeline = dynamic_cast<PipelineWrapper&>(p);
    py::gil_scoped_acquire acquire;
    py::object object;

    const Pipeline::StateTransitionMessage* state_transition = dynamic_cast<const Pipeline::StateTransitionMessage*>(&m);
    const Pipeline::DynamicSourceMessage* dynamic_source = dynamic_cast<const Pipeline::DynamicSourceMessage*>(&m);
    const Pipeline::EOSMessage* eos = dynamic_cast<const Pipeline::EOSMessage*>(&m);
    if (state_transition) {
        Pipeline::StateTransitionMessage message = *state_transition;
        object = py::cast(message);
    } else if (dynamic_source) {
        Pipeline::DynamicSourceMessage message = *dynamic_source;
        object = py::cast(message);
    } else if (eos) {
        Pipeline::EOSMessage message = *eos;
        object = py::cast(message);
    }

    if (pipeline.on_message && !object.is_none()) {
        // invoke python callback
        pipeline.on_message(object);
    }
}

void module_pipeline_bind(py::module &m);

void module_pipeline_bind(py::module &m) {
    py::enum_<Pipeline::State>(m, "PipelineState", pydeepstreamdoc::pipeline::PipelineStateDoc::descr)
        .value("INVALID", Pipeline::State::INVALID)
        .value("EMPTY", Pipeline::State::EMPTY)
        .value("READY", Pipeline::State::READY)
        .value("PAUSED", Pipeline::State::PAUSED)
        .value("PLAYING", Pipeline::State::PLAYING);
    auto pipeline_message = py::class_<Pipeline::Message>(m, "PipelineMessage", pydeepstreamdoc::pipeline::PipelineMessageDoc::descr);
    auto eos_message = py::class_<Pipeline::EOSMessage, Pipeline::Message>(m, "EOSMessage", pydeepstreamdoc::pipeline::EOSMessageDoc::descr);
    py::class_<Pipeline::StateTransitionMessage, Pipeline::Message>(m, "StateTransitionMessage", pydeepstreamdoc::pipeline::StateTransitionMessageDoc::descr)
        .def_property_readonly("old_state", [](Pipeline::StateTransitionMessage& self){
            Pipeline::State states[2] = {Pipeline::State::INVALID,Pipeline::State::INVALID };
            self.getState(states[0], states[1]);
            return states[0];
        })
        .def_property_readonly("new_state", [](Pipeline::StateTransitionMessage& self){
            Pipeline::State states[2] = {Pipeline::State::INVALID,Pipeline::State::INVALID };
            self.getState(states[0], states[1]);
            return states[1];
        })
        .def_property_readonly("origin", [](Pipeline::StateTransitionMessage& self) {
            string name = self.getName();
            return name;
        })
        .def("__repr__", [](Pipeline::StateTransitionMessage& self) {
            Pipeline::State states[2] = {Pipeline::State::INVALID,Pipeline::State::INVALID };
            self.getState(states[0], states[1]);
            return "StateTransitionMessage(old_state=" + std::to_string(states[0]) + 
            ", new_state=" + std::to_string(states[1]) + 
            ", origin=" + self.getName() + ")";
        });
    py::class_<Pipeline::DynamicSourceMessage, Pipeline::Message>(m, "DynamicSourceMessage", pydeepstreamdoc::pipeline::DynamicSourceMessageDoc::descr)
        .def_property_readonly("source_added", &Pipeline::DynamicSourceMessage::isSourceAdded)
        .def_property_readonly("source_id", &Pipeline::DynamicSourceMessage::getSourceId)
        .def_property_readonly("sensor_id", &Pipeline::DynamicSourceMessage::getSensorId)
        .def_property_readonly("sensor_name", &Pipeline::DynamicSourceMessage::getSensorName)
        .def_property_readonly("uri", &Pipeline::DynamicSourceMessage::getUri)
        .def("__repr__", [](Pipeline::DynamicSourceMessage& self) {
            return "DynamicSourceMessage(source_added=" + std::to_string(self.isSourceAdded()) + 
            ", source_id=" + std::to_string(self.getSourceId()) + 
            ", sensor_id=" + self.getSensorId() + 
            ", sensor_name=" + self.getSensorName() + 
            ", uri=" + self.getUri() + ")";
        });
    py::class_<RecordingInfo>(m, "RecordingInfo", pydeepstreamdoc::pipeline::RecordingInfoDoc::descr)
        .def_property_readonly("session_id", &RecordingInfo::getSessionId)
        .def_property_readonly("file_name", &RecordingInfo::getFileName)
        .def_property_readonly("file_directory", &RecordingInfo::getFileDirectory)
        .def_property_readonly("duration", &RecordingInfo::getDuration)
        .def_property_readonly("container_type", &RecordingInfo::getContainerType)
        .def_property_readonly("width", &RecordingInfo::getWidth)
        .def_property_readonly("height", &RecordingInfo::getHeight)
        .def_property_readonly("contains_video", &RecordingInfo::containsVideo)
        .def_property_readonly("contains_audio", &RecordingInfo::containsAudio)
        .def_property_readonly("channels", &RecordingInfo::getChannels)
        .def_property_readonly("sampling_rate", &RecordingInfo::getSamplingRate)
        .def("__repr__", [](RecordingInfo& self) {
            return "RecordingInfo(session_id=" + std::to_string(self.getSessionId()) + 
            ", file_name=" + self.getFileName() + 
            ", file_directory=" + self.getFileDirectory() + 
            ", duration=" + std::to_string(self.getDuration()) + 
            ", container_type=" + self.getContainerType() + 
            ", width=" + std::to_string(self.getWidth()) + 
            ", height=" + std::to_string(self.getHeight()) + 
            ", contains_video=" + std::to_string(self.containsVideo()) + 
            ", contains_audio=" + std::to_string(self.containsAudio()) + 
            ", channels=" + std::to_string(self.getChannels()) + 
            ", sampling_rate=" + std::to_string(self.getSamplingRate()) + ")";
        });
    py::class_<PipelineWrapper>(m, "Pipeline", pydeepstreamdoc::pipeline::PipelineDoc::descr)
        .def(py::init<const string&>(), py::arg("name"), pydeepstreamdoc::pipeline::PipelineDoc::init)
        .def(py::init<const string&, const string&>(), py::arg("name"), py::arg("config_file"), pydeepstreamdoc::pipeline::PipelineDoc::init_with_config)
        .def("add", [](PipelineWrapper& self, const string& type_name, const string& name) {
            self.add(type_name, name);
        }, pydeepstreamdoc::pipeline::PipelineDoc::add)
        .def("attach", (Pipeline& (Pipeline::*)(const string&, const string&, const string&, const string)) &Pipeline::attach, pydeepstreamdoc::pipeline::PipelineDoc::attach)
        .def("__getitem__", [](PipelineWrapper& self, const string& name) -> Node& {
            /** Must be casted to wrapper class for python to access */
            return reinterpret_cast<Node&>(self[name]);
        }, pydeepstreamdoc::pipeline::PipelineDoc::getitem)
        .def("start", (Pipeline& (Pipeline::*)()) &Pipeline::start, pydeepstreamdoc::pipeline::PipelineDoc::start)
        .def("start_with_callback", [](PipelineWrapper& self, std::function<void(py::object)> on_message){
            self.on_message = on_message;
            self.start(on_pipeline_message);
        }, pydeepstreamdoc::pipeline::PipelineDoc::start_with_callback)
        .def("prepare", (int (Pipeline::*)()) &Pipeline::prepare, pydeepstreamdoc::pipeline::PipelineDoc::prepare)
        .def("prepare_with_callback", [](PipelineWrapper& self, std::function<void(py::object)> on_message){
            self.on_message = on_message;
            self.prepare(on_pipeline_message);
        }, pydeepstreamdoc::pipeline::PipelineDoc::prepare_with_callback)
        .def("activate", (Pipeline& (Pipeline::*)()) &Pipeline::activate, pydeepstreamdoc::pipeline::PipelineDoc::activate)
        .def("wait", [](PipelineWrapper& self) {
            py::gil_scoped_release release;
            self.wait();
        }, pydeepstreamdoc::pipeline::PipelineDoc::wait)
        .def("stop", &Pipeline::stop, pydeepstreamdoc::pipeline::PipelineDoc::stop)
        .def("start_rtsp_server", &Pipeline::startRTSP, pydeepstreamdoc::pipeline::PipelineDoc::start_rtsp_server)
        .def("seek",  [](PipelineWrapper& self, uint64_t timestamp) {
            py::gil_scoped_release release;
            self.seek(timestamp);
        }, pydeepstreamdoc::pipeline::PipelineDoc::seek)
        .def("start_recording", [](PipelineWrapper& self, const string& source_name, uint64_t start_time, uint64_t duration, std::function<void(RecordingInfo)> callback = nullptr) {
            py::gil_scoped_release release;
            if (callback) {
                return self.startRecording(source_name, start_time, duration, callback);
            } else {
                return self.startRecording(source_name, start_time, duration);
            }
        }, pydeepstreamdoc::pipeline::PipelineDoc::start_recording)
        .def("stop_recording", [](PipelineWrapper& self, const string& source_name) {
            py::gil_scoped_release release;
            return self.stopRecording(source_name);
        }, pydeepstreamdoc::pipeline::PipelineDoc::stop_recording)
        .def("stop_recording_by_session_id", [](PipelineWrapper& self, uint32_t session_id) {
            py::gil_scoped_release release;
            return self.stopRecording(session_id);
        }, pydeepstreamdoc::pipeline::PipelineDoc::stop_recording_by_session_id);

}