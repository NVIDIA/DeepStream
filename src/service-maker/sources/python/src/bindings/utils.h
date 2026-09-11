/*
 * SPDX-FileCopyrightText: Copyright (c) 2017-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "utilsdoc.h"

class EngineFileMonitor : public NvDsModelEngineWatchOTFTrigger {
public:
    EngineFileMonitor(Node& e, const string &file) : NvDsModelEngineWatchOTFTrigger(&e, file) {}
};

void module_utils_bind(py::module &m);

void module_utils_bind(py::module &m) {
    py::module utils_module = m.def_submodule("utils");
    py::class_<PerfMonitor>(utils_module, "PerfMonitor", pydeepstreamdoc::utils::PerfMonitorDoc::descr)
        .def(py::init<unsigned int, uint64_t, string&, bool>(), py::arg("batch_size"), py::arg("interval"), py::arg("source_type"), py::arg("show_name"))
        .def("apply", [](PerfMonitor& self, Node& node, string& tips) {
            self.apply(node, tips);
        },
        pydeepstreamdoc::utils::PerfMonitorDoc::apply,
        py::arg("node"), py::arg("tips"))
        .def("pause", &PerfMonitor::pause, pydeepstreamdoc::utils::PerfMonitorDoc::pause)
        .def("resume", &PerfMonitor::resume, pydeepstreamdoc::utils::PerfMonitorDoc::resume)
        .def("add_stream", [](PerfMonitor& self, uint32_t source_id, string& uri, string& sensor_id, string& sensor_name) {
            self.addStream(source_id, uri.c_str(), sensor_id.c_str(), sensor_name.c_str());
        }, pydeepstreamdoc::utils::PerfMonitorDoc::add_stream, py::arg("source_id"), py::arg("uri"), py::arg("sensor_id"), py::arg("sensor_name"))
        .def("remove_stream", &PerfMonitor::removeStream, pydeepstreamdoc::utils::PerfMonitorDoc::remove_stream);
    py::class_<EngineFileMonitor>(utils_module, "EngineFileMonitor", py::dynamic_attr(), pydeepstreamdoc::utils::EngineFileMonitorDoc::descr)
        .def(py::init<Node&, string>(), pydeepstreamdoc::utils::EngineFileMonitorDoc::init)
        .def_readwrite("started", &NvDsModelEngineWatchOTFTrigger::started, pydeepstreamdoc::utils::EngineFileMonitorDoc::started)
        .def("start", [](EngineFileMonitor& self) {
            self.start();
        }, pydeepstreamdoc::utils::EngineFileMonitorDoc::start)
        .def("stop", [](EngineFileMonitor& self) {
            self.stop();
        }, pydeepstreamdoc::utils::EngineFileMonitorDoc::stop)
        .def("__repr__", [](EngineFileMonitor& self) {
            return "EngineFileMonitor(started=" + std::to_string(self.started) + ")";
        });
    py::class_<AudioStreamInfo>(utils_module, "AudioStreamInfo", pydeepstreamdoc::utils::AudioStreamInfoDoc::descr)
        .def_property_readonly("codec", [](const AudioStreamInfo& self) { return self.codec; })
        .def_property_readonly("channels", [](const AudioStreamInfo& self) { return self.channels; })
        .def_property_readonly("sampling_rate", [](const AudioStreamInfo& self) { return self.sampling_rate; })
        .def("__repr__", [](const AudioStreamInfo& self) {
            return "AudioStreamInfo(codec=" + self.codec +
            ", channels=" + std::to_string(self.channels) +
            ", sampling_rate=" + std::to_string(self.sampling_rate) + ")";
        });
    py::class_<VideoStreamInfo>(utils_module, "VideoStreamInfo", pydeepstreamdoc::utils::VideoStreamInfoDoc::descr)
        .def_property_readonly("codec", [](const VideoStreamInfo& self) { return self.codec; })
        .def_property_readonly("width", [](const VideoStreamInfo& self) { return self.width; })
        .def_property_readonly("height", [](const VideoStreamInfo& self) { return self.height; })
        .def_property_readonly("framerate", [](const VideoStreamInfo& self) {
            return pybind11::make_tuple(self.framerate.num, self.framerate.denom);
        })
        .def("__repr__", [](const VideoStreamInfo& self) {
            return "VideoStreamInfo(codec=" + self.codec + 
            ", width=" + std::to_string(self.width) + 
            ", height=" + std::to_string(self.height) + 
            ", framerate=" + std::to_string(self.framerate.num) + "/" + std::to_string(self.framerate.denom) + ")";
        });
    py::class_<MediaInfo>(utils_module, "MediaInfo", pydeepstreamdoc::utils::MediaInfoDoc::descr)
        .def_property_readonly("live", [](const MediaInfo& self) { return self.live; })
        .def_property_readonly("duration", [](const MediaInfo& self) { return self.duration; })
        .def_property_readonly("streams", [](const MediaInfo& self) {
            pybind11::list py_list;
            for (auto& s : self.streams) {
                StreamInfo* p = s.get();
                AudioStreamInfo* audio = dynamic_cast<AudioStreamInfo*>(p);
                VideoStreamInfo* video = dynamic_cast<VideoStreamInfo*>(p);
                if (audio) {
                    py_list.append(*audio);
                } else if (video) {
                    py_list.append(*video);
                }
            }
            return py_list;
        })
        .def("__bool__", [](const MediaInfo& self) { return !self.error; })
        .def("discover", &MediaInfo::discover, pydeepstreamdoc::utils::MediaInfoDoc::discover)
        .def("__repr__", [](const MediaInfo& self) {
            pybind11::list py_list;
            for (auto& s : self.streams) {
                StreamInfo* p = s.get();
                AudioStreamInfo* audio = dynamic_cast<AudioStreamInfo*>(p);
                VideoStreamInfo* video = dynamic_cast<VideoStreamInfo*>(p);
                if (audio) {
                    py_list.append(*audio);
                } else if (video) {
                    py_list.append(*video);
                }
            }
            return "MediaInfo(live=" + std::to_string(self.live) + 
            ", duration=" + std::to_string(self.duration) + 
            ", streams=" + py::repr(py_list).cast<std::string>() + ")";
        });
}
