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

#include "osddoc.h"
#include <cstring>
#include <tiler_event_handler.hpp>

typedef enum {
    Font_Serif = 0,
    Font_Max
} FontFamily;

/** Carrier class to handle string transfer */
struct Text : public NvOSD_TextParams {
    std::string text;
};

/** Tiler event handler wrapper */
class EventHandler: public NvDsTilerEventHandler {
  public:
    EventHandler(Node& tiler, Node& osd, Node& renderer)
    : NvDsTilerEventHandler(&tiler, &osd, &renderer)
    {}
};

static const char* font_table[] = { "Serif" };

void module_osd_bind(py::module &m);

void module_osd_bind(py::module &m) {
    py::module osd_module = m.def_submodule("osd");
    py::enum_<FontFamily>(osd_module, "FontFamily", py::module_local(), pydeepstreamdoc::osd::FontFamilyDoc::descr)
        .value("Serif", FontFamily::Font_Serif, pydeepstreamdoc::osd::FontFamilyDoc::serif);
    py::enum_<NvOSD_Arrow_Head_Direction>(osd_module, "ArrowHead", py::module_local(), pydeepstreamdoc::osd::ArrowHeadDoc::descr)
        .value("Start", NvOSD_Arrow_Head_Direction::START_HEAD, pydeepstreamdoc::osd::ArrowHeadDoc::start)
        .value("End", NvOSD_Arrow_Head_Direction::END_HEAD, pydeepstreamdoc::osd::ArrowHeadDoc::end)
        .value("Both", NvOSD_Arrow_Head_Direction::BOTH_HEAD, pydeepstreamdoc::osd::ArrowHeadDoc::both);
    py::class_<NvOSD_FontParams>(osd_module, "Font", py::module_local(), pydeepstreamdoc::osd::FontDoc::descr)
        .def_property("name",
                     // Getter
                     [](const NvOSD_FontParams& self) {
                        return py::str(self.font_name);
                     },
                     // Setter
                     [](NvOSD_FontParams& self, const FontFamily font) {
                        unsigned int index = static_cast<unsigned int>(font);
                        if (index < Font_Max) {
                            self.font_name = (char *)font_table[index];
                        }
                     }
        )
        .def_readwrite("size", &NvOSD_FontParams::font_size)
        .def_readwrite("color", &NvOSD_FontParams::font_color)
        .def("__repr__", [](const NvOSD_FontParams& self) {
            std::string font_name_str = self.font_name ? std::string(self.font_name) : "null";
            return "Font(name=" + font_name_str + 
                ", size=" + std::to_string(self.font_size) + 
                ", color=" + py::repr(py::cast(self.font_color)).cast<std::string>() + ")";
        });
    py::class_<NvOSD_ColorParams>(osd_module, "Color", py::module_local(), pydeepstreamdoc::osd::ColorDoc::descr)
        .def(py::init<double, double, double, double>())
        .def_readwrite("r", &NvOSD_ColorParams::red)
        .def_readwrite("g", &NvOSD_ColorParams::green)
        .def_readwrite("b", &NvOSD_ColorParams::blue)
        .def_readwrite("a", &NvOSD_ColorParams::alpha)
        .def("__repr__", [](const NvOSD_ColorParams& self) {
            return "Color(r=" + std::to_string(self.red) + 
                ", g=" + std::to_string(self.green) + 
                ", b=" + std::to_string(self.blue) + 
                ", a=" + std::to_string(self.alpha) + ")";
        });
    py::class_<Text>(osd_module, "Text", py::module_local(), pydeepstreamdoc::osd::TextDoc::descr)
        .def(py::init<>())
        .def_property("display_text",
                     // Getter
                     [](const Text& self) {
                        return py::str(self.display_text);
                     },
                     // Setter
                     [](Text& self, const string &name) {
                        self.text = name;
                        self.display_text = (char *) self.text.c_str();
                     }
        )
        .def_readwrite("x_offset", &NvOSD_TextParams::x_offset)
        .def_readwrite("y_offset", &NvOSD_TextParams::y_offset)
        .def_readwrite("font", &NvOSD_TextParams::font_params)
        .def_readwrite("set_bg_color", &NvOSD_TextParams::set_bg_clr)
        .def_readwrite("bg_color", &NvOSD_TextParams::text_bg_clr)
        .def("__repr__", [](const Text& self) {
            std::string display_text_str = self.display_text ? std::string(self.display_text) : "null";
            std::string font_name_str = self.font_params.font_name ? std::string(self.font_params.font_name) : "null";
            return "Text(display_text='" + display_text_str +
                "', x_offset=" + std::to_string(self.x_offset) +
                ", y_offset=" + std::to_string(self.y_offset) +
                ", font_name='" + font_name_str +
                "', font_size=" + std::to_string(self.font_params.font_size) +
                ", bg_color=" + py::repr(py::cast(self.text_bg_clr)).cast<std::string>() + ")";
        });
        py::class_<NvOSD_TextParams>(osd_module, "TextParams", py::module_local(), pydeepstreamdoc::osd::TextParamsDoc::descr)
        .def(py::init<>())
        .def_property("display_text",
                    [](const NvOSD_TextParams& self) {
                        return py::str(self.display_text ? self.display_text : "");
                    },
                    [](NvOSD_TextParams& self, const string &text) {
                        // Free existing memory if allocated
                        if (self.display_text) {
                            delete[] self.display_text;
                        }
                        // Allocate new memory and copy string
                        self.display_text = new char[text.length() + 1];
                        std::strcpy(self.display_text, text.c_str());
                    }
        )
        .def_readwrite("x_offset", &NvOSD_TextParams::x_offset)
        .def_readwrite("y_offset", &NvOSD_TextParams::y_offset)
        .def_readwrite("font_params", &NvOSD_TextParams::font_params)
        .def_readwrite("set_bg_clr", &NvOSD_TextParams::set_bg_clr)
        .def_readwrite("text_bg_clr", &NvOSD_TextParams::text_bg_clr)
        .def("__repr__", [](const NvOSD_TextParams& self) {
            return "TextParams(display_text='" + std::string(self.display_text ? self.display_text : "") +
                "', x_offset=" + std::to_string(self.x_offset) +
                ", y_offset=" + std::to_string(self.y_offset) +
                ", font_params=" + py::repr(py::cast(self.font_params)).cast<std::string>() +
                ", set_bg_clr=" + std::to_string(self.set_bg_clr) +
                ", text_bg_clr=" + py::repr(py::cast(self.text_bg_clr)).cast<std::string>() + ")";
        });
    py::class_<NvOSD_RectParams>(osd_module, "Rect", py::module_local(), pydeepstreamdoc::osd::RectDoc::descr)
        .def(py::init<>())
        .def_readwrite("left", &NvOSD_RectParams::left)
        .def_readwrite("top", &NvOSD_RectParams::top)
        .def_readwrite("width", &NvOSD_RectParams::width)
        .def_readwrite("height", &NvOSD_RectParams::height)
        .def_readwrite("rotation_angle", &NvOSD_RectParams::rotation_angle)
        .def_readwrite("border_width", &NvOSD_RectParams::border_width)
        .def_readwrite("border_color", &NvOSD_RectParams::border_color)
        .def_readwrite("has_bg_color", &NvOSD_RectParams::has_bg_color)
        .def_readwrite("bg_color", &NvOSD_RectParams::bg_color)
        .def("__repr__", [](const NvOSD_RectParams& self) {
            return "Rect(left=" + std::to_string(self.left) + 
                ", top=" + std::to_string(self.top) + 
                ", width=" + std::to_string(self.width) + 
                ", height=" + std::to_string(self.height) + 
                ", rotation_angle=" + std::to_string(self.rotation_angle) +
                ", border_width=" + std::to_string(self.border_width) + 
                ", border_color=" + py::repr(py::cast(self.border_color)).cast<std::string>() + 
                ", has_bg_color=" + std::to_string(self.has_bg_color) + 
                ", bg_color=" + py::repr(py::cast(self.bg_color)).cast<std::string>() + ")";
        });;
    py::class_<NvOSD_MaskParams>(osd_module, "Mask", py::module_local(), pydeepstreamdoc::osd::MaskDoc::descr)
        .def(py::init<>())
        .def_readwrite("threshold", &NvOSD_MaskParams::threshold)
        .def_property("mask_array",
                     // Getter
                     [](NvOSD_MaskParams &self) -> py::array {
                        const float* data = self.data;
                        if (!data) {
                            return py::array_t<float>();
                        }
                        return py::array_t<float>(
                            {self.height, self.width},
                            {self.width * sizeof(float), sizeof(float)},
                            data
                        );
                     },
                     // Setter
                     [](NvOSD_MaskParams &self, const py::array &array) {
                        self.data = (float *)array.request().ptr;
                        self.size = array.size() * sizeof(float);
                        self.height = array.shape(0);
                        self.width = array.shape(1);
                     }
        )
        .def("__repr__", [](const NvOSD_MaskParams& self) {
            const float* data = self.data;
            py::array mask_array = py::array_t<float>();
            if (data) {
                mask_array = py::array_t<float>(
                    {self.height, self.width},
                    {self.width * sizeof(float), sizeof(float)},
                    data
                );
            }
            return "Mask(height=" + std::to_string(self.height) + 
                ", width=" + std::to_string(self.width) + 
                ", threshold=" + std::to_string(self.threshold) + 
                ", mask_array=" + py::repr(mask_array).cast<std::string>() + ")";
        });
    py::class_<NvOSD_LineParams>(osd_module, "Line", py::module_local(), pydeepstreamdoc::osd::LineDoc::descr)
        .def(py::init<>())
        .def_readwrite("x1", &NvOSD_LineParams::x1)
        .def_readwrite("y1", &NvOSD_LineParams::y1)
        .def_readwrite("x2", &NvOSD_LineParams::x2)
        .def_readwrite("y2", &NvOSD_LineParams::y2)
        .def_readwrite("width", &NvOSD_LineParams::line_width)
        .def_readwrite("color", &NvOSD_LineParams::line_color)
        .def("__repr__", [](const NvOSD_LineParams& self) {
            return "Line(x1=" + std::to_string(self.x1) + 
                ", y1=" + std::to_string(self.y1) + 
                ", x2=" + std::to_string(self.x2) + 
                ", y2=" + std::to_string(self.y2) + 
                ", width=" + std::to_string(self.line_width) + 
                ", color=" + py::repr(py::cast(self.line_color)).cast<std::string>() + ")";
        });
    py::class_<NvOSD_ArrowParams>(osd_module, "Arrow", py::module_local(), pydeepstreamdoc::osd::ArrowDoc::descr)
        .def(py::init<>())
        .def_readwrite("x1", &NvOSD_ArrowParams::x1)
        .def_readwrite("y1", &NvOSD_ArrowParams::y1)
        .def_readwrite("x2", &NvOSD_ArrowParams::x2)
        .def_readwrite("y2", &NvOSD_ArrowParams::y2)
        .def_readwrite("width", &NvOSD_ArrowParams::arrow_width)
        .def_readwrite("color", &NvOSD_ArrowParams::arrow_color)
        .def_readwrite("head", &NvOSD_ArrowParams::arrow_head)
        .def("__repr__", [](const NvOSD_ArrowParams& self) {
            return "Arrow(x1=" + std::to_string(self.x1) + 
                ", y1=" + std::to_string(self.y1) + 
                ", x2=" + std::to_string(self.x2) + 
                ", y2=" + std::to_string(self.y2) + 
                ", width=" + std::to_string(self.arrow_width) + 
                ", color=" + py::repr(py::cast(self.arrow_color)).cast<std::string>() + 
                ", head=" + std::to_string(self.arrow_head) + ")";
        });
    py::class_<NvOSD_CircleParams>(osd_module, "Circle", py::module_local(), pydeepstreamdoc::osd::CircleDoc::descr)
        .def(py::init<>())
        .def_readwrite("xc", &NvOSD_CircleParams::xc)
        .def_readwrite("yc", &NvOSD_CircleParams::yc)
        .def_readwrite("radius", &NvOSD_CircleParams::radius)
        .def_readwrite("color", &NvOSD_CircleParams::circle_color)
        .def_readwrite("has_bg_color", &NvOSD_CircleParams::has_bg_color)
        .def_readwrite("bg_color", &NvOSD_CircleParams::bg_color)
        .def_readwrite("width", &NvOSD_CircleParams::circle_width)
        .def("__repr__", [](const NvOSD_CircleParams& self) {
            return "Circle(xc=" + std::to_string(self.xc) + 
                ", yc=" + std::to_string(self.yc) + 
                ", radius=" + std::to_string(self.radius) + 
                ", color=" + py::repr(py::cast(self.circle_color)).cast<std::string>() + 
                ", has_bg_color=" + std::to_string(self.has_bg_color) + 
                ", bg_color=" + py::repr(py::cast(self.bg_color)).cast<std::string>() + 
                ", width=" + std::to_string(self.circle_width) + ")";
        });
    py::class_<EventHandler>(osd_module, "EventHandler", py::module_local(), py::dynamic_attr(), pydeepstreamdoc::osd::EventHandlerDoc::descr)
        .def(py::init<Node &, Node &, Node &>(), py::arg("tiler"), py::arg("osd"), py::arg("renderer"), pydeepstreamdoc::osd::EventHandlerDoc::init)
        .def_readonly("started", &NvDsTilerEventHandler::started)
        .def("start", &NvDsTilerEventHandler::start, pydeepstreamdoc::osd::EventHandlerDoc::start)
        .def("stop", &NvDsTilerEventHandler::stop, pydeepstreamdoc::osd::EventHandlerDoc::stop)
        .def("__repr__", [](const EventHandler& self) {
            return "EventHandler(started=" + std::to_string(self.started) + ")";
        });
}