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

namespace pybind11 {
namespace detail {

template<> struct type_caster<Object::Value> : type_caster_base<Object::Value> {
public:
    type_caster() = default;
    type_caster(type_caster&&) noexcept = default;
    type_caster& operator=(type_caster&&) noexcept = default;

    /**
     * Associate _deepstream.Value with C++ class deepstrea::Object::Value, though the latter
     * is not supposed to be exposed to the python application
    */
    PYBIND11_TYPE_CASTER(Object::Value, const_name("_deepstream.Value"));

    bool load(handle src, bool) {
        /* Walk through all the supported types */
        PyTypeObject *srctype = Py_TYPE(src.ptr());
        if (strcmp(srctype->tp_name, "str") == 0) {
            auto tmp = reinterpret_borrow<py::str>(src).cast<string>();
            value = Object::Value(tmp);
        } else if (strcmp(srctype->tp_name, "int") == 0) {
            auto tmp = reinterpret_borrow<py::int_>(src).cast<int>();
            value = Object::Value(tmp);
        } else if (strcmp(srctype->tp_name, "bool") == 0) {
            auto tmp = reinterpret_borrow<py::bool_>(src).cast<bool>();
            value = Object::Value(tmp);
        } else if (strcmp(srctype->tp_name, "float") == 0) {
            auto tmp = reinterpret_borrow<py::float_>(src).cast<float>();
            value = Object::Value(tmp);
        } else {
            printf("Type not supported for value: %s\n", srctype->tp_name);
            return false;
        }

        return true;
    }

    // Conversion from C++ type to Python object
    static py::handle cast(const Object::Value& src, py::return_value_policy /* policy */, py::handle /* parent */) {
        if (src.isInteger()) {
            auto value = (int)src;
            return PyLong_FromLong((long)value);
        } else if (src.isUnsignedInteger()) {
            auto value = (unsigned int)src;
            return PyLong_FromLong((long)value);
        } else if (src.isString()) {
            auto value = (string)src;
            return PyUnicode_FromString(value.c_str());
        } else if (src.isFloat()) {
            auto value = (float)src;
            return PyFloat_FromDouble((double)value);
        } else if (src.isDouble()) {
            auto value = (double)src;
            return PyFloat_FromDouble(value);
        } else if (src.isBoolean()) {
            auto value = (bool)src;
            PyObject* object = value ? Py_True:Py_False;
            return object;
        } else {
            printf("Type from Object::Value is not supported\n");
        }

        return py::none();;
    }
};

}
}