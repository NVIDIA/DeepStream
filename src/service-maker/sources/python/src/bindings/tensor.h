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

#include "tensordoc.h"

#include <dlpack/dlpack.h>
#include <iostream>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <memory>

namespace py = pybind11;

class __attribute__ ((visibility("hidden"))) DlpackTensorContext : public Tensor::Context {
 public:
  DlpackTensorContext(py::object dlm_tensor): dlm_tensor_(dlm_tensor) {}
  virtual ~DlpackTensorContext() {}
 protected:
  py::object dlm_tensor_;
};

enum class ColorFormat {
    RGB = NVBUF_COLOR_FORMAT_RGB,
    RGBA = NVBUF_COLOR_FORMAT_RGBA,
    I420 = NVBUF_COLOR_FORMAT_YUV420
};

class TensorWrapper {
public:
    TensorWrapper() {}

    explicit TensorWrapper(Tensor* tensor)
    : tensor_(tensor)
    {}

    explicit TensorWrapper(py::object object, std::string format) {
        py::tuple dlpack_device = object.attr("__dlpack_device__")().cast<py::tuple>();
        auto      device_type      = static_cast<DLDeviceType>(dlpack_device[0].cast<int>());
        py::capsule cap;
        if (device_type == kDLCUDAHost || device_type == kDLCUDA || device_type == kDLCUDAManaged) {
            auto stream_id = 1;
            cap = object.attr("__dlpack__")(py::arg("stream") = stream_id).cast<py::capsule>();
        } else {
            cap = object.attr("__dlpack__")().cast<py::capsule>();
        }
        if (!PyCapsule_IsValid(cap.ptr(), "dltensor")) {
            throw std::runtime_error("Invalid capsule for dltensor");
        }
        DLManagedTensor *dl_managed_tensor = cap.get_pointer<DLManagedTensor>();
        const DLTensor& dl_tensor = dl_managed_tensor->dl_tensor;
        unsigned int device_id = (unsigned int )dl_tensor.device.device_id;
        auto dtype_code = dl_tensor.dtype.code;
        auto device = Tensor::DeviceType::NONE;
        if (device_type == kDLCPU) {
            device = Tensor::DeviceType::CPU;
        } else if (device_type == kDLCUDAHost || device_type == kDLCUDA || device_type == kDLCUDAManaged) {
            device = Tensor::DeviceType::GPU;
        } else {
            throw std::runtime_error("Unsupported Tensor Device Type");
        }

        Tensor::DataType dtype = Tensor::INVALID;
        if (dtype_code == kDLInt) {
            dtype = Tensor::SIGNED;
        } else if (dtype_code == kDLUInt || dtype_code == kDLBool) {
            dtype = Tensor::UNSIGNED;
        } else if (dtype_code == kDLFloat) {
            dtype = Tensor::FLOAT;
        } else if (dtype_code == kDLComplex) {
            dtype = Tensor::COMPLEX;
        } else {
            std::string dtype_str = std::to_string(dtype_code);
            throw std::runtime_error("Unsupported Tensor dtype: " + dtype_str);
        }

        tensor_ = make_unique<Tensor>(
            dl_tensor.ndim,
            dtype,
            dl_tensor.dtype.bits,
            dl_tensor.shape,
            dl_tensor.strides,
            dl_tensor.data,
            format,
            device_id,
            device,
            new DlpackTensorContext(object)
        );
    }

    std::vector<uint64_t> shape_vec() const {
        return tensor_->shape();
    }

    std::vector<int64_t> strides_vec() const {
        std::vector<int64_t> strides;
        for (size_t i = 0; i < tensor_->rank(); i++) {
            strides.push_back(tensor_->stride(i));
        }
        return strides;
    }

    py::tuple shape() const {
        auto ndim = tensor_->rank();
        auto tensor_shape = tensor_->shape();
        py::tuple shape(ndim);
        for (size_t i = 0; i < ndim; i++) {
            shape[i] = tensor_shape[i];
        }
        return shape;
    }

    py::tuple strides() const {
        auto ndim = tensor_->rank();
        py::tuple strides(ndim);
        for (size_t d = 0; d < ndim; d++) {
            strides[d] = tensor_->stride(d);
        }
        return strides;
    }

    Tensor::DataType c_dtype() const {
        return tensor_->dtype();
    }

    py::dtype dtype() const {
        auto dtype = tensor_->dtype();
        std::string dtype_str;
        if (dtype == Tensor::UNSIGNED) {
            dtype_str = "uint";
        } else if (dtype == Tensor::SIGNED) {
            dtype_str = "int";
        } else if (dtype == Tensor::FLOAT) {
            dtype_str = "float";
        } else if (dtype == Tensor::COMPLEX) {
            dtype_str = "complex";
        }
        if (dtype_str.length()) {
            dtype_str += std::to_string(tensor_->bits());
            return py::dtype(dtype_str);
        }

        return py::dtype();
    }

    py::capsule dlpack(PyObject* py_object_ptr, py::object stream) {
        unique_ptr<DLManagedTensor> dl_managed_tensor(new DLManagedTensor);
        dl_managed_tensor->manager_ctx = (void*) py_object_ptr;
        dl_managed_tensor->deleter = [](DLManagedTensor *tensor) {
            py::gil_scoped_acquire acquire;
            delete tensor->dl_tensor.shape;
            delete tensor->dl_tensor.strides;
            auto ctx = static_cast<PyObject*>(tensor->manager_ctx);
            assert(ctx != NULL);
            Py_DECREF(ctx);
            tensor->manager_ctx = NULL;
            delete tensor;
        };

        DLTensor &dl_tensor = dl_managed_tensor->dl_tensor;
        dl_tensor.device.device_type = tensor_->deviceType() == Tensor::DeviceType::GPU ? kDLCUDA : kDLCPU;
        dl_tensor.device.device_id = tensor_->deviceId();
        dl_tensor.ndim = tensor_->rank();

        unique_ptr<int64_t[]> shape_ptr(new int64_t[dl_tensor.ndim]);
        copy_n(tensor_->shape().begin(), dl_tensor.ndim, shape_ptr.get());
        dl_tensor.shape = shape_ptr.release();

        unique_ptr<int64_t[]> strides_ptr(new int64_t[dl_tensor.ndim]);
        for (int n = 0; n < dl_tensor.ndim; n++) {
            strides_ptr[n] = tensor_->stride(n);
        }
        dl_tensor.strides = strides_ptr.release();

        auto dtype = tensor_->dtype();
        uint8_t dtype_code = 0;
        if (dtype == Tensor::UNSIGNED) {
            dtype_code = kDLUInt;
        } else if (dtype == Tensor::SIGNED) {
            dtype_code = kDLInt;
        } else if (dtype == Tensor::FLOAT) {
            dtype_code = kDLFloat;
        } else if (dtype == Tensor::COMPLEX) {
            dtype_code = kDLComplex;
        }
        dl_tensor.dtype = {dtype_code, (uint8_t)tensor_->bits(), 1};

        dl_tensor.data = tensor_->data();
        dl_tensor.byte_offset = 0;

        py::capsule cap(dl_managed_tensor.release(), "dltensor", [](PyObject *ptr) {
            if (PyCapsule_IsValid(ptr, "dltensor")) {
                if(auto *tensor = static_cast<DLManagedTensor *>(PyCapsule_GetPointer(ptr, "dltensor"))) {
                    if (tensor->deleter) {
                        tensor->deleter(tensor);
                    }
                }
            }
        });

        return cap;
    }

    std::pair<int, int> dlpackDevice() const {
        return {kDLCUDA, tensor_->deviceId()};
    }

    Buffer wrap(ColorFormat format) {
        if (tensor_) {
            return tensor_->wrap(static_cast<NvBufSurfaceColorFormat>(format));
        } else {
            return Buffer();
        }
    }

    TensorWrapper* clone() const {
        if (tensor_) {
            return new TensorWrapper(tensor_->clone());
        } else {
            return nullptr;
        }
    }

    Tensor* release() {
        return tensor_.release();
    }

    uint64_t size() const {
        if (tensor_) {
            return tensor_->size();
        } else {
            return 0;
        }
    }

    TensorWrapper* toGPU(unsigned int device_id) const {
        if (tensor_) {
            return new TensorWrapper(tensor_->toGPU(device_id));
        } else {
            return nullptr;
        }
    }

    std::string deviceType() const {
        if (tensor_) {
            auto dtype = tensor_->deviceType();
            if (dtype == Tensor::DeviceType::GPU) return "GPU";
            if (dtype == Tensor::DeviceType::CPU) return "CPU";
        }
        return "NONE";
    }

    void* data() const {
        return tensor_->data();
    }

    bool hasTensor() const { return tensor_ != nullptr; }

protected:
    unique_ptr<Tensor> tensor_;
};

template<typename T>
void print_nd_summary(const void* base, const std::vector<uint64_t>& shape, const std::vector<int64_t>& strides, size_t dim = 0, ssize_t offset = 0) {
    if (dim == shape.size()) {
        const T* ptr = reinterpret_cast<const T*>(reinterpret_cast<const char*>(base) + offset);
        std::cout << *ptr;
        return;
    }
    std::cout << "[";
    ssize_t n = shape[dim];
    ssize_t print_head = std::min<ssize_t>(3, n);
    ssize_t print_tail = (n > 6) ? 3 : std::max<ssize_t>(0, n - 3);

    // Print first 3
    for (ssize_t i = 0; i < print_head; ++i) {
        if (i > 0) std::cout << ", ";
        print_nd_summary<T>(base, shape, strides, dim + 1, offset + i * strides[dim]);
    }
    // Ellipsis if needed
    if (n > 6) std::cout << ", ...";
    // Print last 3
    for (ssize_t i = n - print_tail; i < n; ++i) {
        if (i > 0) std::cout << ", ";
        print_nd_summary<T>(base, shape, strides, dim + 1, offset + i * strides[dim]);
    }
    std::cout << "]";
}

void module_tensor_bind(py::module &m);

void module_tensor_bind(py::module &m) {
    py::enum_<ColorFormat>(m, "ColorFormat", pydeepstreamdoc::tensor::ColorFormatDoc::descr)
        .value("RGB", ColorFormat::RGB)
        .value("I420", ColorFormat::I420)
        .value("RGBA", ColorFormat::RGBA);
    py::class_<TensorWrapper>(m, "Tensor", pydeepstreamdoc::tensor::TensorDoc::descr)
      .def_property_readonly("shape", &TensorWrapper::shape)
      .def_property_readonly("strides", &TensorWrapper::strides)
      .def_property_readonly("dtype", &TensorWrapper::dtype)
      .def_property_readonly("device_type", &TensorWrapper::deviceType)
      .def("__dlpack__", [](py::object& self, py::object stream){
        TensorWrapper* tensor = self.cast<TensorWrapper*>();
        Py_INCREF(self.ptr());
        return tensor->dlpack(self.ptr(), stream);
      }, pydeepstreamdoc::tensor::TensorDoc::dlpack, py::arg("stream") = py::none())
      .def("__dlpack_device__", &TensorWrapper::dlpackDevice, pydeepstreamdoc::tensor::TensorDoc::dlpack_device)
      .def("wrap", &TensorWrapper::wrap, pydeepstreamdoc::tensor::TensorDoc::wrap)
      .def("clone", &TensorWrapper::clone, py::return_value_policy::take_ownership, pydeepstreamdoc::tensor::TensorDoc::clone)
      .def("to_gpu", &TensorWrapper::toGPU, py::arg("device_id"), py::return_value_policy::take_ownership, pydeepstreamdoc::tensor::TensorDoc::to_gpu)
      .def("size", &TensorWrapper::size, pydeepstreamdoc::tensor::TensorDoc::size)
      .def("__bool__", [](TensorWrapper& self) { return self.hasTensor(); })
      .def("__repr__", [](const TensorWrapper& self) {
        if (self.hasTensor()) {
            if (self.deviceType() == "CPU") {
                return "Tensor(shape=" + py::repr(self.shape()).cast<std::string>() + 
                ", strides=" + py::repr(self.strides()).cast<std::string>() + 
                ", dtype=" + py::repr(self.dtype()).cast<std::string>() + 
                ", __dlpack_device__=" + py::repr(py::cast(self.dlpackDevice())).cast<std::string>() +
                ", device_type=" + self.deviceType() +
                ", size=" + std::to_string(self.size()) + 
                ", data=" + py::repr(py::array(self.dtype(), self.shape_vec(), self.strides_vec(), self.data())).cast<std::string>() +
                ")";
            }
            return "Tensor(shape=" + py::repr(self.shape()).cast<std::string>() + 
                ", strides=" + py::repr(self.strides()).cast<std::string>() + 
                ", dtype=" + py::repr(self.dtype()).cast<std::string>() + 
                ", __dlpack_device__=" + py::repr(py::cast(self.dlpackDevice())).cast<std::string>() +
                ", device_type=" + self.deviceType() +
                ", size=" + std::to_string(self.size()) + 
                ", data=" + py::repr(py::cast(self.data())).cast<std::string>() +
                ")";
        }
        return "Tensor(shape=" + py::repr(self.shape()).cast<std::string>() + 
                ", strides=" + py::repr(self.strides()).cast<std::string>() + 
                ", dtype=" + py::repr(self.dtype()).cast<std::string>() + 
                ", __dlpack_device__=" + py::repr(py::cast(self.dlpackDevice())).cast<std::string>() +
                ", size=" + std::to_string(self.size()) + 
                ", data=[])";
    });
    m.def("as_tensor", [](py::object object, std::string format) {
        if (hasattr(object, "__dlpack_device__")) {
            return make_unique<TensorWrapper>(object, format);
        }
        return make_unique<TensorWrapper>();
    }, pydeepstreamdoc::tensor::TensorDoc::as_tensor)
;
}