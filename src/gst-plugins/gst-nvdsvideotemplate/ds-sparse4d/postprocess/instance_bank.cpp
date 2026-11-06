/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "instance_bank.h"
#include <iostream>
#include <vector>

/**
 * Helper function to perform top-k selection from a confidence tensor with additional inputs
 *
 * @param confidence Input confidence tensor [batch_size, N, ...]
 * @param k Number of top elements to select
 * @param inputs Vector of tensors to select from based on confidence indices
 * @return Tuple containing selected confidence tensor and vector of selected input tensors
 */
static std::tuple<torch::Tensor, std::vector<torch::Tensor>> topk(
    const torch::Tensor& confidence,
    int k,
    const std::vector<torch::Tensor>& inputs) {

    // Get batch size and number of elements
    auto bs = confidence.size(0);
    auto N = confidence.size(1);

    // Safety check: ensure k doesn't exceed available elements
    k = std::min(k, static_cast<int>(N));

    // Handle edge case where k is zero or less
    if (k <= 0) {
        // Return empty tensors with appropriate dimensions
        auto empty_conf = torch::empty({bs, 0}, confidence.options());
        std::vector<torch::Tensor> empty_outputs;
        for (const auto& input : inputs) {
            auto last_dims = input.sizes().slice(2);
            std::vector<int64_t> output_shape = {bs, 0};
            for (auto dim : last_dims) {
                output_shape.push_back(dim);
            }
            empty_outputs.push_back(torch::empty(output_shape, input.options()));
        }
        return std::make_tuple(empty_conf, empty_outputs);
    }

    // Get top-k confidence values and indices
    auto result = torch::topk(confidence, k, 1);
    auto conf_topk = std::get<0>(result);
    auto indices = std::get<1>(result);

    // Create flattened indices with batch offset
    auto batch_offset = torch::arange(bs, torch::TensorOptions().device(indices.device()));
    batch_offset = batch_offset.unsqueeze(1) * N;
    auto flat_indices = (indices + batch_offset).reshape(-1);

    // Collect outputs by gathering from each input tensor
    std::vector<torch::Tensor> outputs;
    for (const auto& input : inputs) {
        // Skip if input has invalid dimensions
        if (input.dim() < 2 || input.size(0) != bs || input.size(1) != N) {
            // Create an empty tensor with appropriate dimensions as a placeholder
            auto last_dims = input.dim() > 2 ? input.sizes().slice(2) : std::vector<int64_t>{1};
            std::vector<int64_t> output_shape = {bs, k};
            for (auto dim : last_dims) {
                output_shape.push_back(dim);
            }
            outputs.push_back(torch::zeros(output_shape, input.options()));
            continue;
        }

        auto flattened = input.flatten(0, 1);
        auto selected = flattened.index({flat_indices});
        outputs.push_back(selected.reshape({bs, k, -1}));
    }

    return std::make_tuple(conf_topk, outputs);
}

InstanceBank::InstanceBank(
    int num_anchor,
    int embed_dims,
    const torch::Tensor& anchor,
    AnchorHandler* anchor_handler,
    int num_temp_instances,
    float default_time_interval,
    float confidence_decay,
    bool anchor_grad,
    bool feat_grad,
    float max_time_interval,
    int reid_dims,
    int gpu_id)
    : m_embed_dims(embed_dims),
      m_reid_dims(reid_dims),
      m_num_temp_instances(num_temp_instances),
      m_default_time_interval(default_time_interval),
      m_confidence_decay(confidence_decay),
      m_max_time_interval(max_time_interval),
      m_anchor_handler(anchor_handler),
      m_prev_id(0),
      m_device(gpu_id >= 0 ? torch::Device(torch::kCUDA, gpu_id) : torch::Device(torch::kCPU)) {


    // Process anchor data and move to correct device
    auto anchor_tensor = anchor.to(m_device);
    m_num_anchor = std::min(static_cast<int>(anchor_tensor.size(0)), num_anchor);
    anchor_tensor = anchor_tensor.slice(0, 0, m_num_anchor);

    // Create anchor parameter on correct device
    m_anchor = register_parameter(
        "anchor",
        anchor_tensor.clone(),
        anchor_grad);

    // Store initial anchor for reset
    m_anchor_init = anchor_tensor.clone();

    // Create instance feature parameter on correct device
    auto instance_feature_tensor = torch::zeros({m_anchor.size(0), embed_dims},
                                              torch::TensorOptions().device(m_device));
    m_instance_feature = register_parameter(
        "instance_feature",
        instance_feature_tensor,
        feat_grad);

    // Create reid feature parameter if needed
    if (reid_dims > 0) {
        auto reid_feature_tensor = torch::zeros({m_anchor.size(0), reid_dims},
                                              torch::TensorOptions().device(m_device));
        m_reid_feature = register_parameter(
            "reid_feature",
            reid_feature_tensor,
            feat_grad);
    }

    // Initialize state
    reset();
}

void InstanceBank::init_weight() {
    // Reset anchor to initial value
    m_anchor.data() = m_anchor_init.clone();

    // Initialize instance feature with Xavier/Glorot uniform if requires grad
    if (m_instance_feature.requires_grad()) {
        torch::nn::init::xavier_uniform_(m_instance_feature);
    }

    // Initialize reid feature with Xavier/Glorot uniform if exists and requires grad
    if (m_reid_feature.defined() && m_reid_feature.requires_grad()) {
        torch::nn::init::xavier_uniform_(m_reid_feature);
    }
}

void InstanceBank::reset() {
    m_cached_feature = torch::Tensor();
    m_cached_anchor = torch::Tensor();
    m_metas.clear();
    m_mask = torch::Tensor();
    m_confidence = torch::Tensor();
    m_temp_confidence = torch::Tensor();
    m_instance_id = torch::Tensor();
    m_prev_id = 0;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
InstanceBank::get(int batch_size, const MetaData& metas, const DNMetaData* dn_metas) {
    // Tile instance feature for batch
    auto batch_instance_feature = m_instance_feature.unsqueeze(0).repeat({batch_size, 1, 1});
    // Tile anchor for batch
    auto batch_anchor = m_anchor.unsqueeze(0).repeat({batch_size, 1, 1});

    // Tile reid feature for batch if defined
    torch::Tensor batch_reid_feature;
    if (m_reid_feature.defined()) {
        batch_reid_feature = m_reid_feature.unsqueeze(0).repeat({batch_size, 1, 1});
    }

    // Default time interval tensor - ensure it's on the correct device
    torch::Tensor time_interval = torch::full({batch_size}, m_default_time_interval, 
                              torch::TensorOptions().device(m_device).dtype(batch_instance_feature.dtype()));

    // Handle cached anchor if available
    if (m_cached_anchor.defined() && batch_size == m_cached_anchor.size(0)) {
        // Calculate time interval - properly convert IValue to Tensor
        auto history_time = m_metas.at("timestamp").toTensor().to(m_device);
        auto current_time = metas.at("timestamp").toTensor().to(m_device);
        time_interval = current_time.to(batch_instance_feature.dtype()) - history_time;

        // Update mask based on time interval - store in class member
        m_mask = torch::abs(time_interval) <= m_max_time_interval;
        m_mask = m_mask.to(m_device);

        // Project anchor if anchor handler is available
        if (m_anchor_handler != nullptr) {
            bool has_T_global = false;

            // Check if img_metas has T_global info
            if (metas.count("img_metas") > 0) {
                const auto& img_metas = metas.at("img_metas");
                if (img_metas.isList() && img_metas.toListRef().size() > 0) {
                    const auto& first_meta = img_metas.toList().get(0);
                    if (first_meta.isGenericDict() && first_meta.toGenericDict().contains("T_global")) {
                        has_T_global = true;
                    }
                }
            }

            if (has_T_global) {
                // Create transformation matrices
                std::vector<torch::Tensor> T_matrices;
                const auto& current_img_metas = metas.at("img_metas").toList();
                const auto& prev_img_metas = m_metas.at("img_metas").toList();

                for (int i = 0; i < batch_size; i++) {
                    auto T_global_inv = prev_img_metas.get(i).toGenericDict().at("T_global_inv").toTensor().to(m_device);
                    auto T_global = current_img_metas.get(i).toGenericDict().at("T_global").toTensor().to(m_device);
                    T_matrices.push_back(torch::matmul(T_global_inv, T_global));
                }
                auto T_temp2cur = torch::stack(T_matrices);

                // Project cached anchors
                auto neg_time_interval = -time_interval;
                std::vector<torch::Tensor> neg_time_intervals = {neg_time_interval};
                std::vector<torch::Tensor> transforms = {T_temp2cur};

                // Use anchor_handler safely (assuming anchor_projection returns vector of tensors)
                auto projection_results = m_anchor_handler->anchor_projection(
                    m_cached_anchor,
                    transforms,
                    neg_time_intervals);

                if (!projection_results.empty()) {
                    m_cached_anchor = projection_results[0];
                }
            }

            // Handle dn_metas if provided
            if (dn_metas != nullptr && batch_size == dn_metas->at("dn_anchor").size(0)) {
                auto dn_anchor_tensor = dn_metas->at("dn_anchor").to(m_device);
                int num_dn_group = dn_anchor_tensor.size(1);
                int num_dn = dn_anchor_tensor.size(2);

                // Project dn anchors
                auto flattened_dn_anchor = dn_anchor_tensor.reshape({batch_size, num_dn_group * num_dn, -1});
                auto neg_time_interval = -time_interval;
                std::vector<torch::Tensor> neg_time_intervals = {neg_time_interval};
                std::vector<torch::Tensor> transforms = {m_cached_anchor.new_zeros({1})};  // Placeholder if T_temp2cur not defined

                // Use anchor_handler safely
                auto projection_results = m_anchor_handler->anchor_projection(
                    flattened_dn_anchor,
                    transforms,
                    neg_time_intervals);

                if (!projection_results.empty()) {
                    auto projected_dn_anchor = projection_results[0];
                    const_cast<DNMetaData*>(dn_metas)->at("dn_anchor") = 
                        projected_dn_anchor.reshape({batch_size, num_dn_group, num_dn, -1});
                }
            }

            // Handle time interval
            time_interval = torch::where(
                torch::logical_and(time_interval != 0, m_mask),
                time_interval,
                torch::full_like(time_interval, m_default_time_interval));
        }
    } else {
        // Reset if no cached anchor or batch size mismatch
        reset();
    }

    return std::make_tuple(
        batch_instance_feature,
        batch_anchor,
        m_cached_feature,
        m_cached_anchor,
        time_interval,
        batch_reid_feature);
}

std::tuple<torch::Tensor, torch::Tensor>
InstanceBank::update(torch::Tensor instance_feature, torch::Tensor anchor, torch::Tensor confidence) {
    // Return if no cached feature
    if (!m_cached_feature.defined()) {
        return std::make_tuple(instance_feature, anchor);
    }
    // Handle denoising (dn) instances if present
    int num_dn = 0;
    torch::Tensor dn_instance_feature, dn_anchor;

    if (instance_feature.size(1) > m_num_anchor) {
        num_dn = instance_feature.size(1) - m_num_anchor;
        dn_instance_feature = instance_feature.slice(1, -num_dn);
        dn_anchor = anchor.slice(1, -num_dn);
        instance_feature = instance_feature.slice(1, 0, m_num_anchor);
        anchor = anchor.slice(1, 0, m_num_anchor);
        confidence = confidence.slice(1, 0, m_num_anchor);
    }

    // Get number of instances to select
    int N = m_num_anchor - m_num_temp_instances;
    // Get max confidence across classes
    confidence = std::get<0>(confidence.max(2));

    // Select top-k instances
    auto result = topk(confidence, N, {instance_feature, anchor});
    auto selected_feature = std::get<1>(result)[0];
    auto selected_anchor = std::get<1>(result)[1];

    // Concatenate with cached features
    selected_feature = torch::cat({m_cached_feature, selected_feature}, 1);
    selected_anchor = torch::cat({m_cached_anchor, selected_anchor}, 1);

    // Update based on mask
    instance_feature = torch::where(
        m_mask.unsqueeze(0).unsqueeze(1).unsqueeze(2),
        selected_feature,
        instance_feature);

    anchor = torch::where(
        m_mask.unsqueeze(0).unsqueeze(1).unsqueeze(2),
        selected_anchor,
        anchor);

    // Update instance IDs if available
    if (m_instance_id.defined()) {
        m_instance_id = torch::where(
            m_mask.unsqueeze(0).unsqueeze(1),
            m_instance_id,
            torch::full_like(m_instance_id, -1));
    }

    // Add back denoising instances if any
    if (num_dn > 0) {
        instance_feature = torch::cat({instance_feature, dn_instance_feature}, 1);
        anchor = torch::cat({anchor, dn_anchor}, 1);
    }

    return std::make_tuple(instance_feature, anchor);
}

void InstanceBank::cache(
    torch::Tensor instance_feature,
    torch::Tensor anchor,
    torch::Tensor confidence,
    const MetaData* metas,
    const std::vector<torch::Tensor>* feature_maps) {

    // Return if no temporary instances needed
    if (m_num_temp_instances <= 0) {
        return;
    }

    // Detach tensors to avoid gradient propagation
    instance_feature = instance_feature.detach();
    anchor = anchor.detach();
    confidence = confidence.detach();

    // Store metadata if provided
    if (metas != nullptr) {
        m_metas = *metas;
    }

    // Calculate confidence (max across classes, then sigmoid)
    confidence = std::get<0>(confidence.max(2)).sigmoid();

    // Apply confidence decay if previous confidence exists
    if (m_confidence.defined()) {
        auto decayed_confidence = m_confidence * m_confidence_decay;
        confidence.slice(1, 0, m_num_temp_instances) = torch::maximum(
            decayed_confidence,
            confidence.slice(1, 0, m_num_temp_instances));
    }

    // Store temporary confidence
    m_temp_confidence = confidence;

    // Select top-k instances
    auto result = topk(confidence, m_num_temp_instances, {instance_feature, anchor});
    m_confidence = std::get<0>(result);
    m_cached_feature = std::get<1>(result)[0];
    m_cached_anchor = std::get<1>(result)[1];
}

torch::Tensor InstanceBank::get_instance_id(
    torch::Tensor confidence,
    torch::Tensor anchor,
    float threshold) {

    // Calculate confidence scores (max across classes, then sigmoid)
    confidence = std::get<0>(confidence.max(2)).sigmoid();

    // Initialize instance IDs as all -1
    auto instance_id_tensor = torch::full_like(confidence, -1, torch::kLong);

    // Copy existing instance IDs if available
    if (m_instance_id.defined() && m_instance_id.size(0) == instance_id_tensor.size(0)) {
        instance_id_tensor.slice(1, 0, m_instance_id.size(1)) = m_instance_id;
    }

    // Create mask for new instances
    auto mask = instance_id_tensor < 0;
    if (threshold > 0) {
        mask = mask & (confidence >= threshold);
    }

    // Get indices where mask is true
    // In C++, torch::where returns a tuple of tensors for each dimension
    auto indices = torch::where(mask);

    // Only move to device if not already there
    auto row_indices = indices[0];
    auto col_indices = indices[1];

    if (row_indices.device() != m_device) {
        row_indices = row_indices.to(m_device);
        col_indices = col_indices.to(m_device);
    }

    // Get number of new instances from the length of indices instead of sum().item()
    auto num_new_instances_tensor = row_indices.size(0);

    if (num_new_instances_tensor > 0) {
        // Create new IDs directly on device without CPU synchronization
        auto new_ids = torch::arange(num_new_instances_tensor, 
                                    torch::TensorOptions().dtype(torch::kLong).device(m_device))
                      + m_prev_id;

        // Assign new IDs
        instance_id_tensor.index_put_({row_indices, col_indices}, new_ids);

        // Update prev_id for next call
        m_prev_id += num_new_instances_tensor;
    }

    // Update cached instance IDs for temporary instances
    if (m_num_temp_instances > 0) {
        update_instance_id(instance_id_tensor, confidence);
    }

    return instance_id_tensor;
}

void InstanceBank::update_instance_id(
    torch::Tensor instance_id_tensor,
    torch::Tensor confidence) {

    // Use either provided confidence or temporary confidence
    torch::Tensor temp_conf;
    if (!m_temp_confidence.defined()) {
        if (confidence.dim() == 3) {  // bs, num_anchor, num_cls
            temp_conf = std::get<0>(confidence.max(2));
        } else {  // bs, num_anchor
            temp_conf = confidence;
        }
    } else {
        temp_conf = m_temp_confidence;
    }

    // Safety check: ensure we don't try to get more elements than available
    int k = std::min(m_num_temp_instances, static_cast<int>(temp_conf.size(1)));

    // Skip if k is 0 or if tensor dimensions are invalid
    if (k <= 0 || temp_conf.size(0) == 0) {
        m_instance_id = torch::full({temp_conf.size(0), m_num_anchor}, -1, torch::kLong);
        return;
    }

    // Get top-k instance IDs based on confidence
    std::vector<torch::Tensor> inputs = {instance_id_tensor};
    auto result = topk(temp_conf, k, inputs);
    auto selected_ids = std::get<1>(result)[0];

    // Handle tensor dimension carefully
    if (selected_ids.dim() >= 3) {
        selected_ids = selected_ids.squeeze(-1);
    }

    // Pad with -1 to fill remaining slots
    m_instance_id = torch::constant_pad_nd(
        selected_ids, 
        {0, m_num_anchor - k},
        -1);
}

torch::Tensor load_anchor(const std::string& filename) {
    try {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open NPY file: " + filename);
        }

        // Parse NPY header
        char magic[6];
        file.read(magic, 6);

        // Check magic string: 147 78 149 150 032 001
        if (magic[0] != '\x93' || magic[1] != 'N' || magic[2] != 'U' ||
            magic[3] != 'M' || magic[4] != 'P' || magic[5] != 'Y') {
            throw std::runtime_error("Invalid NPY file format");
        }

        uint8_t major, minor;
        file.read(reinterpret_cast<char*>(&major), 1);
        file.read(reinterpret_cast<char*>(&minor), 1);

        uint16_t header_len;
        file.read(reinterpret_cast<char*>(&header_len), 2);

        // Read header
        std::vector<char> header(header_len);
        file.read(header.data(), header_len);

        // Parse header to get shape and dtype
        std::string header_str(header.begin(), header.end());

        // Extract shape
        size_t shape_start = header_str.find("(");
        size_t shape_end = header_str.find(")", shape_start);
        std::string shape_str = header_str.substr(shape_start + 1, shape_end - shape_start - 1);

        // Parse shape string into dimensions
        std::vector<int64_t> shape;
        size_t pos = 0;
        size_t comma_pos;
        while ((comma_pos = shape_str.find(",", pos)) != std::string::npos) {
            std::string dim_str = shape_str.substr(pos, comma_pos - pos);
            // Trim whitespace
            dim_str.erase(0, dim_str.find_first_not_of(" "));
            dim_str.erase(dim_str.find_last_not_of(" ") + 1);
            if (!dim_str.empty()) {
                shape.push_back(std::stoll(dim_str));
            }
            pos = comma_pos + 1;
        }

        // Get the last dimension if there is one
        std::string last_dim = shape_str.substr(pos);
        last_dim.erase(0, last_dim.find_first_not_of(" "));
        last_dim.erase(last_dim.find_last_not_of(" ") + 1);
        if (!last_dim.empty()) {
            shape.push_back(std::stoll(last_dim));
        }

        // Calculate total elements
        int64_t total_elements = 1;
        for (int64_t dim : shape) {
            total_elements *= dim;
        }

        // Check for various data types
        bool is_float32 = header_str.find("float32") != std::string::npos || header_str.find("'<f4'") != std::string::npos;
        bool is_float64 = header_str.find("float64") != std::string::npos || header_str.find("'<f8'") != std::string::npos;
        bool is_int32 = header_str.find("int32") != std::string::npos || header_str.find("'<i4'") != std::string::npos;
        bool is_int64 = header_str.find("int64") != std::string::npos || header_str.find("'<i8'") != std::string::npos;
        bool is_int16 = header_str.find("int16") != std::string::npos || header_str.find("'<i2'") != std::string::npos;
        bool is_int8 = header_str.find("int8") != std::string::npos || header_str.find("'<i1'") != std::string::npos;
        bool is_uint8 = header_str.find("uint8") != std::string::npos || header_str.find("'<u1'") != std::string::npos;
        bool is_bool = header_str.find("bool") != std::string::npos || header_str.find("'<b1'") != std::string::npos;

        // Try to extract the actual data type from descr field
        size_t dtype_start = header_str.find("'descr'");
        size_t dtype_value_start = header_str.find("'", dtype_start + 8);
        size_t dtype_value_end = header_str.find("'", dtype_value_start + 1);
        std::string dtype_value = "unknown";

        if (dtype_value_start != std::string::npos && dtype_value_end != std::string::npos) {
            dtype_value = header_str.substr(dtype_value_start + 1, dtype_value_end - dtype_value_start - 1);
            
            // Handle NumPy's encoded format strings
            if (dtype_value == "<f8") is_float64 = true;
            else if (dtype_value == "<f4") is_float32 = true;
            else if (dtype_value == "<i8") is_int64 = true;
            else if (dtype_value == "<i4") is_int32 = true;
            else if (dtype_value == "<i2") is_int16 = true;
            else if (dtype_value == "<i1") is_int8 = true;
            else if (dtype_value == "<u1") is_uint8 = true;
            else if (dtype_value == "<b1") is_bool = true;
        }

        if (is_float32) {
            // Read float32 data
            std::vector<float> data(total_elements);
            file.read(reinterpret_cast<char*>(data.data()), total_elements * sizeof(float));
            return torch::from_blob(data.data(), shape, torch::kFloat32).clone();
        }
        else if (is_float64) {
            // Read float64 data and convert to float32
            std::vector<double> data_double(total_elements);
            file.read(reinterpret_cast<char*>(data_double.data()), total_elements * sizeof(double));

            // Convert to float32
            auto options = torch::TensorOptions().dtype(torch::kFloat32);
            auto tensor = torch::empty(shape, options);
            auto tensor_data = tensor.data_ptr<float>();

            // Copy and convert data
            for (int64_t i = 0; i < total_elements; i++) {
                tensor_data[i] = static_cast<float>(data_double[i]);
            }

            return tensor;
        }
        else if (is_int32) {
            // Read int32 data
            std::vector<int32_t> data(total_elements);
            file.read(reinterpret_cast<char*>(data.data()), total_elements * sizeof(int32_t));
            return torch::from_blob(data.data(), shape, torch::kInt32).clone();
        }
        else if (is_int64) {
            // Read int64 data
            std::vector<int64_t> data(total_elements);
            file.read(reinterpret_cast<char*>(data.data()), total_elements * sizeof(int64_t));
            return torch::from_blob(data.data(), shape, torch::kInt64).clone();
        }
        else if (is_int16) {
            // Read int16 data
            std::vector<int16_t> data(total_elements);
            file.read(reinterpret_cast<char*>(data.data()), total_elements * sizeof(int16_t));

            // Convert to int32
            std::vector<int32_t> data_int32(total_elements);
            for (int64_t i = 0; i < total_elements; i++) {
                data_int32[i] = static_cast<int32_t>(data[i]);
            }

            return torch::from_blob(data_int32.data(), shape, torch::kInt32).clone();
        }
        else if (is_int8) {
            // Read int8 data
            std::vector<int8_t> data(total_elements);
            file.read(reinterpret_cast<char*>(data.data()), total_elements * sizeof(int8_t));
            return torch::from_blob(data.data(), shape, torch::kInt8).clone();
        }
        else if (is_uint8) {
            // Read uint8 data
            std::vector<uint8_t> data(total_elements);
            file.read(reinterpret_cast<char*>(data.data()), total_elements * sizeof(uint8_t));
            return torch::from_blob(data.data(), shape, torch::kUInt8).clone();
        }
        else if (is_bool) {
            // Read bool data
            std::vector<bool> data_bool(total_elements);
            std::vector<uint8_t> data_raw(total_elements);
            file.read(reinterpret_cast<char*>(data_raw.data()), total_elements * sizeof(uint8_t));

            // Convert to uint8
            std::vector<uint8_t> data_uint8(total_elements);
            for (int64_t i = 0; i < total_elements; i++) {
                data_uint8[i] = data_raw[i] ? 1 : 0;
            }

            return torch::from_blob(data_uint8.data(), shape, torch::kUInt8).clone();
        }
        else {
            throw std::runtime_error("Unsupported NumPy data type: " + dtype_value + "\nHeader: " + header_str);
        }
    } catch (const std::exception& e) {
        std::cerr << "Error loading NPY file (" << filename << "): " << e.what() << std::endl;
        // Return an empty tensor on error
        return torch::Tensor();
    }
}
