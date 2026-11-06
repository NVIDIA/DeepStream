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

#include "config_parser.h"
#include <iostream>
#include <fstream>
#include <sys/stat.h>

// Helper function to check if file exists
static bool fileExists(const std::string& path) {
    struct stat buffer;
    return (stat(path.c_str(), &buffer) == 0);
}

ConfigParser::ConfigParser(const std::string& configPath)
    : m_ConfigPath("")
    , m_IsLoaded(false)
{
    loadConfig(configPath);
}

bool ConfigParser::loadConfig(const std::string& configPath) {
    // First check if the provided path exists
    if (fileExists(configPath)) {
        try {
            m_Config = YAML::LoadFile(configPath);
            m_ConfigPath = configPath;
            m_IsLoaded = true;

            // Update the properties based on the loaded config
            updatePropertiesFromConfig();

            return true;
        } catch (const YAML::Exception& e) {
            std::cerr << "Error parsing YAML config file: " << configPath << ": " << e.what() << std::endl;
        }
    } else {
        std::cerr << "Config file not found at: " << configPath << std::endl;

        // Try to load default config if different from provided path
        if (configPath != "/opt/nvidia/deepstream/deepstream/sources/sparse4d/configs/config.yaml" && fileExists("/opt/nvidia/deepstream/deepstream/sources/sparse4d/configs/config.yaml")) {
            std::cout << "Attempting to load default config file instead..." << std::endl;
            return loadConfig("/opt/nvidia/deepstream/deepstream/sources/sparse4d/configs/config.yaml");
        }
    }

    m_IsLoaded = false;
    return false;
}

void ConfigParser::updatePropertiesFromConfig() {
    if (!m_IsLoaded) {
        return;
    }

    // Update inference properties
    m_Properties.onnx_file = getString("onnx_file", m_Properties.onnx_file);
    m_Properties.engine_file = getString("engine_file", m_Properties.engine_file);
    m_Properties.labels_file = getString("labels_file", m_Properties.labels_file);

    // Parse batch_size parameter
    m_Properties.batch_size = getInt("batch_size", m_Properties.batch_size);

    // Parse num_sensors parameter
    m_Properties.num_sensors = getInt("num_sensors", m_Properties.num_sensors);

    m_Properties.enable_fp16 = getBool("enable_fp16", m_Properties.enable_fp16);
    m_Properties.force_engine_rebuild = getBool("force_engine_rebuild", m_Properties.force_engine_rebuild);
    m_Properties.feedback = getBool("feedback", m_Properties.feedback);
    m_Properties.partial_batch = getBool("partial_batch", m_Properties.partial_batch);
    m_Properties.interval = getInt("interval", m_Properties.interval);

    // Parse GPU ID
    if (m_Config["gpu_id"]) {
        try {
            m_Properties.gpu_id = m_Config["gpu_id"].as<int>();
        } catch (const YAML::Exception& e) {
            std::cout << "Warning: Invalid gpu_id in config, using default (0)" << std::endl;
        }
    }

    m_Properties.preprocessed_height = getInt("preprocessed_height", m_Properties.preprocessed_height);
    m_Properties.preprocessed_width = getInt("preprocessed_width", m_Properties.preprocessed_width);

    // Parse instance bank properties
    m_Properties.num_anchor = getInt("num_anchor", m_Properties.num_anchor);
    m_Properties.embed_dims = getInt("embed_dims", m_Properties.embed_dims);
    m_Properties.anchor = getString("anchor", m_Properties.anchor);
    m_Properties.anchor_handler = getString("anchor_handler", m_Properties.anchor_handler);
    m_Properties.num_temp_instances = getInt("num_temp_instances", m_Properties.num_temp_instances);
    m_Properties.default_time_interval = getFloat("default_time_interval", m_Properties.default_time_interval);
    m_Properties.confidence_decay = getFloat("confidence_decay", m_Properties.confidence_decay);
    m_Properties.anchor_grad = getBool("anchor_grad", m_Properties.anchor_grad);
    m_Properties.feat_grad = getBool("feat_grad", m_Properties.feat_grad);
    m_Properties.max_time_interval = getInt("max_time_interval", m_Properties.max_time_interval);
    m_Properties.reid_dims = getInt("reid_dims", m_Properties.reid_dims);

    // Parse decoder properties
    m_Properties.num_output = getInt("num_output", m_Properties.num_output);
    m_Properties.score_threshold = getFloat("score_threshold", m_Properties.score_threshold);

    // Parse performance properties
    m_Properties.num_torch_threads = getInt("num_torch_threads", m_Properties.num_torch_threads);

    // Parse visualization properties
    m_Properties.generate_3d_bbox = getBool("generate_3d_bbox", m_Properties.generate_3d_bbox);
    m_Properties.generate_2d_bbox = getBool("generate_2d_bbox", m_Properties.generate_2d_bbox);
    m_Properties.bbox_color_mode = getString("bbox_color_mode", m_Properties.bbox_color_mode);
    m_Properties.display_class_labels = getBool("display_class_labels", m_Properties.display_class_labels);
    m_Properties.display_instance_ids = getBool("display_instance_ids", m_Properties.display_instance_ids);
    m_Properties.display_class_score = getBool("display_class_score", m_Properties.display_class_score);
    m_Properties.display_3d_bbox = getBool("display_3d_bbox", m_Properties.display_3d_bbox);
    m_Properties.display_2d_bbox = getBool("display_2d_bbox", m_Properties.display_2d_bbox);

    // Parse calibration properties
    m_Properties.calib_file_path = getString("calib_file_path", m_Properties.calib_file_path);
    m_Properties.bev_group_name = getString("bev_group_name", m_Properties.bev_group_name);
    m_Properties.calib_mode = getString("calib_mode", m_Properties.calib_mode);
    m_Properties.use_camera_groups = getBool("use_camera_groups", m_Properties.use_camera_groups);
    m_Properties.recentering = getBool("recentering", m_Properties.recentering);

    // Update debugging properties
    m_Properties.log_level = getString("log_level", m_Properties.log_level);
    m_Properties.display_partial_batch_info = getBool("display_partial_batch_info", m_Properties.display_partial_batch_info);
    m_Properties.display_tensor_info = getBool("display_tensor_info", m_Properties.display_tensor_info);
    m_Properties.display_bbox_list_info = getBool("display_bbox_list_info", m_Properties.display_bbox_list_info);
    m_Properties.max_objects_to_display = getInt("max_objects_to_display", m_Properties.max_objects_to_display);
    m_Properties.dump_frames = getBool("dump_frames", m_Properties.dump_frames);
    m_Properties.dump_max_frames = getInt("dump_max_frames", m_Properties.dump_max_frames);
    m_Properties.fixed_input_tensor_path = getString("fixed_input_tensor_path", m_Properties.fixed_input_tensor_path);
    m_Properties.gpu_postprocess = getBool("gpu_postprocess", m_Properties.gpu_postprocess);
    m_Properties.use_static_frame_num = getBool("use_static_frame_num", m_Properties.use_static_frame_num);

    // Parse TensorRT properties
    m_Properties.trt_logger_severity = getInt("trt_logger_severity", m_Properties.trt_logger_severity);
    m_Properties.enable_profiler = getBool("enable_profiler", m_Properties.enable_profiler);

    // Parse augmentation configuration
    if (m_Config["aug_configs"]) {
        try {
            const YAML::Node& aug_node = m_Config["aug_configs"];

            // Parse resize factor
            if (aug_node["resize"]) {
                m_Properties.aug_config.resize = aug_node["resize"].as<float>();
            }

            // Parse resize dimensions
            if (aug_node["resize_dims"]) {
                if (aug_node["resize_dims"].IsSequence() && aug_node["resize_dims"].size() == 2) {
                    m_Properties.aug_config.resize_dims = {
                        aug_node["resize_dims"][0].as<int>(),
                        aug_node["resize_dims"][1].as<int>()
                    };
                } else {
                    std::cout << "Warning: Invalid resize_dims format in config, must be [width, height]" << std::endl;
                }
            }

            // Parse crop parameters
            if (aug_node["crop"]) {
                if (aug_node["crop"].IsSequence() && aug_node["crop"].size() == 4) {
                    m_Properties.aug_config.crop = {
                        aug_node["crop"][0].as<int>(),
                        aug_node["crop"][1].as<int>(),
                        aug_node["crop"][2].as<int>(),
                        aug_node["crop"][3].as<int>()
                    };
                } else {
                    std::cout << "Warning: Invalid crop format in config, must be [x, y, width, height]" << std::endl;
                }
            }

            // Parse flip parameter
            if (aug_node["flip"]) {
                m_Properties.aug_config.flip = aug_node["flip"].as<bool>();
            }

            // Parse rotation angle
            if (aug_node["rotate"]) {
                m_Properties.aug_config.rotate = aug_node["rotate"].as<float>();
            }

            // Parse 3D rotation angle
            if (aug_node["rotate_3d"]) {
                m_Properties.aug_config.rotate_3d = aug_node["rotate_3d"].as<float>();
            }

        } catch (const YAML::Exception& e) {
            std::cout << "Warning: Error parsing augmentation configuration: " << e.what() << std::endl;
        }
    }

    // Add more property updates here as they are added to the ConfigProperties struct
}

std::string ConfigParser::getString(const std::string& key, const std::string& defaultValue) const {
    if (!m_IsLoaded || !m_Config[key]) {
        return defaultValue;
    }
    return m_Config[key].as<std::string>();
}

int ConfigParser::getInt(const std::string& key, int defaultValue) const {
    if (!m_IsLoaded || !m_Config[key]) {
        return defaultValue;
    }
    return m_Config[key].as<int>();
}

float ConfigParser::getFloat(const std::string& key, float defaultValue) const {
    if (!m_IsLoaded || !m_Config[key]) {
        return defaultValue;
    }
    return m_Config[key].as<float>();
}

bool ConfigParser::getBool(const std::string& key, bool defaultValue) const {
    if (!m_IsLoaded || !m_Config[key]) {
        return defaultValue;
    }
    return m_Config[key].as<bool>();
}

bool ConfigParser::hasKey(const std::string& key) const {
    if (!m_IsLoaded) {
        return false;
    }
    return m_Config[key].IsDefined();
}

std::string ConfigParser::getConfigPath() const {
    return m_ConfigPath;
}

bool ConfigParser::isLoaded() const {
    return m_IsLoaded;
}

const ConfigProperties& ConfigParser::getProperties() const {
    return m_Properties;
}
