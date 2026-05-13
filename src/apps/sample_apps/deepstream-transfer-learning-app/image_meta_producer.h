/*
 * SPDX-FileCopyrightText: Copyright (c) 2020 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once

#include <sstream>
#include <iomanip>
#include <fstream>
#include <chrono>
#include <ctime>
#include "image_meta_consumer.h"
#include "concurrent_queue.h"

class ImageMetaProducer
{
public:
    // left: filepath for multiple file, right: file content
    typedef std::pair<std::string, std::string> string_pair;
    /// Content that will converted to string and sent to consumer
    struct IPData
    {
        float confidence = 0.f;
        bool within_confidence;
        unsigned class_id = 0;
        unsigned current_frame = 0;
        unsigned video_stream_nb = 0;
        std::string class_name;
        std::string video_path;
        std::string image_full_frame_path_saved;
        std::string image_cropped_obj_path_saved;
        std::string datetime;
        unsigned img_height = 0;
        unsigned img_width = 0;
        unsigned img_top = 0;
        unsigned img_left = 0;
    };

    /// Constructor registering a consumer
    /// @param ic The image consumer
    ImageMetaProducer(ImageMetaConsumer &ic);
    /// store metadata information locally before send it.
    bool stack_obj_data(IPData &data);
    /// send metadata stored (after some processing for KITTI type).
    /// Empty the metadata list and full frame path
    void send_and_flush_obj_data();
    /// Create and store locally a path to a complete image.
    void generate_image_full_frame_path(const unsigned stream_source_id,
                                        const std::string &datetime_iso8601);
    /// Returns locally stored path to complete image
    std::string get_image_full_frame_path_saved();

private:

    /// Format a string to csv and return it.
    std::string make_csv_data(const IPData &data);
    /// Format a string to Json and return it.
    std::string make_json_data(const IPData &data);
    /// Format a string to KITTI and return it.
    std::string make_kitti_data(const IPData &data);
    /// Makes a path for KITTI metadata save and return it.
    std::string make_kitti_save_path() const;

    std::string image_full_frame_path_saved_;
    std::vector<std::string> obj_data_csv_;
    std::vector<std::string> obj_data_json_;
    std::vector<std::string> obj_data_kitti_;
    ImageMetaConsumer &ic_;
};