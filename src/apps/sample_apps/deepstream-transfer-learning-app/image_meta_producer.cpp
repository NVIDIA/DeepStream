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

#include "image_meta_producer.h"

typedef std::pair<std::string, std::string> string_pair;

ImageMetaProducer::ImageMetaProducer(ImageMetaConsumer &ic)
        : ic_(ic) {
}


static std::string get_filename(const std::string &filepath) {
    std::string filename = filepath;
    const size_t last_slash_idx = filename.find_last_of('/');
    if (std::string::npos != last_slash_idx)
        filename.erase(0, last_slash_idx + 1);

    const size_t period_idx = filename.rfind('.');
    if (std::string::npos != period_idx)
        filename.erase(period_idx);
    return filename;
}

std::string ImageMetaProducer::make_kitti_save_path() const {
    std::string name_copy = get_filename(image_full_frame_path_saved_);
    name_copy += ".txt";
    return name_copy;
}

void ImageMetaProducer::send_and_flush_obj_data() {


    for (const auto &elm: obj_data_json_)
        ic_.add_meta_json(elm);
    obj_data_json_.clear();

    for (const auto &elm: obj_data_csv_)
        ic_.add_meta_csv(elm);
    obj_data_csv_.clear();

    if (!obj_data_kitti_.empty()) {
        std::string res;
        for (const auto &elm: obj_data_kitti_) {
            res += elm + "\n";
        }
        string_pair sp(make_kitti_save_path(), res);
        ic_.add_meta_kitti(sp);
    }

    obj_data_kitti_.clear();
    image_full_frame_path_saved_.clear();
}

void ImageMetaProducer::generate_image_full_frame_path(const unsigned stream_source_id,
                                                       const std::string &datetime_iso8601) {
    image_full_frame_path_saved_ = ic_.make_img_path(ImageMetaConsumer::FULL_FRAME,
                                                     stream_source_id,
                                                     datetime_iso8601);
}

std::string ImageMetaProducer::get_image_full_frame_path_saved() {
    return image_full_frame_path_saved_;
}

bool ImageMetaProducer::stack_obj_data(IPData &data) {
    if (ic_.get_is_stopped())
        return false;

    if (ic_.get_save_full_frame_enabled())
        data.image_full_frame_path_saved = image_full_frame_path_saved_;

    obj_data_csv_.push_back(make_csv_data(data));
    obj_data_json_.push_back(make_json_data(data));
    obj_data_kitti_.push_back(make_kitti_data(data));
    return true;
}

static std::string format_json_string(const std::string &str) {
    return "\"" + str + "\"";
}

std::string ImageMetaProducer::make_json_data(const IPData &data) {
    const std::string path_full_frame = ic_.get_save_full_frame_enabled() ? data.image_full_frame_path_saved : "";
    const std::string path_cropped_obj = ic_.get_save_cropped_images_enabled() ? data.image_cropped_obj_path_saved : "";

    std::stringstream ss;
    ss << format_json_string(get_filename(data.image_cropped_obj_path_saved)) << " : ";
    ss << "{\n";
    ss << "  " << format_json_string("class_id") << " : " << data.class_id << ",\n";
    ss << "  " << format_json_string("class_name") << " : " << format_json_string(data.class_name) << ",\n";
    ss << "  " << format_json_string("confidence") << " : " << data.confidence << ",\n";
    ss << "  " << format_json_string("within_confidence") << " : " << data.within_confidence << ",\n";
    ss << "  " << format_json_string("current_frame") << " : " << data.current_frame << ",\n";
    ss << "  " << format_json_string("image_cropped_obj_path_saved") << " : "
       << format_json_string(path_cropped_obj) << ",\n";
    ss << "  " << format_json_string("image_full_frame_path_saved") << " : "
       << format_json_string(path_full_frame) << ",\n";
    ss << "  " << format_json_string("datetime") << " : "
       << format_json_string(data.datetime) << ",\n";
    ss << "  " << format_json_string("img_height") << " : " << data.img_height << ",\n";
    ss << "  " << format_json_string("img_width") << " : " << data.img_width << ",\n";
    ss << "  " << format_json_string("img_top") << " : " << data.img_top << ",\n";
    ss << "  " << format_json_string("img_left") << " : " << data.img_left << ",\n";
    ss << "  " << format_json_string("video_path") << " : " << format_json_string(data.video_path) << ",\n";
    ss << "  " << format_json_string("video_stream_nb") << " : " << data.video_stream_nb << "\n";
    ss << "}\n";
    return ss.str();
}

std::string ImageMetaProducer::make_csv_data(const IPData &data) {
    const std::string path_full_frame = ic_.get_save_full_frame_enabled() ? data.image_full_frame_path_saved : "";
    const std::string path_cropped_obj = ic_.get_save_cropped_images_enabled() ? data.image_cropped_obj_path_saved : "";

    std::stringstream ss;
    ss << data.class_id << ",";
    ss << data.class_name << ",";
    ss << data.confidence << ",";
    ss << data.within_confidence << ",";
    ss << data.current_frame << ",";
    ss << path_cropped_obj << ",";
    ss << path_full_frame << ",";
    ss << data.datetime << ",";
    ss << data.img_height << ",";
    ss << data.img_width << ",";
    ss << data.img_top << ",";
    ss << data.img_left << ",";
    ss << data.video_path << ",";
    ss << data.video_stream_nb;
    return ss.str();
}

std::string ImageMetaProducer::make_kitti_data(const IPData &data) {

    std::stringstream ss;
    // Please refer to :
    // https://docs.nvidia.com/tao/tao-toolkit/text/data_annotation_format.html#object-detection-kitti-format
    ss << data.class_name << " "; // Class names
    ss << "0.0" << " "; // Truncation (No data default value)
    ss << "3" << " "; // Occlusion [ 0 = fully visible, 1 = partly visible, 2 = largely occluded, 3 = unknown].
    ss << "0.0" << " "; // Alpha (No data default value)
    // Bounding box coordinates:
    ss << data.img_left << " "; // ymin
    ss << data.img_top << " "; // xmin
    ss << (data.img_left + data.img_width) << " "; // ymax
    ss << (data.img_top + data.img_height) << " "; // xmax
    ss << "0.0 0.0 0.0" << " "; // 3-D dimension (No data default value)
    ss << "0.0 0.0 0.0" << " "; // Location (No data default value)
    ss << "0.0" << " "; // Rotation_y (No data default value)
    return ss.str();
}
