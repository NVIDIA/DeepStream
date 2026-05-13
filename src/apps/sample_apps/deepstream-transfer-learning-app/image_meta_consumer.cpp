/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "image_meta_consumer.h"

constexpr unsigned seconds_in_one_day = 86400;

static int is_dir(const char *path) {
    struct stat path_stat;
    stat(path, &path_stat);
    return S_ISDIR(path_stat.st_mode);
}

ImageMetaConsumer::ImageMetaConsumer()
        : is_stopped_(true), save_full_frame_enabled_(true), save_cropped_obj_enabled_(false),
          obj_ctx_handle_((NvDsObjEncCtxHandle) 0), image_saving_library_is_init_(false) {
}

ImageMetaConsumer::~ImageMetaConsumer() {
    stop();
}

unsigned int ImageMetaConsumer::get_unique_id() {
    mutex_unique_index_.lock();
    auto res = unique_index_++;
    mutex_unique_index_.unlock();
    return res;
}

void ImageMetaConsumer::add_meta_csv(const std::string &meta) {
    if (is_stopped_) {
        std::cerr << __func__ << ": Could not add meta when Consumer is stopped.\n";
        return;
    }
    queue_csv_.push(meta);
    /// tell the reading thread to work
    cv_csv_.notify_one();
}

void ImageMetaConsumer::add_meta_json(const std::string &meta) {
    if (is_stopped_) {
        std::cerr << __func__ << ": Could not add meta when Consumer is stopped.\n";
        return;
    }
    queue_json_.push(meta);
    /// tell the reading thread to work
    cv_json_.notify_one();
}

void ImageMetaConsumer::add_meta_kitti(const std::pair<std::string, std::string> &meta) {
    if (is_stopped_) {
        std::cerr << __func__ << ": Could not add meta when Consumer is stopped.\n";
        return;
    }
    queue_kitti_.push(meta);
    /// tell the reading thread to work
    cv_kitti_.notify_one();
}

void ImageMetaConsumer::stop() {
    if (is_stopped_)
        return;
    is_stopped_ = true;
    /// tell the reading thread to work one last time
    cv_kitti_.notify_one();
    cv_json_.notify_one();
    cv_csv_.notify_one();
    th_kitti_.join();
    th_json_.join();
    th_csv_.join();
    mutex_image_save_init_.lock();
    if (image_saving_library_is_init_) {
        nvds_obj_enc_destroy_context(obj_ctx_handle_);
    }
    mutex_image_save_init_.unlock();
}

void ImageMetaConsumer::init(const unsigned gpu_id,
                             const std::string &output_folder_path, const std::string &frame_to_skip_rules_path,
                             const float min_box_confidence, const float max_box_confidence,
                             const unsigned min_box_width, const unsigned min_box_height,
                             const bool save_full_frame_enabled, const bool save_cropped_obj_enabled,
                             const unsigned seconds_to_skip_interval, const unsigned source_nb) {
    if (!is_stopped_) {
        std::cerr << "Consummer already running.\n";
        return;
    }
    if (!is_dir(output_folder_path.c_str())) {
        std::cerr << "Missing directory: " << output_folder_path << ".\n";
        return;
    }
    output_folder_path_ = output_folder_path;
    if (output_folder_path_.back() != '/')
        output_folder_path_ += '/';

    ctr_.init(frame_to_skip_rules_path, seconds_to_skip_interval);
    if (!ctr_.is_init_())
        return;

    if (!setup_folders()) {
        std::cerr << "Could not create " << images_cropped_obj_output_folder_
                  << " , " << images_full_frame_output_folder_
                  << " and " << labels_output_folder_ << "\n";
        return;
    }

    if (!setup_files()) {
        std::cerr << "Could not create metadata.json and metadata.csv\n";
        return;
    }

    gpu_id_ = gpu_id;
    min_confidence_ = min_box_confidence;
    max_confidence_ = max_box_confidence;
    min_box_width_ = min_box_width;
    min_box_height_ = min_box_height;
    save_full_frame_enabled_ = save_full_frame_enabled;
    save_cropped_obj_enabled_ = save_cropped_obj_enabled;

    auto stsi = std::chrono::seconds(seconds_in_one_day);
    for (unsigned i = 0; i < source_nb; ++i)
        time_last_frame_saved_list_.push_back(std::chrono::system_clock::now() - stsi);

    is_stopped_ = false;
    run();
}

bool ImageMetaConsumer::setup_folders() {
    images_cropped_obj_output_folder_ = output_folder_path_ + "images_cropped/";
    images_full_frame_output_folder_ = output_folder_path_ + "images/";
    labels_output_folder_ = output_folder_path_ + "labels/";
    unsigned permissions = 0755;
    mkdir(images_cropped_obj_output_folder_.c_str(), permissions);
    mkdir(images_full_frame_output_folder_.c_str(), permissions);
    mkdir(labels_output_folder_.c_str(), permissions);
    return is_dir(images_cropped_obj_output_folder_.c_str())
           && is_dir(images_full_frame_output_folder_.c_str())
           && is_dir(labels_output_folder_.c_str());
}


void ImageMetaConsumer::write_intro(std::ofstream &os, OutputType &ot) {
    switch (ot) {
        case OutputType::JSON:
            os << "{\n";
            os << "\"meta_array\" : {\n";
            break;
        case OutputType::CSV:
            os << "class_id,";
            os << "class_name,";
            os << "confidence,";
            os << "within_confidence,";
            os << "current_frame,";
            os << "image_cropped_obj_path_saved,";
            os << "image_full_frame_path_saved,";
            os << "datetime,";
            os << "img_height,";
            os << "img_width,";
            os << "img_top,";
            os << "img_left,";
            os << "video_path,";
            os << "video_stream_nb";
            os << "\n";
            break;
        case KITTI:
            break;
    }
}

void ImageMetaConsumer::write_mid_separator(std::ofstream &os, OutputType &ot) {
    switch (ot) {
        case JSON:
            os << ",\n";
            break;
        case CSV:
            os << "\n";
            break;
        case KITTI:
            break;
    }
}

void ImageMetaConsumer::write_end(std::ofstream &os, OutputType &ot, unsigned total_nb) {
    switch (ot) {
        case JSON:
            os << "},\n";
            os << "\"medatada_nb\" : " << total_nb << ",\n";
            os << "\"min_confidence\" : " << get_min_confidence() << ",\n";
            os << "\"max_confidence\" : " << get_max_confidence() << "\n";
            os << "}\n";
            break;
        case CSV:
            break;
        case KITTI:
            break;
    }
}

void ImageMetaConsumer::single_metadata_maker(const std::string &extension,
                                              ConcurrentQueue<std::string> &queue,
                                              std::mutex &mutex, std::condition_variable &cv,
                                              OutputType ot) {
    std::string meta_path = output_folder_path_ + "metadata." + extension;
    std::ofstream output(meta_path, std::ios::trunc);
    if (!output.good()) {
        std::cerr << "Could not create " << meta_path << std::endl;
        is_stopped_ = true;
        return;
    }
    write_intro(output, ot);

    bool first_time = true;
    unsigned long meta_nb = 0;
    while (!is_stopped_) {
        {
            std::unique_lock<std::mutex> lk(mutex);
            cv.wait(lk);
        }
        while (!queue.is_empty()) {
            auto meta = queue.pop();

            if (first_time)
                first_time = false;
            else
                write_mid_separator(output, ot);
            output << meta;
            meta_nb++;
        }
    }
    write_end(output, ot, meta_nb);
}

bool ImageMetaConsumer::setup_files() {
    std::string p1 = output_folder_path_ + "metadata.csv";
    std::ofstream output1(p1, std::ios::trunc);
    std::string p2 = output_folder_path_ + "metadata.json";
    std::ofstream output2(p2, std::ios::trunc);
    return output1.good() && output2.good();
}

void ImageMetaConsumer::multi_metadata_maker(ConcurrentQueue<std::pair<std::string, std::string>> &queue,
                                             std::mutex &mutex, std::condition_variable &cv) {

    while (!is_stopped_) {
        {
            std::unique_lock<std::mutex> lk(mutex);
            cv.wait(lk);
        }
        while (!queue.is_empty()) {
            auto meta = queue.pop();
            std::ofstream output(labels_output_folder_ + meta.first, std::ios::trunc);
            if (!output.good()) {
                std::cerr << "Could not create " << labels_output_folder_ << meta.first << std::endl;
                is_stopped_ = true;
                return;
            }
            output << meta.second;
        }
    }
}

void ImageMetaConsumer::run() {
    th_kitti_ = std::thread([this]() {
        multi_metadata_maker(queue_kitti_, m_kitti_, cv_kitti_);
    });
    th_json_ = std::thread([this]() {
        single_metadata_maker("json", queue_json_, m_json_, cv_json_, JSON);
    });
    th_csv_ = std::thread([this]() {
        single_metadata_maker("csv", queue_csv_, m_csv_, cv_csv_, CSV);
    });

}

std::string ImageMetaConsumer::make_img_path(const ImageMetaConsumer::ImageSizeType ist,
                                             const unsigned stream_source_id,
                                             const std::string &datetime_iso8601) {
    unsigned long id = get_unique_id();
    std::stringstream ss;
    switch (ist) {
        case FULL_FRAME:
            ss << images_full_frame_output_folder_;
            break;
        case CROPPED_TO_OBJECT:
            ss << images_cropped_obj_output_folder_;
            break;
    }
    ss << "camera-" << stream_source_id << "_";
    ss << datetime_iso8601 << "_";
    ss << std::setw(10) << std::setfill('0') << id;
    ss << ".jpg";
    return ss.str();
}


NvDsObjEncCtxHandle ImageMetaConsumer::get_obj_ctx_handle() {
    return obj_ctx_handle_;
}

float ImageMetaConsumer::get_min_confidence() const {
    return min_confidence_;
}

float ImageMetaConsumer::get_max_confidence() const {
    return max_confidence_;
}

unsigned ImageMetaConsumer::get_gpu_id() const {
    return gpu_id_;
}

unsigned ImageMetaConsumer::get_min_box_width() const {
    return min_box_width_;
}

unsigned ImageMetaConsumer::get_min_box_height() const {
    return min_box_height_;
}

bool ImageMetaConsumer::get_is_stopped() const {
    return is_stopped_;
}

bool ImageMetaConsumer::get_save_full_frame_enabled() const {
    return save_full_frame_enabled_;
}

bool ImageMetaConsumer::get_save_cropped_images_enabled() const {
    return save_cropped_obj_enabled_;
}

void ImageMetaConsumer::init_image_save_library_on_first_time() {
    mutex_image_save_init_.lock();
    if (!image_saving_library_is_init_) {
        m_kitti_.lock();
        m_json_.lock();
        m_csv_.lock();
        if (!image_saving_library_is_init_
            && (save_cropped_obj_enabled_ || save_full_frame_enabled_)) {
            obj_ctx_handle_ = nvds_obj_enc_create_context(gpu_id_);
            if (obj_ctx_handle_)
                image_saving_library_is_init_ = true;
            else
                std::cerr << "Unable to create encoding context\n";
        }
        m_csv_.unlock();
        m_json_.unlock();
        m_kitti_.unlock();
    }
    mutex_image_save_init_.unlock();
}

bool ImageMetaConsumer::should_save_data(unsigned source_id) {
    auto time_interval = ctr_.getCurrentTimeInterval();
    auto now = std::chrono::system_clock::now();
    if (now < time_last_frame_saved_list_[source_id]) {
        std::cerr << "The current time seems to have moved to the past\n"
                  << "Reseting time for last frame save for source" << source_id << "\n";
        time_last_frame_saved_list_[source_id] = now;
    }
    return now > time_interval + time_last_frame_saved_list_[source_id];
}

void ImageMetaConsumer::lock_source_nb(unsigned source_id) {
    mutex_frame_saved_list_[source_id].lock();
}

void ImageMetaConsumer::data_was_saved_for_source(unsigned source_id) {
    time_last_frame_saved_list_[source_id] = std::chrono::system_clock::now();
}

void ImageMetaConsumer::unlock_source_nb(unsigned source_id) {
    mutex_frame_saved_list_[source_id].unlock();
}