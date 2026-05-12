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

#pragma once

#include <condition_variable>
#include <array>
#include <mutex>
#include <string>
#include <atomic>
#include <thread>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <ios>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <deepstream_config.h>
#include "gstnvdsmeta.h"
#include "nvbufsurface.h"
#include "gst-nvmessage.h"
#include "nvds_obj_encode.h"
#include "concurrent_queue.h"
#include "capture_time_rules.h"

class ImageMetaConsumer {
public:
    enum OutputType {
        KITTI = 0,
        JSON = 1,
        CSV = 2
    };

    enum ImageSizeType {
        FULL_FRAME,
        CROPPED_TO_OBJECT
    };

    /// Init an object and set that the queue is stopped
    ImageMetaConsumer();

    /// End the job of the thread dequeing by calling stop()
    ~ImageMetaConsumer();

    /// Fill information in the Consumer. This has to be done when these informations
    /// available.
    /// @param [in] output_folder_path Path where all the metadata will be saved.
    /// @param [in] frame_to_skip_rules_path Path where the rules for the amount of time to
    /// skip between 2 frames is stored.
    /// @param [in] min_box_confidence Minimum confidence that an object detected
    /// on a frame should have, in order to be kept.
    /// @param [in] max_box_confidence Maximum confidence that an object detected
    /// on a frame should have, in order to be kept.
    /// @param [in] gpu_id GPU ID of the gpu which is to be used for encoding
    /// @param [in] min_box_width Minimum box width that an object detected
    /// on a frame should have, in order to be kept.
    /// @param [in] min_box_height Minimum box height that an object detected
    /// on a frame should have, in order to be kept.
    /// @param [in] save_full_frame_enabled Enable/Disable the save of complete images.
    /// @param [in] save_cropped_obj_enabled Enable/Disable the save of cropped images
    /// @param [in] seconds_to_skip_interval Unsigned integer giving the number of seconds to skip.
    /// @param [in] source_nb Unsigned integer giving the number of sources that are currently giving a stream.
    void init(unsigned gpu_id, const std::string &output_folder_path, const std::string &frame_to_skip_rules_path,
              float min_box_confidence, float max_box_confidence,
              unsigned min_box_width, unsigned min_box_height,
              bool save_full_frame_enabled, bool save_cropped_obj_enabled,
              unsigned seconds_to_skip_interval, unsigned source_nb);

    /// Add metadata to the stored concurrent queue.
    /// @param [in] meta Metadata as CSV string
    void add_meta_csv(const std::string &meta);

    /// Add metadata to the stored concurrent queue.
    /// @param [in] meta Metadata as JSON string
    void add_meta_json(const std::string &meta);

    /// Add metadata to the stored concurrent queue.
    /// @param [in] meta Metadata, left is the path needed for multi_metadata_maker()
    /// right is the content to write.
    void add_meta_kitti(const std::pair<std::string, std::string> &meta);

    /// End the job of the current thread reading from the queue.
    void stop();


    /// Min confidence getter.
    /// @return The minimum confidence required for an object.
    float get_min_confidence() const;

    /// Max confidence getter.
    /// @return The maximum confidence required for an object.
    float get_max_confidence() const;

    /// GPU ID getter.
    /// @return The gpu id used for encoding an object.
    unsigned get_gpu_id() const;

    /// Minimum box width getter.
    /// @return The minimum box width required for an object.
    unsigned get_min_box_width() const;

    /// Minimum box height getter.
    /// @return The minimum box height required for an object.
    unsigned get_min_box_height() const;

    /// Dequeuing thread status getter.
    /// @return The status of the thread reading metadata.
    bool get_is_stopped() const;

    /// Dequeuing full frame enabled getter.
    /// @return If complete images must be saved.
    bool get_save_full_frame_enabled() const;

    /// Dequeuing cropped image enable getter.
    /// @return If cropped images must be saved.
    bool get_save_cropped_images_enabled() const;

    /// Image Save Thread Handler
    /// @return the thread handler
    NvDsObjEncCtxHandle get_obj_ctx_handle();

    /// Metadata writer for a unique file (Json or Csv)
    /// @param extension of file requested (Json or Csv)
    /// @param stream_source_id Unique number identifying the stream source
    /// @param datetime_iso8601 current datetime formatted to iso 8601
    std::string make_img_path(ImageMetaConsumer::ImageSizeType ist,
                              unsigned stream_source_id,
                              const std::string &datetime_iso8601);

    /// Create image saving context if needed
    void init_image_save_library_on_first_time();

    /// This function checks if a certain amount of time passed before the
    /// last image was saved.
    /// \param source_id video stream number to check
    /// \return True if the required amount of time has passed before last
    /// the image was saved.
    bool should_save_data(unsigned source_id);

    /// Lock a stream when an image save is needed in order not to save multiple
    /// images from the same stream without waiting a certain amount of time.
    /// \param source_id The stream number to lock
    void lock_source_nb(unsigned source_id);

    /// If an image was saved, then the timestamp of the last image saved is stored.
    /// \param source_id Stream number for which the timestamp is saved.
    void data_was_saved_for_source(unsigned source_id);

    /// Unlock a stream when a stream was locked for image saving
    /// \param source_id The stream number to unlock
    void unlock_source_nb(unsigned source_id);

private:

    /// Function launching 3 threads for KITTI JSON and CSV output.
    void run();

    /// Metadata writer for a file per metadata (KITTI)
    void multi_metadata_maker(ConcurrentQueue<std::pair<std::string, std::string>> &queue,
                              std::mutex &mutex, std::condition_variable &cv);

    /// Set up config files
    bool setup_files();

    /// Creates folder for images and metadata output.
    bool setup_folders();

    /// Write at the beginning of a file depending of the output type.
    void write_intro(std::ofstream &os, OutputType &ot);

    /// Write at the middle of a file between metadata depending of the output type.
    void write_mid_separator(std::ofstream &os, OutputType &ot);

    /// Write at the end of a file depending of the output type.
    void write_end(std::ofstream &os, OutputType &ot, unsigned total_nb);

    /// Creates a unique id for the current consumer.
    unsigned int get_unique_id();

    /// Metadata writer for a unique file (Json or Csv)
    /// @param extension of file requested (Json or Csv)
    void single_metadata_maker(const std::string &extension,
                               ConcurrentQueue<std::string> &queue,
                               std::mutex &mutex, std::condition_variable &cv,
                               OutputType ot);

    ConcurrentQueue<std::pair<std::string, std::string>> queue_kitti_;
    ConcurrentQueue<std::string> queue_csv_;
    ConcurrentQueue<std::string> queue_json_;
    std::mutex m_kitti_;
    std::mutex m_json_;
    std::mutex m_csv_;
    std::condition_variable cv_kitti_;
    std::condition_variable cv_json_;
    std::condition_variable cv_csv_;
    std::atomic<bool> is_stopped_;
    std::string output_folder_path_;
    std::string images_cropped_obj_output_folder_;
    std::string images_full_frame_output_folder_;
    std::string labels_output_folder_;
    std::thread th_kitti_;
    std::thread th_json_;
    std::thread th_csv_;
    std::mutex mutex_unique_index_;
    unsigned int unique_index_ = 0;
    float min_confidence_;
    float max_confidence_;
    unsigned gpu_id_;
    unsigned min_box_width_;
    unsigned min_box_height_;
    bool save_full_frame_enabled_;
    bool save_cropped_obj_enabled_;
    std::vector<std::chrono::_V2::system_clock::time_point> time_last_frame_saved_list_;
    std::array<std::mutex, MAX_SOURCE_BINS> mutex_frame_saved_list_;
    CaptureTimeRules ctr_;
    NvDsObjEncCtxHandle obj_ctx_handle_;
    bool image_saving_library_is_init_;
    std::mutex mutex_image_save_init_;
};