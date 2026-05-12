/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

namespace deepstream {

class Element;


class PerfMonitor {
 public:
  /**
   * @brief constructor for a performance monitor instance
   *
   * @param[in] batch_size  batch size of the pipeline
   * @param[in] interval    monitor interval in seconds
   * @param[in] src_type    type name of the source bin
   * @param[in] show_name   show the stream name in perf log
   *
  */
  PerfMonitor(unsigned int batch_size, uint64_t interval,
    const std::string src_type, bool show_name=true
  );

  virtual ~PerfMonitor();

  /**
   * @brief Apply the performance monitory on an element
   *
   * @param[in] element reference to the targeted element
   * @param[in] tips    name of the pad
  */
  void apply(Element& element, const std::string&tips);

  /**
   * @brief Pause the monitor
  */
  void pause();

  /**
   * @brief Resume the monitor
  */
  void resume();

  /**
   * @brief Add a new stream
  */
  void addStream(uint32_t source_id, const char* uri, const char* sensor_id, const char* sensor_name);

  /**
   * @brief Add a new stream
  */
  void removeStream(uint32_t source_id);

  void print(void* info);

 protected:
  unsigned int batch_size_;
  uint64_t     interval_sec_;
  void*        priv_;
  void*        fps_data_;
};

}