/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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