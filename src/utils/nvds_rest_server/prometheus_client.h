/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "prometheus/registry.h"
#include <memory>
#include <atomic>
#include <string>
#include <mutex>

#define GET_PROMETHEUS PrometheusClient::getInstance

class PrometheusClient
{
public:
    PrometheusClient();
    static PrometheusClient* getInstance();
    std::shared_ptr<prometheus::Registry> getMetricsRegistry() const { return m_metrics_registry; }
    std::shared_ptr<prometheus::Registry> getStreamInfoRegistry() const { return m_stream_info_registry; }
    std::shared_ptr<prometheus::Registry> getReadyRegistry() const { return m_ready_registry; }
    std::shared_ptr<prometheus::Registry> getLiveRegistry() const { return m_live_registry; }
    std::shared_ptr<prometheus::Registry> getStartupRegistry() const { return m_startup_registry; }

    // Methods for stream statistics from metrics-info JSON
    void updateStreamStats(double fps, uint64_t frame_number, double latency_ms, 
                          const std::string& sensor_id, const std::string& sensor_name, int source_id);
    void updateStreamCount(int count);
    void clearStreamMetrics();

    // Methods for system statistics from metrics-info JSON
    void updateSystemGpuMemory(double gpu_gb);
    void updateSystemRamMemory(double ram_gb);
    void updateSystemCpuUtilization(double cpu_util);
    void updateSystemGpuUtilization(double gpu_util);

    // Methods for health statistics from health-info JSON
    void updateDsLiveness(bool is_alive);
    void updateDsReady(bool is_ready);
    void updateDsStartup(bool is_startup);

    void updateStreamInfoCount(int count);

private:
    static PrometheusClient* _instance;
    static std::once_flag _init_flag;

    std::shared_ptr<prometheus::Registry>                   m_ready_registry = nullptr;
    std::shared_ptr<prometheus::Registry>                   m_live_registry = nullptr;
    std::shared_ptr<prometheus::Registry>                   m_startup_registry = nullptr;
    std::shared_ptr<prometheus::Registry>                   m_metrics_registry = nullptr;
    std::shared_ptr<prometheus::Registry>                   m_stream_info_registry = nullptr;
    prometheus::Family<prometheus::Gauge>*                  m_fps_gauge_family = nullptr;
    prometheus::Family<prometheus::Gauge>*                  m_latency_gauge_family = nullptr;
    prometheus::Family<prometheus::Gauge>*                  m_frame_number_gauge_family = nullptr;
    prometheus::Family<prometheus::Gauge>*                  m_memory_gauge_family = nullptr;
    prometheus::Family<prometheus::Gauge>*                  m_utilization_gauge_family = nullptr;
    prometheus::Family<prometheus::Gauge>*                  m_stream_count_gauge = nullptr;
    prometheus::Family<prometheus::Gauge>*                  m_ready_gauge_family = nullptr;
    prometheus::Family<prometheus::Gauge>*                  m_live_gauge_family = nullptr;
    prometheus::Family<prometheus::Gauge>*                  m_startup_gauge_family = nullptr;
    prometheus::Family<prometheus::Gauge>*                  m_stream_info_count_gauge = nullptr;
    void BuildGaugeFamily();

};