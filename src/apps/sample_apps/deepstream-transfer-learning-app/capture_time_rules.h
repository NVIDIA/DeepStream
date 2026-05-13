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

#include <string>
#include <iostream>
#include <chrono>
#include <vector>

class CaptureTimeRules {
    typedef std::chrono::time_point<std::chrono::system_clock> t_time_pt;
    typedef std::chrono::duration<unsigned long long> t_duration;
public:
    /// Fills the time rules with the content of the file in path
    /// \param path containing the time rules
    /// \param default_second_interval When no rules are present for a
    /// certain time interval, this default duration is used.
    void init(const std::string &path, unsigned default_second_interval);

    /// Compute the correct time interval using the local computer time.
    /// \return the computed time interval to skip for current time
    t_duration getCurrentTimeInterval();

    /// \return True if the construction of the the object went well.
    /// False otherwise
    bool is_init_();

private:
    struct TimeRule {
        unsigned begin_time_hour;
        unsigned begin_time_minute;
        unsigned end_time_hour;
        unsigned end_time_minute;
        unsigned interval_between_frame_capture_seconds;
        bool end_time_is_next_day;
    };

    /// Test if a timestamp is between 2 other timestamps
    /// \param t structure containing the left and right bounding
    /// bounding timestamps
    /// \param now the current time
    /// \return True if the current time is between the 2 bounding
    /// timestamps
    static bool isInTimeRule(const TimeRule &t, const tm &now);

    enum ParseResult{
        PARSE_RESULT_OK,
        PARSE_RESULT_BAD_CHARS,
        PARSE_RESULT_OUT_OF_BOUND,
        PARSE_RESULT_EMPTY
    };

    static ParseResult stoi_err_handling(unsigned& dst, const std::string &src, unsigned max_bound);
    static bool parsing_contains_error(const std::vector<ParseResult>& parse_res_list,
            const std::vector<std::string>& str_list, const std::string& curr_line,
                             unsigned line_number);
    bool single_time_rule_parser(const std::string &path, const std::string &line,
                                 unsigned line_number);

    std::chrono::seconds default_duration_;
    t_time_pt end_of_current_time_interval_;
    t_duration current_time_interval_;
    std::vector<TimeRule> rules_;
    bool init_ = false;
};


