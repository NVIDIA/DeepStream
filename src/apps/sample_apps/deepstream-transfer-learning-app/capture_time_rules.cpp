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

#include <fstream>
#include <sstream>
#include <iomanip>
#include "capture_time_rules.h"

constexpr unsigned seconds_in_day = 86400;


static std::vector<std::string> split_string(const std::string &str, char split_char) {
    std::vector<std::string> elems;
    std::stringstream ss(str);
    std::string item;
    while (std::getline(ss, item, split_char))
        elems.push_back(item);
    return elems;
}

CaptureTimeRules::ParseResult
CaptureTimeRules::stoi_err_handling(unsigned &dst, const std::string &src, unsigned max_bound) {
    if (src.empty())
        return PARSE_RESULT_EMPTY;
    unsigned size_number = 0;
    unsigned zeros = 0;
    unsigned idx = 0;
    while (src[idx] == ' ')
        idx++;
    while (src[idx] == '0') {
        zeros++;
        idx++;
    }
    while (src[idx] >= '0' && src[idx] <= '9') {
        size_number++;
        idx++;
    }
    while (src[idx] == ' ')
        idx++;
    if (src.size() != idx)
        return PARSE_RESULT_BAD_CHARS;
    if (size_number > 3)
        return PARSE_RESULT_OUT_OF_BOUND;

    if (size_number == 0 && zeros == 0)
        return PARSE_RESULT_BAD_CHARS;
    else
        dst = std::stoi(src);
    if (dst >= max_bound)
        return PARSE_RESULT_OUT_OF_BOUND;
    return PARSE_RESULT_OK;
}

bool CaptureTimeRules::parsing_contains_error(const std::vector<ParseResult> &parse_res_list,
                                              const std::vector<std::string> &str_list,
                                              const std::string &curr_line,
                                              unsigned int line_number) {
    static const std::vector<unsigned> max_time_list = {24, 60, 24, 60, 24, 60, 60};
    bool contains_error = false;
    std::stringstream ss;
    int shift = 0;

    for (unsigned i = 0; i < 7; ++i) {
        unsigned tab_nb = 0;
        for (const auto& elm: str_list[i])
            if(elm == '\t')
                tab_nb++;
        if (parse_res_list[i] == PARSE_RESULT_OK) {
            shift += str_list[i].size() + 1;
            continue;
        }
        contains_error = true;
        ss << curr_line << " <-- line " << line_number << "\n";
        for (int j = 0; j < shift - 1; ++j)
            ss << ' ';
        if (parse_res_list[i] == PARSE_RESULT_EMPTY)
            ss << '^';
        else if (shift > 0)
            ss << ' ';
        for (unsigned k = 0; k < str_list[i].size(); ++k) {
            ss << '^';
        }
        ss << '\n';
        for (int j = 0; j < shift; ++j)
            ss << ' ';
        switch (parse_res_list[i]) {
            case PARSE_RESULT_OK: // should not happen
                break;
            case PARSE_RESULT_BAD_CHARS:
                ss << "replace this part by ";
                break;
            case PARSE_RESULT_OUT_OF_BOUND:
                ss << "replace this number by ";
                break;
            case PARSE_RESULT_EMPTY:
                ss << "after that character add ";
                break;
        }
        ss << "a number bounded between 0 (included) and "
           << max_time_list[i] << " (excluded).\n\n";
        shift += str_list[i].size() + 1;
    }
    std::cerr << ss.str();
    return contains_error;
}


bool CaptureTimeRules::single_time_rule_parser(const std::string &path, const std::string &line,
                                               unsigned line_number) {
    if (line.empty()) {
        return true;
    }
    bool parse_error_line = false;
    std::vector<std::string> time1;
    std::vector<std::string> time2;
    std::vector<std::string> time_to_skip;
    do {
        auto split_line = split_string(line, ',');
        if (split_line.size() != 3) {
            parse_error_line = true;
            break;
        }
        time1 = split_string(split_line[0], ':');
        if (time1.size() != 2) {
            parse_error_line = true;
            break;
        }
        time2 = split_string(split_line[1], ':');
        if (time2.size() != 2) {
            parse_error_line = true;
            break;
        }
        time_to_skip = split_string(split_line[2], ':');
        if (time_to_skip.size() != 3) {
            parse_error_line = true;
            break;
        }
    } while (false);

    if (parse_error_line) {
        std::cerr << "Parsing error " << path << ":" << (line_number) << "\n"
                  << line << "\n"
                  << "Each line from the second one should have the following format:\n"
                  << "<hours>:<minutes>,<hours>:<minutes>,<hours>:<minutes>:<seconds>\n";
        return false;
    }
    unsigned tts_h;
    unsigned tts_m;
    unsigned tts_s;
    TimeRule t;
    std::vector<ParseResult> parse_res_list;
    parse_res_list.push_back(stoi_err_handling(t.begin_time_hour, time1[0], 24));
    parse_res_list.push_back(stoi_err_handling(t.begin_time_minute, time1[1], 60));
    parse_res_list.push_back(stoi_err_handling(t.end_time_hour, time2[0], 24));
    parse_res_list.push_back(stoi_err_handling(t.end_time_minute, time2[1], 60));
    parse_res_list.push_back(stoi_err_handling(tts_h, time_to_skip[0], 24));
    parse_res_list.push_back(stoi_err_handling(tts_m, time_to_skip[1], 60));
    parse_res_list.push_back(stoi_err_handling(tts_s, time_to_skip[2], 60));

    std::vector<std::string> elm_list = {time1[0], time1[1], time2[0], time2[1],
                                         time_to_skip[0], time_to_skip[1], time_to_skip[2]};

    if (parsing_contains_error(parse_res_list, elm_list, line, line_number)) {
        return false;
    }
    t.end_time_is_next_day = (t.end_time_hour < t.begin_time_hour
                              || (t.end_time_hour == t.begin_time_hour
                                  && t.end_time_minute <= t.begin_time_minute));

    t.interval_between_frame_capture_seconds = ((tts_h * 60) + tts_m) * 60 + tts_s;
    rules_.push_back(t);
    return true;
}

void CaptureTimeRules::init(const std::string &path, unsigned int default_second_interval) {
    default_duration_ = std::chrono::seconds(default_second_interval);
    end_of_current_time_interval_ = std::chrono::system_clock::now() - default_duration_;
    std::ifstream file(path);
    if (!file.good()) {
        std::cerr << "Could not open " << path << ".\n";
        return;
    }
    std::string line;
    // discarding first line
    std::getline(file, line);
    bool no_error = true;
    unsigned line_number = 2;
    while (std::getline(file, line)) {
        no_error &= single_time_rule_parser(path, line, line_number);
        line_number++;
    }
    init_ = no_error;
}


CaptureTimeRules::t_duration CaptureTimeRules::getCurrentTimeInterval() {
    auto now = std::chrono::system_clock::now();
    if (now < end_of_current_time_interval_) {
        return current_time_interval_;
    }

    time_t tt = std::chrono::system_clock::to_time_t(now);
    tm local_tm = *localtime(&tt);
    for (const auto &elm: rules_) {
        if (isInTimeRule(elm, local_tm)) {
            local_tm.tm_hour = elm.end_time_hour;
            local_tm.tm_min = elm.end_time_minute;
            auto tp = std::chrono::system_clock::from_time_t(std::mktime(&local_tm));
            if (elm.end_time_is_next_day)
                tp += std::chrono::hours(24);
            end_of_current_time_interval_ = tp;
            current_time_interval_ = std::chrono::seconds(elm.interval_between_frame_capture_seconds);
            return current_time_interval_;
        }
    }
    current_time_interval_ = default_duration_;
    tm tmp_tm = *localtime(&tt);
    t_duration time_diff = std::chrono::seconds(seconds_in_day);
    for (const auto &elm: rules_) {
        tmp_tm.tm_hour = elm.begin_time_hour;
        tmp_tm.tm_min = elm.begin_time_minute;
        auto tp = std::chrono::system_clock::from_time_t(std::mktime(&tmp_tm));
        if (now > tp)
            continue;
        t_duration diff = std::chrono::duration_cast<std::chrono::seconds>(tp - now);
        if (diff < time_diff)
            time_diff = diff;
    }
    if (time_diff < std::chrono::seconds(seconds_in_day - 1)) {
        end_of_current_time_interval_ = now + time_diff;
    }
    return current_time_interval_;
}

bool CaptureTimeRules::isInTimeRule(const CaptureTimeRules::TimeRule &t, const tm &now) {
    unsigned n_h = static_cast<unsigned>(now.tm_hour);
    unsigned n_m = static_cast<unsigned>(now.tm_min);

    bool is_after_begin = n_h > t.begin_time_hour
                          || (n_h == t.begin_time_hour
                              && n_m >= t.begin_time_minute);
    bool is_before_end = n_h < t.end_time_hour
                         || (n_h == t.end_time_hour
                             && n_m < t.end_time_minute);
    if (!t.end_time_is_next_day)
        return is_after_begin && is_before_end;

    bool is_after_begin_plus_24 = (n_h + 24) > t.begin_time_hour
                                  || ((n_h + 24) == t.begin_time_hour
                                      && n_m >= t.begin_time_minute);
    bool is_before_end_plus_24 = n_h < (t.end_time_hour + 24)
                                 || (n_h == (t.end_time_hour + 24)
                                     && n_m < t.end_time_minute);

    bool next_day_condition = (is_after_begin && is_before_end_plus_24)
                              || (is_after_begin_plus_24 && is_before_end);

    return next_day_condition;
}

bool CaptureTimeRules::is_init_() {
    return init_;
}


