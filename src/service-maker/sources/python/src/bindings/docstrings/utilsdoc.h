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

#pragma once

namespace pydeepstreamdoc
{
    namespace utils
    {
        namespace PerfMonitorDoc
        {
            constexpr const char* descr = R"pydeepstream(Class for monitoring the framerate.)pydeepstream";
            constexpr const char* init = R"pydeepstream(
                Initialize the performance monitor.

                :arg batch_size: *int*, Batch size of the pipeline
                :arg interval: *int*, Interval to monitor performance.
                :arg source_type: *str*, Type name of the source.
                :arg show_name: *bool*, Whether to show the stream name in perf log.)pydeepstream";
            constexpr const char* apply = R"pydeepstream(
                Apply the performance monitor on a specific node(pad) within a pipeline.

                :arg node: *:class:`Node`*, Node to apply the monitor.
                :arg tips: *str*, Name of the pad.)pydeepstream";
            constexpr const char* pause = R"pydeepstream(Pause the performance monitor.)pydeepstream";
            constexpr const char* resume = R"pydeepstream(Resume the performance monitor.)pydeepstream";
            constexpr const char* add_stream = R"pydeepstream(
                Add a new stream to the monitor.

                :arg source_id: *int*, Source ID of the stream.
                :arg uri: *str*, URI of the stream.
                :arg sensor_id: *str*, Sensor ID.
                :arg sensor_name: *str*, Sensor name.)pydeepstream";
            constexpr const char* remove_stream = R"pydeepstream(Remove a stream from the monitor given the source ID.)pydeepstream";
        }

        namespace EngineFileMonitorDoc
        {
            constexpr const char* descr = R"pydeepstream(Class for monitoring the model engine file.)pydeepstream";
            constexpr const char* init = R"pydeepstream(Initialize the engine file monitor from the inference :class:`Node` and model engine file path.)pydeepstream";
            constexpr const char* start = R"pydeepstream(Start the engine file monitor.)pydeepstream";
            constexpr const char* stop = R"pydeepstream(Stop the engine file monitor.)pydeepstream";
            constexpr const char* started = R"pydeepstream(Whether the engine file monitor is started.)pydeepstream";
        }

        namespace AudioStreamInfoDoc
        {
            constexpr const char* descr = R"pydeepstream(
                Class for acquiring audio stream information.

                :ivar codec: *str*, Audio codec information.
                :ivar channels: *int*, Number of channels.
                :ivar sampling_rate: *int*, Sampling rate of the audio stream.)pydeepstream";
        }

        namespace VideoStreamInfoDoc
        {
            constexpr const char* descr = R"pydeepstream(
                Class for acquiring video stream information.

                :ivar codec: *str*, Video codec information.
                :ivar width: *int*, Width of the video stream.
                :ivar height: *int*, Height of the video stream.
                :ivar framerate: *tuple*, Framerate of the video stream in the form of (numerator, denominator).)pydeepstream";
        }

        namespace MediaInfoDoc
        {
            constexpr const char* descr = R"pydeepstream(
                Class for acquiring media information.

                :ivar live: *bool*, Indicator of live media.
                :ivar duration: *int*, Duration of the media in nanoseconds.
                :ivar streams: *list*, List of media streams.)pydeepstream";
            constexpr const char* discover = R"pydeepstream(
                Discover the media information from the given URI.

                :arg uri: *str*, URI of the media.
                :return: *:class:`MediaInfo`*, Media information.)pydeepstream";
        }
    }
}
