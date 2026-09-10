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
    namespace pipeline
    {
        namespace PipelineStateDoc
        {
            constexpr const char* descr = R"pydeepstream(State of the pipeline.)pydeepstream";
        }

        namespace PipelineMessageDoc
        {
            constexpr const char* descr = R"pydeepstream(Base class for pipeline message.)pydeepstream";
        }

        namespace EOSMessageDoc
        {
            constexpr const char* descr = R"pydeepstream(Pipeline message on EOS.)pydeepstream";
        }

        namespace StateTransitionMessageDoc
        {
            constexpr const char* descr = R"pydeepstream(
                Pipeline message on state transition

                :ivar old_state: *PipelineState*, Old state of the pipeline.
                :ivar new_state: *PipelineState*, New state of the pipeline.
                :ivar origin: *str*, Name of the origin of the state transition.)pydeepstream";
        }

        namespace DynamicSourceMessageDoc
        {
            constexpr const char* descr = R"pydeepstream(
                Pipeline message on dynamic source.

                :ivar source_added: *bool*, Whether a source has been added; false indicates source removal or source completion.
                :ivar source_id: *int*, ID of the source.
                :ivar sensor_id: *str*, ID of the sensor.
                :ivar sensor_name: *str*, Name of the sensor.
                :ivar uri: *str*, URI of the source.)pydeepstream";
        }

        namespace RecordingInfoDoc
        {
            constexpr const char* descr = R"pydeepstream(
                Information about a recording session.

                :ivar session_id: *int*, Session id of the recording.
                :ivar file_name: *str*, Name of the recording file.
                :ivar file_directory: *str*, Directory of the recording file.
                :ivar duration: *int*, Duration of the recording.
                :ivar container_type: *str*, Container type of the recording.
                :ivar width: *int*, Width of the recording.
                :ivar height: *int*, Height of the recording.
                :ivar contains_video: *bool*, Whether the recording contains video.
                :ivar contains_audio: *bool*, Whether the recording contains audio.
                :ivar channels: *int*, Channels of the recording.
                :ivar sampling_rate: *int*, Sampling rate of the recording.)pydeepstream";
        }

        namespace PipelineDoc
        {
            constexpr const char* descr = R"pydeepstream(Media processing pipeline.)pydeepstream";
            constexpr const char* init = R"pydeepstream(Initialize the pipeline with a name.)pydeepstream";
            constexpr const char* init_with_config = R"pydeepstream(Initialize the pipeline with a name and config file path.)pydeepstream";
            constexpr const char* add = R"pydeepstream(
                Add an element to the pipeline.

                :arg type_name: *str*, Type of the element.
                :arg name: *str*, Name of the element.)pydeepstream";
            constexpr const char* attach = R"pydeepstream(
                Attach an object to an element in the pipeline. Object must be a probe or a signal handler.

                :arg element_name: *str*, Name of the element to which the object attaches.
                :arg plugin_name: *str*, Name of the plugin to create the custom object from.
                :arg object: *str*, Name of the custom object.
                :arg tip: *str*, Extra information. Pad name for buffer probes, signal name for signal handlers.)pydeepstream";
            constexpr const char* getitem = R"pydeepstream(Retrieve a :class:`Node` from the pipeline by name.)pydeepstream";
            constexpr const char* start = R"pydeepstream(Start the pipeline.)pydeepstream";
            constexpr const char* start_with_callback = R"pydeepstream(Start the pipeline with a callback to capture messages.)pydeepstream";
            constexpr const char* prepare = R"pydeepstream(Initialize the pipeline.)pydeepstream";
            constexpr const char* prepare_with_callback = R"pydeepstream(Initialize the pipeline with a callback to capture messages.)pydeepstream";
            constexpr const char* activate = R"pydeepstream(Start the pipeline after it is already initialized.)pydeepstream";
            constexpr const char* wait = R"pydeepstream(Wait for the pipeline to finish.)pydeepstream";
            constexpr const char* stop = R"pydeepstream(Stop the pipeline.)pydeepstream";
            constexpr const char* start_rtsp_server = R"pydeepstream(Start the RTSP server.)pydeepstream";
            constexpr const char* seek = R"pydeepstream(Seek to a specified timestamp for processing data within the pipeline.)pydeepstream";
            constexpr const char* start_recording = R"pydeepstream(
                Start recording from a specified source.

                :arg source_name: *str*, Name of the source to record.
                :arg start_time: *int*, Start time of the recording.
                :arg duration: *int*, Duration of the recording.)pydeepstream";
            constexpr const char* stop_recording = R"pydeepstream(
                Stop recording from a specified source.

                :arg source_name: *str*, Name of the source to record.)pydeepstream";
            constexpr const char* stop_recording_by_session_id = R"pydeepstream(
                Stop recording from a specified session id.

                :arg session_id: *int*, Session id to stop recording.)pydeepstream";
        }
    }
}
