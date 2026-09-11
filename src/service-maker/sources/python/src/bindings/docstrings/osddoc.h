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

#pragma once

namespace pydeepstreamdoc
{
    namespace osd
    {
        namespace FontFamilyDoc
        {
            constexpr const char* descr = R"pydeepstream(Font family.)pydeepstream";
            constexpr const char* serif = R"pydeepstream(Serif.)pydeepstream";
        }

        namespace ArrowHeadDoc
        {
            constexpr const char* descr = R"pydeepstream(Arrow head positions.)pydeepstream";
            constexpr const char* start = R"pydeepstream(Arrow head only at start.)pydeepstream";
            constexpr const char* end = R"pydeepstream(Arrow head only at end.)pydeepstream";
            constexpr const char* both = R"pydeepstream(Arrow head at both sides.)pydeepstream";
        }

        namespace FontDoc
        {
            constexpr const char* descr = R"pydeepstream(
                Holds the font parameters of the text to be overlaid.

                :ivar name: *str*, Name of the font.
                :ivar size: *int*, Size of the font.
                :ivar color: :class:`Color`, Color parameters of the font.)pydeepstream";
        }

        namespace ColorDoc
        {
            constexpr const char* descr = R"pydeepstream(
                Holds the color parameters of the text to be overlaid.

                :ivar r: *float*, Red component of the color. Value must be in range [0, 1].
                :ivar g: *float*, Green component of the color. Value must be in range [0, 1].
                :ivar b: *float*, Blue component of the color. Value must be in range [0, 1].
                :ivar a: *float*, Alpha component of the color. Value must be in range [0, 1].)pydeepstream";
        }

        namespace TextDoc
        {
            constexpr const char* descr = R"pydeepstream(
                Holds the text parameters to be overlaid.

                :ivar display_text: *str*, Text to be overlaid.
                :ivar x_offset: *int*, Horizentol offset w.r.t top left pixel of the frame.
                :ivar y_offset: *int*, Vertical offset w.r.t top left pixel of the frame.
                :ivar font: :class:`Font`, Font parameters of the text.
                :ivar set_bg_color: *bool*, Whether the text has background color.
                :ivar bg_color: :class:`Color`, Color parameters of the text background, if specified.)pydeepstream";
        }

        namespace TextParamsDoc
        {
            constexpr const char* descr = R"pydeepstream(
                Holds the text parameters to be overlaid.

                :ivar display_text: *str*, Text to be overlaid.
                :ivar x_offset: *int*, Horizentol offset w.r.t top left pixel of the frame.
                :ivar y_offset: *int*, Vertical offset w.r.t top left pixel of the frame.
                :ivar font_params: :class:`Font`, Font parameters of the text.
                :ivar set_bg_clr: *bool*, Whether the text has background color.
                :ivar text_bg_clr: :class:`Color`, Color parameters of the text background, if specified.)pydeepstream";
        }

        namespace RectDoc
        {
            constexpr const char* descr = R"pydeepstream(
                Holds the rectangle parameters of the box to be overlaid.

                :ivar left: *int*, Left coordinate of the rectangle in pixels.
                :ivar top: *int*, Top coordinate of the rectangle in pixels.
                :ivar width: *int*, Width of the rectangle in pixels.
                :ivar height: *int*, Height of the rectangle in pixels.
                :ivar border_width: *int*, Width of the border of the rectangle in pixels.
                :ivar border_color: :class:`Color`, Color parameters of the border of the rectangle.
                :ivar has_bg_color: *bool*, Whether the rectangle has background color.
                :ivar bg_color: :class:`Color`, Color parameters of the rectangle background, if specified.)pydeepstream";
        }

        namespace MaskDoc
        {
            constexpr const char* descr = R"pydeepstream(
                Holds the parameters of the instance segment to be overlaid.

                :ivar threshold: *float*, Threshold for binarization.
                :ivar mask_array: *list of float*, Mask buffer of instance.)pydeepstream";
        }

        namespace LineDoc
        {
            constexpr const char* descr = R"pydeepstream(
                Holds the parameters of the line to be overlaid.

                :ivar x1: *int*, X coordinate of the start point of the line in pixels.
                :ivar y1: *int*, Y coordinate of the start point of the line in pixels.
                :ivar x2: *int*, X coordinate of the end point of the line in pixels.
                :ivar y2: *int*, Y coordinate of the end point of the line in pixels.
                :ivar width: *int*, Width of the line in pixels.
                :ivar color: :class:`Color`, Color parameters of the line.)pydeepstream";
        }

        namespace ArrowDoc
        {
            constexpr const char* descr = R"pydeepstream(
                Holds the parameters of the arrow to be overlaid.

                :ivar x1: *int*, X coordinate of the start point of the arrow in pixels.
                :ivar y1: *int*, Y coordinate of the start point of the arrow in pixels.
                :ivar x2: *int*, X coordinate of the end point of the arrow in pixels.
                :ivar y2: *int*, Y coordinate of the end point of the arrow in pixels.
                :ivar width: *int*, Width of the arrow shaft in pixels.
                :ivar color: :class:`Color`, Color parameters of the arrow.
                :ivar head: :class:`ArrowHead`, Position of the arrow head.)pydeepstream";
        }

        namespace CircleDoc
        {
            constexpr const char* descr = R"pydeepstream(
                Holds the parameters of the circle to be overlaid.

                :ivar xc: *int*, X coordinate of the center of the circle in pixels.
                :ivar yc: *int*, Y coordinate of the center of the circle in pixels.
                :ivar radius: *int*, Radius of the circle in pixels.
                :ivar color: :class:`Color`, Color parameters of the circle.
                :ivar has_bg_color: *bool*, Whether the circle has background color.
                :ivar bg_color: :class:`Color`, Color parameters of the circle background, if specified.
                :ivar width: *int*, Width of the circle in pixels.)pydeepstream";
        }

        namespace EventHandlerDoc
        {
            constexpr const char* descr = R"pydeepstream(
                Handler for capturing mouse events from the display. Once the event capturing is started,
                users can toggle the OSD text using the mouse.

                :ivar started: *bool*, Whether the event handler has been started.)pydeepstream";
            constexpr const char* init = R"pydeepstream(Initialize the event handler from tiler, osd, and renderer :class:`Node`s.)pydeepstream";
            constexpr const char* start = R"pydeepstream(Start the event handler.)pydeepstream";
            constexpr const char* stop = R"pydeepstream(Stop the event handler.)pydeepstream";
        }
    }
}