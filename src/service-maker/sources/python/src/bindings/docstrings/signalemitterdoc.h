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
    namespace signalemitter
    {
        namespace EmitterDoc
        {
            constexpr const char* descr = R"pydeepstream(
                Signal emitter object.
                Can only be attached to elements that support certain actions.
                Can be attached to multiple elements on multiple actions.)pydeepstream";
            constexpr const char* attach = R"pydeepstream(
                Attach the signal emitter to an object on specified action.

                :arg action: name of the action
                :arg object: object :class:`Node` to attach the emitter to.)pydeepstream";
        }

        namespace SourceManagerDoc
        {
            constexpr const char* descr = R"pydeepstream(:class:`SignalEmitter` type that can be used to add sources to dynamicsrcbin on fly.)pydeepstream";
            constexpr const char* add_source = R"pydeepstream(Add a source to the manager.

                :arg source_name: name of the source to be added
                :return: a unique ID for the source, -1 if failed)pydeepstream";
            constexpr const char* remove_source = R"pydeepstream(Remove a source from the manager.

                :arg source_id: ID of the source to be removed)pydeepstream";
            constexpr const char* terminate = R"pydeepstream(Terminate the manager.

                This will send an END_OF_STREAM signal from the manager and stop the manager.)pydeepstream";
        }
    }
}