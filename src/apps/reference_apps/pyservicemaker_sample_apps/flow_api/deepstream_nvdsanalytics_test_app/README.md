<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

## Purpose

The sample app demonstrates how to simplify [deepstream_nvds_analytics_test_app](../../pipeline_api/deepstream_nvdsanalytics_test_app)
using Flow API.
Flow APIs effectively abstract away the underlying pipeline details, allowing 
developers to focus solely on the goals of their specific tasks in a pythonic style.

## Usage
```
$ python3 deepstream_nvdsanalytics.py <uri1> [uri2] ... [uriN]
```

For URI(s) with special characters like @,& etc, you need to pass the uri within quotes

```
$ python3 deepstream_nvdsanalytics.py 'rtsp://user@ip/cam/realmonitor?channel=1&subtype=0' 'rtsp://user@ip/cam/realmonitor?channel=1&subtype=0'
```