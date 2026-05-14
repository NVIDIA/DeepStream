{#
 SPDX-FileCopyrightText: Copyright (c) <year> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#}

    async def {{ name }}(self, request, body, **kwargs):
        stream_info = self.process_request("{{ name }}", body)
        if "url" not in stream_info or not stream_info["url"]:
            return 400, "No url provided"
        try:
            asset = self._asset_manager.add_live_stream(
                stream_info["url"],
                stream_info["description"],
                stream_info["username"],
                stream_info["password"])
        except Exception as e:
            return 500, str(e)
        response = self.process_response("{{ name }}", request, asset.to_dict())
        return 200, response

