# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from lib.asset_manager import AssetManager
from lib.utils import create_jinja2_env, stack_tensors_in_dict, convert_list, get_logger
from omegaconf.errors import ConfigKeyError
from omegaconf import OmegaConf
import base64
from config import global_config
import json
import re
from pydantic import BaseModel
from typing import Dict, Any

jinja2_env = create_jinja2_env()

class ResponderBase:
    def __init__(self):
        self._action_map = {}
        self._inference = None
        self._request_templates = {}
        self._response_templates = {}
        self._asset_manager = AssetManager()
        self.logger = get_logger(__name__)

        # load the request and response templates
        try:
            input_config = OmegaConf.to_container(global_config.server.responders)
        except ConfigKeyError:
            raise ValueError("No responders found in the config")
        for rp, value in input_config.items():
            req_tpls = value.get("requests", {})
            for k, tpl in req_tpls.items():
                if rp not in self._request_templates:
                    self._request_templates[rp] = {}
                if isinstance(tpl, list):
                    self._request_templates[rp][k] = tpl
                else:
                    self._request_templates[rp][k] = base64.b64decode(tpl).decode()
            res_tpls = value.get("responses", {})
            for k, tpl in res_tpls.items():
                if rp not in self._response_templates:
                    self._response_templates[rp] = {}
                if isinstance(tpl, list):
                    self._response_templates[rp][k] = tpl
                else:
                    self._response_templates[rp][k] = base64.b64decode(tpl).decode()

    async def take_action(self, action_name:str, *args):
        action = self._action_map.get(action_name, None)
        if not action:
            raise ValueError(f"Unknown action: {action_name}")
        return await action(*args)

    def healthy_ready(self):
        return self._inference and self._inference.is_healthy()

    def process_request(self, responder: str, request: BaseModel) -> Dict[str, Any]:
        self.logger.debug(f"Processing request {request}")

        result = json.loads(request.model_dump_json())

        # find the request template for the endpoint
        templates = self._request_templates.get(responder, None)
        if not templates:
            # no template to be applied on the request
            return result

        request_class = next(iter(templates.keys()))
        template = templates[request_class]
        json_string = jinja2_env.from_string(template).render(request=result)
        try:
            result = json.loads(json_string)
        except Exception as e:
            self.logger.error(f"Error parsing request template: {e}")
            self.logger.error(f"Json string: {json_string}")
            raise e

        # template filters on each field
        for key, value in templates.items():
            if not key in result:
                continue
            if isinstance(value, list):
                # this is regex filter for fields
                text = result.pop(key)
                if not isinstance(text, str):
                    continue
                regx = value[0]
                keys = value[1:]
                matches = re.findall(regx, text)
                for match in matches:
                    if len(keys) != len(match):
                        continue
                    for x, y in zip(keys, match):
                        if y:
                            result.setdefault(x, []).append(y)
        return result

    def process_response(self, responder: str, request, response: Dict[str, Any]) -> str:
        self.logger.debug(f"Processing response {response}")
        # Load the response template for the endpoint
        templates = self._response_templates.get(responder, None)
        if not templates:
            # no template to be applied on the response
            return response

        response_class = next(iter(templates.keys()))
        template = templates[response_class]
        json_string = jinja2_env.from_string(template).render(request=request, response=response)
        return json_string

    def process_streamed_response(self, responder: str, request, response: Dict[str, Any]) -> str:
        self.logger.debug(f"Processing streamed response {response}")
        # Load the response template for the endpoint
        templates = self._response_templates.get(responder, None)
        if not templates:
            # no template to be applied on the response
            return response
        keys = list(templates.keys())
        if len(keys) < 2:
            return response
        template = templates[keys[1]]
        json_string = jinja2_env.from_string(template).render(request=request, response=response)
        return json_string
