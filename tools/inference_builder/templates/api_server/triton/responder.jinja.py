{#
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
#}
{{ license }}

import json
from .data_model import {{ triton.request_class }}, {{ triton.response_class }}, {{ triton.streaming_response_class }}
from config import global_config
from lib.utils import create_jinja2_env, convert_list, get_logger, to_unicode_string
from typing import Dict, Any, Optional, List, Union
import numpy as np
import torch
from typing import Dict, Any
from lib.responder import ResponderBase
from inferencemodeltoolkit.interfaces.fastapi.triton import (
    TritonInferenceHandler,
    TritonPydanticValidator,
)

{% for responder in responders %}
{% if responder.name == "infer" %}
infer_operation = "{{responder.operation}}"
{% endif %}
{% endfor %}

jinja2_env = create_jinja2_env()
class TritonInferenceHandlerBridge(TritonInferenceHandler):
    def __init__(self, responder):
        super().__init__(
             triton_url="grpc://localhost:8001",
             model_name="{{ service_name }}",
             stream_triton=True,
             stream_http=False)
        self._responder = responder
        self.logger = get_logger(__name__)

    def process_request(
            self,
            request: {{ triton.request_class }},
            headers: Dict[str, str]
    ):
        result = self._responder.process_request("infer", request)
        # Triton inference handler transforms the values to numpy arrays based on the model metadata
        return result

    def process_response(
        self,
        request: {{ triton.request_class }},
        response: Dict[str, np.ndarray],
        previous_responses: Optional[List[Dict[str, np.ndarray]]],
        headers
    ) -> Union[{{ triton.response_class }}, {{ triton.streaming_response_class }}]:
        self.logger.debug(f"Processing response {response}")
        type_map = { i.name: i.data_type for i in global_config.output}
        # Formulating streaming response
        if hasattr(request, 'stream') and request.stream:
            streamed = dict()
            for name, value in response.items():
                if value is None:
                    self.logger.error(f"{name} in response is None")
                    continue
                expected_type = type_map[name]
                if isinstance(value, np.ndarray) or isinstance(value, torch.Tensor):
                    l = value.tolist()
                    if expected_type == "TYPE_STRING":
                        l = convert_list(l, to_unicode_string)
                    streamed[name] = l
                else:
                    streamed[name] = value
            json_string = self._responder.process_streamed_response("infer", request, streamed)
            return {{ triton.streaming_response_class }}(**json.loads(json_string))

        # Formulating aggregated response from all the responses
        responses = previous_responses + [response]  if previous_responses else [response]
        acc = dict()
        # aggregate the data
        for response in responses:
            for name, value in response.items():
                if value is None:
                    logger.error(f"{name} in response is None")
                    continue
                if isinstance(value, np.ndarray):
                    if name in acc:
                        acc[name] = np.append(acc[name], value)
                    else:
                        acc[name] = value
                else:
                    acc[name] = acc[name] + value if name in acc else value
        # transform numpy ndarray to universal value types
        for name in acc:
            expected_type = type_map[name]
            if isinstance(acc[name], np.ndarray):
                l = acc[name].tolist()
                if expected_type == "TYPE_STRING":
                    acc[name] = convert_list(l, to_unicode_string)
                elif len(acc[name].shape) == 1 and len(l) == 1:
                    acc[name] = l[0]
                else:
                    acc[name] = l
        json_string = self._responder.process_response("infer", request, acc)
        return {{ triton.response_class }}(**json.loads(json_string))

class TritonResponder(ResponderBase):
    def __init__(self, operations, app):
        super().__init__()
        self._inference = TritonInferenceHandlerBridge(self)
        validator = TritonPydanticValidator()
        _, additional_responses = validator._validate_pydantic_hints(self._inference)
        infer_path = operations[infer_operation]
        # override the infer operation
        app.router.routes = [
            route for route in app.router.routes if not (getattr(route, "path", None) == infer_path)
        ]
        app.post(
            infer_path,
            response_model_exclude_none=True,
            responses=additional_responses,
        )(self._inference._infer)
        # initialize the action map for the other operations
        {% for responder in responders %}
        {% if responder.name != "infer"  %}
        self._action_map["{{ responder.operation }}"] = self.{{ responder.name }}
        {% endif %}
        {% endfor %}

{% for responder in responders %}
{% if responder.name != "infer" %}
{{ responder.implementation }}
{% endif %}
{% endfor %}
