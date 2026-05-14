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
{{ license }}

from config import global_config
from lib.inference import *
from lib.utils import *
from lib.errors import ErrorFactory
from omegaconf import OmegaConf
import json
import os
from typing import List, Dict
import asyncio
import numpy as np
import torch
from pathlib import Path
import dataclasses

logger = get_logger(__name__)

{% for backend in backends %}
{{ backend }}
{% endfor %}

class GenericInference(InferenceBase):
    """The class that drives a generic inference flow"""

    def _create_backend(self, backend_spec: List[str], model_config: Dict, model_home: str):
        if backend_spec[0] == 'triton':
            backend_class = TritonBackend
        elif backend_spec[0] == 'deepstream':
            backend_class = DeepstreamBackend
        elif backend_spec[0] == 'polygraphy':
            backend_class = PolygraphBackend
        elif backend_spec[0] == 'tensorrtllm':
            backend_class = TensorRTLLMBackend
        elif backend_spec[0] == 'dummy':
            backend_class = DummyBackend
        elif backend_spec[0] == 'pytorch':
            backend_class = PytorchBackend
        elif backend_spec[0] == 'vllm':
            backend_class = VLLMBackend
        else:
            return None
        return backend_class(model_config, model_home)

    async def execute(self, request):
        """ execute a list of requests"""
        if self._async_dispatcher is None:
            self._async_dispatcher = AsyncDispatcher(self._collector, loop=asyncio.get_running_loop())

        logger.debug(f"Received request {request}")
        if not self._inject_tensors(request):
            error = ErrorFactory.create(
                "ERR_DF_006",
                caller=self,
                input_data={"request_keys": list(request.keys())},
                tensor_names=[n for i in self._inputs for n in i.in_names]
            )
            error.log(logger)
            InferenceBase._export_errors()
            yield error
            return

        async_queue = asyncio.Queue()
        self._async_dispatcher.append_async_queue(async_queue)
        # Wait for all the results from one inference request
        while not self._stop_event.is_set():
            try:
                logger.debug("Waiting for tensors from async queue")
                result = await async_queue.get()
                if isinstance(result, Stop):
                    logger.info(f"Got Stop: {result.reason}")
                    return
                if isinstance(result, Error):
                    InferenceBase._export_errors()
                else:
                    logger.debug(f"Got result: {result}")
                    # post-process the data
                    result = self._post_process(result)
                # Yield control to allow other requests to be processed
                await asyncio.sleep(0)
                yield result
            except Exception as e:
                error = ErrorFactory.from_exception(e, caller=self)
                error.log(logger)
                yield error

    def exec_sync(self, request):
        """ execute a list of requests synchronously"""
        from queue import Empty

        logger.debug(f"Received request {request}")
        if not self._inject_tensors(request):
            error = ErrorFactory.create(
                "ERR_DF_006",
                caller=self,
                input_data={"request_keys": list(request.keys())},
                tensor_names=[n for i in self._inputs for n in i.in_names]
            )
            error.log(logger)
            yield error
            return

        # Wait for all the results from one inference request
        while not self._stop_event.is_set():
            try:
                logger.debug("Waiting for tensors from collector")
                result = self._collector.collect()
                if isinstance(result, Stop):
                    logger.info(f"Got Stop: {result.reason}")
                    return
                if isinstance(result, Error):
                    InferenceBase._export_errors()
                else:
                    logger.debug(f"Got result: {result}")
                    # post-process the data
                    result = self._post_process(result)
                yield result
            except Empty:
                continue
            except Exception as e:
                error = ErrorFactory.from_exception(e, caller=self)
                error.log(logger)
                yield error

    def finalize(self):
        self._stop_event.set()
        super().finalize()

    def _inject_tensors(self, request: Dict) -> bool:
        """Inject tensors into the input flows"""
        injected = False
        matched = [[n for n in i.in_names if n in request and request[n]] for i in self._inputs]
        reshuffled = sorted(range(len(matched)), key=lambda x: len(matched[x]), reverse=True)
        for i in reshuffled:
            input_flow = self._inputs[i]
            tensors = {}
            # select and transform the tensors for the input
            for n in matched[i]:
                if n in request:
                    tensors[n] = np.array(request.pop(n))
            if tensors:
                logger.debug(f"Injecting tensors {tensors}")
                input_flow.put(tensors)
                input_flow.put(Stop(reason="end"))
                injected = True
        return injected

    def _post_process(self, data: Dict):
        processed = {k: v for k, v in data.items()}
        data_type_names = { i['name']: i['data_type'] for i in self._output_config}
        for processor in self._processors:
            if not all([i in data for i in processor.input]):
                logger.warning(f"Input settings invalid for the processor: {processor.name}")
                continue
            input = [processed.pop(i) for i in processor.input]
            logger.debug(f"Post-processor invoked with given input {input}")
            output = processor(*input)
            logger.debug(f"Post-processor generated output {output}")
            if len(output) != len(processor.output):
                logger.warning(f"Number of postprocessing output doesn't match the configuration, expecting {len(processor.output)}, while getting {len(output)}")
                continue
            # update as processed
            for key, value in zip(processor.output, output):
                processed[key] = value
                # correct data type
                if key not in data_type_names:
                    logger.warning(f"Invalid output parsed: {key}")
                    continue
                data_type =  data_type_names[key]
                if isinstance(value, np.ndarray):
                    data_type = np_datatype_mapping[data_type]
                    if value.dtype != data_type:
                        processed[key]  = value.astype(data_type)
                elif isinstance(value, torch.Tensor):
                    data_type = torch_datatype_mapping[data_type]
                    if value.dtype != data_type:
                        processed[key]  = value.to(data_type)
                else:
                    processed[key] = value
        # convert numpy and torch tensors to list for server to process
        def convert_to_list(v, is_string_type):
            """Convert numpy/torch tensor to list with proper string decoding"""
            if isinstance(v, np.ndarray):
                if is_string_type and (np.issubdtype(v.dtype, np.bytes_) or np.issubdtype(v.dtype, np.str_) or v.dtype == np.object_):
                    # Decode string arrays
                    flat_list = v.flatten().tolist()
                    decoded = [i.decode("utf-8", "ignore") if isinstance(i, bytes) else str(i) for i in flat_list]
                    return np.array(decoded).reshape(v.shape).tolist()
                else:
                    return v.tolist()
            elif isinstance(v, torch.Tensor):
                if v.dtype == torch.uint8 and is_string_type:
                    # Decode byte tensors as strings
                    flat_list = v.flatten().tolist()
                    decoded = [bytes([i]).decode("utf-8", "ignore") if isinstance(i, int) else str(i) for i in flat_list]
                    return decoded if v.dim() == 1 else np.array(decoded).reshape(v.shape).tolist()
                else:
                    return v.tolist()
            elif dataclasses.is_dataclass(v):
                return dataclasses.asdict(v)
            else:
                return v

        for key, value in processed.items():
            is_string_type = key in data_type_names and 'TYPE_STRING' in data_type_names[key]
            if isinstance(value, list):
                # this is a batch of data
                processed[key] = [convert_to_list(v, is_string_type) for v in value]
            else:
                processed[key] = convert_to_list(value, is_string_type)
        return processed
