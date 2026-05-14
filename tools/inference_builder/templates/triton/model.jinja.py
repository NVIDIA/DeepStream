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

from config import global_config
import triton_python_backend_utils as pb_utils
from lib.inference import *
from lib.utils import *
from omegaconf import OmegaConf
import json
import os
from typing import List, Dict
import asyncio
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from pathlib import Path

logger = get_logger(__name__)

CHECKPOINTS_DIR = os.getenv("CHECKPOINTS_DIR", Path(__file__).resolve().parent.parent.parent)

triton_input_type_map = {
    "TYPE_CUSTOM_BINARY_BASE64": "TYPE_STRING",
    "TYPE_CUSTOM_IMAGE_BASE64": "TYPE_STRING"
}

{% for backend in backends %}
{{ backend }}
{% endfor %}

{% if top_level %}

class TritonPythonModel(InferenceBase):
    """The top level python model that drives the inference flow"""

    @staticmethod
    def auto_complete_config(auto_complete_model_config):
        # create a minimum config
        auto_complete_model_config.set_max_batch_size(1)
        auto_complete_model_config.set_dynamic_batching()
        auto_complete_model_config.set_model_transaction_policy({"decoupled": True})
        for input in global_config.input:
            input_config = OmegaConf.to_container(input)
            data_type = input_config["data_type"]
            if data_type.startswith("TYPE_CUSTOM_"):
                data_type = triton_input_type_map[data_type] if data_type in triton_input_type_map else None
                if data_type is None:
                    raise Exception(f"Unsupported input type: {data_type}")
                input_config["data_type"] = data_type
            auto_complete_model_config.add_input(input_config)
        for output in global_config.output:
            output_config = OmegaConf.to_container(output)
            auto_complete_model_config.add_output(output_config)
        logger.info(f"Model configuration completed as {auto_complete_model_config.as_dict()}")
        return auto_complete_model_config

    def _create_backend(self, backend_spec: List[str], model_config: Dict, model_home: str):
        if backend_spec[0] == 'triton':
            return TritonBackend(model_config, model_home)
        return None

    async def execute(self, requests):
        """ execute a list of requests"""
        async def async_put(queue, item):
            await queue.put(item)
        def thread_to_async_bridge(thread_queue, async_queue, loop):
            while True:
                try:
                    item = thread_queue.get()
                    asyncio.run_coroutine_threadsafe(async_put(async_queue, item), loop)
                    if isinstance(item, Stop) or isinstance(item, Error):
                        break
                except Empty:
                    continue


        logger.info(f"Received {len(requests)} request(s)")
        for request in requests:
            response_sender = request.get_response_sender()
            if request.is_cancelled():
                response_sender.send(pb_utils.InferenceResponse(
                    error=pb_utils.TritonError("request is cancelled", pb_utils.TritonError.CANCELLED)),
                    flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)
                continue
            for input in self._inputs:
                # select the tensors for the input
                tensors = {n: pb_utils.get_input_tensor_by_name(request, n) for n in input.in_names}
                # the tensors need to be transformed to generic type
                for name in tensors:
                    tensor = tensors[name]
                    config = next((c for c in self._input_config if c['name'] == name), None)
                    if config is None:
                        logger.warning(f"Invalid input parsed: {name}")
                        continue
                    dims = config['dims']
                    if pb_utils.Tensor.is_cpu(tensor):
                        tensor = tensor.as_numpy()
                        # auto reshape
                        if len(tensor.shape) == (len(dims)+1):
                            tensor = np.squeeze(tensor, 0)
                    else:
                        tensor = torch.utils.dlpack.from_dlpack(tensor.to_dlpack())
                        # auto reshape
                        if len(tensor.shape) == (len(dims)+1):
                            tensor = torch.squeeze(tensor, 0)
                    tensors[name] = tensor
                if tensors:
                    logger.debug(f"Injecting tensors {tensors}")
                    input.put(tensors)
                    input.put(Stop(reason="end"))
            # fetch result
            loop = asyncio.get_event_loop()
            for a_output, output in zip(self._async_outputs, self._outputs):
                logger.error(f"Submitting {output._timeout} to async executor")
                self._async_executor.submit(thread_to_async_bridge, output, a_output, loop)
            stop = False
            while not stop:
                try:
                    logger.debug("Waiting for tensors from async queue")
                    response_data = dict()
                    results = await asyncio.gather(*(ao.get() for ao in self._async_outputs))
                    for data in results:
                        logger.debug(f"Got output data: {data}")
                        if isinstance(data, Error):
                            error = pb_utils.TritonError(f"steam llm_response, error received: {data.message}")
                            response_sender.send(
                                pb_utils.InferenceResponse(error=error),
                                flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL
                            )
                            return
                        elif isinstance(data, Stop):
                            stop = True
                            continue
                        # collect the output
                        for k, v in data.items():
                            response_data[k] = v
                    response_data = self._post_process(response_data)
                    # response with partial data
                    response_sender.send(pb_utils.InferenceResponse(
                        output_tensors=[
                            pb_utils.Tensor(name, tensor) for name, tensor in response_data.items()
                        ]
                    ))
                except Exception as e:
                    logger.exception(e)

            response_sender.send(flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)
            logger.debug(f"Finalizing the response for requests: {request}")
        logger.debug(f"All request done")

    def finalize(self):
        super().finalize()

    def _post_process(self, data: Dict):
        processed = {k: v for k, v in data.items()}
        for processor in self._processors:
            if not all([i in data for i in processor.input]):
                logger.warning(f"Input settings invalid for the processor: {processor}")
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
        return processed

{% else %}
class TritonPythonModel:
    def initialize(self, args):
        """
        This function allows
        the model to initialize any state associated with this model.
        """
        model_name = args["model_name"]
        model_home = os.path.join(global_config.model_repo, model_name, "1")
        model_config = next((m for m in global_config.models if m.name == model_name), None)
        if model_config is None:
            raise Exception("Model config not found")
        self._model_backend = None
        self._model_config = OmegaConf.to_container(model_config)
        self._in_config = {i["name"]: i for i in self._model_config["input"]}
        self._out_config = {o["name"]: o for o in self._model_config["output"]}
        self._model_repo = args["model_repository"]
        self._device_id = int(json.loads(args["model_instance_device_id"]))
        # default model path
        engine_path = os.path.join(self._model_repo, "1", "model.plan")
        if "tensorrt_engine" in self._model_config:
            self._model_config["tensorrt_engine"] = os.path.join(CHECKPOINTS_DIR, self._model_config["tensorrt_engine"])
        else:
            self._model_config["tensorrt_engine"] = engine_path
        # create backend
        BackendClass = None
        backend_spec = model_config.backend.split('/')
        if backend_spec[-1] == "tensorrtllm":
            BackendClass = TensorRTLLMBackend
        elif backend_spec[-1]  == "polygraphy":
            BackendClass = PolygraphBackend
        elif backend_spec[-1] == "dummy":
            BackendClass = DummyBackend
        if BackendClass is not None:
            self._model_backend = BackendClass(self._model_config, model_home,self._device_id)
        else:
            raise Exception(f"Backend not supported: {model_config.backend}")


    def execute(self, requests):
        """
        execute each request
        """
        logger.info(f"Model {self._model_config['name']} received {len(requests)} requests")

        r_list = []
        responses = []
        for request in requests:
            inputs = {k: None for k in self._in_config}
            for name, config in self._in_config.items():
                dims = config['dims']
                pb_tensor = pb_utils.get_input_tensor_by_name(request, name)
                if pb_utils.Tensor.is_cpu(pb_tensor):
                    tensor = pb_tensor.as_numpy()
                    # auto reshape
                    if len(dims) > 1 and len(tensor.shape) == (len(dims)+1):
                        tensor = np.squeeze(tensor, 0)
                else:
                    tensor = torch.utils.dlpack.from_dlpack(pb_tensor.to_dlpack())
                    # auto reshape
                    if len(dims) > 1 and len(tensor.shape) == (len(dims)+1):
                        tensor = torch.squeeze(tensor, 0)
                inputs[name] = tensor

            for output in self._model_backend(**inputs):
                r_list.append(output)
        for r in r_list:
            logger.info(f"Model {self._model_config['name']} generating responses {r}")
            for k in r:
                v = r[k]
                force_cpu = True if "force_cpu" in self._out_config[k] and self._out_config[k]["force_cpu"] else False
                if isinstance(v, torch.Tensor):
                    tensor = pb_utils.Tensor(k, v.cpu().numpy()) if force_cpu else pb_utils.Tensor.from_dlpack(k, torch.utils.dlpack.to_dlpack(v))
                elif isinstance(v, np.ndarray):
                    tensor = pb_utils.Tensor(k, v)
                elif hasattr(v, "__dlpack__") and hasattr(v, "__dlpack_device__"):
                    tensor =  pb_utils.Tensor(k, np.from_dlpack(v)) if force_cpu else pb_utils.Tensor.from_dlpack(k, v)
                else:
                    responses.append(
                        pb_utils.InferenceResponse(
                            pb_utils.TritonError(f"Unsupported tensor format from the backend. {v}")
                        )
                    )
                    break
                r[k] = tensor
            responses.append(pb_utils.InferenceResponse(output_tensors=[r[k] for k in r]))

        return responses
{% endif %}