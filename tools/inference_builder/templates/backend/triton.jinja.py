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

{% if server_type == "triton" %}

class TritonBackend(ModelBackend):
    """Triton Python Backend"""
    def __init__(self, model_config: Dict, model_home: str):
        logger.debug(f"model_config: {model_config}, model_home: {model_home}")
        super().__init__(model_config, model_home)
        self._model_name = model_config["name"]
        self._input_names = [i['name'] for i in model_config['input']]
        self._output_names = [o['name'] for o in model_config['output']]
        logger.debug(f"TritonBackend created for {self._model_name} with inputs {self._input_names} and outputs {self._output_names}")

    def __call__(self, *args, **kwargs):
        in_data_list = args if args else [kwargs]
        input_config = self._model_config["input"]
        # to determine if we need to stack the input, TODO: below logic needs be more generic
        need_stack = False
        for in_data in in_data_list:
            for key, value in in_data.items():
                input_config = next((i for i in self._model_config["input"] if i['name'] == key), None)
                if input_config is None:
                    logger.error(f"Unexpected input: {key}")
                    continue
                expected_dims = len(input_config["dims"])
                if expected_dims == len(value.shape) + 1:
                    need_stack = True
                    break
        if need_stack:
            in_data_list = [stack_tensors_in_dict(in_data_list)]
        for in_data in in_data_list:
            tensors = []
            for k in in_data:
                tensor = in_data[k]
                batched = "max_batch_size" in self._model_config and self._model_config["max_batch_size"] > 0
                if isinstance(tensor, np.ndarray):
                    tensor = pb_utils.Tensor(k, np.expand_dims(tensor, 0)) if batched else pb_utils.Tensor(k, tensor)
                elif isinstance(tensor, torch.Tensor):
                    if batched:
                        tensor = tensor.unsqueeze(0)
                    tensor = pb_utils.Tensor.from_dlpack(k, torch.utils.dlpack.to_dlpack(tensor))
                elif hasattr(tensor, "__dlpack__") and hasattr(tensor, "__dlpack_device__"):
                    tensor = pb_utils.Tensor.from_dlpack(k, tensor)
                else:
                    yield Error(message="Unsupported input tensor format")
                    return
                tensors.append(tensor)

            llm_request = pb_utils.InferenceRequest(
                model_name = self._model_name,
                requested_output_names = self._output_names,
                inputs = tensors
            )
            finish_reason = "done"
            for idx, response in enumerate(llm_request.exec(decoupled=True)):
                expected = {n: None for n in self._output_names}
                if response.has_error():
                    yield Error(message=f"{response.error().message()}, stream_id={idx}")
                if not response.output_tensors():
                    continue
                for name in expected:
                    config = next((c for c in self._model_config['output'] if c['name'] == name), None)
                    dims = config['dims']
                    output = pb_utils.get_output_tensor_by_name(response, name)
                    if not output:
                        continue
                    if pb_utils.Tensor.is_cpu(output):
                        tensor = output.as_numpy()
                        if tensor.size != 0:
                            expected[name] = np.squeeze(tensor, 0) if len(tensor.shape) == (len(dims)+1) else tensor
                    else:
                        tensor = torch.utils.dlpack.from_dlpack(output.to_dlpack())
                        if tensor.numel() != 0:
                            expected[name] = torch.squeeze(tensor, 0) if len(tensor.shape) == (len(dims)+1) else tensor
                logger.debug(f"TritonBackend saved inference results to: {expected}")
                if all([expected[k] is not None for k in expected]):
                    yield expected
        logger.info(f"Infernece with TritonBackend {self._model_name} accomplished")

{% else %}

import tritonclient.http as httpclient
from tritonclient.utils import *

class TritonBackend(ModelBackend):
    """Triton Python Backend"""
    def __init__(self, model_config: Dict, model_home: str):
        logger.debug(f"model_config: {model_config}")
        super().__init__(model_config, model_home)
        self._model_name = model_config["name"]
        self._input_names = [i['name'] for i in model_config['input']]
        self._output_names = [o['name'] for o in model_config['output']]

        # Wait for Triton server to be ready
        max_retries = 60
        retry_interval = 1
        for attempt in range(max_retries):
            try:
                with httpclient.InferenceServerClient("localhost:8000") as client:
                    if client.is_server_ready():
                        logger.info(f"Triton server is ready after {attempt + 1} attempt(s)")
                        break
            except Exception as e:
                logger.debug(f"Attempt {attempt + 1}/{max_retries}: Triton server not ready yet - {e}")
            if attempt < max_retries - 1:
                import time
                time.sleep(retry_interval)
        else:
            logger.error(f"Triton server did not become ready after {max_retries} attempts")
            raise RuntimeError("Triton server is not ready")

        logger.debug(f"TritonBackend created for {self._model_name} with inputs {self._input_names} and outputs {self._output_names}")

    def __call__(self, *args, **kwargs):
        in_data_list = args if args else [kwargs]
        input_config = self._model_config["input"]
        # to determine if we need to stack the input, TODO: below logic needs be more generic
        need_stack = False
        for in_data in in_data_list:
            for key, value in in_data.items():
                input_config = next((i for i in self._model_config["input"] if i['name'] == key), None)
                if input_config is None:
                    logger.error(f"Unexpected input: {key}")
                    continue
                expected_dims = len(input_config["dims"])
                if expected_dims == len(value.shape) + 1:
                    need_stack = True
                    break
        if need_stack:
            in_data_list = [stack_tensors_in_dict(in_data_list)]
        for in_data in in_data_list:
            inputs = []
            for k in in_data:
                tensor = in_data[k]
                batched = "max_batch_size" in self._model_config and self._model_config["max_batch_size"] > 0
                if isinstance(tensor, torch.Tensor):
                    tensor = tensor.cpu().numpy()
                    if batched:
                        tensor = np.expand_dims(tensor, 0)
                elif not isinstance(tensor, np.ndarray):
                    yield Error(message="Unsupported input tensor format")
                    return
                input = httpclient.InferInput(k, tensor.shape, np_to_triton_dtype(tensor.dtype))
                input.set_data_from_numpy(tensor)
                inputs.append(input)
            outputs = [httpclient.InferRequestedOutput(n) for n in self._output_names]
            with httpclient.InferenceServerClient("localhost:8000") as client:
                response = client.infer(self._model_name, inputs, request_id=str(1), outputs=outputs)

            result = response.get_response()
            expected = {n: None for n in self._output_names}
#            if response.has_error():
#                yield Error(message=f"{response.error().message()}, stream_id={idx}")
            for name in expected:
                config = next((c for c in self._model_config['output'] if c['name'] == name), None)
                dims = config['dims']
                output = response.as_numpy(name)
                expected[name] = np.squeeze(output, 0) if len(output.shape) == (len(dims)+1) else output
            logger.debug(f"TritonBackend saved inference results to: {expected}")
            if all([expected[k] is not None for k in expected]):
                yield expected
        logger.info(f"Inference with TritonBackend {self._model_name} accomplished")

{% endif %}