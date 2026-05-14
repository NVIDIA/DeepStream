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

import importlib

class PytorchBackend(ModelBackend):
    """Pytorch Backend to run models from Huggingface"""
    def __init__(self, model_config:Dict, model_home: str, device_id: int=0):
        super().__init__(model_config, model_home, device_id)
        self._model_name = model_config["name"]
        self._device = f"cuda:{device_id}"
        self._output_names = [o["name"] for o in model_config["output"]]
        self._model_class = model_config["parameters"].get("model_class", "AutoModelForCausalLM")
        self._module = importlib.import_module("transformers")
        model_class = getattr(self._module, self._model_class)
        logger.info(f"Loading pre-trained model {self._model_name} of type {self._model_class} from {self._model_home}")
        self._model = model_class.from_pretrained(self._model_home, torch_dtype="auto", device_map="auto")
        self._model.to(self._device)
        logger.info(f"Model {self._model_name} loaded from {self._model_home}")

    def __call__(self, *args, **kwargs):
        in_data_list = args if args else [kwargs]
        for in_data in in_data_list:
            result = self._model.generate(**in_data)
            if len(self._output_names) != len(result):
                raise ValueError(f"Number of output names ({len(self._output_names)}) does not match the number of output tensors ({len(result)})")
            yield {self._output_names[i]: tensor for i, tensor in enumerate(result)}
